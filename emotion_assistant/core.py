"""Local, dependency-free alignment, fusion and voluntary intervention logic.

All observation times are seconds on one session-relative monotonic clock.
Quality is an engineering measurement, never a probability of psychological truth.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import threading
import time
import uuid

from .evidence import jensen_shannon_divergence, raw_prediction_diagnostics

LABELS = ("positive", "neutral", "negative", "uncertain")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def interval_union(intervals) -> list[list[float]]:
    """Merge overlapping/touching valid intervals; never count overlaps twice."""
    clean = []
    for interval in intervals:
        try:
            start, end = map(float, interval)
        except (TypeError, ValueError):
            continue
        if math.isfinite(start) and math.isfinite(end) and end > start >= 0:
            clean.append((start, end))
    merged = []
    for start, end in sorted(clean):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def interval_duration(intervals) -> float:
    return sum(end - start for start, end in interval_union(intervals))


def _scores(value) -> dict | None:
    if not isinstance(value, dict):
        return None
    try:
        values = {label: float(value.get(label, 0.0)) for label in LABELS}
    except (ValueError, TypeError):
        return None
    if any(not math.isfinite(v) or v < 0 for v in values.values()):
        return None
    total = sum(values.values())
    if total <= 0:
        return None
    # Normalize across ALL four labels, including uncertain mass.
    return {label: value / total for label, value in values.items()}


def _interval(observation) -> tuple[float, float] | None:
    try:
        start, end = float(observation["t_start"]), float(observation["t_end"])
    except (KeyError, ValueError, TypeError):
        return None
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        return None
    return start, end


def _quality(observation) -> float:
    try:
        quality = float(observation.get("quality", 0))
    except (ValueError, TypeError):
        return 0.0
    return min(1.0, max(0.0, quality)) if math.isfinite(quality) else 0.0


class FusionEngine:
    """Fuse audio only with visual evidence from its actual recorded interval.

    History is bounded by both count and session time. Duplicate event IDs replace
    their previous observation rather than amplifying the same evidence. No audio
    result is cached into subsequent visual-only summaries.
    """

    def __init__(self, min_quality=0.2, history_seconds=120.0, max_history=3000,
                 visual_weight=1.0, audio_weight=1.0):
        if not 0 <= min_quality <= 1:
            raise ValueError("min_quality must be in [0, 1]")
        if history_seconds <= 0 or max_history < 1:
            raise ValueError("history limits must be positive")
        if visual_weight <= 0 or audio_weight <= 0:
            raise ValueError("base weights must be positive")
        self.min_quality = float(min_quality)
        self.history_seconds = float(history_seconds)
        self.max_history = int(max_history)
        self.visual_weight, self.audio_weight = float(visual_weight), float(audio_weight)
        self.config = {"min_quality": self.min_quality, "visual_weight": self.visual_weight,
                       "audio_weight": self.audio_weight, "conflict_js_threshold": .18,
                       "conflict_dominance": .6, "version": "quality-aligned-v2"}
        self.config_hash = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        self._visual = {}
        self._session_id = None
        self._latest_end = 0.0
        self._lock = threading.RLock()

    def clear(self):
        with self._lock:
            self._visual.clear()
            self._session_id = None
            self._latest_end = 0.0

    def add_visual(self, observation: dict) -> None:
        if observation.get("modality") != "visual":
            raise ValueError("add_visual requires a visual observation")
        if not observation.get("event_id") or _interval(observation) is None:
            raise ValueError("visual observation needs an event_id and valid interval")
        with self._lock:
            session_id = observation.get("session_id")
            if self._session_id != session_id:
                self._visual.clear()
                self._latest_end = 0.0
                self._session_id = session_id
            self._latest_end = max(self._latest_end, float(observation["t_end"]))
            self._visual[observation["event_id"]] = deepcopy(observation)
            cutoff = self._latest_end - self.history_seconds
            self._visual = {key: item for key, item in self._visual.items()
                            if float(item["t_end"]) > cutoff}
            if len(self._visual) > self.max_history:
                ordered = sorted(self._visual.values(), key=lambda item: float(item["t_end"]))
                self._visual = {item["event_id"]: item for item in ordered[-self.max_history:]}

    def _eligible(self, observation):
        return (bool(observation.get("valid")) and _quality(observation) >= self.min_quality
                and _scores(observation.get("scores")) is not None
                and _interval(observation) is not None
                and raw_prediction_diagnostics(observation.get("raw_scores"))["decision_eligible"])

    def _visual_in_interval(self, start, end):
        candidates = []
        for item in self._visual.values():
            if not self._eligible(item):
                continue
            left, right = max(start, float(item["t_start"])), min(end, float(item["t_end"]))
            if right > left:
                candidates.append((item, left, right))
        if not candidates:
            return None
        # Integrate over a partition so overlapping visual windows cannot double
        # count duration or quality. Equal active windows share each tiny interval.
        boundaries = sorted({p for _, left, right in candidates for p in (left, right)})
        mass = dict.fromkeys(LABELS, 0.0)
        quality_integral, covered = 0.0, 0.0
        for left, right in zip(boundaries, boundaries[1:]):
            active = [item for item, a, b in candidates if a < right and b > left]
            if not active:
                continue
            duration = right - left
            covered += duration
            quality_sum = sum(_quality(item) for item in active)
            local_quality = quality_sum / len(active)
            quality_integral += duration * local_quality
            for item in active:
                weight = duration * _quality(item) / len(active)
                distribution = _scores(item["scores"])
                for label in LABELS:
                    mass[label] += weight * distribution[label]
        coverage = covered / (end - start)
        # Partial evidence lowers the quality of the whole requested interval.
        quality = quality_integral / (end - start)
        return {"scores": _scores(mass), "quality": quality, "coverage": coverage,
                "source_ids": [item["event_id"] for item, _, _ in candidates],
                "model_versions": sorted({str(item["model_version"]) for item, _, _ in candidates if item.get("model_version")}),
                "intervals": interval_union([(left, right) for _, left, right in candidates]),
                "covered_seconds": covered}

    @staticmethod
    def _empty(modality, start, end, reason):
        return {"event_id": str(uuid.uuid4()), "modality": modality,
                "t_start": start, "t_end": end, "scores": None, "quality": 0.0,
                "weights": {"visual": 0.0, "audio": 0.0}, "valid": False,
                "reason": reason, "status": "unavailable", "conflict": False,
                "source_ids": []}

    def visual_summary(self, start, end) -> dict:
        start, end = float(start), float(end)
        result = self._empty("visual_summary", start, end, "no_usable_visual_in_interval")
        if _interval(result) is None:
            result["reason"] = "invalid_interval"
            return result
        with self._lock:
            visual = self._visual_in_interval(start, end)
            result["session_id"] = self._session_id
            result.update(fusion_config_hash=self.config_hash, fusion_config=deepcopy(self.config),
                          model_versions={}, modalities=[], modality_intervals={})
            if visual:
                result.update({"coverage": visual["coverage"],
                               "covered_seconds": visual["covered_seconds"]})
            if visual and visual["quality"] >= self.min_quality:
                result.update(scores=visual["scores"], quality=visual["quality"],
                              valid=True, status="visual_only", reason="visual_only",
                              weights={"visual": 1.0, "audio": 0.0}, source_ids=visual["source_ids"],
                              model_versions={"visual": visual["model_versions"]}, modalities=["visual"],
                              modality_intervals={"visual": visual["intervals"]})
        return result

    def fuse_audio(self, observation: dict) -> dict:
        if observation.get("modality") != "audio":
            raise ValueError("fuse_audio requires an audio observation")
        interval = _interval(observation)
        start, end = interval or (observation.get("t_start", 0), observation.get("t_end", 0))
        result = self._empty("fusion", start, end, "no_usable_modalities")
        result["session_id"] = observation.get("session_id")
        result["audio_event_id"] = observation.get("event_id")
        result.update(fusion_config_hash=self.config_hash, fusion_config=deepcopy(self.config),
                      model_versions={}, modalities=[], modality_intervals={})
        if interval is None:
            result["reason"] = "invalid_audio_interval"
            return result
        with self._lock:
            visual = (self._visual_in_interval(start, end)
                      if observation.get("session_id") == self._session_id else None)
            visual_ok = bool(visual and visual["quality"] >= self.min_quality)
            audio_ok = self._eligible(observation)
            if visual:
                result["visual_coverage"] = visual["coverage"]
            if not (visual_ok or audio_ok):
                return result
            v_weight = self.visual_weight * visual["quality"] if visual_ok else 0.0
            a_weight = self.audio_weight * _quality(observation) if audio_ok else 0.0
            denominator = v_weight + a_weight
            weights = {"visual": v_weight / denominator, "audio": a_weight / denominator}
            audio_scores = _scores(observation.get("scores")) if audio_ok else None
            scores = {label: ((weights["visual"] * visual["scores"][label] if visual_ok else 0)
                              + (weights["audio"] * audio_scores[label] if audio_ok else 0))
                      for label in LABELS}
            status = "joint" if visual_ok and audio_ok else "visual_only" if visual_ok else "audio_only"
            # A 100% relative weight does NOT turn weak evidence into high quality.
            quality = ((v_weight * visual["quality"] if visual_ok else 0)
                       + (a_weight * _quality(observation) if audio_ok else 0)) / denominator
            conflict = False
            divergence = None
            if visual_ok and audio_ok:
                v, a = visual["scores"], audio_scores
                divergence = jensen_shannon_divergence(v, a)
                v_label, a_label = max(v, key=v.get), max(a, key=a.get)
                conflict = ((v["positive"] >= 0.5 and a["negative"] >= 0.5)
                            or (a["positive"] >= 0.5 and v["negative"] >= 0.5)
                            or (v_label != a_label and "uncertain" not in (v_label, a_label)
                                and v[v_label] >= .6 and a[a_label] >= .6
                                and divergence is not None and divergence >= .18))
            source_ids = list(visual["source_ids"]) if visual_ok else []
            if audio_ok and observation.get("event_id"):
                source_ids.append(observation["event_id"])
            versions, intervals = {}, {}
            if visual_ok:
                versions["visual"] = visual["model_versions"]
                intervals["visual"] = visual["intervals"]
            if audio_ok:
                versions["audio"] = [str(observation["model_version"])] if observation.get("model_version") else []
                intervals["audio"] = [[start, end]]
            result.update(scores=scores, quality=quality, weights=weights, valid=True,
                          status=status, conflict=conflict, source_ids=source_ids,
                          model_versions=versions, modalities=sorted(versions), modality_intervals=intervals,
                          conflict_diagnostics={"jensen_shannon": divergence, "threshold": .18},
                          disagreement_js=divergence if divergence is not None else 0.0,
                          reason="modalities_conflict_user_confirmation" if conflict else status)
        return result


class InterventionFlow:
    """An explicit user-controlled observation → confirmation → activity flow."""

    def __init__(self):
        self._state = "observing"
        self._current = None
        self._started_monotonic = None
        self._lock = threading.RLock()

    @property
    def state(self):
        with self._lock:
            return self._state

    @property
    def current(self):
        with self._lock:
            return deepcopy(self._current)

    def request(self) -> bool:
        with self._lock:
            if self._state not in ("observing", "cooldown"):
                return False
            self._state = "confirming"
            self._current = None
            self._started_monotonic = None
            return True

    def confirm(self, accepted: bool) -> None:
        with self._lock:
            if self._state != "confirming":
                raise ValueError("confirmation requires a pending request")
            self._state = "ready" if accepted else "cooldown"

    @staticmethod
    def _distress(value, optional=False):
        if optional and value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10:
            raise ValueError("distress must be an integer from 0 to 10")
        return value

    def start(self, activity: str, pre: int, duration: int = 60) -> dict:
        pre = self._distress(pre)
        if not isinstance(activity, str) or not activity.strip():
            raise ValueError("activity must be a nonempty name")
        if isinstance(duration, bool) or not isinstance(duration, int) or duration < 1:
            raise ValueError("duration must be a positive number of seconds")
        with self._lock:
            if self._state != "ready":
                raise ValueError("activity requires user confirmation first")
            self._current = {"id": str(uuid.uuid4()), "activity": activity.strip(),
                             "pre": pre, "post": None, "delta": None,
                             "duration": duration, "started_at": utc_now(),
                             "ended_at": None, "helpfulness": None,
                             "status": "running", "evaluation": "insufficient",
                             "step_index": 0, "steps": [], "checkins": []}
            self._started_monotonic = time.monotonic()
            self._state = "running"
            return deepcopy(self._current)

    def annotate(self, fields: dict) -> dict:
        """Attach workflow evidence without changing the protected state fields."""
        if not isinstance(fields, dict):
            raise ValueError("activity metadata must be a dictionary")
        protected = {"id", "activity", "pre", "post", "delta", "status", "evaluation",
                     "started_at", "ended_at", "helpfulness", "duration"}
        if protected.intersection(fields):
            raise ValueError("protected activity fields cannot be replaced by metadata")
        with self._lock:
            if self._current is None:
                raise ValueError("there is no activity to annotate")
            self._current.update(deepcopy(fields))
            return deepcopy(self._current)

    def advance_step(self, max_steps: int) -> dict:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("activity must contain at least one step")
        with self._lock:
            if self._state != "running":
                raise ValueError("there is no running activity")
            self._current["step_index"] = min(max_steps - 1, self._current.get("step_index", 0) + 1)
            return deepcopy(self._current)

    def checkin(self, feeling: str) -> dict:
        if feeling not in ("comfortable", "no_change", "uncomfortable"):
            raise ValueError("unsupported activity feedback")
        with self._lock:
            if self._state != "running":
                raise ValueError("there is no running activity")
            checkin = {"feeling": feeling, "created_at": utc_now(),
                       "elapsed_seconds": max(0, time.monotonic() - self._started_monotonic),
                       "step_index": self._current.get("step_index", 0)}
            self._current["checkins"] = (self._current.get("checkins", []) + [checkin])[-100:]
            self._current["last_checkin"] = checkin
            return deepcopy(self._current)

    def complete(self, post: int | None, helpfulness: str) -> dict:
        post = self._distress(post, optional=True)
        if not isinstance(helpfulness, str):
            raise ValueError("helpfulness must be text")
        with self._lock:
            if self._state != "running":
                raise ValueError("there is no running activity to complete")
            delta = self._current["pre"] - post if post is not None else None
            evaluation = ("insufficient" if delta is None else "improved" if delta > 0
                          else "worse" if delta < 0 else "unchanged")
            numeric_evaluation = evaluation
            discomfort = (helpfulness in ("worse", "更不舒服") or self._current.get("discomfort") is True
                          or any(item.get("feeling") == "uncomfortable" for item in self._current.get("checkins", [])))
            if discomfort:
                evaluation = "discomfort"
            self._current.update(post=post, delta=delta, helpfulness=helpfulness,
                                 ended_at=utc_now(), status="completed", evaluation=evaluation,
                                 numeric_evaluation=numeric_evaluation, discomfort=discomfort,
                                 elapsed_seconds=max(0, time.monotonic() - self._started_monotonic))
            self._state = "cooldown"
            return deepcopy(self._current)

    def cancel(self, reason="user_cancelled") -> dict:
        with self._lock:
            if self._state not in ("confirming", "ready", "running"):
                raise ValueError("there is no active request or activity to cancel")
            if self._current is None:
                self._current = {"id": str(uuid.uuid4()), "activity": None,
                                 "pre": None, "post": None, "delta": None,
                                 "started_at": None, "evaluation": "insufficient"}
            discomfort = (reason in ("uncomfortable", "discomfort") or self._current.get("discomfort") is True
                          or self._current.get("helpfulness") in ("worse", "更不舒服")
                          or any(item.get("feeling") == "uncomfortable" for item in self._current.get("checkins", [])))
            self._current.update(ended_at=utc_now(), status="cancelled", post=None, delta=None,
                                 evaluation="discomfort" if discomfort else "insufficient",
                                 numeric_evaluation="insufficient", discomfort=discomfort, stop_reason=reason)
            if self._started_monotonic is not None:
                self._current["elapsed_seconds"] = max(0, time.monotonic() - self._started_monotonic)
            self._state = "cooldown"
            return deepcopy(self._current)
