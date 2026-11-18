"""Independent pre/post windows on the capture clock, with bounded late arrivals."""
from __future__ import annotations

from copy import deepcopy
import math
import uuid

from .core import LABELS, _interval, _quality, _scores, interval_union
from .evidence import raw_prediction_diagnostics


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _modalities(item):
    names = item.get("modalities") or [key for key, weight in (item.get("weights") or {}).items() if weight > 0]
    if not names:
        names = [item.get("modality")]
    return sorted(set(names).intersection(("visual", "audio")))


def _support(item, left, right):
    ranges = item.get("modality_intervals") or {}
    return {name: interval_union([(max(left, a), min(right, b)) for a, b in ranges.get(name, [(left, right)])
                                  if min(right, b) > max(left, a)]) for name in _modalities(item)}


class AssessmentWindowCollector:
    def __init__(self, window_seconds=30, min_coverage=.5, min_quality=.2, max_records=3000,
                 settle_seconds=3):
        if (not all(_finite(v) for v in (window_seconds, min_coverage, min_quality, settle_seconds))
                or window_seconds <= 0 or not 0 < min_coverage <= 1 or not 0 <= min_quality <= 1
                or settle_seconds < 0 or not isinstance(max_records, int)
                or isinstance(max_records, bool) or max_records < 1):
            raise ValueError("invalid assessment policy")
        self.window_seconds, self.min_coverage = float(window_seconds), float(min_coverage)
        self.min_quality, self.max_records = float(min_quality), max_records
        self.settle_seconds = float(settle_seconds)
        self.current = self._final = None
        self._records, self._exclusions, self._sequence = {}, {}, 0

    @property
    def lifecycle(self):
        return self.current["lifecycle"] if self.current else None

    def begin(self, session_id, intervention_id, phase, now, comparison_protocol=None):
        if (not session_id or not intervention_id or phase not in ("pre", "post")
                or not _finite(now) or now < 0):
            raise ValueError("assessment needs session/activity identity, phase and valid time")
        protocol = {"version": "matched-modality-v1", "score_scale": "canonical_before_calibration",
                    "weights": {"visual": .5, "audio": .5},
                    "max_mean_quality_gap": .15, "window_seconds": self.window_seconds,
                    "min_coverage": self.min_coverage, "min_quality": self.min_quality}
        if comparison_protocol is not None and comparison_protocol != protocol:
            raise ValueError("pre/post assessment policies must match")
        self.current = {"assessment_id": str(uuid.uuid4()), "session_id": session_id,
                        "intervention_id": intervention_id, "phase": phase,
                        "window_start": float(now), "window_end": now+self.window_seconds,
                        "window_seconds": self.window_seconds, "settle_until": now+self.window_seconds+self.settle_seconds,
                        "status": "collecting", "lifecycle": "collecting", "comparison_protocol": protocol}
        self._records, self._exclusions, self._sequence, self._final = {}, {}, 0, None
        return self.snapshot(now)

    def _exclude(self, reason):
        self._exclusions[reason] = self._exclusions.get(reason, 0)+1
        return False

    def advance(self, now):
        """Advance by arrival time even if the timer callback was delayed."""
        if self.lifecycle == "collecting" and now >= self.current["window_end"]:
            self.current.update(lifecycle="settling", status="settling")
        if self.lifecycle == "settling" and now >= self.current["settle_until"]:
            self._finalize(now)

    def add(self, observation, received_at=None):
        if self.current is None or self._final is not None:
            return False
        if not isinstance(observation, dict) or observation.get("session_id") != self.current["session_id"]:
            return self._exclude("different_session")
        arrival = observation.get("processed_at") if received_at is None else received_at
        if arrival is not None:
            if not _finite(arrival):
                return self._exclude("invalid_arrival_time")
            self.advance(arrival)
            if self._final is not None:
                return False
        key = observation.get("audio_event_id") or observation.get("event_id")
        if not key:
            return self._exclude("missing_event_id")
        # A corrected result replaces the old observation even if the new one is unusable.
        self._records.pop(str(key), None)
        interval = _interval(observation)
        if interval is None:
            return self._exclude("invalid_interval")
        start, end = self.current["window_start"], self.current["window_end"]
        if interval[1] <= start or interval[0] >= end:
            return self._exclude("outside_window")
        if interval[0] < start or interval[1] > end:
            return self._exclude("crosses_window_boundary")
        if observation.get("stale") or observation.get("paused") or observation.get("system_playback"):
            return self._exclude("stale_paused_or_system_playback")
        if observation.get("conflict"):
            return self._exclude("modalities_conflict")
        if (not observation.get("valid") or _quality(observation) < self.min_quality
                or observation.get("decision") == "abstain" or _scores(observation.get("scores")) is None
                or not raw_prediction_diagnostics(observation.get("raw_scores"))["decision_eligible"]):
            return self._exclude("invalid_or_uncertain_observation")
        self._sequence += 1
        self._records[str(key)] = {**deepcopy(observation), "_sequence": self._sequence}
        if len(self._records) > self.max_records:
            self._records.pop(next(iter(self._records)))
            self._exclude("record_limit")
        return True

    def _statistics(self, now):
        record = deepcopy(self.current)
        start, end = record["window_start"], min(record["window_end"], max(record["window_start"], now))
        # Include actual sub-interval boundaries. A partially observed visual stream
        # must not contribute to an entire audio segment.
        spans, points = [], set()
        for item in self._records.values():
            segments = _support(item, max(start, item["t_start"]), min(end, item["t_end"]))
            for ranges in segments.values():
                for a, b in ranges:
                    points.update((a, b))
            spans.append((item, segments))
        points = sorted(points)
        mass = covered = 0.
        integrated = dict.fromkeys(LABELS, 0.)
        selected, versions, config_hashes = {}, {}, set()
        mix = dict.fromkeys(("joint", "visual_only", "audio_only"), 0.)
        modality_seconds, score_seconds, quality_time, score_time, weight_time = {}, {}, {}, {}, {}
        for a, b in zip(points, points[1:]):
            active = []
            for item, supports in spans:
                names = [m for m, ranges in supports.items() if any(left < b and right > a for left, right in ranges)]
                if names:
                    active.append((item, names))
            if not active:
                continue
            item, names = max(active, key=lambda pair: (len(pair[1]), pair[0]["_sequence"]))
            selected[item["_sequence"]] = item
            config_hashes.add(item.get("fusion_config_hash"))
            seconds = b-a
            covered += seconds
            mix["joint" if len(names) == 2 else names[0]+"_only"] += seconds
            weighted = seconds*_quality(item)
            mass += weighted
            for label, score in _scores(item["scores"]).items():
                integrated[label] += weighted*score
            item_modalities = _modalities(item)
            for name in names:
                values = (item.get("model_versions") or {}).get(name)
                if not values:
                    values = item.get("model_version") if item.get("modality") == name else None
                versions.setdefault(name, set()).update([values] if isinstance(values, str) else values or [None])
                modality_seconds[name] = modality_seconds.get(name, 0.)+seconds
                quality = (item.get("per_modality_quality") or {}).get(name)
                distribution = _scores((item.get("per_modality_scores") or {}).get(name))
                if len(item_modalities) == 1:
                    quality = quality if quality is not None else _quality(item)
                    distribution = distribution or _scores(item["scores"])
                if distribution and _finite(quality) and self.min_quality <= quality <= 1:
                    score_seconds[name] = score_seconds.get(name, 0.)+seconds
                    quality_time[name] = quality_time.get(name, 0.)+seconds*quality
                    scores = score_time.setdefault(name, dict.fromkeys(LABELS, 0.))
                    for label in LABELS:
                        scores[label] += seconds*distribution[label]
            # Preserve the actual fusion weights separately from protocol weights.
            weights = item.get("weights") or ({item_modalities[0]: 1.} if len(item_modalities) == 1 else {})
            for name, weight in weights.items():
                if _finite(weight) and weight >= 0:
                    weight_time[name] = weight_time.get(name, 0.)+seconds*weight
        versions_ok = bool(versions) and all(len(v) == 1 and None not in v for v in versions.values())
        config_ok = len(config_hashes) == 1 and None not in config_hashes
        coverage = covered/self.window_seconds
        by_modality = {m: s/self.window_seconds for m, s in modality_seconds.items()}
        sufficient = coverage >= self.min_coverage and all(value >= self.min_coverage for value in by_modality.values())
        reasons = list(self._exclusions)
        if not sufficient:
            reasons.append("insufficient_valid_coverage")
        if not versions_ok:
            reasons.append("missing_or_changed_model_versions")
        if not config_ok:
            reasons.append("missing_or_changed_fusion_configuration")
        scores = {k: v/mass for k, v in integrated.items()} if mass else None
        record.update(scores=scores, mean_scores=scores, covered_seconds=covered, coverage=coverage,
                      coverage_by_modality=by_modality, modalities=sorted(versions),
                      model_versions={m: sorted(v for v in values if v) for m, values in versions.items()},
                      incomplete_model_versions=sorted(m for m, values in versions.items() if None in values),
                      fusion_config_hash=next(iter(config_hashes)) if config_ok else None,
                      per_modality_scores={m: {k: v/score_seconds[m] for k, v in scores.items()}
                                           for m, scores in score_time.items()},
                      score_coverage_by_modality={m: t/self.window_seconds for m, t in score_seconds.items()},
                      mean_quality_by_modality={m: v/score_seconds[m] for m, v in quality_time.items()},
                      modality_mix={k: {"seconds": s, "fraction": s/covered if covered else 0.} for k, s in mix.items()},
                      weight_summary={k: v/covered for k, v in weight_time.items()} if covered else {},
                      source_ids=list(dict.fromkeys(source for item in selected.values()
                                                   for source in item.get("source_ids", [item.get("event_id")]) if source)),
                      exclusion_reasons=list(dict.fromkeys(reasons)), exclusion_counts=deepcopy(self._exclusions),
                      remaining_seconds=max(0., record["window_end"]-now),
                      settle_remaining_seconds=max(0., record["settle_until"]-now) if self.lifecycle == "settling" else 0.)
        return record, bool(sufficient and versions_ok and config_ok and scores)

    def _finalize(self, now, skip=False):
        record, sufficient = self._statistics(now)
        record.update(lifecycle="skipped" if skip else "finalized",
                      status="skipped" if skip else "complete" if sufficient else "insufficient",
                      finalized_at=now, remaining_seconds=0., settle_remaining_seconds=0.)
        if skip:
            record["exclusion_reasons"].append("user_skipped")
        self.current.update(lifecycle=record["lifecycle"], status=record["status"])
        self._final = record

    def snapshot(self, now):
        if self.current is None:
            return {"status": "insufficient", "exclusion_reasons": ["window_not_started"],
                    "coverage": 0., "scores": None, "source_ids": [], "modalities": []}
        self.advance(now)
        return deepcopy(self._final) if self._final is not None else self._statistics(now)[0]

    def finish(self, now, skip=False):
        """An explicit user transition may end the wait once capture has ended."""
        self.advance(now)
        if self._final is None and self.current and (skip or self.lifecycle == "settling"):
            self._finalize(now, skip)
        return self.snapshot(now)


def compare_assessments(pre, post):
    pre, post = pre or {}, post or {}
    reasons = []
    for field, reason in (("session_id", "different_session"), ("intervention_id", "different_activity"),
                          ("fusion_config_hash", "different_fusion_configuration"),
                          ("comparison_protocol", "different_or_missing_comparison_protocol")):
        if not pre.get(field) or pre[field] != post.get(field):
            reasons.append(reason)
    if pre.get("phase") != "pre" or post.get("phase") != "post":
        reasons.append("independent_phases_required")
    if post.get("window_start", -1) < pre.get("window_end", 0):
        reasons.append("windows_overlap")
    if not pre.get("source_ids") or not post.get("source_ids"):
        reasons.append("missing_source_evidence")
    elif set(pre["source_ids"]) & set(post["source_ids"]):
        reasons.append("reused_source_evidence")
    windows_final = all(w.get("lifecycle") == "finalized" and w.get("status") in ("complete", "insufficient")
                        for w in (pre, post))
    can_match = windows_final and not reasons
    for phase, window in (("pre", pre), ("post", post)):
        if window.get("status") != "complete":
            reasons.append(phase+"_window_incomplete")
    modalities = set(pre.get("modalities", [])) & set(post.get("modalities", []))
    if not modalities or sorted(pre.get("modalities", [])) != sorted(post.get("modalities", [])):
        reasons.append("different_modalities")
    protocol = pre.get("comparison_protocol") or {}
    per_delta, exclusions = {}, {}
    if can_match:
        for name in sorted(modalities):
            conditions = []
            v1 = (pre.get("model_versions") or {}).get(name)
            v2 = (post.get("model_versions") or {}).get(name)
            if len(v1 or []) != 1 or v1 != v2 or any(name in w.get("incomplete_model_versions", []) for w in (pre, post)):
                conditions.append("different_model_versions")
            qs = [w["mean_quality_by_modality"].get(name) for w in (pre, post)]
            if any(q is None for q in qs) or abs(qs[0]-qs[1]) > protocol["max_mean_quality_gap"]:
                conditions.append("different_or_missing_quality")
            if any(w["score_coverage_by_modality"].get(name, 0.) < protocol["min_coverage"] for w in (pre, post)):
                conditions.append("insufficient_modality_score_coverage")
            if conditions:
                exclusions[name] = conditions
                reasons.extend(conditions)
            else:
                before, after = (w["per_modality_scores"][name] for w in (pre, post))
                per_delta[name] = {label: before[label]-after[label] for label in LABELS}
    aggregate = None
    if not reasons and per_delta:
        weights = protocol["weights"]
        denominator = sum(weights[m] for m in per_delta)
        aggregate = {label: sum(weights[m]*change[label] for m, change in per_delta.items())/denominator
                     for label in LABELS}
    return {"comparable": aggregate is not None, "model_delta": aggregate,
            "scope": "protocol" if aggregate is not None else "per_modality_only" if per_delta else "insufficient",
            "per_modality_delta": per_delta, "modality_exclusions": exclusions,
            "reasons": list(dict.fromkeys(reasons)), "comparison_protocol": protocol,
            "interpretation": "辅助模型观察的分布变化，不能替代用户感受或推断调节疗效。",
            "pre_assessment_id": pre.get("assessment_id"), "post_assessment_id": post.get("assessment_id")}
