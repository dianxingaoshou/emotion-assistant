"""Independent, auditable pre/post observation windows on the session clock."""
from __future__ import annotations

from copy import deepcopy
import math
import uuid

from .core import LABELS, _interval, _quality, _scores, interval_duration, interval_union
from .evidence import raw_prediction_diagnostics


class AssessmentWindowCollector:
    def __init__(self, window_seconds=30, min_coverage=.5, min_quality=.2, max_records=3000):
        if (not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                    for value in (window_seconds, min_coverage, min_quality))
                or window_seconds <= 0 or not 0 < min_coverage <= 1 or not 0 <= min_quality <= 1
                or not isinstance(max_records, int) or isinstance(max_records, bool) or max_records < 1):
            raise ValueError("invalid assessment policy")
        self.window_seconds, self.min_coverage = float(window_seconds), float(min_coverage)
        self.min_quality, self.max_records = float(min_quality), max_records
        self.current = None
        self._records = {}
        self._exclusions = {}
        self._sequence = 0

    def begin(self, session_id, intervention_id, phase, now):
        if (not session_id or not intervention_id or phase not in ("pre", "post")
                or not isinstance(now, (int, float)) or not math.isfinite(now) or now < 0):
            raise ValueError("assessment needs session/activity identity, phase and valid time")
        self.current = {"assessment_id": str(uuid.uuid4()), "session_id": session_id,
                        "intervention_id": intervention_id, "phase": phase,
                        "window_start": float(now), "window_end": float(now)+self.window_seconds,
                        "window_seconds": self.window_seconds, "status": "collecting"}
        self._records, self._exclusions, self._sequence = {}, {}, 0
        return self.snapshot(now)

    def _exclude(self, reason):
        self._exclusions[reason] = self._exclusions.get(reason, 0)+1
        return False

    def add(self, observation):
        if self.current is None or self.current["status"] != "collecting":
            return False
        if not isinstance(observation, dict) or observation.get("session_id") != self.current["session_id"]:
            return self._exclude("different_session")
        interval = _interval(observation)
        if interval is None:
            return self._exclude("invalid_interval")
        if interval[1] <= self.current["window_start"] or interval[0] >= self.current["window_end"]:
            return self._exclude("outside_window")
        # A classifier output describes its whole input segment. Cropping only
        # its timestamps cannot remove activity-period samples from that output.
        if interval[0] < self.current["window_start"] or interval[1] > self.current["window_end"]:
            return self._exclude("crosses_window_boundary")
        key = observation.get("audio_event_id") or observation.get("event_id")
        if not key:
            return self._exclude("missing_event_id")
        # Replace a prior revision even when it is now invalid or conflicting.
        self._records.pop(str(key), None)
        if observation.get("stale") or observation.get("paused") or observation.get("system_playback"):
            return self._exclude("stale_paused_or_system_playback")
        if observation.get("conflict"):
            return self._exclude("modalities_conflict")
        if (not observation.get("valid") or _quality(observation) < self.min_quality
                or _scores(observation.get("scores")) is None
                or not raw_prediction_diagnostics(observation.get("raw_scores"))["decision_eligible"]):
            return self._exclude("invalid_or_uncertain_observation")
        self._sequence += 1
        self._records[str(key)] = {**deepcopy(observation), "_sequence": self._sequence}
        if len(self._records) > self.max_records:
            first = min(self._records, key=lambda item: self._records[item]["_sequence"])
            self._records.pop(first)
            self._exclude("record_limit")
        return True

    def snapshot(self, now):
        if self.current is None:
            return {"status": "insufficient", "exclusion_reasons": ["window_not_started"],
                    "coverage": 0.0, "scores": None, "source_ids": [], "modalities": []}
        record = deepcopy(self.current)
        start, end = record["window_start"], min(record["window_end"], max(record["window_start"], now))
        candidates = [(item, max(start, item["t_start"]), min(end, item["t_end"]))
                      for item in self._records.values() if item["t_start"] < end and item["t_end"] > start]
        candidates = [(item, a, b) for item, a, b in candidates if b > a]
        points = sorted({point for _, a, b in candidates for point in (a, b)})
        score_integrals = dict.fromkeys(LABELS, 0.0)
        mass = covered = 0.0
        selected, version_sets, config_hashes, modal_intervals = {}, {}, set(), {}
        for a, b in zip(points, points[1:]):
            active = [item for item, left, right in candidates if left < b and right > a]
            if not active:
                continue
            # A joint observation supersedes a rolling visual-only summary.
            item = max(active, key=lambda value: (value.get("status") == "joint", value["_sequence"]))
            selected[item["_sequence"]] = item
            weight = (b-a)*_quality(item)
            mass += weight
            covered += b-a
            scores = _scores(item["scores"])
            for label in LABELS:
                score_integrals[label] += weight*scores[label]
            modalities = item.get("modalities") or [key for key, value in (item.get("weights") or {}).items() if value > 0]
            if not modalities and item.get("modality") in ("visual", "audio"):
                modalities = [item["modality"]]
            versions = item.get("model_versions") or {}
            for modality in modalities:
                values = versions.get(modality, [])
                if isinstance(values, str):
                    values = [values]
                if not values and item.get("modality") == modality and item.get("model_version"):
                    values = [item["model_version"]]
                version_sets.setdefault(modality, set()).update(str(value) for value in values)
                ranges = (item.get("modality_intervals") or {}).get(modality, [(a, b)])
                modal_intervals.setdefault(modality, []).extend((max(a, left), min(b, right)) for left, right in ranges if min(b, right) > max(a, left))
            if item.get("fusion_config_hash"):
                config_hashes.add(str(item["fusion_config_hash"]))
        scores = {label: score_integrals[label]/mass for label in LABELS} if mass else None
        versions = {key: sorted(values) for key, values in sorted(version_sets.items())}
        coverage_by_modality = {key: interval_duration(values)/record["window_seconds"] for key, values in modal_intervals.items()}
        coverage = covered/record["window_seconds"]
        reasons = list(self._exclusions)
        sufficient = bool(scores and versions and coverage >= self.min_coverage
                          and all(value >= self.min_coverage for value in coverage_by_modality.values()))
        if not sufficient:
            reasons.append("insufficient_valid_coverage")
        if any(len(values) != 1 for values in versions.values()):
            reasons.append("missing_or_changed_model_versions")
            sufficient = False
        if len(config_hashes) != 1:
            reasons.append("missing_or_changed_fusion_configuration")
            sufficient = False
        if record["status"] == "collecting" and now >= record["window_end"]:
            record["status"] = "complete" if sufficient else "insufficient"
        record.update(scores=scores, mean_scores=scores, covered_seconds=covered, coverage=coverage,
                      coverage_by_modality=coverage_by_modality, modalities=sorted(versions),
                      model_versions=versions, fusion_config_hash=next(iter(config_hashes)) if len(config_hashes) == 1 else None,
                      source_ids=list(dict.fromkeys(source for item in selected.values()
                                                   for source in item.get("source_ids", [item.get("event_id")]) if source)),
                      exclusion_reasons=list(dict.fromkeys(reasons)), exclusion_counts=deepcopy(self._exclusions),
                      remaining_seconds=max(0.0, record["window_end"]-now))
        return record

    def finish(self, now, skip=False):
        result = self.snapshot(now)
        if skip:
            result["status"] = "skipped"
            result["exclusion_reasons"] = list(dict.fromkeys(result["exclusion_reasons"]+["user_skipped"]))
        if self.current is not None and result["status"] != "collecting":
            self.current["status"] = result["status"]
        return result


def compare_assessments(pre, post):
    reasons = []
    pre, post = pre or {}, post or {}
    if pre.get("status") != "complete":
        reasons.append("pre_window_incomplete")
    if post.get("status") != "complete":
        reasons.append("post_window_incomplete")
    if not pre.get("session_id") or pre.get("session_id") != post.get("session_id"):
        reasons.append("different_session")
    if not pre.get("intervention_id") or pre.get("intervention_id") != post.get("intervention_id"):
        reasons.append("different_activity")
    if pre.get("phase") != "pre" or post.get("phase") != "post":
        reasons.append("independent_phases_required")
    if post.get("window_start", -1) < pre.get("window_end", 0):
        reasons.append("windows_overlap")
    if not pre.get("modalities") or pre.get("modalities") != post.get("modalities"):
        reasons.append("different_modalities")
    if not pre.get("model_versions") or pre.get("model_versions") != post.get("model_versions"):
        reasons.append("different_model_versions")
    if not pre.get("fusion_config_hash") or pre.get("fusion_config_hash") != post.get("fusion_config_hash"):
        reasons.append("different_fusion_configuration")
    if set(pre.get("source_ids", [])) & set(post.get("source_ids", [])):
        reasons.append("reused_source_evidence")
    before, after = _scores(pre.get("scores")), _scores(post.get("scores"))
    if before is None or after is None:
        reasons.append("missing_scores")
    return {"comparable": not reasons, "reasons": reasons,
            "model_delta": {label: before[label]-after[label] for label in LABELS} if not reasons else None,
            "interpretation": "辅助模型观察的分布变化，不能替代用户感受或推断调节疗效。",
            "pre_assessment_id": pre.get("assessment_id"), "post_assessment_id": post.get("assessment_id")}
