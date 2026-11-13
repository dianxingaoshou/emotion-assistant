"""Prediction diagnostics and bounded, revisable sustained-evidence decisions.

These measures describe engineering evidence, not calibrated correctness.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math


def raw_prediction_diagnostics(raw_scores):
    if raw_scores is None or raw_scores == {}:
        return {"available": False, "valid": False, "normalized_entropy": None,
                "top1": None, "margin": None, "near_uniform": False,
                "decision_eligible": True, "reason": "raw_scores_missing"}
    invalid = {"available": False, "valid": False, "normalized_entropy": None,
               "top1": None, "margin": None, "near_uniform": False,
               "decision_eligible": False, "reason": "invalid_raw_scores"}
    if not isinstance(raw_scores, dict):
        return invalid
    try:
        values = [float(value) for value in raw_scores.values()]
    except (TypeError, ValueError, OverflowError):
        return invalid
    if (len(values) < 2 or any(isinstance(value, bool) for value in raw_scores.values())
            or any(not math.isfinite(value) or value < 0 for value in values)
            or sum(values) <= 0):
        return invalid
    probabilities = sorted((value / sum(values) for value in values), reverse=True)
    entropy = -sum(p * math.log(p) for p in probabilities if p) / math.log(len(values))
    top1, margin = probabilities[0], probabilities[0] - probabilities[1]
    near_uniform = entropy >= .98 and top1 <= 1 / len(values) + .08
    return {"available": True, "valid": True, "normalized_entropy": entropy,
            "top1": top1, "margin": margin, "near_uniform": near_uniform,
            "decision_eligible": not near_uniform,
            "reason": "prediction_near_uniform" if near_uniform else "ok",
            "raw_entropy": entropy, "raw_top1": top1, "raw_margin": margin,
            "diagnostic_version": "raw-dispersion-v1"}


distribution_diagnostics = raw_prediction_diagnostics


def jensen_shannon_divergence(p, q):
    """Base-2 JSD in [0,1]; malformed distributions return None."""
    if not isinstance(p, dict) or not isinstance(q, dict):
        return None
    labels = sorted(set(p) | set(q))
    try:
        left, right = [[float(item.get(key, 0)) for key in labels] for item in (p, q)]
    except (TypeError, ValueError, OverflowError):
        return None
    if (not labels or any(not math.isfinite(v) or v < 0 for v in left + right)
            or not sum(left) or not sum(right)):
        return None
    left, right = [v / sum(left) for v in left], [v / sum(right) for v in right]
    middle = [(a+b)/2 for a, b in zip(left, right)]
    return min(1.0, max(0.0, .5 * sum(v * math.log2(v/m) for v, m in zip(left, middle) if v)
                       + .5 * sum(v * math.log2(v/m) for v, m in zip(right, middle) if v)))


def js_divergence(first, second):
    value = jensen_shannon_divergence(first, second)
    if value is None:
        raise ValueError("distributions must contain finite nonnegative mass")
    return value


@dataclass(frozen=True)
class PromptPolicy:
    required_negative_seconds: float = 20.0
    history_seconds: float = 60.0
    max_gap_seconds: float = 1.5
    min_density: float = .8
    min_quality: float = .5
    negative_threshold: float = .65
    uncertain_threshold: float = .4
    settle_seconds: float = 3.0
    max_result_age_seconds: float = 8.0
    max_items: int = 4096

    def __post_init__(self):
        times = (self.required_negative_seconds, self.history_seconds, self.max_gap_seconds,
                 self.settle_seconds, self.max_result_age_seconds)
        if (any(not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value < 0 for value in times)
                or self.required_negative_seconds <= 0 or self.history_seconds < self.required_negative_seconds):
            raise ValueError("invalid evidence time policy")
        for value in (self.min_density, self.min_quality, self.negative_threshold, self.uncertain_threshold):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("policy fractions must be in [0,1]")
        if not isinstance(self.max_items, int) or isinstance(self.max_items, bool) or self.max_items < 1:
            raise ValueError("max_items must be a positive integer")


class SustainedEvidenceGate:
    """Count negative interval union after a finite settlement delay.

    Higher-specificity joint/audio evidence revises overlapping visual summaries.
    Previously emitted prompts cannot be withdrawn by arbitrarily late results.
    """
    def __init__(self, policy=None, required_seconds=20, min_density=.8, max_gap=1.5,
                 settle_seconds=3, max_age=8, history_seconds=60, max_records=4096):
        if isinstance(policy, PromptPolicy):
            required_seconds, min_density, max_gap = policy.required_negative_seconds, policy.min_density, policy.max_gap_seconds
            settle_seconds, max_age, history_seconds, max_records = policy.settle_seconds, policy.max_result_age_seconds, policy.history_seconds, policy.max_items
        elif policy is not None:
            raise ValueError("policy must be a PromptPolicy")
        if (not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in (required_seconds, min_density, max_gap, settle_seconds, max_age, history_seconds))
                or required_seconds <= 0 or not 0 < min_density <= 1 or max_gap < 0
                or settle_seconds < 0 or max_age <= 0 or history_seconds < required_seconds
                or not isinstance(max_records, int) or isinstance(max_records, bool) or max_records < 1):
            raise ValueError("invalid evidence policy")
        self.required_seconds, self.min_density = float(required_seconds), float(min_density)
        self.max_gap, self.settle_seconds, self.max_age = float(max_gap), float(settle_seconds), float(max_age)
        self.history_seconds, self.max_records = float(history_seconds), max_records
        self.policy = policy or PromptPolicy(required_negative_seconds=required_seconds, min_density=min_density,
                    max_gap_seconds=max_gap, settle_seconds=settle_seconds, max_result_age_seconds=max_age,
                    history_seconds=history_seconds, max_items=max_records)
        self.reset()

    def reset(self, session_id=None):
        self.session_id = session_id
        self._records = {}
        self._revision = 0

    @staticmethod
    def _interval(record):
        try:
            start, end = float(record["t_start"]), float(record["t_end"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        return (start, end) if math.isfinite(start) and math.isfinite(end) and 0 <= start < end else None

    def _qualifies(self, record):
        scores = record.get("scores") or {}
        try:
            values = [float(scores.get(label, 0)) for label in ("positive", "neutral", "negative", "uncertain")]
            if any(not math.isfinite(value) or value < 0 for value in values) or sum(values) <= 0:
                return False
            scores = {label: value/sum(values) for label, value in zip(("positive", "neutral", "negative", "uncertain"), values)}
            return (record.get("valid") is True and float(record.get("quality", 0)) >= self.policy.min_quality
                    and not record.get("conflict") and float(scores.get("uncertain", 1)) < self.policy.uncertain_threshold
                    and float(scores.get("negative", 0)) >= self.policy.negative_threshold
                    and raw_prediction_diagnostics(record.get("raw_scores"))["decision_eligible"])
        except (TypeError, ValueError, OverflowError):
            return False

    def observe(self, result, now):
        if not isinstance(result, dict) or not result.get("session_id"):
            return False
        interval = self._interval(result)
        if (interval is None or not isinstance(now, (int, float)) or not math.isfinite(now)
                or interval[1] > now + 1e-6 or result.get("stale")):
            return False
        if self.session_id != result["session_id"]:
            self.reset(result["session_id"])
        identity = result.get("audio_event_id") or result.get("event_id")
        if not identity:
            return False
        self._revision += 1
        item = deepcopy(result)
        item["_revision"] = self._revision
        self._records[str(identity)] = item
        cutoff = now - self.history_seconds
        self._records = {key: value for key, value in self._records.items() if value["t_end"] > cutoff}
        if len(self._records) > self.max_records:
            self._records = dict(sorted(self._records.items(), key=lambda pair: pair[1]["t_end"])[-self.max_records:])
        return self.ready(now)

    def diagnostics(self, now):
        empty = {"session_id": self.session_id, "effective_seconds": 0.0, "span_seconds": 0.0,
                 "coverage_density": 0.0, "required_seconds": self.required_seconds,
                 "settle_seconds": self.settle_seconds, "settled_until": now-self.settle_seconds,
                 "ready": False, "source_ids": [], "reason": "insufficient_effective_evidence"}
        mature = now - self.settle_seconds
        records = [(item, item["t_start"], min(item["t_end"], mature)) for item in self._records.values()
                   if item["t_start"] < mature and item["t_end"] > now-self.history_seconds]
        if not records:
            return empty
        boundaries = sorted({point for _, a, b in records for point in (a, b)})
        pieces = []
        for a, b in zip(boundaries, boundaries[1:]):
            active = [item for item, left, right in records if left < b and right > a
                      and (item.get("valid") is True or item.get("conflict"))]
            if not active:
                pieces.append((a, b, None, []))
                continue
            def precedence(item):
                return (3 if item.get("status") == "joint" else 2 if item.get("modality") == "fusion"
                        or item.get("status") == "audio_only" else 1, item["_revision"])
            winner = max(active, key=precedence)
            pieces.append((a, b, self._qualifies(winner), winner.get("source_ids") or [winner.get("event_id")]))
        # Explicit contrary evidence breaks a chain; blank intervals only add
        # gaps and never contribute seconds. Consider the latest settled chain.
        chain = []
        for a, b, negative, ids in pieces:
            if negative is False:
                chain = []
            elif negative is True:
                if chain and a-chain[-1][1] > self.max_gap:
                    chain = []
                chain.append((a, b, ids))
        if not chain:
            return empty
        covered = sum(b-a for a, b, _ in chain)
        span = chain[-1][1]-chain[0][0]
        density = covered/span if span else 0
        fresh = now-chain[-1][1] <= self.max_age
        ready = covered + 1e-6 >= self.required_seconds and density + 1e-6 >= self.min_density and fresh
        return {**empty, "effective_seconds": covered, "span_seconds": span,
                "coverage_density": density, "ready": ready,
                "source_ids": list(dict.fromkeys(source for _, _, ids in chain for source in ids if source)),
                "window_start": chain[0][0], "window_end": chain[-1][1],
                "reason": "ready" if ready else "stale_evidence" if not fresh else "insufficient_effective_evidence"}

    def ready(self, now):
        return self.diagnostics(now)["ready"]

    def update(self, result, now):
        if not isinstance(now, (int, float)) or not math.isfinite(now) or now < 0:
            raise ValueError("now must use the nonnegative session-relative clock")
        if not isinstance(result, dict) or not result.get("session_id"):
            return {"should_prompt": False, "reason": "missing_session_id"}
        interval = self._interval(result)
        if interval is None or interval[1] > now + 1e-6:
            return {"should_prompt": False, "reason": "invalid_or_future_interval"}
        self.observe(result, now)
        detail = self.diagnostics(now)
        return {**detail, "should_prompt": detail["ready"], "negative_seconds": detail["effective_seconds"],
                "density": detail["coverage_density"], "t_start": detail.get("window_start"),
                "t_end": detail.get("window_end"), "watermark": detail["settled_until"]}
