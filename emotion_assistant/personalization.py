"""Evidence-linked subjective feedback and frozen, descriptive personal references."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from statistics import median
import uuid


FEEDBACK_LABELS = {
    "subjective_correction": {"positive", "neutral", "negative", "uncertain"},
    "distress_report": {"distressed"},
    "do_not_disturb": {"busy"},
    "typical_reference": {"usual"},
}
MIN_REFERENCES = 5
MIN_SESSIONS = 2
MAX_PER_GROUP = 120


def _digest(value):
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode()).hexdigest()


@dataclass(frozen=True)
class FeedbackRecord:
    feedback_id: str
    user_id: str
    session_id: str
    feedback_type: str
    user_label: str
    learning_consent: bool
    target_event_ids: list
    window_start: float | None
    window_end: float | None
    model_versions: dict
    fusion_config_hash: str | None
    evidence: dict

    @classmethod
    def create(cls, user_id, session_id, kind, label, allowed, result, now):
        if label not in FEEDBACK_LABELS.get(kind, set()):
            raise ValueError("请选择有效的反馈类型和感受")
        result = result if result and result.get("session_id") == session_id else {}
        keys = ("event_id", "source_ids", "revision", "status", "scores", "quality", "valid",
                "conflict", "modalities", "modality_intervals", "model_versions", "fusion_basis_hash",
                "reference_context", "reference_observing", "t_start", "t_end", "decision",
                "per_modality_quality")
        evidence = {key: result[key] for key in keys if key in result}
        end = result.get("t_end")
        evidence["fresh_at_feedback"] = isinstance(end, (int, float)) and 0 <= now - end <= 8
        targets = list(dict.fromkeys([result["event_id"], *result.get("source_ids", [])])) if result.get("event_id") else []
        return cls(uuid.uuid4().hex, user_id, session_id, kind, label, allowed is True,
                   targets, result.get("t_start"), end, result.get("model_versions", {}),
                   result.get("fusion_config_hash"), evidence)

    def payload(self):
        return asdict(self)


def _covered_fraction(intervals, start, end):
    pieces = sorted((max(start, left), min(end, right)) for left, right in intervals)
    covered, cursor = 0., start
    for left, right in pieces:
        covered += max(0., right - max(cursor, left))
        cursor = max(cursor, right)
    return covered / (end-start)


def comparison_key(event):
    """Require complete configured inputs and stable device/model/quality conditions."""
    context = event.get("reference_context") or {}
    modes = sorted(context.get("enabled_modalities", []))
    versions = event.get("model_versions") or {}
    if (not modes or modes != sorted(event.get("modalities", []))
            or modes != sorted(versions) or any(not versions[m] for m in modes)
            or not event.get("fusion_basis_hash") or not event.get("source_ids")
            or not event.get("valid") or event.get("conflict") or event.get("decision") == "abstain"
            or event.get("reference_observing") is not True):
        return None
    try:
        quality = float(event["quality"])
        start, end = event["t_start"], event["t_end"]
        score = float(event["scores"]["negative"])
        qualities = event.get("per_modality_quality") or {m: quality for m in modes}
        if (not .65 <= quality <= 1 or not 0 <= score <= 1 or not 0 <= start < end
                or not math.isfinite(end)
                or any(not .65 <= qualities[m] <= 1 for m in modes)
                or any(_covered_fraction(event["modality_intervals"][m], start, end) < .8 for m in modes)):
            return None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return _digest([context, modes, versions, event["fusion_basis_hash"],
                    "high" if round(quality, 6) >= .85 else "moderate"])


def _quantile(values, fraction):
    position = (len(values)-1)*fraction
    low = int(position)
    return values[low] + (values[min(low+1, len(values)-1)] - values[low])*(position-low)


def matches_final_revision(record, final):
    evidence = record["evidence"]
    key = comparison_key(evidence)
    return bool(key and key == comparison_key(final)
                and evidence.get("revision") == final.get("revision")
                and evidence["source_ids"] == final.get("source_ids")
                and evidence["scores"] == final.get("scores"))


def build_reference(records):
    """Build only from past closed sessions. Caller supplies rows newest first."""
    groups = defaultdict(list)
    for record in records:
        evidence = record["evidence"]
        key = comparison_key(evidence)
        if (not key or record["feedback_type"] != "typical_reference"
                or not record["learning_consent"] or not evidence.get("fresh_at_feedback")):
            continue
        samples = groups[key]
        if len(samples) >= MAX_PER_GROUP:
            continue
        if any(old["session_id"] == record["session_id"] and
               ((old["window_start"] < record["window_end"] and record["window_start"] < old["window_end"])
                or not set(old["evidence"]["source_ids"]).isdisjoint(evidence["source_ids"]))
               for old in samples):
            continue
        samples.append(record)
    summary = {}
    for key, records_for_group in groups.items():
        values = sorted(r["evidence"]["scores"]["negative"] for r in records_for_group)
        center = median(values)
        sessions = len({r["session_id"] for r in records_for_group})
        evidence = records_for_group[0]["evidence"]
        summary[key] = {
            "count": len(values), "sessions": sessions, "median": center,
            "q25": _quantile(values, .25), "q75": _quantile(values, .75),
            "mad": median(abs(v-center) for v in values),
            "ready": len(values) >= MIN_REFERENCES and sessions >= MIN_SESSIONS,
            "modalities": evidence["modalities"], "model_versions": evidence["model_versions"],
            "capture_context": evidence["reference_context"],
            "feedback_ids": [r["feedback_id"] for r in records_for_group],
        }
    return {"version": _digest(["personal-reference-v1", summary]) if summary else None, "groups": summary}


def compare_reference(snapshot, event):
    group = snapshot["groups"].get(comparison_key(event))
    if not group or not group["ready"]:
        return {"status": "insufficient", "version": snapshot["version"]}
    score = event["scores"]["negative"]
    above = score > group["q75"] + max(.05, group["mad"])
    return {"status": "above_reference" if above else "within_or_below_reference",
            "version": snapshot["version"], "median": group["median"], "count": group["count"],
            "message": ("当前负向线索高于你确认的近期同类参考。" if above else "当前负向线索位于个人参考范围内或以下。")}
