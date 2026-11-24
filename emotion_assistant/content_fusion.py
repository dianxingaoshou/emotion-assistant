"""Late fusion for the polarity expressed in an utterance. Never drives a prompt."""

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

from .english_polarity import ASSET as ENGLISH_MODEL

CLASSES = ("negative", "neutral", "positive")
MODALITIES = ("text", "audio", "visual")
MODEL_VERSION = "aligned-utterance-polarity-v1"
ASSET = Path(__file__).parent / "assets" / "content-policy.json"
AV_CAP = 0.25


@lru_cache(maxsize=1)
def deployed_policy():
    """Missing or incompatible selection evidence must not enable a candidate."""
    try:
        doc = json.loads(ASSET.read_text("utf-8"))
        if (
            doc["model_version"] == MODEL_VERSION
            and doc["av_cap"] == AV_CAP
            and doc["policy"] == "blend"
            and doc["english_classifier_sha256"] == hashlib.sha256(ENGLISH_MODEL.read_bytes()).hexdigest()
        ):
            return "blend"
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return "text_first"


def _unit(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _evidence(value, is_text):
    if not isinstance(value, dict) or not isinstance(value.get("scores"), dict):
        return None
    scores = value["scores"]
    quality, coverage = value.get("quality", 1.0), value.get("coverage", 1.0)
    mapped = [scores.get(key) for key in CLASSES]
    unknown = scores.get("uncertain", 0.0)
    if not all(_unit(x) for x in (*mapped, unknown, quality, coverage)):
        return None
    mass = sum(mapped)
    if abs(mass + unknown - 1.0) > 0.025 or mass <= 0 or quality * coverage <= 0:
        return None
    if unknown >= max(mapped) or (is_text and max(mapped) / mass < 0.4):
        return None
    return ({key: value / mass for key, value in zip(CLASSES, mapped)}, quality * coverage * mass)


def _weighted_scores(branches, weights):
    return {k: sum(weights[m] * branches[m][0][k] for m in weights) for k in CLASSES}


def _accepted_label(scores):
    if not scores:
        return None
    leader = max(scores, key=scores.get)
    return leader if scores[leader] >= 0.4 else None


def _set_candidate_estimate(result, scores, weights):
    label = _accepted_label(scores)
    if label is not None:
        result.update(
            candidate_label=label,
            candidate_decision="estimate",
            candidate_estimate_scores=scores,
            candidate_estimate_weights=weights,
        )


def _fuse(text, visual, audio, policy):
    inputs = {"text": text, "audio": audio, "visual": visual}
    branches = {}
    for modality, value in inputs.items():
        branch = _evidence(value, modality == "text")
        if branch:
            branches[modality] = branch
    leaders = {max(scores, key=scores.get) for scores, _ in branches.values()}
    av = {m: branches[m][1] for m in ("audio", "visual") if m in branches}
    result = {
        "model_version": MODEL_VERSION,
        "policy": policy,
        "label": None,
        "scores": None,
        "decision": "abstain",
        "status": "no_evidence",
        "available_modalities": list(branches),
        "used_modalities": [],
        "weights": {},
        "candidate_weights": {},
        "candidate_scores": None,
        "candidate_label": None,
        "candidate_decision": "abstain",
        "candidate_estimate_scores": None,
        "candidate_estimate_weights": {},
        "disagreement": len(leaders) > 1,
    }
    if not branches:
        return result
    av_mass = sum(av.values())
    auxiliary_share = AV_CAP * min(1.0, av_mass / 2.0) if "text" in branches else 1.0
    candidate_weights = {m: auxiliary_share * strength / av_mass for m, strength in av.items()}
    if "text" in branches:
        candidate_weights["text"] = 1.0 - auxiliary_share
    candidate_scores = _weighted_scores(branches, candidate_weights)
    result.update(candidate_scores=candidate_scores, candidate_weights=candidate_weights)
    if "text" not in branches:
        result["status"] = "no_text"
        return result
    _set_candidate_estimate(result, candidate_scores, candidate_weights)
    if not av:
        status, weights = "text_only", {"text": 1.0}
    elif policy == "blend":
        status, weights = "multimodal", candidate_weights
    else:
        status, weights = "text_preferred", {"text": 1.0}
    scores = _weighted_scores(branches, weights)
    label = _accepted_label(scores)
    if label is None:
        result["status"] = "low_confidence"
        return result
    result.update(
        scores=scores,
        label=label,
        decision="estimate",
        status=status,
        used_modalities=list(weights),
        weights=weights,
    )
    return result


def fuse_content(*, text=None, visual=None, audio=None):
    """Score one aligned interval. Four-way A/V cues are evidence, never implicit text."""
    return _fuse(text, visual, audio, deployed_policy())


def _average(rows, key):
    total = sum(seconds for seconds, _ in rows)
    if total <= 0:
        return None
    aggregate = {}
    for seconds, row in rows:
        for name, value in row[key].items():
            aggregate[name] = aggregate.get(name, 0.0) + value * seconds / total
    return aggregate


def summarize_content(rows):
    """Combine nonoverlapping interval results; avoid treating uncovered time as evidence.

    When some text exists, the joint candidate summarizes text-supported windows
    only. In an A/V-only file the candidate instead describes the A/V proxy.
    """
    result = _fuse(None, None, None, deployed_policy())
    valid_windows = []
    end = None
    for row in sorted(rows, key=lambda item: item["t_start"]):
        start, stop = row["t_start"], row["t_end"]
        if not all(isinstance(x, (float, int)) and math.isfinite(x) for x in (start, stop)) or stop <= start:
            raise ValueError("content intervals require finite increasing bounds")
        if end is not None and start < end - 1e-6:
            raise ValueError("overlapping content intervals must be aligned before aggregation")
        valid_windows.append((stop - start, row))
        end = stop
    scored = [
        (dur, row) for dur, row in valid_windows if row.get("decision") == "estimate" and row.get("scores")
    ]
    # Intensity is a separately fitted regression. A neutral value or a
    # polarity abstention can both carry a valid regression estimate.
    intensities = [(dur, row["intensity"]) for dur, row in valid_windows if row.get("intensity") is not None]
    intensity_seconds = sum(dur for dur, _ in intensities)
    intensity = None
    if intensities:
        value = sum(dur * estimate["value"] for dur, estimate in intensities) / intensity_seconds
        intensity = {**intensities[0][1], "value": min(3.0, max(-3.0, value))}
    has_text = any("text" in row["available_modalities"] for _, row in valid_windows)
    candidates = [
        (dur, row)
        for dur, row in valid_windows
        if row.get("candidate_scores") and (not has_text or "text" in row["available_modalities"])
    ]
    # Keep the diagnostic average above for A/V proxies. A benchmark-equivalent
    # candidate uses only text-bearing windows accepted by the blend policy.
    accepted_candidates = [
        (dur, row)
        for dur, row in candidates
        if "text" in row["available_modalities"] and _accepted_label(row["candidate_scores"])
    ]
    _set_candidate_estimate(
        result,
        _average(accepted_candidates, "candidate_scores"),
        _average(accepted_candidates, "candidate_weights"),
    )
    result.update(
        available_modalities=[
            m for m in MODALITIES if any(m in row["available_modalities"] for _, row in valid_windows)
        ],
        disagreement=any(row["disagreement"] for _, row in valid_windows),
        weights=_average(scored, "weights") or {},
        candidate_weights=_average(candidates, "candidate_weights") or {},
        candidate_scores=_average(candidates, "candidate_scores"),
        scores=_average(scored, "scores"),
        interval_count=len(valid_windows),
        covered_seconds=sum(d for d, _ in scored),
        candidate_seconds=sum(d for d, _ in candidates),
        candidate_estimate_seconds=sum(d for d, _ in accepted_candidates),
        observed_seconds=sum(d for d, _ in valid_windows),
        intensity=intensity,
        intensity_covered_seconds=intensity_seconds,
    )
    if valid_windows:
        result["policy"] = valid_windows[0][1]["policy"]
    if scored:
        top = _accepted_label(result["scores"])
        if top is None:
            result.update(scores=None, weights={}, status="low_confidence")
            return result
        result.update(used_modalities=list(result["weights"]), label=top, decision="estimate")
        statuses = {row["status"] for _, row in scored}
        result["status"] = next(s for s in ("multimodal", "text_preferred", "text_only") if s in statuses)
    elif has_text:
        result["status"] = "low_confidence"
    elif candidates:
        result["status"] = "no_text"
    return result
