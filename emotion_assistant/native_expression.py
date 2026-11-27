"""Rank expression classes within one native model, preserving residual categories."""
from __future__ import annotations

from math import isfinite

from .evidence import raw_prediction_diagnostics


NAMES = {"anger": "愤怒", "contempt": "轻蔑", "disgust": "厌恶", "fear": "恐惧",
         "happiness": "愉快", "neutral": "中性", "sadness": "悲伤", "surprise": "惊讶",
         "other": "其他", "unknown": "未知"}
ALIASES = {"angry": "anger", "disgusted": "disgust", "fearful": "fear", "happy": "happiness",
           "sad": "sadness", "surprised": "surprise", "<unk>": "unknown"}
MODALITIES = {"visual": "表情", "audio": "语调"}
CLASS_SETS = {"visual": set(NAMES) - {"other", "unknown"}, "audio": set(NAMES) - {"contempt"}}


def _score_vector(observation, modality):
    raw = observation.get("raw_scores")
    quality = observation.get("quality")
    if (observation.get("modality") != modality or observation.get("valid") is not True
            or observation.get("eligible") is False or isinstance(quality, bool)
            or not isinstance(quality, (float, int)) or not .2 <= quality <= 1
            or not isinstance(raw, dict) or not raw):
        return None
    scores = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            return None
        key = key.split("/")[-1].strip().lower()
        key = ALIASES.get(key, key)
        if (key not in NAMES or key in scores or isinstance(value, bool)
                or not isinstance(value, (float, int)) or not isfinite(value) or value < 0):
            return None
        scores[key] = value
    total = sum(scores.values())
    if (scores.keys() != CLASS_SETS[modality] or not .98 <= total <= 1.02
            or not raw_prediction_diagnostics(scores)["decision_eligible"]):
        return None
    return {key: value / total for key, value in scores.items()}


def expression_profile(observations, modality, *, still=False):
    """Average normalized within-model scores using quality × valid interval duration."""
    if modality not in MODALITIES:
        raise ValueError("Unsupported expression modality")
    sums = {}
    count, mass = 0, 0.
    for obs in observations:
        scores = _score_vector(obs, modality)
        if not scores:
            continue
        duration = 1. if still else obs["t_end"] - obs["t_start"]
        if not isfinite(duration) or duration <= 0:
            continue
        weight = obs["quality"] * duration
        count += 1
        mass += weight
        for key, value in scores.items():
            sums[key] = sums.get(key, 0.) + value * weight
    result = {"modality": modality, "status": "unavailable", "eligible_observations": count,
              "aggregation": "single_observation" if still else "duration_times_quality",
              "scores": {}, "top_two": [], "margin": None}
    if not mass:
        return result
    scores = {key: value / mass for key, value in sums.items()}
    result["scores"] = scores
    diagnostics = raw_prediction_diagnostics(scores)
    if not diagnostics["decision_eligible"]:
        result["status"] = "ambiguous"
        return result
    ranked = sorted(scores, key=lambda key: (-scores[key], key))
    if scores[ranked[0]] == scores[ranked[1]]:
        result["status"] = "ambiguous"
        return result
    result.update(status="ok", top_two=ranked[:2], margin=diagnostics["margin"])
    return result


def _percentage_points(value):
    if .999 < value < 1:
        return ">99.9"
    if 0 < value < .001:
        return "<0.1"
    return f"{100 * value:.1f}"


def format_expression(profile):
    prefix = MODALITIES[profile["modality"]]
    if profile["status"] != "ok":
        reason = "类别分数难以区分" if profile["status"] == "ambiguous" else "缺少有效原始分数"
        return f"{prefix}细分类暂缓 · {reason}"
    first, second = profile["top_two"]
    scores = profile["scores"]
    residual = "　具体类别未定" if first in ("other", "unknown") else ""
    return (f"{prefix}细分类（模型分）　{NAMES[first]} {_percentage_points(scores[first])}%、"
            f"{NAMES[second]} {_percentage_points(scores[second])}%"
            f"　差 {_percentage_points(profile['margin'])} 个百分点" + residual)
