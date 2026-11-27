"""Offline reader-perceived valence/arousal and optional local Qwen combination."""

from collections import Counter
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import zlib
from io import BytesIO

import numpy as np

from .chinese_affect import AXES, HAN, is_chinese_text
from .chinese_affect_calibration import FITTED_MODEL, FITTED_RATER, VERSION as QWEN_CALIBRATION, apply_chinese_calibration
from .chinese_emotions import char_ngrams, normalized_text


MODEL = "cvas-zh-char-ridge-v1"
HYBRID = "cvas-zh-char-qwen-mixture-v1"
BUCKETS = 65536
ASSET = Path(__file__).with_name("assets") / "chinese-dimensions-v1.npz"
ASSET_SHA256 = "e5e73aa857d413a336f7c95b42891b25c5a820a6a7b47d5367e274de7bed7f0a"


def feature_counts(normalized):
    return Counter(zlib.crc32(term.encode("utf-8")) % BUCKETS for term in char_ngrams(normalized))


@lru_cache(maxsize=1)
def bundled_model():
    try:
        raw = ASSET.read_bytes()
        if hashlib.sha256(raw).hexdigest() != ASSET_SHA256:
            return None
        with np.load(BytesIO(raw), allow_pickle=False) as weights:
            data = {name: weights[name] for name in ("known_chars", "trad", "simp", "weights", "idf", "intercept", "mixture")}
        data["translation"] = dict(zip(data.pop("trad").tolist(), data.pop("simp").tolist()))
        data["known"] = frozenset(data.pop("known_chars").tolist())
        return data
    except (OSError, ValueError, KeyError):
        return None


def evidence(normalized, known):
    characters = HAN.findall(normalized)
    matched = sum(ord(ch) in known for ch in characters)
    return matched >= 2 and 2 * matched >= len(characters), {
        "han_characters": len(characters), "known_characters": matched,
    }


def score_normalized(normalized, model):
    counts = feature_counts(normalized)
    columns = np.fromiter((bucket for bucket in counts if model["idf"][bucket] > 0), dtype=int)
    if not len(columns):
        return model["intercept"].copy()
    weights = np.array([(1 + math.log(counts[i])) * model["idf"][i] for i in columns])
    return np.clip(model["intercept"] + model["weights"][:, columns] @ (weights / np.linalg.norm(weights)), 1, 9)


def predict_chinese_dimensions(text):
    base = {"model": MODEL, "scope": "expressed_text", "scale": [1, 9], "dimensions": {}}
    if not isinstance(text, str):
        return {**base, "status": "unsupported_language"}
    if len(text) > 512:
        return {**base, "status": "too_long"}
    if not is_chinese_text(text):
        return {**base, "status": "unsupported_language"}
    model = bundled_model()
    if model is None:
        return {**base, "status": "model_unavailable"}
    normalized = normalized_text(text, model["translation"])
    supported, details = evidence(normalized, model["known"])
    if not supported:
        return {**base, "status": "insufficient_evidence", "evidence": details}
    scores = score_normalized(normalized, model)
    return {**base, "status": "ok", "evidence": details,
            "dimensions": {axis: {"score": round(float(value), 4)} for axis, value in zip(AXES, scores)}}


def blend_chinese_dimensions(offline, calibrated_qwen):
    """A separate estimate, made only when both source-specific models have valid ratings."""
    base = {"model": HYBRID, "scope": "expressed_text", "scale": [1, 9], "dimensions": {}}
    if (not isinstance(offline, dict) or not isinstance(calibrated_qwen, dict)
            or offline.get("model") != MODEL or offline.get("scope") != "expressed_text"
            or offline.get("scale") != [1, 9] or offline.get("status") != "ok"
            or calibrated_qwen.get("status") != "ok" or calibrated_qwen.get("model") != FITTED_MODEL
            or calibrated_qwen.get("version") != FITTED_RATER or calibrated_qwen.get("scope") != "expressed_text"
            or calibrated_qwen.get("calibration_version") != QWEN_CALIBRATION):
        return {**base, "status": "insufficient_evidence"}
    if any(type(calibrated_qwen.get(axis)) is not int or not 1 <= calibrated_qwen[axis] <= 9 for axis in AXES):
        return {**base, "status": "insufficient_evidence"}
    expected = apply_chinese_calibration(calibrated_qwen)["calibrated"]
    model = bundled_model()
    if model is None:
        return {**base, "status": "model_unavailable"}
    dimensions = {}
    for axis, weight in zip(AXES, model["mixture"]):
        try:
            lexical = offline["dimensions"][axis]["score"]
            semantic = calibrated_qwen["calibrated"][axis]
        except (KeyError, TypeError):
            return {**base, "status": "insufficient_evidence"}
        if (any(type(value) not in (int, float) or not 1 <= value <= 9 for value in (lexical, semantic))
                or not math.isclose(semantic, expected[axis], abs_tol=1e-7)):
            return {**base, "status": "insufficient_evidence"}
        dimensions[axis] = {"score": round(float((1 - weight) * lexical + weight * semantic), 4),
                            "qwen_weight": round(float(weight), 4)}
    return {**base, "status": "ok", "dimensions": dimensions}
