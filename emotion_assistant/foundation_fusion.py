"""Clip-level CH-SIMS expression classifier. Runtime inference requires only NumPy."""

from functools import lru_cache
from pathlib import Path
import unicodedata
import zlib
from zipfile import BadZipFile

import numpy as np

from .chinese_affect import HAN
from .foundation_embeddings import AUDIO_DIM, TEXT_DIM, TEXT_LIMIT


ASSET = Path(__file__).with_name("assets") / "chsims-foundation-v1.npz"
VERSION = "chsims-foundation-v1"
LABELS = ("negative", "neutral", "positive")
EXPERTS = ("text", "audio_embedding", "audio_scores", "visual_scores")


def lexical_view(text, salt):
    """Two independent, length-normalized projections of signed character n-gram counts."""
    if not isinstance(text, str):
        return None
    text = text.strip()[:TEXT_LIMIT]
    if not HAN.search(text):
        return None
    words = "^" + "".join(unicodedata.normalize("NFKC", text).lower().split()) + "$"
    vector = np.zeros(TEXT_DIM, dtype=np.float64)
    for n in (1, 2, 3):
        for i in range(len(words) - n + 1):
            checksum = zlib.crc32(words[i:i+n].encode("utf-8"), salt)
            checksum = ((checksum ^ (checksum >> 16)) * 0x85ebca6b) & 0xffffffff
            checksum ^= checksum >> 13
            vector[checksum % TEXT_DIM] += 1 if checksum & 0x80000000 else -1
    return unit(vector, TEXT_DIM)


def unit(vector, size):
    if vector is None:
        return None
    try:
        value = np.asarray(vector, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if value.shape != (size,) or not np.isfinite(value).all():
        return None
    norm = np.linalg.norm(value)
    return value / norm if np.isfinite(norm) and norm > 1e-10 else None


def text_input(text, embedding, with_qwen):
    first = lexical_view(text, 0)
    if first is None:
        return None
    second = unit(embedding, TEXT_DIM) if with_qwen else lexical_view(text, 0x6c8e9cf5)
    return np.concatenate((first, second)) if second is not None else None


def probabilities(logits):
    centered = np.asarray(logits) - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(centered)
    return exp / exp.sum(axis=-1, keepdims=True)


def learned_scores(asset, key, vector):
    if vector is None or key + "_coef" not in asset:
        return None
    centered = vector - asset[key + "_center"]
    norm = np.linalg.norm(centered, axis=-1, keepdims=True)
    normalized = centered / np.maximum(norm, 1e-10)
    return probabilities(normalized @ asset[key + "_coef"].T + asset[key + "_bias"])


def cue_scores(cue):
    if not isinstance(cue, dict) or not isinstance(cue.get("scores"), dict):
        return None, 0.
    try:
        scores = np.array([cue["scores"][k] for k in LABELS], dtype=float)
        unknown = float(cue["scores"].get("uncertain", 0.))
        quality = float(cue["quality"])
        coverage = float(cue["coverage"])
    except (ValueError, TypeError, KeyError):
        return None, 0.
    if (not (0 < quality <= 1 and 0 < coverage <= 1 and 0 <= unknown <= 1)
            or not np.isfinite(scores).all() or np.any(scores < 0.)
            or abs(scores.sum() + unknown - 1.) > .03 or unknown >= scores.max()):
        return None, 0.
    mass = scores.sum()
    return scores / mass, quality * coverage * mass


@lru_cache(maxsize=1)
def bundled_head():
    try:
        with np.load(ASSET, allow_pickle=False) as archive:
            result = {k: archive[k] for k in archive.files}
        if result["version"].item() != VERSION or tuple(result["labels"]) != LABELS:
            return None
        for name, values in result.items():
            if name not in ("version", "labels") and (values.dtype.kind != "f" or not np.isfinite(values).all()):
                return None
        for key, dim in (("base", TEXT_DIM * 2), ("qwen", TEXT_DIM * 2), ("audio", AUDIO_DIM)):
            if (result[key + "_coef"].shape != (3, dim) or result[key + "_bias"].shape != (3,)
                    or result[key + "_center"].shape != (dim,)):
                return None
        for variant in ("base", "no_qwen", "foundation"):
            weights = result[variant + "_weights"]
            if weights.shape != (4,) or np.any(weights < 0) or not weights.any():
                return None
        return result
    except (TypeError, ValueError, EOFError, KeyError, OSError, BadZipFile):
        return None


def predict_with_head(asset, text, *, audio=None, visual=None, embedding=None, audio_embedding=None,
                      variant="foundation", adaptive=None):
    version = VERSION if adaptive is None else "chsims-adaptive-v2"
    base = {"version": version, "status": "insufficient_evidence", "label": None, "scores": {},
            "used_modalities": [], "foundation_used": False, "audio_embedding_used": False, "experts": {}}
    if asset is None:
        return {**base, "status": "head_unavailable"}
    use_qwen = variant == "foundation" and unit(embedding, TEXT_DIM) is not None
    text_key = "qwen" if use_qwen else "base"
    vector = text_input(text, embedding, use_qwen)
    text_probs = learned_scores(asset, text_key, vector)
    audio_probs = learned_scores(asset, "audio", unit(audio_embedding, AUDIO_DIM)) if variant != "base" else None
    acoustic_cue, acoustic_quality = cue_scores(audio)
    visual_cue, visual_quality = cue_scores(visual)
    experts = (text_probs, audio_probs, acoustic_cue, visual_cue)
    quality = (1., 1., acoustic_quality, visual_quality)
    # When text is available but the semantic encoder is offline, use the matched character head.
    policy = "no_qwen" if variant == "foundation" and text_probs is not None and not use_qwen else variant
    if adaptive is None:
        weights = asset[policy + "_weights"] * np.array(
            [q if p is not None else 0. for p, q in zip(experts, quality)])
        if weights.sum() <= 0:
            return base
        weights /= weights.sum()
        mixed = sum(w * p for w, p in zip(weights, experts) if w > 0.)
        calibrated = experts
    else:
        from .adaptive_clip_fusion import blend
        mixed, weights, details = blend(adaptive, policy, experts, quality)
        if mixed is None:
            return base
        calibrated = details["expert_scores"]
    modes = []
    trace = {}
    for name, mode, raw, value, weight in zip(EXPERTS, ("text", "audio", "audio", "visual"),
                                               experts, calibrated, weights):
        if weight > 0:
            trace[name] = {"weight": float(weight), "label": LABELS[int(np.argmax(value))]}
            if adaptive is not None:
                trace[name]["raw_label"] = LABELS[int(np.argmax(raw))]
                trace[name]["scores"] = dict(zip(LABELS, map(float, value)))
            if mode not in modes:
                modes.append(mode)
    return {**base, "status": "ok", "label": LABELS[int(np.argmax(mixed))],
            "scores": dict(zip(LABELS, mixed.tolist())), "used_modalities": modes, "experts": trace,
            "foundation_used": bool(use_qwen and weights[0] > 0),
            "audio_embedding_used": bool(weights[1] > 0)}


def predict_foundation_clip(text, *, audio=None, visual=None, embedding=None, audio_embedding=None,
                            adaptive=False):
    head = bundled_head()
    parameters = None
    if adaptive:
        from .adaptive_clip_fusion import adaptive_parameters
        parameters = adaptive_parameters()
        if parameters is None:
            return {"version": "chsims-adaptive-v2", "status": "head_unavailable", "label": None,
                    "scores": {}, "used_modalities": [], "foundation_used": False,
                    "audio_embedding_used": False, "experts": {}}
    return predict_with_head(head, text, audio=audio, visual=visual, embedding=embedding,
                             audio_embedding=audio_embedding, adaptive=parameters)
