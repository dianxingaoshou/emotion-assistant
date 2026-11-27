"""Fuse frozen CH-SIMS experts with trained probability calibration and reliability weights."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np


ASSET = Path(__file__).with_name("assets") / "chsims-adaptive-v2.json"
VERSION = "chsims-adaptive-v2"
HEAD_SHA256 = "f7c5ae8fe83399abe95a53099a5087c16aed6f47048eae2d713a2e7c78a23393"
LABELS = ("negative", "neutral", "positive")


def validate(parameters):
    if (parameters["version"] != VERSION or parameters["head_sha256"] != HEAD_SHA256
            or tuple(parameters["labels"]) != LABELS):
        raise ValueError("Unknown adaptive classifier")
    for name in ("foundation", "no_qwen", "av"):
        policy = parameters["policies"][name]
        gate = np.asarray(policy["gate"], dtype=float)
        if gate.shape != (4, 3) or not np.isfinite(gate).all():
            raise ValueError("Invalid reliability model")
        if len(policy["calibration"]) != 4:
            raise ValueError("Expected four expert slots")
        for j, calibration in enumerate(policy["calibration"]):
            if j == 0 and name == "av" and calibration is None:
                continue
            for key, shape in (("mean", (3,)), ("scale", (3,)), ("coef", (3, 3)), ("intercept", (3,))):
                values = np.asarray(calibration[key], dtype=float)
                if values.shape != shape or not np.isfinite(values).all():
                    raise ValueError("Invalid expert calibration")
                if key == "scale" and (values <= 0.).any():
                    raise ValueError("Invalid calibration scale")


@lru_cache(maxsize=1)
def adaptive_parameters():
    from .foundation_fusion import ASSET as expert_head
    try:
        params = json.loads(ASSET.read_text(encoding="utf8"))
        validate(params)
        with expert_head.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != params["head_sha256"]:
                return None
        return params
    except (OSError, TypeError, ValueError, KeyError, OverflowError):
        return None


def calibration_features(probs):
    logp = np.log(np.clip(probs, 1e-5, 1.))
    neutral = logp[..., 1] - logp[..., 0]
    positive = logp[..., 2] - logp[..., 0]
    return np.stack((neutral, positive, np.abs(positive)), axis=-1)


def calibrate(values, quality, calibration):
    """Batch interface shared by training, validation and inference."""
    output = np.zeros_like(values, dtype=float)
    for j, params in enumerate(calibration):
        use = quality[:, j] > 0.
        if params is None or not use.any():
            continue
        x = (calibration_features(values[use, j]) - params["mean"]) / params["scale"]
        logits = x @ np.asarray(params["coef"]).T + params["intercept"]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        output[use, j] = exp / exp.sum(axis=1, keepdims=True)
    return output


def gate_features(probs, quality):
    """A margin measures strength. Agreement counts only other available experts."""
    ranked = np.sort(probs, axis=-1)
    margin = ranked[..., -1] - ranked[..., -2]
    labels = probs.argmax(axis=-1)
    matches = labels[:, :, None] == labels[:, None, :]
    other = ~np.eye(4, dtype=bool)[None, :, :]
    masses = other * quality[:, None, :]
    totals = masses.sum(axis=-1)
    agreement = np.divide((matches * masses).sum(axis=-1), totals, where=totals > 0.,
                          out=np.zeros_like(totals))
    return np.stack((np.ones_like(quality), margin, agreement), axis=-1)


def gate_weights(features, quality, gate):
    logits = np.einsum("nej,ej->ne", features, gate)
    # Unavailable experts have zero mass. An all-unavailable sample remains zero.
    logits = np.where(quality > 0., logits, -np.inf)
    maximum = logits.max(axis=1, keepdims=True)
    mass = quality * np.exp(logits - np.where(np.isfinite(maximum), maximum, 0.))
    total = mass.sum(axis=1, keepdims=True)
    return np.divide(mass, total, out=np.zeros_like(mass), where=total > 0.)


def blend(parameters, variant, probabilities, quality):
    """Return class probabilities, normalized expert weights and the calibrated evidence."""
    weights = np.zeros(4)
    details = {"expert_scores": [None] * 4}
    if parameters is None:
        return None, weights, details
    values = np.zeros((1, 4, 3))
    available = np.zeros((1, 4))
    for j, (p, q) in enumerate(zip(probabilities, quality, strict=True)):
        if p is None or not np.isfinite(q) or not 0 < q <= 1:
            continue
        p = np.asarray(p, dtype=float)
        if p.shape == (3,) and np.isfinite(p).all() and (p >= 0.).all() and np.isclose(p.sum(), 1.):
            values[0, j] = p
            available[0, j] = q
    if not available.any():
        return None, weights, details
    policy_name = variant if available[0, 0] else "av"
    policy = parameters["policies"][policy_name]
    calibrated = calibrate(values, available, policy["calibration"])
    features = gate_features(calibrated, available)
    weights = gate_weights(features, available, policy["gate"])[0]
    scores = weights @ calibrated[0]
    details["expert_scores"] = [p.tolist() if w > 0. else None for p, w in zip(calibrated[0], weights)]
    details["policy"] = policy_name
    return scores, weights, details
