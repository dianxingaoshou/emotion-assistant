"""Offline regression of sentiment expressed in English words, on the signed MOSEI scale."""

from collections import Counter
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import zlib

from .english_polarity import terms, tokens

ASSET = Path(__file__).parent / "assets" / "english-intensity.json.gz"
SCHEMA = "en-intensity-tfidf-ridge-v1"


def version_of(model):
    payload = {k: v for k, v in model.items() if k != "model_version"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return SCHEMA + "-" + hashlib.sha256(canonical).hexdigest()[:16]


@lru_cache(maxsize=1)
def _bundled():
    try:
        with gzip.open(ASSET, "rt", encoding="utf-8") as stream:
            model = json.load(stream)
        if model["schema"] == SCHEMA and model["model_version"] == version_of(model):
            return model
    except (OSError, KeyError, TypeError, ValueError, EOFError, zlib.error):
        pass
    return None


def predict_segments_with_model(segments, model):
    """A removed span separates surviving word pairs. Never form a bigram across the gap."""
    if not segments or not all(isinstance(s, str) for s in segments):
        return None
    words = tokens(" ".join(segments))
    vocab = model["terms"]
    if len(words) < 2 or sum(w in vocab for w in words) < 0.25 * len(words):
        return None
    counts = Counter()
    for segment in segments:
        counts.update(terms(tokens(segment)))
    numerator = 0.0
    squared_norm = 0.0
    for key, n in counts.items():
        if key in vocab:
            idf, coefficient = vocab[key]
            weight = (1 + math.log(n)) * idf
            numerator += weight * coefficient
            squared_norm += weight**2
    if not squared_norm:
        return None
    value = model["intercept"] + numerator / math.sqrt(squared_norm)
    return {"value": min(3.0, max(-3.0, value)), "model_version": model["model_version"],
            "used_modalities": ["text"]}


def predict_english_intensity_segments(segments):
    """Score the surviving parts of one utterance with known gaps between them."""
    model = _bundled()
    return predict_segments_with_model(segments, model) if model is not None else None


def predict_english_intensity(text):
    """Return a signed intensity estimate, or None for unsupported/insufficient words."""
    return predict_english_intensity_segments([text])
