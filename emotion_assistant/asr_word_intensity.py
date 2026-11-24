"""Local ASR-specific estimator. Word evidence and within-word character evidence share the score."""
from collections import Counter
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import zlib

from .english_intensity import predict_segments_with_model
from .english_polarity import tokens

ASSET = Path(__file__).parent / "assets/asr-character-intensity.json.gz"
SCHEMA = "en-asr-within-word-char-ridge-v1"


def char_terms(text):
    for word in tokens(text):
        padded = " " + word + " "
        for n in (3, 4, 5):
            for i in range(len(padded) - n + 1):
                yield padded[i:i+n]


def version_of(model):
    body = {k: v for k, v in model.items() if k != "model_version"}
    return SCHEMA + "-" + hashlib.sha256(json.dumps(body, separators=(",", ":"), sort_keys=True,
                                                     allow_nan=False).encode()).hexdigest()[:16]


@lru_cache(maxsize=1)
def bundled_char():
    try:
        with gzip.open(ASSET, "rt", encoding="utf-8") as fh:
            model = json.load(fh)
        if model["schema"] == SCHEMA and model["model_version"] == version_of(model):
            return model
    except (KeyError, TypeError, ValueError, OSError, EOFError, zlib.error):
        pass
    return None


def robust_value(text, word_model, char_model):
    word = predict_segments_with_model([text], word_model)
    if word is None or char_model is None:
        return None
    norm = linear = 0.
    vocab = char_model["terms"]
    for term, count in Counter(char_terms(text)).items():
        if term in vocab:
            idf, coefficient = vocab[term]
            weight = (1 + math.log(count)) * idf
            linear += coefficient * weight
            norm += weight**2
    if not norm:
        return None
    char_value = max(-3., min(3., char_model["intercept"] + linear / math.sqrt(norm)))
    return (word["value"] + char_value) / 2
