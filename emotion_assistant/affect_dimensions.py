"""Local English text VAD estimates on the 1–5 EmoBank annotation scale."""

from collections import Counter
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import re


AXES = ("valence", "arousal", "dominance")
MODEL = "emobank-tfidf-ridge-v1"
LEVEL = 0.80
ASSET = Path(__file__).with_name("assets") / "emobank-vad.json.gz"
MODEL_SHA256 = "82ee41ea3859e30d4b2f39494803e1505f4eb0189151cd5b553c9b76f32a812b"
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
MAX_WORDS = 80


def words_in(text):
    return WORD.findall(text.lower().replace("’", "'"))


def terms_in(text):
    """Same analyzer for the training vectorizer and the standard-library scorer."""
    words = words_in(text)
    terms = ["w:" + word for word in words]
    terms.extend("b:" + left + " " + right for left, right in zip(words, words[1:]))
    for word in words:
        padded = "^" + word + "$"
        for width in (3, 4):
            terms.extend("c:" + padded[i:i + width] for i in range(len(padded) - width + 1))
    for mark in ("!", "?", "..."):
        if mark in text:
            terms.append("p:" + mark)
    return terms


@lru_cache(maxsize=1)
def bundled_model():
    try:
        encoded = ASSET.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != MODEL_SHA256:
            return None
        return json.loads(gzip.decompress(encoded))
    except (ValueError, OSError, EOFError):
        return None


def evidence_for(text, model):
    letters = sum(char.isalpha() for char in text)
    if not letters or sum(char.isascii() and char.isalpha() for char in text) < 0.8 * letters:
        return "unsupported_language", {}
    words = words_in(text)
    known = sum("w:" + word in model["features"] for word in words)
    evidence = {"word_count": len(words), "known_words": known}
    if len(words) > MAX_WORDS:
        return "too_long", evidence
    if known < 2 or 2 * known < len(words):
        return "insufficient_words", evidence
    return "ok", evidence


def score_with_model(text, model):
    features = model["features"]
    weighted = []
    for term, count in Counter(terms_in(text)).items():
        if term in features:
            row = features[term]
            weighted.append((row, row[0] * (1 + math.log(count))))
    norm = math.sqrt(sum(tfidf ** 2 for _, tfidf in weighted))
    result = model["intercept"].copy()
    for row, tfidf in weighted:
        for j in range(len(AXES)):
            result[j] += (tfidf / norm) * row[j + 1]
    return [min(5.0, max(1.0, value)) for value in result]


def predict_affect_dimensions(text):
    result = {"model": MODEL, "scope": "expressed_text", "scale": [1, 5],
              "interval_level": LEVEL, "dimensions": {}}
    if not isinstance(text, str) or not text.strip():
        return {**result, "status": "empty"}
    if len(text) > 16000:
        return {**result, "status": "too_long"}
    model = bundled_model()
    if model is None:
        return {**result, "status": "model_unavailable"}
    status, evidence = evidence_for(text, model)
    result.update(status=status, evidence=evidence)
    if status == "ok":
        result["asset_sha256"] = MODEL_SHA256
        for name, score, radius in zip(AXES, score_with_model(text, model), model["radii"]):
            result["dimensions"][name] = {
                "score": round(score, 4),
                "interval": [round(max(1.0, score - radius), 4), round(min(5.0, score + radius), 4)],
            }
    return result
