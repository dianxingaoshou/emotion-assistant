"""Offline classifier of sentiment expressed by English words in a transcript."""

from collections import Counter
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import re


CLASSES = ("negative", "neutral", "positive")
ASSET = Path(__file__).parent / "assets" / "english-polarity.json.gz"
TOKEN = re.compile("[a-z]+(?:'[a-z]+)?")


def tokens(text):
    if not isinstance(text, str):
        return []
    normalized = text.lower().replace("’", "'")
    alphabetic = sum(c.isalpha() for c in normalized)
    english = sum("a" <= c <= "z" for c in normalized)
    if not english or english < 0.8 * alphabetic:
        return []
    return TOKEN.findall(normalized)


def terms(words):
    counts = Counter(words)
    counts.update(a + " " + b for a, b in zip(words, words[1:]))
    return counts


def predict_with_model(text, model):
    words = tokens(text)
    if len(words) < 2:
        return None
    vocab = model["terms"]
    if sum(word in vocab for word in words) < 0.25 * len(words):
        return None
    known = [(vocab[term], n) for term, n in terms(words).items() if term in vocab]
    if not known or not any(max(row[1:]) - min(row[1:]) >= 0.35 for row, _ in known):
        return None
    weights = [(1 + math.log(n)) * row[0] for row, n in known]
    norm = math.sqrt(sum(value**2 for value in weights))
    logits = list(model["intercept"])
    for (row, _), weight in zip(known, weights):
        for index in range(len(CLASSES)):
            logits[index] += row[index + 1] * weight / norm
    maximum = max(logits)
    exp = [math.exp(value - maximum) for value in logits]
    total = sum(exp)
    scores = dict(zip(CLASSES, (value / total for value in exp)))
    label = max(scores, key=scores.get)
    if scores[label] < 0.40:
        return None
    return {"label": label, "scores": scores, "model_version": model["model_version"]}


@lru_cache(maxsize=1)
def _bundled():
    try:
        with gzip.open(ASSET, "rt", encoding="utf-8") as stream:
            doc = json.load(stream)
        if doc["label_order"] != list(CLASSES) or doc["schema"] != "english-tfidf-v1":
            return None
        payload = {key: value for key, value in doc.items() if key != "model_version"}
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if doc["model_version"] != "competition-en-polarity-tfidf-" + digest[:16]:
            return None
        return doc
    except (OSError, KeyError, TypeError, ValueError, EOFError):
        return None


def predict_english_polarity(text):
    """Return expressed-text scores, or None for unsupported/insufficient text."""
    model = _bundled()
    return predict_with_model(text, model) if model is not None else None
