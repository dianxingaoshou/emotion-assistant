"""An English expressed-polarity reference alongside the opt-in semantic review."""

from collections import Counter
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

from . import semantic_review
from .chinese_affect import format_chinese_affect, is_chinese_text, rate_chinese_expression
from .chinese_affect_calibration import apply_chinese_calibration
from .english_polarity import CLASSES


ASSET = Path(__file__).parent / "assets/goemotions-reference.json.gz"
SCHEMA = "goemotions-word-bigram-tfidf-lr-v1"
TOKEN = re.compile(r"(?u)\b\w\w+\b")
MODEL_SHA256 = "87aec5ced9d772712d76978003f3a13462929fa9051351e0bb8b4a16706677e4"


@lru_cache(maxsize=1)
def bundled_model():
    try:
        encoded = ASSET.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != MODEL_SHA256:
            return None
        return json.loads(gzip.decompress(encoded))
    except (ValueError, EOFError, OSError):
        return None


def _language_supported(text):
    letters = sum(ch.isalpha() for ch in text)
    ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in text)
    return letters > 0 and 5 * ascii_letters >= 4 * letters


def predict_with_model(text, model, *, gate=True):
    """Reproduce sklearn word TF-IDF scores; desktop additionally gates the input."""
    if gate and not _language_supported(text):
        return None
    words = TOKEN.findall(text.lower())
    features = model["terms"]
    if gate:
        candidates = [word for word in words if word.isalpha()]
        known = sum(word in features for word in candidates)
        if known < 2 or known * 2 < len(candidates):
            return None
    counts = Counter(words)
    counts.update(f"{left} {right}" for left, right in zip(words, words[1:]))
    weighted = [(features[term], (1 + math.log(n)) * features[term][0])
                for term, n in counts.items() if term in features]
    norm = math.sqrt(sum(value * value for _, value in weighted))
    scores = list(model["intercept"])
    for row, value in weighted:
        for i in range(len(CLASSES)):
            scores[i] += value / norm * row[i + 1]
    return CLASSES[max(range(len(CLASSES)), key=scores.__getitem__)]


def review_with_reference(text):
    """Review the selected text; add a second opinion on expressed polarity only."""
    result = semantic_review.review_text(text)
    if result.get("status") != "ok":
        return result
    if is_chinese_text(text):
        ratings = apply_chinese_calibration(rate_chinese_expression(text))
        return {**result, "chinese_dimensions": ratings}
    if not _language_supported(text):
        return result
    model = bundled_model()
    label = predict_with_model(text, model) if model is not None else None
    if label is None:
        return result
    content = result.get("content_polarity")
    agreement = "unresolved" if content not in CLASSES else ("agree" if content == label else "disagree")
    return {**result, "english_reference": {
        "model": model["schema"], "asset_sha256": MODEL_SHA256,
        "label": label, "agreement": agreement,
    }}


def format_research_review(result):
    rendered = semantic_review.format_review(result)
    if result.get("status") == "ok" and isinstance(result.get("chinese_dimensions"), dict):
        return rendered + "\n" + format_chinese_affect(result["chinese_dimensions"])
    reference = result.get("english_reference")
    if result.get("status") != "ok" or not isinstance(reference, dict):
        return rendered
    label = semantic_review.POLARITY.get(reference.get("label"))
    detail = {
        "agree": "与大模型对发言倾向的判断一致。",
        "disagree": "与大模型判断不同，请核对原话。",
        "unresolved": "大模型对发言倾向的依据不足，请核对原话。",
    }.get(reference.get("agreement"))
    if not detail or not label:
        return rendered
    return f"{rendered}\n英文发言倾向参照（GoEmotions）为{label}。{detail}"
