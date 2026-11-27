"""Local multi-label scores for emotions expressed in an English utterance."""

from collections import Counter
from functools import lru_cache
import hashlib
from io import BytesIO
import math
from pathlib import Path

import numpy as np

from .research_text_review import TOKEN, _language_supported


MODEL = "goemotions-28-word-tfidf-ovr-v1"
ASSET = Path(__file__).parent / "assets/goemotions-fine-v1.npz"
ASSET_SHA256 = "1eb4bcd758ab5254594134930538e7ea88516a7ee080b7e1df9b43fc1610bef6"
# Fragments made entirely of grammatical scaffolding do not carry a usable expression.
FUNCTION_WORDS = frozenset("a an the and or but of with for in to by from at as on is are be "
                           "been was were am has had have it that this those these he she they "
                           "we you them their my your our his her its".split())
LABEL_NAMES = {
    "admiration": "钦佩", "amusement": "觉得有趣", "anger": "愤怒", "annoyance": "烦恼",
    "approval": "赞同", "caring": "关心", "confusion": "困惑", "curiosity": "好奇",
    "desire": "渴望", "disappointment": "失望", "disapproval": "不赞同", "disgust": "厌恶",
    "embarrassment": "尴尬", "excitement": "兴奋", "fear": "害怕", "gratitude": "感激",
    "grief": "悲痛", "joy": "喜悦", "love": "喜爱", "nervousness": "紧张",
    "optimism": "乐观", "pride": "自豪", "realization": "恍然领悟", "relief": "如释重负",
    "remorse": "懊悔", "sadness": "悲伤", "surprise": "惊讶", "neutral": "中性表达",
}


@lru_cache(maxsize=1)
def _bundled_model():
    try:
        data = ASSET.read_bytes()
    except OSError:
        return None
    if hashlib.sha256(data).hexdigest() != ASSET_SHA256:
        return None
    return unpack_model(data)


def unpack_model(data):
    with np.load(BytesIO(data), allow_pickle=False) as arrays:
        labels = tuple(arrays["labels"].tolist())
        if labels != tuple(LABEL_NAMES):
            raise ValueError("Model label order differs from the desktop taxonomy")
        return {
            "features": {word: index for index, word in enumerate(arrays["terms"].tolist())},
            "idf": arrays["idf"], "weights": arrays["weights"], "bias": arrays["bias"],
            "thresholds": arrays["thresholds"], "labels": labels,
        }


def score_text(text, model):
    """Return independent sigmoid scores after the same lexical gate used in the app."""
    if not isinstance(text, str) or not text.strip():
        return "no_text", None
    if len(text.encode("utf-8")) > 16 * 1024:
        return "too_long", None
    if not _language_supported(text):
        return "unsupported_language", None
    words = TOKEN.findall(text.lower())
    lexical = [word for word in words if word.isascii() and word.isalpha()]
    features = model["features"]
    known = sum(word in features for word in lexical)
    if (known == 0 or known * 2 < len(lexical)
            or not any(word in features and word not in FUNCTION_WORDS for word in lexical)):
        return "insufficient_evidence", None
    counts = Counter(words)
    counts.update(f"{a} {b}" for a, b in zip(words, words[1:]))
    indices, values = [], []
    for term, count in counts.items():
        index = features.get(term)
        if index is not None:
            indices.append(index)
            values.append((1 + math.log(count)) * model["idf"][index])
    vector = np.asarray(values)
    logits = model["bias"] + (vector / np.linalg.norm(vector)) @ model["weights"][indices]
    return "ok", 1.0 / (1.0 + np.exp(-logits))


def analyze_fine_emotions(text):
    """Multi-label expression, with no substitution of a neutral class on abstention."""
    result = {"model": MODEL, "scope": "expressed_text", "score_type": "independent_label_score",
              "labels": []}
    model = _bundled_model()
    if model is None:
        return {**result, "status": "model_unavailable"}
    status, scores = score_text(text, model)
    if scores is None:
        return {**result, "status": status}
    for i in np.argsort(-scores, kind="stable"):
        if scores[i] >= model["thresholds"][i]:
            label = model["labels"][i]
            result["labels"].append({"id": label, "name_zh": LABEL_NAMES[label],
                                     "score": round(float(scores[i]), 6),
                                     "threshold": round(float(model["thresholds"][i]), 6)})
    result["status"] = "ok" if result["labels"] else "below_threshold"
    return result
