"""Offline classification of the expression in one Chinese utterance on the CPED taxonomy."""

from collections import Counter
from functools import lru_cache
import hashlib
from io import BytesIO
import math
from pathlib import Path
import unicodedata

import numpy as np

from .chinese_affect import HAN, is_chinese_text


MODEL = "cped-13-char-tfidf-v1"
ASSET = Path(__file__).parent / "assets" / "cped-emotions-v1.npz"
ASSET_SHA256 = "32f1324704a4e0d9f376f453e1c7d93bf01538a4a3df1be44aa174ff72453871"
LABEL_NAMES = {
    "anger": "生气", "astonished": "惊讶", "depress": "低落", "disgust": "厌恶",
    "fear": "害怕", "grateful": "感激", "happy": "开心", "negative-other": "其他负向表达",
    "neutral": "中性表达", "positive-other": "其他正向表达", "relaxed": "放松",
    "sadness": "悲伤", "worried": "担心",
}
MAX_CHARS = 512
MIN_MARGIN = 0.06


def normalized_text(text, translation):
    return "".join(unicodedata.normalize("NFKC", text).lower().translate(translation).split())


def char_ngrams(text):
    for width in (1, 2, 3):
        for i in range(len(text) - width + 1):
            yield text[i:i + width]


def unpack_model(data):
    with np.load(BytesIO(data), allow_pickle=False) as asset:
        labels = tuple(asset["labels"].tolist())
        if labels != tuple(LABEL_NAMES):
            raise ValueError("Unexpected CPED taxonomy")
        terms = {term: i for i, term in enumerate(asset["terms"].tolist())}
        translation = str.maketrans(asset["traditional"].item(), asset["simplified"].item())
        return {"terms": terms, "translation": translation, "labels": labels, "idf": asset["idf"],
                "weights": asset["weights"], "bias": asset["bias"], "threshold": asset["threshold"].item()}


@lru_cache(maxsize=1)
def _bundled_model():
    try:
        data = ASSET.read_bytes()
    except OSError:
        return None
    if hashlib.sha256(data).hexdigest() != ASSET_SHA256:
        return None
    return unpack_model(data)


def score_text(text, model):
    if not isinstance(text, str) or not text.strip():
        return "no_text", None
    if len(text) > MAX_CHARS:
        return "too_long", None
    if not is_chinese_text(text):
        return "unsupported_language", None
    normalized = normalized_text(text, model["translation"])
    han = HAN.findall(normalized)
    if not han or sum(char in model["terms"] for char in han) < max(2, len(han) / 2):
        return "insufficient_evidence", None
    indices, values = [], []
    for term, count in Counter(char_ngrams(normalized)).items():
        index = model["terms"].get(term)
        if index is not None:
            indices.append(index)
            values.append((1 + math.log(count)) * model["idf"][index])
    features = np.asarray(values)
    logits = model["bias"] + (features / np.linalg.norm(features)) @ model["weights"][indices]
    exp = np.exp(logits - logits.max())
    return "ok", exp / exp.sum()


def select_label(scores, threshold):
    order = np.argsort(-scores, kind="stable")
    top, second = order[:2]
    if scores[top] < threshold or scores[top] - scores[second] < MIN_MARGIN:
        return None
    return int(top)


def analyze_chinese_emotions(text):
    """Expose one supported category, or an abstention, without contacting an external model."""
    result = {"model": MODEL, "language": "zh", "scope": "expressed_text",
              "score_type": "relative_class_score", "labels": [], "top_scores": []}
    model = _bundled_model()
    if model is None:
        return {**result, "status": "model_unavailable"}
    status, scores = score_text(text, model)
    if scores is None:
        return {**result, "status": status}
    for i in np.argsort(-scores, kind="stable")[:2]:
        label = model["labels"][i]
        result["top_scores"].append({"id": label, "name_zh": LABEL_NAMES[label],
                                     "score": round(float(scores[i]), 6)})
    result["threshold"] = model["threshold"]
    result["minimum_margin"] = MIN_MARGIN
    if select_label(scores, model["threshold"]) is None:
        return {**result, "status": "below_threshold"}
    return {**result, "status": "ok", "labels": result["top_scores"][:1]}
