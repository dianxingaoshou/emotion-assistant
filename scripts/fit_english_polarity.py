"""Fit on the competition label.xlsx train partition and score the untouched official splits."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from xml.etree import ElementTree as ET
import zipfile

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emotion_assistant.english_polarity import ASSET, CLASSES, predict_with_model, terms, tokens


def iter_label_rows(path):
    """Yield official workbook values keyed by their column headers."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as workbook:
        strings = ["".join(x.itertext()) for x in ET.fromstring(workbook.read("xl/sharedStrings.xml"))]
        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
    header = {}
    for row in sheet.iter(ns + "row"):
        values = {}
        for cell in row.iter(ns + "c"):
            item = cell.find(ns + "v")
            if item is not None:
                name = re.match(r"[A-Z]+", cell.attrib["r"])[0]
                values[name] = strings[int(item.text)] if cell.get("t") == "s" else item.text
        if not values:
            continue
        if not header:
            header = values
            continue
        yield {header[col]: value for col, value in values.items() if col in header}


def load_rows(path):
    seen = set()
    splits = {k: [] for k in ("train", "valid", "test")}
    for row in iter_label_rows(path):
        video, clip, text, label, mode = (row[k] for k in ("video_id", "clip_id", "text", "annotation", "mode"))
        if (video, clip) in seen or mode not in splits or label.lower() not in CLASSES:
            raise ValueError("duplicate clips, unrecognized partition or unsupported label")
        seen.add((video, clip))
        splits[mode].append((video, text, label.lower()))
    return splits


def fit(rows):
    by_document = [terms(tokens(text)) for _, text, _ in rows]
    df = Counter()
    for counts in by_document:
        df.update(counts.keys())
    vocab = sorted((k for k in df if df[k] >= 3), key=lambda k: (-df[k], k))[:5000]
    index = {k: i for i, k in enumerate(vocab)}
    idf = [1 + math.log((len(rows) + 1) / (df[k] + 1)) for k in vocab]
    y = np.array([CLASSES.index(label) for _, _, label in rows])
    values, indices, indptr = [], [], [0]
    for doc in by_document:
        active = [(index[k], n) for k, n in doc.items() if k in index]
        weights = [(1 + math.log(n)) * idf[i] for i, n in active]
        norm = math.sqrt(sum(w * w for w in weights))
        indices.extend(i for i, _ in active)
        values.extend(w / norm for w in weights)
        indptr.append(len(values))
    X = csr_matrix((values, indices, indptr), shape=(len(rows), len(vocab)))
    class_weight = len(y) / (len(CLASSES) * np.bincount(y))
    sample_weights = class_weight[y]
    target = np.eye(len(CLASSES))[y]
    shape = (len(vocab) + 1, len(CLASSES))

    def loss(params):
        w = params.reshape(shape)
        logits = X @ w[:-1] + w[-1]
        norm = logsumexp(logits, axis=1)
        value = sample_weights @ (norm - logits[np.arange(len(y)), y]) + 0.125 * np.square(w[:-1]).sum()
        err = (np.exp(logits - norm[:, None]) - target) * sample_weights[:, None]
        grad = np.vstack((X.T @ err + 0.25 * w[:-1], err.sum(axis=0)))
        return float(value), grad.ravel()

    result = minimize(loss, np.zeros(np.prod(shape)), method="L-BFGS-B", jac=True, options={"maxiter": 500})
    if not result.success:
        raise ValueError(result.message)
    w = result.x.reshape(shape)
    doc = {
        "schema": "english-tfidf-v1",
        "label_order": list(CLASSES),
        "intercept": w[-1].round(7).tolist(),
        "terms": {k: [round(idf[i], 7), *w[i].round(7).tolist()] for i, k in enumerate(vocab)},
    }
    digest = hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    doc["model_version"] = "competition-en-polarity-tfidf-" + digest[:16]
    return doc


def split_score(matrix):
    c = np.array(matrix)
    hits = c[:, :3].diagonal()
    true, guesses = c.sum(axis=1), c[:, :3].sum(axis=0)
    f1 = 2 * hits / np.maximum(1, true + guesses)
    return {
        "n": int(c.sum()),
        "accuracy": float(hits.sum() / c.sum()) if c.sum() else None,
        "macro_f1": float(f1.mean()) if c.sum() else None,
        "per_class_recall": dict(zip(CLASSES, (hits / np.maximum(1, true)).tolist())),
        "per_class_f1": dict(zip(CLASSES, f1.tolist())),
    }


def evaluate(rows, model, majority):
    confusion = np.zeros((3, 4), dtype=int)
    for _, text, label in rows:
        pred = predict_with_model(text, model)
        j = CLASSES.index(pred["label"]) if pred else 3
        confusion[CLASSES.index(label), j] += 1
    accepted = confusion[:, :3]

    def baseline(counts):
        mat = np.zeros_like(confusion)
        mat[:, CLASSES.index(majority)] = counts
        return split_score(mat)

    return {
        "n": len(rows),
        "videos": len({v for v, _, _ in rows}),
        "coverage": int(accepted.sum()) / len(rows) if rows else None,
        "confusion_rows_truth_columns_negative_neutral_positive_abstain": confusion.tolist(),
        "including_abstention_as_miss": split_score(confusion),
        "conditional_on_score": split_score(accepted),
        "majority_all": baseline(confusion.sum(axis=1)),
        "majority_on_model_scored": baseline(accepted.sum(axis=1)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--model", type=Path, default=ASSET)
    parser.add_argument("--report", type=Path, default=ROOT / "docs/results/english-polarity.json")
    args = parser.parse_args()
    splits = load_rows(args.workbook)
    model = fit(splits["train"])
    text_key = lambda text: " ".join(tokens(text))
    text_sets = {k: {text_key(t) for _, t, _ in rows} for k, rows in splits.items()}
    videos = {k: {v for v, _, _ in rows} for k, rows in splits.items()}
    pairs = (("train", "valid"), ("train", "test"), ("valid", "test"))
    majority = Counter(label for _, _, label in splits["train"]).most_common(1)[0][0]
    body = json.dumps(
        model, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode()
    args.model.write_bytes(gzip.compress(body, mtime=0))
    report = {
        "protocol": "attachment2-english-text-v1",
        "workbook_sha256": hashlib.sha256(args.workbook.read_bytes()).hexdigest(),
        "source": "E题 附件2-数据集特征文件/label.xlsx; text and official annotation, video_id, clip_id and mode only",
        "model_version": model["model_version"],
        "artifact_uncompressed_sha256": hashlib.sha256(body).hexdigest(),
        "method": "Lowercase English words and adjacent pairs, min train document frequency 3, up to 5000 terms, sublinear TF, smoothed IDF and L2 normalization. Class-balanced three-way multinomial logistic regression, C=4. Scores are softmax model outputs. No feature, IDF, coefficient or threshold uses validation or test.",
        "abstention": "Need >=80% ASCII English alphabetic characters, >=2 tokens, >=25% known tokens, >=1 feature with learned class coefficient spread >=0.35 and maximum score >=0.40. All gates fixed before evaluation.",
        "scope": "Sentiment expressed by English transcript. Local standalone text or separately opted-in transcript; no raw text is bundled. No Chinese classifier; no inference of the speaker's own state.",
        "n_terms": len(model["terms"]),
        "class_order": list(CLASSES),
        "train_majority": majority,
        "counts": {k: dict(Counter(label for _, _, label in rows)) for k, rows in splits.items()},
        "video_overlap_counts": {a + "_" + b: len(videos[a] & videos[b]) for a, b in pairs},
        "normalized_text_overlap_distinct": {a + "_" + b: len(text_sets[a] & text_sets[b]) for a, b in pairs},
        "repeated_normalized_text_within_split": {k: len(v) - len(text_sets[k]) for k, v in splits.items()},
        "official": {k: evaluate(splits[k], model, majority) for k in ("valid", "test")},
    }
    clean = [
        (v, t, l)
        for v, t, l in splits["test"]
        if v not in videos["train"] | videos["valid"]
        and text_key(t) not in text_sets["train"] | text_sets["valid"]
    ]
    report["test_excluding_text_or_video_seen_in_train_or_valid"] = evaluate(clean, model, majority)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8")
    for k, score in report["official"].items():
        print(k, json.dumps(score["including_abstention_as_miss"]), "coverage", score["coverage"])


if __name__ == "__main__":
    main()
