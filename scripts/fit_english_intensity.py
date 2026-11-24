"""Fit an English expressed-sentiment regressor on attachment 2 and evaluate official splits."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emotion_assistant.english_intensity import ASSET, SCHEMA, predict_segments_with_model, version_of
from emotion_assistant.english_polarity import ASSET as POLARITY_ASSET
from emotion_assistant.english_polarity import CLASSES, predict_english_polarity, terms, tokens
from scripts.fit_english_polarity import iter_label_rows

ALPHAS = (0.25, 1.0, 4.0, 16.0)


def read_splits(path):
    splits = {k: [] for k in ("train", "valid", "test")}
    seen = set()
    for row in iter_label_rows(path):
        source, clip = row["video_id"], row["clip_id"]
        y = float(row["label"])
        label, mode = row["annotation"].lower(), row["mode"]
        if not -3 <= y <= 3 or label not in CLASSES or mode not in splits or (source, clip) in seen:
            raise ValueError("Invalid official annotation, duplicate clip or unexpected split.")
        seen.add((source, clip))
        splits[mode].append((source, row["text"], y, label))
    sources = {k: {r[0] for r in rows} for k, rows in splits.items()}
    if any(not rows for rows in splits.values()):
        raise ValueError("An official split is empty.")
    if any(sources[a] & sources[b] for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))):
        raise ValueError("Source videos overlap across official splits.")
    return splits


def train_design(rows):
    docs = [terms(tokens(row[1])) for row in rows]
    df = Counter()
    for doc in docs:
        df.update(doc.keys())
    vocab = sorted((k for k in df if df[k] >= 3), key=lambda k: (-df[k], k))[:5000]
    index = {k: i for i, k in enumerate(vocab)}
    idf = [1 + math.log((len(docs) + 1) / (df[k] + 1)) for k in vocab]
    columns, values, end = [], [], [0]
    for doc in docs:
        weights = [(index[k], (1 + math.log(n)) * idf[index[k]]) for k, n in doc.items() if k in index]
        norm = math.sqrt(sum(v**2 for _, v in weights))
        columns.extend(i for i, _ in weights)
        values.extend(v / norm for _, v in weights)
        end.append(len(columns))
    return csr_matrix((values, columns, end), shape=(len(docs), len(vocab))), vocab, idf


def fit(train, design, alpha):
    from sklearn.linear_model import Ridge

    matrix, vocab, idf = design
    ridge = Ridge(alpha=alpha, solver="lsqr", tol=1e-9)
    ridge.fit(matrix, [row[2] for row in train])
    model = {
        "schema": SCHEMA,
        "alpha": alpha,
        "intercept": round(float(ridge.intercept_), 9),
        "terms": {k: [round(idf[i], 9), round(float(ridge.coef_[i]), 9)] for i, k in enumerate(vocab)},
    }
    model["model_version"] = version_of(model)
    return model


def regression_metrics(truth, predicted):
    y, p = np.asarray(truth, dtype=float), np.asarray(predicted, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("Expected equally sized finite 1D vectors.")
    if not len(y):
        return {"n": 0, "mae": None, "pearson": None}
    corr = None
    if len(y) >= 2 and np.ptp(y) > 0 and np.ptp(p) > 0:
        corr = float(np.corrcoef(y, p)[0, 1])
    return {"n": len(y), "mae": float(np.abs(y - p).mean()), "pearson": corr}


def estimates(rows, model):
    result = []
    for _, text, _, _ in rows:
        pred = predict_segments_with_model([text], model)
        result.append(pred["value"] if pred is not None else np.nan)
    return np.array(result)


def paired_intervals(sources, truth, predicted, baseline):
    """Resample complete source videos to preserve within-source dependence."""
    labels, inverse = np.unique(sources, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(labels))]
    rng = np.random.default_rng(20261123)
    mae_delta, pearson = [], []
    for _ in range(2000):
        indices = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        y, p, b = truth[indices], predicted[indices], baseline[indices]
        mae_delta.append(np.mean(np.abs(y - p) - np.abs(y - b)))
        if np.ptp(y) > 0 and np.ptp(p) > 0:
            pearson.append(np.corrcoef(y, p)[0, 1])
    return {
        "draws": 2000,
        "mae_delta_vs_train_median_95pct": np.quantile(mae_delta, [.025, .975]).tolist(),
        "pearson_95pct": np.quantile(pearson, [.025, .975]).tolist() if pearson else None,
    }


def evaluate(rows, model, median, polarity_medians):
    y = np.array([row[2] for row in rows])
    p = estimates(rows, model)
    valid = np.isfinite(p)
    constant = np.full(len(rows), median)
    # The executable baseline uses a predicted class (never the test gold class).
    previous = []
    for _, text, _, _ in rows:
        polarity = predict_english_polarity(text)
        previous.append(polarity_medians[polarity["label"]] if polarity is not None else median)
    p_all = np.where(valid, p, median)
    return {
        "clips": len(rows), "sources": len({r[0] for r in rows}),
        "scored": int(valid.sum()), "abstained": int((~valid).sum()), "coverage": float(valid.mean()),
        "regressor_scored": regression_metrics(y[valid], p[valid]),
        "median_on_same_scored": regression_metrics(y[valid], constant[valid]),
        "regressor_all_with_train_median_fallback": regression_metrics(y, p_all),
        "train_median_all": regression_metrics(y, constant),
        "existing_polarity_class_to_train_median_all": regression_metrics(y, previous),
        "all_with_fallback_clustered_intervals": paired_intervals(
            [row[0] for row in rows], y, p_all, constant
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--model", type=Path, default=ASSET)
    parser.add_argument("--report", type=Path, default=ROOT / "docs/results/english-intensity.json")
    args = parser.parse_args()
    splits = read_splits(args.workbook)
    train, valid = splits["train"], splits["valid"]
    median = float(np.median([row[2] for row in train]))
    by_class = {k: float(np.median([r[2] for r in train if r[3] == k])) for k in CLASSES}
    design = train_design(train)
    grid, model, best = [], None, float("inf")
    for alpha in ALPHAS:
        candidate = fit(train, design, alpha)
        p = estimates(valid, candidate)
        score = regression_metrics([r[2] for r in valid], np.where(np.isfinite(p), p, median))
        grid.append(dict(alpha=alpha, **score))
        if score["mae"] < best:
            best, model = score["mae"], candidate
    artifact = json.dumps(model, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    text_sets = {k: {tuple(tokens(r[1])) for r in rows} for k, rows in splits.items()}
    report = {
        "protocol": "attachment2-english-expressed-intensity-v1",
        "workbook_sha256": hashlib.sha256(args.workbook.read_bytes()).hexdigest(),
        "model_version": model["model_version"], "artifact_uncompressed_sha256": hashlib.sha256(artifact).hexdigest(),
        "source": "Attachment 2 label.xlsx, official English transcript and continuous label for each clip, original mode splits.",
        "target": "Signed expressed sentiment intensity, clipped to [-3,+3]; independent of polarity classification and user self-assessment.",
        "method": "Train-only English unigram/bigram TF-IDF (train df>=3, at most 5000 terms), sublinear TF, L2 normalization, ridge regression on continuous labels. Entirely local inference.",
        "selection": "Lowest validation MAE over all clips, imputing train median for abstentions. Fixed grid ordered by alpha. No refit on validation. Test opened for final frozen model evaluation.",
        "abstention": "Same gates in all splits and desktop: >=80% ASCII English alphabetic characters, >=2 words, >=25% training vocabulary coverage and nonempty TF-IDF. No numerical value on abstention in the UI. The median fallback is used for all-clip comparisons only.",
        "metrics": "MAE and Pearson on identical denominators; Pearson is null for a constant. Intervals use 2000 paired source-video bootstrap draws, seed 20261123.",
        "scope": "Reference English transcript experiment, not an acoustic or visual intensity model. Individual clause and ASR-generated transcript intensity need separate evaluation.",
        "train_median": median, "train_median_by_gold_polarity": by_class,
        "prior_classifier_artifact_sha256": hashlib.sha256(POLARITY_ASSET.read_bytes()).hexdigest(),
        "train_clips": len(train), "train_sources": len({r[0] for r in train}),
        "source_overlap_across_partitions": 0,
        "normalized_transcript_overlap_distinct": {
            a + "_" + b: len(text_sets[a] & text_sets[b])
            for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))
        },
        "valid_selection_grid": grid,
        "chosen_alpha": model["alpha"], "n_terms": len(model["terms"]),
        "official": {k: evaluate(splits[k], model, median, by_class) for k in ("valid", "test")},
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.model.write_bytes(gzip.compress(artifact, mtime=0))
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8")
    print(json.dumps(report["official"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
