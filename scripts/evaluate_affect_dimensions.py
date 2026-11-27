"""Fit sentence VAD from the official EmoBank train split, calibrate on dev, score test."""

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from urllib.request import urlopen

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emotion_assistant import affect_dimensions as vad
from emotion_assistant.english_polarity import CLASSES, predict_english_polarity

COMMIT = "248ce2a43e165a66d31aeaed83cff9641d6654e0"
SOURCE = f"https://raw.githubusercontent.com/JULIELab/EmoBank/{COMMIT}/corpus/emobank.csv"
SOURCE_SHA256 = "1ade4a4a453e880c0f39d0536d2b355e8a716e0cf88236c64cdb4d438cf9605b"
ATTRIBUTION = {
    "dataset": "EmoBank, Sven Buechel and Udo Hahn, JULIE Lab, 2017",
    "source": SOURCE, "commit": COMMIT, "sha256": SOURCE_SHA256,
    "papers": ["https://aclanthology.org/E17-2092/", "https://aclanthology.org/W17-0801/"],
    "license_statement": f"https://github.com/JULIELab/EmoBank/blob/{COMMIT}/README.md#license",
    "data_and_derived_parameters_license": "CC BY-SA 4.0",
    "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
    "changes": "Fitted sentence-level TF-IDF ridge coefficients and dev residual radii.",
}


def load_splits(cache, download):
    if not cache.exists() and download:
        with urlopen(SOURCE, timeout=60) as stream:
            raw = stream.read()
        if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
            raise ValueError("Unexpected EmoBank checksum")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(raw)
    raw = cache.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Unexpected EmoBank checksum")
    splits = {key: [] for key in ("train", "dev", "test")}
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8"))):
        scores = [float(row[axis]) for axis in ("V", "A", "D")]
        if not all(1 <= score <= 5 for score in scores):
            raise ValueError("Out-of-range annotation")
        splits[row["split"]].append((row["id"], row["text"], scores))
    return splits


def polarity_features(texts):
    # A supervised conversion gives the existing three-polarity scores a fair
    # attempt at predicting each dimension. A fourth indicator records abstention.
    rows = []
    for text in texts:
        pred = predict_english_polarity(text)
        if pred:
            rows.append([pred["scores"][axis] for axis in CLASSES] + [0.0])
        else:
            rows.append([0.0, 0.0, 0.0, 1.0])
    return np.array(rows)


def metrics(target, preds):
    output = {}
    for index, axis in enumerate(vad.AXES):
        truth = target[:, index]
        values = preds[:, index]
        corr = None
        if np.ptp(values) > 1e-12 and np.ptp(truth) > 1e-12:
            corr = round(float(np.corrcoef(truth, values)[0, 1]), 6)
        output[axis] = {"mae": round(float(np.mean(abs(truth - values))), 6), "pearson": corr}
    return output


def export_bytes(model):
    data = json.dumps(model, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as stream:
        stream.write(data)
    return buf.getvalue()


def evaluate(splits):
    texts = {key: [row[1] for row in rows] for key, rows in splits.items()}
    targets = {key: np.array([row[2] for row in rows]) for key, rows in splits.items()}
    vectorizer = TfidfVectorizer(analyzer=vad.terms_in, min_df=2, sublinear_tf=True, norm="l2")
    train = vectorizer.fit_transform(texts["train"])
    ridge = Ridge(alpha=2, solver="lsqr", tol=1e-7).fit(train, targets["train"])
    idfs = vectorizer.idf_
    coefs = ridge.coef_.T
    model = {
        "schema": vad.MODEL, "axes": list(vad.AXES), "attribution": ATTRIBUTION,
        "intercept": ridge.intercept_.round(10).tolist(),
        "features": {term: [round(float(idfs[i]), 10), *coefs[i].round(10).tolist()]
                     for i, term in enumerate(vectorizer.get_feature_names_out())},
    }
    predicted = {key: np.array([vad.score_with_model(t, model) for t in texts[key]])
                 for key in ("dev", "test")}
    statuses = {key: [vad.evidence_for(t, model)[0] for t in texts[key]] for key in ("dev", "test")}
    mask = {key: np.array([status == "ok" for status in statuses[key]]) for key in statuses}
    errors = abs(targets["dev"][mask["dev"]] - predicted["dev"][mask["dev"]])
    rank = math.ceil((len(errors) + 1) * vad.LEVEL)
    model["radii"] = np.sort(errors, axis=0)[rank - 1].tolist()
    test = targets["test"]
    scores = predicted["test"]
    mean_pred = np.tile(np.mean(targets["train"], axis=0), (len(test), 1))
    polarity_ridge = Ridge(alpha=2).fit(polarity_features(texts["train"]), targets["train"])
    polarity_pred = polarity_ridge.predict(polarity_features(texts["test"])).clip(1, 5)
    retained = mask["test"]
    lower = np.maximum(1, scores[retained] - model["radii"])
    upper = np.minimum(5, scores[retained] + model["radii"])
    contained = (lower <= test[retained]) & (test[retained] <= upper)
    check = ridge.predict(vectorizer.transform(texts["test"])).clip(1, 5)
    max_error = float(np.max(abs(check - scores)))
    if max_error > 1e-7:
        raise ValueError("Bundled scorer predictions differ from training implementation")
    train_texts = set(texts["train"])
    fresh_test = np.array([t not in train_texts for t in texts["test"]]) & retained
    report = {
        "model": vad.MODEL, "dataset": ATTRIBUTION, "sklearn_version": sklearn.__version__,
        "target": "Reader and writer perspectives weighted by number of annotators; English sentence-level ratings from 1 to 5.",
        "counts": {key: len(rows) for key, rows in splits.items()}, "vocabulary_size": len(model["features"]),
        "training": "Training split only; word unigrams/bigrams, within-word 3/4-grams and punctuation, min_df 2, log-count TF-IDF, L2 norm, multioutput ridge alpha 2. Fixed before test.",
        "baseline": "Constant train mean and train-fitted ridge conversion of original desktop three polarity scores plus abstention indicator.",
        "exact_train_test_text_overlap_test_rows": sum(t in train_texts for t in texts["test"]),
        "scorer_max_abs_difference_vs_sklearn": max_error,
        "coverage_by_status": {key: dict(Counter(values)) for key, values in statuses.items()},
        "all_test_before_gate": {"train_mean": metrics(test, mean_pred),
                                 "desktop_polarity_conversion": metrics(test, polarity_pred), "vad": metrics(test, scores)},
        "desktop_supported_test": {"n": int(retained.sum()), "coverage": float(np.mean(retained)),
                                    "train_mean": metrics(test[retained], mean_pred[retained]),
                                    "desktop_polarity_conversion": metrics(test[retained], polarity_pred[retained]),
                                    "vad": metrics(test[retained], scores[retained])},
        "supported_test_excluding_verbatim_train_text": {
            "n": int(fresh_test.sum()),
            "desktop_polarity_conversion": metrics(test[fresh_test], polarity_pred[fresh_test]),
            "vad": metrics(test[fresh_test], scores[fresh_test]),
        },
        "intervals": {
            "level": vad.LEVEL, "calibration_split": "dev, supported only",
            "method": "Per-axis absolute residual order statistic ceil((n+1)*level), clipped to annotation scale. Marginal, not simultaneous.",
            "n_calibration": len(errors), "rank": rank,
            "per_axis_test": {axis: {"radius": round(model["radii"][j], 6),
                                    "covered": int(contained[:, j].sum()),
                                    "coverage": round(float(contained[:, j].mean()), 6),
                                    "mean_width": round(float((upper - lower)[:, j].mean()), 6)}
                             for j, axis in enumerate(vad.AXES)},
        },
    }
    return model, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch official data if absent")
    parser.add_argument("--cache", type=Path, default=ROOT / ".runtime/emobank/emobank.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/affect-dimensions.json")
    parser.add_argument("--export-model", type=Path, help="Write the compact .json.gz desktop parameters")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".json" or (args.export_model and not args.export_model.name.endswith(".json.gz")):
        parser.error("Choose .json for the report and .json.gz for the optional model")
    if args.cache.resolve() in {path.resolve() for path in (args.output, args.export_model) if path}:
        parser.error("Keep input and output paths separate")
    model, report = evaluate(load_splits(args.cache, args.download))
    encoded = export_bytes(model)
    report["asset_sha256"] = hashlib.sha256(encoded).hexdigest()
    if args.export_model:
        args.export_model.parent.mkdir(parents=True, exist_ok=True)
        args.export_model.write_bytes(encoded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("counts", "desktop_supported_test", "intervals", "asset_sha256")}, indent=2))


if __name__ == "__main__":
    main()
