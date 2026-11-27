"""Train and evaluate the desktop's 28-label English expression model on the official splits."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
from urllib.request import urlopen

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emotion_assistant import fine_emotions as runtime
from scripts.evaluate_goemotions import BASE_URL, COMMIT, SHA256, load_files
from scripts.evaluate_goemotions_linear import ATTRIBUTION, TRAIN_SHA256, load_train


DEV_SHA256 = "575489c079c9de1097062a01738f998590d6b7ead66dd1c9fd1d2ba01fd8bc62"
CLASSES = tuple(runtime.LABEL_NAMES)
GRID = np.round(np.arange(.05, .81, .025), 3)


def load_development(directory, download):
    path = directory / "dev.tsv"
    if path.is_file():
        data = path.read_bytes()
    elif download:
        with urlopen(BASE_URL + "/dev.tsv", timeout=60) as response:
            data = response.read()
    else:
        raise ValueError(f"Missing {path}; add --download to obtain the pinned official file")
    if hashlib.sha256(data).hexdigest() != DEV_SHA256:
        raise ValueError("SHA-256 mismatch: dev.tsv")
    if not path.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data.decode("utf-8")


def parse_split(contents):
    texts, identities = [], set()
    rows = contents.splitlines()
    truth = np.zeros((len(rows), len(CLASSES)), dtype=np.uint8)
    for i, row in enumerate(rows):
        text, labels, identity = row.rsplit("\t", 2)
        if identity in identities or not text.strip():
            raise ValueError("Duplicate ID or empty text in the official split")
        identities.add(identity)
        texts.append(text)
        for raw in labels.split(","):
            k = int(raw)
            if not 0 <= k < len(CLASSES):
                raise ValueError("Invalid label index")
            truth[i, k] = 1
    return texts, identities, truth


def choose_thresholds(scores, truth):
    """Maximize each label's development F1 on a fixed grid; ties favor the higher threshold."""
    candidates = []
    for level in GRID[::-1]:
        predicted = scores >= level
        hits = (truth * predicted).sum(axis=0)
        counts = truth.sum(axis=0) + predicted.sum(axis=0)
        candidates.append(np.divide(2. * hits, counts, out=np.zeros(len(CLASSES)), where=counts > 0))
    return GRID[::-1][np.argmax(candidates, axis=0)]


def metrics(truth, predicted):
    counts = predicted.sum(axis=1)
    return {"samples": len(truth), "micro_f1": float(f1_score(truth, predicted, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
            "exact_label_set_matches": int(np.all(predicted == truth, axis=1).sum()),
            "abstained": int((counts == 0).sum()), "coverage": float(np.mean(counts > 0)),
            "mean_predicted_labels": float(counts.mean()), "multi_label_predictions": int((counts > 1).sum())}


def train_and_evaluate(files, asset_output):
    if tuple(files["emotions.txt"].splitlines()) != CLASSES:
        raise ValueError("Taxonomy differs from published label order")
    train, train_ids, y_train = parse_split(files["train.tsv"])
    dev, dev_ids, y_dev = parse_split(files["dev.tsv"])
    test, test_ids, y_test = parse_split(files["test.tsv"])
    if train_ids & dev_ids or train_ids & test_ids or dev_ids & test_ids:
        raise ValueError("Overlapping comment IDs")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30_000,
                                 sublinear_tf=True, norm="l2", token_pattern=runtime.TOKEN.pattern)
    classifier = OneVsRestClassifier(LogisticRegression(solver="liblinear", C=5, max_iter=250, random_state=0))
    with threadpool_limits(limits=1):
        classifier.fit(vectorizer.fit_transform(train), y_train)
        if any(int(est.n_iter_.max()) >= 250 for est in classifier.estimators_):
            raise RuntimeError("Estimator reached iteration limit")
        dev_scores = classifier.predict_proba(vectorizer.transform(dev))
        thresholds = choose_thresholds(dev_scores, y_dev)
        # Test data is scored only after feature selection, training and threshold choice.
        test_scores = classifier.predict_proba(vectorizer.transform(test))
    payload = BytesIO()
    np.savez_compressed(payload, terms=np.asarray(vectorizer.get_feature_names_out(), dtype=str),
                        idf=vectorizer.idf_, labels=np.asarray(CLASSES),
                        weights=np.stack([est.coef_[0] for est in classifier.estimators_], axis=1).astype("float32"),
                        bias=np.array([est.intercept_[0] for est in classifier.estimators_]),
                        thresholds=thresholds,
                        attribution=json.dumps({**ATTRIBUTION, "changes": "Trained new multi-label coefficients."}))
    data = payload.getvalue()
    model = runtime.unpack_model(data)
    predicted = np.zeros_like(y_test, dtype=bool)
    eligible = np.zeros(len(test), dtype=bool)
    max_difference = 0.
    for i, text in enumerate(test):
        _, desktop_scores = runtime.score_text(text, model)
        if desktop_scores is None:
            continue
        eligible[i] = True
        predicted[i] = desktop_scores >= thresholds
        max_difference = max(max_difference, float(np.abs(desktop_scores - test_scores[i]).max()))
    reference = (test_scores >= thresholds) & eligible[:, None]
    if max_difference > 1e-5 or not np.array_equal(reference, predicted):
        raise ValueError("Exported implementation disagrees with scikit-learn on the held-out set")
    baseline = np.zeros_like(y_test)
    baseline[:, np.argmax(y_train.sum(axis=0))] = 1
    precision, recall, f1, support = precision_recall_fscore_support(y_test, predicted, zero_division=0)
    label_scores = {label: {"test_support": int(support[i]), "test_predicted": int(predicted[:, i].sum()),
                            "precision": float(precision[i]), "recall": float(recall[i]),
                            "f1": float(f1[i]), "dev_threshold": float(thresholds[i])}
                    for i, label in enumerate(CLASSES)}
    train_set = set(train)
    duplicate = np.array([text in train_set for text in test])
    report = {
        "model": runtime.MODEL, "task": "English text, all 28 independent expression labels",
        "dataset": {**ATTRIBUTION, "changes": "Original multi-label targets retained and new weights fitted.",
                    "commit": COMMIT, "input_sha256": {k: v for k, v in {
                        **SHA256, "train.tsv": TRAIN_SHA256, "dev.tsv": DEV_SHA256}.items() if k != "sentiment_mapping.json"}},
        "training": {"train_n": len(train), "dev_n": len(dev), "test_n": len(test), "overlapping_ids": 0,
                     "test_identical_text_in_train": int(duplicate.sum()), "vocabulary_size": len(model["features"]),
                     "features": "Word 1-2 grams, min_df=2, max_features=30000, sublinear TF, L2 normalization",
                     "estimator": "One-vs-rest logistic regression, C=5, liblinear, no class weights, seed=0",
                     "threshold_selection": "Per-label max dev F1; 0.05..0.80 by 0.025, ties choose higher value; no test tuning",
                     "sklearn": sklearn.__version__, "numpy": np.__version__},
        "scores": "Independent sigmoid classifier scores; neither subjective intensity nor calibrated confidence.",
        "mean_true_labels": float(y_test.sum(axis=1).mean()),
        "multi_label_ground_truth": int((y_test.sum(axis=1) > 1).sum()),
        "full_test": metrics(y_test, predicted), "per_label": label_scores,
        "baseline_most_frequent_label": {"label": CLASSES[np.argmax(y_train.sum(axis=0))], **metrics(y_test, baseline)},
        "same_model_threshold_0_5": metrics(y_test, (test_scores >= .5) & eligible[:, None]),
        "excluding_train_text_duplicates": metrics(y_test[~duplicate], predicted[~duplicate]),
        "raw_score_macro_average_precision_on_eligible": float(average_precision_score(y_test[eligible], test_scores[eligible], average="macro")),
        "desktop_parity": {"eligible": int(eligible.sum()), "pre_score_abstentions": int((~eligible).sum()),
                           "post_threshold_label_set_mismatches": 0, "max_absolute_score_error": max_difference},
        "asset": {"file": runtime.ASSET.name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)},
    }
    if asset_output:
        asset_output.parent.mkdir(parents=True, exist_ok=True)
        asset_output.write_bytes(data)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".runtime/goemotions" / COMMIT)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--export-model", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/goemotions-fine.json")
    args = parser.parse_args()
    files = load_files(args.data_dir, args.download)
    files["train.tsv"] = load_train(args.data_dir, args.download)
    files["dev.tsv"] = load_development(args.data_dir, args.download)
    result = train_and_evaluate(files, args.export_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({k: result[k] for k in ("full_test", "same_model_threshold_0_5", "asset", "desktop_parity")}, indent=2))
