"""Fit a supervised GoEmotions polarity baseline and score fixed official test cohorts.

Run with scikit-learn installed in the evaluation environment. The report
contains only aggregate results. Use --export-model to write the vocabulary
and coefficients used by the standard-library desktop scorer.
"""

import argparse
import gzip
from io import BytesIO
import json
from pathlib import Path
import sys
from urllib.request import urlopen

import numpy
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_goemotions import (
    ABSTAIN, BASE_URL, CLASSES, COMMIT, ROOT, SHA256, group_split, load_files,
    metrics, paired, select_sample, sha256,
)
from emotion_assistant import research_text_review as reference


TRAIN_SHA256 = "1c254a142be5c00e80d819b9ae1bbd36d94b2eeb8f4b1271846508d57e57d9c5"
PILOT_IDS_SHA256 = "455b3a91b13114ea845a46d86dbffc54db547045dd4ba7aee7c757c3e43bd289"
CONFIRMATION_IDS_SHA256 = "0f1028d7425a644132e854b600e756e51ee9ea4c30b088069f43d1f7203d8c62"
VECTORIZER = dict(analyzer="word", token_pattern=r"(?u)\b\w\w+\b", lowercase=True,
                  ngram_range=(1, 2), min_df=2, max_features=60000,
                  sublinear_tf=True, norm="l2")
CLASSIFIER = dict(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=300)
ATTRIBUTION = {
    "dataset": "GoEmotions, Google Research; Demszky, Movshovitz-Attias, Ko, Cowen, Nemade and Ravi (2020).",
    "source": f"https://github.com/google-research/google-research/tree/{COMMIT}/goemotions",
    "paper": "https://aclanthology.org/2020.acl-main.372/",
    "source_license_statement": f"https://github.com/google-research/google-research/blob/{COMMIT}/README.md",
    "dataset_license": "CC BY 4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "changes": "Collapsed eligible official training labels to three polarities and fitted new weights.",
}


def load_train(directory, download):
    path = directory / "train.tsv"
    if path.is_file():
        data = path.read_bytes()
    elif download:
        with urlopen(f"{BASE_URL}/train.tsv", timeout=60) as response:
            data = response.read()
    else:
        raise ValueError(f"Missing {path}. Use --download to fetch the pinned official split.")
    if sha256(data) != TRAIN_SHA256:
        raise ValueError("SHA-256 mismatch: train.tsv")
    if not path.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data.decode("utf-8")


def export_payload(vectorizer, classifier):
    if list(classifier.classes_) != list(CLASSES):
        raise ValueError("Unexpected fitted label order")
    terms = {str(term): [float(vectorizer.idf_[i]), *classifier.coef_[:, i].tolist()]
             for i, term in enumerate(vectorizer.get_feature_names_out())}
    payload = {
        "schema": reference.SCHEMA, "label_order": list(CLASSES), "attribution": ATTRIBUTION,
        "train_sha256": TRAIN_SHA256, "intercept": classifier.intercept_.tolist(), "terms": terms,
    }
    raw = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    packed = BytesIO()
    with gzip.GzipFile(fileobj=packed, mode="wb", filename="", mtime=0, compresslevel=9) as stream:
        stream.write(raw)
    return packed.getvalue()


def compare_review(rows, classifier_by_id, gated_by_id, review_decisions):
    """Match the same held-out IDs and labels before comparing concordant errors."""
    decisions = json.loads(review_decisions.read_text(encoding="utf-8"))
    if not isinstance(decisions, list) or len(decisions) != len(rows):
        raise ValueError("Expected one review decision per confirmation comment")
    if not all(isinstance(row, dict) and isinstance(row.get("id"), str) for row in decisions):
        raise ValueError("Malformed review decision")
    by_id = {row["id"]: row for row in decisions}
    if by_id.keys() != {row[1] for row in rows}:
        raise ValueError("Review decision IDs do not match the confirmation cohort")
    outcomes = []
    for _, comment_id, truth, _ in rows:
        decision = by_id[comment_id]
        if decision.get("truth") != truth or decision.get("review") not in (*CLASSES, ABSTAIN):
            raise ValueError("Review decision has an inconsistent truth or unknown prediction")
        outcomes.append({"truth": truth, "baseline": classifier_by_id[comment_id], "review": decision["review"]})
    agreement = {"agree": 0, "disagree": 0, "reference_unavailable": 0, "review_unresolved": 0}
    selective = []
    for item in decisions:
        label = gated_by_id[item["id"]]
        if label is None:
            key = "reference_unavailable"
        elif item["review"] == ABSTAIN:
            key = "review_unresolved"
        else:
            key = "agree" if item["review"] == label else "disagree"
        agreement[key] += 1
        selective.append({"truth": item["truth"], "prediction": label if key == "agree" else ABSTAIN})
    return {
        "baseline": "in-domain TF-IDF + logistic regression (before desktop language/lexical gate)",
        "review": "desktop semantic review content_polarity", **paired(outcomes),
        "desktop_agreement": {
            "rule": "Retain a polarity only when the gated desktop reference and the review agree.",
            "denominator": len(rows), "disposition": agreement, **metrics(selective, "prediction"),
        },
    }


def evaluate(files, review_decisions=None, export_model=None):
    grouped, train_counts = group_split(files, "train.tsv")
    train = sorted(row for rows in grouped.values() for row in rows)
    train_texts = [row[3] for row in train]
    train_labels = [row[2] for row in train]
    train_text_set = set(train_texts)
    model = make_pipeline(TfidfVectorizer(**VECTORIZER), LogisticRegression(**CLASSIFIER))
    with threadpool_limits(limits=1):
        model.fit(train_texts, train_labels)
    vectorizer, classifier = model.steps[0][1], model.steps[1][1]
    if max(classifier.n_iter_) >= CLASSIFIER["max_iter"]:
        raise RuntimeError("Training hit its fixed iteration limit; check convergence")

    grouped, test_counts = group_split(files, "test.tsv")
    test = sorted(row for rows in grouped.values() for row in rows)
    # Audit by official comment ID, including excluded records in both splits.
    train_ids, test_ids = (
        {line.rsplit("\t", 1)[1] for line in files[name].splitlines()}
        for name in ("train.tsv", "test.tsv")
    )
    if train_ids & test_ids:
        raise ValueError("Official train and test share a comment ID")
    with threadpool_limits(limits=1):
        predictions = model.predict([row[3] for row in test])
    by_id = {row[1]: label for row, label in zip(test, predictions, strict=True)}
    packed = export_payload(vectorizer, classifier)
    exported = json.loads(gzip.decompress(packed))
    mismatches = sum(reference.predict_with_model(text, exported, gate=False) != by_id[row_id]
                     for _, row_id, _, text in test)
    if mismatches:
        raise ValueError(f"Exported runtime and sklearn disagree on {mismatches} held-out decisions")
    gated = {row_id: reference.predict_with_model(text, exported) for _, row_id, _, text in test}
    if export_model is not None:
        export_model.parent.mkdir(parents=True, exist_ok=True)
        export_model.write_bytes(packed)
    pilot, pilot_selection = select_sample(files, 20)
    confirmation, confirmation_selection = select_sample(files, 60, 20)
    if pilot_selection["ordered_ids_sha256"] != PILOT_IDS_SHA256:
        raise ValueError("Selection does not match the frozen pilot")
    if confirmation_selection["ordered_ids_sha256"] != CONFIRMATION_IDS_SHA256:
        raise ValueError("Selection does not match the frozen confirmation cohort")
    if {row[1] for row in pilot} & {row[1] for row in confirmation}:
        raise ValueError("Confirmation set overlaps the pilot")

    def summarize(rows):
        outcomes = [{"truth": truth, "prediction": by_id[row_id]} for _, row_id, truth, _ in rows]
        return {
            "sample_count": len(rows),
            "exact_text_also_in_eligible_training": sum(row[3] in train_text_set for row in rows),
            **metrics(outcomes, "prediction"),
        }

    report = {
        "task": "expressed content polarity",
        "dataset": {"repository": "google-research/google-research/goemotions", "commit": COMMIT,
                    "sha256": {**SHA256, "train.tsv": TRAIN_SHA256}, "attribution": ATTRIBUTION},
        "protocol": {
            "training": "Official train.tsv only; vocabulary, IDF and classifier fit on eligible training comments.",
            "labels": list(CLASSES),
            "mapping": "Use the official sentiment taxonomy; exclude ambiguous or cross-polarity comments.",
            "selection": "Model settings fixed before evaluating either cohort; no tuning or threshold selection.",
            "decision": "Highest-scoring class for every comment.",
            "overlap_audit": "Raw case-sensitive identical comment texts across eligible splits are counted, not removed.",
            "official_train_test_shared_ids": 0,
        },
        "software": {"python": sys.version.split()[0], "scikit_learn": sklearn.__version__,
                     "numpy": numpy.__version__, "scipy": scipy.__version__},
        "model": {
            "vectorizer": {"type": "TfidfVectorizer", **VECTORIZER},
            "classifier": {"type": "LogisticRegression", **CLASSIFIER},
            "fit_order": "sorted by the fixed per-ID hash", "library_threads": 1,
            "actual_vocabulary_size": len(vectorizer.vocabulary_),
            "solver_iterations": classifier.n_iter_.tolist(),
        },
        "desktop_asset": {
            "schema": reference.SCHEMA, "compressed_bytes": len(packed), "sha256": sha256(packed),
            "check": "Deserialized standard-library scorer versus sklearn before input gating.",
            "compared": len(test), "label_mismatches": mismatches,
            "gate": "ASCII letters >=80% of letters; at least two vocabulary matches among alphabetic tokens and >=50% of those tokens known.",
            "full_eligible_test_passing_gate": sum(label is not None for label in gated.values()),
        },
        "train": {"eligible_count": len(train), **train_counts},
        "pilot_60": {"selection": pilot_selection, **summarize(pilot)},
        "confirmation_180": {"selection": confirmation_selection, **summarize(confirmation)},
        "full_eligible_test": {"split_counts": test_counts, **summarize(test)},
    }
    if review_decisions is not None:
        report["confirmation_180"]["paired_review"] = compare_review(confirmation, by_id, gated, review_decisions)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".runtime/goemotions" / COMMIT)
    parser.add_argument("--download", action="store_true", help="Cache missing verified official source files")
    parser.add_argument("--review-decisions", type=Path,
                        help="Compare to confirmation IDs and decisions from evaluate_goemotions.py --predictions-output")
    parser.add_argument("--export-model", type=Path, metavar="FILE.json.gz",
                        help="Export fitted vocabulary, IDF and coefficients for local inference")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/goemotions-linear.json")
    args = parser.parse_args()
    source_names = (*SHA256, "train.tsv")
    inputs = {(args.data_dir / name).resolve() for name in source_names}
    if args.review_decisions is not None:
        inputs.add(args.review_decisions.resolve())
    if args.output.suffix.lower() != ".json" or args.output.resolve() in inputs:
        parser.error("Choose a JSON output separate from the source files")
    if args.export_model and (not args.export_model.name.endswith(".json.gz") or args.export_model.resolve() in inputs):
        parser.error("Choose a separate .json.gz model asset file")
    try:
        files = load_files(args.data_dir, args.download)
        files["train.tsv"] = load_train(args.data_dir, args.download)
        report = evaluate(files, args.review_decisions, args.export_model)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name in ("pilot_60", "confirmation_180", "full_eligible_test"):
        entry = report[name]
        print(f"{name}: {entry['correct']}/{entry['sample_count']}, macro-F1 {entry['macro_f1_all']:.4f}")


if __name__ == "__main__":
    main()
