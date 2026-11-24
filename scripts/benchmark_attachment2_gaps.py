"""Evaluate aligned whole-word omissions and simultaneous multimodal gaps on Attachment 2."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.attachment2_features import (  # noqa: E402 - supports direct CLI and project imports
    CHANNELS,
    VOCAB_SHA256,
    conditions,
    load_splits,
    masked_inputs,
)


def identity(terms):
    return terms


class CueModel:
    """Three independently supervised experts: TF-IDF words, audio stats and face stats."""

    def fit(self, split):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.preprocessing import StandardScaler

        docs, dense, seen, _ = masked_inputs(split, conditions()[0])
        self.words = TfidfVectorizer(
            analyzer=identity, min_df=3, sublinear_tf=True, max_features=8000
        )
        features = {"T": self.words.fit_transform(docs)}
        seen["T"] = features["T"].getnnz(axis=1) > 0
        self.scalers = {}
        for k, x in dense.items():
            self.scalers[k] = StandardScaler().fit(x[seen[k]])
            features[k] = np.clip(self.scalers[k].transform(x), -6.0, 6.0)
        self.heads = {}
        for k, x in features.items():
            usable = seen[k]
            # Fix learner and regularization on all branches, ahead of any test scores.
            classifier = LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=800
            )
            classifier.fit(x[usable], split.classes[usable])
            regressor = Ridge(alpha=8.0, solver="lsqr", tol=1e-6)
            regressor.fit(x[usable], split.intensities[usable])
            self.heads[k] = classifier, regressor
        return self

    def evidence(self, split, gap):
        docs, dense, seen, ratios = masked_inputs(split, gap)
        text = self.words.transform(docs)
        seen["T"] = text.getnnz(axis=1) > 0
        features = {"T": text}
        for k, x in dense.items():
            features[k] = np.clip(self.scalers[k].transform(x), -6.0, 6.0)
        probs, intensity = [], []
        for k in CHANNELS:
            classifier, regressor = self.heads[k]
            probs.append(classifier.predict_proba(features[k]))
            intensity.append(regressor.predict(features[k]))
        return (
            np.stack(probs, axis=1),
            np.stack(intensity, axis=1),
            np.column_stack([seen[k] for k in CHANNELS]),
        ), ratios


def combine(evidence, class_av_weight, intensity_av_weight, *, text_only=False):
    probabilities, intensities, seen = evidence

    def weights(av_weight):
        weights = seen.astype(float)
        if text_only:
            weights[:, 1:] = 0.0
        else:
            weights[seen[:, 0], 1:] *= av_weight
        return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1.0)

    class_weights = weights(class_av_weight)
    regression_weights = weights(intensity_av_weight)
    distribution = np.einsum("nmc,nm->nc", probabilities, class_weights)
    predicted = distribution.argmax(axis=1)
    predicted[distribution.max(axis=1) < 0.40] = -1
    scores = np.clip((intensities * regression_weights).sum(axis=1), -3.0, 3.0)
    scores[regression_weights.sum(axis=1) == 0] = np.nan
    return predicted, scores


def metrics(split, outputs):
    predicted, intensity = outputs
    classes = split.classes
    valid = np.isfinite(intensity)
    n = len(classes)
    f1 = []
    for k in range(3):
        tp = np.sum((classes == k) & (predicted == k))
        f1.append(2 * tp / max(1, np.sum(classes == k) + np.sum(predicted == k)))
    obs, truth = intensity[valid], split.intensities[valid]
    pearson = None
    if len(obs) > 1 and np.ptp(obs) > 1e-12 and np.ptp(truth) > 1e-12:
        pearson = float(np.corrcoef(obs, truth)[0, 1])
    correct = int(np.sum(predicted == classes))
    abstain = int(np.sum(predicted < 0))
    return {
        "n": n,
        "correct": correct,
        "incorrect": n - correct - abstain,
        "abstain": abstain,
        "accuracy_all": correct / n,
        "macro_f1_all": float(np.mean(f1)),
        "intensity_scored": int(valid.sum()),
        "intensity_abstain": int((~valid).sum()),
        "mae": float(np.abs(obs - truth).mean()) if valid.any() else None,
        "pearson": pearson,
    }


def paired_intervals(split, first, second):
    """Paired accuracy on all clips and MAE on common covered clips; cluster by source."""
    _, inverse = np.unique(split.sources, return_inverse=True)
    n_sources = np.max(inverse) + 1
    both = np.isfinite(first[1]) & np.isfinite(second[1])
    acc = (first[0] == split.classes).astype(float) - (
        second[0] == split.classes
    ).astype(float)
    aerr = np.abs(first[1] - split.intensities) - np.abs(second[1] - split.intensities)
    aggregation = np.column_stack(
        [
            np.bincount(inverse, weights=values, minlength=n_sources)
            for values in (acc, np.ones(len(acc)), np.where(both, aerr, 0.0), both)
        ]
    )
    rng = np.random.default_rng(42)
    indices = rng.integers(0, n_sources, (2000, n_sources))
    totals = aggregation[indices].sum(axis=1)

    def interval(num, den):
        ok = den > 0
        return (
            np.quantile(num[ok] / den[ok], [0.025, 0.975]).tolist()
            if ok.any()
            else None
        )

    return {
        "accuracy_difference_95": interval(totals[:, 0], totals[:, 1]),
        "common_intensity": int(both.sum()),
        "mae_difference_95": interval(totals[:, 2], totals[:, 3]),
    }


def run(aligned, vocab):
    splits = load_splits(aligned, vocab)
    train, valid, test = (splits[k] for k in ("train", "valid", "test"))
    model = CueModel().fit(train)
    cases = conditions()
    # Same predeclared three views; each validation clip has identical total weight.
    tuning = [
        case
        for case in cases
        if case.name in ("complete", "T_50_middle", "TAV_50_middle")
    ]
    cache = [model.evidence(valid, case)[0] for case in tuning]
    choices = []
    for alpha in (0.0, 0.25, 0.5, 1.0):
        scores = [metrics(valid, combine(e, alpha, alpha)) for e in cache]
        choices.append(
            {
                "weight": alpha,
                "macro_f1": np.mean([s["macro_f1_all"] for s in scores]),
                "mae": np.mean([s["mae"] for s in scores]),
            }
        )
    class_av = max(choices, key=lambda c: c["macro_f1"])["weight"]
    intensity_av = min(choices, key=lambda c: c["mae"])["weight"]
    # Held-out labels are first scored after freezing both weights.
    majority = int(np.argmax(np.bincount(train.classes)))
    median = float(np.median(train.intensities))
    constant = np.full(len(test.ids), majority), np.full(len(test.ids), median)
    report = {
        "protocol": "attachment2-aligned-wordgap-v1",
        "official_aligned_sha256": file_sha(aligned),
        "bert_vocab_sha256": VOCAB_SHA256,
        "splits": {
            k: {"clips": len(s.ids), "sources": len(set(s.sources))}
            for k, s in splits.items()
        },
        "cross_partition_source_overlap": 0,
        "targets": "Official class 0/1/2 and real sentiment in [-3,3].",
        "inputs": "Original aligned_50 at most 48 content WordPieces; rebuild words from non-contextual token IDs; split ## pieces and contractions joined. No pretrained contextual text features, reference full sentences, ASR or raw media. Finite nonzero A/V rows inside attended content slots only.",
        "geometry": "Remove ceil(proportion * original lexical word count) adjacent words at start, center (floor left), or end. Extend through all corresponding aligned slots including interword punctuation. Simultaneous gaps remove the SAME interval; staggered uses text/start, audio/middle, vision/end. Ratios are pooled fractions of originally observable words or A/V rows actually removed. No fabricated bigrams across gaps. Repeated conditions are paired, not new clips.",
        "model": "Train-only TF-IDF words + within-run bigrams; A/V mean/std across surviving aligned rows, train-only scaling per expert. Separate balanced logistic (C=1) and ridge (alpha=8) predictors. Mean available predictions with equal A/V relative weights; absent text falls back to mean of available A/V. Missing ALL inputs abstains for both tasks. Max class score <0.40 abstains. Regression clipped to [-3,3].",
        "evaluation": "All 727 test clips are retained in every condition, abstentions count as misses for accuracy/F1. MAE/Pearson are on that arm's finite outputs only; constant/empty correlation is null. Source bootstrap uses 2000 paired resamples. Companion label.xlsx is not needed.",
        "selection": {
            "valid_views": [c.name for c in tuning],
            "validation_grid": choices,
            "selected_class_weight": class_av,
            "selected_intensity_weight": intensity_av,
        },
        "constant": {
            "train_majority_class": majority,
            "train_median": median,
            "test": metrics(test, constant),
        },
        "test": [],
    }
    for gap in cases:
        evidence, ratios = model.evidence(test, gap)
        joint = combine(evidence, class_av, intensity_av)
        text = combine(evidence, 0.0, 0.0, text_only=True)
        row = {
            "condition": gap.name,
            "removed_observations_ratio": ratios,
            "multimodal": metrics(test, joint),
            "same_surviving_text_only": metrics(test, text),
            "fixed_equal_available": metrics(test, combine(evidence, 1.0, 1.0)),
        }
        if gap.name in (
            "complete",
            "T_50_middle",
            "TAV_50_middle",
            "TAV_50_staggered",
            "T_100",
        ):
            row["multimodal_minus_text"] = paired_intervals(test, joint, text)
            row["multimodal_minus_constant"] = paired_intervals(test, joint, constant)
        report["test"].append(row)
    return report


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    from threadpoolctl import threadpool_limits

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aligned",
        type=Path,
        required=True,
        help="Official Attachment 2 aligned_50.pkl",
    )
    parser.add_argument(
        "--vocab", type=Path, required=True, help="Original bert-base-uncased vocab.txt"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "reports/attachment2-gaps.json"
    )
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        report = run(args.aligned, args.vocab)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf8"
    )
    print("validation selection", report["selection"])
    for row in report["test"]:
        if row["condition"] in (
            "complete",
            "T_50_middle",
            "TAV_50_middle",
            "TAV_50_staggered",
            "T_100",
            "TAV_100",
        ):
            print(row["condition"], row["multimodal"])


if __name__ == "__main__":
    main()
