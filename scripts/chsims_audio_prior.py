"""Replay the class-weight correction for the frozen Omni audio teacher."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_teacher_cross_source import group_ids, labels, metrics, mix, prior_adjustment
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file
from scripts.chsims_teacher_supervision import (
    available, compare, design, fit_classifier, normalize_fit, numbers, read_blocks, read_full_report,
    source_folds, temperature,
)


def fit_audio_head(blocks, truth, head, selected):
    selected = selected[available(blocks)[selected]]
    names = head["features"]
    params = normalize_fit(blocks, names, selected)
    model = fit_classifier(design(blocks, params, selected), truth[selected], head["C"], head["balance_exponent"])
    return {**head, "coef": model.coef_.tolist(), "intercept": model.intercept_.tolist(),
            "normalizer": {name: [p.tolist() for p in stats] for name, stats in params.items()}}


def audio_logits(blocks, head, selected):
    indices = np.flatnonzero(available(blocks)[selected])
    params = {name: tuple(np.asarray(p, np.float32) for p in stats)
              for name, stats in head["normalizer"].items()}
    output = np.full((len(selected), len(LABELS)), np.nan)
    x = design(blocks, params, selected[indices])
    output[indices] = x @ np.asarray(head["coef"]).T + np.asarray(head["intercept"])
    return output


def collection_ids(rows):
    return np.array([re.sub(r"[_-]?\d+$", "", source) for source in group_ids(rows)])


def policy_scores(logits, adjustment, scales):
    scores = {"omni_original": softmax(logits["omni"] / scales["omni"], axis=1),
              "omni_prior": softmax((logits["omni"] + adjustment) / scales["omni_prior"], axis=1),
              "desktop_scores": softmax(logits["desktop_scores"] / scales["desktop_scores"], axis=1),
              "desktop_embedding": softmax(logits["desktop_embedding"] / scales["desktop_embedding"], axis=1)}
    return scores


def select_policy(validation, truth, collections):
    """Give each development collection equal influence on the fusion weights."""
    options = [{"omni_prior": 1.}]
    for desktop in ("desktop_scores", "desktop_embedding"):
        options.extend({"omni_prior": weight, desktop: 1 - weight} for weight in (.25, .5, .75))
    groups = [collections == name for name in sorted(set(collections))]

    def rank(weights):
        prediction = mix(validation, weights)
        per_group = [metrics(truth, prediction, group) for group in groups]
        return (float(np.mean([score["accuracy"] for score in per_group])),
                metrics(truth, prediction)["macro_f1"])

    ranked = [(rank(weights), weights) for weights in options]
    return max(ranked, key=lambda candidate: candidate[0])[1], [
        {"weights": weights, "mean_collection_accuracy": rank[0], "macro_f1": rank[1]}
        for rank, weights in ranked]


def legacy_heads(checkpoint):
    heads = checkpoint["heads"]["audio"]
    return {"omni": heads["omni_selected"], "desktop_scores": heads["desktop_calibrated"],
            "desktop_embedding": heads["desktop_acoustic"]}


def held_out_collections(training, validation):
    for group in sorted(set(validation)):
        yield group, np.flatnonzero(training != group), np.flatnonzero(validation == group)


def fit(cache):
    reports = {split: read_full_report(cache, split) for split in ("train", "valid")}
    checkpoint_path = cache / "teacher-supervision.json"
    checkpoint = json.loads(checkpoint_path.read_text("utf-8"))
    for split, report in reports.items():
        if checkpoint["meta_sha256"] != report["meta_sha256"] or checkpoint["extractors"] != report["models"]:
            raise ValueError(f"The frozen teacher does not match the {split} data")
        if checkpoint["splits"][split]["report_sha256"] != sha_file(cache / f"evaluation-{split}.json"):
            raise ValueError(f"The {split} report changed")
    training, validation = (read_blocks(cache, report)["audio"] for report in reports.values())
    rows = {split: report["rows"] for split, report in reports.items()}
    groups = {split: collection_ids(part) for split, part in rows.items()}
    y_train, y_valid = (numbers(rows[s], "audio") for s in ("train", "valid"))
    heads = legacy_heads(checkpoint)
    raw = {head: np.full((len(y_valid), len(LABELS)), np.nan) for head in heads}
    shifted = raw["omni"].copy()
    folds = []
    for name, ti, vi in held_out_collections(groups["train"], groups["valid"]):
        folds.append({"collection": name, "train": len(ti), "valid": len(vi)})
        for expert, head in heads.items():
            arm = "teacher" if expert == "omni" else "baseline"
            fitted = fit_audio_head(training[arm], y_train, head, ti)
            raw[expert][vi] = audio_logits(validation[arm], fitted, vi)
        ti = ti[available(training["teacher"])[ti]]
        shifted[vi] = raw["omni"][vi] + prior_adjustment(y_train[ti], heads["omni"]["balance_exponent"], 1.)

    calibration = {}
    for expert, logits in {**raw, "omni_prior": shifted}.items():
        observed = np.isfinite(logits[:, 0])
        calibration[expert] = temperature(logits[observed], y_valid[observed])
    probabilities = policy_scores(raw, shifted - raw["omni"], calibration)
    chosen, candidates = select_policy(probabilities, y_valid, groups["valid"])
    probabilities["fused"] = mix(probabilities, chosen)

    observed = available(training["teacher"])
    adjustment = prior_adjustment(y_train[observed], heads["omni"]["balance_exponent"], 1.).tolist()
    targets = []
    for fold, (ti, vi) in enumerate(source_folds(rows["train"])):
        model = fit_audio_head(training["teacher"], y_train, heads["omni"], ti)
        odds = audio_logits(training["teacher"], model, vi)
        fold_adjustment = prior_adjustment(y_train[ti[observed[ti]]], heads["omni"]["balance_exponent"], 1.)
        guess = softmax((odds + fold_adjustment) / calibration["omni_prior"], axis=1)
        targets.extend({"id": rows["train"][i]["id"], "fold": fold,
                        "probabilities": prob.tolist() if np.isfinite(prob[0]) else None}
                       for i, prob in zip(vi, guess, strict=True))
    target_path = cache / "audio-prior-training-targets.json"
    save_json(target_path, {"classes": LABELS, "split": "train", "group_by": "video_id", "rows": targets})
    score = {name: metrics(y_valid, pred) for name, pred in probabilities.items()}
    frozen_path = cache / "audio-prior-calibration.json"
    save_json(frozen_path, {"code_sha256": sha_file(Path(__file__)), "teacher_sha256": sha_file(checkpoint_path),
                            "classes": LABELS, "training_counts": np.bincount(y_train[observed]).tolist(),
                            "correction": adjustment, "temperature": calibration, "weights": chosen,
                            "validation": score, "validation_collections": folds, "candidates": candidates,
                            "targets_sha256": sha_file(target_path)})
    print("Frozen audio correction", adjustment, chosen, sha_file(frozen_path), flush=True)
    for name, value in score.items():
        print(name, value["correct"], f"{value['macro_f1']:.4f}", value["predicted_neutral"], flush=True)


def evaluate(cache, output):
    model_path = cache / "audio-prior-calibration.json"
    checkpoint_path = cache / "teacher-supervision.json"
    model = json.loads(model_path.read_text("utf-8"))
    if model["code_sha256"] != sha_file(Path(__file__)) or model["teacher_sha256"] != sha_file(checkpoint_path):
        raise ValueError("Run fit with this version and the frozen reference heads")
    checkpoint = json.loads(checkpoint_path.read_text("utf-8"))
    test = read_full_report(cache, "test")
    if checkpoint["meta_sha256"] != test["meta_sha256"] or checkpoint["extractors"] != test["models"]:
        raise ValueError("The frozen audio heads use different test data")
    development_sources = set()
    test_ids = {row["id"] for row in test["rows"]}
    for split in ("train", "valid"):
        if checkpoint["splits"][split]["report_sha256"] != sha_file(cache / f"evaluation-{split}.json"):
            raise ValueError(f"The {split} report changed")
        rows = read_full_report(cache, split)["rows"]
        if test_ids & {row["id"] for row in rows}:
            raise ValueError(f"Test and {split} IDs overlap")
        development_sources.update(group_ids(rows))
    rows = test["rows"]
    blocks = read_blocks(cache, test)["audio"]
    index = np.arange(len(rows))
    logits = {name: audio_logits(blocks["teacher" if name == "omni" else "baseline"], head, index)
              for name, head in legacy_heads(checkpoint).items()}
    probabilities = policy_scores(logits, model["correction"], model["temperature"])
    probabilities["fused"] = mix(probabilities, model["weights"])
    y = numbers(rows, "audio")
    sources = group_ids(rows)
    old = logits["omni"].argmax(1)
    shifted = (logits["omni"] + model["correction"]).argmax(1)
    groups = {"official": np.ones(len(rows), bool), "unseen_sources": ~np.isin(sources, list(development_sources)),
              "audio_text_disagreement": np.array([row["truth"]["text"] != row["truth"]["audio"] for row in rows])}
    evidence = {}
    neutral = LABELS.index("neutral")
    predictions = {name: labels(scores) for name, scores in probabilities.items()}
    for name, group in groups.items():
        values = {arm: metrics(y, scores, group) for arm, scores in probabilities.items()}
        selected = np.flatnonzero(group)
        comparisons = {}
        for candidate in ("omni_prior", "fused"):
            comparisons[candidate] = {reference: compare(rows, "audio", predictions[reference],
                                                         predictions[candidate], selected)
                                      for reference in ("omni_original", "desktop_scores", "desktop_embedding")}
            score = values[candidate]
            print(name, candidate, score["correct"], f"{score['macro_f1']:.4f}", score["predicted_neutral"], flush=True)
        false_neutral = group & (old == neutral) & (y != neutral)
        evidence[name] = {"n": len(selected), "sources": len(set(sources[group])),
                          "true_labels": dict(Counter(LABELS[i] for i in y[group])), "metrics": values,
                          "comparisons": comparisons,
                          "neutral_audit": {
                              "previous_false_neutral": int(false_neutral.sum()),
                              "correct_direction_among_previous_false_neutral":
                                  int((logits["omni"][false_neutral, ::2].argmax(axis=1) * 2 == y[false_neutral]).sum()),
                              "false_neutral_corrected": int((false_neutral & (shifted == y)).sum()),
                              "true_neutrals_lost": int((group & (y == neutral) & (old == y) & (shifted != y)).sum()),
                          }}
    save_json(cache / "audio-prior-test-predictions.json", {
        "checkpoint_sha256": sha_file(model_path), "predictions": {
            row["id"]: {arm: guesses[i] for arm, guesses in predictions.items()} for i, row in enumerate(rows)}})
    save_json(output, {"test_ids_sha256": test["selected_ids_sha256"], "checkpoint_sha256": sha_file(model_path),
                       "calibration": model, "groups": evidence})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("fit", "evaluate"))
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=4, user_api="blas"):
        if args.step == "fit":
            fit(args.cache)
        else:
            evaluate(args.cache, args.output or args.cache / "audio-prior-results.json")


if __name__ == "__main__":
    main()
