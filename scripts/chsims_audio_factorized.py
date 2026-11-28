"""Separate recognition of neutral audio from the direction of emotional audio."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_audio_prior import collection_ids, held_out_collections
from scripts.chsims_teacher_cross_source import group_ids, labels, load_blocks, metrics
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file
from scripts.chsims_teacher_supervision import compare, design, head_predictions, normalize_fit, numbers, read_full_report


BLOCKS = {"omni": ("context", "acoustic"),
          "speech_and_omni": ("context", "acoustic", "speech", "expert_scores", "expert_quality"),
          "speech": ("speech", "expert_scores", "expert_quality")}
NEUTRAL = LABELS.index("neutral")
REGULARIZATION = (.3, 2., 15.)
SOURCE_WEIGHT = (0., .5)
NEUTRAL_COSTS = (.75, 1., 1.25)
ESTIMATORS = ("direction", "neutral")


def fit_binary(x, y, selected, objective, groups, strength, source_power):
    observed = y[selected]
    group_names, ids = np.unique(groups[selected], return_inverse=True)
    counts = np.bincount(ids, minlength=len(group_names))
    weight = counts[ids] ** -source_power
    binary = observed == (2 if objective == "direction" else NEUTRAL)
    if binary.all() or not binary.any():
        raise ValueError(f"Need both categories for {objective}")
    model = LogisticRegression(C=strength, max_iter=600)
    model.fit(x[selected], binary, sample_weight=weight / weight.mean())
    return {"coef": model.coef_[0].tolist(), "bias": float(model.intercept_[0])}


def join_logits(direction, neutrality, penalty):
    neutral = expit(neutrality - np.log(penalty))
    positive = expit(direction)
    return np.column_stack(((1 - neutral) * (1 - positive), neutral, (1 - neutral) * positive))


def score_groups(truth, prediction, groups):
    items = [metrics(truth, prediction, group) for group in groups]
    return (np.mean([item["accuracy"] for item in items]), metrics(truth, prediction)["macro_f1"])


def fit(cache):
    train, valid = (read_full_report(cache, split) for split in ("train", "valid"))
    if train["models"] != valid["models"] or train["meta_sha256"] != valid["meta_sha256"]:
        raise ValueError("Training and validation need the same dataset and extractors")
    train_rows, valid_rows = train["rows"], valid["rows"]
    if {row["id"] for row in train_rows} & {row["id"] for row in valid_rows}:
        raise ValueError("Training and validation IDs overlap")
    training, validation = (load_blocks(cache, report, ("audio",))[0]["audio"] for report in (train, valid))
    y_train, y_valid = numbers(train_rows, "audio"), numbers(valid_rows, "audio")
    train_sources, valid_sources = (collection_ids(rows) for rows in (train_rows, valid_rows))
    folds = list(held_out_collections(train_sources, valid_sources))
    groups = [valid_sources == group for group, _, _ in folds]
    train_available = np.isfinite(training["acoustic"][:, 0])
    valid_available = np.isfinite(validation["acoustic"][:, 0])
    candidates = {}
    for name, features in BLOCKS.items():
        for objective in ESTIMATORS:
            candidates[name, objective] = {(reg, source): np.full(len(y_valid), np.nan)
                                           for reg in REGULARIZATION for source in SOURCE_WEIGHT}
        for _, ti, vi in folds:
            ti, vi = ti[train_available[ti]], vi[valid_available[vi]]
            norm = normalize_fit(training, features, ti)
            values = design(training, norm, ti)
            observed = design(validation, norm, vi)
            use = {"direction": np.flatnonzero(y_train[ti] != NEUTRAL), "neutral": np.arange(len(ti))}
            for objective in ESTIMATORS:
                for (reg, source), output in candidates[name, objective].items():
                    model = fit_binary(values, y_train[ti], use[objective], objective, train_sources[ti], reg, source)
                    output[vi] = observed @ np.asarray(model["coef"]) + model["bias"]
        print(f"Validation folds completed for {name}", flush=True)

    # Each binary head selects its settings on held-out collections before joining the posteriors.
    selected = {}
    signs = y_valid != NEUTRAL
    for (name, objective), options in candidates.items():
        def rank(option):
            prediction = options[option]
            use = np.isfinite(prediction) & (signs if objective == "direction" else True)
            labels_for_head = y_valid == (2 if objective == "direction" else NEUTRAL)
            loss = np.logaddexp(0., np.where(labels_for_head, -prediction, prediction))
            return -float(np.mean([loss[group & use].mean() for group in groups if (group & use).any()]))

        selected[name, objective] = max(options, key=rank)
    reports = {}
    for direction in ("omni", "speech_and_omni"):
        for neutral in BLOCKS:
            sign_scores = candidates[direction, "direction"][selected[direction, "direction"]]
            neutral_scores = candidates[neutral, "neutral"][selected[neutral, "neutral"]]
            for penalty in NEUTRAL_COSTS:
                key = direction, neutral, penalty
                values = join_logits(sign_scores, neutral_scores, penalty)
                rank = score_groups(y_valid, values, groups)
                reports[key] = {**metrics(y_valid, values), "mean_collection_accuracy": float(rank[0])}
    chosen = max(reports, key=lambda key: (reports[key]["mean_collection_accuracy"], reports[key]["macro_f1"]))
    heads = {}
    ti = np.flatnonzero(train_available)
    for name, objective in zip(chosen[:2], ESTIMATORS, strict=True):
        reg, source = selected[name, objective]
        features = BLOCKS[name]
        norm = normalize_fit(training, features, ti)
        values = design(training, norm, ti)
        use = np.flatnonzero(y_train[ti] != NEUTRAL) if objective == "direction" else np.arange(len(ti))
        model = fit_binary(values, y_train[ti], use, objective, train_sources[ti], reg, source)
        heads[objective] = {**model, "features": features, "C": reg, "source_power": source,
                            "normalizer": {name: [p.tolist() for p in parts] for name, parts in norm.items()}}
    model = cache / "audio-factorized.json"
    save_json(model, {"code_sha256": sha_file(Path(__file__)), "extractors": train["models"],
                      "meta_sha256": train["meta_sha256"], "heads": heads, "choice": chosen,
                      "splits": {s: sha_file(cache / f"evaluation-{s}.json") for s in ("train", "valid")},
                      "validation": [{"choice": key, **val} for key, val in reports.items()]})
    print(f"Frozen direction / neutral heads {chosen}: {reports[chosen]['correct']}/{len(y_valid)}", flush=True)


def evaluate(cache, output):
    checkpoint = cache / "audio-factorized.json"
    model = json.loads(checkpoint.read_text("utf-8"))
    if model["code_sha256"] != sha_file(Path(__file__)):
        raise ValueError("Run fit with this version")
    test = read_full_report(cache, "test")
    if model["extractors"] != test["models"] or model["meta_sha256"] != test["meta_sha256"]:
        raise ValueError("The test cache uses different extractors or data")
    rows = test["rows"]
    test_ids, sources = {row["id"] for row in rows}, set()
    for s, digest in model["splits"].items():
        if digest != sha_file(cache / f"evaluation-{s}.json"):
            raise ValueError(f"The {s} report changed after fitting")
        fitting = read_full_report(cache, s)["rows"]
        if test_ids & {row["id"] for row in fitting}:
            raise ValueError(f"Test and {s} IDs overlap")
        sources.update(group_ids(fitting))
    blocks, old = load_blocks(cache, test, ("audio",))
    active = np.flatnonzero(np.isfinite(blocks["audio"]["acoustic"][:, 0]))
    logits = {}
    for name, head in model["heads"].items():
        norm = {name: tuple(np.asarray(p, np.float32) for p in parts) for name, parts in head["normalizer"].items()}
        values = design(blocks["audio"], norm, active)
        logits[name] = values @ np.asarray(head["coef"]) + head["bias"]
    probability = np.full((len(rows), len(LABELS)), np.nan)
    probability[active] = join_logits(logits["direction"], logits["neutral"], model["choice"][2])
    selected = labels(probability)
    original = json.loads((cache / "teacher-supervision.json").read_text("utf-8"))["heads"]["audio"]
    prior = json.loads((cache / "audio-prior-test-predictions.json").read_text("utf-8"))["predictions"]
    arms = {"prior_correction": [prior[row["id"]]["omni_prior"] for row in rows],
            **{name: head_predictions(old["audio"]["baseline" if name.startswith("desktop") else "teacher"], head)
               for name, head in original.items()}}
    seen = np.isin(group_ids(rows), list(sources))
    groups = {"official": np.arange(len(rows)), "unseen_sources": np.flatnonzero(~seen),
              "audio_text_disagreement": np.flatnonzero([row["truth"]["audio"] != row["truth"]["text"] for row in rows])}
    test_metrics = {}
    for name, indices in groups.items():
        test_metrics[name] = {"metrics": metrics(numbers(rows, "audio"), probability, indices),
                              "comparisons": {key: compare(rows, "audio", guesses, selected, indices)
                                              for key, guesses in arms.items()}}
        part = test_metrics[name]["metrics"]
        print(name, part["correct"], f"{part['macro_f1']:.4f}", part["predicted_neutral"], flush=True)
    save_json(output, {"checkpoint_sha256": sha_file(checkpoint), "test_ids_sha256": test["selected_ids_sha256"],
                       "choice": model["choice"], "groups": test_metrics, "validation": model["validation"]})
    save_json(cache / "audio-factorized-test-predictions.json", {
        "model_sha256": sha_file(checkpoint),
        "predictions": dict(zip((r["id"] for r in rows), selected, strict=True))})


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
            evaluate(args.cache, args.output or args.cache / "audio-factorized-results.json")


if __name__ == "__main__":
    main()
