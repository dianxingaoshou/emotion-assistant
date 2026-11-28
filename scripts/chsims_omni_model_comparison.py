"""Compare two native multimodal audio encoders in the same source-held-out protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_minicpmo_audio import ASSETS as MINI_ASSETS
from scripts.chsims_teacher_cross_source import group_ids, labels, load_blocks, metrics
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file
from scripts.chsims_teacher_supervision import (
    compare, head_predictions, normalize_fit, numbers, read_full_report, temperature,
)
from scripts.chsims_audio_prior import audio_logits
from scipy.special import softmax


TEXT_DESK = ("text_desk",)
AUDIO_DESK = ("audio_desk", "audio_speech", "audio_expert_scores", "audio_expert_quality")
VISUAL_DESK = ("visual_desk", "visual_expert_scores", "visual_expert_quality")
AUDIO_QWEN = ("audio_context", "audio_acoustic")
AUDIO_MINI = ("audio_mini_middle", "audio_mini_native")
QWEN_TEXT_VISUAL = ("text_context", "visual_context", "visual_frames", "visual_face_context", "visual_face_frames")
DESKTOP = (*TEXT_DESK, *AUDIO_DESK, *VISUAL_DESK)
ARMS = {
    "audio": {"desktop_matched": AUDIO_DESK, "qwen": AUDIO_QWEN, "minicpm": AUDIO_MINI,
              "qwen_and_desktop": (*AUDIO_DESK, *AUDIO_QWEN),
              "minicpm_and_desktop": (*AUDIO_DESK, *AUDIO_MINI)},
    "full": {"desktop_matched": DESKTOP, "qwen": (*QWEN_TEXT_VISUAL, *AUDIO_QWEN),
             "qwen_and_desktop": (*DESKTOP, *QWEN_TEXT_VISUAL, *AUDIO_QWEN),
             "minicpm_audio_and_desktop": (*DESKTOP, *QWEN_TEXT_VISUAL, *AUDIO_MINI)},
}
CONDITIONS = {"audio": ("all",), "full": ("all", "text", "audio", "visual")}
NEUTRAL = LABELS.index("neutral")


def read_minicpm(cache, rows):
    vectors = {"audio_mini_middle": np.full((len(rows), 2048), np.nan, np.float32),
               "audio_mini_native": np.full((len(rows), 2048), np.nan, np.float32)}
    for i, row in enumerate(rows):
        key = row["id"]
        with np.load(cache / "media" / f"{key}.npz", allow_pickle=False) as media:
            pcm = media["pcm"]
        with np.load(cache / "minicpmo" / f"{key}.npz", allow_pickle=False) as entry:
            info = json.loads(entry["info"].item())
            if (info["model"]["assets"] != MINI_ASSETS
                    or info["audio_sha256"] != hashlib.sha256(np.ascontiguousarray(pcm)).hexdigest()):
                raise ValueError(f"MiniCPM-o evidence changed for {key}")
            if len(pcm) >= 400:
                for name, place in vectors.items():
                    value = entry[name.removeprefix("audio_")]
                    if value.shape != (2048,) or not np.isfinite(value).all():
                        raise ValueError(f"Invalid MiniCPM-o feature for {key}")
                    place[i] = value
    return vectors


def development_inputs(cache, report):
    new, old = load_blocks(cache, report, ("audio", "visual"))
    cleaned = {"text_desk": old["text"]["baseline"]["scores"], "text_context": old["text"]["teacher"]["context"],
               "audio_desk": old["audio"]["baseline"]["scores"], "visual_desk": old["visual"]["baseline"]["scores"]}
    for view, names in (("audio", ("context", "acoustic", "speech", "expert_scores", "expert_quality")),
                        ("visual", ("context", "frames", "face_context", "face_frames", "expert_scores", "expert_quality"))):
        cleaned.update({f"{view}_{name}": new[view][name] for name in names})
    cleaned.update(read_minicpm(cache, report["rows"]))
    return cleaned


def matrix(blocks, scales, indices, remove):
    arrays, active = [], np.zeros(len(indices), dtype=bool)
    for name, (center, scale) in scales.items():
        part = blocks[name][indices]
        visible = np.isfinite(part[:, 0]) & (name.split("_", 1)[0] != remove)
        normalized = np.zeros_like(part)
        normalized[visible] = (part[visible] - center) / scale
        arrays.extend((normalized, visible[:, None]))
        active |= visible
    return np.column_stack(arrays).astype(np.float64, copy=False), active


def fitting_data(blocks, names, indices, cases, y, sources):
    scales = normalize_fit(blocks, names, indices)
    _, ownership, members = np.unique(sources[indices], return_inverse=True, return_counts=True)
    label_counts = np.bincount(y[indices], minlength=3)
    weights = members[ownership] ** -.25 * label_counts[y[indices]] ** -.25
    arrays, labels_for_cases, case_weights = [], [], []
    for case in cases:
        values, has_evidence = matrix(blocks, scales, indices, case)
        arrays.append(values[has_evidence])
        labels_for_cases.append(y[indices[has_evidence]])
        factor = 1. if case == "all" else .35
        case_weights.append(factor * weights[has_evidence])
    fit_weight = np.concatenate(case_weights)
    return scales, np.concatenate(arrays), np.concatenate(labels_for_cases), fit_weight / fit_weight.mean()


def posterior(values, scale, shift=0.):
    shifted = values / scale
    shifted[:, NEUTRAL] += shift
    return softmax(shifted, axis=1)


def choose(candidates, y, view, baseline):
    def quality(score):
        return .55 * score["macro_f1"] + .45 * score["accuracy"]

    minimum_neutral = baseline["confusion"][NEUTRAL][NEUTRAL]
    by_setting = {}
    for strength, cases in candidates.items():
        logits = cases["all"]
        observed = np.isfinite(logits[:, 0])
        scale = temperature(logits[observed], y[observed])
        for shift in (-.25, 0., .25, .5, .75):
            measures = {case: metrics(y, posterior(value, scale, shift)) for case, value in cases.items()}
            full = measures["all"]
            aggregate = quality(full) if view == "audio" else .7 * quality(full) + .3 * np.mean([
                quality(measures[case]) for case in ("text", "audio", "visual")])
            score = (full["confusion"][NEUTRAL][NEUTRAL] >= minimum_neutral, aggregate, -abs(shift), -strength)
            values = {"C": strength, "temperature": scale, "neutral_shift": shift, "validation": measures,
                      "validation_neutral_floor": minimum_neutral}
            by_setting[strength, shift] = (score, values)
    return max(by_setting.values(), key=lambda item: item[0])[1]


def train_estimator(x, y, sample_weight, strength):
    return LogisticRegression(C=strength, max_iter=500).fit(x, y, sample_weight=sample_weight)


def fit_arm(view, names, train, valid, y, vy, sources, folds, baseline):
    conditions = CONDITIONS[view]
    options = {c: {case: np.full((len(vy), 3), np.nan) for case in conditions} for c in (.3, 2., 10.)}
    for ti, vi in folds:
        scales, x, target, weights = fitting_data(train, names, ti, conditions, y, sources)
        for strength, candidates in options.items():
            model = train_estimator(x, target, weights, strength)
            for case, predictions in candidates.items():
                z, active = matrix(valid, scales, vi, case)
                if active.any():
                    predictions[vi[active]] = model.decision_function(z[active])
    method = choose(options, vy, view, baseline)
    scales, x, target, weights = fitting_data(train, names, np.arange(len(y)), conditions, y, sources)
    model = train_estimator(x, target, weights, method["C"])
    return {**method, "features": names, "scales": {name: [p.tolist() for p in values]
                                                   for name, values in scales.items()},
            "weights": model.coef_.tolist(), "bias": model.intercept_.tolist()}


def fit(cache, views):
    train, valid = (read_full_report(cache, name) for name in ("train", "valid"))
    if train["meta_sha256"] != valid["meta_sha256"] or train["models"] != valid["models"]:
        raise ValueError("Training and validation use different sources")
    if {r["id"] for r in train["rows"]} & {r["id"] for r in valid["rows"]}:
        raise ValueError("Training and validation IDs overlap")
    sources, sources_v = group_ids(train["rows"]), group_ids(valid["rows"])
    folds = [(np.flatnonzero(~np.isin(sources, sources_v[vi])), vi)
             for _, vi in GroupKFold(n_splits=3).split(sources_v, groups=sources_v)]
    tr, va = (development_inputs(cache, report) for report in (train, valid))
    legacy = json.loads((cache / "teacher-supervision.json").read_text("utf-8"))["heads"]
    for view in views:
        y, vy = numbers(train["rows"], view), numbers(valid["rows"], view)
        reference = legacy[view]["desktop_calibrated"]
        raw = {"scores": va["audio_desk"]} if view == "audio" else {
            "scores": np.array([full_ratio(row["baseline"]["full"]) for row in valid["rows"]])}
        baseline = metrics(vy, posterior(audio_logits(raw, reference, np.arange(len(vy))), reference["temperature"]))
        fitted = {}
        for name, features in ARMS[view].items():
            model = fit_arm(view, features, tr, va, y, vy, sources, folds, baseline)
            fitted[name] = model
            summary = model["validation"]["all"]
            print(f"{view} {name}: valid {summary['correct']}/{len(vy)} F1 {summary['macro_f1']:.4f} "
                  f"neutral {summary['confusion'][NEUTRAL][NEUTRAL]}", flush=True)
        path = cache / f"omni-comparison-{view}.json"
        save_json(path, {"code": sha_file(Path(__file__)), "meta": train["meta_sha256"], "frozen_extractors": train["models"],
                         "splits": {s: sha_file(cache / f"evaluation-{s}.json") for s in ("train", "valid")},
                         "mini": MINI_ASSETS, "heads": fitted})
        print("Frozen comparison", view, sha_file(path), flush=True)


def full_ratio(record):
    from scripts.calibrate_chsims_teacher import log_ratios

    value = log_ratios(record)
    return value if value is not None else np.full(2, np.nan)


def predict(model, blocks, case):
    scales = {name: tuple(np.asarray(p, np.float32) for p in m) for name, m in model["scales"].items()}
    values, active = matrix(blocks, scales, np.arange(len(next(iter(blocks.values())))), case)
    logits = np.full((len(active), 3), np.nan)
    logits[active] = values[active] @ np.asarray(model["weights"]).T + np.asarray(model["bias"])
    return posterior(logits, model["temperature"], model["neutral_shift"])


def evaluate(cache, views, output):
    test = read_full_report(cache, "test")
    rows = test["rows"]
    test_ids = {r["id"] for r in rows}
    known = set()
    for split in ("train", "valid"):
        development = read_full_report(cache, split)["rows"]
        if test_ids & {r["id"] for r in development}:
            raise ValueError(f"Test and {split} overlap")
        known.update(group_ids(development))
    unseen = np.flatnonzero(~np.isin(group_ids(rows), list(known)))
    all_ids = np.arange(len(rows))
    blocks = development_inputs(cache, test)
    ref = json.loads((cache / "teacher-supervision.json").read_text("utf-8"))["heads"]
    report = {"test_ids_sha256": test["selected_ids_sha256"], "views": {}}
    for view in views:
        path = cache / f"omni-comparison-{view}.json"
        checkpoint = json.loads(path.read_text("utf-8"))
        if (checkpoint["code"] != sha_file(Path(__file__)) or checkpoint["meta"] != test["meta_sha256"]
                or checkpoint["frozen_extractors"] != test["models"]
                or any(digest != sha_file(cache / f"evaluation-{split}.json")
                       for split, digest in checkpoint["splits"].items())):
            raise ValueError(f"Incompatible comparison checkpoint for {view}")
        models = checkpoint["heads"]
        y = numbers(rows, view)
        results, all_predictions = {}, {}
        for condition in CONDITIONS[view]:
            predictions = {name: predict(model, blocks, condition) for name, model in models.items()}
            choices = {name: labels(value) for name, value in predictions.items()}
            if condition == "all":
                reference = {"scores": blocks["audio_desk"]} if view == "audio" else {
                    "scores": np.array([full_ratio(row["baseline"]["full"]) for row in rows])}
                choices["desktop_raw"] = [row["baseline"][view]["label"] for row in rows]
                choices["desktop_calibrated"] = head_predictions(reference, ref[view]["desktop_calibrated"])
            cohorts = {"official": all_ids, "unseen_sources": unseen}
            all_predictions[condition] = choices
            results[condition] = {}
            for group, indices in cohorts.items():
                compare_to = ["desktop_matched", "qwen"]
                if condition == "all":
                    compare_to += ["desktop_calibrated", "desktop_raw"]
                comparisons = {name: {baseline: compare(rows, view, choices[baseline], guess, indices)
                                      for baseline in compare_to} for name, guess in choices.items()}
                results[condition][group] = {
                    "comparisons": comparisons, "metrics": {name: metrics(y, scores, indices)
                                                             for name, scores in predictions.items()}}
                print(view, condition, group, [(name, part["correct"], round(part["macro_f1"], 4))
                                               for name, part in results[condition][group]["metrics"].items()], flush=True)
        save_json(cache / f"omni-comparison-{view}-test-predictions.json", {
            "ids": [r["id"] for r in rows], "predictions": all_predictions})
        report["views"][view] = {"results": results, "head_sha256": sha_file(path),
                                 "heads": {name: {k: head[k] for k in ("C", "neutral_shift", "temperature", "validation")}
                                           for name, head in models.items()}}
    save_json(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("fit", "evaluate"))
    parser.add_argument("--view", choices=("audio", "full", "both"), default="both")
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    views = ("audio", "full") if args.view == "both" else (args.view,)
    with threadpool_limits(limits=4, user_api="blas"):
        if args.step == "fit":
            fit(args.cache, views)
        else:
            evaluate(args.cache, views, args.output or args.cache / "omni-model-comparison.json")


if __name__ == "__main__":
    main()
