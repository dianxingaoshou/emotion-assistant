"""Fit modality-specific heads on frozen Omni features and export out-of-fold targets."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotion_assistant.foundation_embeddings import AUDIO_DIM
from scripts.benchmark_chsims_teacher import paired_metrics, save_json, scores_from_confusion
from scripts.calibrate_chsims_teacher import log_ratios, read_report
from scripts.chsims_samples import LABEL_SHA256
from scripts.chsims_teacher_data import CACHE, LABELS, VIEWS, ranked_ids, sha_file


SCORE = ("scores",)
CONTEXT = ("context",)
NATIVE = {"text": (), "audio": ("acoustic",), "visual": ("frames",), "full": ("acoustic", "frames")}
INPUT_FOR = {"acoustic": "audio", "frames": "video", "speech": "audio"}
PILOT_TEST_SIZE = 256
FOLD_COUNT = 5
REGULARIZATION = (.1, 1., 10., 100.)
BALANCE = (0., .5, 1.)
ABSTAIN = len(LABELS)


def read_full_report(cache, split):
    path = cache / f"evaluation-{split}.json"
    report = read_report(path, split)
    ids = [row["id"] for row in report["rows"]]
    digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    if len(set(ids)) != len(ids) or len(ids) != report["split_size"]:
        raise ValueError(f"Expected every official {split} clip exactly once")
    if digest != report["selected_ids_sha256"]:
        raise ValueError(f"Unexpected ordering or IDs in {path}")
    if report["models"]["teacher"].get("representations") is not True:
        raise ValueError("Score the teacher with --representations")
    return report


def expected_features(predictions):
    names = set()
    for view, allowed in VIEWS.items():
        prediction = predictions[view]
        seen = set(prediction["seen"])
        if not seen.issubset(allowed) or bool(seen) != bool(prediction["scores"]):
            raise ValueError(f"Invalid input or score provenance in the {view} view")
        if seen:
            names.add(f"{view}_context")
            names.update(f"{view}_{name}" for name in NATIVE[view] if INPUT_FOR[name] in seen)
    return names


def read_blocks(cache, report):
    rows = report["rows"]
    blocks = {view: {arm: {"scores": np.full((len(rows), 2), np.nan, np.float32)}
                     for arm in ("teacher", "baseline")} for view in VIEWS}
    speech_blocks = np.full((len(rows), AUDIO_DIM), np.nan, np.float32)
    blocks["audio"]["baseline"]["speech"] = speech_blocks
    for position, row in enumerate(rows):
        clip = row["id"]
        record = json.loads((cache / "teacher" / f"{clip}.json").read_text("utf-8"))
        path = cache / "representations" / f"{clip}.npz"
        media = cache / "media" / f"{clip}.npz"
        if (record["model"] != report["models"]["teacher"] or record["predictions"] != row["teacher"]
                or record["representations_sha256"] != sha_file(path) or record["media_sha256"] != sha_file(media)):
            raise ValueError(f"Inconsistent representation cache for {clip}")
        with np.load(media, allow_pickle=False) as asset:
            speech = asset["speech"]
        if speech.size:
            if speech.shape != (AUDIO_DIM,) or not np.isfinite(speech).all():
                raise ValueError(f"Invalid desktop speech embedding for {clip}")
            speech_blocks[position] = speech
        with np.load(path, allow_pickle=False) as vectors:
            if set(vectors.files) != expected_features(row["teacher"]):
                raise ValueError(f"Representation would use a missing or masked input for {clip}")
            for name in vectors.files:
                view, block = name.split("_", 1)
                vector = vectors[name]
                if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
                    raise ValueError(f"Invalid {name} for {clip}")
                feature = blocks[view]["teacher"].get(block)
                if feature is None:
                    feature = np.full((len(rows), len(vector)), np.nan, np.float32)
                    blocks[view]["teacher"][block] = feature
                if feature.shape[1] != len(vector):
                    raise ValueError(f"Inconsistent {name} width for {clip}")
                feature[position] = vector
        for view in VIEWS:
            for arm in ("teacher", "baseline"):
                ratios = log_ratios(row[arm][view])
                if ratios is not None:
                    blocks[view][arm]["scores"][position] = ratios
    return blocks


def available(blocks):
    return np.isfinite(blocks["scores"][:, 0])


def normalize_fit(blocks, names, indices):
    parameters = {}
    for name in names:
        block = blocks[name][indices]
        observed = block[np.isfinite(block[:, 0])]
        if not len(observed):
            raise ValueError(f"No fitting inputs for {name}")
        floor = .05 if name == "scores" else 1e-3
        center = observed.mean(axis=0)
        scale = np.maximum(observed.std(axis=0), floor) * np.sqrt(block.shape[1])
        parameters[name] = (center, scale)
    return parameters


def design(blocks, parameters, indices):
    parts = []
    for name, (center, scale) in parameters.items():
        block = blocks[name][indices]
        if block.shape[1] != len(center):
            raise ValueError(f"Unexpected feature width for {name}")
        present = np.isfinite(block[:, 0])
        standardized = np.zeros(block.shape, dtype=np.float32)
        standardized[present] = (block[present] - center) / scale
        parts.append(standardized)
        if name in INPUT_FOR:
            parts.append(present[:, None])
    return np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float64)


def fit_classifier(x, y, strength, balance):
    counts = np.bincount(y, minlength=len(LABELS))
    if (counts == 0).any():
        raise ValueError("Each fitting partition needs all three labels")
    weights = (len(y) / (len(LABELS) * counts[y])) ** balance
    weights /= weights.mean()
    return LogisticRegression(C=strength, max_iter=600).fit(x, y, sample_weight=weights)


def numbers(rows, view):
    return np.array([LABELS.index(row["truth"][view]) for row in rows], dtype=int)


def confusion(truth, predictions):
    return np.bincount(truth * (ABSTAIN + 1) + predictions,
                       minlength=ABSTAIN * (ABSTAIN + 1)).reshape(ABSTAIN, ABSTAIN + 1)


def classification_metrics(truth, indices, logits):
    predictions = np.full(len(truth), ABSTAIN)
    predictions[indices] = logits.argmax(axis=1)
    matrix = confusion(truth, predictions)
    accuracy, macro_f1 = scores_from_confusion(matrix)
    return {"accuracy": float(accuracy), "macro_f1": float(macro_f1), "correct": int(matrix.trace()),
            "abstain": int(matrix[:, -1].sum()), "confusion": matrix.tolist()}


def temperature(logits, y):
    def loss(log_scale):
        probabilities = softmax(logits / np.exp(log_scale), axis=1)
        return -np.log(np.maximum(probabilities[np.arange(len(y)), y], 1e-12)).mean()

    optimum = minimize_scalar(loss, bounds=(np.log(.25), np.log(8.)), method="bounded")
    if not optimum.success:
        raise RuntimeError("Could not calibrate the teacher probabilities")
    return float(np.exp(optimum.x))


def select_head(train, valid, y_train, y_valid, candidates):
    train_index, valid_index = (np.flatnonzero(available(blocks)) for blocks in (train, valid))
    selections = {}
    for names in candidates:
        parameters = normalize_fit(train, names, train_index)
        x_train = design(train, parameters, train_index)
        x_valid = design(valid, parameters, valid_index)
        best = None
        for strength in REGULARIZATION:
            for balance in BALANCE:
                model = fit_classifier(x_train, y_train[train_index], strength, balance)
                logits = model.decision_function(x_valid)
                metrics = classification_metrics(y_valid, valid_index, logits)
                rank = (metrics["macro_f1"], metrics["accuracy"], -len(names), -strength, -balance)
                if best is None or rank > best[0]:
                    best = rank, model, strength, balance, metrics, logits
        rank, model, strength, balance, metrics, logits = best
        print(f"Validation {names}: {metrics['correct']}/{len(y_valid)}, F1 {metrics['macro_f1']:.4f}", flush=True)
        selections[names] = (rank, {
            "features": list(names), "C": strength, "balance_exponent": balance, "fitted": len(train_index),
            "validation": metrics, "temperature": temperature(logits, y_valid[valid_index]),
            "normalizer": {key: [center.tolist(), scale.tolist()] for key, (center, scale) in parameters.items()},
            "coef": model.coef_.tolist(), "intercept": model.intercept_.tolist(),
        })
    return selections


def teacher_candidates(view):
    native = NATIVE[view]
    candidates = (SCORE, CONTEXT, (*SCORE, *CONTEXT))
    if not native:
        return candidates
    if view != "full":
        candidates = (*candidates, native)
    # Full-input candidates also need to use the evidence on a text-only clip.
    return (*candidates, (*SCORE, *native), (*CONTEXT, *native), (*SCORE, *CONTEXT, *native))


def source_folds(rows):
    groups = [row["id"].split("/")[0] for row in rows]
    return list(GroupKFold(n_splits=FOLD_COUNT).split(np.arange(len(rows)), groups=groups))


def out_of_fold(blocks, truth, head, folds):
    scores = np.full((len(truth), len(LABELS)), np.nan)
    active = available(blocks)
    for training, held_out in folds:
        training, held_out = training[active[training]], held_out[active[held_out]]
        if not len(held_out):
            continue
        parameters = normalize_fit(blocks, head["features"], training)
        model = fit_classifier(design(blocks, parameters, training), truth[training], head["C"],
                               head["balance_exponent"])
        scores[held_out] = softmax(model.decision_function(design(blocks, parameters, held_out))
                                   / head["temperature"], axis=1)
    if not np.array_equal(np.isfinite(scores[:, 0]), active):
        raise ValueError("Out-of-fold predictions do not cover all supplied evidence")
    return scores


def fit(cache):
    train, valid = (read_full_report(cache, split) for split in ("train", "valid"))
    if train["models"] != valid["models"] or train["meta_sha256"] != valid["meta_sha256"]:
        raise ValueError("Training and validation must share the data release and frozen extractors")
    train_rows, valid_rows = train["rows"], valid["rows"]
    if {row["id"] for row in train_rows} & {row["id"] for row in valid_rows}:
        raise ValueError("Training and validation IDs overlap")
    train_blocks, valid_blocks = (read_blocks(cache, report) for report in (train, valid))
    folds = source_folds(train_rows)
    targets = [{"id": row["id"], "views": {}} for row in train_rows]
    for fold, (_, held_out) in enumerate(folds):
        for index in held_out:
            targets[index]["fold"] = fold
    heads, selection, target_quality = {}, {}, {}
    for view in VIEWS:
        print(f"Fitting {view}", flush=True)
        y_train, y_valid = numbers(train_rows, view), numbers(valid_rows, view)
        desktop_features = (*SCORE, "speech")
        desktop = select_head(train_blocks[view]["baseline"], valid_blocks[view]["baseline"], y_train, y_valid,
                              (SCORE, desktop_features) if view == "audio" else (SCORE,))
        teacher = select_head(train_blocks[view]["teacher"], valid_blocks[view]["teacher"],
                              y_train, y_valid, teacher_candidates(view))
        _, winner = max(teacher.values(), key=lambda item: item[0])
        heads[view] = {"desktop_calibrated": desktop[SCORE][1], "omni_calibrated": teacher[SCORE][1],
                       "omni_selected": winner}
        if view == "audio":
            heads[view]["desktop_acoustic"] = desktop[desktop_features][1]
        selection[view] = {"+".join(names): {key: candidate[key] for key in
                                            ("C", "balance_exponent", "validation")}
                           for names, (_, candidate) in teacher.items()}
        predictions = out_of_fold(train_blocks[view]["teacher"], y_train, winner, folds)
        covered = np.flatnonzero(np.isfinite(predictions[:, 0]))
        target_quality[view] = classification_metrics(y_train, covered, predictions[covered])
        for index, target in enumerate(targets):
            target["views"][view] = {"seen": train_rows[index]["teacher"][view]["seen"],
                                     "probabilities": predictions[index].tolist()
                                     if np.isfinite(predictions[index, 0]) else None}
    provenance = {split: {"n": report["split_size"], "ids_sha256": report["selected_ids_sha256"],
                          "report_sha256": sha_file(cache / f"evaluation-{split}.json")}
                  for split, report in (("train", train), ("valid", valid))}
    model = cache / "teacher-supervision.json"
    save_json(model, {"protocol_sha256": sha_file(Path(__file__)), "classes": LABELS,
                      "meta_sha256": train["meta_sha256"], "extractors": train["models"],
                      "splits": provenance, "heads": heads, "validation_candidates": selection})
    save_json(cache / "teacher-training-targets.json",
              {"split": "train", "classes": LABELS, "group_by": "video_id", "folds": FOLD_COUNT,
               "ids_sha256": provenance["train"]["ids_sha256"], "head_record_sha256": sha_file(model),
               "quality": target_quality, "rows": targets})
    print("Frozen supervision", sha_file(model), flush=True)


def head_predictions(blocks, head):
    indices = np.flatnonzero(available(blocks))
    parameters = {name: [np.asarray(part, np.float32) for part in head["normalizer"][name]]
                  for name in head["features"]}
    values = design(blocks, parameters, indices)
    logits = values @ np.asarray(head["coef"]).T + np.asarray(head["intercept"])
    probabilities = np.full((len(blocks["scores"]), len(LABELS)), np.nan)
    probabilities[indices] = softmax(logits / head["temperature"], axis=1)
    return [LABELS[int(scores.argmax())] if np.isfinite(scores[0]) else "abstain"
            for scores in probabilities]


def compare(rows, view, first, second, indices):
    entries = [(rows[i]["id"], rows[i]["truth"], {view: {"label": first[i]}}, {view: {"label": second[i]}})
               for i in indices]
    comparison = paired_metrics(entries, view)
    if comparison is None:
        return None
    comparison["reference"] = comparison.pop("baseline")
    comparison["candidate"] = comparison.pop("teacher")
    comparison["candidate_minus_reference_95_source_bootstrap"] = comparison.pop(
        "teacher_minus_baseline_95_source_bootstrap")
    return comparison


def historical_ids(v1_labels):
    if sha_file(v1_labels) != LABEL_SHA256:
        raise ValueError("Expected the pinned CH-SIMS v1 label file")
    with v1_labels.open(encoding="utf-8-sig", newline="") as stream:
        return {f"{row[0]}/{row[1]}" for row in csv.reader(stream) if row and row[-1] == "test"}


def evaluate(cache, output, v1_labels=None):
    checkpoint = cache / "teacher-supervision.json"
    model = json.loads(checkpoint.read_text("utf-8"))
    if model["protocol_sha256"] != sha_file(Path(__file__)) or model["classes"] != list(LABELS):
        raise ValueError("Run fit with this version before evaluating")
    test = read_full_report(cache, "test")
    if model["extractors"] != test["models"] or model["meta_sha256"] != test["meta_sha256"]:
        raise ValueError("Test evidence uses different frozen extractors or dataset")
    rows = test["rows"]
    test_ids = {row["id"] for row in rows}
    fitting_sources = set()
    for split in ("train", "valid"):
        if model["splits"][split]["report_sha256"] != sha_file(cache / f"evaluation-{split}.json"):
            raise ValueError(f"{split} report changed after freezing the heads")
        fitting_ids = {row["id"] for row in read_full_report(cache, split)["rows"]}
        if test_ids & fitting_ids:
            raise ValueError(f"Official {split} and test IDs overlap")
        fitting_sources.update(clip.split("/")[0] for clip in fitting_ids)
    blocks = read_blocks(cache, test)
    targets = json.loads((cache / "teacher-training-targets.json").read_text("utf-8"))
    if targets["head_record_sha256"] != sha_file(checkpoint) or targets["split"] != "train":
        raise ValueError("Training targets do not match the frozen teacher")
    target_ids = [row["id"] for row in targets["rows"]]
    digest = hashlib.sha256("\n".join(target_ids).encode("utf-8")).hexdigest()
    if digest != model["splits"]["train"]["ids_sha256"] or test_ids & set(target_ids):
        raise ValueError("Training targets do not use the expected official IDs")
    pilot_ids = set(ranked_ids({"test": test_ids}, "test")[:PILOT_TEST_SIZE])
    cohorts = {"official_test": list(range(len(rows))),
               "outside_teacher_pilot": [i for i, row in enumerate(rows) if row["id"] not in pilot_ids],
               "unseen_sources": [i for i, row in enumerate(rows) if row["id"].split("/")[0] not in fitting_sources]}
    if v1_labels is not None:
        old_ids = historical_ids(v1_labels)
        cohorts["outside_pilot_and_v1_test"] = [i for i in cohorts["outside_teacher_pilot"]
                                               if rows[i]["id"] not in old_ids]
    results = {"meta_sha256": test["meta_sha256"], "head_record_sha256": sha_file(checkpoint),
               "test_report_sha256": sha_file(cache / "evaluation-test.json"),
               "test_ids_sha256": test["selected_ids_sha256"], "extractors": model["extractors"],
               "splits": {key: value["n"] for key, value in model["splits"].items()} | {"test": len(rows)},
               "inputs_on_test": test["used_teacher_inputs"],
               "cohort_sizes": {name: len(indices) for name, indices in cohorts.items()},
               "validation_candidates": model["validation_candidates"],
               "targets": {"folds_by_video_source": dict(Counter(row["fold"] for row in targets["rows"])),
                           "available": {view: sum(row["views"][view]["probabilities"] is not None
                                                   for row in targets["rows"]) for view in VIEWS},
                           "quality": targets["quality"],
                           "sha256": sha_file(cache / "teacher-training-targets.json")}, "views": {}}
    predictions = {row["id"]: {} for row in rows}
    for view in VIEWS:
        heads = model["heads"][view]
        labels = {"desktop_raw": [row["baseline"][view]["label"] for row in rows],
                  "omni_raw": [row["teacher"][view]["label"] for row in rows]}
        labels.update({name: head_predictions(blocks[view]["baseline" if name.startswith("desktop_") else "teacher"],
                                              head)
                       for name, head in heads.items()})
        for index, row in enumerate(rows):
            predictions[row["id"]][view] = {name: values[index] for name, values in labels.items()}
        selected = labels["omni_selected"]
        comparisons = {cohort: {name: compare(rows, view, reference, selected, indices)
                                for name, reference in labels.items() if name != "omni_selected"}
                       for cohort, indices in cohorts.items()}
        both = [i for i, label in enumerate(labels["desktop_calibrated"])
                if label != "abstain" and selected[i] != "abstain"]
        comparisons["same_coverage"] = compare(rows, view, labels["desktop_calibrated"], selected, both)
        record = {"heads": {name: {key: head[key] for key in ("features", "C", "balance_exponent", "fitted",
                                                              "validation", "temperature")}
                            for name, head in heads.items()}, "comparisons": comparisons}
        if view == "audio":
            disagreement = [i for i, row in enumerate(rows) if row["truth"]["audio"] != row["truth"]["text"]]
            record["audio_text_label_disagreement"] = {
                "comparison": compare(rows, view, labels["desktop_calibrated"], selected, disagreement),
                "matched_text_label": {name: sum(values[i] == rows[i]["truth"]["text"] for i in disagreement)
                                       for name, values in labels.items()}}
        if view == "full":
            complete = [i for i, row in enumerate(rows) if set(row["teacher"][view]["seen"]) == set(VIEWS[view])]
            record["all_inputs_present"] = compare(rows, view, labels["desktop_calibrated"], selected, complete)
        results["views"][view] = record
        measured = comparisons["official_test"]["desktop_calibrated"]
        print(view, heads["omni_selected"]["features"], measured["candidate"]["correct"],
              f"{measured['candidate']['macro_f1']:.4f}", flush=True)
    save_json(cache / "supervision-test-predictions.json",
              {"classes": LABELS, "model_sha256": sha_file(checkpoint), "predictions": predictions})
    save_json(output, results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "evaluate"))
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--v1-labels", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=4, user_api="blas"):
        if args.stage == "fit":
            fit(args.cache)
        else:
            evaluate(args.cache, args.output or args.cache / "supervision-test-results.json", args.v1_labels)


if __name__ == "__main__":
    main()
