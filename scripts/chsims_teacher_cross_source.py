"""Calibrate audio and face-aware teachers using validation from held-out video sources."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.special import softmax
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_chsims_teacher import save_json
from scripts.calibrate_chsims_teacher import log_ratios
from scripts.chsims_face_teacher import source_identity
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file
from scripts.chsims_teacher_supervision import (
    classification_metrics, compare, design, fit_classifier, head_predictions,
    normalize_fit, numbers, read_blocks, read_full_report, source_folds, temperature,
)


FAMILIES = {
    "audio": {
        "omni": ("context", "acoustic"),
        "encoder": ("acoustic",),
        "specialist": ("speech", "expert_scores", "expert_quality"),
        "joint": ("context", "acoustic", "speech", "expert_scores", "expert_quality"),
    },
    "visual": {
        "scene": ("context", "frames"),
        "face": ("face_context", "face_frames"),
        "specialist": ("expert_scores", "expert_quality"),
        "joint": ("context", "frames", "face_context", "face_frames", "expert_scores", "expert_quality"),
    },
}
OPTIONAL = ("face_frames", "expert_scores")
SCORE_KEYS = ("accuracy", "macro_f1")
NEUTRAL = LABELS.index("neutral")


def group_ids(rows):
    return np.array([row["id"].split("/")[0] for row in rows])


def held_out_validation(train_groups, valid_groups):
    folds = GroupKFold(n_splits=5)
    for _, held_out in folds.split(np.arange(len(valid_groups)), groups=valid_groups):
        training = np.flatnonzero(~np.isin(train_groups, valid_groups[held_out]))
        yield training, held_out


def load_blocks(cache, report, views):
    old = read_blocks(cache, report)
    rows = report["rows"]
    result = {view: dict(old[view]["teacher"]) for view in views}
    if "audio" in result:
        result["audio"]["speech"] = old["audio"]["baseline"]["speech"]
    for view in result:
        result[view]["expert_scores"] = np.full((len(rows), 2), np.nan, np.float32)
        result[view]["expert_quality"] = np.full((len(rows), 3), np.nan, np.float32)
    identity = source_identity(cache) if "visual" in result else None
    for i, row in enumerate(rows):
        clip = row["id"]
        media = cache / "media" / f"{clip}.npz"
        with np.load(media, allow_pickle=False) as asset:
            info = json.loads(asset["info"].item())
        for view, blocks in result.items():
            if info[view] is not None:
                observation = info[view]
                blocks["expert_scores"][i] = log_ratios(observation)
                blocks["expert_quality"][i] = (observation["quality"], observation["coverage"],
                                                observation["scores"]["uncertain"])
        if identity is None:
            continue
        path = cache / "face" / f"{clip}.npz"
        with np.load(path, allow_pickle=False) as asset:
            face = json.loads(asset["info"].item())
            if face["provenance"] != {"source": sha_file(media), "extractor": identity}:
                raise ValueError(f"Face evidence changed for {clip}")
            active = face["face_frames"] > 0
            expected = {"info", "context", "frames"} if active else {"info"}
            if set(asset.files) != expected or (active and face["prediction"]["seen"] != ["video"]):
                raise ValueError(f"Unexpected face evidence for {clip}")
            for key in ("context", "frames") if active else ():
                values = asset[key]
                if values.ndim != 1 or not np.isfinite(values).all():
                    raise ValueError(f"Invalid face vector for {clip}")
                feature = "face_" + key
                blocks = result["visual"]
                if feature not in blocks:
                    blocks[feature] = np.full((len(rows), len(values)), np.nan, np.float32)
                blocks[feature][i] = values
    return result, old


def supported(blocks, names):
    return np.logical_or.reduce([np.isfinite(blocks[name][:, 0]) for name in names])


def features(blocks, normalizer, indices):
    values = design(blocks, normalizer, indices)
    flags = [np.isfinite(blocks[name][indices, :1]) for name in OPTIONAL if name in normalizer]
    return np.column_stack((values, *flags)) if flags else values


def source_directions(x, y, groups, rank):
    """Find source offsets after removing class means, using fitting sources only."""
    residuals = x.copy()
    for label in range(len(LABELS)):
        member = y == label
        residuals[member] -= x[member].mean(0)
    means = []
    for source in np.unique(groups):
        members = residuals[groups == source]
        means.append(members.sum(0) / (len(members) + 8))
    means = np.stack(means)
    eigenvalues, vectors = np.linalg.eigh(means @ means.T)
    kept = np.flatnonzero(eigenvalues > 1e-8)[-rank:]
    return (vectors[:, kept].T @ means) / np.sqrt(eigenvalues[kept, None])


def remove_source_directions(x, basis):
    return x - (x @ basis.T) @ basis if basis.size else x


def prior_adjustment(truth, balance, correction):
    counts = np.bincount(truth, minlength=len(LABELS))
    if (counts == 0).any():
        raise ValueError("Every training partition must contain all three classes")
    return balance * correction * np.log(counts / counts.sum())


def metrics(truth, predictions, selected=None):
    if selected is not None:
        truth, predictions = truth[selected], predictions[selected]
    observed = np.flatnonzero(np.isfinite(predictions[:, 0]))
    outcome = classification_metrics(truth, observed, predictions[observed])
    outcome["n"] = len(truth)
    outcome["predicted_neutral"] = int(np.asarray(outcome["confusion"])[:, NEUTRAL].sum())
    return outcome


def validation_rank(view, truth, predictions, desktop_coverage):
    measured = metrics(truth, predictions)
    accuracy, macro = (measured[key] for key in SCORE_KEYS)
    if view == "visual" and desktop_coverage.any():
        covered = metrics(truth, predictions, desktop_coverage)
        return (.7 * covered["accuracy"] + .3 * accuracy, .7 * covered["macro_f1"] + .3 * macro)
    return (.55 * accuracy + .45 * macro, accuracy)


def choose_head(view, names, training, validation, y_train, y_valid, sources, folds, coverage):
    train_supported, valid_supported = supported(training, names), supported(validation, names)
    ranks = (0, 16) if view == "audio" and "acoustic" in names else (0,)
    balances = (0., .5, 1.) if view == "audio" else (0., .5)
    strengths = (.3, 2., 15.)
    choices = [(rank, strength, balance, correction)
               for rank in ranks for strength in strengths for balance in balances
               for correction in ((0., .5, 1.) if balance and view == "audio" else (0.,))]
    out = {choice: np.full((len(y_valid), len(LABELS)), np.nan) for choice in choices}
    for train_index, valid_index in folds:
        ti = train_index[train_supported[train_index]]
        vi = valid_index[valid_supported[valid_index]]
        normalizer = normalize_fit(training, names, ti)
        train_x = features(training, normalizer, ti)
        valid_x = features(validation, normalizer, vi)
        for rank in ranks:
            basis = source_directions(train_x, y_train[ti], sources[ti], rank) if rank else np.empty((0, 0))
            x = remove_source_directions(train_x, basis)
            z = remove_source_directions(valid_x, basis)
            for strength in strengths:
                for balance in balances:
                    estimator = fit_classifier(x, y_train[ti], strength, balance)
                    logits = estimator.decision_function(z)
                    for correction in ((0., .5, 1.) if balance and view == "audio" else (0.,)):
                        adjustment = prior_adjustment(y_train[ti], balance, correction)
                        out[rank, strength, balance, correction][vi] = logits + adjustment
    winner = max(out, key=lambda choice: (*validation_rank(view, y_valid, out[choice], coverage),
                                           -choice[0], -choice[1], -choice[2], choice[3]))
    logits = out[winner]
    covered = np.flatnonzero(valid_supported)
    if not np.isfinite(logits[covered]).all():
        raise ValueError(f"Incomplete source-held-out predictions for {names}")
    rank, strength, balance, correction = winner
    scale = temperature(logits[covered], y_valid[covered])
    probabilities = np.full_like(logits, np.nan)
    probabilities[covered] = softmax(logits[covered] / scale, axis=1)
    settings = {"features": names, "source_rank": rank, "C": strength, "balance": balance,
                "prior_correction": correction, "temperature": scale,
                "validation": metrics(y_valid, probabilities)}
    print(f"{view} {names}: {settings['validation']['correct']}/{len(y_valid)} "
          f"F1 {settings['validation']['macro_f1']:.4f} params {winner}", flush=True)
    return settings, probabilities


def fit_head(blocks, y, groups, settings, indices):
    names = settings["features"]
    indices = indices[supported(blocks, names)[indices]]
    normalizer = normalize_fit(blocks, names, indices)
    values = features(blocks, normalizer, indices)
    rank = settings["source_rank"]
    basis = source_directions(values, y[indices], groups[indices], rank) if rank else np.empty((0, 0))
    model = fit_classifier(remove_source_directions(values, basis), y[indices], settings["C"], settings["balance"])
    return {**settings, "fitted": len(indices),
            "normalizer": {name: [part.tolist() for part in stats] for name, stats in normalizer.items()},
            "basis": basis.tolist(), "coef": model.coef_.tolist(),
            "bias": (model.intercept_ + prior_adjustment(
                y[indices], settings["balance"], settings["prior_correction"])).tolist()}


def predict_head(blocks, head, indices):
    normalizer = {name: tuple(np.asarray(part, np.float32) for part in head["normalizer"][name])
                  for name in head["features"]}
    active = supported(blocks, head["features"])[indices]
    values = features(blocks, normalizer, indices[active])
    values = remove_source_directions(values, np.asarray(head["basis"]))
    probabilities = np.full((len(indices), len(LABELS)), np.nan)
    if len(values):
        logits = values @ np.asarray(head["coef"]).T + np.asarray(head["bias"])
        probabilities[active] = softmax(logits / head["temperature"], axis=1)
    return probabilities


def mix(probabilities, weights, fallback=None):
    total = np.zeros_like(next(iter(probabilities.values())))
    mass = np.zeros((len(total), 1))
    for name, weight in weights.items():
        pred = probabilities[name]
        present = np.isfinite(pred[:, :1])
        total += weight * np.where(present, pred, 0.)
        mass += weight * present
    result = np.divide(total, mass, out=np.full_like(total, np.nan), where=mass > 0)
    if fallback is not None:
        absent = mass[:, 0] == 0
        result[absent] = fallback[absent]
    return result


def policies(view, predictions, y, coverage):
    def select(options, fallback=None):
        def rank(weights):
            return validation_rank(view, y, mix(predictions, weights, fallback), coverage)

        return max(options, key=rank)

    if view == "audio":
        omni_options = ({"omni": 1.}, {"encoder": 1.}, {"omni": .5, "encoder": .5})
    else:
        omni_options = ({"scene": 1.}, *({"scene": 1 - w, "face": w} for w in (.35, .65, .9)))
    pure = select(omni_options)
    pure_scores = mix(predictions, pure)
    all_options = ({"joint": 1.},
                   *({name: (1 - w) * weight for name, weight in pure.items()} | {"specialist": w}
                     for w in (.25, .5, .75, .9)),
                   *({"joint": 1 - w, "specialist": w} for w in (.45, .75)))
    hybrid = select(all_options, pure_scores)
    return {"pure_omni": pure, "hybrid": hybrid}


def freeze_view(cache, train, valid, view, train_blocks, valid_blocks):
    train_rows, valid_rows = train["rows"], valid["rows"]
    sources, valid_sources = group_ids(train_rows), group_ids(valid_rows)
    folds = list(held_out_validation(sources, valid_sources))
    y_train, y_valid = numbers(train_rows, view), numbers(valid_rows, view)
    coverage = np.array([bool(row["baseline"][view]["scores"]) for row in valid_rows])
    settings, predictions = {}, {}
    for name, features_for_head in FAMILIES[view].items():
        settings[name], predictions[name] = choose_head(view, features_for_head, train_blocks, valid_blocks,
                                                       y_train, y_valid, sources, folds, coverage)
    chosen = policies(view, predictions, y_valid, coverage)
    pure = mix(predictions, chosen["pure_omni"])
    valid_arms = {**predictions, "pure_omni": pure, "hybrid": mix(predictions, chosen["hybrid"], pure)}
    train_indices = np.arange(len(train_rows))
    heads = {name: fit_head(train_blocks, y_train, sources, option, train_indices)
             for name, option in settings.items()}
    record = {"heads": heads, "policies": chosen, "validation_folds": [
        {"train": len(ti), "valid": len(vi), "held_out_sources": sorted(set(valid_sources[vi]))}
        for ti, vi in folds],
        "validation": {name: metrics(y_valid, pred) for name, pred in valid_arms.items()},
        "validation_desktop_coverage": {name: metrics(y_valid, pred, coverage)
                                        for name, pred in valid_arms.items()}}
    folded = np.full((len(train_rows), len(LABELS)), np.nan)
    for fit_indices, held_out in source_folds(train_rows):
        options = chosen["pure_omni"].keys() | chosen["hybrid"].keys()
        scores = {name: predict_head(train_blocks, fit_head(train_blocks, y_train, sources, settings[name],
                                                          fit_indices), held_out) for name in sorted(options)}
        folded[held_out] = mix(scores, chosen["hybrid"], mix(scores, chosen["pure_omni"]))
    target_path = cache / f"cross-source-{view}-targets.json"
    save_json(target_path, {"labels": LABELS, "split": "train", "metrics": metrics(y_train, folded),
                            "rows": [{"id": row["id"], "probabilities": pred.tolist()
                                      if np.isfinite(pred[0]) else None}
                                     for row, pred in zip(train_rows, folded, strict=True)]})
    record["targets_sha256"] = sha_file(target_path)
    return record


def fit(cache, views):
    train, valid = (read_full_report(cache, split) for split in ("train", "valid"))
    if train["models"] != valid["models"] or train["meta_sha256"] != valid["meta_sha256"]:
        raise ValueError("Development reports use different datasets or checkpoints")
    if {row["id"] for row in train["rows"]} & {row["id"] for row in valid["rows"]}:
        raise ValueError("Training and validation IDs overlap")
    tr, va = (load_blocks(cache, report, views)[0] for report in (train, valid))
    for view in views:
        started = perf_counter()
        record = freeze_view(cache, train, valid, view, tr[view], va[view])
        record.update({"script_sha256": sha_file(Path(__file__)), "meta_sha256": train["meta_sha256"],
                       "extractors": train["models"], "split_sha256": {s: sha_file(cache / f"evaluation-{s}.json")
                                                                       for s in ("train", "valid")}})
        save_json(cache / f"cross-source-{view}.json", record)
        print(f"Frozen {view} {record['policies']} in {perf_counter() - started:.1f}s", flush=True)


def labels(probabilities):
    return [LABELS[p.argmax()] if np.isfinite(p[0]) else "abstain" for p in probabilities]


def compare_choices(rows, view, predictions, cohorts):
    references = ("omni_selected", "desktop_calibrated", "specialist")
    if view == "audio":
        references += ("desktop_acoustic",)
    return {name: {candidate: {
        reference: compare(rows, view, predictions[reference], predictions[candidate], selected)
        for reference in references}
        for candidate in ("pure_omni", "hybrid")} for name, selected in cohorts.items()}


def error_audit(rows, view, predictions, cohorts):
    output = {}
    for group, indices in cohorts.items():
        gold = [rows[i]["truth"][view] for i in indices]
        counts = Counter(gold)
        predictions_by_arm = {}
        for name, values in predictions.items():
            guesses = [values[i] for i in indices]
            report = compare(rows, view, values, values, indices)
            if report is not None:
                predictions_by_arm[name] = {
                    **report["candidate"],
                    "prediction_count": dict(Counter(guesses)),
                    "false_neutral_by_label": {label: sum(a == label and b == "neutral"
                                                         for a, b in zip(gold, guesses, strict=True))
                                               for label in ("negative", "positive")},
                }
        output[group] = {"gold_count": dict(counts), "arms": predictions_by_arm}
    return output


def evaluate(cache, output, views):
    test = read_full_report(cache, "test")
    rows = test["rows"]
    fitting_sources = set()
    for split in ("train", "valid"):
        development = read_full_report(cache, split)["rows"]
        if {row["id"] for row in rows} & {row["id"] for row in development}:
            raise ValueError(f"Test and {split} IDs overlap")
        fitting_sources.update(group_ids(development))
    indices = np.arange(len(rows))
    unseen = [i for i, source in enumerate(group_ids(rows)) if source not in fitting_sources]
    blocks, legacy_blocks = load_blocks(cache, test, views)
    legacy_path = cache / "teacher-supervision.json"
    legacy = json.loads(legacy_path.read_text("utf-8"))
    if legacy["meta_sha256"] != test["meta_sha256"] or legacy["extractors"] != test["models"]:
        raise ValueError("Previous teacher and test cache use different data or extractors")
    result = {"test_ids_sha256": test["selected_ids_sha256"], "meta_sha256": test["meta_sha256"],
              "legacy_checkpoint_sha256": sha_file(legacy_path), "views": {}}
    for view in views:
        path = cache / f"cross-source-{view}.json"
        frozen = json.loads(path.read_text("utf-8"))
        if frozen["script_sha256"] != sha_file(Path(__file__)) or frozen["meta_sha256"] != test["meta_sha256"] \
                or frozen["extractors"] != test["models"] or any(
                    digest != sha_file(cache / f"evaluation-{split}.json")
                    for split, digest in frozen["split_sha256"].items()):
            raise ValueError(f"Incompatible frozen {view} configuration")
        legacy_head = legacy["heads"][view]
        predictions = {key: head_predictions(legacy_blocks[view][
            "baseline" if key.startswith("desktop_") else "teacher"], head) for key, head in legacy_head.items()}
        scores = {name: predict_head(blocks[view], head, indices) for name, head in frozen["heads"].items()}
        pure = mix(scores, frozen["policies"]["pure_omni"])
        predictions.update({name: labels(pred) for name, pred in scores.items()})
        predictions["pure_omni"] = labels(pure)
        predictions["hybrid"] = labels(mix(scores, frozen["policies"]["hybrid"], pure))
        cohorts = {"official": indices.tolist(), "unseen_sources": unseen,
                   "old_desktop_coverage": [i for i, pred in enumerate(predictions["desktop_calibrated"])
                                            if pred != "abstain"],
                   "old_desktop_abstained": [i for i, pred in enumerate(predictions["desktop_calibrated"])
                                             if pred == "abstain"]}
        if view == "visual":
            cohorts["desktop_and_face"] = [i for i in cohorts["old_desktop_coverage"]
                                           if predictions["face"][i] != "abstain"]
        else:
            cohorts["audio_text_disagreement"] = [i for i, row in enumerate(rows)
                                                   if row["truth"]["audio"] != row["truth"]["text"]]
        comparisons = compare_choices(rows, view, predictions, cohorts)
        for key, selected in cohorts.items():
            if not selected:
                continue
            target = comparisons[key]["hybrid"]["desktop_calibrated"]["candidate"]
            print(f"{view} {key}: {target['correct']}/{len(selected)} F1 {target['macro_f1']:.4f} "
                  f"neutral {sum(row[NEUTRAL] for row in target['confusion'])}", flush=True)
        save_json(cache / f"cross-source-{view}-predictions.json",
                  {"head_sha256": sha_file(path), "predictions": {
                      row["id"]: {name: preds[i] for name, preds in predictions.items()}
                      for i, row in enumerate(rows)}})
        result["views"][view] = {"head_sha256": sha_file(path), "policies": frozen["policies"],
                                 "targets_sha256": frozen["targets_sha256"],
                                 "validation_folds": frozen["validation_folds"],
                                 "validation": frozen["validation"],
                                 "validation_desktop_coverage": frozen["validation_desktop_coverage"],
                                 "settings": {name: {key: head[key] for key in (
                                     "features", "source_rank", "C", "balance", "prior_correction", "temperature")}
                                              for name, head in frozen["heads"].items()},
                                 "comparisons": comparisons,
                                 "errors": error_audit(rows, view, predictions, cohorts)}
    save_json(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "evaluate"))
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    parser.add_argument("--view", choices=(*FAMILIES, "both"), default="both")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    views = tuple(FAMILIES) if args.view == "both" else (args.view,)
    with threadpool_limits(limits=4, user_api="blas"):
        if args.stage == "fit":
            fit(args.cache, views)
        else:
            evaluate(args.cache, args.output or args.cache / "cross-source-results.json", views)


if __name__ == "__main__":
    main()
