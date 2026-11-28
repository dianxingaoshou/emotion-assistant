"""Train and evaluate evidence-specific CH-SIMS fusion heads under missing inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_chsims_teacher import CLASSES, save_json, scores_from_confusion
from scripts.chsims_omni_model_comparison import ARMS, DESKTOP as DESKTOP_FEATURES, matrix, predict as old_predict
from scripts.chsims_teacher_cross_source import group_ids, labels, load_blocks, metrics, mix
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file
from scripts.chsims_teacher_supervision import compare, normalize_fit, numbers, read_full_report, temperature
from scripts.prepare_chsims_teacher_gaps import DESKTOP, FEATURES, FRACTION, PREPARED, SEED


CHANNEL = {"text": 1, "audio": 2, "visual": 4}
WHOLE = {"complete": 7, "no_text": 6, "no_audio": 5, "no_visual": 3,
         "text_only": 1, "audio_only": 2, "visual_only": 4, "no_evidence": 0}
PARTIAL = {"text_gap": ("text",), "audio_gap": ("audio",), "visual_gap": ("visual",),
           "audio_visual_gap": ("audio", "visual"), "synchronous_gap": ("joint_text", "audio", "visual")}
CHANNEL_FOR = {"joint_text": "text", "text": "text", "audio": "audio", "visual": "visual"}
HEADS = {"desktop": DESKTOP_FEATURES, "teacher": ARMS["full"]["qwen_and_desktop"]}
STRENGTHS = (.15, .75, 3.)
NEUTRAL = LABELS.index("neutral")
RELIABILITY_COVERAGE = .95


def modality(name):
    return name.split("_", 1)[0]


def block_names(names, mask):
    return tuple(name for name in names if CHANNEL[modality(name)] & mask)


def observed_mask(blocks):
    return sum(bit * np.isfinite(blocks[f"{name}_context"][:, 0]) for name, bit in CHANNEL.items()).astype(np.uint8)


def original_blocks(cache, report):
    extra, original = load_blocks(cache, report, ("audio", "visual"))
    clean = {"text_desk": original["text"]["baseline"]["scores"],
             "text_context": original["text"]["teacher"]["context"]}
    for view, blocks in extra.items():
        clean[f"{view}_desk"] = original[view]["baseline"]["scores"]
        names = [name for name in blocks if f"{view}_{name}" in HEADS["teacher"]]
        clean.update({f"{view}_{name}": blocks[name] for name in names})
    return clean


def read_gap_blocks(cache, report, clean):
    n = len(report["rows"])
    partial = {name: np.full_like(value, np.nan) for name, value in clean.items()}
    alternative_text = {name: np.full_like(clean[name], np.nan) for name in ("text_context", "text_desk")}
    proportions = np.zeros((n, 3), dtype=float)
    reencoded, desktop_versions, prepared_versions = set(), set(), set()
    face_names = {"face_context": "visual_face_context", "face_frames": "visual_face_frames"}
    for i, row in enumerate(report["rows"]):
        key = row["id"]
        source = cache / "media" / f"{key}.npz"
        prepared = cache / PREPARED / f"{key}.npz"
        stamp = sha_file(prepared)
        with np.load(prepared, allow_pickle=False) as entry:
            gap = json.loads(entry["info"].item())
        if gap["provenance"]["input"] != sha_file(source):
            raise ValueError(f"Original input changed for {key}")
        prepared_versions.add(json.dumps({k: v for k, v in gap["provenance"].items() if k != "input"}, sort_keys=True))
        for j, name in enumerate(("text", "pcm", "frames")):
            total = gap[f"{name}_total"]
            proportions[i, j] = gap[f"{name}_lost"] / total if total else np.nan
        with np.load(cache / FEATURES / f"{key}.npz", allow_pickle=False) as entry:
            info = json.loads(entry["info"].item())
            for view, seen in info["seen"].items():
                expected = "video" if view in ("visual", "face") else CHANNEL_FOR[view]
                if seen != [expected]:
                    raise ValueError(f"Gap encoding used invalid evidence for {key}")
            model = info["provenance"]["model"]
            if info["provenance"]["input"] != stamp or model != {
                    k: v for k, v in report["models"]["teacher"].items() if k != "representations"}:
                raise ValueError(f"Gap teacher provenance differs for {key}")
            for field in entry.files:
                if field == "info":
                    continue
                if field == "joint_text_context":
                    alternative_text["text_context"][i] = entry[field]
                else:
                    partial[face_names.get(field, field)][i] = entry[field]
            reencoded.add(json.dumps(model, sort_keys=True))
        with np.load(cache / DESKTOP / f"{key}.npz", allow_pickle=False) as entry:
            stamp_info = json.loads(entry["info"].item())["provenance"]
            if (stamp_info["input"] != stamp or stamp_info["asset"] != report["models"]["baseline"]["classifier_sha256"]
                    or stamp_info["text_digest"] != report["models"]["baseline"]["embedding_digest"]):
                raise ValueError(f"Gap desktop provenance differs for {key}")
            desktop_versions.add(json.dumps({k: v for k, v in stamp_info.items() if k != "input"}, sort_keys=True))
            for field in entry.files:
                if field == "info":
                    continue
                if field == "joint_text_desk":
                    alternative_text["text_desk"][i] = entry[field]
                else:
                    partial[field][i] = entry[field]
    if not all(len(versions) == 1 for versions in (prepared_versions, reencoded, desktop_versions)):
        raise ValueError("Each split must use one gap and extractor protocol")
    identity = {name: json.loads(next(iter(versions))) for name, versions in (
        ("prepare", prepared_versions), ("omni", reencoded), ("desktop", desktop_versions))}
    return partial, alternative_text, proportions, identity


def inputs(cache, report):
    clean = original_blocks(cache, report)
    partial, joint_text, removed, identity = read_gap_blocks(cache, report, clean)
    variants = {"complete": clean}
    for case, corrupted in PARTIAL.items():
        replacements = {CHANNEL_FOR[name] for name in corrupted}
        variants[case] = {name: partial[name] if modality(name) in replacements else value
                          for name, value in clean.items()}
        if "joint_text" in corrupted:
            variants[case].update(joint_text)
    return variants, removed, identity


def applicable_gaps(mask):
    recipes = {}
    for case, changes in PARTIAL.items():
        projected = tuple(name for name in changes if CHANNEL[CHANNEL_FOR[name]] & mask)
        if projected:
            recipes.setdefault(projected, case)
    return tuple(recipes.values())


def sample_weights(y, sources, members):
    _, positions, total = np.unique(sources[members], return_inverse=True, return_counts=True)
    counts = np.bincount(y[members], minlength=len(LABELS))
    return counts[y[members]].astype(float) ** -.28 * total[positions].astype(float) ** -.32


def fit_data(variants, names, mask, y, sources, indices, assignment):
    scales = normalize_fit(variants["complete"], names, indices)
    groups = ("complete", *applicable_gaps(mask))
    pieces, truth, weight = [], [], []
    initial = sample_weights(y, sources, indices)
    for part, case in enumerate(groups):
        selected = indices if case == "complete" else indices[assignment[indices] % (len(groups) - 1) == part - 1]
        if not len(selected):
            continue
        eligible = (observed_mask(variants[case])[selected] & mask) == mask
        selected = selected[eligible]
        if not len(selected):
            continue
        design, available = matrix(variants[case], scales, selected, None)
        kept = selected[available]
        factor = 1. if case == "complete" else .7
        pieces.append(design[available])
        truth.append(y[kept])
        weight.append(initial[np.searchsorted(indices, kept)] * factor)
    weights = np.concatenate(weight)
    return scales, np.concatenate(pieces), np.concatenate(truth), weights / weights.mean()


def calibrated(logits, scale, shift):
    values = logits / scale
    values[:, NEUTRAL] += shift
    return softmax(values, axis=1)


def validation_quality(truth, predictions, weights, sources):
    """Give intact and corrupted evidence consistent influence on model selection."""
    summary, source_accuracy = 0., 0.
    for name, prob in predictions.items():
        stats = metrics(truth, prob)
        summary += weights[name] * (.57 * stats["macro_f1"] + .43 * stats["accuracy"])
        yes = np.isfinite(prob[:, 0]) & (np.where(np.isfinite(prob), prob, -1.).argmax(axis=1) == truth)
        ids, source_ids = np.unique(sources, return_inverse=True)
        by_source = np.bincount(source_ids, weights=yes, minlength=len(ids)) / np.bincount(source_ids)
        source_accuracy += weights[name] * by_source.mean()
    return .86 * summary + .14 * source_accuracy


def choose_head(y, sources, cases, options):
    clean_weight = 1. if len(cases) == 1 else .64
    weights = {name: clean_weight if name == "complete" else (1 - clean_weight) / (len(cases) - 1)
               for name in cases}
    if not any(np.isfinite(values["complete"][:, 0]).any() for values in options.values()):
        raise ValueError("No validation evidence for the fusion head")
    best = None
    for strength, values in options.items():
        complete = values["complete"]
        active = np.isfinite(complete[:, 0])
        scale = temperature(complete[active], y[active])
        for shift in (-.25, 0., .25, .5, .75):
            predictions = {name: calibrated(value, scale, shift) for name, value in values.items()}
            score = validation_quality(y, predictions, weights, sources)
            rank = (score, -abs(shift), -strength)
            if best is None or rank > best[0]:
                best = (rank, {"C": strength, "temperature": scale, "neutral_shift": shift}, predictions)
    _, settings, predictions = best
    return settings, predictions, weights


def estimator(x, y, weights, strength):
    return LogisticRegression(C=strength, max_iter=500).fit(x, y, sample_weight=weights)


def fit_head(variants, names, mask, y, sources, indices, assignment, settings):
    scales, x, targets, weights = fit_data(variants, names, mask, y, sources, indices, assignment)
    model = estimator(x, targets, weights, settings["C"])
    return {**settings, "features": names, "scales": {name: [p.tolist() for p in params]
                                                     for name, params in scales.items()},
            "weights": model.coef_.tolist(), "bias": model.intercept_.tolist(),
            "training_rows": len(targets)}


def learn(mask, family, train, valid, y, vy, sources, validation_sources, folds, assignment):
    names = block_names(HEADS[family], mask)
    cases = ("complete", *applicable_gaps(mask))
    options = {strength: {case: np.full((len(vy), len(LABELS)), np.nan) for case in cases}
               for strength in STRENGTHS}
    for training, held_out in folds:
        scales, x, y_augmented, weights = fit_data(train, names, mask, y, sources, training, assignment)
        tests = {}
        for case in cases:
            present = (observed_mask(valid[case])[held_out] & mask) == mask
            indices = held_out[present]
            values, active = matrix(valid[case], scales, indices, None)
            tests[case] = (indices[active], values[active])
        for strength, candidates in options.items():
            model = estimator(x, y_augmented, weights, strength)
            for case, (indices, data) in tests.items():
                if len(indices):
                    candidates[case][indices] = model.decision_function(data)
    settings, predictions, condition_weights = choose_head(vy, validation_sources, cases, options)
    head = fit_head(train, names, mask, y, sources, np.arange(len(y)), assignment, settings)
    return head, predictions, condition_weights


def blend_choices(teacher, desktop, y, condition_weights, sources):
    candidates = []
    for first in (.25, .5, .75, 1.):
        probabilities = {name: mix({"teacher": value, "desktop": desktop[name]},
                                   {"teacher": first, "desktop": 1. - first}, fallback=value)
                         for name, value in teacher.items()}
        rank = validation_quality(y, probabilities, condition_weights, sources)
        candidates.append((rank, first, probabilities))
    _, weight, selected = max(candidates, key=lambda candidate: candidate[:2])
    confidence = np.concatenate([part.max(1) for part in selected.values()])
    observed = np.isfinite(confidence)
    minimum = float(np.quantile(confidence[observed], 1 - RELIABILITY_COVERAGE))
    return weight, minimum, {name: metrics(y, values) for name, values in selected.items()}


def fit(cache):
    root = cache / "missing-evidence"
    train, valid = (read_full_report(cache, phase) for phase in ("train", "valid"))
    if train["meta_sha256"] != valid["meta_sha256"] or train["models"] != valid["models"]:
        raise ValueError("Development sets use different releases or extractors")
    if {row["id"] for row in train["rows"]} & {row["id"] for row in valid["rows"]}:
        raise ValueError("Training and validation IDs overlap")
    tr, _, extraction = inputs(cache, train)
    va, _, extract_valid = inputs(cache, valid)
    if extraction != extract_valid:
        raise ValueError("Gap preparation differs across development sets")
    y, vy = (numbers(report["rows"], "full") for report in (train, valid))
    sources, validation_sources = (group_ids(report["rows"]) for report in (train, valid))
    folds = [(np.flatnonzero(~np.isin(sources, validation_sources[vi])), vi)
             for _, vi in GroupKFold(n_splits=3).split(vy, groups=validation_sources)]
    assignment = case_assignment(train["rows"])
    started = perf_counter()
    models = {}
    for mask in range(1, 8):
        phase = perf_counter()
        heads, validation = {}, {}
        for family in HEADS:
            heads[family], validation[family], weights = learn(mask, family, tr, va, y, vy, sources,
                                                               validation_sources, folds, assignment)
        amount, threshold, estimates = blend_choices(validation["teacher"], validation["desktop"], vy, weights,
                                                     validation_sources)
        models[str(mask)] = {"heads": heads, "teacher_weight": amount, "confidence_threshold": threshold,
                             "validation": estimates}
        current = estimates["complete"]
        print(f"Mask {mask}: validation {current['correct']}/{current['n']} F1 {current['macro_f1']:.4f}, "
              f"teacher {amount:.2f}, {perf_counter() - phase:.1f}s", flush=True)
    previous = cache / "omni-comparison-full.json"
    model = root / "fusion-heads.json"
    save_json(model, {"implementation": sha_file(Path(__file__)), "metadata": train["meta_sha256"],
                      "splits": {phase: sha_file(cache / f"evaluation-{phase}.json") for phase in ("train", "valid")},
                      "models": train["models"], "gap_extractors": extraction, "previous": sha_file(previous),
                      "labels": LABELS, "heads": models})
    print(f"Frozen missing-evidence fusion in {perf_counter() - started:.1f}s", sha_file(model), flush=True)


def case_assignment(rows):
    return np.array([int.from_bytes(hashlib.sha256(row["id"].encode("ascii")).digest()[:4], "big")
                     for row in rows], dtype=np.uint64)


def robust_predict(variants, mask, model):
    # Route by actual evidence after both artificial and natural losses.
    available = observed_mask(variants) & mask
    shape = (len(available), len(LABELS))
    result = {name: np.full(shape, np.nan) for name in ("desktop", "teacher", "selective")}
    for mask, settings in model["heads"].items():
        indices = np.flatnonzero(available == int(mask))
        if not len(indices):
            continue
        selected = {name: value[indices] for name, value in variants.items()}
        scores = {name: old_predict(head, selected, None) for name, head in settings["heads"].items()}
        weight = settings["teacher_weight"]
        teacher = mix(scores, {"teacher": weight, "desktop": 1 - weight}, fallback=scores["teacher"])
        result["desktop"][indices] = scores["desktop"]
        result["teacher"][indices] = teacher
        credible = np.isfinite(teacher[:, 0]) & (teacher.max(axis=1) >= settings["confidence_threshold"])
        result["selective"][indices[credible]] = teacher[credible]
    return result, available


def export_supervision(cache):
    root = cache / "missing-evidence"
    path = root / "fusion-heads.json"
    model = json.loads(path.read_text("utf-8"))
    if (model["implementation"] != sha_file(Path(__file__))
            or model["splits"]["train"] != sha_file(cache / "evaluation-train.json")):
        raise ValueError("Export requires the frozen fusion configuration and original train split")
    train = read_full_report(cache, "train")
    variants, _, extraction = inputs(cache, train)
    if extraction != model["gap_extractors"]:
        raise ValueError("Training inputs differ from the frozen configuration")
    rows = train["rows"]
    y, sources, assignment = numbers(rows, "full"), group_ids(rows), case_assignment(rows)
    targets = [{"id": row["id"], "conditions": {}} for row in rows]
    conditions = {**WHOLE, **dict.fromkeys(PARTIAL, WHOLE["complete"])}
    counts = dict.fromkeys(conditions, 0)
    for fold, (training, retained) in enumerate(GroupKFold(n_splits=5).split(y, groups=sources)):
        heads = {}
        for mask, params in model["heads"].items():
            heads[mask] = {**params, "heads": {
                name: fit_head(variants, config["features"], int(mask), y, sources, training, assignment, {
                    key: config[key] for key in ("C", "temperature", "neutral_shift")})
                for name, config in params["heads"].items()}}
        for condition, mask in conditions.items():
            blocks = variants.get(condition, variants["complete"])
            estimates, known = robust_predict({name: values[retained] for name, values in blocks.items()},
                                              mask, {"heads": heads})
            for index, actual, values in zip(retained, known, estimates["teacher"], strict=True):
                names = [name for name, flag in CHANNEL.items() if actual & flag]
                scores = values.tolist() if np.isfinite(values[0]) else None
                targets[index]["fold"] = fold
                targets[index]["conditions"][condition] = {"seen": names, "probabilities": scores}
                counts[condition] += scores is not None
        print(f"Supervision fold {fold + 1}/5", flush=True)
    output = root / "training-targets.json"
    save_json(output, {"split": "train", "classes": LABELS, "model": sha_file(path),
                       "grouped_by": "video_id", "available": counts, "rows": targets})
    print(f"Exported {len(rows)} supervised training clips to {output}", flush=True)


def previous_predict(blocks, mask, heads):
    restricted = {name: values if CHANNEL[modality(name)] & mask else np.full_like(values, np.nan)
                  for name, values in blocks.items()}
    return {name: old_predict(heads[arm], restricted, None) for name, arm in (
        ("previous_teacher", "qwen_and_desktop"), ("previous_desktop", "desktop_matched"))}


def report_condition(rows, view, groups, probabilities):
    classes = {name: labels(scores) for name, scores in probabilities.items()}
    values = {}
    for group, indices in groups.items():
        pair = compare(rows, "full", classes["previous_teacher"], classes["teacher"], indices)
        results = {name: metrics(view, scores, indices) for name, scores in probabilities.items()}
        for stats in results.values():
            stats["coverage"] = 1 - stats["abstain"] / stats["n"]
            judged = stats["n"] - stats["abstain"]
            stats["accuracy_if_judged"] = stats["correct"] / judged if judged else None
            stats["recall"] = {label: {"hits": stats["confusion"][i][i], "total": sum(stats["confusion"][i])}
                               for i, label in enumerate(LABELS)}
        values[group] = {"metrics": results, "compared_to_previous_teacher": pair,
                         "compared_to_same_training_desktop": compare(
                             rows, "full", classes["desktop"], classes["teacher"], indices)}
    return values, classes


def average_delta(rows, truth, predictions, conditions, indices, baseline):
    sources = group_ids(rows)[indices]
    unique, source_index = np.unique(sources, return_inverse=True)
    blocks = np.zeros((len(unique), len(conditions), 2, 3, 4), dtype=int)
    class_index = {name: i for i, name in enumerate(CLASSES)}
    gold = truth[indices]
    for condition_index, condition in enumerate(conditions):
        for method_index, method in enumerate((baseline, "teacher")):
            values = np.array([class_index[predictions[condition][method][i]] for i in indices])
            np.add.at(blocks, (source_index, condition_index, method_index, gold, values), 1)
    samples = np.random.default_rng(813).multinomial(len(unique), np.full(len(unique), 1. / len(unique)), size=2000)
    pooled = (samples @ blocks.reshape(len(unique), -1)).reshape(-1, len(conditions), 2, 3, 4)
    averaged = scores_from_confusion(pooled).mean(axis=1)
    intervals = np.quantile(averaged[:, 1] - averaged[:, 0], [.025, .975], axis=0).T
    return dict(zip(("accuracy", "macro_f1"), intervals.tolist(), strict=True))


def evaluate(cache, output):
    root = cache / "missing-evidence"
    path = root / "fusion-heads.json"
    model = json.loads(path.read_text("utf-8"))
    previous = cache / "omni-comparison-full.json"
    if model["implementation"] != sha_file(Path(__file__)) or model["previous"] != sha_file(previous):
        raise ValueError("Use the model frozen by this protocol and the matching previous teacher")
    reference = json.loads(previous.read_text("utf-8"))
    test = read_full_report(cache, "test")
    if model["metadata"] != test["meta_sha256"] or model["models"] != test["models"]:
        raise ValueError("Test data uses a different release or extractor")
    rows = test["rows"]
    test_ids = {row["id"] for row in rows}
    known = set()
    for phase, digest in model["splits"].items():
        if sha_file(cache / f"evaluation-{phase}.json") != digest:
            raise ValueError(f"{phase} changed after freezing")
        development = read_full_report(cache, phase)["rows"]
        if test_ids & {row["id"] for row in development}:
            raise ValueError(f"Test and {phase} overlap")
        known.update(group_ids(development))
    observed = group_ids(rows)
    cohorts = {"official": np.arange(len(rows)), "unseen_sources": np.flatnonzero(~np.isin(observed, list(known)))}
    if not len(cohorts["unseen_sources"]):
        del cohorts["unseen_sources"]
    all_variants, removed, extraction = inputs(cache, test)
    if extraction != model["gap_extractors"]:
        raise ValueError("Gap extraction differs from development")
    truth = numbers(rows, "full")
    predictions, cases = {}, {}
    for case, mask in {**WHOLE, **dict.fromkeys(PARTIAL, WHOLE["complete"])}.items():
        blocks = all_variants[case] if case in PARTIAL else all_variants["complete"]
        estimates, availability = robust_predict(blocks, mask, model)
        estimates.update(previous_predict(blocks, mask, reference["heads"]))
        groups, guesses = report_condition(rows, truth, cohorts, estimates)
        cases[case] = {"groups": groups, "available_masks": np.bincount(availability, minlength=8).tolist()}
        predictions[case] = guesses
        score = groups["official"]["metrics"]
        print(f"{case}: {score['previous_teacher']['correct']} -> {score['teacher']['correct']}/{len(rows)}, "
              f"F1 {score['teacher']['macro_f1']:.4f}, abstain {score['teacher']['abstain']}", flush=True)
    if any(case != "abstain" for values in predictions["no_evidence"].values() for case in values):
        raise ValueError("A method guessed an emotion without evidence")
    summaries, intervals = {}, {}
    for tag, names in (("one_missing", ("no_text", "no_audio", "no_visual")),
                       ("two_missing", ("text_only", "audio_only", "visual_only")),
                       ("partial", tuple(PARTIAL)), ("all_missing_conditions", (*tuple(WHOLE)[1:-1], *PARTIAL))):
        summaries[tag] = {group: {method: {stat: float(np.mean([
            cases[case]["groups"][group]["metrics"][method][stat] for case in names]))
            for stat in ("accuracy", "macro_f1", "coverage")} for method in score}
            for group in cohorts}
        intervals[tag] = {group: {baseline: average_delta(rows, truth, predictions, names, members, baseline)
                                  for baseline in ("previous_teacher", "desktop")}
                          for group, members in cohorts.items()}
    save_json(output, {"model": sha_file(path), "previous": sha_file(previous),
                       "test_ids_sha256": test["selected_ids_sha256"], "meta_sha256": test["meta_sha256"],
                       "sizes": {group: len(indices) for group, indices in cohorts.items()},
                       "gap": {"nominal_fraction": FRACTION, "seed": SEED, "median_actual_fraction": {
                           name: float(np.nanmedian(removed[:, i])) for i, name in enumerate(CHANNEL)}},
                       "head_selection": {mask: {key: settings[key] for key in (
                           "teacher_weight", "confidence_threshold", "validation")}
                                          for mask, settings in model["heads"].items()},
                       "cases": cases, "averages": summaries, "average_delta_95_source_bootstrap": intervals})
    save_json(root / "test-predictions.json", {"ids": [row["id"] for row in rows], "cases": predictions})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("fit", "evaluate", "export"))
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=5, user_api="blas"):
        if args.step == "fit":
            fit(args.cache)
        elif args.step == "export":
            export_supervision(args.cache)
        else:
            evaluate(args.cache, args.output or args.cache / "missing-evidence" / "results.json")


if __name__ == "__main__":
    main()
