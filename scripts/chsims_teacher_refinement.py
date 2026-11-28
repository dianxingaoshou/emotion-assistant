"""Fit and freeze source-validated nonlinear heads for incomplete CH-SIMS evidence."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import softmax
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import chsims_teacher_robustness as previous
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_minicpmo_audio import ASSETS as MINI_ASSETS, GAP_AUDIO, MODEL as MINI_MODEL
from scripts.chsims_omni_model_comparison import ARMS, AUDIO_MINI, matrix, predict as linear_predict, read_minicpm
from scripts.chsims_teacher_cross_source import group_ids, metrics, mix
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file, write_atomic
from scripts.chsims_teacher_supervision import numbers, read_full_report, temperature


DIRECTORY = "teacher-refinement"
FEATURES = {"qwen": previous.HEADS["teacher"], "desktop": previous.HEADS["desktop"],
            "mini_hybrid": ARMS["full"]["minicpm_audio_and_desktop"]}
STRENGTHS = (1., 5., 20.)
NEUTRAL = LABELS.index("neutral")
LANDMARKS_PER_CLASS = 64


def read_gap_minicpm(cache, rows):
    blocks = {name: np.full((len(rows), 2048), np.nan, np.float32) for name in AUDIO_MINI}
    for i, row in enumerate(rows):
        clip = row["id"]
        with np.load(cache / previous.PREPARED / f"{clip}.npz", allow_pickle=False) as media:
            pcm = media["pcm"]
        with np.load(cache / GAP_AUDIO / f"{clip}.npz", allow_pickle=False) as entry:
            info = json.loads(entry["info"].item())
            if (info["model"]["model"] != MINI_MODEL or info["model"]["assets"] != MINI_ASSETS
                    or info["audio_sha256"] != hashlib.sha256(np.ascontiguousarray(pcm)).hexdigest()):
                raise ValueError(f"MiniCPM-o did not encode the matching damaged track for {clip}")
            if len(pcm) >= 400:
                for name, target in blocks.items():
                    vector = entry[name.removeprefix("audio_")]
                    if vector.shape != (2048,) or not np.isfinite(vector).all():
                        raise ValueError(f"Invalid damaged audio representation for {clip}")
                    target[i] = vector
    return blocks


def mini_variants(cache, rows, variants):
    clean = read_minicpm(cache, rows)
    gap = read_gap_minicpm(cache, rows)
    output = {}
    for case, blocks in variants.items():
        audio = gap if "audio" in previous.PARTIAL.get(case, ()) else clean
        # Availability is obtained from MiniCPM-o. Neither Qwen audio vector enters this head.
        output[case] = {**blocks, **audio, "audio_context": audio[AUDIO_MINI[0]]}
    return output


def radial(x, anchors, gamma):
    distances = x @ anchors.T
    distances *= -2
    distances += np.einsum("ij,ij->i", x, x)[:, None]
    distances += np.einsum("ij,ij->i", anchors, anchors)[None, :]
    np.maximum(distances, 0., out=distances)
    return np.exp(-gamma * distances)


def basis_transform(x, transform):
    anchors, gamma = transform["anchors"], transform["gamma"]
    kernels = radial(x, anchors, gamma)
    interactions = (kernels - transform["radial_mean"]) / transform["radial_scale"]
    return np.column_stack((x, interactions))


def fit_transform(x, y, feature_count):
    rng = np.random.default_rng(412)
    members = [np.flatnonzero(y == c) for c in range(len(LABELS))]
    sites = np.concatenate([rng.choice(ids, min(len(ids), LANDMARKS_PER_CLASS), replace=False)
                            for ids in members])
    anchors = x[sites]
    gamma = np.float32(.5 / feature_count)
    kernels = radial(x, anchors, gamma)
    center = kernels.mean(0)
    scale = np.maximum(kernels.std(0), .025) * np.sqrt(np.float32(len(sites)))
    transform = {"anchors": anchors, "gamma": gamma, "radial_mean": center, "radial_scale": scale}
    return transform, np.column_stack((x, (kernels - center) / scale))


def ridge_path(x, targets, weights, strengths):
    weights = weights.astype(np.float32)
    targets = np.eye(len(LABELS), dtype=np.float32)[targets]
    center = np.average(x, axis=0, weights=weights)
    intercept = np.average(targets, axis=0, weights=weights)
    x -= center
    gram = x @ x.T
    centered_targets = targets - intercept
    result = {}
    for strength in strengths:
        fit = gram.copy()
        fit.ravel()[::len(fit) + 1] += strength / weights
        dual = cho_solve(cho_factor(fit, overwrite_a=True, check_finite=False), centered_targets, check_finite=False)
        coefficients = x.T @ dual
        result[strength] = (coefficients, intercept - center @ coefficients)
    return result


def fit_candidates(variants, names, mask, truth, sources, training, augmentation, strengths):
    scales, x, target, weights = previous.fit_data(variants, names, mask, truth, sources, training, augmentation)
    transform, x = fit_transform(x.astype(np.float32), target, len(names))
    parameters = ridge_path(x, target, weights, strengths)
    return scales, transform, parameters


def logits_for(blocks, indices, scales, transform, parameters):
    x, active = matrix(blocks, scales, indices, None)
    x = basis_transform(x[active].astype(np.float32), transform) if active.any() else None
    logits = {strength: x @ coef + intercept if x is not None else np.empty((0, len(LABELS)))
              for strength, (coef, intercept) in parameters.items()}
    return active, logits


def neutral_policy(new, base, weight, offset, conflict_bonus):
    if base is not None:
        scores = mix({"new": new, "base": base}, {"new": weight, "base": 1 - weight},
                     fallback=new if weight == 0 else base)
        new_direction = new[:, 2] - new[:, 0]
        base_direction = base[:, 2] - base[:, 0]
        conflict = (new_direction * base_direction < 0) & (new.argmax(1) != NEUTRAL) & (base.argmax(1) != NEUTRAL)
    else:
        scores, conflict = new.copy(), False
    scores[:, NEUTRAL] *= np.exp(offset + conflict_bonus * conflict)
    return scores / scores.sum(1, keepdims=True)


def validation_summary(truth, probabilities, groups, weights):
    scores = {case: metrics(truth, values) for case, values in probabilities.items()}
    neutral = sum(weights[case] * s["confusion"][NEUTRAL][NEUTRAL] / sum(s["confusion"][NEUTRAL])
                  for case, s in scores.items())
    accuracy = sum(weights[case] * s["accuracy"] for case, s in scores.items())
    macro = sum(weights[case] * s["macro_f1"] for case, s in scores.items())
    rank = previous.validation_quality(truth, probabilities, weights, groups) + .09 * neutral
    return {"quality": rank, "accuracy": accuracy, "macro_f1": macro, "neutral_recall": neutral,
            "complete_accuracy": scores["complete"]["accuracy"]}


def validation_weights(cases):
    if len(cases) == 1:
        return {"complete": 1.}
    return {case: .5 if case == "complete" else .5 / (len(cases) - 1) for case in cases}


def calibrated_candidates(logits, base, settings):
    return {case: neutral_policy(softmax(values / settings["temperature"], axis=1),
                                 None if base is None else base[case], settings["kernel_weight"],
                                 settings["neutral_offset"], settings["neutral_conflict_bonus"])
            for case, values in logits[settings["strength"]].items()}


def choose_policy(options, base, truth, groups):
    cases = tuple(next(iter(options.values())))
    weights = validation_weights(cases)
    original = validation_summary(truth, base, groups, weights) if base is not None else None
    candidate_weights = (0., .25, .5, .75, 1.) if base is not None else (1.,)
    best = None
    for strength, logits in options.items():
        intact = logits["complete"]
        active = np.isfinite(intact[:, 0])
        scale = temperature(3. * intact[active], truth[active]) / 3.
        new = {case: softmax(values / scale, axis=1) for case, values in logits.items()}
        for weight in candidate_weights:
            for offset in (-.3, 0., .25, .5, .75, 1.):
                for bonus in ((0., .7) if 0 < weight < 1. else (0.,)):
                    predicted = {case: neutral_policy(values, None if base is None else base[case],
                                                       weight, offset, bonus) for case, values in new.items()}
                    summary = validation_summary(truth, predicted, groups, weights)
                    # Keep clean and stressed accuracy close to the previous teacher on development data.
                    safe = original is None or (summary["complete_accuracy"] >= original["complete_accuracy"] - .01
                                                and summary["accuracy"] >= original["accuracy"] - .01)
                    rank = (safe, summary["quality"], -abs(offset), -bonus, -weight, strength)
                    if best is None or rank > best[0]:
                        best = rank, {"strength": strength, "temperature": scale, "kernel_weight": weight,
                                      "neutral_offset": offset, "neutral_conflict_bonus": bonus,
                                      "validation": summary, "previous_validation": original}
    return best[1]


def blend_audio_encoders(qwen, mini, share, when_uncertain):
    result = qwen.copy()
    active = np.isfinite(qwen[:, 0]) & np.isfinite(mini[:, 0])
    if when_uncertain:
        qwen_max, mini_max = qwen.max(1), mini.max(1)
        active &= (qwen_max < .65) & (mini_max > qwen_max) & (qwen.argmax(1) != mini.argmax(1))
    result[active] = (1 - share) * qwen[active] + share * mini[active]
    return result


def choose_audio_blend(qwen, mini, truth, groups, mask):
    weights = validation_weights(qwen)
    base = validation_summary(truth, qwen, groups, weights)
    options = [(0., False)]
    if mask & previous.CHANNEL["audio"]:
        options.extend((share, gate) for share in (.15, .30, .50) for gate in (False, True))
    best = None
    for share, conditional in options:
        scores = {case: blend_audio_encoders(values, mini[case], share, conditional) for case, values in qwen.items()}
        measure = validation_summary(truth, scores, groups, weights)
        safe = (measure["complete_accuracy"] >= base["complete_accuracy"] - .01
                and measure["accuracy"] >= base["accuracy"] - .01)
        rank = (safe, measure["quality"], -share, not conditional)
        if best is None or rank > best[0]:
            best = rank, {"mini_share": share, "when_uncertain": conditional, "validation": measure}
    return best[1]


def accept_corrections(base, balanced, alternate, settings):
    if settings["proposal"] == "previous":
        return base.copy()
    proposed, support = (balanced, alternate) if settings["proposal"] == "balanced" else (alternate, balanced)
    rows = np.arange(len(base))
    proposed_labels = proposed.argmax(1)
    margin = base.max(1) - base[rows, proposed_labels]
    active = (np.isfinite(base[:, 0]) & np.isfinite(proposed[:, 0]) & np.isfinite(support[:, 0])
              & (proposed_labels != base.argmax(1)) & (proposed_labels == support.argmax(1))
              & (margin <= settings["max_margin"]))
    result = base.copy()
    result[active] = proposed[active]
    return result


def select_corrections(base, balanced, alternate, truth, groups):
    weights = validation_weights(base)
    reference = validation_summary(truth, base, groups, weights)
    fallback = {"proposal": "previous", "max_margin": 0., "validation": reference}
    best = (reference["quality"], reference["accuracy"], 0.), fallback
    for proposal in ("balanced", "mini_hybrid"):
        for margin in (.025, .05, .10, .20):
            settings = {"proposal": proposal, "max_margin": margin}
            scores = {case: accept_corrections(values, balanced[case], alternate[case], settings)
                      for case, values in base.items()}
            measured = validation_summary(truth, scores, groups, weights)
            if (measured["complete_accuracy"] + 1e-12 < reference["complete_accuracy"]
                    or measured["accuracy"] + 1e-12 < reference["accuracy"]):
                continue
            rank = (measured["quality"], measured["accuracy"], -margin)
            if rank > best[0]:
                best = rank, {**settings, "validation": measured}
    return best[1]


def collect_validation(mask, prior, train, valid, truth, validation_truth, sources, folds, augmentation):
    cases = ("complete", *previous.applicable_gaps(mask))
    shape = (len(validation_truth), len(LABELS))
    past = {case: np.full(shape, np.nan) for case in cases}
    all_logits = {arm: {strength: {case: np.full(shape, np.nan) for case in cases} for strength in STRENGTHS}
                  for arm in FEATURES}
    for training, validation in folds:
        past_heads = {arm: previous.fit_head(train["qwen"], settings["features"], mask, truth, sources,
                                             training, augmentation, {key: settings[key] for key in (
                                                 "C", "temperature", "neutral_shift")})
                      for arm, settings in prior["heads"].items()}
        for case in cases:
            blocks = valid["qwen"][case]
            active = validation[(previous.observed_mask(blocks)[validation] & mask) == mask]
            if len(active):
                selected = {name: part[active] for name, part in blocks.items()}
                scores = {arm: linear_predict(head, selected, None) for arm, head in past_heads.items()}
                weight = prior["teacher_weight"]
                past[case][active] = mix(scores, {"teacher": weight, "desktop": 1 - weight}, fallback=scores["teacher"])
        for arm, features in FEATURES.items():
            names = previous.block_names(features, mask)
            scales, transform, params = fit_candidates(train[arm], names, mask, truth, sources,
                                                        training, augmentation, STRENGTHS)
            for case in cases:
                blocks = valid[arm][case]
                present = validation[(previous.observed_mask(blocks)[validation] & mask) == mask]
                if len(present):
                    active, logits = logits_for(blocks, present, scales, transform, params)
                    for strength, values in logits.items():
                        all_logits[arm][strength][case][present[active]] = values
    return past, all_logits


def save_head(path, names, scales, transform, parameters):
    center, scale = zip(*(scales[name] for name in names), strict=True)
    coef, intercept = parameters
    payload = io.BytesIO()
    np.savez(payload, names=names, center=np.concatenate(center), scale=np.concatenate(scale),
             widths=[len(values) for values in center], coef=coef, bias=intercept, **transform)
    write_atomic(path, payload.getvalue())


def fit(cache):
    path = cache / "missing-evidence/fusion-heads.json"
    baseline = json.loads(path.read_text("utf8"))
    train, valid = (read_full_report(cache, phase) for phase in ("train", "valid"))
    hashes = {phase: sha_file(cache / f"evaluation-{phase}.json") for phase in ("train", "valid")}
    if baseline["implementation"] != sha_file(Path(previous.__file__)) or hashes != baseline["splits"]:
        raise ValueError("Use the frozen previous teacher with its development reports")
    variant_sets = []
    for report in (train, valid):
        variants, _, extraction = previous.inputs(cache, report)
        if extraction != baseline["gap_extractors"]:
            raise ValueError("Development extraction has changed since the prior experiment")
        variant_sets.append({"qwen": variants, "desktop": variants,
                             "mini_hybrid": mini_variants(cache, report["rows"], variants)})
    training, validation = variant_sets
    y, vy = (numbers(report["rows"], "full") for report in (train, valid))
    sources, vs = (group_ids(report["rows"]) for report in (train, valid))
    folds = [(np.flatnonzero(~np.isin(sources, vs[vi])), vi) for _, vi in GroupKFold(3).split(vy, groups=vs)]
    augmentation = previous.case_assignment(train["rows"])
    indices = np.arange(len(y))
    config = {"implementation": sha_file(Path(__file__)), "prior": sha_file(path), "splits": hashes,
              "metadata": train["meta_sha256"], "extractors": extraction, "labels": LABELS,
              "heads": {}, "audio_blends": {}, "corrections": {}}
    for mask in range(1, 8):
        started = perf_counter()
        past, logits = collect_validation(mask, baseline["heads"][str(mask)], training, validation, y, vy,
                                          sources, folds, augmentation)
        config["heads"][str(mask)] = heads = {}
        for arm, features in FEATURES.items():
            settings = choose_policy(logits[arm], past if arm == "qwen" else None, vy, vs)
            names = previous.block_names(features, mask)
            strength = settings["strength"]
            scales, transform, fitted = fit_candidates(training[arm], names, mask, y, sources,
                                                         indices, augmentation, (strength,))
            filename = f"{arm}-{mask}.npz"
            head = cache / DIRECTORY / filename
            save_head(head, names, scales, transform, fitted[strength])
            heads[arm] = {**settings, "artifact": filename, "sha256": sha_file(head)}
            value = settings["validation"]
            print(f"Mask {mask} {arm}: {value['accuracy']:.3f} accuracy {value['macro_f1']:.3f} F1, "
                  f"{value['neutral_recall']:.3f} neutral, weight {settings['kernel_weight']:.2f}, "
                  f"offset {settings['neutral_offset']:.2f}", flush=True)
        qwen, mini = (calibrated_candidates(logits[arm], past if arm == "qwen" else None, heads[arm])
                      for arm in ("qwen", "mini_hybrid"))
        gate = choose_audio_blend(qwen, mini, vy, vs, mask)
        config["audio_blends"][str(mask)] = gate
        balanced = {case: blend_audio_encoders(prob, mini[case], gate["mini_share"], gate["when_uncertain"])
                    for case, prob in qwen.items()}
        correction = select_corrections(past, balanced, mini, vy, vs)
        config["corrections"][str(mask)] = correction
        print(f"Mask {mask} MiniCPM-o share {gate['mini_share']:.2f}, "
              f"conditional {gate['when_uncertain']}, F1 {gate['validation']['macro_f1']:.3f}", flush=True)
        print(f"Mask {mask} correction: {correction['proposal']}, margin {correction['max_margin']:.3f}", flush=True)
        print(f"Mask {mask} frozen in {perf_counter() - started:.1f}s", flush=True)
    save_json(cache / DIRECTORY / "heads.json", config)


def new_predict(cache, variants, requested, configs, baseline):
    available = previous.observed_mask(variants) & requested
    result = np.full((len(available), len(LABELS)), np.nan)
    for mask, settings in configs.items():
        indices = np.flatnonzero(available == int(mask))
        if not len(indices):
            continue
        with np.load(cache / DIRECTORY / settings["artifact"], allow_pickle=False) as model:
            ends = np.cumsum(model["widths"])[:-1]
            scales = dict(zip(model["names"].tolist(), zip(np.split(model["center"], ends),
                              np.split(model["scale"], ends), strict=True), strict=True))
            transform = {key: model[key] for key in ("anchors", "gamma", "radial_mean", "radial_scale")}
            active, logits = logits_for(variants, indices, scales, transform, {0: (model["coef"], model["bias"])})
        values = np.full((len(indices), len(LABELS)), np.nan)
        values[active] = softmax(logits[0] / settings["temperature"], axis=1)
        result[indices] = neutral_policy(values, baseline[indices] if baseline is not None else None,
                                         settings["kernel_weight"], settings["neutral_offset"],
                                         settings["neutral_conflict_bonus"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    args = parser.parse_args()
    with threadpool_limits(limits=6, user_api="blas"):
        fit(args.cache)


if __name__ == "__main__":
    main()
