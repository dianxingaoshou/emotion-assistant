"""Train modality-supervised heads and learn when to trust aligned evidence."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, softmax
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotion_assistant.foundation_embeddings import TEXT_DIGEST
from emotion_assistant.foundation_fusion import predict_foundation_clip
from scripts import chsims_teacher_robustness as prior
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_omni_model_comparison import predict
from scripts.chsims_teacher_cross_source import group_ids, metrics, mix
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file
from scripts.chsims_teacher_supervision import numbers, read_full_report
from scripts.encode_chsims_timeline_visual import VISUAL
from scripts.prepare_chsims_teacher_timeline import DIRECTORY, SEGMENTS, TEXT, text_units


TEXT_FEATURES = ("text_context", "text_embedding", "text_desk")
AUDIO_FEATURES = ("audio_context", "audio_acoustic", "audio_speech", "audio_expert_scores", "audio_desk")
SPECIALISTS = {"text": TEXT_FEATURES, "audio": AUDIO_FEATURES,
               "window_text": ("text_embedding",), "window_audio": ("audio_speech",),
               "window_visual": ("visual_frames",)}
CASES = {"no_visual": ("complete", 3), "synchronous_gap": ("synchronous_gap", 7)}
N_CLASSES = len(LABELS)
NEUTRAL = LABELS.index("neutral")


def text_quality(text, seconds, coverage=1.):
    words = [match.group().lower() for match in text_units(text)]
    if not words:
        return 0.
    repeats = sum(a == b for a, b in zip(words, words[1:])) / len(words)
    density = len(words) / max(.25, seconds)
    rate = min(1., density / 2., 9. / density)
    return float(np.clip(rate * (1. - repeats) * coverage, 0., 1.))


def read_timeline(cache, rows):
    global_stats, windows, embedding = [], [], []
    versions, vision, desktop = set(), set(), []
    for row in rows:
        key = row["id"]
        source = cache / SEGMENTS / f"{key}.npz"
        with (np.load(source, allow_pickle=False) as item,
              np.load(cache / TEXT / f"{key}.npz", allow_pickle=False) as text,
              np.load(cache / VISUAL / f"{key}.npz", allow_pickle=False) as video):
            info = json.loads(item["info"].item())
            receipt = json.loads(text["info"].item())["provenance"]
            if receipt != {"segments": sha_file(source), "embedding": TEXT_DIGEST}:
                raise ValueError(f"Timeline text was extracted from different observations for {key}")
            stamp = info["provenance"]
            raw = cache / "media" / f"{key}.npz"
            raw_gap = sha_file(cache / prior.PREPARED / f"{key}.npz")
            if stamp["media"] != sha_file(raw) or stamp["gap"] != raw_gap:
                raise ValueError(f"Timeline evidence differs from the same-clip protocols for {key}")
            versions.add(json.dumps({k: v for k, v in stamp.items() if k not in ("media", "gap")}, sort_keys=True))
            video_info = json.loads(video["info"].item())
            if video_info["provenance"]["input"] != raw_gap:
                raise ValueError(f"Timeline vision used a different gap for {key}")
            vision.add(json.dumps(video_info["provenance"]["encoder"], sort_keys=True))
            clean = info["clean"]
            voiced = clean["voiced"]
            with np.load(raw, allow_pickle=False) as original:
                speech = original["speech"]
                raw_info = json.loads(original["info"].item())
            decision = predict_foundation_clip(clean["text"], audio=raw_info["audio"], embedding=text["clean"],
                                               audio_embedding=speech if speech.size else None)
            desktop.append([decision["scores"].get(label, np.nan) for label in LABELS])
            global_stats.append([text_quality(clean["text"], voiced), voiced, clean["snr"]])
            embedding.append(np.stack([text[name] for name in ("clean", "damaged", "joint")]))
            local = []
            for index, window in enumerate(info["windows"]):
                seconds = window["voiced"]
                coverage = min(1., window["asr_seconds"] / max(.05, seconds))
                vision_key = f"window-{index}"
                visual = video[vision_key] if vision_key in video else np.full(4096, np.nan, np.float32)
                local.append([text[f"window-{index}"], item["audio_embeddings"][index],
                              visual,
                              [text_quality(window["text"], seconds, coverage), seconds, window["snr"]],
                              [window["end"] - window["start"], video_info["coverage"][index]]])
            windows.append(local)
    if len(versions) != 1 or len(vision) != 1:
        raise ValueError("Select a single timeline evidence version")
    window_arrays = {name: np.asarray([[item[i] for item in pair] for pair in windows])
                     for i, name in enumerate(("text_embedding", "audio_speech", "visual_frames", "quality", "duration"))}
    return {"clean": np.asarray(global_stats), "desktop": np.asarray(desktop), "windows": window_arrays,
            "embeddings": np.stack(embedding),
            "provenance": {"observations": json.loads(versions.pop()), "vision": json.loads(vision.pop())}}


def read_data(cache, report):
    variants, _, receipt = prior.inputs(cache, report)
    timeline = read_timeline(cache, report["rows"])
    for name, blocks in variants.items():
        index = 2 if name == "synchronous_gap" else int(name == "text_gap")
        blocks["text_embedding"] = timeline["embeddings"][:, index]
    return {"variants": variants, "time": timeline, "receipt": receipt}


def fit_specialists(data, rows, groups, indices, augmentation):
    heads = {}
    for arm, features in SPECIALISTS.items():
        modality = arm.removeprefix("window_")
        target = numbers(rows, modality)
        heads[arm] = prior.fit_head(data["variants"], features, prior.CHANNEL[modality], target, groups, indices,
                                    augmentation, {"C": .75, "temperature": 1., "neutral_shift": 0.})
    return heads


def fit_prior(data, rows, frozen, groups, indices, augmentation):
    y = numbers(rows, "full")
    result = {}
    for mask in (3, 7):
        settings = frozen["heads"][str(mask)]
        heads = {name: prior.fit_head(data["variants"], head["features"], mask, y, groups, indices, augmentation,
                                     {key: head[key] for key in ("C", "temperature", "neutral_shift")})
                 for name, head in settings["heads"].items() if settings["teacher_weight"] != 1 or name == "teacher"}
        result[str(mask)] = {**settings, "heads": heads}
    return result


def source_predict(blocks, heads, available):
    values = {name: predict(head, blocks, None) for name, head in heads["heads"].items()}
    weight = heads["teacher_weight"]
    result = mix(values, {name: weight if name == "teacher" else 1 - weight for name in values},
                 fallback=values["teacher"])
    result[~available] = np.nan
    return result


def score_examples(data, specialists, reference, indices):
    results = {}
    for case, (variant, mask) in CASES.items():
        blocks = {k: v[indices] for k, v in data["variants"][variant].items()}
        eligible = (prior.observed_mask(blocks) & mask) == mask
        results[case] = {"reference": source_predict(blocks, reference[str(mask)], eligible),
                         **{channel: predict(specialists[channel], blocks, None) for channel in ("text", "audio")}}
    local = data["time"]["windows"]
    results["synchronous_gap"]["windows"] = {
        name: np.stack([predict(specialists[name], {feature: local[feature][indices, j]}, None)
                        for j in range(2)], axis=1)
        for name in ("window_text", "window_audio", "window_visual") for feature in SPECIALISTS[name]
    }
    return results


def quality_features(text, audio, quality):
    t, a = np.nan_to_num(text), np.nan_to_num(audio)
    text_margin = t.max(1) - np.sort(t, axis=1)[:, -2]
    audio_margin = a.max(1) - np.sort(a, axis=1)[:, -2]
    disagreement = np.abs(t - a).sum(1) * .5
    conflict = np.sign(t[:, 2] - t[:, 0]) != np.sign(a[:, 2] - a[:, 0])
    length = np.log1p(np.clip(quality[:, 1], 0., 10.)) / np.log(11.)
    return np.column_stack((np.ones(len(t)), quality[:, 0], length, quality[:, 2], text_margin - audio_margin,
                            disagreement, conflict * (quality[:, 0] - length)))


def learn_text_weight(text, audio, quality, truth, weights, penalty=.018):
    active = np.isfinite(text[:, 0]) & np.isfinite(audio[:, 0])
    target, sample_weight = truth[active], weights[active]
    x = quality_features(text, audio, quality)[active]
    t, a = text[active, target], audio[active, target]
    sample_weight /= sample_weight.sum()

    def objective(params):
        weight = expit(x @ params)
        probability = np.maximum(weight * t + (1 - weight) * a, 1e-9)
        loss = -(sample_weight * np.log(probability)).sum() + penalty * (params @ params)
        change = -(sample_weight / probability) * (t - a) * weight * (1 - weight)
        return loss, x.T @ change + 2 * penalty * params

    solution = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
    if not solution.success:
        raise RuntimeError("Could not fit text–audio reliability")
    return solution.x


def weighted_probabilities(probabilities, weights):
    available = np.isfinite(probabilities[..., 0])
    active = weights * available
    denominator = active.sum(-1, keepdims=True)
    pooled = (np.nan_to_num(probabilities) * active[..., None]).sum(-2)
    return np.divide(pooled, denominator, out=np.full_like(pooled, np.nan), where=denominator > 0)


def combine_text_audio(text, audio, quality, coefficients):
    weight = expit(quality_features(text, audio, quality) @ coefficients) if coefficients is not None else np.full(len(text), .5)
    weight = np.clip(weight, .015, .985)
    fused = weighted_probabilities(np.stack((text, audio), axis=1), np.column_stack((weight, 1 - weight)))
    return fused, weight


def temporal_probability(windows, local, coefficients):
    text, audio = windows["window_text"], windows["window_audio"]
    visual, quality, duration = windows["window_visual"], local["quality"], local["duration"]
    aggregate, masses = [], []
    for j in range(text.shape[1]):
        ta, weight = combine_text_audio(text[:, j], audio[:, j], quality[:, j], coefficients)
        t, a = np.isfinite(text[:, j, 0]), np.isfinite(audio[:, j, 0])
        if coefficients is None:
            ta_mass = np.ones(len(ta))
            v_mass = np.ones(len(ta))
            time_mass = np.ones(len(ta))
        else:
            transcription = quality[:, j, 0]
            voice = np.sqrt(np.clip(quality[:, j, 1] / 1.4, 0., 1.)) * (.55 + .45 * quality[:, j, 2])
            ta_mass = (weight * transcription * t + (1 - weight) * voice * a) / np.maximum(weight * t + (1 - weight) * a, .015)
            v_mass = duration[:, j, 1]
            time_mass = duration[:, j, 0] * np.maximum(ta_mass, v_mass)
        fused = weighted_probabilities(np.stack((ta, visual[:, j]), axis=1), np.column_stack((ta_mass, v_mass)))
        aggregate.append(fused)
        masses.append(time_mass)
    time_weights = np.column_stack(masses)
    return weighted_probabilities(np.stack(aggregate, axis=1), time_weights), time_weights


def example_features(predictions, timeline, case, coefficients):
    text, audio = predictions["text"], predictions["audio"]
    if case == "no_visual":
        quality = timeline["clean"]
        dynamic, weight = combine_text_audio(text, audio, quality, coefficients)
        supporting = [text, audio, timeline["desktop"]]
        reliability = quality_features(text, audio, quality)
    else:
        dynamic, weight = temporal_probability(predictions["windows"], timeline["windows"], coefficients)
        supporting = [text, audio]
        reliability = np.column_stack((weight.sum(1), np.count_nonzero(weight > 0, axis=1)))
    reference = predictions["reference"]
    active = np.isfinite(reference[:, 0]) & np.isfinite(dynamic[:, 0])
    parts = []
    for probabilities in [reference, dynamic, *supporting]:
        ratios = np.log(np.maximum(np.nan_to_num(probabilities, nan=1. / N_CLASSES), 1e-7))
        parts.append(ratios[:, [0, 2]] - ratios[:, [NEUTRAL]])
    return np.column_stack((*parts, reliability)), reference, active, weight, dynamic


def learn_residual(features, reference, truth, weights, penalty):
    center, scale = features.mean(0), np.maximum(features.std(0), .04)
    x = np.column_stack((np.ones(len(features)), (features - center) / scale))
    baseline = np.log(np.maximum(reference, 1e-8))
    expected = np.eye(N_CLASSES)[truth]
    weights = weights / weights.sum()

    def objective(vector):
        coefficients = vector.reshape(x.shape[1], N_CLASSES)
        probabilities = softmax(baseline + x @ coefficients, axis=1)
        loss = -(weights * np.log(np.maximum(probabilities[np.arange(len(truth)), truth], 1e-9))).sum()
        derivative = x.T @ ((probabilities - expected) * weights[:, None]) + 2 * penalty * coefficients
        return loss + penalty * (vector @ vector), derivative.ravel()

    solution = minimize(objective, np.zeros(x.shape[1] * N_CLASSES), jac=True, method="L-BFGS-B")
    if not solution.success:
        raise RuntimeError("Could not fit evidence correction")
    return {"center": center.tolist(), "scale": scale.tolist(),
            "coefficients": solution.x.reshape(x.shape[1], N_CLASSES).tolist(), "penalty": penalty}


def apply_residual(features, reference, policy, eligible):
    result = reference.copy()
    if policy["share"] == 0 or not eligible.any():
        return result
    x = np.column_stack((np.ones(int(eligible.sum())), (features[eligible] - policy["center"]) / policy["scale"]))
    logits = np.log(np.maximum(reference[eligible], 1e-8)) + policy["share"] * (x @ np.asarray(policy["coefficients"]))
    logits[:, NEUTRAL] += policy["neutral_offset"]
    result[eligible] = softmax(logits, axis=1)
    return result


def validation_rank(truth, predicted, reference, sources, scenario):
    before, after = metrics(truth, reference), metrics(truth, predicted)
    baseline_neutral, neutral = (entry["confusion"][NEUTRAL][NEUTRAL] for entry in (before, after))
    _, source_indices, counts = np.unique(sources, return_inverse=True, return_counts=True)
    hit = np.isfinite(predicted[:, 0]) & (predicted.argmax(1) == truth)
    per_source = np.mean(np.bincount(source_indices, weights=hit) / counts)
    neutral_recall = neutral / max(1, np.count_nonzero(truth == NEUTRAL))
    if scenario == "synchronous_gap":
        quality = .45 * after["accuracy"] + .37 * after["macro_f1"] + .12 * neutral_recall + .06 * per_source
    else:
        quality = .58 * after["accuracy"] + .25 * after["macro_f1"] + .17 * per_source
    admissible = after["accuracy"] >= before["accuracy"] - .003 and neutral >= baseline_neutral
    return (admissible, quality, after["accuracy"]), after


def tune_case(name, train, valid, observations, validation_observations, y, vy, sources, vs, coefficients):
    fit_x, old, active, _, _ = example_features(observations, train["time"], name, coefficients)
    val_x, val_old, val_active, _, _ = example_features(validation_observations, valid["time"], name, coefficients)
    weights = prior.sample_weights(y, sources, np.flatnonzero(active))
    rank, measured = validation_rank(vy, val_old, val_old, vs, name)
    best = (*rank, 0., 0.), {"share": 0., "neutral_offset": 0., "validation": measured}
    for penalty in (.015, .075, .28):
        fitted = learn_residual(fit_x[active], old[active], y[active], weights, penalty)
        for share in (.25, .5, .75, 1.):
            for offset in (-.20, 0., .20, .40):
                policy = {**fitted, "share": share, "neutral_offset": offset}
                probabilities = apply_residual(val_x, val_old, policy, val_active)
                outcome, score = validation_rank(vy, probabilities, val_old, vs, name)
                rank = (*outcome, -abs(offset), -share)
                if rank > best[0]:
                    best = rank, {**policy, "validation": score}
    return {**best[1], "previous_validation": metrics(vy, val_old)}


def protect_directions(reference, candidate, local, margin, corroboration):
    if margin is None:
        return candidate
    indices = np.arange(len(reference))
    old, new = reference.argmax(1), candidate.argmax(1)
    allowed_margin = np.where((new == NEUTRAL) & (old != NEUTRAL), margin * .5, margin)
    available = np.isfinite(reference[:, 0]) & np.isfinite(candidate[:, 0]) & np.isfinite(local[:, 0])
    accept = available & (reference[indices, old] - reference[indices, new] <= allowed_margin)
    if corroboration:
        accept &= local[indices, new] >= local[indices, old]
    return np.where(accept[:, None], candidate, reference)


def tune_protected(name, train, valid, observations, validation_observations, y, vy, sources, vs, weights):
    reference = validation_observations["reference"]
    initial, before = validation_rank(vy, reference, reference, vs, name)
    recall = np.diag(np.asarray(before["confusion"])[:, :N_CLASSES])
    best = (*initial, 0., 0.), {"share": 0., "neutral_offset": 0., "max_margin": None, "corroboration": False,
                               "validation": before, "text_weight": weights[0].tolist()}
    for coefficients in weights:
        x, old, active, _, _ = example_features(observations, train["time"], name, coefficients)
        z, old_valid, available, _, local = example_features(validation_observations, valid["time"], name, coefficients)
        sample = prior.sample_weights(y, sources, np.flatnonzero(active))
        for penalty in (.015, .075):
            fitted = learn_residual(x[active], old[active], y[active], sample, penalty)
            for share in (.25, .5, .75, 1.):
                for offset in (-.2, 0., .15, .3, .4):
                    coefficients_to_apply = {**fitted, "share": share, "neutral_offset": offset}
                    candidate = apply_residual(z, old_valid, coefficients_to_apply, available)
                    for margin in (.07, .15, .28, None):
                        for support in ((False, True) if margin is not None else (False,)):
                            predicted = protect_directions(old_valid, candidate, local, margin, support)
                            rank, score = validation_rank(vy, predicted, old_valid, vs, name)
                            if np.any(np.diag(np.asarray(score["confusion"])[:, :N_CLASSES]) < recall):
                                continue
                            rank = (*rank, -share, -abs(offset))
                            if rank > best[0]:
                                best = rank, {**coefficients_to_apply, "max_margin": margin, "corroboration": support,
                                              "text_weight": coefficients.tolist(), "validation": score}
    return {**best[1], "previous_validation": before}


def reassemble(pieces, count):
    data = {}
    for name, example in pieces[0][1].items():
        if isinstance(example, dict):
            data[name] = reassemble([(rows, recorded[name]) for rows, recorded in pieces], count)
        else:
            data[name] = np.full((count, *example.shape[1:]), np.nan)
            for indices, block in pieces:
                data[name][indices] = block[name]
    return data


def fit(cache):
    old_path = cache / "missing-evidence/fusion-heads.json"
    reference = json.loads(old_path.read_text("utf8"))
    reports = {phase: read_full_report(cache, phase) for phase in ("train", "valid")}
    digests = {phase: sha_file(cache / f"evaluation-{phase}.json") for phase in reports}
    if digests != reference["splits"] or sha_file(Path(prior.__file__)) != reference["implementation"]:
        raise ValueError("Use the original frozen comparison and development reports")
    data = {phase: read_data(cache, report) for phase, report in reports.items()}
    for phase, values in data.items():
        if values["receipt"] != reference["gap_extractors"]:
            raise ValueError(f"The {phase} input differs from the original comparison")
    train, valid = data["train"], data["valid"]
    if train["time"]["provenance"] != valid["time"]["provenance"]:
        raise ValueError("Development timelines require the same extractor")
    rows, validation = (reports[phase]["rows"] for phase in ("train", "valid"))
    sources, vs = group_ids(rows), group_ids(validation)
    all_sources = np.concatenate((sources, vs))
    assignment = prior.case_assignment(rows)
    folds = {phase: {case: [] for case in CASES} for phase in reports}
    for fold, (_, hidden) in enumerate(GroupKFold(4).split(all_sources, groups=all_sources), 1):
        holdout = set(all_sources[hidden])
        stay = np.flatnonzero(~np.isin(sources, list(holdout)))
        specialties = fit_specialists(train, rows, sources, stay, assignment)
        old = fit_prior(train, rows, reference, sources, stay, assignment)
        for phase, groups in (("train", sources), ("valid", vs)):
            indices = np.flatnonzero(np.isin(groups, list(holdout)))
            scores = score_examples(data[phase], specialties, old, indices)
            for case, output in scores.items():
                folds[phase][case].append((indices, output))
        print(f"Fitted modality labels and references excluding sources in fold {fold}/4", flush=True)
    oof = {phase: {name: reassemble(values, len(reports[phase]["rows"])) for name, values in cases.items()}
           for phase, cases in folds.items()}
    y, vy = numbers(rows, "full"), numbers(validation, "full")
    weights = prior.sample_weights(y, sources, np.arange(len(rows)))
    overview = oof["train"]["no_visual"]
    coefficients = learn_text_weight(overview["text"], overview["audio"], train["time"]["clean"], y, weights)
    flexible = learn_text_weight(overview["text"], overview["audio"], train["time"]["clean"], y, weights, penalty=.004)
    policies = {name: tune_case(name, train, valid, oof["train"][name], oof["valid"][name], y, vy, sources, vs,
                                coefficients) for name in CASES}
    protected = {name: tune_protected(name, train, valid, oof["train"][name], oof["valid"][name], y, vy, sources, vs,
                                      (coefficients, flexible)) for name in CASES}
    for name, policy in policies.items():
        print(name, f"validation {policy['previous_validation']['correct']} -> {policy['validation']['correct']}, "
              f"Macro-F1 {policy['validation']['macro_f1']:.4f}, share={policy['share']}", flush=True)
        selected = protected[name]
        print(name, f"protected validation {selected['previous_validation']['correct']} -> "
              f"{selected['validation']['correct']}, F1 {selected['validation']['macro_f1']:.4f}, "
              f"share={selected['share']} margin={selected['max_margin']}", flush=True)
    for modality in ("text", "audio"):
        score = metrics(numbers(validation, modality), oof["valid"]["no_visual"][modality])
        print(f"{modality} validated on its own labels: {score['correct']}/{len(validation)}, F1 {score['macro_f1']:.4f}",
              flush=True)
    specialists = fit_specialists(train, rows, sources, np.arange(len(rows)), assignment)
    output = {"code": sha_file(Path(__file__)), "splits": digests, "metadata": reports["train"]["meta_sha256"],
              "reference": sha_file(old_path), "extractors": train["receipt"], "timeline": train["time"]["provenance"],
              "text_weight": coefficients.tolist(), "policies": policies, "protected": protected,
              "specialists": specialists}
    save_json(cache / DIRECTORY / "timeline-fusion.json", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    args = parser.parse_args()
    with threadpool_limits(limits=5, user_api="blas"):
        fit(args.cache)


if __name__ == "__main__":
    main()
