"""Fit a source-held-out MiniCPM-o voice review for two missing-evidence conditions."""

import argparse
import io
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import softmax
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import chsims_teacher_directional as directional
from scripts import chsims_teacher_directional_pair as pair
from scripts import chsims_teacher_refinement as refinement
from scripts import chsims_teacher_robustness as robust
from scripts import chsims_teacher_source_guard as guard
from scripts import chsims_teacher_source_review as review
from scripts import chsims_teacher_timeline as timeline
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_omni_model_comparison import AUDIO_MINI, predict as linear_predict
from scripts.chsims_teacher_cross_source import group_ids, metrics
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file, write_atomic
from scripts.chsims_teacher_supervision import numbers, read_full_report


MODEL = "minicpm-review.json"
DEVELOPMENT = "minicpm-review-development.npz"
NEUTRAL = LABELS.index("neutral")


def voice_quality(data, condition):
    if condition == "no_visual":
        return data["clean"]
    quality = data["windows"]["quality"]
    speech = quality[:, :, 1]
    total = speech.sum(1)
    weighted = (quality * speech[:, :, None]).sum(1) / np.maximum(total[:, None], 1e-6)
    return np.column_stack((weighted[:, 0], total, weighted[:, 2]))


def decision_quality(probability):
    sorted_scores = np.sort(probability, axis=1)
    return np.nan_to_num(sorted_scores[:, -1] - sorted_scores[:, -2])


def design(batch, audio, hybrid, previous, quality):
    local = batch["support"][:, 1]
    eligible = (batch["active"] & np.isfinite(previous[:, 0]) & np.isfinite(audio[:, 0]) &
                np.isfinite(hybrid[:, 0]) & (quality[:, 1] > 0.))
    voice = np.clip(quality[:, 1] / 1.5, 0., 1.) * (.55 + .45 * quality[:, 2])
    blocks = [previous, hybrid, audio, local, batch["reference"]]
    cues = np.column_stack([*(directional.log_ratios(value) for value in blocks), quality[:, 0],
                            np.log1p(quality[:, 1]), quality[:, 2], decision_quality(audio),
                            decision_quality(previous), voice * (np.nan_to_num(hybrid[:, NEUTRAL]) -
                                                                 np.nan_to_num(previous[:, NEUTRAL]))])
    return {"prior": previous, "mini": hybrid, "voice": audio, "voice_quality": voice,
            "design": cues, "active": eligible}


def use_voice(batch, proposed, gate):
    previous = batch["prior"]
    audio = batch["voice"]
    old, offered = previous.argmax(1), proposed.argmax(1)
    items = np.arange(len(old))
    gain = proposed[items, offered] - proposed[items, old]
    voice_gain = audio[items, offered] - audio[items, old]
    support = (audio.argmax(1) == offered) if gate["strict"] else (voice_gain >= 0.)
    center_shift = (old == NEUTRAL) | (offered == NEUTRAL)
    minimum = np.where(offered == NEUTRAL, gate["neutral_margin"], gate["polar_margin"])
    chosen = (batch["active"] & (batch["voice_quality"] >= gate["voice_quality"]) & support &
              (decision_quality(audio) >= gate["voice_confidence"]) & (old != offered) & (gain >= minimum))
    if gate["center_only"]:
        chosen &= center_shift
    return np.where(chosen[:, None], proposed, previous)


def scores(batch, method, residual):
    if method == "mini":
        return batch["mini"]
    if method == "voice_mix":
        return .60 * batch["mini"] + .40 * batch["voice"]
    settings = {**residual, "share": .75, "neutral_offset": 0.}
    return timeline.apply_residual(batch["design"], batch["prior"], settings, batch["active"])


def infer(batch, policy):
    if policy["gate"] is None:
        return batch["prior"]
    return use_voice(batch, scores(batch, policy["method"], policy["residual"]), policy["gate"])


def baseline(batch, case, model):
    preceding = pair.predict(batch, model["preceding"]["policies"][case])
    if case == review.CONDITION:
        proposed, _ = review.predict(batch, preceding, model["review"])
        return guard.predict(batch, preceding, proposed, model["guard"])
    return preceding


def hybrid_on_fold(mini, src, mask, head, selected, fitted):
    probabilities = np.full((len(selected), len(LABELS)), np.nan)
    eligible = (robust.observed_mask(mini[src])[selected] & mask) == mask
    present = selected[eligible]
    if len(present):
        strength = head["strength"]
        active, logits = refinement.logits_for(mini[src], present, *fitted)
        soft = np.full((len(present), len(LABELS)), np.nan)
        soft[active] = softmax(logits[strength] / head["temperature"], axis=1)
        probabilities[eligible] = refinement.neutral_policy(soft, None, 1., head["neutral_offset"], 0.)
    return probabilities


def fit_native(variants, truth, sources, selected, augmentation):
    return robust.fit_head(variants, AUDIO_MINI, robust.CHANNEL["audio"], truth, sources, selected, augmentation,
                           {"C": .75, "temperature": 1., "neutral_shift": 0.})


def development(cache, checkpoint, previous, original, acoustic, provenance):
    archive = cache / timeline.DIRECTORY / DEVELOPMENT
    if archive.exists():
        with np.load(archive, allow_pickle=False) as data:
            if json.loads(data["provenance"].item()) == provenance:
                records = {phase: {case: {key.removeprefix(phase + "/" + case + "/"): data[key] for key in data.files
                                          if key.startswith(phase + "/" + case + "/")}
                                   for case in timeline.CASES} for phase in ("train", "valid")}
                return records, json.loads(data["voice"].item())
    reports = {phase: read_full_report(cache, phase) for phase in ("train", "valid")}
    inputs = {phase: timeline.read_data(cache, report) for phase, report in reports.items()}
    for data in inputs.values():
        if data["receipt"] != checkpoint["extractors"] or data["time"]["provenance"] != checkpoint["timeline"]:
            raise ValueError("The development evidence differs from the frozen teacher")
    rows = {phase: doc["rows"] for phase, doc in reports.items()}
    mini = {phase: refinement.mini_variants(cache, rows[phase], data["variants"]) for phase, data in inputs.items()}
    people = {phase: group_ids(values) for phase, values in rows.items()}
    labels = {phase: numbers(values, "full") for phase, values in rows.items()}
    audio_truth = numbers(rows["train"], "audio")
    groups = np.concatenate(tuple(people.values()))
    assignment = robust.case_assignment(rows["train"])
    parts = {phase: {case: [] for case in timeline.CASES} for phase in rows}
    train = inputs["train"]
    for fold, (_, held) in enumerate(GroupKFold(4).split(groups, groups=groups), 1):
        withheld = set(groups[held])
        kept = np.flatnonzero(~np.isin(people["train"], list(withheld)))
        specialties = timeline.fit_specialists(train, rows["train"], people["train"], kept, assignment)
        references = timeline.fit_prior(train, rows["train"], original, people["train"], kept, assignment)
        voice = fit_native(mini["train"], audio_truth, people["train"], kept, assignment)
        heads = {}
        for case, (_, mask) in timeline.CASES.items():
            head = acoustic["heads"][str(mask)]["mini_hybrid"]
            names = robust.block_names(refinement.FEATURES["mini_hybrid"], mask)
            fitted = refinement.fit_candidates(mini["train"], names, mask, labels["train"], people["train"], kept,
                                               assignment, (head["strength"],))
            heads[case] = head, fitted
        for phase, data in inputs.items():
            indices = np.flatnonzero(np.isin(people[phase], list(withheld)))
            observations = timeline.score_examples(data, specialties, references, indices)
            for case, (src, mask) in timeline.CASES.items():
                head, fitted = heads[case]
                hybrid = hybrid_on_fold(mini[phase], src, mask, head, indices, fitted)
                vectors = {name: mini[phase][src][name][indices] for name in AUDIO_MINI}
                audio = linear_predict(voice, vectors, None)
                parts[phase][case].append((indices, {"observations": observations[case], "mini": hybrid, "voice": audio}))
        print(f"Fitted MiniCPM-o sound and hybrid heads without held sources {fold}/4", flush=True)

    records = {phase: {} for phase in rows}
    for phase, data in inputs.items():
        time = data["time"]
        desktop = {"no_visual": time["desktop"], "synchronous_gap": directional.desktop_gap_scores(cache, rows[phase], time)}
        for case in timeline.CASES:
            parts_by_id = timeline.reassemble(parts[phase][case], len(rows[phase]))
            batch = directional.describe_evidence(parts_by_id["observations"], time, case, checkpoint["protected"][case],
                                                  desktop[case])
            record = design(batch, parts_by_id["voice"], parts_by_id["mini"], baseline(batch, case, previous),
                            voice_quality(time, case))
            records[phase][case] = {**record, "target": labels[phase], "source": people[phase],
                                    "audio_target": numbers(rows[phase], "audio")}
    voice = fit_native(mini["train"], audio_truth, people["train"], np.arange(len(audio_truth)), assignment)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, provenance=json.dumps(provenance), voice=json.dumps(voice), **{f"{phase}/{case}/{name}": value
                        for phase, collection in records.items() for case, entry in collection.items()
                        for name, value in entry.items()})
    write_atomic(archive, buffer.getvalue())
    return records, voice


def choose(train, valid):
    y, previous = valid["target"], valid["prior"]
    before = metrics(y, previous)
    floors = np.diag(np.asarray(before["confusion"])[:, :len(LABELS)])
    decision = previous.argmax(1)
    base_correct = np.isfinite(previous[:, 0]) & (decision == y)
    _, sources, size = np.unique(valid["source"], return_inverse=True, return_counts=True)
    winner = {"method": None, "residual": None, "gate": None, "before": before, "after": before,
              "voice_validation_label_A": metrics(valid["audio_target"], valid["voice"])}
    best_rank = (0., 0., 0)
    fit = np.flatnonzero(train["active"])
    weight = robust.sample_weights(train["target"], train["source"], fit)
    methods = [(name, None) for name in ("mini", "voice_mix")]
    methods.extend(("learned", timeline.learn_residual(train["design"][fit], train["prior"][fit],
                                                       train["target"][fit], weight, penalty)) for penalty in (.015, .06))
    for method, calibrated in methods:
        offered = scores(valid, method, calibrated)
        for quality in (.25, .45, .65):
            for confidence in (.03, .10, .20):
                for strict in (False, True):
                    for center_only in (True, False):
                        for to_center in (0., .05, .12, .24):
                            for away in (.04, .12, .24, .55):
                                options = {"voice_quality": quality, "voice_confidence": confidence, "strict": strict,
                                           "center_only": center_only, "neutral_margin": to_center, "polar_margin": away}
                                predicted = use_voice(valid, offered, options)
                                after = metrics(y, predicted)
                                hits = np.diag(np.asarray(after["confusion"])[:, :len(LABELS)])
                                if np.any(hits < floors) or hits[NEUTRAL] <= floors[NEUTRAL] or after["macro_f1"] <= before["macro_f1"]:
                                    continue
                                top = predicted.argmax(1)
                                correct = np.isfinite(predicted[:, 0]) & (top == y)
                                gain = (np.bincount(sources, weights=correct.astype(int) - base_correct) / size).mean()
                                value = .35 * (after["accuracy"] - before["accuracy"]) + .40 * (
                                    after["macro_f1"] - before["macro_f1"]) + .25 * gain
                                rank = (value, float(gain), -int(np.count_nonzero(top != decision)))
                                if rank > best_rank:
                                    best_rank = rank
                                    winner.update(method=method, residual=calibrated, gate=options, after=after)
    return winner


def paths(cache):
    return cache / timeline.DIRECTORY / guard.MODEL, cache / refinement.DIRECTORY / "heads.json"


def fit(cache):
    reference, checkpoint, original = directional.frozen_inputs(cache)
    previous = guard.load(cache, reference)
    previous_path, acoustic_path = paths(cache)
    acoustic = json.loads(acoustic_path.read_text("utf8"))
    provenance = {"guard": sha_file(previous_path), "refinement": sha_file(acoustic_path),
                  "implementation": sha_file(Path(__file__)), "checkpoint": sha_file(reference),
                  "splits": checkpoint["splits"]}
    if acoustic["implementation"] != sha_file(Path(refinement.__file__)) or acoustic["splits"] != provenance["splits"]:
        raise ValueError("Use the frozen MiniCPM-o comparison")
    for phase, sha in provenance["splits"].items():
        if sha_file(cache / f"evaluation-{phase}.json") != sha:
            raise ValueError(f"The {phase} data changed")
    dev, voice = development(cache, checkpoint, previous, original, acoustic, provenance)
    policies = {case: choose(dev["train"][case], dev["valid"][case]) for case in timeline.CASES}
    for case, policy in policies.items():
        print(case, policy["method"], policy["gate"], policy["before"]["correct"], "->", policy["after"]["correct"],
              "F1", policy["after"]["macro_f1"], flush=True)
    save_json(cache / timeline.DIRECTORY / MODEL, {**provenance, "policies": policies, "voice": voice})


def load(cache, reference):
    settings = json.loads((cache / timeline.DIRECTORY / MODEL).read_text("utf8"))
    previous, acoustic = paths(cache)
    expected = {"guard": sha_file(previous), "refinement": sha_file(acoustic),
                "implementation": sha_file(Path(__file__)), "checkpoint": sha_file(reference)}
    if any(settings[k] != value for k, value in expected.items()):
        raise ValueError("Fit a new MiniCPM-o review with the frozen development inputs")
    return {**guard.load(cache, reference), "kind": "minicpm_review", "audio_review": settings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    args = parser.parse_args()
    with threadpool_limits(limits=5, user_api="blas"):
        fit(args.cache)


if __name__ == "__main__":
    main()
