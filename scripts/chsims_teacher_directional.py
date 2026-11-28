"""Train a selective correction for source changes under two missing-evidence conditions."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotion_assistant.foundation_fusion import predict_foundation_clip
from scripts import chsims_teacher_robustness as robust
from scripts import chsims_teacher_timeline as timeline
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_teacher_cross_source import group_ids, metrics
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file
from scripts.chsims_teacher_supervision import numbers, read_full_report


DIRECTIONAL = "directional-fusion.json"
NEGATIVE, NEUTRAL, POSITIVE = (LABELS.index(label) for label in ("negative", "neutral", "positive"))


def frozen_inputs(cache):
    previous = cache / timeline.DIRECTORY / "timeline-fusion.json"
    original = cache / "missing-evidence/fusion-heads.json"
    model = json.loads(previous.read_text("utf8"))
    if model["code"] != sha_file(Path(timeline.__file__)) or model["reference"] != sha_file(original):
        raise ValueError("The preceding timeline model and its reference must match their frozen implementations")
    return previous, model, json.loads(original.read_text("utf8"))


def desktop_gap_scores(cache, rows, timeline_data):
    scores = []
    for row, vector in zip(rows, timeline_data["embeddings"][:, 2], strict=True):
        with np.load(cache / robust.PREPARED / f"{row['id']}.npz", allow_pickle=False) as asset:
            gap = json.loads(asset["info"].item())
            speech = asset["speech"]
        result = predict_foundation_clip(gap["joint_text"], embedding=vector, audio=gap["audio"],
                                         visual=gap["visual"], audio_embedding=speech if speech.size else None)
        if result["status"] not in ("ok", "insufficient_evidence"):
            raise ValueError(f"Could not replay missing-evidence desktop scores for {row['id']}")
        scores.append([result["scores"].get(label, np.nan) for label in LABELS])
    return np.asarray(scores)


def local_channel_scores(windows, observations):
    quality, duration = observations["quality"], observations["duration"]
    span, video = duration[:, :, 0], duration[:, :, 1]
    text = quality[:, :, 0]
    voiced = np.minimum(quality[:, :, 1], span) * (.55 + .45 * quality[:, :, 2])
    return [timeline.weighted_probabilities(windows[f"window_{name}"], mass) for name, mass in (
        ("text", text * span), ("audio", voiced), ("visual", video * span))]


def prior_timeline(scores, time, name, policy):
    x, reference, active, mass, local = timeline.example_features(scores, time, name, np.asarray(policy["text_weight"]))
    proposed = timeline.apply_residual(x, reference, policy, active)
    previous = timeline.protect_directions(reference, proposed, local, policy["max_margin"], policy["corroboration"])
    return previous, reference, active, local, x, mass


def log_ratios(probabilities):
    logs = np.log(np.clip(np.nan_to_num(probabilities, nan=1. / len(LABELS)), 1e-7, 1.))
    return logs[:, [NEGATIVE, POSITIVE]] - logs[:, [NEUTRAL]]


def describe_evidence(scores, data, name, policy, desktop, development_previous=None):
    previous, reference, active, local, _, mass = prior_timeline(scores, data, name, policy)
    if development_previous is not None:
        previous = development_previous
    text, audio = scores["text"], scores["audio"]
    blocks = [previous, reference, local, desktop, text, audio]
    support = [reference, local, desktop]
    if name == "no_visual":
        quality = timeline.quality_features(text, audio, data["clean"])[:, 1:]
    else:
        windows = data["windows"]
        short_text, short_audio, short_visual = local_channel_scores(scores["windows"], windows)
        blocks.extend((short_text, short_audio, short_visual))
        support.extend((short_audio, short_visual))
        neutral_disagreement = np.nan_to_num(audio[:, NEUTRAL] - short_audio[:, NEUTRAL])
        local_disagreement = np.nan_to_num(audio[:, NEUTRAL] - local[:, NEUTRAL])
        quality = np.column_stack((windows["quality"][:, :, 0].mean(1),
                                   np.log1p(windows["quality"][:, :, 1].sum(1)),
                                   windows["duration"][:, :, 1].mean(1), mass.sum(1),
                                   (mass > 0).sum(1), neutral_disagreement, local_disagreement))
    design = np.column_stack([*(log_ratios(value) for value in blocks),
                              *(np.isfinite(value[:, 0]) for value in blocks[3:]), quality])
    return {"design": design, "previous": previous, "reference": reference, "active": active,
            "support": np.stack(support, axis=1)}


def crossfit_previous(scores, data, condition, settings, truth, sources, fold_index):
    _, reference, active, local, design, _ = prior_timeline(scores, data, condition, settings)
    outcome = reference.copy()
    for fold in np.unique(fold_index):
        included = np.flatnonzero((fold_index != fold) & active)
        excluded = (fold_index == fold) & active
        fitted = timeline.learn_residual(design[included], reference[included], truth[included],
                                         robust.sample_weights(truth, sources, included), settings["penalty"])
        recalibrated = timeline.apply_residual(design, reference, {**settings, **fitted}, excluded)
        guarded = timeline.protect_directions(reference, recalibrated, local,
                                               settings["max_margin"], settings["corroboration"])
        outcome[excluded] = guarded[excluded]
    return outcome


def support_counts(evidence, proposal):
    return np.count_nonzero(np.isfinite(evidence[:, :, 0]) &
                            (evidence.argmax(2) == proposal[:, None]), axis=1)


def accept(current, candidate, evidence, eligible, threshold, support):
    """Change a label only for a supported improvement in estimated correctness."""
    previous, proposed = current.argmax(1), candidate.argmax(1)
    indices = np.arange(len(current))
    gain = candidate[indices, proposed] - candidate[indices, previous]
    accepted = (eligible & np.isfinite(current[:, 0]) & np.isfinite(candidate[:, 0]) &
                (previous != proposed) & (gain >= np.asarray(threshold)[proposed]) &
                (support_counts(evidence, proposed) >= support))
    return np.where(accepted[:, None], candidate, current)


def propose(batch, model):
    base = batch[model["anchor"]]
    return timeline.apply_residual(batch["design"], base, model, batch["active"])


def predict(batch, settings):
    base = batch["previous"]
    if settings["rule"] is None:
        return base
    proposal = propose(batch, settings["rule"])
    return accept(base, proposal, batch["support"], batch["active"], settings["threshold"], settings["support"])


def select_policy(condition, training, validation, truth, targets, sources, validation_sources):
    previous = validation["previous"]
    before = metrics(targets, previous)
    hits = np.diag(np.asarray(before["confusion"])[:, :len(LABELS)])
    chosen = {"rule": None, "previous_validation": before, "validation": before, "development_changes": 0}
    recent = previous.argmax(1)
    _, group, counts = np.unique(validation_sources, return_inverse=True, return_counts=True)
    before_hits = np.isfinite(previous[:, 0]) & (recent == targets)
    active = np.flatnonzero(training["active"])
    sample = robust.sample_weights(truth, sources, active)
    best_rank = (0., 0., 0.)
    for anchor in ("previous", "reference"):
        for penalty in (.008, .035, .12):
            fitter = timeline.learn_residual(training["design"][active], training[anchor][active],
                                             truth[active], sample, penalty)
            for share in (.5, 1.):
                rule = {**fitter, "anchor": anchor, "share": share, "neutral_offset": 0.}
                proposal = propose(validation, rule)
                for support in (0, 1, 2):
                    for negative in (0., .04, .10, .20):
                        for others in (0., .05, .15, .30):
                            threshold = [negative, others, others]
                            output = accept(previous, proposal, validation["support"], validation["active"], threshold, support)
                            after = metrics(targets, output)
                            recall = np.diag(np.asarray(after["confusion"])[:, :len(LABELS)])
                            if (np.any(recall < hits) or after["accuracy"] <= before["accuracy"] or
                                    after["macro_f1"] <= before["macro_f1"]):
                                continue
                            top = output.argmax(1)
                            gained = (np.isfinite(output[:, 0]) & (top == targets)).astype(int) - before_hits
                            source_delta = np.bincount(group, weights=gained) / counts
                            accuracy_delta = after["accuracy"] - before["accuracy"]
                            macro_delta = after["macro_f1"] - before["macro_f1"]
                            negative_delta = (recall[NEGATIVE] - hits[NEGATIVE]) / max(1, (targets == NEGATIVE).sum())
                            changes = int(np.count_nonzero(top != recent))
                            score = .33 * accuracy_delta + .32 * macro_delta + .25 * source_delta.mean() + .10 * negative_delta
                            rank = (score, float(source_delta.mean()), -changes)
                            if rank > best_rank:
                                best_rank = rank
                                chosen = {"rule": rule, "threshold": threshold, "support": support,
                                          "previous_validation": before, "validation": after,
                                          "development_changes": changes, "source_accuracy_delta": float(source_delta.mean())}
    print(f"{condition} development: {before['correct']} -> {chosen['validation']['correct']}; Macro-F1 "
          f"{before['macro_f1']:.4f} -> {chosen['validation']['macro_f1']:.4f}", flush=True)
    return chosen


def development_data(cache, model, original):
    reports = {split: read_full_report(cache, split) for split in ("train", "valid")}
    for split in reports:
        if sha_file(cache / f"evaluation-{split}.json") != model["splits"][split]:
            raise ValueError(f"The {split} split changed after the preceding teacher was fitted")
    inputs = {split: timeline.read_data(cache, report) for split, report in reports.items()}
    for values in inputs.values():
        if values["receipt"] != model["extractors"] or values["time"]["provenance"] != model["timeline"]:
            raise ValueError("Development evidence differs from the preceding timeline teacher")
    rows = {split: report["rows"] for split, report in reports.items()}
    groups = {split: group_ids(part) for split, part in rows.items()}
    joint = np.concatenate(tuple(groups.values()))
    members = {split: np.full(len(part), -1, dtype=int) for split, part in rows.items()}
    pieces = {split: defaultdict(list) for split in rows}
    augmentation = robust.case_assignment(rows["train"])
    for fold, (_, out) in enumerate(GroupKFold(4).split(joint, groups=joint)):
        held_sources = set(joint[out])
        kept = np.flatnonzero(~np.isin(groups["train"], list(held_sources)))
        specialists = timeline.fit_specialists(inputs["train"], rows["train"], groups["train"], kept, augmentation)
        references = timeline.fit_prior(inputs["train"], rows["train"], original, groups["train"], kept, augmentation)
        for split, values in inputs.items():
            held = np.flatnonzero(np.isin(groups[split], list(held_sources)))
            members[split][held] = fold
            observations = timeline.score_examples(values, specialists, references, held)
            for case, part in observations.items():
                pieces[split][case].append((held, part))
        print(f"Scored source-held-out development fold {fold + 1}/4", flush=True)
    truth = {split: numbers(part, "full") for split, part in rows.items()}
    result = {split: {} for split in rows}
    for split, values in inputs.items():
        time = values["time"]
        desktop = {"no_visual": time["desktop"], "synchronous_gap": desktop_gap_scores(cache, rows[split], time)}
        for case, segment in pieces[split].items():
            scores = timeline.reassemble(segment, len(rows[split]))
            policy = model["protected"][case]
            crossfit = None
            if split == "train" and policy["share"]:
                crossfit = crossfit_previous(scores, time, case, policy, truth[split], groups[split], members[split])
            result[split][case] = describe_evidence(scores, time, case, policy, desktop[case], crossfit)
    return result, truth, groups


def fit(cache):
    previous, model, original = frozen_inputs(cache)
    batches, truth, groups = development_data(cache, model, original)
    policies = {case: select_policy(case, batches["train"][case], batches["valid"][case], truth["train"],
                                    truth["valid"], groups["train"], groups["valid"]) for case in timeline.CASES}
    save_json(cache / timeline.DIRECTORY / DIRECTIONAL,
              {"implementation": sha_file(Path(__file__)), "previous_model": sha_file(previous),
               "splits": model["splits"], "policies": policies})


def load(cache, previous):
    settings = json.loads((cache / timeline.DIRECTORY / DIRECTIONAL).read_text("utf8"))
    if settings["implementation"] != sha_file(Path(__file__)) or settings["previous_model"] != sha_file(previous):
        raise ValueError("Fit the directional correction on development data with the current implementation first")
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    args = parser.parse_args()
    with threadpool_limits(limits=5, user_api="blas"):
        fit(args.cache)


if __name__ == "__main__":
    main()
