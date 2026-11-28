"""Review neutral predictions using source-held-out text and audio calibration."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import chsims_teacher_audio_evidence as audio_evidence
from scripts import chsims_teacher_context_evidence as evidence
from scripts import chsims_teacher_dynamic_conflict as dynamic
from scripts import chsims_teacher_neutral_guard as neutral_guard
from scripts import chsims_teacher_precision_review as precision
from scripts import chsims_teacher_source_calibration as calibrated
from scripts import chsims_teacher_timeline as timeline
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_teacher_cross_source import metrics
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file
from scripts.chsims_teacher_directional import log_ratios
from scripts.chsims_teacher_supervision import read_full_report


MODEL = "neutral-consensus-review.json"
ACOUSTIC_MODEL = "neutral-consensus-acoustic.json"
NEUTRAL = LABELS.index("neutral")
POSITIVE = LABELS.index("positive")
POLARITIES = (LABELS.index("negative"), LABELS.index("positive"))
PHASES = ("train", "valid")


def retained_text_audio(window_scores, windows):
    """Aggregate only text and audio repeated measurements, in temporal order."""
    quality = windows["quality"]
    span = windows["duration"][:, :, 0]
    text_mass = quality[:, :, 0] * span
    audio_mass = np.minimum(quality[:, :, 1], span) * (.55 + .45 * quality[:, :, 2])
    return np.stack((timeline.weighted_probabilities(window_scores["window_text"], text_mass),
                     timeline.weighted_probabilities(window_scores["window_audio"], audio_mass)), axis=1)


def observe(batch, incumbent, acoustic_support=False):
    record = precision.describe(batch, incumbent)
    retained = batch["retained"]
    window_features = [log_ratios(retained[:, i]) for i in range(retained.shape[1])]
    record["features"] = np.column_stack((record["features"], *window_features, np.isfinite(retained).all(2)))
    record["eligible"] = (record["active"] & (incumbent.argmax(1) == NEUTRAL) &
                          (batch["prior"].argmax(1) == NEUTRAL) &
                          np.isfinite(record["support"]).all(axis=(1, 2)))
    if acoustic_support:
        acoustic, mini = batch["acoustic"], batch["voice"]
        record["eligible"] &= ((acoustic.argmax(1) == POSITIVE) |
                               (np.isfinite(mini).all(1) & (mini.argmax(1) == POSITIVE)))
    return record


def fit_head(record, batch, indices, strength):
    training = indices[record["active"][indices]]
    features = record["features"][training]
    weights = calibrated.source_weights(batch["source"][training]) * len(training)
    center, scale = features.mean(0), np.maximum(features.std(0), .05)
    model = LogisticRegression(C=strength, max_iter=400)
    model.fit((features - center) / scale, batch["target"][training], sample_weight=weights)
    if model.classes_.tolist() != list(range(len(LABELS))):
        raise ValueError("Each held-source fit needs every emotion class")
    return {"center": center.tolist(), "scale": scale.tolist(), "coef": model.coef_.tolist(),
            "bias": model.intercept_.tolist()}


def scores(record, head):
    x = (record["features"] - head["center"]) / head["scale"]
    return softmax(x @ np.asarray(head["coef"]).T + head["bias"], axis=1)


def cross_fitted(records, batches, strength):
    sources = {phase: batch["source"] for phase, batch in batches.items()}
    combined = np.concatenate(tuple(sources.values()))
    result = {phase: np.full_like(record["initial"], np.nan) for phase, record in records.items()}
    for _, indices in GroupKFold(4).split(combined, groups=combined):
        held = np.unique(combined[indices])
        training = np.flatnonzero(~np.isin(sources["train"], held))
        head = fit_head(records["train"], batches["train"], training, strength)
        for phase, record in records.items():
            selected = record["eligible"] & np.isin(sources[phase], held)
            result[phase][selected] = scores(record, head)[selected]
    if any(not np.isfinite(result[k][v["eligible"]]).all() for k, v in records.items()):
        raise ValueError("The grouped fit did not score every eligible clip")
    return result


def select(record, proposal, rule):
    target = rule["target"]
    return (record["eligible"] & (proposal.argmax(1) == target) &
            (record["initial"][:, NEUTRAL] - record["initial"][:, target] <= rule["max_gap"]) &
            (proposal[:, target] - proposal[:, NEUTRAL] >= rule["min_lead"]))


def choose(records, batches, acoustic_support=False):
    baseline = {phase: metrics(batches[phase]["target"], record["initial"]) for phase, record in records.items()}
    weights = {phase: calibrated.source_weights(batch["source"]) for phase, batch in batches.items()}
    best = {"rule": None, "before": baseline, "after": baseline}
    rank = (0., 0., 0.)
    directions = (POSITIVE,) if acoustic_support else POLARITIES
    for strength in (.02, .10, .40, 2.):
        offered = cross_fitted(records, batches, strength)
        for mix in (.45, .70, 1.):
            proposals = {phase: mix * offered[phase] + (1. - mix) * record["initial"]
                         for phase, record in records.items()}
            for target, gap, lead in product(directions, (.04, .08, .14, .24, .40, .75),
                                              (.01, .05, .10, .17, .26, .40)):
                rule = {"target": target, "mix": mix, "max_gap": gap, "min_lead": lead}
                after, gains, changes = {}, {}, 0
                for phase, record in records.items():
                    chosen = select(record, proposals[phase], rule)
                    truth = batches[phase]["target"]
                    selected_truth = truth[chosen]
                    if np.any(selected_truth == NEUTRAL) or not np.any(selected_truth == target):
                        break
                    revised = np.where(chosen[:, None], proposals[phase], record["initial"])
                    result = metrics(truth, revised)
                    if result["macro_f1"] < baseline[phase]["macro_f1"]:
                        break
                    after[phase] = result
                    gains[phase] = float(weights[phase][chosen][selected_truth == target].sum())
                    changes += int(chosen.sum())
                if len(after) != len(PHASES):
                    continue
                quality = (.35 * gains["train"] + .65 * gains["valid"], -changes, after["valid"]["macro_f1"])
                if quality > rank:
                    rank = quality
                    best = {"rule": rule, "strength": strength, "before": baseline, "after": after}
    if best["rule"] is not None:
        best["head"] = fit_head(records["train"], batches["train"], np.arange(len(records["train"]["initial"])),
                                best["strength"])
    return best


def infer(batch, incumbent, policy):
    rule = policy["rule"]
    if rule is None:
        return incumbent
    record = observe(batch, incumbent, policy["acoustic_support"])
    proposal = rule["mix"] * scores(record, policy["head"]) + (1. - rule["mix"]) * incumbent
    return np.where(select(record, proposal, rule)[:, None], proposal, incumbent)


def provenance(directory, splits):
    return {"code": sha_file(Path(__file__)), "features": sha_file(Path(precision.__file__)),
            "reference": sha_file(directory / neutral_guard.MODEL), "splits": splits}


def fit(cache, acoustic_support=False):
    directory = cache / timeline.DIRECTORY
    model = neutral_guard.load(cache, directory / "timeline-fusion.json")
    acoustic = audio_evidence.load(cache)
    development = evidence.load_development(cache)
    batches = {phase: bundle[precision.CONDITION] for phase, bundle in development.items()}
    for phase, batch in batches.items():
        batch.update(calibrated.acoustic_inputs(read_full_report(cache, phase)["rows"], acoustic[phase]))
        batch["quality"] = dynamic.quality_input(batch)
        batch["retained"] = development[phase]["synchronous_gap"]["context"][:, :2]
    prior = precision.development_incumbent(batches, model["audio_review"]["policies"][precision.CONDITION])
    records = {phase: observe(batch, prior[phase], acoustic_support) for phase, batch in batches.items()}
    policy = {**choose(records, batches, acoustic_support), "acoustic_support": acoustic_support}
    path = directory / (ACOUSTIC_MODEL if acoustic_support else MODEL)
    save_json(path, {**provenance(directory, model["splits"]), "policy": policy})
    print(policy["rule"], policy.get("strength"),
          {k: (v["correct"], v["macro_f1"], np.diag(v["confusion"]).tolist()) for k, v in policy["after"].items()},
          flush=True)


def load(cache, splits, acoustic_support=False):
    directory = cache / timeline.DIRECTORY
    path = directory / (ACOUSTIC_MODEL if acoustic_support else MODEL)
    model = json.loads(path.read_text("utf8"))
    if (any(model.get(key) != value for key, value in provenance(directory, splits).items())
            or model["policy"]["acoustic_support"] != acoustic_support):
        raise ValueError("Fit the neutral consensus review on frozen development data first")
    return model["policy"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    parser.add_argument("--acoustic-support", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=5, user_api="blas"):
        fit(args.cache, args.acoustic_support)
