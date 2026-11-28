"""Calibrate individual no-visual neutral-to-polar candidates by held-out source."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit, softmax
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


MODEL = "neutral-precision-review.json"
PHASES = ("train", "valid")
NEUTRAL = LABELS.index("neutral")
POLARITIES = (LABELS.index("negative"), LABELS.index("positive"))
CUES = ("both_families", "global_acoustic", "temporal", "temporal_and_other")


def features(record, target, family):
    initial, text, acoustic, mini, retained = (
        record[k] for k in ("initial", "text", "acoustic", "mini", "retained"))
    present = np.isfinite(retained).all(axis=2)
    signed = [v[:, target] - v[:, NEUTRAL] for v in (initial, text, acoustic, mini, retained[:, 0], retained[:, 1])]
    compact = np.column_stack((*signed, present))
    if family == "compact":
        return np.nan_to_num(compact, nan=0.)
    disagreement = np.column_stack((
        retained[:, 0, target] - text[:, target], retained[:, 1, target] - acoustic[:, target],
        np.maximum(text[:, target], retained[:, 0, target]) -
        np.maximum(text[:, NEUTRAL], retained[:, 0, NEUTRAL]),
        np.maximum(acoustic[:, target], retained[:, 1, target]) -
        np.maximum(acoustic[:, NEUTRAL], retained[:, 1, NEUTRAL]),
    ))
    if family == "disagreement":
        columns = (compact, disagreement, record["full_features"][:, 12:])
    else:
        columns = (compact, disagreement, record["full_features"],
                   log_ratios(retained[:, 0]), log_ratios(retained[:, 1]))
    return np.nan_to_num(np.column_stack(columns), nan=0.)


def observe(batch, incumbent):
    retained = batch["retained"]
    if retained.shape != (len(incumbent), 2, len(LABELS)):
        raise ValueError("Expected retained text and acoustic probabilities")
    raw = precision.describe(batch, incumbent)
    neutral = (raw["active"] & (incumbent.argmax(1) == NEUTRAL)
               & np.isfinite(raw["support"]).all(axis=(1, 2)))
    return {"initial": incumbent, "full_features": raw["features"], "active": raw["active"],
            "neutral": neutral, "old_neutral": batch["prior"].argmax(1) == NEUTRAL,
            "text": raw["support"][:, 0], "acoustic": raw["support"][:, 1],
            "mini": batch["voice"], "retained": retained}


def support(record, target):
    def first(scores):
        return (scores.argmax(axis=1) == target) & np.isfinite(scores).all(axis=1)

    text = first(record["text"])
    acoustic = first(record["acoustic"]) | first(record["mini"])
    retained_text, retained_audio = (first(record["retained"][:, i]) for i in range(2))
    return {"both_families": (text | retained_text) & (acoustic | retained_audio),
            "global_acoustic": acoustic, "temporal": retained_text | retained_audio,
            "temporal_and_other": (retained_text & (acoustic | retained_audio)) |
                                  (retained_audio & (text | retained_text))}


def fit_head(record, batch, design, available, target, mode, subset, strength):
    selected = available & record[subset]
    x, truth = design[selected], batch["target"][selected]
    classes = (truth == target).astype(int) if mode == "binary" else truth
    if len(np.unique(classes)) != (2 if mode == "binary" else len(LABELS)):
        raise ValueError("Held-source precision fitting needs every label")
    center, scale = x.mean(axis=0), np.maximum(x.std(axis=0), .045)
    weights = calibrated.source_weights(batch["source"][selected]) * len(x)
    model = LogisticRegression(C=strength, max_iter=500)
    model.fit((x - center) / scale, classes, sample_weight=weights)
    return {"center": center.tolist(), "scale": scale.tolist(), "coef": model.coef_.tolist(),
            "bias": model.intercept_.tolist()}


def predict(design, head, mode, target):
    logits = ((design - head["center"]) / head["scale"]) @ np.asarray(head["coef"]).T + head["bias"]
    if mode == "binary":
        return expit(logits[:, 0])
    return softmax(logits, axis=1)[:, target]


def cross_fitted(records, batches, design, target, mode, subset, strength):
    grouped = np.concatenate([batches[k]["source"] for k in PHASES])
    offered = {phase: np.full(len(values), np.nan) for phase, values in design.items()}
    for _, picked in GroupKFold(4).split(grouped, groups=grouped):
        held = np.unique(grouped[picked])
        available = ~np.isin(batches["train"]["source"], held)
        head = fit_head(records["train"], batches["train"], design["train"], available, target, mode, subset, strength)
        for phase, values in design.items():
            chosen = np.isin(batches[phase]["source"], held)
            offered[phase][chosen] = predict(values[chosen], head, mode, target)
    if any(not np.isfinite(values).all() for values in offered.values()):
        raise ValueError("Expected one held-source calibration score per development clip")
    return offered


def proposal(initial, confidence, target, mix):
    # Reallocate the non-target mass in its incumbent proportions.
    adjusted = initial * ((1. - confidence) / np.maximum(1. - initial[:, target], 1.e-8))[:, None]
    adjusted[:, target] = confidence
    return (1. - mix) * initial + mix * adjusted


def select(record, proposed, confidence, rule, supported):
    initial = record["initial"]
    target = rule["target"]
    return (record["neutral"] & (record["old_neutral"] | (not rule["anchor_neutral"])) & supported
            & (proposed.argmax(axis=1) == target) & (initial[:, NEUTRAL] - initial[:, target] <= rule["max_gap"])
            & (confidence >= rule["min_target"]) & (proposed[:, target] - proposed[:, NEUTRAL] >= rule["min_lead"]))


def choose(records, batches):
    before = {phase: metrics(batch["target"], records[phase]["initial"]) for phase, batch in batches.items()}
    weights = {phase: calibrated.source_weights(batch["source"]) for phase, batch in batches.items()}
    best = {"rule": None, "before": before, "after": before}
    rank = (0., 0., 0., 0.)
    for target, family in product(POLARITIES, ("compact", "disagreement", "expanded")):
        design = {phase: features(record, target, family) for phase, record in records.items()}
        cues = {phase: support(record, target) for phase, record in records.items()}
        for mode, subset, strength in product(("binary", "multinomial"), ("neutral", "active"), (.02, .1, .5, 1.5)):
            confidence = cross_fitted(records, batches, design, target, mode, subset, strength)
            for mix in (.55, .75, 1.):
                proposals = {phase: proposal(record["initial"], confidence[phase], target, mix)
                             for phase, record in records.items()}
                for anchor, cue, gap, minimum, lead in product(
                        (True, False), CUES, (.08, .13, .2, .35, .6), (0., .50, .60, .70), (.03, .08, .15)):
                    rule = {"target": target, "family": family, "mode": mode, "subset": subset, "strength": strength,
                            "mix": mix, "anchor_neutral": anchor, "cue": cue, "max_gap": gap,
                            "min_target": minimum, "min_lead": lead}
                    masks = {phase: select(record, proposals[phase], confidence[phase], rule, cues[phase][cue])
                             for phase, record in records.items()}
                    truth = {phase: batch["target"][masks[phase]] for phase, batch in batches.items()}
                    if any(not y.size or np.any(y != target) for y in truth.values()):
                        continue
                    utility = .35 * weights["train"][masks["train"]].sum() + .65 * weights["valid"][masks["valid"]].sum()
                    # Equal held-source improvements prefer narrower admitted margins and stronger estimates.
                    quality = (float(utility), -gap, minimum, lead)
                    if quality <= rank:
                        continue
                    after = {phase: metrics(batch["target"], np.where(
                        masks[phase][:, None], proposals[phase], records[phase]["initial"]))
                             for phase, batch in batches.items()}
                    if any(after[k]["macro_f1"] < before[k]["macro_f1"] or
                           np.any(np.diag(after[k]["confusion"]) < np.diag(before[k]["confusion"])) for k in PHASES):
                        continue
                    best, rank = {"rule": rule, "before": before, "after": after}, quality
    if best["rule"] is not None:
        chosen = best["rule"]
        train = records["train"]
        x = features(train, chosen["target"], chosen["family"])
        best["head"] = fit_head(train, batches["train"], x, np.ones(len(x), dtype=bool),
                                chosen["target"], chosen["mode"], chosen["subset"], chosen["strength"])
    return best


def infer(batch, incumbent, policy):
    rule = policy["rule"]
    if rule is None:
        return incumbent
    record = observe(batch, incumbent)
    target = rule["target"]
    x = features(record, target, rule["family"])
    confidence = predict(x, policy["head"], rule["mode"], target)
    proposed = proposal(incumbent, confidence, target, rule["mix"])
    chosen = select(record, proposed, confidence, rule, support(record, target)[rule["cue"]])
    return np.where(chosen[:, None], proposed, incumbent)


def provenance(directory, splits):
    return {"code": sha_file(Path(__file__)), "reference": sha_file(directory / neutral_guard.MODEL),
            "features": sha_file(Path(precision.__file__)),
            "retained_development": sha_file(directory / evidence.DEVELOPMENT), "splits": splits}


def fit(cache):
    directory = cache / timeline.DIRECTORY
    previous = neutral_guard.load(cache, directory / "timeline-fusion.json")
    acoustic = audio_evidence.load(cache)
    development = evidence.load_development(cache)
    batches = {phase: cases[precision.CONDITION] for phase, cases in development.items()}
    for phase, batch in batches.items():
        batch.update(calibrated.acoustic_inputs(read_full_report(cache, phase)["rows"], acoustic[phase]))
        batch["quality"] = dynamic.quality_input(batch)
        batch["retained"] = development[phase]["synchronous_gap"]["context"][:, :2]
    incumbent = precision.development_incumbent(batches, previous["audio_review"]["policies"][precision.CONDITION])
    records = {phase: observe(batch, incumbent[phase]) for phase, batch in batches.items()}
    policy = choose(records, batches)
    save_json(directory / MODEL, {**provenance(directory, previous["splits"]), "policy": policy})
    print(policy["rule"], flush=True)
    for phase in PHASES:
        print(phase, {stage: (policy[stage][phase]["correct"], policy[stage][phase]["macro_f1"],
                             np.diag(policy[stage][phase]["confusion"]).tolist())
                      for stage in ("before", "after")}, flush=True)


def load(cache, splits):
    directory = cache / timeline.DIRECTORY
    artifact = json.loads((directory / MODEL).read_text("utf8"))
    if any(artifact.get(k) != v for k, v in provenance(directory, splits).items()):
        raise ValueError("Fit candidate precision on the frozen development evidence first")
    return artifact["policy"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    with threadpool_limits(limits=5, user_api="blas"):
        fit(parser.parse_args().cache)
