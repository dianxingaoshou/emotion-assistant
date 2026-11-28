"""Limit positive evidence from overlapping acoustic views without video."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import chsims_teacher_acoustic_continuity_review as continuity
from scripts import chsims_teacher_neutral_pair_review as pair
from scripts import chsims_teacher_neutral_precision_review as published
from scripts import chsims_teacher_source_calibration as calibrated
from scripts import chsims_teacher_timeline as timeline
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_teacher_cross_source import metrics
from scripts.chsims_teacher_data import CACHE, sha_file


MODEL = "acoustic-overlap-review.json"


def neutral_agreement(record):
    text = calibrated.signed_margin(record["text"], pair.NEUTRAL)
    acoustic = calibrated.signed_margin(record["acoustic"], pair.NEUTRAL)
    return np.nan_to_num(np.maximum(0., np.minimum(text, acoustic)), nan=0.)


def damp(margin, agreement, rule):
    overlap = np.maximum(0., agreement - rule["floor"])
    return np.where(margin > 0., margin / (1. + rule["strength"] * overlap), margin)


def apply(record, confidence, old, acoustic, rule):
    baseline = continuity.apply(record, confidence, old, acoustic)
    if rule is None:
        return baseline
    _, previously_selected = pair.reference_result(record, confidence, old)
    margin = continuity.acoustic_margin(record)
    revised_margin = damp(margin, neutral_agreement(record), rule)
    adjusted = continuity.adjusted_scores(confidence, revised_margin, acoustic["gain"])
    revised = published.proposal(record["initial"], adjusted, pair.POSITIVE, old["rule"]["mix"])
    threshold = {**old["rule"], "min_target": acoustic["min_target"], "min_lead": acoustic["min_lead"]}
    supported = published.support(record, pair.POSITIVE)[threshold["cue"]]
    accepted = published.select(record, revised, adjusted, threshold, supported)
    challenged = ((revised_margin < margin) & ~previously_selected &
                  (baseline.argmax(1) == pair.POSITIVE) & (record["initial"].argmax(1) == pair.NEUTRAL))
    return np.where(challenged[:, None], np.where(accepted[:, None], revised, record["initial"]), baseline)


def choose(records, batches, old, acoustic):
    confidence = pair.reference_confidence(records, batches, old)
    baseline = {phase: continuity.apply(record, confidence[phase], old, acoustic)
                for phase, record in records.items()}
    before = {phase: metrics(batches[phase]["target"], result) for phase, result in baseline.items()}
    groups = {}
    for phase, record in records.items():
        margin = continuity.acoustic_margin(record)
        agreement = neutral_agreement(record)
        eligible = record["active"] & (margin > 0.) & (agreement > 0.)
        weights = calibrated.source_weights(batches[phase]["source"])
        groups[phase] = (margin[eligible], agreement[eligible], weights[eligible], eligible)
    best = {"rule": None, "before": before, "after": before}
    rank = (0., 0., 0.)
    for floor, strength in product((0., .04, .08, .12, .16, .20, .24), (.5, 1., 2., 4., 8., 16., 32.)):
        rule = {"floor": floor, "strength": strength}
        gains, after, improvements = {}, {}, {}
        for phase, record in records.items():
            truth = batches[phase]["target"]
            revised = apply(record, confidence[phase], old, acoustic, rule)
            prior_hits = baseline[phase].argmax(1) == truth
            hits = revised.argmax(1) == truth
            if np.any(prior_hits & ~hits):
                break
            result = metrics(truth, revised)
            if result["macro_f1"] < before[phase]["macro_f1"]:
                break
            margin, agreement, weights, eligible = groups[phase]
            old_score = continuity.adjusted_scores(confidence[phase][eligible], margin, acoustic["gain"])
            new_score = continuity.adjusted_scores(confidence[phase][eligible], damp(margin, agreement, rule),
                                                   acoustic["gain"])
            label = truth[eligible] == pair.POSITIVE
            improvements[phase] = float(((old_score - label)**2 - (new_score - label)**2) @ weights)
            gains[phase] = int(hits.sum() - prior_hits.sum())
            after[phase] = result
        if len(after) != len(batches) or min(improvements.values()) < 0.:
            continue
        quality = (sum(gains.values()), .35 * improvements["train"] + .65 * improvements["valid"], -strength)
        if quality > rank:
            rank = quality
            best = {"rule": rule, "before": before, "after": after,
                    "calibration_development": {"improvement": improvements,
                                                "counts": {k: len(v[0]) for k, v in groups.items()}}}
    return best


def infer(batch, incumbent, policy):
    old = policy["reference"]["reference"]
    record = published.observe(batch, incumbent)
    rule = old["rule"]
    design = published.features(record, pair.POSITIVE, rule["family"])
    confidence = published.predict(design, old["head"], rule["mode"], pair.POSITIVE)
    return apply(record, confidence, old, policy["reference"]["rule"], policy["rule"])


def provenance(directory, splits):
    return {"code": sha_file(Path(__file__)), "continuity": sha_file(directory / continuity.MODEL), "splits": splits}


def fit(cache):
    records, batches, old, splits = pair.development_records(cache)
    directory = cache / timeline.DIRECTORY
    reference = continuity.load(cache, splits)
    if reference["rule"] is None:
        raise ValueError("A frozen acoustic continuity rule is required")
    policy = choose(records, batches, old, reference["rule"])
    save_json(directory / MODEL, {**provenance(directory, splits), "policy": policy})
    print(policy["rule"], policy.get("calibration_development"), flush=True)
    for phase in batches:
        print(phase, {key: (policy[key][phase]["correct"], policy[key][phase]["macro_f1"])
                      for key in ("before", "after")}, flush=True)


def load(cache, splits):
    directory = cache / timeline.DIRECTORY
    model = json.loads((directory / MODEL).read_text("utf8"))
    if any(model.get(key) != value for key, value in provenance(directory, splits).items()):
        raise ValueError("Fit acoustic overlap on frozen development evidence first")
    return {**model["policy"], "reference": continuity.load(cache, splits)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    with threadpool_limits(limits=5, user_api="blas"):
        fit(parser.parse_args().cache)
