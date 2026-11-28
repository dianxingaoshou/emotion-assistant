"""Use retained speech to calibrate a lone positive MiniCPM-o vote without video."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import chsims_teacher_neutral_pair_review as pair
from scripts import chsims_teacher_neutral_precision_review as review
from scripts import chsims_teacher_source_calibration as calibrated
from scripts import chsims_teacher_timeline as timeline
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_teacher_cross_source import metrics
from scripts.chsims_teacher_data import CACHE, sha_file


MODEL = "acoustic-continuity-review.json"


def acoustic_margin(record):
    local = record["retained"][:, 1]
    available = np.isfinite(local).all(1) & np.isfinite(record["mini"]).all(1)
    isolated = ((record["mini"].argmax(1) == pair.POSITIVE) &
                (record["acoustic"].argmax(1) != pair.POSITIVE))
    return np.where(available & isolated, calibrated.signed_margin(local, pair.POSITIVE), 0.)


def adjusted_scores(confidence, margin, gain):
    adjusted = expit(logit(np.clip(confidence, 1e-8, 1. - 1e-8)) + gain * margin)
    return np.where(margin != 0., adjusted, confidence)


def apply(record, probability, old, settings):
    old_rule = old["rule"]
    if settings is None:
        proposal, selected = pair.reference_result(record, probability, old)
    else:
        rule = {**old_rule, "min_target": settings["min_target"], "min_lead": settings["min_lead"]}
        margin = acoustic_margin(record)
        improved = adjusted_scores(probability, margin, settings["gain"])
        proposed = review.proposal(record["initial"], improved, pair.POSITIVE, old_rule["mix"])
        selected = review.select(record, proposed, improved, rule, review.support(record, pair.POSITIVE)[rule["cue"]])
        unadjusted, previously_selected = pair.reference_result(record, probability, old)
        changed_cue = margin != 0.
        proposal = np.where(changed_cue[:, None], proposed, unadjusted)
        selected = np.where(changed_cue, selected, previously_selected)
    return np.where(selected[:, None], proposal, record["initial"])


def choose(records, batches, old):
    posterior = pair.reference_confidence(records, batches, old)
    previous = {phase: apply(record, posterior[phase], old, None) for phase, record in records.items()}
    before = {phase: metrics(batches[phase]["target"], scores) for phase, scores in previous.items()}
    weights = {phase: calibrated.source_weights(batch["source"]) for phase, batch in batches.items()}
    adjusted_group = {phase: record["active"] & (acoustic_margin(record) != 0.)
                      for phase, record in records.items()}
    best = {"rule": None, "before": before, "after": before}
    best_rank = (0., 0., 0., 0.)
    for gain, target, lead in product((.25, .5, 1., 1.5, 2., 3., 4.), (.45, .5, .55, .6), (.1, .15, .2)):
        settings = {"gain": gain, "min_target": target, "min_lead": lead}
        utility, brier_gain, changes, after, split_gain = 0., 0., 0, {}, {}
        for phase, record in records.items():
            truth = batches[phase]["target"]
            new = apply(record, posterior[phase], old, settings)
            prior_hits = previous[phase].argmax(1) == truth
            hits = new.argmax(1) == truth
            if np.any(prior_hits & ~hits):
                break
            result = metrics(truth, new)
            if result["macro_f1"] < before[phase]["macro_f1"]:
                break
            after[phase] = result
            scale = .35 if phase == "train" else .65
            utility += scale * float((hits.astype(int) - prior_hits) @ weights[phase])
            mask = adjusted_group[phase]
            label = (truth == pair.POSITIVE)[mask]
            raw = posterior[phase][mask]
            revised = adjusted_scores(raw, acoustic_margin(record)[mask], gain)
            split_gain[phase] = float(((raw - label)**2 - (revised - label)**2) @ weights[phase][mask])
            brier_gain += scale * split_gain[phase]
            changes += int((new.argmax(1) != previous[phase].argmax(1)).sum())
        drift = abs(target - old["rule"]["min_target"]) + abs(lead - old["rule"]["min_lead"])
        rank = utility, brier_gain, -changes, -drift
        if len(after) == len(batches) and rank > best_rank:
            best_rank = rank
            best = {"rule": settings, "before": before, "after": after,
                    "calibration_development": {"improvement": split_gain,
                                                "counts": {k: int(v.sum()) for k, v in adjusted_group.items()}}}
    return best


def infer(batch, incumbent, policy):
    old = policy["reference"]
    record = review.observe(batch, incumbent)
    rule = old["rule"]
    design = review.features(record, pair.POSITIVE, rule["family"])
    confidence = review.predict(design, old["head"], rule["mode"], pair.POSITIVE)
    return apply(record, confidence, old, policy["rule"])


def provenance(directory, splits):
    return {"code": sha_file(Path(__file__)), "features": sha_file(Path(pair.__file__)),
            "reference": sha_file(directory / review.MODEL), "splits": splits}


def fit(cache):
    records, batches, old, splits = pair.development_records(cache)
    directory = cache / timeline.DIRECTORY
    policy = choose(records, batches, old)
    save_json(directory / MODEL, {**provenance(directory, splits), "policy": policy})
    print(policy["rule"], flush=True)
    for phase in batches:
        print(phase, {key: (policy[key][phase]["correct"], policy[key][phase]["macro_f1"],
                           np.diag(policy[key][phase]["confusion"]).tolist()) for key in ("before", "after")}, flush=True)


def load(cache, splits):
    directory = cache / timeline.DIRECTORY
    model = json.loads((directory / MODEL).read_text("utf8"))
    if any(model.get(key) != value for key, value in provenance(directory, splits).items()):
        raise ValueError("Fit acoustic continuity on the frozen development evidence first")
    return {**model["policy"], "reference": review.load(cache, splits)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    with threadpool_limits(limits=5, user_api="blas"):
        fit(parser.parse_args().cache)
