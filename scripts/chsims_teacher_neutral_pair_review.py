"""Compare positive and neutral evidence directly before revising a no-video result."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import chsims_teacher_audio_evidence as audio_evidence
from scripts import chsims_teacher_context_evidence as context_evidence
from scripts import chsims_teacher_dynamic_conflict as dynamic
from scripts import chsims_teacher_neutral_guard as neutral_guard
from scripts import chsims_teacher_neutral_precision_review as review
from scripts import chsims_teacher_precision_review as precision
from scripts import chsims_teacher_source_calibration as calibrated
from scripts import chsims_teacher_timeline as timeline
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_teacher_cross_source import metrics
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file
from scripts.chsims_teacher_supervision import read_full_report


MODEL = "neutral-pair-review.json"
NEGATIVE, NEUTRAL, POSITIVE = (LABELS.index(label) for label in ("negative", "neutral", "positive"))


def reference_result(record, confidence, policy):
    rule = policy["rule"]
    proposal = review.proposal(record["initial"], confidence, POSITIVE, rule["mix"])
    selected = review.select(record, proposal, confidence, rule, review.support(record, POSITIVE)[rule["cue"]])
    return proposal, selected


def paired_scores(incumbent, confidence):
    positive = (1. - incumbent[:, NEGATIVE]) * confidence
    return np.column_stack((incumbent[:, NEGATIVE], 1. - incumbent[:, NEGATIVE] - positive, positive))


def apply(record, reference, paired, rule):
    incumbent = record["initial"]
    proposal, selected = reference
    if rule is None:
        return np.where(selected[:, None], proposal, incumbent)
    scores, confidence = paired
    kept = selected & (confidence >= rule["keep_floor"])
    positive_cues = review.support(record, POSITIVE)
    admitted = (record["neutral"] & record["old_neutral"] & ~selected & positive_cues[rule["cue"]] &
                (incumbent[:, NEUTRAL] - incumbent[:, POSITIVE] <= rule["max_gap"]) &
                (confidence >= rule["entry_floor"]) & (scores.argmax(1) == POSITIVE))
    return np.where(kept[:, None], proposal, np.where(admitted[:, None], scores, incumbent))


def reference_confidence(records, batches, previous):
    reference_rule = previous["rule"]
    reference_design = {phase: review.features(record, POSITIVE, reference_rule["family"])
                        for phase, record in records.items()}
    return review.cross_fitted(records, batches, reference_design, POSITIVE,
                               reference_rule["mode"], reference_rule["subset"], reference_rule["strength"])


def choose(records, batches, previous):
    old = reference_confidence(records, batches, previous)
    proposals = {phase: reference_result(record, old[phase], previous) for phase, record in records.items()}
    initial = {phase: apply(record, proposals[phase], None, None) for phase, record in records.items()}
    before = {phase: metrics(batch["target"], initial[phase]) for phase, batch in batches.items()}
    weights = {phase: calibrated.source_weights(batch["source"]) for phase, batch in batches.items()}
    best = {"rule": None, "before": before, "after": before}
    best_rank = (0., 0., 0.)
    for family, subset, strength in product(("compact", "disagreement", "expanded"),
                                            ("pair_available", "pair_undecided"), (.02, .12, .6)):
        design = {phase: review.features(record, POSITIVE, family) for phase, record in records.items()}
        probability = review.cross_fitted(records, batches, design, POSITIVE, "binary", subset, strength)
        paired = {phase: (paired_scores(record["initial"], probability[phase]), probability[phase])
                  for phase, record in records.items()}
        for keep, entry, gap, cue in product((0., .5, .58, .66, .74), (.54, .61, .68, .75, .82),
                                             (.12, .2, .3, .42), review.CUES):
            rule = {"keep_floor": keep, "entry_floor": entry, "max_gap": gap, "cue": cue,
                    "family": family, "subset": subset, "strength": strength}
            utilities, after, changes = {}, {}, 0
            for phase, batch in batches.items():
                truth = batch["target"]
                scores = apply(records[phase], proposals[phase], paired[phase], rule)
                hits = scores.argmax(1) == truth
                old_hits = initial[phase].argmax(1) == truth
                delta = hits.astype(int) - old_hits
                changed = scores.argmax(1) != initial[phase].argmax(1)
                if (np.any(np.bincount(truth, weights=delta, minlength=len(LABELS)) < 0) or
                        np.any(changed & (truth == NEUTRAL) & ~hits)):
                    break
                result = metrics(truth, scores)
                if result["macro_f1"] < before[phase]["macro_f1"]:
                    break
                after[phase] = result
                utilities[phase] = float(delta @ weights[phase])
                changes += int(changed.sum())
            if len(after) != len(batches):
                continue
            rank = (.35 * utilities["train"] + .65 * utilities["valid"],
                    after["valid"]["macro_f1"] - before["valid"]["macro_f1"], -changes)
            if rank > best_rank:
                best_rank = rank
                best = {"rule": rule, "before": before, "after": after}
    if best["rule"] is not None:
        rule = best["rule"]
        train = records["train"]
        design = review.features(train, POSITIVE, rule["family"])
        best["head"] = review.fit_head(train, batches["train"], design, np.ones(len(design), dtype=bool),
                                       POSITIVE, "binary", rule["subset"], rule["strength"])
    return best


def infer(batch, incumbent, policy):
    previous = policy["reference"]
    if policy["rule"] is None:
        return review.infer(batch, incumbent, previous)
    record = review.observe(batch, incumbent)
    reference_rule = previous["rule"]
    design = review.features(record, POSITIVE, reference_rule["family"])
    old = review.predict(design, previous["head"], reference_rule["mode"], POSITIVE)
    rule = policy["rule"]
    design = review.features(record, POSITIVE, rule["family"])
    probability = review.predict(design, policy["head"], "binary", POSITIVE)
    return apply(record, reference_result(record, old, previous),
                 (paired_scores(incumbent, probability), probability), rule)


def provenance(directory, splits):
    return {"code": sha_file(Path(__file__)), "released_review": sha_file(directory / review.MODEL), "splits": splits}


def development_records(cache):
    directory = cache / timeline.DIRECTORY
    checkpoint = neutral_guard.load(cache, directory / "timeline-fusion.json")
    previous = review.load(cache, checkpoint["splits"])
    if not previous["rule"] or previous["rule"]["target"] != POSITIVE:
        raise ValueError("The reference must review neutral-to-positive decisions")
    acoustic = audio_evidence.load(cache)
    development = context_evidence.load_development(cache)
    batches = {phase: conditions[precision.CONDITION] for phase, conditions in development.items()}
    for phase, batch in batches.items():
        batch.update(calibrated.acoustic_inputs(read_full_report(cache, phase)["rows"], acoustic[phase]))
        batch["quality"] = dynamic.quality_input(batch)
        batch["retained"] = development[phase]["synchronous_gap"]["context"][:, :2]
    incumbent = precision.development_incumbent(batches, checkpoint["audio_review"]["policies"][precision.CONDITION])
    records = {}
    for phase, batch in batches.items():
        record = review.observe(batch, incumbent[phase])
        pair = np.isin(batch["target"], (NEUTRAL, POSITIVE))
        records[phase] = {**record, "pair_available": pair & record["active"],
                          "pair_undecided": pair & record["neutral"]}
    return records, batches, previous, checkpoint["splits"]


def fit(cache):
    directory = cache / timeline.DIRECTORY
    records, batches, previous, splits = development_records(cache)
    policy = choose(records, batches, previous)
    save_json(directory / MODEL, {**provenance(directory, splits), "policy": policy})
    print(policy["rule"], flush=True)
    for phase in batches:
        print(phase, {key: (policy[key][phase]["correct"], policy[key][phase]["macro_f1"],
                           np.diag(policy[key][phase]["confusion"]).tolist()) for key in ("before", "after")}, flush=True)


def load(cache, splits):
    directory = cache / timeline.DIRECTORY
    model = json.loads((directory / MODEL).read_text("utf8"))
    if any(model.get(key) != value for key, value in provenance(directory, splits).items()):
        raise ValueError("Fit the paired review on the frozen development evidence first")
    return {**model["policy"], "reference": review.load(cache, splits)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    with threadpool_limits(limits=5, user_api="blas"):
        fit(parser.parse_args().cache)
