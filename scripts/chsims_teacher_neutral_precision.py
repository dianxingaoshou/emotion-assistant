"""Fit a source-held-out review of neutral predictions without picture evidence."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
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
from scripts.chsims_teacher_supervision import read_full_report


MODEL = "neutral-precision.json"
CONDITION = precision.CONDITION
NEUTRAL = LABELS.index("neutral")
POLARITIES = (LABELS.index("negative"), LABELS.index("positive"))
PHASES = ("train", "valid")
PENALTIES = (.005, .02, .08, .25, .8)
SHARES = (.6, 1., 1.5)
GAPS = (.02, .04, .06, .09, .12, .18, .25, .35, .50, 1.)
LEADS = (.001, .025, .05, .08, .10, .15, .20, .30)
SUPPORT = ("text_and_audio", "text_or_audio", "mini_audio", "text_or_mini")


def observe(batch, incumbent):
    record = precision.describe(batch, incumbent)
    record["active"] &= (incumbent.argmax(1) == NEUTRAL) & np.isfinite(record["support"]).all(axis=(1, 2))
    record["mini_audio"] = batch["voice"]
    return record


def train_head(record, batch, selected, penalty):
    return timeline.learn_residual(record["features"][selected], record["initial"][selected],
                                   batch["target"][selected], calibrated.source_weights(batch["source"][selected]),
                                   penalty)


def propose(record, head, share, selected):
    policy = {**head, "share": share, "neutral_offset": 0.}
    return timeline.apply_residual(record["features"], record["initial"], policy, selected)


def development_proposals(records, batches, penalty, share):
    sources = np.concatenate([batches[phase]["source"] for phase in PHASES])
    proposals = {phase: record["initial"].copy() for phase, record in records.items()}
    for _, held in GroupKFold(4).split(sources, groups=sources):
        withheld = np.unique(sources[held])
        selected = records["train"]["active"] & ~np.isin(batches["train"]["source"], withheld)
        head = train_head(records["train"], batches["train"], selected, penalty)
        for phase, record in records.items():
            selected = record["active"] & np.isin(batches[phase]["source"], withheld)
            scores = propose(record, head, share, selected)
            proposals[phase][selected] = scores[selected]
    return proposals


def supporting(record, target, support):
    text, audio = record["support"].transpose(1, 0, 2)
    text_votes = text[:, target] > text[:, NEUTRAL]
    audio_votes = audio[:, target] > audio[:, NEUTRAL]
    mini_votes = record["mini_audio"][:, target] > record["mini_audio"][:, NEUTRAL]
    if support == "text_and_audio":
        return text_votes & audio_votes
    if support == "text_or_audio":
        return text_votes | audio_votes
    if support == "mini_audio":
        return mini_votes
    if support == "text_or_mini":
        return text_votes | mini_votes
    raise ValueError(f"Unknown corroboration: {support}")


def offer(record, scores, target, support):
    return (record["active"] & (scores.argmax(1) == target) & supporting(record, target, support),
            record["initial"][:, NEUTRAL] - record["initial"][:, target], scores[:, target] - scores[:, NEUTRAL])


def choose_mask(offered, maximum_gap, minimum_lead):
    active, old_gap, new_lead = offered
    return active & (old_gap <= maximum_gap) & (new_lead >= minimum_lead)


def candidate_rules(records, batches):
    best = {}
    for penalty, share in product(PENALTIES, SHARES):
        scores = development_proposals(records, batches, penalty, share)
        for target, support in product(POLARITIES, SUPPORT):
            offered = {phase: offer(record, scores[phase], target, support) for phase, record in records.items()}
            for gap, lead in product(GAPS, LEADS):
                masks = {phase: choose_mask(offered[phase], gap, lead) for phase in PHASES}
                truth = {phase: batch["target"][masks[phase]] for phase, batch in batches.items()}
                if any(np.any(y == NEUTRAL) or not np.any(y == target) for y in truth.values()):
                    continue
                # Retain the most useful rule for each distinct decision set.
                key = target, *(mask.tobytes() for mask in masks.values())
                rank = -gap, lead, penalty, -share
                if key in best and rank <= best[key][0]:
                    continue
                rule = {"target": target, "support": support, "penalty": penalty, "share": share,
                        "max_initial_gap": gap, "min_proposal_lead": lead}
                best[key] = rank, rule, masks, scores
    return [item[1:] for item in best.values()]


def apply_offers(initial, offers):
    predictions = initial.copy()
    counts = np.zeros(len(initial), dtype=np.uint8)
    for mask, scores in offers:
        counts += mask
        predictions[mask] = scores[mask]
    predictions[counts > 1] = initial[counts > 1]
    return predictions


def choose(records, batches):
    baseline = {phase: metrics(batches[phase]["target"], record["initial"]) for phase, record in records.items()}
    weights = {phase: calibrated.source_weights(batch["source"]) for phase, batch in batches.items()}
    candidates = candidate_rules(records, batches)
    grouped = [[None] + [candidate for candidate in candidates if candidate[0]["target"] == target]
               for target in POLARITIES]
    rank = (0., 0., 0.)
    selected, after = [], baseline
    for pair in product(*grouped):
        used = [candidate for candidate in pair if candidate is not None]
        if not used:
            continue
        measured, utility = {}, []
        for phase in PHASES:
            truth, initial = batches[phase]["target"], records[phase]["initial"]
            predicted = apply_offers(initial, [(masks[phase], scores[phase]) for _, masks, scores in used])
            measured[phase] = metrics(truth, predicted)
            hits = np.diag(measured[phase]["confusion"])
            if (np.any(hits < np.diag(baseline[phase]["confusion"]))
                    or measured[phase]["macro_f1"] < baseline[phase]["macro_f1"]):
                break
            gain = (predicted.argmax(1) == truth).astype(int) - (initial.argmax(1) == truth)
            utility.append(float(gain @ weights[phase]))
        if len(utility) != len(PHASES):
            continue
        quality = .35 * utility[0] + .65 * utility[1], measured["valid"]["macro_f1"], -len(used)
        if quality > rank:
            selected, after, rank = [rule for rule, _, _ in used], measured, quality
    return {"rules": selected, "before": baseline, "after": after}


def infer(batch, incumbent, policy):
    if not policy["rules"]:
        return incumbent
    record = observe(batch, incumbent)
    offers = []
    for rule in policy["rules"]:
        scores = propose(record, rule["head"], rule["share"], record["active"])
        offered = offer(record, scores, rule["target"], rule["support"])
        offers.append((choose_mask(offered, rule["max_initial_gap"], rule["min_proposal_lead"]), scores))
    return apply_offers(incumbent, offers)


def provenance(directory, splits):
    return {"code": sha_file(Path(__file__)), "features": sha_file(Path(precision.__file__)),
            "reference": sha_file(directory / neutral_guard.MODEL), "splits": splits}


def fit(cache):
    directory = cache / timeline.DIRECTORY
    previous = neutral_guard.load(cache, directory / "timeline-fusion.json")
    acoustic = audio_evidence.load(cache)
    batches = {phase: doc[CONDITION] for phase, doc in evidence.load_development(cache).items()}
    for phase, batch in batches.items():
        batch.update(calibrated.acoustic_inputs(read_full_report(cache, phase)["rows"], acoustic[phase]))
        batch["quality"] = dynamic.quality_input(batch)
    incumbent = precision.development_incumbent(batches, previous["audio_review"]["policies"][CONDITION])
    records = {phase: observe(batch, incumbent[phase]) for phase, batch in batches.items()}
    policy = choose(records, batches)
    print(policy["rules"], {phase: (score["correct"], round(score["macro_f1"], 5),
                                   np.diag(score["confusion"]).tolist())
                            for phase, score in policy["after"].items()}, flush=True)
    for rule in policy["rules"]:
        rule["head"] = train_head(records["train"], batches["train"], records["train"]["active"], rule["penalty"])
    save_json(directory / MODEL, {**provenance(directory, previous["splits"]), "policy": policy})


def load(cache, splits):
    directory = cache / timeline.DIRECTORY
    checkpoint = json.loads((directory / MODEL).read_text("utf8"))
    if any(checkpoint.get(key) != value for key, value in provenance(directory, splits).items()):
        raise ValueError("Fit the neutral precision review on the frozen development data first")
    return checkpoint["policy"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    with threadpool_limits(limits=5, user_api="blas"):
        fit(parser.parse_args().cache)
