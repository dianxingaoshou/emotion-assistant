"""Review neutral predictions using retained text and acoustic windows."""

import argparse
import json
from itertools import product
from pathlib import Path
import sys

import numpy as np

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


MODEL = "temporal-neutral-review.json"
CONDITION = precision.CONDITION
CHANNELS = ("text", "audio")
PHASES = ("train", "valid")
NEUTRAL = LABELS.index("neutral")
POLARITIES = (LABELS.index("negative"), LABELS.index("positive"))


def observe(batch, incumbent):
    retained = batch["retained"]
    if retained.shape != (len(incumbent), len(CHANNELS), len(LABELS)):
        raise ValueError("Expected retained-window text and acoustic probabilities")
    available = np.isfinite(retained).all(axis=2)
    original = precision.describe(batch, incumbent)
    active = (original["active"] & np.isfinite(original["support"]).all(axis=(1, 2))
              & (incumbent.argmax(axis=1) == NEUTRAL))
    return {"initial": incumbent, "retained": retained, "active": active,
            "available": available, "top": retained.argmax(axis=2)}


def propose(record, channel, weight):
    return (1. - weight) * record["initial"] + weight * record["retained"][:, channel]


def select(record, proposed, rule):
    target = rule["target"]
    initial = record["initial"]
    support = (record["available"] & (record["top"] == target)).any(axis=1)
    return (record["active"] & record["available"][:, rule["channel"]] & support
            & (proposed.argmax(axis=1) == target)
            & (initial[:, NEUTRAL] - initial[:, target] <= rule["max_gap"])
            & (proposed[:, target] - proposed[:, NEUTRAL] >= rule["min_lead"]))


def choose(records, batches):
    before = {phase: metrics(batch["target"], records[phase]["initial"]) for phase, batch in batches.items()}
    weights = {phase: calibrated.source_weights(batch["source"]) for phase, batch in batches.items()}
    best = {"rule": None, "before": before, "after": before}
    rank = (0., 0., 0, 0., 0., 0.)
    for channel, weight in product(range(len(CHANNELS)), (.25, .40, .55, .70, .85, 1.)):
        scores = {phase: propose(record, channel, weight) for phase, record in records.items()}
        for target, gap, lead in product(POLARITIES, (.04, .08, .13, .20, .32, .50, 1.),
                                         (.015, .035, .06, .11, .19, .28)):
            rule = {"channel": channel, "weight": weight, "target": target,
                    "max_gap": gap, "min_lead": lead}
            picked = {phase: select(record, scores[phase], rule) for phase, record in records.items()}
            truth = {phase: batch["target"][picked[phase]] for phase, batch in batches.items()}
            if any(np.any(values == NEUTRAL) or not np.any(values == target) for values in truth.values()):
                continue
            output = {phase: np.where(mask[:, None], scores[phase], records[phase]["initial"])
                      for phase, mask in picked.items()}
            after = {phase: metrics(batch["target"], output[phase]) for phase, batch in batches.items()}
            if any(after[phase]["macro_f1"] < before[phase]["macro_f1"]
                   or np.any(np.diag(after[phase]["confusion"]) < np.diag(before[phase]["confusion"]))
                   for phase in PHASES):
                continue
            gain = {phase: weights[phase][picked[phase]][truth[phase] == target].sum() for phase in PHASES}
            count = sum(mask.sum() for mask in picked.values())
            quality = (.35 * gain["train"] + .65 * gain["valid"] - .000004 * count,
                       after["valid"]["macro_f1"], -count, -gap, -weight, lead)
            if quality > rank:
                best, rank = {"rule": rule, "before": before, "after": after}, quality
    return best


def infer(batch, incumbent, policy):
    rule = policy["rule"]
    if rule is None:
        return incumbent
    record = observe(batch, incumbent)
    proposal = propose(record, rule["channel"], rule["weight"])
    return np.where(select(record, proposal, rule)[:, None], proposal, incumbent)


def provenance(directory, splits):
    return {"code": sha_file(Path(__file__)), "reference": sha_file(directory / neutral_guard.MODEL),
            "features": sha_file(Path(precision.__file__)),
            "retained_development": sha_file(directory / evidence.DEVELOPMENT), "splits": splits}


def fit(cache):
    directory = cache / timeline.DIRECTORY
    previous = neutral_guard.load(cache, directory / "timeline-fusion.json")
    acoustic = audio_evidence.load(cache)
    development = evidence.load_development(cache)
    batches = {phase: cases[CONDITION] for phase, cases in development.items()}
    for phase, batch in batches.items():
        rows = read_full_report(cache, phase)["rows"]
        batch.update(calibrated.acoustic_inputs(rows, acoustic[phase]))
        batch["quality"] = dynamic.quality_input(batch)
        batch["retained"] = development[phase]["synchronous_gap"]["context"][:, :len(CHANNELS)]
    prior_policy = previous["audio_review"]["policies"][CONDITION]
    incumbent = precision.development_incumbent(batches, prior_policy)
    records = {phase: observe(batch, incumbent[phase]) for phase, batch in batches.items()}
    policy = choose(records, batches)
    signature = provenance(directory, previous["splits"])
    save_json(directory / MODEL, {**signature, "policy": policy})
    for phase in PHASES:
        print(phase, {step: (policy[step][phase]["correct"], policy[step][phase]["macro_f1"],
                            np.diag(policy[step][phase]["confusion"]).tolist()) for step in ("before", "after")},
              flush=True)
    print(policy["rule"], flush=True)


def load(cache, splits):
    directory = cache / timeline.DIRECTORY
    model = json.loads((directory / MODEL).read_text("utf8"))
    if any(model.get(key) != value for key, value in provenance(directory, splits).items()):
        raise ValueError("Fit temporal neutral review on the frozen development evidence first")
    return model["policy"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    fit(parser.parse_args().cache)
