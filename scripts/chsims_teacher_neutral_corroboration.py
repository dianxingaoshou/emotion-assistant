"""Select corroborated neutral-to-polar corrections without visual evidence."""

import argparse
from itertools import product
import json
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


MODEL = "neutral-corroboration.json"
CONDITION = precision.CONDITION
NEUTRAL = LABELS.index("neutral")
POLARITIES = (LABELS.index("negative"), LABELS.index("positive"))
PHASES = ("train", "valid")


def observe(batch, initial):
    evidence = np.stack((batch["context"][:, 0], batch["acoustic"], batch["voice"]), axis=1)
    active = (batch["active"] & (initial.argmax(1) == NEUTRAL) & np.isfinite(initial).all(1)
              & np.isfinite(evidence).all(axis=(1, 2)))
    return {"initial": initial, "voice": evidence[:, 2], "active": active,
            "labels": evidence.argmax(2), "anchor": batch["prior"].argmax(1)}


def propose(record, voice_weight):
    return (1. - voice_weight) * record["initial"] + voice_weight * record["voice"]


def select(record, scores, target, support, max_gap, min_lead):
    text, audio, mini = (record["labels"] == target).T
    if support == "two_acoustic_models":
        agreed = audio & mini
    elif support == "text_and_acoustic":
        agreed = text & (audio | mini)
    elif support == "two_models":
        agreed = (text & (audio | mini)) | (audio & mini)
    elif support == "prior_and_acoustic":
        agreed = (record["anchor"] == target) & (audio | mini)
    else:
        raise ValueError(f"Unknown corroboration: {support}")
    initial = record["initial"]
    return (record["active"] & agreed & (scores.argmax(1) == target)
            & (initial[:, NEUTRAL] - initial[:, target] <= max_gap)
            & (scores[:, target] - scores[:, NEUTRAL] >= min_lead))


def choose(batches, records):
    before = {split: metrics(batch["target"], records[split]["initial"]) for split, batch in batches.items()}
    source_weights = {split: calibrated.source_weights(batch["source"]) for split, batch in batches.items()}
    best = {"rule": None, "before": before, "after": before}
    rank = (0., 0., 0., 0., 0., 0.)
    modes = ("two_models", "two_acoustic_models", "text_and_acoustic", "prior_and_acoustic")
    proposals = {mix: {split: propose(record, mix) for split, record in records.items()} for mix in (.3, .6, 1.)}
    for (mix, scores), target, support, gap, lead in product(
            proposals.items(), POLARITIES, modes, (.02, .04, .06, .09, .12, .18, .25, .35, .5, 1.),
            (.001, .025, .05, .08, .1, .15, .2, .3)):
        selected = {split: select(record, scores[split], target, support, gap, lead) for split, record in records.items()}
        outcomes = {split: batch["target"][selected[split]] for split, batch in batches.items()}
        if any(np.any(y == NEUTRAL) or not np.any(y == target) for y in outcomes.values()):
            continue
        weights = {split: source_weights[split][selected[split]][y == target].sum() for split, y in outcomes.items()}
        predicted = {split: np.where(mask[:, None], scores[split], records[split]["initial"])
                     for split, mask in selected.items()}
        after = {split: metrics(batches[split]["target"], values) for split, values in predicted.items()}
        if any(after[split]["macro_f1"] < before[split]["macro_f1"] for split in PHASES):
            continue
        quality = (.35 * weights["train"] + .65 * weights["valid"], after["valid"]["macro_f1"],
                   -sum(mask.sum() for mask in selected.values()), modes.index(support), -gap, lead)
        if quality > rank:
            rank = quality
            best = {"rule": {"target": target, "support": support, "voice_weight": mix,
                            "max_gap": gap, "min_lead": lead}, "before": before, "after": after}
    return best


def infer(batch, incumbent, policy):
    rule = policy["rule"]
    if rule is None:
        return incumbent
    record = observe(batch, incumbent)
    scores = propose(record, rule["voice_weight"])
    selected = select(record, scores, rule["target"], rule["support"], rule["max_gap"], rule["min_lead"])
    return np.where(selected[:, None], scores, incumbent)


def provenance(directory, splits):
    return {"code": sha_file(Path(__file__)), "reference": sha_file(directory / neutral_guard.MODEL),
            "development_code": sha_file(Path(precision.__file__)), "splits": splits}


def fit(cache):
    directory = cache / timeline.DIRECTORY
    previous = neutral_guard.load(cache, directory / "timeline-fusion.json")
    acoustic = audio_evidence.load(cache)
    batches = {split: doc[CONDITION] for split, doc in evidence.load_development(cache).items()}
    for split, batch in batches.items():
        batch.update(calibrated.acoustic_inputs(read_full_report(cache, split)["rows"], acoustic[split]))
        batch["quality"] = dynamic.quality_input(batch)
    incumbent = precision.development_incumbent(batches, previous["audio_review"]["policies"][CONDITION])
    records = {split: observe(batch, incumbent[split]) for split, batch in batches.items()}
    policy = choose(batches, records)
    print(policy["rule"], {split: (m["correct"], round(m["macro_f1"], 5), np.diag(m["confusion"]).tolist())
                           for split, m in policy["after"].items()}, flush=True)
    save_json(directory / MODEL, {**provenance(directory, previous["splits"]), "policy": policy})


def load(cache, splits):
    directory = cache / timeline.DIRECTORY
    checkpoint = json.loads((directory / MODEL).read_text("utf8"))
    if any(checkpoint.get(key) != value for key, value in provenance(directory, splits).items()):
        raise ValueError("Fit the corroborated review on frozen development data first")
    return checkpoint["policy"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    fit(parser.parse_args().cache)
