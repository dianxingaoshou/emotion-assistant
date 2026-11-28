"""Protect neutral decisions when text outweighs contrary voice evidence."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import chsims_teacher_audio_evidence as audio_evidence
from scripts import chsims_teacher_context_evidence as evidence
from scripts import chsims_teacher_dynamic_conflict as dynamic
from scripts import chsims_teacher_source_calibration as calibrated
from scripts import chsims_teacher_timeline as timeline
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_teacher_cross_source import metrics
from scripts.chsims_teacher_data import CACHE, sha_file
from scripts.chsims_teacher_supervision import read_full_report


MODEL = "neutral-guard.json"
NEGATIVE, NEUTRAL = calibrated.NEGATIVE, calibrated.NEUTRAL


def trustworthy_negative(observed, native, text_floor):
    acoustic = observed["audio"]
    pick = observed["audio_pick"]
    acoustic_top = acoustic[np.arange(len(pick)), pick] >= acoustic.max(axis=1)
    contrary_voice = (pick == NEUTRAL) & (native[:, NEUTRAL] > native[:, NEGATIVE])
    text = observed["text"]
    clear_text = text[:, NEGATIVE] - text[:, NEUTRAL] >= text_floor
    return acoustic_top & (~contrary_voice | clear_text)


def veto(batch, observed, changed, text_floor):
    return changed & ~trustworthy_negative(observed, batch["voice"], text_floor)


def choose(batches, previous):
    base = {split: calibrated.infer(batch, previous["reference"]) for split, batch in batches.items()}
    observed = {split: dynamic.describe(batch, base[split]) for split, batch in batches.items()}
    shares = dynamic.held_shares(observed, batches, previous["strength"])
    former = previous["rule"]
    proposals = {split: dynamic.fuse(item, np.clip(shares[split] + former["offset"], 0., 1.),
                                     base[split], former["anchor"]) for split, item in observed.items()}
    changed = {split: dynamic.accept(item, proposals[split], former) for split, item in observed.items()}
    targets = {split: batch["target"] for split, batch in batches.items()}
    weights = {split: calibrated.source_weights(batch["source"]) for split, batch in batches.items()}
    prior = {split: np.where(mask[:, None], proposals[split], base[split]) for split, mask in changed.items()}
    baseline = {split: metrics(targets[split], scores) for split, scores in prior.items()}
    macro_floor = {split: metrics(targets[split], scores)["macro_f1"] for split, scores in base.items()}
    rank = (0., 0.)
    best = {"text_floor": None, "before": baseline, "after": baseline}
    for floor in (.00, .05, .10, .15, .20, .25, .30):
        blocked = {split: veto(batch, observed[split], changed[split], floor) for split, batch in batches.items()}
        if any(np.any(mask & (targets[split] == NEGATIVE)) for split, mask in blocked.items()):
            continue
        reports = {split: metrics(targets[split], np.where(mask[:, None], base[split], prior[split]))
                   for split, mask in blocked.items()}
        if any(reports[split]["macro_f1"] < macro_floor[split] for split in reports):
            continue
        recovered = {split: mask & (targets[split] == NEUTRAL) for split, mask in blocked.items()}
        utility = .35 * weights["train"][recovered["train"]].sum() + .65 * weights["valid"][recovered["valid"]].sum()
        # For equal development performance, demand more support before dropping neutral.
        quality = float(utility), floor
        if quality > rank:
            rank = quality
            best = {"text_floor": floor, "before": baseline, "after": reports}
    return best


def infer(batch, policy):
    previous = policy["reference"]
    current = dynamic.infer(batch, previous)
    if previous["case"] != calibrated.NO_VISUAL or policy["text_floor"] is None:
        return current
    base = calibrated.infer(batch, previous["reference"])
    observed = dynamic.describe(batch, base)
    changed = (observed["old"] == NEUTRAL) & (current.argmax(1) == NEGATIVE)
    blocked = veto(batch, observed, changed, policy["text_floor"])
    return np.where(blocked[:, None], base, current)


def provenance(directory, splits):
    return {"implementation": sha_file(Path(__file__)), "previous": sha_file(directory / dynamic.MODEL),
            "splits": splits}


def fit(cache):
    directory = cache / timeline.DIRECTORY
    previous = dynamic.load(cache, directory / "timeline-fusion.json")
    acoustic = audio_evidence.load(cache)
    batches = {split: cases[calibrated.NO_VISUAL] for split, cases in evidence.load_development(cache).items()}
    for split, batch in batches.items():
        batch.update(calibrated.acoustic_inputs(read_full_report(cache, split)["rows"], acoustic[split]))
        batch["quality"] = dynamic.quality_input(batch)
    policy = choose(batches, previous["audio_review"]["policies"][calibrated.NO_VISUAL])
    print(policy["text_floor"], {split: (m["correct"], round(m["macro_f1"], 6), np.diag(m["confusion"]).tolist())
                                 for split, m in policy["after"].items()}, flush=True)
    save_json(directory / MODEL, {**provenance(directory, previous["splits"]), "policy": policy})


def load(cache, reference):
    directory = cache / timeline.DIRECTORY
    previous = dynamic.load(cache, reference)
    checkpoint = json.loads((directory / MODEL).read_text("utf8"))
    if any(checkpoint.get(key) != val for key, val in provenance(directory, previous["splits"]).items()):
        raise ValueError("Fit neutral protection on the frozen development inputs first")
    audio = previous["audio_review"]
    policies = {case: {**checkpoint["policy"], "reference": policy} for case, policy in audio["policies"].items()}
    return {**previous, "kind": "neutral_guard", "audio_review": {**audio, "policies": policies}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    fit(parser.parse_args().cache)
