"""Confirm acoustic-continuity revisions with a frozen neutral reliability model."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import chsims_teacher_acoustic_continuity_review as continuity
from scripts import chsims_teacher_neutral_pair_review as pair
from scripts import chsims_teacher_neutral_precision_review as published
from scripts import chsims_teacher_neutral_reliability_review as reliability
from scripts import chsims_teacher_timeline as timeline
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_teacher_cross_source import metrics
from scripts.chsims_teacher_data import CACHE, sha_file


MODEL = "protected-continuity-review.json"


def infer(batch, incumbent, policy):
    proposed = continuity.infer(batch, incumbent, policy["continuity"])
    neutral = policy["reliability"]
    rule = neutral["rule"]
    record = published.observe(batch, incumbent)
    estimates = reliability.probabilities(reliability.features(record, rule["family"]), neutral["heads"])
    return reliability.protect(incumbent, proposed, estimates, rule)


def provenance(directory, splits):
    return {"code": sha_file(Path(__file__)), "continuity": sha_file(directory / continuity.MODEL),
            "reliability": sha_file(directory / reliability.MODEL), "splits": splits}


def fit(cache):
    records, batches, old, splits = pair.development_records(cache)
    acoustic = continuity.load(cache, splits)
    neutral = reliability.load(cache, splits)
    settings = neutral["rule"]
    if settings is None or acoustic["rule"] is None:
        raise ValueError("Fit both development reviews before combining them")
    design = {phase: reliability.features(record, settings["family"]) for phase, record in records.items()}
    estimates = reliability.held_out_heads(records, batches, design, settings["subset"], settings["strength"])
    confidence = pair.reference_confidence(records, batches, old)
    before, after = {}, {}
    for phase, record in records.items():
        target = batches[phase]["target"]
        reference = continuity.apply(record, confidence[phase], old, None)
        candidate = continuity.apply(record, confidence[phase], old, acoustic["rule"])
        result = reliability.protect(record["initial"], candidate, estimates[phase], settings)
        if np.any((reference.argmax(1) == target) & (result.argmax(1) != target)):
            raise ValueError("The combined reviews lost an existing held-source correct result")
        before[phase] = metrics(target, reference)
        after[phase] = metrics(target, result)
    policy = {"rule": {"continuity": acoustic["rule"], "reliability": settings}, "before": before, "after": after}
    directory = cache / timeline.DIRECTORY
    save_json(directory / MODEL, {**provenance(directory, splits), "policy": policy})
    print({phase: (value["correct"], value["macro_f1"], np.diag(value["confusion"]).tolist())
           for phase, value in after.items()}, flush=True)


def load(cache, splits):
    directory = cache / timeline.DIRECTORY
    checkpoint = json.loads((directory / MODEL).read_text("utf8"))
    if any(checkpoint.get(key) != value for key, value in provenance(directory, splits).items()):
        raise ValueError("Verify the joint review on the frozen development evidence first")
    return {**checkpoint["policy"], "continuity": continuity.load(cache, splits),
            "reliability": reliability.load(cache, splits)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    with threadpool_limits(limits=5, user_api="blas"):
        fit(parser.parse_args().cache)
