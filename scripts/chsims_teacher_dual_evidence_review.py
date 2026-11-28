"""Confirm proposed positive reviews with whole-clip evidence on held sources."""

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
from scripts import chsims_teacher_neutral_consensus_review as consensus
from scripts import chsims_teacher_neutral_guard as neutral_guard
from scripts import chsims_teacher_precision_review as precision
from scripts import chsims_teacher_source_calibration as calibrated
from scripts import chsims_teacher_timeline as timeline
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_teacher_cross_source import metrics
from scripts.chsims_teacher_data import CACHE, sha_file
from scripts.chsims_teacher_supervision import read_full_report


MODEL = "dual-evidence-review.json"


def margin(scores):
    positive = scores[:, consensus.POSITIVE]
    competing = np.max(np.delete(scores, consensus.POSITIVE, axis=1), axis=1)
    return np.nan_to_num(positive - competing, nan=-1.)


def confirmation(batch, guard):
    if guard is None:
        return np.ones(len(batch["acoustic"]), dtype=bool)
    text = margin(batch["context"][:, 0])
    prosody = margin(batch["acoustic"])
    mini = margin(batch["voice"])
    local_text = margin(batch["retained"][:, 0])
    local_audio = margin(batch["retained"][:, 1])
    text_supported = ((text >= guard["text_floor"]) & (local_text >= guard["local_floor"]) &
                      (np.maximum(prosody, mini) >= guard["voice_floor"]))
    audio_supported = ((np.minimum(prosody, mini) >= guard["dual_audio_floor"]) &
                       (local_audio >= guard["local_audio_floor"]))
    return text_supported | audio_supported


def choose(batches, records, proposals, proposal_rule):
    neutral, positive = consensus.NEUTRAL, consensus.POSITIVE
    selected = {phase: consensus.select(record, proposals[phase], proposal_rule)
                for phase, record in records.items()}
    # Extra, borderline proposals give a less sparse estimate of false revisions.
    nearby_rule = {**proposal_rule, "min_lead": 0.0}
    nearby = {phase: consensus.select(record, proposals[phase], nearby_rule)
              for phase, record in records.items()}
    weights = {phase: calibrated.source_weights(batch["source"]) for phase, batch in batches.items()}
    baseline = {phase: metrics(batch["target"], records[phase]["initial"]) for phase, batch in batches.items()}
    previous = {phase: metrics(batch["target"], np.where(selected[phase][:, None], proposals[phase],
                                                        records[phase]["initial"]))
                for phase, batch in batches.items()}
    candidates = ({"text_floor": text, "local_floor": local, "voice_floor": voice, "dual_audio_floor": dual,
                   "local_audio_floor": audio}
                  for text, local, voice, dual, audio in product((-.25, -.15, -.05, .05, .25),
                                                                (-1., -.1, 0., .10, .20), (0., .05, .10, .25),
                                                                (0., .10, .25, .40, .60), (-1., -.20, -.10, 0., .10)))
    best = {"guard": None, "before": baseline, "previous": previous, "after": previous}
    rank = (0., 0.)
    for guard in candidates:
        utility, removed, after = 0., 0, {}
        for phase, batch in batches.items():
            target = batch["target"]
            supported = confirmation(batch, guard)
            permitted = selected[phase] & supported
            result = metrics(target, np.where(permitted[:, None], proposals[phase], records[phase]["initial"]))
            if (result["correct"] <= baseline[phase]["correct"] or
                    np.any(np.diag(result["confusion"]) < np.diag(baseline[phase]["confusion"])) or
                    result["macro_f1"] <= baseline[phase]["macro_f1"]):
                break
            after[phase] = result
            skipped = nearby[phase] & ~supported
            gain = (target == neutral).astype(float) - (target == positive).astype(float)
            utility += (.35 if phase == "train" else .65) * float(gain[skipped] @ weights[phase][skipped])
            removed += int(skipped.sum())
        if len(after) == len(batches) and (utility, -removed) > rank:
            rank = utility, -removed
            best = {"guard": guard, "before": baseline, "previous": previous, "after": after,
                    "nearby_source_balanced_gain": utility}
    return best


def infer(batch, incumbent, policy):
    proposal = consensus.infer(batch, incumbent, policy["proposal"])
    return np.where(confirmation(batch, policy["guard"])[:, None], proposal, incumbent)


def provenance(directory, splits):
    return {"code": sha_file(Path(__file__)), "consensus": sha_file(directory / consensus.MODEL), "splits": splits}


def fit(cache):
    directory = cache / timeline.DIRECTORY
    checkpoint = neutral_guard.load(cache, directory / "timeline-fusion.json")
    previous = consensus.load(cache, checkpoint["splits"])
    if previous["rule"] is None or previous["rule"]["target"] != consensus.POSITIVE:
        raise ValueError("A frozen positive consensus review is required")
    acoustic = audio_evidence.load(cache)
    development = context_evidence.load_development(cache)
    batches = {phase: cases[precision.CONDITION] for phase, cases in development.items()}
    for phase, batch in batches.items():
        batch.update(calibrated.acoustic_inputs(read_full_report(cache, phase)["rows"], acoustic[phase]))
        batch["quality"] = dynamic.quality_input(batch)
        batch["retained"] = development[phase]["synchronous_gap"]["context"][:, :2]
    incumbent = precision.development_incumbent(batches, checkpoint["audio_review"]["policies"][precision.CONDITION])
    records = {phase: consensus.observe(batch, incumbent[phase]) for phase, batch in batches.items()}
    offered = consensus.cross_fitted(records, batches, previous["strength"])
    mix = previous["rule"]["mix"]
    proposals = {phase: mix * scores + (1. - mix) * incumbent[phase] for phase, scores in offered.items()}
    policy = choose(batches, records, proposals, previous["rule"])
    save_json(directory / MODEL, {**provenance(directory, checkpoint["splits"]), "policy": policy})
    print(policy["guard"], {key: {phase: (v["correct"], v["macro_f1"], np.diag(v["confusion"]).tolist())
                                for phase, v in policy[key].items()}
                            for key in ("before", "previous", "after")}, flush=True)


def load(cache, splits):
    directory = cache / timeline.DIRECTORY
    archive = json.loads((directory / MODEL).read_text("utf8"))
    if any(archive.get(key) != value for key, value in provenance(directory, splits).items()):
        raise ValueError("Fit the dual evidence review on frozen development data first")
    return {**archive["policy"], "proposal": consensus.load(cache, splits)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    args = parser.parse_args()
    with threadpool_limits(limits=5, user_api="blas"):
        fit(args.cache)
