"""Learn when to trust text or acoustics under source-held-out disagreement."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import chsims_teacher_audio_evidence as audio_evidence
from scripts import chsims_teacher_context_evidence as evidence
from scripts import chsims_teacher_source_calibration as calibrated
from scripts import chsims_teacher_timeline as timeline
from scripts import chsims_teacher_window_agreement as agreement
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_teacher_cross_source import metrics
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file
from scripts.chsims_teacher_supervision import read_full_report


MODEL = "dynamic-conflict.json"
NEGATIVE, NEUTRAL = calibrated.NEGATIVE, calibrated.NEUTRAL


def describe(batch, before):
    text, audio = batch["context"][:, 0], batch["acoustic"]
    t = np.isfinite(text).all(axis=1)
    a = np.isfinite(audio).all(axis=1)
    text = np.where(t[:, None], text, 0.)
    audio = np.where(a[:, None], audio, 0.)
    text_pick = np.where(text[:, NEGATIVE] > text[:, NEUTRAL], NEGATIVE, NEUTRAL)
    audio_pick = np.where(audio[:, NEGATIVE] > audio[:, NEUTRAL], NEGATIVE, NEUTRAL)
    old = before.argmax(1)
    opposing = (t & a & batch["active"] & (text_pick != audio_pick) &
                (old <= NEUTRAL))
    t_gap, a_gap = np.abs(text[:, NEGATIVE] - text[:, NEUTRAL]), np.abs(audio[:, NEGATIVE] - audio[:, NEUTRAL])
    opposing &= (t_gap > 0.) & (a_gap > 0.)
    rows = np.arange(len(text))
    # The MiniCPM-o audio head can add support, but receives no second modality vote.
    mini = batch["voice"]
    mini_agreement = mini[rows, audio_pick] - mini[rows, text_pick]
    prior = batch["prior"]
    prior_agreement = prior[rows, audio_pick] - prior[rows, text_pick]
    quality = batch["quality"]
    design = np.column_stack((t_gap, a_gap, quality[:, 0], np.log1p(quality[:, 1]), quality[:, 2],
                              mini_agreement, prior_agreement, audio_pick == NEGATIVE))
    opposing &= np.isfinite(design).all(axis=1)
    return {"text": text, "audio": audio, "audio_pick": audio_pick, "design": design,
            "old": old, "gap": np.abs(before[:, NEGATIVE] - before[:, NEUTRAL]), "opposing": opposing}


def fit_head(record, truth, sources, indices, strength):
    used = indices[record["opposing"][indices] & (truth[indices] <= NEUTRAL)]
    x = record["design"][used]
    wins = record["audio_pick"][used] == truth[used]
    weights = calibrated.source_weights(sources[used]) * len(used)
    center, scale = x.mean(0), np.maximum(x.std(0), .01)
    model = LogisticRegression(C=strength, max_iter=500)
    model.fit((x - center) / scale, wins, sample_weight=weights)
    return {"center": center.tolist(), "scale": scale.tolist(), "weights": model.coef_[0].tolist(),
            "bias": float(model.intercept_[0])}


def acoustic_share(record, head):
    x = np.where(np.isfinite(record["design"]), record["design"], 0.)
    return expit(((x - head["center"]) / head["scale"]) @ head["weights"] + head["bias"])


def fuse(record, share, before, anchor):
    proposals = share[:, None] * record["audio"] + (1. - share[:, None]) * record["text"]
    return anchor * before + (1. - anchor) * proposals


def accept(record, proposal, rule):
    pick = proposal.argmax(axis=1)
    gap = proposal[:, NEGATIVE] - proposal[:, NEUTRAL]
    return (record["opposing"] & (record["old"] == NEUTRAL) & (pick == NEGATIVE) & (gap > 0.) &
            (gap >= rule["min_margin"]) & (record["gap"] <= rule["max_previous_gap"]))


def held_shares(records, batches, strength):
    sources = {split: batch["source"] for split, batch in batches.items()}
    combined = np.concatenate(tuple(sources.values()))
    shares = {split: np.full(len(groups), np.nan) for split, groups in sources.items()}
    for _, held in GroupKFold(4).split(combined, groups=combined):
        withheld = np.unique(combined[held])
        training = np.flatnonzero(~np.isin(sources["train"], withheld))
        model = fit_head(records["train"], batches["train"]["target"], sources["train"], training, strength)
        for split, groups in sources.items():
            out = np.isin(groups, withheld)
            shares[split][out] = acoustic_share(records[split], model)[out]
    if any(not np.isfinite(values).all() for values in shares.values()):
        raise ValueError("Every development clip needs a source-held-out share")
    return shares


def choose(batches, previous):
    before = {split: calibrated.infer(batch, previous) for split, batch in batches.items()}
    observed = {split: describe(batch, before[split]) for split, batch in batches.items()}
    baseline = {split: metrics(batch["target"], before[split]) for split, batch in batches.items()}
    floor = {"train": np.diag(metrics(batches["train"]["target"], batches["train"]["prior"])["confusion"]),
             "valid": np.diag(baseline["valid"]["confusion"])}
    weights = {split: calibrated.source_weights(batch["source"]) for split, batch in batches.items()}
    rank = (0., 0., 0.)
    selected_strength = None
    best = {"rule": None, "before": baseline, "after": baseline}
    for strength in (.1, 1., 10.):
        shares = held_shares(observed, batches, strength)
        for anchor, offset in product((.0, .3, .6), (-.3, -.15, 0., .15, .3)):
            proposals = {split: fuse(record, np.clip(shares[split] + offset, 0., 1.), before[split], anchor)
                         for split, record in observed.items()}
            for minimum, maximum in product((.0, .025, .05, .10, .15, .20), (.08, .12, .18, .24, .40, 1.)):
                rule = {"min_margin": minimum, "max_previous_gap": maximum, "anchor": anchor, "offset": offset}
                candidate = evaluate_rule(rule, batches, before, floor, observed, proposals, weights)
                if candidate is None:
                    continue
                after, utility, changes = candidate
                improvement = {split: after[split]["macro_f1"] - baseline[split]["macro_f1"] for split in after}
                quality = (utility, improvement["valid"], -changes)
                if min(improvement.values()) >= 0. and quality > rank:
                    rank = quality
                    selected_strength = strength
                    best = {"rule": rule, "strength": strength, "before": baseline, "after": after}
    if selected_strength is not None:
        batch = batches["train"]
        best["head"] = fit_head(observed["train"], batch["target"], batch["source"],
                                np.arange(len(before["train"])), selected_strength)
    return best


def evaluate_rule(rule, batches, before, floor, observed, proposals, weights):
    after = {}
    utility = {}
    changes = 0
    for split, batch in batches.items():
        chosen = accept(observed[split], proposals[split], rule)
        changed = before[split].copy()
        changed[chosen] = proposals[split][chosen]
        truth = batch["target"]
        report = metrics(truth, changed)
        if np.any(np.diag(report["confusion"]) < floor[split]):
            return None
        gain = (changed[chosen].argmax(1) == truth[chosen]).astype(int) - (truth[chosen] == NEUTRAL)
        after[split] = report
        utility[split] = float(gain @ weights[split][chosen])
        changes += int(chosen.sum())
    return after, .35 * utility["train"] + .65 * utility["valid"], changes


def infer(batch, policy):
    reference = policy["reference"]
    if policy["case"] != calibrated.NO_VISUAL:
        return agreement.infer(batch, reference)
    before = calibrated.infer(batch, reference)
    if policy["rule"] is None:
        return before
    observed = describe(batch, before)
    rule = policy["rule"]
    share = np.clip(acoustic_share(observed, policy["head"]) + rule["offset"], 0., 1.)
    proposal = fuse(observed, share, before, rule["anchor"])
    selected = accept(observed, proposal, rule)
    result = before.copy()
    result[selected] = proposal[selected]
    return result


def provenance(directory, splits):
    return {"implementation": sha_file(Path(__file__)),
            "previous": sha_file(directory / calibrated.MODEL), "splits": splits}


def quality_input(batch):
    # These three columns are text quality, log voice duration and the SNR proxy.
    return np.column_stack((batch["design"][:, 10], np.expm1(batch["design"][:, 11]), batch["design"][:, 12]))


def fit(cache):
    directory = cache / timeline.DIRECTORY
    baseline = calibrated.load(cache, directory / "timeline-fusion.json")
    batch = evidence.load_development(cache)
    acoustic = audio_evidence.load(cache)
    samples = {split: groups[calibrated.NO_VISUAL] for split, groups in batch.items()}
    for split, sample in samples.items():
        sample.update(calibrated.acoustic_inputs(read_full_report(cache, split)["rows"], acoustic[split]))
        sample["quality"] = quality_input(sample)
    policy = choose(samples, baseline["audio_review"]["policies"][calibrated.NO_VISUAL])
    print(policy.get("strength"), policy["rule"],
          {k: (r["correct"], r["macro_f1"], np.diag(r["confusion"]).tolist())
           for k, r in policy["after"].items()}, flush=True)
    save_json(directory / MODEL, {**provenance(directory, baseline["splits"]), "no_visual": policy})


def load(cache, reference):
    directory = cache / timeline.DIRECTORY
    baseline = calibrated.load(cache, reference)
    checkpoint = json.loads((directory / MODEL).read_text("utf8"))
    if any(checkpoint.get(key) != value for key, value in provenance(directory, baseline["splits"]).items()):
        raise ValueError("Fit dynamic conflict on the frozen development inputs first")
    audio = baseline["audio_review"]
    policies = {
        calibrated.NO_VISUAL: {**checkpoint["no_visual"], "case": calibrated.NO_VISUAL,
                               "reference": audio["policies"][calibrated.NO_VISUAL]},
        calibrated.SYNC_GAP: {"case": calibrated.SYNC_GAP, "reference": audio["policies"][calibrated.SYNC_GAP]["reference"]},
    }
    return {**baseline, "kind": "dynamic_conflict", "audio_review": {**audio, "policies": policies}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    fit(parser.parse_args().cache)
