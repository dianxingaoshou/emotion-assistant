"""Calibrate missing-evidence review on source-held-out development predictions."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import chsims_teacher_audio_evidence as audio_evidence
from scripts import chsims_teacher_context_evidence as evidence
from scripts import chsims_teacher_timeline as timeline
from scripts import chsims_teacher_window_agreement as agreement
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_teacher_cross_source import metrics
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file
from scripts.chsims_teacher_supervision import read_full_report


MODEL = "source-calibration.json"
NO_VISUAL = "no_visual"
SYNC_GAP = agreement.CASE
NEGATIVE, NEUTRAL, POSITIVE = (LABELS.index(label) for label in ("negative", "neutral", "positive"))


def signed_margin(scores, label):
    competitors = [other for other in range(len(LABELS)) if other != label]
    return scores[..., label] - scores[..., competitors].max(axis=-1)


def acoustic_inputs(rows, acoustic):
    desktop = np.asarray([[row["baseline"]["audio"]["scores"].get(label, np.nan) for label in LABELS]
                          for row in rows])
    return {"acoustic": acoustic, "desktop_audio": desktop}


def visual_absence_evidence(batch, before):
    acoustic = signed_margin(batch["acoustic"], NEGATIVE)
    desktop = signed_margin(batch["desktop_audio"], NEGATIVE)
    return {
        "proposed_neutral": (batch["prior"].argmax(1) == NEGATIVE) & (before.argmax(1) == NEUTRAL),
        "text_support": signed_margin(batch["context"][:, 0], NEUTRAL),
        "voice_support": batch["voice"][:, NEUTRAL] - batch["voice"][:, NEGATIVE],
        "prior_support": signed_margin(batch["prior"], NEGATIVE),
        "acoustic_conflict": np.minimum(acoustic, desktop),
    }


def protect_negative(observed, rule):
    text, voice = observed["text_support"], observed["voice_support"]
    verified = ((text > 0.) & (text >= rule["text_margin"]) & (voice >= rule["voice_margin"]) &
                (np.minimum(text, voice) >= rule["prior_ratio"] * observed["prior_support"]))
    cutoff = rule["acoustic_margin"]
    conflict = np.zeros(verified.shape, dtype=bool) if cutoff is None else observed["acoustic_conflict"] >= cutoff
    return observed["proposed_neutral"] & (~verified | conflict)


def window_evidence(batch, before, hybrid_share):
    context = batch["context"]
    present = np.isfinite(context).all(axis=2)
    available = np.where(present[:, :, None], context, 0.)
    neutral_votes = signed_margin(available, NEUTRAL)
    second = np.sort(neutral_votes, axis=1)[:, -2]
    old = batch["prior"].argmax(axis=1)
    original = np.where(old[:, None] == NEGATIVE, signed_margin(available, NEGATIVE),
                        signed_margin(available, POSITIVE)).max(axis=1)
    mean = available.sum(axis=1) / np.maximum(present.sum(axis=1)[:, None], 1)
    scores = np.ma.masked_array(context, mask=np.broadcast_to(~present[:, :, None], context.shape))
    median = np.ma.median(scores, axis=1).filled(0.)
    median = median / np.maximum(median.sum(axis=1, keepdims=True), 1e-9)
    hybrid = hybrid_share * batch["mini"] + (1. - hybrid_share) * batch["voice"]
    return {
        "active": batch["active"] & (old == before.argmax(axis=1)),
        "old": old,
        "second": second,
        "original": np.maximum(original, 0.),
        "prior_gap": batch["prior"].max(axis=1) - batch["prior"][:, NEUTRAL],
        "mean": mean,
        "median": median,
        "hybrid": hybrid,
    }


def neutral_proposal(observed, rule):
    share = rule["window_share"]
    return share * observed[rule["pool"]] + (1. - share) * observed["hybrid"]


def recover_neutral(observed, rule):
    second = observed["second"]
    proposal = neutral_proposal(observed, rule)
    return (observed["active"] & (observed["old"] == rule["from"]) & (second > 0.) &
            (second >= rule["second_margin"]) & (second >= rule["support_ratio"] * observed["original"]) &
            (observed["prior_gap"] <= rule["prior_gap"]) & (signed_margin(proposal, NEUTRAL) > 0.))


def source_weights(sources):
    _, groups, counts = np.unique(sources, return_inverse=True, return_counts=True)
    return 1. / (len(counts) * counts[groups])


def evaluate_development(records, select, label_scores):
    """Compare class counts and source-balanced accuracy with the preceding policy."""
    reports = {}
    utility = {}
    changes = 0
    for split, (batch, before, observed, weights) in records.items():
        mask = select(observed)
        corrected = before.copy()
        corrected[mask] = label_scores(batch, observed)[mask]
        truth = batch["target"]
        gain = (corrected.argmax(1) == truth).astype(int) - (before.argmax(1) == truth)
        if np.any(np.bincount(truth, weights=gain, minlength=len(LABELS)) < 0):
            return None
        reports[split] = metrics(truth, corrected)
        utility[split] = float(gain @ weights)
        changes += int(mask.sum())
    return reports, .35 * utility["train"] + .65 * utility["valid"], changes


def choose_no_visual(batches, reference):
    records = {}
    baseline = {}
    for split, batch in batches.items():
        scores = agreement.infer(batch, reference)
        records[split] = (batch, scores, visual_absence_evidence(batch, scores), source_weights(batch["source"]))
        baseline[split] = metrics(batch["target"], scores)
    rank = (0., baseline["valid"]["macro_f1"], baseline["train"]["macro_f1"], 0)
    best = {"rule": None, "before": baseline, "after": baseline}
    for acoustic, text, voice, ratio in product(
            (None, .04, .10, .18, .28, .40), (0., .04, .08, .12, .20),
            (0., .06, .12, .20), (0., .35, .65, 1.)):
        rule = {"acoustic_margin": acoustic, "text_margin": text, "voice_margin": voice, "prior_ratio": ratio}
        candidate = evaluate_development(records, lambda observed: protect_negative(observed, rule),
                                         lambda batch, observed: batch["prior"])
        if candidate is None:
            continue
        after, utility, changes = candidate
        quality = (utility, after["valid"]["macro_f1"], after["train"]["macro_f1"], -changes)
        if quality > rank:
            rank = quality
            best = {"rule": rule, "before": baseline, "after": after}
    return best


def choose_synchronous_gap(batches, reference):
    records = {}
    baseline = {}
    for split, batch in batches.items():
        scores = agreement.infer(batch, reference)
        observed = window_evidence(batch, scores, reference["share"])
        records[split] = (batch, scores, observed, source_weights(batch["source"]))
        baseline[split] = metrics(batch["target"], scores)
    rank = (0., 0., 0., 0)
    best = {"rule": None, "before": baseline, "after": baseline}
    for pool, old, share, margin, ratio, gap in product(
            ("mean", "median"), (NEGATIVE, POSITIVE), (.30, .50, .70),
            (.04, .05, .08), (.5, 1., 1.5), (.08, .12, .20)):
        rule = {"from": old, "pool": pool, "window_share": share, "second_margin": margin,
                "support_ratio": ratio, "prior_gap": gap}
        candidate = evaluate_development(records, lambda observed: recover_neutral(observed, rule),
                                         lambda batch, observed: neutral_proposal(observed, rule))
        if candidate is None:
            continue
        after, utility, changes = candidate
        delta = tuple(after[split]["macro_f1"] - baseline[split]["macro_f1"] for split in ("train", "valid"))
        if min(delta) <= 0.:
            continue
        quality = (utility, delta[1], delta[0], -changes)
        if quality > rank:
            rank = quality
            best = {"rule": rule, "before": baseline, "after": after}
    return best


def infer(batch, policy):
    reference = policy["reference"]
    before = agreement.infer(batch, reference)
    rule = policy["rule"]
    if rule is None:
        return before
    result = before.copy()
    if policy["case"] == NO_VISUAL:
        blocked = protect_negative(visual_absence_evidence(batch, before), rule)
        result[blocked] = batch["prior"][blocked]
    else:
        observed = window_evidence(batch, before, reference["share"])
        selected = recover_neutral(observed, rule)
        result[selected] = neutral_proposal(observed, rule)[selected]
    return result


def checkpoint_provenance(directory, splits):
    return {"implementation": sha_file(Path(__file__)),
            "window_agreement": sha_file(directory / agreement.MODEL),
            "development": sha_file(directory / evidence.DEVELOPMENT),
            "acoustic_development": sha_file(directory / audio_evidence.DEVELOPMENT),
            "splits": splits}


def fit(cache):
    directory = cache / timeline.DIRECTORY
    model = agreement.load(cache, directory / "timeline-fusion.json")
    batches = evidence.load_development(cache)
    acoustic = audio_evidence.load(cache)
    for split, cases in batches.items():
        cases[NO_VISUAL].update(acoustic_inputs(read_full_report(cache, split)["rows"], acoustic[split]))
    previous = model["audio_review"]["policies"]
    policies = {}
    for case, choose in ((NO_VISUAL, choose_no_visual), (SYNC_GAP, choose_synchronous_gap)):
        policies[case] = choose({split: data[case] for split, data in batches.items()}, previous[case])
        policy = policies[case]
        values = {split: (score["correct"], round(score["macro_f1"], 5),
                          np.diag(score["confusion"], k=0).tolist())
                  for split, score in policy["after"].items()}
        print(case, policy["rule"], values, flush=True)
    save_json(directory / MODEL, {**checkpoint_provenance(directory, model["splits"]), "policies": policies})


def load(cache, reference):
    directory = cache / timeline.DIRECTORY
    model = agreement.load(cache, reference)
    checkpoint = json.loads((directory / MODEL).read_text("utf8"))
    if any(checkpoint.get(key) != value for key, value in checkpoint_provenance(directory, model["splits"]).items()):
        raise ValueError("Fit source calibration on the frozen development inputs first")
    audio = model["audio_review"]
    policies = {case: {**policy, "case": case, "reference": audio["policies"][case]}
                for case, policy in checkpoint["policies"].items()}
    return {**model, "kind": "source_calibration", "audio_review": {**audio, "policies": policies}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    fit(parser.parse_args().cache)
