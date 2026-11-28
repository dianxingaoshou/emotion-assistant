"""Review MiniCPM-o neutral suggestions with separately observed context."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import chsims_teacher_context_evidence as evidence
from scripts import chsims_teacher_minicpm_review as mini
from scripts import chsims_teacher_timeline as timeline
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_teacher_cross_source import metrics
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file


MODEL = "context-review.json"
NEUTRAL = LABELS.index("neutral")
DIRECTIONS = tuple((a, b) for a in range(len(LABELS)) for b in range(len(LABELS))
                   if a != b and NEUTRAL in (a, b))


def propose(batch, share):
    context = batch["context"]
    previous = batch["prior"]
    present = np.isfinite(context[:, :, 0])
    available = present.any(1)
    pooled = np.nan_to_num(context).sum(1) / np.maximum(present.sum(1)[:, None], 1)
    hybrid = share * batch["mini"] + (1. - share) * batch["voice"]
    old = previous.argmax(1)
    new = np.where(old[:, None] == NEUTRAL, pooled, hybrid)
    target = new.argmax(1)
    row = np.arange(len(old))
    # A leading contextual pole must match the established decision to veto a change.
    margins = mini.decision_quality(context.reshape(-1, len(LABELS))).reshape(present.shape)
    old_support = np.max(np.where(present & (context.argmax(2) == old[:, None]), margins, 0.), axis=1)
    delta = context[:, :, NEUTRAL] - context[row, :, old]
    neutral_context = np.max(np.where(present, delta, -np.inf), axis=1)
    audio = batch["voice"]
    eligible = batch["active"] & available & ((old == NEUTRAL) | (batch["mini"].argmax(1) == NEUTRAL))
    return {"old": old, "target": target, "proposed": new, "eligible": eligible, "old_support": old_support,
            "neutral_context": neutral_context, "voice_quality": batch["voice_quality"],
            "voice_delta": audio[row, target] - audio[row, old], "gain": new[row, target] - new[row, old]}


def accept(proposal, rule):
    selected = (proposal["eligible"] & (proposal["old"] == rule["from"]) &
                (proposal["target"] == rule["to"]) & (proposal["gain"] >= rule["gain"]) &
                (proposal["voice_delta"] >= rule["audio_support"]))
    if rule["to"] == NEUTRAL:
        selected &= ((proposal["voice_quality"] >= rule["quality"]) &
                     (proposal["old_support"] < rule["veto"]) &
                     (proposal["neutral_context"] >= rule["context_floor"]))
    return selected


def infer(batch, policy):
    if not policy["rules"]:
        return batch["prior"]
    proposal = propose(batch, policy["share"])
    result = batch["prior"].copy()
    for rule in policy["rules"]:
        selected = accept(proposal, rule)
        result[selected] = proposal["proposed"][selected]
    return result


def candidates(direction):
    first, second = direction
    if second == NEUTRAL:
        for gain, quality, veto, floor in product((.00, .05, .12), (.25, .45, .65), (.12, .25, .40), (-.35, -.15, 0.)):
            yield {"from": first, "to": second, "gain": gain, "quality": quality,
                   "veto": veto, "context_floor": floor, "audio_support": 0.}
    else:
        for gain, support in product((.08, .15, .25, .40), (-.20, -.08, 0., .08)):
            yield {"from": first, "to": second, "gain": gain, "audio_support": support}


def rules_for(train, valid, share):
    records = []
    for batch in (train, valid):
        proposal = propose(batch, share)
        target = batch["target"]
        _, group, counts = np.unique(batch["source"], return_inverse=True, return_counts=True)
        source_weight = 1. / (len(counts) * counts[group])
        delta = (target == proposal["target"]).astype(int) - (target == proposal["old"])
        records.append((proposal, target, delta, source_weight))
    columns, losses, rules = [], [], []
    for i, direction in enumerate(DIRECTIONS):
        options = {(0,) * (2 * len(LABELS)): (0., None)}
        for rule in candidates(direction):
            totals, weighted, changed = [], [], 0
            for proposal, target, delta, source_weight in records:
                selected = accept(proposal, rule)
                totals.extend(np.bincount(target[selected], weights=delta[selected], minlength=len(LABELS)).astype(int))
                weighted.append(float(delta[selected] @ source_weight[selected]))
                changed += int(selected.sum())
            key = tuple(totals)
            score = .35 * weighted[0] + .65 * weighted[1] - .000005 * changed
            if key not in options or score > options[key][0]:
                options[key] = score, rule
        for delta, (score, rule) in options.items():
            columns.append(np.r_[np.eye(len(DIRECTIONS))[i], delta])
            losses.append(-score)
            rules.append(rule)
    lower = np.r_[np.ones(len(DIRECTIONS)), np.zeros(2 * len(LABELS))]
    lower[len(DIRECTIONS) + len(LABELS) + NEUTRAL] = 1
    upper = np.r_[np.ones(len(DIRECTIONS)), np.full(2 * len(LABELS), np.inf)]
    solution = milp(losses, integrality=np.ones(len(losses)), bounds=Bounds(0., 1.),
                    constraints=LinearConstraint(np.asarray(columns).T, lower, upper))
    if not solution.success:
        return []
    return [rule for rule, chosen in zip(rules, solution.x, strict=True) if rule is not None and chosen > .5]


def choose(train, valid):
    before = {name: metrics(batch["target"], batch["prior"]) for name, batch in (("train", train), ("valid", valid))}
    best = {"rules": [], "before": before, "after": before}
    rank = (0., 0.)
    for share in (.60, .80, 1.):
        rules = rules_for(train, valid, share)
        if not rules:
            continue
        proposed = {"rules": rules, "share": share}
        after = {name: metrics(batch["target"], infer(batch, proposed)) for name, batch in (("train", train), ("valid", valid))}
        delta = {name: after[name]["macro_f1"] - before[name]["macro_f1"] for name in before}
        quality = (delta["valid"], delta["train"])
        if delta["train"] >= 0. and quality > rank:
            rank = quality
            best = {**proposed, "before": before, "after": after}
    return best


def fit(cache):
    directory = cache / timeline.DIRECTORY
    calibration = mini.load(cache, directory / "timeline-fusion.json")["audio_review"]
    data = evidence.load_development(cache)
    policies = {case: choose(data["train"][case], data["valid"][case]) for case in timeline.CASES}
    for case, policy in policies.items():
        print(case, policy["rules"], {k: (r["correct"], round(r["macro_f1"], 4))
                                     for k, r in policy["after"].items()}, flush=True)
    save_json(directory / MODEL, {"implementation": sha_file(Path(__file__)),
                                 "evidence": sha_file(Path(evidence.__file__)),
                                 "mini": sha_file(directory / mini.MODEL),
                                 "splits": calibration["splits"], "policies": policies})


def load(cache, reference):
    directory = cache / timeline.DIRECTORY
    model = mini.load(cache, reference)
    checkpoint = json.loads((directory / MODEL).read_text("utf8"))
    if (checkpoint["implementation"] != sha_file(Path(__file__)) or
            checkpoint["evidence"] != sha_file(Path(evidence.__file__)) or
            checkpoint["mini"] != sha_file(directory / mini.MODEL) or
            checkpoint["splits"] != model["splits"]):
        raise ValueError("Fit the context review on the frozen development inputs first")
    return {**model, "kind": "context_review", "audio_review": {**model["audio_review"],
                                                                "policies": checkpoint["policies"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    args = parser.parse_args()
    fit(args.cache)


if __name__ == "__main__":
    main()
