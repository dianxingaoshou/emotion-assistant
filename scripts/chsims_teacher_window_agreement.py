"""Fit MiniCPM-o review gates with agreement between retained modalities."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import chsims_teacher_context_evidence as evidence
from scripts import chsims_teacher_context_review as context_review
from scripts import chsims_teacher_timeline as timeline
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_teacher_cross_source import metrics
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file


MODEL = "window-agreement.json"
CASE = "synchronous_gap"
NEUTRAL = context_review.NEUTRAL
DIRECTIONS = context_review.DIRECTIONS


def propose(batch, share):
    proposal = context_review.propose(batch, share)
    # Each head can vote once. Missing heads and tied top scores provide no vote.
    context = batch["context"]
    present = np.isfinite(context).all(axis=2)
    scores = np.where(present[:, :, None], context, 0.)
    top = scores.argmax(axis=2)
    ordered = np.sort(scores, axis=2)
    margin = ordered[:, :, -1] - ordered[:, :, -2]
    support = np.where(present & (top == proposal["target"][:, None]), margin, 0.)
    proposal["second_margin"] = np.sort(support, axis=1)[:, -2]
    return proposal


def accept(proposal, rule):
    second = proposal["second_margin"]
    selected = (proposal["eligible"] & (proposal["old"] == rule["from"]) &
                (proposal["target"] == rule["to"]) & (proposal["gain"] >= rule["gain"]) &
                (proposal["voice_delta"] >= rule["audio_support"]) & (second > 0.) &
                (second >= rule["margin"]) & (second >= rule["ratio"] * proposal["old_support"]))
    if rule["to"] == NEUTRAL:
        selected &= proposal["voice_quality"] >= rule["quality"]
    return selected


def infer(batch, policy):
    if policy["kind"] == "context_review":
        return context_review.infer(batch, policy)
    result = batch["prior"].copy()
    if policy["rules"]:
        proposal = propose(batch, policy["share"])
        for rule in policy["rules"]:
            selected = accept(proposal, rule)
            result[selected] = proposal["proposed"][selected]
    return result


def candidates(direction):
    original, target = direction
    if target == NEUTRAL:
        quality_grid = (.25, .45, .65)
        gain_grid = (.00, .05, .12, .20)
        audio_grid = (0.,)
    else:
        quality_grid = (0.,)
        gain_grid = (.08, .15, .25, .40)
        audio_grid = (-.20, -.08, 0., .08)
    for margin, ratio, quality, gain, audio_support in product(
            (.00, .04, .08), (.5, 1., 1.5), quality_grid, gain_grid, audio_grid):
        yield {"from": original, "to": target, "margin": margin, "ratio": ratio,
               "quality": quality, "gain": gain, "audio_support": audio_support}


def development_record(batch, share):
    proposal = propose(batch, share)
    target = batch["target"]
    _, groups, counts = np.unique(batch["source"], return_inverse=True, return_counts=True)
    source_weight = 1. / (len(counts) * counts[groups])
    gain = (target == proposal["target"]).astype(int) - (target == proposal["old"])
    return proposal, target, source_weight, gain


def development_rules(train, valid, share):
    records = [development_record(batch, share) for batch in (train, valid)]
    width = 2 * len(LABELS)
    columns, costs, rules = [], [], []
    for direction, indicator in zip(DIRECTIONS, np.eye(len(DIRECTIONS)), strict=True):
        options = {(0,) * width: (0., None)}
        for rule in candidates(direction):
            totals, utility, changes = [], [], 0
            for proposal, target, weight, gain in records:
                selected = accept(proposal, rule)
                totals.extend(np.bincount(target[selected], weights=gain[selected],
                                         minlength=len(LABELS)).astype(int))
                utility.append(float(gain[selected] @ weight[selected]))
                changes += int(selected.sum())
            key = tuple(totals)
            score = .35 * utility[0] + .65 * utility[1] - .000005 * changes
            if key not in options or score > options[key][0]:
                options[key] = score, rule
        for totals, (utility, rule) in options.items():
            columns.append(np.r_[indicator, totals])
            costs.append(-utility)
            rules.append(rule)
    # Keep each class's correct count on both development splits.
    lower = np.r_[np.ones(len(DIRECTIONS)), np.zeros(width)]
    upper = np.r_[np.ones(len(DIRECTIONS)), np.full(width, np.inf)]
    result = milp(costs, integrality=np.ones(len(costs)), bounds=Bounds(0., 1.),
                  constraints=LinearConstraint(np.asarray(columns).T, lower, upper))
    if not result.success:
        return []
    return [rule for rule, selected in zip(rules, result.x, strict=True) if rule is not None and selected > .5]


def choose(train, valid):
    splits = {"train": train, "valid": valid}
    before = {phase: metrics(batch["target"], batch["prior"]) for phase, batch in splits.items()}
    best = {"kind": "window_agreement", "rules": [], "before": before, "after": before}
    rank = (0., 0.)
    for share in (.60, .80, 1.):
        rules = development_rules(train, valid, share)
        if not rules:
            continue
        policy = {"kind": "window_agreement", "share": share, "rules": rules}
        after = {phase: metrics(batch["target"], infer(batch, policy)) for phase, batch in splits.items()}
        delta = {phase: after[phase]["macro_f1"] - before[phase]["macro_f1"] for phase in splits}
        quality = delta["valid"], delta["train"]
        if delta["train"] >= 0. and quality > rank:
            rank = quality
            best = {**policy, "before": before, "after": after}
    return best


def fit(cache):
    directory = cache / timeline.DIRECTORY
    model = context_review.load(cache, directory / "timeline-fusion.json")
    cases = evidence.load_development(cache)
    policy = choose(cases["train"][CASE], cases["valid"][CASE])
    save_json(directory / MODEL, {"implementation": sha_file(Path(__file__)),
                                 "context_review": sha_file(directory / context_review.MODEL),
                                 "development": sha_file(directory / evidence.DEVELOPMENT),
                                 "splits": model["splits"], "policy": policy})
    print(policy["rules"], {key: (row["correct"], round(row["macro_f1"], 5))
                            for key, row in policy["after"].items()}, flush=True)


def load(cache, reference):
    directory = cache / timeline.DIRECTORY
    model = context_review.load(cache, reference)
    checkpoint = json.loads((directory / MODEL).read_text("utf8"))
    if (checkpoint["implementation"] != sha_file(Path(__file__)) or
            checkpoint["context_review"] != sha_file(directory / context_review.MODEL) or
            checkpoint["development"] != sha_file(directory / evidence.DEVELOPMENT) or
            checkpoint["splits"] != model["splits"]):
        raise ValueError("Fit window agreement on the frozen development inputs first")
    audio = model["audio_review"]
    policies = {**audio["policies"], CASE: checkpoint["policy"],
                "no_visual": {**audio["policies"]["no_visual"], "kind": "context_review"}}
    return {**model, "kind": "window_agreement", "audio_review": {**audio, "policies": policies}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    fit(parser.parse_args().cache)
