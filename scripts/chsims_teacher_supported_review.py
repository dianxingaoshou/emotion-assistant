"""Review complete-input decisions when modality heads support the new class."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.special import softmax
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import chsims_teacher_complete_directional_review as directional
from scripts import chsims_teacher_complete_review as review
from scripts import chsims_teacher_robustness as robust
from scripts import chsims_teacher_timeline as timeline
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_teacher_cross_source import group_ids, metrics
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file
from scripts.chsims_teacher_supervision import numbers, read_full_report


MODEL = "supported-review.json"
NEUTRAL = LABELS.index("neutral")


def describe(design, incumbent, candidate, eligible):
    expert_logits = np.stack((design[:, 2::2], np.zeros_like(design[:, 2::2]), design[:, 3::2]), axis=2)
    experts = softmax(expert_logits, axis=2)
    old, new = incumbent.argmax(1), candidate.argmax(1)
    rows = np.arange(len(incumbent))
    support_new = np.take_along_axis(experts, new[:, None, None], axis=2)[:, :, 0]
    support_old = np.take_along_axis(experts, old[:, None, None], axis=2)[:, :, 0]
    sorted_scores = np.sort(experts, axis=2)
    expert_margin = sorted_scores[:, :, -1] - sorted_scores[:, :, -2]
    expert_top = experts.argmax(2)
    top = (expert_top == new[:, None]) & (expert_margin > 0.)
    difference = support_new - support_old
    second_support = np.sort(difference, axis=1)[:, 1]
    opposition = np.where(expert_top == old[:, None], -difference, 0.).max(1)
    return {"old": old, "new": new, "eligible": eligible, "votes": top.sum(1),
            "pair_support": (support_new > support_old).sum(1),
            "strongest": np.where(top, expert_margin, 0.).max(1),
            "second_support": second_support, "opposition": opposition,
            "original_margin": incumbent[rows, old] - incumbent[rows, new],
            "proposed_margin": candidate[rows, new] - candidate[rows, old]}


def accept(scope, rule):
    return (scope["eligible"] & (scope["old"] == rule["from"]) & (scope["new"] == rule["to"]) &
            (scope["original_margin"] <= rule["original_margin"]) &
            (scope["proposed_margin"] >= rule["proposed_margin"]) &
            (scope["votes"] >= rule["votes"]) & (scope["strongest"] >= rule["min_expert_margin"]) &
            (scope["second_support"] >= rule["opposition_ratio"] * scope["opposition"]) &
            (scope["pair_support"] >= rule["pair_support"]))


def selection(scope, rules):
    chosen = np.zeros(len(scope["eligible"]), dtype=bool)
    for rule in rules:
        chosen |= accept(scope, rule)
    return chosen


def options_for(direction, records, scopes, weights):
    old, new = direction
    options = {(0,) * (2 * len(LABELS)): (0., None)}
    for gap, minimum, count, strongest, pair, opposing in product(
            (.04, .07, .10, .16, .22, .30), (.01, .03, .06, .10), (1, 2), (.0, .035, .08), (2, 3), (.5, 1.)):
        rule = {"from": old, "to": new, "original_margin": gap, "proposed_margin": minimum,
                "votes": count, "min_expert_margin": strongest, "pair_support": pair,
                "opposition_ratio": opposing}
        utility, totals = 0., []
        for split in review.STAGES:
            chosen = accept(scopes[split], rule)
            truth = records[split]["truth"][chosen]
            gain = (truth == new).astype(int) - (truth == old)
            totals.extend(np.bincount(truth, weights=gain, minlength=len(LABELS)).astype(int))
            utility += (.35 if split == "train" else .65) * float(gain @ weights[split][chosen])
            utility -= .000003 * int(chosen.sum())
        key = tuple(totals)
        if key not in options or utility > options[key][0]:
            options[key] = utility, rule
    return options


def choose_rules(records, offered, weights):
    scopes = {split: describe(item["design"], item["incumbent"], offered[split], item["eligible"])
              for split, item in records.items()}
    candidates, costs, columns = [], [], []
    directions = directional.DIRECTIONS
    for direction, indicator in zip(directions, np.eye(len(directions)), strict=True):
        for outcome, (utility, rule) in options_for(direction, records, scopes, weights).items():
            candidates.append(rule)
            costs.append(-utility)
            columns.append(np.r_[indicator, outcome])
    minimum = np.zeros(2 * len(LABELS))
    minimum[[NEUTRAL, len(LABELS) + NEUTRAL]] = 1
    lower = np.r_[np.ones(len(directions)), minimum]
    upper = np.r_[np.ones(len(directions)), np.full(len(minimum), np.inf)]
    found = milp(costs, integrality=np.ones(len(costs)), bounds=Bounds(0., 1.),
                 constraints=LinearConstraint(np.asarray(columns).T, lower, upper))
    if not found.success:
        return [], None
    rules = [r for r, use in zip(candidates, found.x, strict=True) if r is not None and use > .5]
    after = {split: metrics(item["truth"], np.where(selection(scopes[split], rules)[:, None],
                                                    offered[split], item["incumbent"]))
             for split, item in records.items()}
    return rules, after


def choose(records):
    baseline = {split: metrics(item["truth"], item["incumbent"]) for split, item in records.items()}
    weights = {split: directional.source_weight(item["sources"]) for split, item in records.items()}
    best = {"rules": [], "before": baseline, "after": baseline}
    rank = (0., 0.)
    for penalty in (.02, .08, .30):
        for offered, share, offset in directional.proposal_options(records, penalty):
            rules, after = choose_rules(records, offered, weights)
            if not rules:
                continue
            quality = tuple(.60 * (after[p]["accuracy"] - baseline[p]["accuracy"]) +
                            .40 * (after[p]["macro_f1"] - baseline[p]["macro_f1"]) for p in ("valid", "train"))
            if quality > rank and min(quality) >= 0.:
                rank = quality
                best = {"rules": rules, "penalty": penalty, "before": baseline, "after": after,
                        "residual": {"share": share, "neutral_offset": offset}}
    return best


def infer(blocks, incumbent, policy):
    if not policy["rules"]:
        return incumbent
    design, available = review.observe(blocks, incumbent, policy["experts"])
    proposal = timeline.apply_residual(design, incumbent, policy["residual"], available)
    scope = describe(design, incumbent, proposal, available)
    return np.where(selection(scope, policy["rules"])[:, None], proposal, incumbent)


def provenance(cache):
    return {**directional.provenance(cache), "directional": sha_file(Path(directional.__file__)),
            "support_implementation": sha_file(Path(__file__))}


def fit(cache):
    signature = provenance(cache)
    frozen = json.loads((cache / "missing-evidence/fusion-heads.json").read_text("utf8"))
    refinement = json.loads((cache / "teacher-refinement/heads.json").read_text("utf8"))
    if (frozen["splits"] != signature["splits"] or frozen["implementation"] != sha_file(Path(robust.__file__))
            or refinement["corrections"][str(review.FULL)]["proposal"] != "previous"):
        raise ValueError("Reproduce the frozen complete-input teacher before fitting a review")
    reports = {split: read_full_report(cache, split) for split in review.STAGES}
    variants = {}
    for split, report in reports.items():
        variants[split], _, extraction = robust.inputs(cache, report)
        if extraction != frozen["gap_extractors"]:
            raise ValueError("Expected frozen development extraction")
    labels = {split: {name: numbers(report["rows"], name) for name in (*review.MODALITIES, "full")}
              for split, report in reports.items()}
    sources = {split: group_ids(report["rows"]) for split, report in reports.items()}
    assignment = robust.case_assignment(reports["train"]["rows"])
    records = directional.build_records(variants, labels, sources, assignment, frozen)
    selected = choose(records)
    if selected["rules"]:
        training = np.arange(len(sources["train"]))
        selected["residual"].update(review.fit_residual(records["train"], training, selected["penalty"]))
        selected["experts"] = review.fit_experts(variants["train"], labels["train"], sources["train"], training,
                                                assignment, frozen)
    save_json(cache / timeline.DIRECTORY / MODEL, {**signature, "review": selected})
    for split in review.STAGES:
        print(split, {stage: (selected[stage][split]["correct"], round(selected[stage][split]["macro_f1"], 6),
                              np.diag(selected[stage][split]["confusion"]).tolist())
                      for stage in ("before", "after")}, flush=True)
    print(selected["rules"], flush=True)


def load(cache):
    payload = json.loads((cache / timeline.DIRECTORY / MODEL).read_text("utf8"))
    if any(payload.get(key) != value for key, value in provenance(cache).items()):
        raise ValueError("Fit supported review on the frozen development evidence first")
    return payload["review"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    with threadpool_limits(limits=5, user_api="blas"):
        fit(parser.parse_args().cache)
