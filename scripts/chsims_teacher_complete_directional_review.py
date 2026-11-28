"""Choose complete-input direction gates without weakening class recall on development."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import chsims_teacher_complete_review as review
from scripts import chsims_teacher_robustness as robust
from scripts import chsims_teacher_timeline as timeline
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_teacher_cross_source import group_ids, metrics
from scripts.chsims_teacher_data import CACHE, sha_file
from scripts.chsims_teacher_supervision import numbers, read_full_report


MODEL = "complete-directional-review.json"
DIRECTIONS = tuple((old, new) for old in range(3) for new in range(3) if old != new)


def describe(design, incumbent, proposal, eligible):
    experts = np.stack((design[:, 2::2], np.zeros_like(design[:, 2::2]), design[:, 3::2]), axis=-1)
    old = incumbent.argmax(axis=1)
    new = proposal.argmax(axis=1)
    rows = np.arange(len(old))
    ordered = np.sort(incumbent, axis=1)
    new_support = np.take_along_axis(experts, new[:, None, None], axis=2)[:, :, 0]
    old_support = np.take_along_axis(experts, old[:, None, None], axis=2)[:, :, 0]
    return {"eligible": eligible, "old": old, "new": new, "old_gap": ordered[:, -1] - ordered[:, -2],
            "new_gap": proposal[rows, new] - proposal[rows, old],
            "votes": (experts.argmax(axis=2) == new[:, None]).sum(axis=1),
            "pair_support": (new_support > old_support).sum(axis=1)}


def accepted(item, rule):
    return (item["eligible"] & (item["old"] == rule["from"]) & (item["new"] == rule["to"]) &
            (item["old_gap"] <= rule["old_gap"]) & (item["new_gap"] >= rule["new_gap"]) &
            (item["votes"] >= rule["votes"]) & (item["pair_support"] >= rule["pair_support"]))


def apply(design, incumbent, eligible, policy):
    if not policy["rules"]:
        return incumbent
    proposal = timeline.apply_residual(design, incumbent, policy["residual"], eligible)
    item = describe(design, incumbent, proposal, eligible)
    chosen = np.logical_or.reduce([accepted(item, rule) for rule in policy["rules"]])
    return np.where(chosen[:, None], proposal, incumbent)


def infer(blocks, incumbent_scores, policy):
    if not policy["rules"]:
        return incumbent_scores
    design, eligible = review.observe(blocks, incumbent_scores, policy["experts"])
    return apply(design, incumbent_scores, eligible, policy)


def source_weight(sources):
    _, indices, counts = np.unique(sources, return_inverse=True, return_counts=True)
    return 1. / (len(counts) * counts[indices])


def source_folds(splits):
    sources = np.concatenate(tuple(splits.values()))
    for _, picked in GroupKFold(4).split(sources, groups=sources):
        held = np.unique(sources[picked])
        fitting = np.flatnonzero(~np.isin(splits["train"], held))
        yield fitting, {split: np.flatnonzero(np.isin(groups, held)) for split, groups in splits.items()}


def build_records(variants, labels, sources, assignment, frozen):
    records = {split: {"incumbent": np.full((len(ids), 3), np.nan), "eligible": np.zeros(len(ids), dtype=bool),
                       "design": np.zeros((len(ids), 8)), "truth": labels[split]["full"], "sources": ids}
               for split, ids in sources.items()}
    for fold, (fitting, testing) in enumerate(source_folds(sources), start=1):
        anchor = review.fit_anchor(variants["train"], labels["train"]["full"], sources["train"], fitting,
                                   assignment, frozen)
        experts = review.fit_experts(variants["train"], labels["train"], sources["train"], fitting,
                                     assignment, frozen)
        for split, indices in testing.items():
            blocks = {name: scores[indices] for name, scores in variants[split]["complete"].items()}
            incumbent = review.predict(anchor, blocks, None)
            complete = robust.observed_mask(blocks) == review.FULL
            if not complete.all():
                remaining, _ = robust.robust_predict(blocks, review.FULL, frozen)
                incumbent[~complete] = remaining["teacher"][~complete]
            design, eligible = review.observe(blocks, incumbent, experts)
            for key, values in (("incumbent", incumbent), ("design", design), ("eligible", eligible)):
                records[split][key][indices] = values
        print(f"Scored complete evidence without the held sources {fold}/4", flush=True)
    return records


def proposal_options(records, penalty):
    sources = {split: item["sources"] for split, item in records.items()}
    for share, offset in product((.5, 1.), (-.20, 0., .20, .40)):
        offered = {split: item["incumbent"].copy() for split, item in records.items()}
        for fitting, testing in source_folds(sources):
            model = review.fit_residual(records["train"], fitting, penalty)
            model.update(share=share, neutral_offset=offset)
            for split, indices in testing.items():
                item = records[split]
                offered[split][indices] = timeline.apply_residual(item["design"][indices],
                                                                 item["incumbent"][indices],
                                                                 model, item["eligible"][indices])
        yield offered, share, offset


def options_for_direction(start, end, records, scopes):
    choices = {(0,) * 6: (0., None)}
    for old_gap, new_gap, votes, pair_support in product((.04, .07, .10, .15, .20, .30, .45, .60),
                                                         (.0, .03, .06, .12), (0, 1, 2), (0, 2, 3)):
        rule = {"from": start, "to": end, "old_gap": old_gap, "new_gap": new_gap,
                "votes": votes, "pair_support": pair_support}
        utility, totals = 0., []
        for split, item in records.items():
            observed = scopes[split]
            selected = accepted(observed, rule)
            y = item["truth"][selected]
            delta = (y == end).astype(int) - (y == start)
            totals.extend(np.bincount(y, weights=delta, minlength=3).astype(int))
            utility += (.35 if split == "train" else .65) * float(delta @ observed["source_weight"][selected])
            utility -= .000003 * int(selected.sum())
        index = tuple(totals)
        if index not in choices or utility > choices[index][0]:
            choices[index] = utility, rule
    return choices


def choose_rules(records, offered, weights):
    scopes = {split: {**describe(item["design"], item["incumbent"], offered[split], item["eligible"]),
                      "source_weight": weights[split]} for split, item in records.items()}
    candidates = []
    for direction, (start, end) in enumerate(DIRECTIONS):
        candidates.extend((direction, delta, utility, rule) for delta, (utility, rule)
                          in options_for_direction(start, end, records, scopes).items())
    columns = np.asarray([np.r_[np.eye(len(DIRECTIONS))[i], delta] for i, delta, _, _ in candidates]).T
    lower = np.r_[np.ones(len(DIRECTIONS)), np.zeros(6)]
    upper = np.r_[np.ones(len(DIRECTIONS)), np.full(6, np.inf)]
    solution = milp(-np.array([item[2] for item in candidates]), integrality=np.ones(len(candidates)),
                    bounds=Bounds(0., 1.), constraints=LinearConstraint(columns, lower, upper))
    if not solution.success:
        raise RuntimeError("The exact complete-input direction search did not converge")
    rules = [item[3] for item, selected in zip(candidates, solution.x, strict=True)
             if item[3] is not None and selected > .5]
    results = {}
    for split, item in records.items():
        use = np.logical_or.reduce([accepted(scopes[split], rule) for rule in rules]) if rules else item["eligible"] & False
        results[split] = metrics(item["truth"], np.where(use[:, None], offered[split], item["incumbent"]))
    return rules, results


def choose(records):
    before = {split: metrics(item["truth"], item["incumbent"]) for split, item in records.items()}
    weights = {split: source_weight(item["sources"]) for split, item in records.items()}
    best = {"rules": [], "before": before, "after": before}
    rank = (0., 0.)
    for penalty in (.02, .08, .30):
        for proposed, share, offset in proposal_options(records, penalty):
            rules, after = choose_rules(records, proposed, weights)
            quality = tuple(.60 * (after[p]["accuracy"] - before[p]["accuracy"]) +
                            .40 * (after[p]["macro_f1"] - before[p]["macro_f1"]) for p in ("valid", "train"))
            if quality > rank and min(quality) >= 0.:
                rank = quality
                best = {"rules": rules, "penalty": penalty, "before": before, "after": after,
                        "residual": {"share": share, "neutral_offset": offset}}
    return best


def provenance(cache):
    return {**review.provenance(cache), "base_review": sha_file(Path(review.__file__)),
            "implementation": sha_file(Path(__file__))}


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
    records = build_records(variants, labels, sources, assignment, frozen)
    policy = choose(records)
    if policy["rules"]:
        training = np.arange(len(sources["train"]))
        policy["residual"].update(review.fit_residual(records["train"], training, policy["penalty"]))
        policy["experts"] = review.fit_experts(variants["train"], labels["train"], sources["train"], training,
                                              assignment, frozen)
    save_json(cache / timeline.DIRECTORY / MODEL, {**signature, "review": policy})
    for split in review.STAGES:
        print(split, {stage: (policy[stage][split]["correct"], policy[stage][split]["macro_f1"],
                              np.diag(policy[stage][split]["confusion"]).tolist())
                      for stage in ("before", "after")}, flush=True)


def load(cache):
    model = json.loads((cache / timeline.DIRECTORY / MODEL).read_text("utf8"))
    if any(model.get(key) != value for key, value in provenance(cache).items()):
        raise ValueError("Fit complete-input direction gates against the frozen development evidence first")
    return model["review"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    with threadpool_limits(limits=5, user_api="blas"):
        fit(parser.parse_args().cache)
