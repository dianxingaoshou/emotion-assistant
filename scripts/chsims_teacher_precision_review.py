"""Fit a source-held-out precision review for clips without visual evidence."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import chsims_teacher_audio_evidence as audio_evidence
from scripts import chsims_teacher_context_evidence as evidence
from scripts import chsims_teacher_dynamic_conflict as dynamic
from scripts import chsims_teacher_neutral_guard as neutral_guard
from scripts import chsims_teacher_source_calibration as calibrated
from scripts import chsims_teacher_timeline as timeline
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_teacher_cross_source import metrics
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file
from scripts.chsims_teacher_directional import log_ratios
from scripts.chsims_teacher_supervision import read_full_report


MODEL = "precision-review.json"
JOINT_MODEL = "precision-joint-review.json"
CONDITION = calibrated.NO_VISUAL
DIRECTIONS = tuple((a, b) for a, b in product(range(len(LABELS)), repeat=2) if a != b)


def development_incumbent(batches, policy):
    previous = policy["reference"]
    before = {phase: calibrated.infer(batch, previous["reference"]) for phase, batch in batches.items()}
    observed = {phase: dynamic.describe(batch, before[phase]) for phase, batch in batches.items()}
    shares = dynamic.held_shares(observed, batches, previous["strength"])
    rule = previous["rule"]
    scores = {}
    for phase, batch in batches.items():
        p = dynamic.fuse(observed[phase], np.clip(shares[phase] + rule["offset"], 0., 1.),
                         before[phase], rule["anchor"])
        changed = dynamic.accept(observed[phase], p, rule)
        if policy["text_floor"] is not None:
            changed &= ~neutral_guard.veto(batch, observed[phase], changed, policy["text_floor"])
        scores[phase] = np.where(changed[:, None], p, before[phase])
    return scores


def describe(batch, incumbent):
    text = batch["context"][:, 0]
    acoustic = batch["acoustic"]
    scores = (incumbent, batch["prior"], text, acoustic, batch["voice"], batch["mini"])
    raw = np.column_stack((*[log_ratios(p) for p in scores], batch["design"][:, 10:13]))
    active = batch["active"] & np.isfinite(incumbent).all(axis=1) & np.isfinite(raw).all(axis=1)
    return {"features": raw, "initial": incumbent, "active": active,
            "support": np.stack((text, acoustic), axis=1)}


def train_head(record, batch, indices, penalty):
    used = indices[record["active"][indices]]
    weights = calibrated.source_weights(batch["source"][used])
    return timeline.learn_residual(record["features"][used], record["initial"][used],
                                   batch["target"][used], weights, penalty)


def oof_proposals(data, batches, penalty):
    groups = {phase: batch["source"] for phase, batch in batches.items()}
    all_sources = np.concatenate(tuple(groups.values()))
    values = {phase: item["initial"].copy() for phase, item in data.items()}
    for _, held in GroupKFold(4).split(all_sources, groups=all_sources):
        sources = np.unique(all_sources[held])
        train = np.flatnonzero(~np.isin(groups["train"], sources))
        head = train_head(data["train"], batches["train"], train, penalty)
        for phase, item in data.items():
            selected = item["active"] & np.isin(groups[phase], sources)
            proposal = timeline.apply_residual(item["features"], item["initial"],
                                               {**head, "share": 1., "neutral_offset": 0.}, selected)
            values[phase][selected] = proposal[selected]
    return values


def selection(record, proposed, rule):
    reference = record["initial"]
    rows = np.arange(len(reference))
    old, new = reference.argmax(1), proposed.argmax(1)
    old_gap = reference[rows, old] - reference[rows, new]
    new_gap = proposed[rows, new] - proposed[rows, old]
    support = record["support"]
    votes = np.any(np.isfinite(support).all(2) & (support.argmax(2) == new[:, None]), axis=1)
    selected = (record["active"] & (old != new) & (old_gap <= rule["max_initial_gap"]) &
                (new_gap >= rule["min_proposed_gap"]))
    if rule["corroborate"]:
        selected &= votes
    if rule["direction"] is not None:
        initial, revised = rule["direction"]
        selected &= (old == initial) & (new == revised)
    return selected


def choose(batches, previous):
    incumbent = development_incumbent(batches, previous)
    data = {phase: describe(batch, incumbent[phase]) for phase, batch in batches.items()}
    baseline = {phase: metrics(batch["target"], incumbent[phase]) for phase, batch in batches.items()}
    weights = {phase: calibrated.source_weights(batch["source"]) for phase, batch in batches.items()}
    rank = (0., 0., 0.)
    best = {"before": baseline, "after": baseline, "rule": None}
    directions = (None, *DIRECTIONS)
    for penalty in (.03, .1, .3):
        predictions = oof_proposals(data, batches, penalty)
        for limit, minimum, corroborate, direction in product((.04, .08, .12, .20, .30, .45),
                                                              (.02, .06, .12, .20), (True, False), directions):
            rule = {"max_initial_gap": limit, "min_proposed_gap": minimum, "corroborate": corroborate,
                    "direction": direction}
            changes = {phase: selection(state, predictions[phase], rule) for phase, state in data.items()}
            after = {phase: metrics(batch["target"], np.where(changes[phase][:, None], predictions[phase],
                                                               incumbent[phase]))
                     for phase, batch in batches.items()}
            if any(np.any(np.diag(after[k]["confusion"]) < np.diag(v["confusion"])) for k, v in baseline.items()):
                continue
            delta = {k: after[k]["macro_f1"] - v["macro_f1"] for k, v in baseline.items()}
            if any(d < 0. for d in delta.values()):
                continue
            utility = {}
            for phase, batch in batches.items():
                used = changes[phase]
                truth = batch["target"][used]
                gain = (predictions[phase][used].argmax(1) == truth).astype(int) - (
                    incumbent[phase][used].argmax(1) == truth)
                utility[phase] = float(gain @ weights[phase][used])
            quality = (.35 * utility["train"] + .65 * utility["valid"], delta["valid"],
                       -sum(int(used.sum()) for used in changes.values()))
            if quality > rank:
                rank = quality
                best = {"rule": rule, "penalty": penalty, "before": baseline, "after": after}
    if best["rule"] is not None:
        best["head"] = train_head(data["train"], batches["train"], np.arange(len(incumbent["train"])), best["penalty"])
    return best


def direction_options(data, proposed, batches, direction):
    empty = (0,) * (2 * len(LABELS))
    options = {empty: (0., None)}
    weights = {phase: calibrated.source_weights(batch["source"]) for phase, batch in batches.items()}
    old, new = direction
    for limit, minimum, corroborate in product((.04, .08, .12, .20, .30, .45), (.02, .06, .12, .20), (True, False)):
        rule = {"max_initial_gap": limit, "min_proposed_gap": minimum, "corroborate": corroborate,
                "direction": direction}
        delta, utility = [], 0.
        for phase, scale in (("train", .35), ("valid", .65)):
            changed = selection(data[phase], proposed[phase], rule)
            truth = batches[phase]["target"][changed]
            gain = (truth == new).astype(int) - (truth == old)
            delta.extend(np.bincount(truth, weights=gain, minlength=len(LABELS)).astype(int))
            utility += scale * (gain @ weights[phase][changed]) - .000005 * len(truth)
        key = tuple(delta)
        if key not in options or utility > options[key][0]:
            options[key] = float(utility), rule
    return options


def joint_rules(data, proposed, batches):
    columns, costs, rules = [], [], []
    for direction, indicator in zip(DIRECTIONS, np.eye(len(DIRECTIONS)), strict=True):
        for delta, (utility, rule) in direction_options(data, proposed, batches, direction).items():
            columns.append(np.r_[indicator, delta])
            costs.append(-utility)
            rules.append(rule)
    # Select one option per direction while retaining every class count on both splits.
    lower = np.r_[np.ones(len(DIRECTIONS)), np.zeros(2 * len(LABELS))]
    upper = np.r_[np.ones(len(DIRECTIONS)), np.full(2 * len(LABELS), np.inf)]
    solved = milp(costs, integrality=np.ones(len(costs)), bounds=Bounds(0., 1.),
                  constraints=LinearConstraint(np.asarray(columns).T, lower, upper))
    if not solved.success:
        return [], 0.
    return [rule for rule, weight in zip(rules, solved.x, strict=True) if rule is not None and weight > .5], -solved.fun


def accept_any(state, proposed, rules):
    mask = np.zeros(len(proposed), dtype=bool)
    for rule in rules:
        mask |= selection(state, proposed, rule)
    return mask


def choose_joint(batches, previous):
    initial = development_incumbent(batches, previous)
    data = {phase: describe(batch, initial[phase]) for phase, batch in batches.items()}
    baseline = {phase: metrics(batch["target"], initial[phase]) for phase, batch in batches.items()}
    rank = (0., 0.)
    best = {"rules": [], "before": baseline, "after": baseline}
    for penalty in (.03, .1, .3):
        proposed = oof_proposals(data, batches, penalty)
        rules, utility = joint_rules(data, proposed, batches)
        after = {phase: metrics(batch["target"], np.where(
            accept_any(data[phase], proposed[phase], rules)[:, None], proposed[phase], initial[phase]))
            for phase, batch in batches.items()}
        quality = utility, after["valid"]["macro_f1"] - baseline["valid"]["macro_f1"]
        if quality > rank and all(after[k]["macro_f1"] >= v["macro_f1"] for k, v in baseline.items()):
            rank = quality
            best = {"rules": rules, "penalty": penalty, "before": baseline, "after": after}
    if best["rules"]:
        best["head"] = train_head(data["train"], batches["train"], np.arange(len(initial["train"])), best["penalty"])
    return best


def infer(batch, incumbent, policy):
    rules = policy.get("rules")
    if rules is None:
        rules = [] if policy["rule"] is None else [policy["rule"]]
    if not rules:
        return incumbent
    state = describe(batch, incumbent)
    proposal = timeline.apply_residual(state["features"], incumbent,
                                       {**policy["head"], "share": 1., "neutral_offset": 0.}, state["active"])
    return np.where(accept_any(state, proposal, rules)[:, None], proposal, incumbent)


def provenance(directory, splits):
    return {"code": sha_file(Path(__file__)), "reference": sha_file(directory / neutral_guard.MODEL), "splits": splits}


def fit(cache, joint=False):
    directory = cache / timeline.DIRECTORY
    previous = neutral_guard.load(cache, directory / "timeline-fusion.json")
    acoustic = audio_evidence.load(cache)
    batches = {phase: doc[CONDITION] for phase, doc in evidence.load_development(cache).items()}
    for phase, batch in batches.items():
        batch.update(calibrated.acoustic_inputs(read_full_report(cache, phase)["rows"], acoustic[phase]))
        batch["quality"] = dynamic.quality_input(batch)
    chooser = choose_joint if joint else choose
    selected = chooser(batches, previous["audio_review"]["policies"][CONDITION])
    print(selected.get("rules", selected.get("rule")),
          {phase: (m["correct"], round(m["macro_f1"], 5), np.diag(m["confusion"]).tolist())
           for phase, m in selected["after"].items()}, flush=True)
    save_json(directory / (JOINT_MODEL if joint else MODEL), {**provenance(directory, previous["splits"]),
                                                            "policy": selected})


def load(cache, splits, joint=False):
    directory = cache / timeline.DIRECTORY
    checkpoint = json.loads((directory / (JOINT_MODEL if joint else MODEL)).read_text("utf8"))
    if any(checkpoint.get(key) != value for key, value in provenance(directory, splits).items()):
        raise ValueError("Fit the precision review on the frozen development data first")
    return checkpoint["policy"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    parser.add_argument("--joint", action="store_true")
    args = parser.parse_args()
    fit(args.cache, args.joint)
