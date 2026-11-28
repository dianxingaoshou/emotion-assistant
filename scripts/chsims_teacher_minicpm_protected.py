"""Protect established polarity when using MiniCPM-o to recover neutral decisions."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import chsims_teacher_minicpm_review as review
from scripts import chsims_teacher_robustness as robust
from scripts import chsims_teacher_timeline as timeline
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_teacher_cross_source import metrics
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file


MODEL = "minicpm-protected.json"
TRANSITIONS = tuple((first, second) for first in range(len(LABELS)) for second in range(len(LABELS)) if first != second)


def protect(batch, offered, gate, maximum_polar_confidence):
    previous = batch["prior"]
    to_neutral = (previous.argmax(1) != review.NEUTRAL) & (offered.argmax(1) == review.NEUTRAL)
    protected = to_neutral & (review.decision_quality(previous) > maximum_polar_confidence)
    return review.use_voice({**batch, "active": batch["active"] & ~protected}, offered, gate)


def infer(batch, policy):
    if not policy["rules"]:
        return batch["prior"]
    offered = review.scores(batch, policy["method"], policy["residual"])
    previous = batch["prior"]
    old, new = previous.argmax(1), offered.argmax(1)
    result = previous.copy()
    for rule in policy["rules"]:
        selected = batch["active"] & (old == rule["from"]) & (new == rule["to"])
        values = protect({**batch, "active": selected}, offered, rule["gate"], rule["maximum_polar_confidence"])
        result[selected] = values[selected]
    return result


def transition_data(batch, proposed, first, second):
    previous = batch["prior"]
    selected = batch["active"] & (previous.argmax(1) == first) & (proposed.argmax(1) == second)
    chosen = np.flatnonzero(selected)
    _, group, count = np.unique(batch["source"], return_inverse=True, return_counts=True)
    gain = (batch["target"][chosen] == second).astype(int) - (batch["target"][chosen] == first)
    return {"batch": {key: value[chosen] for key, value in batch.items()}, "proposed": proposed[chosen],
            "gain": gain, "weight": 1. / (len(count) * count[group[chosen]])}


def options_for(pair, train, valid):
    first, second = pair
    options = {tuple(np.zeros(2 * len(LABELS), dtype=int)): (0., None)}
    if not len(train["gain"]) and not len(valid["gain"]):
        return options
    to_center = second == review.NEUTRAL
    for maximum in ((.15, .30, .50, 1.) if to_center else (1.,)):
        for quality in (.25, .45, .65):
            for certainty in (.03, .10, .20):
                for strict in (False, True):
                    for threshold in (0., .05, .12, .24, .40, .60):
                        gate = {"voice_quality": quality, "voice_confidence": certainty, "strict": strict,
                                "center_only": False, "neutral_margin": threshold, "polar_margin": threshold}
                        deltas, values, changed = [], [], 0
                        for entry in (train, valid):
                            batch = entry["batch"]
                            values_for_entry = protect(batch, entry["proposed"], gate, maximum)
                            moves = values_for_entry.argmax(1) == second
                            weights = entry["gain"] * moves
                            deltas.extend(np.bincount(batch["target"], weights=weights, minlength=len(LABELS)).astype(int))
                            values.append(float(weights @ entry["weight"]))
                            changed += int(moves.sum())
                        score = .35 * values[0] + .65 * values[1] - .000001 * changed
                        key = tuple(deltas)
                        if key not in options or score > options[key][0]:
                            options[key] = (score, {"from": first, "to": second, "gate": gate,
                                                   "maximum_polar_confidence": maximum})
    return options


def select_rules(train, valid, proposed_train, proposed_valid):
    scores, columns, rules = [], [], []
    for index, (first, second) in enumerate(TRANSITIONS):
        train_pair = transition_data(train, proposed_train, first, second)
        valid_pair = transition_data(valid, proposed_valid, first, second)
        choice = np.eye(len(TRANSITIONS))[index]
        for delta, (value, rule) in options_for((first, second), train_pair, valid_pair).items():
            columns.append(np.r_[choice, delta])
            scores.append(-value)
            rules.append(rule)
    constraints = np.ones(len(TRANSITIONS))
    minimum = np.r_[constraints, np.zeros(2 * len(LABELS))]
    minimum[len(TRANSITIONS) + len(LABELS) + review.NEUTRAL] = 1
    maximum = np.r_[constraints, np.full(2 * len(LABELS), np.inf)]
    solution = milp(scores, integrality=np.ones(len(scores)), bounds=Bounds(0., 1.),
                    constraints=LinearConstraint(np.asarray(columns).T, minimum, maximum))
    if not solution.success:
        return []
    return [rule for rule, included in zip(rules, solution.x, strict=True) if rule is not None and included > .5]


def choose(train, valid):
    train_y, valid_y = train["target"], valid["target"]
    training = metrics(train_y, train["prior"])
    validation = metrics(valid_y, valid["prior"])
    best = {"method": None, "residual": None, "rules": [], "before": validation, "after": validation,
            "train_before": training, "train_after": training,
            "voice_validation_label_A": metrics(valid["audio_target"], valid["voice"])}
    active = np.flatnonzero(train["active"])
    weight = robust.sample_weights(train_y, train["source"], active)
    methods = [(name, None) for name in ("mini", "voice_mix")]
    methods.extend(("learned", timeline.learn_residual(train["design"][active], train["prior"][active],
                    train_y[active], weight, penalty)) for penalty in (.015, .06))
    rank = (0., 0., 0.)
    for method, residual in methods:
        rules = select_rules(train, valid, review.scores(train, method, residual), review.scores(valid, method, residual))
        if not rules:
            continue
        policy = {"method": method, "residual": residual, "rules": rules}
        trained = metrics(train_y, infer(train, policy))
        measured = metrics(valid_y, infer(valid, policy))
        if measured["macro_f1"] <= validation["macro_f1"] or trained["macro_f1"] < training["macro_f1"]:
            continue
        quality = (.30 * (measured["accuracy"] - validation["accuracy"]) +
                   .55 * (measured["macro_f1"] - validation["macro_f1"]) +
                   .15 * (trained["macro_f1"] - training["macro_f1"]))
        key = (quality, measured["macro_f1"], -len(rules))
        if key > rank:
            rank = key
            best.update(**policy, after=measured, train_after=trained)
    return best


def fit(cache):
    directory = cache / timeline.DIRECTORY
    reference = directory / "timeline-fusion.json"
    original = review.load(cache, reference)["audio_review"]
    provenance = {key: original[key] for key in ("guard", "refinement", "implementation", "checkpoint", "splits")}
    for phase, digest in provenance["splits"].items():
        if sha_file(cache / f"evaluation-{phase}.json") != digest:
            raise ValueError(f"The {phase} data changed")
    with np.load(directory / review.DEVELOPMENT, allow_pickle=False) as stored:
        if json.loads(stored["provenance"].item()) != provenance:
            raise ValueError("Run the MiniCPM-o development experiment first")
        data = {phase: {case: {name.removeprefix(phase + "/" + case + "/"): stored[name] for name in stored.files
                              if name.startswith(phase + "/" + case + "/")} for case in timeline.CASES}
                for phase in ("train", "valid")}
    policies = {case: choose(data["train"][case], data["valid"][case]) for case in timeline.CASES}
    for case, policy in policies.items():
        print(case, policy["method"], policy["rules"], policy["before"]["correct"], "->", policy["after"]["correct"],
              "training", policy["train_before"]["correct"], "->", policy["train_after"]["correct"], flush=True)
    save_json(directory / MODEL, {"implementation": sha_file(Path(__file__)),
                                  "review_sha256": sha_file(directory / review.MODEL), "policies": policies})


def load(cache, reference):
    directory = cache / timeline.DIRECTORY
    model = review.load(cache, reference)
    settings = json.loads((directory / MODEL).read_text("utf8"))
    if (settings["implementation"] != sha_file(Path(__file__)) or
            settings["review_sha256"] != sha_file(directory / review.MODEL)):
        raise ValueError("Fit the polarity protection using the frozen MiniCPM-o review")
    return {**model, "kind": "minicpm_protected", "audio_review": {**model["audio_review"],
                                                                   "policies": settings["policies"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    args = parser.parse_args()
    with threadpool_limits(limits=5, user_api="blas"):
        fit(args.cache)


if __name__ == "__main__":
    main()
