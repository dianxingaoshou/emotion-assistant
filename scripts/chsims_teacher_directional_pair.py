"""Fit a focused negative–neutral correction without replacing a positive decision."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import chsims_teacher_directional as directional
from scripts import chsims_teacher_robustness as robust
from scripts import chsims_teacher_timeline as timeline
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_teacher_cross_source import metrics
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file


MODEL = "directional-pair.json"
NEGATIVE, NEUTRAL, POSITIVE = directional.NEGATIVE, directional.NEUTRAL, directional.POSITIVE


def pair_posterior(batch, classifier):
    features = (batch["design"] - classifier["center"]) / classifier["scale"]
    return expit(features @ classifier["coef"] + classifier["intercept"])


def review_pair(previous, negative_probability, evidence, eligible, recover_at, neutral_at, required_votes):
    old = previous.argmax(1)
    negative_support = directional.support_counts(evidence, np.full(len(old), NEGATIVE))
    neutral_support = directional.support_counts(evidence, np.full(len(old), NEUTRAL))
    recover = (old == NEUTRAL) & (negative_probability >= recover_at) & (negative_support >= required_votes)
    neutral = (old == NEGATIVE) & (negative_probability <= neutral_at) & (neutral_support >= 1)
    allowed = eligible & np.isfinite(previous[:, 0]) & (recover | neutral)
    # Leave the third class's score intact. The pair gets its original total probability.
    proposal = previous.copy()
    # Moving the class requires the pair's inferred ordering to agree with the proposed direction.
    strength = np.where(recover, np.maximum(negative_probability, .501), np.minimum(negative_probability, .499))
    pair_mass = previous[:, NEGATIVE] + previous[:, NEUTRAL]
    proposal[:, NEGATIVE] = pair_mass * strength
    proposal[:, NEUTRAL] = pair_mass * (1. - strength)
    decision = np.where(recover, NEGATIVE, NEUTRAL)
    return np.where((allowed & (proposal.argmax(1) == decision))[:, None], proposal, previous)


def predict(batch, policy):
    if policy["rule"] is None:
        return batch["previous"]
    posterior = pair_posterior(batch, policy["rule"])
    return review_pair(batch["previous"], posterior, batch["support"], batch["active"],
                       policy["recover_at"], policy["neutral_at"], policy["required_votes"])


def fit_pair(data, truth, groups, focus, strength):
    current = data["previous"].argmax(1)
    candidates = data["active"] & (current != POSITIVE) & (truth != POSITIVE)
    votes = directional.support_counts(data["support"], current)
    examples = np.flatnonzero(candidates & (votes < data["support"].shape[1]) if focus else candidates)
    design = data["design"][examples]
    center, scale = design.mean(0), np.maximum(design.std(0), .04)
    weights = robust.sample_weights(truth, groups, examples)
    model = LogisticRegression(C=strength, max_iter=500)
    model.fit((design - center) / scale, truth[examples] == NEGATIVE, sample_weight=weights / weights.mean())
    return {"center": center.tolist(), "scale": scale.tolist(), "coef": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]), "focus_on_disagreement": focus, "C": strength}


def choose(name, train, valid, truth, targets, source, validation_source):
    previous = valid["previous"]
    before = metrics(targets, previous)
    old_hits = np.diag(np.asarray(before["confusion"])[:, :len(LABELS)])
    candidate = {"rule": None, "previous_validation": before, "validation": before, "development_changes": 0}
    before_top = previous.argmax(1)
    before_correct = np.isfinite(previous[:, 0]) & (before_top == targets)
    _, group, count = np.unique(validation_source, return_counts=True, return_inverse=True)
    rank = (0., 0, 0)
    for focus in (False, True):
        for strength in (.03, .15, .75):
            classifier = fit_pair(train, truth, source, focus, strength)
            posterior = pair_posterior(valid, classifier)
            for recover_at in (.42, .50, .58, .66, .74, .82, .90, .96):
                for neutral_at in (0., .16, .24, .32, .40, .48):
                    for required_votes in (2, 3):
                        output = review_pair(previous, posterior, valid["support"], valid["active"],
                                             recover_at, neutral_at, required_votes)
                        measured = metrics(targets, output)
                        recall = np.diag(np.asarray(measured["confusion"])[:, :len(LABELS)])
                        if (np.any(recall < old_hits) or recall[NEGATIVE] <= old_hits[NEGATIVE] or
                                measured["macro_f1"] <= before["macro_f1"]):
                            continue
                        top = output.argmax(1)
                        gains = (np.isfinite(output[:, 0]) & (top == targets)).astype(int) - before_correct
                        source_gain = np.mean(np.bincount(group, weights=gains) / count)
                        changes = int(np.count_nonzero(top != before_top))
                        score = (.35 * (measured["accuracy"] - before["accuracy"]) +
                                 .35 * (measured["macro_f1"] - before["macro_f1"]) + .30 * source_gain)
                        quality = (score, int(recall[NEGATIVE]), -changes)
                        if quality > rank:
                            candidate = {"rule": classifier, "recover_at": recover_at, "neutral_at": neutral_at,
                                         "required_votes": required_votes, "previous_validation": before,
                                         "validation": measured, "development_changes": changes,
                                         "source_accuracy_delta": float(source_gain)}
                            rank = quality
    print(f"{name} pair validation {before['correct']} -> {candidate['validation']['correct']}; "
          f"Macro-F1 {before['macro_f1']:.4f} -> {candidate['validation']['macro_f1']:.4f}", flush=True)
    return candidate


def fit(cache):
    previous, model, original = directional.frozen_inputs(cache)
    data, truth, sources = directional.development_data(cache, model, original)
    policies = {case: choose(case, data["train"][case], data["valid"][case], truth["train"], truth["valid"],
                             sources["train"], sources["valid"]) for case in timeline.CASES}
    save_json(cache / timeline.DIRECTORY / MODEL, {"kind": "pair", "implementation": sha_file(Path(__file__)),
              "features": sha_file(Path(directional.__file__)), "previous_model": sha_file(previous),
              "splits": model["splits"], "policies": policies})


def load(cache, previous):
    result = json.loads((cache / timeline.DIRECTORY / MODEL).read_text("utf8"))
    if (result["implementation"] != sha_file(Path(__file__)) or result["features"] != sha_file(Path(directional.__file__))
            or result["previous_model"] != sha_file(previous)):
        raise ValueError("Fit the pair correction on development data using the current implementation")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    args = parser.parse_args()
    with threadpool_limits(limits=5, user_api="blas"):
        fit(args.cache)


if __name__ == "__main__":
    main()
