"""Limit isolated audio revisions when independent whole-clip cues agree on neutral."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import chsims_teacher_acoustic_continuity_review as continuity


MODEL = "neutral-agreement-review.json"
review = continuity.review
calibrated = continuity.calibrated
pair = continuity.pair
NEUTRAL, POSITIVE = pair.NEUTRAL, pair.POSITIVE


def independent_evidence(record):
    text, acoustic = record["text"], record["acoustic"]
    tm = calibrated.signed_margin(text, NEUTRAL)
    am = calibrated.signed_margin(acoustic, NEUTRAL)
    consensus = np.minimum(tm, am)
    primary = np.column_stack((text, acoustic))
    contrast = np.column_stack((review.log_ratios(text), review.log_ratios(acoustic),
                                tm, am, consensus, text[:, POSITIVE] - acoustic[:, POSITIVE]))
    auxiliary = ((calibrated.signed_margin(record["mini"], POSITIVE) > 0.) |
                 (calibrated.signed_margin(record["retained"][:, 1], POSITIVE) > 0.))
    return {"simple": np.nan_to_num(primary, nan=0.), "extended": np.nan_to_num(contrast, nan=0.),
            "consensus": consensus, "auxiliary": auxiliary}


def scope(record, evidence, probability, rule):
    return (record["active"] & (evidence["consensus"] > 0.) & evidence["auxiliary"] &
            (evidence["consensus"] >= rule["margin"]) & (probability >= rule["min_probability"]))


def reduce_confidence(probability, independent, share):
    return probability - share * np.maximum(probability - independent, 0.)


def introduced_by_acoustics(record, base, reference):
    _, accepted = pair.reference_result(record, base, reference)
    return ~accepted & (continuity.acoustic_margin(record) > 0.)


def revised_decision(record, previous, probability, independent, protected, rule, reference, settings):
    initial = record["initial"]
    changed = (initial.argmax(1) == NEUTRAL) & (previous.argmax(1) == POSITIVE)
    if not np.any(changed & protected):
        return previous
    lower = reduce_confidence(probability, independent, rule["share"])
    proposal = review.proposal(initial, lower, POSITIVE, reference["mix"])
    min_target, min_lead = reference["min_target"], reference["min_lead"]
    if settings is not None:
        adjusted = continuity.acoustic_margin(record) != 0.
        min_target = np.where(adjusted, settings["min_target"], min_target)
        min_lead = np.where(adjusted, settings["min_lead"], min_lead)
    accepted = ((lower >= min_target) & (proposal.argmax(1) == POSITIVE) &
                (proposal[:, POSITIVE] - proposal[:, NEUTRAL] >= min_lead))
    return np.where((changed & protected)[:, None], np.where(accepted[:, None], proposal, initial), previous)


def confidence(reference, record, settings):
    if settings is None:
        return reference
    return continuity.adjusted_scores(reference, continuity.acoustic_margin(record), settings["gain"])


def choose(records, batches, reference, settings):
    folded = pair.reference_confidence(records, batches, reference)
    previous = {k: continuity.apply(r, folded[k], reference, settings) for k, r in records.items()}
    before = {k: continuity.metrics(b["target"], previous[k]) for k, b in batches.items()}
    best = {"rule": None, "before": before, "after": before}
    rank = (0., 0.)
    primary = {k: independent_evidence(r) for k, r in records.items()}
    posterior = {k: confidence(folded[k], r, settings) for k, r in records.items()}
    added = {k: introduced_by_acoustics(r, folded[k], reference) for k, r in records.items()}
    observed = {k: {**r, "both_neutral": r["active"] & (primary[k]["consensus"] > 0.)}
                for k, r in records.items()}
    for family, subset, strength in product(("simple", "extended"), ("active", "both_neutral"), (.01, .05, .25, 1.)):
        design = {k: p[family] for k, p in primary.items()}
        independent = review.cross_fitted(observed, batches, design, POSITIVE, "binary", subset, strength)
        for margin, share, minimum in product((0., .05, .1, .15, .2), (.25, .5, .75, 1.), (0., .1, .2, .3, .4)):
            rule = {"margin": margin, "share": share, "min_probability": minimum, "family": family,
                    "subset": subset, "strength": strength}
            checks, after = {}, {}
            for phase, record in records.items():
                protected = scope(record, primary[phase], posterior[phase], rule)
                truth = batches[phase]["target"]
                sources = batches[phase]["source"][protected]
                groups = len(np.unique(sources))
                if groups < 5:
                    break
                scores = revised_decision(record, previous[phase], posterior[phase], independent[phase],
                                          protected & added[phase], rule, reference["rule"], settings)
                new_labels, old_labels = scores.argmax(1), previous[phase].argmax(1)
                if np.any((truth == old_labels) & (truth != new_labels)):
                    break
                report = continuity.metrics(truth, scores)
                if report["macro_f1"] < before[phase]["macro_f1"]:
                    break
                after[phase] = report
                expected = truth[protected] == POSITIVE
                old = posterior[phase][protected]
                new = reduce_confidence(old, independent[phase][protected], share)
                weight = calibrated.source_weights(sources)
                brier = [float(weight @ ((prob - expected) ** 2)) for prob in (old, new)]
                if brier[1] >= brier[0]:
                    break
                checks[phase] = {"brier_before": brier[0], "brier_after": brier[1], "n": len(expected),
                                 "sources": groups, "positive": int(expected.sum()),
                                 "downweighted": int((old > new).sum()),
                                 "decisions_changed": int((new_labels != old_labels).sum())}
            if len(checks) != len(batches):
                continue
            gain = .35 * (checks["train"]["brier_before"] - checks["train"]["brier_after"])
            gain += .65 * (checks["valid"]["brier_before"] - checks["valid"]["brier_after"])
            quality = (gain, -share)
            if quality > rank:
                best = {"rule": rule, "before": before, "after": after, "calibration": checks}
                rank = quality
    if best["rule"] is not None:
        rule = best["rule"]
        features = primary["train"][rule["family"]]
        best["head"] = review.fit_head(observed["train"], batches["train"], features,
                                       np.ones(len(features), dtype=bool), POSITIVE, "binary",
                                       rule["subset"], rule["strength"])
    return best


def infer(batch, incumbent, policy):
    rule, old = policy["rule"], policy["continuity"]
    reference, settings = old["reference"], old["rule"]
    record = review.observe(batch, incumbent)
    config = reference["rule"]
    design = review.features(record, POSITIVE, config["family"])
    base = review.predict(design, reference["head"], config["mode"], POSITIVE)
    previous = continuity.apply(record, base, reference, settings)
    if rule is None:
        return previous
    evidence = independent_evidence(record)
    probability = confidence(base, record, settings)
    protected = scope(record, evidence, probability, rule) & introduced_by_acoustics(record, base, reference)
    independent = review.predict(evidence[rule["family"]], policy["head"], "binary", POSITIVE)
    return revised_decision(record, previous, probability, independent, protected, rule, config, settings)


def provenance(directory, splits):
    return {"code": continuity.sha_file(Path(__file__)), "splits": splits,
            "continuity": continuity.sha_file(directory / continuity.MODEL),
            "continuity_code": continuity.sha_file(Path(continuity.__file__)),
            "evidence": continuity.sha_file(directory / pair.context_evidence.DEVELOPMENT)}


def fit(cache):
    records, batches, reference, splits = pair.development_records(cache)
    old = continuity.load(cache, splits)
    policy = choose(records, batches, reference, old["rule"])
    directory = cache / continuity.timeline.DIRECTORY
    continuity.save_json(directory / MODEL, {**provenance(directory, splits), "policy": policy})
    print(policy["rule"], flush=True)
    print(policy.get("calibration"), flush=True)
    for phase in batches:
        print(phase, {key: (policy[key][phase]["correct"], policy[key][phase]["macro_f1"],
                            np.diag(policy[key][phase]["confusion"]).tolist())
                      for key in ("before", "after")}, flush=True)


def load(cache, splits):
    directory = cache / continuity.timeline.DIRECTORY
    model = json.loads((directory / MODEL).read_text("utf8"))
    if any(model.get(key) != value for key, value in provenance(directory, splits).items()):
        raise ValueError("Fit neutral agreement on the frozen development evidence first")
    return {**model["policy"], "continuity": continuity.load(cache, splits)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=continuity.CACHE / "full")
    with threadpool_limits(limits=5, user_api="blas"):
        fit(parser.parse_args().cache)
