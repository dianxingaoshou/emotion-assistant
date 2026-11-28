"""Calibrate neutral reliability against full-clip human annotations."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import chsims_teacher_neutral_precision_review as published


MODEL = "neutral-reliability-review.json"
NEUTRAL = published.NEUTRAL
NEGATIVE, POSITIVE = published.POLARITIES
PHASES = published.PHASES
CONDITION = published.precision.CONDITION


def features(record, family):
    retained = record["retained"]
    cues = [record[name] for name in ("initial", "text", "acoustic", "mini")]
    cues.extend((retained[:, 0], retained[:, 1]))
    neutral = np.column_stack([s[:, NEUTRAL] - np.maximum(s[:, NEGATIVE], s[:, POSITIVE]) for s in cues])
    polarity = np.column_stack([s[:, POSITIVE] - s[:, NEGATIVE] for s in cues])
    present = np.isfinite(retained).all(axis=2)
    columns = [neutral, polarity, present]
    if family != "confidence":
        pairs = ((1, 2), (1, 4), (2, 5))
        columns.extend(values[:, i] - values[:, j] for values in (neutral, polarity) for i, j in pairs)
        columns.append(record["full_features"] if family == "expanded" else record["full_features"][:, 12:])
    if family == "expanded":
        columns.extend(published.log_ratios(retained[:, i]) for i in range(2))
    return np.nan_to_num(np.column_stack(columns), nan=0.)


def fit_binary(design, label, source, strength):
    center = design.mean(axis=0)
    scale = np.maximum(design.std(axis=0), .05)
    model = LogisticRegression(C=strength, max_iter=300)
    weights = published.calibrated.source_weights(source) * len(design)
    model.fit((design - center) / scale, label, sample_weight=weights)
    return {"center": center.tolist(), "scale": scale.tolist(), "coef": model.coef_[0].tolist(),
            "bias": float(model.intercept_[0])}


def train_heads(design, batch, selected, strength):
    truth = batch["target"]
    polar = selected & (truth != NEUTRAL)
    return {
        "neutral": fit_binary(design[selected], truth[selected] == NEUTRAL, batch["source"][selected], strength),
        "positive": fit_binary(design[polar], truth[polar] == POSITIVE, batch["source"][polar], strength),
    }


def probabilities(design, heads):
    return np.column_stack([
        expit((design - head["center"]) / head["scale"] @ np.asarray(head["coef"]) + head["bias"])
        for head in (heads["neutral"], heads["positive"])
    ])


def held_out_heads(records, batches, design, subset, strength):
    sources = np.concatenate([batches[phase]["source"] for phase in PHASES])
    result = {phase: np.full((len(x), 2), np.nan) for phase, x in design.items()}
    for _, indices in GroupKFold(4).split(sources, groups=sources):
        held = np.unique(sources[indices])
        train = records["train"][subset] & ~np.isin(batches["train"]["source"], held)
        heads = train_heads(design["train"], batches["train"], train, strength)
        for phase, x in design.items():
            selected = np.isin(batches[phase]["source"], held)
            result[phase][selected] = probabilities(x[selected], heads)
    if any(not np.isfinite(values).all() for values in result.values()):
        raise ValueError("Expected one reliability score for every development clip")
    return result


def veto(estimates, rule):
    neutral, direction = estimates.T
    return (neutral >= rule["neutral_veto"]) | (direction < rule["minimum_direction"])


def protect(incumbent, proposal, estimates, rule):
    changed = (incumbent.argmax(1) == NEUTRAL) & (proposal.argmax(1) == POSITIVE)
    return np.where((changed & veto(estimates, rule))[:, None], incumbent, proposal)


def replay_reference(records, batches, previous):
    rule = previous["rule"]
    x = {phase: published.features(record, POSITIVE, rule["family"]) for phase, record in records.items()}
    confidence = published.cross_fitted(records, batches, x, POSITIVE, rule["mode"], rule["subset"], rule["strength"])
    picked, predictions = {}, {}
    for phase, record in records.items():
        proposal = published.proposal(record["initial"], confidence[phase], POSITIVE, rule["mix"])
        cue = published.support(record, POSITIVE)[rule["cue"]]
        picked[phase] = published.select(record, proposal, confidence[phase], rule, cue)
        predictions[phase] = np.where(picked[phase][:, None], proposal, record["initial"])
    return confidence, picked, predictions


def choose(records, batches, previous):
    _, reference, outputs = replay_reference(records, batches, previous)
    baseline = {phase: published.metrics(batch["target"], outputs[phase]) for phase, batch in batches.items()}
    weights = {phase: published.calibrated.source_weights(batch["source"]) for phase, batch in batches.items()}
    truth = {phase: batch["target"] for phase, batch in batches.items()}
    scope = {phase: record["neutral"] & record["old_neutral"] &
             published.support(record, POSITIVE)["global_acoustic"]
             for phase, record in records.items()}
    totals = {phase: np.bincount(truth[phase][scope[phase]], minlength=3).tolist() for phase in PHASES}
    best = {"rule": None, "before": baseline, "after": baseline}
    rank = (0., 0., 0., 0.)
    for family in ("confidence", "contrast", "expanded"):
        design = {phase: features(record, family) for phase, record in records.items()}
        for subset, strength in product(("neutral", "active"), (.015, .05, .20, .80, 2.)):
            estimates = held_out_heads(records, batches, design, subset, strength)
            # Calibrate in a wider proposal pool, then deploy only on previously accepted changes.
            for neutral_limit, direction_floor in product((.32, .38, .43, .48, .53, .58, .63, .68, .75),
                                                           (.0, .40, .48, .56, .64, .72, .80)):
                rule = {"neutral_veto": neutral_limit, "minimum_direction": direction_floor}
                rejected = {phase: scope[phase] & veto(estimates[phase], rule) for phase in PHASES}
                if any(np.any(mask & (reference[k] | (truth[k] == POSITIVE)))
                       or not np.any(mask & (truth[k] == NEUTRAL)) for k, mask in rejected.items()):
                    continue
                gains = {phase: weights[phase][mask & (truth[phase] == NEUTRAL)].sum()
                         for phase, mask in rejected.items()}
                other = {phase: weights[phase][mask & (truth[phase] == NEGATIVE)].sum()
                         for phase, mask in rejected.items()}
                quality = (.35 * gains["train"] + .65 * gains["valid"],
                           .35 * other["train"] + .65 * other["valid"], direction_floor, -neutral_limit)
                if quality <= rank:
                    continue
                counts = {phase: np.bincount(truth[phase][mask], minlength=3).tolist()
                          for phase, mask in rejected.items()}
                best, rank = {
                    "rule": {**rule, "family": family, "strength": strength, "subset": subset},
                    "before": baseline, "after": baseline,
                    "broad_scope": {"total": totals, "rejected": counts},
                }, quality
    rule = best["rule"]
    if rule is not None:
        train = records["train"]
        best["heads"] = train_heads(features(train, rule["family"]), batches["train"], train[rule["subset"]],
                                   rule["strength"])
    return best


def infer(batch, incumbent, policy):
    rule = policy["rule"]
    proposal = published.infer(batch, incumbent, policy["published"])
    if rule is None:
        return proposal
    record = published.observe(batch, incumbent)
    estimates = probabilities(features(record, rule["family"]), policy["heads"])
    return protect(incumbent, proposal, estimates, rule)


def provenance(directory, splits):
    return {"code": published.sha_file(Path(__file__)), "splits": splits,
            "published": published.sha_file(directory / published.MODEL),
            "published_features": published.sha_file(Path(published.__file__)),
            "retained_development": published.sha_file(directory / published.evidence.DEVELOPMENT)}


def fit(cache):
    directory = cache / published.timeline.DIRECTORY
    prior = published.neutral_guard.load(cache, directory / "timeline-fusion.json")
    previous = published.load(cache, prior["splits"])
    acoustic = published.audio_evidence.load(cache)
    development = published.evidence.load_development(cache)
    batches = {phase: items[CONDITION] for phase, items in development.items()}
    for phase, batch in batches.items():
        rows = published.read_full_report(cache, phase)["rows"]
        batch.update(published.calibrated.acoustic_inputs(rows, acoustic[phase]))
        batch["quality"] = published.dynamic.quality_input(batch)
        batch["retained"] = development[phase]["synchronous_gap"]["context"][:, :2]
    incumbent = published.precision.development_incumbent(batches, prior["audio_review"]["policies"][CONDITION])
    records = {phase: published.observe(batch, incumbent[phase]) for phase, batch in batches.items()}
    policy = {**choose(records, batches, previous), "published": previous}
    published.save_json(directory / MODEL, {**provenance(directory, prior["splits"]), "policy": policy})
    print(policy["rule"], flush=True)
    print(policy.get("broad_scope"), flush=True)
    for phase in PHASES:
        print(phase, {key: (policy[key][phase]["correct"], policy[key][phase]["macro_f1"],
                            np.diag(policy[key][phase]["confusion"]).tolist())
                      for key in ("before", "after")}, flush=True)


def load(cache, splits):
    directory = cache / published.timeline.DIRECTORY
    model = json.loads((directory / MODEL).read_text("utf8"))
    if any(model.get(key) != value for key, value in provenance(directory, splits).items()):
        raise ValueError("Fit neutral reliability on frozen development evidence first")
    return model["policy"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=published.CACHE / "full")
    with threadpool_limits(limits=5, user_api="blas"):
        fit(parser.parse_args().cache)
