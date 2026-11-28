"""Fit a source-held-out review for complete text, sound, and image input."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import chsims_teacher_robustness as robust
from scripts import chsims_teacher_timeline as timeline
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_omni_model_comparison import predict
from scripts.chsims_teacher_cross_source import group_ids, metrics
from scripts.chsims_teacher_data import CACHE, sha_file
from scripts.chsims_teacher_supervision import numbers, read_full_report


MODEL = "complete-review.json"
MODALITIES = ("text", "audio", "visual")
FULL = 7
STAGES = ("train", "valid")


def fit_experts(variants, labels, sources, indices, augmentation, frozen):
    experts = {}
    for name in MODALITIES:
        mask = robust.CHANNEL[name]
        features = frozen["heads"][str(mask)]["heads"]["teacher"]["features"]
        experts[name] = robust.fit_head(variants, features, mask, labels[name], sources, indices, augmentation,
                                        {"C": .15, "temperature": 1., "neutral_shift": 0.})
    return experts


def fit_anchor(variants, truth, sources, indices, augmentation, frozen):
    settings = frozen["heads"][str(FULL)]
    if settings["teacher_weight"] != 1.:
        raise ValueError("Expected the frozen full teacher without a desktop score contribution")
    head = settings["heads"]["teacher"]
    return robust.fit_head(variants, head["features"], FULL, truth, sources, indices, augmentation,
                           {key: head[key] for key in ("C", "temperature", "neutral_shift")})


def observe(blocks, incumbent, experts):
    eligible = (robust.observed_mask(blocks) == FULL) & np.all(np.isfinite(incumbent), axis=1)
    probabilities = [incumbent, *(predict(experts[name], blocks, None) for name in MODALITIES)]
    for scores in probabilities[1:]:
        eligible &= np.all(np.isfinite(scores), axis=1)
    contrasts = []
    for scores in probabilities:
        logs = np.log(np.maximum(np.where(np.isfinite(scores), scores, 1. / 3.), 1e-8))
        contrasts.append(logs[:, [0, 2]] - logs[:, [1]])
    return np.column_stack(contrasts), eligible


def apply_review(features, incumbent, eligible, policy):
    if policy["rule"] is None:
        return incumbent
    rule = policy["rule"]
    offered = timeline.apply_residual(features, incumbent, rule, eligible)
    old, new = incumbent.argmax(axis=1), offered.argmax(axis=1)
    ordered = np.sort(incumbent, axis=1)
    gap = ordered[:, -1] - ordered[:, -2]
    positions = np.arange(len(old))
    lead = offered[positions, new] - offered[positions, old]
    change = eligible & (old != new) & (gap <= rule["max_anchor_margin"]) & (lead >= rule["min_proposal_margin"])
    return np.where(change[:, None], offered, incumbent)


def infer(blocks, incumbent_scores, policy):
    if policy["rule"] is None:
        return incumbent_scores
    features, eligible = observe(blocks, incumbent_scores, policy["experts"])
    return apply_review(features, incumbent_scores, eligible, policy)


def source_score(truth, probabilities, sources):
    won = np.isfinite(probabilities[:, 0]) & (truth == probabilities.argmax(axis=1))
    _, parts, counts = np.unique(sources, return_inverse=True, return_counts=True)
    return float((np.bincount(parts, weights=won) / counts).mean())


def fit_residual(record, training, penalty):
    incumbent = record["incumbent"]
    eligible = training[record["eligible"][training]]
    weights = robust.sample_weights(record["truth"], record["sources"], eligible)
    return timeline.learn_residual(record["design"][eligible], incumbent[eligible], record["truth"][eligible],
                                   weights, penalty)


def score_rule(record, settings, rule):
    incumbent = record["incumbent"]
    scores = incumbent.copy()
    for indices, head in settings:
        scores[indices] = apply_review(record["design"][indices], incumbent[indices], record["eligible"][indices],
                                       {"rule": {**head, **rule}})
    result = metrics(record["truth"], scores)
    return result, source_score(record["truth"], scores, record["sources"])


def choose(records, folds):
    baseline = {phase: metrics(record["truth"], record["incumbent"]) for phase, record in records.items()}
    source_baseline = {phase: source_score(record["truth"], record["incumbent"], record["sources"])
                       for phase, record in records.items()}
    best = {"rule": None, "before": baseline, "after": baseline}
    rank = (0., 0., 0.)
    for penalty in (.02, .08, .30):
        heads = {phase: [] for phase in STAGES}
        for training, held in folds:
            head = fit_residual(records["train"], training, penalty)
            for phase in STAGES:
                heads[phase].append((np.flatnonzero(np.isin(records[phase]["sources"], held)), head))
        for share, offset, anchor, proposed in product((.25, .50, .75, 1.), (-.20, 0., .20, .40, .60),
                                                         (.10, .20, .35, .55), (.0, .04, .08)):
            rule = {"share": share, "neutral_offset": offset, "max_anchor_margin": anchor,
                    "min_proposal_margin": proposed}
            after, source = {}, {}
            for phase in STAGES:
                after[phase], source[phase] = score_rule(records[phase], heads[phase], rule)
            change = {phase: after[phase]["accuracy"] - baseline[phase]["accuracy"] for phase in STAGES}
            macro = {phase: after[phase]["macro_f1"] - baseline[phase]["macro_f1"] for phase in STAGES}
            if min(change.values()) < 0. or min(macro.values()) < 0.:
                continue
            if any(after[phase]["confusion"][1][1] < baseline[phase]["confusion"][1][1] for phase in STAGES):
                continue
            validation_gain = .55 * change["valid"] + .30 * macro["valid"] + .15 * (source["valid"] -
                                                                                  source_baseline["valid"])
            training_gain = .55 * change["train"] + .30 * macro["train"] + .15 * (source["train"] -
                                                                                 source_baseline["train"])
            quality = (validation_gain, training_gain, -share)
            if quality > rank:
                rank = quality
                best = {"rule": rule, "penalty": penalty, "before": baseline, "after": after}
    return best


def provenance(cache):
    frozen = cache / "missing-evidence/fusion-heads.json"
    refinement = cache / "teacher-refinement/heads.json"
    return {"implementation": sha_file(Path(__file__)), "frozen": sha_file(frozen),
            "refinement": sha_file(refinement),
            "splits": {phase: sha_file(cache / f"evaluation-{phase}.json") for phase in STAGES}}


def fit(cache):
    record = provenance(cache)
    frozen = json.loads((cache / "missing-evidence/fusion-heads.json").read_text("utf8"))
    refinement = json.loads((cache / "teacher-refinement/heads.json").read_text("utf8"))
    if (frozen["splits"] != record["splits"] or frozen["implementation"] != sha_file(Path(robust.__file__))
            or refinement["corrections"][str(FULL)]["proposal"] != "previous"):
        raise ValueError("Reproduce the frozen complete-input teacher before fitting a review")
    documents = {phase: read_full_report(cache, phase) for phase in STAGES}
    variants, inputs = {}, {}
    for phase, report in documents.items():
        variants[phase], _, extraction = robust.inputs(cache, report)
        if extraction != frozen["gap_extractors"]:
            raise ValueError("Expected frozen development extraction")
        inputs[phase] = variants[phase]["complete"]
        print(f"Loaded complete-input {phase} features", flush=True)
    y = {phase: {name: numbers(doc["rows"], name) for name in (*MODALITIES, "full")}
         for phase, doc in documents.items()}
    groups = {phase: group_ids(doc["rows"]) for phase, doc in documents.items()}
    augmentation = robust.case_assignment(documents["train"]["rows"])
    combined = np.concatenate(tuple(groups.values()))
    folds = []
    records = {phase: {"incumbent": np.full((len(ids), 3), np.nan), "eligible": np.zeros(len(ids), dtype=bool),
                       "design": np.zeros((len(ids), 8)), "truth": y[phase]["full"], "sources": ids}
               for phase, ids in groups.items()}
    for split, (_, withheld) in enumerate(GroupKFold(4).split(combined, groups=combined), start=1):
        held = np.unique(combined[withheld])
        train = np.flatnonzero(~np.isin(groups["train"], held))
        folds.append((train, held))
        anchor = fit_anchor(variants["train"], y["train"]["full"], groups["train"], train, augmentation, frozen)
        experts = fit_experts(variants["train"], y["train"], groups["train"], train, augmentation, frozen)
        for phase, ids in groups.items():
            index = np.flatnonzero(np.isin(ids, held))
            blocks = {name: values[index] for name, values in inputs[phase].items()}
            incumbent = predict(anchor, blocks, None)
            complete = robust.observed_mask(blocks) == FULL
            # Natural losses are handled by the old teacher throughout training and inference.
            if not complete.all():
                backed_up, _ = robust.robust_predict(blocks, FULL, frozen)
                incumbent[~complete] = backed_up["teacher"][~complete]
            design, eligible = observe(blocks, incumbent, experts)
            records[phase]["incumbent"][index] = incumbent
            records[phase]["eligible"][index] = eligible
            records[phase]["design"][index] = design
        print(f"Scored development without held sources {split}/4", flush=True)
    review = choose(records, folds)
    if review["rule"] is not None:
        train = np.arange(len(groups["train"]))
        review["rule"].update(fit_residual(records["train"], train, review["penalty"]))
        review["experts"] = fit_experts(variants["train"], y["train"], groups["train"], train, augmentation, frozen)
    save_json(cache / timeline.DIRECTORY / MODEL, {**record, "review": review})
    for phase in STAGES:
        print(phase, {stage: (review[stage][phase]["correct"], review[stage][phase]["macro_f1"],
                              np.diag(review[stage][phase]["confusion"]).tolist())
                      for stage in ("before", "after")}, flush=True)


def load(cache):
    model = json.loads((cache / timeline.DIRECTORY / MODEL).read_text("utf8"))
    if any(model.get(key) != value for key, value in provenance(cache).items()):
        raise ValueError("Fit complete-input review on the frozen development reports first")
    return model["review"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    with threadpool_limits(limits=5, user_api="blas"):
        fit(parser.parse_args().cache)
