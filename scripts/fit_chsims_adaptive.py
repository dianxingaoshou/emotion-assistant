"""Learn reliability weights on source-held-out train predictions; select on official validation only."""

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from fit_chsims_foundation import (DATA, SCENARIOS, checksums_for_raw, components, dataset, embeddings,
                                   file_sha, frozen_cohort, fusion, inputs, metrics, read_json, target, write_json)
from emotion_assistant import adaptive_clip_fusion as adaptive


PROTOCOL = {
    "version": adaptive.VERSION, "folds": 5, "groups": "CH-SIMS video source",
    "expert_training": "use original v1 C and class balance. Exclude every caption/audio clip of a held source.",
    "calibration": "regularized multinomial regression on log-neutral/negative, log-positive/negative and its absolute magnitude",
    "C": [.3, 3.], "balance": [0., .5, 1.], "gate_l2": [.01, .1],
    "gate": "quality * exp(intercept + b1*top2_margin + b2*quality_weighted_agreement_with_other_experts)",
    "training_masks": {"observed": .5, "hide_audio": .25, "hide_visual": .25},
    "av_masks": {"hide_text": .75, "only_visual": .25},
    "validation_objective": ".75 observed + .125 hide_audio + .125 hide_visual; each (accuracy+2*macro-F1+balanced accuracy)/4",
    "av_objective": "same metric on hide_text; shared by both arms when text is absent",
    "ties": "observed positive recall (AV: hide_text), then original grid order",
}


def load_train_valid(data):
    manifest = frozen_cohort(data, "raw-cohort.json")
    rows = []
    with (data / "rows.jsonl").open(encoding="utf8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] in ("train", "valid"):
                rows.append(row)
    wanted = {r["id"]: hashlib.sha256(r["text"].encode()).hexdigest() for r in rows if r["split"] == "train"}
    actual_text, actual_audio = checksums_for_raw(data, manifest["train"] + manifest["valid"])
    raw_embeddings = embeddings(data / "embeddings-text-asr.npz", fusion.TEXT_DIM, expected=actual_text)
    acoustic = embeddings(data / "embeddings-audio.npz", fusion.AUDIO_DIM, expected=actual_audio)
    captions = dataset(data, rows, set(wanted), embeddings(data / "embeddings-text.npz", fusion.TEXT_DIM,
                       expected=wanted), {}, raw=False)
    raw = {}
    for split in ("train", "valid"):
        ids = set(manifest[split])
        if not ids <= {r["id"] for r in rows if r["split"] == split}:
            raise ValueError(f"Raw cohort crosses the official {split} split")
        raw[split] = dataset(data, rows, ids, raw_embeddings, acoustic, raw=True)
        if any(r["text"] and fusion.HAN.search(r["text"]) and r["embedding"] is None for r in raw[split]):
            raise ValueError(f"Missing ASR embedding in {split}")
    return captions, raw["train"], raw["valid"]


def sources(rows):
    return np.array([r["id"].split("/")[0] for r in rows])


def balanced_weights(y, exponent):
    n = np.bincount(y, minlength=3)
    if np.any(n == 0):
        raise ValueError("A training fold lacks one of the three classes")
    return (len(y) / (3 * n)) ** exponent


def source_held_out_experts(captions, train, record):
    """Build train features without fitting on the held clip or other clips from its source."""
    groups = sources(train)
    folds = list(GroupKFold(n_splits=PROTOCOL["folds"]).split(train, groups=groups))
    matrices = {}
    for key in ("base", "qwen", "audio"):
        rows = train if key == "audio" else captions
        x, valid = inputs(rows, key)
        matrices[key] = (x, target(rows)[valid], sources(rows)[valid])
    by_variant = {v: (np.zeros((len(train), 4, 3)), np.zeros((len(train), 4)))
                  for v in ("no_qwen", "foundation")}
    for _, held in folds:
        held_sources = np.unique(groups[held])
        model = {}
        for key, (x, y, source) in matrices.items():
            keep = ~np.isin(source, held_sources)
            meta = record["heads"][key]
            center = x[keep].mean(axis=0)
            shifted = x[keep] - center
            xx = shifted / np.maximum(np.linalg.norm(shifted, axis=1, keepdims=True), 1e-10)
            weights = balanced_weights(y[keep], meta["class_exponent"])
            clf = LogisticRegression(C=meta["C"], solver="lbfgs", class_weight=dict(enumerate(weights)),
                                     max_iter=3000, tol=1e-6).fit(xx, y[keep])
            if np.max(clf.n_iter_) == clf.max_iter:
                raise RuntimeError("Source-held-out expert did not converge")
            model[key + "_center"] = center
            model[key + "_coef"] = clf.coef_
            model[key + "_bias"] = clf.intercept_
        held_rows = [train[i] for i in held]
        for v, (values, quality) in by_variant.items():
            values[held], quality[held] = components(held_rows, model, v)
    return by_variant, folds


def fit_calibration(values, quality, y, C, balance, *, av_only):
    calibrations = []
    for j in range(4):
        if av_only and j == 0:
            calibrations.append(None)
            continue
        use = quality[:, j] > 0.
        x = adaptive.calibration_features(values[use, j])
        mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), .05)
        weight = balanced_weights(y[use], balance)
        lr = LogisticRegression(C=C, class_weight=dict(enumerate(weight)), solver="lbfgs",
                                max_iter=1000, tol=1e-8).fit((x - mean) / scale, y[use])
        if np.max(lr.n_iter_) == lr.max_iter:
            raise RuntimeError("Expert calibration did not converge")
        calibrations.append({k: a.tolist() for k, a in (
            ("mean", mean), ("scale", scale), ("coef", lr.coef_), ("intercept", lr.intercept_))})
    return calibrations


def masked(quality, scenario):
    tx, au, vi = SCENARIOS[scenario]
    return quality * np.array([tx, au, au, vi])


def fit_gate(probs, quality, y, balance, penalty, masks):
    # Available views receive the predeclared observed/missing weights for their clip.
    pieces = []
    class_weights = balanced_weights(y, balance)[y]
    for scenario, fraction in masks.items():
        q = masked(quality, scenario)
        use = q.any(axis=1)
        features = adaptive.gate_features(probs, q)
        true_scores = np.take_along_axis(probs, y[:, None, None], axis=2).squeeze(axis=2)
        pieces.append((features[use], q[use], true_scores[use], class_weights[use] * fraction))
    features, present, true_scores, importance = (np.concatenate(part) for part in zip(*pieces))
    importance = importance / importance.sum()

    def objective(coefficients):
        gate = coefficients.reshape(4, 3)
        weights = adaptive.gate_weights(features, present, gate)
        likelihood = np.maximum((weights * true_scores).sum(axis=1), 1e-12)
        loss = -np.sum(importance * np.log(likelihood)) + .5 * penalty * np.sum(gate ** 2)
        gradient = importance[:, None] * weights * (1. - true_scores / likelihood[:, None])
        grad = np.einsum("ne,nej->ej", gradient, features) + penalty * gate
        return loss, grad.ravel()

    result = minimize(objective, np.zeros(12), method="L-BFGS-B", jac=True,
                      bounds=[(-5., 5.)] * 12, options={"ftol": 1e-12, "gtol": 1e-7})
    if not result.success:
        raise RuntimeError(f"Reliability gate did not converge: {result.message}")
    return result.x.reshape(4, 3)


def evaluate(probs, q, gate, y, condition):
    quality = masked(q, condition)
    weights = adaptive.gate_weights(adaptive.gate_features(probs, quality), quality, gate)
    output = np.einsum("ne,nec->nc", weights, probs)
    predictions = np.where(weights.any(axis=1), output.argmax(axis=1), -1)
    m = metrics(y, predictions)
    cm = np.array(m["confusion_gold_by_prediction"])
    recall = np.diag(cm[:, :3]) / np.maximum(cm.sum(axis=1), 1)
    m["neutral_recall"] = float(recall[1])
    m["balanced_accuracy"] = float(recall.mean())
    return m


def selection_value(results, av_only):
    scenarios = {"hide_text": 1.} if av_only else {"observed": .75, "hide_audio": .125, "hide_visual": .125}
    val = 0.
    for name, weight in scenarios.items():
        observed = results[name]
        val += weight * (observed["accuracy"] + 2 * observed["macro_f1"] + observed["balanced_accuracy"]) / 4.
    return val, results["hide_text" if av_only else "observed"]["positive_recall"]


def select(train_blocks, y, folds, valid_blocks, valid_y, *, av_only):
    values, q = train_blocks
    val_values, val_q = valid_blocks
    if av_only:
        q, val_q = masked(q, "hide_text"), masked(val_q, "hide_text")
    masks = PROTOCOL["av_masks" if av_only else "training_masks"]
    best_rank, winner = (-1., -1.), None
    trials = []
    for C, balance in itertools.product(PROTOCOL["C"], PROTOCOL["balance"]):
        # Fit gate weights on calibrated scores unseen by that fold's calibrator.
        held_scores = np.zeros_like(values)
        for fit, held in folds:
            cal = fit_calibration(values[fit], q[fit], y[fit], C, balance, av_only=av_only)
            held_scores[held] = adaptive.calibrate(values[held], q[held], cal)
        cal = fit_calibration(values, q, y, C, balance, av_only=av_only)
        valid_scores = adaptive.calibrate(val_values, val_q, cal)
        for penalty in PROTOCOL["gate_l2"]:
            gate = fit_gate(held_scores, q, y, balance, penalty, masks)
            result = {scenario: evaluate(valid_scores, val_q, gate, valid_y, scenario) for scenario in masks}
            rank = selection_value(result, av_only)
            config = {"C": C, "balance": balance, "gate_l2": penalty}
            trials.append({**config, "objective": rank[0], "positive_recall_tiebreak": rank[1]})
            if rank > best_rank:
                best_rank = rank
                winner = ({"calibration": cal, "gate": gate.tolist()}, {"config": config, "validation": result})
    parameters, detail = winner
    return parameters, {**detail, "grid": trials}


def existing_diagnosis(blocks, rows, head, variant):
    values, quality = blocks
    y = target(rows)
    neutral = (y == 1)
    w = head[variant + "_weights"] * quality
    z = (values * w[:, :, None]).sum(axis=1)
    av = (values[:, 1:] * w[:, 1:, None]).sum(axis=1)
    conflict = (w[:, 0] > 0) & (w[:, 1:].sum(axis=1) > 0) & (values[:, 0].argmax(axis=1) != av.argmax(axis=1))
    def tally(prediction, use):
        return dict(sorted(Counter(fusion.LABELS[p] for p in prediction[use]).items()))
    return {"neutral_n": int(neutral.sum()), "neutral_predictions": tally(z.argmax(axis=1), neutral & w.any(axis=1)),
            "neutral_expert_top": {key: tally(values[:, j].argmax(axis=1), neutral & (w[:, j] > 0))
                                   for j, key in enumerate(fusion.EXPERTS)},
            "text_av_conflict": int(sum(conflict)), "text_correct_in_conflict": int(sum(conflict & (values[:,0].argmax(axis=1) == y))),
            "av_correct_in_conflict": int(sum(conflict & (av.argmax(axis=1) == y)))}


def main(data, output, record):
    expert_record = read_json(data / "fit.json")
    if file_sha(fusion.ASSET) != expert_record["model_sha256"] or expert_record["model_sha256"] != adaptive.HEAD_SHA256:
        raise ValueError("Unexpected expert model")
    with np.load(fusion.ASSET, allow_pickle=False) as archive:
        head = dict(archive)
    captions, train, valid = load_train_valid(data)
    valid_blocks = {v: components(valid, head, v) for v in ("foundation", "no_qwen")}
    diagnosis = {"validation": {v: existing_diagnosis(b, valid, head, v) for v, b in valid_blocks.items()}}
    print("v1 validation:", json.dumps(diagnosis["validation"]), flush=True)
    oof, folds = source_held_out_experts(captions, train, expert_record)
    diagnosis["source_held_out_training"] = {v: existing_diagnosis(b, train, head, v) for v, b in oof.items()}
    policies, details = {}, {}
    for regime in ("no_qwen", "foundation", "av"):
        view = "no_qwen" if regime == "av" else regime
        policies[regime], details[regime] = select(oof[view], target(train), folds, valid_blocks[view],
                                                  target(valid), av_only=regime == "av")
        print(regime, json.dumps(details[regime]["config"]), flush=True)
    # Diagnostic only. These switches do not participate in model selection.
    ablation = {}
    for variant in ("foundation", "no_qwen"):
        values, quality = valid_blocks[variant]
        policy = policies[variant]
        scores = adaptive.calibrate(values, quality, policy["calibration"])
        old_gate = np.zeros((4, 3))
        old_gate[:, 0] = np.log(head[variant + "_weights"])
        constant_gate = np.array(policy["gate"])
        constant_gate[:, 1:] = 0.
        ablation[variant] = {
            "raw_old_weights": evaluate(values, quality, old_gate, target(valid), "observed"),
            "calibrated_old_weights": evaluate(scores, quality, old_gate, target(valid), "observed"),
            "calibrated_new_intercepts_only": evaluate(scores, quality, constant_gate, target(valid), "observed"),
            "calibrated_new_gate": details[variant]["validation"]["observed"],
        }
    params = {"version": adaptive.VERSION, "labels": list(fusion.LABELS),
              "head_sha256": adaptive.HEAD_SHA256, "policies": policies}
    adaptive.validate(params)
    write_json(output, params)
    write_json(record, {"protocol": PROTOCOL, "model_sha256": file_sha(output), "head_sha256": adaptive.HEAD_SHA256,
                        "train_captions": len(captions), "train_raw": len(train), "valid_raw": len(valid),
                        "folds": [{"sources": np.unique(sources(train)[held]).tolist(), "n": len(held)}
                                  for _, held in folds],
                        "diagnosis_v1": diagnosis, "selection": details, "validation_ablation": ablation})
    print("Frozen", file_sha(output), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=adaptive.ASSET)
    parser.add_argument("--record", type=Path, default=DATA / "adaptive-fit.json")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        main(args.data, args.output, args.record)
