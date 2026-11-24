"""Fit single-channel corrections on fitting actors, select on policy actors, evaluate on held-out actors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bench_real_ravdess_data import specimens, split_for
from emotion_assistant.calibration import LABELS, MAPPING_VERSION, TARGET, fingerprint
from emotion_assistant.fusion_policy import FusionConfig, FusionPolicy
from emotion_assistant.models import VERSIONS
from emotion_assistant.single_cue import SingleCueCalibration, log_features


def load_corpus(directory):
    features = [
        json.loads(line)
        for line in (directory / "features.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    labels = json.loads((directory / "rater-labels.json").read_text("utf-8"))
    votes = json.loads((directory / "rater-votes.json").read_text("utf-8"))
    expected = {s for actor in range(1, 25) for s in specimens(actor)}
    if (
        len(features) != len(expected)
        or {r["id"] for r in features} != expected
        or len({r["sha256"] for r in features}) != len(expected)
    ):
        raise ValueError("expected the original 192 unique RAVDESS recordings")
    rows = {"fit": [], "policy": [], "test": []}
    for row in sorted(features, key=lambda r: r["id"]):
        if row["protocol"] != "ravdess-half-second-clip-v1":
            raise ValueError("unexpected extraction protocol")
        label = labels.get(row["id"])
        if label != votes[row["id"]]["label"]:
            raise ValueError("label and vote table disagree")
        if label is None:
            continue  # independent audiovisual rater plurality is tied
        label = "uncertain" if label == "unmapped" else label
        context = {}
        for key, b in row["modalities"].items():
            if b["model_version"] != VERSIONS[key]:
                raise ValueError("model version mismatch")
            context[key] = {
                "scores": b["scores"],
                "versions": [b["model_version"]],
                "quality": b["quality"],
                "coverage": b["coverage"],
                "age": b["age_seconds"],
                "duration": b.get("speech_seconds", 0),
            }
        actor = int(row["id"][-2:])
        rows[split_for(actor)].append((row["id"], actor, label, context))
    return rows


def metrics(probabilities, labels):
    p, y = np.asarray(probabilities), np.asarray(labels, dtype=int)
    top = p.argmax(axis=1)
    confusion = np.bincount(y * 4 + top, minlength=16).reshape(4, 4)
    den = confusion.sum(axis=0) + confusion.sum(axis=1)
    return {
        "accuracy": float((top == y).mean()),
        "macro_f1": float(np.divide(2 * confusion.diagonal(), den, where=den > 0, out=np.zeros(4)).mean()),
        "brier": float(np.square(p - np.eye(4)[y]).sum(axis=1).mean()),
    }


def fit_matrix(design, targets, ridge):
    # Identity is the previous class mapping; finite ridge shrinks the fit toward it.
    identity = np.column_stack([np.eye(4), np.zeros(4)])

    def objective(flat):
        w = flat.reshape(identity.shape)
        logits = design @ w.T
        norm = logsumexp(logits, axis=1)
        delta = w - identity
        loss = np.sum(norm - logits[np.arange(len(targets)), targets]) + 0.5 * ridge * np.sum(delta**2)
        residual = np.exp(logits - norm[:, None]) - np.eye(4)[targets]
        return float(loss), (residual.T @ design + ridge * delta).ravel()

    sol = minimize(objective, identity.ravel(), method="L-BFGS-B", jac=True)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.x.reshape(identity.shape)


def make_artifact(rows):
    chosen, selections = {}, {}
    for key in ("visual", "audio"):
        fit = [(ctx[key], LABELS.index(label)) for _, _, label, ctx in rows["fit"] if key in ctx]
        policy = [(ctx[key], LABELS.index(label)) for _, _, label, ctx in rows["policy"] if key in ctx]
        p = np.array([[b["scores"][k] for k in LABELS] for b, _ in policy])
        labels = np.array([y for _, y in policy])
        reference = metrics(p, labels)
        best = reference
        best_fit = None
        X = np.array([log_features(b["scores"]) for b, _ in fit])
        y = np.array([y for _, y in fit])
        validation = np.array([log_features(b["scores"]) for b, _ in policy])
        for ridge in (1.0, 4.0, 16.0):
            weights = fit_matrix(X, y, ridge)
            logits = validation @ weights.T
            out = np.exp(logits - logsumexp(logits, axis=1)[:, None])
            for blend in (0.5, 0.75, 1.0):
                scores = metrics((1 - blend) * p + blend * out, labels)
                if (
                    scores["brier"] <= reference["brier"] - 0.005
                    and scores["brier"] < best["brier"]
                    and scores["accuracy"] >= reference["accuracy"]
                    and scores["macro_f1"] >= reference["macro_f1"]
                ):
                    best = scores
                    best_fit = {
                        "coefficients": weights.round(10).tolist(),
                        "blend": blend,
                        "ridge": ridge,
                        "model_version": VERSIONS[key],
                        "n_fit": len(fit),
                        "n_policy": len(policy),
                    }
        if best_fit is not None:
            chosen[key] = best_fit
        selections[key] = {
            "eligible_fit": len(fit),
            "eligible_policy": len(policy),
            "base": reference,
            "selected": best,
            "adapted": best_fit is not None,
        }
    doc = {
        "schema": "single-cue-log-probability-v1",
        "target": TARGET,
        "mapping_version": MAPPING_VERSION,
        "label_order": list(LABELS),
        "training": "RAVDESS/Zenodo 1188976; Livingstone & Russo 2018; CC BY-NC-SA 4.0",
        "labels": "unique plurality of independent full audiovisual clip ratings (PLOS S5), including after input masking",
        "fit_actors": sorted({actor for _, actor, _, _ in rows["fit"]}),
        "policy_actors": sorted({actor for _, actor, _, _ in rows["policy"]}),
        "branches": chosen,
    }
    return {**doc, "version": fingerprint(doc)}, selections


def scenarios(context):
    def attenuate(key, attr, ratio):
        changed = {**context, key: {**context[key], attr: context[key][attr] * ratio}}
        return {k: b for k, b in changed.items() if b["quality"] * b["coverage"] >= 0.2}

    return {
        "both": context,
        "visual_missing": {"audio": context["audio"]},
        "audio_missing": {"visual": context["visual"]},
        "visual_coverage_halved": attenuate("visual", "coverage", 0.5),
        "acoustic_quality_quartered": attenuate("audio", "quality", 0.25),
        "both_missing": {},
    }


def compare(cases, before, after):
    rows = [
        (actor, LABELS.index(label), before._evaluate(ctx), after._evaluate(ctx))
        for actor, label, ctx in cases
    ]
    valid = [r for r in rows if r[2].get("scores") is not None and r[3].get("scores") is not None]
    out = {"n": len(rows), "scored": len(valid)}
    if not valid:
        out["both_abstained"] = sum(
            old["decision"] == new["decision"] == "abstain" for _, _, old, new in rows
        )
        return out
    labels = np.array([r[1] for r in valid])
    probs = [np.array([[r[j]["scores"][k] for k in LABELS] for r in valid]) for j in (2, 3)]
    for j, (policy, p) in enumerate(zip(("before", "after"), probs), start=2):
        accepted = np.array([r[j]["decision"] != "abstain" for r in valid])
        out[policy] = {
            **metrics(p, labels),
            "accepted": int(accepted.sum()),
            "accepted_errors": int((accepted & (p.argmax(axis=1) != labels)).sum()),
        }
    out["adapted"] = sum(r[3]["missing_modality_calibration"] == "applied" for r in valid)
    # Paired resampling at actor level; masks of the same clip never become independent subjects.
    brier_change = (np.square(probs[1] - np.eye(4)[labels]) - np.square(probs[0] - np.eye(4)[labels])).sum(
        axis=1
    )
    actors = np.array([r[0] for r in valid])
    pools = [np.flatnonzero(actors == actor) for actor in sorted(set(actors))]
    rng = np.random.default_rng(20261123)
    draws = [
        float(brier_change[np.concatenate([pools[i] for i in indices])].mean())
        for indices in rng.integers(len(pools), size=(2000, len(pools)))
    ]
    out["paired_brier_change_ci95_actor"] = np.percentile(draws, [2.5, 97.5]).tolist()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, default=ROOT / "reports/ravdess")
    p.add_argument("--report", type=Path, default=ROOT / "docs/results/missing-modality-benchmark.json")
    p.add_argument("--write-artifact", type=Path, help="write newly fitted single-cue calibration")
    args = p.parse_args()
    rows = load_corpus(args.dir)
    doc, validation = make_artifact(rows)
    if args.write_artifact:
        args.write_artifact.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n", "utf-8")
    fitted = SingleCueCalibration(doc)
    before = FusionPolicy(FusionConfig())
    after = FusionPolicy(FusionConfig(adapt_single_modality=True), single_cue_model=fitted)
    cases = {}
    for _, actor, label, context in rows["test"]:
        if len(context) != 2:
            continue
        for name, masked in scenarios(context).items():
            cases.setdefault(name, []).append((actor, label, masked))
    report = {
        "protocol": "ravdess-conditioned-dropout-v1",
        "dataset": "10.5281/zenodo.1188976",
        "human_labels": "10.1371/journal.pone.0196391.s008",
        "unit": "Paired held-out clips with two originally eligible modes and a unique audiovisual rater plurality. No scripted emotion label used.",
        "masking": "Deterministic masking and reduced coverage/quality on frozen clip features AFTER inference. No raw noise/occlusion synthesis. No text; not a field or clinical evaluation.",
        "selection": "fit 4x5 ridge multinomial map vs identity on fitting actors. On policy actors minimize Brier, require >=0.005 Brier improvement and no loss in accuracy/macro-F1, otherwise retain identity.",
        "actor_splits": {k: sorted({a for _, a, _, _ in v}) for k, v in rows.items()},
        "counts_after_excluding_rater_ties": {k: len(v) for k, v in rows.items()},
        "input_sha256": {
            n: hashlib.sha256((args.dir / n).read_bytes()).hexdigest()
            for n in ("features.jsonl", "rater-labels.json")
        },
        "artifact": doc["version"],
        "validation": validation,
        "scenarios": {k: compare(v, before, after) for k, v in cases.items()},
    }
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", "utf-8")
    print(json.dumps(report["scenarios"], indent=2))


if __name__ == "__main__":
    main()
