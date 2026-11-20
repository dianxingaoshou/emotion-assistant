"""Actor-disjoint, paired, four-policy evaluation of real RAVDESS speech clips."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path

import numpy as np

from bench_real_ravdess_data import LABEL, ROOT, specimens, split_for
from calibrate_fusion import Calibration, FusionConfig, FusionPolicy, LABELS, fit_branch, fit_groups
from emotion_assistant.models import VERSIONS

POLICIES = ("visual_only", "audio_only", "quality_v2", "calibrated_fusion")
SEED = 20261120


def normalized(label):
    return "uncertain" if label == "unmapped" else label


def context(features):
    return {k: {"scores": dict(v["scores"]), "quality": v["quality"], "coverage": v["coverage"],
                "versions": [v["model_version"]], "age": v["age_seconds"],
                **({"duration": v["speech_seconds"]} if k == "audio" else {})}
            for k, v in features.items()}


def stats(p, y, tie):
    """All methods have exactly the same labelled clips; no abstention counts as correct."""
    guess = np.argmax(p, axis=1)
    confusion = np.bincount(y*4 + guess, minlength=16).reshape(4, 4)
    denom = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(2*confusion.diagonal(), denom, out=np.zeros(4, dtype=float), where=denom>0)
    brier = np.square(p - np.eye(4)[y]).sum(axis=1).mean()
    # Fixed 70% score-ranked operating point, independent of deployment abstention.
    # Hash tie-breaker does not use the source emotion encoded in the file name.
    k = max(1, math.floor(.70*len(y)))
    by_confidence = np.lexsort((tie, -p.max(axis=1)))
    selected = by_confidence[:k]
    return np.array([f1.mean(), brier, np.mean(guess[selected] != y[selected])])


def score_methods(rows):
    if not rows:
        return {"n": 0}
    y = np.array([LABELS.index(r["label"]) for r in rows])
    tie = np.array([int(hashlib.sha256(("tie-v1:"+r["id"]).encode()).hexdigest()[:14],16) for r in rows])
    arr = np.array([[[r["outputs"][name]["scores"][label] for label in LABELS] for r in rows]
                    for name in POLICIES])
    methods = [stats(p,y,tie) for p in arr]
    actors = np.array([r["actor"] for r in rows])
    groups = sorted(set(actors))
    pools = [np.flatnonzero(actors == a) for a in groups]
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(2000):
        ix = np.concatenate([pools[j] for j in rng.integers(0, len(groups), len(groups))])
        draws.append([stats(p[ix], y[ix], tie[ix]) for p in arr])
    samples = np.asarray(draws)
    names = ("macro_f1", "brier", "risk_at_70pct")
    def intervals(draws):
        ci = np.percentile(draws,[2.5,97.5],axis=0)
        return {n: list(ci[:,i]) for i,n in enumerate(names)}
    out = {}
    for i, method in enumerate(POLICIES):
        pred = arr[i].argmax(axis=1)
        accepted = np.array([r["outputs"][method]["decision"] != "abstain" for r in rows])
        out[method] = {**dict(zip(names, methods[i])), "ci95_actor": intervals(samples[:,i,:]),
                       "confusion_true_by_predicted": np.bincount(y*4+pred,minlength=16).reshape(4,4).tolist(),
                       "accepted_n": int(accepted.sum()), "accepted_errors": int(sum(accepted & (pred != y))),
                       "fallbacks": sum(r["outputs"][method]["calibration_status"] != "fitted"
                                        for r in rows) if method == "calibrated_fusion" else None}
    # Secondary operational comparison: compare errors at the same number of
    # decisions, selecting only actually accepted clips. This is descriptive,
    # not a new deployable threshold chosen on test subjects.
    matched_k = min(v["accepted_n"] for v in out.values())
    for i,method in enumerate(POLICIES):
        eligible = np.flatnonzero([r["outputs"][method]["decision"] != "abstain" for r in rows])
        ranked = eligible[np.lexsort((tie[eligible], -arr[i,eligible].max(axis=1)))][:matched_k]
        out[method]["matched_accepted_errors"] = int((arr[i,ranked].argmax(axis=1) != y[ranked]).sum()) if matched_k else None
    return {"n":len(rows), "actors":len(groups), "support":dict(Counter(r["label"] for r in rows)),
            "top70_n":max(1,math.floor(.70*len(rows))), "matched_accepted_n":matched_k, "methods":out,
            "paired_calibrated_minus_quality": dict(zip(names, methods[3] - methods[2])),
            "paired_calibrated_minus_quality_ci95_actor": intervals(samples[:,3,:] - samples[:,2,:]),
            "paired_calibrated_minus_each": {
                method:{"difference":dict(zip(names,methods[3] - methods[i])),
                        "ci95_actor":intervals(samples[:,3,:] - samples[:,i,:])}
                for i,method in enumerate(POLICIES[:3])}}


def evaluate(features, labels, votes):
    ids = [n for a in range(1,25) for n in specimens(a)]
    if len(features) != len(ids) or {r["id"] for r in features} != set(ids):
        raise ValueError("expected exactly the complete fixed 192-clip subset")
    if any(r["protocol"] != "ravdess-half-second-clip-v1" for r in features):
        raise ValueError("feature protocol mismatch")
    if len({r["sha256"] for r in features}) != len(ids):
        raise ValueError("duplicate media across fixed corpus")
    if any(v["model_version"] != VERSIONS[k] for r in features for k,v in r["modalities"].items()):
        raise ValueError("feature model version mismatch")
    if any((name not in votes or labels.get(name) != votes[name]["label"]) for name in ids):
        raise ValueError("ratings missing, or vote/label files inconsistent")
    counts, fitted_rows, prepared = {}, {"fit":[], "policy":[]}, []
    for row in features:
        name = row["id"]
        actor = int(name[-2:])
        split = split_for(actor)
        c = counts.setdefault(split, {"actors":[], "selected":0, "tied":0, "unique_below_60pct":0,
                                      "no_evidence":0, "both_modalities":0, "visual_eligible":0, "audio_eligible":0})
        if actor not in c["actors"]:
            c["actors"].append(actor)
        c["selected"] += 1
        branches = context(row["modalities"])
        for k in branches:
            c[k+"_eligible"] += 1
        c["both_modalities"] += int(len(branches) == 2)
        c["no_evidence"] += int(not branches)
        label = normalized(labels.get(name))
        if not label:
            c["tied"] += 1
        else:
            c["unique_below_60pct"] += int(votes[name]["winning_votes"] / sum(votes[name]["votes"].values()) < .6)
        if split in fitted_rows and branches and label:
            fitted_rows[split].append((branches,label))
        if split == "test":
            prepared.append((name,actor,branches,label,normalized(LABEL[name.split("-")[2]])))
    fits = {k:fit_branch(fitted_rows["fit"],k) for k in ("visual","audio")}
    if not all(fits.values()):
        raise ValueError("not enough real labelled fitting data; do not report an unfitted calibration")
    artifact = fit_groups(fits, fitted_rows["policy"], .25)
    policy = FusionPolicy(FusionConfig())
    calibrated = FusionPolicy(FusionConfig(strategy="calibrated_fusion"), Calibration.from_document(artifact))
    if calibrated.calibration.status != "fitted":
        raise ValueError("calibration failed to load")
    recordings, primary, intended = [], [], []
    for name, actor, branches, label, script in prepared:
        outputs = {k+"_only": policy.evaluate({k:branches[k]} if k in branches else {}) for k in ("visual","audio")}
        outputs.update(quality_v2=policy.evaluate(branches),calibrated_fusion=calibrated.evaluate(branches))
        row = {"id":name, "actor":actor, "label":label, "intended":script, "outputs":outputs}
        recordings.append(row)
        # Four methods compared on the identical population with both inputs.
        if len(branches) == 2:
            intended.append({**row,"label":script})
            if label:
                primary.append(row)
    available = {}
    for method in POLICIES:
        unique = [r for r in recordings if r["label"]]
        usable = [r for r in unique if r["outputs"][method].get("scores")]
        accepted = [r for r in usable if r["outputs"][method]["decision"] != "abstain"]
        def predicted(r):
            s = r["outputs"][method]["scores"]
            return max(LABELS,key=s.get)
        available[method] = {"all_unique_test_labels":len(unique), "scored":len(usable), "accepted":len(accepted),
                             "accepted_errors": sum(predicted(r) != r["label"] for r in accepted)}
    report = {"protocol":"ravdess-half-second-clip-v1", "dataset_doi":"10.5281/zenodo.1188976",
              "ratings_doi":"10.1371/journal.pone.0196391.s008",
              "unit":"one paired audio-video release clip; no live device or VAD segmentation",
              "primary_label":"unique plurality of collapsed audiovisual human rater votes; ties excluded",
              "sensitivity_label":"actor's intended expression, using the unchanged rater-trained model",
              "scope":"actor disjoint for this project's fit, policy and test; upstream model training overlap unknown",
              "bootstrap":{"unit":"test actor, paired with replacement", "replicates":2000, "seed":SEED},
              "model_versions":{k:VERSIONS[k] for k in ("face","visual","audio")},
              "selected_media_sha256":hashlib.sha256(
                  "\n".join(f"{r['id']} {r['sha256']}" for r in sorted(features,key=lambda v:v["id"]))
                  .encode("ascii")).hexdigest(),
              "rating_workbook_sha256":"b304b75a1a1a8928be557b0fe6a248c40db193e95e4d5b02073dce8dd034fcee",
              "scoring_versions":{k:version(k) for k in ("numpy","scipy")},
              "class_order":list(LABELS), "splits":counts, "calibration_version":artifact["version"],
              "fitting_counts":{k:len(v) for k,v in fitted_rows.items()},
              "primary_common_clips":score_methods(primary), "intended_common_clips":score_methods(intended),
              "all_primary_test_clips":available}
    return report, recordings, artifact


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, default=ROOT / "reports/ravdess")
    args = p.parse_args()
    read = lambda fn: (args.dir/fn).read_text(encoding="utf8")
    rows = [json.loads(line) for line in read("features.jsonl").splitlines() if line.strip()]
    report, predictions, artifact = evaluate(rows,json.loads(read("rater-labels.json")),json.loads(read("rater-votes.json")))
    for name, data in [("recognition-results.json",report),("heldout-predictions.json",predictions),
                       ("research-calibration.json",artifact)]:
        (args.dir/name).write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps(report["primary_common_clips"],ensure_ascii=True,indent=2))


if __name__ == "__main__":
    main()
