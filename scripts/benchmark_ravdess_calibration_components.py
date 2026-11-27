"""Replay the frozen RAVDESS clip scores with temperature and gating independently enabled.

First run bench_real_ravdess_eval.py to generate the paired held-out predictions.
The four comparisons reuse the fitted parameters and score every eligible paired clip.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from bench_real_ravdess_eval import LABELS, ROOT, stats


def evaluate_components(predictions, reference):
    baseline, calibrated = "quality_v2", "calibrated_fusion"
    variants = {
        baseline: (baseline, baseline),
        "temperature_only": (calibrated, baseline),
        "gate_only": (baseline, calibrated),
        calibrated: (calibrated, calibrated),
    }
    primary = reference["primary_common_clips"]
    rows = [row for row in predictions if row["label"] and row["outputs"][baseline]["status"] == "joint"]
    if not rows or len(rows) != primary["n"] or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("paired valid clips do not match the reference cohort")
    if any(row["outputs"][calibrated]["calibration_status"] != "fitted"
           or row["outputs"][calibrated]["calibration_version"] != reference["calibration_version"]
           for row in rows):
        raise ValueError("expected the frozen full calibration on each clip")

    targets = np.array([LABELS.index(row["label"]) for row in rows])
    methods = {}
    for name, (distribution_source, weight_source) in variants.items():
        scores = []
        for row in rows:
            outputs = row["outputs"]
            distributions = outputs[distribution_source]["per_modality_output_scores"]
            weights = outputs[weight_source]["weights"]
            # Temperatures transform each branch's four-class distribution. The gate
            # uses pre-temperature features, so the weight vector can be toggled independently.
            probs = [sum(weights[k] * distributions[k][label] for k in ("visual", "audio"))
                     for label in LABELS]
            if name in (baseline, calibrated):
                recorded = [outputs[name]["scores"][label] for label in LABELS]
                if not np.allclose(probs, recorded, rtol=0, atol=1e-12):
                    raise ValueError(f"replayed {name} does not match archived scores for {row['id']}")
            scores.append(probs)
        values = np.array(scores)
        correct = int((values.argmax(axis=1) == targets).sum())
        f1, brier, _ = stats(values, targets, np.arange(len(rows)))
        methods[name] = {"correct": correct, "accuracy": correct / len(rows),
                         "macro_f1": float(f1), "brier": float(brier)}
        if name in (baseline, calibrated):
            old = primary["methods"][name]
            if (correct != np.trace(old["confusion_true_by_predicted"])
                    or not np.allclose([f1, brier], [old["macro_f1"], old["brier"]], rtol=0, atol=1e-12)):
                raise ValueError(f"{name} metrics differ from the published reference")
    return {
        "protocol": "ravdess-paired-temperature-gate-replay-v1",
        "reference": "ravdess-recognition.json",
        "selected_media_sha256": reference["selected_media_sha256"],
        "calibration_version": reference["calibration_version"],
        "cohort": "Same unique audiovisual rater label and valid visual/audio pair as primary_common_clips.",
        "n": len(rows), "actors": len({row["actor"] for row in rows}),
        "scoring": "Top class on every clip, before acceptance thresholds. Lower multiclass Brier is better.",
        "design": "Cross the saved raw/temperature-scaled branch distributions with the saved quality/gated weights. No refitting or threshold selection.",
        "methods": methods,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "reports/ravdess/heldout-predictions.json")
    parser.add_argument("--reference", type=Path, default=ROOT / "docs/results/ravdess-recognition.json")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/ravdess/calibration-ablation.json")
    args = parser.parse_args()
    if args.out.resolve() in {args.predictions.resolve(), args.reference.resolve()}:
        parser.error("--out must be separate from the input predictions and reference")
    result = evaluate_components(json.loads(args.predictions.read_text(encoding="utf-8")),
                                 json.loads(args.reference.read_text(encoding="utf-8")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["methods"], indent=2))
