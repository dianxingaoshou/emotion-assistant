"""Fit and freeze a score-only calibration before evaluating the CH-SIMS test set."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_chsims_teacher import paired_metrics, save_json, sha_file
from scripts.chsims_teacher_data import CACHE, LABELS, VIEWS


def read_report(path, split):
    report = json.loads(path.read_text("utf8"))
    if report["split"] != split or len(report["rows"]) != report["results"]["full"]["n"]:
        raise ValueError(f"Unexpected {split} evaluation file")
    return report


def log_ratios(prediction):
    scores = prediction["scores"]
    if not scores:
        return None
    negative, neutral, positive = np.log(np.maximum(1e-12, [scores[k] for k in LABELS]))
    return np.array((negative - neutral, positive - neutral))


def calibrated(prediction, parameters):
    ratios = log_ratios(prediction)
    if ratios is None:
        return prediction
    logits = (ratios - parameters["center"]) / parameters["scale"] @ parameters["coef"].T + parameters["bias"]
    scores = np.exp(logits - logits.max())
    scores /= scores.sum()
    return {"label": LABELS[int(scores.argmax())], "scores": dict(zip(LABELS, scores.tolist())),
            "seen": prediction["seen"]}


def rows_for_view(rows, view, parameters, arm="teacher"):
    return [(row["id"], row["truth"], row["baseline"], {view: calibrated(row[arm][view], parameters)})
            for row in rows]


def read_parameters(record):
    return {name: np.asarray(record[name]) for name in ("center", "scale", "coef", "bias")}


def fit(cache, output, arm):
    training, validation = (read_report(cache / f"evaluation-{split}.json", split) for split in ("train", "valid"))
    if training["models"] != validation["models"] or training["meta_sha256"] != validation["meta_sha256"]:
        raise ValueError("Training and validation must share the dataset and frozen models")
    if {row["id"] for row in training["rows"]} & {row["id"] for row in validation["rows"]}:
        raise ValueError("Training and validation IDs overlap")
    models = {}
    for view in VIEWS:
        inputs = [(x, row["truth"][view]) for row in training["rows"]
                  if (x := log_ratios(row[arm][view])) is not None]
        x, y = zip(*inputs, strict=True)
        x = np.vstack(x)
        center, scale = x.mean(axis=0), np.maximum(x.std(axis=0), .05)
        candidates = []
        for reg in (.03, .1, .3, 1., 3.):
            for balance in (None, "balanced"):
                model = LogisticRegression(C=reg, class_weight=balance, max_iter=1000).fit((x - center) / scale, y)
                if tuple(model.classes_) != LABELS:
                    raise ValueError(f"Incomplete training labels for {view}")
                params = {"center": center, "scale": scale, "coef": model.coef_, "bias": model.intercept_}
                measured = paired_metrics(rows_for_view(validation["rows"], view, params, arm), view)["teacher"]
                rank = (measured["macro_f1"], measured["accuracy"], -reg, balance is None)
                candidates.append((rank, dict(C=reg, class_weight=balance, fitted=len(y),
                                              validation=measured, **{k: v.tolist() for k, v in params.items()})))
        models[view] = max(candidates, key=lambda item: item[0])[1]
    save_json(output, {"protocol": f"two {arm} log odds, multinomial logistic regression, train fit; "
                                    "validation selects Macro-F1 then accuracy, then stronger regularization",
                       "meta_sha256": training["meta_sha256"], arm: training["models"][arm],
                       "train_ids_sha256": training["selected_ids_sha256"],
                       "valid_ids_sha256": validation["selected_ids_sha256"],
                       "splits": {"train": len(training["rows"]), "valid": len(validation["rows"])},
                       "models": models})
    print("Frozen calibration", sha_file(output), flush=True)
    for view, model in models.items():
        print(view, model["validation"]["correct"], round(model["validation"]["macro_f1"], 4), flush=True)


def evaluate(cache, model_path, output, arm):
    calibration = json.loads(model_path.read_text("utf8"))
    test = read_report(cache / "evaluation-test.json", "test")
    if calibration[arm] != test["models"][arm] or calibration["meta_sha256"] != test["meta_sha256"]:
        raise ValueError("Incompatible test data or teacher checkpoint")
    result = {"calibration_sha256": sha_file(model_path), "raw_test_sha256": sha_file(cache / "evaluation-test.json"),
              "calibration": calibration, "results": {}}
    for view in VIEWS:
        params = read_parameters(calibration["models"][view])
        result["results"][view] = paired_metrics(rows_for_view(test["rows"], view, params, arm), view)
        stage = result["results"][view]["teacher"]
        print(view, stage["correct"], round(stage["macro_f1"], 4), flush=True)
    save_json(output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("fit", "evaluate"))
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--arm", choices=("teacher", "baseline"), default="teacher")
    args = parser.parse_args()
    model = args.cache / f"{args.arm}-calibration.json"
    if args.step == "fit":
        fit(args.cache, model, args.arm)
    else:
        evaluate(args.cache, model, args.cache / f"{args.arm}-calibrated-test.json", args.arm)
