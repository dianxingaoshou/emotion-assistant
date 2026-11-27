"""Fit two source-specific monotone affine calibrators, then evaluate on untouched CVAS sentences."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
from http.client import HTTPConnection
import inspect
import io
import json
from pathlib import Path
from random import Random
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emotion_assistant import chinese_affect as rater
from scripts.evaluate_chinese_affect import DIGEST, URL, errors

VERSION = "cvas-zh-affine-v1"
CALIBRATION_IDS = "40f28eeee70d75a8f0e7bd225b6a483dab23f7b88b290e1cf4e418df6c4d723a"
TEST_IDS = "0663e627be2bbc59a0f054c11e6a31d0657e289cd222db7cb1cf8f8b08a54284"
MODEL_DIGEST = "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepared(path):
    raw = path.read_bytes()
    if digest(raw) != DIGEST:
        raise ValueError("Unexpected official CVAS source checksum")
    rows = []
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter="\t"):
        text = row["Text"].strip()
        rows.append({"text": text, "id": digest(("zh-va-v1|" + text).encode()),
                     "valence": float(row["Valence_Mean"]), "arousal": float(row["Arousal_Mean"])})
    rows.sort(key=lambda r: r["id"])
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate official sentences")
    for part, expected in ((rows[96:288], CALIBRATION_IDS), (rows[288:448], TEST_IDS)):
        if digest("\n".join(r["id"] for r in part).encode()) != expected:
            raise ValueError("Evaluation cohort differs from the frozen selection")
    return rows


def verify_local_model():
    conn = HTTPConnection("127.0.0.1", 11434, timeout=10)
    try:
        conn.request("GET", "/api/tags")
        tags = json.loads(conn.getresponse().read())
    finally:
        conn.close()
    if not any(m["name"] == rater.MODEL and m["digest"] == MODEL_DIGEST for m in tags["models"]):
        raise ValueError("Local model digest differs from the frozen Qwen3 8B")


def frozen_backend(offline):
    if not offline:
        verify_local_model()
    source = "\n".join(inspect.getsource(f) for f in (rater.rate_chinese_expression, rater.validate_ratings))
    return digest((rater.PROMPT + json.dumps(rater.SCHEMA, sort_keys=True) + source).encode())


def infer(rows, path, fingerprint, workers, offline):
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = {"model_digest": MODEL_DIGEST, "rater": fingerprint, "dataset_sha256": DIGEST}
    cached = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["identity"] != identity:
                raise ValueError("Use a fresh cache for a different model or rater")
            cached[record["id"]] = record["rating"]
    pending = [r for r in rows if r["id"] not in cached]
    if offline and pending:
        raise ValueError("Requested ratings absent from the local cache")
    if not pending:
        return [dict(row, result=cached[row["id"]]) for row in rows]

    def query(row):
        result = rater.rate_chinese_expression(row["text"])
        if result["status"] == "service_error":
            raise RuntimeError("Local model unavailable; retry with the same validated cache")
        return row["id"], {k: result[k] for k in ("status", *rater.AXES) if k in result}

    with path.open("a", encoding="utf8") as out, ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (key, rating) in enumerate(pool.map(query, pending), 1):
            out.write(json.dumps({"identity": identity, "id": key, "rating": rating}) + "\n")
            out.flush()
            cached[key] = rating
            if index % 48 == 0:
                print(f"{index}/{len(pending)} ratings obtained", flush=True)
    return [dict(row, result=cached[row["id"]]) for row in rows]


def fit(calibration):
    valid = [r for r in calibration if r["result"]["status"] == "ok"]
    if len(valid) < 2:
        raise ValueError("Too few valid calibration outputs")
    fitted = {}
    for axis in rater.AXES:
        x_mean = mean(r["result"][axis] for r in valid)
        y_mean = mean(r[axis] for r in valid)
        xx = sum((r["result"][axis] - x_mean) ** 2 for r in valid)
        xy = sum((r["result"][axis] - x_mean) * (r[axis] - y_mean) for r in valid)
        slope = max(0.0, xy / xx) if xx else 0.0
        fitted[axis] = {"slope": slope, "intercept": y_mean - slope * x_mean}
    return fitted, len(valid)


def metrics(test, params, calibration_prior, unused_prior):
    valid = [r for r in test if r["result"]["status"] == "ok"]
    outputs = {}
    for axis in rater.AXES:
        coefficient = params[axis]
        prior = calibration_prior[axis]
        def adjusted(row):
            value = coefficient["intercept"] + coefficient["slope"] * row["result"][axis]
            return max(1.0, min(9.0, value))
        outputs[axis] = {
            "valid_raw": errors(valid, axis, lambda r: r["result"][axis]),
            "valid_calibrated": errors(valid, axis, adjusted),
            "valid_calibration_mean": errors(valid, axis, lambda _: prior),
            "valid_unused_mean": errors(valid, axis, lambda _: unused_prior[axis]),
            "all_raw_with_calibration_prior": errors(test, axis, lambda r: r["result"].get(axis, prior)),
            "all_calibrated_with_calibration_prior": errors(test, axis, lambda r: adjusted(r) if r["result"]["status"] == "ok" else prior),
            "all_calibration_mean": errors(test, axis, lambda _: prior),
            "all_unused_mean": errors(test, axis, lambda _: unused_prior[axis]),
        }
        if valid:
            rng = Random(20261127)
            deltas = {}
            for name, predictions in (
                    ("raw", [r["result"][axis] for r in valid]),
                    ("calibration_mean", [prior] * len(valid)),
                    ("unused_mean", [unused_prior[axis]] * len(valid))):
                # Positive differences mean the calibrated absolute error is smaller.
                delta = [abs(r[axis] - baseline) - abs(r[axis] - adjusted(r))
                         for r, baseline in zip(valid, predictions)]
                replicates = sorted(sum(rng.choices(delta, k=len(delta))) / len(delta) for _ in range(2000))
                deltas[name] = {"mae_gain": round(mean(delta), 4),
                                "paired_bootstrap_95_percentile": [round(replicates[49], 4), round(replicates[1949], 4)]}
            outputs[axis]["exploratory_relative_mae_gain_on_valid"] = deltas
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / ".runtime/chinese-emobank/CVAS_all.csv")
    parser.add_argument("--cache", type=Path, default=ROOT / ".runtime/chinese-emobank/qwen-zh-va-calibration.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/chinese-cvas-calibration.json")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--offline", action="store_true", help="Verify and re-score a complete existing cache without Ollama")
    args = parser.parse_args()
    if len({p.resolve() for p in (args.cache, args.output, args.data)}) != 3 or args.output.suffix != ".json" or args.cache.suffix != ".jsonl":
        parser.error("Choose distinct paths for source CSV, JSONL cache and JSON report")
    rows = prepared(args.data)
    fingerprint = frozen_backend(args.offline)
    calibration = infer(rows[96:288], args.cache, fingerprint, args.workers, args.offline)
    params, n_fit = fit(calibration)
    print("Frozen coefficients", n_fit, json.dumps(params), flush=True)
    prior = {axis: mean(r[axis] for r in calibration) for axis in rater.AXES}
    unused = {axis: mean(r[axis] for r in rows[448:]) for axis in rater.AXES}
    test = infer(rows[288:448], args.cache, fingerprint, args.workers, args.offline)
    status = Counter(r["result"]["status"] for r in test)
    report = {
        "version": VERSION, "model": rater.MODEL, "model_digest": MODEL_DIGEST, "rater_version": rater.VERSION,
        "rater_fingerprint": fingerprint, "scale": [1, 9], "source": URL, "source_sha256": DIGEST,
        "source_use": "CVAS Chinese-EmoBank academic research / noncommercial; derived aggregate coefficients; no sentences or individual annotations.",
        "source_terms": "https://www.mis.yzu.edu.tw/nlplab/resources/chinese-emobank_download.html",
        "attribution": "Lee, Li & Yu (2022), Chinese EmoBank; Yu et al. (2016), Building Chinese affective resources in valence-arousal dimensions.",
        "selection": "Sorted SHA256(zh-va-v1|stripped original text). Excluded pilot ranks 1–96. Fit on 97–288; independent confirmation 289–448. No prompt or parameter search.",
        "calibration_ids_sha256": CALIBRATION_IDS, "test_ids_sha256": TEST_IDS,
        "n_fit": n_fit, "calibration_attempted": len(calibration),
        "calibration_status_counts": dict(Counter(r["result"]["status"] for r in calibration)),
        "calibration_prior": prior, "unused_prior": unused, "unused_prior_rows": len(rows) - 448,
        "method": "Least squares per axis, nonnegative slope, intercept refit at means. Clip prediction to [1,9]. Fallbacks apply to evaluation only.",
        "coefficients": params, "test_attempted": len(test), "test_status_counts": dict(status),
        "test_coverage": status["ok"] / len(test), "test_metrics": metrics(test, params, prior, unused),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
