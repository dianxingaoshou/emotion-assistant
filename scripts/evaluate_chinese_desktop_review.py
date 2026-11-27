"""Measure the two-stage local desktop reviewer on the frozen 160-sentence CVAS cohort."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emotion_assistant import chinese_affect, chinese_affect_calibration, research_text_review, semantic_review
from emotion_assistant.chinese_dimensions import ASSET_SHA256, predict_chinese_dimensions
from scripts.calibrate_chinese_affect import MODEL_DIGEST, frozen_backend, infer, prepared, verify_local_model
from scripts.evaluate_chinese_affect import DIGEST, URL
from scripts.evaluate_chinese_dimension_hybrid import gains, metrics

COHORT = "25fd5493c0d0adc38f3404090bc3deb1dfa07d53f1137eaf78220e926d793a33"


def run_review(row):
    # Use exactly the entry point called by SemanticReviewQueue. It performs its own
    # schema and verbatim-quote checks before attaching Chinese dimension scores.
    result = research_text_review.review_with_reference(row["text"])
    dimensions = result.get("chinese_dimensions") or {}
    status = dimensions.get("status", "not_run")
    scores = dimensions.get("calibrated") if status == "ok" else None
    if scores and (dimensions["evidence"] not in row["text"] or
                   dimensions.get("calibration_version") != chinese_affect_calibration.VERSION):
        raise ValueError("Unexpected desktop evidence or calibration version")
    if status == "ok" and not scores:
        raise ValueError("No calibrated ratings returned by the desktop reviewer")
    return {"id": row["id"], "semantic": result["status"], "dimension": status, "scores": scores}


def collect(rows, cache, workers, offline, identity):
    by_id = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            stored = json.loads(line)
            if stored["identity"] != identity or stored["record"]["id"] in by_id:
                raise ValueError("Incompatible or duplicate evaluation cache record")
            record = stored["record"]
            by_id[record["id"]] = record
    pending = [row for row in rows if row["id"] not in by_id]
    if pending and offline:
        raise ValueError("First run the full local Qwen evaluation to populate its cache")
    if pending:
        verify_local_model()
        cache.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with cache.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=workers) as pool:
            for i, record in enumerate(pool.map(run_review, pending), 1):
                stream.write(json.dumps({"identity": identity, "record": record}) + "\n")
                stream.flush()
                by_id[record["id"]] = record
                if i == 1 or i % 10 == 0 or i == len(pending):
                    print(f"{i}/{len(pending)} pending reviews completed in {time.monotonic()-started:.1f}s", flush=True)
    return [by_id[row["id"]] for row in rows]


def correlation_changes(local, reviewed, gold):
    if len(gold) < 3:
        return {}
    indices = np.random.default_rng(91732).choice(len(gold), (5000, len(gold)))
    def correlations(x, y):
        x = x - x.mean(axis=1, keepdims=True)
        y = y - y.mean(axis=1, keepdims=True)
        denominator = np.sqrt((x*x).sum(axis=1) * (y*y).sum(axis=1))
        return np.divide((x*y).sum(axis=1), denominator, where=denominator > 0,
                         out=np.full_like(denominator, np.nan))
    out = {}
    for i, axis in enumerate(chinese_affect.AXES):
        y = gold[indices, i]
        differences = correlations(reviewed[indices, i], y) - correlations(local[indices, i], y)
        finite = differences[np.isfinite(differences)]
        if len(finite):
            point = correlations(reviewed[:, i][None, :], gold[:, i][None, :])[0]
            point -= correlations(local[:, i][None, :], gold[:, i][None, :])[0]
            out[axis] = {"pearson_increase": round(float(point), 4),
                         "paired_bootstrap_95": np.quantile(finite, [.025, .975]).round(4).tolist(),
                         "valid_resamples": len(finite)}
    return out


def summarize(rows, reviews, standalone_cache):
    axes = chinese_affect.AXES
    local, reviewed, gold = [], [], []
    for row, record in zip(rows, reviews):
        offline = predict_chinese_dimensions(row["text"])
        if offline["status"] == "model_unavailable":
            raise ValueError("Bundled Chinese dimension weights not available")
        local.append([offline["dimensions"][a]["score"] if offline["status"] == "ok" else np.nan for a in axes])
        reviewed.append([record["scores"][a] if record["scores"] else np.nan for a in axes])
        gold.append([row[a] for a in axes])
    local, reviewed, gold = map(np.array, (local, reviewed, gold))
    paired = np.isfinite(local).all(axis=1) & np.isfinite(reviewed).all(axis=1)
    status = Counter(record["semantic"] for record in reviews)
    second_stage = Counter(record["dimension"] for record in reviews)
    result = {"n": len(rows), "first_stage_semantic": dict(status), "second_stage_dimensions": dict(second_stage),
              "offline_valid": int(np.isfinite(local).all(axis=1).sum()),
              "review_valid": int(np.isfinite(reviewed).all(axis=1).sum()), "paired_valid": int(paired.sum()),
              "paired": {"offline_character_ridge": metrics(local[paired], gold[paired]),
                         "desktop_qwen": metrics(reviewed[paired], gold[paired]),
                         "mae_reduction_vs_offline": gains(local[paired], reviewed[paired], gold[paired]),
                         "pearson_increase_vs_offline": correlation_changes(local[paired], reviewed[paired], gold[paired])}}
    if np.isfinite(local).all():
        replay = np.where(np.isfinite(reviewed), reviewed, local)
        result["all_attempts_fallback_replay"] = {
            "policy": "Hypothetical reader prioritizes Qwen when valid, otherwise keeps visible offline score.",
            "offline_character_ridge": metrics(local, gold), "prefer_qwen_else_offline": metrics(replay, gold),
            "mae_reduction_vs_offline": gains(local, replay, gold)}
    if standalone_cache.is_file():
        previous = infer(rows, standalone_cache, frozen_backend(True), 1, True)
        ratings = [chinese_affect_calibration.apply_chinese_calibration({
            **r["result"], "model": semantic_review.MODEL, "version": chinese_affect.VERSION,
            "scope": "expressed_text"}) for r in previous]
        earlier = np.array([[r.get("calibrated", {}).get(a, np.nan) for a in axes] for r in ratings])
        direct_valid = np.isfinite(earlier).all(axis=1)
        end_valid = np.isfinite(reviewed).all(axis=1)
        both = direct_valid & end_valid
        result["versus_previous_dimension_only_run"] = {
            "valid_in_both": int(both.sum()), "only_standalone_valid": int((direct_valid & ~end_valid).sum()),
            "only_desktop_valid": int((end_valid & ~direct_valid).sum()),
            "neither_valid": int((~direct_valid & ~end_valid).sum()),
            "both_scores_identical_when_valid": int((earlier[both] == reviewed[both]).all(axis=1).sum()),
            "standalone_on_shared": metrics(earlier[both], gold[both]),
            "desktop_on_shared": metrics(reviewed[both], gold[both])}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / ".runtime/chinese-emobank/CVAS_all.csv")
    parser.add_argument("--cache", type=Path, default=ROOT / ".runtime/chinese-emobank/desktop-isolated-cvas-v1.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/chinese-desktop-review.json")
    parser.add_argument("--standalone-cache", type=Path,
                        default=ROOT / ".runtime/chinese-emobank/qwen-zh-va-new-confirmation.jsonl",
                        help="If available, compare the earlier independent VA-only run after completing this run")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--offline", action="store_true", help="Re-score a complete compatible cache without Ollama")
    args = parser.parse_args()
    if (len({p.resolve() for p in (args.data, args.cache, args.output, args.standalone_cache)}) != 4 or
            args.output.suffix != ".json" or args.cache.suffix != ".jsonl"):
        parser.error("Choose distinct source, JSONL cache and JSON report paths")
    rows = prepared(args.data)[448:608]
    signature = hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest()
    if signature != COHORT:
        raise ValueError("Not the same official CVAS cohort as the dimension-only study")
    sources = {m.__name__: hashlib.sha256(Path(m.__file__).read_text(encoding="utf-8").encode()).hexdigest()
               for m in (chinese_affect, chinese_affect_calibration, semantic_review, research_text_review)}
    identity = {"cohort_sha256": COHORT, "model_digest": MODEL_DIGEST, "dataset_sha256": DIGEST, "workers": args.workers,
                "review_source_sha256": sources}
    reviews = collect(rows, args.cache, args.workers, args.offline, identity)
    result = summarize(rows, reviews, args.standalone_cache)
    report = {"source": URL, "identity": identity, "offline_model_sha256": ASSET_SHA256,
              "scope": "Desktop back end after the user selects Chinese text and requests a review. "
                       "The test supplies the corpus transcription directly and evaluates reader-rated VA only. "
                       "It does not score ASR errors, Qt event delivery, semantic fields, or effects on users.",
              "method": "All hash-ranked CVAS sentences 449–608. Original selected text sent to the production "
                        "review_with_reference function. Each new record calls the actual semantic reviewer and, "
                        "only on a structurally valid review, the actual Chinese dimension rater. No Qwen output "
                        "from the earlier dimension-only experiment is reused. API failures count as missing output. "
                        "Scores describe the expressed passage. The optional all-160 fallback is an analysis policy.",
              "intervals": "Absolute-error reduction (offline minus Qwen) and Pearson increase (Qwen minus offline); "
                           "5,000 matched sentence bootstrap replicates, percentile 95%, seed 91732; "
                           "no multiple-testing adjustment.",
              "results": result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
