"""Measure warm, serial processing times for the desktop text profile and its optional local review."""
import argparse
from collections import Counter
import hashlib
from http.client import HTTPConnection
import json
from pathlib import Path
import platform
from statistics import median, quantiles
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotion_assistant.expression_profile import text_expression_profile
from emotion_assistant.research_text_review import review_with_reference
from scripts import evaluate_goemotions as goemotions
from scripts.calibrate_chinese_affect import MODEL_DIGEST, prepared
from scripts.evaluate_chinese_affect import DIGEST as CVAS_DIGEST, URL as CVAS_URL


COUNT = 12
SALT = "desktop-text-latency-v1"


def fixed_samples(rows, language):
    # Order distinct texts solely by salted hash; labels and earlier predictions have no role.
    texts = {row.strip() for row in rows if row.strip()}
    ranked = sorted((hashlib.sha256(f"{SALT}|{language}|{text}".encode()).hexdigest(), text) for text in texts)
    selection = ranked[1:COUNT + 1]
    if len(selection) != COUNT:
        raise ValueError("Not enough distinct texts to recreate the cohort")
    ids_digest = hashlib.sha256("\n".join(item[0] for item in selection).encode()).hexdigest()
    return ranked[0][1], [text for _, text in selection], ids_digest


def current_backend():
    metadata = goemotions.ollama_metadata()
    if metadata["manifest_digest"] != MODEL_DIGEST:
        raise ValueError("The local model differs from this benchmark's frozen Qwen digest")
    return metadata


def has_vram_offload():
    connection = HTTPConnection("127.0.0.1", 11434, timeout=10)
    try:
        connection.request("GET", "/api/ps")
        response = connection.getresponse()
        if response.status != 200:
            return None
        for model in json.load(response).get("models", []):
            if model.get("digest") == MODEL_DIGEST and "size_vram" in model:
                return model["size_vram"] > 0
    finally:
        connection.close()
    return None


def timed(function, text):
    start = perf_counter_ns()
    result = function(text)
    return result, (perf_counter_ns() - start) / 1_000_000


def percentiles(values):
    return {"p50": round(median(values), 4),
            "p90": round(quantiles(values, n=10, method="inclusive")[8], 4)}


def run(output):
    english_dir = ROOT / ".runtime/goemotions" / goemotions.COMMIT
    english = goemotions.load_files(english_dir, download=False)["test.tsv"]
    datasets = {
        "en": fixed_samples((line.split("\t")[0] for line in english.splitlines()), "en"),
        "zh": fixed_samples((row["text"] for row in prepared(ROOT / ".runtime/chinese-emobank/CVAS_all.csv")), "zh"),
    }
    model = current_backend()
    for warmup, _, _ in datasets.values():
        offline = text_expression_profile(warmup)
        review = review_with_reference(warmup)
        rating_status = review.get("chinese_dimensions", {}).get("status")
        if (not offline or review["status"] != "ok" or
                (offline["language"] == "zh" and rating_status not in {"ok", "insufficient_evidence"})):
            raise RuntimeError("The optional local review and offline models must be available during warm-up")
    model["resident_in_vram_after_warmup"] = has_vram_offload()
    rows = {"en": [], "zh": []}
    # Alternate languages, complete the same two sequential operations for each text.
    for pair in zip(datasets["en"][1], datasets["zh"][1], strict=True):
        for language, text in zip(("en", "zh"), pair):
            offline, offline_ms = timed(text_expression_profile, text)
            review, review_ms = timed(review_with_reference, text)
            if (review["status"] in {"unavailable", "model_missing", "service_error"} or
                    review.get("chinese_dimensions", {}).get("status") == "service_error"):
                raise RuntimeError("The local Qwen backend was unavailable during the benchmark")
            rows[language].append((offline or {}, review, offline_ms, review_ms, len(text)))
    results = {}
    for language, trials in rows.items():
        results[language] = {
            "n": len(trials), "cohort_sha256": datasets[language][2],
            "text_length_characters": percentiles([row[4] for row in trials]),
            "offline_text_profile_ms": percentiles([row[2] for row in trials]),
            "optional_review_ms": percentiles([row[3] for row in trials]),
            "sequential_offline_plus_review_ms": percentiles([row[2] + row[3] for row in trials]),
            "offline_category_status": dict(Counter(row[0].get("fine_emotions", {}).get("status", "unsupported")
                                                   for row in trials)),
            "offline_dimensions_status": dict(Counter(row[0].get("affect_dimensions", {}).get("status", "unsupported")
                                                     for row in trials)),
            "semantic_review_status": dict(Counter(row[1]["status"] for row in trials)),
        }
        if language == "zh":
            results[language]["qwen_dimensions_status"] = dict(Counter(
                row[1].get("chinese_dimensions", {}).get("status", "not_run") for row in trials))
    report = {
        "unit": "milliseconds of elapsed wall-clock time",
        "scope": "Warm desktop recognition functions, with or without opt-in localhost Qwen. Excludes application "
                 "startup, audio transcription, UI scheduling/rendering and time spent reading or choosing.",
        "protocol": "A single client alternated English and Chinese. Text profiles, followed by optional reviews, "
                    "were executed serially with no cached model responses. One disjoint text per language warmed "
                    "the models first. All subsequent calls including abstentions enter the quantiles. English "
                    "review uses one semantic request and an offline reference when supported. A valid Chinese "
                    "semantic review is followed by a second request for grounded valence and arousal.",
        "selection": f"Order distinct stripped texts by SHA256({SALT}|language|text). First warms each language, "
                     f"next {COUNT} are timed. Labels and predictions are unused.",
        "percentiles": "Median and linearly interpolated 90th percentile (inclusive); small descriptive samples.",
        "sources": {"en": {"url": goemotions.BASE_URL + "/test.tsv", "sha256": goemotions.SHA256["test.tsv"]},
                    "zh": {"url": CVAS_URL, "sha256": CVAS_DIGEST}},
        "backend": model, "os": platform.system(), "python": platform.python_version(), "results": results,
        "limitations": "One workstation, one pass after warm-up. Timings depend on the hardware, model serving "
                       "configuration and text. These modules produce different information; latency is separate "
                       "from quality. No individual texts, predictions or host identifiers are included.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/official-text-latency.json")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".json":
        parser.error("Use a JSON report path")
    run(args.output)
