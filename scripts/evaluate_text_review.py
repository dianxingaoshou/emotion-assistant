"""Compare the desktop text classifier and local semantic review on identical English transcripts."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emotion_assistant.english_polarity import CLASSES, predict_english_polarity
from emotion_assistant.semantic_review import MODEL, REVIEW_VERSION, review_text
from scripts.fit_english_polarity import iter_label_rows

SELECTION_SEED = "semantic-review-pilot-v1"


def sample_order(row):
    material = "|".join((SELECTION_SEED, row["video_id"], row["clip_id"]))
    return hashlib.sha256(material.encode("utf-8")).digest()


def select_sample(labels, per_class):
    if per_class < 1:
        raise ValueError("--per-class must be positive")
    rows = [row for row in iter_label_rows(labels) if row["mode"] == "test"]
    identities = {(row["video_id"], row["clip_id"]) for row in rows}
    if len(rows) != 727 or len(identities) != len(rows):
        raise ValueError("Expected the 727 unique official Attachment 2 test clips")
    groups = {label: [] for label in CLASSES}
    for row in sorted(rows, key=sample_order):
        label = row["annotation"].lower()
        if label not in groups or not row["text"].strip():
            raise ValueError("Invalid official sentiment label or missing test transcript")
        groups[label].append(row)
    if any(len(group) < per_class for group in groups.values()):
        raise ValueError(f"--per-class exceeds the smallest class ({min(map(len, groups.values()))})")
    return sorted((row for group in groups.values() for row in group[:per_class]), key=sample_order)


def metrics(records, prediction_key):
    hits = {label: 0 for label in CLASSES}
    abstentions = {label: 0 for label in CLASSES}
    for row in records:
        truth, predicted = row["human"], row[prediction_key]
        if predicted == truth:
            hits[truth] += 1
        elif predicted not in CLASSES:
            abstentions[truth] += 1
    correct, abstained = sum(hits.values()), sum(abstentions.values())
    return {"correct": correct, "correct_per_class": hits, "abstained": abstained,
            "abstained_per_class": abstentions, "incorrect": len(records) - correct - abstained}


def evaluate(labels, per_class):
    records = []
    for row in select_sample(labels, per_class):
        text = row["text"]
        baseline = predict_english_polarity(text)
        start = monotonic()
        review = review_text(text)
        elapsed = monotonic() - start
        if review["status"] in ("unavailable", "model_missing", "service_error"):
            raise RuntimeError(f"Local semantic review could not run ({review['status']})")
        records.append({
            "id": row["video_id"] + "$_$" + row["clip_id"], "human": row["annotation"].lower(),
            "baseline": baseline["label"] if baseline is not None else None,
            "review": review.get("content_polarity"), "status": review["status"],
            "seconds": round(elapsed, 3),
        })
    return {
        "model": MODEL, "version": REVIEW_VERSION, "n": len(records), "samples_per_class": per_class,
        "selection": (f"first {per_class} per true class ordered by sha256({SELECTION_SEED}|video_id|clip_id); "
                      "held-out official full transcripts"),
        "baseline": metrics(records, "baseline"), "semantic_review": metrics(records, "review"),
        "status": dict(Counter(row["status"] for row in records)),
        "latency_median_s": median(row["seconds"] for row in records), "samples": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path, help="Official Attachment 2 label.xlsx")
    parser.add_argument("--per-class", type=int, default=10, help="Number of clips per true class (default: 10)")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/semantic-review/evaluation.json",
                        help="JSON report with clip IDs, predictions and timings; excludes transcript text")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".json" or args.output.resolve() == args.labels.resolve():
        parser.error("--output must be a JSON file separate from the input workbook")
    try:
        result = evaluate(args.labels, args.per_class)
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
