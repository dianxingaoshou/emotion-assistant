"""Run the shipped English content models on paired deletions in attachment 2 test transcripts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from fit_english_intensity import regression_metrics
from fit_english_polarity import ROOT, iter_label_rows
from emotion_assistant.english_polarity import ASSET as POLARITY_ASSET, TOKEN, predict_english_polarity
from emotion_assistant.english_intensity import ASSET as INTENSITY_ASSET, predict_english_intensity


def drop_words(text, ratio, position):
    normalized = text.lower().replace("’", "'")
    words = list(TOKEN.finditer(normalized))
    if not words or not ratio:
        return text, 0, len(words)
    size = min(len(words), math.ceil(len(words) * ratio))
    free = len(words) - size
    start = {"start": 0, "middle": free // 2, "end": free}[position]
    return normalized[:words[start].start()] + " " + normalized[words[start + size - 1].end():], size, len(words)


def score(rows, ratio, position, train_median):
    correct = classified = removed = words = 0
    truth, predictions = [], []
    for row in rows:
        text, n_removed, n_words = drop_words(row["text"], ratio, position)
        polarity = predict_english_polarity(text)
        intensity = predict_english_intensity(text)
        if polarity:
            classified += 1
            correct += polarity["label"] == row["annotation"].strip().lower()
        if intensity:
            truth.append(float(row["label"]))
            predictions.append(intensity["value"])
        removed += n_removed
        words += n_words
    return {
        "clips": len(rows),
        "removed_words": removed,
        "total_words": words,
        "correct": correct,
        "incorrect": classified - correct,
        "abstained": len(rows) - classified,
        "intensity_scored": len(truth),
        "intensity_abstained": len(rows) - len(truth),
        "intensity": regression_metrics(truth, predictions),
        "train_median_same_clips": regression_metrics(truth, [train_median] * len(truth)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path, help="attachment 2 label.xlsx")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/transcript-gaps.json")
    args = parser.parse_args()
    rows = list(iter_label_rows(args.workbook))
    test = [r for r in rows if r["mode"] == "test"]
    known = {r["video_id"] for r in rows if r["mode"] in ("train", "valid")}
    if len(test) != 727 or any(r["video_id"] in known for r in test):
        raise ValueError("Expected official 727-clip, source-disjoint attachment 2 test")
    train_median = median(float(r["label"]) for r in rows if r["mode"] == "train")
    conditions = {"intact": score(test, 0, "start", train_median)}
    for ratio in (0.25, 0.5):
        for position in ("start", "middle", "end"):
            conditions[f"drop_{int(100*ratio)}pct_{position}"] = score(test, ratio, position, train_median)
    conditions["drop_all_text"] = score(test, 1, "start", train_median)
    result = {"protocol": "attachment2-shipped-transcript-gaps-v1",
              "scope": "Actual shipped English text APIs. Full workbook transcripts. "
                       "Remove ceil(ratio * lexical word count) consecutive words and pass the remaining string "
                       "as a single transcript. Missing words are invisible to the model, as with an ASR omission. "
                       "No A/V inputs or reference labels as features. Intensity metrics on scored clips only.",
              "sources": len({r["video_id"] for r in test}),
              "workbook_sha256": hashlib.sha256(args.workbook.read_bytes()).hexdigest(),
              "polarity_sha256": hashlib.sha256(POLARITY_ASSET.read_bytes()).hexdigest(),
              "intensity_sha256": hashlib.sha256(INTENSITY_ASSET.read_bytes()).hexdigest(),
              "train_median": train_median, "conditions": conditions}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n", "utf-8")
    print(f"Audited {len(test)} transcripts under {len(conditions)} paired conditions")


if __name__ == "__main__":
    main()
