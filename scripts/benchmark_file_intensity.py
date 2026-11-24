"""Check the file importer's utterance-weighted intensity against raw attachment 1 labels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark_mosei_upload import ROOT, attachment_rows
from fit_english_intensity import paired_intervals, regression_metrics
from fit_english_polarity import iter_label_rows


def evaluate(rows, key, median):
    scored = [row for row in rows if row[key] is not None]
    truth = [row["target"] for row in scored]
    result = {
        "total": len(rows),
        "scored": len(scored),
        "abstained": len(rows) - len(scored),
        "prediction": regression_metrics(truth, [row[key] for row in scored]),
        "train_median_same_clips": regression_metrics(truth, [median] * len(scored)),
    }
    if len(scored) > 1:
        result["paired_95pct_by_source"] = paired_intervals(
            [row["id"].split("/")[0] for row in scored], np.asarray(truth),
            np.asarray([row[key] for row in scored]), np.full(len(scored), median))
    return result


def summaries(rows, median):
    common = [row for row in rows if row["file"] is not None and row["official_text"] is not None]
    return {"file": evaluate(rows, "file", median), "official_text": evaluate(rows, "official_text", median),
            "official_text_on_common": evaluate(common, "official_text", median),
            "file_on_common": evaluate(common, "file", median)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/file-intensity.json")
    args = parser.parse_args()
    from emotion_assistant.paths import MODELS, configure_environment
    configure_environment()
    import cv2
    from emotion_assistant.english_intensity import ASSET, predict_english_intensity
    from emotion_assistant.local_media import analyze_media
    from emotion_assistant.models import ModelHub, VERSIONS

    cv2.setNumThreads(2)
    protocol = json.loads((ROOT / "docs/results/content-fusion.json").read_text("utf-8"))
    held = {row["id"] for row in protocol["clips"]}
    intensity_run = json.loads((ROOT / "docs/results/english-intensity.json").read_text("utf-8"))
    median = intensity_run["train_median"]
    book, = args.data.rglob("label-100.xlsx")
    targets = {(r["video_id"], r["clip_id"]): float(r["label"]) for r in iter_label_rows(book)}
    source_workbook, = args.data.rglob("label.xlsx")
    seen_sources = {r["video_id"] for r in iter_label_rows(source_workbook) if r["mode"] in {"train", "valid"}}
    raw = [r for r in attachment_rows(book) if r[0] in held]
    if (len(raw) != len(held) or len(held) != 43
            or any(path.parent.name in seen_sources for _, path, _, _ in raw)):
        raise ValueError("Expected the 43 previously reserved clips from independent sources")
    hub = ModelHub(MODELS, enable_asr=True)
    asr = hub.load_asr()["asr"]
    if not asr["ready"]:
        raise RuntimeError(asr["reason"])
    rows = []
    for index, (key, path, _, official_words) in enumerate(raw, 1):
        output = analyze_media(path, model_dir=MODELS, model_hub=hub, transcribe=True,
                               on_sample=lambda *_: None, cancelled=lambda: False)
        published = output["content_summary"]["intensity"]
        reference = predict_english_intensity(official_words)
        rows.append({"id": key, "target": targets[path.parent.name, path.stem],
                     "file": published["value"] if published else None,
                     "official_text": reference["value"] if reference else None})
        if index % 10 == 0:
            print(f"Scored {index}/{len(raw)} raw files", flush=True)
    report = {
        "scope": "Previously assessed attachment 1 holdout, disjoint sources from attachment 2 fit/selection. "
                 "Actual local file inference with opt-in ASR and duration-weighted intensity on scored speech. "
                 "Official transcripts used only for separate text-only comparison.",
        "source_count": len({p.parent.name for _, p, _, _ in raw}),
        "model_sha256": hashlib.sha256(ASSET.read_bytes()).hexdigest(),
        "workbook_sha256": hashlib.sha256(book.read_bytes()).hexdigest(),
        "asr_version": VERSIONS["asr"],
        "train_median": median,
        **summaries(rows, median),
        "clips": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8")
    print(json.dumps({k: report[k] for k in ("file", "official_text")}), flush=True)


if __name__ == "__main__":
    main()
