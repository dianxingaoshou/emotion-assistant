"""Validate English content polarity from the file importer's real ASR output.

The raw ASR text is used in memory only. Official reference transcripts are
evaluated separately, on source groups disjoint from training and validation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from benchmark_mosei_upload import ROOT, attachment_rows, metrics, paired_accuracy_interval
from fit_english_polarity import load_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Directory containing attachments 1 and 2")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/mosei-asr.json")
    args = parser.parse_args()
    from emotion_assistant.paths import configure_environment, MODELS
    configure_environment()
    import cv2
    from emotion_assistant.english_polarity import ASSET, predict_english_polarity
    from emotion_assistant.local_media import analyze_media
    from emotion_assistant.models import ModelHub, VERSIONS
    cv2.setNumThreads(2)
    files = list(args.data.rglob("label.xlsx"))
    if len(files) != 1:
        raise ValueError("Expected one attachment 2 label.xlsx")
    splits = load_rows(files[0])
    named_groups = {mode: {speaker for speaker, _, _ in rows} for mode, rows in splits.items()}
    seen_text = {" ".join(text.lower().split()) for key in ("train", "valid") for _, text, _ in splits[key]}
    majority = Counter(label for _, _, label in splits["train"]).most_common(1)[0][0]
    hub = ModelHub(MODELS, enable_asr=True)
    status = hub.load_asr()["asr"]
    if not status["ready"]:
        raise RuntimeError(status["reason"])
    rows = []
    inputs = attachment_rows(args.data)
    for index, (identifier, path, label, transcript_reference) in enumerate(inputs, 1):
        source = path.parent.name
        overlap = [key for key in named_groups if source in named_groups[key]]
        disjoint = not ("train" in overlap or "valid" in overlap)
        result = analyze_media(path, model_dir=MODELS, model_hub=hub, transcribe=True,
                               on_sample=lambda *_: None, cancelled=lambda: False)
        segments = sorted(result["transcripts"], key=lambda t: t["t_start"])
        words = " ".join(s["text"].strip() for s in segments if s["valid"] and s["text"].strip())
        actual = result.get("semantic_context", {}).get("english_polarity")
        independent = predict_english_polarity(words)
        if (actual or {}).get("label") != (independent or {}).get("label"):
            raise ValueError("Published file semantic result differs from its real ASR transcript")
        # The file importer never receives the supplied reference transcript.
        reference = predict_english_polarity(transcript_reference) if disjoint else None
        rows.append({"id": identifier, "label": label, "source_partitions": overlap,
                     "no_train_valid_source_overlap": disjoint, "has_asr_text": bool(words),
                     "reference_seen_in_train_valid": " ".join(transcript_reference.lower().split()) in seen_text,
                     "segments": sum(bool(s["valid"] and s["text"].strip()) for s in segments),
                     "asr": actual["label"] if actual else None,
                     "reference_text": reference["label"] if reference else None,
                     "majority": majority})
        if index % 10 == 0:
            print(f"Real ASR clips {index}/{len(inputs)}", flush=True)
    held = [r for r in rows if r["no_train_valid_source_overlap"]]
    def group_count(items):
        return {"clips": len(items), "sources": len({r["id"].split("/")[0] for r in items})}
    report = {
        "scope": "Raw E attachment 1, opt-in production VAD + actual SenseVoice ASR + fixed English expressed-content model. "
                 "Only sources excluded from both attachment 2 train and valid are scored; no thresholds fitted here. "
                 "Reference-text results are a separate input comparison. Neither semantic score is a visual or acoustic score.",
        "asr_version": VERSIONS["asr"], "english_classifier_sha256": hashlib.sha256(ASSET.read_bytes()).hexdigest(),
        "official_split_workbook_sha256": hashlib.sha256(files[0].read_bytes()).hexdigest(),
        "attachment_1_workbook_sha256": hashlib.sha256(next(args.data.rglob("label-100.xlsx")).read_bytes()).hexdigest(),
        "all_inputs": group_count(rows), "files_with_real_transcript": sum(r["has_asr_text"] for r in rows),
        "overlap_with_attachment_2_by_source": {
            key: group_count([r for r in rows if key in r["source_partitions"]]) for key in named_groups},
        "train_valid_disjoint": group_count(held), "held_labels": dict(Counter(r["label"] for r in held)),
        "held_reference_texts_repeated_in_train_valid": sum(r["reference_seen_in_train_valid"] for r in held),
        "held_asr_text_available": sum(r["has_asr_text"] for r in held),
        "train_majority": majority,
        "held_metrics": {key: metrics(held, key) for key in ("asr", "reference_text", "majority")},
        "held_delta_vs_majority_95pct_by_source": {
            key: paired_accuracy_interval(held, "majority", key) for key in ("asr", "reference_text")},
        "clips": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf8")
    print(json.dumps({k: v for k, v in report.items() if k != "clips"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
