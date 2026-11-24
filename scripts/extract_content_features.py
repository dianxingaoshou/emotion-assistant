"""Extract aligned model outputs from competition E videos using the actual file importer.

The cache and output contain timestamped scores and clip labels, never raw video,
speech, transcript text, or official reference transcripts.
"""

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_mosei_upload import attachment_rows, runtime_configuration
from scripts.fit_english_polarity import fit, load_rows


AV_FIELDS = (
    "t_start",
    "t_end",
    "scores",
    "status",
    "valid",
    "decision",
    "decision_reasons",
    "conflict",
    "modalities",
    "weights",
    "per_modality_scores",
    "per_modality_output_scores",
    "per_modality_quality",
    "per_modality_coverage",
    "reason",
    "quality",
    "event_id",
    "audio_event_id",
    "source_ids",
)


def strip_private_data(result, identifier, development_model):
    from emotion_assistant.english_polarity import predict_with_model

    intervals = []
    for row in result["fusion_intervals"]:
        item = {k: row[k] for k in AV_FIELDS if k in row}
        item["session_id"] = identifier
        intervals.append(item)
    transcripts = []
    for row in result["transcripts"]:
        if row.get("valid"):
            transcript = {
                "t_start": row["t_start"],
                "t_end": row["t_end"],
                "valid": True,
                "session_id": identifier,
                "event_id": row.get("event_id"),
                "semantic_context": {
                    "english_polarity": (row.get("semantic_context") or {}).get("english_polarity")
                },
            }
            if development_model is not None:
                transcript["dev_prediction"] = predict_with_model(row["text"], development_model)
            transcripts.append(transcript)
    return {
        "av_intervals": intervals,
        "transcripts": transcripts,
        "audio_samples": [
            {k: row[k] for k in ("t_start", "t_end", "event_id")} for row in result["audio_samples"]
        ],
        "clip_text": (result.get("semantic_context") or {}).get("english_polarity"),
        "clip_av": result.get("scores"),
        "duration_s": result["timeline_coverage"]["duration_s"],
    }


def disjoint_source_partitions(splits):
    partitions = {}
    for split, rows in splits.items():
        for name in {name for name, _, _ in rows}:
            if name in partitions:
                raise ValueError(f"Source belongs to more than one official partition: {name}")
            partitions[name] = split
    return partitions


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def align_features(features, is_development):
    """Use the desktop's own time alignment for both the original and development-supervised scores."""
    from emotion_assistant.content_timeline import iter_content_features

    live_rows = list(
        iter_content_features(features["av_intervals"], features["transcripts"], features["audio_samples"])
    )
    if is_development:
        surrogate_transcripts = [
            {**row, "semantic_context": {"english_polarity": row["dev_prediction"]}}
            for row in features["transcripts"]
        ]
        training_rows = list(
            iter_content_features(features["av_intervals"], surrogate_transcripts, features["audio_samples"])
        )
        if [(r["t_start"], r["t_end"]) for r in live_rows] != [
            (r["t_start"], r["t_end"]) for r in training_rows
        ]:
            raise ValueError("Changing only transcript scoring changed interval boundaries")
        for row, dev_row in zip(live_rows, training_rows, strict=True):
            row["dev_text"] = dev_row["text"]
    # Runtime event IDs are random UUIDs for per-session provenance. Source
    # video hashes plus original interval timestamps identify our research unit.
    for row in live_rows:
        row.pop("source_ids", None)
    return {
        "duration_s": features["duration_s"],
        "clip_text": features["clip_text"],
        "clip_av": features["clip_av"],
        "intervals": live_rows,
    }


def write_atomic(path, data):
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temp.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Folder containing attachments 1 and 2")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/content-features.jsonl")
    parser.add_argument("--snapshot", type=Path, help="Also save a gzip snapshot of the same numeric rows")
    parser.add_argument("--cache", type=Path, default=ROOT / ".runtime/content-feature-cache")
    parser.add_argument(
        "--fresh", action="store_true", help="Re-run raw models and replace cached branch scores"
    )
    args = parser.parse_args()
    from emotion_assistant.paths import MODELS, configure_environment

    configure_environment()
    import cv2
    from emotion_assistant.english_polarity import ASSET, tokens
    from emotion_assistant.local_media import analyze_media
    from emotion_assistant.models import ModelHub, VERSIONS

    cv2.setNumThreads(2)
    books = list(args.data.rglob("label.xlsx"))
    if len(books) != 1:
        raise ValueError("Expected one attachment 2 label.xlsx below --data")
    splits = load_rows(books[0])
    base_partitions = disjoint_source_partitions(splits)
    inputs = attachment_rows(args.data)
    raw_sources = {path.parent.name for _, path, _, _ in inputs}
    raw_texts = {" ".join(tokens(reference)) for _, _, _, reference in inputs}
    base_texts = {" ".join(tokens(text)) for split in ("train", "valid") for _, text, _ in splits[split]}
    repeated = sum(
        " ".join(tokens(reference)) in base_texts
        for _, path, _, reference in inputs
        if base_partitions.get(path.parent.name) not in ("train", "valid")
    )
    if repeated:
        raise ValueError("External evaluation contains reference text seen in base training or validation")
    surrogate_rows = [
        (source, text, label)
        for source, text, label in splits["train"]
        if source not in raw_sources and " ".join(tokens(text)) not in raw_texts
    ]
    development_text_model = fit(surrogate_rows)
    frozen_inputs = {
        row["id"]: row
        for row in json.loads((ROOT / "docs/results/mosei-upload.json").read_text("utf-8"))["clips"]
    }
    metadata = {
        "protocol": "content-features-v1",
        "runtime": runtime_configuration(),
        "asr": VERSIONS["asr"],
        "text_artifact_sha256": hash_file(ASSET),
        "surrogate_text_model": development_text_model["model_version"],
        "surrogate_train_clips": len(surrogate_rows),
        "surrogate_train_sources": len({source for source, _, _ in surrogate_rows}),
        "surrogate_excludes_raw_sources": len(raw_sources),
        "official_split_sha256": hash_file(books[0]),
        "input_code_sha256": {
            name: hash_file(ROOT / "emotion_assistant" / name)
            for name in ("file_tracks.py", "models.py", "english_polarity.py", "core.py", "fusion_policy.py")
        },
    }
    signature = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    hub = ModelHub(MODELS, enable_asr=True)
    models_checked = False
    output = []
    for index, (identifier, path, label, _) in enumerate(inputs, 1):
        source_hash = hash_file(path)
        old = frozen_inputs[identifier]
        if old["label"] != label or old["sha256"] != source_hash:
            raise ValueError(f"Source or label differs from previous frozen run: {identifier}")
        cache = args.cache / (source_hash + "-" + signature[:16] + ".json")
        base_split = base_partitions.get(path.parent.name, "unseen")
        if cache.exists() and not args.fresh:
            features = json.loads(cache.read_text("utf-8"))
        else:
            if not models_checked:
                statuses = {**hub.load_visual(), **hub.load_audio_emotion(), **hub.load_asr()}
                if not all(state["ready"] for state in statuses.values()):
                    raise RuntimeError(f"Extraction requires the real visual, audio and ASR weights: {statuses}")
                models_checked = True
            result = analyze_media(
                path,
                model_dir=MODELS,
                model_hub=hub,
                transcribe=True,
                on_sample=lambda *_: None,
                cancelled=lambda: False,
            )
            development_model = development_text_model if base_split in ("train", "valid") else None
            features = strip_private_data(result, identifier, development_model)
            write_atomic(cache, json.dumps(features, ensure_ascii=False, allow_nan=False))
        row = {
            "id": identifier,
            "source_id": identifier.split("/")[0],
            "label": label,
            "base_split": base_split,
            "partition": "development" if base_split in ("train", "valid") else "holdout",
            "sha256": source_hash,
            "extraction_hash": signature,
            **align_features(features, base_split in ("train", "valid")),
        }
        output.append(row)
        if index % 10 == 0:
            print(f"Feature extraction {index}/100", flush=True)
    if len(output) != 100:
        raise ValueError("Expected 100 labeled clips")
    write_atomic(args.out, "".join(json.dumps(row, allow_nan=False) + "\n" for row in output))
    if args.snapshot is not None:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(args.snapshot, gzip.compress(args.out.read_bytes(), mtime=0))
    details = {
        **metadata,
        "holdout_reference_texts_repeated_in_base_train_valid": repeated,
        "timeline_code_sha256": hash_file(ROOT / "emotion_assistant/content_timeline.py"),
        "output_sha256": hash_file(args.out),
        "partitions": {
            split: {
                "clips": sum(row["partition"] == split for row in output),
                "sources": len({row["source_id"] for row in output if row["partition"] == split}),
            }
            for split in ("development", "holdout")
        },
    }
    write_atomic(args.out.with_suffix(".metadata.json"), json.dumps(details, indent=2) + "\n")
    if args.snapshot is not None:
        write_atomic(args.snapshot.with_suffix("").with_suffix(".metadata.json"),
                     json.dumps(details, indent=2) + "\n")
    print(Counter(r["partition"] for r in output))


if __name__ == "__main__":
    main()
