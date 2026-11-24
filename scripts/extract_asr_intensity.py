"""Recreate the private reference/actual-ASR pairs used for the file intensity audit."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from benchmark_mosei_upload import ROOT, attachment_rows
from fit_english_intensity import fit, read_splits, train_design
from fit_english_polarity import iter_label_rows
from emotion_assistant.english_polarity import tokens


PRIVATE = ROOT / ".runtime/asr-intensity-study"
BASE_SEED = 20261124


def dump(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    args = parser.parse_args()
    from emotion_assistant.paths import configure_environment, MODELS
    configure_environment()
    import cv2
    import torch
    from emotion_assistant.english_intensity import ASSET
    from emotion_assistant.file_asr_intensity import english_speech
    from emotion_assistant.models import ModelHub, VERSIONS
    from emotion_assistant.local_media import analyze_media

    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    raw_book, = args.data.rglob("label-100.xlsx")
    fit_book, = args.data.rglob("label.xlsx")
    clips = attachment_rows(raw_book)
    target_by_key = {(r["video_id"], r["clip_id"]): float(r["label"]) for r in iter_label_rows(raw_book)}
    # Remove every raw video source and all identical normalized reference strings from shadow training.
    raw_sources = {path.parent.name for _, path, _, _ in clips}
    raw_words = {tuple(tokens(ref)) for _, _, _, ref in clips}
    splits = read_splits(fit_book)
    base_sources = {r[0] for k in ("train", "valid") for r in splits[k]}
    train = [r for r in splits["train"] if r[0] not in raw_sources and tuple(tokens(r[1])) not in raw_words]
    with gzip.open(ASSET, "rt", encoding="utf-8") as fh:
        frozen_model = json.load(fh)
    shadow = fit(train, train_design(train), frozen_model["alpha"])
    PRIVATE.mkdir(parents=True, exist_ok=True)
    dump(PRIVATE / "shadow.json", shadow)
    archived = json.loads((ROOT / "docs/results/file-intensity.json").read_text("utf-8"))
    held = {r["id"]: r for r in archived["clips"]}
    media = {r["id"]: r["sha256"] for r in json.loads(
        (ROOT / "docs/results/mosei-upload.json").read_text("utf-8"))["clips"]}
    if len(held) != 43 or set(held) != {k for k, p, _, _ in clips if p.parent.name not in base_sources}:
        raise ValueError("Attachment 1 sources no longer match the original holdout")
    hub = ModelHub(MODELS, enable_asr=True)
    ready = {**hub.load_visual(), **hub.load_audio_emotion(), **hub.load_asr()}
    if not all(item["ready"] for item in ready.values()):
        raise RuntimeError(f"Real media model unavailable: {ready}")
    rows = []
    language_gates = {}
    language_tags = {}
    # The fixed seed per clip controls SenseVoice frontend dither independently of order.
    for key, path, _, reference in sorted(clips, key=lambda clip: (clip[0] in held, clip[0])):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != media[key]:
            raise ValueError("Raw media no longer matches the frozen attachment")
        seed = (int(digest[:8], 16) + BASE_SEED) % 2**32
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        output = analyze_media(path, model_dir=MODELS, on_sample=lambda *_: None,
                               cancelled=lambda: False, model_hub=hub, transcribe=True,
                               refine_file_intensity=False)
        if any(t["reason"].startswith("inference_error") for t in output["transcripts"]):
            raise RuntimeError("ASR inference failure")
        valid = [t for t in output["transcripts"] if t["valid"]]
        language_gates[key] = english_speech(valid)
        language_tags[key] = [t.get("asr_language") for t in valid]
        estimate = output["content_summary"]["intensity"]
        row = {"id": key, "source": key.split("/")[0],
               "split": "holdout" if key in held else "development",
               "target": target_by_key[path.parent.name, path.stem], "official_text": reference,
               "segments": [{"start": t["t_start"], "end": t["t_end"], "text": t["text"]} for t in valid],
               "original_file_value": estimate["value"] if estimate is not None else None,
               "media_sha256": digest}
        rows.append(row)
        if len(rows) in (57, 100):
            out = PRIVATE / ("development.jsonl" if len(rows) == 57 else "utterances.jsonl")
            temporary = out.with_suffix(".tmp")
            temporary.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows),
                                 "utf-8", newline="\n")
            temporary.replace(out)
            prefix = "development-" if len(rows) == 57 else ""
            dump(PRIVATE / (prefix + "language-gates.json"), language_gates)
            dump(PRIVATE / (prefix + "language-tags.json"), language_tags)
        if len(rows) % 10 == 0 or len(rows) == 57:
            print(f"Actual baseline file inference {len(rows)}/100", flush=True)
    meta = {"base_seed": BASE_SEED, "asr": VERSIONS["asr"], "source_groups_excluded": len(raw_sources),
            "shadow_fit_clips": len(train), "shadow_fit_sources": len({r[0] for r in train}),
            "shadow_model_version": shadow["model_version"], "deployed_model_version": frozen_model["model_version"],
            "raw_book_sha256": hashlib.sha256(raw_book.read_bytes()).hexdigest(),
            "train_book_sha256": hashlib.sha256(fit_book.read_bytes()).hexdigest(),
            "private_snapshot_sha256": hashlib.sha256((PRIVATE / "utterances.jsonl").read_bytes()).hexdigest()}
    dump(PRIVATE / "metadata.json", meta)


if __name__ == "__main__":
    main()
