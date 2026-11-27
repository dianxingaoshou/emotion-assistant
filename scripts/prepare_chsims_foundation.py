"""Prepare official supervised captions and a deterministic original-video CH-SIMS cohort."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from scripts.chsims_samples import CACHE, cohort, fetch_members, fetch_selected, write_atomic

DEST = ROOT / ".runtime/chsims-foundation"
SIZES = {"train": 256, "valid": 128}
SEED = "chsims-deep-fusion-v1"
AUDIT_SEED = "chsims-deep-fusion-unseen-audit-v1"


def selected_rows(data, dest, *, audit_only=False):
    tests = cohort(data / "label.csv")
    with (data / "label.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    out, selected = [], {}
    for split in (("audit",) if audit_only else ("train", "valid", "test")):
        official_split = "test" if split == "audit" else split
        entries = {f"{r[0]}/{r[1]}": r for r in rows if r[-1] == official_split}
        if split in SIZES:
            selected[split] = sorted(entries, key=lambda key: hashlib.sha256(
                f"{SEED}|{split}|{key}".encode("utf8")).digest())[:SIZES[split]]
        elif split == "test":
            selected[split] = list(tests)
        else:
            selected[split] = sorted(set(entries) - set(tests), key=lambda key: hashlib.sha256(
                f"{AUDIT_SEED}|{key}".encode("utf8")).digest())[:48]
        for key, r in entries.items():
            if official_split == "test" and key not in selected[split]:
                continue
            text, source = r[2], "official_caption"
            if official_split == "test":
                fresh = dest / "observations" / f"{key}.json"
                prior = data / "observations" / f"{key}.json"
                if fresh.exists():
                    text = json.loads(fresh.read_text("utf8"))["text"]
                elif prior.exists():
                    obs = json.loads(prior.read_text("utf8"))
                    text = " ".join(w["text"].strip() for w in obs["windows"] if w["text"].strip())
                else:
                    text = ""
                source = "desktop_asr"
            out.append({"id": key, "split": split, "label": r[-2].lower(),
                        "text": text, "text_source": source, "raw_selected": key in selected[split]})
    return out, selected


def prepare(data, dest, *, download=False, audit_only=False):
    if download and not (data / "label.csv").is_file():
        fetch_selected(data, 48)
    rows, selection = selected_rows(data, dest, audit_only=audit_only)
    dest.mkdir(parents=True, exist_ok=True)
    row_file = "audit-rows.jsonl" if audit_only else "rows.jsonl"
    cohort_file = "audit-cohort.json" if audit_only else "raw-cohort.json"
    write_atomic(dest / row_file, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf8"))
    write_atomic(dest / cohort_file, (json.dumps(selection, indent=2) + "\n").encode("utf8"))
    print(f"{len(rows)} rows; fixed raw split:", {k: len(v) for k, v in selection.items()}, flush=True)
    if download:
        keys = [key for split in ("test", "train", "valid", "audit") for key in selection.get(split, ())]
        fetch_members(data, keys)
    return selection


def available_videos(data, selection, wait):
    pending = [key for split in ("test", "valid", "train", "audit") for key in selection.get(split, ())]
    deadline = time.monotonic() + 600
    while pending:
        waiting = []
        for key in pending:
            video = data / "Raw" / f"{key}.mp4"
            if video.is_file():
                yield key, video
                deadline = time.monotonic() + 600
            elif wait:
                waiting.append(key)
            else:
                raise FileNotFoundError(video)
        pending = waiting
        if pending:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{len(pending)} original clips still unavailable, starting with {pending[0]}")
            time.sleep(1)


def extract_observations(data, dest, selection, wait_for_files):
    import cv2
    import numpy as np
    import torch
    from emotion_assistant.paths import configure_environment, MODELS
    configure_environment()
    from emotion_assistant.models import ModelHub
    from emotion_assistant.local_media import analyze_media
    from emotion_assistant.clip_inputs import clip_inputs
    from scripts.benchmark_chsims_conflict import EXTRACT_MODULES, code_signature, sha_file

    versions = {"production_code": code_signature((*EXTRACT_MODULES, "clip_inputs.py")),
                "models_manifest": sha_file(MODELS / "manifest.json")}
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    hub = ModelHub(MODELS, enable_asr=True)
    model_status = {**hub.load_visual(), **hub.load_audio()}
    if not all(item["ready"] for item in model_status.values()):
        raise RuntimeError(str(model_status))
    total = sum(map(len, selection.values()))
    for index, (key, video) in enumerate(available_videos(data, selection, wait_for_files), 1):
        checksum = sha_file(video)
        out = dest / "observations" / f"{key}.json"
        if out.exists():
            old = json.loads(out.read_text("utf8"))
            if old.get("sha256") == checksum and old.get("versions") == versions:
                continue
        seed = int(checksum[:8], 16)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        result = analyze_media(video, model_dir=MODELS, model_hub=hub, transcribe=True,
                               on_sample=lambda *_: None, cancelled=lambda: False)
        if any(str(t.get("reason", "")).startswith("inference_error") for t in result["transcripts"]):
            raise RuntimeError(f"ASR failed on {key}")
        fields = clip_inputs(result)
        payload = {"id": key, "sha256": checksum, "versions": versions, **fields}
        write_atomic(out, json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf8"))
        if index % 12 == 0 or index == total:
            print(f"Processed {index}/{total} raw clips", flush=True)
    prepare(data, dest, audit_only="audit" in selection)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=CACHE)
    parser.add_argument("--dest", type=Path, default=DEST)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--wait-for-files", action="store_true", help="Allow a concurrent downloader to finish")
    parser.add_argument("--audit", action="store_true", help="Only the prespecified disjoint 48 test clips")
    args = parser.parse_args()
    selected = prepare(args.data, args.dest, download=args.download, audit_only=args.audit)
    if args.extract:
        extract_observations(args.data, args.dest, selected, args.wait_for_files)
