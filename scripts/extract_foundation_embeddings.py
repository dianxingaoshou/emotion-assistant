"""Extract the same frozen language/acoustic features used by the file detector."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emotion_assistant.foundation_embeddings import (
    AUDIO_DIM, AUDIO_SECONDS, TEXT_DIGEST, TEXT_DIM, TEXT_PREFIX, clip_audio_embedding, embed_texts,
)
from emotion_assistant.models import ModelHub, VERSIONS
from emotion_assistant.paths import MODELS, configure_environment


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def signature(mode):
    sources = ["foundation_embeddings.py"]
    if mode == "audio":
        sources.extend(["models.py", "file_tracks.py", "capture.py"])
    parts = [(ROOT / "emotion_assistant" / s).read_text(encoding="utf-8") for s in sources]
    parts.append(VERSIONS["audio"] + str(AUDIO_SECONDS) if mode == "audio" else TEXT_DIGEST + TEXT_PREFIX)
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def load_cache(path, identity):
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if data["identity"].item() == identity:
                return {str(key): (str(check), feature) for key, check, feature in zip(
                    data["ids"], data["checksums"], data["values"], strict=True)}
    return {}


def save(path, cache, identity, dim):
    keys = sorted(cache)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.stem + "-", suffix=".npz",
                                     delete=False) as stream:
        temp = Path(stream.name)
    try:
        np.savez(temp, ids=np.asarray(keys, dtype=str),
                 values=np.asarray([cache[key][1] for key in keys], dtype=np.float32).reshape(len(keys), dim),
                 checksums=np.asarray([cache[key][0] for key in keys], dtype=str), identity=identity)
        for attempt in range(40):
            try:
                temp.replace(path)
                break
            except PermissionError:
                # An open reader on Windows can briefly block atomic replacement.
                if attempt == 39:
                    raise
                time.sleep(0.25)
    finally:
        temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modality", choices=["text", "text-asr", "audio"])
    parser.add_argument("--cache", type=Path, default=ROOT / ".runtime/chsims-foundation")
    parser.add_argument("--raw", type=Path, default=ROOT / ".runtime/chsims/Raw")
    parser.add_argument("--wait-seconds", type=float, default=0,
                        help="Allow this much time for missing video or ASR inputs to arrive")
    parser.add_argument("--include-audit", action="store_true",
                        help="Also encode the separately locked audit MP4s (ASR only for text)")
    args = parser.parse_args()
    data = args.cache
    rows = [json.loads(line) for line in (data / "rows.jsonl").read_text(encoding="utf8").splitlines()]
    mode = args.modality
    if mode != "text":
        chosen = json.loads((data / "raw-cohort.json").read_text(encoding="utf8"))
        keys = set().union(*chosen.values())
        rows = [r for r in rows if r["id"] in keys]
    if args.include_audit:
        rows.extend(json.loads(line) for line in (data / "audit-rows.jsonl").read_text(encoding="utf8").splitlines())
    output = data / f"embeddings-{mode}.npz"
    identity = signature(mode)
    cache = load_cache(output, identity)
    dim = AUDIO_DIM if mode == "audio" else TEXT_DIM
    hub = None
    if mode == "audio":
        configure_environment()
        import torch
        torch.set_num_threads(2)
        hub = ModelHub(MODELS, enable_asr=False)
        if not hub.load_audio_emotion()["audio"]["ready"]:
            raise RuntimeError("Pinned speech emotion encoder unavailable")
    deadline = time.monotonic() + args.wait_seconds
    pending = rows
    checkpoint = len(cache)
    while pending:
        missing, batch = [], []
        def flush():
            if batch:
                vectors = embed_texts([item[2] for item in batch])
                for (key, check, _), feature in zip(batch, vectors, strict=True):
                    cache[key] = check, feature
                batch.clear()
        for row in pending:
            key = row["id"]
            if mode == "audio":
                path = args.raw / (key + ".mp4")
            elif mode == "text-asr" or row["split"] == "audit":
                path = data / "observations" / (key + ".json")
            else:
                path = None
            if path is not None and not path.exists():
                missing.append(row)
                continue
            if mode == "audio":
                check = file_hash(path)
                if key not in cache or cache[key][0] != check:
                    feature = clip_audio_embedding(path, model_hub=hub)
                    cache[key] = check, feature if feature is not None else np.zeros(dim, dtype=np.float32)
            else:
                text = row["text"] if path is None else json.loads(path.read_text(encoding="utf8"))["text"]
                check = hashlib.sha256(text.encode()).hexdigest()
                if key not in cache or cache[key][0] != check:
                    if text.strip():
                        batch.append((key, check, text))
                    else:
                        cache[key] = check, np.zeros(dim, dtype=np.float32)
                    if len(batch) == 24:
                        flush()
            if len(cache) - checkpoint >= 128:
                save(output, cache, identity, dim)
                checkpoint = len(cache)
        flush()
        save(output, cache, identity, dim)
        print(f"{mode}: {len(rows)-len(missing)}/{len(rows)} inputs; {len(cache)} cached", flush=True)
        if not missing or time.monotonic() >= deadline:
            break
        time.sleep(3)
        pending = missing
    if missing:
        raise SystemExit(f"Waiting for {len(missing)} source files. Re-run to resume.")


if __name__ == "__main__":
    main()
