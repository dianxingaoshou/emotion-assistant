"""Run a bounded English speech second pass on the original files, without labels or reference text."""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from benchmark_mosei_upload import attachment_rows
from extract_asr_intensity import BASE_SEED, PRIVATE
from emotion_assistant.english_polarity import tokens
from emotion_assistant.file_tracks import RATE, pcm_parts


def continuous_audio(path, track, origin, check):
    if (track is None or track.start is None or track.duration is None or
            not math.isfinite(track.duration) or not .2 <= track.duration <= 30):
        return None
    begin = end = None
    chunks = []
    size = 0
    for stamp, block in pcm_parts(path, track, origin, check):
        check()
        if begin is None:
            begin = stamp
        elif abs(stamp - end) > 2 / RATE:
            return None
        size += len(block)
        if size > 30 * RATE:
            return None
        chunks.append(block)
        end = stamp + len(block) / RATE
    if begin is None or size < int(.2 * RATE):
        return None
    return np.concatenate(chunks), max(0., begin)


def transcribe_context(path, track, origin, model_hub, source_id, check):
    pcm = continuous_audio(path, track, origin, check)
    if pcm is None:
        return None
    samples, start = pcm
    check()
    result = model_hub.transcribe_audio(samples, source_id, start, start + len(samples)/RATE,
                                        asr_allowed=True, language="en")
    check()
    return result if result.get("valid") and result.get("text") else None


def select_words(segments, retry, method):
    first = " ".join(row["text"].strip() for row in segments if row.get("text"))
    if method == "pooled" or not retry:
        return first, "first_pass_pooled"
    if method == "longer" and len(tokens(retry)) <= len(tokens(first)):
        return first, "first_pass_pooled"
    return retry, "english_context"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    args = parser.parse_args()
    from emotion_assistant.paths import configure_environment, MODELS
    configure_environment()
    import torch
    from emotion_assistant.file_tracks import media_info
    from emotion_assistant.models import ModelHub

    torch.set_num_threads(2)
    book, = args.data.rglob("label-100.xlsx")
    hub = ModelHub(MODELS, enable_asr=True)
    if not hub.load_asr()["asr"]["ready"]:
        raise RuntimeError("Local SenseVoice model unavailable")
    PRIVATE.mkdir(parents=True, exist_ok=True)
    # Complete the independent development rows first. The reference is never passed to ASR.
    held = {r["id"] for r in json.loads((PRIVATE.parents[1] / "docs/results/file-intensity.json").read_text("utf-8"))["clips"]}
    rows = sorted(attachment_rows(book), key=lambda row: (row[0] in held, row[0]))
    output = PRIVATE / "english-context.jsonl"
    partial = output.with_suffix(".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as out:
        for n, (key, path, _, _) in enumerate(rows, 1):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            seed = (int(digest[:8], 16) + BASE_SEED) % 2**32
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            tracks, origin = media_info(path, True)
            start, cpu = time.perf_counter(), time.process_time()
            result = transcribe_context(path, tracks.get("audio"), origin, hub, key, lambda: None)
            event = {"id": key, "media_sha256": digest, "text": result["text"] if result else None,
                     "wall_seconds": time.perf_counter() - start, "cpu_seconds": time.process_time() - cpu}
            out.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            out.flush()
            if n % 10 == 0 or n == 57:
                print(f"Context speech inference {n}/{len(rows)}", flush=True)
    partial.replace(output)


if __name__ == "__main__":
    main()
