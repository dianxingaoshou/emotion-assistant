"""Replay the pinned public paired AV clips, without opening a camera or microphone."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from bench_real_ravdess_data import CACHE, ROOT, specimens

sys.path.insert(0, str(ROOT))
from emotion_assistant.paths import configure_environment, MODELS

PROTOCOL = "ravdess-half-second-clip-v1"


def infer_clip(hub, path):
    import av
    import numpy as np
    from emotion_assistant.core import FusionEngine

    # Benchmark unit is one released speech utterance, supplied already clipped.
    # We do not emulate VAD endpoints, hardware, processing age, or real-time latency.
    with av.open(str(path)) as media:
        res = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        samples = [out.to_ndarray().reshape(-1) for frame in media.decode(audio=0)
                   for out in res.resample(frame)]
        samples.extend(out.to_ndarray().reshape(-1) for out in res.resample(None))
        audio = np.concatenate(samples).astype(np.float32)
    duration = len(audio) / 16000
    engine = FusionEngine()
    # First available frame at each fixed 0.5s bin midpoint; each represents
    # only its own bin. Short final bin is clipped at the utterance end.
    bins = [(i*.5, min(duration, (i+1)*.5)) for i in range(math.ceil(duration/.5))]
    reasons = Counter()
    with av.open(str(path)) as media:
        i = 0
        for frame in media.decode(video=0):
            if frame.time is None or i >= len(bins):
                continue
            start, end = bins[i]
            if frame.time + 1e-7 < (start + end)/2:
                continue
            observation = hub.analyze_frame(frame.to_ndarray(format="bgr24"), path.stem, start, end)
            reasons[observation["reason"]] += 1
            engine.add_visual(observation)
            i += 1
        reasons["no_video_frame"] += len(bins) - i
    acoustic = hub.analyze_audio_emotion(audio, path.stem, 0, duration)
    if any("inference_error" in str(reason) or "model_unavailable" in str(reason)
           for reason in [*reasons, acoustic["reason"]]):
        raise RuntimeError(f"inference error on {path.stem}: {dict(reasons)}, {acoustic['reason']}")
    fused = engine.fuse_audio(acoustic)
    modalities = {}
    for k in fused["modalities"]:
        modalities[k] = {"scores": fused["per_modality_scores"][k],
                         "quality": fused["per_modality_quality"][k],
                         "coverage": fused["per_modality_coverage"][k],
                         "model_version": fused["model_versions"][k][0],
                         "age_seconds": 0.0}
        if k == "audio":
            modalities[k]["speech_seconds"] = fused["speech_seconds"]
    return {"id": path.stem, "protocol": PROTOCOL, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "modalities": modalities, "visual_bins": len(bins), "visual_reasons": dict(reasons),
            "audio_reason": acoustic["reason"], "clip_seconds": duration}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--out", type=Path, default=ROOT / "reports/ravdess/features.jsonl")
    p.add_argument("--wait-minutes", type=float, default=0,
                   help="Process already completed media while the downloader runs.")
    args = p.parse_args()
    configure_environment()
    import cv2
    from emotion_assistant.models import ModelHub
    cv2.setNumThreads(2)
    hub = ModelHub(MODELS, enable_asr=False)
    status = {**hub.load_visual(), **hub.load_audio_emotion()}
    if not all(m["ready"] for m in status.values()):
        raise RuntimeError(status)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cache = args.out.parent / "inference-cache"
    cache.mkdir(exist_ok=True)
    names = [name for a in range(1,25) for name in specimens(a)]
    done = {}
    deadline = time.monotonic() + 60 * args.wait_minutes
    while len(done) < len(names):
        for name in names:
            path = args.cache / (name + ".mp4")
            if name in done or not path.exists():
                continue
            result = cache / (name + ".json")
            row = json.loads(result.read_text("utf8")) if result.exists() else {}
            if row.get("protocol") != PROTOCOL or row.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
                row = infer_clip(hub, path)
                tmp = result.with_suffix(".tmp")
                tmp.write_text(json.dumps(row), encoding="utf8")
                os.replace(tmp, result)
            done[name] = row
            if len(done) % 8 == 0:
                print(f"Inferred {len(done)}/{len(names)} clips", flush=True)
        if len(done) < len(names):
            if time.monotonic() >= deadline:
                raise SystemExit(f"{len(names)-len(done)} input files missing; run bench_real_ravdess_data.py")
            time.sleep(3)
    args.out.write_text("".join(json.dumps(done[name], sort_keys=True)+"\n" for name in names), encoding="utf8")


if __name__ == "__main__":
    main()
