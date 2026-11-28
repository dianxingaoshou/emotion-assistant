"""Measure observed speech and construct aligned evidence on both sides of a CH-SIMS gap."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.chsims_teacher_data import CACHE, FRAME_PERIOD, ROOT, SPLITS, ranked_ids, read_rows, sha_file
from scripts.prepare_chsims_teacher_gaps import PREPARED, RATE, check_result, save_archive, text_units, with_cache


DIRECTORY = "teacher-timeline"
SEGMENTS = f"{DIRECTORY}/segments"
TEXT = f"{DIRECTORY}/text-embeddings"


def remaining_spans(duration, gap):
    left, right = gap
    if not 0. <= left < right <= duration:
        raise ValueError("The missing interval must be inside the original timeline")
    return ((0., left), (right, duration))


def speech_rows(pcm, span, vad):
    from emotion_assistant.file_tracks import speech_segments

    lo, hi = (round(seconds * RATE) for seconds in span)
    waveform = pcm[lo:hi]
    vad.reset()
    return list(speech_segments([(lo / RATE, waveform)], vad, lambda: None)) if len(waveform) else []


def speech_quality(rows):
    seconds = snr = total = 0.
    for _, start, end, detail in rows:
        duration = end - start
        voiced, estimate = detail.get("voiced_seconds"), detail.get("estimated_snr_db")
        seconds += duration if voiced is None else min(duration, max(0., voiced))
        snr += duration * (.5 if estimate is None else np.clip(estimate / 25., 0., 1.))
        total += duration
    return float(seconds), float(snr / total) if total else 0.


def evidence_window(pcm, frames, times, span, hub, vad, min_quality):
    from emotion_assistant.clip_inputs import clip_inputs
    from emotion_assistant.foundation_embeddings import AUDIO_DIM

    left, right = span
    rows = speech_rows(pcm, span, vad)
    voiced, snr = speech_quality(rows)
    speech, seconds, asr_seconds = np.zeros(AUDIO_DIM, dtype=np.float64), 0., 0.
    transcripts = []
    for waveform, start, end, _ in rows:
        vector = hub.audio_representation(waveform, start, end)
        if vector is not None:
            speech += (end - start) * vector
            seconds += end - start
        transcript = hub.transcribe_audio(waveform, "teacher-timeline", start, end, asr_allowed=True)
        check_result(transcript)
        if transcript["valid"] and text_units(transcript["text"]):
            transcripts.append(transcript["text"])
            asr_seconds += end - start

    visuals = []
    for frame, stamp in zip(frames, times, strict=True):
        if left <= stamp < right:
            end = min(stamp + FRAME_PERIOD, right)
            observation = hub.analyze_frame(np.ascontiguousarray(frame[:, :, ::-1]), "teacher-timeline", stamp, end)
            check_result(observation)
            observation["eligible"] = observation["valid"] and observation["quality"] >= min_quality
            visuals.append(observation)
    visual = clip_inputs({"samples": visuals, "timeline_coverage": {"duration_s": right}})["visual"]
    if visual is not None:
        visual["coverage"] = min(1., visual["coverage"] * right / (right - left))
    text = " ".join(transcripts)
    stats = {"start": left, "end": right, "voiced": voiced, "asr_seconds": asr_seconds, "snr": snr,
             "text": text, "visual": visual}
    result = (speech / seconds).astype(np.float32) if seconds else np.full(AUDIO_DIM, np.nan, dtype=np.float32)
    return stats, result


def prepare(cache, keys, device):
    if device == "cuda":
        sys.path.append(str(ROOT / ".venv/Lib/site-packages"))
    import cv2
    import torch
    from emotion_assistant.capture import SileroVAD
    from emotion_assistant.core import FusionEngine
    from emotion_assistant.models import ModelHub, VERSIONS
    from emotion_assistant.paths import DATA, MODELS, configure_environment

    configure_environment()
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Select the isolated CUDA environment for timeline preprocessing")

    class TimelineHub(ModelHub):
        def _load_funasr(self, folder, required):
            model = super()._load_funasr(folder, required)
            if device == "cuda":
                model.model.to(device)
                model.kwargs["device"] = device
            return model

    hub = TimelineHub(MODELS)
    status = {**hub.load_visual(), **hub.load_audio()}
    if not all(model["ready"] for model in status.values()):
        raise RuntimeError(f"Timeline models unavailable: {status}")
    vad_path = MODELS / "vad/silero_vad.onnx"
    vad = SileroVAD(vad_path)
    min_quality = FusionEngine.from_project(DATA).min_quality
    identity = {"implementation": sha_file(Path(__file__)), "models": VERSIONS, "vad": sha_file(vad_path),
                "torch": torch.__version__, "device": device, "min_quality": min_quality}
    started = perf_counter()
    for i, key in enumerate(keys, 1):
        raw = cache / "media" / f"{key}.npz"
        gap = cache / PREPARED / f"{key}.npz"
        dst = cache / SEGMENTS / f"{key}.npz"
        provenance = {**identity, "media": sha_file(raw), "gap": sha_file(gap)}
        if not with_cache(dst, provenance, "provenance"):
            with np.load(raw, allow_pickle=False) as original, np.load(gap, allow_pickle=False) as damaged:
                clean = json.loads(original["info"].item())
                changed = json.loads(damaged["info"].item())
                duration = clean["duration_s"]
                voiced, snr = speech_quality(speech_rows(original["pcm"], (0., duration), vad))
                pcm, frames, times = damaged["pcm"], damaged["video"], changed["frame_times"]
                windows, vectors = zip(*(evidence_window(pcm, frames, times, span, hub, vad, min_quality)
                                         for span in remaining_spans(duration, changed["gap"])), strict=True)
            data = {"provenance": provenance, "duration": duration, "gap": changed["gap"],
                    "clean": {"text": clean["text"], "voiced": voiced, "snr": snr},
                    "damaged_text": changed["text"], "joint_text": changed["joint_text"], "windows": windows}
            save_archive(dst, data, audio_embeddings=np.stack(vectors))
        if i % 128 == 0 or i == len(keys):
            print(f"Temporal evidence {i}/{len(keys)} in {perf_counter() - started:.1f}s", flush=True)


def embed(cache, keys):
    from emotion_assistant.chinese_affect import HAN
    from emotion_assistant.foundation_embeddings import TEXT_DIGEST, TEXT_DIM, embed_texts

    started = perf_counter()
    names = ("window-0", "window-1")
    for offset in range(0, len(keys), 6):
        pending = []
        for key in keys[offset:offset + 6]:
            source = cache / SEGMENTS / f"{key}.npz"
            dst = cache / TEXT / f"{key}.npz"
            identity = {"segments": sha_file(source), "embedding": TEXT_DIGEST}
            if not with_cache(dst, identity, "provenance"):
                with np.load(source, allow_pickle=False) as evidence:
                    info = json.loads(evidence["info"].item())
                text = {"clean": info["clean"]["text"], "damaged": info["damaged_text"], "joint": info["joint_text"],
                        **dict(zip(names, (w["text"] for w in info["windows"]), strict=True))}
                pending.append((dst, identity, text))
        phrases = list(dict.fromkeys(value for _, _, texts in pending for value in texts.values() if HAN.search(value)))
        features = dict(zip(phrases, embed_texts(phrases) if phrases else [], strict=True))
        for dst, identity, texts in pending:
            vectors = {name: features.get(value, np.full(TEXT_DIM, np.nan, np.float32)) for name, value in texts.items()}
            save_archive(dst, {"provenance": identity}, **vectors)
        done = min(offset + 6, len(keys))
        if done % 120 == 0 or done == len(keys):
            print(f"Temporal text {done}/{len(keys)} in {perf_counter() - started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("prepare", "embed"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    rows = read_rows(args.data)
    splits = SPLITS if args.split == "all" else (args.split,)
    keys = [key for phase in splits for key in ranked_ids(rows, phase)]
    if args.step == "prepare":
        prepare(args.cache, keys, args.device)
    else:
        embed(args.cache, keys)


if __name__ == "__main__":
    main()
