"""Re-encode locally damaged CH-SIMS evidence for the missing-modality experiment."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
import logging
from pathlib import Path
import re
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.chsims_face_teacher import face_crops
from scripts.chsims_teacher_data import CACHE, FRAME_PERIOD, SPLITS, ranked_ids, read_rows, sha_file, write_atomic


ROOT = Path(__file__).resolve().parents[1]
FRACTION = .4
SEED = "chsims-v2-teacher-local-gap-01"
TEXT_UNIT = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*|[^\W_]")
RATE = 16000
PREPARED = "missing-evidence/inputs"
FEATURES = "missing-evidence/gap-teacher"
DESKTOP = "missing-evidence/gap-desktop"


def random_position(key, evidence):
    digest = hashlib.sha256(f"{SEED}|{key}|{evidence}".encode("utf-8")).digest()
    return int.from_bytes(digest[:6], "big") / 2**48


def text_units(text):
    return list(TEXT_UNIT.finditer(text))


def drop_text(text, position):
    tokens = text_units(text)
    if not tokens:
        return "", 0, 0
    lost = max(1, round(FRACTION * len(tokens)))
    begin = min(len(tokens) - lost, int(position * (len(tokens) - lost + 1)))
    end = begin + lost
    remaining = text[:tokens[begin].start()] + " " + text[tokens[end - 1].end():]
    return (remaining.strip() if lost < len(tokens) else ""), lost, len(tokens)


def timed_gap(duration, position):
    start = duration * (.1 + .4 * position)
    return start, start + duration * FRACTION


def cut_tracks(waveform, frames, times, gap):
    left, right = gap
    low = min(len(waveform), max(0, round(left * RATE)))
    high = min(len(waveform), max(low, round(right * RATE)))
    remaining = waveform.copy()
    remaining[low:high] = 0.
    keep = (times < left) | (times >= right)
    fragments = [(a / RATE, waveform[a:b]) for a, b in ((0, low), (high, len(waveform))) if b > a]
    return remaining, frames[keep], keep, fragments, high - low


def save_archive(path, info, **arrays):
    out = io.BytesIO()
    np.savez(out, info=json.dumps(info, ensure_ascii=False, allow_nan=False), **arrays)
    write_atomic(path, out.getvalue())


def with_cache(destination, identity, key):
    if not destination.is_file():
        return False
    with np.load(destination, allow_pickle=False) as previous:
        return json.loads(previous["info"].item()).get(key) == identity


def speech_inputs(fragments, duration, hub, vad, quality):
    from emotion_assistant.clip_inputs import clip_inputs
    from emotion_assistant.file_tracks import speech_segments

    speech, seconds, observed, transcripts = np.zeros(768, np.float64), 0., [], []
    for timestamp, samples in fragments:
        vad.reset()
        intervals = speech_segments([(timestamp, samples)], vad, lambda: None)
        for wave, start, end, details in intervals:
            result = hub.analyze_audio_emotion(wave, "teacher-gap", start, end, details)
            check_result(result)
            observed.append({**result, "eligible": result["valid"] and result["quality"] >= quality})
            if start < 30 and len(wave) >= RATE * .2:
                vector = hub.audio_representation(wave, start, end)
                if vector is not None:
                    speech += (end - start) * vector
                    seconds += end - start
            record = hub.transcribe_audio(wave, "teacher-gap", start, end, asr_allowed=True)
            check_result(record)
            transcripts.append(record)
    features = clip_inputs({"audio_samples": observed, "transcripts": transcripts,
                            "timeline_coverage": {"duration_s": duration}})
    embedding = (speech / seconds).astype(np.float32) if seconds else np.empty(0, np.float32)
    return features, embedding


def check_result(observation):
    if observation["reason"].startswith(("model_unavailable", "inference_error")):
        raise RuntimeError(observation["reason"])


def visual_inputs(video, times, keep, duration, gap, hub, quality):
    from emotion_assistant.clip_inputs import clip_inputs

    observed = []
    retained = np.flatnonzero(keep)

    def observe(image, session, index, _end):
        position = retained[index]
        stamp = times[position]
        end = times[position + 1] if position + 1 < len(times) else duration
        if stamp < gap[0]:
            end = min(end, gap[0])
        result = hub.analyze_frame(image, session, stamp, stamp + .001)
        check_result(result)
        result.update(t_end=min(end, stamp + FRAME_PERIOD),
                      eligible=result["valid"] and result["quality"] >= quality)
        observed.append(result)
        return result

    faces, _ = face_crops(video, observe)
    result = clip_inputs({"samples": observed, "timeline_coverage": {"duration_s": duration}})
    return result["visual"], faces


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
        raise RuntimeError("Select the isolated CUDA environment for GPU gap preprocessing")

    class GapHub(ModelHub):
        def _load_funasr(self, folder, required):
            model = super()._load_funasr(folder, required)
            if device == "cuda":
                model.model.to(device)
                model.kwargs["device"] = device
            return model

    hub = GapHub(MODELS)
    status = {**hub.load_visual(), **hub.load_audio()}
    if not all(value["ready"] for value in status.values()):
        raise RuntimeError(f"Gap preprocessing models unavailable: {status}")
    vad = SileroVAD(MODELS / "vad/silero_vad.onnx")
    quality = FusionEngine.from_project(DATA).min_quality
    identity = {"code": sha_file(Path(__file__)), "models": VERSIONS, "assets": sha_file(MODELS / "manifest.json"),
                "device": device, "min_quality": quality, "torch": torch.__version__}
    started = perf_counter()
    for index, key in enumerate(keys, 1):
        source = cache / "media" / f"{key}.npz"
        path = cache / PREPARED / f"{key}.npz"
        receipt = {**identity, "input": sha_file(source)}
        if not with_cache(path, receipt, "provenance"):
            with np.load(source, allow_pickle=False) as data:
                info = json.loads(data["info"].item())
                pcm, video = data["pcm"], data["video"]
            duration = info["duration_s"]
            times = np.asarray(info["frame_times"])
            gap = timed_gap(duration, random_position(key, "timed"))
            audio, video, keep, fragments, samples_lost = cut_tracks(pcm, video, times, gap)
            if not len(audio) or np.abs(audio).max() < .002:
                audio = np.empty(0, dtype=np.float32)
            text, units_lost, units_total = drop_text(info["text"], random_position(key, "text"))
            spoken, speech = speech_inputs(fragments, duration, hub, vad, quality)
            visual, faces = visual_inputs(video, times, keep, duration, gap, hub, quality)
            result = {"provenance": receipt, "text": text, "joint_text": spoken["text"], "audio": spoken["audio"],
                      "visual": visual, "duration": duration, "gap": list(gap), "pcm_lost": samples_lost,
                      "pcm_total": len(pcm), "frames_lost": int((~keep).sum()), "frames_total": len(times),
                      "frame_times": times[keep].tolist(), "text_lost": units_lost, "text_total": units_total,
                      "faces": len(faces)}
            save_archive(path, result, pcm=audio, video=video, face=faces, speech=speech)
        if index % 128 == 0 or index == len(keys):
            print(f"Gap inputs {index}/{len(keys)} in {perf_counter() - started:.1f}s", flush=True)


def encode(cache, keys, batch_size):
    from scripts.chsims_omni_teacher import OmniTeacher

    logging.getLogger().addFilter(
        lambda record: not record.getMessage().startswith("System prompt modified, audio output"))
    model = OmniTeacher(CACHE / "omni-3b")
    started = perf_counter()
    for start in range(0, len(keys), batch_size):
        entries, batches = {}, defaultdict(list)
        for key in keys[start:start + batch_size]:
            source = cache / PREPARED / f"{key}.npz"
            path = cache / FEATURES / f"{key}.npz"
            identity = {"model": model.identity, "input": sha_file(source)}
            if with_cache(path, identity, "provenance"):
                continue
            with np.load(source, allow_pickle=False) as data:
                info = json.loads(data["info"].item())
                views = {"text": ("text", info["text"]), "joint_text": ("text", info["joint_text"]),
                         "audio": ("audio", data["pcm"]), "visual": ("video", data["video"]),
                         "face": ("video", data["face"])}
            for name, (channel, value) in views.items():
                if len(value):
                    batches[name].append((key, {channel: value}))
            entries[key] = (path, {"provenance": identity, "seen": {}}, {})
        for name, batch in batches.items():
            view = {"joint_text": "text", "face": "visual"}.get(name, name)
            scores, vectors = model.score(view, [item for _, item in batch], capture=True)
            for (key, _), score, features in zip(batch, scores, vectors, strict=True):
                _, identity, blocks = entries[key]
                identity["seen"][name] = score["seen"]
                blocks.update({f"{name}_{field}": value for field, value in features.items()})
        for path, info, data in entries.values():
            save_archive(path, info, **data)
        done = min(start + batch_size, len(keys))
        if done % 128 < batch_size or done == len(keys):
            print(f"Gap teacher {done}/{len(keys)} in {perf_counter() - started:.1f}s", flush=True)


def score_desktop(cache, keys):
    from emotion_assistant.chinese_affect import HAN
    from emotion_assistant.foundation_embeddings import TEXT_DIGEST, embed_texts
    from emotion_assistant.foundation_fusion import ASSET, predict_foundation_clip
    from scripts.calibrate_chsims_teacher import log_ratios

    receipt = {"text_digest": TEXT_DIGEST, "asset": sha_file(ASSET), "code": sha_file(Path(__file__))}
    started = perf_counter()
    for offset in range(0, len(keys), 12):
        inputs, transcripts = [], []
        for key in keys[offset:offset + 12]:
            source = cache / PREPARED / f"{key}.npz"
            path = cache / DESKTOP / f"{key}.npz"
            identity = {**receipt, "input": sha_file(source)}
            if with_cache(path, identity, "provenance"):
                continue
            with np.load(source, allow_pickle=False) as data:
                info, speech = json.loads(data["info"].item()), data["speech"]
            inputs.append((key, path, identity, info, speech))
            transcripts.extend((key, view, info[view]) for view in ("text", "joint_text") if HAN.search(info[view]))
        vectors = embed_texts([text for _, _, text in transcripts]) if transcripts else []
        embeddings = {(key, view): vector for (key, view, _), vector in zip(transcripts, vectors, strict=True)}
        for key, path, identity, info, speech in inputs:
            predictions = {"text_desk": predict_foundation_clip(info["text"], embedding=embeddings.get((key, "text"))),
                           "joint_text_desk": predict_foundation_clip(info["joint_text"],
                                                                     embedding=embeddings.get((key, "joint_text"))),
                           "audio_desk": predict_foundation_clip("", audio=info["audio"],
                                                                 audio_embedding=speech if speech.size else None),
                           "visual_desk": predict_foundation_clip("", visual=info["visual"])}
            blocks = {name: vector for name, outcome in predictions.items()
                      if (vector := log_ratios(outcome)) is not None}
            for view in ("audio", "visual"):
                cue = info[view]
                if cue is not None:
                    blocks[f"{view}_expert_scores"] = log_ratios(cue)
                    blocks[f"{view}_expert_quality"] = np.asarray(
                        [cue["quality"], cue["coverage"], cue["scores"]["uncertain"]])
            if speech.size:
                blocks["audio_speech"] = speech
            save_archive(path, {"provenance": identity}, **blocks)
        done = min(offset + 12, len(keys))
        if done % 132 < 12 or done == len(keys):
            print(f"Gap desktop {done}/{len(keys)} in {perf_counter() - started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("prepare", "encode", "desktop"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Use positive sizes")
    rows = read_rows(args.data)
    splits = SPLITS if args.split == "all" else (args.split,)
    keys = [key for split in splits for key in ranked_ids(rows, split)][:args.limit]
    if args.step == "prepare":
        prepare(args.cache, keys, args.device)
    elif args.step == "encode":
        encode(args.cache, keys, args.batch_size)
    else:
        score_desktop(args.cache, keys)


if __name__ == "__main__":
    main()
