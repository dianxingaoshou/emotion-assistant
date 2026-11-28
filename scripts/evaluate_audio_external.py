"""Evaluate frozen audio models on the authors' Mandarin MCAE-SPPS recordings."""
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import gc
import hashlib
import io
import json
import logging
from pathlib import Path
import re
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1] / ".venv/Lib/site-packages"))
import numpy as np
from scipy.special import softmax
from scripts.benchmark_chsims_teacher import save_json
from scripts.calibrate_chsims_teacher import log_ratios
from scripts.chsims_audio_prior import audio_logits
from scripts.chsims_teacher_cross_source import labels, metrics, mix
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file, write_atomic
from scripts.chsims_teacher_supervision import compare


ARCHIVES = {
    "anger": (10, "negative", "ec0e652a3a85a38cefac3e184ecb5d2a497e093ad66a6260d4243b154dc3cc5d"),
    "sadness": (10, "negative", "cb9f37c7a2d5ec9469fd92a84b884d8b0627d9d3396dbb2c780cde53c1a47275"),
    "neutral": (20, "neutral", "1c1670f0296a0bf75516784d0579ba2ddb11251baa916b6978cfb21ac9e53bf0"),
    "happiness": (20, "positive", "59f708d5c93ad209e302f533d90acc01ae2ab7efcb3c9278b1c89bf8ca6fbd05"),
}
RATINGS = (
    "9d91a2f900c2386216cd22697c9986db6b4bab7c5538020c42020bee42367027",
    "48b0a3703cad045f2220e1a5ce527bc50360d6a2c5cd55af138f2bb5e74cc0a7",
    "19218d4355314872dc1f6273ea513dd5d7ce56a5e2cc56bdba3b250c104a9b53",
    "fe47b7c746a22be28ccfe723a00f5732305c21f01a036c3a748df2c04e74c02f",
    "a38defa145705b3307b5dc940e58ea2c1b0667fa2180463f1969dce97baa1e6b",
    "928404f6185e7ead32648a5a9c88210304396dae72121940efbb40dfa37ebb15",
)
SELECTION = "mcae-spps-2026-audio-only-1"
MODEL_REVIEW_WARNING = "System prompt modified, audio output"


def cohort(directory, archives):
    selections = []
    for actor in range(1, 7):
        votes = {}
        with (directory / f"ratings-{actor}.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                votes.setdefault(row["filename"], Counter())[row["emotionresult"]] += 1
        for emotion, archive in archives.items():
            count, label, _ = ARCHIVES[emotion]
            choices = []
            for name in archive.namelist():
                match = re.fullmatch(rf"{emotion}/({actor}\d{{4}})\.wav", name)
                if match is None:
                    continue
                stem = match[1]
                rating = votes.get(stem, Counter())
                total = sum(rating.values())
                if total < 30 or rating[emotion] <= total / 2:
                    continue
                order = hashlib.sha256(f"{SELECTION}|{emotion}|{stem}".encode("ascii")).digest()
                choices.append((order, name, stem, rating[emotion], total))
            if len(choices) < count:
                raise ValueError(f"Insufficient majority-rated samples for actor {actor}, {emotion}")
            for _, file, stem, agree, total in sorted(choices)[:count]:
                selections.append({"id": f"actor{actor}/{stem}", "emotion": emotion, "member": file,
                                   "truth": {"audio": label}, "ratings": [agree, total]})
    if len({row["id"] for row in selections}) != len(selections):
        raise ValueError("Repeated external sample")
    return selections


def decode(wave_bytes):
    import av

    resample = av.AudioResampler(format="fltp", layout="mono", rate=16000)
    samples = []
    with av.open(io.BytesIO(wave_bytes)) as media:
        for frame in media.decode(audio=0):
            samples.extend(p.to_ndarray().reshape(-1) for p in resample.resample(frame))
        samples.extend(p.to_ndarray().reshape(-1) for p in resample.resample(None))
    pcm = np.concatenate(samples).astype(np.float32, copy=False)
    if not np.isfinite(pcm).all() or len(pcm) < 400:
        raise ValueError("Invalid external recording")
    return pcm


def prepare(directory, cache):
    import torch
    from emotion_assistant.capture import SileroVAD
    from emotion_assistant.clip_inputs import clip_inputs
    from emotion_assistant.core import FusionEngine
    from emotion_assistant.file_tracks import speech_segments
    from emotion_assistant.models import ModelHub
    from emotion_assistant.paths import DATA, MODELS, configure_environment

    configure_environment()
    hub = ModelHub(MODELS, enable_asr=False)
    if not hub.load_audio_emotion()["audio"]["ready"]:
        raise RuntimeError("The desktop voice encoder is unavailable")
    torch.set_num_threads(2)
    vad = SileroVAD(MODELS / "vad/silero_vad.onnx")
    minimum = FusionEngine.from_project(DATA).min_quality
    with ExitStack() as opened:
        archives = {}
        for emotion, (_, _, digest) in ARCHIVES.items():
            path = directory / f"{emotion}.zip"
            if sha_file(path) != digest:
                raise ValueError(f"Incorrect OSF archive {path}")
            archives[emotion] = opened.enter_context(ZipFile(path))
        rating_hashes = {str(i): sha_file(directory / f"ratings-{i}.csv") for i in range(1, 7)}
        if tuple(rating_hashes.values()) != RATINGS:
            raise ValueError("The external listener ratings differ from the published checksums")
        rows = cohort(directory, archives)
        receipt = {"selection": SELECTION, "archive_sha256": {k: value[2] for k, value in ARCHIVES.items()},
                   "ratings_sha256": rating_hashes,
                   "desktop_assets": sha_file(MODELS / "manifest.json"), "rows": rows}
        save_json(cache / "cohort.json", receipt)
        for index, row in enumerate(rows, 1):
            dst = cache / "media" / f"{row['id']}.npz"
            if dst.is_file():
                continue
            pcm = decode(archives[row["emotion"]].read(row["member"]))
            vad.reset()
            vectors, audio, seconds = np.zeros(768), [], 0.
            for samples, left, right, quality in speech_segments([(0., pcm)], vad, lambda: None):
                observation = hub.analyze_audio_emotion(samples, row["id"], left, right, vad_metrics=quality)
                if observation["reason"].startswith("inference_error"):
                    raise RuntimeError(observation["reason"])
                observation["eligible"] = observation["valid"] and observation["quality"] >= minimum
                audio.append(observation)
                vector = hub.audio_representation(samples, left, right)
                if vector is not None:
                    vectors += (right - left) * vector
                    seconds += right - left
            features = clip_inputs({"audio_samples": audio, "timeline_coverage": {"duration_s": len(pcm) / 16000}})
            stream = io.BytesIO()
            np.savez(stream, pcm=pcm, speech=(vectors / seconds).astype(np.float32) if seconds else np.empty(0, np.float32),
                     info=json.dumps(features))
            write_atomic(dst, stream.getvalue())
            if index % 60 == 0 or index == len(rows):
                print(f"External desktop evidence {index}/{len(rows)}", flush=True)


def encode(cache):
    import torch
    from scripts.chsims_minicpmo_audio import MiniCPMAudio, extract
    from scripts.chsims_omni_audio_encoder import AudioTower, encode_batch
    from scripts.chsims_omni_teacher import OmniTeacher

    rows = json.loads((cache / "cohort.json").read_text("utf-8"))["rows"]
    keys = [row["id"] for row in rows]
    waves = {}
    for key in keys:
        with np.load(cache / "media" / f"{key}.npz", allow_pickle=False) as asset:
            waves[key] = asset["pcm"]
    tower = AudioTower(CACHE / "omni-3b")
    for start in range(0, len(keys), 8):
        encode_batch(tower, [(waves[key], cache / "audio-tokens" / f"{key}.npz") for key in keys[start:start+8]])
    del tower
    gc.collect()
    torch.cuda.empty_cache()
    logging.getLogger().addFilter(lambda item: not item.getMessage().startswith(MODEL_REVIEW_WARNING))
    model = OmniTeacher(CACHE / "omni-3b")
    for start in range(0, len(keys), 8):
        batch = keys[start:start+8]
        batch = [key for key in batch if not (cache / "representations" / f"{key}.npz").is_file()]
        if not batch:
            continue
        scores, features = model.score("audio", [{"audio": waves[key]} for key in batch], capture=True)
        for key, value, score in zip(batch, features, scores, strict=True):
            stream = io.BytesIO()
            np.savez(stream, audio_acoustic=value["acoustic"], audio_context=value["context"],
                     info=json.dumps(score))
            write_atomic(cache / "representations" / f"{key}.npz", stream.getvalue())
        if (start + 8) % 80 == 0 or start + 8 >= len(keys):
            print(f"External Omni evidence {min(start + 8, len(keys))}/{len(keys)}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    extract(MiniCPMAudio(CACHE / "minicpm-o-4_5"), cache, keys, 8)


def desktop_blocks(cache, clips, rows):
    from emotion_assistant.foundation_fusion import predict_foundation_clip

    scores = np.full((len(rows), 2), np.nan, np.float32)
    speech = np.full((len(rows), 768), np.nan, np.float32)
    cue_scores = np.full((len(rows), 2), np.nan, np.float32)
    cue_quality = np.full((len(rows), 3), np.nan, np.float32)
    raw = []
    for i, (row, clip) in enumerate(zip(rows, clips, strict=True)):
        with np.load(cache / "media" / f"{row['id']}.npz", allow_pickle=False) as cache_file:
            info = json.loads(cache_file["info"].item())
        vector = clip.desktop[:768] if clip.desktop[-1] else None
        result = predict_foundation_clip("", audio=info["audio"], audio_embedding=vector)
        raw.append(result["label"] if result["label"] in LABELS else "abstain")
        if result["scores"]:
            scores[i] = log_ratios(result)
        if vector is not None:
            speech[i] = vector
        if info["audio"] is not None:
            observed = info["audio"]
            cue_scores[i] = log_ratios(observed)
            cue_quality[i] = (observed["quality"], observed["coverage"], observed["scores"]["uncertain"])
    return {"scores": scores, "speech": speech, "expert_scores": cue_scores, "expert_quality": cue_quality}, raw


def evaluate(cache, models, output):
    from scripts.chsims_audio_lora import custom_predictions, read_clip
    from scripts.chsims_audio_review_gate import from_blocks as review_audio
    from scripts.chsims_minicpmo_audio import ASSETS as MINI_ASSETS
    from scripts.chsims_omni_model_comparison import predict as compare_predict, read_minicpm
    from scripts.chsims_omni_audio_encoder import AudioTower

    rows = json.loads((cache / "cohort.json").read_text("utf-8"))["rows"]
    tower = AudioTower(CACHE / "omni-3b")
    clips = [read_clip(cache, row["id"]) for row in rows]
    probabilities, model = custom_predictions(models, clips, tower)
    legacy = json.loads((models / "teacher-supervision.json").read_text("utf-8"))["heads"]["audio"]
    prior = json.loads((models / "audio-prior-calibration.json").read_text("utf-8"))
    raw_ratios, acoustic = [], []
    for row in rows:
        with np.load(cache / "representations" / f"{row['id']}.npz", allow_pickle=False) as asset:
            raw_ratios.append(log_ratios(json.loads(asset["info"].item())))
            acoustic.append(asset["audio_acoustic"])
    context = {"scores": np.stack(raw_ratios), "context": np.stack([clip.context for clip in clips]),
               "acoustic": np.stack(acoustic)}
    indices = np.arange(len(rows))
    omni = audio_logits(context, legacy["omni_selected"], indices)
    probabilities["qwen_prior"] = softmax((omni + prior["correction"]) / prior["temperature"]["omni_prior"], axis=1)
    desktop, raw = desktop_blocks(cache, clips, rows)
    for name in ("desktop_calibrated", "desktop_acoustic"):
        head = legacy[name]
        probabilities[name] = softmax(audio_logits(desktop, head, indices) / head["temperature"], axis=1)
    probabilities["lora_and_desktop"] = mix(probabilities, model["blend"])
    review, _ = review_audio(models, desktop, context)
    probabilities.update({f"qwen_{name}": value for name, value in review.items()})
    comparison_path = models / "omni-comparison-audio.json"
    matched_models = json.loads(comparison_path.read_text("utf-8"))
    if matched_models["mini"] != MINI_ASSETS:
        raise ValueError("Independent scores require the corresponding MiniCPM-o checkpoint")
    blocks = {"audio_desk": desktop["scores"], "audio_context": context["context"],
              "audio_acoustic": context["acoustic"], "audio_speech": desktop["speech"],
              "audio_expert_scores": desktop["expert_scores"], "audio_expert_quality": desktop["expert_quality"],
              **read_minicpm(cache, rows)}
    probabilities.update({f"comparison_{arm}": compare_predict(head, blocks, "all")
                          for arm, head in matched_models["heads"].items()})
    truth = np.array([LABELS.index(row["truth"]["audio"]) for row in rows])
    predictions = {name: labels(values) for name, values in probabilities.items()}
    predictions["desktop_raw"] = raw
    results = {name: compare(rows, "audio", raw, guess, indices)["candidate"]
               for name, guess in predictions.items()}
    for name, value in results.items():
        print(name, value["correct"], f"{value['macro_f1']:.4f}", value["recall"]["neutral"], flush=True)
    report = {"dataset": "MCAE-SPPS", "cohort_sha256": sha_file(cache / "cohort.json"),
              "comparison_sha256": sha_file(comparison_path),
              "n": len(rows), "results": results,
              "comparisons_vs_desktop_calibrated": {name: compare(rows, "audio", predictions["desktop_calibrated"],
                                                                guess, indices)
                                                    for name, guess in predictions.items()},
              "neutral_counts": {name: metrics(truth, values)["predicted_neutral"]
                                 for name, values in probabilities.items()}}
    save_json(output, report)
    save_json(cache / "predictions.json", {"ids": [row["id"] for row in rows], "labels": predictions})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("prepare", "encode", "evaluate"))
    parser.add_argument("--models", type=Path, default=CACHE / "full")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.models / "mcae-spps"
    cache = directory / "experiment"
    if args.step == "prepare":
        prepare(directory, cache)
    elif args.step == "encode":
        encode(cache)
    else:
        evaluate(cache, args.models, args.output or cache / "results.json")


if __name__ == "__main__":
    main()
