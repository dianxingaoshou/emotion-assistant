"""Replay held-out raw RAVDESS clips after removing a contiguous time block of one stream."""

import argparse
from collections import Counter, defaultdict
from copy import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emotion_assistant.paths import MODELS, configure_environment
from emotion_assistant.calibration import LABELS
from emotion_assistant.core import FusionEngine
from emotion_assistant.fusion_policy import FusionConfig
from emotion_assistant.local_media import _file_intervals, _mean_scores
from emotion_assistant.models import ModelHub
from scripts.benchmark_missing_modalities import load_corpus


RATE = 16000


def read_raw(path, hub):
    import av
    import numpy as np

    with av.open(str(path)) as src:
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=RATE)
        chunks = [
            out.to_ndarray().ravel() for frame in src.decode(audio=0) for out in resampler.resample(frame)
        ]
        chunks.extend(out.to_ndarray().ravel() for out in resampler.resample(None))
    wave = np.concatenate(chunks).astype(np.float32)
    duration = len(wave) / RATE
    visual = []
    idx = 0
    with av.open(str(path)) as src:
        for frame in src.decode(video=0):
            a, b = 0.5 * idx, min(duration, 0.5 * (idx + 1))
            if a >= b:
                break
            if frame.time is None or frame.time + 1e-7 < (a + b) / 2:
                continue
            obs = hub.analyze_frame(frame.to_ndarray(format="bgr24"), path.stem, a, b)
            if obs["reason"].startswith(("inference_error", "model_unavailable")):
                raise RuntimeError(obs["reason"])
            obs["sample_time_s"] = float(frame.time)
            visual.append(obs)
            idx += 1
    return wave, visual


def mask_visual(observations, start, end):
    """Retain only the visible portion of a bin that contains its actual sampled frame."""
    remaining = []
    for obs in observations:
        stamp = obs["sample_time_s"]
        if start <= stamp < end:
            continue
        clipped = copy(obs)
        if stamp < start:
            clipped["t_end"] = min(obs["t_end"], start)
        else:
            clipped["t_start"] = max(obs["t_start"], end)
        if clipped["t_start"] < clipped["t_end"]:
            remaining.append(clipped)
    return remaining


def acoustic_chunks(wave, spans, cache, hub, identifier):
    observations = []
    for lo, hi in spans:
        if hi <= lo:
            continue
        if (lo, hi) not in cache:
            obs = hub.analyze_audio_emotion(wave[lo:hi], identifier, lo / RATE, hi / RATE)
            if obs["reason"].startswith(("inference_error", "model_unavailable")):
                raise RuntimeError(obs["reason"])
            obs["eligible"] = bool(obs.get("valid") and obs["scores"] and obs["quality"] >= 0.2)
            cache[lo, hi] = obs
        observations.append(cache[lo, hi])
    return observations


def matched_scores(visual, audio):
    results = []
    for enabled in (False, True):
        engine = FusionEngine(config=FusionConfig(adapt_single_modality=enabled))
        intervals = _file_intervals(engine, visual, audio, True)
        scores = _mean_scores(intervals, duration_weighted=True)
        covered = sum(r["t_end"] - r["t_start"] for r in intervals if r.get("scores"))
        results.append((scores, covered))
    return results


def summarize(rows):
    import numpy as np

    output = {
        "cases": len(rows),
        "total_seconds": sum(r["duration"] for r in rows),
        "removed_source_seconds": sum(r["removed"] for r in rows),
        "visual_frames_before": sum(r["n_frames"] for r in rows),
        "visual_frames_retained": sum(r["kept_frames"] for r in rows),
        "raw_audio_chunks_retained": sum(r["chunks"] for r in rows),
        "eligible_acoustic_chunks": sum(r["eligible_audio"] for r in rows),
        "rejected_audio_by_reason": dict(sum((Counter(r["audio_reasons"]) for r in rows), Counter())),
    }
    for index, name in enumerate(("before", "after")):
        matrix = np.zeros((4, 5), dtype=int)
        loss, scored_seconds = [], 0.0
        for r in rows:
            scores, seconds = r["results"][index]
            truth = LABELS.index(r["label"])
            predicted = LABELS.index(max(scores, key=scores.get)) if scores else 4
            matrix[truth, predicted] += 1
            if scores:
                loss.append(sum((scores[k] - (k == r["label"])) ** 2 for k in LABELS))
            scored_seconds += seconds
        diag = matrix[:, :4].diagonal()
        den = matrix.sum(axis=1) + matrix[:, :4].sum(axis=0)
        output[name] = {
            "scored_clips": int(matrix[:, :4].sum()),
            "coverage_clips": float(matrix[:, :4].sum() / len(rows)),
            "scored_fraction_of_timeline": scored_seconds / output["total_seconds"],
            "accuracy_all": float(diag.sum() / len(rows)),
            "macro_f1_all": float(np.divide(2 * diag, den, out=np.zeros(4), where=den > 0).mean()),
            "brier_scored": float(np.mean(loss)) if loss else None,
            "confusion_truth_by_pred_plus_abstention": matrix.tolist(),
        }
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, default=ROOT / "reports/ravdess")
    p.add_argument("--media", type=Path, default=ROOT / ".runtime/ravdess")
    p.add_argument("--report", type=Path, default=ROOT / "docs/results/continuous-gaps.json")
    args = p.parse_args()
    configure_environment()
    import cv2

    cv2.setNumThreads(2)
    hub = ModelHub(MODELS, enable_asr=False)
    state = {**hub.load_visual(), **hub.load_audio_emotion()}
    if not all(x["ready"] for x in state.values()):
        raise RuntimeError(state)
    selected = [(i, a, l) for i, a, l, c in load_corpus(args.dir)["test"] if len(c) == 2]
    media_hashes = {
        r["id"]: r["sha256"]
        for r in map(json.loads, (args.dir / "features.jsonl").read_text("utf-8").splitlines())
    }
    by_condition = defaultdict(list)
    for n, (identifier, actor, label) in enumerate(selected, 1):
        path = args.media / (identifier + ".mp4")
        if hashlib.sha256(path.read_bytes()).hexdigest() != media_hashes[identifier]:
            raise ValueError(f"media hash mismatch: {identifier}")
        wave, visual = read_raw(path, hub)
        length = len(wave)
        cache = {}

        def record(name, observations, spans, removed):
            audio = acoustic_chunks(wave, spans, cache, hub, identifier)
            by_condition[name].append(
                {
                    "actor": actor,
                    "label": label,
                    "duration": length / RATE,
                    "removed": removed / RATE,
                    "n_frames": len(visual),
                    "kept_frames": len(observations),
                    "chunks": len(audio),
                    "eligible_audio": sum(o["eligible"] for o in audio),
                    "audio_reasons": [
                        o["reason"] if not o["valid"] else "below_min_quality"
                        for o in audio
                        if not o["eligible"]
                    ],
                    "results": matched_scores(observations, audio),
                }
            )

        record("intact", visual, [(0, length)], 0)
        for ratio in (0.25, 0.5):
            size = round(length * ratio)
            for part, lo in (("early", 0), ("middle", (length - size) // 2), ("late", length - size)):
                hi = lo + size
                suffix = f"{int(ratio * 100)}_{part}"
                record("visual_" + suffix, mask_visual(visual, lo / RATE, hi / RATE), [(0, length)], size)
                record("audio_" + suffix, visual, [(0, lo), (hi, length)], size)
        if n % 8 == 0 or n == len(selected):
            print(f"{n}/{len(selected)} raw held-out clips", flush=True)
    all_masks = [r for key, items in by_condition.items() if key != "intact" for r in items]
    doc = {
        "protocol": "contiguous-raw-stream-gap-v1",
        "source": "RAVDESS 10.5281/zenodo.1188976, audiovisual human votes PLOS S5",
        "labels": list(LABELS),
        "heldout_unique_clips": len(selected),
        "heldout_actor_ids": sorted({a for _, a, _ in selected}),
        "class_counts": dict(Counter(l for _, _, l in selected)),
        "method": "One fixed 25% or 50% contiguous block at the beginning, middle, or end, in exactly one stream. Visual cue uses the actual PTS of each 0.5s sampled frame; rejected source frames never enter remaining bins. Audio uses physically excised 16 kHz raw samples: run emotion2vec independently on both remaining contiguous pieces, no cross-gap concatenation or whole-utterance reuse. Segments shorter than 2s cannot enter fusion. Same native time intervals, quality gates and duration-weighted clip aggregation for both versions. Fixed project baseline quality_v2 vs bundled single-cue adaptation, no refit.",
        "scope": "62 paired research clips selected before masking; other covers silence, scripted English utterances and eight acting subjects. No live VAD segmentation; released utterance defines the acoustic segment. Predict whole-clip audiovisual human perception throughout. Averages over repeated masks of the same clips do not constitute independent subjects or new samples.",
        "metrics": "Accuracy and four-class macro-F1 include unscored clips as missed; Brier only on scored clips. Timeline coverage is duration with a score over entire source duration. Scored labels include descriptive probabilities from decision-level abstentions.",
        "conditions": {k: summarize(v) for k, v in by_condition.items()},
        "pooled_twelve_masks": summarize(all_masks),
        "pooled_by_masked_channel": {
            key: summarize([r for n, rr in by_condition.items() if n.startswith(key + "_") for r in rr])
            for key in ("audio", "visual")
        },
    }
    args.report.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n", "utf-8")
    print(json.dumps(doc["pooled_twelve_masks"]))


if __name__ == "__main__":
    main()
