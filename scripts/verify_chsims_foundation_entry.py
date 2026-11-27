"""Replay the paired comparison through the same raw-MP4 entry as the desktop."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def keep_track(source, dest, track):
    import av
    with av.open(str(source)) as src, av.open(str(dest), "w") as out:
        original = getattr(src.streams, track)[0]
        stream = out.add_stream_from_template(original, opaque=True)
        for packet in src.demux(original):
            if packet.dts is not None:
                packet.stream = stream
                out.mux(packet)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / ".runtime/chsims/Raw")
    parser.add_argument("--adaptive", action="store_true", help="Replay the adaptive weights used in the desktop")
    parser.add_argument("--cohort", choices=("primary48", "audit48"), default="primary48")
    parser.add_argument("--study", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from emotion_assistant.paths import configure_environment, MODELS
    configure_environment()
    import cv2
    import numpy as np
    import torch
    from emotion_assistant.clip_recognition import analyze_foundation_media
    from emotion_assistant.foundation_fusion import ASSET
    from emotion_assistant.models import ModelHub

    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    results_dir = ROOT / "docs/results"
    study_path = args.study or results_dir / (
        "chsims-adaptive-fusion.json" if args.adaptive else "chsims-foundation-fusion.json")
    output_path = args.output or results_dir / (
        f"chsims-adaptive-entry-{args.cohort}.json" if args.adaptive else
        "chsims-foundation-entry.json" if args.cohort == "primary48" else "chsims-foundation-entry-audit48.json")
    study = json.loads(study_path.read_text(encoding="utf-8"))
    model_digest = hashlib.sha256(ASSET.read_bytes()).hexdigest()
    fit = study["fit"]
    if model_digest != fit["head_sha256" if args.adaptive else "model_sha256"]:
        raise ValueError("Different classifier from the one evaluated")
    if args.adaptive:
        from emotion_assistant.adaptive_clip_fusion import ASSET as ADAPTIVE_ASSET
        if hashlib.sha256(ADAPTIVE_ASSET.read_bytes()).hexdigest() != fit["model_sha256"]:
            raise ValueError("Different calibration parameters from the ones evaluated")
    cohort = study[args.cohort]
    hub = ModelHub(MODELS, enable_asr=True)
    counts = Counter()
    mismatches = []
    expert_mismatches = []
    largest_score_difference = 0.

    def analyze(path, transcribe):
        result = analyze_foundation_media(path, model_dir=MODELS, model_hub=hub, transcribe=transcribe,
                                          on_sample=lambda *_: None, cancelled=lambda: False, adaptive=args.adaptive)
        return result["foundation_clip"]

    for index, clip in enumerate(cohort["clips"], 1):
        path = args.raw / (clip["id"] + ".mp4")
        with path.open("rb") as stream:
            seed = int(hashlib.file_digest(stream, "sha256").hexdigest()[:8], 16)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        prediction = analyze(path, True)
        expected = clip["predictions"]["observed"]["foundation"]
        if prediction["label"] != expected:
            mismatches.append(clip["id"])
        cached = clip["observed_foundation"]
        actual_experts = prediction["experts"]
        same_experts = cached["experts"].keys() == actual_experts.keys() and all(
            ref["label"] == actual_experts[k]["label"] and abs(ref["weight"] - actual_experts[k]["weight"]) <= 1e-6
            for k, ref in cached["experts"].items())
        if (set(cached["scores"]) != set(prediction["scores"]) or not same_experts
                or cached["qwen_used"] != prediction.get("foundation_used")):
            expert_mismatches.append(clip["id"])
        for name, score in cached["scores"].items():
            if name in prediction["scores"]:
                largest_score_difference = max(largest_score_difference, abs(prediction["scores"][name] - score))
        counts["correct"] += prediction["label"] == clip["gold"]
        counts["qwen_used"] += bool(prediction.get("foundation_used"))
        counts["audio_vector_used"] += bool(prediction.get("audio_embedding_used"))
        if index % 12 == 0:
            print(f"Replayed {index} original videos", flush=True)
    matched = len(cohort["clips"]) - len(mismatches)
    passed = (matched == cohort["n"] == 48 and not expert_mismatches and largest_score_difference <= 1e-6
              and counts["qwen_used"] == cohort["qwen_text_embeddings_used"]
              and counts["audio_vector_used"] == cohort["audio_embeddings_used"]
              and counts["correct"] == cohort["conditions"]["observed"]["metrics"]["foundation"]["correct"])
    first = cohort["clips"][0]
    original = args.raw / (first["id"] + ".mp4")
    track_checks = {}
    with TemporaryDirectory(dir=ROOT / ".runtime") as folder:
        for condition, track, allowed in (("hide_text", None, {"audio", "visual"}),
                                          ("hide_visual", "audio", {"audio", "text"}),
                                          ("only_visual", "video", {"visual"})):
            path = original
            if track:
                path = Path(folder) / f"{track}.mp4"
                keep_track(original, path, track)
            prediction = analyze(path, condition != "hide_text")
            used = prediction["used_modalities"]
            expected = first["predictions"][condition]["foundation"]
            check = set(used) <= allowed and prediction["label"] == expected
            track_checks[condition] = {"modalities": used, "label": prediction["label"], "passed": check}
            passed = passed and check
    report = {"entry": "emotion_assistant.clip_recognition.analyze_foundation_media",
              "cohort": args.cohort, "clips": cohort["n"], "matched": matched, **counts,
              "mismatched_ids": mismatches, "expert_mismatches": expert_mismatches,
              "max_class_score_difference": largest_score_difference,
              "track_check_id": first["id"], "track_checks": track_checks,
              "passed": passed, "head_sha256": model_digest}
    if args.adaptive:
        report["adaptive_sha256"] = fit["model_sha256"]
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
