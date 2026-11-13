"""Real local model smoke check, with explicitly optional official sample download.

No device is opened. No accuracy, fusion gain, or intervention effect is measured.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import platform
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLES = {
    "lena.jpg": {
        "url": "https://raw.githubusercontent.com/opencv/opencv/4.11.0/samples/data/lena.jpg",
        "sha256": "7de7ed51a1594fff247f4cae2301eceacf5313d6011e37b4a4c8733f7bb72c07"},
    "emotion2vec_test.wav": {
        "url": "https://huggingface.co/emotion2vec/emotion2vec_plus_seed/resolve/7be9a30d52bdc793cc24e6f7d25358d6648927c1/example/test.wav",
        "official_mirror": "https://modelscope.cn/models/iic/emotion2vec_plus_seed/resolve/master/example/test.wav",
        "sha256": "a4839eaaa3d54bd2db6eb48aa3d40def1b5c5004df3fd163a8dcd045097f8a23"},
    "sensevoice_zh.mp3": {
        "url": "https://huggingface.co/FunAudioLLM/SenseVoiceSmall/resolve/3847d57b6bdf2dd8875cb1508d2af43d80a16bf7/example/zh.mp3",
        "official_mirror": "https://modelscope.cn/models/iic/SenseVoiceSmall/resolve/master/example/zh.mp3",
        "sha256": "0e64de19e4ff9a02e682955c9112f32d2317cfdbb5bc2f3504664044c993f195"},
}


def prepare_samples(directory: Path, allow_download=False):
    directory.mkdir(parents=True, exist_ok=True)
    for name, spec in SAMPLES.items():
        path = directory / name
        if not path.is_file():
            if not allow_download:
                raise FileNotFoundError(f"Missing official sample {path}. Use --download-samples explicitly to fetch it.")
            errors = []
            for source in dict.fromkeys([spec.get("official_mirror", spec["url"]), spec["url"]]):
                try:
                    with urllib.request.urlopen(source, timeout=60) as response:
                        data = response.read()
                    if hashlib.sha256(data).hexdigest() != spec["sha256"]:
                        raise ValueError("Official sample SHA256 mismatch")
                    path.write_bytes(data)
                    break
                except Exception as exc:
                    errors.append(str(exc))
            else:
                raise RuntimeError(f"Cannot download {name}: {errors}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError(f"Sample checksum mismatch: {path}")


def read_audio(path):
    import numpy as np
    import soundfile
    from scipy.signal import resample_poly
    samples, rate = soundfile.read(path, dtype="float32")
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if rate != 16000:
        divisor = math.gcd(int(rate), 16000)
        samples = resample_poly(samples, 16000 // divisor, int(rate) // divisor)
    return np.asarray(samples, np.float32), int(rate)


def vad_check(model_dir, speech):
    import numpy as np
    from emotion_assistant.capture import SileroVAD, SpeechSegmenter
    vad = SileroVAD(model_dir / "vad/silero_vad.onnx")

    def replay(samples):
        vad.reset()
        segmenter = SpeechSegmenter()
        probabilities, intervals = [], []
        for index in range(math.ceil(len(samples) / 512)):
            block = samples[index*512:(index+1)*512]
            if len(block) < 512:
                block = np.pad(block, (0, 512-len(block)))
            probability = vad(block)
            probabilities.append(probability)
            result = segmenter.feed(block, index * 512 / 16000, probability)
            if result:
                audio, start, end = result
                intervals.append({"t_start": start, "t_end": end,
                                  "sample_duration_s": len(audio)/16000})
        return {"input_duration_s": len(samples)/16000, "blocks": len(probabilities),
                "probability_min": min(probabilities), "probability_max": max(probabilities),
                "blocks_above_0_5": sum(p >= 0.5 for p in probabilities), "segments": intervals}

    silence = replay(np.zeros(32000, np.float32))
    speech_with_padding = replay(np.concatenate((np.zeros(4800, np.float32), speech, np.zeros(16000, np.float32))))
    assert not silence["segments"], "Silence unexpectedly produced a segment"
    assert speech_with_padding["segments"], "Official speech sample did not produce a segment"
    return {"model": "Silero VAD v5.1.2 ONNX", "silence": silence,
            "official_speech_with_0_3s_prefix_1s_suffix": speech_with_padding,
            "scope": "Measured interval outputs only; no human speech-boundary ground truth or VAD accuracy measured."}


def worker_check(model_dir, image, audio):
    """Exercise the same spawn target and IPC as runtime, without device capture."""
    import queue
    from emotion_assistant.capture import inference_process
    context = mp.get_context("spawn")
    output = context.Queue(64)
    stop = context.Event()
    permission = context.Event()
    permission.set()
    generation = context.Value("q", 0)
    consent_generation = context.Value("q", 0)
    jobs = {modality: context.Queue(2) for modality in ("visual", "audio", "asr")}
    processes = {}
    report = {"scope": "Independent official samples, separate session IDs; no cross-sample fusion or device capture.",
              "timestamp_source": "synthetic replay intervals; end-to-end durations are not a live benchmark",
              "worker_contract": "independent visual, acoustic-emotion and optional text workers",
              "queue_maxsize": {modality: 2 for modality in jobs},
              "loaded": {}, "observations": {}, "utterances": [], "exited": {}}
    try:
        for modality, data, end in (("visual", image, .5), ("audio", audio, len(audio)/16000),
                                    ("asr", audio, len(audio)/16000)):
            origin = time.monotonic()-max(10, end)-1
            process = context.Process(target=inference_process,
                args=(modality, str(model_dir), "official_"+modality+"_worker_smoke",
                      origin, jobs[modality], output, stop, modality == "asr",
                      permission, generation, consent_generation),
                name="model-smoke-"+modality)
            process.start()
            processes[modality] = process
            queued_at = time.monotonic()-origin
            jobs[modality].put(("official_"+modality+"_sample", data, 0.0, end, 0,
                                queued_at, 0, queued_at if modality != "visual" else None))
        deadline = time.monotonic()+120
        done = set()
        while len(done) < 3 and time.monotonic() < deadline:
            try:
                event = output.get(timeout=1)
            except queue.Empty:
                if any(process.exitcode is not None for process in processes.values()):
                    raise RuntimeError("Inference worker exited before delivering its result")
                continue
            kind = event.get("kind")
            if kind == "loaded":
                report["loaded"][event["modality"]] = event["status"]
            elif kind == "observation":
                observation = event["observation"]
                report["observations"][observation["modality"]] = observation
            elif kind == "utterance":
                report["utterances"].append(event["utterance"])
            elif kind == "done":
                done.add(event["modality"])
            elif kind == "error":
                raise RuntimeError(event.get("message", "worker error"))
        assert done == {"visual", "audio", "asr"}, "Inference worker IPC timed out"
        assert all(report["observations"].get(modality, {}).get("valid") for modality in ("visual", "audio"))
        assert report["loaded"]["audio"].keys() == {"audio"}, "Acoustic worker must not load ASR"
        assert report["loaded"]["asr"].keys() == {"asr"}, "ASR worker must not load the emotion model"
        assert report["utterances"] and report["utterances"][0]["text"] and report["utterances"][0]["valid"]
        assert "scores" not in report["utterances"][0], "Text must not become a third emotion vote"
        report["done"] = sorted(done)
    finally:
        stop.set()
        for process in processes.values():
            process.join(10)
            if process.is_alive():
                process.terminate()
                process.join(5)
        for modality, process in processes.items():
            report["exited"][modality] = {"exitcode": process.exitcode, "alive": process.is_alive()}
        for work_queue in (*jobs.values(), output):
            work_queue.close()
            work_queue.join_thread()
    assert all(item["exitcode"] == 0 and not item["alive"] for item in report["exited"].values())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-samples", action="store_true",
                        help="explicitly permit downloading the three pinned official samples")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/model-smoke.json")
    parser.add_argument("--workers", action="store_true",
                        help="also exercise the actual Windows spawn inference workers and IPC")
    args = parser.parse_args()
    from emotion_assistant.paths import configure_environment, MODELS, RUNTIME
    configure_environment()
    sample_dir = RUNTIME / "test_assets"
    report = {"schema_version": 1, "checked_at": datetime.now(timezone.utc).isoformat(),
              "python_version": platform.python_version(), "samples": SAMPLES, "checks": {},
              "limitations": ["Official samples only; no camera or microphone opened.",
                              "Single-call durations are smoke values, not a performance benchmark.",
                              "No accuracy, fusion benefit, intervention efficacy or real-device session acceptance measured."]}
    try:
        prepare_samples(sample_dir, args.download_samples)
        import cv2
        import numpy as np
        from emotion_assistant.models import ModelHub
        hub = ModelHub(MODELS)
        report["visual_load"] = hub.load_visual()
        image = cv2.imdecode(np.fromfile(sample_dir / "lena.jpg", np.uint8), cv2.IMREAD_COLOR)
        visual = hub.analyze_frame(image, "official_sample_smoke", 0, 0.5)
        report["visual"] = visual
        assert visual["valid"], visual["reason"]
        report["checks"]["visual_real_inference"] = True
        refusals = {}
        for label, frame in (("no_face", np.zeros_like(image)),
                             ("multiple_faces", np.concatenate([image, image], axis=1)),
                             ("small_image", cv2.resize(image, (60, 60)))):
            observation = hub.analyze_frame(frame, "official_sample_smoke", 0, .5)
            refusals[label] = {"valid": observation["valid"], "reason": observation["reason"]}
            assert not observation["valid"] and observation["scores"] is None
        report["visual_refusals"] = refusals
        report["checks"]["visual_refusals"] = True
        report["audio_load"] = hub.load_audio()
        assert all(status["ready"] for status in report["audio_load"].values()), report["audio_load"]
        for name, allowed in (("emotion2vec_test.wav", False), ("sensevoice_zh.mp3", True)):
            samples, source_rate = read_audio(sample_dir / name)
            observation, utterance = hub.analyze_audio(samples, "official_sample_smoke", 0,
                                                      len(samples)/16000, asr_allowed=allowed)
            report[name] = {"source_sample_rate": source_rate, "analysis_sample_rate": 16000,
                            "observation": observation, "utterance": utterance}
            assert observation["valid"], observation["reason"]
            assert abs(sum(observation["scores"].values())-1) < 1e-6
            if allowed:
                assert utterance and utterance["text"], "ASR did not return text"
            else:
                assert utterance is None
        report["checks"]["audio_real_inference"] = True
        report["checks"]["asr_real_inference"] = True
        speech, _ = read_audio(sample_dir / "emotion2vec_test.wav")
        report["vad"] = vad_check(MODELS, speech)
        report["checks"]["vad_real_inference_and_segmentation"] = True
        if args.workers:
            # Release direct models before loading another audio worker copy.
            import gc
            del hub
            gc.collect()
            worker_audio, _ = read_audio(sample_dir / "sensevoice_zh.mp3")
            report["workers"] = worker_check(MODELS, image, worker_audio)
            report["checks"]["spawn_workers_real_inference_and_clean_exit"] = True
        report["passed"] = True
    except Exception as exc:
        report["passed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({"passed": report["passed"], "checks": report["checks"],
                      "report": str(args.report), "error": report.get("error")}, ensure_ascii=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    mp.freeze_support()
    main()
