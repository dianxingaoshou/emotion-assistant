"""Read the supervised CH-SIMS v2 release and prepare isolated local evidence."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.chsims_samples import write_atomic

CACHE = ROOT / ".runtime/chsims-teacher"
SPLITS = ("train", "valid", "test")
VIEWS = {"text": ("text",), "audio": ("audio",), "visual": ("video",),
         "full": ("text", "audio", "video")}
COLUMNS = {"text": "label_T", "audio": "label_A", "visual": "label_V", "full": "label"}
LABELS = ("negative", "neutral", "positive")
FRAME_PERIOD = 0.5
MAX_FRAMES = 8
MAX_SECONDS = 60
SEED = "chsims-v2-omni-supervision-01"


def sha_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_rows(data):
    splits = {split: {} for split in SPLITS}
    seen = set()
    with (data / "meta.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            source, clip, split = row["video_id"], row["clip_id"], row["mode"]
            if not all(re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in (source, clip)):
                raise ValueError("Unexpected release clip identifier")
            key = f"{source}/{clip}"
            if key in seen or split not in splits:
                raise ValueError(f"Duplicate or invalid split for {key}")
            labels = {}
            for view, column in COLUMNS.items():
                number = float(row[column])
                if not -1 <= number <= 1:
                    raise ValueError(f"Invalid {column} for {key}")
                labels[view] = LABELS[1 + (number > 0) - (number < 0)]
            splits[split][key] = labels
            seen.add(key)
    if not all(splits.values()):
        raise ValueError("Supply the supervised release with all three author splits")
    return splits


def ranked_ids(rows, split):
    return sorted(rows[split], key=lambda key: hashlib.sha256(f"{SEED}|{split}|{key}".encode()).digest())


def media_path(data, key):
    return data / "Raw" / f"{key}.mp4"


def evidence_for(view, transcript, waveform, video):
    evidence = {"text": transcript, "audio": waveform, "video": video}
    return {key: evidence[key] for key in VIEWS[view] if len(evidence[key]) > 0}


def redact_subtitles(frame, detector):
    import cv2

    clean = frame.copy()
    height, width = clean.shape[:2]
    boxes, _ = detector(frame)
    pad = max(4, round(min(height, width) * .015))
    if boxes is not None:
        for box in boxes:
            x, y, w, h = cv2.boundingRect(box.astype(np.float32))
            clean[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad] = 0
    # Film captions in the lower band may be too small for the detector.
    clean[round(height * .72):] = 0
    return clean, len(boxes) if boxes is not None else 0


def model_frame(bgr):
    import cv2

    height, width = bgr.shape[:2]
    edge = 224
    scale = edge / max(height, width)
    height, width = max(1, round(height * scale)), max(1, round(width * scale))
    rgb = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)[:, :, ::-1]
    canvas = np.zeros((edge, edge, 3), np.uint8)
    top, left = (edge - height) // 2, (edge - width) // 2
    canvas[top:top + height, left:left + width] = rgb
    return canvas


def waveform_from_track(path, track, origin):
    from emotion_assistant.file_tracks import RATE, pcm_parts

    samples = np.zeros(MAX_SECONDS * RATE, np.float32)
    end = 0
    for stamp, pcm in pcm_parts(path, track, origin, lambda: None):
        start = round(stamp * RATE)
        left, right = max(0, start), min(len(samples), start + len(pcm))
        if right > left:
            samples[left:right] = pcm[left - start:right - start]
            end = max(end, right)
    if not end or np.max(np.abs(samples[:end])) < .002:
        return samples[:0]
    return samples[:end].copy()


def extract_clip(path, hub, detector, min_quality):
    from emotion_assistant.clip_inputs import clip_inputs
    from emotion_assistant.file_tracks import media_info, video_frames
    from emotion_assistant.foundation_embeddings import clip_audio_embedding
    from emotion_assistant.local_media import _analyze_frame, _audio_pass

    tracks, origin = media_info(path, want_video=True)
    if any(t.duration is not None and t.duration > MAX_SECONDS for t in tracks.values()):
        raise ValueError(f"Clip exceeds {MAX_SECONDS} seconds: {path}")
    frames, visual, stamps, text_boxes = [], [], [], 0
    limits, issues = {}, []
    if "video" in tracks:
        for image, _, stamp, _ in video_frames(path, tracks["video"], origin, lambda: None, limits,
                                               MAX_FRAMES, FRAME_PERIOD):
            clean, count = redact_subtitles(image, detector)
            frames.append(model_frame(clean))
            text_boxes += count
            stamps.append(stamp)
            observed = _analyze_frame(hub, clean, "teacher-study", stamp)
            if observed["reason"].startswith(("inference_error", "model_unavailable")):
                raise RuntimeError(observed["reason"])
            observed["eligible"] = observed["valid"] and observed["quality"] >= min_quality
            visual.append(observed)
        for observed, following in zip(visual, [*stamps[1:], limits["end"]], strict=True):
            observed["t_end"] = min(following, observed["t_start"] + FRAME_PERIOD)
    track = tracks.get("audio")
    audio, transcripts, _, _ = _audio_pass(path, track, origin, hub, "teacher-study", hub.model_dir,
                                           lambda: None, True, issues, min_quality)
    for row in [*audio, *transcripts]:
        if row["reason"].startswith(("inference_error", "model_unavailable")):
            raise RuntimeError(row["reason"])
    if issues:
        raise RuntimeError(" / ".join(issues))
    waveform = waveform_from_track(path, track, origin) if track else np.empty(0, np.float32)
    duration = max(limits.get("end", 0), len(waveform) / 16000)
    inputs = clip_inputs({"samples": visual, "audio_samples": audio, "transcripts": transcripts,
                          "timeline_coverage": {"duration_s": duration}})
    speech = clip_audio_embedding(path, model_hub=hub) if track else None
    info = dict(inputs, duration_s=duration, frame_times=stamps, ocr_boxes=text_boxes)
    video = np.asarray(frames, dtype=np.uint8).reshape(-1, 224, 224, 3)
    return info, waveform, video, speech if speech is not None else np.empty(0, np.float32)


def make_preparer(cache, *, device="cpu"):
    # In the CUDA venv, use its PyTorch and torchaudio and fall back to the
    # project's pinned decoder, vision and FunASR packages for preprocessing.
    if device == "cuda":
        sys.path.append(str(ROOT / ".venv/Lib/site-packages"))
    import cv2
    import torch
    from emotion_assistant.core import FusionEngine
    from emotion_assistant.models import ModelHub, VERSIONS
    from emotion_assistant.paths import DATA, MODELS, configure_environment

    configure_environment()
    ocr_dir = cache / "ocr"
    sys.path.insert(0, str(ocr_dir if ocr_dir.is_dir() else CACHE / "ocr"))
    from rapidocr_onnxruntime import RapidOCR

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA preprocessing requires the isolated CUDA environment")

    class ExperimentModelHub(ModelHub):
        def _load_funasr(self, folder, required):
            model = super()._load_funasr(folder, required)
            if device == "cuda":
                model.model.to(device)
                model.kwargs["device"] = device
            return model

    torch.set_num_threads(int(os.environ.get("CHSIMS_PREPARATION_THREADS", "2")))
    cv2.setNumThreads(1)
    ocr = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1, det_limit_type="max",
                   det_limit_side_len=640, det_thresh=.2, det_box_thresh=.4)
    detector_backend = "CPUExecutionProvider"
    if device == "cuda":
        import onnxruntime as ort

        ort.preload_dlls(directory=str(Path(torch.__file__).parent / "lib"))
        original = ocr.text_det.infer.session
        session = ort.InferenceSession(
            original._model_path, sess_options=original.get_session_options(),
            providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC", "use_tf32": 0,
                                                   "arena_extend_strategy": "kSameAsRequested"}),
                       "CPUExecutionProvider"],
        )
        detector_backend = session.get_providers()[0]
        if detector_backend != "CUDAExecutionProvider":
            raise RuntimeError("The CUDA subtitle detector could not be initialized")
        ocr.text_det.infer.session = session
    hub = ExperimentModelHub(MODELS)
    state = {**hub.load_visual(), **hub.load_audio()}
    if not all(model["ready"] for model in state.values()):
        raise RuntimeError(str(state))
    min_quality = FusionEngine.from_project(DATA).min_quality
    identity = {"source_code": sha_file(Path(__file__)), "models": VERSIONS, "device": device,
                "torch": torch.__version__, "subtitle_detector_backend": detector_backend,
                "manifest": sha_file(MODELS / "manifest.json"), "min_quality": min_quality}

    def prepare(source, destination):
        stamp = sha_file(source)
        if destination.exists():
            with np.load(destination, allow_pickle=False) as saved:
                info = json.loads(saved["info"].item())
                if info.get("identity") == identity and info.get("source_sha256") == stamp:
                    return
        seed = int(stamp[:8], 16)
        np.random.seed(seed)
        torch.manual_seed(seed)
        info, pcm, video, speech = extract_clip(source, hub, ocr.text_det, min_quality)
        info.update(source_sha256=stamp, identity=identity)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, info=json.dumps(info, ensure_ascii=False), pcm=pcm, video=video, speech=speech)
        write_atomic(destination, buffer.getvalue())

    return prepare
