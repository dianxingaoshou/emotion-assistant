"""Offline adapters for real neural models; unavailable evidence stays unavailable.

No model package is imported until explicitly loaded. Emotion scores are model
outputs, while quality metrics are deliberately separate engineering measures.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time
import uuid

from .evidence import raw_prediction_diagnostics

VISUAL_LABELS = ("anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise")
CANONICAL = ("positive", "neutral", "negative", "uncertain")
VERSIONS = {
    "face": "YuNet-2023mar@48b10fa505ae",
    "visual": "EmotiEffLib/enet_b0_8_best_afew@520a051c64cd",
    "audio": "emotion2vec_plus_seed@7be9a30d52bd",
    "asr": "SenseVoiceSmall@3847d57b6bdf",
}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _canonical(raw):
    """Keep ALL mass, particularly surprise, other and unknown categories."""
    result = dict.fromkeys(CANONICAL, 0.0)
    for label, score in raw.items():
        key = str(label).casefold()
        if any(word in key for word in ("happy", "happiness", "开心")):
            category = "positive"
        elif "neutral" in key or "中立" in key or "中性" in key:
            category = "neutral"
        elif any(word in key for word in ("angry", "anger", "contempt", "disgust", "fear", "sad", "生气", "厌恶", "恐惧", "难过", "悲伤")):
            category = "negative"
        else:
            category = "uncertain"
        result[category] += float(score)
    return result


class ModelHub:
    """Load verified local weights and analyze one frame / one 16 kHz utterance.

    Call load_visual/load_audio from workers. Each inference method serializes
    its own model, so camera and audio inference can run independently.
    """

    def __init__(self, model_dir: Path, enable_asr: bool = True):
        self.model_dir = Path(model_dir).resolve()
        self.enable_asr = bool(enable_asr)
        self._detector = self._visual_model = self._audio_model = self._asr_model = None
        self._visual_lock = threading.RLock()
        self._audio_lock = threading.RLock()
        self._asr_lock = threading.RLock()
        self._verified = {}
        self.status = {key: self._state(key, False, "not_loaded") for key in VERSIONS}

    @staticmethod
    def _state(key, ready, reason):
        return {"ready": bool(ready), "reason": str(reason), "model_version": VERSIONS[key]}

    def _verify(self, relative_paths):
        manifest_path = self.model_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("model_manifest_missing: run scripts/download_models.py")
        manifest = json.loads(manifest_path.read_text("utf-8"))
        records = {record["path"]: record for record in manifest.get("assets", [])}
        for relative in relative_paths:
            path = self.model_dir / relative
            if not path.is_file():
                raise FileNotFoundError(f"model_file_missing: {relative}")
            expected = records.get(relative, {}).get("sha256")
            if not expected:
                raise ValueError(f"model_not_in_manifest: {relative}")
            stat = path.stat()
            stamp = (stat.st_size, stat.st_mtime_ns, expected)
            if self._verified.get(relative) == stamp:
                continue
            checksum = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    checksum.update(chunk)
            if checksum.hexdigest() != expected:
                raise ValueError(f"model_checksum_mismatch: {relative}")
            self._verified[relative] = stamp

    def load_visual(self) -> dict:
        with self._visual_lock:
            try:
                import cv2
                import numpy as np
                self._verify(["visual/face_detection_yunet_2023mar.onnx"])
                if self._detector is None:
                    # Python opens Unicode Windows paths correctly; OpenCV's
                    # native filename overload does not on some Windows builds.
                    model_bytes = (self.model_dir / "visual/face_detection_yunet_2023mar.onnx").read_bytes()
                    self._detector = cv2.FaceDetectorYN.create(
                        "onnx", np.frombuffer(model_bytes, dtype=np.uint8),
                        np.array([], dtype=np.uint8), (640, 480), 0.8, 0.3, 5000)
                self.status["face"] = self._state("face", True, "ready")
            except Exception as exc:
                self._detector = None
                self.status["face"] = self._state("face", False, f"{type(exc).__name__}: {exc}")
            try:
                import onnxruntime as ort
                self._verify(["visual/enet_b0_8_best_afew.onnx"])
                if self._visual_model is None:
                    options = ort.SessionOptions()
                    options.intra_op_num_threads = 2
                    options.inter_op_num_threads = 1
                    self._visual_model = ort.InferenceSession(
                        str(self.model_dir / "visual/enet_b0_8_best_afew.onnx"),
                        sess_options=options, providers=["CPUExecutionProvider"])
                    output = self._visual_model.get_outputs()[0]
                    if output.shape[-1] != 8:
                        raise ValueError(f"unexpected EmotiEffLib output shape: {output.shape}")
                self.status["visual"] = self._state("visual", True, "ready")
            except Exception as exc:
                self._visual_model = None
                self.status["visual"] = self._state("visual", False, f"{type(exc).__name__}: {exc}")
            return {key: dict(self.status[key]) for key in ("face", "visual")}

    def _load_funasr(self, folder, required):
        # Absolute existing paths, check_latest=False, and no remote code avoid
        # model hub downloads and package installation during application use.
        self._verify([f"{folder}/{filename}" for filename in required])
        from funasr import AutoModel
        extra = {}
        if folder == "SenseVoiceSmall":
            # SentencePiece's C++ file loader also has a Unicode-path issue on
            # Windows. Register a distinct local subclass that reads bytes with
            # Python; keep the official tokenizer class and package untouched.
            from funasr.register import tables
            from funasr.tokenizer.sentencepiece_tokenizer import SentencepiecesTokenizer
            if "LocalBytesSentencepiecesTokenizer" not in tables.tokenizer_classes:
                @tables.register("tokenizer_classes", "LocalBytesSentencepiecesTokenizer")
                class LocalBytesSentencepiecesTokenizer(SentencepiecesTokenizer):
                    def _build_sentence_piece_processor(self):
                        if self.sp is None:
                            import sentencepiece
                            self.sp = sentencepiece.SentencePieceProcessor(
                                model_proto=Path(self.bpemodel).read_bytes())
            extra["tokenizer"] = "LocalBytesSentencepiecesTokenizer"
        return AutoModel(model=str(self.model_dir / folder), device="cpu", ncpu=2,
                         disable_update=True, check_latest=False, trust_remote_code=False,
                         disable_pbar=True, disable_log=True, hub="ms", **extra)

    def load_audio_emotion(self) -> dict:
        """Load only the acoustic classifier; ASR has its own worker and lock."""
        with self._audio_lock:
            try:
                if self._audio_model is None:
                    self._audio_model = self._load_funasr(
                        "emotion2vec_plus_seed", ("config.yaml", "configuration.json", "model.pt", "tokens.txt"))
                self.status["audio"] = self._state("audio", True, "ready")
            except Exception as exc:
                self._audio_model = None
                self.status["audio"] = self._state("audio", False, f"{type(exc).__name__}: {exc}")
            return {"audio": dict(self.status["audio"])}

    def load_asr(self) -> dict:
        with self._asr_lock:
            if not self.enable_asr:
                self.status["asr"] = self._state("asr", False, "disabled_by_configuration")
            else:
                try:
                    if self._asr_model is None:
                        self._asr_model = self._load_funasr(
                            "SenseVoiceSmall", ("config.yaml", "configuration.json", "model.pt", "am.mvn",
                                                "chn_jpn_yue_eng_ko_spectok.bpe.model"))
                    self.status["asr"] = self._state("asr", True, "ready")
                except Exception as exc:
                    self._asr_model = None
                    self.status["asr"] = self._state("asr", False, f"{type(exc).__name__}: {exc}")
            return {"asr": dict(self.status["asr"])}

    def load_audio(self) -> dict:
        """Compatibility loader. The runtime calls the two loaders separately."""
        return {**self.load_audio_emotion(), **self.load_asr()}

    @staticmethod
    def _observation(modality, session_id, t_start, t_end):
        return {"event_id": str(uuid.uuid4()), "session_id": str(session_id),
                "modality": modality, "t_start": float(t_start), "t_end": float(t_end),
                "processed_at": _now(), "scores": None, "raw_scores": {}, "quality": 0.0,
                "quality_metrics": {}, "valid": False, "reason": "not_analyzed",
                "model_version": VERSIONS["visual" if modality == "visual" else "audio"],
                "latency_ms": 0.0}

    @staticmethod
    def _finish(observation, started, reason=None):
        observation["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        observation["processed_at"] = _now()
        if reason:
            observation["reason"] = reason
        return observation

    def analyze_frame(self, frame, session_id: str, t_start: float, t_end: float) -> dict:
        started = time.perf_counter()
        observation = self._observation("visual", session_id, t_start, t_end)
        with self._visual_lock:
            if self._detector is None or self._visual_model is None:
                missing = [key for key in ("face", "visual") if not self.status[key]["ready"]]
                return self._finish(observation, started, "model_unavailable: " + "; ".join(
                    f"{key}={self.status[key]['reason']}" for key in missing))
            try:
                import cv2
                import numpy as np
                if (not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3
                        or frame.dtype != np.uint8 or min(frame.shape[:2]) < 20):
                    return self._finish(observation, started, "invalid_bgr_frame")
                if not (math.isfinite(t_start) and math.isfinite(t_end) and 0 <= t_start < t_end):
                    return self._finish(observation, started, "invalid_interval")
                height, width = frame.shape[:2]
                self._detector.setInputSize((width, height))
                _, faces = self._detector.detect(frame)
                count = 0 if faces is None else len(faces)
                observation["quality_metrics"]["face_count"] = count
                if count != 1:
                    return self._finish(observation, started, "no_face" if count == 0 else "multiple_faces")
                face = faces[0]
                x, y, w, h = map(float, face[:4])
                left, top = max(0, int(x)), max(0, int(y))
                right, bottom = min(width, int(math.ceil(x + w))), min(height, int(math.ceil(y + h)))
                observation["face_box"] = [left, top, max(0, right-left), max(0, bottom-top)]
                if right <= left or bottom <= top:
                    return self._finish(observation, started, "invalid_face_box")
                crop = frame[top:bottom, left:right]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                brightness = float(gray.mean())
                blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                eye_distance = float(np.linalg.norm(face[4:6] - face[6:8]))
                eye_mid = (face[4:6] + face[6:8]) / 2
                nose_offset = abs(float(face[8] - eye_mid[0])) / max(eye_distance, 1.0)
                eye_tilt = abs(float(face[5] - face[7])) / max(eye_distance, 1.0)
                metrics = observation["quality_metrics"]
                metrics.update(face_size_px=float(min(right-left, bottom-top)), brightness=brightness,
                               blur_variance=blur, detection_score=float(face[-1]),
                               eye_distance_px=eye_distance, nose_offset_ratio=nose_offset,
                               eye_tilt_ratio=eye_tilt)
                if min(right-left, bottom-top) < 70:
                    return self._finish(observation, started, "face_too_small")
                if brightness < 30 or brightness > 225:
                    return self._finish(observation, started, "poor_lighting")
                if blur < 25:
                    return self._finish(observation, started, "blurred_face")
                if eye_distance < 0.18*w or nose_offset > 0.65 or eye_tilt > 0.65:
                    return self._finish(observation, started, "large_pose_or_unreliable_landmarks")
                quality = min(1.0, min(right-left, bottom-top)/160.0)
                quality *= min(1.0, blur/120.0) * min(1.0, min(brightness, 255-brightness)/65.0)
                quality *= float(face[-1]) * max(0.2, 1.0 - nose_offset)
                # Match official ONNX implementation: RGB, cv2 linear resize,
                # /255, ImageNet channel mean/std, then NCHW float32.
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                image = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
                image = (image - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
                tensor = image.transpose(2, 0, 1)[None].astype(np.float32)
                name = self._visual_model.get_inputs()[0].name
                logits = np.asarray(self._visual_model.run(None, {name: tensor})[0], dtype=np.float64).reshape(-1)
                if logits.size != 8 or not np.isfinite(logits).all():
                    raise ValueError("invalid_visual_output")
                probabilities = np.exp(logits - logits.max())
                probabilities /= probabilities.sum()
                raw = dict(zip(VISUAL_LABELS, map(float, probabilities)))
                observation.update(scores=_canonical(raw), raw_scores=raw,
                                   raw_logits=dict(zip(VISUAL_LABELS, map(float, logits))),
                                   quality=float(max(0.0, min(1.0, quality))), valid=True, reason="ok")
                self._prediction_gate(observation)
            except Exception as exc:
                observation.update(valid=False, scores=None, quality=0.0,
                                   reason=f"inference_error: {type(exc).__name__}: {exc}")
            return self._finish(observation, started)

    @staticmethod
    def _prediction_gate(observation):
        diagnostics = raw_prediction_diagnostics(observation.get("raw_scores"))
        observation["prediction_diagnostics"] = diagnostics
        if not diagnostics["decision_eligible"]:
            observation.update(valid=False, scores=None, reason=diagnostics["reason"])

    @staticmethod
    def _audio_input(samples, t_start, t_end):
        import numpy as np
        # The model owns this small snapshot; third-party tensor conversion need
        # not mutate or borrow a read-only capture queue's array.
        audio = np.array(samples, dtype=np.float32, copy=True)
        if audio.ndim != 1 or not np.isfinite(audio).all() or len(audio) == 0:
            return audio, {}, "invalid_mono_audio"
        if not (math.isfinite(t_start) and math.isfinite(t_end) and 0 <= t_start < t_end):
            return audio, {}, "invalid_interval"
        duration = len(audio) / 16000.0
        rms = float(np.sqrt(np.mean(audio.astype(np.float64)**2)))
        clipped = float(np.mean(np.abs(audio) >= 0.995))
        metrics = {"sample_rate": 16000, "duration_s": duration, "rms": rms,
                   "clipping_ratio": clipped, "peak": float(np.max(np.abs(audio)))}
        if duration > 30:
            reason = "audio_too_long"
        elif abs((t_end-t_start)-duration) > max(0.15, duration*0.05):
            reason = "audio_timestamp_length_mismatch"
        elif duration < 0.2:
            reason = "audio_too_short"
        elif clipped > 0.05:
            reason = "audio_clipped"
        elif rms < 0.002:
            reason = "audio_silent_or_too_quiet"
        else:
            reason = None
        return audio, metrics, reason

    def analyze_audio_emotion(self, samples, session_id: str, t_start: float, t_end: float) -> dict:
        """Return acoustic evidence immediately, without calling or waiting for ASR."""
        started = time.perf_counter()
        observation = self._observation("audio", session_id, t_start, t_end)
        with self._audio_lock:
            try:
                import numpy as np
                audio, metrics, reason = self._audio_input(samples, t_start, t_end)
                observation["quality_metrics"] = metrics
                if reason:
                    return self._finish(observation, started, reason)
                duration, rms, clipped = metrics["duration_s"], metrics["rms"], metrics["clipping_ratio"]
                # Audio is already speech-segmented by capture/Silero. This check
                # rejects short/poor input; it is not a second emotion classifier.
                if duration < 2.0:
                    observation["reason"] = "insufficient_speech_duration"
                elif self._audio_model is None:
                    observation["reason"] = "model_unavailable: " + self.status["audio"]["reason"]
                else:
                    try:
                        result = self._audio_model.generate(input=audio, fs=16000,
                            granularity="utterance", extract_embedding=False, cache={})
                        item = result[0]
                        values = np.asarray(item["scores"], dtype=np.float64).reshape(-1)
                        labels = item["labels"]
                        if (len(labels) != len(values) or len(values) == 0 or not np.isfinite(values).all()
                                or np.any(values < 0) or abs(float(values.sum())-1.0) > 0.02):
                            raise ValueError("invalid_audio_probability_output")
                        values /= values.sum()
                        raw = {str(label): float(score) for label, score in zip(labels, values)}
                        quality = min(1.0, duration / 4.0) * min(1.0, rms / 0.015) * max(0.0, 1.0-clipped/0.05)
                        observation.update(scores=_canonical(raw), raw_scores=raw, quality=float(quality),
                                           valid=True, reason="ok")
                        self._prediction_gate(observation)
                    except Exception as exc:
                        observation["reason"] = f"inference_error: {type(exc).__name__}: {exc}"
                return self._finish(observation, started)
            except Exception as exc:
                return self._finish(observation, started, f"inference_error: {type(exc).__name__}: {exc}")

    def transcribe_audio(self, samples, session_id: str, t_start: float, t_end: float,
                         asr_allowed: bool = False) -> dict:
        """Independent text result. Only publish valid results; tags never vote."""
        started = time.perf_counter()
        transcript = {"event_id": str(uuid.uuid4()), "session_id": str(session_id),
                      "t_start": float(t_start), "t_end": float(t_end), "processed_at": _now(),
                      "text": "", "model_version": VERSIONS["asr"], "valid": False,
                      "reason": "not_analyzed", "is_current_concern": False, "latency_ms": 0.0}
        if asr_allowed is not True or not self.enable_asr:
            return self._finish(transcript, started, "consent_not_granted" if asr_allowed is not True
                                else "disabled_by_configuration")
        with self._asr_lock:
            try:
                audio, _, reason = self._audio_input(samples, t_start, t_end)
                if reason:
                    return self._finish(transcript, started, reason)
                if self._asr_model is None:
                    return self._finish(transcript, started, "model_unavailable: " + self.status["asr"]["reason"])
                result = self._asr_model.generate(input=audio, fs=16000, cache={},
                    language="auto", use_itn=True, batch_size=1)
                text = re.sub(r"<\|[^|]*\|>", "", str(result[0].get("text", ""))).strip()
                transcript.update(text=text, valid=bool(text), reason="ok" if text else "empty_transcript")
            except Exception as exc:
                transcript["reason"] = f"inference_error: {type(exc).__name__}: {exc}"
        return self._finish(transcript, started)

    def analyze_audio(self, samples, session_id: str, t_start: float, t_end: float,
                      asr_allowed: bool = False) -> tuple[dict, dict | None]:
        """Legacy synchronous pair; CaptureRuntime uses the independent methods."""
        started = time.perf_counter()
        observation = self.analyze_audio_emotion(samples, session_id, t_start, t_end)
        try:
            transcript = self.transcribe_audio(samples, session_id, t_start, t_end, asr_allowed)
            utterance = transcript if transcript.get("valid") else None
            observation["asr_reason"] = transcript["reason"]
        except Exception as exc:
            utterance = None
            observation["asr_reason"] = f"inference_error: {type(exc).__name__}: {exc}"
        observation["wrapper_latency_ms"] = round((time.perf_counter()-started)*1000, 3)
        return observation, utterance
