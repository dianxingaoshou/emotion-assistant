"""Bounded capture and inference processes. No neural work in audio callback."""
from __future__ import annotations

from collections import deque
from math import gcd
from pathlib import Path
import multiprocessing as mp
import queue
import threading
import time
import uuid

import numpy as np


def invalid_observation(session_id, modality, start, end, reason):
    return dict(event_id=uuid.uuid4().hex, session_id=session_id,
                modality=modality, t_start=start, t_end=end, processed_at=end,
                scores=None, raw_scores={}, quality=0.0, valid=False,
                reason=reason, model_version="unavailable", latency_ms=0.0)


def _offer(target, value, latest=False):
    try:
        target.put_nowait(value)
        return True
    except queue.Full:
        if latest:
            try:
                target.get_nowait()
            except queue.Empty:
                pass
            try:
                target.put_nowait(value)
                return True
            except queue.Full:
                pass
        return False


class SileroVAD:
    """16 kHz ONNX streaming state/context as specified by Silero's wrapper."""
    def __init__(self, path):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1
        self.model = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.context = np.zeros((1, 64), np.float32)

    def __call__(self, block):
        if len(block) != 512:
            raise ValueError("Silero requires 512 samples at 16 kHz")
        sample = np.concatenate((self.context, np.asarray(block, np.float32)[None, :]), axis=1)
        output, self.state = self.model.run(None, {"input": sample, "state": self.state,
                                                  "sr": np.array(16000, np.int64)})
        self.context = sample[:, -64:].copy()
        return float(output[0, 0])


class SpeechSegmenter:
    """Probability-based endpointing, 0.6s silence / 8s maximum, 0.2s preroll."""
    def __init__(self, silence=0.6, maximum=8.0):
        self.silence = silence
        self.maximum = maximum
        self.reset()

    def reset(self):
        self.preroll = deque(maxlen=6)
        self.blocks = []
        self.start = None
        self.last_voice = None
        self.expected_next = None

    def feed(self, block, start, probability):
        block = np.asarray(block, np.float32)
        if self.expected_next is not None and abs(start - self.expected_next) > 0.01:
            self.reset()
        end = start + len(block) / 16000
        self.expected_next = end
        voice = probability >= 0.5
        if self.start is None:
            if not voice:
                self.preroll.append((start, block.copy()))
                return None
            self.start = self.preroll[0][0] if self.preroll else start
            self.blocks = [b for _, b in self.preroll]
            self.preroll.clear()
        self.blocks.append(block.copy())
        if voice:
            self.last_voice = end
        if ((self.last_voice is not None and end - self.last_voice >= self.silence)
                or end - self.start >= self.maximum):
            result = (np.concatenate(self.blocks), self.start, end)
            self.reset()
            return result
        return None


def camera_process(index, origin, stop, pause, frame_queue, status_queue):
    import cv2
    capture = None
    last_error = 0.0
    try:
        while not stop.is_set():
            if pause.is_set():
                if capture is not None:
                    capture.release()
                    capture = None
                stop.wait(0.05)
                continue
            if capture is None:
                capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                if not capture.isOpened():
                    capture.release()
                    capture = None
                    if time.monotonic() - last_error > 5:
                        _offer(status_queue, {"kind": "error", "message": "摄像头不可用；将继续尝试重连"})
                        last_error = time.monotonic()
                    stop.wait(2)
                    continue
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                capture.set(cv2.CAP_PROP_FPS, 15)
            begin = time.monotonic()
            ok, frame = capture.read()
            if not ok:
                capture.release()
                capture = None
                _offer(status_queue, {"kind": "error", "message": "摄像头读取失败；正在重连"})
                stop.wait(1)
                continue
            stamp = time.monotonic() - origin
            _offer(frame_queue, (stamp, frame), latest=True)
            stop.wait(max(0, 1 / 15 - (time.monotonic() - begin)))
    finally:
        if capture is not None:
            capture.release()


def audio_process(device, model_dir, origin, stop, pause, segment_queue, status_queue):
    import sounddevice as sd
    import soxr
    try:
        vad = SileroVAD(Path(model_dir) / "vad" / "silero_vad.onnx")
    except Exception as exc:
        _offer(status_queue, {"kind": "error", "message": f"VAD不可用，语音采集未启动：{type(exc).__name__}；请先下载模型"})
        return
    while not stop.is_set():
        if pause.is_set():
            stop.wait(0.05)
            continue
        try:
            info = sd.query_devices(device, "input")
            rate = int(info["default_samplerate"])
            blocksize = round(rate * 0.032)
            ring = np.zeros((128, blocksize), np.float32)
            stamps = np.zeros(128, np.float64)
            positions = [0, 0]  # write / read, monotonic indices
            drops = [0]
            ring_lock = threading.Lock()

            def callback(indata, frames, timing, status):
                # Only copies into preallocated buffers. No model, file I/O or waits.
                if frames != blocksize or not ring_lock.acquire(blocking=False):
                    drops[0] += 1
                    return
                try:
                    if positions[0] - positions[1] >= len(ring):
                        drops[0] += 1
                        return
                    index = positions[0] % len(ring)
                    np.copyto(ring[index], indata[:, 0])
                    # Both PortAudio fields use the same device clock; map ADC time
                    # into our monotonic session clock rather than callback arrival.
                    offset = float(timing.inputBufferAdcTime) - float(timing.currentTime)
                    if not np.isfinite(offset) or abs(offset) > 10:
                        offset = -frames / rate
                    stamps[index] = time.monotonic() - origin + offset
                    positions[0] += 1
                    if status:
                        drops[0] += 1
                finally:
                    ring_lock.release()

            segmenter = SpeechSegmenter()
            vad.reset()
            pending = np.empty(0, np.float32)
            pending_start = 0.0
            resampler = soxr.ResampleStream(rate, 16000, 1, dtype="float32", quality="HQ")
            expected_native = None
            output_origin = None
            output_count = 0
            with sd.InputStream(device=device, samplerate=rate, channels=1, dtype="float32",
                                blocksize=blocksize, callback=callback) as stream:
                _offer(status_queue, {"kind": "info", "message": f"麦克风已启动：{rate} Hz → 16000 Hz；Silero语音分段"})
                while not stop.is_set() and not pause.is_set():
                    item = None
                    with ring_lock:
                        if positions[1] < positions[0]:
                            index = positions[1] % len(ring)
                            item = (ring[index].copy(), stamps[index])
                            positions[1] += 1
                    if drops[0]:
                        count = drops[0]
                        drops[0] = 0
                        _offer(status_queue, {"kind": "gap", "message": f"音频缓冲缺口：{count}个块；当前语段已丢弃",
                                              "count": count, "t": time.monotonic() - origin})
                        segmenter.reset()
                        vad.reset()
                        pending = np.empty(0, np.float32)
                        resampler.clear()
                        expected_native = output_origin = None
                        output_count = 0
                        with ring_lock:
                            positions[1] = positions[0]
                        item = None
                    if item is None:
                        if not stream.active:
                            raise RuntimeError("audio stream disconnected")
                        stop.wait(0.005)
                        continue
                    samples, stamp = item
                    if expected_native is not None and abs(stamp - expected_native) > 0.02:
                        segmenter.reset()
                        vad.reset()
                        resampler.clear()
                        pending = np.empty(0, np.float32)
                        output_origin = None
                        output_count = 0
                        _offer(status_queue, {"kind": "gap", "message": "音频时间不连续，当前语段已丢弃", "t": stamp})
                    expected_native = stamp + len(samples) / rate
                    if output_origin is None:
                        output_origin = stamp
                    samples = resampler.resample_chunk(samples)
                    if not len(pending):
                        pending_start = output_origin + output_count / 16000
                    output_count += len(samples)
                    pending = np.concatenate((pending, samples))
                    while len(pending) >= 512:
                        block, pending = pending[:512], pending[512:]
                        segment = segmenter.feed(block, pending_start, vad(block))
                        pending_start += 512 / 16000
                        ready_segment = (*segment, time.monotonic()-origin) if segment else None
                        if ready_segment and not _offer(segment_queue, ready_segment):
                            _offer(status_queue, {"kind": "gap", "message": "语音队列已满，本语段未分析",
                                                  "t": pending_start})
                # Unfinished segments are deliberately discarded across pause/end.
        except Exception as exc:
            _offer(status_queue, {"kind": "error", "message": f"麦克风不可用（{type(exc).__name__}）；2秒后重连"})
            stop.wait(2)


def inference_process(modality, model_dir, session_id, origin, jobs, results, stop, asr_allowed,
                      permission=None, generation_value=None, consent_value=None):
    from .paths import configure_environment
    configure_environment()
    from .models import ModelHub
    try:
        hub = ModelHub(Path(model_dir), enable_asr=modality == "asr" and asr_allowed)
        status = (hub.load_visual() if modality == "visual" else
                  hub.load_audio_emotion() if modality == "audio" else hub.load_asr())
        _offer(results, {"kind": "loaded", "modality": modality, "status": status})
        while not stop.is_set():
            try:
                item = jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            token, data, start, end, generation = item[:5]
            queued_at = item[5] if len(item) > 5 else time.monotonic() - origin
            vad_ready_at = item[7] if len(item) > 7 else None
            consent_generation = item[6] if len(item) > 6 else 0
            if (generation_value is not None and generation != generation_value.value
                    or modality == "asr" and (permission is not None and not permission.is_set()
                    or consent_value is not None and consent_generation != consent_value.value)):
                continue
            _offer(results, {"kind": "started", "modality": modality, "token": token,
                             "generation": generation})
            inference_started = time.monotonic()
            try:
                if modality == "visual":
                    payload = hub.analyze_frame(data, session_id, start, end)
                elif modality == "audio":
                    payload = hub.analyze_audio_emotion(data, session_id, start, end)
                else:
                    payload = hub.transcribe_audio(data, session_id, start, end,
                        asr_allowed=asr_allowed and (permission is None or permission.is_set()))
                if (payload is not None and (modality != "asr" or payload.get("valid"))
                        and not stop.is_set()):
                    finished = time.monotonic() - origin
                    payload.update(processed_at=finished,
                        queue_wait_ms=max(0, (inference_started-origin-queued_at)*1000),
                        inference_ms=max(0, (time.monotonic()-inference_started)*1000),
                        end_to_end_ms=max(0, (finished-end)*1000),
                        timing={"captured_end": end, "queued_at": queued_at,
                                "inference_started": inference_started-origin,
                                "inference_finished": finished})
                    if vad_ready_at is not None:
                        payload["timing"]["vad_ready_at"] = vad_ready_at
                    kind = "utterance" if modality == "asr" else "observation"
                    _offer(results, {"kind": kind, kind: payload, "generation": generation,
                                     "consent_generation": consent_generation})
            except Exception as exc:
                if modality == "asr":
                    _offer(results, {"kind": "error", "stage": "asr", "reason": "inference_error",
                        "message": f"转写失败：{type(exc).__name__}；声学观察不受影响"})
                else:
                    obs = invalid_observation(session_id, modality, start, end,
                                              f"推理失败：{type(exc).__name__}")
                    obs["processed_at"] = time.monotonic() - origin
                    obs["end_to_end_ms"] = max(0, (obs["processed_at"] - end) * 1000)
                    obs["latency_ms"] = (time.monotonic() - inference_started) * 1000
                    _offer(results, {"kind": "observation", "observation": obs, "generation": generation})
            _offer(results, {"kind": "done", "modality": modality, "token": token,
                             "generation": generation})
    except Exception as exc:
        _offer(results, {"kind": "error", "message": f"{modality}模型加载失败：{type(exc).__name__}；采集仍可继续"})


class CaptureRuntime:
    """Owns every process/queue; stop forcibly reaps hung device/model calls."""
    def __init__(self, session_id, camera, microphone, model_dir, on_frame, on_event,
                 asr_allowed=False):
        self.session_id = session_id
        self.camera = camera
        self.microphone = microphone
        self.model_dir = str(model_dir)
        self.on_frame, self.on_event = on_frame, on_event
        self.asr_allowed = asr_allowed is True
        self._asr_accept_after = 0.0
        self.ctx = mp.get_context("spawn")
        self.stop_event = self.ctx.Event()
        self.pause_event = self.ctx.Event()
        self.frames = self.ctx.Queue(2)
        self.segments = self.ctx.Queue(2)
        self.results = self.ctx.Queue(64)
        self.jobs = {"visual": self.ctx.Queue(1), "audio": self.ctx.Queue(2), "asr": self.ctx.Queue(2)}
        self.processes = {}
        self.worker_started_at = {}
        self.generation = 0
        self._generation_value = self.ctx.Value("q", 0)
        self._consent_value = self.ctx.Value("q", 0)
        self._asr_permission = self.ctx.Event()
        if self.asr_allowed:
            self._asr_permission.set()
        self.origin = time.monotonic()
        self.accept_after = 0.0
        self.thread = None

    def _spawn(self, name, target, args):
        process = self.ctx.Process(target=target, args=args, name=f"emotion-{name}", daemon=True)
        process.start()
        self.processes[name] = process
        self.worker_started_at[name] = time.monotonic()

    def _spawn_inference(self, modality):
        self._spawn(modality, inference_process,
                    (modality, self.model_dir, self.session_id, self.origin,
                     self.jobs[modality], self.results, self.stop_event,
                     modality == "asr", self._asr_permission,
                     self._generation_value, self._consent_value))

    def start(self):
        for modality, enabled in (("visual", self.camera >= 0), ("audio", self.microphone is not None)):
            if enabled:
                self._spawn_inference(modality)
        if self.microphone is not None and self.asr_allowed:
            self._spawn_inference("asr")
        if self.camera >= 0:
            self._spawn("camera", camera_process, (self.camera, self.origin, self.stop_event,
                                                  self.pause_event, self.frames, self.results))
        if self.microphone is not None:
            self._spawn("microphone", audio_process, (self.microphone, self.model_dir, self.origin,
                                                      self.stop_event, self.pause_event,
                                                      self.segments, self.results))
        self.thread = threading.Thread(target=self._monitor, name="capture-monitor", daemon=True)
        self.thread.start()

    def pause(self, paused):
        self.generation += 1
        if hasattr(self, "_generation_value"):
            self._generation_value.value = self.generation
        self.accept_after = time.monotonic() - self.origin
        self.pause_event.set() if paused else self.pause_event.clear()
        for q in (self.frames, self.segments, *self.jobs.values()):
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass

    def set_asr_allowed(self, allowed):
        """Invalidate queued and in-flight text on every consent transition."""
        allowed = allowed is True
        if self.stop_event.is_set() or allowed == self.asr_allowed:
            return
        self.asr_allowed = allowed
        self._consent_value.value += 1
        if allowed:
            self._asr_accept_after = time.monotonic() - self.origin
        self._asr_permission.set() if allowed else self._asr_permission.clear()
        try:
            while True:
                self.jobs["asr"].get_nowait()
        except queue.Empty:
            pass
        if allowed and self.microphone is not None and "asr" not in self.processes:
            self._spawn_inference("asr")
        self.on_event({"kind": "info", "stage": "asr", "reason": "consent_changed",
            "message": "转写已开启" if allowed else "转写授权已撤回；旧转写结果不再发布"})

    def _publish_result(self, event):
        if self.stop_event.is_set() or self.pause_event.is_set() or event.get("generation") != self.generation:
            return
        if event["kind"] == "utterance" and (not getattr(self, "asr_allowed", False)
                or event.get("consent_generation") != self._consent_value.value):
            return
        payload = event[event["kind"]]
        published_at = time.monotonic() - self.origin
        payload.setdefault("timing", {})["published_at"] = published_at
        if "t_end" in payload:
            payload["end_to_end_ms"] = max(0, (published_at - payload["t_end"]) * 1000)
        self.on_event(event)

    def _queue_segment(self, samples, start, end, timed_out, vad_ready_at=None):
        """Independent queues own separate PCM snapshots, never a callback buffer."""
        generation = self.generation
        consent_generation = self._consent_value.value if hasattr(self, "_consent_value") else 0
        for stage in ("audio", "asr"):
            if stage not in self.jobs or stage in timed_out:
                continue
            if stage == "asr" and not getattr(self, "asr_allowed", False):
                continue
            if stage == "asr" and start < getattr(self, "_asr_accept_after", 0):
                continue
            snapshot = np.array(samples, dtype=np.float32, copy=True)
            snapshot.flags.writeable = False
            job = (uuid.uuid4().hex, snapshot, start, end, generation,
                   time.monotonic() - self.origin, consent_generation, vad_ready_at)
            if not _offer(self.jobs[stage], job):
                self.on_event({"kind": "gap", "stage": stage, "reason": "queue_full",
                    "dropped_count": 1, "t": end,
                    "message": ("声学" if stage == "audio" else "转写") + "推理积压，当前任务已跳过"})

    def _monitor(self):
        last_visual = -10.0
        running = {}
        startup = time.monotonic()
        loaded = set()
        timed_out = set()
        while not self.stop_event.is_set():
            try:
                while True:
                    event = self.results.get_nowait()
                    kind = event.get("kind")
                    if kind == "started":
                        if event.get("generation", self.generation) == self.generation:
                            running[event["modality"]] = (event.get("token"), time.monotonic())
                    elif kind == "done":
                        if running.get(event["modality"], (None,))[0] == event.get("token"):
                            running.pop(event["modality"], None)
                    elif kind == "loaded":
                        loaded.add(event["modality"])
                        self.on_event({"kind": "info", "message": f"{event['modality']} 模型状态：{event['status']}"})
                    elif kind in ("observation", "utterance"):
                        self._publish_result(event)
                    else:
                        self.on_event(event)
            except queue.Empty:
                pass
            if not self.pause_event.is_set():
                latest = None
                try:
                    while True:
                        latest = self.frames.get_nowait()
                except queue.Empty:
                    pass
                if latest and latest[0] >= self.accept_after:
                    stamp, frame = latest
                    self.on_frame(frame)
                    if stamp - last_visual >= 0.5 and "visual" not in timed_out:
                        _offer(self.jobs["visual"], (uuid.uuid4().hex, frame, max(self.accept_after, stamp - 0.5),
                                                     stamp, self.generation,
                                                     time.monotonic()-self.origin), latest=True)
                        last_visual = stamp
                try:
                    segment = self.segments.get_nowait()
                    samples, start, end = segment[:3]
                    if start >= self.accept_after:
                        self._queue_segment(samples, start, end, timed_out,
                                            segment[3] if len(segment) > 3 else None)
                except queue.Empty:
                    pass
            for modality in ("visual", "audio", "asr"):
                process = self.processes.get(modality)
                if process is None or modality in timed_out:
                    continue
                deadline = 10 if modality == "visual" else 40 if modality == "audio" else 60
                expired = (modality in running and time.monotonic() - running[modality][1] > deadline)
                load_start = getattr(self, "worker_started_at", {}).get(modality, startup)
                load_expired = modality not in loaded and time.monotonic() - load_start > 180
                if expired or load_expired or not process.is_alive():
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=1)
                    timed_out.add(modality)
                    now = time.monotonic() - self.origin
                    if modality != "asr":
                        self.on_event({"kind": "observation", "observation": invalid_observation(
                            self.session_id, modality, max(0, now - .5), now, "模型超时或退出；结束后重新开始可重载")})
                    self.on_event({"kind": "error", "message": f"{modality}模型已降级，界面及其余采集继续运行"})
            self.stop_event.wait(0.015)

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)
            if hasattr(self.thread, "is_alive") and self.thread.is_alive():
                raise RuntimeError("采集监测线程未退出，尚未关闭进程句柄")
        for q in self.jobs.values():
            _offer(q, None, latest=True)
        deadline = time.monotonic() + 3
        for process in self.processes.values():
            process.join(timeout=max(0, deadline - time.monotonic()))
        for process in self.processes.values():
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
                if process.is_alive():
                    raise RuntimeError("采集或模型进程未退出")
            process.close()
        for q in (self.frames, self.segments, self.results, *self.jobs.values()):
            q.cancel_join_thread()
            q.close()
