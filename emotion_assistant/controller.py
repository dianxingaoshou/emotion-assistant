"""Qt-facing orchestration. Device/model work is isolated from the UI process."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import threading
import time
import uuid

from PySide6.QtCore import QObject, QTimer, Signal, Slot, Qt

from .capture import CaptureConfig, CaptureRuntime, invalid_observation
from .core import FusionEngine, InterventionFlow
from .content_timeline import LiveContentTimeline
from .paths import DATA, MODELS
from .storage import Store
from .regulation import ACTIVITIES, NEEDS, evaluate_outcome, summarize_feedback
from .prompt_policy import ConflictGate, PromptTimingPolicy
from .recommendation_policy import contextual_decision
from .crossover import trial_order, trial_prompt, write_report
from .evidence import SustainedEvidenceGate
from .assessment import AssessmentWindowCollector, compare_assessments
from .personalization import FeedbackRecord, build_reference, compare_reference


def simulation_observation(session_id, modality, start, end, negative=0.7, quality=0.85):
    """Explicit fixture generator ONLY used by the user-selected simulation mode."""
    return {"event_id": uuid.uuid4().hex, "session_id": session_id, "modality": modality,
            "t_start": start, "t_end": end, "processed_at": end + 0.1,
            "scores": {"negative": negative, "neutral": 0.85 - negative,
                       "positive": 0.1, "uncertain": 0.05}, "raw_scores": {},
            "quality": quality, "valid": True, "reason": "模拟输入，非实测",
            "model_version": "SIMULATION-FIXTURE-v1", "latency_ms": 0,
            "simulation": True}


class Controller(QObject):
    frame_ready = Signal(object)
    observation_ready = Signal(dict)
    fusion_ready = Signal(dict)
    transcript_ready = Signal(dict)
    content_ready = Signal(dict)
    status_changed = Signal(str)
    session_changed = Signal(dict)
    devices_ready = Signal(dict)
    intervention_changed = Signal(dict)
    history_ready = Signal(list)
    recommendations_ready = Signal(dict)
    _event_received = Signal(object)
    _stopped = Signal(str)
    _history_received = Signal(list)
    _devices_received = Signal(dict)
    _recommendations_received = Signal(dict)
    _write_completed = Signal(object)
    _stop_failed = Signal(dict)
    evidence_changed = Signal(dict)
    transcript_cleanup_changed = Signal()
    _transcript_cleanup_finished = Signal(object)

    def __init__(self, database=None, parent=None):
        super().__init__(parent)
        self.store = Store(database or DATA / "sessions.sqlite3")
        self.transcript_cleanup_inflight = {}
        self.transcript_cleanup_failed = set(self.store.pending_transcript_cleanup())
        self._transcript_cleanup_finished.connect(self._on_transcript_cleanup, Qt.ConnectionType.QueuedConnection)
        self.writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sqlite-writer")
        self._write_slots = threading.BoundedSemaphore(256)
        self._observation_slots = threading.BoundedSemaphore(192)
        self._last_write_warning = 0.0
        self.background = ThreadPoolExecutor(max_workers=2, thread_name_prefix="app-background")
        self._event_received.connect(self._handle_event)
        self._stopped.connect(self._finish_stop)
        self._history_received.connect(self.history_ready)
        self._devices_received.connect(self.devices_ready)
        self._recommendations_received.connect(self._publish_recommendations)
        self._write_completed.connect(self._handle_write_completion, Qt.ConnectionType.QueuedConnection)
        self._stop_failed.connect(self._handle_stop_failure, Qt.ConnectionType.QueuedConnection)
        self.session_id = None
        self.trial = None
        self._trial_active_since = None
        self._trial_seconds = 0.
        self._trial_capture_started = False
        self._trial_unwanted_at = None
        self.state = "idle"
        self.mode = "real"
        self.runtime = None
        self.fusion = FusionEngine.from_project(DATA, baseline_version=None)
        self.personal_learning = False
        self._personal_reference_state = "disabled"
        self.frozen_reference = build_reference([])
        self._reference_context = {}
        self._reference_observing_since = 0.
        self.flow = InterventionFlow()
        self.started = time.monotonic()
        self.activity_started = None
        self.task_details = {}
        self.cooldown_until = 0.0
        self.negative_start = None
        self.negative_end = None
        self.latest_summary = None
        self.pre_model = None
        self.transcript_allowed = False
        self.closed = False
        self.last_observed = {"visual": 0.0, "audio": 0.0}
        self.enabled_modalities = set()
        self.current_user_id = None
        self.current_user_name = ""
        self.regulation_need = "unsure"
        self.regulation_seconds = 120
        self.recommendation_pre = None
        self.prompt_timing = PromptTimingPolicy()
        self.conflict_gate = ConflictGate()
        self._request_source = None
        self._request_decision_id = None
        self._prompt_signature = None
        self._offered_decision = None
        self.personalization = {}
        self._recommendation_serial = 0
        self._last_due_signature = ()
        self._pending_previous_id = None
        self._pending_root_id = None
        self._known_discomfort = set()
        self._avoid_writes = {}
        self.evidence_gate = SustainedEvidenceGate()
        self.assessment = None
        self.pre_assessment = self.post_assessment = None
        self._assessment_phase = None
        self._write_context = {}
        self._record_write_versions = {}
        self._audio_evidence = OrderedDict()
        self._content_timeline = LiveContentTimeline()
        self._simulation_timer = QTimer(self)
        self._simulation_timer.setInterval(500)
        self._simulation_timer.timeout.connect(self._simulate_tick)
        self._activity_timer = QTimer(self)
        self._activity_timer.setInterval(500)
        self._activity_timer.timeout.connect(self._activity_tick)
        self._activity_timer.start()
        self._resource_timer = QTimer(self)
        self._resource_timer.setInterval(10000)
        self._resource_timer.timeout.connect(self._record_resources)
        self._resource_timer.start()
        self._freshness_timer = QTimer(self)
        self._freshness_timer.setInterval(1000)
        self._freshness_timer.timeout.connect(self._check_freshness)
        self._freshness_timer.start()

    def _write(self, function, *args):
        observation = getattr(function, "__name__", "") == "save_observation"
        observation_slot = self._observation_slots.acquire(blocking=False) if observation else False
        acquired = (not observation or observation_slot) and self._write_slots.acquire(blocking=False)
        if not acquired:
            if observation_slot:
                self._observation_slots.release()
            if time.monotonic() - self._last_write_warning > 5:
                self._last_write_warning = time.monotonic()
                self.status_changed.emit("数据库写入积压，部分记录未保存；请结束会话并检查磁盘")
            rejected = Future()
            rejected.set_exception(RuntimeError("database queue full"))
            return rejected
        try:
            future = self.writer.submit(function, *args)
        except Exception as exc:
            self._write_slots.release()
            if observation_slot:
                self._observation_slots.release()
            future = Future()
            future.set_exception(exc)
            self.status_changed.emit("本地存储提交失败，请重试或导出记录")
            return future
        def finished(result):
            self._write_slots.release()
            if observation_slot:
                self._observation_slots.release()
            try:
                result.result()
            except Exception as exc:
                self.status_changed.emit(f"本地存储失败：{type(exc).__name__}；请检查磁盘和导出记录")
        future.add_done_callback(finished)
        return future

    @Slot(object)
    def _handle_write_completion(self, payload):
        if self.closed:
            return
        same_profile = (payload.get("user_id") == self.current_user_id and payload.get("mode") == self.mode)
        same_record = (payload.get("session_id") == self.session_id and self.flow.current
                       and payload.get("record_id") == self.flow.current.get("id"))
        if same_record:
            fields = {}
            if payload.get("write_revision") == self._record_write_versions.get(payload.get("record_id")):
                fields["save_status"] = "saved" if payload["success"] else "failed"
            updated = payload.get("updated")
            if isinstance(updated, dict) and updated.get("followup") is not None:
                fields.update(followup=updated["followup"], discomfort=updated.get("discomfort", False))
            if fields:
                self.flow.annotate(fields)
                self._intervention_signal()
        if payload["success"]:
            if same_profile:
                if isinstance(payload.get("updated"), dict) and payload["updated"].get("discomfort"):
                    self._known_discomfort.add(payload["updated"]["activity"])
                if payload.get("message"):
                    self.status_changed.emit(payload["message"])
                self.refresh_recommendations()
            self.refresh_history()
        elif same_profile:
            self.status_changed.emit("反馈保存失败，尚未确认写入；请重试保存或检查磁盘")

    def refresh_devices(self):
        def detect():
            result = {"cameras": [], "microphones": [], "errors": []}
            try:
                from PySide6.QtMultimedia import QMediaDevices
                result["cameras"] = [{"index": index, "name": item.description()}
                                     for index, item in enumerate(QMediaDevices.videoInputs())]
            except Exception as exc:
                result["errors"].append(f"摄像头枚举：{type(exc).__name__}")
            try:
                import sounddevice as sd
                devices = sd.query_devices()
                result["microphones"] = [{"index": i, "name": d["name"]}
                                         for i, d in enumerate(devices) if d["max_input_channels"] > 0]
            except Exception as exc:
                result["errors"].append(f"麦克风枚举：{type(exc).__name__}")
            if not self.closed:
                self._devices_received.emit(result)
        self.background.submit(detect)

    def _session_signal(self):
        self.session_changed.emit(
            {
                "state": self.state,
                "session_id": self.session_id,
                "mode": self.mode,
                "user_name": self.current_user_name,
                "trial": deepcopy(self.trial),
            }
        )

    def trial_status(self, user_name, mode):
        if (
            self.state in ("running", "paused", "stopping", "stop_failed")
            and self.trial
            and self.current_user_name == user_name.strip()
            and mode == self.mode
        ):
            return dict(self.trial, status="in_progress", next_period=None)
        uid = next((u["id"] for u in self.store.list_users() if u["name"] == user_name.strip()), None)
        return self.store.trial_status(uid, mode)

    def withdraw_trial(self, user_name=None, mode=None):
        name = (self.current_user_name if user_name is None else user_name).strip()
        selected = self.mode if mode is None else mode
        if not name or selected not in ("real", "simulation"):
            return False
        uid = next((u["id"] for u in self.store.list_users() if u["name"] == name), None)
        if not self.store.withdraw_trial(uid, selected):
            return False
        if uid == self.current_user_id and selected == self.mode:
            self.trial = self._trial_active_since = self._trial_unwanted_at = None
            self._session_signal()
            self.refresh_recommendations()
        return True

    def rate_trial_prompt(self, wanted):
        if (
            not self.trial
            or self.state not in ("running", "paused")
            or self.flow.state != "confirming"
            or self._request_source not in ("sustained_observation", "clarify_conflict")
            or not self._request_decision_id
            or type(wanted) is not bool
        ):
            return False
        # Serialize after prompt_asked; never infer a burden label from rejection.
        try:
            saved = self._write(
                self.store.rate_trial_prompt, self.session_id, self._request_decision_id, wanted
            ).result(timeout=5)
        except Exception:
            self.status_changed.emit("本次提示评价未确认保存，可重试")
            return False
        if saved and not wanted:
            self._trial_unwanted_at = time.monotonic()
        return saved

    def trial_results(self, mode="real"):
        return self.store.analyze_trial(mode)

    def export_trial_results(self, path, mode="real"):
        if Path(path).suffix.lower() != ".json" or (
            self.store.path and self.store.path.resolve() == Path(path).resolve()
        ):
            raise ValueError("请选择独立的 JSON 报告文件")
        return write_report(path, self.trial_results(mode))

    def _freeze_trial_exposure(self):
        if self._trial_active_since is not None:
            self._trial_seconds += max(0.0, time.monotonic() - self._trial_active_since)
            self._trial_active_since = None
        # A failed device startup consumes the assigned period, but contributes
        # no completed period or exposure to the crossover analysis.
        return self._trial_seconds if self.trial and self._trial_capture_started else None

    def start_session(self, user_name, camera_index, mic_index, consent, mode="real"):
        if self.state in ("running", "paused", "stopping", "stop_failed"):
            self.status_changed.emit("请先结束当前会话")
            return
        if (self.flow.current or {}).get("save_status") in ("pending", "failed"):
            self.status_changed.emit("请先完成上一条反馈的保存；失败时可重试保存")
            return
        if not consent.get("collection"):
            self.status_changed.emit("勾选自愿开始本地采集后，才能开始会话")
            return
        if not str(user_name).strip():
            self.status_changed.emit("请填写个人档案名称")
            return
        strategy = consent.get("fusion_strategy")
        if strategy not in (None, "quality_v2", "calibrated_fusion"):
            self.status_changed.emit("请选择有效的融合策略")
            return
        try:
            capture_config = CaptureConfig() if mode == "simulation" else CaptureConfig.from_project(DATA)
        except (ValueError, TypeError, OSError) as exc:
            self.status_changed.emit(f"采集配置无效，请检查 data/capture-config.json（{exc}）")
            return
        self.mode = "simulation" if mode == "simulation" else "real"
        users = self.store.list_users()
        user_id = next((u["id"] for u in users if u["name"] == user_name.strip()), None)
        user_id = user_id or self.store.create_user(user_name.strip())
        self.current_user_id, self.current_user_name = user_id, user_name.strip()
        self._recommendation_serial += 1
        self.personalization = {}
        self._known_discomfort = set()
        self._pending_previous_id = self._pending_root_id = None
        self.personal_learning = consent.get("personal_learning") is True
        self._personal_reference_state = "active" if self.personal_learning else "disabled"
        requested_devices = consent.get("device_names") or {}
        self.frozen_reference = (self.store.personal_reference(user_id, self.mode)
                                 if self.personal_learning else build_reference([]))
        consent = {"collection": True, "transcript": consent.get("transcript") is True,
                   "personal_learning": self.personal_learning,
                   "crossover_trial": consent.get("crossover_trial") is True,
                   "save_transcript": consent.get("transcript") is True, "raw_media": False}
        self.transcript_allowed = consent["transcript"]
        try:
            sid = self.store.start_session(user_id, consent, self.mode)
        except ValueError as exc:
            self.status_changed.emit(str(exc))
            return
        self.session_id = sid
        self.trial = self.store.trial_assignment(sid) if consent["crossover_trial"] else None
        self.state = "running"
        self.started = time.monotonic()
        self._trial_seconds = 0.
        self._trial_capture_started = self.mode == "simulation"
        self._trial_active_since = self.started if self.trial else None
        self._trial_unwanted_at = None
        self.fusion = FusionEngine.from_project(DATA, baseline_version=self.frozen_reference["version"], strategy=strategy)
        self.flow = InterventionFlow()
        self.evidence_gate.reset(self.session_id)
        self._audio_evidence.clear()
        self._clear_content()
        self.conflict_gate.reset(self.session_id)
        since = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(timespec="milliseconds")
        self.prompt_timing.restore(self.store.recent_prompt_history(user_id, self.mode, since))
        self._prompt_signature = self._offered_decision = self._request_source = None
        self._request_decision_id = None
        self.recommendation_pre = None
        self.assessment = None
        self.pre_assessment = self.post_assessment = None
        self._assessment_phase = None
        self.negative_start = self.negative_end = None
        self.latest_summary = self.pre_model = None
        self.last_observed = {"visual": 0.0, "audio": 0.0}
        self.enabled_modalities = ({"visual"} if camera_index is not None and camera_index >= 0 else set())
        if mic_index is not None:
            self.enabled_modalities.add("audio")
        if self.mode == "simulation":
            self._reference_context = {"enabled_modalities": ["audio", "visual"], "fixture": "simulation-v1"}
        else:
            self._reference_context = {"enabled_modalities": sorted(self.enabled_modalities),
                "camera": [camera_index, str(requested_devices.get("visual") or "")],
                "microphone": [mic_index, str(requested_devices.get("audio") or "")],
                "visual_period": capture_config.visual_period, "audio_quality_version": "vad-noise-v1"}
        self._reference_observing_since = 0.
        self.cooldown_until = 0
        self.task_details = {}
        self.activity_started = None
        self._session_signal()
        self._intervention_signal()
        self._write(self.store.append_event, self.session_id, "configuration",
                    {"camera_index": camera_index, "mic_index": mic_index,
                     "video_resolution": [640, 480], "preview_fps_target": 15,
                     "visual_inference_fps_target": 1/capture_config.visual_period,
                     "capture_config": asdict(capture_config), "audio_sample_rate": 16000,
                     "fusion_min_quality": self.fusion.min_quality, "fusion_config": self.fusion.config,
                     "fusion_config_hash": self.fusion.config_hash, "sustained_seconds": self.evidence_gate.required_seconds,
                     "cooldown_seconds": 1200, "raw_media_saved": False,
                     "app_version": "0.1.0", "simulation": self.mode == "simulation"})
        if self.mode == "simulation":
            self._sim_tick = 0
            self._simulation_timer.start()
            self.status_changed.emit("模拟回放已启动：全部线索为测试输入，不是摄像头/麦克风实测")
        else:
            try:
                self.runtime = CaptureRuntime(self.session_id, int(camera_index if camera_index is not None else -1),
                                              mic_index, MODELS, self.frame_ready.emit,
                                              self._event_received.emit, consent["transcript"], config=capture_config)
                self.runtime.start()
                self._trial_capture_started = True
                self.status_changed.emit("本地会话已开始；模型加载期间只显示可用输入。随时可暂停或结束")
            except Exception as exc:
                self.status_changed.emit(f"设备启动失败：{type(exc).__name__}")
                self.stop_session()
        self.refresh_recommendations()

    def _simulate_tick(self):
        if self.state != "running":
            return
        self._sim_tick += 1
        end = time.monotonic() - self.started
        start = max(0, end - 0.5)
        negative = 0.68 if self.flow.state != "running" else 0.35
        visual = simulation_observation(self.session_id, "visual", start, end, negative)
        if 12 <= self._sim_tick % 40 <= 16:
            visual.update(valid=False, scores=None, quality=0, reason="模拟遮挡，非实测")
        self._handle_event({"kind": "observation", "observation": visual})
        if self._sim_tick % 8 == 0:
            self._handle_event({"kind": "observation", "observation": simulation_observation(
                self.session_id, "audio", max(0, end - 3), end, 0.5)})

    @Slot(object)
    def _handle_event(self, event):
        if self.state != "running" or not self.session_id:
            return
        kind = event.get("kind")
        if (self.mode == "real" and kind in ("observation", "utterance") and "generation" in event
                and (self.runtime is None or not self.runtime.accepts_result(event))):
            return
        payload = event.get(kind, {})
        timing = payload.get("timing", {}) if isinstance(payload, dict) else {}
        if ("published_at" in timing and payload.get("session_id") == self.session_id
                and self.runtime is not None and hasattr(self.runtime, "record_timing")):
            arrival = time.monotonic()-self.runtime.origin
            timing["ui_received_at"] = arrival
            self.runtime.record_timing("asr" if kind == "utterance" else payload["modality"],
                                       ui_dispatch=max(0., arrival-timing["published_at"])*1000)
        if kind == "observation":
            observation = event["observation"]
            if observation.get("session_id") != self.session_id:
                return
            self.observation_ready.emit(observation)
            if not observation.get("stale"):
                self.last_observed[observation["modality"]] = max(
                    self.last_observed.get(observation["modality"], 0), observation["t_end"])
            self._write(self.store.save_observation, self.session_id, observation)
            if observation["modality"] == "visual":
                if observation["t_end"] > observation["t_start"] >= 0:
                    self.fusion.add_visual(observation)
                result = self.fusion.visual_summary(observation["t_start"], observation["t_end"])
                self._revise_audio_evidence(observation)
            else:
                self._queue_audio_evidence(observation)
                return
            self._publish_fusion(result)
        elif kind == "utterance":
            utterance = event["utterance"]
            sid = utterance.get("session_id")
            if not self.transcript_allowed or sid != self.session_id or utterance.get("stale"):
                return
            self.transcript_ready.emit(utterance)
            if self.state == "running" and self.transcript_allowed and self.session_id == sid:
                self._emit_content(self._content_timeline.observe_transcript(utterance))
            if self.state == "running" and self.transcript_allowed and self.session_id == sid:
                self._write(self.store.save_utterance, sid, utterance, True)
        else:
            self.status_changed.emit(event.get("message", str(kind)))
            self._write(self.store.append_event, self.session_id, kind or "status", event)

    def _publish_fusion(self, result):
        result.update(session_id=self.session_id, processed_at=time.monotonic()-self.started,
                      simulation=self.mode == "simulation")
        result.update(reference_context=self._reference_context,
                      reference_observing=(self._reference_observing_since is not None
                          and result["t_start"] >= self._reference_observing_since and self.state == "running"))
        if self.personal_learning:
            reference = compare_reference(self.frozen_reference, result)
        else:
            reference = {"status": self._personal_reference_state, "version": None}
        reference["session_frozen_version"] = self.fusion.config.get("baseline_version")
        reference["applied"] = reference["status"] in ("above_reference", "within_or_below_reference")
        result["personal_reference"] = reference
        self.fusion_ready.emit(result)
        if self.state != "running" or self.session_id != result["session_id"]:
            return
        self._emit_content(self._content_timeline.observe_fusion(result))
        if self.state != "running" or self.session_id != result["session_id"]:
            return
        self._write(self.store.save_observation, result["session_id"], result)
        priority = {"joint": 2, "audio_only": 1, "visual_only": 1, "unavailable": 0}
        previous = self.latest_summary
        if (previous is None or result["t_end"] > previous["t_end"]
                or (result["t_end"] == previous["t_end"]
                    and (result.get("event_id") == previous.get("event_id")
                         or priority.get(result.get("status"), 0) >= priority.get(previous.get("status"), 0)))):
            self.latest_summary = result
        self._consider_prompt(result)
        if self.assessment is not None:
            self.assessment.add(result)

    def _clear_content(self, minimum_start=0.):
        self._content_timeline.reset(minimum_start)
        self.content_ready.emit({})

    def _emit_content(self, result):
        if result is not None and self.state == "running":
            self.content_ready.emit(dict(result, session_id=self.session_id, simulation=self.mode == "simulation"))

    def _queue_audio_evidence(self, observation):
        eid = observation.get("event_id")
        if not eid:
            return
        now = time.monotonic()-self.started
        previous = self._audio_evidence.get(eid, {})
        self._audio_evidence[eid] = {"observation": deepcopy(observation), "deadline": now+3,
                                     "revision": previous.get("revision", 0), "published": False}
        self._audio_evidence.move_to_end(eid)
        if "visual" not in self.enabled_modalities or self.last_observed["visual"] >= observation["t_end"]:
            self._fuse_pending_audio(eid)
        while len(self._audio_evidence) > 64:
            first = next(iter(self._audio_evidence))
            if not self._audio_evidence[first]["published"]:
                self._fuse_pending_audio(first)
            self._audio_evidence.pop(first)
            self._write(self.store.append_event, self.session_id, "fusion_buffer_eviction", {"audio_event_id": first})

    def _fuse_pending_audio(self, eid):
        entry = self._audio_evidence[eid]
        result = self.fusion.fuse_audio(entry["observation"])
        entry["revision"] += 1
        result.update(event_id="fusion:"+eid, audio_event_id=eid,
                      revision=entry["revision"], revised=entry["published"])
        if entry["observation"].get("segment_id"):
            result["segment_id"] = entry["observation"]["segment_id"]
        entry["published"] = True
        self._publish_fusion(result)

    def _revise_audio_evidence(self, visual):
        for eid, entry in list(self._audio_evidence.items()):
            audio = entry["observation"]
            if visual["t_start"] < audio["t_end"] and visual["t_end"] > audio["t_start"]:
                if entry["published"] or self.last_observed["visual"] >= audio["t_end"]:
                    self._fuse_pending_audio(eid)

    def set_transcript_allowed(self, allowed):
        if allowed is not False or self.state not in ("running", "paused"):
            return
        self.transcript_allowed = False
        self._clear_content(time.monotonic() - self.started)
        if self.runtime:
            try:
                self.runtime.set_asr_allowed(False)
            except (OSError, RuntimeError, ValueError):
                self.status_changed.emit("转写输出已拦截，采集进程确认失败。")
        return self._submit_transcript_cleanup(self.session_id)

    def _submit_transcript_cleanup(self, session_id):
        if session_id in self.transcript_cleanup_inflight:
            return self.transcript_cleanup_inflight[session_id]
        try:
            # Independent of writer queue capacity, including already queued ASR results.
            self.store.request_transcript_cleanup(session_id)
        except Exception:
            # The in-memory barrier is installed first. Still try the full transaction.
            self.status_changed.emit("转写已停，清理请求暂未写入磁盘，正在尝试清理。")
        self.transcript_cleanup_failed.discard(session_id)
        future = self._write(self.store.set_transcript_consent, session_id, False)
        self.transcript_cleanup_inflight[session_id] = future
        self.transcript_cleanup_changed.emit()

        def finished(result):
            if not self.closed:
                self._transcript_cleanup_finished.emit((session_id, not result.cancelled() and result.exception() is None))
        future.add_done_callback(finished)
        return future

    @Slot(object)
    def _on_transcript_cleanup(self, result):
        if self.closed:
            return
        session_id, success = result
        self.transcript_cleanup_inflight.pop(session_id, None)
        if success:
            self.transcript_cleanup_failed.discard(session_id)
            self.status_changed.emit("转写已停止，已清除语境标注，授权期间的原转写文字保留。")
        else:
            self.transcript_cleanup_failed.add(session_id)
            self.status_changed.emit("转写已停止，语境标注清理失败，可在转写面板重试。")
        self.transcript_cleanup_changed.emit()

    def retry_transcript_cleanup(self):
        for session_id in self.store.pending_transcript_cleanup():
            self._submit_transcript_cleanup(session_id)

    def _consider_prompt(self, result):
        if self.flow.state not in ("observing", "cooldown"):
            return
        now = time.monotonic() - self.started
        self.evidence_gate.observe(result, now)
        self.conflict_gate.observe(result, now)
        self._evaluate_prompt(now)

    def set_prompt_availability(self, mode):
        if mode in self.prompt_timing.modes:
            self.prompt_timing.mode = mode
            self.evidence_gate.reset(self.session_id)
            self.conflict_gate.reset(self.session_id)
            self._prompt_signature = None

    def note_prompt_refusal(self, source=None):
        self.prompt_timing.history.append(("declined", datetime.now(timezone.utc).timestamp()))
        self._write(self.store.append_event, self.session_id, "prompt_declined",
                    {"source": source or self._request_source, "policy_version": self.prompt_timing.version,
                     "decision_id": None if source else self._request_decision_id})

    def _evaluate_prompt(self, now):
        negative = self.evidence_gate.diagnostics(now)
        conflict = self.conflict_gate.diagnostics(now)
        source = ("clarify_conflict" if conflict["ready"] else
                  "sustained_observation" if negative["ready"] else None)
        condition = self.trial["condition"] if self.trial else None
        if condition == "basic" and negative["ready"]:
            source = "sustained_observation"
        evidence = conflict if source == "clarify_conflict" else negative
        policy = self.prompt_timing.evaluate(datetime.now(timezone.utc).timestamp(), source, time.monotonic() < self.cooldown_until)
        policy = trial_prompt(policy,condition,negative["ready"],conflict["ready"],
                              self._trial_unwanted_at is not None and time.monotonic()-self._trial_unwanted_at < 3600)
        if source and (self.flow.current or {}).get("save_status") in ("failed", "pending"):
            policy.update(action="observe", reason="awaiting_feedback_save")
        self.evidence_changed.emit(dict(negative, timing_reason=policy["reason"],
                                        conflict_seconds=conflict["effective_seconds"]))
        signature = (source, policy["action"], policy["reason"])
        decision_id = None
        if source and (policy["action"] == "ask" or self._prompt_signature != signature):
            decision_id = uuid.uuid4().hex
            self._write(self.store.append_event, self.session_id, "prompt_decision",
                        dict(policy, decision_id=decision_id, source_ids=evidence["source_ids"],
                             window_start=evidence.get("window_start"), window_end=evidence.get("window_end")))
        self._prompt_signature = signature
        if policy["action"] == "ask":
            self.request_intervention(source=source, timing_decision_id=decision_id)

    def pause_session(self, paused):
        if self.state not in ("running", "paused"):
            return
        if self.trial:
            if paused:
                self._freeze_trial_exposure()
            elif self.state == "paused":
                self._trial_active_since = time.monotonic()
        self.state = "paused" if paused else "running"
        self._reference_observing_since = (time.monotonic()-self.started
            if not paused and self.flow.state in ("observing", "cooldown") else None)
        self.conflict_gate.reset(self.session_id)
        if self.runtime:
            self.runtime.pause(paused)
        self.negative_start = self.negative_end = None
        self.evidence_gate.reset(self.session_id)
        self._audio_evidence.clear()
        self._clear_content(time.monotonic() - self.started)
        if paused and self.assessment is not None:
            self.skip_assessment(reason="paused")
        self._write(self.store.append_event, self.session_id, "pause" if paused else "resume",
                    {"t": time.monotonic() - self.started})
        self._session_signal()
        self.status_changed.emit("已暂停；设备正在释放，跨暂停语段不保留" if paused else "已恢复采集")

    def stop_session(self):
        if self.state not in ("running", "paused", "stop_failed"):
            return
        if self.flow.state in ("confirming", "ready", "running"):
            self.cancel_activity(reason="session_stopped")
        self.state = "stopping"
        self._clear_content()
        self._simulation_timer.stop()
        self._session_signal()
        sid, runtime = self.session_id, self.runtime
        active_seconds = self._freeze_trial_exposure()
        def release():
            try:
                if runtime:
                    runtime.stop()
                args = (sid,) if active_seconds is None else (sid,active_seconds)
                self._write(self.store.finish_session, *args).result(timeout=10)
            except Exception as exc:
                if not self.closed:
                    self._stop_failed.emit({"session_id": sid, "error": type(exc).__name__})
            else:
                if not self.closed:
                    self._stopped.emit(sid)
        self.background.submit(release)

    def _finish_stop(self, session_id):
        if session_id != self.session_id:
            return
        self.state = "stopped"
        if self.trial:
            self.trial = self.store.trial_assignment(session_id)
        self.runtime = None
        self._session_signal()
        self.status_changed.emit("本次会话已结束；采集和推理进程已退出，设备已释放")
        self.refresh_history()
        self.refresh_recommendations()

    @Slot(dict)
    def _handle_stop_failure(self, result):
        if result.get("session_id") != self.session_id or self.closed:
            return
        self.state = "stop_failed"
        self._session_signal()
        self.status_changed.emit("结束会话未完全成功（" + result["error"] + "），请重试结束；尚未确认设备释放与会话保存")

    @staticmethod
    def _helpfulness(value):
        if not isinstance(value, str):
            raise ValueError("请选择主观帮助程度")
        aliases = {"有帮助": "helpful", "觉得有帮助": "helpful", "无明显帮助": "no_change",
                   "更不舒服": "worse", "无法评价": "unknown", "暂时无法评价": "unknown"}
        value = aliases.get(value, value)
        if value not in ("helpful", "no_change", "worse", "unknown"):
            raise ValueError("请选择主观帮助程度")
        return value

    def refresh_recommendations(self, need=None, available_seconds=None, user_name=None, mode=None):
        """Read only this profile/mode's feedback; late requests cannot cross profiles."""
        empty = Future()
        if self.closed:
            empty.set_result(None)
            return empty
        if need is not None:
            if need not in NEEDS:
                self.status_changed.emit("请选择列表中的当前需要")
                empty.set_result(None)
                return empty
            self.regulation_need = need
        if available_seconds is not None:
            if isinstance(available_seconds, bool) or not isinstance(available_seconds, int) or not 1 <= available_seconds <= 3600:
                self.status_changed.emit("可用时间应为1至3600秒")
                empty.set_result(None)
                return empty
            self.regulation_seconds = available_seconds
        active = self.state in ("running", "paused", "stopping", "stop_failed")
        if mode is not None:
            if mode not in ("real", "simulation") or (active and mode != self.mode):
                self.status_changed.emit("当前会话的来源不能在运行中切换")
                empty.set_result(None)
                return empty
            self.mode = mode
        if user_name is not None:
            name = str(user_name).strip()
            if active and name != self.current_user_name:
                self.status_changed.emit("当前会话结束后才能查看另一档案的推荐")
                empty.set_result(None)
                return empty
            if not active:
                found = next((u for u in self.store.list_users() if u["name"] == name), None)
                self.current_user_id = found["id"] if found else None
                self.current_user_name = name
        self._recommendation_serial += 1
        serial = self._recommendation_serial
        profile_id, profile_name, selected_mode = self.current_user_id, self.current_user_name, self.mode
        selected_need, seconds, pre = self.regulation_need, self.regulation_seconds, self.recommendation_pre
        pending_avoids = self._pending_avoids(profile_id)
        condition = self.trial["condition"] if active and self.trial else None

        def read():
            self._write(lambda: None).result(timeout=10)
            history = self.store.list_interventions(profile_id, selected_mode) if profile_id else []
            preferences = self.store.get_activity_preferences(profile_id) if profile_id else {}
            preferences.update(dict.fromkeys(pending_avoids, "avoid"))
            pending = self.store.list_pending_followups(profile_id, selected_mode) if profile_id else []
            excluded = self.store.historical_discomfort(profile_id, selected_mode) if len(history) >= 200 else set()
            items, decision = contextual_decision(history, selected_need, seconds, pre, preferences, selected_mode, excluded)
            items, decision = trial_order(items,decision,condition)
            payload = {"need": selected_need, "available_seconds": seconds,
                       "items": items, "decision": decision,
                       "stats": summarize_feedback(history, selected_mode), "preferences": preferences,
                       "pending_followups": pending, "mode": selected_mode,
                       "user_name": profile_name, "user_id": profile_id, "history_limit": 200,
                       "_serial": serial}
            if not self.closed:
                self._recommendations_received.emit(payload)
            return payload
        try:
            return self.background.submit(read)
        except RuntimeError:
            empty.set_result(None)
            return empty

    @staticmethod
    def _due_signature(records):
        now = datetime.now(timezone.utc)
        signature = []
        for record in records:
            due = (record.get("followup") or {}).get("due_at")
            try:
                due_time = datetime.fromisoformat(str(due).replace("Z", "+00:00"))
                if due_time.tzinfo is None:
                    due_time = due_time.replace(tzinfo=timezone.utc)
                is_due = due_time <= now
            except (TypeError, ValueError):
                is_due = False
            signature.append((record.get("session_id"), record.get("id"), is_due))
        return tuple(sorted(signature))

    @Slot(dict)
    def _publish_recommendations(self, payload):
        if (self.closed or payload.get("_serial") != self._recommendation_serial
                or payload.get("user_id") != self.current_user_id
                or payload.get("mode") != self.mode or payload.get("user_name") != self.current_user_name):
            return
        payload = deepcopy(payload)
        payload.pop("_serial", None)
        self._last_due_signature = self._due_signature(payload["pending_followups"])
        due = {(sid, rid): is_due for sid, rid, is_due in self._last_due_signature}
        for record in payload["pending_followups"]:
            record["is_due"] = due.get((record.get("session_id"), record.get("id")), False)
        self.personalization = payload
        if self.state in ("running", "paused") and self.flow.state in ("confirming", "ready"):
            self._record_recommendation(payload["decision"])
        self._known_discomfort = {item["activity"] for item in payload["stats"].get("per_activity", [])
                                  if item.get("discomfort_count", 0) > 0}
        self.recommendations_ready.emit(payload)

    def set_recommendation_pre(self, value):
        if value != self.recommendation_pre:
            self.recommendation_pre = value
            self.refresh_recommendations()

    def _record_recommendation(self, decision):
        if self._offered_decision is None or self._offered_decision[0] != decision:
            identity = uuid.uuid4().hex
            self._offered_decision = (deepcopy(decision), identity)
            self._write(self.store.append_event, self.session_id, "recommendation_offered",
                        dict(decision, decision_id=identity))

    def _refresh_after_write(self, future, message=None, record_id=None, require_result=False):
        context = {"user_id": self.current_user_id, "mode": self.mode,
                   "session_id": self.session_id, "record_id": record_id, "message": message,
                   "write_revision": self._record_write_versions.get(record_id)}
        def finished(result):
            try:
                updated = result.result()
                payload = dict(context, success=True, updated=updated)
                if require_result and updated is None:
                    payload["message"] = None
            except Exception as exc:
                payload = dict(context, success=False, error=type(exc).__name__)
            if not self.closed:
                self._write_completed.emit(payload)
        future.add_done_callback(finished)
        return future

    def set_activity_preference(self, activity, preference):
        if activity not in ACTIVITIES or preference not in ("prefer", "avoid", "neutral"):
            self.status_changed.emit("请选择有效的活动与偏好")
            return None
        if not self.current_user_id:
            self.status_changed.emit("请先选择已有档案，或开始一次会话后保存偏好")
            return None
        self._recommendation_serial += 1
        future = self._write(self.store.save_activity_preference, self.current_user_id, activity, preference)
        key = (self.current_user_id, activity)
        if preference == "avoid" or key in self._avoid_writes:
            self._avoid_writes[key] = future
        return self._refresh_after_write(future)

    def _pending_avoids(self, profile):
        """Honor an explicit refusal until the latest preference write has committed."""
        result = set()
        for (owner, activity), future in list(self._avoid_writes.items()):
            if future.done() and not future.cancelled() and future.exception() is None:
                del self._avoid_writes[owner, activity]
            elif owner == profile:
                result.add(activity)
        return result

    def _intervention_signal(self):
        if self.flow.state not in ("observing", "cooldown") or self.state != "running":
            self._reference_observing_since = None
        elif self._reference_observing_since is None:
            self._reference_observing_since = time.monotonic()-self.started
        current = self.flow.current
        if current and self.task_details:
            current["task_details"] = self.task_details
        remaining = None
        if current and self.flow.state == "running" and self.activity_started is not None:
            remaining = max(0, current["duration"] - int(time.monotonic() - self.activity_started))
        self.intervention_changed.emit({"state": self.flow.state, "current": current, "source": self._request_source,
                                        "remaining_seconds": remaining,
                                        "assessment": self.assessment.snapshot(time.monotonic()-self.started) if self.assessment else None,
                                        "need": self.regulation_need, "mode": self.mode})

    def request_intervention(self, source="user", timing_decision_id=None):
        if self.state not in ("running", "paused"):
            self.status_changed.emit("先开始一次会话，再记录调节过程")
            return
        if (self.flow.current or {}).get("save_status") == "failed":
            self.status_changed.emit("请先等待本次反馈保存，失败时可点击重试保存")
            return
        if self.flow.request():
            self._request_source = source
            self._request_decision_id = timing_decision_id
            self._offered_decision = None
            self.recommendation_pre = None
            if source in ("sustained_observation", "clarify_conflict"):
                self.prompt_timing.history.append(("asked", datetime.now(timezone.utc).timestamp()))
                self._write(self.store.append_event, self.session_id, "prompt_asked",
                            {"source": source, "decision_id": timing_decision_id})
            self.task_details = {}
            self.pre_model = None
            self.assessment = None
            self.pre_assessment = self.post_assessment = None
            self._assessment_phase = None
            self.activity_started = None
            if source != "alternative":
                self._pending_previous_id = self._pending_root_id = None
            self._write(self.store.append_event, self.session_id, "intervention_requested",
                        {"source": source, "decision_id": timing_decision_id, "t": time.monotonic() - self.started})
            self._intervention_signal()
            self.refresh_recommendations()

    def confirm_intervention(self, accepted):
        try:
            self.flow.confirm(bool(accepted))
            self._write(self.store.append_event, self.session_id, "intervention_confirmation",
                        {"accepted": bool(accepted), "decision_id": self._request_decision_id,
                         "t": time.monotonic() - self.started})
            if not accepted:
                self.cooldown_until = time.monotonic() + 1200
                if self._request_source in ("sustained_observation", "clarify_conflict"):
                    self.note_prompt_refusal()
            else:
                self.assessment = AssessmentWindowCollector()
                self._assessment_phase = "pre"
                self.assessment.begin(self.session_id, uuid.uuid4().hex, "pre", time.monotonic()-self.started)
            self._intervention_signal()
        except ValueError as exc:
            self.status_changed.emit(str(exc))

    def start_activity(self, activity, pre, duration, task_details=None, context=None):
        if self.state not in ("running", "paused"):
            return
        if activity not in ACTIVITIES:
            self.status_changed.emit("请选择列表中的活动")
            return
        try:
            context = deepcopy(context or {})
            if not isinstance(context, dict):
                raise ValueError("活动情境应为结构化信息")
            need = context.get("need", self.regulation_need)
            if need not in NEEDS:
                raise ValueError("请选择列表中的当前需要")
            seconds = context.get("available_seconds", self.regulation_seconds)
            if isinstance(seconds, bool) or not isinstance(seconds, int) or not 1 <= seconds <= 3600:
                raise ValueError("可用时间应为1至3600秒")
            history = self.store.list_interventions(self.current_user_id, self.mode)
            pending_avoids = self._pending_avoids(self.current_user_id)
            preferences = self.store.get_activity_preferences(self.current_user_id)
            preferences.update(dict.fromkeys(pending_avoids, "avoid"))
            excluded = self.store.historical_discomfort(self.current_user_id, self.mode) if len(history) >= 200 else set()
            candidates, decision = contextual_decision(history, need, seconds, pre, preferences, self.mode, excluded)
            candidates, decision = trial_order(candidates,decision,self.trial["condition"] if self.trial else None)
            recommendation = next(item for item in candidates if item["activity"] == activity)
            if recommendation.get("blocked") or activity in self._known_discomfort:
                raise ValueError("此方法已因偏好或不适反馈排除，请选择其他活动或调整偏好")
            record = self.flow.start(activity, pre, min(600, max(1, int(duration))))
            offer = self._offered_decision
            decision = dict(decision, decision_id=uuid.uuid4().hex, offered_decision_id=offer[1] if offer else None,
                            prompt_decision_id=self._request_decision_id, user_selected_action=activity, outcome_id=record["id"])
            self.activity_started = time.monotonic()
            self.task_details = task_details or {}
            self._finish_assessment_for_activity(record["id"])
            self.pre_model = self.pre_assessment if self.pre_assessment and self.pre_assessment.get("status") == "complete" else None
            self.regulation_need, self.regulation_seconds = need, seconds
            context.update(need=need, available_seconds=seconds)
            record = self.flow.annotate({
                "task_details": self.task_details, "pre_model": self.pre_model,
                "assessments": {"pre": self.pre_assessment, "post": None},
                "assessment_phase": "during", "save_status": "pending",
                "started_session_t": self.activity_started - self.started,
                "context": context, "steps": deepcopy(ACTIVITIES[activity]["steps"]),
                "recommendation": recommendation, "recommendation_mode": self.mode, "decision_log": decision,
                "selection": "user_selected", "user_id": self.current_user_id,
                "mode": self.mode, "session_id": self.session_id,
                "previous_intervention_id": self._pending_previous_id,
                "root_intervention_id": self._pending_root_id or record["id"],
                "adaptation": "按自己的节奏逐步进行，可以跳过或随时停止。",
            })
            self._pending_previous_id = self._pending_root_id = None
            self._save_bundle(record, [{"phase": "pre", "distress": pre}], event_kind="activity_started")
            self._intervention_signal()
        except (ValueError, TypeError) as exc:
            self.status_changed.emit(f"活动未开始：{exc}")

    def _finish_assessment_for_activity(self, intervention_id):
        if self.assessment is not None and self._assessment_phase == "pre":
            now = time.monotonic()-self.started
            self.assessment.advance(now)
            self.pre_assessment = self.assessment.finish(now, skip=self.assessment.lifecycle == "collecting")
            self.pre_assessment["intervention_id"] = intervention_id
        self.assessment = None
        self._assessment_phase = "during"

    def begin_post_assessment(self):
        if self.flow.state != "running" or self.state != "running":
            return
        if self._assessment_phase == "post":
            return
        self.assessment = AssessmentWindowCollector()
        self._assessment_phase = "post"
        self.post_assessment = None
        protocol = (self.pre_assessment or {}).get("comparison_protocol")
        self.assessment.begin(self.session_id, self.flow.current["id"], "post",
                              time.monotonic()-self.started, comparison_protocol=protocol)
        self.flow.annotate({"assessment_phase": "post"})
        self._intervention_signal()

    def resume_activity_guidance(self):
        if self.flow.state != "running":
            return
        if self.assessment is not None and self._assessment_phase == "post":
            self.skip_assessment(reason="returned_to_guidance")
        self.assessment = None
        self.post_assessment = None
        self._assessment_phase = "during"
        self.flow.annotate({"assessment_phase": "during"})
        self._intervention_signal()

    def skip_assessment(self, reason="user_skipped"):
        if self.assessment is None:
            return
        result = self.assessment.finish(time.monotonic()-self.started, skip=True)
        if result["status"] != "skipped":
            return
        result["skip_reason"] = reason if isinstance(reason, str) else "user_skipped"
        if self._assessment_phase == "pre":
            self.pre_assessment = result
        else:
            self.post_assessment = result
        if self.flow.current:
            self.flow.annotate({"assessments": {"pre": self.pre_assessment, "post": self.post_assessment}})
        self._intervention_signal()

    def _save_bundle(self, record, reports=None, preference="unchanged", event_kind=None, message=None):
        self._record_write_versions[record["id"]] = self._record_write_versions.get(record["id"], 0) + 1
        persisted = deepcopy(record)
        persisted["save_status"] = "saved"
        reports = list(reports or [])
        if isinstance(record.get("pre"), int) and not isinstance(record.get("pre"), bool) and not any(item["phase"] == "pre" for item in reports):
            reports.append({"phase": "pre", "distress": record["pre"]})
        args = (self.session_id, persisted, reports, preference, event_kind,
                {"intervention_id": record["id"], "loop_result": record.get("loop_result"),
                 "simulation": self.mode == "simulation"})
        self._write_context[record["id"]] = (self.store.save_activity_bundle, args, message)
        return self._refresh_after_write(self._write(self.store.save_activity_bundle, *args),
                                         message, record["id"])

    def retry_activity_save(self):
        current = self.flow.current or {}
        pending = self._write_context.get(current.get("id"))
        if pending and current.get("save_status") == "failed":
            function, args, message = pending
            self.flow.annotate({"save_status": "pending"})
            self._record_write_versions[current["id"]] = self._record_write_versions.get(current["id"], 0) + 1
            self._intervention_signal()
            self._refresh_after_write(self._write(function, *args), message, current["id"])

    def advance_activity_step(self):
        try:
            record = self.flow.current or {}
            steps = record.get("steps") or ACTIVITIES.get(record.get("activity"), {}).get("steps", [])
            record = self.flow.advance_step(len(steps))
            self._write(self.store.save_intervention, self.session_id, record)
            self._write(self.store.append_event, self.session_id, "activity_step",
                        {"intervention_id": record["id"], "step_index": record["step_index"]})
            self._intervention_signal()
        except ValueError as exc:
            self.status_changed.emit(str(exc))

    def activity_checkin(self, feeling):
        try:
            record = self.flow.checkin(feeling)
            messages = {"comfortable": "已记录这一步的帮助感受，可按自己的节奏继续。",
                        "no_change": "没有变化也可以记录；可跳到下一步，或由你选择更换方法。",
                        "uncomfortable": "已停止引导并记录不适；此方法不再进入当前档案的优先推荐。"}
            self.flow.annotate({"adaptation": messages[feeling]})
            self._write(self.store.append_event, self.session_id, "activity_checkin",
                        {"intervention_id": record["id"], "feeling": feeling,
                         "step_index": record.get("step_index", 0)})
            if feeling == "uncomfortable":
                self._recommendation_serial += 1
                self._known_discomfort.add(record["activity"])
                self.cancel_activity(reason="uncomfortable")
            else:
                self._write(self.store.save_intervention, self.session_id, self.flow.current)
                self._intervention_signal()
            self.status_changed.emit(messages[feeling])
        except ValueError as exc:
            self.status_changed.emit(str(exc))

    def request_alternative(self):
        current = self.flow.current
        if self.flow.state != "running" or not current:
            return
        previous_id = current["id"]
        root_id = current.get("root_intervention_id") or previous_id
        self.cancel_activity(reason="manual_switch")
        self._pending_previous_id, self._pending_root_id = previous_id, root_id
        self.request_intervention(source="alternative")

    def complete_activity(self, post, helpfulness, followup_seconds=None,
                          next_step="", preference="unchanged", feedback_note=""):
        try:
            helpfulness = self._helpfulness(helpfulness)
            if preference not in ("unchanged", "neutral", "prefer", "avoid"):
                raise ValueError("请选择有效的下次推荐偏好")
            if followup_seconds is not None and (isinstance(followup_seconds, bool)
                    or not isinstance(followup_seconds, int) or not 30 <= followup_seconds <= 3600):
                raise ValueError("延迟复测时间应为30至3600秒")
            record = self.flow.complete(post, helpfulness)
            self._recommendation_serial += 1
            if self.assessment is not None and self._assessment_phase == "post":
                now_t = time.monotonic()-self.started
                self.assessment.advance(now_t)
                self.post_assessment = self.assessment.finish(now_t, skip=self.assessment.lifecycle == "collecting")
            comparison = compare_assessments(self.pre_assessment, self.post_assessment)
            comparable = comparison.get("comparable", False)
            now = datetime.now(timezone.utc)
            followup = ({"status": "pending", "scheduled_at": now.isoformat(timespec="milliseconds"),
                         "due_at": (now + timedelta(seconds=followup_seconds)).isoformat(timespec="milliseconds"),
                         "distress": None, "helpfulness": None} if followup_seconds is not None else None)
            record = self.flow.annotate({
                "task_details": self.task_details, "pre_model": self.pre_model,
                "post_model": self.post_assessment,
                "assessments": {"pre": self.pre_assessment, "post": self.post_assessment},
                "assessment_comparison": comparison, "assessment_phase": "finished", "save_status": "pending",
                "started_session_t": (self.activity_started or self.started) - self.started,
                "model_comparison": ("independent_windows_observation_only" if comparable else
                                     "per_modality_only" if comparison.get("per_modality_delta") else "insufficient"),
                "completed_session_t": time.monotonic() - self.started,
                "loop_result": evaluate_outcome(record["pre"], post, helpfulness),
                "next_step": str(next_step).strip()[:1000], "feedback_note": str(feedback_note).strip()[:2000],
                "preference_action": preference, "followup": followup,
                "discomfort": bool(record.get("discomfort") or helpfulness == "worse"),
            })
            if helpfulness == "worse":
                self._known_discomfort.add(record["activity"])
            self.cooldown_until = time.monotonic() + 1200
            reports = [{"phase": "post", "distress": post}] if post is not None else []
            self._save_bundle(record, reports, preference, "activity_completed",
                              "已保存即时复测与反馈" + ("，稍后可在应用内完成延迟复测" if followup else ""))
            self.assessment = None
            self._intervention_signal()
        except ValueError as exc:
            self.status_changed.emit(str(exc))

    def cancel_activity(self, reason="user_cancelled"):
        # QPushButton.clicked(bool) may supply False to an optional argument.
        # Keep that native signal from becoming a boolean stop reason.
        if not isinstance(reason, str) or not reason:
            reason = "user_cancelled"
        try:
            record = self.flow.cancel(reason)
            metadata = {"task_details": self.task_details, "pre_model": self.pre_model,
                        "stop_reason": reason, "discomfort": bool(record.get("discomfort") or reason == "uncomfortable"),
                        "save_status": "pending",
                        "loop_result": evaluate_outcome(record.get("pre"), None,
                                                        "worse" if reason == "uncomfortable" else "unknown")}
            if self.activity_started is not None:
                metadata["started_session_t"] = self.activity_started - self.started
            record = self.flow.annotate(metadata)
            self.assessment = None
            self._save_bundle(record, event_kind="activity_cancelled", message="已保存中止记录与反馈")
            self.cooldown_until = time.monotonic() + 1200
            self._intervention_signal()
        except ValueError:
            pass

    def submit_followup(self, session_id, intervention_id, distress, helpfulness="unknown"):
        """Resolve an opt-in follow-up once, against its original profile/session."""
        try:
            distress = InterventionFlow._distress(distress, optional=True)
            helpfulness = self._helpfulness(helpfulness)
        except ValueError as exc:
            self.status_changed.emit(str(exc))
            return None
        if not self.current_user_id:
            self.status_changed.emit("请先选择原会话所属的个人档案")
            return None
        profile_id, selected_mode = self.current_user_id, self.mode
        self._record_write_versions[intervention_id] = self._record_write_versions.get(intervention_id, 0) + 1
        def resolve_followup():
            try:
                return self.store.complete_followup(session_id, intervention_id, profile_id,
                                                    selected_mode, distress, helpfulness)
            except ValueError:
                self.status_changed.emit("延迟复测不属于当前档案或来源，未写入记录")
                return None
        future = self._write(resolve_followup)
        return self._refresh_after_write(future, "已保存延迟复测；下次推荐将参考这次反馈",
                                         intervention_id, require_result=True)

    def feedback_target(self):
        """Freeze what was displayed when the user opened the feedback dialog."""
        return {"session_id": self.session_id, "observation": deepcopy(self.latest_summary),
                "viewed_at": time.monotonic()-self.started}

    def correct_observation(self, label, feedback_type="subjective_correction", target=None):
        if not self.session_id or self.state not in ("running", "paused"):
            return
        target = self.feedback_target() if target is None else target
        if target["session_id"] != self.session_id:
            self.status_changed.emit("原会话已结束，请重新选择当前观察。")
            return
        try:
            record = FeedbackRecord.create(self.current_user_id, self.session_id, feedback_type, label,
                self.personal_learning and self.state == "running", target["observation"], target["viewed_at"])
        except ValueError as exc:
            self.status_changed.emit(str(exc))
            return
        if feedback_type in ("do_not_disturb", "subjective_correction"):
            self.cooldown_until = time.monotonic() + 1200
            self.negative_start = self.negative_end = None
            self.evidence_gate.reset(self.session_id)
        if feedback_type == "do_not_disturb":
            self.note_prompt_refusal("user_timing_feedback")
        messages = {"subjective_correction": "已记录你的主观修正，主动询问暂停 20 分钟。",
                    "distress_report": "已记录当前困扰，可在调节页自行选择活动。",
                    "do_not_disturb": "已记录时机反馈，主动询问暂停 20 分钟。",
                    "typical_reference": "已记录日常参考反馈，会话结束后按来源和采集条件检查。"}
        return self._refresh_after_write(self._write(self.store.save_observation_feedback, record),
                                         messages[feedback_type])

    def pause_personal_learning(self):
        if self.state not in ("running", "paused") or not self.personal_learning:
            return
        self.store.pause_personal_learning(self.session_id)
        self.personal_learning = False
        self._personal_reference_state = "paused"
        self.status_changed.emit("本次个人学习已暂停。")

    def reference_for_profile(self, name, mode):
        if self.state in ("running", "paused", "stopping", "stop_failed"):
            reference = (self.store.personal_reference(self.current_user_id, self.mode)
                         if self._personal_reference_state == "disabled" else self.frozen_reference)
            return self.current_user_id, reference
        user_id = next((u["id"] for u in self.store.list_users() if u["name"] == name.strip()), None)
        return user_id, self.store.personal_reference(user_id, mode) if user_id else build_reference([])

    def reset_personal_reference(self, name, mode):
        uid, _ = self.reference_for_profile(name, mode)
        if uid:
            active = self.state in ("running", "paused", "stopping", "stop_failed")
            self.store.reset_personal_reference(uid, self.mode if active else mode)
            if active:
                self.frozen_reference = build_reference([])
                self.personal_learning = False
                self._personal_reference_state = "reset"
            self.status_changed.emit("已清除该档案在当前来源下的个人参考。")

    def _activity_tick(self):
        now = time.monotonic()-self.started
        if self.state == "running":
            for eid, entry in list(self._audio_evidence.items()):
                if not entry["published"] and entry["deadline"] <= now:
                    self._fuse_pending_audio(eid)
                if entry["observation"]["t_end"] < now-120:
                    self._audio_evidence.pop(eid, None)
        if self.state == "running" and self.flow.state in ("observing", "cooldown"):
            self._evaluate_prompt(now)
        if self.assessment is not None:
            self.assessment.advance(now)
            if self.assessment.lifecycle == "finalized":
                if self._assessment_phase == "pre" and self.pre_assessment is None:
                    self.pre_assessment = self.assessment.snapshot(now)
                elif self._assessment_phase == "post" and self.post_assessment is None:
                    self.post_assessment = self.assessment.snapshot(now)
            if self.flow.state == "ready":
                self._intervention_signal()
        if self.flow.state == "running":
            self._intervention_signal()
        pending = self.personalization.get("pending_followups", [])
        signature = self._due_signature(pending)
        if not self.closed and signature != self._last_due_signature:
            self._last_due_signature = signature
            self.refresh_recommendations()

    def _check_freshness(self):
        if self.state != "running" or self.mode != "real":
            return
        now = time.monotonic() - self.started
        for modality in self.enabled_modalities:
            maximum_age = 3 if modality == "visual" else 10
            if now - self.last_observed[modality] <= maximum_age:
                continue
            reason = ("视频暂不可用或尚未返回结果" if modality == "visual"
                      else "暂无新的有效语音片段；静音不记为中性")
            observation = invalid_observation(self.session_id, modality, max(0, now - 1), now, reason)
            observation["stale"] = True
            self._handle_event({"kind": "observation", "observation": observation})

    def _record_resources(self):
        if self.state not in ("running", "paused") or not self.session_id:
            return
        try:
            import psutil
        except ImportError:
            return
        try:
            process = psutil.Process()
            all_processes = [process] + process.children(recursive=True)
            rss, times = 0, {}
            for child in all_processes:
                try:
                    with child.oneshot():
                        cpu = child.cpu_times()
                        size = child.memory_info().rss
                        times[(child.pid, child.create_time())] = cpu.user+cpu.system
                        rss += size
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            now = time.monotonic()
            previous = getattr(self, "_last_resource_sample", None)
            percent = None
            if previous and previous[0] == self.session_id and now > previous[1]:
                delta = sum(max(0., value-previous[2][key]) for key, value in times.items() if key in previous[2])
                percent = 100*delta/(now-previous[1])
            self._last_resource_sample = (self.session_id, now, times)
            capture = self.runtime.telemetry() if self.runtime is not None and hasattr(self.runtime, "telemetry") else None
            self._write(self.store.append_event, self.session_id, "resource",
                        {"t": now-self.started, "rss_bytes": rss, "processes": len(times),
                         "cpu_percent_one_core": percent, "capture": capture})
        except (psutil.Error, OSError):
            return

    def refresh_history(self):
        if self.closed:
            return
        def read():
            # This barrier ensures pending writes are visible before reading.
            self._write(lambda: None).result(timeout=10)
            history = self.store.list_sessions()
            if not self.closed:
                self._history_received.emit(history)
        try:
            self.background.submit(read)
        except RuntimeError:
            return

    def export_session(self, session_id, path):
        def export():
            target = self.store.export_session(session_id, path)
            self.status_changed.emit(f"已导出完整会话：{target}")
        self._write(export)

    def delete_session(self, session_id):
        if session_id == self.session_id and self.state in ("running", "paused", "stopping", "stop_failed"):
            self.status_changed.emit("请先结束当前会话，再删除记录")
            return
        future = self._write(self.store.delete_session, session_id)
        self._recommendation_serial += 1
        self._refresh_after_write(future)

    def shutdown(self):
        if self.closed:
            return
        # Freeze at the user's exit, before saving/cancelling an activity and
        # waiting for device workers to release resources.
        exposure = self._freeze_trial_exposure()
        self._simulation_timer.stop()
        self._activity_timer.stop()
        self._resource_timer.stop()
        self._freshness_timer.stop()
        if self.flow.state in ("confirming", "ready", "running") and self.session_id:
            self.cancel_activity(reason="app_closed")
        self.closed = True
        self._clear_content()
        if self.runtime:
            self.runtime.stop()
            self.runtime = None
        if self.session_id and self.state in ("running", "paused"):
            args = (self.session_id,) if exposure is None else (self.session_id,exposure)
            self._write(self.store.finish_session, *args)
        self.state = "stopped"
        self.background.shutdown(wait=True, cancel_futures=True)
        self.writer.shutdown(wait=True)
        self.store.close()
