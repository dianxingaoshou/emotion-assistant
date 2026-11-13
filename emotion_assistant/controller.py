"""Qt-facing orchestration. Device/model work is isolated from the UI process."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from collections import OrderedDict
from copy import deepcopy
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import threading
import time
import uuid

from PySide6.QtCore import QObject, QTimer, Signal, Slot, Qt

from .capture import CaptureRuntime, invalid_observation
from .core import FusionEngine, InterventionFlow
from .paths import DATA, MODELS
from .storage import Store
from .regulation import ACTIVITIES, NEEDS, evaluate_outcome, recommend_activities, summarize_feedback
from .evidence import SustainedEvidenceGate
from .assessment import AssessmentWindowCollector, compare_assessments


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

    def __init__(self, database=None, parent=None):
        super().__init__(parent)
        self.store = Store(database or DATA / "sessions.sqlite3")
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
        self.state = "idle"
        self.mode = "real"
        self.runtime = None
        self.fusion = FusionEngine()
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
        self.personalization = {}
        self._recommendation_serial = 0
        self._last_due_signature = ()
        self._pending_previous_id = None
        self._pending_root_id = None
        self._known_discomfort = set()
        self.evidence_gate = SustainedEvidenceGate()
        self.assessment = None
        self.pre_assessment = self.post_assessment = None
        self._assessment_phase = None
        self._write_context = {}
        self._record_write_versions = {}
        self._audio_evidence = OrderedDict()
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
        self.session_changed.emit({"state": self.state, "session_id": self.session_id,
                                   "mode": self.mode, "user_name": self.current_user_name})

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
        self.mode = "simulation" if mode == "simulation" else "real"
        users = self.store.list_users()
        user_id = next((u["id"] for u in users if u["name"] == user_name.strip()), None)
        user_id = user_id or self.store.create_user(user_name.strip())
        self.current_user_id, self.current_user_name = user_id, user_name.strip()
        self._recommendation_serial += 1
        self.personalization = {}
        self._known_discomfort = set()
        self._pending_previous_id = self._pending_root_id = None
        consent = {"collection": True, "transcript": consent.get("transcript") is True,
                   "save_transcript": consent.get("transcript") is True, "raw_media": False}
        self.transcript_allowed = consent["transcript"]
        self.session_id = self.store.start_session(user_id, consent, self.mode)
        self.state = "running"
        self.started = time.monotonic()
        self.fusion = FusionEngine()
        self.flow = InterventionFlow()
        self.evidence_gate.reset(self.session_id)
        self._audio_evidence.clear()
        self.assessment = None
        self.pre_assessment = self.post_assessment = None
        self._assessment_phase = None
        self.negative_start = self.negative_end = None
        self.latest_summary = self.pre_model = None
        self.last_observed = {"visual": 0.0, "audio": 0.0}
        self.enabled_modalities = ({"visual"} if camera_index is not None and camera_index >= 0 else set())
        if mic_index is not None:
            self.enabled_modalities.add("audio")
        self.cooldown_until = 0
        self.task_details = {}
        self.activity_started = None
        self._session_signal()
        self._intervention_signal()
        self._write(self.store.append_event, self.session_id, "configuration",
                    {"camera_index": camera_index, "mic_index": mic_index,
                     "video_resolution": [640, 480], "preview_fps_target": 15,
                     "visual_inference_fps_target": 2, "audio_sample_rate": 16000,
                     "fusion_min_quality": 0.2, "sustained_seconds": 20,
                     "cooldown_seconds": 1200, "raw_media_saved": False,
                     "app_version": "0.1.0", "simulation": self.mode == "simulation"})
        if self.mode == "simulation":
            self._sim_tick = 0
            self._simulation_timer.start()
            self.status_changed.emit("模拟回放已启动：全部线索为测试输入，不是摄像头/麦克风实测")
        else:
            self.runtime = CaptureRuntime(self.session_id, int(camera_index if camera_index is not None else -1),
                                          mic_index, MODELS, self.frame_ready.emit,
                                          self._event_received.emit, consent["transcript"])
            try:
                self.runtime.start()
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
            if not self.transcript_allowed or utterance.get("session_id", self.session_id) != self.session_id:
                return
            utterance.setdefault("session_id", self.session_id)
            self.transcript_ready.emit(utterance)
            self._write(self.store.save_utterance, self.session_id, utterance, True)
        else:
            self.status_changed.emit(event.get("message", str(kind)))
            self._write(self.store.append_event, self.session_id, kind or "status", event)

    def _publish_fusion(self, result):
        result.update(session_id=self.session_id, processed_at=time.monotonic()-self.started,
                      simulation=self.mode == "simulation")
        self.fusion_ready.emit(result)
        self._write(self.store.save_observation, self.session_id, result)
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
        if self.runtime:
            self.runtime.set_asr_allowed(False)
        future = self._write(self.store.set_transcript_consent, self.session_id, False)
        return self._refresh_after_write(future, "已停止本次转写，之前获授权的历史保留")

    def _consider_prompt(self, result):
        if self.flow.state not in ("observing", "cooldown") or time.monotonic() < self.cooldown_until:
            return
        now = time.monotonic() - self.started
        if self.evidence_gate.observe(result, now):
            self.request_intervention(source="sustained_observation")
        self.evidence_changed.emit(self.evidence_gate.diagnostics(now))

    def pause_session(self, paused):
        if self.state not in ("running", "paused"):
            return
        self.state = "paused" if paused else "running"
        if self.runtime:
            self.runtime.pause(paused)
        self.negative_start = self.negative_end = None
        self.evidence_gate.reset(self.session_id)
        self._audio_evidence.clear()
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
        self._simulation_timer.stop()
        self._session_signal()
        sid, runtime = self.session_id, self.runtime
        def release():
            try:
                if runtime:
                    runtime.stop()
                self._write(self.store.finish_session, sid).result(timeout=10)
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
        selected_need, seconds = self.regulation_need, self.regulation_seconds

        def read():
            self._write(lambda: None).result(timeout=10)
            history = self.store.list_interventions(profile_id, selected_mode) if profile_id else []
            preferences = self.store.get_activity_preferences(profile_id) if profile_id else {}
            pending = self.store.list_pending_followups(profile_id, selected_mode) if profile_id else []
            payload = {"need": selected_need, "available_seconds": seconds,
                       "items": recommend_activities(history, selected_need, seconds, preferences, selected_mode),
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
        self._known_discomfort = {item["activity"] for item in payload["stats"].get("per_activity", [])
                                  if item.get("discomfort_count", 0) > 0}
        self.recommendations_ready.emit(payload)

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
        return self._refresh_after_write(future)

    def _intervention_signal(self):
        current = self.flow.current
        if current and self.task_details:
            current["task_details"] = self.task_details
        remaining = None
        if current and self.flow.state == "running" and self.activity_started is not None:
            remaining = max(0, current["duration"] - int(time.monotonic() - self.activity_started))
        self.intervention_changed.emit({"state": self.flow.state, "current": current,
                                        "remaining_seconds": remaining,
                                        "assessment": self.assessment.snapshot(time.monotonic()-self.started) if self.assessment else None,
                                        "need": self.regulation_need, "mode": self.mode})

    def request_intervention(self, source="user"):
        if self.state not in ("running", "paused"):
            self.status_changed.emit("先开始一次会话，再记录调节过程")
            return
        if (self.flow.current or {}).get("save_status") == "failed":
            self.status_changed.emit("请先等待本次反馈保存，失败时可点击重试保存")
            return
        if self.flow.request():
            self.task_details = {}
            self.pre_model = None
            self.assessment = None
            self.pre_assessment = self.post_assessment = None
            self._assessment_phase = None
            self.activity_started = None
            if source != "alternative":
                self._pending_previous_id = self._pending_root_id = None
            self._write(self.store.append_event, self.session_id, "intervention_requested",
                        {"source": source, "t": time.monotonic() - self.started})
            self._intervention_signal()
            self.refresh_recommendations()

    def confirm_intervention(self, accepted):
        try:
            self.flow.confirm(bool(accepted))
            self._write(self.store.append_event, self.session_id, "intervention_confirmation",
                        {"accepted": bool(accepted), "t": time.monotonic() - self.started})
            if not accepted:
                self.cooldown_until = time.monotonic() + 1200
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
            preferences = self.store.get_activity_preferences(self.current_user_id)
            candidates = recommend_activities(history, need, seconds, preferences, self.mode)
            recommendation = next(item for item in candidates if item["activity"] == activity)
            if recommendation.get("blocked") or activity in self._known_discomfort:
                raise ValueError("此方法已因偏好或不适反馈排除，请选择其他活动或调整偏好")
            record = self.flow.start(activity, pre, min(600, max(1, int(duration))))
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
                "recommendation": recommendation, "recommendation_mode": self.mode,
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
            snapshot = self.assessment.snapshot(now)
            self.pre_assessment = self.assessment.finish(now, skip=snapshot.get("status") == "collecting")
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
        self.assessment.begin(self.session_id, self.flow.current["id"], "post", time.monotonic()-self.started)
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
                self.post_assessment = self.assessment.finish(now_t, skip=self.assessment.snapshot(now_t).get("status") == "collecting")
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
                "model_comparison": "independent_windows_observation_only" if comparable else "insufficient",
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

    def correct_observation(self, label):
        if not self.session_id or self.state not in ("running", "paused"):
            return
        self._write(self.store.append_event, self.session_id, "user_correction",
                    {"label": str(label), "t": time.monotonic() - self.started})
        self.cooldown_until = time.monotonic() + 1200
        self.negative_start = self.negative_end = None
        self.evidence_gate.reset(self.session_id)
        self.status_changed.emit("已记录你的修正；20分钟内停止主动询问，仍可随时主动调节")

    def _activity_tick(self):
        now = time.monotonic()-self.started
        if self.state == "running":
            for eid, entry in list(self._audio_evidence.items()):
                if not entry["published"] and entry["deadline"] <= now:
                    self._fuse_pending_audio(eid)
                if entry["observation"]["t_end"] < now-120:
                    self._audio_evidence.pop(eid, None)
        if (self.state == "running" and self.flow.state in ("observing", "cooldown")
                and time.monotonic() >= self.cooldown_until and self.evidence_gate.ready(now)):
            self.request_intervention(source="sustained_observation")
        if self.assessment is not None:
            snapshot = self.assessment.snapshot(now)
            if snapshot.get("remaining_seconds", 0) <= 0 and snapshot.get("status") == "collecting":
                result = self.assessment.finish(now)
                if self._assessment_phase == "pre":
                    self.pre_assessment = result
                else:
                    self.post_assessment = result
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
            process = psutil.Process()
            all_processes = [process] + process.children(recursive=True)
            rss = sum(p.memory_info().rss for p in all_processes if p.is_running())
            self._write(self.store.append_event, self.session_id, "resource",
                        {"t": time.monotonic() - self.started, "rss_bytes": rss,
                         "processes": len(all_processes)})
        except Exception:
            pass

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
        self._simulation_timer.stop()
        self._activity_timer.stop()
        self._resource_timer.stop()
        self._freshness_timer.stop()
        if self.flow.state in ("confirming", "ready", "running") and self.session_id:
            self.cancel_activity(reason="app_closed")
        self.closed = True
        if self.runtime:
            self.runtime.stop()
            self.runtime = None
        if self.session_id and self.state in ("running", "paused"):
            self._write(self.store.finish_session, self.session_id)
        self.state = "stopped"
        self.background.shutdown(wait=True, cancel_futures=True)
        self.writer.shutdown(wait=True)
        self.store.close()
