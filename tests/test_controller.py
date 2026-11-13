"""Integration tests for the user-visible closed loop; no real devices opened."""
import json
import os
from pathlib import Path
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtWidgets import QApplication
import pytest

from emotion_assistant.controller import Controller, simulation_observation


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def controller(app):
    instance = Controller(database=":memory:")
    yield instance
    instance.shutdown()


def begin(controller):
    controller.start_session("测试档案", -1, None,
                             {"collection": True, "transcript": False}, "simulation")
    controller._simulation_timer.stop()
    return controller.session_id


def snapshot(controller, tmp_path):
    controller.writer.submit(lambda: None).result(timeout=5)
    path = controller.store.export_session(controller.session_id, tmp_path / "session.json")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_consent_required_and_start_does_not_create_real_runtime(controller):
    controller.start_session("测试", -1, None, {"collection": False}, "simulation")
    assert controller.session_id is None
    begin(controller)
    assert controller.mode == "simulation" and controller.runtime is None


def test_full_closed_loop_persists_ratings_activity_and_trace(controller, tmp_path):
    sid = begin(controller)
    for start, end in ((0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2)):
        obs = simulation_observation(sid, "visual", start, end)
        controller._handle_event({"kind": "observation", "observation": obs})
    controller._handle_event({"kind": "observation", "observation": simulation_observation(sid, "audio", 0, 2)})
    controller.request_intervention()
    controller.confirm_intervention(True)
    controller.start_activity("task_split", 7, 60, {"blocker": "调试卡住", "next_step": "复现问题"})
    controller.complete_activity(4, "有帮助")
    exported = snapshot(controller, tmp_path)
    assert exported["session"]["mode"] == "simulation"
    assert {item["phase"] for item in exported["self_reports"]} == {"pre", "post"}
    activity = exported["interventions"][0]
    assert activity["delta"] == 3 and activity["evaluation"] == "improved"
    assert activity["task_details"]["next_step"] == "复现问题"
    assert exported["coverage"]["joint_seconds"] == 2
    assert any(item["kind"] == "intervention_confirmation" for item in exported["events"])


def test_missing_post_does_not_claim_improvement(controller, tmp_path):
    begin(controller)
    controller.request_intervention()
    controller.confirm_intervention(True)
    controller.start_activity("break", 7, 60)
    controller.complete_activity(None, "无法评价")
    result = snapshot(controller, tmp_path)["interventions"][0]
    assert result["delta"] is None and result["evaluation"] == "insufficient"
    assert result["model_comparison"] == "insufficient"


def test_pause_discards_pending_observations(controller, tmp_path):
    sid = begin(controller)
    controller.pause_session(True)
    controller._handle_event({"kind": "observation", "observation": simulation_observation(sid, "visual", 0, 1)})
    assert not snapshot(controller, tmp_path)["observations"]


def test_rejection_and_correction_suppress_auto_but_allow_user_request(controller):
    sid = begin(controller)
    controller.request_intervention()
    controller.confirm_intervention(False)
    for i in range(50):
        controller._handle_event({"kind": "observation", "observation": simulation_observation(sid, "visual", i * .5, (i + 1) * .5)})
    assert controller.flow.state == "cooldown"
    controller.request_intervention()
    assert controller.flow.state == "confirming"


def test_same_late_interval_does_not_create_sustained_prompt(controller):
    sid = begin(controller)
    for _ in range(60):
        controller._handle_event({"kind": "observation", "observation": simulation_observation(sid, "visual", 0, .5)})
    assert controller.flow.state == "observing"


def test_twenty_seconds_unique_evidence_requests_confirmation(controller):
    sid = begin(controller)
    for i in range(41):
        # Simulate arrival on the same clock as the observation, not in its future.
        controller.started = time.monotonic() - (i + 1) * .5
        controller._handle_event({"kind": "observation", "observation": simulation_observation(sid, "visual", i * .5, (i + 1) * .5)})
    assert controller.flow.state == "observing"  # Three-second settlement window.
    controller.started = time.monotonic() - 24
    controller._activity_tick()
    assert controller.flow.state == "confirming"


def test_late_conflicting_audio_resets_visual_streak(controller):
    sid = begin(controller)
    for i in range(30):
        controller._handle_event({"kind": "observation", "observation": simulation_observation(sid, "visual", i * .5, (i + 1) * .5)})
    audio = simulation_observation(sid, "audio", 10, 14, negative=0.05)
    audio["scores"] = {"positive": .8, "negative": .05, "neutral": .1, "uncertain": .05}
    controller._handle_event({"kind": "observation", "observation": audio})
    assert controller.negative_start is None
    assert controller.flow.state == "observing"


def test_previous_session_transcript_cannot_enter_current_session(controller, tmp_path):
    begin(controller)
    controller._handle_event({"kind": "utterance", "utterance": {"session_id": "old-session", "text": "旧记录"}})
    assert not snapshot(controller, tmp_path)["utterances"]


def test_zero_length_timeout_is_unavailable_not_crash(controller, tmp_path):
    sid = begin(controller)
    obs = simulation_observation(sid, "visual", 2, 2)
    obs.update(valid=False, scores=None, quality=0, reason="timeout")
    controller._handle_event({"kind": "observation", "observation": obs})
    assert controller.latest_summary["valid"] is False


def test_expired_audio_marks_gap_instead_of_carrying_old_scores(controller):
    sid = begin(controller)
    controller.mode = "real"  # Only exercise the heartbeat, never start devices.
    controller.enabled_modalities = {"audio"}
    controller.started = time.monotonic() - 20
    events = []
    controller.observation_ready.connect(events.append)
    controller._check_freshness()
    assert len(events) == 1 and events[0]["stale"]
    assert events[0]["scores"] is None and not events[0]["valid"]


def test_stale_pre_model_is_not_used_as_retest_baseline(controller, tmp_path):
    sid = begin(controller)
    controller.started = time.monotonic() - 30
    controller._handle_event({"kind": "observation", "observation": simulation_observation(sid, "visual", 0, .5)})
    controller.request_intervention()
    controller.confirm_intervention(True)
    controller.start_activity("break", 5, 60)
    controller.complete_activity(3, "有帮助")
    activity = snapshot(controller, tmp_path)["interventions"][0]
    assert activity["pre_model"] is None and activity["model_comparison"] == "insufficient"
