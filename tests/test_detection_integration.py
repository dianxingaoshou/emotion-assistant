"""Real Controller/Qt evidence wiring with an accelerated session clock.

No camera, microphone or model process is started. Observation scores are
explicit simulation fixtures, not measured emotion or intervention outcomes.
"""
from copy import deepcopy
import json
import os
import time
from types import SimpleNamespace

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import emotion_assistant.controller as controller_module
from emotion_assistant.controller import Controller, simulation_observation
from emotion_assistant.ui import MainWindow
from emotion_assistant.ui_widgets import TrendChart


class DetectionController(Controller):
    def refresh_devices(self):
        self.devices_ready.emit({"cameras": [], "microphones": [], "errors": []})


@pytest.fixture(scope="module")
def app():
    result = QApplication.instance() or QApplication([])
    result.setQuitOnLastWindowClosed(False)
    return result


@pytest.fixture
def rig(app, tmp_path, monkeypatch):
    clock = [1000.0]
    # Replace this module's clock reference, not the global time module used by
    # Qt polling, other test modules or database deadlines.
    monkeypatch.setattr(controller_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    controller = DetectionController(tmp_path / "detection-integration.sqlite3")
    controller._activity_timer.stop()
    controller._resource_timer.stop()
    controller._freshness_timer.stop()
    controller.start_session("检测接线模拟验收", -1, None, {"collection": True}, "simulation")
    controller._simulation_timer.stop()
    yield controller, clock
    controller.shutdown()
    app.processEvents()


def at(rig, seconds):
    controller, clock = rig
    clock[0] = controller.started+seconds


def wait_for(app, predicate, timeout=4):
    deadline = time.perf_counter()+timeout
    while not predicate():
        if time.perf_counter() >= deadline:
            raise AssertionError("Timed out waiting for Qt/database completion")
        QTest.qWait(5)
        app.processEvents()


def barrier(controller, app):
    controller._write(lambda: None).result(timeout=4)
    app.processEvents()


def send(controller, modality, start, end, event_id, negative=.7):
    record = simulation_observation(controller.session_id, modality, start, end, negative)
    record["event_id"] = event_id
    controller._handle_event({"kind": "observation", "observation": record})
    return record


def rows(controller, event_id):
    with controller.store._lock:
        return [json.loads(row[0]) for row in controller.store._db.execute(
            "SELECT payload_json FROM observations WHERE session_id=? AND event_id=?",
            (controller.session_id, event_id))]


def prepare_pre(rig):
    controller, _ = rig
    controller.request_intervention()
    controller.confirm_intervention(True)
    assert controller.assessment.current["phase"] == "pre"
    for second in range(30):
        at(rig, second+1)
        send(controller, "visual", second, second+1, f"pre-{second}")
    controller.start_activity("break", 7, 120)
    assert controller.pre_assessment["status"] == "complete"
    return controller.flow.current["id"]


def test_audio_waits_for_visual_watermark_and_publishes_joint(rig, app):
    controller, _ = rig
    controller.enabled_modalities = {"visual", "audio"}
    published = []
    controller.fusion_ready.connect(published.append)
    at(rig, 2)
    send(controller, "audio", 0, 2, "audio-watermark")
    assert not published
    assert not controller._audio_evidence["audio-watermark"]["published"]
    at(rig, 2.5)
    send(controller, "visual", 0, 2, "visual-watermark")
    audio = [item for item in published if item.get("audio_event_id") == "audio-watermark"]
    assert len(audio) == 1 and audio[0]["status"] == "joint"
    assert audio[0]["event_id"] == "fusion:audio-watermark" and audio[0]["revision"] == 1
    assert set(audio[0]["source_ids"]) == {"audio-watermark", "visual-watermark"}
    barrier(controller, app)
    assert rows(controller, "fusion:audio-watermark")[0]["status"] == "joint"


def test_deadline_then_late_visual_revises_one_audio_and_one_database_row(rig, app):
    controller, _ = rig
    controller.enabled_modalities = {"visual", "audio"}
    published = []
    controller.fusion_ready.connect(published.append)
    at(rig, 2)
    send(controller, "audio", 0, 2, "audio-late")
    at(rig, 4.99)
    controller._activity_tick()
    assert not published
    at(rig, 5)
    controller._activity_tick()
    first = [item for item in published if item.get("audio_event_id") == "audio-late"]
    assert len(first) == 1 and first[0]["status"] == "audio_only"
    at(rig, 6)
    send(controller, "visual", 0, 2, "late-visual")
    revisions = [item for item in published if item.get("audio_event_id") == "audio-late"]
    assert len(revisions) == 2
    assert revisions[-1]["status"] == "joint" and revisions[-1]["revised"] is True
    assert revisions[-1]["revision"] == 2
    assert revisions[0]["event_id"] == revisions[1]["event_id"]
    barrier(controller, app)
    stored = rows(controller, "fusion:audio-late")
    assert len(stored) == 1 and stored[0]["revision"] == 2
    assert set(stored[0]["source_ids"]) == {"audio-late", "late-visual"}


def test_audio_pending_bounded_64_and_eviction_is_explicit(rig, app):
    controller, _ = rig
    controller.enabled_modalities = {"visual", "audio"}
    at(rig, 10)
    for index in range(65):
        send(controller, "audio", index/10, (index+1)/10, f"queued-{index}")
    assert len(controller._audio_evidence) == 64
    assert "queued-0" not in controller._audio_evidence
    assert "queued-64" in controller._audio_evidence
    barrier(controller, app)
    assert len(rows(controller, "fusion:queued-0")) == 1
    with controller.store._lock:
        events = list(controller.store._db.execute(
            "SELECT payload_json FROM events WHERE session_id=? AND kind='fusion_buffer_eviction'",
            (controller.session_id,)))
    assert len(events) == 1 and json.loads(events[0][0])["audio_event_id"] == "queued-0"


def test_pause_clears_pending_and_old_results_cannot_publish(rig, app):
    controller, _ = rig
    controller.enabled_modalities = {"visual", "audio"}
    at(rig, 2)
    send(controller, "audio", 0, 2, "before-pause")
    assert controller._audio_evidence
    published = []
    controller.fusion_ready.connect(published.append)
    controller.pause_session(True)
    assert controller._audio_evidence == {}
    at(rig, 8)
    controller._activity_tick()
    send(controller, "visual", 0, 2, "paused-old-visual")
    assert published == []
    barrier(controller, app)
    assert rows(controller, "fusion:before-pause") == []


def test_pre_during_and_new_post_window_are_independent_and_persisted(rig, app):
    controller, _ = rig
    record_id = prepare_pre(rig)
    for second in range(30, 60):
        at(rig, second+1)
        send(controller, "visual", second, second+1, f"during-{second}", .8)
    assert controller.post_assessment is None
    assert controller.assessment is None
    controller.begin_post_assessment()
    assert controller.assessment.current["window_start"] == 60
    for second in range(60, 90):
        at(rig, second+1)
        send(controller, "visual", second, second+1, f"post-{second}", .3)
    controller.complete_activity(4, "helpful")
    record = controller.flow.current
    comparison = record["assessment_comparison"]
    assert comparison["comparable"] is True and comparison["reasons"] == []
    assert comparison["model_delta"]["negative"] == pytest.approx(.4)
    assert record["model_comparison"] == "independent_windows_observation_only"
    assert record["assessments"]["pre"]["intervention_id"] == record_id
    assert record["assessments"]["post"]["intervention_id"] == record_id
    assert all(source.startswith("pre-") for source in record["assessments"]["pre"]["source_ids"])
    assert all(source.startswith("post-") for source in record["assessments"]["post"]["source_ids"])
    barrier(controller, app)
    saved = controller.store.get_intervention(controller.session_id, record_id)
    assert saved["assessment_comparison"]["comparable"] is True
    assert saved["assessments"]["post"]["coverage"] == 1


@pytest.mark.parametrize("post_case", ["not_started", "skipped", "cross_boundary"])
def test_no_independent_complete_post_window_never_reports_model_comparison(rig, app, post_case):
    controller, _ = rig
    record_id = prepare_pre(rig)
    at(rig, 40)
    # This strong activity-period result must not become a post measurement.
    send(controller, "visual", 30, 40, "during-only", .1)
    if post_case != "not_started":
        controller.begin_post_assessment()
        if post_case == "skipped":
            controller.skip_assessment()
        else:
            at(rig, 45)
            send(controller, "visual", 35, 45, "crossing-activity-post", .1)
            at(rig, 70)
    controller.complete_activity(3, "helpful")
    record = controller.flow.current
    assert record["assessment_comparison"]["comparable"] is False
    assert record["assessment_comparison"]["model_delta"] is None
    assert record["model_comparison"] == "insufficient"
    post = record["assessments"]["post"]
    if post_case == "not_started":
        assert post is None
    elif post_case == "skipped":
        assert post["status"] == "skipped"
    else:
        assert post["status"] == "insufficient"
        assert "crosses_window_boundary" in post["exclusion_reasons"]
        assert post["source_ids"] == []
    barrier(controller, app)
    assert controller.store.get_intervention(controller.session_id, record_id)["assessment_comparison"]["comparable"] is False


def test_chart_revisions_replace_same_event_without_duplicating_or_moving_time(app):
    chart = TrendChart()
    original = {"event_id": "fusion:a", "t_start": 0, "t_end": 2, "status": "audio_only",
                "scores": {"negative": .8, "neutral": .1, "positive": .05, "uncertain": .05}, "valid": True}
    chart.add_observation(original)
    revised = deepcopy(original)
    revised.update(status="joint", scores={"negative": .3, "neutral": .6, "positive": .05, "uncertain": .05})
    chart.add_observation(revised)
    assert len(chart._samples) == 1 and len(chart.points) == 1
    assert chart.points[0] == (2, 30, "joint")
    assert chart._event_samples["fusion:a"][0] == 2


def test_late_revision_updates_chart_but_never_overwrites_current_ui(rig, app):
    controller, _ = rig
    window = MainWindow(controller)
    window.user_name.setText(controller.current_user_name)
    window.mode_combo.setCurrentIndex(window.mode_combo.findData("simulation"))
    window.show()
    try:
        current = simulation_observation(controller.session_id, "visual_summary", 29, 30, .7)
        current.update(event_id="current", status="visual_only", weights={"visual": 1, "audio": 0})
        old = simulation_observation(controller.session_id, "fusion", 0, 2, .2)
        old.update(event_id="fusion:old", status="joint", weights={"visual": .5, "audio": .5})
        window._on_fusion(current)
        before = (window.fusion_label.text(), window.fusion_mode.text(), window.fusion_interval.text())
        window._on_fusion(old)
        after = (window.fusion_label.text(), window.fusion_mode.text(), window.fusion_interval.text())
        assert before == after and window._fusion_display_end == 30
        assert any(item[0] == 2 for item in window.chart.points)
        assert controller.latest_summary is None  # UI display alone never alters controller evidence.
    finally:
        window.close()


def test_joint_conclusion_not_downgraded_by_visual_summary_at_same_end(rig, app):
    controller, _ = rig
    controller.enabled_modalities = {"visual", "audio"}
    window = MainWindow(controller)
    window.user_name.setText(controller.current_user_name)
    window.mode_combo.setCurrentIndex(window.mode_combo.findData("simulation"))
    window.show()
    try:
        app.processEvents()
        at(rig, 2)
        send(controller, "audio", 0, 2, "same-end-audio")
        send(controller, "visual", 0, 2, "same-end-visual")
        # The visual arrival first resolves the pending joint result and then
        # emits its own visual summary. Equal event time must prefer the joint
        # conclusion, while both series remain available in the chart.
        assert controller.latest_summary["status"] == "joint"
        assert window.fusion_mode.text() == "音视频联合观察"
        assert {item[2] for item in window.chart.points} == {"visual", "joint"}
    finally:
        window.close()
