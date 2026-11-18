"""Synthetic closed-session references. No real person, devices or model labels."""
import json
import os
import threading
import time

import pytest

from emotion_assistant.personalization import FeedbackRecord, compare_reference, comparison_key
from emotion_assistant.storage import Store


def observation(sid, i, negative=.25):
    return {"session_id": sid, "event_id": f"fusion-{i}", "source_ids": [f"visual-{i}"],
            "modality": "visual_summary", "valid": True, "status": "visual_only",
            "t_start": i, "t_end": i+1, "quality": .9, "scores": {"negative": negative},
            "model_versions": {"visual": ["v1"]}, "modalities": ["visual"],
            "modality_intervals": {"visual": [[i, i+1]]}, "fusion_config_hash": "with-baseline",
            "fusion_basis_hash": "independent-of-baseline", "reference_observing": True,
            "reference_context": {"enabled_modalities": ["visual"], "camera": ["cam-a"]}}


def add(store, uid, sid, i, *, allowed=True, kind="typical_reference", label="usual",
        fresh=True, save_raw=True, event=None):
    event = event or observation(sid, i)
    if save_raw:
        raw = dict(event, event_id=event["source_ids"][0], modality="visual")
        store.save_observation(sid, raw)
    store.save_observation(sid, event)
    record = FeedbackRecord.create(uid, sid, kind, label, allowed, event, i+(1 if fresh else 30))
    return store.save_observation_feedback(record)


@pytest.fixture
def db():
    store = Store(":memory:")
    yield store
    store.close()


def test_reference_uses_only_closed_prior_sessions_and_checks_device_model_and_basis(db):
    uid = db.create_user("本人")
    consent = {"personal_learning": True}
    a = db.start_session(uid, consent, "real")
    for i in (0, 2, 4):
        add(db, uid, a, i)
    assert not db.personal_reference(uid, "real")["version"]
    db.finish_session(a)
    frozen = db.personal_reference(uid, "real")
    b = db.start_session(uid, consent, "real")
    for i in (0, 2):
        add(db, uid, b, i)
    assert db.personal_reference(uid, "real") == frozen
    db.finish_session(b)
    updated = db.personal_reference(uid, "real")
    assert frozen["version"] != updated["version"]
    future = observation("future", 10, negative=.8)
    future["fusion_config_hash"] = "hash-changes-with-new-frozen-baseline"
    assert compare_reference(frozen, future)["status"] == "insufficient"
    assert compare_reference(updated, future)["status"] == "above_reference"
    group, = updated["groups"].values()
    assert (group["count"], group["sessions"], group["median"], group["mad"]) == (5, 2, .25, 0)
    for key, value in (("fusion_basis_hash", "new-calibration"), ("model_versions", {"visual": ["v2"]}),
                       ("reference_context", {"enabled_modalities": ["visual"], "camera": ["cam-b"]})):
        assert compare_reference(updated, dict(future, **{key: value}))["status"] == "insufficient"
    assert not db.personal_reference(uid, "simulation")["groups"]
    assert not db.personal_reference(db.create_user("另一档案"), "real")["groups"]
    db.delete_session(a)
    assert not next(iter(db.personal_reference(uid, "real")["groups"].values()))["ready"]


@pytest.mark.parametrize("overrides", [
    {"conflict": True}, {"quality": .4}, {"valid": False}, {"source_ids": []},
    {"reference_context": {"enabled_modalities": ["visual", "audio"]}}, {"decision": "abstain"},
    {"reference_observing": False}, {"fusion_basis_hash": None}, {"model_versions": {"visual": []}},
    {"modality_intervals": {"visual": [[0,.2]]}}, {"scores": {"negative": float("nan")}},
])
def test_unstable_or_incomplete_inputs_do_not_learn(overrides):
    assert comparison_key(dict(observation("s", 0), **overrides)) is None


def test_consent_at_write_time_target_revision_dedup_and_json_export(db, tmp_path):
    uid = db.create_user("反馈")
    sid = db.start_session(uid, {"personal_learning": True}, "real")
    add(db, uid, sid, 0)
    add(db, uid, sid, 0)
    add(db, uid, sid, 2, fresh=False)
    add(db, uid, sid, 4, save_raw=False)
    add(db, uid, sid, 6, allowed=False)
    add(db, uid, sid, 8)
    db.save_observation(sid, dict(observation(sid, 8), revision=2, conflict=True))
    add(db, uid, sid, 10, kind="subjective_correction", label="negative")
    add(db, uid, sid, 12, kind="distress_report", label="distressed")
    add(db, uid, sid, 14, kind="do_not_disturb", label="busy")
    add(db, uid, sid, 16)
    add(db, uid, sid, 16, kind="subjective_correction", label="neutral")
    db.pause_personal_learning(sid)
    assert not add(db, uid, sid, 18)["learning_consent"]
    db.finish_session(sid)
    groups = db.personal_reference(uid, "real")["groups"]
    assert len(groups) == 1 and next(iter(groups.values()))["count"] == 1
    result = json.loads(db.export_session(sid, tmp_path / "feedback.json").read_text("utf-8"))
    record = result["observation_feedback"][0]
    assert record["window_start"] == 0 and record["window_end"] == 1
    assert record["target_event_ids"] == ["fusion-0", "visual-0"] and record["user_id"] == uid
    assert record["model_versions"] == {"visual": ["v1"]} and record["fusion_config_hash"] == "with-baseline"


def test_reset_during_session_blocks_inflight_learning_and_retains_history(db, tmp_path):
    uid = db.create_user("重置")
    sid = db.start_session(uid, {"personal_learning": True}, "real")
    add(db, uid, sid, 0)
    db.reset_personal_reference(uid, "real")
    assert not add(db, uid, sid, 2)["learning_consent"]
    db.finish_session(sid)
    assert db.personal_reference(uid, "real") == {"version": None, "groups": {}}
    assert len(json.loads(db.export_session(sid, tmp_path/"original.json").read_text("utf-8"))["observation_feedback"]) == 2


def test_cannot_attach_foreign_event_or_write_under_another_profile(db):
    user, other = db.create_user("甲"), db.create_user("乙")
    sid = db.start_session(user, {"personal_learning": True}, "simulation")
    rec = FeedbackRecord.create(user, sid, "subjective_correction", "neutral", False,
                                observation("other-session", 0), 1)
    assert not rec.target_event_ids and rec.window_start is None
    wrong_profile = FeedbackRecord.create(other, sid, "subjective_correction", "neutral", False, None, 1)
    with pytest.raises(ValueError):
        db.save_observation_feedback(wrong_profile)
    with pytest.raises(ValueError):
        FeedbackRecord.create(user, sid, "do_not_disturb", "negative", True, None, 1)


def test_window_feedback_targets_displayed_interval_across_live_updates(monkeypatch, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QInputDialog
    from emotion_assistant.controller import Controller, simulation_observation
    from emotion_assistant.ui import MainWindow
    app = QApplication.instance() or QApplication([])
    ctl = Controller(tmp_path/"ui.sqlite3")
    monkeypatch.setattr(ctl, "refresh_devices", lambda: None)
    window = MainWindow(ctl)
    try:
        window.mode_combo.setCurrentIndex(window.mode_combo.findData("simulation"))
        window.user_name.setText("窗口测试")
        window.consent.setChecked(True)
        window.learning_consent.setChecked(True)
        window.start_button.click()
        ctl._simulation_timer.stop()
        sid = ctl.session_id
        assert sid and ctl.personal_learning
        ctl.started = time.monotonic()-4
        def publish(start, end):
            for modality in ("visual", "audio"):
                obs = simulation_observation(sid, modality, start, end, negative=.6)
                ctl._handle_event({"kind":"observation", "observation":obs})
        publish(1, 2)
        displayed = ctl.latest_summary
        def while_choosing(*args):
            publish(2, 3)
            return "这是我的日常状态，用作个人参考", True
        monkeypatch.setattr(QInputDialog, "getItem", while_choosing)
        window.correct_button.click()
        ctl._write(lambda: None).result(timeout=5)
        file = ctl.store.export_session(sid, tmp_path/"ui-feedback.json")
        feedback, = json.loads(file.read_text("utf-8"))["observation_feedback"]
        assert feedback["target_event_ids"][0] == displayed["event_id"] != ctl.latest_summary["event_id"]
        assert feedback["reference_candidate_at_feedback"] and feedback["fusion_config_hash"] == ctl.fusion.config_hash
        assert not ctl.frozen_reference["version"]
        ctl.pause_personal_learning()
        assert not ctl.personal_learning
        ctl.store.finish_session(sid)
        snapshot = ctl.store.personal_reference(ctl.current_user_id, "simulation")
        assert next(iter(snapshot["groups"].values()))["count"] == 1
    finally:
        window.close()
        ctl.shutdown()
        app.processEvents()


def test_forward_audit_uses_only_prior_closed_exports_and_independent_labels(db, tmp_path):
    from scripts.evaluate_personal_reference import evaluate
    uid = db.create_user("时序比较")
    exports = []
    for start, end, scores in [(0, 1, [.25]*3), (2, 3, [.25]*2), (4, 7, [.95]*8), (5, 6, [.55])]:
        sid = db.start_session(uid, {"personal_learning": True}, "real")
        for i, score in enumerate(scores):
            event = observation(sid, i, score)
            event["personal_reference"] = {"status":"insufficient"}
            add(db, uid, sid, i, event=event)
        db.append_event(sid, "prompt_decision", {"decision_id": "d", "action": "ask"})
        db.finish_session(sid)
        doc = json.loads(db.export_session(sid, tmp_path/f"{start}.json").read_text("utf-8"))
        doc["session"].update(started_at=f"2026-01-01T{start:02d}:00:00+00:00",
                              ended_at=f"2026-01-01T{end:02d}:00:00+00:00")
        exports.append(doc)
    labels = [{"session_id": sid, "event_id": "fusion-0", "above_usual": True},
              {"session_id": sid, "decision_id": "d", "prompt_wanted": False}]
    result = evaluate(reversed(exports), labels)["sessions"]
    assert result[0]["prior_reference_version"] is None and result[0]["unwanted_rate_among_labeled_prompts"] is None
    assert result[-1]["coverage"] == 1 and result[-1]["independently_labeled_events"] == 1
    assert result[-1]["personal_reference_error_rate"] == 0 and result[-1]["fixed_065_error_rate_on_same_events"] == 1
    assert result[-1]["unwanted_rate_among_labeled_prompts"] == 1 and result[-1]["feedback_actions_per_session_hour"] == 1


@pytest.mark.parametrize("command,expected_status", [("pause", "paused"), ("reset", "reset")])
def test_active_session_disables_reference_and_blocks_queued_feedback_without_changing_fusion(command, expected_status, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from emotion_assistant.controller import Controller, simulation_observation
    app = QApplication.instance() or QApplication([])
    ctl = Controller(tmp_path/"disable.sqlite3")
    release = threading.Event()
    def start():
        ctl.start_session("学习权限", -1, None, {"collection": True, "personal_learning": True}, "simulation")
        ctl._simulation_timer.stop()
    def publish(i, negative):
        ctl.started = time.monotonic() - (i+1.1)
        for m in ("visual", "audio"):
            item = simulation_observation(ctl.session_id, m, i, i+1, negative=negative)
            ctl._handle_event({"kind":"observation", "observation":item})
    try:
        for count in (3, 2):
            start()
            for i in range(count):
                publish(i, .25)
                ctl.correct_observation("usual", "typical_reference").result(timeout=5)
            ctl.store.finish_session(ctl.session_id)
            ctl._finish_stop(ctl.session_id)
        start()
        version, fingerprint = ctl.frozen_reference["version"], ctl.fusion.config_hash
        publish(0, .75)
        assert version and ctl.latest_summary["personal_reference"]["status"] == "above_reference"
        ctl._write(release.wait, 5)
        queued = ctl.correct_observation("usual", "typical_reference")
        if command == "reset":
            ctl.reset_personal_reference("学习权限", "simulation")
        else:
            ctl.pause_personal_learning()
        release.set()
        assert not queued.result(timeout=5)["learning_consent"]
        publish(2, .75)
        assert ctl.latest_summary["personal_reference"] == {"status":expected_status, "version":None,
            "session_frozen_version":version, "applied":False}
        assert ctl.latest_summary["fusion_config_hash"] == fingerprint
        ctl.store.finish_session(ctl.session_id)
        persisted = ctl.store.personal_reference(ctl.current_user_id, "simulation")
        assert persisted["version"] == (version if command == "pause" else None)
    finally:
        release.set()
        ctl.shutdown()
        app.processEvents()
