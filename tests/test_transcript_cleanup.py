import json
import os
import sqlite3
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
import pytest

from emotion_assistant.controller import Controller
from emotion_assistant.storage import Store
from emotion_assistant.ui import MainWindow


def utterance(key):
    return {"event_id": key, "text": "已授权的原话", "segment_id": "same-source",
            "semantic_context": {"subject": "self", "need_hint": "stuck", "evidence_text": "已授权的原话"}}


def exported(store, sid, directory):
    return json.loads(store.export_session(sid, directory / "history.json").read_text(encoding="utf-8"))


def test_revoke_journal_survives_failed_cleanup_and_recovers_on_reopen(tmp_path):
    path = tmp_path / "history.sqlite3"
    store = Store(path)
    sid = store.start_session(store.create_user("fixture"), {"transcript": True, "save_transcript": True}, "real")
    try:
        store.save_utterance(sid, utterance("old"), True)
        assert exported(store, sid, tmp_path)["utterances"][0]["semantic_context"]
        store.request_transcript_cleanup(sid)
        store._db.executescript("""
            CREATE TRIGGER fail_cleanup BEFORE INSERT ON events
            WHEN NEW.kind='transcript_consent_changed' BEGIN SELECT RAISE(ABORT,'failure'); END;
        """)
        with pytest.raises(sqlite3.DatabaseError):
            store.set_transcript_consent(sid, False)
        assert store.save_utterance(sid, utterance("late"), True) is None
        assert store.pending_transcript_cleanup() == [sid]
    finally:
        store.close()
    other = Store(path)
    try:
        data = exported(other, sid, tmp_path)
        assert data["session"]["consent"]["semantic_cleanup_pending"] and not data["session"]["consent"]["transcript"]
        assert [u["event_id"] for u in data["utterances"]] == ["old"]
        assert data["utterances"][0]["text"] == "已授权的原话" and "semantic_context" not in data["utterances"][0]
        assert other.save_utterance(sid, utterance("late"), True) is None
        other._db.executescript("DROP TRIGGER fail_cleanup")
    finally:
        other.close()
    recovered = Store(path)
    try:
        assert recovered.pending_transcript_cleanup() == []
        saved = exported(recovered, sid, tmp_path)
        assert saved["session"]["consent"]["transcript"] is False
        assert len([e for e in saved["events"] if e["kind"] == "transcript_consent_changed"]) == 1
        assert "semantic_context" not in saved["utterances"][0] and saved["utterances"][0]["text"] == "已授权的原话"
    finally:
        recovered.close()


def test_gui_failure_and_retry_keep_queued_asr_out_of_export(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    controller = Controller(tmp_path / "session.sqlite3")
    monkeypatch.setattr(controller, "refresh_devices", lambda: None)
    window = MainWindow(controller)
    gate = threading.Event()
    try:
        window.mode_combo.setCurrentIndex(window.mode_combo.findData("simulation"))
        window.consent.setChecked(True)
        window.transcript_consent.setChecked(True)
        window.start_button.click()
        controller._simulation_timer.stop()
        sid = controller.session_id
        controller.store.save_utterance(sid, utterance("before"), True)
        controller.writer.submit(gate.wait, 3)
        queued = controller._write(controller.store.save_utterance, sid, utterance("late"), True)
        original = controller.store.set_transcript_consent
        def fail_once(*args):
            raise sqlite3.OperationalError("test cleanup failure")
        monkeypatch.setattr(controller.store, "set_transcript_consent", fail_once)
        window.revoke_transcript_button.click()
        pending = exported(controller.store, sid, tmp_path)
        assert "semantic_context" not in pending["utterances"][0]
        assert not controller.transcript_allowed
        gate.set()
        assert queued.result(timeout=2) is None
        controller.writer.submit(lambda: None).result(timeout=2)
        app.processEvents()
        assert not window.retry_transcript_button.isHidden() and "尚待清理" in window.transcript_notice.text()
        monkeypatch.setattr(controller.store, "set_transcript_consent", original)
        window.retry_transcript_button.click()
        controller.writer.submit(lambda: None).result(timeout=2)
        app.processEvents()
        assert window.retry_transcript_button.isHidden() and "已清除" in window.transcript_notice.text()
        saved = exported(controller.store, sid, tmp_path)
        assert [r["event_id"] for r in saved["utterances"]] == ["before"]
        assert saved["utterances"][0]["text"] == "已授权的原话"
    finally:
        gate.set()
        window.close()
        app.processEvents()
