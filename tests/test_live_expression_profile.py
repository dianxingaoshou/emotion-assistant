"""Exercise the actual ASR adapter, controller, Qt signal, and SQLite export boundaries."""
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from emotion_assistant import controller as controller_module
from emotion_assistant.models import ModelHub
from emotion_assistant.ui import MainWindow


@pytest.fixture
def desktop(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(controller_module, "DATA", tmp_path)
    controller = controller_module.Controller(tmp_path / "session.sqlite3")
    monkeypatch.setattr(controller, "refresh_devices", lambda: None)
    window = MainWindow(controller)
    try:
        yield app, controller, window, tmp_path
    finally:
        window.close()
        app.processEvents()


def recorded_utterance(folder, session_id, tagged_text="<|en|>I feel calm and relaxed today."):
    class Recognizer:
        def generate(self, **kwargs):
            return [{"text": tagged_text}]

    hub = ModelHub(folder)
    hub._asr_model = Recognizer()
    return hub.transcribe_audio(np.full(32000, 0.03), session_id, 0, 2, asr_allowed=True)


def exported_rows(app, controller, path):
    controller.writer.submit(lambda: None).result(timeout=5)
    app.processEvents()
    exported = controller.store.export_session(controller.session_id, path)
    return json.loads(exported.read_text(encoding="utf-8"))["utterances"]


def test_authorized_asr_quantification_reaches_live_ui_and_export_then_revokes(desktop):
    app, controller, window, folder = desktop
    window.transcript_consent.setChecked(True)
    controller.start_session("现场文字验收", -1, None, {"collection": True, "transcript": True}, "simulation")
    row = recorded_utterance(folder, controller.session_id)
    assert row["valid"]
    profile = row["semantic_context"]["expression_profile"]
    assert profile["affect_dimensions"]["status"] == "ok"
    controller._handle_event({"kind": "utterance", "utterance": row})
    saved = exported_rows(app, controller, folder / "before.json")
    assert saved[0]["semantic_context"]["expression_profile"] == profile
    assert "激活度" in window.transcript.toPlainText() and "掌控感" in window.transcript.toPlainText()

    window._revoke_transcript()
    controller._handle_event({"kind": "utterance", "utterance": dict(row, event_id="late")})
    saved = exported_rows(app, controller, folder / "after.json")
    assert len(saved) == 1 and saved[0]["text"] == row["text"]
    assert "semantic_context" not in saved[0] and window.transcript.toPlainText() == ""


def test_wrong_session_stale_and_unconsented_events_never_publish_quantification(desktop):
    app, controller, window, folder = desktop
    window.transcript_consent.setChecked(True)
    controller.start_session("拒绝过期文字", -1, None, {"collection": True, "transcript": True}, "simulation")
    row = recorded_utterance(folder, controller.session_id)
    for blocked in (dict(row, session_id="other-session"), dict(row, stale=True)):
        controller._handle_event({"kind": "utterance", "utterance": blocked})
    app.processEvents()
    assert not window.transcript.toPlainText()
    window._revoke_transcript()
    controller._handle_event({"kind": "utterance", "utterance": row})
    assert not window.transcript.toPlainText()
    assert not exported_rows(app, controller, folder / "blocked.json")


def test_chinese_asr_shows_two_offline_dimensions_before_optional_review(desktop):
    app, controller, window, folder = desktop
    window.transcript_consent.setChecked(True)
    controller.start_session("中文实时输入", -1, None, {"collection": True, "transcript": True}, "simulation")
    row = recorded_utterance(folder, controller.session_id, "<|zh|>谢谢你")
    controller._handle_event({"kind": "utterance", "utterance": row})
    exported = exported_rows(app, controller, folder / "zh.json")[0]
    profile = exported["semantic_context"]["expression_profile"]
    assert profile["language"] == "zh"
    assert profile["fine_emotions"]["labels"][0]["id"] == "grateful"
    assert set(profile["affect_dimensions"]["dimensions"]) == {"valence", "arousal"}
    assert "感激" in window.transcript.toPlainText()
    assert "激活度" in window.transcript.toPlainText()
    assert "semantic_review" not in exported
