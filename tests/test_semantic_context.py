"""Fixed Chinese utterance labels, independent of the acoustic emotion scores."""
import json
import pytest

from emotion_assistant.semantic_context import interpret_context
from emotion_assistant.storage import Store


@pytest.mark.parametrize("utterance,subject,scope,quote,hint", [
    ("我现在卡住了，这个问题一直解决不了", "self", "current", False, "stuck"),
    ("我现在好累，想休息一下", "self", "current", False, "fatigue"),
    ("我有点紧张", "self", "current", False, "tense"),
    ("我现在反复想同一件事", "self", "current", False, "rumination"),
    ("我昨天很难受", "self", "past", False, None),
    ("他说他现在很难受", "other", "current", True, None),
    ("我在朗读一句话：我现在很紧张", "self", "current", True, None),
    ("要是明天我卡住了呢", "self", "hypothetical", False, None),
    ("我现在并不紧张", "self", "current", False, None),
    ("我现在不需要放松", "self", "current", False, None),
    ("这个问题怎么解决", "unknown", "unknown", False, None),
])
def test_annotated_utterances(utterance, subject, scope, quote, hint):
    result = interpret_context(utterance)
    assert (result["subject"], result["time_scope"], result["is_quote"], result["need_hint"]) == (subject, scope, quote, hint)
    assert "scores" not in result and result["requires_confirmation"]


def test_permission_removes_derivatives_but_retains_authorized_transcript(tmp_path):
    store = Store(":memory:")
    try:
        sid = store.start_session(store.create_user("fixture"), {"transcript": True}, "simulation")
        utterance = {"event_id": "u", "text": "我想休息", "segment_id": "a",
                     "semantic_context": interpret_context("我想休息")}
        store.save_utterance(sid, utterance, True)
        store.set_transcript_consent(sid, False)
        assert store.save_utterance(sid, dict(utterance, event_id="late"), True) is None
        saved = json.loads(store.export_session(sid, tmp_path / "export.json").read_text(encoding="utf-8"))
        assert len(saved["utterances"]) == 1
        assert saved["utterances"][0]["text"] == "我想休息" and "semantic_context" not in saved["utterances"][0]
    finally:
        store.close()


def test_desktop_strategy_selector_and_opt_in_need_hint(tmp_path, monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from emotion_assistant import controller as module
    from emotion_assistant.ui import MainWindow
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(module, "DATA", tmp_path)
    controller = module.Controller(tmp_path / "test.sqlite3")
    monkeypatch.setattr(controller, "refresh_devices", lambda: None)
    window = MainWindow(controller)
    try:
        window.mode_combo.setCurrentIndex(window.mode_combo.findData("simulation"))
        window.fusion_strategy.setCurrentIndex(window.fusion_strategy.findData("calibrated_fusion"))
        window.consent.setChecked(True)
        window.transcript_consent.setChecked(True)
        window.start_button.click()
        controller._simulation_timer.stop()
        assert not window.fusion_strategy.isEnabled()
        controller._simulate_tick()
        assert controller.latest_summary["calibration_status"] == "unfitted"
        assert "未拟合" in window.fusion_weights.text()
        window.prompt_availability.setCurrentIndex(window.prompt_availability.findData("quiet"))
        assert controller.prompt_timing.mode == "quiet"
        text = "我现在很紧张"
        current = window.need_combo.currentData()
        payload = {"session_id": controller.session_id, "t_end": .1, "text": text,
                   "semantic_context": interpret_context(text)}
        controller._handle_event({"kind": "utterance", "utterance": payload})
        assert not window.context_button.isHidden() and window.need_combo.currentData() == current
        old_text = "我现在想休息"
        controller._handle_event({"kind": "utterance", "utterance": dict(payload, t_end=0., text=old_text,
                                          semantic_context=interpret_context(old_text))})
        window.context_button.click()
        assert window.need_combo.currentData() == "tense"
        controller._handle_event({"kind": "utterance", "utterance": payload})
        window.revoke_transcript_button.click()
        assert window.context_button.isHidden()
        controller._handle_event({"kind": "utterance", "utterance": payload})
        assert window.context_button.isHidden()
    finally:
        window.close()
        app.processEvents()
