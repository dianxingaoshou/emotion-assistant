import json
import os
import time
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from emotion_assistant import ai_activity, ai_activity_widget
from emotion_assistant.controller import Controller
from emotion_assistant.ui import MainWindow


def context():
    items = [{"activity": "break", "blocked": False, "notes": "PRIVATE"},
             {"activity": "desk_reset", "blocked": False},
             {"activity": "grounding", "blocked": False, "discomfort_count": 1},
             {"activity": "breathing", "blocked": False},
             {"activity": "task_split", "blocked": True}]
    return ai_activity.activity_context("fatigue", 120, 6, items,
                                        {"break": "prefer", "breathing": "avoid"})


@pytest.mark.parametrize("value, valid", [
    ({"activities": ["break", "desk_reset"]}, True), ({"activities": ["desk_reset"]}, True),
    ({"activities": []}, False), ({"activities": ["breathing"]}, False),
    ({"activities": ["break", "break"]}, False), ({"activities": ["break"] * 3}, False),
    ({"activities": ["break"], "advice": "untrusted"}, False),
])
def test_bounded_catalog_accepts_only_distinct_allowed_ids(monkeypatch, value, valid):
    response = Mock(status=200)
    response.read.return_value = json.dumps({"done": True, "message": {"content": json.dumps(value)}}).encode()
    connection = Mock()
    connection.getresponse.return_value = response
    factory = Mock(return_value=connection)
    monkeypatch.setattr(ai_activity, "HTTPConnection", factory)
    ctx = context()
    assert ctx.eligible == ("break", "desk_reset")
    result = ai_activity.recommend_with_ai(ctx)
    factory.assert_called_once_with("127.0.0.1", 11434, timeout=ai_activity.TIMEOUT_SECONDS)
    _, endpoint, body, _ = connection.request.call_args.args
    assert endpoint == "/api/chat"
    request = json.loads(body)
    data = json.loads(request["messages"][1]["content"])
    assert "PRIVATE" not in str(data)
    assert set(data) == {"confirmed_goal", "available_seconds", "self_rated_distress_0_to_10", "candidates"}
    assert [item["id"] for item in data["candidates"]] == list(ctx.eligible)
    assert data["candidates"][0]["preferred"] and "偏好" in ai_activity.recommendation_reason("break", ctx)
    assert result["status"] == ("ok" if valid else "invalid_response")
    assert result.get("activities") == (value["activities"] if valid else None)
    connection.close.assert_called_once()


def test_no_candidates_or_missing_self_rating_prevents_offers(monkeypatch):
    item = {"activity": "break", "blocked": False}
    assert ai_activity.activity_context("fatigue", 15, 4, [item], {}) is None
    assert ai_activity.activity_context("fatigue", 60, None, [item], {}) is None
    assert ai_activity.activity_context("fatigue", 60, 0, [item], {}) is not None
    connection = Mock()
    connection.request.side_effect = ConnectionRefusedError()
    monkeypatch.setattr(ai_activity, "HTTPConnection", Mock(return_value=connection))
    assert ai_activity.recommend_with_ai(context()) == {"status": "unavailable"}


class PendingOffers(QObject):
    finished = Signal(int, object)

    def __init__(self, parent, *, callback):
        super().__init__(parent)
        self.requests = []

    def clear(self):
        pass  # A deliberately late reply exercises the widget's own scope check.

    def submit(self, ctx):
        self.requests.append(ctx)
        return len(self.requests)


def test_old_session_response_and_selection_do_not_cross_sessions(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(ai_activity_widget, "SemanticReviewQueue", PendingOffers)
    panel = ai_activity_widget.AiActivitySuggestions()
    ctx = context()
    try:
        panel.set_context("first session", ctx)
        panel.ask.click()
        panel.set_context("second session", ctx)
        panel.queue.finished.emit(1, {"status": "ok", "activities": ["break"]})
        assert panel.options[0][0].isHidden() and panel.selection_record("break") is None
        panel.ask.click()
        panel.queue.finished.emit(2, {"status": "ok", "activities": ["break"]})
        panel.options[0][0].click()
        assert panel.selection_record("break")["adopted_id"] == "break"
        panel.set_context(None, None)
        assert panel.selection_record("break") is None
    finally:
        panel.close()
        app.processEvents()


@pytest.mark.parametrize("replace_ai_selection", [True, False])
def test_ai_offers_require_confirmation_self_report_and_explicit_selection(tmp_path, monkeypatch, replace_ai_selection):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(ai_activity_widget, "SemanticReviewQueue", PendingOffers)
    controller = Controller(tmp_path / "activity.sqlite3")
    monkeypatch.setattr(controller, "refresh_devices", lambda: None)
    window = MainWindow(controller)
    panel = window.ai_activity
    window.show()
    try:
        window.user_name.setText("UI-test-profile")
        window.mode_combo.setCurrentIndex(window.mode_combo.findData("simulation"))
        window.consent.setChecked(True)
        window.start_button.click()
        controller._simulation_timer.stop()
        window.activity_request_button.click()
        assert controller.flow.state == "confirming" and panel.isHidden()
        window.accept_button.click()
        assert controller.flow.state == "ready" and panel.isHidden()
        window.pre_combo.setCurrentIndex(window.pre_combo.findData(4))
        deadline = time.monotonic() + 5
        while panel.isHidden() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.005)
        assert panel.isVisible() and not panel.queue.requests
        # Both crossover arms use their assigned protocol.
        for condition in ("basic", "adaptive"):
            window._active_trial = {"condition": condition}
            window._sync_ai_activity()
            assert panel.isHidden()
        window._active_trial = None
        window._sync_ai_activity()
        panel.ask.click()
        ctx = panel.queue.requests[-1]
        assert ctx.distress == 4
        initial = window.activity_combo.currentData()
        key = next(k for k in ctx.eligible if k != initial)
        panel.queue.finished.emit(1, {"status": "ok", "activities": [key]})
        assert window.activity_combo.currentData() == initial and panel.options[0][0].isVisible()
        panel.dismiss.click()
        assert not panel.options[0][0].isVisible() and window.activity_combo.currentData() == initial
        panel.ask.click()
        panel.queue.finished.emit(2, {"status": "unavailable"})
        assert "手动" in panel.status.text() and window.begin_activity_button.isEnabled()
        panel.ask.click()
        panel.queue.finished.emit(3, {"status": "ok", "activities": [key]})
        panel.options[0][0].click()
        assert window.activity_combo.currentData() == key and controller.flow.state == "ready"
        panel.ask.click()
        window.duration_combo.setCurrentIndex(0)
        panel.queue.finished.emit(4, {"status": "ok", "activities": [key]})
        assert panel.options[0][0].isHidden()
        panel.ask.click()
        panel.queue.finished.emit(5, {"status": "ok", "activities": [key]})
        panel.options[0][0].click()
        final_id = initial if replace_ai_selection else key
        if replace_ai_selection:
            window.activity_combo.setCurrentIndex(window.activity_combo.findData(final_id))
        window.begin_activity_button.click()
        assert controller.flow.state == "running" and controller.flow.current["activity"] == final_id
        assert panel.isHidden()
        controller.writer.submit(lambda: None).result(timeout=5)
        export = controller.store.export_session(controller.session_id, tmp_path / "export.json")
        record = json.loads(export.read_text(encoding="utf-8"))["interventions"][0]["context"]["ai_activity_offer"]
        assert record == {"model": "qwen3:8b", "eligible_ids": list(ctx.eligible),
                          "suggested_ids": [key], "selected_id": final_id,
                          "selection_source": "manual" if replace_ai_selection else "ai_suggestion",
                          "adopted_id": None if replace_ai_selection else key}
    finally:
        window.close()
        app.processEvents()
