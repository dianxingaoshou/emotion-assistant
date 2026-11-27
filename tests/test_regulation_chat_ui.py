import json
import os
from threading import Event
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from emotion_assistant.controller import Controller
from emotion_assistant.regulation import ACTIVITIES
from emotion_assistant.regulation_chat import ChatTurn
from emotion_assistant.ui import MainWindow


@pytest.fixture
def window(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(Controller, "refresh_devices", lambda self: None)
    controller = Controller(":memory:")
    view = MainWindow(controller)
    view.show()
    controller.start_session("聊天界面测试", -1, None, {"collection": True}, "simulation")
    controller._simulation_timer.stop()
    view.tabs.setCurrentIndex(1)
    view.regulation_tabs.setCurrentWidget(view.regulation_chat)
    app.processEvents()
    yield view
    view.close()
    view.deleteLater()
    app.processEvents()


def start_activity(controller, key="break"):
    controller.request_intervention()
    controller.confirm_intervention(True)
    controller.start_activity(key, 5, 120)
    assert controller.flow.state == "running"


def wait_for(predicate):
    deadline = monotonic() + 4
    while not predicate() and monotonic() < deadline:
        QTest.qWait(10)
    assert predicate()


@pytest.mark.parametrize("command,mode,reason", [
    ("停一下", "quiet", "chat_stop"), ("我只想聊聊", "listening", "chat_listen"),
    ("这个练习让我不舒服", "quiet", "uncomfortable"),
])
def test_typed_commands_change_real_activity_without_model(window, command, mode, reason, tmp_path):
    chat, controller = window.regulation_chat, window.controller
    start_activity(controller)
    chat.editor.setPlainText(command)
    QTest.keyClick(chat.editor, Qt.Key.Key_Return)
    assert chat.editor.toPlainText() == ""
    assert controller.flow.state == "cooldown" and controller.chat_mode == mode
    assert controller.flow.current["stop_reason"] == reason
    assert not window.guide_widget.isVisible()
    assert chat.history[-2] == ("user", command)
    assert not chat.queue._running
    controller.writer.submit(lambda: None).result(timeout=5)
    path = controller.store.export_session(controller.session_id, tmp_path / "chat.json")
    record = json.loads(path.read_text(encoding="utf-8"))
    event = next(row["payload"] for row in record["events"] if row["kind"] == "regulation_chat_turn")
    assert (event["source"], event["state_before"], event["state_after"]) == ("local", "running", "cooldown")
    assert "message" not in event and "text" not in event


def test_switch_changes_selector_and_waits_for_user_to_start(window):
    chat, controller = window.regulation_chat, window.controller
    start_activity(controller)
    first = controller.flow.current["id"]
    chat.shortcuts[1].click()
    wait_for(lambda: controller.flow.state == "confirming")
    assert window.regulation_tabs.currentWidget() is chat
    selected = window.activity_combo.currentData()
    assert selected != "break" and ACTIVITIES[selected]["title"] in chat.history[-1][1]
    chat.activity_link.click()
    assert window.regulation_tabs.currentIndex() == 0
    window.accept_button.click()
    window.pre_combo.setCurrentIndex(window.pre_combo.findData(4))
    window.begin_activity_button.click()
    current = controller.flow.current
    assert current["activity"] == selected and current["previous_intervention_id"] == first


def test_no_eligible_switch_stops_previous_guidance(window):
    chat, controller = window.regulation_chat, window.controller
    start_activity(controller)
    for key in ACTIVITIES:
        if key != "break":
            controller.set_activity_preference(key, "avoid")
    chat.submit("换一种")
    assert controller.flow.state == "cooldown"
    assert "没有其他合适活动" in chat.history[-1][1]


def test_switch_retains_cancelled_activity_for_save_retry(window, monkeypatch):
    chat, controller = window.regulation_chat, window.controller
    start_activity(controller)
    first = controller.flow.current["id"]
    original = controller.store.save_activity_bundle
    failed = False
    def fail(*args):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("database unavailable")
        return original(*args)
    monkeypatch.setattr(controller.store, "save_activity_bundle", fail)
    chat.submit("换一种")
    wait_for(lambda: controller.flow.current["save_status"] == "failed")
    assert controller.flow.current["id"] == first and controller.flow.state == "cooldown"
    assert "重试保存" in chat.history[-1][1]
    assert controller._chat_switch is None
    controller.retry_activity_save()
    wait_for(lambda: controller.flow.current["save_status"] == "saved")
    chat.submit("换一种")
    assert controller.flow.state == "confirming" and controller._pending_previous_id == first


def test_delayed_model_reply_cannot_restart_after_stop(window):
    chat, controller = window.regulation_chat, window.controller
    release = Event()
    payloads = []

    def delayed(payload):
        payloads.append(payload)
        release.wait(3)
        return ChatTurn("switch", "旧回复", "qwen")

    chat.queue._callback = delayed
    start_activity(controller)
    try:
        chat.submit("这个办法跟上次的感觉不一样")
        wait_for(lambda: bool(payloads))
        assert payloads[0]["activity"] == "break"
        chat.shortcuts[0].click()
        assert controller.flow.state == "cooldown"
        expected = list(chat.history)
    finally:
        release.set()
    wait_for(lambda: not chat.queue._running)
    assert chat.history == expected and controller.flow.state == "cooldown"


def test_stop_during_switch_save_invalidates_the_waiting_alternative(window, monkeypatch):
    chat, controller = window.regulation_chat, window.controller
    start_activity(controller)
    save = controller.store.save_activity_bundle
    release = Event()
    def delayed(*args):
        release.wait(3)
        return save(*args)
    monkeypatch.setattr(controller.store, "save_activity_bundle", delayed)
    try:
        chat.submit("换一种")
        assert controller._chat_switch and controller.flow.state == "cooldown"
        chat.submit("停一下")
    finally:
        release.set()
    wait_for(lambda: controller.flow.current["save_status"] == "saved")
    assert controller.flow.state == "cooldown" and controller.chat_mode == "quiet"
    assert not any("另一种活动是" in msg for _, msg in chat.history)


def test_listening_mode_suppresses_an_otherwise_eligible_automatic_prompt(window, monkeypatch):
    chat, controller = window.regulation_chat, window.controller
    monkeypatch.setattr(controller.prompt_timing, "evaluate",
                        lambda *_args: {"action": "ask", "reason": "ready"})
    chat.submit("我只想聊聊")
    controller._evaluate_prompt(1.0)
    assert controller.flow.state == "observing"
    controller.request_intervention()
    assert controller.chat_mode == "open" and controller.flow.state == "confirming"


def test_model_receives_session_context_and_session_end_erases_text(window, tmp_path):
    chat, controller = window.regulation_chat, window.controller
    payloads = []

    def echo(payload):
        payloads.append(payload)
        return ChatTurn("reply", "想先说说哪一页？", "qwen")

    chat.queue._callback = echo
    chat.submit("我只想聊聊")
    chat.editor.setPlainText("明天我要在灯塔汇报")
    QTest.keyClick(chat.editor, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
    assert "明天我要在灯塔汇报" in chat.editor.toPlainText() and not payloads
    chat.send.click()
    wait_for(lambda: chat.history[-1][1] == "想先说说哪一页？")
    assert payloads[0]["listening"] and ("user", "我只想聊聊") in payloads[0]["history"]
    controller.writer.submit(lambda: None).result(timeout=5)
    exported = controller.store.export_session(controller.session_id, tmp_path / "chat.json")
    assert "灯塔" not in str(json.loads(exported.read_text(encoding="utf-8")))
    controller.stop_session()
    wait_for(lambda: not chat.editor.isEnabled())
    assert not chat.history and not chat.submit("换一种")
