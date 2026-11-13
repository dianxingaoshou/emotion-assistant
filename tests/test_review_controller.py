"""Real Qt/SQLite integration checks for write ownership and failure recovery.

All sessions are explicit simulation fixtures; no capture runtime is started.
"""
from concurrent.futures import Future
import os
import threading
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PySide6.QtCore import QThread
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from emotion_assistant.controller import Controller
from emotion_assistant.ui import MainWindow


class ReviewController(Controller):
    def refresh_devices(self):
        # Construct the genuine UI while keeping device enumeration out of tests.
        self.devices_ready.emit({"cameras": [], "microphones": [], "errors": []})


@pytest.fixture(scope="module")
def app():
    result = QApplication.instance() or QApplication([])
    result.setQuitOnLastWindowClosed(False)
    return result


@pytest.fixture
def controller(app, tmp_path):
    result = ReviewController(tmp_path / "review-controller.sqlite3")
    result._activity_timer.stop()
    result._resource_timer.stop()
    result._freshness_timer.stop()
    yield result
    result.shutdown()
    app.processEvents()


def wait_for(app, predicate, timeout=4):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for Qt/database completion")
        QTest.qWait(5)
        app.processEvents()


def begin(controller, app):
    controller.start_session("线程与保存验收档案", -1, None, {"collection": True}, "simulation")
    controller._simulation_timer.stop()
    controller.request_intervention()
    controller.confirm_intervention(True)
    controller.start_activity("break", 7, 120)
    wait_for(app, lambda: (controller.flow.current or {}).get("save_status") == "saved")
    return controller.flow.current["id"]


def completion_notice(messages):
    return any(message.startswith("已保存即时复测与反馈") for message in messages)


def test_transaction_failure_never_claims_saved_and_ui_can_retry(controller, app, monkeypatch):
    window = MainWindow(controller)
    window.user_name.setText("线程与保存验收档案")
    window.mode_combo.setCurrentIndex(window.mode_combo.findData("simulation"))
    window.show()
    messages = []
    controller.status_changed.connect(messages.append)
    original = controller.store.save_activity_bundle
    fail = [False]

    def guarded_save(*args, **kwargs):
        if fail[0] and args[4] == "activity_completed":
            raise OSError("injected transaction failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(controller.store, "save_activity_bundle", guarded_save)
    try:
        record_id = begin(controller, app)
        window.tabs.setCurrentIndex(1)
        window.regulation_tabs.setCurrentIndex(0)
        fail[0] = True
        messages.clear()
        controller.complete_activity(3, "helpful")
        wait_for(app, lambda: controller.flow.current.get("save_status") == "failed")
        assert not completion_notice(messages)
        assert window.activity_current["save_status"] == "failed"
        assert "尚未保存" in window.activity_result.text()
        assert not window.retry_save_button.isHidden() and window.retry_save_button.isEnabled()
        assert controller.store.get_intervention(controller.session_id, record_id)["status"] == "running"

        fail[0] = False
        window.retry_save_button.click()
        wait_for(app, lambda: controller.flow.current.get("save_status") == "saved")
        assert completion_notice(messages)
        assert window.retry_save_button.isHidden()
        assert window.activity_result.text().startswith("已保存")
        assert controller.store.get_intervention(controller.session_id, record_id)["post"] == 3
    finally:
        window.close()


def test_completed_notice_waits_for_transaction_and_callback_mutates_on_owner_thread(controller, app, monkeypatch):
    record_id = begin(controller, app)
    messages, annotation_threads, refresh_threads = [], [], []
    controller.status_changed.connect(messages.append)
    entered, release = threading.Event(), threading.Event()
    original_save = controller.store.save_activity_bundle
    original_annotate = controller.flow.annotate
    original_refresh = controller.refresh_recommendations

    def blocked_save(*args, **kwargs):
        entered.set()
        if not release.wait(4):
            raise TimeoutError("test did not release transaction")
        return original_save(*args, **kwargs)

    def record_annotation(fields):
        if fields.get("save_status") in ("saved", "failed"):
            annotation_threads.append(QThread.currentThread())
        return original_annotate(fields)

    def record_refresh(*args, **kwargs):
        refresh_threads.append(QThread.currentThread())
        return original_refresh(*args, **kwargs)

    monkeypatch.setattr(controller.store, "save_activity_bundle", blocked_save)
    monkeypatch.setattr(controller.flow, "annotate", record_annotation)
    monkeypatch.setattr(controller, "refresh_recommendations", record_refresh)
    try:
        messages.clear()
        controller.complete_activity(4, "helpful")
        wait_for(app, entered.is_set)
        assert controller.flow.current["save_status"] == "pending"
        assert not completion_notice(messages)
        assert controller.store.get_intervention(controller.session_id, record_id)["status"] == "running"
        release.set()
        wait_for(app, lambda: controller.flow.current["save_status"] == "saved")
        assert completion_notice(messages)
        assert annotation_threads and all(thread == controller.thread() for thread in annotation_threads)
        assert refresh_threads and all(thread == controller.thread() for thread in refresh_threads)
        assert controller.store.get_intervention(controller.session_id, record_id)["post"] == 4
    finally:
        release.set()


def test_followup_worker_does_not_mutate_flow_until_owner_processes_queued_signal(controller, app, monkeypatch):
    record_id = begin(controller, app)
    controller.complete_activity(4, "helpful", followup_seconds=120)
    wait_for(app, lambda: controller.flow.current["save_status"] == "saved")
    annotation_threads = []
    original = controller.flow.annotate

    def annotate(fields):
        if "followup" in fields:
            annotation_threads.append(QThread.currentThread())
        return original(fields)

    monkeypatch.setattr(controller.flow, "annotate", annotate)
    future = controller.submit_followup(controller.session_id, record_id, 5, "no_change")
    assert future.result(timeout=4)["followup"]["status"] == "completed"
    # Future.result waits for SQLite, not for the GUI's queued completion slot.
    assert controller.flow.current["followup"]["status"] == "pending"
    assert annotation_threads == []
    wait_for(app, lambda: controller.flow.current["followup"]["status"] == "completed")
    assert annotation_threads and all(thread == controller.thread() for thread in annotation_threads)


@pytest.mark.parametrize("observation", [False, True])
def test_executor_submission_failure_releases_every_reserved_slot(controller, monkeypatch, observation):
    def rejected_submit(*args, **kwargs):
        raise RuntimeError("injected executor shutdown")

    function = controller.store.save_observation if observation else lambda: None
    with monkeypatch.context() as patch:
        patch.setattr(controller.writer, "submit", rejected_submit)
        future = controller._write(function)
        assert isinstance(future, Future)
        with pytest.raises(RuntimeError):
            future.result(timeout=1)
    # Count actual available reservations rather than relying on private counters.
    for semaphore, expected in ((controller._write_slots, 256), (controller._observation_slots, 192)):
        count = 0
        while semaphore.acquire(blocking=False):
            count += 1
        for _ in range(count):
            semaphore.release()
        assert count == expected
    assert controller._write(lambda: "next write still works").result(timeout=2) == "next write still works"


@pytest.mark.parametrize("failure", ["device", "database"])
def test_stop_failure_is_not_reported_as_released_and_can_be_retried(controller, app, monkeypatch, failure):
    controller.start_session("停止失败验收档案", -1, None, {"collection": True}, "simulation")
    controller._simulation_timer.stop()
    messages = []
    controller.status_changed.connect(messages.append)
    original_finish = controller.store.finish_session
    device_calls, database_calls = [], []

    class RuntimeFixture:
        def stop(self):
            device_calls.append(True)
            if failure == "device" and len(device_calls) == 1:
                raise OSError("injected device stop failure")

    def finish(session_id):
        database_calls.append(True)
        if failure == "database" and len(database_calls) == 1:
            raise OSError("injected finish persistence failure")
        return original_finish(session_id)

    runtime = RuntimeFixture()
    controller.runtime = runtime
    monkeypatch.setattr(controller.store, "finish_session", finish)
    controller.stop_session()
    wait_for(app, lambda: controller.state == "stop_failed")
    assert controller.runtime is runtime
    assert not any("设备已释放" in message for message in messages)
    session = next(item for item in controller.store.list_sessions() if item["id"] == controller.session_id)
    assert session["ended_at"] is None
    assert any("请重试结束" in message for message in messages)

    controller.stop_session()
    wait_for(app, lambda: controller.state == "stopped")
    assert controller.runtime is None and len(device_calls) == 2
    assert any("设备已释放" in message for message in messages)
    session = next(item for item in controller.store.list_sessions() if item["id"] == controller.session_id)
    assert session["ended_at"]


@pytest.mark.parametrize("foreign", ["profile", "mode"])
def test_foreign_followup_keeps_legacy_none_contract_without_writing(controller, app, foreign):
    record_id = begin(controller, app)
    controller.complete_activity(4, "helpful", followup_seconds=120)
    wait_for(app, lambda: controller.flow.current["save_status"] == "saved")
    if foreign == "profile":
        controller.current_user_id = "foreign-profile"
    else:
        controller.mode = "real"
    future = controller.submit_followup(controller.session_id, record_id, 2, "helpful")
    assert future.result(timeout=4) is None
    assert controller.store.get_intervention(controller.session_id, record_id)["followup"]["status"] == "pending"
