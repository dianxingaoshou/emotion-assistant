"""Exercise the voluntary two-period flow with actual widgets and an isolated DB."""
import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from emotion_assistant.controller import Controller
from emotion_assistant.ui import MainWindow


class NoDeviceController(Controller):
    def refresh_devices(self):
        self.devices_ready.emit({"cameras": [], "microphones": [], "errors": []})


def test_two_visits_ratings_export_and_withdrawal(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    controller = NoDeviceController(tmp_path / "synthetic.sqlite3")
    window = MainWindow(controller)
    page = window.trial_page
    window.show()
    app.processEvents()

    def open_experiment():
        window.nav_buttons[4].click()
        assert window.tabs.currentWidget() is page

    def enroll_and_start():
        open_experiment()
        assert page.consent.isEnabled() and not page.consent.isChecked()
        page.consent.click()
        page.setup_button.click()
        assert window.tabs.currentIndex() == 0 and "实验" in window.start_button.text()
        window.consent.setChecked(True)
        window.start_button.click()
        assert controller.state == "running"
        controller._simulation_timer.stop()
        assert not page.consent.isChecked()

    def stop():
        window.stop_button.click()
        deadline = time.monotonic() + 6
        while controller.state != "stopped" and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.005)
        assert controller.state == "stopped"

    try:
        open_experiment()
        assert not page.consent.isEnabled() and "代号" in page.progress.text()
        window.user_name.setText("P-test")
        page.consent.click()
        window.mode_combo.setCurrentIndex(window.mode_combo.findData("simulation"))
        assert not page.consent.isChecked() and "非真人" in page.progress.text()
        page.report_button.click()
        assert "还没有" in page.summary.text()

        enroll_and_start()
        first = controller.trial.copy()
        assert first["period"] == 1 and "实验第 1 场" in window.session_summary.text()
        controller.request_intervention("sustained_observation", "scripted-prompt")
        feedback = window.trial_prompt_feedback
        assert feedback.isVisible() and window.accept_button.isEnabled()
        feedback.unwanted.click()
        assert controller.flow.state == "confirming" and "已记录" in feedback.note.text()
        window.accept_button.click()  # Independently accepts an activity despite disliking the timing.
        assert controller.flow.state == "ready" and not feedback.isVisible()
        stop()

        open_experiment()
        assert "第 2 场" in page.progress.text()
        window.nav_buttons[0].click()
        window.start_button.click()
        assert controller.state == "running" and controller.trial is None
        stop()

        enroll_and_start()
        assert controller.trial["period"] == 2 and controller.trial["condition"] != first["condition"]
        window.request_button.click()
        assert controller.flow.state == "confirming" and not window.trial_prompt_feedback.isVisible()
        window.reject_button.click()
        stop()
        open_experiment()
        assert not page.consent.isEnabled() and "已完成" in page.progress.text()
        page.report_button.click()
        assert page.results.isVisible() and "1 位完成" in page.summary.text()
        export = tmp_path / "aggregate.json"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(export), "JSON (*.json)"))
        page.export_button.click()
        data = json.loads(export.read_text(encoding="utf-8"))
        arm = data["by_condition"][first["condition"]]
        assert data["mode"] == "simulation" and data["periods_assigned"] == 2
        assert (arm["prompts_asked"], arm["unwanted_prompts"], arm["prompt_accepted"]) == (1, 1, 1)
        assert "P-test" not in export.read_text(encoding="utf-8")
        window.mode_combo.setCurrentIndex(window.mode_combo.findData("real"))
        page.report_button.click()
        assert not page.results.isVisible() and "真人测试" in page.summary.text() and "还没有" in page.summary.text()
        window.mode_combo.setCurrentIndex(window.mode_combo.findData("simulation"))
        monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
        page.withdraw_button.click()
        assert "已退出" in page.progress.text()
        page.report_button.click()
        assert "还没有" in page.summary.text()
    finally:
        window.close()
        app.processEvents()
