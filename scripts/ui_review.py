"""Render native desktop UI fixtures without opening any capture device.

The images and metrics describe UI layout only. Scores, times and self reports
are scripted simulation inputs, not inference, performance or effect results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["PYTHONPYCACHEPREFIX"] = str(ROOT / ".runtime" / "cache" / "pycache")
sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]
sys.path.insert(0, str(ROOT))

from emotion_assistant.paths import configure_environment

configure_environment()

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QPushButton, QScrollArea

from emotion_assistant.controller import Controller, simulation_observation
from emotion_assistant.ui import MainWindow


class ReviewController(Controller):
    """Keep constructor refreshes entirely inside the scripted fixture."""
    def refresh_devices(self):
        self.devices_ready.emit({"cameras": [], "microphones": [], "errors": []})

    def refresh_history(self):
        if not self.closed:
            self.writer.submit(lambda: None).result(timeout=5)
            self.history_ready.emit(self.store.list_sessions())


def settle(milliseconds=35):
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


def primary_buttons(window):
    return [{"text": button.text(), "enabled": button.isEnabled(),
             "geometry": [button.x(), button.y(), button.width(), button.height()]}
            for button in window.findChildren(QPushButton)
            if button.objectName() == "primary" and button.isVisible()]


def capture(window, output_dir, name, requested_size):
    window.resize(*requested_size)
    settle()
    scrolls = []
    for scroll in window.tabs.currentWidget().findChildren(QScrollArea):
        if scroll.isVisible():
            scrolls.append({"horizontal_maximum": scroll.horizontalScrollBar().maximum(),
                            "vertical_maximum": scroll.verticalScrollBar().maximum(),
                            "viewport_size": [scroll.viewport().width(), scroll.viewport().height()]})
    page = window.tabs.currentWidget()
    if isinstance(page, QScrollArea):
        scrolls.insert(0, {"horizontal_maximum": page.horizontalScrollBar().maximum(),
                          "vertical_maximum": page.verticalScrollBar().maximum(),
                          "viewport_size": [page.viewport().width(), page.viewport().height()]})
    target = output_dir / f"{name}-{requested_size[0]}.png"
    if not window.grab().save(str(target)):
        raise RuntimeError(f"Could not save {target}")
    return {"state": name, "requested_size": list(requested_size),
            "actual_size": [window.width(), window.height()],
            "minimum_size": [window.minimumWidth(), window.minimumHeight()],
            "page": window.tabs.currentIndex(), "scroll_areas": scrolls,
            "horizontal_overflow": any(row["horizontal_maximum"] > 0 for row in scrolls),
            "visible_primary_actions": primary_buttons(window), "screenshot": str(target),
            "simulation": True, "real_devices_opened": False}


def render_state(window, output_dir, name, results):
    for size in ((1280, 900), (1040, 760)):
        results.append(capture(window, output_dir, name, size))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/refactoring-ui/after")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(False)
    controller = ReviewController(":memory:")
    window = MainWindow(controller)
    snapshots = []
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "simulation": True, "real_devices_opened": False,
              "model_inference_performed": False,
              "purpose": "Native Qt layout verification; every score and rating is a scripted fixture.",
              "snapshots": snapshots}
    try:
        window.show()
        settle()
        window.user_name.setText("界面评审（模拟）")
        window.mode_combo.setCurrentIndex(window.mode_combo.findData("simulation"))
        window._on_status("界面评审 · 预设模拟内容 · 未打开摄像头或麦克风")
        render_state(window, output_dir, "idle", snapshots)
        window.tabs.setCurrentIndex(2)
        window._on_history([])
        render_state(window, output_dir, "history-empty", snapshots)
        window.tabs.setCurrentIndex(0)
        window.consent.setChecked(True)
        window.start_button.click()
        controller._simulation_timer.stop()
        assert controller.runtime is None and controller.mode == "simulation"
        controller.started = time.monotonic() - 22
        window._session_started = controller.started
        window._update_clock()
        for index in range(40):
            start, end = index * .5, (index + 1) * .5
            observation = simulation_observation(controller.session_id, "visual", start, end,
                                                 negative=.35 + .25 * index / 39)
            if 28 <= index < 31:
                observation.update(valid=False, scores=None, quality=0,
                                   reason="模拟遮挡，非实测")
            controller._handle_event({"kind": "observation", "observation": observation})
            if index % 8 == 7:
                audio = simulation_observation(controller.session_id, "audio", end - 4, end,
                                               negative=.50 + .04 * index / 39)
                controller._handle_event({"kind": "observation", "observation": audio})
        window._on_status("界面评审 · 所有线索均为预设模拟输入，非实测")
        render_state(window, output_dir, "live", snapshots)
        probe = capture(window, output_dir, "mobile-probe", (375, 812))
        probe["supported"] = probe["actual_size"][0] == 375
        probe["limitation"] = ("This native desktop application preserves its declared minimum width; "
                               "a 375px mobile viewport is outside its supported platform scope.")
        report["mobile_probe"] = probe
        controller.request_intervention()
        controller.confirm_intervention(True)
        window.pre_combo.setCurrentIndex(window.pre_combo.findData(5))
        window.activity_combo.setCurrentIndex(window.activity_combo.findData("breathing"))
        window.duration_combo.setCurrentIndex(window.duration_combo.findData(60))
        render_state(window, output_dir, "activity-ready", snapshots)
        window.begin_activity_button.click()
        controller.activity_started = time.monotonic() - 15
        controller._activity_tick()
        render_state(window, output_dir, "activity-running", snapshots)
        window.open_retest_button.click()
        window.post_combo.setCurrentIndex(window.post_combo.findData(4))
        window.helpfulness_combo.setCurrentIndex(window.helpfulness_combo.findData("unknown"))
        window.complete_button.click()
        controller.writer.submit(lambda: None).result(timeout=5)
        controller.store.finish_session(controller.session_id)
        controller.state = "stopped"
        controller._session_signal()
        window.tabs.setCurrentIndex(2)
        window._on_history(controller.store.list_sessions())
        window._on_status("界面评审 · 历史及自评分数均为预设测试记录，非真实改善")
        render_state(window, output_dir, "history", snapshots)
        report["passed"] = all(not item["horizontal_overflow"] for item in snapshots)
        report["desktop_sizes"] = [[1280, 900], [1040, 760]]
        report["mobile_scope"] = "Not supported: native desktop minimum width is preserved."
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        report["passed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        window.close()
        controller.shutdown()
        (output_dir / "layout.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "snapshots": len(snapshots),
                      "output_dir": str(output_dir), "mobile_actual": report["mobile_probe"]["actual_size"]}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
