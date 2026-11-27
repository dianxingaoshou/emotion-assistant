"""Capture the Qt chat using a simulated activity, scripted messages and the real local Qwen."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from regulation_smoke import QApplication, MainWindow, SmokeController, barrier, settle, wait_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/regulation/chat.png"))
    args = parser.parse_args()
    app = QApplication.instance() or QApplication([])
    controller = SmokeController(":memory:")
    window = MainWindow(controller)
    try:
        window.resize(1280, 900)
        window.show()
        controller.start_session("对话演示", -1, None, {"collection": True}, "simulation")
        controller._simulation_timer.stop()
        controller.request_intervention()
        controller.confirm_intervention(True)
        controller.start_activity("break", 5, 120)
        window.tabs.setCurrentIndex(1)
        panel = window.regulation_chat
        window.regulation_tabs.setCurrentWidget(panel)
        panel.shortcuts[2].click()
        assert controller.chat_mode == "listening" and controller.flow.state == "cooldown"
        panel.editor.setPlainText("明天要汇报，我怕一上台就忘了开场白。")
        panel.send.click()
        wait_for(lambda: panel.history[-1][0] == "assistant", timeout=90, label="local Qwen response")
        barrier(controller)
        with TemporaryDirectory() as scratch:
            exported = controller.store.export_session(controller.session_id, Path(scratch) / "session.json")
            events = json.loads(exported.read_text(encoding="utf-8"))["events"]
        event = next(item["payload"] for item in reversed(events) if item["kind"] == "regulation_chat_turn")
        if event["source"] != "qwen" or event["action"] != "reply":
            raise RuntimeError(f"Expected Qwen conversation reply, got {event['source']}/{event['action']}")
        settle(150)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not window.grab().save(str(args.output)):
            raise OSError(f"Could not save screenshot to {args.output}")
        print(f"Qt chat capture passed: {event['source']}/{event['action']}, {controller.chat_mode}")
    finally:
        window.close()
        app.processEvents()


if __name__ == "__main__":
    main()
