from __future__ import annotations

import argparse
import multiprocessing
import os
import sys

from .paths import configure_environment, REPORTS


def main():
    configure_environment()
    parser = argparse.ArgumentParser(description="桌面情绪线索与自我调节助手（本地原型）")
    parser.add_argument("--smoke", action="store_true", help="离屏界面启动验证，不打开设备")
    parser.add_argument("--screenshot", action="store_true", help="离屏模拟截图，不打开设备")
    args = parser.parse_args()
    if args.smoke or args.screenshot:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from .controller import Controller
    from .ui import MainWindow
    app = QApplication(sys.argv[:1])
    app.setApplicationName("桌面情绪线索助手")
    controller = Controller(database=":memory:" if args.smoke or args.screenshot else None)
    window = MainWindow(controller)
    window.show()
    controller.refresh_devices()
    controller.refresh_history()
    if args.screenshot:
        QTimer.singleShot(300, lambda: controller.start_session("模拟演示", -1, None,
                          {"collection": True, "transcript": False}, "simulation"))
        QTimer.singleShot(3500, lambda: window.grab().save(str(REPORTS / "desktop-preview.png")))
    if args.smoke or args.screenshot:
        QTimer.singleShot(4500 if args.screenshot else 500, window.close)
    app.aboutToQuit.connect(controller.shutdown)
    return app.exec()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
