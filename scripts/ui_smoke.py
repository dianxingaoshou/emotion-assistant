"""Exercise the explicit desktop workflow without opening capture devices.

Run with .venv/Scripts/python.exe scripts/ui_smoke.py.  All events, exports and
screenshots are labelled simulation and cannot be used as real model results.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from emotion_assistant.paths import REPORTS, configure_environment

configure_environment()

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from emotion_assistant.controller import Controller
from emotion_assistant.ui import MainWindow


def main() -> int:
    app = QApplication(sys.argv[:1])
    controller = Controller(":memory:")
    window = MainWindow(controller)
    window.show()
    export_path = REPORTS / "ui-smoke-session.json"
    export_path.unlink(missing_ok=True)
    result = {"simulation": True, "real_devices_opened": False, "checks": [], "screenshots": []}
    finished = False

    def check(condition, message):
        if not condition:
            raise AssertionError(message)
        result["checks"].append(message)

    def save(name):
        target = REPORTS / name
        check(window.grab().save(str(target)), "保存截图 " + name)
        result["screenshots"].append(str(target))

    def fail(exc):
        nonlocal finished
        if finished:
            return
        finished = True
        result["passed"] = False
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        (REPORTS / "ui-smoke-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("UI smoke failed:", exc)
        controller.shutdown()
        app.exit(1)

    def phase(function):
        def callback():
            if not finished:
                try:
                    function()
                except Exception as exc:
                    fail(exc)
        return callback

    def start():
        check(controller.state == "idle", "启动后不自动采集")
        check(window.mic_combo.currentData() is None, "麦克风默认关闭")
        check(not window.transcript_consent.isChecked(), "转写保存默认关闭")
        check(not window.start_button.isEnabled(), "缺少采集授权时不可开始")
        window.nav_buttons[2].click()
        check(window.tabs.currentIndex() == 2 and window.nav_buttons[2].isChecked(), "侧栏导航切换历史页面")
        check(sum(button.isChecked() for button in window.nav_buttons) == 1, "侧栏只突出当前页面")
        check(window.history_empty.isVisible() and not window.history_export.isEnabled(), "空历史显示指引并禁用导出")
        check(window.history_tools.isHidden() and window.history_stats.isHidden()
              and window.history_detail_card.isHidden() and window.history_empty_start.isVisible(),
              "首次空历史隐藏无效工具并保留开始入口")
        window.nav_buttons[0].click()
        window.user_name.setText("界面验收（模拟）")
        window.mode_combo.setCurrentIndex(window.mode_combo.findData("simulation"))
        window.consent.setChecked(True)
        window.start_button.click()
        check(controller.state == "running" and controller.mode == "simulation", "显式开始模拟会话")
        check(controller.runtime is None, "模拟模式不创建真实采集运行时")
        check(window.settings_details.isHidden() and not window.settings_toggle.isChecked(), "开始后收起设备设置")
        check(window.pause_button.isVisible() and window.stop_button.isVisible(), "采集中保留暂停与结束操作")

    def activity():
        check(window.visual_card.state.text() == "可用", "界面接收模拟视觉事件")
        check(window.audio_card.state.text() == "可用", "界面接收模拟语音事件")
        save("ui-live.png")
        window.request_button.click()
        check(controller.flow.state == "confirming", "主动请求进入确认")
        check(window.nav_buttons[1].isChecked() and window.confirm_widget.isVisible(), "确认流程同步侧栏与当前步骤")
        check(window.activity_flow._stage == 0, "确认时突出第一步")
        window.accept_button.click()
        check(controller.flow.state == "ready", "用户确认后允许选择活动")
        check(window.activity_flow._stage == 1 and window.setup_widget.isVisible(), "确认后突出调节前自评")
        window.pre_combo.setCurrentIndex(window.pre_combo.findData(5))
        window.activity_combo.setCurrentIndex(window.activity_combo.findData("breathing"))
        window.duration_combo.setCurrentIndex(window.duration_combo.findData(60))
        window.begin_activity_button.click()
        check(controller.flow.state == "running", "自评后开始活动")
        check(window.activity_flow._stage == 2 and window.feedback_widget.isHidden()
              and window.open_retest_button.isVisible(), "活动中突出执行并提供主动复测入口")
        check(window.setup_widget.isHidden() and window.activity_summary.isVisible(), "活动中用选择摘要替代设置表单")
        QTimer.singleShot(900, phase(complete))

    def complete():
        save("ui-activity.png")
        window.open_retest_button.click()
        check(window.feedback_widget.isVisible(), "主动进入复测后显示前后反馈表单")
        window.post_combo.setCurrentIndex(window.post_combo.findData(3))
        window.helpfulness_combo.setCurrentIndex(window.helpfulness_combo.findData("helpful"))
        window.complete_button.click()
        check(controller.flow.state == "cooldown", "完成活动进入冷却期")
        check(controller.flow.current["delta"] == 2, "前后自评分别记录并计算变化")
        check(window.activity_flow._stage == 3 and window.feedback_widget.isHidden(), "提交后进入反馈完成阶段")
        saved_poll.start()

    def after_save():
        # A completed flow is not proof of durable storage. New activity entry
        # stays disabled until the real queued SQLite completion reaches Qt.
        if (controller.flow.current or {}).get("save_status") != "saved":
            return
        saved_poll.stop()
        check(window.activity_request_button.isEnabled(), "写入成功后恢复新活动入口")
        window.activity_request_button.click()
        check(controller.flow.state == "confirming" and window.activity_result.objectName() != "savedFeedback"
              and not window.activity_result.text().startswith("已保存"),
              "再次请求活动时清除上次完成提示")
        window.reject_button.click()
        window.pause_button.click()
        check(controller.state == "paused", "暂停控件有效")
        window.pause_button.click()
        check(controller.state == "running", "继续采集控件有效")
        window.stop_button.click()
        poll.start()

    def poll_history():
        if controller.state != "stopped":
            return
        if window.tabs.currentIndex() != 2:
            window.tabs.setCurrentIndex(2)
            controller.refresh_history()
        if window.history_table.rowCount() < 1:
            return
        poll.stop()
        check(controller.state == "stopped", "结束会话完成设备释放流程")
        check(window._selected_session() is not None and window.history_export.isEnabled(), "历史自动选择首条记录并允许导出")
        check(window.history_tools.isVisible() and window.history_stats.isVisible()
              and window.history_detail_card.isVisible() and window.history_empty_start.isHidden()
              and window.history_export.objectName() == "primary", "有记录后恢复工具并突出所选记录导出")
        window.history_filter.setCurrentIndex(window.history_filter.findData("real"))
        check(window.history_table.rowCount() == 0 and window.history_empty.isVisible(), "真实来源筛选排除模拟记录")
        check(not window.history_export.isEnabled() and not window.history_delete.isEnabled(), "筛选为空时禁用记录操作")
        window.history_filter.setCurrentIndex(window.history_filter.findData("simulation"))
        check(window.history_table.rowCount() == 1, "模拟来源筛选恢复模拟记录")
        window.history_search.setText("no-such-profile-for-ui-smoke")
        check(window.history_table.rowCount() == 0 and window.history_empty.isVisible(), "搜索无匹配时显示空状态")
        window.history_search.clear()
        window.history_filter.setCurrentIndex(window.history_filter.findData("all"))
        check(window.history_table.rowCount() == 1 and window._selected_session() is not None, "清除筛选后恢复记录与选择")
        window.history_table.selectRow(0)
        save("ui-history.png")
        controller.export_session(controller.session_id, str(export_path))
        export_poll.start()

    def verify_export():
        nonlocal finished
        if not export_path.exists():
            return
        try:
            exported = json.loads(export_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        export_poll.stop()
        check(exported["session"]["mode"] == "simulation", "导出明确保存模拟标记")
        check(len(exported["observations"]) > 0, "结构化观察已保存")
        check(len(exported["self_reports"]) == 2, "调节前后两次自评已保存")
        check(any(record.get("status") == "completed" for record in exported["interventions"]), "完整活动反馈已保存")
        check(not exported["utterances"], "未授权转写时不保存转写文本")
        window.close()
        check(controller.closed, "关闭界面停止控制器")
        result["passed"] = True
        result["export"] = str(export_path)
        (REPORTS / "ui-smoke-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        finished = True
        print(f"UI smoke passed: {len(result['checks'])} checks; screenshots in {REPORTS}")
        app.quit()

    poll = QTimer()
    poll.setInterval(100)
    poll.timeout.connect(phase(poll_history))
    saved_poll = QTimer()
    saved_poll.setInterval(50)
    saved_poll.timeout.connect(phase(after_save))
    export_poll = QTimer()
    export_poll.setInterval(100)
    export_poll.timeout.connect(phase(verify_export))
    QTimer.singleShot(100, phase(start))
    QTimer.singleShot(5300, phase(activity))
    QTimer.singleShot(20000, lambda: fail(TimeoutError("界面闭环验收超过 20 秒")) if not finished else None)
    app.aboutToQuit.connect(controller.shutdown)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
