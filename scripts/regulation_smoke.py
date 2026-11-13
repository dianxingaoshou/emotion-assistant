"""Verify the regulation loop with scripted ratings and an isolated database.

No camera/microphone is opened. Screenshots show simulation fixtures, and the
scheduled follow-up clock is explicitly accelerated rather than actually waited.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["PYTHONPYCACHEPREFIX"] = str(ROOT / ".runtime" / "cache" / "pycache")
sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]
sys.path.insert(0, str(ROOT))

from emotion_assistant.paths import configure_environment

configure_environment()

from PySide6.QtCore import QEventLoop, QPoint, QRect, QTimer
from PySide6.QtWidgets import QApplication, QScrollArea

from emotion_assistant.controller import Controller
from emotion_assistant.regulation import ACTIVITIES, recommend_activities
from emotion_assistant.ui import MainWindow


class SmokeController(Controller):
    """Keep constructor refreshes inside the fixture; never enumerate hardware."""
    def refresh_devices(self):
        self.devices_ready.emit({"cameras": [], "microphones": [], "errors": []})


def settle(milliseconds=25):
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


def wait_for(predicate, timeout=5.0, label="condition"):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for " + label)
        settle()


def choose(combo, value):
    index = combo.findData(value)
    if index < 0:
        raise AssertionError(f"Missing combo option: {value}")
    combo.setCurrentIndex(index)


def barrier(controller):
    controller.writer.submit(lambda: None).result(timeout=5)
    settle()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/regulation")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(False)
    report = {"simulation": True, "real_devices_opened": False,
              "model_inference_performed": False, "ratings_are_scripted": True,
              "followup_clock_accelerated": True,
              "purpose": "Software workflow verification, not measured regulation efficacy.",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "checks": [], "screenshots": [], "sessions": [], "guided_controls": []}
    controller = window = None
    temporary_context = tempfile.TemporaryDirectory(prefix="regulation-smoke-", dir=ROOT / ".runtime" / "tmp")

    def check(condition, message):
        if not condition:
            raise AssertionError(message)
        report["checks"].append(message)

    def snapshot(name):
        for width, height in ((1280, 900), (1040, 760)):
            window.resize(width, height)
            settle()
            target = output / f"{name}-{width}.png"
            check(window.grab().save(str(target)), f"保存 {name} {width}px 截图")
            current = window.tabs.currentWidget()
            scrolls = current.findChildren(QScrollArea)
            if isinstance(current, QScrollArea):
                scrolls = [current] + scrolls
            visible = [scroll for scroll in scrolls if scroll.isVisible()]
            check(not any(scroll.horizontalScrollBar().maximum() > 0 for scroll in visible),
                  f"{name} {width}px 没有横向溢出")
            if name == "guided":
                for button in (window.open_retest_button, window.process_uncomfortable):
                    ancestors = [scroll for scroll in visible if scroll.isAncestorOf(button)]
                    geometry = []
                    for scroll in ancestors:
                        viewport = scroll.viewport()
                        rect = QRect(button.mapTo(viewport, QPoint(0, 0)), button.size())
                        geometry.append({"button_rect": [rect.x(), rect.y(), rect.width(), rect.height()],
                                         "viewport_rect": [0, 0, viewport.width(), viewport.height()],
                                         "contained": viewport.rect().contains(rect)})
                    inside = button.isVisible() and bool(geometry) and all(row["contained"] for row in geometry)
                    report["guided_controls"].append({"size": [width, height], "button": button.text(),
                                                       "fully_inside_scroll_viewports": inside,
                                                       "geometry": geometry})
                    check(inside, f"guided {width}px 的“{button.text()}”完整位于滚动视口内，无需滚动")
            report["screenshots"].append(str(target))
        window.resize(1280, 900)
        settle()

    def open_session(database):
        nonlocal controller, window
        controller = SmokeController(database)
        window = MainWindow(controller)
        window.show()
        settle()
        window.user_name.setText("调节闭环验收（模拟）")
        choose(window.mode_combo, "simulation")
        window.consent.setChecked(True)
        window.start_button.click()
        controller._simulation_timer.stop()
        check(controller.state == "running" and controller.mode == "simulation", "显式开始模拟会话")
        check(controller.runtime is None, "模拟会话不创建采集运行时")
        window._on_status("闭环验收 · 预设模拟评分 · 未打开摄像头或麦克风")
        return controller.session_id

    def prepare(activity, pre, need="unsure"):
        window.tabs.setCurrentIndex(1)
        window.regulation_tabs.setCurrentIndex(0)
        window.activity_request_button.click()
        check(controller.flow.state == "confirming", "开始活动前先确认需要")
        window.accept_button.click()
        check(controller.flow.state == "ready", "确认后进入活动选择")
        choose(window.need_combo, need)
        choose(window.available_combo, 120)
        choose(window.pre_combo, pre)
        choose(window.activity_combo, activity)
        choose(window.duration_combo, 120)

    try:
        database = Path(temporary_context.name) / "sessions.sqlite3"
        first_session = open_session(database)
        check(set(ACTIVITIES) == {"break", "breathing", "grounding", "task_split", "thought_label",
                                  "self_kindness", "values_action", "desk_reset"}, "活动库包含八种方法")
        check(window.activity_combo.count() == 8, "界面提供八种活动选择")
        prepare("thought_label", 6, "rumination")
        latest = []
        controller.recommendations_ready.connect(latest.append)
        controller.refresh_recommendations("rumination", 120)
        wait_for(lambda: latest and latest[-1].get("need") == "rumination", label="recommendations")
        check(all(item.get("reasons") for item in latest[-1]["items"]), "每个候选提供推荐或排除理由")
        check(window.activity_combo.currentData() == "thought_label", "异步推荐刷新保留用户已明确选择的方法")
        snapshot("recommendations")
        check(window.activity_combo.currentData() == "thought_label", "等待界面刷新后仍保留用户明确选择")
        window.begin_activity_button.click()
        check(controller.flow.state == "running", "分步骤活动开始")
        check((controller.flow.current or {}).get("activity") == "thought_label", "实际启动方法与用户选择一致")
        start_step = (controller.flow.current or {}).get("step_index", 0)
        window.next_guide_button.click()
        check((controller.flow.current or {}).get("step_index", 0) > start_step, "用户主动推进步骤")
        window.process_comfortable.click()
        window.process_no_change.click()
        barrier(controller)
        current = controller.flow.current
        first_activity = current["id"]
        record = controller.store.get_intervention(first_session, first_activity)
        check(record.get("activity") == "thought_label", "持久化活动方法与用户选择一致")
        check(len(record.get("checkins", [])) >= 2, "过程反馈保存到活动记录")
        snapshot("guided")
        window.open_retest_button.click()
        check(window.feedback_widget.isVisible(), "用户选择进入复测后展开反馈表单")
        choose(window.post_combo, 4)
        choose(window.helpfulness_combo, "helpful")
        window.followup_optin.setChecked(True)
        choose(window.followup_delay_combo, 120)
        window.complete_button.click()
        check(controller.flow.state == "cooldown", "即时反馈完成后进入冷却")
        barrier(controller)
        record = controller.store.get_intervention(first_session, first_activity)
        check(record.get("delta") == 2, "前后自评变化按脚本输入分别记录")
        check((record.get("followup") or {}).get("status") == "pending", "延迟复测仅在主动选择后创建")
        window.regulation_tabs.setCurrentIndex(1)
        choose(window.pref_activity_combo, "thought_label")
        choose(window.pref_state_combo, "prefer")
        window.save_preference_button.click()
        barrier(controller)
        user_id = controller.store.list_users()[0]["id"]
        check(controller.store.get_activity_preferences(user_id).get("thought_label") == "prefer", "主动偏好写入本地档案")

        # Reopen the same database before answering the follow-up. The due
        # time change is an explicit accelerated fixture, never a real wait.
        window.close()
        controller = window = None
        second_session = open_session(database)
        record = controller.store.get_intervention(first_session, first_activity)
        check((record.get("followup") or {}).get("status") == "pending", "关闭再打开后保留待复测记录")
        followup = dict(record["followup"])
        followup["due_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        followup["test_clock_accelerated"] = True
        controller.store.patch_intervention(first_session, first_activity, {"followup": followup})
        controller.refresh_recommendations("rumination", 120)
        wait_for(lambda: any((window.pending_followup_combo.itemData(index) or {}).get("id") == first_activity
                             for index in range(window.pending_followup_combo.count())), label="pending follow-up UI")
        window.regulation_tabs.setCurrentIndex(2)
        pending_index = next((index for index in range(window.pending_followup_combo.count())
                              if (window.pending_followup_combo.itemData(index) or {}).get("id") == first_activity), -1)
        check(pending_index >= 0, "当前档案界面显示重启后到期的复测")
        window.pending_followup_combo.setCurrentIndex(pending_index)
        choose(window.followup_distress_combo, 5)
        choose(window.followup_helpfulness_combo, "no_change")
        window.submit_followup_button.click()
        barrier(controller)
        saved_followup = controller.store.get_intervention(first_session, first_activity).get("followup") or {}
        check(saved_followup.get("status") == "completed", "延迟复测回写原活动")
        check(saved_followup.get("delta") == 1 and saved_followup.get("delta_from_immediate") == -1,
              "延迟复测分别比较活动前与即时后测，不把回升隐藏为持续改善")
        check(saved_followup.get("helpfulness") == "no_change", "主观无明显帮助与数值变化分别保存")
        controller.submit_followup(first_session, first_activity, 5, "no_change")
        barrier(controller)
        first_export = controller.store.export_session(first_session, output / "session-with-followup.json")
        exported = json.loads(first_export.read_text(encoding="utf-8"))
        check(sum(item["phase"] == "followup" for item in exported["self_reports"]) == 1,
              "重复提交不会重复生成延迟自评")

        window.regulation_tabs.setCurrentIndex(0)
        prepare("breathing", 5, "tense")
        window.begin_activity_button.click()
        check(controller.flow.state == "running", "第二种方法开始")
        check(controller.flow.current.get("activity") == "breathing", "不适分支实际执行用户选择的呼吸方法")
        bad_activity = controller.flow.current["id"]
        window.process_uncomfortable.click()
        check(controller.flow.state == "cooldown", "不舒服反馈立即停止本次引导")
        barrier(controller)
        bad = controller.store.get_intervention(second_session, bad_activity)
        check(bad.get("discomfort") is True and bad.get("status") == "cancelled", "不适与中止原因可追溯")
        check(bad.get("post") is None and bad.get("delta") is None, "中止缺失后测不会形成伪改善")
        recommendations = recommend_activities(controller.store.list_interventions(user_id, mode="simulation"),
                                                 need="tense", available_seconds=120,
                                                 preferences=controller.store.get_activity_preferences(user_id), mode="simulation")
        check(next(item for item in recommendations if item["activity"] == "breathing")["blocked"], "不适方法不再进入自动推荐候选")

        prepare("values_action", 4, "unsure")
        window.begin_activity_button.click()
        check(controller.flow.current.get("activity") == "values_action", "缺失后测分支实际执行用户选择的价值微行动")
        missing_activity = controller.flow.current["id"]
        window.open_retest_button.click()
        choose(window.post_combo, None)
        choose(window.helpfulness_combo, "unknown")
        window.followup_optin.setChecked(False)
        window.complete_button.click()
        barrier(controller)
        missing = controller.store.get_intervention(second_session, missing_activity)
        check(missing.get("delta") is None and missing.get("evaluation") == "insufficient", "未填写即时后测时保持无法评价")
        real_history = controller.store.list_interventions(user_id, mode="real")
        real_items = recommend_activities(real_history, need="unsure", available_seconds=120, mode="real")
        check(all(item["evidence_count"] == 0 for item in real_items), "模拟活动不进入真实模式历史统计")

        # A rendering-only payload exercises contradictory feedback without
        # inventing another persisted activity or an actual efficacy result.
        actual_history = controller.store.list_sessions()
        first_history = next(item for item in actual_history if item["id"] == first_session)
        discomfort_payload = dict(first_history, id="render-only-discomfort-fixture",
                                  user_name="历史文案验收（模拟）", last_evaluation="improved",
                                  last_delta=2, last_helpfulness="worse",
                                  latest_loop_state="discomfort", latest_discomfort=True,
                                  latest_followup_status=None,
                                  latest_loop_caption="预设渲染输入：用户反馈更不舒服，优先保留不适。")
        window.tabs.setCurrentIndex(2)
        choose(window.history_filter, "all")
        window.history_search.clear()
        window._on_history([discomfort_payload])
        check(window.history_table.item(0, 6).text() == "报告不适",
              "历史摘要对数值下降但主观更不舒服优先显示报告不适")
        discomfort_details = window.history_details.toPlainText()
        check("前测减后测 = 2" in discomfort_details and "即时主观帮助程度：更不舒服" in discomfort_details,
              "不适详情同时保留即时数值与主观不适，不覆盖矛盾信息")
        window._on_history([first_history])
        check(window.history_table.item(0, 6).text() == "稍后自评下降",
              "历史摘要明确显示已完成的稍后自评结果")
        followup_details = window.history_details.toPlainText()
        check("即时前后自评：用户自评困扰下降（前测减后测 = 2）" in followup_details,
              "完成延迟复测后的详情仍保留即时前后自评数值")
        check("稍后复测：已完成" in followup_details,
              "历史详情区分即时反馈与已完成的稍后复测状态")
        window._on_history(actual_history)
        check(window.history_table.rowCount() == len(actual_history)
              and all(item.get("id") != "render-only-discomfort-fixture" for item in window._all_history),
              "历史渲染验收后恢复数据库真实测试会话")
        window.tabs.setCurrentIndex(1)
        window.regulation_tabs.setCurrentIndex(1)
        window.refresh_feedback_button.click()
        wait_for(lambda: window.feedback_table.rowCount() > 0, label="personal feedback table")
        snapshot("feedback")
        barrier(controller)
        second_export = controller.store.export_session(second_session, output / "session-process-feedback.json")
        report["sessions"] = [str(first_export), str(second_export)]
        check(controller.runtime is None, "全程保持无真实采集运行时")
        window.close()
        controller = window = None
        report["passed"] = True
    except Exception as exc:
        report["passed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        if window is not None:
            window.close()
        elif controller is not None:
            controller.shutdown()
        temporary_path = Path(temporary_context.name).resolve()
        if not temporary_path.is_relative_to((ROOT / ".runtime" / "tmp").resolve()):
            raise RuntimeError("Temporary cleanup target is outside project runtime")
        temporary_context.cleanup()
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": len(report["checks"]),
                      "report": str(output / "result.json"), "error": report.get("error")}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
