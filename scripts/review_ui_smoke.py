"""Exercise review interfaces through the real controller and native Qt UI.

All observations, ratings, timing values, failures and transcripts are fixtures.
No device or neural model is opened; these checks do not measure efficacy or
hardware performance. Source data are kept in a temporary project-local DB.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import traceback

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
from emotion_assistant.evidence import raw_prediction_diagnostics
from emotion_assistant.ui import MainWindow


class ReviewController(Controller):
    def refresh_devices(self):
        self.devices_ready.emit({"cameras": [], "microphones": [], "errors": []})


def settle(milliseconds=30):
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


def wait_for(predicate, label, timeout=5):
    deadline = time.monotonic()+timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError(label)
        settle()


def choose(combo, value):
    index = combo.findData(value)
    if index < 0:
        raise AssertionError(f"Missing UI option {value}")
    combo.setCurrentIndex(index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/review")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT/output
    output.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(False)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "passed": False,
              "simulation": True, "real_devices_opened": False, "model_inference_performed": False,
              "ratings_are_scripted": True, "timing_values_are_scripted": True,
              "storage_and_stop_failures_are_injected": True,
              "purpose": "UI/controller contract verification; not model accuracy, performance or regulation efficacy.",
              "checks": [], "screenshots": [], "snapshots": []}
    scratch = tempfile.TemporaryDirectory(prefix="review-ui-", dir=ROOT/".runtime"/"tmp")
    controller = window = None
    release_write = threading.Event()
    original_bundle = None

    def check(condition, message):
        if not condition:
            raise AssertionError(message)
        report["checks"].append(message)

    def barrier():
        controller.writer.submit(lambda: None).result(timeout=5)
        settle()

    def snapshot(name, focus=None):
        for width, height in ((1280, 900), (1040, 760)):
            window.resize(width, height)
            settle()
            page = window.tabs.currentWidget()
            if focus is not None and isinstance(page, QScrollArea):
                page.ensureWidgetVisible(focus)
                settle()
            scrolls = page.findChildren(QScrollArea)
            if isinstance(page, QScrollArea):
                scrolls = [page]+scrolls
            metrics = [{"horizontal_maximum": scroll.horizontalScrollBar().maximum(),
                        "viewport": [scroll.viewport().width(), scroll.viewport().height()]}
                       for scroll in scrolls if scroll.isVisible()]
            check(not any(item["horizontal_maximum"] for item in metrics), f"{name} {width}px 无横向溢出")
            primaries = [button.text() for button in window.findChildren(QPushButton)
                         if button.isVisible() and button.isEnabled() and button.objectName() == "primary"]
            check(len(primaries) <= 1, f"{name} {width}px 可用主操作最多一个")
            target = output/f"{name}-{width}.png"
            check(window.grab().save(str(target)), f"保存 {name} {width}px 界面截图")
            report["screenshots"].append(str(target))
            report["snapshots"].append({"state": name, "size": [width, height],
                "scroll_areas": metrics, "available_primary_actions": primaries, "screenshot": str(target)})
        window.resize(1280, 900)
        settle()

    try:
        controller = ReviewController(Path(scratch.name)/"sessions.sqlite3")
        window = MainWindow(controller)
        window.show()
        settle()
        statuses = []
        controller.status_changed.connect(statuses.append)
        window.user_name.setText("代码审查接口验收（模拟）")
        choose(window.mode_combo, "simulation")
        window.transcript_consent.setChecked(True)
        window.consent.setChecked(True)
        window.start_button.click()
        for timer in (controller._simulation_timer, controller._activity_timer,
                      controller._resource_timer, controller._freshness_timer):
            timer.stop()
        check(controller.state == "running" and controller.mode == "simulation", "用户显式开启模拟会话")
        check(controller.runtime is None, "未创建摄像头或麦克风采集运行时")
        check("转写已开启" in window.transcript.placeholderText() and "未开启" not in window.transcript.placeholderText(),
              "已授权运行中的转写占位与撤回按钮语义一致")
        controller.started = time.monotonic()-10
        session = controller.session_id

        uncertain = simulation_observation(session, "visual", 8, 8.5)
        uncertain.update(valid=False, scores=None, quality=.98, reason="prediction_near_uniform",
                         raw_scores={key: 1/8 for key in ("anger", "contempt", "disgust", "fear",
                             "happiness", "neutral", "sadness", "surprise")},
                         queue_wait_ms=23, inference_ms=45, end_to_end_ms=120)
        uncertain["prediction_diagnostics"] = raw_prediction_diagnostics(uncertain["raw_scores"])
        controller._handle_event({"kind": "observation", "observation": uncertain})
        window.visual_card.details_toggle.setChecked(True)
        check("接近均匀" in window.visual_card.reason.text(), "原始近均匀诊断接入视觉卡片说明")
        check(window.visual_card.state.text() == "未参与判断", "近均匀预测不显示为有效情绪判断")
        check("98%" in window.visual_card.quality_label.text(), "清晰输入质量与预测不确定性分别显示")
        check("原始分布熵" in window.visual_card.metrics.text() and "最高类别" in window.visual_card.metrics.text(),
              "详情展示原始预测熵与最高类别")
        check("排队 23 ms" in window.visual_card.metrics.text() and "推理 45 ms" in window.visual_card.metrics.text()
              and "采集结束至返回 120 ms" in window.visual_card.metrics.text(), "详情分别展示排队、推理和端到端耗时")
        controller.evidence_gate.reset(session)
        for index in range(8):
            controller._handle_event({"kind": "observation", "observation":
                simulation_observation(session, "visual", index*.5, (index+1)*.5, negative=.8)})
        # Restore the diagnostic example after recording four seconds of valid
        # evidence; its later invalid interval cannot add to the settled total.
        controller._handle_event({"kind": "observation", "observation": uncertain})
        check("4.0 / 20 秒" in window.evidence_progress.text(), "有效时长接口显示4秒区间并集而非首尾跨度")
        check("100%" in window.evidence_progress.text() and "3 秒" in window.evidence_progress.text(),
              "证据密度与有限等待参数接入实时页面")
        check(controller.flow.state == "observing", "四秒证据不会触发二十秒询问")
        snapshot("review-diagnostics")
        snapshot("review-model-details", window.visual_card)

        transcript = {"event_id": "authorized-transcript", "session_id": session,
                      "t_start": 5, "t_end": 6, "text": "已授权的模拟转写", "valid": True}
        controller._handle_event({"kind": "utterance", "utterance": transcript})
        barrier()
        check("已授权的模拟转写" in window.transcript.toPlainText(), "授权时转写接口进入UI")
        check(window.revoke_transcript_button.isVisible(), "运行中提供撤回转写入口")
        window.revoke_transcript_button.click()
        barrier()
        before = window.transcript.toPlainText()
        controller._handle_event({"kind": "utterance", "utterance":
            {**transcript, "event_id": "late-transcript", "text": "禁止发布的迟到文本"}})
        check(not controller.transcript_allowed and not window.transcript_consent.isChecked(), "撤回按钮同步控制器授权和界面勾选")
        check(window.transcript.toPlainText() == before, "撤回后旧转写结果不进入界面")
        consent = controller.store.list_sessions()[0]["consent"]
        check(consent.get("save_transcript") is False, "撤回授权已持久化")

        window.request_button.click()
        window.accept_button.click()
        check(controller._assessment_phase == "pre" and controller.assessment is not None,
              "确认调节后创建独立辅助前测窗口")
        check(window.pre_assessment_hint.isVisible() and "30" in window.pre_assessment_hint.text(),
              "前测窗口状态与可直接开始说明接入准备页面")
        choose(window.pre_combo, 6)
        choose(window.activity_combo, "thought_label")
        window.begin_activity_button.click()
        wait_for(lambda: (controller.flow.current or {}).get("save_status") == "saved", "activity start write")
        check(controller.pre_assessment["status"] == "skipped", "未等待的前测保留跳过状态，不伪造有效观察")
        window.open_retest_button.click()
        first_post = controller.assessment.snapshot(time.monotonic()-controller.started)
        check(first_post["phase"] == "post" and first_post["window_start"] >= controller.activity_started-controller.started,
              "主动复测创建结束练习后开始的独立后测窗")
        check(window.skip_assessment_button.isVisible() and "独立后测" in window.assessment_hint.text(),
              "独立后测说明与跳过入口接入表单")
        check("00:30" in window.countdown.text(), "独立后测显示30秒窗口倒计时而非练习计划时长")
        window.skip_assessment_button.click()
        check(controller.post_assessment["status"] == "skipped" and "已跳过" in window.assessment_hint.text(),
              "跳过按钮调用后端并同步显示真实状态")
        check(window.skip_assessment_button.isHidden(), "跳过后不再重复提供同一观察窗跳过操作")
        window.back_to_guide_button.click()
        check(not window._retest_open and controller._assessment_phase == "during" and controller.assessment is None,
              "返回练习将后端阶段恢复during，过程不会混入后测")
        window.open_retest_button.click()
        second_post = controller.assessment.snapshot(time.monotonic()-controller.started)
        check(second_post["assessment_id"] != first_post["assessment_id"] and second_post["window_start"] >= first_post["window_start"],
              "再次复测创建新窗口，不复用旧观察")
        snapshot("review-post-window")
        window.skip_assessment_button.click()

        original_bundle = controller.store.save_activity_bundle
        write_calls = []
        def fail_once(*args):
            write_calls.append(args)
            if len(write_calls) == 1:
                if not release_write.wait(10):
                    raise TimeoutError("fixture release")
                raise OSError("injected temporary disk error")
            return original_bundle(*args)
        controller.store.save_activity_bundle = fail_once
        choose(window.post_combo, 4)
        choose(window.helpfulness_combo, "helpful")
        window.complete_button.click()
        check(controller.flow.current["save_status"] == "pending", "提交反馈先进入pending状态")
        check("正在保存" in window.outcome_card.heading.text() and not window.activity_result.text().startswith("已保存"),
              "写入未完成时UI不提前宣称已保存")
        check(window.retry_save_button.isHidden(), "写入进行中不显示失败重试入口")
        check(not window.activity_request_button.isEnabled() and not window.request_button.isEnabled(),
              "写入进行中禁用新活动入口，保留当前未保存反馈")
        check("保存成功后" in window.outcome_card.next_action.text(), "写入未完成不提前宣称反馈已用于推荐或待办")
        release_write.set()
        wait_for(lambda: controller.flow.current.get("save_status") == "failed", "injected failed write")
        check(window.retry_save_button.isVisible() and "尚未保存" in window.outcome_card.heading.text(),
              "真实失败回调接入结果卡与重试按钮")
        check(not window.activity_request_button.isEnabled() and window.activity_request_button.objectName() == "ghost"
              and window.retry_save_button.objectName() == "primary", "保存失败时重试为主操作，新活动禁用且降级")
        failed_id = controller.flow.current["id"]
        controller.request_intervention()
        check(controller.flow.state == "cooldown" and controller.flow.current["id"] == failed_id,
              "控制器也拒绝用新活动覆盖尚未保存的失败反馈")
        check(controller.store.get_intervention(session, controller.flow.current["id"])["status"] == "running",
              "失败事务未部分保存完成状态")
        snapshot("review-save-failure")
        window.retry_save_button.click()
        wait_for(lambda: controller.flow.current.get("save_status") == "saved", "retry activity write")
        check(len(write_calls) == 2 and window.retry_save_button.isHidden(), "重试按钮接到后端并在成功后隐藏")
        check(window.activity_request_button.isEnabled(), "成功保存后恢复主动新活动入口")
        check("已保存" in window.outcome_card.heading.text(), "仅成功写入后显示已保存")
        check(controller.flow.current["assessment_comparison"]["comparable"] is False
              and "不可比" in window.outcome_card.detail.text(), "跳过独立观察不生成模型改善结论")
        barrier()
        exported = controller.store.export_session(session, output/"review-ui-session.json")
        data = json.loads(exported.read_text(encoding="utf-8"))
        check(len(data["self_reports"]) == 2, "失败后重试只保留一条前测与一条后测")
        check(len(data["utterances"]) == 1 and all("禁止发布" not in item.get("text", "") for item in data["utterances"]),
              "撤回后的迟到文本未持久化，之前获授权的转写保留")
        window.view_feedback_button.click()
        wait_for(lambda: "独立活动样本 1 次" in window.personal_stats_label.text(), "personal independent count")
        check(window.feedback_table.horizontalHeaderItem(2).text() == "独立活动样本", "个人表格使用独立活动样本新接口")
        row = next(index for index in range(window.feedback_table.rowCount())
                   if window.feedback_table.item(index, 0).text() == "给想法命名")
        check(window.feedback_table.item(row, 2).text() == "1", "即时与延迟统计以独立活动为单位呈现")
        snapshot("review-independent-feedback")

        # Check missing source separately from an explicitly unknown source.
        # Both are fixtures in this temporary database/view, never real history.
        window.tabs.setCurrentIndex(2)
        settle()
        actual_history = controller.store.list_sessions()
        missing_source = {**actual_history[0], "id": "missing-source-ui-fixture"}
        missing_source.pop("mode", None)
        choose(window.history_filter, "all")
        window._on_history([missing_source])
        check(window.history_table.item(0, 2).text() == "来源未核实"
              and "来源未核实" in window.history_details.toPlainText(), "缺失mode时摘要与详情都保留来源未核实")
        choose(window.history_filter, "real")
        check(window.history_table.rowCount() == 0, "缺失来源记录不进入真实来源筛选")
        unknown_session = controller.store.start_session(controller.current_user_id, {}, "unknown")
        controller.store.finish_session(unknown_session)
        window._on_history(controller.store.list_sessions())
        choose(window.history_filter, "unknown")
        check(window.history_table.rowCount() == 1 and window._selected_session()["id"] == unknown_session,
              "来源未核实筛选显示未知来源真实存储记录")
        check(window.history_table.item(0, 2).text() == "来源未核实"
              and "不纳入真实反馈统计" in window.history_details.toPlainText(), "未知来源详情明确排除真实反馈统计")
        snapshot("review-unknown-source")
        choose(window.history_filter, "simulation")
        check(window.history_table.rowCount() == 1, "模拟筛选排除未知来源")
        choose(window.history_filter, "all")
        check(window.history_table.rowCount() == 2, "全部来源保留模拟与未核实记录供核对")

        class FailingStopRuntime:
            calls = 0
            def stop(self):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("injected device-release failure")
        runtime = FailingStopRuntime()
        controller.runtime = runtime
        statuses.clear()
        window.stop_button.click()
        wait_for(lambda: controller.state == "stop_failed", "stop failure callback")
        check("未完全成功" in window.session_badge.text() and window.stop_button.isEnabled(),
              "stop_failed显示失败状态并允许重新结束")
        check(not window.pause_button.isEnabled() and not window.user_name.isEnabled() and window.start_button.isHidden(),
              "停止失败时保持会话上下文，禁用新会话与暂停")
        check(not any("设备已释放" in item for item in statuses), "设备释放失败不显示成功消息")
        check(next(item for item in controller.store.list_sessions() if item["id"] == session)["ended_at"] is None,
              "释放失败不把会话伪记为已结束")
        snapshot("review-stop-failure")
        window.stop_button.click()
        wait_for(lambda: controller.state == "stopped", "retry session stop")
        check(runtime.calls == 2 and controller.runtime is None, "结束重试复用保留上下文并完成释放")
        check(bool(next(item for item in controller.store.list_sessions() if item["id"] == session)["ended_at"]),
              "成功重试后才保存会话结束时间")
        report["passed"] = True
        report["export"] = str(exported)
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        release_write.set()
        if controller and original_bundle:
            controller.store.save_activity_bundle = original_bundle
        if window:
            window.close()
        if controller:
            controller.shutdown()
        scratch.cleanup()
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        (output/"ui-contracts.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": len(report["checks"]),
                      "snapshots": len(report["snapshots"]), "report": str(output/"ui-contracts.json"),
                      "error": report.get("error")}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
