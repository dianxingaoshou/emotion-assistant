"""Repeatable offline acceptance checks; all example data are scripted simulation.

Run with .venv/Scripts/python.exe scripts/verify.py. This runner never starts a
camera, microphone, ASR or emotion model. A passed result verifies implementation
and a synthetic closed loop; it is not a real-session or model-accuracy result.
"""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import ExitStack
import json
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT))


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def as_text(value):
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""


def run_command(command, output_path, timeout):
    started = time.monotonic()
    result = {"command": command, "passed": False, "returncode": None}
    try:
        process = subprocess.run(command, cwd=ROOT, env=os.environ.copy(),
                                 capture_output=True, text=True, encoding="utf-8",
                                 errors="replace", timeout=timeout, check=False)
        result.update(returncode=process.returncode, stdout=process.stdout, stderr=process.stderr,
                      passed=process.returncode == 0)
    except subprocess.TimeoutExpired as exc:
        result.update(stdout=as_text(exc.stdout), stderr=as_text(exc.stderr),
                      error=f"acceptance subprocess exceeded {timeout} seconds")
    except Exception:
        result.update(stdout="", stderr="", error=traceback.format_exc())
    result["runner_elapsed_seconds"] = round(time.monotonic() - started, 3)
    output_path.write_text(result.get("stdout", "") + "\n" + result.get("stderr", "")
                           + "\n" + result.get("error", ""), encoding="utf-8")
    result["output_file"] = str(output_path)
    return result


def junit_counts(path):
    root = ET.parse(path).getroot()
    suites = [node for node in root.iter("testsuite") if not node.findall("testsuite")]
    return {key: sum(int(node.get(key, 0)) for node in suites)
            for key in ("tests", "failures", "errors", "skipped")}


def simulated_closed_loop():
    # Configure before importing Qt or any project code. No real model is loaded.
    from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer
    import emotion_assistant.controller as controller_module

    app = QCoreApplication.instance() or QCoreApplication(["offline-acceptance"])
    app.setApplicationName("Emotion assistant offline acceptance")
    result = {"passed": False, "mode": "simulation", "hardware_opened": False,
              "model_inference_performed": False, "accelerated_replay": True,
              "fixture_span_seconds": 30.0, "scripted_ratings": {"pre": 7, "post": 4},
              "ratings_are_test_inputs": True,
              "interpretation": "Scripted workflow verification only; no measured emotional improvement or accuracy."}
    statuses = []
    states = []
    controller = None
    original_runtime = controller_module.CaptureRuntime

    class ForbiddenHardwareRuntime:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("acceptance simulation attempted to initialize hardware")

    controller_module.CaptureRuntime = ForbiddenHardwareRuntime
    try:
        # LIFO cleanup closes SQLite/workers before TemporaryDirectory removes
        # files, including when an acceptance assertion fails on Windows.
        with ExitStack() as cleanup:
            scratch = cleanup.enter_context(tempfile.TemporaryDirectory(prefix="acceptance-", dir=REPORTS))
            database_path = Path(scratch) / "simulation.sqlite3"
            controller = controller_module.Controller(database=database_path)
            cleanup.callback(controller.shutdown)
            controller.status_changed.connect(statuses.append)
            controller.session_changed.connect(lambda event: states.append(event["state"]))
            controller.start_session("自动验收模拟档案（非真实用户）", -1, None,
                                     {"collection": True, "transcript": False}, "simulation")
            if controller.state != "running" or controller.runtime is not None:
                raise RuntimeError("simulation session did not start in isolated mode")
            controller._simulation_timer.stop()
            controller._resource_timer.stop()

            # These observations replay a 30-second synthetic timeline immediately.
            # Moving the controller's logical origin keeps its freshness checks
            # meaningful during accelerated replay. UTC wall duration is separate.
            controller.started = time.monotonic() - 30
            session_id = controller.session_id
            controller._write(controller.store.append_event, session_id, "acceptance_fixture", {
                "simulation": True, "accelerated_replay": True,
                "ratings_are_scripted_test_inputs": True, "fixture_span_seconds": 30,
                "scripted_ratings": {"pre": 7, "post": 4},
                "note": "模拟回放与预设自评；不是实际采集、模型推理或调节效果证据。"})
            for index in range(60):
                start, end = index * .5, (index + 1) * .5
                visual = controller_module.simulation_observation(
                    session_id, "visual", start, end, negative=.55)
                if 20 <= index < 24:
                    visual.update(valid=False, scores=None, quality=0,
                                  reason="模拟摄像头遮挡缺口，非实测")
                controller._handle_event({"kind": "observation", "observation": visual})
                if (index + 1) % 6 == 0:
                    audio = controller_module.simulation_observation(
                        session_id, "audio", end - 3, end, negative=.55)
                    controller._handle_event({"kind": "observation", "observation": audio})
            controller.request_intervention(source="scripted_acceptance_user")
            controller.confirm_intervention(True)
            controller.start_activity("task_split", 7, 1, {
                "blocker": "模拟测试：下一步尚未拆分", "tried": "模拟测试：检查输入",
                "next_step": "模拟测试：列出一个可执行的小步骤", "simulation": True})
            if controller.flow.state != "running":
                raise RuntimeError("confirmed scripted activity did not enter running state")

            loop = QEventLoop()
            poll = QTimer()
            poll.setInterval(25)
            watchdog = QTimer()
            watchdog.setSingleShot(True)
            failure = []
            stop_started = [None]

            def abort(exc):
                failure.append(f"{type(exc).__name__}: {exc}")
                loop.quit()

            def complete_scripted_activity():
                try:
                    end = time.monotonic() - controller.started
                    for modality in ("visual", "audio"):
                        controller._handle_event({"kind": "observation", "observation":
                            controller_module.simulation_observation(
                                session_id, modality, end - .5, end, negative=.55)})
                    controller.complete_activity(4, "helpful",
                        feedback_note="模拟预设：有帮助；仅用于流程测试，不是实际效果反馈。")
                    if controller.flow.state != "cooldown":
                        raise RuntimeError("scripted post-rating was not completed")
                    stop_started[0] = time.monotonic()
                    controller.stop_session()
                    poll.start()
                except Exception as exc:
                    abort(exc)

            def check_stopped():
                if controller.state == "stopped":
                    loop.quit()
                elif stop_started[0] is not None and time.monotonic() - stop_started[0] > 10:
                    abort(TimeoutError("controller did not stop within 10 seconds"))

            poll.timeout.connect(check_stopped)
            watchdog.timeout.connect(lambda: abort(TimeoutError("scripted acceptance loop exceeded 15 seconds")))
            QTimer.singleShot(1100, complete_scripted_activity)
            watchdog.start(15000)
            loop.exec()
            poll.stop()
            watchdog.stop()
            if failure:
                raise RuntimeError("; ".join(failure))
            if controller.state != "stopped":
                raise RuntimeError("controller did not finish its session")
            controller._write(lambda: None).result(timeout=10)
            export_path = controller.store.export_session(session_id, REPORTS / "example-session.json")
            exported = json.loads(export_path.read_text(encoding="utf-8"))
            completed = [item for item in exported["interventions"] if item.get("status") == "completed"]
            checks = {
                "simulation_session": exported["session"]["mode"] == "simulation",
                "utc_session_finished": bool(exported["session"]["ended_at"]),
                "observations_include_fusion": any(item.get("status") == "joint" for item in exported["observations"]),
                "all_observations_marked_simulation": all(item.get("simulation") is True for item in exported["observations"]),
                "one_completed_activity": len(completed) == 1,
                "two_self_reports": len(exported["self_reports"]) == 2,
                "scripted_pre_post_saved": bool(completed and completed[0].get("pre") == 7
                                                 and completed[0].get("post") == 4
                                                 and completed[0].get("delta") == 3),
                "transcript_not_saved_without_consent": not exported["utterances"],
                "camera_gap_not_filled": exported["coverage"]["visual_seconds"] < exported["coverage"]["audio_seconds"],
                "no_storage_failure": not any(message.startswith("本地存储失败") for message in statuses),
            }
            result.update(checks=checks, session_id=session_id, session_states=states,
                          counts={key: len(exported[key]) for key in
                                  ("observations", "utterances", "self_reports", "interventions", "events")},
                          coverage=exported["coverage"], export=str(export_path))
            if not all(checks.values()):
                raise RuntimeError("closed-loop checks failed: " + ", ".join(key for key, value in checks.items() if not value))
            controller.shutdown()
            controller = None
            final_database = REPORTS / "example-session.sqlite3"
            shutil.copyfile(database_path, final_database)
            connection = sqlite3.connect(final_database)
            try:
                if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise RuntimeError("exported simulation database failed integrity check")
                if connection.execute("SELECT count(*) FROM sessions").fetchone()[0] != 1:
                    raise RuntimeError("simulation database is not isolated to one acceptance session")
            finally:
                connection.close()
            result.update(passed=True, database=str(final_database),
                          storage_isolated_to_reports=True, status_messages=statuses)
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        if controller is not None:
            try:
                controller.shutdown()
            except Exception:
                result["cleanup_error"] = traceback.format_exc()
                result["passed"] = False
        controller_module.CaptureRuntime = original_runtime
    return result


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "created_at_utc": utc_now(), "passed": False,
              "python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "isolated": True,
              "offline": True, "qt_platform": "offscreen",
              "hardware_session_tested": False,
              "scope": "Automated software checks and scripted simulation only; hardware, model accuracy and intervention efficacy remain untested."}
    expected_prefix = (ROOT / ".venv").resolve()
    if Path(sys.prefix).resolve() != expected_prefix or sys.prefix == sys.base_prefix:
        report["error"] = f"Project virtual environment required: {expected_prefix / 'Scripts' / 'python.exe'} scripts/verify.py"
    elif sys.version_info[:2] != (3, 12):
        report["error"] = "The locked project environment requires Python 3.12."
    else:
        try:
            from emotion_assistant.paths import configure_environment
            configure_environment()
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
            os.environ["PYTHONIOENCODING"] = "utf-8"
            test_path = REPORTS / "tests.xml"
            # A failed pytest startup must not inherit a previous run's counts.
            test_path.unlink(missing_ok=True)
            report["tests"] = run_command(
                [sys.executable, "-m", "pytest", "-q", f"--junitxml={test_path}"],
                REPORTS / "tests-output.txt", timeout=180)
            if test_path.exists():
                try:
                    report["tests"]["junit"] = junit_counts(test_path)
                except Exception:
                    report["tests"].update(passed=False, junit_error=traceback.format_exc())
            elif report["tests"]["passed"]:
                report["tests"].update(passed=False, error="pytest did not produce its JUnit report")
            report["ui_smoke"] = run_command(
                [sys.executable, "-m", "emotion_assistant", "--smoke"],
                REPORTS / "ui-smoke-output.txt", timeout=30)
            report["closed_loop"] = simulated_closed_loop()
            report["passed"] = all(report[key]["passed"] for key in ("tests", "ui_smoke", "closed_loop"))
        except Exception:
            report["error"] = traceback.format_exc()
    report["finished_at_utc"] = utc_now()
    destination = REPORTS / "acceptance.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(("PASS" if report["passed"] else "FAIL") + f": {destination}")
    if not report["passed"]:
        print(report.get("error", "Inspect acceptance.json and its step output files for failures."))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
