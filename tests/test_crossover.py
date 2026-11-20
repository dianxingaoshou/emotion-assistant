"""Participant-period randomization, deployed policies, denominators and missingness."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication

from emotion_assistant.controller import Controller
from emotion_assistant.crossover import analyze, trial_order
from emotion_assistant.storage import Store


OPT = {"collection": True, "crossover_trial": True}
ROOT = Path(__file__).resolve().parents[1]


def test_assignment_persisted_withdrawal_erases_links_and_ratings_only(tmp_path, monkeypatch):
    monkeypatch.setattr("emotion_assistant.storage.secrets.choice", lambda seq: "BA")
    db = tmp_path / "visits.db"
    store = Store(db)
    uid = store.create_user("not-for-export")
    s1 = store.start_session(uid, OPT, "real")
    ordinary = store.start_session(uid, {"collection": True}, "real")
    assert not store.trial_assignment(ordinary)
    assert store.trial_assignment(s1)["condition"] == "adaptive"
    store.append_event(s1, "prompt_asked", {"decision_id": "real-offer"})
    assert not store.rate_trial_prompt(s1, "not-asked", False)
    assert store.rate_trial_prompt(s1, "real-offer", False)
    assert not store.rate_trial_prompt(s1, "real-offer", True)
    store.finish_session(s1, 1200)
    store.close()
    store = Store(db)
    try:
        s2 = store.start_session(uid, OPT, "real")
        sim = store.start_session(uid, OPT, "simulation")
        assert (
            store.trial_assignment(s2)["condition"] == "basic"
            and store.trial_assignment(sim)["period"] == 1
        )
        with pytest.raises(ValueError):
            store.start_session(uid, OPT, "real")
        assert len(store.list_sessions()) == 4
        assert store.withdraw_trial(uid, "real")
        assert store.trial_status(uid, "real")["status"] == "withdrawn"
        assert store.trial_periods("real") == [] and store.trial_assignment(sim)
        assert not store.rate_trial_prompt(s1, "real-offer", False)
        assert len(store.list_sessions()) == 4
        result = store.analyze_trial("real")
        assert (result["randomized"], result["withdrawn"], result["participants"]) == (1, 1, 0)
        assert "not-for-export" not in json.dumps(result) and uid not in json.dumps(result)
        assert store._db.execute("SELECT count(*) FROM trial_ratings").fetchone()[0] == 0
    finally:
        store.close()


def test_interrupted_and_deleted_periods_are_missing_not_reallocated(tmp_path):
    database = tmp_path / "crash.db"
    st = Store(database)
    u = st.create_user("test")
    st.start_session(u, OPT, "real")
    st.close()
    st = Store(database)
    try:
        assert st.trial_status(u, "real")["incomplete_periods"] == 1
        sid = st.start_session(u, OPT, "real")
        assert st.trial_assignment(sid)["period"] == 2
        st.finish_session(sid, 120)
        assert st.trial_status(u, "real")["status"] == "incomplete"
        r = st.analyze_trial("real")
        assert (r["periods_assigned"], r["periods_completed"], r["paired_participants"]) == (2, 1, 0)
        st.delete_session(sid)
        assert st.analyze_trial("real")["periods_incomplete"] == 2
        with pytest.raises(ValueError):
            st.start_session(u, OPT, "real")
    finally:
        st.close()


def fixture(uid, arm, wanted=True, completed=True, post=2, followup="completed", observed=3600):
    seq = "AB" if uid == "p" else "BA"
    return {
        "mode": "real",
        "user_id": uid,
        "sequence": seq,
        "condition": arm,
        "period": seq.index("A" if arm == "basic" else "B") + 1,
        "active_seconds": 3600.0,
        "observed_seconds": observed,
        "ratings": {} if wanted is None else {"d": wanted},
        "events": [
            ("prompt_decision", {"decision_id": "d"}),
            ("prompt_asked", {"decision_id": "d"}),
            ("intervention_confirmation", {"decision_id": "d", "accepted": True}),
            ("intervention_confirmation", {"decision_id": "manual", "accepted": False}),
        ],
        "activities": [
            {
                "activity": "break",
                "status": "completed" if completed else "cancelled",
                "pre": 5,
                "post": post,
                "followup_status": followup,
                "followup_distress": 3 if followup == "completed" else None,
            }
        ],
    }


def test_failed_capture_consumes_period_without_making_a_completed_pair(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])

    class FailedCapture:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("test device startup failed")

        def stop(self):
            pass

    monkeypatch.setattr("emotion_assistant.controller.CaptureRuntime", FailedCapture)
    ctrl = Controller(tmp_path / "failed.db")
    try:
        for _ in range(2):
            ctrl.start_session("P", 0, None, OPT, "real")
            deadline = time.monotonic() + 5
            while ctrl.state != "stopped" and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(.005)
            assert ctrl.state == "stopped"
        report = ctrl.trial_results("real")
        assert report["periods_assigned"] == report["periods_incomplete"] == 2
        assert report["periods_completed"] == report["paired_participants"] == 0
        assert ctrl.trial_status("P", "real")["status"] == "incomplete"
        assert all(s["ended_at"] for s in ctrl.store.list_sessions())
    finally:
        ctrl.shutdown()


def test_close_excludes_worker_teardown_from_trial_exposure(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    clock = [100.]
    monkeypatch.setattr("emotion_assistant.controller.time", SimpleNamespace(monotonic=lambda: clock[0]))

    class SlowStop:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            clock[0] += 30.

    db = tmp_path / "teardown.db"
    monkeypatch.setattr("emotion_assistant.controller.CaptureRuntime", SlowStop)
    ctrl = Controller(db)
    try:
        ctrl.start_session("P", 0, None, OPT, "real")
        assert ctrl.state == "running"
        clock[0] += 120.
    finally:
        ctrl.shutdown()
    report = Store.read_trial_report(db, "real")
    assert report["periods_completed"] == 1
    assert sum(v["active_hours"] for v in report["by_condition"].values()) == pytest.approx(120. / 3600)


def test_paired_and_weighted_metrics_zero_versus_missing():
    rows = [
        fixture("p", "basic", False, post=4),
        fixture("p", "adaptive", True, post=2),
        fixture("z", "basic", False, post=4, followup="skipped"),
        fixture("z", "adaptive", False, post=None, followup="pending"),
        fixture("single", "basic", None, completed=False),
        dict(fixture("fake", "adaptive", False), mode="simulation"),
    ]
    report = analyze(rows)
    a, b = (report["by_condition"][x] for x in ("basic", "adaptive"))
    assert report["participants"] == 3 and report["paired_participants"] == 2
    assert a["prompts_asked"] == 3 and a["prompt_accepted"] == 3 and a["unwanted_per_hour"] is None
    assert a["unwanted_per_hour_bounds"] == pytest.approx([2 / 3, 1])
    assert b["unwanted_per_hour"] == 0.5
    assert a["completion_rate"] == pytest.approx(2 / 3) and a["followup_skipped"] == 1
    assert a["post_missing"] == 0 and a["post_missing_all_starts"] == 1
    assert b["post_missing"] == 1 and b["followup_missing"] == 1 and b["followup_pending"] == 1
    assert report["paired"]["unwanted_per_hour"] == {"n": 2, "mean_difference": -0.5, "ci95": [-1.0, 0.0]}
    assert report["paired"]["immediate_mean_delta"] == {"n": 1, "mean_difference": 2.0, "ci95": None}
    sparse = analyze([fixture("p", "basic", wanted=True, observed=1799)])
    arm = sparse["by_condition"]["basic"]
    assert arm["unwanted_per_hour"] is None and arm["burden_eligible_periods"] == 0


def test_readonly_cli_with_repeated_modality_coverage(tmp_path):
    path = tmp_path / "visits.db"
    store = Store(path)
    u = store.create_user("not-for-export")
    s = store.start_session(u, OPT, "real")
    arm = store.trial_assignment(s)["condition"]
    for sensor in ("visual", "audio"):
        store.save_observation(
            s, dict(event_id=sensor, modality=sensor, valid=True, quality=0.8, t_start=0, t_end=30)
        )
    store.finish_session(s, 70)
    store.close()
    out = tmp_path / "result.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/analyze_trial.py"),
            "--database",
            str(path),
            "--out",
            str(out),
        ],
        capture_output=True,
        check=True,
    )
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["by_condition"][arm]["observed_hours"] == pytest.approx(30 / 3600)
    assert result["by_condition"][arm]["unwanted_per_hour"] is None
    assert "not-for-export" not in out.read_text(encoding="utf-8")


def test_actual_adaptive_order_only_and_fidelity():
    def choice(key, rank, blocked=False):
        return {"activity": key, "rank": rank, "blocked": blocked}

    decision = {"ranking": ["break", "grounding"], "shadow": {"ranking": ["grounding", "break"]}}
    items = [choice("break", 1), choice("grounding", 2), choice("breathing", 3, True)]
    for arm in (None, "basic"):
        unmodified, log = trial_order(items, decision, arm)
        assert [i["activity"] for i in unmodified][:2] == log["ranking"] == ["break", "grounding"]
    ordered, log = trial_order(items, decision, "adaptive")
    assert [(i["activity"], i["rank"]) for i in ordered] == [
        ("grounding", 1),
        ("break", 2),
        ("breathing", 3),
    ]
    assert log["ranking"] == ["grounding", "break"] and log["baseline_ranking"] == ["break", "grounding"]


@pytest.mark.parametrize("sequence,expected", [("AB", "observing"), ("BA", "confirming")])
def test_controller_enforces_arm_on_conflict_and_labels_offer(tmp_path, monkeypatch, sequence, expected):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr("emotion_assistant.storage.secrets.choice", lambda seq: sequence)
    controller = Controller(tmp_path / "trial.db")
    try:
        controller.start_session("practice", -1, None, OPT, "simulation")
        controller._simulation_timer.stop()
        # The common gate accepts these two distinct quality-qualified conflicting segments.
        for ident, lo, hi in (("a", 0, 6), ("b", 6, 12)):
            controller.conflict_gate.observe(
                {
                    "session_id": controller.session_id,
                    "status": "joint",
                    "valid": True,
                    "conflict": True,
                    "event_id": "fusion:" + ident,
                    "audio_event_id": ident,
                    "t_start": lo,
                    "t_end": hi,
                    "visual_coverage": 1,
                    "per_modality_quality": {"audio": 0.9, "visual": 0.9},
                },
                15,
            )
        controller.set_prompt_availability(
            "available"
        )  # resets gates; inject equivalent ready diagnostics below
        diag = {
            "ready": True,
            "source_ids": ["a", "b"],
            "effective_seconds": 12.0,
            "window_start": 0.0,
            "window_end": 12.0,
        }
        monkeypatch.setattr(controller.conflict_gate, "diagnostics", lambda now: diag)
        controller.prompt_timing.mode = "quiet"
        controller._evaluate_prompt(15)
        assert controller.flow.state == "observing"
        controller.prompt_timing.mode = "available"
        controller._evaluate_prompt(15)
        assert controller.flow.state == expected
        if expected == "confirming":
            assert controller.rate_trial_prompt(False) and not controller.rate_trial_prompt(True)
            # A person can dislike the timing AND accept; don't infer one from the other.
            controller.confirm_intervention(True)
        else:
            assert not controller.rate_trial_prompt(False)
        controller.writer.submit(lambda: None).result(timeout=5)
        assert controller.store.trial_periods("simulation")[0]["ratings"] == (
            {controller._request_decision_id: False} if expected == "confirming" else {}
        )
    finally:
        controller.shutdown()
