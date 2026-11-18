"""Device-free regression checks for prompt costs, shadow ranking and decision joins."""
import json
import os
import time
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication

from emotion_assistant.controller import Controller
from emotion_assistant.prompt_policy import ConflictGate, PromptTimingPolicy
from emotion_assistant.recommendation_policy import contextual_decision
from emotion_assistant.regulation import recommend_activities


def test_budget_refusal_and_availability():
    timing = PromptTimingPolicy()
    assert timing.evaluate(2000, None)["action"] == "observe"
    for mode in ("quiet", "focused"):
        timing.mode = mode
        assert timing.evaluate(2000, "clarify_conflict")["reason"] == mode
    timing.mode = "available"
    timing.restore([("asked", 100), ("asked", 200)])
    assert timing.evaluate(2000, "sustained_observation")["reason"] == "hourly_budget"
    assert timing.evaluate(3900, "sustained_observation")["action"] == "ask"
    timing.restore([("declined", 100), ("declined", 300)])
    assert timing.evaluate(600, "clarify_conflict")["reason"] == "cooldown"
    assert timing.evaluate(2000, "clarify_conflict")["reason"] == "recent_declines"


def joint(identity, start, end, **fields):
    return dict({"session_id": "s", "status": "joint", "valid": True, "conflict": True,
                 "audio_event_id": identity, "event_id": "fusion:" + identity,
                 "modality": "fusion", "t_start": start, "t_end": end, "visual_coverage": 1,
                 "per_modality_quality": {"visual": .8, "audio": .9}}, **fields)


def test_clarification_counts_independent_revisable_coverage():
    gate = ConflictGate()
    gate.observe(joint("a", 0, 12), 12)
    assert not gate.ready(15)
    gate.observe(joint("a", 0, 6), 12)
    gate.observe(joint("b", 6, 12, visual_coverage=.2), 12)
    assert not gate.ready(15)
    gate.observe(joint("b", 6, 12), 13)
    assert gate.ready(15)
    gate.observe(joint("b", 6, 12, conflict=False), 16)
    assert not gate.ready(16)


def test_shadow_estimates_match_context_and_leave_live_ranking_intact():
    history = []
    for i, fields in enumerate([{}, {"post": 6}, {"pre": 9}, {"post": None},
                                {"context": {"available_seconds": 600, "need": "tense"}},
                                {"context": {"available_seconds": 120, "need": "fatigue"}},
                                {"previous_intervention_id": "old"}, {"mode": "simulation"}]):
        history.append(dict({"id": str(i), "session_id": str(i), "user_id": "u", "activity": "break",
                             "mode": "real", "status": "completed", "pre": 5, "post": 3,
                             "context": {"available_seconds": 120, "need": "tense"}}, **fields))
    prefs = {"breathing": "avoid"}
    items, log = contextual_decision(history, "tense", 120, 5, prefs, "real", {"desk_reset"})
    expected = recommend_activities(history, "tense", 120, prefs, "real")
    assert [i["activity"] for i in items] == [i["activity"] for i in expected]
    outcome = log["shadow"]["estimates"]["break"]
    assert outcome["count"] == 2 and outcome["mean_delta"] == .5 and outcome["sample_sd"] > 2
    assert not {"breathing", "desk_reset"}.intersection(log["shadow"]["ranking"])
    assert log["selection_probability"] is None and log["user_selected_action"] is None
    _, empty = contextual_decision([], "tense", 120, None, {}, "real")
    assert empty["shadow"]["ranking"] == empty["ranking"]


def test_quiet_does_not_block_manual_choice_and_outcome_links(tmp_path):
    app = QApplication.instance() or QApplication([])
    controller = Controller(tmp_path / "sessions.sqlite3")
    try:
        controller.start_session("simulated", -1, None, {"collection": True}, "simulation")
        controller._simulation_timer.stop()
        controller.set_prompt_availability("quiet")
        controller.started = time.monotonic() - 15
        for record in (joint("a", 0, 6), joint("b", 6, 12)):
            record["session_id"] = controller.session_id
            controller._consider_prompt(record)
        assert controller.flow.state == "observing"
        controller.request_intervention()
        controller.confirm_intervention(True)
        controller.set_recommendation_pre(5)
        payload = controller.refresh_recommendations(need="tense").result(timeout=5)
        app.processEvents()
        controller.start_activity("grounding", 5, 120)
        controller.complete_activity(4, "helpful")
        controller.writer.submit(lambda: None).result(timeout=5)
        session = json.loads(controller.store.export_session(controller.session_id, tmp_path / "export.json").read_text(encoding="utf-8"))
        decision = session["interventions"][0]["decision_log"]
        assert decision["ranking"] == payload["decision"]["ranking"]
        assert decision["user_selected_action"] == "grounding"
        assert decision["outcome_id"] == session["interventions"][0]["id"]
        offers = {e["payload"]["decision_id"] for e in session["events"] if e["kind"] == "recommendation_offered"}
        assert decision["offered_decision_id"] in offers
        assert any(e["kind"] == "prompt_decision" and e["payload"]["reason"] == "quiet" for e in session["events"])
        assert not any(e["kind"] == "prompt_asked" for e in session["events"])
    finally:
        controller.shutdown()


def test_preference_is_respected_while_writer_is_busy():
    app = QApplication.instance() or QApplication([])
    controller = Controller(":memory:")
    release = threading.Event()
    try:
        controller.start_session("test", -1, None, {"collection": True}, "simulation")
        controller._simulation_timer.stop()
        controller.request_intervention()
        controller.confirm_intervention(True)
        controller.writer.submit(release.wait, 5)
        avoid = controller.set_activity_preference("break", "avoid")
        controller.start_activity("break", 5, 120)
        assert controller.flow.state == "ready" and not avoid.done()
        release.set()
        avoid.result(timeout=5)
        controller.set_activity_preference("break", "neutral").result(timeout=5)
        controller.start_activity("break", 5, 120)
        assert controller.flow.state == "running"
    finally:
        release.set()
        controller.shutdown()
