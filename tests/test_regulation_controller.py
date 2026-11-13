"""Meaningful closed-loop integration checks; every session is an explicit fixture."""
from datetime import datetime, timedelta, timezone
import json
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import pytest
from PySide6.QtWidgets import QApplication

from emotion_assistant.controller import Controller
from emotion_assistant.core import InterventionFlow
from emotion_assistant.regulation import ACTIVITIES


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def controller(app):
    value = Controller(":memory:")
    value.start_session("闭环测试档案", -1, None, {"collection": True}, "simulation")
    value._simulation_timer.stop()
    yield value
    value.shutdown()


def flush(controller):
    controller.writer.submit(lambda: None).result(timeout=5)


def refresh(controller, app, **kwargs):
    flush(controller)
    payload = controller.refresh_recommendations(**kwargs).result(timeout=5)
    app.processEvents()
    return payload


def start(controller, activity="break", pre=6, **kwargs):
    controller.request_intervention()
    controller.confirm_intervention(True)
    controller.start_activity(activity, pre, 120, **kwargs)
    return controller.flow.current


def exported(controller, tmp_path):
    flush(controller)
    path = controller.store.export_session(controller.session_id, tmp_path / "loop.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_eight_activities_save_guidance_context_and_user_feedback(controller, tmp_path):
    for activity in ACTIVITIES:
        current = start(controller, activity, context={"need": "unsure", "available_seconds": 120})
        assert current["activity"] == activity and len(current["steps"]) >= 3
        assert current["context"]["need"] == "unsure"
        assert current["recommendation"]["reasons"]
        controller.advance_activity_step()
        controller.activity_checkin("comfortable")
        controller.complete_activity(4, "helpful", next_step="回到一个小任务")
    result = exported(controller, tmp_path)
    assert len(result["interventions"]) == 8
    assert len(result["self_reports"]) == 16
    assert all(record["step_index"] == 1 and record["checkins"][0]["feeling"] == "comfortable"
               for record in result["interventions"])
    assert all(record["loop_result"]["state"] == "improved" and record["next_step"] for record in result["interventions"])
    assert all(record["followup"] is None for record in result["interventions"])


def test_uncomfortable_immediately_stops_and_is_not_an_improvement(controller, app, tmp_path):
    current = start(controller, "breathing")
    controller.activity_checkin("uncomfortable")
    assert controller.flow.state == "cooldown"
    result = exported(controller, tmp_path)
    stopped = result["interventions"][0]
    assert stopped["id"] == current["id"] and stopped["status"] == "cancelled"
    assert stopped["stop_reason"] == "uncomfortable" and stopped["discomfort"]
    assert stopped["post"] is None and stopped["delta"] is None
    assert stopped["loop_result"]["state"] == "discomfort"
    assert [item["phase"] for item in result["self_reports"]] == ["pre"]
    recommendations = refresh(controller, app)["items"]
    assert next(item for item in recommendations if item["activity"] == "breathing")["blocked"]
    controller.request_intervention()
    controller.confirm_intervention(True)
    controller.start_activity("breathing", 5, 120)
    assert controller.flow.state == "ready"  # It cannot silently start again.


def test_no_change_and_explicit_switch_preserve_the_chain(controller, tmp_path):
    first = start(controller, "grounding")
    controller.activity_checkin("no_change")
    assert controller.flow.state == "running" and controller.flow.current["last_checkin"]["feeling"] == "no_change"
    controller.request_alternative()
    assert controller.flow.state == "confirming"
    controller.confirm_intervention(True)
    controller.start_activity("task_split", 6, 120)
    second = controller.flow.current
    assert second["previous_intervention_id"] == first["id"]
    assert second["root_intervention_id"] == first["id"]
    controller.complete_activity(5, "no_change")
    records = exported(controller, tmp_path)["interventions"]
    original = next(item for item in records if item["id"] == first["id"])
    assert original["stop_reason"] == "manual_switch" and original["evaluation"] == "insufficient"
    assert original["last_checkin"]["feeling"] == "no_change"


def test_adverse_subjective_feedback_wins_over_numeric_drop(controller, tmp_path):
    start(controller, "desk_reset", pre=7)
    controller.complete_activity(3, "worse")
    record = exported(controller, tmp_path)["interventions"][0]
    assert record["delta"] == 4  # Preserve the raw arithmetic without reinterpreting it.
    assert record["loop_result"]["state"] == "discomfort" and record["discomfort"]


def test_missing_post_keeps_unknown_result_and_optional_preference_unchanged(controller, tmp_path):
    controller.store.save_activity_preference(controller.current_user_id, "break", "prefer")
    start(controller)
    controller.complete_activity(None, "unknown")
    record = exported(controller, tmp_path)["interventions"][0]
    assert record["loop_result"]["state"] == "insufficient"
    assert record["delta"] is None and record["followup"] is None
    assert controller.store.get_activity_preferences(controller.current_user_id)["break"] == "prefer"


def test_followup_is_opt_in_and_same_activity_is_not_counted_twice(controller, app, tmp_path):
    current = start(controller)
    controller.complete_activity(4, "helpful", followup_seconds=120)
    first = refresh(controller, app)
    assert len(first["pending_followups"]) == 1
    pending = first["pending_followups"][0]
    assert pending["id"] == current["id"]
    assert not pending.get("is_due", pending.get("due", False))
    first_submit = controller.submit_followup(controller.session_id, current["id"], 3, "helpful")
    second_submit = controller.submit_followup(controller.session_id, current["id"], 2, "helpful")
    first_submit.result(timeout=5)
    assert second_submit.result(timeout=5) is None
    record = exported(controller, tmp_path)
    assert len(record["interventions"]) == 1
    assert [item["phase"] for item in record["self_reports"]].count("followup") == 1
    followup = record["interventions"][0]["followup"]
    assert followup["status"] == "completed" and followup["distress"] == 3
    assert followup["delta"] == 3 and followup["delta_from_immediate"] == 1
    assert followup["answered_early"] is True
    app.processEvents()  # Worker commits; controller state applies on its Qt thread.
    assert controller.flow.current["followup"]["status"] == "completed"
    payload = refresh(controller, app)
    assert payload["pending_followups"] == []
    assert payload["stats"]["evidence_count"] == 1 and payload["stats"]["followup_count"] == 1


def test_missing_followup_rating_is_skipped_without_fake_zero(controller, tmp_path):
    current = start(controller)
    controller.complete_activity(4, "helpful", followup_seconds=120)
    flush(controller)
    controller.submit_followup(controller.session_id, current["id"], None, "unknown").result(timeout=5)
    result = exported(controller, tmp_path)
    followup = result["interventions"][0]["followup"]
    assert followup["status"] == "skipped" and followup["distress"] is None
    assert followup["delta"] is None and followup["loop_result"]["state"] == "insufficient"
    assert len(result["self_reports"]) == 2


def test_followup_restores_after_restart_and_stays_in_its_mode(app, tmp_path):
    database = tmp_path / "persistent.sqlite3"
    first = Controller(database)
    first.start_session("持久档案", -1, None, {"collection": True}, "simulation")
    first._simulation_timer.stop()
    current = start(first, "thought_label")
    sid = first.session_id
    first.complete_activity(4, "helpful", followup_seconds=120)
    first.shutdown()
    restored = Controller(database)
    try:
        simulated = refresh(restored, app, user_name="持久档案", mode="simulation")
        assert simulated["pending_followups"][0]["id"] == current["id"]
        assert simulated["pending_followups"][0]["session_id"] == sid
        real = refresh(restored, app, user_name="持久档案", mode="real")
        assert real["pending_followups"] == [] and real["stats"]["evidence_count"] == 0
        wrong_mode = restored.submit_followup(sid, current["id"], 3, "helpful")
        assert wrong_mode.result(timeout=5) is None
        stored = restored.store.get_intervention(sid, current["id"])
        assert stored["followup"]["status"] == "pending"
    finally:
        restored.shutdown()


def test_followup_cannot_be_written_under_another_profile(controller, app, tmp_path):
    current = start(controller)
    controller.complete_activity(4, "helpful", followup_seconds=120)
    flush(controller)
    owner = controller.current_user_id
    other = controller.store.create_user("另一个档案")
    controller.state = "stopped"  # No runtime exists in this explicit simulation.
    refresh(controller, app, user_name="另一个档案")
    assert controller.current_user_id == other
    assert controller.submit_followup(controller.session_id, current["id"], 2, "helpful").result(timeout=5) is None
    record = controller.store.get_intervention(controller.session_id, current["id"])
    assert record["user_id"] == owner and record["followup"]["status"] == "pending"


def test_preference_avoid_blocks_until_explicitly_changed(controller, app):
    controller.set_activity_preference("break", "avoid").result(timeout=5)
    payload = refresh(controller, app)
    assert next(item for item in payload["items"] if item["activity"] == "break")["blocked"]
    controller.request_intervention()
    controller.confirm_intervention(True)
    controller.start_activity("break", 6, 120)
    assert controller.flow.state == "ready"
    controller.set_activity_preference("break", "neutral").result(timeout=5)
    refresh(controller, app)
    controller.start_activity("break", 6, 120)
    assert controller.flow.state == "running"


def test_stale_recommendation_results_cannot_replace_current_profile(controller, app):
    payload = refresh(controller, app)
    events = []
    controller.recommendations_ready.connect(events.append)
    stale = dict(payload, _serial=controller._recommendation_serial - 1)
    controller._publish_recommendations(stale)
    foreign = dict(payload, _serial=controller._recommendation_serial, user_id="different-profile")
    controller._publish_recommendations(foreign)
    assert events == []


def test_due_boundary_refreshes_once_instead_of_repeated_reminders(controller, monkeypatch):
    import emotion_assistant.controller as module
    now = datetime.now(timezone.utc)
    pending = [{"session_id": controller.session_id, "id": "pending-fixture", "followup": {
        "status": "pending", "due_at": (now + timedelta(seconds=1)).isoformat()}}]
    controller.personalization = {"pending_followups": pending}
    controller._last_due_signature = controller._due_signature(pending)
    class FutureClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now + timedelta(seconds=2)
    monkeypatch.setattr(module, "datetime", FutureClock)
    calls = []
    monkeypatch.setattr(controller, "refresh_recommendations", lambda *a, **k: calls.append(True))
    controller._activity_tick()
    controller._activity_tick()
    assert len(calls) == 1


def test_step_cannot_advance_after_stop_and_metadata_cannot_replace_identity():
    flow = InterventionFlow()
    flow.request()
    flow.confirm(True)
    flow.start("break", 5, 60)
    for _ in range(10):
        flow.advance_step(3)
    assert flow.current["step_index"] == 2
    with pytest.raises(ValueError):
        flow.annotate({"id": "changed"})
    flow.cancel()
    with pytest.raises(ValueError):
        flow.advance_step(3)


def test_native_cancel_button_bool_cannot_become_a_stop_reason(controller):
    from PySide6.QtWidgets import QPushButton
    start(controller)
    button = QPushButton()
    button.clicked.connect(controller.cancel_activity)
    button.click()
    assert controller.flow.current["stop_reason"] == "user_cancelled"
