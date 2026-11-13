from datetime import datetime, timedelta, timezone
import json

import pytest

from emotion_assistant.storage import Store


@pytest.fixture
def store(tmp_path):
    database = Store(tmp_path / "feedback.sqlite3")
    yield database
    database.close()


def profiles(store):
    one, two = store.create_user("甲"), store.create_user("乙")
    real = store.start_session(one, {"save_transcript": False}, "real")
    simulation = store.start_session(one, {"save_transcript": False}, "simulation")
    other = store.start_session(two, {"save_transcript": False}, "real")
    return one, two, real, simulation, other


def activity(identity="one", **extra):
    return {"id": identity, "activity": "breathing", "status": "completed",
            "pre": 7, "post": 4, "helpfulness": "helpful", **extra}


def test_filtered_history_is_profile_and_mode_scoped_with_authoritative_identity(store):
    one, two, real, simulation, other = profiles(store)
    store.save_intervention(real, activity(mode="simulation", user_id=two))
    store.save_intervention(simulation, activity())
    store.save_intervention(other, activity())
    records = store.list_interventions(one, "real")
    assert len(records) == 1
    assert records[0]["mode"] == "real" and records[0]["user_id"] == one
    assert store.get_intervention(real, "one")["session_id"] == real
    assert len(store.list_interventions(one, "simulation")) == 1


def test_atomic_patch_preserves_activity_and_transcript_consent(store):
    _, _, real, _, _ = profiles(store)
    store.save_intervention(real, activity(task_details={"next_step": "复现问题"}))
    result = store.patch_intervention(real, "one", {"checkins": [{"feeling": "comfortable", "text": "不可保存"}],
                                                    "followup": {"status": "pending", "due_at": "2030-01-01T00:00:00Z"}})
    assert result["pre"] == 7 and result["post"] == 4
    assert result["task_details"]["next_step"] == "复现问题"
    assert "text" not in result["checkins"][0]
    with pytest.raises(ValueError):
        store.patch_intervention(real, "one", {"session_id": "another-session"})
    assert store.get_intervention(real, "one")["pre"] == 7


def test_preferences_persist_per_profile_and_export_with_session(store, tmp_path):
    one, two, real, _, _ = profiles(store)
    store.save_activity_preference(one, "breathing", "avoid")
    store.save_activity_preference(two, "breathing", "prefer")
    assert store.get_activity_preferences(one) == {"breathing": "avoid"}
    store.save_activity_preference(one, "breathing", "neutral")
    assert store.get_activity_preferences(two)["breathing"] == "prefer"
    exported = json.loads(store.export_session(real, tmp_path / "export.json").read_text(encoding="utf-8"))
    assert exported["profile_preferences"] == {"breathing": "neutral"}
    with pytest.raises(ValueError):
        store.save_activity_preference(one, "unknown", "prefer")
    with pytest.raises(ValueError):
        store.save_activity_preference(one, "breathing", "always")


def test_followups_normalize_time_zones_and_respect_due_state_and_mode(store):
    one, _, real, simulation, _ = profiles(store)
    now = datetime(2030, 1, 1, tzinfo=timezone.utc)
    for identity, session, due, status in (
            ("past", real, "2030-01-01T07:59:00+08:00", "completed"),
            ("future", real, "2030-01-01T00:02:00Z", "completed"),
            ("unfinished", real, "2029-12-31T00:00:00Z", "running"),
            ("sim", simulation, "2029-12-31T00:00:00Z", "completed")):
        store.save_intervention(session, activity(identity, status=status,
            followup={"status": "pending", "due_at": due, "scheduled_at": "2029-12-31T23:00:00Z"}))
    pending = store.list_pending_followups(one, "real", now=now)
    assert [record["id"] for record in pending] == ["past", "future"]
    assert pending[0]["due"] and pending[0]["remaining_seconds"] == 0
    assert pending[1]["remaining_seconds"] == 120
    assert len(store.list_pending_followups(one, "real", now=now, due_only=True)) == 1


def test_completed_followup_is_not_pending_and_deleting_session_removes_pending(store):
    one, _, real, _, _ = profiles(store)
    store.save_intervention(real, activity(followup={"status": "pending", "due_at": "2020-01-01T00:00:00Z"}))
    assert len(store.list_pending_followups(one)) == 1
    completed = store.patch_intervention(real, "one", {"followup": {"status": "completed", "distress": 3, "helpfulness": "helpful"}})
    assert completed["pre"] == 7 and not store.list_pending_followups(one)
    store.patch_intervention(real, "one", {"followup": {"status": "pending", "due_at": "2020-01-01T00:00:00Z"}})
    store.delete_session(real)
    assert store.get_intervention(real, "one") is None and not store.list_pending_followups(one)


def test_missing_or_malformed_followup_does_not_break_other_results(store):
    one, _, real, _, _ = profiles(store)
    store.save_intervention(real, activity("bad", followup={"status": "pending", "due_at": "not-a-date"}))
    store.save_intervention(real, activity("good", followup={"status": "pending", "due_at": "2030-01-01T00:00:00Z"}))
    assert [record["id"] for record in store.list_pending_followups(one)] == ["good"]
    assert store.get_intervention(real, "missing") is None
    with pytest.raises(ValueError):
        store.patch_intervention(real, "missing", {})


def test_pending_feedback_and_explicit_preference_survive_database_reopen(tmp_path):
    path = tmp_path / "persisted.sqlite3"
    first = Store(path)
    user = first.create_user("恢复测试")
    session = first.start_session(user, {}, "real")
    first.save_activity_preference(user, "breathing", "avoid")
    first.save_intervention(session, activity(followup={"status": "pending", "due_at": "2030-01-01T00:00:00Z"}))
    first.close()
    reopened = Store(path)
    try:
        assert reopened.get_activity_preferences(user) == {"breathing": "avoid"}
        assert reopened.list_pending_followups(user, "real")[0]["id"] == "one"
    finally:
        reopened.close()


def test_history_retains_numeric_improvement_but_exposes_adverse_feedback_priority(store):
    _, _, real, _, _ = profiles(store)
    store.save_intervention(real, activity(evaluation="improved", delta=3,
        helpfulness="worse", loop_result={"state": "discomfort", "caption": "即时不适优先"},
        followup={"status": "pending", "due_at": "2030-01-01T00:00:00Z"}))
    latest = next(item for item in store.list_sessions() if item["id"] == real)
    assert latest["last_evaluation"] == "improved" and latest["last_delta"] == 3
    assert latest["latest_loop_state"] == "discomfort" and latest["latest_discomfort"] is True
    assert latest["latest_loop_caption"] == "即时不适优先"
    assert latest["latest_followup_status"] == "pending"


def test_history_delayed_discomfort_cannot_be_hidden_by_immediate_positive_result(store):
    _, _, real, _, _ = profiles(store)
    store.save_intervention(real, activity(evaluation="improved", delta=3,
        loop_result={"state": "improved", "caption": "即时下降"},
        followup={"status": "completed", "distress": 8, "helpfulness": "worse",
                  "loop_result": {"state": "discomfort", "caption": "延迟不适优先"}}))
    latest = next(item for item in store.list_sessions() if item["id"] == real)
    assert latest["last_evaluation"] == "improved"
    assert latest["latest_loop_state"] == "discomfort" and latest["latest_discomfort"] is True
    assert latest["latest_loop_caption"] == "延迟不适优先"
    assert latest["latest_followup_status"] == "completed"
