"""Regression checks for independent evidence, provenance and atomic feedback."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import sqlite3

import pytest

from emotion_assistant.regulation import recommend_activities, summarize_feedback
from emotion_assistant.storage import Store


def activity(identity="one", **extra):
    return {"id": identity, "session_id": "session", "mode": "real",
            "activity": "breathing", "status": "completed", "pre": 8, "post": 5,
            "helpfulness": "unknown", **extra}


def breathing(items):
    return next(item for item in items if item["activity"] == "breathing")


def stats(records):
    return breathing(summarize_feedback(records)["per_activity"])


def test_disjoint_immediate_delayed_records_count_union_not_max():
    records = [activity("immediate"), activity("delayed", post=None,
               followup={"status": "completed", "distress": 4})]
    result = stats(records)
    assert result["evidence_count"] == result["followup_count"] == 1
    assert result["independent_evidence_count"] == 2
    assert result["effective_mean_delta"] == 3.5


def test_feedback_is_paired_inside_activity_before_cross_activity_average():
    records = [activity("paired", pre=10, post=10,
                followup={"status": "completed", "distress": 0}),
               activity("immediate", pre=10, post=2)]
    result = stats(records)
    assert result["independent_evidence_count"] == 2
    assert result["effective_mean_delta"] == 5.5  # (0*.7 + 10*.3 + 8) / 2
    assert result["mean_delta"] == 4 and result["mean_followup_delta"] == 10


def test_explicit_helpfulness_without_post_can_affect_choice_but_not_numeric_effect():
    record = activity(post=None, helpfulness="helpful")
    result = stats([record])
    assert result["independent_evidence_count"] == 0
    assert result["effective_mean_delta"] is None
    assert result["subjective_count"] == result["helpful_count"] == 1
    assert breathing(recommend_activities([record]))["score"] > breathing(recommend_activities([]))["score"]


def test_delayed_subjective_answer_revises_one_sample_and_adverse_wins():
    record = activity(helpfulness="helpful", followup={"status": "completed", "distress": 3,
                                                       "helpfulness": "no_change"})
    result = stats([record])
    assert result["subjective_count"] == 1 and result["helpful_count"] == 0
    record["helpfulness"] = "worse"
    assert stats([record])["discomfort_count"] == 1
    assert breathing(recommend_activities([record]))["blocked"]


def test_newer_correction_supersedes_completed_snapshot():
    original = activity(updated_at="2030-01-01T00:00:00Z")
    correction = activity(status="cancelled", post=None, discomfort=True,
                          updated_at="2030-01-01T00:00:01Z")
    result = stats([correction, original])
    assert result["completed_count"] == result["independent_evidence_count"] == 0
    assert result["discomfort_count"] == 1


@pytest.mark.parametrize("mode", [None, "unknown", "", "REAL"])
def test_unknown_provenance_is_not_real(mode):
    assert stats([activity(mode=mode)])["completed_count"] == 0


def test_replay_is_simulation_and_not_real():
    record = activity(mode="replay")
    assert stats([record])["independent_evidence_count"] == 0
    assert summarize_feedback([record], "simulation")["independent_evidence_count"] == 1


def test_method_chain_is_not_attributed_but_discomfort_crosses_context():
    original = activity("root", status="cancelled", post=None, stop_reason="manual_switch",
                        discomfort=True, context={"need": "tense"})
    following = activity("last", previous_intervention_id="root", root_intervention_id="root",
                         context={"need": "rumination"})
    result = stats([original, following])
    assert result["chain_excluded_count"] == 1
    assert result["independent_evidence_count"] == 0
    assert result["discomfort_count"] == 1
    assert breathing(recommend_activities([original, following], need="rumination"))["blocked"]


def test_same_need_history_preferred_over_unrelated_positive_history():
    positive = [activity(str(i), pre=10, post=0, context={"need": "rumination"}) for i in range(6)]
    same_need = activity("same", pre=0, post=10, context={"need": "tense"})
    baseline = breathing(recommend_activities(positive, need="tense"))
    chosen = breathing(recommend_activities(positive + [same_need], need="tense"))
    assert chosen["context_used"] and chosen["history_scope"] == "same_need"
    assert chosen["context_evidence_count"] == chosen["history_evidence_count"] == 1
    assert chosen["history_mean_delta"] == -10 and chosen["score"] < baseline["score"]


@pytest.mark.parametrize("context", [None, [], "legacy", {"need": []}])
def test_malformed_legacy_context_does_not_crash_or_become_same_need(context):
    result = breathing(recommend_activities([activity(context=context)], need="tense"))
    assert not result["context_used"] and result["history_scope"] == "all_needs"


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "atomic.sqlite3")
    yield value
    value.close()


def profile(store):
    user = store.create_user("原子事务测试")
    return user, store.start_session(user, {"save_transcript": False}, "real")


def stored_activity(**extra):
    record = activity(**extra)
    record.pop("session_id")
    record.pop("mode")
    return record


def export(store, session, tmp_path):
    return json.loads(store.export_session(session, tmp_path / "out.json").read_text(encoding="utf-8"))


def test_atomic_bundle_persists_new_evidence_fields_and_privacy(store, tmp_path):
    user, session = profile(store)
    record = stored_activity(assessment={"assessment_id": "a", "phase": "post", "coverage": .8,
                              "source_ids": ["v1"], "text": "private transcript"})
    result = store.save_activity_bundle(session, record,
        reports=[{"phase": "pre", "distress": 8}, {"phase": "post", "distress": 5}],
        preference="prefer", event_kind="activity_completed", event_payload={"new_diagnostics": True})
    assert result["session_id"] == session and result["user_id"] == user
    reopened = store.get_intervention(session, "one")
    assert reopened["assessment"]["source_ids"] == ["v1"]
    assert "text" not in reopened["assessment"]
    data = export(store, session, tmp_path)
    assert len(data["self_reports"]) == 2 and len(data["events"]) == 1
    assert data["profile_preferences"] == {"breathing": "prefer"}


def test_invalid_report_rolls_back_activity_reports_preference_and_event(store, tmp_path):
    _, session = profile(store)
    with pytest.raises(ValueError):
        store.save_activity_bundle(session, stored_activity(),
            reports=[{"phase": "pre", "distress": 8}, {"phase": "post", "distress": True}],
            preference="prefer", event_kind="activity_completed")
    data = export(store, session, tmp_path)
    assert data["interventions"] == data["self_reports"] == data["events"] == []
    assert data["profile_preferences"] == {}


def test_failure_at_last_event_rolls_back_all_earlier_writes(store, tmp_path):
    user, session = profile(store)
    store.save_intervention(session, stored_activity(status="running", post=None))
    store._db.execute("""CREATE TRIGGER fail_completion BEFORE INSERT ON events
        WHEN NEW.kind='activity_completed' BEGIN SELECT RAISE(ABORT,'injected disk-like failure'); END""")
    store._db.commit()
    with pytest.raises(sqlite3.IntegrityError):
        store.save_activity_bundle(session, stored_activity(),
            reports=[{"phase": "post", "distress": 5}], preference="prefer", event_kind="activity_completed")
    assert store.get_intervention(session, "one")["status"] == "running"
    data = export(store, session, tmp_path)
    assert data["self_reports"] == data["events"] == []
    assert store.get_activity_preferences(user) == {}


def test_bundle_report_retry_does_not_duplicate_self_report_or_clear_preference(store, tmp_path):
    user, session = profile(store)
    store.save_activity_preference(user, "breathing", "avoid")
    for _ in range(2):
        store.save_activity_bundle(session, stored_activity(), reports=[{"phase": "post", "distress": 5}])
    assert len(export(store, session, tmp_path)["self_reports"]) == 1
    assert store.get_activity_preferences(user) == {"breathing": "avoid"}


def pending(store, session):
    store.save_intervention(session, stored_activity(
        evaluation="improved", delta=3,
        followup={"status": "pending", "due_at": "2030-01-01T00:00:00Z"}))


def test_followup_is_atomic_idempotent_and_retains_both_deltas(store, tmp_path):
    user, session = profile(store)
    pending(store, session)
    updated = store.complete_followup(session, "one", user, "real", 6, "worse")
    assert updated["followup"]["delta"] == 2 and updated["followup"]["delta_from_immediate"] == -1
    assert updated["evaluation"] == "discomfort" and updated["numeric_evaluation"] == "improved"
    assert store.complete_followup(session, "one", user, "real", 1, "helpful") is None
    data = export(store, session, tmp_path)
    assert len(data["self_reports"]) == len(data["events"]) == 1


def test_followup_failure_keeps_pending_and_no_partial_report(store, tmp_path):
    user, session = profile(store)
    pending(store, session)
    store._db.execute("""CREATE TRIGGER fail_followup BEFORE INSERT ON events
        WHEN NEW.kind='followup_completed' BEGIN SELECT RAISE(ABORT,'injected failure'); END""")
    store._db.commit()
    with pytest.raises(sqlite3.IntegrityError):
        store.complete_followup(session, "one", user, "real", 4, "helpful")
    assert store.get_intervention(session, "one")["followup"]["status"] == "pending"
    assert export(store, session, tmp_path)["self_reports"] == []


def test_foreign_followup_rejected_and_missing_rating_kept_missing(store, tmp_path):
    user, session = profile(store)
    pending(store, session)
    with pytest.raises(ValueError):
        store.complete_followup(session, "one", "other-user", "real", 4)
    with pytest.raises(ValueError):
        store.complete_followup(session, "one", user, "simulation", 4)
    result = store.complete_followup(session, "one", user, "real", None, "helpful")
    assert result["followup"]["status"] == "skipped" and result["followup"]["delta"] is None
    assert export(store, session, tmp_path)["self_reports"] == []


def test_two_connections_cannot_complete_one_pending_followup_twice(store, tmp_path):
    user, session = profile(store)
    pending(store, session)
    other = Store(store.path)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(db.complete_followup, session, "one", user, "real", 4, "helpful")
                       for db in (store, other)]
            results = [future.result(timeout=5) for future in futures]
        assert sum(result is not None for result in results) == 1
        data = export(store, session, tmp_path)
        assert len(data["self_reports"]) == len(data["events"]) == 1
    finally:
        other.close()


def test_transcript_revocation_preserves_authorized_history_and_blocks_new_results(store, tmp_path):
    user = store.create_user("转写授权测试")
    session = store.start_session(user, {"transcript": True, "save_transcript": True}, "real")
    store.save_utterance(session, {"id": "authorized", "text": "撤回前主动授权保存"}, True)
    consent = store.set_transcript_consent(session, False)
    assert not consent["transcript"] and not consent["save_transcript"]
    assert consent["transcript_revoked_at"] and consent["transcript_consent_revision"] == 1
    assert store.save_utterance(session, {"id": "late", "text": "旧任务迟到结果"}, True) is None
    assert store.save_utterance(session, {"id": "legacy", "text": "默认接口也不能绕过撤回"}) is None
    data = export(store, session, tmp_path)
    assert [item["id"] for item in data["utterances"]] == ["authorized"]
    assert data["utterances"][0]["text"] == "撤回前主动授权保存"
    assert data["events"][-1]["kind"] == "transcript_consent_changed"
    assert data["events"][-1]["payload"]["retains_authorized_history"] is True


def test_strict_transcript_writer_checks_permission_at_execution(store, tmp_path):
    _, session = profile(store)
    assert store.save_utterance(session, {"id": "never-authorized", "text": "不保存"}, True) is None
    assert export(store, session, tmp_path)["utterances"] == []
    with pytest.raises(ValueError):
        store.set_transcript_consent(session, "false")


def test_consent_event_failure_rolls_back_permission_change(store, tmp_path):
    user = store.create_user("授权事务失败测试")
    session = store.start_session(user, {"transcript": True, "save_transcript": True}, "real")
    store._db.execute("""CREATE TRIGGER fail_consent BEFORE INSERT ON events
        WHEN NEW.kind='transcript_consent_changed' BEGIN SELECT RAISE(ABORT,'injected failure'); END""")
    store._db.commit()
    with pytest.raises(sqlite3.IntegrityError):
        store.set_transcript_consent(session, False)
    data = export(store, session, tmp_path)
    assert data["session"]["consent"]["save_transcript"] is True
    assert data["events"] == []


def test_patch_compatible_feedback_aliases_match_independent_statistics():
    result = stats([activity(followup={"status": "completed", "distress": 4})])
    assert result["feedback_count"] == result["independent_evidence_count"] == 1
    assert result["feedback_mean_delta"] == result["effective_mean_delta"] == pytest.approx(3.3)
    assert result["feedback_ids"] == ["one"]
    chosen = breathing(recommend_activities([activity(context={"need": "tense"})], need="tense"))
    assert chosen["personalization_scope"] == "same_need" and chosen["rank_feedback_count"] == 1
