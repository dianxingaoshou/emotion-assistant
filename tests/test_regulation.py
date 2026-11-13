from copy import deepcopy

import pytest

from emotion_assistant.regulation import ACTIVITIES, NEEDS, evaluate_outcome, recommend_activities, summarize_feedback


def record(identity="one", activity="breathing", pre=7, post=4, helpfulness="helpful", mode="real", **extra):
    return {"id": identity, "session_id": "session-one", "activity": activity,
            "status": "completed", "pre": pre, "post": post,
            "helpfulness": helpfulness, "mode": mode, **extra}


def by_activity(items, activity="breathing"):
    return next(item for item in items if item["activity"] == activity)


def test_library_keeps_existing_methods_and_supplies_original_steps_for_all_eight():
    assert len(ACTIVITIES) == 8
    assert {"break", "breathing", "grounding", "task_split"} <= set(ACTIVITIES)
    for activity in ACTIVITIES.values():
        assert 3 <= len(activity["steps"]) <= 4
        assert all(step["title"] and step["body"] for step in activity["steps"])
        assert set(activity["needs"]) <= set(NEEDS)
        assert activity["source"] and activity["min_seconds"] <= activity["default_seconds"]


def test_simulation_feedback_cannot_become_real_recommendation_evidence():
    history = [record(str(i), pre=10, post=0, mode="simulation") for i in range(5)]
    real = by_activity(recommend_activities(history, mode="real"))
    simulated = by_activity(recommend_activities(history, mode="simulation"))
    assert real["evidence_count"] == 0 and real["mean_delta"] is None
    assert simulated["evidence_count"] == 5 and simulated["mean_delta"] == 10
    assert real["score"] == by_activity(recommend_activities([], mode="real"))["score"]


def test_duplicate_activity_and_followup_does_not_count_as_independent_evidence():
    item = record(followup={"status": "completed", "distress": 3, "helpfulness": "helpful"})
    stats = summarize_feedback([item, deepcopy(item), deepcopy(item)])
    one = by_activity(stats["per_activity"])
    assert one["completed_count"] == one["evidence_count"] == one["followup_count"] == 1
    assert one["mean_delta"] == 3 and one["mean_followup_delta"] == 4


def test_missing_cancelled_and_boolean_ratings_cannot_improve_statistics():
    history = [record("missing", post=None), record("cancelled", status="cancelled"),
               record("bad-pre", pre=True), record("bad-post", post=4.0)]
    item = by_activity(recommend_activities(history))
    assert item["completed_count"] == 3 and item["evidence_count"] == 0
    assert item["mean_delta"] is None and item["independent_evidence_count"] == 0
    assert item["subjective_count"] == 2 and item["helpful_rate"] == 1
    assert any("反馈不足" in reason for reason in item["reasons"])


def test_numeric_improvement_never_overrides_reported_discomfort():
    item = record(helpfulness="worse")
    outcome = evaluate_outcome(item["pre"], item["post"], item["helpfulness"])
    assert outcome["delta"] == 3 and outcome["state"] == "discomfort"
    assert "优先" in outcome["caption"]
    assert by_activity(recommend_activities([item]))["blocked"]
    assert evaluate_outcome(7, 4, "更不舒服")["state"] == "discomfort"


@pytest.mark.parametrize("extra", [
    {"checkins": [{"feeling": "uncomfortable"}], "status": "cancelled"},
    {"discomfort": True, "status": "cancelled"},
    {"followup": {"status": "completed", "distress": 2, "helpfulness": "worse"}},
    {"followup": {"status": "completed", "distress": 2, "discomfort": True}},
])
def test_process_or_delayed_discomfort_excludes_method_even_if_delta_positive(extra):
    assert by_activity(recommend_activities([record(**extra)]))["blocked"]


def test_user_preference_and_time_budget_have_explanations_and_do_not_run_activities():
    plain = recommend_activities([], need="stuck", available_seconds=180)
    preferred = recommend_activities([], need="stuck", available_seconds=180,
                                     preferences={"task_split": "prefer", "breathing": "avoid"})
    assert by_activity(preferred, "task_split")["score"] > by_activity(plain, "task_split")["score"]
    assert by_activity(preferred)["blocked"]
    assert any("明确偏好" in reason for reason in by_activity(preferred, "task_split")["reasons"])
    assert all(item["blocked"] for item in recommend_activities([], available_seconds=30))


def test_top_options_are_diverse_and_historical_effect_is_not_overclaimed_when_sparse():
    ranked = recommend_activities([record()], need="unsure")
    assert len({item["category"] for item in ranked[:3]}) == 3
    assert [item["rank"] for item in ranked] == list(range(1, 9))
    assert any("反馈不足" in reason for reason in by_activity(ranked)["reasons"])
    assert not any("最佳" in reason for item in ranked for reason in item["reasons"])


def test_missing_post_is_explicit_and_legacy_records_without_mode_have_unknown_source():
    assert evaluate_outcome(7, None, "helpful")["state"] == "insufficient"
    assert evaluate_outcome(True, 2, "helpful")["delta"] is None
    legacy = record()
    legacy.pop("mode")
    assert summarize_feedback([legacy], "real")["evidence_count"] == 0
    assert summarize_feedback([legacy], "simulation")["evidence_count"] == 0


def test_valid_delayed_rating_can_inform_next_choice_when_immediate_post_was_skipped():
    baseline = by_activity(recommend_activities([]))
    item = record(post=None, followup={"status": "completed", "distress": 3, "helpfulness": "helpful"})
    ranked = by_activity(recommend_activities([item]))
    assert ranked["evidence_count"] == 0 and ranked["followup_count"] == 1
    assert ranked["score"] > baseline["score"]
    assert any("反馈不足" in reason for reason in ranked["reasons"])


def test_recommendation_does_not_mutate_private_history_or_preferences():
    history = [record()]
    preferences = {"grounding": "prefer"}
    expected = deepcopy((history, preferences))
    recommend_activities(history, preferences=preferences)
    assert (history, preferences) == expected
    with pytest.raises(ValueError):
        recommend_activities(history, available_seconds=True)


def test_cancelled_confirmation_without_an_activity_is_not_counted_as_activity_use():
    stats = summarize_feedback([record(activity=None, status="cancelled")])
    assert stats["total_count"] == stats["completed_count"] == stats["evidence_count"] == 0


def test_explicit_subjective_feedback_remains_valid_without_numeric_post():
    history = [record("helpful"), record("no-change", helpfulness="no_change"),
               record("unknown", helpfulness="unknown"), record("no-post", post=None),
               record("cancelled", status="cancelled"), record("bool", pre=True)]
    item = by_activity(summarize_feedback(history)["per_activity"])
    assert item["evidence_count"] == 3
    assert item["subjective_count"] == 3 and item["helpful_count"] == 2
    assert item["helpful_rate"] == 2 / 3
    empty = by_activity(summarize_feedback([])["per_activity"])
    assert empty["subjective_count"] == empty["helpful_count"] == 0
    assert empty["helpful_rate"] is None
