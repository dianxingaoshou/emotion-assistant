"""Compare a contextual shadow ranking to the rule ranking shown to the user."""
from statistics import mean, stdev

from .regulation import (_chain_members, _discomfort, _need, _rating, _records,
                         recommend_activities)


def _time_band(seconds):
    if type(seconds) is not int or seconds <= 0:
        return None
    return 0 if seconds <= 120 else 1 if seconds <= 300 else 2


def _distress_band(pre):
    if _rating(pre) is None:
        return None
    return 0 if pre <= 3 else 1 if pre <= 6 else 2


def contextual_decision(history, need, seconds, pre, preferences, mode, excluded=()):
    items = recommend_activities(history, need, seconds, preferences, mode)
    records = _records(history, mode)
    chains = _chain_members(records)
    outcomes = {}
    for entry in records:
        key = (str(entry.get("user_id", "")), str(entry.get("session_id", "")), str(entry["id"]))
        before = _rating(entry.get("pre"))
        if (entry.get("status") != "completed" or key in chains or _discomfort(entry)
                or before is None or _distress_band(before) != _distress_band(pre)
                or _need(entry) != need
                or _time_band((entry.get("context") or {}).get("available_seconds")) != _time_band(seconds)):
            continue
        followup = entry.get("followup") or {}
        delayed = _rating(followup.get("distress")) if followup.get("status") == "completed" else None
        post = _rating(entry.get("post"))
        if post is None and delayed is None:
            continue
        after = (.7 * post + .3 * delayed if post is not None and delayed is not None
                 else post if post is not None else delayed)
        outcomes.setdefault(entry.get("activity"), []).append(before - after)

    shadow_scores = {}
    summaries = {}
    for item in items:
        action = item["activity"]
        if action in excluded and not item["blocked"]:
            item["blocked"] = True
            item["reasons"].insert(0, "此档案在相同来源中记录过不适。")
        values = outcomes.get(action, [])
        count = len(values)
        average = mean(values) if values else None
        spread = stdev(values) if count >= 2 else None
        summaries[action] = {"count": count, "mean_delta": average, "sample_sd": spread}
        item["matched_feedback"] = summaries[action]
        if not item["blocked"]:
            # Penalize sampling uncertainty. A missing estimate gets no adjustment.
            adjustment = 0 if count < 2 else (average - spread / count ** .5) * count / (count + 3)
            shadow_scores[action] = -item["rank"] + adjustment
    actual = [item["activity"] for item in items if not item["blocked"]]
    shadow = sorted(actual, key=shadow_scores.get, reverse=True)
    return items, {"context": {"need": need, "available_seconds": seconds, "pre_distress": pre, "mode": mode},
                   "eligible_actions": actual, "ranking": actual, "policy_version": "rules-v1",
                   "suggested_action": next(iter(actual), None), "user_selected_action": None,
                   "selection_probability": None, "outcome_id": None,
                   "shadow": {"policy_version": "matched-feedback-v1", "ranking": shadow,
                              "suggested_action": next(iter(shadow), None), "estimates": summaries}}
