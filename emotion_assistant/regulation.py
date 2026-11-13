"""Local activity guidance and transparent, feedback-aware recommendation rules.

The ranking score is an engineering ordering aid, not an efficacy probability.
Only the user's explicit preferences and feedback are used for personalization.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone


WHO_SOURCE = "https://www.who.int/publications/i/item/9789240003927/"
NHS_SOURCE = "https://www.nhs.uk/mental-health/self-help/guides-tools-and-activities/breathing-exercises-for-stress/"

NEEDS = {
    "tense": "身体紧绷，想放松一下",
    "rumination": "反复想一件事，想回到当下",
    "stuck": "任务卡住，想找到下一小步",
    "fatigue": "精力不足，想暂停恢复",
    "unsure": "暂时说不清，先试一种短活动",
}


def _activity(title, summary, needs, category, seconds, steps, source, source_url=""):
    return {"title": title, "summary": summary, "needs": list(needs),
            "category": category, "categories": [category], "default_seconds": seconds,
            "min_seconds": 60, "steps": [{"title": name, "body": body} for name, body in steps],
            "source": source, "source_url": source_url}


ACTIVITIES = {
    "break": _activity("短时休息", "暂时离开任务，给自己一点缓冲。", ("fatigue", "tense", "unsure"), "休息与环境", 120,
        [("暂时放下任务", "把手边任务放一放，让视线离开屏幕。无需立刻解决问题。"),
         ("照顾舒适感", "按自己的习惯喝一点水，或安静坐一会儿。是否离开座位由你决定。"),
         ("留意此刻需要", "注意自己现在需要继续休息、换个方法，还是回到一个很小的任务。")],
        "产品内置短时休息提示，不是治疗方案。"),
    "breathing": _activity("舒适呼吸", "用可选视觉节奏留意自然呼吸，不憋气、不勉强。", ("tense", "unsure"), "身体与当下", 120,
        [("找到舒适姿势", "让身体有支撑，保持自然呼吸。圆圈只是可选提示，不需要跟上它。"),
         ("留意自然吸气", "让气息按舒适的深度自然进入，不刻意吸满，也不追求固定拍数。"),
         ("缓缓自然呼出", "用自己舒服的方式呼出，不憋气、不用力。感到不舒服可以立即停止。"),
         ("回到自己的节奏", "放开对节奏的要求，留意此刻的感受，再决定是否继续。")],
        "参考 NHS 的舒适、轻柔、不强迫呼吸原则；应用短活动不等于已验证的疗程。", NHS_SOURCE),
    "grounding": _activity("感官注意", "把注意力带回眼前环境，可自由跳过任何步骤。", ("rumination", "tense", "unsure"), "身体与当下", 120,
        [("看看身边", "留意眼前几件物品的颜色、形状或位置。不需要找满指定数量。"),
         ("感受接触", "注意脚与地面、身体与座椅的接触。选择一种让自己舒适的触感。"),
         ("听见此刻", "留意环境里一种声音，再把注意力慢慢带回你愿意做的事。")],
        "依据 WHO《Doing What Matters in Times of Stress》的 grounding 原则原创简短引导。", WHO_SOURCE),
    "task_split": _activity("任务拆解", "把卡点、已尝试的方法和下一步写清楚。", ("stuck", "rumination", "unsure"), "任务与行动", 180,
        [("只说一个卡点", "用一句具体的话描述现在卡在哪里，先不评价自己做得好不好。"),
         ("记录已尝试", "写下已经试过的方法。避免不停重复一个没有带来新信息的动作。"),
         ("缩成下一小步", "选一个能开始、能检查的小动作。把它填入下一步，完成后再决定后续。")],
        "产品内置结构化任务表单，不替代专业建议。"),
    "thought_label": _activity("给想法命名", "观察一个反复出现的想法，不急着判断或赶走它。", ("rumination", "stuck"), "想法与回应", 120,
        [("注意到它", "留意正在反复出现的想法。无需把它说出来，也不必强迫自己停止想。"),
         ("加一个开头", "可以在心里说：我注意到，自己正在想……。这是一种观察方式，不要求相信另一种结论。"),
         ("回到可见之处", "看看眼前的一件物品或感受脚下接触，再选择一个愿意做的小动作。")],
        "依据 WHO 指南的 noticing/naming 与 unhooking 原则原创简短引导。", WHO_SOURCE),
    "self_kindness": _activity("友善自我对话", "用对朋友的语气回应自己，允许暂时有困难。", ("rumination", "tense", "unsure"), "想法与回应", 120,
        [("承认有点难", "可以先承认：这件事现在让我觉得不容易。无需否定或放大这种感受。"),
         ("换一种语气", "想一想，如果朋友遇到同样的事，你愿意对他说什么。选择一句自己能接受的话。"),
         ("照顾眼前需要", "留意现在需要一点休息、支持，还是一个可完成的小动作。没有标准答案。")],
        "依据 WHO 指南 being kind 的原则原创简短引导。", WHO_SOURCE),
    "values_action": _activity("价值微行动", "从你在意的方向里，选择一个当下可控的小动作。", ("stuck", "rumination", "fatigue"), "任务与行动", 180,
        [("选一个在意的方向", "想一想此刻比较在意什么，例如认真、学习、照顾自己或与人合作。由你决定，不需要填标准答案。"),
         ("缩成可控行动", "把这个方向缩成一个当下能开始的小动作，而不是一个必须马上实现的大结果。"),
         ("试一个小动作", "如果愿意，就从这个动作开始；也可以把它写入下一步，稍后再做。")],
        "依据 WHO 指南 acting on your values 的原则原创简短引导。", WHO_SOURCE),
    "desk_reset": _activity("桌面舒适调整", "调整坐姿与注意焦点，活动范围以舒适为准。", ("fatigue", "tense", "unsure"), "休息与环境", 60,
        [("先离开任务焦点", "把目光从任务上移开，看看周围。无需突然起身，也不需要做大幅动作。"),
         ("选择舒适小动作", "如果愿意，可以自然活动手指或轻轻放松肩部。只在舒适范围内进行，疼痛或不适时停止。"),
         ("找到下一种安排", "调整到舒服的姿势，再选择继续休息、换一种活动或回到任务。")],
        "产品内置桌面舒适提醒，不是专业运动或康复方案。"),
}


def _rating(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 10 else None


def _mode(value):
    if value in ("simulation", "demo", "replay"):
        return "simulation"
    return "real" if value == "real" else None


def _helpfulness(value):
    if not isinstance(value, str):
        return None
    return {"有帮助": "helpful", "觉得有帮助": "helpful", "无明显帮助": "no_change",
            "更不舒服": "worse", "无法评价": "unknown", "暂时无法评价": "unknown"}.get(value, value)


def _discomfort(record):
    if record.get("discomfort") is True or _helpfulness(record.get("helpfulness")) == "worse":
        return True
    if record.get("stop_reason") in ("uncomfortable", "discomfort"):
        return True
    for field in ("checkins", "checkpoints", "process_feedback"):
        events = record.get(field) or []
        if isinstance(events, dict):
            events = [events]
        if isinstance(events, (list, tuple)):
            for event in events:
                values = event.values() if isinstance(event, dict) else [event]
                if any(value in ("uncomfortable", "discomfort", "worse", "更不舒服")
                       for value in values if isinstance(value, str)):
                    return True
    followup = record.get("followup") or {}
    return isinstance(followup, dict) and (followup.get("discomfort") is True
                                           or _helpfulness(followup.get("helpfulness")) == "worse")


def evaluate_outcome(pre, post, helpfulness):
    """Keep numeric self-report change separate from contradictory discomfort."""
    before, after = _rating(pre), _rating(post)
    delta = before - after if before is not None and after is not None else None
    if _helpfulness(helpfulness) == "worse":
        state, caption = "discomfort", "你报告更不舒服；这次反馈优先于分数下降。"
    elif delta is None:
        state, caption = "insufficient", "缺少有效前后自评，暂时无法评价变化。"
    elif delta > 0:
        state, caption = "improved", f"本次自评困扰下降 {delta} 分，主观帮助程度另行记录。"
    elif delta < 0:
        state, caption = "worse", f"本次自评困扰上升 {-delta} 分，可停止或换一种方式。"
    else:
        state, caption = "unchanged", "本次自评困扰没有变化，可以按自己的需要决定下一步。"
    return {"state": state, "delta": delta, "caption": caption}


def _timestamp(value):
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.timestamp()
    except (ValueError, TypeError, OverflowError):
        return 0.0


def _records(history, mode):
    normalized_mode = _mode(mode)
    if normalized_mode is None:
        raise ValueError("mode must be real or simulation")
    latest = {}
    for index, item in enumerate(history or []):
        if not isinstance(item, dict) or _mode(item.get("mode")) != normalized_mode or not item.get("id"):
            continue
        identity = (str(item.get("user_id", "")), str(item.get("session_id", "")), str(item["id"]))
        # Revisions supersede earlier completion snapshots, including a later
        # cancellation/discomfort correction. Missing provenance stays unknown.
        revision = item.get("revision", 0)
        revision = revision if isinstance(revision, int) and not isinstance(revision, bool) else 0
        priority = (max(_timestamp(item.get(key)) for key in ("updated_at", "ended_at", "started_at")), revision, index)
        if identity not in latest or priority >= latest[identity][0]:
            latest[identity] = (priority, item)
    return [deepcopy(value[1]) for value in latest.values()]


def _need(record):
    context = record.get("context")
    value = context.get("need") if isinstance(context, dict) else None
    return value if isinstance(value, str) and value in NEEDS else None


def _chain_members(records):
    """Do not attribute a multi-method sequence to its final method."""
    members = set()
    for entry in records:
        session, identity = (str(entry.get("user_id", "")), str(entry.get("session_id", ""))), str(entry["id"])
        previous, root = entry.get("previous_intervention_id"), entry.get("root_intervention_id")
        if previous or (root and str(root) != identity) or entry.get("stop_reason") == "manual_switch":
            members.add((*session, identity))
            for linked in (previous, root):
                if linked:
                    members.add((*session, str(linked)))
    return members


def _subjective_feedback(record):
    values = [_helpfulness(record.get("helpfulness"))]
    followup = record.get("followup")
    if isinstance(followup, dict) and followup.get("status") in ("completed", "skipped"):
        values.append(_helpfulness(followup.get("helpfulness")))
    if "worse" in values:
        return "worse"
    # One subjective observation per independent activity; a valid later answer
    # revises the earlier one instead of adding another independent sample.
    return next((value for value in reversed(values) if value in ("helpful", "no_change")), None)


def summarize_feedback(history, mode="real", need=None):
    records = [entry for entry in _records(history, mode) if entry.get("activity") in ACTIVITIES]
    chain_members = _chain_members(records)
    by_activity = defaultdict(list)
    for item in records:
        if item.get("activity") in ACTIVITIES:
            by_activity[item["activity"]].append(item)
    per_activity = []
    for activity, specification in ACTIVITIES.items():
        entries = by_activity[activity]
        if need is not None:
            entries = [entry for entry in entries if _need(entry) == need]
        completed = [entry for entry in entries if entry.get("status") == "completed"]
        independent = [entry for entry in completed
                       if (str(entry.get("user_id", "")), str(entry.get("session_id", "")), str(entry["id"])) not in chain_members]
        evidence = [entry for entry in completed if _rating(entry.get("pre")) is not None
                    and _rating(entry.get("post")) is not None]
        deltas = [entry["pre"] - entry["post"] for entry in evidence]
        # An explicit response remains useful even when the numeric post-test
        # was skipped. Invalid numeric pre-ratings are still excluded.
        subjective = [_subjective_feedback(entry) for entry in completed
                      if _rating(entry.get("pre")) is not None
                      if _subjective_feedback(entry) is not None]
        rank_subjective = [_subjective_feedback(entry) for entry in independent
                          if _rating(entry.get("pre")) is not None and _subjective_feedback(entry) is not None]
        followups, effective_deltas, feedback_ids = [], [], []
        for entry in completed:
            followup = entry.get("followup")
            if (isinstance(followup, dict) and followup.get("status") == "completed"
                    and _rating(entry.get("pre")) is not None and _rating(followup.get("distress")) is not None):
                followups.append(entry["pre"] - followup["distress"])
        for entry in independent:
            pre, post = _rating(entry.get("pre")), _rating(entry.get("post"))
            immediate = pre - post if pre is not None and post is not None else None
            followup = entry.get("followup") or {}
            delayed = None
            if (isinstance(followup, dict) and followup.get("status") == "completed"
                    and pre is not None and _rating(followup.get("distress")) is not None):
                delayed = pre - followup["distress"]
            if immediate is not None or delayed is not None:
                # Pair inside one intervention, then average distinct activities.
                # The .7/.3 weighting is an engineering setting, not causality.
                effective_deltas.append(immediate * .7 + delayed * .3
                                        if immediate is not None and delayed is not None
                                        else immediate if immediate is not None else delayed)
                feedback_ids.append(entry["id"])
        per_activity.append({"activity": activity, "title": specification["title"],
            "completed_count": len(completed), "evidence_count": len(evidence),
            "mean_delta": sum(deltas) / len(deltas) if deltas else None,
            "independent_evidence_count": len(effective_deltas),
            "effective_mean_delta": sum(effective_deltas) / len(effective_deltas) if effective_deltas else None,
            "chain_excluded_count": len(completed) - len(independent),
            "feedback_count": len(effective_deltas), "feedback_ids": feedback_ids,
            "feedback_mean_delta": sum(effective_deltas) / len(effective_deltas) if effective_deltas else None,
            "excluded_chain_count": len(completed) - len(independent),
            "ranking_subjective_count": len(rank_subjective),
            "ranking_helpful_rate": rank_subjective.count("helpful") / len(rank_subjective) if rank_subjective else None,
            "subjective_count": len(subjective),
            "helpful_count": subjective.count("helpful"),
            "helpful_rate": subjective.count("helpful") / len(subjective) if subjective else None,
            "discomfort_count": sum(_discomfort(entry) for entry in entries),
            "followup_count": len(followups),
            "mean_followup_delta": sum(followups) / len(followups) if followups else None})
    return {"mode": _mode(mode), "total_count": sum(need is None or _need(entry) == need for entry in records),
            "completed_count": sum(item["completed_count"] for item in per_activity),
            "evidence_count": sum(item["evidence_count"] for item in per_activity),
            "independent_evidence_count": sum(item["independent_evidence_count"] for item in per_activity),
            "chain_excluded_count": sum(item["chain_excluded_count"] for item in per_activity),
            "independent_episode_count": len({(str(entry.get("user_id", "")), str(entry.get("session_id", "")),
                                                str(entry.get("root_intervention_id") or entry["id"])) for entry in records}),
            "discomfort_count": sum(item["discomfort_count"] for item in per_activity),
            "followup_count": sum(item["followup_count"] for item in per_activity),
            "per_activity": per_activity,
            "need": need,
            "note": "即时与延迟反馈先在同一活动内配对，再按独立活动统计；换方法链不归因，未知来源不计入真实历史。"}


def recommend_activities(history, need="unsure", available_seconds=120, preferences=None, mode="real"):
    """Rank all activities with explicit reasons and conservative sparse history.

    The first three unblocked candidates favor different categories. Feedback is
    shrunk toward a neutral prior using n/(n+3), not treated as proven efficacy.
    """
    if isinstance(available_seconds, bool) or not isinstance(available_seconds, int) or available_seconds <= 0:
        raise ValueError("available_seconds must be a positive integer")
    need = need if isinstance(need, str) and need in NEEDS else "unsure"
    preferences = preferences if isinstance(preferences, dict) else {}
    statistics = summarize_feedback(history, mode)
    per_activity = {entry["activity"]: entry for entry in statistics["per_activity"]}
    context_statistics = {entry["activity"]: entry for entry in summarize_feedback(history, mode, need)["per_activity"]}
    candidates = []
    for activity, specification in ACTIVITIES.items():
        stats = per_activity[activity]
        context_stats = context_statistics[activity]
        use_context = bool(context_stats["independent_evidence_count"] or context_stats["ranking_subjective_count"])
        selected_stats = context_stats if use_context else stats
        count = stats["evidence_count"]
        preference = preferences.get(activity, "neutral")
        reasons, blocked = [], False
        matched = need in specification["needs"]
        score = 40.0 + (20 if matched else 5 if need == "unsure" else 0)
        if matched:
            reasons.append("匹配你选择的需要：" + NEEDS[need])
        else:
            reasons.append("提供另一种活动方式，最终由你选择。")
        if available_seconds < specification["min_seconds"]:
            blocked = True
            reasons.append(f"当前可用时间不足，建议至少预留 {specification['min_seconds']} 秒。")
        elif specification["default_seconds"] <= available_seconds:
            score += 10
            reasons.append(f"默认约 {specification['default_seconds'] // 60} 分钟，符合当前可用时间。")
        else:
            score += 5
            reasons.append(f"可按当前 {available_seconds} 秒的时间安排，步骤允许跳过。")
        if preference == "avoid":
            blocked = True
            reasons.append("你选择了暂不推荐此方法。")
        elif preference == "prefer":
            score += 8
            reasons.append("你明确偏好此方法。")
        if stats["discomfort_count"]:
            blocked = True
            reasons.append(f"记录过 {stats['discomfort_count']} 次不适反馈，暂不推荐。")
        # A valid delayed rating may still inform the next choice when the
        # immediate post-rating was skipped. Never add the two counts: one
        # activity with two ratings is not two independent activities.
        history_count = selected_stats["independent_evidence_count"]
        history_scale = 1.0 if use_context or need == "unsure" else .35
        if history_count:
            mean = selected_stats["effective_mean_delta"]
            score += history_scale * 12 * max(-1, min(1, mean / 5)) * history_count / (history_count + 3)
        subjective_count = selected_stats["ranking_subjective_count"]
        if subjective_count:
            score += history_scale * 12 * (selected_stats["ranking_helpful_rate"] - .5) * subjective_count / (subjective_count + 3)
        if use_context:
            reasons.append(f"优先参考相同需求的 {history_count} 次独立活动和 {subjective_count} 次主观反馈。")
        elif history_count or subjective_count:
            reasons.append("尚无同需求反馈，跨需求历史以较低权重参考。")
        if history_count < 3:
            reasons.append(f"独立有效活动 {history_count} 次，反馈不足；主要依据需求、时间与偏好排序。")
        else:
            reasons.append(f"独立有效活动 {history_count} 次，活动内配对后平均自评变化 {selected_stats['effective_mean_delta']:.1f} 分；仅作个人反馈参考。")
        if stats["followup_count"]:
            reasons.append(f"另有 {stats['followup_count']} 次有效延迟复测，未把重复反馈当成新活动。")
        if stats["chain_excluded_count"]:
            reasons.append(f"{stats['chain_excluded_count']} 次换方法链记录不计入单一方法收益。")
        candidates.append({**stats, "category": specification["category"],
                           "history_evidence_count": history_count,
                           "history_mean_delta": selected_stats["effective_mean_delta"],
                           "context_evidence_count": context_stats["independent_evidence_count"],
                           "context_subjective_count": context_stats["subjective_count"],
                           "context_used": use_context,
                           "rank_feedback_count": history_count,
                           "personalization_scope": "same_need" if use_context else "all_needs_fallback",
                           "history_scope": "same_need" if use_context else "all_needs" if history_count or subjective_count else "none",
                           "score": round(max(0, min(100, score)), 3), "reasons": reasons,
                           "blocked": blocked})
    candidates.sort(key=lambda entry: (entry["blocked"], -entry["score"], entry["activity"]))
    chosen, remaining, categories = [], [], set()
    for entry in candidates:
        if len(chosen) < 3 and not entry["blocked"] and entry["category"] not in categories:
            chosen.append(entry)
            categories.add(entry["category"])
        else:
            remaining.append(entry)
    result = chosen + remaining
    for index, item in enumerate(result, 1):
        item["rank"] = index
    return result
