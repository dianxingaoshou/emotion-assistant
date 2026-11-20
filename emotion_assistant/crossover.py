"""Two-period randomized crossover; aggregate only consented periods of one source."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean


VERSION = "crossover-ab-v1"
ARMS = ("basic", "adaptive")
METRICS = (
    "unwanted_per_hour",
    "acceptance_rate",
    "completion_rate",
    "immediate_mean_delta",
    "followup_mean_delta",
)


def arm_for(sequence, period):
    return {"A": "basic", "B": "adaptive"}[sequence[period - 1]]


def trial_prompt(policy, condition, negative_ready, conflict_ready, recent_unwanted):
    """Both conditions retain all common refusal, availability and safety limits."""
    if not condition:
        return policy
    reason = policy["reason"]
    if policy["action"] == "ask":
        if (condition == "basic" and conflict_ready and not negative_ready) or (
            condition == "adaptive" and recent_unwanted
        ):
            # The enrollment and explicit burden labels live only in the deletable
            # study tables, not in ordinary history or queued event writes.
            reason = "timing_deferred"
    return dict(policy, action=policy["action"] if reason == policy["reason"] else "observe", reason=reason)


def trial_order(items, decision, condition):
    if condition != "adaptive":
        return items, decision
    ranking = decision["shadow"]["ranking"]
    lookup = {name: i for i, name in enumerate(ranking)}
    # The ordinary rule ranking and matched estimates are retained for audit.
    items = sorted(items, key=lambda item: lookup.get(item["activity"], len(ranking) + item["rank"]))
    for i, item in enumerate(items, 1):
        item["rank"] = i
    return items, dict(
        decision,
        baseline_ranking=decision["ranking"],
        ranking=ranking,
        suggested_action=next(iter(ranking), None),
        policy_version="matched-feedback-v1",
    )


def _ratio(n, d):
    return n / d if d else None


def _valid(x):
    return type(x) is int and 0 <= x <= 10


def summary(periods):
    count = defaultdict(int)
    immediate, delayed = [], []
    active_seconds = observed_seconds = 0.0
    primary_seconds = 0.0
    primary_unwanted = primary_missing = 0
    for p in periods:
        active_seconds += p["active_seconds"]
        covered = min(p["active_seconds"], p["observed_seconds"])
        observed_seconds += covered
        asked, responses, seen = set(), {}, set()
        for kind, item in p["events"]:
            did = item.get("decision_id")
            if not did:
                continue
            if kind == "prompt_decision":
                seen.add(did)
            elif kind == "prompt_asked":
                asked.add(did)
            elif kind == "intervention_confirmation" and type(item.get("accepted")) is bool:
                responses[did] = item["accepted"]
            elif kind == "recommendation_offered":
                count["recommendation_offers"] += 1
                count["offers_where_rankings_differ"] += int(item["ranking_changed"])
        count["opportunity_decisions"] += len(seen)
        count["prompts_asked"] += len(asked)
        count["prompt_responses"] += sum(d in responses for d in asked)
        count["prompt_accepted"] += sum(responses.get(d) is True for d in asked)
        count["prompt_rated"] += sum(d in p["ratings"] for d in asked)
        count["unwanted_prompts"] += sum(p["ratings"].get(d) is False for d in asked)
        # No sensor input cannot be interpreted as zero unwanted prompts. Preserve
        # all periods/counters separately; only the burden rate has this minimum.
        if p["active_seconds"] > 0 and covered >= 0.5 * p["active_seconds"]:
            count["burden_eligible_periods"] += 1
            primary_seconds += p["active_seconds"]
            primary_unwanted += sum(p["ratings"].get(d) is False for d in asked)
            primary_missing += sum(d not in p["ratings"] for d in asked)
        for a in p["activities"]:
            # A request dismissed before starting a named activity is not an activity.
            if not a.get("activity") or not _valid(a.get("pre")):
                continue
            count["activities_started"] += 1
            if a.get("status") != "completed":
                continue
            count["activities_completed"] += 1
            if _valid(a.get("post")):
                immediate.append(a["pre"] - a["post"])
            if a.get("followup_status") in ("pending", "completed", "skipped"):
                count["followup_scheduled"] += 1
            if a.get("followup_status") in ("pending", "skipped"):
                count["followup_" + a["followup_status"]] += 1
            if a.get("followup_status") == "completed" and _valid(a.get("followup_distress")):
                delayed.append(a["pre"] - a["followup_distress"])
    hours = active_seconds / 3600
    primary_hours = primary_seconds / 3600
    asked, rated = count["prompts_asked"], count["prompt_rated"]
    started, completed, scheduled = (
        count[k] for k in ("activities_started", "activities_completed", "followup_scheduled")
    )
    return dict(
        periods=len(periods),
        active_hours=hours,
        observed_hours=observed_seconds / 3600,
        burden_eligible_hours=primary_hours,
        burden_known_unwanted=primary_unwanted,
        burden_unrated=primary_missing,
        **{
            k: count[k]
            for k in (
                "recommendation_offers",
                "offers_where_rankings_differ",
                "burden_eligible_periods",
                "followup_pending",
                "followup_skipped",
                "opportunity_decisions",
                "prompts_asked",
                "prompt_rated",
                "unwanted_prompts",
                "prompt_responses",
                "prompt_accepted",
                "activities_started",
                "activities_completed",
                "followup_scheduled",
            )
        },
        prompt_rating_missing=asked - rated,
        prompt_response_missing=asked - count["prompt_responses"],
        unwanted_per_hour=_ratio(primary_unwanted, primary_hours) if primary_missing == 0 else None,
        unwanted_per_hour_bounds=[
            _ratio(primary_unwanted, primary_hours),
            _ratio(primary_unwanted + primary_missing, primary_hours),
        ],
        acceptance_rate=_ratio(count["prompt_accepted"], asked),
        completion_rate=_ratio(completed, started),
        post_valid=len(immediate),
        post_missing=completed - len(immediate),
        post_missing_all_starts=started - len(immediate),
        followup_missing_all_starts=started - len(delayed),
        immediate_mean_delta=_ratio(sum(immediate), len(immediate)),
        followup_valid=len(delayed),
        followup_missing=scheduled - len(delayed),
        followup_not_scheduled=completed - scheduled,
        followup_mean_delta=_ratio(sum(delayed), len(delayed)),
    )


def analyze(periods, mode="real", flow=None):
    if mode not in ("real", "simulation"):
        raise ValueError("mode must be real or simulation")
    selected = [p for p in periods if p["mode"] == mode]
    completed = [p for p in selected if p["active_seconds"] is not None]
    users = defaultdict(dict)
    for p in completed:
        users[p["user_id"]][p["condition"]] = summary([p])
    both = [u for u in users.values() if all(a in u for a in ARMS)]
    paired = {}
    for metric in METRICS:
        ds = [
            u["adaptive"][metric] - u["basic"][metric]
            for u in both
            if all(u[arm][metric] is not None for arm in ARMS)
        ]
        ci = None
        if len(ds) >= 2:
            rng = random.Random(20261120)
            estimates = sorted(sum(rng.choices(ds, k=len(ds))) / len(ds) for _ in range(2000))
            ci = [estimates[49], estimates[1949]]
        paired[metric] = {"n": len(ds), "mean_difference": mean(ds) if ds else None, "ci95": ci}
    active_n = len({p["user_id"] for p in selected})
    flow = flow or {"randomized": active_n, "withdrawn": 0, "periods_allocated": len(selected)}
    n = flow["randomized"] - flow["withdrawn"]
    return {
        "protocol": VERSION,
        "mode": mode,
        "participants": n,
        "randomized": flow["randomized"],
        "withdrawn": flow["withdrawn"],
        "participants_without_complete_pair": n - len(both),
        "paired_participants": len(both),
        "periods_assigned": flow["periods_allocated"],
        "periods_incomplete": flow["periods_allocated"] - len(completed),
        "periods_completed": len(completed),
        "sequence_counts": {
            s: len({p["user_id"] for p in selected if p["sequence"] == s}) for s in ("AB", "BA")
        },
        "by_condition": {a: summary([p for p in completed if p["condition"] == a]) for a in ARMS},
        "by_sequence_period": {
            s: {
                str(i): summary([p for p in completed if p["sequence"] == s and p["period"] == i])
                for i in (1, 2)
            }
            for s in ("AB", "BA")
        },
        "paired": paired,
    }


def write_report(path, report):
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return dest
