"""Exercise the desktop recommendation constraints with scripted contexts and a real local model."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emotion_assistant.ai_activity import activity_context, recommend_with_ai
from emotion_assistant.regulation import ACTIVITIES
from emotion_assistant.recommendation_policy import contextual_decision
from emotion_assistant.semantic_review import MODEL


# goal, available seconds, self-rating, preferences, activities previously marked uncomfortable
SCENARIOS = (
    ("tense", 120, 6, {}, ()),
    ("tense", 120, 6, {"breathing": "avoid"}, ("grounding",)),
    ("rumination", 180, 5, {"thought_label": "prefer"}, ()),
    ("stuck", 180, 5, {"task_split": "prefer"}, ()),
    ("fatigue", 60, 4, {"desk_reset": "prefer"}, ()),
    ("unsure", 60, 3, {}, ()),
    ("fatigue", 120, 7, {"break": "prefer"}, ("break",)),
    ("unsure", 120, 5, {key: "avoid" for key in ACTIVITIES}, ()),
)


def evaluate():
    records = []
    for need, seconds, distress, preferences, excluded in SCENARIOS:
        items, _ = contextual_decision([], need, seconds, distress, preferences, "simulation", excluded)
        context = activity_context(need, seconds, distress, items, preferences)
        result = recommend_with_ai(context) if context else {"status": "no_eligible_options"}
        offered = result.get("activities", [])
        eligible = list(context.eligible) if context else []
        valid = (1 <= len(offered) <= 2 and len(set(offered)) == len(offered)
                 and all(key in eligible and key not in excluded and preferences.get(key) != "avoid"
                         and ACTIVITIES[key]["min_seconds"] <= seconds for key in offered))
        correct_fallback = (not eligible and not offered and result["status"] == "no_eligible_options")
        records.append({"need": need, "seconds": seconds, "distress": distress, "preferences": preferences,
                        "prior_discomfort": list(excluded), "eligible": eligible, "rule_top_two": eligible[:2],
                        "status": result["status"], "ai_suggestions": offered,
                        "offer_or_fallback_valid": valid or correct_fallback,
                        "has_goal_match": any(need in ACTIVITIES[key]["needs"] for key in offered)})
    offered_rows = [row for row in records if row["ai_suggestions"]]
    return {"design": "Eight scripted contexts; real model calls only when eligible. No human outcomes.",
            "model": MODEL, "scenarios": len(records), "model_requests": sum(bool(r["eligible"]) for r in records),
            "offers_returned": len(offered_rows), "valid_offers_or_fallbacks": sum(r["offer_or_fallback_valid"] for r in records),
            "offers_with_goal_match": sum(r["has_goal_match"] for r in offered_rows),
            "distinct_activities_offered": sorted({key for row in offered_rows for key in row["ai_suggestions"]}),
            "records": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/ai-activity-offers.json")
    args = parser.parse_args()
    report = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, ensure_ascii=False))
