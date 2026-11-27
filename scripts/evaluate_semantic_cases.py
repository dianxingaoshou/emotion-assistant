"""Check independently specified linguistic fields against the real local reviewer."""

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotion_assistant.semantic_review import FIELDS, MODEL, REVIEW_VERSION, review_text


# Each case scores only the fields whose meanings the stimulus makes explicit.
CASES = [
    {
        "id": "zh_object", "language": "zh",
        "text": "这盏台灯的开关设计得很糟糕。",
        "expected": {"subject": "object", "content_polarity": "negative",
                     "current_polarity": "unknown", "current_change": "unknown"},
    },
    {
        "id": "en_quotation", "language": "en",
        "text": "My neighbor said, ‘I feel miserable today.’",
        "expected": {"subject": "other", "current_polarity": "unknown",
                     "current_change": "unknown", "help_request": "none"},
    },
    {
        "id": "en_self_positive", "language": "en",
        "text": "I feel delighted right now.",
        "expected": {"subject": "self", "time_scope": "current",
                     "current_polarity": "positive", "current_change": "unknown"},
    },
    {
        "id": "zh_self_help", "language": "zh",
        "text": "我现在很难过。请帮帮我，陪我聊一会儿。",
        "expected": {"subject": "self", "current_polarity": "negative",
                     "current_change": "unknown", "help_request": "yes"},
    },
    {
        "id": "en_relative_improvement", "language": "en",
        "text": "Compared with an hour ago, I feel a little better.",
        "expected": {"current_polarity": "unknown", "current_change": "improving"},
    },
    {
        "id": "zh_past_and_present", "language": "zh",
        "text": "昨晚我很开心，可现在我觉得很伤心。",
        "expected": {"subject": "self", "time_scope": "mixed",
                     "current_polarity": "negative", "current_change": "worsening"},
    },
    {
        "id": "en_fact", "language": "en",
        "text": "The library opens at nine on weekdays.",
        "expected": {"content_polarity": "neutral", "current_polarity": "unknown",
                     "current_change": "unknown", "help_request": "none"},
    },
    {
        "id": "zh_unchanged_declines_help", "language": "zh",
        "text": "我现在的心情和早上一样，既不高兴也不难过。我不需要帮助。",
        "expected": {"current_polarity": "neutral", "current_change": "unchanged",
                     "help_request": "declined"},
    },
]


def main():
    matched = Counter()
    tested = Counter()
    rows = []
    for case in CASES:
        expected = case["expected"]
        for field, value in expected.items():
            assert value in FIELDS[field][1], (case["id"], field, value)
        actual = review_text(case["text"])
        verified = actual.get("status") == "ok"
        checks = {field: verified and actual.get(field) == value for field, value in expected.items()}
        tested.update(checks.keys())
        matched.update(field for field, passed in checks.items() if passed)
        row = {**case, "actual": actual, "passed_by_field": checks, "passed": all(checks.values())}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    summary = {
        "cases_passed": sum(row["passed"] for row in rows),
        "cases_total": len(rows),
        "fields": {field: {"passed": matched[field], "scored": total} for field, total in tested.items()},
    }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL, "review_version": REVIEW_VERSION,
        "design": "Hand-written bilingual challenge cases with expected fields defined before inference. "
                  "An exact match on every specified field passes the case. Service failures count as failures.",
        "cases_sha256": sha256(json.dumps(CASES, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        "summary": summary, "cases": rows,
    }
    output = ROOT / "reports" / "semantic-review" / "cases.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(output)
    return 0 if summary["cases_passed"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
