"""Evaluate the same Chinese utterances with the production rules and local semantic reviewer."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotion_assistant.semantic_context import chinese_context_rules
from emotion_assistant.semantic_review import MODEL, REVIEW_VERSION, review_text


# Two cases per pair. The reference concerns whose feeling is described, when,
# the speaker's explicitly stated current polarity, and any explicit request.
CASES = [
    ("紧张", "我现在很紧张。", "self", "current", "negative", "none"),
    ("紧张", "同桌现在很紧张。", "other", "current", "unknown", "none"),
    ("欣喜", "我此刻满心欢喜。", "self", "current", "positive", "none"),
    ("欣喜", "我昨天满心欢喜。", "self", "past", "unknown", "none"),
    ("恼火", "我眼下对这件事感到恼火。", "self", "current", "negative", "none"),
    ("恼火", "倘若再发生这件事，我会感到恼火。", "self", "hypothetical", "unknown", "none"),
    ("称赞", "这款键盘手感非常好。", "object", "unknown", "unknown", "none"),
    ("称赞", "我目前的心情非常好。", "self", "current", "positive", "none"),
    ("难过", "我此刻很难过，能陪我坐一会儿吗？", "self", "current", "negative", "yes"),
    ("难过", "我此刻很难过，请别陪我，让我独处。", "self", "current", "negative", "declined"),
    ("伤心", "我上周很伤心。", "self", "past", "unknown", "none"),
    ("伤心", "我上周很伤心，现在我感到快乐。", "self", "mixed", "positive", "none"),
    ("放松", "我此刻全身放松，心里很愉悦。", "self", "current", "positive", "none"),
    ("放松", "我希望以后能全身放松、心里愉悦。", "self", "future", "unknown", "none"),
    ("引述", "我此刻心里很害怕。", "self", "current", "negative", "none"),
    ("引述", "室友说：“我此刻心里很害怕。”", "other", "current", "unknown", "none"),
    ("安心", "此刻我非常安心。", "self", "current", "positive", "none"),
    ("安心", "此刻小周非常安心。", "other", "current", "unknown", "none"),
    ("气愤", "我此刻很气愤，请帮我整理思路。", "self", "current", "negative", "yes"),
    ("气愤", "我此刻很气愤，但我不需要任何协助。", "self", "current", "negative", "declined"),
    ("不安", "假如明天独自上台，我会不安。", "self", "hypothetical", "unknown", "none"),
    ("不安", "想到明天要上台，我此刻已经感到不安。", "self", "current", "negative", "none"),
    ("自豪", "我此刻为完成这项任务而感到自豪。", "self", "current", "positive", "none"),
    ("自豪", "教练此刻为完成这项任务而感到自豪。", "other", "current", "unknown", "none"),
]
FIELDS = ("subject", "time_scope", "current_polarity", "help_request")


def evaluate(case):
    pair, text, *reference = case
    rules = chinese_context_rules(text)
    review = review_text(text)
    return {"pair": pair, "text": text, "expected": dict(zip(FIELDS, reference)),
            "rules": {key: rules[key] for key in FIELDS[:2]}, "qwen": review}


def summarize(rows, method, fields):
    def match(row, key):
        return row[method].get(key) == row["expected"][key]
    return {"by_field": {k: sum(match(row, k) for row in rows) for k in fields},
            "all_scored_fields": sum(all(match(row, k) for k in fields) for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/chinese-context-pairs.json")
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(evaluate, CASES))
    if any(row["qwen"]["status"] in {"unavailable", "model_missing", "timeout", "service_error"} for row in rows):
        raise SystemExit("Local Qwen unavailable for at least one case. Re-run with the local service available.")
    score = {"n": len(rows), "pairs": len({row["pair"] for row in rows}),
             "validated_reviews": sum(row["qwen"]["status"] == "ok" for row in rows),
             "rules_shared_fields": summarize(rows, "rules", FIELDS[:2]),
             "qwen_shared_fields": summarize(rows, "qwen", FIELDS[:2]),
             "qwen_all_fields": summarize(rows, "qwen", FIELDS)}
    report = {"design": "Developer-authored Chinese paired sentences. Expected fields fixed before inference. "
                        "Cases share vocabulary within pairs. No population or clinical inference.",
              "model": MODEL, "review_version": REVIEW_VERSION,
              "cases_sha256": hashlib.sha256(json.dumps(CASES, ensure_ascii=False).encode()).hexdigest(),
              "summary": score, "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(score))


if __name__ == "__main__":
    main()
