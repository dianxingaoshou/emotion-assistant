"""Replay scripted Chinese turns through the production resolver, using the local Qwen service."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotion_assistant.regulation_chat import reply_to_turn
from emotion_assistant.semantic_review import MODEL


# These are authored interface scenarios. Expected actions are fixed before invoking Qwen.
SCENARIOS = {
    "remember_in_listening_mode": [
        ("我只想聊聊", "listen", "local"),
        ("明天我得上台汇报，最担心的是上次忘记翻到最后一张幻灯片。", "reply", "qwen"),
        ("刚才我提到自己忘了做哪件事？", "reply", "qwen"),
        ("停一下", "stop", "local"),
    ],
    "switch_during_guidance": [
        ("换一种", "switch", "local"),
    ],
    "current_discomfort": [
        ("这个练习让我不舒服", "discomfort", "local"),
    ],
    "context_and_indirect_request": [
        ("昨天同事跟我说“停一下”，但我现在还想继续跟你说。", "reply", "qwen"),
        ("这些方法都先收起来吧，今天我只需要有个人听我把事情讲完。", "listen", "qwen"),
    ],
    "new_memory_and_correction": [
        ("手工课上我给做的船起名叫银杏号，那是今天最开心的事。", "reply", "qwen"),
        ("刚想起来我记岔了，我最后给船取的是青梨号。", "reply", "qwen"),
        ("你还记得我最后决定用哪个名字吗？", "reply", "qwen"),
    ],
    "new_change_of_intent": [
        ("我现在需要的是一个能听我把话说完的人，暂时不准备做什么。", "listen", "qwen"),
        ("聊过之后我改主意了，可以帮我挑个别的练习吗？", "switch", "qwen"),
    ],
}


def evaluate():
    records = []
    for scenario, turns in SCENARIOS.items():
        listening = False
        history = []
        activity = "breathing" if scenario in {"switch_during_guidance", "current_discomfort"} else None
        for index, (text, action, source) in enumerate(turns, start=1):
            start = perf_counter()
            result = reply_to_turn(text, history, listening=listening, activity=activity)
            record = {
                "scenario": scenario, "turn": index, "input": text, "expected_action": action,
                "expected_source": source, "action": result.action, "source": result.source,
                "status": result.status, "reply": result.message, "seconds": round(perf_counter() - start, 3),
                "matches_action_and_source": result.status == "ok" and result.action == action and result.source == source,
            }
            if scenario == "remember_in_listening_mode" and index == 3:
                record["recalls_final_slide"] = ("最后" in result.message and
                    any(word in result.message for word in ("幻灯片", "PPT", "演示文稿", "投影片")))
            if scenario == "new_memory_and_correction" and index == 3:
                record["recalls_corrected_name"] = "青梨号" in result.message
            records.append(record)
            history.extend([("user", text), ("assistant", result.message)])
            if result.action == "listen":
                listening = True
            if result.action in {"listen", "stop", "switch", "discomfort"}:
                activity = None
    return {"design": "Six development scenarios via the desktop turn resolver. The first four also appear in the initial pilot. No independent holdout or human outcomes.",
            "model": MODEL, "scenarios": len(SCENARIOS), "turns": len(records),
            "expected_local": sum(r["expected_source"] == "local" for r in records),
            "expected_qwen": sum(r["expected_source"] == "qwen" for r in records),
            "matched": sum(r["matches_action_and_source"] for r in records),
            "records": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/regulation-chat.json")
    args = parser.parse_args()
    report = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}))
