"""Forward-only personal-reference audit on exported sessions and optional held-out labels."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotion_assistant.personalization import build_reference, compare_reference, matches_final_revision


def _timestamp(value):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Session timestamps must include a timezone")
    return stamp


def _rate(numerator, denominator):
    return numerator/denominator if denominator else None


def _annotations(rows, field, identity):
    result = {}
    for row in rows:
        if field not in row:
            continue
        key = (row["session_id"], row[identity])
        if type(row[field]) is not bool or key in result:
            raise ValueError(f"Expected one boolean {field} per {identity}")
        result[key] = row[field]
    return result


def _samples(export, observations):
    for item in export.get("observation_feedback", []):
        if item.get("reference_candidate") is not True or not item.get("learning_consent"):
            continue
        evidence = item["evidence"]
        raw = [observations.get(key, {}) for key in evidence["source_ids"]]
        if (all(r.get("valid") and r.get("modality") in ("visual", "audio") for r in raw)
                and matches_final_revision(item, observations.get(evidence["event_id"], {}))):
            yield item


def evaluate(exports, labels=()):
    """Measure later sessions only against references from already finished, same-profile sessions."""
    labels = list(labels)
    feelings = _annotations(labels, "above_usual", "event_id")
    timing = _annotations(labels, "prompt_wanted", "decision_id")
    past, seen, sessions = defaultdict(list), set(), []
    for export in sorted(exports, key=lambda e: _timestamp(e["session"]["started_at"])):
        session = export["session"]
        sid = session["id"]
        if sid in seen:
            raise ValueError(f"Duplicate session export: {sid}")
        seen.add(sid)
        if not session.get("ended_at") or session["mode"] not in ("simulation", "real"):
            continue
        observations = {item["event_id"]: item for item in export["observations"]}
        feedback = [f for f in export.get("observation_feedback", []) if f["learning_consent"]]
        held_out = [o for o in observations.values() if isinstance(o.get("personal_reference"), dict)
                    and o["personal_reference"].get("status") not in ("disabled", "paused", "reset")]
        if not held_out and not feedback:
            continue
        start, end = _timestamp(session["started_at"]), _timestamp(session["ended_at"])
        if end < start:
            raise ValueError("Session ends before it starts")
        scope = (session["user_id"], session["mode"])
        prior = [f for ended, items in past[scope] if ended <= start for f in items]
        prior.sort(key=lambda r: (r["created_at"], r["feedback_id"]), reverse=True)
        snapshot = build_reference(prior[:512])
        covered = annotated = reference_errors = fixed_errors = 0
        for event in held_out:
            result = compare_reference(snapshot, event)
            if result["status"] == "insufficient":
                continue
            covered += 1
            target = feelings.get((sid, event["event_id"]))
            if target is not None:
                annotated += 1
                reference_errors += (result["status"] == "above_reference") != target
                fixed_errors += (event["scores"]["negative"] >= .65) != target
        asked = {e["payload"]["decision_id"] for e in export["events"]
                 if e["kind"] == "prompt_decision" and e["payload"].get("action") == "ask"}
        rated = [timing[sid, d] for d in asked if (sid, d) in timing]
        aligned = [f for f in feedback if f["target_event_ids"]]
        corrections = {f["target_event_ids"][0] for f in aligned if f["feedback_type"] == "subjective_correction"}
        feedback_targets = {f["target_event_ids"][0] for f in aligned}
        sessions.append({"session_id": sid, "mode": session["mode"], "prior_reference_version": snapshot["version"],
            "authorized_fusion_events": len(held_out), "comparable_events": covered,
            "coverage": _rate(covered, len(held_out)), "independently_labeled_events": annotated,
            "personal_reference_error_rate": _rate(reference_errors, annotated),
            "fixed_065_error_rate_on_same_events": _rate(fixed_errors, annotated),
            "automatic_prompts": len(asked), "explicitly_labeled_prompts": len(rated),
            "unwanted_rate_among_labeled_prompts": _rate(rated.count(False), len(rated)),
            "corrected_feedback_targets": len(corrections), "all_feedback_targets": len(feedback_targets),
            "corrected_fraction_of_feedback_targets": _rate(len(corrections), len(feedback_targets)),
            "feedback_actions": len(feedback), "feedback_actions_per_session_hour":
                _rate(len(feedback), (end-start).total_seconds()/3600)})
        past[scope].append((end, list(_samples(export, observations))))
    return {"evaluation": "closed_prior_sessions_to_future_sessions", "sessions": sessions,
            "scope": "Descriptive audit. A missing label is unknown. Prompt timing was observed, not randomized."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exports", type=Path, nargs="+", help="Session JSON exports from History")
    parser.add_argument("--labels", type=Path, help="JSON list: session_id + event_id + above_usual (bool), "
                        "or session_id + decision_id + prompt_wanted (bool)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    inputs = args.exports + ([args.labels] if args.labels else [])
    if args.out.resolve() in {p.resolve() for p in inputs}:
        parser.error("Choose an output separate from the input exports and labels")
    read = lambda path: json.loads(path.read_text(encoding="utf-8-sig"))
    result = evaluate([read(p) for p in args.exports], read(args.labels) if args.labels else [])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
