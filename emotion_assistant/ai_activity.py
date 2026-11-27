"""Optional local selection from activities already allowed by the regulation policy."""

from dataclasses import dataclass
from http.client import HTTPConnection, HTTPException
import json

from .regulation import ACTIVITIES, NEEDS
from .semantic_review import MODEL, MAX_RESPONSE_BYTES, TIMEOUT_SECONDS


@dataclass(frozen=True)
class ActivityContext:
    need: str
    seconds: int
    distress: int
    eligible: tuple[str, ...]
    preferred: frozenset[str]


def activity_context(need, seconds, distress, items, preferences):
    if (need not in NEEDS or type(seconds) is not int or seconds <= 0
            or type(distress) is not int or not 0 <= distress <= 10):
        return None
    eligible = []
    for item in items:
        key = item["activity"]
        if (key in ACTIVITIES and key not in eligible and item.get("blocked") is False
                and not item.get("discomfort_count") and preferences.get(key) != "avoid"
                and ACTIVITIES[key]["min_seconds"] <= seconds):
            eligible.append(key)
    if eligible:
        return ActivityContext(need, seconds, distress, tuple(eligible),
                               frozenset(key for key in eligible if preferences.get(key) == "prefer"))
    return None


def recommendation_reason(key, context):
    activity = ACTIVITIES[key]
    reasons = [activity["summary"]]
    if context.need != "unsure" and context.need in activity["needs"]:
        reasons.append("与本次所选需要相符。")
    if key in context.preferred:
        reasons.append("你已将它设为偏好。")
    reasons.append(f"可在你预留的 {context.seconds // 60} 分钟内尝试。")
    return "".join(reasons)


def recommend_with_ai(context):
    """Send just the self-report and approved catalog options to the local Ollama endpoint."""
    candidates = [{"id": key, "description": ACTIVITIES[key]["summary"],
                   "preferred": key in context.preferred, "matches_goal": context.need in ACTIVITIES[key]["needs"],
                   "usual_seconds": ACTIVITIES[key]["default_seconds"]} for key in context.eligible]
    data = {"confirmed_goal": NEEDS[context.need], "available_seconds": context.seconds,
            "self_rated_distress_0_to_10": context.distress, "candidates": candidates}
    schema = {"type": "object", "properties": {"activities": {"type": "array",
              "items": {"type": "string", "enum": list(context.eligible)}, "minItems": 1,
              "maxItems": min(2, len(candidates)), "uniqueItems": True}},
              "required": ["activities"], "additionalProperties": False}
    system = ("The speaker confirmed they want a brief voluntary activity. Select one or two unique "
              "eligible catalog IDs suitable for their selected goal, available time and explicit preference. "
              "Their self-rating is context only, not a diagnosis. Return only the requested IDs in order. "
              "Do not offer instructions or clinical advice.")
    request = {"model": MODEL, "stream": False, "think": False, "format": schema,
               "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 128},
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": json.dumps(data, ensure_ascii=False)}]}
    connection = HTTPConnection("127.0.0.1", 11434, timeout=TIMEOUT_SECONDS)
    try:
        connection.request("POST", "/api/chat", json.dumps(request).encode("utf-8"),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            return {"status": "model_missing" if response.status == 404 else "service_error"}
        body = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            return {"status": "invalid_response"}
        envelope = json.loads(body)
        if (not isinstance(envelope, dict) or envelope.get("done") is not True
                or envelope.get("done_reason") == "length"):
            return {"status": "invalid_response"}
        result = json.loads(envelope["message"]["content"])
        if not isinstance(result, dict) or set(result) != {"activities"}:
            return {"status": "invalid_response"}
        choices = result["activities"]
        if (not isinstance(choices, list) or not 1 <= len(choices) <= 2
                or any(key not in context.eligible for key in choices) or len(set(choices)) != len(choices)):
            return {"status": "invalid_response"}
        return {"status": "ok", "model": MODEL, "activities": choices}
    except TimeoutError:
        return {"status": "timeout"}
    except (OSError, HTTPException):
        return {"status": "unavailable"}
    except (ValueError, TypeError, KeyError, RecursionError):
        return {"status": "invalid_response"}
    finally:
        connection.close()
