"""Structured generation on the installed loopback-only Ollama service."""

from http.client import HTTPConnection, HTTPException
import json

from .semantic_review import MAX_RESPONSE_BYTES, MODEL, TIMEOUT_SECONDS


def request_json(system, payload, schema, *, num_predict=256, history=()):
    request = {
        "model": MODEL, "stream": False, "think": False, "format": schema,
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": num_predict},
        "messages": [{"role": "system", "content": system}, *history,
                     {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
    }
    conn = HTTPConnection("127.0.0.1", 11434, timeout=TIMEOUT_SECONDS)
    try:
        conn.request("POST", "/api/chat", json.dumps(request).encode(), {"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            return ("model_missing" if response.status == 404 else "service_error"), None
        body = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) <= MAX_RESPONSE_BYTES:
            envelope = json.loads(body)
            if isinstance(envelope, dict) and envelope.get("done") is True and envelope.get("done_reason") != "length":
                parsed = json.loads(envelope["message"]["content"])
                if isinstance(parsed, dict):
                    return "ok", parsed
        return "invalid_response", None
    except TimeoutError:
        return "timeout", None
    except (OSError, HTTPException):
        return "unavailable", None
    except (KeyError, ValueError, TypeError, RecursionError):
        return "invalid_response", None
    finally:
        conn.close()
