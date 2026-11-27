import json

import pytest

from emotion_assistant import chinese_affect, research_text_review


@pytest.mark.parametrize("ratings,status", [
    ({"valence": 8, "arousal": 3, "evidence": "我很安心"}, "ok"),
    ({"valence": 0, "arousal": 0, "evidence": ""}, "insufficient_evidence"),
    ({"valence": 7, "arousal": 0, "evidence": "安心"}, "invalid_response"),
    ({"valence": True, "arousal": 2, "evidence": "安心"}, "invalid_response"),
    ({"valence": 8, "arousal": 2, "evidence": "非常放松"}, "invalid_response"),
])
def test_ratings_require_evidence_and_two_valid_axes(ratings, status):
    result = chinese_affect.validate_ratings(ratings, "我很安心。", {})
    assert result["status"] == status
    assert ("arousal" in result) == (status == "ok")


def test_chinese_review_uses_selected_text_and_keeps_semantics_on_failure(monkeypatch):
    text = "我很安心。"
    existing = {"status": "ok", "subject": "self", "current_polarity": "positive"}
    monkeypatch.setattr(research_text_review.semantic_review, "review_text", lambda _: existing.copy())
    calls = []

    def fail(selected):
        calls.append(selected)
        return {"status": "invalid_response"}

    monkeypatch.setattr(research_text_review, "rate_chinese_expression", fail)
    result = research_text_review.review_with_reference(text)
    assert calls == [text]
    assert result["current_polarity"] == existing["current_polarity"]
    assert result["chinese_dimensions"]["status"] == "invalid_response"


def test_no_request_for_unsupported_or_overlong_text(monkeypatch):
    monkeypatch.setattr(chinese_affect, "HTTPConnection", lambda *_args, **_kwargs: pytest.fail("unexpected call"))
    assert chinese_affect.rate_chinese_expression("Just an English sentence")["status"] == "unsupported"
    assert chinese_affect.rate_chinese_expression("平静" * 257)["status"] == "too_long"


def test_local_request_checks_envelope(monkeypatch):
    response = {"done": True, "message": {"content": json.dumps({"valence": 7, "arousal": 2, "evidence": "安心"})}}

    class Connection:
        def __init__(self, host, port, timeout):
            assert host == "127.0.0.1" and port == 11434 and timeout > 0

        def request(self, method, path, payload, headers):
            assert method == "POST" and path == "/api/chat"
            assert json.loads(json.loads(payload)["messages"][-1]["content"]) == {"text": "觉得安心。"}

        def getresponse(self):
            return type("Response", (), {"status": 200, "read": lambda s, n: json.dumps(response).encode()})()

        def close(self):
            pass

    monkeypatch.setattr(chinese_affect, "HTTPConnection", Connection)
    assert chinese_affect.rate_chinese_expression("觉得安心。")["arousal"] == 2
    response["done_reason"] = "length"
    assert chinese_affect.rate_chinese_expression("觉得安心。")["status"] == "invalid_response"
