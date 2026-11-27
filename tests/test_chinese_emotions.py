import json
from unittest.mock import Mock

import numpy as np

from emotion_assistant import chinese_emotion_review as review
from emotion_assistant import chinese_emotions as emotions


def test_desktop_chinese_recognizes_identical_expression_across_scripts():
    simplified = emotions.analyze_chinese_emotions("谢谢你")
    traditional = emotions.analyze_chinese_emotions("謝謝你")
    assert simplified == traditional
    assert simplified["status"] == "ok"
    assert [item["id"] for item in simplified["labels"]] == ["grateful"]
    assert simplified["score_type"] == "relative_class_score"
    assert len(simplified["top_scores"]) == 2


def test_desktop_abstentions_keep_evidence_separate_from_accepted_labels(monkeypatch):
    for text, status in (("大概就是这个意思。", "below_threshold"), ("早", "unsupported_language"),
                         ("thank you", "unsupported_language"), ("中国の学生", "unsupported_language"),
                         ("非常高兴" * 129, "too_long"), ("", "no_text")):
        result = emotions.analyze_chinese_emotions(text)
        assert result["status"] == status
        assert result["labels"] == []
    assert emotions.select_label(np.array([.47, .44, .09]), .45) is None
    assert emotions.select_label(np.array([.48, .23, .29]), .45) == 0
    monkeypatch.setattr(emotions, "_bundled_model", lambda: None)
    assert emotions.analyze_chinese_emotions("谢谢你")["status"] == "model_unavailable"


def test_review_only_accepts_one_real_label_and_a_verbatim_excerpt():
    base = {"label": None, "evidence": ""}
    assert review.validated_result({"label": "grateful", "evidence": "感谢"}, "我很感谢你。", base)["name_zh"] == "感激"
    for record in ({"label": "grateful", "evidence": "感激"}, {"label": [], "evidence": "谢谢"},
                   {"label": "abstain", "evidence": "谢谢"}, {"label": "other", "evidence": "谢谢"}):
        assert review.validated_result(record, "谢谢", base)["status"] == "invalid_response"
    assert review.validated_result({"label": "abstain", "evidence": ""}, "在吗", base)["status"] == "insufficient_evidence"


def test_local_review_checks_boundaries_and_model_envelope(monkeypatch):
    wire = Mock()
    wire.return_value.getresponse.return_value.status = 200
    monkeypatch.setattr(review, "HTTPConnection", wire)
    assert review.review_chinese_emotion("bonjour")["status"] == "unsupported_language"
    assert review.review_chinese_emotion("开心" * 257)["status"] == "too_long"
    wire.assert_not_called()
    payload = {"label": "grateful", "evidence": "谢谢"}
    response = wire.return_value.getresponse.return_value
    response.read.return_value = json.dumps({"done": True, "message": {"content": json.dumps(payload)}}).encode()
    result = review.review_chinese_emotion("谢谢你")
    assert result["status"] == "ok"
    assert result["name_zh"] == "感激" and result["evidence"] == "谢谢"
    assert wire.call_args.args == ("127.0.0.1", 11434)
    req = json.loads(wire.return_value.request.call_args.args[2])
    assert json.loads(req["messages"][-1]["content"]) == {"text": "谢谢你"}
    response.read.return_value = json.dumps({"done": True, "done_reason": "length", "message": {"content": "{}"}}).encode()
    assert review.review_chinese_emotion("谢谢你")["status"] == "invalid_response"
    response.status = 503
    assert review.review_chinese_emotion("谢谢你")["status"] == "service_error"
