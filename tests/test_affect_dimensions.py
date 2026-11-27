import hashlib

import pytest

from emotion_assistant import affect_dimensions as vad


def test_asset_and_output_contract():
    model = vad.bundled_model()
    assert model["attribution"]["data_and_derived_parameters_license"] == "CC BY-SA 4.0"
    assert hashlib.sha256(vad.ASSET.read_bytes()).hexdigest() == vad.MODEL_SHA256
    result = vad.predict_affect_dimensions("I feel calm and relaxed.")
    assert result["status"] == "ok"
    assert result["scope"] == "expressed_text" and result["scale"] == [1, 5]
    assert result["interval_level"] == 0.8 and tuple(result["dimensions"]) == vad.AXES
    for estimate in result["dimensions"].values():
        lo, hi = estimate["interval"]
        assert 1 <= lo <= estimate["score"] <= hi <= 5


@pytest.mark.parametrize(("text", "reason"), [
    (None, "empty"), ("  ", "empty"), ("我今天没有在说英语", "unsupported_language"),
    ("42 ! ?", "unsupported_language"), ("zxyxy gggqqq", "insufficient_words"),
    ("happy", "insufficient_words"), ("calm " * 81, "too_long"),
])
def test_unsupported_text_has_no_scores(text, reason):
    result = vad.predict_affect_dimensions(text)
    assert result["status"] == reason and result["dimensions"] == {}


def test_unavailable_asset_preserves_abstention(monkeypatch):
    monkeypatch.setattr(vad, "bundled_model", lambda: None)
    result = vad.predict_affect_dimensions("This is a sentence.")
    assert result["status"] == "model_unavailable" and result["dimensions"] == {}


def test_sparse_scoring_uses_weighted_terms_and_respects_scale():
    model = {"intercept": [4, 2, 3], "features": {"w:calm": [2, 2, -2, 0.5]}}
    assert vad.score_with_model("calm", model) == [5, 1, 3.5]
    assert vad.score_with_model("?", model) == [4, 2, 3]
