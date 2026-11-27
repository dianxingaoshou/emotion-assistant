import hashlib

import pytest

from emotion_assistant import fine_emotions as fine


def test_bundled_model_exposes_multiple_independent_labels():
    result = fine.analyze_fine_emotions("Thank you so much, I love it!")
    assert result["status"] == "ok"
    assert result["scope"] == "expressed_text"
    assert result["score_type"] == "independent_label_score"
    labels = result["labels"]
    assert {item["id"] for item in labels} == {"gratitude", "love"}
    assert [item["score"] for item in labels] == sorted((item["score"] for item in labels), reverse=True)
    assert all(item["threshold"] <= item["score"] <= 1 for item in labels)
    assert hashlib.sha256(fine.ASSET.read_bytes()).hexdigest() == fine.ASSET_SHA256


@pytest.mark.parametrize(("text", "status"), [
    ("", "no_text"), (None, "no_text"), ("谢谢你帮我", "unsupported_language"),
    ("qzxqzz kqxqqx", "insufficient_evidence"), ("and the in of", "insufficient_evidence"),
    ("ok " * 5700, "too_long"),
])
def test_text_without_usable_evidence_returns_no_labels(text, status):
    result = fine.analyze_fine_emotions(text)
    assert result["status"] == status
    assert result["labels"] == []


def test_no_label_reaches_threshold_is_distinct_from_neutral(monkeypatch):
    model = {**fine._bundled_model(), "thresholds": [1.] * len(fine.LABEL_NAMES)}
    monkeypatch.setattr(fine, "_bundled_model", lambda: model)
    result = fine.analyze_fine_emotions("I like it")
    assert result["status"] == "below_threshold" and result["labels"] == []


def test_missing_or_replaced_asset_abstains(tmp_path, monkeypatch):
    path = tmp_path / "replaced.npz"
    path.write_bytes(b"wrong digest")
    monkeypatch.setattr(fine, "ASSET", path)
    fine._bundled_model.cache_clear()
    try:
        assert fine.analyze_fine_emotions("I feel happy")["status"] == "model_unavailable"
    finally:
        fine._bundled_model.cache_clear()
