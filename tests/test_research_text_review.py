import pytest

from emotion_assistant import research_text_review as reference


@pytest.fixture
def model():
    bundled = reference.bundled_model()
    assert bundled is not None and bundled["label_order"] == list(reference.CLASSES)
    return bundled


@pytest.mark.parametrize("text", ["", "😀", "有一点失落。", "thanks", "17 22", "xyzxyz abcabc",
                                  "thanks for 中文情绪识别没有足够多的英文线索。"])
def test_reference_skips_unsupported_language_or_lexical_coverage(text, model):
    assert reference.predict_with_model(text, model) is None


def review_stub(polarity, text):
    return {"status": "ok", "subject": "object", "time_scope": "current",
            "content_polarity": polarity, "current_polarity": "unknown",
            "current_change": "unknown", "help_request": "none",
            "evidence": [text], "current_feeling_quote": ""}


@pytest.mark.parametrize("polarity,agreement,phrase", [
    ("positive", "agree", "判断一致"), ("negative", "disagree", "请核对原话"),
    ("unknown", "unresolved", "依据不足"),
])
def test_reference_augments_only_content_and_uses_user_visible_states(monkeypatch, model, polarity, agreement, phrase):
    text = "Thank you so much for the wonderful gift!"
    assert reference.predict_with_model(text, model) == "positive"
    original = review_stub(polarity, text)
    monkeypatch.setattr(reference.semantic_review, "review_text", lambda _: original)
    result = reference.review_with_reference(text)
    assert {k: v for k, v in result.items() if k != "english_reference"} == original
    assert result["english_reference"]["agreement"] == agreement
    assert result["english_reference"]["label"] == "positive"
    assert phrase in reference.format_research_review(result)
    assert "本人当前感受：证据不足" in reference.format_research_review(result)


def test_failed_review_does_not_fall_back_to_model(monkeypatch):
    monkeypatch.setattr(reference.semantic_review, "review_text", lambda _: {"status": "unavailable"})
    monkeypatch.setattr(reference, "bundled_model", lambda: pytest.fail("unexpected reference model lookup"))
    assert reference.review_with_reference("That is great") == {"status": "unavailable"}


def test_missing_asset_preserves_valid_review(monkeypatch):
    text = "This is good."
    original = review_stub("positive", text)
    monkeypatch.setattr(reference.semantic_review, "review_text", lambda _: original)
    monkeypatch.setattr(reference, "bundled_model", lambda: None)
    assert reference.review_with_reference(text) == original
    assert reference.format_research_review(original) == reference.semantic_review.format_review(original)


def test_hash_check_rejects_damaged_asset(tmp_path, monkeypatch):
    path = tmp_path / "asset.gz"
    path.write_bytes(b"not a valid asset")
    monkeypatch.setattr(reference, "ASSET", path)
    reference.bundled_model.cache_clear()
    try:
        assert reference.bundled_model() is None
    finally:
        reference.bundled_model.cache_clear()
