import pytest

from emotion_assistant.english_polarity import CLASSES, predict_english_polarity


@pytest.mark.parametrize("text", [None, "", "123 ???", "我最近很开心", "lovely", "xzzvr zxjxxw"])
def test_insufficient_or_unsupported_text_abstains(text):
    assert predict_english_polarity(text) is None


@pytest.mark.parametrize(("text", "label"), [
    ("I absolutely love this amazing movie.", "positive"),
    ("This is horrible and stupid.", "negative"),
])
def test_bundled_classifier_returns_three_normalized_scores(text, label):
    output = predict_english_polarity(text)
    assert output["label"] == label
    assert output["model_version"].startswith("competition-en-polarity-tfidf-")
    assert set(output["scores"]) == set(CLASSES)
    assert sum(output["scores"].values()) == pytest.approx(1.)


def test_coverage_counts_rejections_as_misses_in_full_population():
    from scripts.fit_english_polarity import evaluate
    from emotion_assistant.english_polarity import _bundled

    rows = [("video1", "xzzvr zxjxxw", "negative"), ("video2", "This is horrible and stupid.", "negative")]
    report = evaluate(rows, _bundled(), "positive")
    assert report["coverage"] == .5
    assert report["conditional_on_score"]["accuracy"] == 1
    assert report["including_abstention_as_miss"]["accuracy"] == .5
    assert report["confusion_rows_truth_columns_negative_neutral_positive_abstain"][0] == [1, 0, 0, 1]
