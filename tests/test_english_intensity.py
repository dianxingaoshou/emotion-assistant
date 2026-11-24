import pytest

from emotion_assistant import english_intensity as intensity
from scripts.fit_english_intensity import regression_metrics


@pytest.mark.parametrize("text", [None, "", "中文很好", "alone", "zxzzx ttyqp"])
def test_unsupported_or_inadequate_text_has_no_numeric_intensity(text):
    assert intensity.predict_english_intensity(text) is None


def test_signed_model_score_and_provenance():
    positive = intensity.predict_english_intensity("I really love this amazing movie")
    negative = intensity.predict_english_intensity("This is horrible and stupid")
    assert -3 <= negative["value"] < 0 < positive["value"] <= 3
    assert positive["model_version"] == negative["model_version"]
    assert positive["used_modalities"] == ["text"]


def test_clipped_values_and_no_spurious_bigram_across_deletion():
    model = {"model_version": "fixture", "intercept": 0., "terms": {
        "never": [1., 0.], "again": [1., 0.], "never again": [1., -10.]}}
    assert intensity.predict_segments_with_model(["never again"], model)["value"] == -3.
    assert intensity.predict_segments_with_model(["never", "again"], model)["value"] == 0.


def test_regression_metrics_with_constant_target_or_estimates():
    assert regression_metrics([], []) == {"n": 0, "mae": None, "pearson": None}
    assert regression_metrics([-1, 1], [0, 0]) == {"n": 2, "mae": 1., "pearson": None}
    assert regression_metrics([-1, 1], [-.5, .5])["pearson"] == pytest.approx(1.)


def test_corrupt_word_model_abstains_for_manual_and_automatic_transcripts(tmp_path, monkeypatch):
    packed = bytearray(intensity.ASSET.read_bytes())
    packed[10] ^= 255
    bad = tmp_path / "corrupt.json.gz"
    bad.write_bytes(packed)
    monkeypatch.setattr(intensity, "ASSET", bad)
    intensity._bundled.cache_clear()
    try:
        assert intensity.predict_english_intensity("I love this movie") is None
    finally:
        intensity._bundled.cache_clear()
