import pytest
import gzip
import hashlib
import json
from pathlib import Path

from emotion_assistant import file_asr_intensity as files
from emotion_assistant import asr_word_intensity as characters
from emotion_assistant.asr_word_intensity import bundled_char, char_terms, robust_value
from emotion_assistant.english_intensity import _bundled


def segment(text, start=1., end=3., **extra):
    return {"valid": True, "session_id": "clip", "t_start": start, "t_end": end, "text": text, **extra}


def test_file_score_uses_sorted_current_words_and_unique_speech_duration():
    rows = [segment("this movie", 2, 4), segment("I really love"), segment("I hate it", valid=False),
            segment("这是错误", stale=True), segment("讨厌", session_id="previous")]
    result = files.estimate_file_intensity(rows, source_id="clip", duration=10.)
    assert result["covered_seconds"] == 3.
    assert result["intensity"]["value"] == robust_value("I really love this movie", _bundled(), bundled_char())
    assert result["intensity"]["used_modalities"] == ["text"]
    assert result["intensity"]["method"] == files.METHOD
    assert bundled_char()["model_version"] in result["intensity"]["model_version"]


@pytest.mark.parametrize("rows,duration", [
    ([], 2.), ([segment("one")], 2.), ([segment("zxczx qqqxxff")], 2.),
    ([segment("I love it"), segment("你好")], 2.), ([segment("I love it", asr_language="zh")], 2.),
    ([segment("I love it", t_end=float("nan"))], 2.), ([segment("I love it")], 30.01),
    ([segment("I love it")], None)])
def test_file_score_abstains_outside_evaluated_scope(rows, duration):
    assert files.estimate_file_intensity(rows, source_id="clip", duration=duration) is None


def test_unknown_character_asset_retains_old_result(monkeypatch):
    monkeypatch.setattr(files, "bundled_char", lambda: None)
    assert files.estimate_file_intensity([segment("great movie")], source_id="clip", duration=6.) is None


def test_packaged_character_asset_matches_evaluated_weights_and_rejects_tampering(tmp_path, monkeypatch):
    report = json.loads((Path(__file__).resolve().parents[1] / "docs/results/asr-intensity-upgrade.json").read_text("utf-8"))
    assert hashlib.sha256(characters.ASSET.read_bytes()).hexdigest() == report["asset"]["sha256"]
    packed = bytearray(characters.ASSET.read_bytes())
    doc = dict(bundled_char(), intercept=123.)
    damaged = tmp_path / "wrong.json.gz"
    damaged.write_bytes(gzip.compress(json.dumps(doc).encode()))
    monkeypatch.setattr(characters, "ASSET", damaged)
    bundled_char.cache_clear()
    try:
        assert bundled_char() is None
        packed[10] ^= 255
        damaged.write_bytes(packed)
        bundled_char.cache_clear()
        assert bundled_char() is None
    finally:
        bundled_char.cache_clear()


def test_character_features_keep_word_boundaries_and_scoring_uses_equal_weights():
    assert list(char_terms("I")) == [" i "]
    assert "t b" not in char_terms("cat bat")
    word = {"intercept": -.4, "terms": {"good": [1., 0.], "film": [1., 0.]}, "model_version": "word"}
    char = {"intercept": 0., "terms": {"goo": [1., 1.4]}}
    assert robust_value("good film", word, char) == pytest.approx(.5)
