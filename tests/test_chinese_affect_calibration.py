import pytest

from emotion_assistant import chinese_affect_calibration as calibration
from emotion_assistant.chinese_affect import VERSION as RATER_VERSION
from emotion_assistant.semantic_review import MODEL


def rating(**override):
    return {"status": "ok", "model": MODEL, "version": RATER_VERSION, "scope": "expressed_text",
            "scale": [1, 9], "valence": 1, "arousal": 9, "evidence": "激动", **override}


def test_augments_only_own_source_and_retains_original_evidence():
    raw = rating()
    result = calibration.apply_chinese_calibration(raw)
    assert "calibrated" not in raw and all(result[key] == val for key, val in raw.items())
    assert result["calibration_version"] == calibration.VERSION and result["calibration_n"] == 188
    assert result["calibrated"]["valence"] == pytest.approx(2.8721406)
    assert result["calibrated"]["arousal"] == pytest.approx(6.4776197)
    assert calibration.apply_chinese_calibration(result) == result


@pytest.mark.parametrize("override", [
    {"version": "old"}, {"status": "insufficient_evidence"}, {"model": "different"},
    {"scope": "self_report"}, {"valence": None}, {"arousal": True}, {"arousal": 10},
])
def test_incompatible_or_missing_scores_do_not_receive_calibration(override):
    original = rating(**override)
    assert calibration.apply_chinese_calibration(original) == original
