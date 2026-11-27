import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from emotion_assistant import chinese_dimensions as zh
from emotion_assistant.chinese_affect_calibration import FITTED_MODEL, FITTED_RATER, apply_chinese_calibration


def raw_review():
    return {"status": "ok", "model": FITTED_MODEL, "version": FITTED_RATER,
            "scope": "expressed_text", "scale": [1, 9], "valence": 8, "arousal": 7}


def test_bundled_estimates_work_in_either_script_offline():
    simplified = zh.predict_chinese_dimensions("我真的很开心，很满足。")
    traditional = zh.predict_chinese_dimensions("我真的很開心，很滿足。")
    assert simplified == traditional
    assert simplified["status"] == "ok" and simplified["scale"] == [1, 9]
    assert all(1 <= simplified["dimensions"][axis]["score"] <= 9 for axis in zh.AXES)
    positive = zh.predict_chinese_dimensions("我今天真的很高兴，谢谢你支持我。")
    worried = zh.predict_chinese_dimensions("我特别害怕这一切都失控了。")
    assert positive["dimensions"]["valence"]["score"] > worried["dimensions"]["valence"]["score"]


@pytest.mark.parametrize("text,status", [
    (None, "unsupported_language"), ("only English", "unsupported_language"),
    ("今日嬉しいです", "unsupported_language"), ("龘靐", "insufficient_evidence"), ("好" * 513, "too_long"),
])
def test_unsupported_content_does_not_receive_baseline_mean(text, status):
    result = zh.predict_chinese_dimensions(text)
    assert result["status"] == status and not result["dimensions"]


def test_bundle_checksum_matches_heldout_report_and_never_loads_unverified_weights(tmp_path, monkeypatch):
    report_path = Path(__file__).resolve().parents[1] / "docs/results/chinese-dimension-hybrid.json"
    report = json.loads(report_path.read_text(encoding="utf8"))
    assert zh.ASSET_SHA256 == hashlib.sha256(zh.ASSET.read_bytes()).hexdigest() == report["weights_sha256"]
    assert report["ridge_export_vs_sklearn_max_abs_delta"] < 1e-5
    assert report["fresh_confirmation"]["attempted"] == 160
    assert report["deduplicated_test_size"] == 160
    with np.load(zh.ASSET, allow_pickle=False) as data:
        assert "Apache License" in str(data["opencc_license"])
        assert "Chinese EmoBank" in str(data["source_attribution"])
    corrupt = tmp_path / "bad.npz"
    corrupt.write_bytes(b"truncated")
    monkeypatch.setattr(zh, "ASSET", corrupt)
    zh.bundled_model.cache_clear()
    try:
        assert zh.predict_chinese_dimensions("让我很高兴")["status"] == "model_unavailable"
    finally:
        zh.bundled_model.cache_clear()


def test_research_mixture_preserves_components_and_uses_each_calibration_weight():
    offline = zh.predict_chinese_dimensions("我真的很开心")
    review = apply_chinese_calibration(raw_review())
    combined = zh.blend_chinese_dimensions(offline, review)
    assert combined["model"] == zh.HYBRID and combined["status"] == "ok"
    for axis, weight in zip(zh.AXES, zh.bundled_model()["mixture"]):
        expected = (1-weight)*offline["dimensions"][axis]["score"] + weight*review["calibrated"][axis]
        assert combined["dimensions"][axis]["score"] == pytest.approx(expected, abs=0.000051)
        assert combined["dimensions"][axis]["qwen_weight"] == round(weight, 4)
    assert 0 < combined["dimensions"]["arousal"]["qwen_weight"] < combined["dimensions"]["valence"]["qwen_weight"] < 1
    assert review["valence"] == 8 and offline["model"] == zh.MODEL


def test_missing_unversioned_or_corrupt_ratings_cannot_be_combined():
    offline = zh.predict_chinese_dimensions("我真的很开心")
    review = apply_chinese_calibration(raw_review())
    for modified in (None, {}, raw_review(), {**review, "version": "older"},
                     {**review, "calibration_version": "unexpected"},
                     {**review, "calibrated": {"valence": float("nan"), "arousal": 6}},
                     {**review, "valence": True}, {**review, "valence": 2},
                     {**review, "scope": "personal_self_report"}):
        assert zh.blend_chinese_dimensions(offline, modified)["status"] == "insufficient_evidence"
    assert zh.blend_chinese_dimensions({**offline, "status": "too_long"}, review)["dimensions"] == {}
