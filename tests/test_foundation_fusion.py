import numpy as np

from emotion_assistant import foundation_fusion as fusion


def mock_head():
    asset = {}
    for k, n in (("base",2048),("qwen",2048),("audio",768)):
        asset[k + "_center"] = np.zeros(n)
        asset[k + "_coef"] = np.zeros((3,n))
        asset[k + "_bias"] = np.array([2.,0.,0.]) if k == "base" else np.array([0.,0.,2.])
    for variant in ("base", "no_qwen", "foundation"):
        asset[variant + "_weights"] = np.array([1.,1. if variant != "base" else 0.,.2,.5])
    return asset


def test_text_representation_switches_the_equal_capacity_head():
    model = mock_head()
    offline = fusion.predict_with_head(model, "明天去学校")
    assert offline["label"] == "negative" and not offline["foundation_used"]
    found = fusion.predict_with_head(model, "明天去学校", embedding=np.ones(1024))
    assert found["label"] == "positive" and found["foundation_used"]
    bad = fusion.predict_with_head(model, "明天去学校", embedding=np.full(1024,np.nan))
    assert bad == offline
    hidden = fusion.predict_with_head(model, "", embedding=np.ones(1024))
    assert hidden["label"] is None


def test_acoustic_prediction_works_without_text_and_visual_only_renormalizes():
    model = mock_head()
    visual = {"scores":{"positive":.05,"negative":.1,"neutral":.8,"uncertain":.05},
              "quality":.7,"coverage":.2}
    face = fusion.predict_with_head(model, "", visual=visual)
    assert face["label"] == "neutral" and face["used_modalities"] == ["visual"]
    assert face["experts"]["visual_scores"]["weight"] == 1
    union = fusion.predict_with_head(model,"",audio_embedding=np.ones(768),visual=visual)
    assert union["label"] == "positive" and union["audio_embedding_used"] and not union["foundation_used"]
    assert sum(e["weight"] for e in union["experts"].values()) == 1
    no_evidence = fusion.predict_with_head(model, "", visual={**visual,"coverage":0})
    assert no_evidence["status"] == "insufficient_evidence" and no_evidence["scores"] == {}


def test_no_evidence_or_model_returns_without_fabricating_a_prediction():
    assert fusion.predict_with_head(None, "短句")["status"] == "head_unavailable"
    r = fusion.predict_with_head(mock_head(), "", audio={"quality":1,"coverage":1,
        "scores":{"negative":.1,"positive":.1,"neutral":.1,"uncertain":.7}})
    assert r["label"] is None and not r["experts"]


def test_corrupted_or_non_numeric_bundle_does_not_crash_file_inference(tmp_path, monkeypatch):
    package = tmp_path / "model.npz"
    monkeypatch.setattr(fusion, "ASSET", package)
    try:
        package.write_bytes(b"PK\x03\x04truncated")
        fusion.bundled_head.cache_clear()
        assert fusion.bundled_head() is None
        data = {**mock_head(), "version": fusion.VERSION, "labels": fusion.LABELS,
                "audio_bias": np.array(["bad", "bad", "bad"])}
        np.savez(package, **data)
        fusion.bundled_head.cache_clear()
        assert fusion.bundled_head() is None
    finally:
        fusion.bundled_head.cache_clear()


def test_desktop_uses_calibration_and_offline_text_matches_the_trained_control():
    from emotion_assistant.adaptive_clip_fusion import adaptive_parameters
    audio = np.linspace(-1., 1., fusion.AUDIO_DIM)
    head, params = fusion.bundled_head(), adaptive_parameters()
    assert head is not None and params is not None
    for text in ("我得想一想", ""):
        deployed = fusion.predict_foundation_clip(text, audio_embedding=audio, adaptive=True)
        control = fusion.predict_with_head(head, text, audio_embedding=audio, variant="no_qwen", adaptive=params)
        assert deployed == control and deployed["version"] == "chsims-adaptive-v2"
        assert deployed["audio_embedding_used"] and not deployed["foundation_used"]
        assert abs(sum(e["weight"] for e in deployed["experts"].values()) - 1) < 1e-8
        for c in fusion.LABELS:
            assert np.isclose(deployed["scores"][c],
                              sum(e["weight"] * e["scores"][c] for e in deployed["experts"].values()))


def test_unavailable_calibration_is_reported_and_v1_can_still_be_replayed(monkeypatch):
    from emotion_assistant import adaptive_clip_fusion
    monkeypatch.setattr(adaptive_clip_fusion, "adaptive_parameters", lambda: None)
    assert fusion.predict_foundation_clip("挺好的", adaptive=True)["status"] == "head_unavailable"
    assert fusion.predict_foundation_clip("挺好的")["version"] == fusion.VERSION
