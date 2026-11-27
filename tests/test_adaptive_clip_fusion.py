import numpy as np
import pytest

from emotion_assistant import adaptive_clip_fusion as adaptive


def test_shipped_policy_reconstructs_scores_from_reported_evidence():
    params = adaptive.adaptive_parameters()
    assert params is not None
    inputs = ([.2, .3, .5], [.6, .25, .15], None, [.4, .4, .2])
    result, weights, detail = adaptive.blend(params, "foundation", inputs, (1., 1., 0., .5))
    assert result.sum() == pytest.approx(1.)
    assert weights.sum() == pytest.approx(1.) and weights[2] == 0
    assert detail["expert_scores"][2] is None
    reconstructed = sum(weights[j] * np.array(p) for j, p in enumerate(detail["expert_scores"]) if p is not None)
    assert np.allclose(result, reconstructed)


def test_missing_text_uses_identical_parameters_for_both_arms():
    params = adaptive.adaptive_parameters()
    inputs = (None, [.6, .3, .1], None, [.4, .4, .2])
    a = adaptive.blend(params, "no_qwen", inputs, (1., 1., 0., .5))
    b = adaptive.blend(params, "foundation", inputs, (1., 1., 0., .5))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]
    assert a[2]["policy"] == "av" and a[1][0] == 0.
    result, weights, _ = adaptive.blend(params, "foundation", (None,) * 4, (0.,) * 4)
    assert result is None and not weights.any()


def test_conflicting_cue_agreement_excludes_missing_peers_and_self():
    z = np.array([[[.7, .2, .1], [.1, .3, .6], [.8, .1, .1], [0., 0., 0.]]])
    q = np.array([[1., 1., .5, 0.]])
    x = adaptive.gate_features(z, q)[0]
    assert x[0].tolist() == pytest.approx([1., .5, 1 / 3])
    assert x[1, 2] == 0. and x[2, 2] == .5
    ignored, w, _ = adaptive.blend(adaptive.adaptive_parameters(), "foundation",
                                   ([np.nan, 0., 1.], None, None, None), (1., 0., 0., 0.))
    assert ignored is None and not w.any()


def test_corrupt_or_replaced_base_rejects_calibration(tmp_path, monkeypatch):
    from emotion_assistant import foundation_fusion
    replaced = tmp_path / "different-experts.npz"
    replaced.write_bytes(b"another fit")
    try:
        monkeypatch.setattr(foundation_fusion, "ASSET", replaced)
        adaptive.adaptive_parameters.cache_clear()
        assert adaptive.adaptive_parameters() is None
    finally:
        adaptive.adaptive_parameters.cache_clear()
