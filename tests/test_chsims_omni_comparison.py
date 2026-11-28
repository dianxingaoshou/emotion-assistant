import numpy as np

from scripts.chsims_audio_review_gate import review
from scripts.chsims_omni_model_comparison import matrix
from scripts.chsims_teacher_refinement import accept_corrections, blend_audio_encoders, neutral_policy
from scripts.chsims_teacher_robustness import applicable_gaps, robust_predict
from scripts.prepare_chsims_teacher_gaps import cut_tracks, drop_text


def test_omni_review_can_protect_neutral_and_abstain_on_missing_audio():
    desktop = np.array([[.2, .6, .2], [.25, .15, .6], [np.nan] * 3, [np.nan] * 3])
    omni = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .2, .7], [np.nan] * 3])
    normal = review(desktop, omni, .55, 1., .75)
    protected = review(desktop, omni, .55, 1., .75, protect_neutral=True)
    np.testing.assert_allclose(normal[:3], omni[:3])
    np.testing.assert_allclose(protected[0], desktop[0])
    np.testing.assert_allclose(protected[1:3], omni[1:3])
    assert np.isnan(protected[3]).all()


def test_audio_blend_requires_both_predictions_and_honors_uncertainty_gate():
    qwen = np.array([[.50, .20, .30], [.70, .20, .10], [.50, .40, .10], [np.nan] * 3, [.5, .4, .1]])
    mini = np.array([[.20, .20, .60], [.05, .05, .90], [.60, .30, .10], [.1, .3, .6], [np.nan] * 3])
    blended = blend_audio_encoders(qwen, mini, .5, when_uncertain=True)
    np.testing.assert_allclose(blended[0], [.35, .20, .45])
    np.testing.assert_allclose(blended[1:], qwen[1:])


def test_neutral_correction_keeps_empty_predictions_empty():
    a = np.array([[.39, .25, .36], [.70, .20, .10], [np.nan] * 3])
    b = np.array([[.31, .23, .46], [.70, .20, .10], [np.nan] * 3])
    adjusted = neutral_policy(a, b, .5, 0., .7)
    assert adjusted[0].argmax() == 1
    np.testing.assert_allclose(adjusted[1], b[1])
    assert np.isnan(adjusted[2]).all()


def test_correction_requires_close_scores_and_another_heads_agreement():
    base = np.array([[.48, .45, .07], [.48, .45, .07], [.75, .15, .10], [np.nan] * 3])
    proposal = np.tile([.40, .50, .10], (4, 1))
    other = np.array([[.30, .60, .10], [.60, .30, .10], [.30, .60, .10], [.30, .60, .10]])
    final = accept_corrections(base, proposal, other, {"proposal": "balanced", "max_margin": .05})
    np.testing.assert_allclose(final[0], proposal[0])
    np.testing.assert_allclose(final[1:], base[1:])


def test_multimodal_fusion_masks_all_features_and_flags_of_removed_input():
    blocks = {"audio_context": np.array([[3.], [9.]], dtype=np.float32),
              "audio_desk": np.array([[7.], [10.]], dtype=np.float32),
              "visual_frames": np.array([[4.], [np.nan]], dtype=np.float32)}
    scales = {name: (np.zeros(1), np.ones(1)) for name in blocks}
    indices = np.arange(2)
    x, active = matrix(blocks, scales, indices, "audio")
    np.testing.assert_array_equal(x, [[0, 0, 0, 0, 4, 1], [0, 0, 0, 0, 0, 0]])
    np.testing.assert_equal(active, [True, False])


def test_synchronous_gap_excludes_the_same_interval_from_audio_and_video():
    wave = np.arange(16000, dtype=np.float32)
    frames = np.arange(5)
    audio, video, keep, fragments, lost = cut_tracks(wave, frames, np.arange(5) * .2, (.2, .6))
    assert lost == 6400 and not audio[3200:9600].any()
    np.testing.assert_array_equal(audio[9600:], wave[9600:])
    np.testing.assert_array_equal(video, [0, 3, 4])
    np.testing.assert_array_equal(keep, [True, False, False, True, True])
    assert [start for start, _ in fragments] == [0., .6]
    np.testing.assert_array_equal(fragments[0][1], wave[:3200])
    np.testing.assert_array_equal(fragments[1][1], wave[9600:])


def test_text_loss_drops_contiguous_units_and_keeps_one_unit_loss_empty():
    assert drop_text("一二三四五", .5) == ("一二 五", 2, 5)
    assert drop_text("happy", .5) == ("", 1, 1)
    assert applicable_gaps(1) == ("text_gap", "synchronous_gap")


def test_teacher_routes_by_actual_remaining_evidence_and_abstains_when_empty():
    blocks = {"text_context": np.array([[2.], [np.nan]]),
              "audio_context": np.array([[np.nan], [np.nan]]),
              "visual_context": np.array([[1.], [1.]])}
    text = {"scales": {"text_context": [[0.], [1.]]}, "weights": [[0., 0.], [1., 0.], [0., 0.]],
            "bias": [0., 0., 0.], "temperature": 1., "neutral_shift": 0.}
    config = {"heads": {"1": {"heads": {"teacher": text, "desktop": text}, "teacher_weight": 1.,
                              "confidence_threshold": .95}}}
    probs, masks = robust_predict(blocks, 1, config)
    np.testing.assert_array_equal(masks, [1, 0])
    assert probs["teacher"][0].argmax() == 1
    assert np.isnan(probs["teacher"][1]).all()
    assert np.isnan(probs["selective"]).all()
