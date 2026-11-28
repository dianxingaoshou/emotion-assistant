import numpy as np
import pytest

from scripts.benchmark_chsims_teacher import paired_metrics
from scripts.chsims_audio_factorized import join_logits
from scripts.chsims_face_teacher import face_crops
from scripts.chsims_teacher_cross_source import held_out_validation, labels, mix, prior_adjustment
from scripts.chsims_teacher_data import evidence_for, read_rows, redact_subtitles
from scripts.chsims_teacher_supervision import expected_features, out_of_fold, source_folds


def test_teacher_isolates_the_four_evidence_inputs():
    audio = np.arange(10, dtype=np.float32)
    frames = np.ones((2, 8, 8, 3), dtype=np.uint8)
    assert evidence_for("text", "台词", audio, frames) == {"text": "台词"}
    assert list(evidence_for("audio", "台词", audio, frames)) == ["audio"]
    assert list(evidence_for("visual", "台词", audio, frames)) == ["video"]
    assert list(evidence_for("full", "台词", audio, frames)) == ["text", "audio", "video"]
    assert evidence_for("visual", "台词", audio, frames[:0]) == {}
    assert list(evidence_for("full", "", audio, frames[:0])) == ["audio"]


def test_subtitle_removal_masks_detected_text_and_lower_band():
    raw = np.full((100, 100, 3), 255, dtype=np.uint8)
    upper_text = np.array([[[18, 14], [26, 14], [26, 22], [18, 22]]])
    clean, count = redact_subtitles(raw, lambda _: (upper_text, 0))
    assert count == 1 and not clean[14:23, 18:27].any()
    assert not clean[72:].any()
    assert clean[35:70, 40:65].min() == raw.min() == 255


def test_score_uses_the_matching_reference_and_preserves_abstentions(tmp_path):
    (tmp_path / "meta.csv").write_text(
        "video_id,clip_id,text,label,label_T,label_A,label_V,annotation,mode\n"
        "v,0001,caption,-1,0,0.2,-0.2,Negative,train\n"
        "v,0002,caption,-1,0,0.2,-0.2,Negative,valid\n"
        "w,0001,caption,-1,0,0.2,-0.2,Negative,test\n", encoding="utf8")
    labels = read_rows(tmp_path)["test"]["w/0001"]
    assert labels == {"full": "negative", "text": "neutral", "audio": "positive", "visual": "negative"}
    result = paired_metrics([("w/0001", labels, {"audio": {"label": "abstain"}},
                              {"audio": {"label": "positive"}})], "audio")
    assert result["baseline"]["abstain"] == 1 and result["baseline"]["accuracy"] == 0
    assert result["teacher"]["accuracy"] == 1 and result["corrected"] == 1


def test_supervision_blocks_only_use_the_actual_inputs():
    predictions = {
        "text": {"seen": [], "scores": None},
        "audio": {"seen": ["audio"], "scores": {"neutral": 1.}},
        "visual": {"seen": ["video"], "scores": {"neutral": 1.}},
        "full": {"seen": ["audio"], "scores": {"neutral": 1.}},
    }
    assert expected_features(predictions) == {"audio_context", "audio_acoustic", "visual_context",
                                              "visual_frames", "full_context", "full_acoustic"}
    predictions["visual"]["seen"].append("audio")
    with pytest.raises(ValueError, match="visual"):
        expected_features(predictions)


def test_training_targets_hold_out_whole_sources_and_preserve_missing_evidence():
    rows = [{"id": f"source{source}/clip{clip}"} for source in range(10) for clip in range(3)]
    truth = np.tile(np.arange(3), 10)
    scores = np.column_stack((truth == 0, truth == 2)).astype(float)
    scores[1] = np.nan
    folds = source_folds(rows)
    for training, held_out in folds:
        assert not set(training // 3) & set(held_out // 3)
    head = {"features": ["scores"], "C": 10., "balance_exponent": .5, "temperature": 1.}
    probabilities = out_of_fold({"scores": scores}, truth, head, folds)
    assert np.isnan(probabilities[1]).all()
    observed = np.arange(len(rows)) != 1
    np.testing.assert_allclose(probabilities[observed].sum(axis=1), 1.)
    np.testing.assert_equal(probabilities[observed].argmax(axis=1), truth[observed])


def test_cross_source_validation_excludes_sources_even_if_in_official_train():
    train_groups = np.array(["a", "a", "b", "c", "d", "e", "e", "f", "g"])
    valid_groups = np.array(["a", "b", "b", "d", "e", "f", "g", "new"])
    seen = []
    for train, valid in held_out_validation(train_groups, valid_groups):
        assert not set(train_groups[train]) & set(valid_groups[valid])
        seen.extend(valid)
    assert sorted(seen) == list(range(len(valid_groups)))


def test_undoing_class_weight_reduces_the_rare_class_logit():
    truth = np.array([0, 0, 0, 1, 2, 2])
    adjustment = prior_adjustment(truth, balance=1., correction=1.)
    assert adjustment[1] < adjustment[0] and adjustment[1] < adjustment[2]
    np.testing.assert_allclose(prior_adjustment(truth, 0., 1.), 0.)
    np.testing.assert_allclose(prior_adjustment(truth, 1., .5), adjustment * .5)


def test_face_fusion_uses_only_available_probabilities():
    face = np.array([[.05, .15, .80], [np.nan] * 3, [np.nan] * 3])
    scene = np.array([[.80, .15, .05], [.65, .15, .2], [np.nan] * 3])
    result = mix({"face": face, "scene": scene}, {"face": .65, "scene": .35})
    assert labels(result) == ["positive", "negative", "abstain"]
    np.testing.assert_allclose(result[1], scene[1])


def test_face_teacher_crops_only_the_prepared_frames():
    frames = np.zeros((3, 224, 224, 3), np.uint8)
    frames[1, 53:118, 76:131] = [210, 140, 90]

    def observe(_frame, _session, stamp, _end):
        return {"reason": "ok", "face_box": [152, 106, 110, 130]} if stamp == 1 else {"reason": "no_face"}

    cropped, sizes = face_crops(frames, observe)
    assert cropped.shape == (1, 224, 224, 3) and sizes == [55]
    np.testing.assert_equal(cropped[0, 112, 112], [210, 140, 90])


def test_factorized_audio_preserves_direction_when_neutral_cost_changes():
    direction = np.array([-2., 2., np.nan])
    neutral = np.array([0., 0., np.nan])
    base = join_logits(direction, neutral, 1.)
    corrected = join_logits(direction, neutral, 2.)
    np.testing.assert_allclose(corrected[:2].sum(axis=1), 1.)
    np.testing.assert_allclose(base[:2, 2] / base[:2, 0], corrected[:2, 2] / corrected[:2, 0])
    assert labels(base) == ["neutral", "neutral", "abstain"]
    assert labels(corrected) == ["negative", "positive", "abstain"]
