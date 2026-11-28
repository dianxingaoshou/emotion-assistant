import numpy as np

from scripts.chsims_teacher_complete_review import apply_review, infer


def test_complete_review_changes_only_an_eligible_close_decision():
    incumbent = np.array([[.45, .40, .15], [.70, .20, .10], [.45, .40, .15], [np.nan] * 3])
    rule = {"max_anchor_margin": .2, "min_proposal_margin": .05, "share": 1., "neutral_offset": 0.,
            "center": [0.], "scale": [1.], "coefficients": [[0., .9, 0.], [0., 0., 0.]]}
    scores = apply_review(np.zeros((4, 1)), incumbent, np.array([True, True, False, False]), {"rule": rule})

    assert scores[0].argmax() == 1
    np.testing.assert_equal(scores[1:], incumbent[1:])


def test_disabled_complete_review_never_extracts_additional_features():
    incumbent = np.array([[.2, .6, .2]])
    np.testing.assert_equal(infer({}, incumbent, {"rule": None}), incumbent)
