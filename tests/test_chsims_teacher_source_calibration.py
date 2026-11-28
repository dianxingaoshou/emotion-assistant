import numpy as np

from scripts.chsims_teacher_source_calibration import infer, protect_negative


def test_no_visual_only_blocks_the_negative_to_neutral_review():
    observed = {
        "proposed_neutral": np.array([True, True, True, True, False]),
        "text_support": np.array([.2, -.03, .3, .3, -.03]),
        "voice_support": np.array([.3, .2, .04, .3, .04]),
        "prior_support": np.array([.2] * 5),
        "acoustic_conflict": np.array([.01, .01, .01, .25, .25]),
    }
    rule = {"text_margin": 0., "voice_margin": .06, "prior_ratio": 0., "acoustic_margin": None}
    np.testing.assert_array_equal(protect_negative(observed, rule), [False, True, True, False, False])
    np.testing.assert_array_equal(protect_negative(observed, {**rule, "acoustic_margin": .18}),
                                  [False, True, True, True, False])


def test_window_neutral_review_can_confirm_a_hybrid_disagreement_with_two_clear_votes():
    strong = [.15, .68, .17]
    other = [.20, .63, .17]
    weak = [.32, .36, .32]
    opposed = [.21, .35, .44]
    missing = [np.nan] * 3
    positive, negative = [.12, .40, .48], [.48, .40, .12]
    batch = {
        "prior": np.array([positive, positive, positive, negative, positive]),
        "mini": np.tile([.18, .32, .50], (5, 1)),
        "voice": np.tile([.15, .45, .40], (5, 1)),
        "active": np.array([True, True, True, True, False]),
        "context": np.array([[strong, other, opposed], [strong, weak, opposed], [strong, missing, opposed],
                             [strong, other, opposed], [strong, other, opposed]]),
    }
    policy = {"case": "synchronous_gap", "reference": {"kind": "window_agreement", "share": .8, "rules": []},
              "rule": {"from": 2, "pool": "median", "window_share": .30,
                       "second_margin": .04, "support_ratio": 1.5, "prior_gap": .12}}
    np.testing.assert_array_equal(infer(batch, policy).argmax(axis=1), [1, 2, 2, 0, 2])
