import numpy as np

from scripts.chsims_teacher_dynamic_conflict import infer


def test_no_visual_changes_conflicting_neutral_only_with_sufficient_weight():
    neutral, positive = [.42, .46, .12], [.15, .25, .60]
    text, audio = [.20, .65, .15], [.72, .20, .08]
    batch = {
        "prior": np.array([neutral, neutral, positive, neutral, neutral]),
        "context": np.array([[text], [text], [text], [[np.nan] * 3], [text]]),
        "acoustic": np.array([audio, audio, audio, audio, text]),
        "voice": np.tile([.60, .28, .12], (5, 1)),
        "quality": np.array([[.6, 4., .5], [.6, .2, .5], [.6, 4., .5], [.6, 4., .5], [.6, 4., .5]]),
        "active": np.ones(5, dtype=bool),
    }
    reference = {"case": "no_visual", "rule": None,
                 "reference": {"kind": "context_review", "rules": []}}
    policy = {
        "case": "no_visual", "reference": reference,
        "rule": {"min_margin": 0., "max_previous_gap": .24, "anchor": 0., "offset": -.30},
        "head": {"center": [0.] * 8, "scale": [1.] * 8,
                 "weights": [0., 0., 0., 5., 0., 0., 0., 0.], "bias": -2.},
    }
    np.testing.assert_array_equal(infer(batch, policy).argmax(axis=1), [0, 1, 2, 1, 1])
