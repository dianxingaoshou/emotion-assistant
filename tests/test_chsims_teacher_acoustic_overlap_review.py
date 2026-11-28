import numpy as np

from scripts.chsims_teacher_acoustic_overlap_review import apply


def test_same_track_limit_preserves_published_acceptances_and_negative_adjustments():
    neutral = [.15, .72, .13]
    negative = [.64, .24, .12]
    positive = [.14, .18, .68]
    record = {
        "initial": np.array([[.12, .46, .42]] * 4),
        "text": np.array([neutral] * 4),
        "acoustic": np.array([neutral, neutral, negative, neutral]),
        "mini": np.array([positive] * 4),
        "retained": np.array([[neutral, positive]] * 3 + [[neutral, negative]]),
        "neutral": np.ones(4, dtype=bool),
        "old_neutral": np.ones(4, dtype=bool),
    }
    previous = {"rule": {"target": 2, "max_gap": .20, "min_target": .50, "min_lead": .15,
                         "cue": "global_acoustic", "anchor_neutral": True, "mix": 1.}}
    acoustic = {"gain": 3., "min_target": .50, "min_lead": .10}
    damped = {"floor": 0., "strength": 32.}
    probability = np.array([.43, .63, .43, .60])
    prior = apply(record, probability, previous, acoustic, None)
    revised = apply(record, probability, previous, acoustic, damped)
    np.testing.assert_array_equal(prior.argmax(1), [2, 2, 2, 1])
    np.testing.assert_array_equal(revised.argmax(1), [1, 2, 2, 1])
    np.testing.assert_array_equal(revised[1:], prior[1:])
