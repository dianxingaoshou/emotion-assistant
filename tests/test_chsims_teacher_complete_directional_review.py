import numpy as np

from scripts.chsims_teacher_complete_directional_review import accepted, describe, infer


def test_direction_gate_uses_pairwise_support_from_distinct_modalities():
    incumbent = np.array([[.48, .45, .07]] * 4)
    offered = np.array([[.36, .56, .08], [.36, .56, .08], [.36, .56, .08], [.36, .56, .08]])
    design = np.array([
        [0., 0., -1., .2, -.4, -.2, .3, -.1],
        [0., 0., -1., .2, .4, -.2, .3, -.1],
        [0., 0., -1., .2, -.4, -.2, .3, -.1],
        [0., 0., -1., .2, -.4, -.2, .3, -.1],
    ])
    observed = describe(design, incumbent, offered, np.array([True, True, False, True]))
    observed["old_gap"][3] = .15
    rule = {"from": 0, "to": 1, "old_gap": .10, "new_gap": .04, "votes": 1, "pair_support": 2}

    np.testing.assert_array_equal(accepted(observed, rule), [True, False, False, False])


def test_empty_direction_policy_returns_the_old_scores():
    original = np.array([[.20, .60, .20]])
    np.testing.assert_equal(infer({}, original, {"rules": []}), original)
