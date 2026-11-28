import numpy as np

from scripts.chsims_teacher_acoustic_continuity_review import acoustic_margin, adjusted_scores
from scripts.chsims_teacher_neutral_pair_review import paired_scores


def test_only_lone_minicpm_support_uses_the_available_acoustic_window():
    positive = [.10, .25, .65]
    negative = [.65, .25, .10]
    neutral = [.10, .75, .15]
    unavailable = [np.nan] * 3
    record = {"retained": np.array([[positive, negative], [negative, positive],
                                    [positive, negative], [negative, unavailable], [positive, positive]]),
              "mini": np.array([positive, positive, positive, positive, neutral]),
              "acoustic": np.array([neutral, neutral, positive, neutral, negative])}
    margin = acoustic_margin(record)
    np.testing.assert_allclose(margin, [-.55, .40, 0., 0., 0.])
    before = np.full(5, .50)
    after = adjusted_scores(before, margin, 2.)
    assert after[0] < before[0] < after[1]
    np.testing.assert_array_equal(after[2:], before[2:])


def test_pair_review_reserves_the_incumbent_negative_mass():
    before = np.array([[.12, .64, .24], [.60, .24, .16]])
    after = paired_scores(before, np.array([.7, .7]))
    np.testing.assert_allclose(after[:, 0], before[:, 0])
    np.testing.assert_allclose(after.sum(axis=1), 1.)
    np.testing.assert_array_equal(after.argmax(axis=1), [2, 0])
