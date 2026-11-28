import numpy as np

from scripts.chsims_teacher_neutral_reliability_review import protect


def test_either_reliability_score_can_keep_a_neutral_prediction():
    incumbent = np.tile([.20, .55, .25], (4, 1))
    proposal = np.tile([.15, .30, .55], (4, 1))
    estimates = np.array([[.68, .88], [.30, .47], [.67, .48], [.69, .10]])
    rule = {"neutral_veto": .68, "minimum_direction": .48}

    actual = protect(incumbent, proposal, estimates, rule)

    np.testing.assert_array_equal(actual, np.vstack((incumbent[:2], proposal[2], incumbent[3])))


def test_veto_is_limited_to_previous_neutral_to_positive_proposals():
    incumbent = np.array([[.20, .55, .25], [.60, .25, .15], [.10, .20, .70], [.20, .55, .25]])
    proposal = np.array([[.60, .25, .15], [.15, .30, .55], [.10, .70, .20], [.20, .55, .25]])
    estimates = np.tile([.99, .01], (4, 1))
    rule = {"neutral_veto": .68, "minimum_direction": .48}

    np.testing.assert_array_equal(protect(incumbent, proposal, estimates, rule), proposal)
