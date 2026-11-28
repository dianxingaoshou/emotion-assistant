import numpy as np

from scripts.chsims_teacher_supported_review import accept, describe


def test_genuine_votes_cannot_overrule_a_strong_opposing_head_or_large_prior_gap():
    incumbent = np.array([[.32, .30, .38]] * 3 + [[.46, .04, .50]])
    candidate = np.array([[.15, .64, .21]] * 4)
    weak_neutral = [.26, .41, .33]
    strong_neutral = [.15, .68, .17]
    strong_positive = [.08, .16, .76]
    strong_negative = [.54, .26, .20]
    heads = np.array([[weak_neutral, weak_neutral, strong_positive],
                      [strong_neutral, strong_neutral, strong_positive],
                      [strong_negative, strong_negative, strong_positive],
                      [strong_neutral, strong_neutral, strong_positive]])
    logs = np.log(heads)
    contrasts = (logs[:, :, [0, 2]] - logs[:, :, [1]]).reshape(4, 6)
    design = np.column_stack((np.zeros((4, 2)), contrasts))
    scope = describe(design, incumbent, candidate, np.ones(4, dtype=bool))
    rule = {"from": 2, "to": 1, "original_margin": .10, "proposed_margin": .01, "votes": 1,
            "min_expert_margin": .08, "pair_support": 2, "opposition_ratio": .5}
    np.testing.assert_array_equal(accept(scope, rule), [False, True, False, False])
