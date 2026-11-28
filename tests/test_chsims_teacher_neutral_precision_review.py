import numpy as np

from scripts.chsims_teacher_neutral_precision_review import infer


def batch_and_policy():
    incumbent = np.array([[.25, .45, .30], [.10, .60, .30], [.25, .45, .30],
                          [.15, .20, .65], [.25, .45, .30], [.25, .45, .30]])
    prior = np.tile([.20, .50, .30], (len(incumbent), 1))
    prior[2] = [.20, .25, .55]
    acoustic = np.tile([.10, .20, .70], (len(incumbent), 1))
    acoustic[4] = np.nan
    acoustic[5] = [.15, .60, .25]
    mini = np.tile([.15, .60, .25], (len(incumbent), 1))
    mini[4] = [.15, .20, .65]
    text = np.tile([.05, .25, .70], (len(incumbent), 1))
    batch = {"context": text[:, None], "acoustic": acoustic, "voice": mini, "prior": prior,
             "mini": mini, "retained": np.stack((text, mini), axis=1),
             "design": np.zeros((len(incumbent), 13)), "active": np.ones(len(incumbent), dtype=bool)}
    policy = {"rule": {"target": 2, "family": "compact", "mode": "multinomial", "mix": 1.,
                       "anchor_neutral": True, "cue": "global_acoustic", "max_gap": .20,
                       "min_target": .50, "min_lead": .15},
              "head": {"center": [0.] * 8, "scale": [1.] * 8,
                       "coef": np.zeros((3, 8)).tolist(), "bias": np.log([.10, .20, .70]).tolist()}}
    return batch, incumbent, policy


def test_review_preserves_old_polarity_large_margin_and_missing_acoustic():
    batch, incumbent, policy = batch_and_policy()
    revised = infer(batch, incumbent, policy)
    np.testing.assert_allclose(revised[0], [(.30 * .25) / .70, (.30 * .45) / .70, .70])
    np.testing.assert_array_equal(revised[1:], incumbent[1:])


def test_target_confidence_and_lead_are_separate_requirements():
    batch, incumbent, policy = batch_and_policy()
    for key, value in (("min_target", .71), ("min_lead", .52)):
        blocked = {**policy, "rule": {**policy["rule"], key: value}}
        np.testing.assert_array_equal(infer(batch, incumbent, blocked), incumbent)
