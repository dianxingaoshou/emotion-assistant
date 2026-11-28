import numpy as np

from scripts import chsims_teacher_neutral_precision as review


def test_neutral_review_requires_evidence_and_keeps_conflicting_or_non_neutral_predictions():
    initial = np.asarray([[.42, .46, .12], [.14, .48, .38], [.10, .80, .10],
                          [.43, .44, .13], [.65, .25, .10], [.42, .46, .12]])
    text = np.asarray([[.2, .7, .1], [.15, .30, .55], [.15, .30, .55],
                       [.15, .30, .55], [.15, .30, .55], [np.nan, np.nan, np.nan]])
    audio = np.asarray([[.2, .7, .1], [.15, .30, .55], [.15, .30, .55],
                        [.15, .30, .55], [.15, .30, .55], [.15, .30, .55]])
    mini = np.asarray([[.6, .3, .1], [.2, .4, .4], [.6, .3, .1],
                       [.6, .3, .1], [.6, .3, .1], [.6, .3, .1]])
    batch = {"context": text[:, None, :], "acoustic": audio, "voice": mini, "mini": mini, "prior": initial,
             "active": np.ones(len(initial), dtype=bool), "design": np.zeros((len(initial), 13))}
    n_features = review.observe(batch, initial)["features"].shape[1]
    rules = []
    for target, support, gap in ((0, "mini_audio", .06), (2, "text_and_audio", .35)):
        coefficients = np.zeros((n_features + 1, len(review.LABELS)))
        coefficients[0, target] = 3.
        rules.append({"target": target, "support": support, "share": 1., "max_initial_gap": gap,
                      "min_proposal_lead": .2, "head": {"coefficients": coefficients.tolist(),
                                                         "center": [0.] * n_features, "scale": [1.] * n_features}})
    predicted = review.infer(batch, initial, {"rules": rules})
    np.testing.assert_array_equal(predicted.argmax(1), [0, 2, 1, 1, 0, 1])
    np.testing.assert_array_equal(predicted[2:], initial[2:])
