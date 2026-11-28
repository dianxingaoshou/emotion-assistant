import numpy as np

from scripts.chsims_teacher_neutral_corroboration import infer


def test_confirming_neutral_override_requires_both_acoustic_models_and_available_evidence():
    initial = np.asarray([[.3, .45, .25], [.3, .45, .25], [.1, .8, .1],
                          [.1, .2, .7], [.3, .45, .25]])
    text = np.asarray([[.7, .2, .1]] * 4 + [[np.nan, np.nan, np.nan]])
    audio = np.asarray([[.7, .2, .1], [.1, .8, .1], [.7, .2, .1], [.7, .2, .1], [.7, .2, .1]])
    voice = np.asarray([[.9, .05, .05]] * len(initial))
    batch = {"context": text[:, None, :], "acoustic": audio, "voice": voice, "prior": initial,
             "active": np.ones(len(initial), dtype=bool)}
    policy = {"rule": {"target": 0, "support": "two_acoustic_models", "voice_weight": .6,
                       "max_gap": .35, "min_lead": .3}}
    predicted = infer(batch, initial, policy)
    np.testing.assert_allclose(predicted[0], .4 * initial[0] + .6 * voice[0])
    np.testing.assert_array_equal(predicted[1:], initial[1:])
