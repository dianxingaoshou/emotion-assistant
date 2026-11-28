import numpy as np

from scripts.chsims_teacher_neutral_consensus_review import observe, retained_text_audio, select


def test_retained_audio_and_text_use_their_own_quality_weights_without_vision():
    positive, negative = [.08, .22, .70], [.70, .22, .08]
    scores = {"window_text": np.array([[positive, negative]]), "window_audio": np.array([[positive, negative]])}
    windows = {"quality": np.array([[[1., 0., .5], [0., 1., .5]]]),
               "duration": np.array([[[1.], [1.]]])}
    np.testing.assert_allclose(retained_text_audio(scores, windows), [[positive, negative]])


def test_protects_neutral_from_preceding_channel_conflict_and_ignores_unavailable_modality():
    neutral, positive = [.20, .47, .33], [.12, .23, .65]
    unavailable = [np.nan] * 3
    batch = {"context": np.array([[neutral], [neutral], [unavailable]]),
             "acoustic": np.array([neutral] * 3), "voice": np.array([positive] * 3),
             "mini": np.array([positive] * 3), "prior": np.array([neutral, positive, neutral]),
             "design": np.zeros((3, 13)), "active": np.ones(3, dtype=bool),
             "retained": np.array([[positive, positive]] * 3)}
    initial = np.array([neutral] * 3)
    proposal = np.array([positive] * 3)
    rule = {"target": 2, "max_gap": .24, "min_lead": .17}
    np.testing.assert_array_equal(select(observe(batch, initial), proposal, rule), [True, False, False])
