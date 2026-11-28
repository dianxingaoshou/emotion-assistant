import numpy as np

from scripts.chsims_teacher_context_review import infer


def test_strong_text_and_missing_context_keep_the_established_prediction():
    prior = np.array([[.11, .28, .61]] * 3)
    neutral = np.array([[.12, .69, .19]] * 3)
    batch = {"prior": prior, "mini": neutral, "voice": neutral, "active": np.ones(3, dtype=bool),
             "context": np.array([[[.08, .12, .80]], [[.20, .43, .37]], [[np.nan] * 3]]),
             "voice_quality": np.array([.80, .80, .80])}
    gate = {"share": .8, "rules": [{"from": 2, "to": 1, "gain": .05, "quality": .45,
                                   "veto": .25, "context_floor": -.15, "audio_support": 0.}]}
    np.testing.assert_array_equal(infer(batch, gate).argmax(1), [2, 1, 2])


def test_one_strong_retained_modality_can_veto_a_voice_only_neutral_suggestion():
    positive = [.10, .25, .65]
    neutral = [.05, .77, .18]
    prior = np.array([positive, positive])
    audio = np.array([neutral, neutral])
    weak = [.20, .41, .39]
    batch = {"prior": prior, "mini": audio, "voice": audio, "active": np.ones(2, dtype=bool),
             "context": np.array([[positive, weak, weak], [weak, weak, weak]]),
             "voice_quality": np.array([.8, .8])}
    policy = {"share": .6, "rules": [{"from": 2, "to": 1, "gain": .05, "quality": .25,
                                     "veto": .25, "context_floor": -.15, "audio_support": 0.}]}
    np.testing.assert_array_equal(infer(batch, policy).argmax(1), [2, 1])
