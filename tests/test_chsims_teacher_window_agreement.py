import numpy as np

from scripts.chsims_teacher_window_agreement import infer


def test_neutral_review_requires_two_clear_votes_from_retained_modalities():
    negative, neutral, positive = [.70, .24, .06], [.12, .75, .13], [.06, .24, .70]
    weak_neutral = [.42, .48, .10]
    absent = [np.nan] * 3
    tied = [.12, .44, .44]
    batch = {
        "prior": np.array([positive, negative, positive, positive, positive, positive]),
        "mini": np.tile(neutral, (6, 1)),
        "voice": np.tile(neutral, (6, 1)),
        "voice_quality": np.array([.8, .8, .8, .8, .8, .1]),
        "active": np.ones(6, dtype=bool),
        "context": np.array([[negative, positive, negative],
                             [weak_neutral, negative, neutral],
                             [neutral, absent, tied],
                             [neutral, neutral, positive],
                             [neutral, absent, positive],
                             [neutral, neutral, positive]]),
    }
    rules = [{"from": label, "to": 1, "margin": 0., "ratio": .5,
              "quality": .25, "gain": 0., "audio_support": 0.} for label in (0, 2)]
    policy = {"kind": "window_agreement", "share": 1., "rules": rules}
    np.testing.assert_array_equal(infer(batch, policy).argmax(axis=1), [2, 0, 2, 1, 2, 2])


def test_missing_visual_keeps_the_previous_independent_text_review():
    batch = {
        "prior": np.array([[.10, .29, .61]]),
        "mini": np.array([[.10, .74, .16]]),
        "voice": np.array([[.10, .74, .16]]),
        "voice_quality": np.array([.8]),
        "active": np.ones(1, dtype=bool),
        "context": np.array([[[.13, .62, .25]]]),
    }
    policy = {"kind": "context_review", "share": 1.,
              "rules": [{"from": 2, "to": 1, "gain": .05, "quality": .25, "veto": .25,
                         "context_floor": 0., "audio_support": 0.}]}
    np.testing.assert_array_equal(infer(batch, policy).argmax(axis=1), [1])
