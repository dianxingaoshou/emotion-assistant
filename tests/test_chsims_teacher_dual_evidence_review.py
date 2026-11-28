import numpy as np

from scripts.chsims_teacher_dual_evidence_review import confirmation


def test_retained_text_cannot_confirm_an_upgrade_without_whole_clip_support():
    neutral = [.1, .8, .1]
    positive = [.1, .2, .7]
    absent = [np.nan, np.nan, np.nan]
    batch = {"context": np.array([[neutral], [positive], [neutral], [absent], [positive]]),
             "acoustic": np.array([positive, neutral, positive, absent, neutral]),
             "voice": np.array([neutral, positive, positive, absent, positive]),
             "retained": np.array([[positive, neutral], [positive, neutral], [absent, absent],
                                   [positive, positive], [absent, positive]])}
    policy = {"text_floor": -.25, "local_floor": 0., "voice_floor": 0., "dual_audio_floor": .25,
              "local_audio_floor": -1.}
    np.testing.assert_array_equal(confirmation(batch, policy), [False, True, True, False, False])
