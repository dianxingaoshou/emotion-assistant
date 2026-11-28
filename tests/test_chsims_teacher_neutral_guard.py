import numpy as np

from scripts.chsims_teacher_neutral_guard import veto


def test_neutral_override_requires_genuine_acoustic_direction_and_text_confirmation():
    observed = {
        "text": np.array([[.41, .30, .29], [.52, .31, .17], [.52, .31, .17], [.42, .34, .24],
                          [.20, .60, .20], [.41, .30, .29]]),
        "audio": np.array([[.20, .59, .21], [.20, .59, .21], [.18, .37, .45], [.20, .59, .21],
                           [.59, .25, .16], [.20, .59, .21]]),
        "audio_pick": np.array([1, 1, 1, 1, 0, 1]),
    }
    batch = {
        "voice": np.array([[.20, .70, .10], [.20, .70, .10], [.20, .70, .10], [.70, .20, .10],
                           [.20, .70, .10], [.20, .70, .10]]),
    }
    already_changed = np.array([True, True, True, True, True, False])
    np.testing.assert_array_equal(veto(batch, observed, already_changed, .20),
                                  [True, False, True, False, False, False])
