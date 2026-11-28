import numpy as np

from scripts.chsims_teacher_directional import accept, local_channel_scores
from scripts.chsims_teacher_directional_pair import review_pair


def test_correction_requires_evidence_support_and_a_threshold_for_the_proposed_direction():
    current = np.array([[.30, .55, .15], [.30, .55, .15], [.16, .30, .54], [np.nan] * 3])
    proposed = np.array([[.57, .35, .08], [.57, .35, .08], [.16, .47, .37], [.70, .20, .10]])
    support = np.stack((current, proposed, proposed), axis=1)
    support[1, 2] = np.nan
    changed = accept(current, proposed, support, np.array([True, True, True, False]),
                     threshold=[.10, .15, .05], support=2)
    np.testing.assert_allclose(changed, [proposed[0], current[1], current[2], current[3]])


def test_local_audio_and_video_use_their_own_observation_lengths():
    left, right = [.75, .15, .10], [.10, .25, .65]
    probabilities = np.array([[left, right]])
    windows = {f"window_{channel}": probabilities for channel in ("text", "audio", "visual")}
    observation = {"quality": np.array([[[1., 0., .7], [0., .8, .7]]]),
                   "duration": np.array([[[1., .8], [1., 0.]]])}
    text, audio, video = local_channel_scores(windows, observation)
    np.testing.assert_allclose([text[0], audio[0], video[0]], [left, right, left])


def test_pair_review_does_not_replace_positive_or_guess_when_evidence_is_missing():
    previous = np.array([[.31, .55, .14], [.30, .54, .16], [.52, .39, .09], [.33, .17, .50], [np.nan] * 3])
    negative, neutral = [.75, .20, .05], [.30, .60, .10]
    evidence = np.array([[negative, negative], [negative, neutral], [neutral, neutral], [negative, negative],
                         [negative, negative]])
    result = review_pair(previous, np.array([.85, .9, .1, .97, .95]), evidence,
                         np.array([True, True, True, True, False]), .70, .25, 2)
    np.testing.assert_array_equal(result[:4].argmax(1), [0, 1, 1, 2])
    np.testing.assert_allclose(result[3:], previous[3:])
    np.testing.assert_allclose(result[:, 2], previous[:, 2])
