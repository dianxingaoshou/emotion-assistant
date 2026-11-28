import numpy as np
import pytest

from scripts.chsims_teacher_timeline import (
    combine_text_audio, protect_directions, temporal_probability, weighted_probabilities,
)
from scripts.prepare_chsims_teacher_timeline import remaining_spans, speech_quality


def test_text_audio_weight_uses_quality_speech_length_and_conflict():
    text = np.array([[.05, .10, .85]] * 2)
    audio = np.array([[.80, .15, .05]] * 2)
    quality = np.array([[1., .3, .5], [.1, 5., .9]])
    coefficients = np.array([0., 3., -3., 0., 0., 0., 1.])
    prediction, text_weight = combine_text_audio(text, audio, quality, coefficients)
    assert text_weight[0] > .5 > text_weight[1]
    np.testing.assert_array_equal(prediction.argmax(1), [2, 0])


def test_temporal_weights_respond_to_evidence_and_ignore_a_missing_window():
    missing = np.full(3, np.nan)
    negative, positive = [.8, .1, .1], [.1, .1, .8]
    observations = {"window_text": np.array([[positive, negative], [positive, missing]]),
                    "window_audio": np.full((2, 2, 3), np.nan), "window_visual": np.full((2, 2, 3), np.nan)}
    local = {"quality": np.array([[[.9, 1., .6], [.2, .2, .5]]] * 2),
             "duration": np.array([[[2., 0.], [1., 0.]]] * 2)}
    probabilities, mass = temporal_probability(observations, local, np.zeros(7))
    assert mass[0, 0] > 5 * mass[0, 1]
    assert mass[1, 1] == 0.
    assert probabilities[0, 2] > .7
    np.testing.assert_allclose(probabilities[1], positive)
    assert np.isnan(weighted_probabilities(np.full((1, 2, 3), np.nan), np.ones((1, 2)))).all()


def test_gap_boundaries_and_optional_snr_are_measured_on_the_original_clock():
    assert remaining_spans(5., (1., 3.)) == ((0., 1.), (3., 5.))
    with pytest.raises(ValueError):
        remaining_spans(2., (1., 3.))
    seconds, quality = speech_quality([(None, 3., 4., {"voiced_seconds": .6, "estimated_snr_db": None})])
    assert seconds == .6
    assert quality == .5


def test_video_in_an_observed_window_can_contribute_without_speech():
    absent = np.full((1, 2, 3), np.nan)
    observations = {"window_text": absent, "window_audio": absent,
                    "window_visual": np.array([[[.15, .2, .65], [.70, .2, .10]]])}
    local = {"quality": np.zeros((1, 2, 3)), "duration": np.array([[[2., .9], [2., .1]]])}
    scores, _ = temporal_probability(observations, local, np.zeros(7))
    np.testing.assert_allclose(scores[0], [.205, .2, .595])


def test_direction_guard_requires_a_smaller_margin_to_replace_either_pole_with_neutral():
    previous = np.array([[.53, .41, .06], [.06, .41, .53], [.05, .45, .50], [.09, .51, .40], [.46, .42, .12]])
    candidate = np.array([[.25, .70, .05], [.05, .70, .25], [.05, .70, .25], [.05, .25, .70], [.25, .70, .05]])
    observed = candidate.copy()
    observed[-1] = [.70, .25, .05]
    scores = protect_directions(previous, candidate, observed, margin=.20, corroboration=True)
    np.testing.assert_allclose(scores, [previous[0], previous[1], candidate[2], candidate[3], previous[4]])
