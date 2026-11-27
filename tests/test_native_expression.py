from copy import deepcopy

import pytest

from emotion_assistant.native_expression import CLASS_SETS, expression_profile, format_expression


def observation(modality, prob, stop=1, quality=.8):
    scores = dict.fromkeys(sorted(CLASS_SETS[modality]), 0.)
    scores.update(prob)
    return {"modality": modality, "raw_scores": scores, "valid": True, "quality": quality,
            "t_start": 0, "t_end": stop}


def test_same_modality_duration_times_quality_weights_and_native_classes():
    short = observation("visual", {"anger": .9, "fear": .1}, 1, .5)
    long = observation("visual", {"anger": .1, "surprise": .9}, 2, 1)
    audio = observation("audio", {"happiness": 1})
    profile = expression_profile([short, long, {**long, "valid": False}, audio], "visual")
    assert profile["eligible_observations"] == 2
    assert profile["top_two"] == ["surprise", "anger"]
    assert profile["scores"]["surprise"] == pytest.approx(.72)
    assert profile["margin"] == pytest.approx(.46)
    assert "惊讶 72.0%" in format_expression(profile)


def test_audio_native_residual_is_preserved_and_named():
    obs = observation("audio", {"unknown": .6, "surprise": .4})
    raw = obs["raw_scores"]
    raw["<unk>"] = raw.pop("unknown")
    raw["吃惊/surprised"] = raw.pop("surprise")
    profile = expression_profile([obs], "audio")
    assert profile["top_two"] == ["unknown", "surprise"]
    assert "具体类别未定" in format_expression(profile)


def test_saturated_scores_are_not_rounded_to_certainty():
    obs = observation("audio", {"happiness": .99999, "surprise": .00001})
    text = format_expression(expression_profile([obs], "audio"))
    assert ">99.9%" in text and "<0.1%" in text and "100.0%" not in text


@pytest.mark.parametrize("bad", ["missing", "four_polarities", "category_missing", "nan", "uniform", "low_quality"])
def test_invalid_evidence_never_acquires_a_fine_label(bad):
    obs = observation("visual", {"anger": 1})
    if bad == "missing":
        obs.pop("raw_scores")
    elif bad == "four_polarities":
        obs["raw_scores"] = {"positive": 1, "negative": 0, "neutral": 0, "uncertain": 0}
    elif bad == "category_missing":
        obs["raw_scores"].pop("contempt")
    elif bad == "nan":
        obs["raw_scores"]["neutral"] = float("nan")
    elif bad == "uniform":
        obs["raw_scores"] = dict.fromkeys(obs["raw_scores"], .125)
    else:
        obs["quality"] = .1
    profile = expression_profile([obs], "visual", still=True)
    assert profile["status"] == "unavailable" and not profile["top_two"]


def test_opposing_valid_observations_keep_tie_unresolved():
    one = observation("visual", {"fear": 1})
    two = deepcopy(one)
    two["raw_scores"].update(fear=0, anger=1)
    profile = expression_profile([one, two], "visual")
    assert profile["status"] == "ambiguous" and profile["top_two"] == []
