"""Reference labels come from independent audiovisual observers, including disagreement."""
import pytest

from scripts.ravdess_rater_labels import adjudicate


def test_plurality_after_coarsening_not_actor_intention_or_single_emotion_vote():
    # No negative fine-grained emotion beats happy alone, but the coarse votes do.
    winner, votes, counts = adjudicate([3, 3, 3, 4, 5, 5, 6, 7, 9, 8])
    assert (winner, votes, counts) == ("negative", 5, {"negative": 5, "positive": 3, "unmapped": 2})


def test_tied_plurality_excluded_and_explicit_none_differs_from_missing_vote():
    label, top, _ = adjudicate([1, 1, 2, 2, 2, 8, 8, 9, 9, 9])
    assert (label, top) == (None, 5)
    with pytest.raises(ValueError):
        adjudicate([1] * 9)
