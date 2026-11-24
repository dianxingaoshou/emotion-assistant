import numpy as np
import pytest

from scripts.attachment2_features import (
    Gap,
    Split,
    conditions,
    lexical_terms,
    masked_inputs,
    word_coordinates,
)
from scripts.benchmark_attachment2_gaps import combine, metrics


def test_word_mask_removes_every_wordpiece_and_never_forms_a_bigram_across_the_hole():
    vocab = ["[PAD]", "[CLS]", "[SEP]", "i", "dis", "##like", "it", "!"]
    ids = np.array([1, 3, 4, 5, 6, 7, 2, 0])
    words, slots = word_coordinates(ids, ids != 0, vocab)
    assert words == ["i", "dislike", "it"] and slots == [1, 2, 4, 6]
    assert lexical_terms(words, 1, 2) == ["i", "it"]
    assert lexical_terms(words, 0, 0) == [
        "i",
        "dislike",
        "i dislike",
        "it",
        "dislike it",
    ]

    # Natural zero rows are already missing: they do not count as extra damage.
    row = np.array([0, 1, 10, 20, 4, 0, 0, 0], dtype=float)
    series = row[None, :, None]
    split = Split(
        ["clip"],
        np.array(["source"]),
        [words],
        [slots],
        {key: series for key in "AV"},
        {key: row[None, :] != 0 for key in "AV"},
        np.array([1]),
        np.array([0.0]),
    )
    docs, features, available, removed = masked_inputs(
        split, Gap("fixture", "TAV", 0.25, "middle")
    )
    assert docs == [["i", "it"]]
    assert removed == pytest.approx({"T": 1 / 3, "A": 0.5, "V": 0.5})
    for key in "AV":
        assert available[key][0]
        assert features[key][0] == pytest.approx([2.5, 1.5])
    assert len({case.name for case in conditions()}) == 48


def test_unknown_word_has_geometry_but_emits_no_feature_and_cannot_glue_its_neighbors():
    vocab = ["[CLS]", "i", "'", "[UNK]", "##e", "agree", "[SEP]"]
    words, slots = word_coordinates(np.arange(7), np.ones(7), vocab)
    assert words == ["i", "[UNK]", "agree"] and slots == [1, 3, 5, 6]
    assert lexical_terms(words, 0, 0) == ["i", "agree"]


def test_modal_fallback_class_abstention_and_undefined_intensity_are_separate():
    probs = np.array([[[0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.7, 0.2, 0.1]]] * 4)
    probs[2, 1] = [0.34, 0.33, 0.33]
    intensities = np.array([[8.0, -1.0, -6.0]] * 4)
    seen = np.array([[1, 1, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=bool)
    evidence = probs, intensities, seen
    classes, values = combine(evidence, 0.0, 0.0)
    assert list(classes) == [2, 0, -1, -1]
    assert values[:3] == pytest.approx([3.0, -3.0, -1.0]) and np.isnan(values[3])
    text, text_intensity = combine(evidence, 0.0, 0.0, text_only=True)
    assert list(text) == [2, -1, -1, -1] and np.isnan(text_intensity[1:]).all()
    split = Split(
        [],
        np.array([]),
        [],
        [],
        {},
        {},
        np.array([2, 0, 1, 0]),
        np.array([2.0, -1.0, 0.0, -3.0]),
    )
    score = metrics(split, (classes, values))
    assert (
        score["correct"] == 2
        and score["abstain"] == 2
        and score["intensity_scored"] == 3
    )
    assert score["mae"] == pytest.approx(4 / 3)
    assert metrics(split, (classes, np.zeros(4)))["pearson"] is None
