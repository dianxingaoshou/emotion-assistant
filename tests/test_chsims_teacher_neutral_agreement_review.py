import numpy as np

from scripts import chsims_teacher_neutral_agreement_review as agreement


def test_only_two_independently_neutral_whole_clip_cues_enable_the_guard():
    record = {
        "text": np.tile([.15, .65, .20], (4, 1)),
        "acoustic": np.array([[.15, .65, .20], [.10, .20, .70], [.15, .65, .20], [.15, .65, .20]]),
        "mini": np.array([[.10, .20, .70], [.10, .20, .70], [.15, .65, .20], [.10, .20, .70]]),
        "retained": np.full((4, 2, 3), np.nan),
        "active": np.array([True, True, True, False]),
    }
    evidence = agreement.independent_evidence(record)
    probability = np.full(4, .60)
    rule = {"margin": 0., "min_probability": .40}

    np.testing.assert_array_equal(agreement.scope(record, evidence, probability, rule), [True, False, False, False])


def test_positive_window_gain_can_be_reduced_only_when_it_introduced_the_revision(monkeypatch):
    monkeypatch.setattr(agreement.pair, "reference_result", lambda *args: (None, np.array([False, True, False])))
    monkeypatch.setattr(agreement.continuity, "acoustic_margin", lambda record: np.array([.25, .25, -.25]))
    initial = np.tile([.20, .55, .25], (3, 1))
    previous = np.array([[.12, .28, .60], [.12, .28, .60], [.20, .55, .25]])
    record = {"initial": initial}
    base = np.array([.42, .57, .60])
    post = previous[:, agreement.POSITIVE]
    reference = {"mix": 1., "min_target": .50, "min_lead": .15}
    settings = {"min_target": .50, "min_lead": .10}
    introduced = agreement.introduced_by_acoustics(record, base, reference)

    scores = agreement.revised_decision(record, previous, post, np.full(3, .20), introduced,
                                        {"share": .75}, reference, settings)

    np.testing.assert_array_equal(introduced, [True, False, False])
    np.testing.assert_array_equal(scores, np.vstack((initial[0], previous[1:])))
