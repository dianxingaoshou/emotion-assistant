import numpy as np

from scripts.chsims_teacher_directional import log_ratios
from scripts.chsims_teacher_source_guard import protect
from scripts.chsims_teacher_source_review import conflict_evidence, predict
from scripts.prepare_cmma_teacher_audit import choose


def test_window_review_only_changes_neutral_on_agreement_in_multiple_retained_channels():
    negative, neutral, positive = [.66, .22, .12], [.22, .66, .12], [.19, .22, .59]
    earlier = np.array([negative] * 5)
    local = np.array([negative] * 5)
    text = np.array([negative, negative, negative, negative, negative])
    audio = np.array([negative, neutral, negative, negative, negative])
    visual = np.array([neutral] * 5)
    full_audio = np.array([neutral, neutral, positive, neutral, neutral])
    previous = np.array([neutral, neutral, neutral, positive, neutral])
    blocks = [previous, earlier, local, earlier, text, full_audio, text, audio, visual]
    design = np.column_stack([*(log_ratios(value) for value in blocks), np.ones((5, 6)), np.zeros((5, 7))])
    batch = {"design": design, "support": np.stack([earlier, local, earlier, audio, visual], axis=1),
             "active": np.array([True, True, True, True, False])}
    policy = {"method": "time", "anchor": "none", "window_support": 2, "recover_at": .60, "neutral_at": 0.}
    result, conflict = predict(batch, previous, {"policy": policy, "calibration": None})
    np.testing.assert_array_equal(conflict, [True, True, False, True, False])
    np.testing.assert_array_equal(result.argmax(1), [0, 1, 1, 2, 1])
    np.testing.assert_allclose(conflict_evidence(batch)["local_ratio"][0], .75)


def test_cmma_sampling_excludes_entire_previous_conversations():
    fieldnames = "CoversationID,UtteranceID,Sentiment,StartTime,EndTime\n"
    entries, paths = [], set()
    for c, sentiment in enumerate(("negative", "neutral", "positive", "negative", "neutral", "positive"), 1):
        for u in (1, 2):
            entries.append(f"C_{c},C_{c}_U_{u},{sentiment},00:00:00,00:00:03\n")
            paths.add(f"CMMA/test/video/C_{c}/C_{c}_U_{u}.mp4")
    labels = (fieldnames + "".join(entries)).encode("gb18030")
    selection = choose(paths, labels, excluded={"C_1", "C_2", "C_3"}, per_class=1, seed="fresh")
    assert {row["id"].split("/")[0] for row in selection} == {"C_4", "C_5", "C_6"}
    assert selection == choose(paths, labels, excluded={"C_3", "C_2", "C_1"}, per_class=1, seed="fresh")


def test_neutral_safeguard_requires_a_separate_negative_anchor_and_no_strong_neutral_cue():
    previous = np.array([[.27, .55, .18], [.27, .55, .18], [.27, .55, .18], [.24, .20, .56]])
    proposed = np.array([[.59, .23, .18]] * 4)
    checks = {"earlier_negative": np.array([True, True, True, True]),
              "earlier_margin": np.array([.20, .09, .20, .20]),
              "strongest_window_neutral": np.array([.14, .05, .21, .14])}
    result = protect(previous, proposed, checks, {"minimum_anchor_margin": .10, "maximum_neutral_lead": .15})
    np.testing.assert_allclose(result, [proposed[0], *previous[1:]])
