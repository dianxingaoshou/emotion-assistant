import numpy as np

from scripts.chsims_teacher_minicpm_protected import infer, select_rules
from scripts.chsims_teacher_minicpm_review import use_voice, voice_quality
from scripts.evaluate_chsims_teacher_timeline import audio_release_metrics
from scripts.prepare_cmma_teacher_audit import choose


def test_voice_review_only_changes_a_supported_class_with_enough_observed_speech():
    negative, neutral, positive = [.65, .25, .10], [.21, .65, .14], [.11, .22, .67]
    old = np.array([negative, negative, negative, positive, negative, negative])
    voice = np.array([neutral, neutral, positive, neutral, negative, [np.nan] * 3])
    proposals = np.array([neutral, neutral, positive, neutral, neutral, neutral])
    data = {"prior": old, "voice": voice, "voice_quality": np.array([.8, .2, .8, .8, .8, 0.]),
            "active": np.array([True, True, True, True, True, False])}
    gate = {"voice_quality": .45, "voice_confidence": .10, "strict": True,
            "center_only": True, "neutral_margin": .1, "polar_margin": .2}
    new = use_voice(data, proposals, gate)
    np.testing.assert_array_equal(new.argmax(1), [1, 0, 0, 1, 0, 0])


def test_gap_voice_quality_uses_only_voiced_time_in_the_retained_windows():
    record = {"windows": {"quality": np.array([[[1., 1.2, .75], [.6, .8, .5]], [[1., 0., 1.], [1., 0., 1.]]])}}
    np.testing.assert_allclose(voice_quality(record, "synchronous_gap"), [[.84, 2., .65], [0., 0., 0.]])


def test_one_video_per_available_conversation_ignores_label_balance_and_excludes_previous_groups():
    head = "CoversationID,UtteranceID,Sentiment,StartTime,EndTime\n"
    labels = ["negative", "positive", "positive", "neutral", "neutral", "neutral"]
    rows, media = [], set()
    for index, label in enumerate(labels, 1):
        for utterance in (1, 2):
            key = f"C_{index}_U_{utterance}"
            rows.append(f"C_{index},{key},{label},00:00:00,00:00:03\n")
            media.add(f"CMMA/test/video/C_{index}/{key}.mp4")
    table = (head + "".join(rows)).encode("gb18030")
    result = choose(media, table, excluded={"C_1", "C_3"}, all_conversations=True, seed="test-expansion")
    assert [item["id"].split("/")[0] for item in result] == ["C_2", "C_4", "C_5", "C_6"]
    assert [item["truth"]["full"] for item in result].count("neutral") == 3


def test_direction_thresholds_can_recover_neutral_without_losing_an_established_polar_class():
    negative, neutral, positive = [.55, .35, .10], [.20, .70, .10], [.10, .20, .70]
    prior = np.array([negative, negative, positive])
    offered = np.array([neutral, neutral, positive])
    examples = {"prior": prior, "mini": offered, "voice": offered, "active": np.array([True] * 3),
                "voice_quality": np.array([.85, .30, .75]), "source": np.array(["a", "b", "c"]),
                "target": np.array([1, 0, 2])}
    gates = select_rules(examples, examples, offered, offered)
    result = infer(examples, {"method": "mini", "residual": None, "rules": gates})
    np.testing.assert_array_equal(result.argmax(1), [1, 0, 2])


def test_a_tied_missing_average_does_not_regress_because_of_rounding():
    old = {"accuracy": .4772727272727273, "macro_f1": .4048,
           "recall": {key: {"hits": 1} for key in ("negative", "neutral", "positive")}}
    best = {**old, "accuracy": .4772727272727272, "macro_f1": .4075}
    arms = {"teacher_previous": old, "teacher": best}
    result = {"cases": {case: {"external": {"metrics": {"teacher": old, "teacher_previous": old}}}
                       for case in ("complete", "no_visual", "synchronous_gap")},
              "averages": {"missing": {"external": {"metrics": arms}}}}
    release = audio_release_metrics(result, external=True)
    assert release["missing_no_regression"]
    assert not release["missing_improves_accuracy_and_macro_f1"]
