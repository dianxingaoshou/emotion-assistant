import numpy as np

from scripts.evaluate_chsims_teacher_refinement import desktop_case_inputs


def test_desktop_control_uses_the_correct_transcript_and_cues_for_each_gap():
    clean = {"text": "完整转写", "audio": "原音频分数", "visual": "原画面分数"}
    gap = {"text": "省略部分文字", "joint_text": "缺段音频重新转写", "audio": "缺段音频分数", "visual": "缺段画面分数"}
    speech, gap_speech = np.array([1.0]), np.array([2.0])
    cases = dict(desktop_case_inputs(clean, speech, gap, gap_speech))

    assert cases["complete"] == {**clean, "audio_embedding": speech}
    assert cases["no_visual"] == {**clean, "visual": None, "audio_embedding": speech}
    assert cases["text_gap"] == {**clean, "text": gap["text"], "audio_embedding": speech}
    assert cases["synchronous_gap"] == {"text": gap["joint_text"], "audio": gap["audio"],
                                        "visual": gap["visual"], "audio_embedding": gap_speech}
    assert cases["audio_only_with_gap"] == {"text": "", "audio": gap["audio"], "visual": None,
                                           "audio_embedding": gap_speech}
    assert cases["no_evidence"] == {"text": "", "audio": None, "visual": None, "audio_embedding": None}
