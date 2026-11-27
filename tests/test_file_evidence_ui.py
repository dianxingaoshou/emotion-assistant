import json
import os

import pytest

from emotion_assistant.local_media import analyze_media
from emotion_assistant.text_file import analyze_text_file


def test_text_file_needs_no_av_models_and_keeps_semantics_separate(tmp_path):
    path = tmp_path / "英文表达.txt"
    path.write_text("I absolutely love this wonderful, amazing movie.", encoding="utf-8-sig")
    result = analyze_media(path, model_dir=tmp_path, on_sample=None, cancelled=lambda: False)
    assert result["kind"] == "text" and result["scores"] is None
    polarity = result["semantic_context"]["english_polarity"]
    assert polarity["label"] == "positive" and sum(polarity["scores"].values()) == pytest.approx(1)
    path.write_text("我现在想休息。", encoding="utf-8")
    result = analyze_text_file(path)
    assert result["semantic_context"]["need_hint"] == "fatigue"
    assert "english_polarity" not in result["semantic_context"]
    path.write_bytes(b"\xff\x00")
    with pytest.raises(ValueError, match="UTF-8"):
        analyze_text_file(path)


def test_file_ui_displays_actual_window_and_abstention(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from emotion_assistant.local_media_widgets import LocalMediaPage
    app = QApplication.instance() or QApplication([])
    page = LocalMediaPage()
    page.show()
    app.processEvents()
    try:
        scores = {"positive": .55, "negative": .35, "neutral": .1, "uncertain": 0}
        exp = {"available": ["audio", "visual"], "target": "positive", "score": .55,
               "without": {"visual": {"target_score": .31}, "audio": {"target_score": .79}}}
        rows = [{"t_start": 0, "t_end": 1, "status": "visual_only", "decision": "observe_only",
                 "scores": scores}, {"t_start": 1, "t_end": 3, "status": "joint", "decision": "abstain",
                                     "conflict": True, "scores": scores, "evidence_explanation": exp}]
        page.result = {"kind": "video", "mode": "mixed", "valid_count": 2, "sampled_count": 2,
                       "audio_valid_count": 1, "scores": scores, "fusion_intervals": rows,
                       "decision_counts": {"observe_only": 1, "abstain": 1}, "reasons": {}}
        page._show_summary()
        assert page.intervals.currentIndex() == 1
        assert "31.0%" in page.interval_detail.text() and "分歧" in page.interval_detail.text()
        page.intervals.setCurrentIndex(0)
        assert "仅观察" in page.interval_detail.text()
        page._reset_scores()
        page.result["decision_counts"] = {"abstain": 2}
        page._show_summary()
        assert not page.aggregate.isVisible() and "暂缓判断" in page.details.text()
        json.dumps(page.result, allow_nan=False)
    finally:
        page.close()


def test_content_view_identifies_used_channels_and_separates_experimental_scores(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from emotion_assistant.local_media_widgets import LocalMediaPage
    app = QApplication.instance() or QApplication([])
    page = LocalMediaPage()
    page.show()
    try:
        text_scores = {"positive": .55, "negative": .44, "neutral": .01}
        candidate = {"positive": .48, "negative": .50, "neutral": .02}
        item = {"decision": "estimate", "status": "text_preferred", "label": "positive",
                "scores": text_scores, "candidate_scores": candidate, "disagreement": True,
                "candidate_estimate_scores": candidate, "candidate_decision": "estimate",
                "candidate_label": "negative", "candidate_estimate_weights": {"text": .80, "visual": .10, "audio": .10},
                "used_modalities": ["text"], "available_modalities": ["text", "audio", "visual"],
                "candidate_weights": {"text": .80, "visual": .10, "audio": .10}}
        page.result = {"kind": "video", "valid_count": 2, "sampled_count": 2, "scores": None,
                       "transcription_status": "available", "content_summary": item, "reasons": {},
                       "content_intervals": [dict(item, t_start=2., t_end=4.)]}
        page._show_summary()
        app.processEvents()
        assert "正向 55.0%" in page.content_readout.headline.text()
        assert "依据：文字" in page.content_readout.basis.text()
        page.content_intervals.setCurrentIndex(1)
        assert "2.0–4.0" in page.content_readout.headline.text()
        page.content_readout.compare.click()
        assert page.content_readout.comparison.isVisible() and "负向 50.0%" in page.content_readout.comparison.text()
        page.content_readout.set_result(dict(item, candidate_label=None, candidate_decision="abstain",
                                             candidate_estimate_scores=None, candidate_estimate_weights={}))
        assert "暂缓判断" in page.content_readout.comparison.text()
        assert "50.0%" not in page.content_readout.comparison.text()
        page._reset_scores()
        assert page.content_readout.isHidden() and not page.content_readout.comparison.text()
    finally:
        page.close()


def test_intensity_only_utterance_is_selectable_and_zero_is_displayed(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from emotion_assistant.local_media_widgets import LocalMediaPage
    app = QApplication.instance() or QApplication([])
    page = LocalMediaPage()
    page.show()
    try:
        row = {"decision": "abstain", "available_modalities": [], "t_start": 3., "t_end": 5.,
               "intensity": {"value": 0., "used_modalities": ["text"]}}
        page.result = {"kind": "audio", "valid_count": 0, "sampled_count": 0, "scores": None,
                       "transcription_status": "available", "reasons": {}, "content_intervals": [row],
                       "content_summary": dict(row, intensity={"value": -.75}, intensity_covered_seconds=2.)}
        page._show_summary()
        app.processEvents()
        assert "-0.75" in page.content_readout.intensity.text() and "2.0 秒" in page.content_readout.intensity.text()
        assert page.content_intervals.count() == 2
        page.content_intervals.setCurrentIndex(1)
        assert "0.00" in page.content_readout.intensity.text()
        assert "暂缓" in page.content_readout.headline.text()
        page.content_readout.set_result(dict(row, intensity=None))
        assert "暂缓" in page.content_readout.intensity.text()
        page.content_readout.set_result({"decision": "abstain"})
        assert page.content_readout.intensity.isHidden()
    finally:
        page.close()


def test_file_review_preserves_automatic_result_and_exports_only_confirmed_edits(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog
    from emotion_assistant.english_intensity import predict_english_intensity
    from emotion_assistant.expression_profile import english_expression_profile
    from emotion_assistant.local_media_widgets import LocalMediaPage

    app = QApplication.instance() or QApplication([])
    page = LocalMediaPage()
    page.show()
    page._path = tmp_path / "interview.mp4"
    original = "I love this"
    corrected = "I do not love this at all"
    automatic = {"value": .45, "used_modalities": ["text"]}
    summary = {"intensity": automatic, "available_modalities": ["text"]}
    page.result = {"kind": "video", "valid_count": 0, "sampled_count": 0, "scores": None, "reasons": {},
                   "transcripts": [], "content_summary": summary,
                   "content_intervals": [dict(summary, t_start=0., t_end=1.)]}
    try:
        page._show_summary()
        assert page.review_button.isHidden()
        page.result.update(transcription_status="available", transcripts=[
            {"valid": True, "t_start": 0., "text": original},
            {"valid": False, "t_start": 1., "text": "discarded"}])
        page._show_summary()
        assert not page.review_button.isHidden()
        replies = iter([(original, False), (corrected, True), ("试" * 6000, True), ("不太喜欢", True), ("", True)])
        initial = []

        def edit(_parent, _title, _label, value):
            initial.append(value)
            return next(replies)

        monkeypatch.setattr(QInputDialog, "getMultiLineText", edit)
        page.review_button.click()
        assert "reviewed_transcript" not in page.result
        page.review_button.click()
        expected = predict_english_intensity(corrected)
        assert page.result["reviewed_transcript"] == {"text": corrected, "source": "user_edited",
                                                     "method": "reviewed_english_text_v1", "intensity": expected,
                                                     "expression_profile": english_expression_profile(corrected)}
        assert "校对后" in page.reviewed_intensity.text() and "+0.45" in page.content_readout.intensity.text()
        page.content_intervals.setCurrentIndex(1)
        assert page.review_button.isHidden() and page.reviewed_intensity.isHidden()
        page.content_intervals.setCurrentIndex(0)
        assert not page.reviewed_intensity.isHidden()
        target = tmp_path / "review.json"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(target), ""))
        page._export()
        exported = json.loads(target.read_text("utf-8"))
        assert exported["transcripts"][0]["text"] == original
        assert exported["content_summary"]["intensity"] == automatic
        assert exported["reviewed_transcript"]["intensity"] == expected
        page.review_button.click()
        assert "16 KiB" in page.status.text() and page.result["reviewed_transcript"]["text"] == corrected
        page.review_button.click()
        assert page.result["reviewed_transcript"]["intensity"] is None and "暂缓" in page.reviewed_intensity.text()
        page.review_button.click()
        assert "reviewed_transcript" not in page.result and page.reviewed_intensity.isHidden()
        assert initial == [original, original, corrected, corrected, "不太喜欢"]
    finally:
        page.close()
