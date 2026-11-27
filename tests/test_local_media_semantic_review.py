import json

import pytest


@pytest.fixture
def file_page(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from emotion_assistant import local_media_widgets

    app = QApplication.instance() or QApplication([])
    page = local_media_widgets.LocalMediaPage()
    submitted = []

    def submit(text):
        submitted.append(text)
        return len(submitted)

    monkeypatch.setattr(page._review_queue, "submit", submit)
    monkeypatch.setattr(local_media_widgets, "format_research_review", lambda review: review["status"])
    yield page, submitted
    page.close()
    app.processEvents()


def test_txt_review_export_and_failed_retry(tmp_path, file_page, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from emotion_assistant.text_file import analyze_text_file

    page, submitted = file_page
    path = tmp_path / "材料.txt"
    path.write_text("我只是引用，不是我现在的感受。", encoding="utf-8")
    page._path = path
    page.result = analyze_text_file(path)
    page._show_summary()
    assert not submitted
    page.semantic_button.click()
    page._review_queue.finished.emit(1, {"status": "unavailable"})
    assert page.semantic_button.isEnabled() and page.semantic_readout.text() == "unavailable"
    assert "semantic_review" not in page.result
    page.semantic_button.click()
    page._review_queue.finished.emit(2, {"status": "ok", "subject": "self"})
    assert submitted == [page.result["text"], page.result["text"]]
    target = tmp_path / "result.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(target), ""))
    page._export()
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved["semantic_review"]["text_source"] == "txt"
    assert saved["semantic_context"] == page.result["semantic_context"]


def test_media_review_tracks_segment_and_edits(file_page, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    page, submitted = file_page
    texts = ["我之前觉得不好。", "现在好多了。"]
    rows = [{"t_start": i, "t_end": i + 1, "available_modalities": []} for i in range(2)]
    page.result = {
        "kind": "video", "valid_count": 0, "sampled_count": 0, "scores": None, "reasons": {},
        "content_summary": {"decision": "abstain"}, "content_intervals": rows,
        "transcription_status": "available",
        "transcripts": [dict(row, text=text, valid=True) for row, text in zip(rows, texts)],
    }
    page._show_summary()
    assert page.content_intervals.count() == 3  # Both Chinese utterances are reviewable.
    page.content_intervals.setCurrentIndex(1)
    page.semantic_button.click()
    page.content_intervals.setCurrentIndex(2)
    page.semantic_button.click()
    page._review_queue.finished.emit(1, {"status": "ok"})
    assert "semantic_review" not in rows[0] and not page.semantic_button.isEnabled()
    page._review_queue.finished.emit(2, {"status": "ok"})
    assert rows[1]["semantic_review"]["text_source"] == "asr_segment"
    assert submitted == texts

    page.content_intervals.setCurrentIndex(0)
    page.semantic_button.click()
    assert submitted[-1] == " ".join(texts)
    monkeypatch.setattr(QInputDialog, "getMultiLineText", lambda *args: ("确认后的文字", True))
    page.review_button.click()
    page._review_queue.finished.emit(3, {"status": "ok"})
    assert "semantic_review" not in page.result
    page.semantic_button.click()
    assert submitted[-1] == "确认后的文字"
    page._review_queue.finished.emit(4, {"status": "ok"})
    assert page.result["reviewed_transcript"]["semantic_review"]["text_source"] == "user_edited"
    assert page.result["transcripts"][0]["text"] == texts[0]
    page.semantic_button.click()
    page.stop_for_close()
    page._review_queue.finished.emit(5, {"status": "unavailable"})
    assert page.semantic_readout.isHidden()


def test_unavailable_or_stale_transcript_has_no_review(file_page):
    page, submitted = file_page
    page.result = {
        "kind": "audio", "valid_count": 0, "sampled_count": 0, "scores": None, "reasons": {},
        "content_summary": {"decision": "abstain"}, "transcripts": [
            {"t_start": 0, "t_end": 1, "valid": True, "text": "stale", "stale": True}],
        "transcription_status": "available",
    }
    page._show_summary()
    assert page.semantic_button.isHidden()
    page._request_semantic_review()
    assert not submitted


def test_new_file_discards_old_review_even_for_identical_text(tmp_path, file_page):
    from emotion_assistant.text_file import analyze_text_file

    page, _ = file_page
    for name in ("first.txt", "second.txt"):
        path = tmp_path / name
        path.write_text("Same words", encoding="utf-8")
        page._reset_scores()
        page.result = analyze_text_file(path)
        page._show_summary()
        page.semantic_button.click()
    page._review_queue.finished.emit(1, {"status": "ok"})
    assert "semantic_review" not in page.result
    assert not page.semantic_button.isEnabled()
    page._review_queue.finished.emit(2, {"status": "ok"})
    assert page.result["semantic_review"]["text_source"] == "txt"


def test_typed_chinese_offline_and_review_source_are_preserved_on_export(tmp_path, file_page, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QInputDialog
    from emotion_assistant.text_file import analyze_text_file

    page, submitted = file_page
    phrase = "謝謝你"
    source = tmp_path / "sample.txt"
    source.write_text(phrase, encoding="utf-8")
    file_profile = analyze_text_file(source)["semantic_context"]["expression_profile"]
    monkeypatch.setattr(QInputDialog, "getMultiLineText", lambda *args: (phrase, True))
    page.type_button.click()
    assert not submitted and not page.content_readout.isVisible()
    assert page.result["semantic_context"]["expression_profile"] == file_profile
    assert "感激" in page.profile_readout.text() and "激活度" in page.profile_readout.text()
    assert page._path is None and page.result["filename"] is None
    page.semantic_button.click()
    assert submitted == [phrase]
    page._review_queue.finished.emit(1, {"status": "ok"})
    target = tmp_path / "typed.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(target), ""))
    page._export()
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved["text"] == phrase
    assert saved["semantic_review"]["text_source"] == saved["text_source"] == "typed"
    assert saved["semantic_context"]["expression_profile"] == file_profile
    monkeypatch.setattr(QInputDialog, "getMultiLineText", lambda *args: ("", False))
    page.type_button.click()
    assert page.result == saved
