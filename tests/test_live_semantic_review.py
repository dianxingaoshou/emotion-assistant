import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from emotion_assistant import controller as controller_module, live_semantic_review
from emotion_assistant.ui import MainWindow


class PendingReviews(QObject):
    finished = Signal(int, object)

    def __init__(self, parent):
        super().__init__(parent)
        self.texts = []

    def submit(self, text):
        self.texts.append(text)
        return len(self.texts)

    def clear(self):
        pass  # Simulate a response that arrives after cancellation.


def test_live_review_is_optional_pinned_and_scoped_to_consent(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(live_semantic_review, "SemanticReviewQueue", PendingReviews)
    monkeypatch.setattr(live_semantic_review, "format_research_review", lambda result: result["message"])
    monkeypatch.setattr(controller_module, "DATA", tmp_path)
    controller = controller_module.Controller(tmp_path / "live.sqlite3")
    monkeypatch.setattr(controller, "refresh_devices", lambda: None)
    window = MainWindow(controller)
    panel = window.live_review
    try:
        window._on_session({"session_id": "s1", "state": "running"})
        first = {"session_id": "s1", "segment_id": "one", "t_end": 2., "text": "a past feeling"}
        window._on_transcript(first)
        assert panel.choices.count() == 0
        window.transcript_consent.setChecked(True)
        window._on_transcript(first)
        assert panel.queue.texts == []
        panel.review.click()
        assert panel.queue.texts == ["a past feeling"]
        window._on_transcript(dict(first, segment_id="two", t_end=4., text="feeling better"))
        assert panel.choices.currentData()[0] == "one"
        panel.queue.finished.emit(1, {"message": "review of one"})
        assert panel.result.text() == "review of one"

        panel.choices.setCurrentIndex(0)
        panel.review.click()
        assert panel.queue.texts[-1] == "feeling better"
        window._on_transcript(dict(first, segment_id="two", t_end=4., text="revised words"))
        panel.queue.finished.emit(2, {"message": "outdated"})
        assert panel.result.isHidden()
        panel.review.click()
        assert panel.queue.texts[-1] == "revised words"
        window.transcript_consent.setChecked(False)
        panel.queue.finished.emit(3, {"message": "revoked"})
        assert panel.isHidden() and panel.choices.count() == 0 and panel.result.text() == ""

        window.transcript_consent.setChecked(True)
        window._on_transcript(first)
        panel.review.click()
        window._on_session({"session_id": "s2", "state": "running"})
        panel.queue.finished.emit(4, {"message": "old session"})
        window._on_transcript(first)
        assert panel.choices.count() == 0 and panel.result.isHidden()
    finally:
        window.close()
        app.processEvents()
