import os
import time
from threading import Event

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication

from emotion_assistant import semantic_review
from emotion_assistant.semantic_review_worker import SemanticReviewQueue


def test_queue_discards_obsolete_results_and_only_runs_latest_pending(monkeypatch):
    app = QApplication.instance() or QApplication([])
    started, release = Event(), Event()
    calls = []

    def review(text):
        calls.append(text)
        if text == "first":
            started.set()
            assert release.wait(3)
        return {"text": text}

    monkeypatch.setattr(semantic_review, "review_text", review)
    worker = SemanticReviewQueue()
    completed = []
    worker.finished.connect(lambda i, result: completed.append((i, result)))
    worker.submit("first")
    try:
        assert started.wait(1)
        worker.submit("obsolete")
        worker.clear()
        latest = worker.submit("corrected")
    finally:
        release.set()
    deadline = time.monotonic() + 3
    while not completed and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    assert calls == ["first", "corrected"]
    assert completed == [(latest, {"text": "corrected"})]
