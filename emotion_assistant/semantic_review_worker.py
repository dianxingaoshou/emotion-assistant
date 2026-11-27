"""Run local text inference away from capture and the GUI, retaining only the latest request."""
from threading import Thread

from PySide6.QtCore import QObject, Qt, Signal, Slot

from .research_text_review import review_with_reference


class SemanticReviewQueue(QObject):
    finished = Signal(int, object)
    _returned = Signal(int, object)

    def __init__(self, parent=None, *, callback=None):
        super().__init__(parent)
        self._callback = callback
        self._revision = 0
        self._running = False
        self._pending = None
        self._returned.connect(self._complete, Qt.ConnectionType.QueuedConnection)

    def clear(self):
        """Forget pending text and invalidate any inference already in progress."""
        self._revision += 1
        self._pending = None

    def submit(self, payload):
        self._revision += 1
        self._pending = (self._revision, payload)
        if not self._running:
            self._start_pending()
        return self._revision

    def _start_pending(self):
        request_id, payload = self._pending
        self._pending = None
        self._running = True
        Thread(target=self._run, args=(request_id, payload),
               name="local-model-request", daemon=True).start()

    def _run(self, request_id, payload):
        try:
            result = (self._callback or review_with_reference)(payload)
        except Exception:
            # Do not expose user text or service response bodies in errors.
            result = {"status": "service_error", "message": "本地文字复核未能完成。"}
        try:
            self._returned.emit(request_id, result)
        except RuntimeError:
            # The owning page was destroyed while the local request was in flight.
            pass

    @Slot(int, object)
    def _complete(self, request_id, result):
        self._running = False
        if self._pending is not None:
            self._start_pending()
        elif request_id == self._revision:
            self.finished.emit(request_id, result)
