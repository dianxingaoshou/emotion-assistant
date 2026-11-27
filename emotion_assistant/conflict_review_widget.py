"""An opt-in review of the currently selected utterance and its synchronized cues."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from .conflict_review import format_conflict, review_conflict
from .semantic_review_worker import SemanticReviewQueue


class ConflictReviewPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.queue = SemanticReviewQueue(self, callback=lambda args: review_conflict(*args))
        self.queue.finished.connect(self._finished)
        self.label = QLabel()
        self.label.setTextFormat(Qt.TextFormat.PlainText)
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.button = QPushButton("用千问复核本段分歧")
        self.button.setToolTip("把所选原话与同一时段的模型分数交给本机千问，核对本段表达。")
        self.button.clicked.connect(self._request)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        layout.addWidget(self.button, 0, Qt.AlignmentFlag.AlignLeft)
        self.clear()

    def clear(self):
        self.queue.clear()
        self._pending = self._record = self._text = None
        self.label.clear()
        self.hide()

    def select(self, record, text):
        evidence = (record or {}).get("conflict_evidence")
        if not text or not evidence or not evidence["conflict"]:
            self.clear()
            return
        if record is self._record and text == self._text:
            return
        self.clear()
        self._record, self._text = record, text
        self.label.setText(format_conflict(evidence, record.get("conflict_review")))
        self.button.setEnabled(True)
        self.show()

    def _request(self):
        if self._record is None or self._pending is not None:
            return
        self._pending = self.queue.submit((self._text, self._record["conflict_evidence"]))
        self.button.setEnabled(False)
        self.label.setText(format_conflict(self._record["conflict_evidence"]) + "\n正在本机复核…")

    def _finished(self, request_id, result):
        if self._record is None or request_id != self._pending:
            return
        self._pending = None
        if result.get("status") == "ok":
            self._record["conflict_review"] = result
        self.label.setText(format_conflict(self._record["conflict_evidence"], result))
        self.button.setEnabled(True)
