"""Optional local review of a user-selected, consented live transcript."""

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .research_text_review import format_research_review
from .semantic_review_worker import SemanticReviewQueue
from .conflict_review_widget import ConflictReviewPanel


class LiveSemanticReview(QWidget):
    """Keep a bounded selection in memory, scoped to one running session."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.queue = SemanticReviewQueue(self)
        self.queue.finished.connect(self._finish)
        self._request = None
        self._pinned = False
        self._aligned = {}
        self.conflict_panel = ConflictReviewPanel()
        self.choices = QComboBox()
        self.choices.setAccessibleName("待复核的发言")
        self.choices.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.choices.setMinimumContentsLength(8)
        self.review = QPushButton("本地大模型复核")
        self.review.setToolTip("仅在点击后，将所选文字交给本机 Ollama 分析。")
        self.result = QLabel()
        self.result.setWordWrap(True)
        self.result.setTextFormat(Qt.TextFormat.PlainText)
        self.result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.result.hide()
        row = QHBoxLayout()
        row.addWidget(self.choices, 1)
        row.addWidget(self.review)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row)
        layout.addWidget(self.result)
        layout.addWidget(self.conflict_panel)
        self.choices.currentIndexChanged.connect(self._choose)
        self.review.clicked.connect(self._submit)
        self.hide()

    def _invalidate(self):
        self.queue.clear()
        self._request = None
        self.result.clear()
        self.result.hide()
        self.review.setEnabled(True)

    def clear(self):
        self._invalidate()
        self.conflict_panel.clear()
        self._aligned.clear()
        self._pinned = False
        with QSignalBlocker(self.choices):
            self.choices.clear()
        self.choices.setToolTip("")
        self.hide()

    def observe(self, segment_id, text, end):
        previous = self.choices.currentData()
        rows = [self.choices.itemData(i) for i in range(self.choices.count())]
        old = next((row for row in rows if row[0] == segment_id), None)
        if old and old != (segment_id, text, end):
            self._aligned.pop(segment_id, None)
        rows = [row for row in rows if row[0] != segment_id]
        if text:
            rows.append((segment_id, text, end))
        rows.sort(key=lambda row: row[2], reverse=True)
        if not rows:
            self.clear()
            return
        # Keep the most recent twelve. A pinned selection survives new speech.
        pinned = next((row for row in rows if self._pinned and previous and row[0] == previous[0]), None)
        rows = rows[:12]
        if pinned and pinned not in rows:
            rows[-1] = pinned
        selected = pinned or rows[0]
        self._pinned = pinned is not None
        with QSignalBlocker(self.choices):
            self.choices.clear()
            for key, value, finish in rows:
                preview = " ".join(value.split())
                if len(preview) > 36:
                    preview = preview[:35] + "…"
                self.choices.addItem(f"{finish:.1f}s · {preview}", (key, value, finish))
            self.choices.setCurrentIndex(rows.index(selected))
        self.choices.setToolTip(selected[1])
        if selected != previous:
            self._invalidate()
        keep = {r[0] for r in rows}
        self._aligned = {k: v for k, v in self._aligned.items() if k in keep}
        self._sync_conflict()
        self.show()

    def observe_content(self, record):
        key = record.get("conflict_segment_id")
        current = next((self.choices.itemData(i) for i in range(self.choices.count())
                        if self.choices.itemData(i)[0] == key), None)
        if current and current[2] == record.get("t_end"):
            previous = self._aligned.get(key)
            if previous and previous.get("t_start") == record.get("t_start") and (
                    previous.get("conflict_evidence") == record.get("conflict_evidence")):
                return
            self._aligned[key] = record
        else:
            # An ambiguous join may invalidate a previous estimate at the same boundary.
            for old_key, row in list(self._aligned.items()):
                if (row.get("t_start"), row.get("t_end")) == (record.get("t_start"), record.get("t_end")):
                    del self._aligned[old_key]
        self._sync_conflict()

    def _sync_conflict(self):
        row = self.choices.currentData()
        self.conflict_panel.select(self._aligned.get(row[0]) if row else None, row[1] if row else None)

    def _choose(self, _index):
        self._pinned = True
        self._invalidate()
        self.choices.setToolTip(self.choices.currentData()[1])
        self._sync_conflict()

    def _submit(self):
        row = self.choices.currentData()
        if not row or self._request is not None:
            return
        self._pinned = True
        self.review.setEnabled(False)
        self.result.setText("正在本机复核所选发言…")
        self.result.show()
        self._request = self.queue.submit(row[1])

    def _finish(self, request, payload):
        if request != self._request:
            return
        self._request = None
        self.review.setEnabled(True)
        self.result.setText(format_research_review(payload))
        self.result.show()
