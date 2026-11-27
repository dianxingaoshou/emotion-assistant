"""Optional local review of a user-selected, consented live transcript."""

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .chinese_affect import HAN
from .clip_recognition import format_clip_result, recognize_cues
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
        self.foundation = QCheckBox("中文发言融合识别")
        self.foundation.setToolTip("在本机用当前发言的文字向量和同期声画分数分类。中国影视短片训练，现场发言尚待验证。")
        self.foundation_policy = QComboBox()
        self.foundation_policy.setAccessibleName("当前中文发言的融合算法")
        self.foundation_policy.addItem("固定权重（v1）", False)
        self.foundation_policy.addItem("自适应权重（v2）", True)
        self.foundation_policy.setToolTip("切换后重新识别当前发言。中文短片对照中，v1 判对的负向片段更多，"
                                          "v2 找回了更多中性和正向片段。实时发言仍待验证。")
        self.foundation_policy.hide()
        # The capture process sends scores, not raw sound, to the desktop.
        self.foundation_queue = SemanticReviewQueue(self, callback=lambda args: recognize_cues(
            args[0], audio=args[1], visual=args[2], adaptive=args[3]))
        self.foundation_queue.finished.connect(self._finish_foundation)
        self._foundation_key = self._foundation_request = None
        self.foundation_readout = QLabel()
        self.foundation_readout.setTextFormat(Qt.TextFormat.PlainText)
        self.foundation_readout.setWordWrap(True)
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
        policy_row = QHBoxLayout()
        policy_row.addWidget(self.foundation)
        policy_row.addWidget(self.foundation_policy)
        policy_row.addStretch()
        layout.addLayout(policy_row)
        layout.addWidget(self.foundation_readout)
        layout.addWidget(self.result)
        layout.addWidget(self.conflict_panel)
        self.choices.currentIndexChanged.connect(self._choose)
        self.review.clicked.connect(self._submit)
        self.foundation.toggled.connect(self.foundation_policy.setVisible)
        self.foundation.toggled.connect(self._sync_foundation)
        self.foundation_policy.currentIndexChanged.connect(self._sync_foundation)
        self.hide()

    def _invalidate(self):
        self.queue.clear()
        self._request = None
        self.result.clear()
        self.result.hide()
        self.review.setEnabled(True)

    def clear(self):
        self._invalidate()
        self.foundation.setChecked(False)
        self.foundation_policy.setCurrentIndex(0)
        self._clear_foundation()
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
        self._sync_foundation()

    def _clear_foundation(self):
        self.foundation_queue.clear()
        self._foundation_key = self._foundation_request = None
        self.foundation_readout.clear()
        self.foundation_readout.hide()

    def _sync_foundation(self):
        row = self.choices.currentData()
        if not self.foundation.isChecked() or not row or not HAN.search(row[1]):
            self._clear_foundation()
            return
        record = self._aligned.get(row[0]) or {}
        cues = (record.get("conflict_evidence") or {}).get("cues", {})
        payload = (row[1], cues.get("audio"), cues.get("visual"), self.foundation_policy.currentData())
        key = (row[0], payload)
        if key != self._foundation_key:
            self._foundation_key = key
            self._foundation_request = self.foundation_queue.submit(payload)
            self.foundation_readout.setText("正在识别本段表达…")
            self.foundation_readout.show()

    def _finish_foundation(self, revision, prediction):
        if revision == self._foundation_request:
            self.foundation_readout.setText(format_clip_result(prediction))

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
