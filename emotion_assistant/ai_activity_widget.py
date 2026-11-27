"""Optional model suggestions inside the confirmed activity setup form."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .ai_activity import recommend_with_ai, recommendation_reason
from .regulation import ACTIVITIES
from .semantic_review import MODEL
from .semantic_review_worker import SemanticReviewQueue


class AiActivitySuggestions(QWidget):
    chosen = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scope = None
        self._context = None
        self._request = None
        self._choices = []
        self._offered = ()
        self._adopted = None
        self.queue = SemanticReviewQueue(self, callback=recommend_with_ai)
        self.queue.finished.connect(self._received)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        actions = QHBoxLayout()
        self.ask = QPushButton("用本机 AI 推荐活动")
        self.ask.setObjectName("ghost")
        self.ask.setToolTip("按本次自评、目标、可用时间和活动偏好，从可用活动中推荐一至两种。")
        self.dismiss = QPushButton("自己选择")
        self.dismiss.setObjectName("ghost")
        actions.addWidget(self.ask)
        actions.addWidget(self.dismiss)
        actions.addStretch()
        layout.addLayout(actions)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setObjectName("smallMuted")
        layout.addWidget(self.status)
        self.options = []
        for index in range(2):
            button = QPushButton()
            button.setObjectName("secondary")
            button.clicked.connect(lambda _checked=False, i=index: self._choose(i))
            reason = QLabel()
            reason.setObjectName("muted")
            reason.setWordWrap(True)
            layout.addWidget(button)
            layout.addWidget(reason)
            self.options.append((button, reason))
        self.ask.clicked.connect(self._ask)
        self.dismiss.clicked.connect(self._dismiss)
        self.reset()
        self.hide()

    def set_context(self, scope, context):
        if (self._scope, self._context) != (scope, context):
            self._scope, self._context = scope, context
            self._offered = ()
            self._adopted = None
            self.reset()
        self.setVisible(context is not None)

    def reset(self):
        self.queue.clear()
        self._request = None
        self._choices = []
        self.ask.setEnabled(self._context is not None)
        self.dismiss.hide()
        self.status.clear()
        self.status.hide()
        for button, reason in self.options:
            button.hide()
            reason.hide()

    def _ask(self):
        if self._context is None or self._request is not None:
            return
        self.reset()
        self._request = self.queue.submit(self._context)
        self.ask.setEnabled(False)
        self.dismiss.show()
        self.status.setText("本机 Qwen 正在比较可用活动，你也可以直接从列表选择。")
        self.status.show()

    def manual_choice(self, *_args):
        self._adopted = None

    def _dismiss(self):
        self.manual_choice()
        self.reset()

    def selection_record(self, activity):
        """Record exposure separately from clicking an AI choice and actually starting it."""
        if not self._offered or self._context is None:
            return None
        adopted = activity if activity == self._adopted else None
        return {"model": MODEL, "eligible_ids": list(self._context.eligible),
                "suggested_ids": list(self._offered), "selected_id": activity,
                "selection_source": "ai_suggestion" if adopted else "manual", "adopted_id": adopted}

    def _received(self, revision, result):
        if self._context is None or self._request != revision:
            return
        self._request = None
        self.ask.setEnabled(True)
        if result.get("status") != "ok":
            reason = {"unavailable": "未连接本机 Ollama", "model_missing": "本机尚未安装 qwen3:8b",
                      "timeout": "本机模型响应超时", "invalid_response": "模型返回的活动未通过校验"}
            self.status.setText(reason.get(result.get("status"), "本机推荐暂不可用") + "。可继续手动选择活动。")
            return
        self._choices = result["activities"]
        self._offered = tuple(self._choices)
        self._adopted = None
        self.status.setText("本机 Qwen 建议以下活动。选定后仍由你决定何时开始。")
        for key, (button, reason) in zip(self._choices, self.options):
            button.setText("选择 · " + ACTIVITIES[key]["title"])
            reason.setText(recommendation_reason(key, self._context))
            button.show()
            reason.show()

    def _choose(self, index):
        if self._context is not None and index < len(self._choices):
            key = self._choices[index]
            if key in self._context.eligible:
                self.chosen.emit(key)
                self._adopted = key
                self.reset()
