"""Session-local chat for steering the desktop regulation workflow."""
# The conversation and send action lead; status and shortcuts support them.
# Use the existing forest palette, 8/12/16/24 spacing and card radius.
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QScrollArea,
    QSizePolicy, QVBoxLayout, QWidget,
)

from .regulation_chat import ChatTurn, HISTORY_MESSAGES, MAX_MESSAGE_CHARS, local_command, reply_to_turn
from .semantic_review_worker import SemanticReviewQueue
from .ui_theme import COLORS, RADIUS, SPACE


class MessageEditor(QPlainTextEdit):
    submitted = Signal()

    def __init__(self):
        super().__init__()
        self._composing = False
        self.setTabChangesFocus(True)
        self.setAccessibleName("调节消息")
        self.setPlaceholderText("输入消息，Enter 发送，Shift + Enter 换行")
        self.setFixedHeight(58)

    def inputMethodEvent(self, event):
        self._composing = bool(event.preeditString())
        super().inputMethodEvent(event)

    def keyPressEvent(self, event):
        if (event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                and event.modifiers() == Qt.KeyboardModifier.NoModifier and not self._composing):
            self.submitted.emit()
            event.accept()
            return
        super().keyPressEvent(event)


def _respond(payload):
    return reply_to_turn(payload["text"], payload["history"],
                         listening=payload["listening"], activity=payload["activity"])


class RegulationChat(QWidget):
    def __init__(self, apply_turn, open_activity, parent=None):
        super().__init__(parent)
        self._apply_turn = apply_turn
        self.queue = SemanticReviewQueue(self, callback=_respond)
        self.queue.finished.connect(self._received)
        self._scope = self._context = self._pending = None
        self.history = []
        self.setStyleSheet(f"""
            QFrame#chatPanel {{background: {COLORS['surface']}; border-radius: {RADIUS['card']}px;}}
            QWidget#thread {{background: {COLORS['background']};}}
            QLabel#chatAssistant, QLabel#chatUser {{border-radius: {RADIUS['control']}px;
                padding: 8px 12px; font-size: 14px;}}
            QLabel#chatAssistant {{background: {COLORS['surface']}; color: {COLORS['text']};}}
            QLabel#chatUser {{background: {COLORS['primary']}; color: {COLORS['surface']};}}
            QLabel#chatStatus {{color: {COLORS['secondary_text']};}}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, SPACE["sm"], 0, 0)
        panel = QFrame()
        panel.setObjectName("chatPanel")
        layout.addWidget(panel)
        body = QVBoxLayout(panel)
        body.setContentsMargins(SPACE["lg"], SPACE["md"], SPACE["lg"], SPACE["md"])
        body.setSpacing(SPACE["sm"])
        bar = QHBoxLayout()
        title = QLabel("和情绪助手聊聊")
        title.setObjectName("sectionTitle")
        bar.addWidget(title)
        self.status = QLabel()
        self.status.setObjectName("chatStatus")
        bar.addWidget(self.status)
        bar.addStretch()
        self.activity_link = QPushButton("查看本次活动")
        self.activity_link.setObjectName("ghost")
        self.activity_link.clicked.connect(open_activity)
        bar.addWidget(self.activity_link)
        body.addLayout(bar)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        thread = QWidget()
        thread.setObjectName("thread")
        self.messages = QVBoxLayout(thread)
        self.messages.setContentsMargins(SPACE["lg"], SPACE["lg"], SPACE["lg"], SPACE["lg"])
        self.messages.setSpacing(SPACE["md"])
        self.messages.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll.setWidget(thread)
        self.scroll.viewport().installEventFilter(self)
        body.addWidget(self.scroll, 1)
        scroll_bar = self.scroll.verticalScrollBar()
        scroll_bar.rangeChanged.connect(lambda _low, high: scroll_bar.setValue(high))
        row = QHBoxLayout()
        row.setSpacing(SPACE["sm"])
        self.shortcuts = []
        for command in ("停一下", "换一种", "我只想聊聊"):
            btn = QPushButton(command)
            btn.clicked.connect(lambda _checked=False, entry=command: self.submit(entry))
            self.shortcuts.append(btn)
            row.addWidget(btn)
        row.addStretch()
        body.addLayout(row)
        composer = QHBoxLayout()
        self.editor = MessageEditor()
        self.editor.submitted.connect(self._send_editor)
        self.editor.textChanged.connect(self._enable_send)
        composer.addWidget(self.editor, 1)
        self.send = QPushButton("发送")
        self.send.setObjectName("primary")
        self.send.setFixedWidth(72)
        self.send.clicked.connect(self._send_editor)
        composer.addWidget(self.send)
        body.addLayout(composer)
        privacy = QLabel("自由表达由本机千问处理。对话原文仅在本次会话显示。")
        privacy.setObjectName("smallMuted")
        privacy.setWordWrap(True)
        body.addWidget(privacy)
        self._reset()

    def _enable_send(self):
        self.send.setEnabled(bool(self._scope and self.editor.toPlainText().strip()))

    def _reset(self):
        self.queue.clear()
        self._pending = None
        self.history.clear()
        self.editor.clear()
        while self.messages.count():
            self._remove_oldest_bubble()
        self._bubble("assistant", "你现在最想说的是什么？需要暂停、换方法，或只是聊一会儿，都可以告诉我。", "情绪助手")

    def set_context(self, scope, state, current, mode, in_trial=False):
        # Local inference may return after a new activity, retest or session.
        if self._scope != scope:
            self._scope = scope
            self._reset()
        context = (state, current.get("id"), current.get("assessment_phase"), mode)
        if self._context != context and self._pending is not None:
            self.queue.clear()
            self._pending = None
        self._context = context
        self._mode = mode
        self._activity = current.get("activity") if state == "running" else None
        self.editor.setEnabled(bool(scope))
        for button in self.shortcuts:
            button.setEnabled(bool(scope))
        self._enable_send()
        if not scope:
            label = "固定对照会话请使用「本次调节」" if in_trial else "先开始一次会话"
        elif self._pending is not None:
            label = "千问正在回复…"
        elif mode == "listening":
            label = "只聊聊 · 已暂停主动提醒"
        elif mode == "quiet":
            label = "已停止引导"
        else:
            label = {"running": "活动进行中", "ready": "等待选择活动", "confirming": "等待本人确认"}.get(state, "可以开始聊")
        self.status.setText(label)

    def _bubble(self, role, text, sender):
        item = QWidget()
        line = QHBoxLayout(item)
        line.setContentsMargins(0, 0, 0, 0)
        column = QVBoxLayout()
        column.setSpacing(SPACE["xs"])
        align = Qt.AlignmentFlag.AlignRight if role == "user" else Qt.AlignmentFlag.AlignLeft
        author = QLabel(sender)
        author.setObjectName("smallMuted")
        column.addWidget(author, 0, align)
        bubble = QLabel(text)
        bubble.setTextFormat(Qt.TextFormat.PlainText)
        bubble.setObjectName("chatUser" if role == "user" else "chatAssistant")
        bubble.setWordWrap(True)
        bubble.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._size_bubble(bubble)
        bubble.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        column.addWidget(bubble, 0, align)
        if role == "user":
            line.addStretch()
        line.addLayout(column)
        if role != "user":
            line.addStretch()
        self.messages.addWidget(item)
        if self.messages.count() > 81:
            self._remove_oldest_bubble()

    def _remove_oldest_bubble(self):
        item = self.messages.takeAt(0).widget()
        item.hide()
        item.deleteLater()

    def _size_bubble(self, bubble):
        bubble.ensurePolished()
        natural = max((bubble.fontMetrics().horizontalAdvance(s) for s in bubble.text().splitlines()), default=0)
        available = max(80, self.scroll.viewport().width() - 2 * SPACE["lg"] - SPACE["xl"])
        bubble.setFixedWidth(min(natural + 2 * SPACE["md"] + SPACE["xs"], available, 470))

    def eventFilter(self, watched, event):
        if watched is self.scroll.viewport() and event.type() == QEvent.Type.Resize:
            for name in ("chatUser", "chatAssistant"):
                for bubble in self.scroll.widget().findChildren(QLabel, name):
                    self._size_bubble(bubble)
        return super().eventFilter(watched, event)

    def _send_editor(self):
        if self.submit(self.editor.toPlainText()):
            self.editor.clear()

    def submit(self, text):
        text = text.strip()
        if not self._scope or not text:
            return False
        if len(text) > MAX_MESSAGE_CHARS:
            self.status.setText(f"每次可发送 {MAX_MESSAGE_CHARS:,} 字，请分条发送。")
            return False
        self.queue.clear()
        self._pending = None
        previous = tuple(self.history[-HISTORY_MESSAGES:])
        self._add("user", text, "我")
        command = local_command(text)
        if command is not None:
            self._show_turn(command)
        else:
            self.status.setText("千问正在回复…")
            self._pending = self.queue.submit({"text": text, "history": previous,
                                               "listening": self._mode == "listening", "activity": self._activity})
        return True

    def _add(self, role, text, label):
        if not self.history and self.messages.count():
            self._remove_oldest_bubble()
        self.history.append((role, text))
        del self.history[:-80]
        self._bubble(role, text, label)

    def _received(self, revision, turn):
        if self._scope and revision == self._pending:
            self._pending = None
            if not isinstance(turn, ChatTurn):
                turn = ChatTurn(action="reply", message="本机回复暂时中断。你仍可以使用下面的快捷指令。",
                                source="fallback", status="service_error")
            self._show_turn(turn)

    def _show_turn(self, turn):
        text = self._apply_turn(turn)
        if self._scope and text:
            sender = {"qwen": "本机千问", "local": "情绪助手 · 即时指令",
                      "fallback": "情绪助手 · 离线回应"}.get(turn.source, "情绪助手")
            if turn.status == "listening_guard":
                sender = "情绪助手 · 倾听"
            elif turn.status == "unverified_recall":
                sender = "情绪助手"
            self._add("assistant", text, sender)

    def activity_update(self, text):
        if self._scope:
            self._add("assistant", text, "情绪助手 · 活动状态")
