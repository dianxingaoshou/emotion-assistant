"""Feedback-oriented regulation components, using the existing Qt theme.

Refactoring UI §2.1/§2.8/§3.6/§8.4 constraint brief:
primary = current guide step and one next action; secondary = user outcome and
recommendation reasons; tertiary = sample counts and comparable-model metadata.
SPACE/TYPE/COLORS come from ui_theme; no additional palette or dependency.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .ui_theme import COLORS, SPACE, TYPE


def label(text, role="muted"):
    widget = QLabel(text)
    widget.setObjectName(role)
    widget.setWordWrap(True)
    return widget


class RecommendationCards(QWidget):
    chosen = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items = []
        self.selected = None
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.layout_.setSpacing(SPACE["md"])
        self.set_items([])

    def set_items(self, items, selected=None):
        self.items = list(items or [])
        self.selected = selected
        while self.layout_.count():
            item = self.layout_.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        candidates = [item for item in self.items if not item.get("blocked")][:3]
        if not candidates:
            self.layout_.addWidget(label("选择当前需要和可用时间后，查看适合尝试的活动。个人反馈会在这里逐步积累。"))
            return
        for item in candidates:
            frame = QFrame()
            frame.setObjectName("infoStrip")
            content = QVBoxLayout(frame)
            content.setContentsMargins(SPACE["md"], SPACE["sm"], SPACE["md"], SPACE["sm"])
            content.setSpacing(SPACE["xs"])
            heading = QHBoxLayout()
            heading.addWidget(label(item.get("title", item["activity"]), "cardTitle"), 1)
            choose = QPushButton("已选择" if item["activity"] == selected else "选择")
            choose.setObjectName("ghost")
            choose.setCursor(Qt.CursorShape.PointingHandCursor)
            choose.clicked.connect(lambda checked=False, activity=item["activity"]: self.chosen.emit(activity))
            heading.addWidget(choose)
            content.addLayout(heading)
            reasons = [str(reason) for reason in item.get("reasons", [])]
            content.addWidget(label("；".join(reasons[:2]) or "根据本次需要提供，可自主选择。", "smallMuted"))
            count = int(item.get("history_evidence_count", item.get("independent_evidence_count", item.get("evidence_count", 0))) or 0)
            scope = "同需求" if item.get("context_used") else "跨需求低权重参考"
            content.addWidget(label(f"{scope}独立反馈 {count} 次" + (" · 反馈仍不足" if count < 3 else " · 仅作个人选择参考"), "smallMuted"))
            self.layout_.addWidget(frame)


class OutcomeCard(QFrame):
    """Self-report and subjective comfort remain separate visible evidence."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("infoStrip")
        box = QVBoxLayout(self)
        box.setContentsMargins(SPACE["lg"], SPACE["lg"], SPACE["lg"], SPACE["lg"])
        box.setSpacing(SPACE["md"])
        self.heading = label("本次反馈已保存", "cardTitle")
        self.caption = label("", "bodyNote")
        self.ratings = label("", "cardTitle")
        self.detail = label("", "muted")
        self.next_action = label("", "muted")
        box.addWidget(self.heading)
        box.addWidget(self.caption)
        box.addWidget(self.ratings)
        box.addWidget(self.detail)
        box.addWidget(self.next_action)
        self.hide()

    def set_record(self, record, mode="real"):
        if not record or record.get("status") not in ("completed", "cancelled"):
            self.hide()
            return
        self.show()
        result = record.get("loop_result") or {}
        followup = record.get("followup") or {}
        if (followup.get("status") == "completed" and
                (followup.get("loop_result") or {}).get("state") == "discomfort"):
            result = followup["loop_result"]
        uncomfortable = (record.get("discomfort") or record.get("helpfulness") == "worse"
                         or result.get("state") == "discomfort")
        self.heading.setText("已记录不适反馈" if uncomfortable else "本次反馈已保存")
        if record.get("save_status") == "pending":
            self.heading.setText("正在保存本次反馈…")
        elif record.get("save_status") == "failed":
            self.heading.setText("反馈尚未保存，请重试")
        self.caption.setText(result.get("caption") or ("这次活动已中止，未把中止视为改善。" if record.get("status") == "cancelled" else "你的前后自评与主观反馈已分别记录。"))
        pre, post = record.get("pre"), record.get("post")
        self.ratings.setText(f"困扰度  {pre if pre is not None else '—'} → {post if post is not None else '无有效复测'}  / 10")
        help_label = {"helpful": "觉得有帮助", "no_change": "无明显帮助", "worse": "更不舒服", "unknown": "暂时无法评价"}.get(record.get("helpfulness"), "未填写")
        window_comparison = record.get("assessment_comparison") or {}
        comparable = window_comparison.get("comparable") is True
        comparison = "独立前后观察窗可比，仅提供辅助观察变化" if comparable else "模型前后条件不可比，不推断改善"
        detail = f"主观反馈：{help_label}\n{comparison}。"
        assessments = record.get("assessments") or {}
        if assessments:
            pre_window, post_window = assessments.get("pre") or {}, assessments.get("post") or {}
            detail += f"\n前窗覆盖 {pre_window.get('coverage', 0):.0%} · 后窗覆盖 {post_window.get('coverage', 0):.0%}。"
            self.detail.setToolTip("；".join(str(item) for item in window_comparison.get("reasons", [])))
        if comparable:
            negative_delta = (window_comparison.get("model_delta") or {}).get("negative")
            if isinstance(negative_delta, (float, int)):
                detail += f"\n模型负向线索变化（前减后）{negative_delta:+.3f}；与用户感受分别记录。"
        if followup.get("status") == "completed":
            delayed_help = {"helpful": "仍觉得有帮助", "no_change": "无明显帮助", "worse": "更不舒服", "unknown": "暂时无法评价"}.get(followup.get("helpfulness"), "未填写")
            detail += f"\n稍后复测：困扰度 {followup.get('distress', '—')} / 10；{delayed_help}。"
        elif followup.get("status") == "skipped":
            detail += "\n稍后复测缺少有效自评，无法评价持续变化。"
        self.detail.setText(detail + ("\n本条为模拟记录，单独统计。" if mode == "simulation" else ""))
        action = str(record.get("next_step", record.get("next_action", "")) or "").strip()
        note = "下一步：" + action if action else "下次推荐会参考这次反馈，仍由你决定是否尝试。"
        if followup.get("status") == "pending":
            note += "\n已加入应用内稍后复测，关闭程序后待办仍保留。"
        if record.get("save_status") in ("pending", "failed"):
            note = "保存成功后，这次反馈才会用于下次推荐；稍后复测待办也会一并保存。"
        self.next_action.setText(note)
        self.setStyleSheet(f"color:{COLORS['warning']}; background:{COLORS['warning_surface']};" if uncomfortable else "")
