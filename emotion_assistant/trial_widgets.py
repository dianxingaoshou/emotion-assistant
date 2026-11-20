"""Qt views for the optional two-period crossover study.

Primary: the participant's next step. Secondary: consent and the two-period
instructions. Tertiary: aggregate results and withdrawal. Reuse the desktop
SPACE scale, cardTitle / smallMuted labels and primary / ghost button roles.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QFrame, QHBoxLayout, QHeaderView, QLabel,
                               QPushButton, QScrollArea, QSizePolicy, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from .ui_theme import SPACE


def _label(text, role="muted"):
    widget = QLabel(text)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    widget.setWordWrap(True)
    widget.setObjectName(role)
    return widget


def _button(text, role="ghost"):
    button = QPushButton(text)
    button.setObjectName(role)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    return button


def _group(layout, title):
    frame = QFrame()
    frame.setObjectName("card")
    inner = QVBoxLayout(frame)
    inner.setContentsMargins(SPACE["xl"], SPACE["lg"], SPACE["xl"], SPACE["lg"])
    inner.setSpacing(SPACE["md"])
    inner.addWidget(_label(title, "cardTitle"))
    layout.addWidget(frame)
    return inner


class TrialPage(QScrollArea):
    prepare = Signal()
    withdraw = Signal()
    refresh = Signal()
    export = Signal()

    def __init__(self):
        super().__init__()
        content = QWidget()
        self.setWidgetResizable(True)
        self.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE["lg"])
        top = _group(layout, "两次会话，体验两种安排")
        self.progress = _label("", "infoStrip")
        self.step = _label("")
        top.addWidget(self.progress)
        top.addWidget(self.step)
        top.addWidget(_label("两次尽量选择相似的时段、任务、时长和设备。提示和活动排序的安排随机决定；"
                             "按平常方式使用即可，不用刻意制造困扰。可拒绝活动，跳过活动后自评，或不预约稍后复测。"))
        self.consent = QCheckBox("我自愿将下一场会话纳入这项对照实验")
        self.consent.setToolTip("授权本次实验记录提示、选择、活动与自评；不授权语音转写。每场需单独勾选。")
        top.addWidget(self.consent)
        top.addWidget(_label("仅在本机汇总提示、活动与自评。未勾选时仍可正常使用；退出后该档案不再参与实验统计。", "smallMuted"))
        actions = QHBoxLayout()
        self.setup_button = _button("前往会话设置", "primary")
        self._needs_alias = False
        self.consent.toggled.connect(lambda checked: self.setup_button.setEnabled(self._needs_alias or checked))
        self.setup_button.clicked.connect(self.prepare)
        self.withdraw_button = _button("退出实验")
        self.withdraw_button.clicked.connect(self.withdraw)
        actions.addWidget(self.setup_button)
        actions.addWidget(self.withdraw_button)
        actions.addStretch()
        top.addLayout(actions)

        report = _group(layout, "本机实验汇总")
        self.summary = _label("")
        report.addWidget(self.summary)
        self.results = QTableWidget(0, 4)
        self.results.setHorizontalHeaderLabels(["指标", "基础策略", "自适应策略", "配对差值（人数）"])
        self.results.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.results.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.results.verticalHeader().hide()
        self.results.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3):
            self.results.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.results.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.results.setAccessibleName("按策略汇总的实验指标")
        report.addWidget(self.results)
        self.results.hide()
        links = QHBoxLayout()
        self.report_button = _button("查看汇总")
        self.report_button.clicked.connect(self.refresh)
        self.export_button = _button("导出汇总 JSON")
        self.export_button.clicked.connect(self.export)
        links.addWidget(self.report_button)
        links.addWidget(self.export_button)
        links.addStretch()
        report.addLayout(links)
        layout.addStretch()

    def set_context(self, name, mode, state, *, active, enrolled):
        source = "模拟演练（非真人结果）" if mode == "simulation" else "真人测试"
        status = state.get("status", "new")
        period = (enrolled or {}).get("period") or state.get("next_period") or 1
        valid_name = bool(name and name != "默认用户")
        if enrolled:
            title = f"正在进行第 {period} 场 / 共 2 场"
            hint = "按平时的方式使用。在系统主动询问时，可以单独评价提示时机；结束会话后再查看下一步。"
        elif active:
            title, hint = "当前是日常会话", "结束本次日常会话后，再为下一场选择是否加入实验。"
        elif status == "complete":
            title, hint = "两场会话已完成", "如已自愿预约稍后复测，可在「调节与复测 → 稍后复测」补填。"
        elif status == "withdrawn":
            title, hint = "已退出实验", "这个档案仍可用于日常观察。"
        elif status == "in_progress":
            title, hint = "有一场实验尚未结束", "请先结束当前实验或选择退出，以免重复分配。"
        elif status == "incomplete":
            title, hint = "两场名额已分配，记录尚不完整", "已中断或删除的实验场次不会重新分配。可查看已有记录的汇总，也可退出实验。"
        elif not valid_name:
            title, hint = "先设置参与者代号", "在上方会话设置中把「默认用户」改为专属代号，例如 P01。两场都使用同一个代号。"
        else:
            title = f"待开始第 {period} 场 / 共 2 场"
            if period == 2:
                first = "第一场缺少完整记录，后续可能无法形成配对。" if state.get("incomplete_periods") else "第一场已经结束。"
                hint = first + "使用相同代号和来源，为第二场重新勾选同意。"
            else:
                hint = "使用专属代号，不需要填写姓名。首次开始实验时，程序会分配两种安排的先后顺序。"
        self.progress.setText(f"{source}  ·  {title}")
        self.step.setText(hint)
        closed = status in ("complete", "withdrawn", "in_progress", "incomplete")
        available = not active and valid_name and not closed
        self.consent.setEnabled(available)
        if not available:
            self.consent.setChecked(False)
        self.setup_button.setVisible(not active and not closed)
        self._needs_alias = not valid_name
        self.setup_button.setEnabled(not valid_name or self.consent.isChecked())
        self.setup_button.setText("前往会话设置" if valid_name else "设置参与者代号")
        self.withdraw_button.setVisible(bool(enrolled) or status in ("in_progress", "next", "complete", "incomplete"))

    def reset_report(self, mode):
        source = "模拟演练" if mode == "simulation" else "真人测试"
        self.summary.setText(f"查看这台电脑上所有参与者的{source}汇总。真实与模拟分别统计。")
        self.results.hide()
        self.report_button.setText("查看汇总")

    def show_report(self, data):
        source = "模拟演练（非真人结果）" if data["mode"] == "simulation" else "真人测试"
        n = data["participants"]
        if n == 0:
            self.summary.setText(f"{source} · 还没有可纳入的实验记录。完成实验会话后可回到这里刷新。")
            self.results.hide()
            return
        self.summary.setText(f"{source} · {n} 位参与者，{data['paired_participants']} 位完成两种安排。"
                             "仅统计已结束的实验场次。差值为同一人「自适应减基础」后取平均；「—」表示不适用或缺少数据。")
        metrics = [("观察时长（小时）", "active_hours", ".2f"),
                   ("可评估提示负担的时长（小时）", "burden_eligible_hours", ".2f"),
                   ("主动提示次数", "prompts_asked", "d"),
                   ("提示时机已评价数", "prompt_rated", "d"),
                   ("明确不想要的提示 / 小时", "unwanted_per_hour", ".2f"),
                   ("提示接受率", "acceptance_rate", ".0%"),
                   ("活动完成率", "completion_rate", ".0%"),
                   ("即时困扰平均下降（分）", "immediate_mean_delta", ".2f"),
                   ("完成活动但缺少即时评分", "post_missing", "d"),
                   ("稍后困扰平均下降（分）", "followup_mean_delta", ".2f"),
                   ("已预约但缺少稍后评分", "followup_missing", "d")]
        self.results.setRowCount(len(metrics))
        for row, (label, key, fmt) in enumerate(metrics):
            self.results.setItem(row, 0, QTableWidgetItem(label))
            for column, condition in enumerate(("basic", "adaptive"), 1):
                value = data["by_condition"][condition].get(key)
                item = QTableWidgetItem("—" if value is None else format(value, fmt))
                if key in ("burden_eligible_hours", "unwanted_per_hour"):
                    item.setToolTip("只纳入有效视觉或声学覆盖达到非暂停时长一半的场次；缺少提示时机评价时不填估计值。")
                self.results.setItem(row, column, item)
            pair = data["paired"].get(key, {})
            diff = pair.get("mean_difference")
            def delta(x):
                return f"{100*x:+.1f} 个百分点" if "%" in fmt else f"{x:+.2f}"
            item = QTableWidgetItem("—" if diff is None else f"{delta(diff)}（{pair['n']} 人）")
            if pair.get("ci95"):
                lo, hi = pair["ci95"]
                item.setToolTip(f"按参与者重采样的 95% 区间：{delta(lo)} 到 {delta(hi)}")
            self.results.setItem(row, 3, item)
        self.results.resizeRowsToContents()
        self.results.setFixedHeight(self.results.horizontalHeader().height() + self.results.verticalHeader().length()
                                    + 2 * self.results.frameWidth() + SPACE["xs"])
        self.results.show()
        self.report_button.setText("刷新汇总")


class PromptTimingFeedback(QWidget):
    rated = Signal(bool)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE["xs"])
        self.note = _label("这条提示来得合适吗？评价时机不影响是否接受活动，可跳过。", "smallMuted")
        layout.addWidget(self.note)
        self.options = QWidget()
        buttons = QHBoxLayout(self.options)
        buttons.setContentsMargins(0, 0, 0, 0)
        self.wanted = _button("此时希望收到")
        self.unwanted = _button("此时不想收到")
        self.wanted.clicked.connect(lambda: self.rated.emit(True))
        self.unwanted.clicked.connect(lambda: self.rated.emit(False))
        buttons.addWidget(self.wanted)
        buttons.addWidget(self.unwanted)
        layout.addWidget(self.options)
        self.hide()

    def reset(self):
        self.note.setText("这条提示来得合适吗？评价时机不影响是否接受活动，可跳过。")
        self.options.show()

    def submitted(self, wanted):
        self.note.setText("已记录：" + ("此时希望收到。" if wanted else "此时不想收到。") + "仍可自由决定是否接受活动。")
        self.options.hide()
