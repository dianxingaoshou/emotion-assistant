"""Offline desktop interface for the voluntary observation and feedback workflow.

The controller owns capture, inference, storage and timers.  This module only
renders signals and forwards explicit user actions to the controller.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QLineF, QRectF, QTimer
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QBoxLayout, QCheckBox, QComboBox, QCompleter, QFileDialog,
    QFormLayout, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QInputDialog,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSplitter,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
    QWidget,
)


from .ui_theme import APP_STYLE, COLORS, SPACE, TYPE, RADIUS
from .ui_widgets import TrendChart, SignalDistribution, StepIndicator, ChartLegend
from .regulation import ACTIVITIES as ACTIVITY_LIBRARY, NEEDS
from .regulation_widgets import RecommendationCards, OutcomeCard


LABELS = {
    "positive": "正向", "neutral": "中性", "negative": "负向",
    "uncertain": "不确定", "unknown": "未知", "other": "其他",
    "happy": "愉快", "happiness": "愉快", "sad": "低落",
    "sadness": "低落", "angry": "愤怒", "anger": "愤怒",
    "fear": "恐惧", "fearful": "恐惧", "disgust": "厌恶",
    "surprise": "惊讶", "surprised": "惊讶", "calm": "平静",
}
ACTIVITIES = {key: item["title"] for key, item in ACTIVITY_LIBRARY.items()}
REASONS = {
    "visual_only": "本时间段仅有视觉证据，未填入语音状态。",
    "audio_only": "本时间段仅有语音证据，视觉未参与。",
    "joint": "使用本段语音实际时间区间内的表情与声音共同判断。",
    "no_usable_visual_in_interval": "本时间段没有可用的视觉证据。",
    "no_usable_modalities": "表情与语音均未通过质量检查。",
    "modalities_conflict_user_confirmation": "两种模态线索不一致，需要用户确认。",
    "invalid_interval": "时间区间无效，本次观察不参与判断。",
    "invalid_audio_interval": "语音时间区间无效，本次观察不参与判断。",
    "no_face": "未检测到人脸。", "multiple_faces": "检测到多人，暂不归入个人观察。",
    "no_speech": "当前没有有效语音。", "silence": "当前没有有效语音。",
    "model_unavailable": "本地模型尚未就绪。", "model_timeout": "模型处理超时，本次观察留空。",
    "ok": "通过当前输入质量检查。", "invalid_bgr_frame": "画面格式无效，本次观察不参与判断。",
    "invalid_face_box": "人脸区域无效。", "face_too_small": "人脸过小，请靠近摄像头。",
    "poor_lighting": "光线过暗或过亮。", "blurred_face": "人脸画面模糊。",
    "large_pose_or_unreliable_landmarks": "人脸角度偏转明显或关键点不可靠。",
    "invalid_mono_audio": "音频输入无效。", "audio_too_long": "语音片段过长。",
    "audio_timestamp_length_mismatch": "语音长度与时间区间不一致，暂不融合。",
    "audio_too_short": "语音片段太短。", "audio_clipped": "音频削波失真。",
    "audio_silent_or_too_quiet": "静音或音量过低。",
    "insufficient_speech_duration": "有效语音不足两秒，暂不生成声学结论。",
}
ACTIVITY_HINTS = {key: item["summary"] for key, item in ACTIVITY_LIBRARY.items()}


def _text(value: Any, default: str = "—") -> str:
    return default if value is None or value == "" else str(value)


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _percent(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else f"{number * 100 if abs(number) <= 1 else number:.0f}%"


def _time_label(value: Any) -> str:
    if value is None:
        return "—"
    china_time = timezone(timedelta(hours=8))
    if isinstance(value, (int, float)) and value > 1_000_000_000:
        return datetime.fromtimestamp(value, china_time).strftime("%m-%d %H:%M:%S")
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(china_time)
        return parsed.strftime("%m-%d %H:%M:%S")
    except ValueError:
        return str(value).replace("T", " ")[:19]


def _reason(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "；".join(_reason(item) for item in value)
    text = str(value)
    if text.startswith("model_unavailable:"):
        return "本地模型尚未就绪，暂不生成分数。"
    if text.startswith("inference_error:"):
        return "模型处理失败，本次观察留空；详细信息见状态与导出记录。"
    return REASONS.get(text, text)


def _ensure_chinese_font() -> None:
    # The Qt offscreen plugin on Windows may expose no system font families.
    # Load installed fonts only; normal desktop startup uses native discovery.
    if "Microsoft YaHei UI" not in QFontDatabase.families():
        font_dir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        for filename in ("msyh.ttc", "msyhbd.ttc"):
            font_path = font_dir / filename
            if font_path.is_file():
                QFontDatabase.addApplicationFont(str(font_path))


def _card(title: str, subtitle: str | None = None) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(SPACE["xl"], SPACE["lg"], SPACE["xl"], SPACE["lg"])
    layout.setSpacing(SPACE["sm"])
    label = QLabel(title)
    label.setObjectName("cardTitle")
    layout.addWidget(label)
    if subtitle:
        label = QLabel(subtitle)
        label.setObjectName("muted")
        label.setWordWrap(True)
        layout.addWidget(label)
    return frame, layout


def _button(label: str, primary: bool = False) -> QPushButton:
    button = QPushButton(label)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setMinimumHeight(40)
    if primary:
        button.setObjectName("primary")
    return button



def _session_time(value) -> str:
    seconds = max(0, int(_number(value, 0) or 0))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _ui_font(role="meta"):
    """§4.1: native drawing shares QSS pixel typography."""
    font = QFont("Microsoft YaHei UI")
    font.setPixelSize(TYPE[role])
    return font


class CameraPreview(QLabel):
    """A useful camera empty state with the same QLabel frame API."""
    def paintEvent(self, event):
        pixmap = self.pixmap()
        if pixmap is not None and not pixmap.isNull():
            super().paintEvent(event)
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#EEF4F0"))
        painter.drawRoundedRect(QRectF(self.rect()), 12, 12)
        cx, cy = self.width() / 2, self.height() / 2 - 28
        painter.setBrush(QColor("#E0EBE3"))
        painter.drawEllipse(QRectF(cx - 33, cy - 33, 66, 66))
        painter.setPen(QPen(QColor("#789487"), 1.8))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRectF(cx - 17, cy - 11, 27, 23), 4, 4)
        lens = QPainterPath()
        lens.moveTo(cx + 10, cy - 5)
        lens.lineTo(cx + 21, cy - 10)
        lens.lineTo(cx + 21, cy + 10)
        lens.lineTo(cx + 10, cy + 5)
        painter.drawPath(lens)
        painter.setPen(QColor(COLORS["tertiary_text"]))
        painter.setFont(_ui_font())
        painter.drawText(QRectF(16, cy + 47, self.width() - 32, 68),
                         Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                         self.text())


class ModalityCard(QFrame):
    def __init__(self, title: str, code: str = ""):
        super().__init__()
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACE["lg"], SPACE["lg"], SPACE["lg"], SPACE["lg"])
        layout.setSpacing(SPACE["sm"])
        header = QHBoxLayout()
        heading = QLabel(title)
        heading.setObjectName("cardTitle")
        header.addWidget(heading)
        if code:
            tag = QLabel(code)
            tag.setObjectName("smallMuted")
            header.addWidget(tag)
        header.addStretch()
        self.state = QLabel("等待数据")
        self.state.setObjectName("smallMuted")
        header.addWidget(self.state)
        layout.addLayout(header)
        self.dominant = QLabel("等待有效输入")
        self.dominant.setObjectName("dominant")
        layout.addWidget(self.dominant)
        self.distribution = SignalDistribution()
        layout.addWidget(self.distribution)
        self.scores = QLabel("正向 —   中性 —   负向 —   不确定 —")
        self.scores.setObjectName("smallMuted")
        self.scores.setWordWrap(True)
        layout.addWidget(self.scores)
        self.reason = QLabel("开始会话后显示本次观察。")
        self.reason.setObjectName("muted")
        self.reason.setWordWrap(True)
        layout.addWidget(self.reason)
        quality_row = QHBoxLayout()
        self.quality_label = QLabel("输入质量 —")
        self.quality_label.setObjectName("smallMuted")
        self.quality = QProgressBar()
        self.quality.setRange(0, 100)
        self.quality.setValue(0)
        self.quality.setTextVisible(False)
        self.quality.setFixedSize(64, 5)
        quality_row.addWidget(self.quality_label)
        quality_row.addWidget(self.quality)
        quality_row.addStretch()
        self.details_toggle = _button("详情")
        self.details_toggle.setObjectName("ghost")
        self.details_toggle.setMinimumHeight(22)
        self.details_toggle.setStyleSheet(f"padding:0; font-size:{TYPE['meta']}px;")
        self.details_toggle.setCheckable(True)
        quality_row.addWidget(self.details_toggle)
        layout.addLayout(quality_row)
        self.observation_time = QLabel("尚无观察区间")
        self.observation_time.setObjectName("smallMuted")
        layout.addWidget(self.observation_time)
        self.details = QWidget()
        detail_l = QVBoxLayout(self.details)
        detail_l.setContentsMargins(0, 3, 0, 0)
        detail_l.setSpacing(4)
        self.model = QLabel("模型：等待加载")
        self.model.setObjectName("smallMuted")
        self.model.setWordWrap(True)
        self.metrics = QLabel()
        self.metrics.setObjectName("smallMuted")
        self.metrics.setWordWrap(True)
        detail_l.addWidget(self.model)
        detail_l.addWidget(self.metrics)
        self.details.hide()
        layout.addWidget(self.details)
        self.details_toggle.toggled.connect(self.details.setVisible)

    def update_result(self, result: dict) -> None:
        valid = result.get("valid", False)
        quality = result.get("quality")
        if isinstance(quality, dict):
            quality = quality.get("score", quality.get("quality"))
        quality = _number(quality)
        eligible = valid and quality is not None and quality >= 0.2
        self.state.setText("可用" if eligible else "质量偏低" if valid else "未参与判断")
        self.state.setStyleSheet(f"color:{COLORS['success']}; font-weight:600;" if eligible else f"color:{COLORS['warning']};")
        self.quality.setValue(round(max(0, min(100, (quality or 0) * 100))))
        self.quality_label.setText(f"输入质量 {_percent(quality)}")
        self.quality.setToolTip("输入质量是工程指标，不等于模型判断正确的概率。")
        scores = result.get("scores") or result.get("mapped_scores") or {}
        self.distribution.set_scores(scores if eligible else None)
        if scores and eligible:
            dominant = max(scores, key=lambda key: scores[key])
            self.dominant.setText(f"{LABELS.get(dominant, dominant)}线索  {_percent(scores[dominant])}")
            self.scores.setText("   ".join(f"{LABELS[key]} {_percent(scores.get(key))}" for key in ("positive", "neutral", "negative", "uncertain")))
        else:
            self.dominant.setText("暂无有效线索")
            self.scores.setText("正向 —   中性 —   负向 —   不确定 —")
        self.reason.setText(_reason(result.get("reason") or "等待有效输入"))
        self.model.setText("模型：" + _text(result.get("model_version", result.get("model"))))
        start, end = _number(result.get("t_start")), _number(result.get("t_end"))
        self.observation_time.setText(f"最近区间 {_session_time(start)} – {_session_time(end)}" if start is not None and end is not None else "尚无观察区间")
        self.observation_time.setToolTip("这是该模态最近一次结果的采集区间；是否参与当前联合判断，以当前观察方式为准。")
        raw = result.get("raw_scores") or {}
        self.scores.setToolTip("原始模型输出：\n" + json.dumps(raw, ensure_ascii=False, indent=2, default=str) if raw else "当前没有原始模型输出。")
        names = {"face_count": "人脸数", "brightness": "亮度", "blur_variance": "清晰度", "rms": "音量 RMS", "duration_s": "片段秒数", "clipping_ratio": "削波比例", "sample_rate": "采样率"}
        metrics = result.get("quality_metrics") or {}
        details = []
        for key, name in names.items():
            if key in metrics:
                value = _number(metrics[key])
                details.append(f"{name} {value:.3g}" if value is not None else f"{name} {metrics[key]}")
        diagnostic = result.get("prediction_diagnostics") or {}
        if diagnostic.get("available"):
            details += [f"原始分布熵 {_number(diagnostic.get('normalized_entropy'), 0):.3f}",
                        f"最高类别 {_percent(diagnostic.get('top1'))}",
                        f"前两类差 {_percent(diagnostic.get('margin'))}"]
            if diagnostic.get("near_uniform"):
                self.reason.setText("原始预测接近均匀，证据不足，本次不参与判断。")
        for key, name in (("queue_wait_ms", "排队"), ("inference_ms", "推理"), ("end_to_end_ms", "采集结束至返回")):
            value = _number(result.get(key))
            if value is not None:
                details.append(f"{name} {value:.0f} ms")
        self.metrics.setText(" · ".join(details))
        self.metrics.setToolTip("预测分散度、输入质量与耗时分别记录；它们不是校准后的正确率。")

class BreathingVisual(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(125)
        self.running = False
        self.started = 0.0
        self.timer = QTimer(self)
        self.timer.setInterval(60)
        self.timer.timeout.connect(self.update)

    def set_running(self, running: bool) -> None:
        if self.running == running:
            return
        self.running = running
        if running:
            self.started = time.monotonic()
            self.timer.start()
        else:
            self.timer.stop()
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        phase = (time.monotonic() - self.started) % 10 if self.running else 0
        scale = (1 - math.cos(phase / 10 * 2 * math.pi)) / 2
        radius = 34 + 20 * scale
        center_x = self.width() / 2
        center_y = 58
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#e2f1ef"))
        painter.drawEllipse(QRectF(center_x - 55, center_y - 55, 110, 110))
        painter.setBrush(QColor("#75b8b0"))
        painter.drawEllipse(QRectF(center_x - radius, center_y - radius, radius * 2, radius * 2))
        painter.setPen(QColor(COLORS["breathing_text"]))
        painter.setFont(_ui_font("body"))
        painter.drawText(QRectF(0, 40, self.width(), 35), Qt.AlignmentFlag.AlignCenter,
                         ("自然吸气" if phase < 5 else "缓缓呼气") if self.running else "按舒适的节奏")
        painter.setFont(_ui_font())
        painter.drawText(QRectF(0, 108, self.width(), 17), Qt.AlignmentFlag.AlignCenter, "无需跟随圆圈 · 随时可以结束")


class MainWindow(QMainWindow):
    def __init__(self, controller):
        super().__init__()
        _ensure_chinese_font()
        self.controller = controller
        self.session_state = "idle"
        self.session_id: str | None = None
        self.activity_state = "observing"
        self.activity_current: dict = {}
        self._preview_image: QImage | None = None
        self._session_mode = "real"
        self._history: list[dict] = []
        self._has_devices = False
        self._retest_open = False
        self._recommendation_items = []
        self._recommendation_profile_id = None
        self._manual_activity_choice = False
        self._applying_recommendation_selection = False
        self._pending_followups = []
        self._activity_preferences = {}
        self._recommendation_user_name = None
        self._recommendation_mode = None
        self.setWindowTitle("情绪线索 · 桌面观察助手")
        self.resize(1440, 920)
        self.setMinimumSize(1040, 700)
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen" and self.screen() is not None:
            available = self.screen().availableGeometry()
            self.resize(min(1440, max(1040, available.width() - 64)),
                        min(920, max(700, available.height() - 64)))
        self._build()
        self._connect()
        self._set_session_controls()
        self._set_activity_controls()
        self._activity_choice_changed()
        QTimer.singleShot(0, self.controller.refresh_devices)
        QTimer.singleShot(0, self.controller.refresh_history)
        QTimer.singleShot(0, self._refresh_recommendations)


    def _build(self) -> None:
        self.setStyleSheet(APP_STYLE)
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        shell = QHBoxLayout(central)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(196)
        side_l = QVBoxLayout(sidebar)
        side_l.setContentsMargins(SPACE["lg"], SPACE["xxl"], SPACE["lg"], SPACE["xl"])
        side_l.setSpacing(8)
        brand = QHBoxLayout()
        brand.setSpacing(SPACE["md"])
        mark = QLabel("观")
        mark.setObjectName("brandMark")
        mark.setFixedSize(42, 44)
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand.addWidget(mark)
        brand_copy = QVBoxLayout()
        brand_copy.setSpacing(SPACE["xs"])
        title = QLabel("情绪线索")
        title.setObjectName("brandTitle")
        sub = QLabel("桌面观察助手")
        sub.setObjectName("brandSubtitle")
        brand_copy.addWidget(title)
        brand_copy.addWidget(sub)
        brand.addLayout(brand_copy)
        side_l.addLayout(brand)
        side_l.addSpacing(42)
        caption = QLabel("你的工作空间")
        caption.setObjectName("sidebarCaption")
        side_l.addWidget(caption)
        side_l.addSpacing(7)
        self.nav_buttons = []
        for index, label in enumerate(("实时观察", "调节与复测", "历史记录", "使用说明")):
            button = _button(f"0{index + 1}    {label}")
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.setMinimumHeight(47)
            button.clicked.connect(lambda checked=False, i=index: self.tabs.setCurrentIndex(i))
            side_l.addWidget(button)
            self.nav_buttons.append(button)
        side_l.addStretch()
        side_note = QLabel("给自己一点空间\n也让每次反馈被看见。")
        side_note.setObjectName("sidebarNote")
        side_note.setWordWrap(True)
        side_note.setStyleSheet("line-height:1.6;")
        side_l.addWidget(side_note)
        side_l.addSpacing(18)
        local = QLabel("●  仅在本机保存")
        local.setObjectName("sidebarNote")
        local.setStyleSheet(f"color:#C0DACE; font-size:{TYPE['meta']}px; padding-top:{SPACE['lg']}px; border-top:1px solid #31534A;")
        side_l.addWidget(local)
        shell.addWidget(sidebar)

        self.content_frame = QWidget()
        main_l = QVBoxLayout(self.content_frame)
        main_l.setContentsMargins(SPACE["xl"], SPACE["xl"], SPACE["xl"], SPACE["lg"])
        main_l.setSpacing(SPACE["xl"])
        heading = QHBoxLayout()
        heading.setSpacing(18)
        heading_copy = QVBoxLayout()
        heading_copy.setSpacing(SPACE["sm"])
        self.page_title = QLabel("实时观察")
        self.page_title.setObjectName("headerTitle")
        self.page_subtitle = QLabel("查看表情与语音线索，按自己的节奏暂停或调节。")
        self.page_subtitle.setObjectName("headerSubtitle")
        self.page_subtitle.setWordWrap(True)
        heading_copy.addWidget(self.page_title)
        heading_copy.addWidget(self.page_subtitle)
        heading.addLayout(heading_copy, 1)
        self.session_clock = QLabel("尚未开始")
        self.session_clock.setObjectName("smallMuted")
        heading.addWidget(self.session_clock)
        self.session_badge = QLabel("准备就绪")
        self.session_badge.setObjectName("sessionBadge")
        heading.addWidget(self.session_badge)
        main_l.addLayout(heading)

        controls = QFrame()
        controls.setObjectName("sessionPanel")
        controls_l = QVBoxLayout(controls)
        controls_l.setContentsMargins(SPACE["lg"], SPACE["md"], SPACE["lg"], SPACE["md"])
        controls_l.setSpacing(SPACE["lg"])
        controls_header = QHBoxLayout()
        controls_header.setSpacing(SPACE["md"])
        control_title = QLabel("本次会话")
        control_title.setObjectName("cardTitle")
        controls_header.addWidget(control_title)
        self.session_summary = QLabel("设置设备后，开始一次属于自己的观察")
        self.session_summary.setObjectName("smallMuted")
        self.session_summary.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        controls_header.addWidget(self.session_summary, 1)
        self.settings_toggle = _button("收起设置")
        self.settings_toggle.setObjectName("ghost")
        self.settings_toggle.setCheckable(True)
        self.settings_toggle.setChecked(True)
        self.settings_toggle.clicked.connect(self._toggle_settings)
        controls_header.addWidget(self.settings_toggle)
        self.start_button = _button("开始会话", True)
        self.pause_button = _button("暂停")
        self.stop_button = _button("结束会话")
        self.stop_button.setObjectName("danger")
        controls_header.addWidget(self.start_button)
        controls_header.addWidget(self.pause_button)
        controls_header.addWidget(self.stop_button)
        controls_l.addLayout(controls_header)

        self.settings_details = QWidget()
        settings_l = QVBoxLayout(self.settings_details)
        settings_l.setContentsMargins(0, 0, 0, 0)
        settings_l.setSpacing(12)
        self.controls_grid = QGridLayout()
        self.controls_grid.setHorizontalSpacing(SPACE["lg"])
        self.controls_grid.setVerticalSpacing(SPACE["sm"])
        self.user_name = QLineEdit("默认用户")
        self.user_name.setPlaceholderText("输入个人档案名称")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("真实设备采集", "real")
        self.mode_combo.addItem("模拟回放（非实测）", "simulation")
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("正在检查摄像头…", -1)
        self.mic_combo = QComboBox()
        self.mic_combo.addItem("不启用麦克风", None)
        for combo in (self.mode_combo, self.camera_combo, self.mic_combo):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
            combo.setMinimumWidth(0)
            combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._control_fields = []
        for caption, widget in (("个人档案", self.user_name), ("会话来源", self.mode_combo),
                                ("摄像头", self.camera_combo), ("麦克风", self.mic_combo)):
            label = QLabel(caption)
            label.setObjectName("smallMuted")
            self._control_fields.append((label, widget))
        self._control_columns = None
        self._reflow_controls(4)
        settings_l.addLayout(self.controls_grid)
        consent_row = QHBoxLayout()
        self.consent = QCheckBox("同意本次本地采集与记录")
        self.consent.setToolTip("我自愿采集并在本机保存结构化观察与活动反馈；原始视频和录音不长期保存。")
        self.transcript_consent = QCheckBox("保存语音转写（可选）")
        self.transcript_consent.setToolTip("开启后将在本地保存识别文字；未授权时不执行转写。")
        consent_row.addWidget(self.consent)
        consent_row.addSpacing(15)
        consent_row.addWidget(self.transcript_consent)
        consent_row.addStretch()
        self.refresh_button = _button("刷新设备")
        self.refresh_button.setObjectName("ghost")
        consent_row.addWidget(self.refresh_button)
        settings_l.addLayout(consent_row)
        self.mode_notice = QLabel("采集只在你点击开始后进行，原始视频与录音不长期保存。")
        self.mode_notice.setObjectName("smallMuted")
        self.mode_notice.setWordWrap(True)
        settings_l.addWidget(self.mode_notice)
        self.input_hint = QLabel()
        self.input_hint.setObjectName("warningPill")
        self.input_hint.setWordWrap(True)
        self.input_hint.hide()
        settings_l.addWidget(self.input_hint)
        controls_l.addWidget(self.settings_details)
        main_l.addWidget(controls)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._live_page(), "实时观察")
        self.tabs.addTab(self._activity_page(), "调节与复测")
        self.tabs.addTab(self._history_page(), "历史记录")
        self.tabs.addTab(self._about_page(), "使用说明")
        self.tabs.tabBar().hide()
        self.tabs.currentChanged.connect(self._on_page_changed)
        main_l.addWidget(self.tabs, 1)
        shell.addWidget(self.content_frame, 1)
        self._on_page_changed(0)
        self.tabs.setFocus(Qt.FocusReason.OtherFocusReason)
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().showMessage("就绪 · 设备仅在开始会话后运行")
        self._session_started = None
        self._clock_timer = QTimer(self)
        self._clock_timer.setInterval(1000)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start()

    def _reflow_controls(self, columns):
        if columns == self._control_columns:
            return
        for label, widget in self._control_fields:
            self.controls_grid.removeWidget(label)
            self.controls_grid.removeWidget(widget)
        for column in range(4):
            self.controls_grid.setColumnStretch(column, 0)
        for index, (label, widget) in enumerate(self._control_fields):
            row, col = (index // columns) * 2, index % columns
            self.controls_grid.addWidget(label, row, col)
            self.controls_grid.addWidget(widget, row + 1, col)
            self.controls_grid.setColumnStretch(col, 1)
        self._control_columns = columns

    def _toggle_settings(self, checked):
        self.settings_details.setVisible(bool(checked))
        self.settings_toggle.setText("收起设置" if checked else "设备设置")

    def _on_page_changed(self, index):
        titles = (("实时观察", "查看表情与语音线索，按自己的节奏暂停或调节。"),
                  ("调节与复测", "先确认自己的需要，再选择一次适合当下的活动。"),
                  ("历史记录", "回看每次观察与反馈，保留对自己有用的信息。"),
                  ("使用说明", "了解观察方式、数据保存和你的自主选择。"))
        title, subtitle = titles[index]
        self.page_title.setText(title)
        self.page_subtitle.setText(subtitle)
        for i, button in enumerate(self.nav_buttons):
            button.setChecked(i == index)
        self._sync_action_hierarchy()

    def _sync_action_hierarchy(self):
        """Refactoring UI §2.8: emphasize the current task, not all actions."""
        if not hasattr(self, "history_export"):
            return
        history_page = self.tabs.currentIndex() == 2
        active_session = self.session_state in ("running", "paused")
        can_request = (active_session and self.activity_state in ("observing", "cooldown", "completed", "cancelled")
                       and self.activity_current.get("save_status") not in ("pending", "failed"))
        personal_page = (self.tabs.currentIndex() == 1 and hasattr(self, "regulation_tabs")
                         and self.regulation_tabs.currentIndex() != 0)
        roles = ((self.start_button, "ghost" if history_page or personal_page else "primary"),
                 (self.history_export, "primary" if history_page and self._selected_session() else "secondary"),
                 (self.retry_save_button, "primary" if self.activity_current.get("save_status") == "failed" else "secondary"),
                 (self.request_button, "primary" if can_request else "ghost"),
                 (self.activity_request_button, "primary" if can_request else "ghost"))
        for widget, role in roles:
            if widget.objectName() != role:
                widget.setObjectName(role)
                widget.style().unpolish(widget)
                widget.style().polish(widget)
                widget.update()

    def _show_session_setup(self):
        self.tabs.setCurrentIndex(0)
        self.settings_toggle.setChecked(True)
        self._toggle_settings(True)
        self.user_name.setFocus()

    def _update_clock(self):
        if self._session_started is not None and self.session_state in ("running", "paused", "stopping"):
            self.session_clock.setText("本次会话  " + _session_time(time.monotonic() - self._session_started))

    def _live_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE["xl"])
        left = QVBoxLayout()
        left.setSpacing(SPACE["xl"])
        fusion_card, fusion_l = _card("当下的观察")
        fusion_l.setSpacing(SPACE["sm"])
        result_row = QHBoxLayout()
        score_column = QVBoxLayout()
        score_column.setSpacing(2)
        score_caption = QLabel("负向线索  /  100")
        score_caption.setObjectName("eyebrow")
        score_column.addWidget(score_caption)
        self.fusion_label = QLabel("—")
        self.fusion_label.setObjectName("scoreValue")

        score_column.addWidget(self.fusion_label)
        result_row.addLayout(score_column)
        result_row.addStretch()
        provenance = QVBoxLayout()
        provenance.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.fusion_mode = QLabel("等待有效输入")
        self.fusion_mode.setObjectName("successPill")
        provenance.addWidget(self.fusion_mode, 0, Qt.AlignmentFlag.AlignRight)
        self.fusion_interval = QLabel("尚无观察区间")
        self.fusion_interval.setObjectName("smallMuted")
        provenance.addWidget(self.fusion_interval, 0, Qt.AlignmentFlag.AlignRight)
        result_row.addLayout(provenance)
        fusion_l.addLayout(result_row)
        self.fusion_reason = QLabel("开始会话后，在这里查看线索及判断依据。")
        self.fusion_reason.setObjectName("muted")
        self.fusion_reason.setWordWrap(True)
        fusion_l.addWidget(self.fusion_reason)
        self.chart = TrendChart()
        fusion_l.addWidget(self.chart, 1)
        # Refactoring UI §5.7: actual line/marker keys work without color.
        meta_row = QVBoxLayout()
        meta_row.setSpacing(SPACE["xs"])
        legend = ChartLegend()
        meta_row.addWidget(legend)
        caution = QLabel("缺失区间留空 · 非心理健康评分")
        caution.setObjectName("smallMuted")
        meta_row.addWidget(caution)
        fusion_l.addLayout(meta_row)
        self.fusion_weights = QLabel("视觉权重 —　·　语音权重 —　·　不确定 —")
        self.fusion_weights.setObjectName("smallMuted")
        self.fusion_weights.setWordWrap(True)
        fusion_l.addWidget(self.fusion_weights)
        self.evidence_progress = QLabel("自动询问需要累计 20 秒有效证据；缺口不计时，等待迟到结果后再判断。")
        self.evidence_progress.setObjectName("smallMuted")
        self.evidence_progress.setWordWrap(True)
        fusion_l.addWidget(self.evidence_progress)
        left.addWidget(fusion_card, 1)
        modals = QHBoxLayout()
        modals.setSpacing(SPACE["lg"])
        self.visual_card = ModalityCard("表情线索", "M1")
        self.audio_card = ModalityCard("语音线索", "M2")
        modals.addWidget(self.visual_card)
        modals.addWidget(self.audio_card)
        left.addLayout(modals)
        layout.addLayout(left, 7)

        right = QVBoxLayout()
        right.setSpacing(SPACE["xl"])
        preview_card, preview_l = _card("实时画面", "仅用于本次观察，不保存原始视频。")
        self.preview = CameraPreview("摄像头尚未运行\n开始会话后显示实时画面")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(230, 180)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview.setStyleSheet(f"background:#EEF4F0; color:{COLORS['tertiary_text']}; border-radius:{RADIUS['card']}px;")
        preview_l.addWidget(self.preview, 1)
        self.preview_state = QLabel("设备仅在你开始会话后运行。")
        self.preview_state.setObjectName("smallMuted")
        self.preview_state.setWordWrap(True)
        preview_l.addWidget(self.preview_state)
        right.addWidget(preview_card, 5)
        transcript_card, transcript_l = _card("语音片段", "中文转写需单独开启。")
        self.transcript = QPlainTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setPlaceholderText("暂未开启转写\n开启后，识别文字会显示在这里。")
        self.transcript.setMaximumBlockCount(60)
        self.transcript.setMinimumHeight(65)
        self.transcript.setMaximumHeight(105)
        self.transcript.setStyleSheet("background:#F8FAF8; border:0; padding:8px;")
        transcript_l.addWidget(self.transcript)
        # §3.3 / §8.4: when transcription is off, a short explanation replaces
        # an empty editor; the supporting card is sized to its actual content.
        self.transcript_notice = QLabel("转写未开启。可在下次会话前于设备设置中单独授权。")
        self.transcript_notice.setObjectName("smallMuted")
        self.transcript_notice.setWordWrap(True)
        transcript_l.addWidget(self.transcript_notice)
        self.revoke_transcript_button = _button("停止本次转写")
        self.revoke_transcript_button.setObjectName("ghost")
        self.revoke_transcript_button.hide()
        transcript_l.addWidget(self.revoke_transcript_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.transcript.hide()
        right.addWidget(transcript_card)
        action_card = QFrame()
        action_card.setObjectName("infoStrip")
        actions = QVBoxLayout(action_card)
        actions.setContentsMargins(SPACE["lg"], SPACE["md"], SPACE["lg"], SPACE["md"])
        actions.setSpacing(SPACE["sm"])
        note = QLabel("你的感受，始终由你来确认。")
        note.setWordWrap(True)
        note.setStyleSheet("font-weight:600; color:#416A58;")
        actions.addWidget(note)
        self.request_button = _button("我想调整一下", True)
        self.correct_button = _button("修正当前观察")
        self.correct_button.setObjectName("ghost")
        self.correct_button.setToolTip("记录你对当前线索的修正，并暂停重复主动提示。")
        # §2.8 / §3.6: one primary action with a quiet adjacent correction.
        self.live_action_row = QHBoxLayout()
        self.live_action_row.setSpacing(SPACE["sm"])
        self.live_action_row.addWidget(self.request_button, 2)
        self.live_action_row.addWidget(self.correct_button, 1)
        self.live_action_row.setDirection(QBoxLayout.Direction.TopToBottom if self.width() < 1280 else QBoxLayout.Direction.LeftToRight)
        actions.addLayout(self.live_action_row)
        right.addWidget(action_card)
        layout.addLayout(right, 4)
        page.setMinimumHeight(555)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        return scroll


    def _activity_page(self) -> QWidget:
        content = QWidget()
        outer = QVBoxLayout(content)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACE["xl"])
        self.activity_flow = StepIndicator()
        outer.addWidget(self.activity_flow)
        self.activity_status = QLabel("先确认自己此刻的需要，再开始一次短时调整。")
        self.activity_status.setObjectName("muted")
        self.activity_status.setWordWrap(True)
        outer.addWidget(self.activity_status)
        columns = QHBoxLayout()
        columns.setSpacing(SPACE["xl"])
        setup, setup_l = _card("从你的需要开始")
        # Refactoring UI §2.1/§2.5: user need comes before method selection.
        self.context_widget = QWidget()
        context_form = QFormLayout(self.context_widget)
        context_form.setContentsMargins(0, 0, 0, 0)
        context_form.setSpacing(SPACE["md"])
        self.need_combo = QComboBox()
        for key, title in NEEDS.items():
            self.need_combo.addItem(title, key)
        self.need_combo.setCurrentIndex(self.need_combo.findData("unsure"))
        self.available_combo = QComboBox()
        for seconds in (60, 120, 180, 300, 600):
            self.available_combo.addItem(f"{seconds // 60} 分钟", seconds)
        self.available_combo.setCurrentIndex(1)
        context_form.addRow("当前需要", self.need_combo)
        context_form.addRow("可用时间", self.available_combo)
        setup_l.addWidget(self.context_widget)
        self.recommendation_reason = QLabel("方法会根据你的需要、时间与个人反馈排序，仍由你选择。")
        self.recommendation_reason.setObjectName("muted")
        self.recommendation_reason.setWordWrap(True)
        setup_l.addWidget(self.recommendation_reason)
        self.recommend_toggle = _button("查看推荐理由与其他建议")
        self.recommend_toggle.setObjectName("ghost")
        self.recommend_toggle.setCheckable(True)
        setup_l.addWidget(self.recommend_toggle)
        self.recommendation_cards = RecommendationCards()
        self.recommendation_cards.hide()
        setup_l.addWidget(self.recommendation_cards)
        self.recommend_toggle.toggled.connect(self.recommendation_cards.setVisible)
        self.activity_options_hint = QLabel("不必等到模型给出提示。\n八类短时方法可自主选择，建议只用于参考。")
        self.activity_options_hint.setObjectName("muted")
        self.activity_options_hint.setWordWrap(True)
        setup_l.addWidget(self.activity_options_hint)
        self.activity_request_button = _button("我想调整一下", True)
        setup_l.addWidget(self.activity_request_button)
        self.confirm_widget = QWidget()
        confirm_row = QHBoxLayout(self.confirm_widget)
        confirm_row.setContentsMargins(0, 0, 0, 0)
        self.accept_button = _button("是，我需要", True)
        self.reject_button = _button("暂时不需要")
        confirm_row.addWidget(self.accept_button)
        confirm_row.addWidget(self.reject_button)
        setup_l.addWidget(self.confirm_widget)
        self.activity_summary = QLabel()
        self.activity_summary.setObjectName("infoStrip")
        self.activity_summary.setWordWrap(True)
        self.activity_summary.hide()
        setup_l.addWidget(self.activity_summary)
        self.pre_assessment_hint = QLabel("可自愿等待 30 秒辅助前测；直接开始会保留观察不足。")
        self.pre_assessment_hint.setObjectName("muted")
        self.pre_assessment_hint.setWordWrap(True)
        setup_l.addWidget(self.pre_assessment_hint)
        self.setup_widget = QWidget()
        form = QFormLayout(self.setup_widget)
        form.setContentsMargins(0, 8, 0, 0)
        form.setVerticalSpacing(SPACE["lg"])
        form.setHorizontalSpacing(SPACE["md"])
        self.pre_combo = self._rating_combo("请选择当前困扰度")
        self.activity_combo = QComboBox()
        for key, title in ACTIVITIES.items():
            self.activity_combo.addItem(title, key)
        self.duration_combo = QComboBox()
        for seconds in (60, 120, 180, 300, 600):
            self.duration_combo.addItem(f"{seconds // 60} 分钟", seconds)
        self.duration_combo.setCurrentIndex(1)
        form.addRow("调节前自评", self.pre_combo)
        form.addRow("选择活动", self.activity_combo)
        form.addRow("计划时长", self.duration_combo)
        self.task_form = QWidget()
        task_l = QFormLayout(self.task_form)
        task_l.setContentsMargins(0, 0, 0, 0)
        self.task_blocker = QLineEdit()
        self.task_blocker.setPlaceholderText("当前具体卡在哪里？")
        self.task_tried = QLineEdit()
        self.task_tried.setPlaceholderText("已经试过什么？")
        self.task_next = QLineEdit()
        self.task_next.setPlaceholderText("接下来能做的一个小动作")
        task_l.addRow("卡点", self.task_blocker)
        task_l.addRow("已尝试", self.task_tried)
        task_l.addRow("下一步", self.task_next)
        form.addRow(self.task_form)
        self.task_form.hide()
        setup_l.addWidget(self.setup_widget)
        self.begin_activity_button = _button("开始活动", True)
        setup_l.addWidget(self.begin_activity_button)
        self.scale_note = QLabel("自评 0 = 没有困扰，10 = 此刻困扰很强。\n你的自评与模型分数分别记录。")
        self.scale_note.setObjectName("smallMuted")
        self.scale_note.setWordWrap(True)
        setup_l.addWidget(self.scale_note)
        setup_l.addStretch()
        gentle = QLabel("过程由你掌握\n\n可以提前结束，也可以随时中止。\n无需刻意微笑、讲话或完成全部步骤。")
        gentle.setObjectName("smallMuted")
        gentle.setWordWrap(True)
        setup_l.addWidget(gentle)
        # §3.3: this is a concise selection summary, not a full-height panel.
        columns.addWidget(setup, 4, Qt.AlignmentFlag.AlignTop)
        practice, practice_l = _card("活动引导与反馈")
        practice_l.setSpacing(SPACE["lg"])
        self.activity_name = QLabel("短时休息")
        self.activity_name.setObjectName("activityName")
        activity_heading = QHBoxLayout()
        activity_heading.addWidget(self.activity_name)
        activity_heading.addStretch()
        self.countdown = QLabel("02:00")
        self.countdown.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.countdown.setObjectName("countdown")
        activity_heading.addWidget(self.countdown)
        practice_l.addLayout(activity_heading)
        self.activity_hint = QLabel(ACTIVITY_HINTS["break"])
        self.activity_hint.setWordWrap(True)
        self.activity_hint.setObjectName("bodyNote")
        self.activity_hint.setMaximumWidth(35 * TYPE["body"])
        practice_l.addWidget(self.activity_hint)
        self.activity_preview = QLabel()
        self.activity_preview.setObjectName("muted")
        self.activity_preview.setWordWrap(True)
        practice_l.addWidget(self.activity_preview)
        self.guide_widget = QFrame()
        self.guide_widget.setObjectName("infoStrip")
        guide_l = QVBoxLayout(self.guide_widget)
        guide_l.setContentsMargins(SPACE["lg"], SPACE["lg"], SPACE["lg"], SPACE["lg"])
        guide_l.setSpacing(SPACE["md"])
        self.guide_step_counter = QLabel("步骤 1")
        self.guide_step_counter.setObjectName("smallMuted")
        self.guide_step_title = QLabel()
        self.guide_step_title.setObjectName("cardTitle")
        self.guide_step_body = QLabel()
        self.guide_step_body.setObjectName("bodyNote")
        self.guide_step_body.setWordWrap(True)
        self.guide_step_body.setMaximumWidth(35 * TYPE["body"])
        guide_l.addWidget(self.guide_step_counter)
        guide_l.addWidget(self.guide_step_title)
        guide_l.addWidget(self.guide_step_body)
        self.next_guide_button = _button("下一步")
        self.next_guide_button.setObjectName("secondary")
        guide_l.addWidget(self.next_guide_button, 0, Qt.AlignmentFlag.AlignLeft)
        practice_l.addWidget(self.guide_widget)
        self.breathing = BreathingVisual()
        self.breathing.setMinimumHeight(128)
        self.breathing.hide()
        practice_l.addWidget(self.breathing)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setFormat("等待开始")
        self.progress.setFixedHeight(20)
        practice_l.addWidget(self.progress)
        self.process_widget = QWidget()
        process_l = QVBoxLayout(self.process_widget)
        process_l.setContentsMargins(0, 0, 0, 0)
        process_l.setSpacing(SPACE["sm"])
        checkin_label = QLabel("到这里，感觉怎样？")
        checkin_label.setObjectName("cardTitle")
        process_l.addWidget(checkin_label)
        checkin_buttons = QGridLayout()
        checkin_buttons.setSpacing(SPACE["sm"])
        self.process_comfortable = _button("目前合适")
        self.process_no_change = _button("暂时没变化")
        self.process_uncomfortable = _button("不舒服，停止")
        self.process_uncomfortable.setObjectName("danger")
        checkin_buttons.addWidget(self.process_comfortable, 0, 0)
        checkin_buttons.addWidget(self.process_no_change, 0, 1)
        process_l.addLayout(checkin_buttons)
        self.process_feedback_hint = QLabel("可跳过任何步骤。出现不适时会停止引导，并保存这条反馈。")
        self.process_feedback_hint.setObjectName("muted")
        self.process_feedback_hint.setWordWrap(True)
        process_l.addWidget(self.process_feedback_hint)
        self.switch_activity_button = _button("换一种方式")
        self.switch_activity_button.setObjectName("ghost")
        process_l.addWidget(self.switch_activity_button, 0, Qt.AlignmentFlag.AlignLeft)
        practice_l.addWidget(self.process_widget)
        self.open_retest_button = _button("结束练习，开始复测", True)
        # §2.8/§3.3: keep ending and stopping beside the concise summary.
        # These controls stay above the fold at the reviewed desktop sizes.
        setup_l.addWidget(self.open_retest_button)
        setup_l.addWidget(self.process_uncomfortable)
        self.back_to_guide_button = _button("返回练习")
        self.back_to_guide_button.setObjectName("ghost")
        practice_l.addWidget(self.back_to_guide_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.feedback_widget = QWidget()
        feedback_l = QVBoxLayout(self.feedback_widget)
        feedback_l.setContentsMargins(0, 5, 0, 0)
        feedback_l.setSpacing(SPACE["lg"])
        feedback_heading = QLabel("结束后，看看自己的感受")
        feedback_heading.setObjectName("cardTitle")
        feedback_l.addWidget(feedback_heading)
        self.assessment_hint = QLabel("活动结束后重新开始 30 秒辅助观察，可跳过；模型不替你填写自评。")
        self.assessment_hint.setObjectName("muted")
        self.assessment_hint.setWordWrap(True)
        feedback_l.addWidget(self.assessment_hint)
        self.skip_assessment_button = _button("跳过辅助观察")
        self.skip_assessment_button.setObjectName("ghost")
        feedback_l.addWidget(self.skip_assessment_button, 0, Qt.AlignmentFlag.AlignLeft)
        post_form = QFormLayout()
        post_form.setVerticalSpacing(12)
        self.post_combo = self._rating_combo("无有效复测 / 暂不填写")
        self.helpfulness_combo = QComboBox()
        self.helpfulness_combo.addItem("请选择主观帮助程度", "")
        for label, key in (("觉得有帮助", "helpful"), ("无明显帮助", "no_change"),
                           ("更不舒服", "worse"), ("暂时无法评价", "unknown")):
            self.helpfulness_combo.addItem(label, key)
        post_form.addRow("调节后自评", self.post_combo)
        post_form.addRow("主观反馈", self.helpfulness_combo)
        feedback_l.addLayout(post_form)
        self.next_action_edit = QLineEdit()
        self.next_action_edit.setPlaceholderText("结束后想尝试的一个小动作（可选）")
        self.feedback_note_edit = QLineEdit()
        self.feedback_note_edit.setPlaceholderText("哪些部分合适或不合适？（可选）")
        self.completion_preference = QComboBox()
        self.completion_preference.addItem("不改变现有偏好", "unchanged")
        self.completion_preference.addItem("下次优先考虑这种方法", "prefer")
        self.completion_preference.addItem("下次先不推荐这种方法", "avoid")
        self.completion_preference.addItem("恢复中性偏好", "neutral")
        feedback_l.addWidget(self.next_action_edit)
        feedback_l.addWidget(self.feedback_note_edit)
        feedback_l.addWidget(self.completion_preference)
        followup_row = QHBoxLayout()
        self.followup_optin = QCheckBox("稍后再复测一次（可选）")
        self.followup_delay_combo = QComboBox()
        for seconds in (120, 300, 600):
            self.followup_delay_combo.addItem(f"{seconds // 60} 分钟后", seconds)
        followup_row.addWidget(self.followup_optin, 1)
        followup_row.addWidget(self.followup_delay_combo)
        feedback_l.addLayout(followup_row)
        followup_note = QLabel("只在本应用中提示。若关闭，待复测记录保留，下次可继续填写。")
        followup_note.setObjectName("smallMuted")
        followup_note.setWordWrap(True)
        feedback_l.addWidget(followup_note)
        feedback_row = QHBoxLayout()
        self.complete_button = _button("结束并保存反馈", True)
        self.cancel_button = _button("中止活动")
        self.cancel_button.setObjectName("ghost")
        feedback_row.addWidget(self.complete_button, 2)
        feedback_row.addWidget(self.cancel_button, 1)
        feedback_l.addLayout(feedback_row)
        practice_l.addWidget(self.feedback_widget)
        self.activity_result = QLabel("活动后以自己的感受为准。未填写复测时，结果会记为无法评价。")
        self.activity_result.setWordWrap(True)
        self.activity_result.setObjectName("smallMuted")
        practice_l.addWidget(self.activity_result)
        self.outcome_card = OutcomeCard()
        practice_l.addWidget(self.outcome_card)
        self.retry_save_button = _button("重试保存反馈")
        self.retry_save_button.setObjectName("secondary")
        self.retry_save_button.hide()
        practice_l.addWidget(self.retry_save_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.view_feedback_button = _button("查看个人方法反馈")
        self.view_feedback_button.setObjectName("ghost")
        practice_l.addWidget(self.view_feedback_button, 0, Qt.AlignmentFlag.AlignLeft)
        practice_l.addStretch()
        columns.addWidget(practice, 6, Qt.AlignmentFlag.AlignTop)
        outer.addLayout(columns, 1)
        content.setMinimumHeight(550)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        self.regulation_tabs = QTabWidget()
        self.regulation_tabs.addTab(scroll, "本次调节")
        self.regulation_tabs.addTab(self._personal_feedback_page(), "个人方法反馈")
        self.regulation_tabs.addTab(self._followup_page(), "稍后复测")
        return self.regulation_tabs

    def _personal_feedback_page(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, SPACE["lg"], 0, 0)
        outer.setSpacing(SPACE["xl"])
        panel, body = _card("个人方法反馈", "仅比较当前档案、同一种会话来源。少量反馈用于参考，不称为最佳方法。")
        self.personal_feedback_scope = QLabel("开始一次会话或加载当前档案后，查看自己的方法记录。")
        self.personal_feedback_scope.setObjectName("muted")
        self.personal_feedback_scope.setWordWrap(True)
        body.addWidget(self.personal_feedback_scope)
        self.refresh_feedback_button = _button("加载 / 刷新当前档案")
        self.refresh_feedback_button.setObjectName("secondary")
        body.addWidget(self.refresh_feedback_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.personal_stats_label = QLabel("有效前后测 0 次 · 稍后复测 0 次 · 不适记录 0 次")
        self.personal_stats_label.setObjectName("cardTitle")
        self.personal_stats_label.setWordWrap(True)
        body.addWidget(self.personal_stats_label)
        self.feedback_table = QTableWidget(0, 6)
        self.feedback_table.setHorizontalHeaderLabels(["方法", "完成数", "独立活动样本", "综合自评变化", "有帮助反馈", "不适记录"])
        self.feedback_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.feedback_table.verticalHeader().hide()
        self.feedback_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.feedback_table.setMinimumHeight(280)
        self.feedback_table.setStyleSheet(f"font-size:{TYPE['meta']}px;")
        body.addWidget(self.feedback_table)
        outer.addWidget(panel)
        pref, pref_l = _card("由你决定下次怎样推荐")
        preference_note = QLabel("显式偏好用于当前档案。已有不适记录的方法，不会仅因设为优先而重新进入推荐。")
        preference_note.setObjectName("muted")
        preference_note.setWordWrap(True)
        pref_l.addWidget(preference_note)
        form = QFormLayout()
        self.pref_activity_combo = QComboBox()
        for key, name in ACTIVITIES.items():
            self.pref_activity_combo.addItem(name, key)
        self.pref_state_combo = QComboBox()
        for name, value in (("优先考虑", "prefer"), ("先不推荐", "avoid"), ("中性偏好", "neutral")):
            self.pref_state_combo.addItem(name, value)
        form.addRow("选择方法", self.pref_activity_combo)
        form.addRow("个人偏好", self.pref_state_combo)
        pref_l.addLayout(form)
        self.save_preference_button = _button("保存偏好", True)
        self.save_preference_button.setEnabled(False)
        pref_l.addWidget(self.save_preference_button, 0, Qt.AlignmentFlag.AlignLeft)
        outer.addWidget(pref)
        outer.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        return scroll

    def _followup_page(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, SPACE["lg"], 0, 0)
        outer.setSpacing(SPACE["xl"])
        panel, body = _card("稍后再看看", "在即时复测之外，记录一段时间后的感受。未填写不能当作持续改善。")
        self.followup_empty = QLabel("目前没有待复测记录。结束活动时，可自愿选择稍后再复测一次。")
        self.followup_empty.setObjectName("muted")
        self.followup_empty.setWordWrap(True)
        body.addWidget(self.followup_empty)
        self.refresh_followup_button = _button("加载 / 刷新当前档案")
        self.refresh_followup_button.setObjectName("secondary")
        body.addWidget(self.refresh_followup_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.followup_form = QWidget()
        followup_l = QVBoxLayout(self.followup_form)
        followup_l.setContentsMargins(0, SPACE["lg"], 0, 0)
        followup_l.setSpacing(SPACE["lg"])
        self.pending_followup_combo = QComboBox()
        self.followup_due_label = QLabel()
        self.followup_due_label.setObjectName("muted")
        self.followup_due_label.setWordWrap(True)
        self.followup_distress_combo = self._rating_combo("无有效复测 / 暂不填写")
        self.followup_helpfulness_combo = QComboBox()
        for title, key in (("暂时无法评价", "unknown"), ("仍觉得有帮助", "helpful"), ("无明显帮助", "no_change"), ("更不舒服", "worse")):
            self.followup_helpfulness_combo.addItem(title, key)
        form = QFormLayout()
        form.addRow("选择待复测", self.pending_followup_combo)
        form.addRow(self.followup_due_label)
        form.addRow("此刻困扰度", self.followup_distress_combo)
        form.addRow("此刻帮助程度", self.followup_helpfulness_combo)
        followup_l.addLayout(form)
        self.submit_followup_button = _button("保存这次稍后复测", True)
        followup_l.addWidget(self.submit_followup_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.followup_form.hide()
        body.addWidget(self.followup_form)
        outer.addWidget(panel)
        outer.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        return scroll

    @staticmethod
    def _rating_combo(prompt: str) -> QComboBox:
        combo = QComboBox()
        combo.addItem(prompt, None)
        for value in range(11):
            combo.addItem(str(value), value)
        return combo


    def _history_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE["xl"])
        self._all_history = []
        self.history_stats = QWidget()
        summary_row = QHBoxLayout(self.history_stats)
        summary_row.setContentsMargins(0, 0, 0, 0)
        summary_row.setSpacing(SPACE["xl"])
        self.history_totals = []
        for caption, initial in (("当前列表会话", "0"), ("已记录活动", "0"), ("有效联合观察", "0 秒")):
            card, card_l = _card(caption)
            value = QLabel(initial)
            value.setObjectName("statValue")
            card_l.addWidget(value)
            self.history_totals.append(value)
            summary_row.addWidget(card)
        layout.addWidget(self.history_stats)
        panel, panel_l = _card("会话记录")
        self.history_tools = QWidget()
        tools = QGridLayout(self.history_tools)
        tools.setContentsMargins(0, 0, 0, 0)
        tools.setHorizontalSpacing(SPACE["md"])
        tools.setVerticalSpacing(SPACE["md"])
        self.history_tools_layout = tools
        self._history_tools_compact = None
        self.history_search = QLineEdit()
        self.history_search.setPlaceholderText("搜索档案或日期…")
        self.history_search.setClearButtonEnabled(True)
        self.history_search.setMaximumWidth(250)
        self.history_filter = QComboBox()
        self.history_filter.addItem("全部来源", "all")
        self.history_filter.addItem("真实采集", "real")
        self.history_filter.addItem("模拟回放", "simulation")
        self.history_filter.addItem("来源未核实", "unknown")
        self.history_refresh = _button("刷新")
        self.history_refresh.setObjectName("ghost")
        self.history_export = _button("导出记录")
        self.history_export.setObjectName("secondary")
        self.history_delete = _button("删除")
        self.history_delete.setObjectName("danger")
        self._reflow_history_tools(self.width() < 1280)
        panel_l.addWidget(self.history_tools)
        self.history_table = QTableWidget(0, 7)
        self.history_table.setHorizontalHeaderLabels(["开始时间", "个人档案", "来源", "状态 / 时长", "观察数", "活动数", "最近反馈"])
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.history_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.history_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.history_table.verticalHeader().hide()
        self.history_table.verticalHeader().setDefaultSectionSize(46)
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setShowGrid(False)
        self.history_table.setStyleSheet(f"border:0; border-radius:0; font-size:{TYPE['meta']}px;")
        self.history_table.setMinimumHeight(120)
        panel_l.addWidget(self.history_table, 1)
        self.history_empty = QLabel("还没有会话记录\n\n完成一次观察与活动后，在这里回看自己的反馈。")
        self.history_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.history_empty.setObjectName("emptyBody")
        self.history_empty.setMinimumHeight(150)
        # §3.6 / §8.4: explanation and next action are one centered group.
        self.history_empty_group = QWidget()
        empty_l = QVBoxLayout(self.history_empty_group)
        empty_l.setContentsMargins(0, 0, 0, 0)
        empty_l.setSpacing(SPACE["lg"])
        empty_l.addStretch()
        self.history_empty.setMinimumHeight(0)
        empty_l.addWidget(self.history_empty)
        self.history_empty_start = _button("开始第一次会话", True)
        self.history_empty_start.clicked.connect(self._show_session_setup)
        empty_l.addWidget(self.history_empty_start, 0, Qt.AlignmentFlag.AlignHCenter)
        empty_l.addStretch()
        panel_l.addWidget(self.history_empty_group, 1)
        layout.addWidget(panel, 1)
        details, detail_l = _card("会话详情", "选中一条记录，查看观察覆盖与活动反馈。")
        self.history_detail_card = details
        self.history_details = QPlainTextEdit()
        self.history_details.setReadOnly(True)
        self.history_details.setPlaceholderText("选择一条会话查看详情。")
        self.history_details.setMinimumHeight(130)
        self.history_details.setMaximumHeight(180)
        self.history_details.setStyleSheet("border:0; background:#F8FAF8; padding:10px;")
        detail_l.addWidget(self.history_details)
        layout.addWidget(details)
        note = QLabel("模拟记录始终单独标识。原始视频与录音不长期保存，完整记录可导出为 JSON。")
        note.setObjectName("smallMuted")
        note.setWordWrap(True)
        layout.addWidget(note)
        page.setMinimumHeight(580)
        self.history_content = page
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.history_search.textChanged.connect(lambda _: self._on_history(self._all_history))
        self.history_filter.currentIndexChanged.connect(lambda _: self._on_history(self._all_history))
        return scroll

    def _reflow_history_tools(self, compact):
        if compact == self._history_tools_compact:
            return
        fields = (self.history_search, self.history_filter, self.history_refresh,
                  self.history_export, self.history_delete)
        for widget in fields:
            self.history_tools_layout.removeWidget(widget)
        for col in range(5):
            self.history_tools_layout.setColumnStretch(col, 0)
        if compact:
            self.history_tools_layout.addWidget(self.history_search, 0, 0, 1, 2)
            self.history_tools_layout.addWidget(self.history_filter, 0, 2)
            for col, widget in enumerate(fields[2:]):
                self.history_tools_layout.addWidget(widget, 1, col, Qt.AlignmentFlag.AlignRight)
        else:
            for col, widget in enumerate(fields):
                self.history_tools_layout.addWidget(widget, 0, col)
        self.history_tools_layout.setColumnStretch(0, 1)
        self._history_tools_compact = compact

    def _about_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE["xl"])
        intro = QLabel("观察线索，记录反馈，把决定留给自己。")
        intro.setObjectName("infoStrip")
        layout.addWidget(intro)
        grid = QGridLayout()
        grid.setSpacing(SPACE["xl"])
        sections = (
            ("01  这个工具能做什么", "在自愿使用的桌面学习或工作场景中，记录表情与语音线索，并提供可自主选择的短时活动。\n\n观察结果不等于你的真实感受，也不用于心理诊断。"),
            ("02  两种线索怎样共同判断", "表情与语音按照实际采集区间对齐，再依据输入质量融合。静音不会被补成中性。\n\n遇到缺失、冲突或不确定性高的情况，系统会显示原因并保留空白。"),
            ("03  数据与选择由你掌控", "开始前需同意本次采集。语音转写单独开启、默认关闭；原始视频和录音不长期保存。\n\n你可以随时暂停、结束或中止活动，也可导出和删除本地会话。"),
            ("04  真实采集与模拟回放", "真实模式只展示设备和本地模型产生的结果。模型未就绪或输入不足时，允许留空。\n\n模拟回放只用于体验操作，所有记录都有模拟标识，不作为实测结果。"),
            ("05  使用前的小提醒", "请在单人、本人同意的环境中使用，选择自己的个人档案。麦克风不能自动确认说话人。\n\n存在明显背景对话时，请关闭麦克风，避免记录混淆。"),
            ("06  怎样理解活动反馈", "活动中无需刻意微笑、讲话或跟随呼吸节奏。离开镜头、闭眼与静音，不代表状态改善。\n\n前后自评与模型线索分别记录；一次自评下降不能证明活动具有治疗效果。"),
        )
        for index, (title, body) in enumerate(sections):
            card, card_l = _card(title)
            text = QLabel(body)
            text.setWordWrap(True)
            text.setObjectName("muted")
            card_l.addWidget(text)
            card_l.addStretch()
            grid.addWidget(card, index // 2, index % 2)
        layout.addLayout(grid)
        layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        return scroll

    def _connect(self) -> None:
        for signal, slot in (
            ("frame_ready", self._on_frame), ("observation_ready", self._on_observation),
            ("fusion_ready", self._on_fusion), ("transcript_ready", self._on_transcript),
            ("status_changed", self._on_status), ("session_changed", self._on_session),
            ("devices_ready", self._on_devices), ("intervention_changed", self._on_intervention),
            ("history_ready", self._on_history),
        ):
            getattr(self.controller, signal).connect(slot)
        if hasattr(self.controller, "recommendations_ready"):
            self.controller.recommendations_ready.connect(self._on_recommendations)
        if hasattr(self.controller, "evidence_changed"):
            self.controller.evidence_changed.connect(self._on_evidence)
        self.revoke_transcript_button.clicked.connect(self._revoke_transcript)
        self.skip_assessment_button.clicked.connect(lambda: self.controller.skip_assessment())
        self.retry_save_button.clicked.connect(self.controller.retry_activity_save)
        self.refresh_button.clicked.connect(self.controller.refresh_devices)
        self.start_button.clicked.connect(self._start_session)
        self.pause_button.clicked.connect(self._pause_session)
        self.stop_button.clicked.connect(self.controller.stop_session)
        self.consent.toggled.connect(self._set_session_controls)
        self.transcript_consent.toggled.connect(self._set_session_controls)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        self.camera_combo.currentIndexChanged.connect(self._set_session_controls)
        self.mic_combo.currentIndexChanged.connect(self._set_session_controls)
        self.camera_combo.currentTextChanged.connect(self.camera_combo.setToolTip)
        self.mic_combo.currentTextChanged.connect(self.mic_combo.setToolTip)
        self.user_name.textChanged.connect(self._set_session_controls)
        self.request_button.clicked.connect(self._request_activity)
        self.correct_button.clicked.connect(self._correct_observation)
        self.activity_request_button.clicked.connect(self._request_activity)
        self.accept_button.clicked.connect(lambda: self.controller.confirm_intervention(True))
        self.reject_button.clicked.connect(lambda: self.controller.confirm_intervention(False))
        self.begin_activity_button.clicked.connect(self._begin_activity)
        self.complete_button.clicked.connect(self._complete_activity)
        self.cancel_button.clicked.connect(self.controller.cancel_activity)
        self.activity_combo.currentIndexChanged.connect(self._activity_choice_changed)
        self.duration_combo.currentIndexChanged.connect(self._duration_changed)
        self.history_refresh.clicked.connect(self.controller.refresh_history)
        self.history_export.clicked.connect(self._export_history)
        self.history_delete.clicked.connect(self._delete_history)
        self.history_table.itemSelectionChanged.connect(self._history_selected)
        self.tabs.currentChanged.connect(lambda index: self.controller.refresh_history() if index == 2 else None)
        self.regulation_tabs.currentChanged.connect(self._regulation_tab_changed)
        self.need_combo.currentIndexChanged.connect(self._context_changed)
        self.available_combo.currentIndexChanged.connect(self._context_changed)
        self.recommendation_cards.chosen.connect(self._choose_recommendation)
        self.activity_combo.activated.connect(lambda _: setattr(self, "_manual_activity_choice", True))
        self.next_guide_button.clicked.connect(self.controller.advance_activity_step)
        self.process_comfortable.clicked.connect(lambda: self.controller.activity_checkin("comfortable"))
        self.process_no_change.clicked.connect(lambda: self.controller.activity_checkin("no_change"))
        self.process_uncomfortable.clicked.connect(lambda: self.controller.activity_checkin("uncomfortable"))
        self.switch_activity_button.clicked.connect(self.controller.request_alternative)
        self.open_retest_button.clicked.connect(self._open_retest)
        self.back_to_guide_button.clicked.connect(self._return_to_guide)
        self.view_feedback_button.clicked.connect(lambda: self.regulation_tabs.setCurrentIndex(1))
        self.refresh_feedback_button.clicked.connect(self._refresh_recommendations)
        self.refresh_followup_button.clicked.connect(self._refresh_recommendations)
        self.save_preference_button.clicked.connect(self._save_activity_preference)
        self.pref_activity_combo.currentIndexChanged.connect(self._preference_activity_changed)
        self.pending_followup_combo.currentIndexChanged.connect(self._followup_selection_changed)
        self.submit_followup_button.clicked.connect(self._submit_followup)
        self.user_name.editingFinished.connect(self._refresh_recommendations)
        self.user_name.textChanged.connect(self._invalidate_personal_view)
        self.mode_combo.currentIndexChanged.connect(self._invalidate_personal_view)
        self.mode_combo.currentIndexChanged.connect(self._refresh_recommendations)

    def _set_session_controls(self, *_args) -> None:
        active = self.session_state in ("running", "paused", "starting", "stopping", "stop_failed")
        for widget in (self.user_name, self.mode_combo, self.camera_combo, self.mic_combo,
                       self.refresh_button, self.consent, self.transcript_consent):
            widget.setEnabled(not active)
        simulated = self.mode_combo.currentData() == "simulation"
        has_input = (self.camera_combo.currentData() not in (None, -1)
                     or self.mic_combo.currentData() is not None)
        self.start_button.setEnabled(not active and self.consent.isChecked() and (simulated or has_input)
                                    and self.activity_current.get("save_status") not in ("pending", "failed"))
        self.start_button.setVisible(not active)
        self.pause_button.setVisible(active)
        self.stop_button.setVisible(active)
        self.pause_button.setEnabled(self.session_state in ("running", "paused"))
        self.pause_button.setText("继续采集" if self.session_state == "paused" else "暂停")
        self.stop_button.setEnabled(self.session_state in ("running", "paused", "stop_failed"))
        self.revoke_transcript_button.setVisible(self.session_state in ("running", "paused") and self.transcript_consent.isChecked())
        if self.transcript_consent.isChecked():
            self.transcript.setPlaceholderText("转写已开启，等待本人的语音片段…" if active else "已选择转写，开始会话后处理语音片段。")
        else:
            self.transcript.setPlaceholderText("暂未开启转写\n开启后，识别文字会显示在这里。")
        self.correct_button.setEnabled(self.session_state == "running")
        if simulated:
            self.camera_combo.setEnabled(False)
            self.mic_combo.setEnabled(False)
        self.input_hint.setVisible(not active and not simulated and not has_input and self._has_devices)
        self.input_hint.setText("没有启用采集设备。连接摄像头或麦克风，也可切换“模拟回放”体验完整流程。")
        if not self.consent.isChecked():
            self.start_button.setToolTip("请先勾选同意本次本地采集与记录。")
        elif not simulated and not has_input:
            self.start_button.setToolTip("请选择摄像头或麦克风，或切换模拟回放。")
        else:
            self.start_button.setToolTip("开始本次会话")
        source = "模拟回放 · 非实测" if simulated else "真实设备采集"
        summary = f"{self.user_name.text().strip() or '个人档案'}  /  {source}"
        self.session_summary.setText(summary)
        self.session_summary.setToolTip(summary)
        self.transcript.setVisible(self.transcript_consent.isChecked())
        self.transcript_notice.setVisible(not self.transcript_consent.isChecked())
        self._sync_action_hierarchy()
        self._set_activity_controls()

    def _mode_changed(self, *_args) -> None:
        simulated = self.mode_combo.currentData() == "simulation"
        self.consent.setText("同意保存本次模拟记录" if simulated else "同意本次本地采集与记录")
        self.mode_notice.setText(
            "模拟回放仅使用预设内容体验流程，不连接真实设备。所有结果均标为模拟。"
            if simulated else "采集只在你点击开始后进行。麦克风默认关闭，原始视频与录音不长期保存。"
        )
        self.mode_notice.setStyleSheet(f"color:{COLORS['warning']};" if simulated else f"color:{COLORS['tertiary_text']};")
        self._set_session_controls()

    def _start_session(self) -> None:
        name = self.user_name.text().strip()
        if not name:
            self.statusBar().showMessage("请输入本地个人档案名称。")
            self.user_name.setFocus()
            return
        mode = self.mode_combo.currentData()
        self.chart.clear()
        self.transcript.clear()
        self._preview_image = None
        self.preview.setPixmap(QPixmap())
        self.preview.setText("模拟回放（非实测）\n不采集摄像头画面" if mode == "simulation" else "正在连接摄像头…")
        self.controller.start_session(
            name, self.camera_combo.currentData() if self.camera_combo.currentData() is not None else -1,
            self.mic_combo.currentData(),
            {"collection": self.consent.isChecked(), "transcript": self.transcript_consent.isChecked(), "raw_media": False},
            mode=mode,
        )

    def _pause_session(self) -> None:
        self.controller.pause_session(self.session_state != "paused")

    def _on_session(self, result: dict) -> None:
        previous_id = self.session_id
        self.session_state = str(result.get("state", self.session_state))
        self.session_id = result.get("session_id", self.session_id)
        self._session_mode = result.get("mode", self._session_mode)
        mode_index = self.mode_combo.findData(self._session_mode)
        if mode_index >= 0 and mode_index != self.mode_combo.currentIndex():
            self.mode_combo.setCurrentIndex(mode_index)
        if self.session_state in ("running", "paused", "stopping"):
            self.consent.setChecked(True)
        if result.get("user_name"):
            self.user_name.setText(str(result["user_name"]))
        names = {"idle": "未开始会话", "starting": "正在启动", "running": "正在观察", "paused": "已暂停采集", "stopping": "正在释放设备", "stopped": "会话已结束", "stop_failed": "结束未完全成功，请重试"}
        prefix = "模拟回放 · 非实测 / " if self._session_mode in ("simulation", "demo", "replay") else "真实采集 / "
        self.session_badge.setText((prefix if self.session_id else "") + names.get(self.session_state, self.session_state))
        self.session_badge.setStyleSheet(f"background:{COLORS['warning_surface']}; color:{COLORS['warning']};" if self._session_mode in ("simulation", "demo", "replay") else "")
        if self.session_state == "paused":
            self.preview_state.setText("采集已暂停，设备不再产生新的观察；可继续或结束会话。")
        elif self.session_state == "stopped":
            self._preview_image = None
            self.preview.clear()
            self.preview.setText("本次会话已结束\n设备已请求释放")
            self.preview_state.setText("会话记录可在“历史记录”查看、导出或删除。")
            self.controller.refresh_history()
        elif self.session_state == "running":
            self.preview_state.setText("模拟回放中，不使用真实设备。" if self._session_mode in ("simulation", "demo", "replay")
                                       else "采集中 · 原始视频和音频不长期保存")
        if self.session_id != previous_id:
            self._session_started = time.monotonic()
            self.session_clock.setText("本次会话  00:00")
            self.settings_toggle.setChecked(False)
            self._toggle_settings(False)
            self.chart.clear()
            self._fusion_display_end = -1
            self._fusion_display_priority = -1
            self._fusion_display_event = None
            self.transcript.clear()
            waiting = {"valid": False, "scores": None, "quality": None,
                       "reason": "等待本次会话的有效输入。", "model_version": "等待本次推理"}
            self.visual_card.update_result(waiting)
            self.audio_card.update_result(waiting)
            self.fusion_label.setText("—")
            self.fusion_mode.setText("尚无模态参与")
            self.fusion_interval.setText("尚无观察区间")
            self.fusion_weights.setText("视觉相对权重 —　·　语音相对权重 —　·　不确定 —")
            self.fusion_reason.setText("新会话尚无有效结果；表情与声音只提供可观察线索。")
            self.fusion_reason.setStyleSheet("")
            self.post_combo.setCurrentIndex(0)
            self.pre_combo.setCurrentIndex(0)
            self.helpfulness_combo.setCurrentIndex(0)
            self.activity_result.setText("活动后请以自己的感受为准。缺少复测时将记录为无法评价。")
        self._set_session_controls()

    def _on_devices(self, devices: dict) -> None:
        current_cam = self.camera_combo.currentData()
        current_mic = self.mic_combo.currentData()
        self.camera_combo.clear()
        self.camera_combo.addItem("不启用 / 未检测到摄像头", -1)
        for camera in devices.get("cameras", []):
            self.camera_combo.addItem(str(camera.get("name", f"摄像头 {camera.get('index')}")), camera.get("index"))
            self.camera_combo.setItemData(self.camera_combo.count() - 1, self.camera_combo.itemText(self.camera_combo.count() - 1), Qt.ItemDataRole.ToolTipRole)
        index = self.camera_combo.findData(current_cam)
        self.camera_combo.setCurrentIndex(index if index > 0 else (1 if self.camera_combo.count() > 1 else 0))
        self.mic_combo.clear()
        self.mic_combo.addItem("不启用麦克风", None)
        for mic in devices.get("microphones", []):
            self.mic_combo.addItem(str(mic.get("name", f"麦克风 {mic.get('index')}")), mic.get("index"))
            self.mic_combo.setItemData(self.mic_combo.count() - 1, self.mic_combo.itemText(self.mic_combo.count() - 1), Qt.ItemDataRole.ToolTipRole)
        index = self.mic_combo.findData(current_mic)
        self.mic_combo.setCurrentIndex(max(0, index))
        self._has_devices = True
        errors = devices.get("errors")
        if errors:
            self._on_status("设备检查：" + ("；".join(map(str, errors)) if isinstance(errors, list) else str(errors)))
        self._set_session_controls()

    def _on_frame(self, frame: Any) -> None:
        if self.session_state != "running" or frame is None:
            return
        try:
            if frame.ndim == 2:
                image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format.Format_Grayscale8).copy()
            elif frame.ndim == 3 and frame.shape[2] == 3:
                image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format.Format_BGR888).copy()
            elif frame.ndim == 3 and frame.shape[2] == 4:
                image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format.Format_ARGB32).copy()
            else:
                return
            self._preview_image = image
            self._resize_preview()
        except (AttributeError, ValueError, TypeError, BufferError):
            self.preview_state.setText("收到无法显示的画面，等待下一帧。")

    def _resize_preview(self) -> None:
        if self._preview_image is not None:
            self.preview.setPixmap(QPixmap.fromImage(self._preview_image).scaled(
                self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._resize_preview()
        if hasattr(self, "content_frame") and hasattr(self, "controls_grid"):
            self._reflow_controls(2 if self.width() - 252 < 950 else 4)
        if hasattr(self, "history_tools_layout"):
            self._reflow_history_tools(self.width() < 1280)
        if hasattr(self, "live_action_row"):
            self.live_action_row.setDirection(QBoxLayout.Direction.TopToBottom if self.width() < 1280 else QBoxLayout.Direction.LeftToRight)

    def _on_observation(self, result: dict) -> None:
        modality = str(result.get("modality", ""))
        if modality in ("visual", "vision", "video", "face"):
            self.visual_card.update_result(result)
        elif modality in ("audio", "speech", "acoustic"):
            self.audio_card.update_result(result)

    def _on_fusion(self, result: dict) -> None:
        self.chart.add_observation(result)
        end = _number(result.get("t_end"), -1)
        if end < getattr(self, "_fusion_display_end", -1):
            return  # Late revisions update historical points without replacing now.
        priority = {"joint": 2, "audio_only": 1, "visual_only": 1, "unavailable": 0}.get(result.get("status"), 0)
        if (end == getattr(self, "_fusion_display_end", -1)
                and result.get("event_id") != getattr(self, "_fusion_display_event", None)
                and priority < getattr(self, "_fusion_display_priority", -1)):
            return
        self._fusion_display_end = end
        self._fusion_display_priority = priority
        self._fusion_display_event = result.get("event_id")
        scores = result.get("scores") or result.get("fused_scores") or {}
        negative = _number(result.get("negative_score"))
        if negative is None:
            negative = _number(scores.get("negative"))
            if negative is not None:
                negative *= 100
        elif negative <= 1:
            negative *= 100
        uncertain = _number(scores.get("uncertain", result.get("uncertainty")))
        valid = result.get("valid", bool(scores)) and result.get("display_score", True)
        if uncertain is not None and uncertain >= 0.5:
            valid = False
        self.fusion_label.setText(f"{negative:.0f}" if valid and negative is not None else "—")
        self.fusion_label.setToolTip("负向线索分数来自当前有效模型输出，不是心理健康评分。" if valid else "证据不足，暂不显示分数。")
        mode = str(result.get("mode", result.get("status", result.get("source", "joint"))))
        modes = {"visual_only": "仅视觉观察", "visual": "仅视觉观察", "audio_only": "仅语音观察", "audio": "仅语音观察",
                 "joint": "音视频联合观察", "fusion": "音视频联合观察", "unavailable": "两模态均不可用", "none": "两模态均不可用"}
        self.fusion_mode.setText(modes.get(mode, mode))
        start, end = _number(result.get("t_start")), _number(result.get("t_end"))
        self.fusion_interval.setText(f"对应区间 {_session_time(start)} – {_session_time(end)}" if start is not None and end is not None else "尚无观察区间")
        reason = result.get("reason", result.get("reasons", ""))
        conflict = result.get("conflict", result.get("is_conflict", False))
        self.fusion_reason.setText(("模态线索不一致，请以自己的当前感受为准。 " if conflict else "") + _reason(reason or "按对应时间区间和有效输入质量进行融合。"))
        self.fusion_reason.setStyleSheet(f"color:{COLORS['warning']};" if conflict else "")
        self.fusion_mode.setStyleSheet(f"background:{COLORS['warning_surface']}; color:{COLORS['warning']};" if conflict or not valid else "")
        weights = result.get("weights") or {}
        visual_weight = weights.get("visual", weights.get("video", result.get("visual_weight")))
        audio_weight = weights.get("audio", result.get("audio_weight"))
        self.fusion_weights.setText(f"视觉权重 {_percent(visual_weight)}　·　语音权重 {_percent(audio_weight)}　·　不确定 {_percent(uncertain)}")
        divergence = _number(result.get("disagreement_js", (result.get("conflict_diagnostics") or {}).get("jensen_shannon")))
        versions = result.get("model_versions") or {}
        self.fusion_reason.setToolTip(f"模态分布差异：{divergence if divergence is not None else '不可比'}\n参与模型：{json.dumps(versions, ensure_ascii=False)}\n融合配置：{result.get('fusion_config_hash', '未记录')}")

    def _on_evidence(self, result):
        seconds = _number(result.get("effective_seconds", result.get("negative_seconds")), 0)
        density = result.get("coverage_density", result.get("density"))
        self.evidence_progress.setText(f"有效负向证据 {seconds:.1f} / 20 秒 · 覆盖密度 {_percent(density)} · 等待迟到结果 3 秒")
        self.evidence_progress.setToolTip("按有效区间并集计时，重复和空白不计；中性、冲突和无效证据不能当作连续负向证据。")

    def _revoke_transcript(self):
        self.controller.set_transcript_allowed(False)
        self.transcript_consent.setChecked(False)
        self._set_session_controls()
        self.transcript_notice.setText("本次转写已停止；之前获授权的历史保留，后续转写任务和旧结果已拦截。")
        self.transcript_notice.show()

    def _on_transcript(self, result: dict) -> None:
        text = str(result.get("text", result.get("transcript", ""))).strip()
        prefix = "[模拟] " if self._session_mode in ("simulation", "demo", "replay") else ""
        stamp = result.get("t_start")
        if isinstance(stamp, (int, float)) and stamp < 1_000_000_000:
            prefix += f"{stamp:.1f}s  "
        if text:
            self.transcript.appendPlainText(prefix + text)
        else:
            reason = result.get("reason", result.get("status", "未生成转写"))
            self.transcript.appendPlainText(prefix + str(reason))

    def _on_status(self, status: str) -> None:
        self.statusBar().showMessage(str(status))

    def _request_activity(self) -> None:
        self.tabs.setCurrentIndex(1)
        self.controller.request_intervention()

    def _correct_observation(self) -> None:
        labels = {"正向线索": "positive", "中性线索": "neutral", "负向线索": "negative", "不确定 / 不符合当前感受": "uncertain"}
        choice, accepted = QInputDialog.getItem(self, "修正当前观察", "请选择更符合你当前情况的线索。\n这条反馈与模型输出分别保存。", list(labels), 3, False)
        if accepted:
            self.controller.correct_observation(labels[choice])

    def _set_activity_controls(self) -> None:
        if not hasattr(self, "activity_request_button"):
            return
        active_session = self.session_state in ("running", "paused")
        state = self.activity_state
        observing = state in ("observing", "idle", "cooldown", "completed", "cancelled")
        confirming = state in ("confirming", "pending", "requested")
        ready = state in ("ready", "confirmed")
        running = state in ("running", "active")
        guiding = running and not self._retest_open
        retesting = running and self._retest_open
        current = self.activity_current
        assessment = getattr(self, "_latest_assessment", None) or {}
        phase = assessment.get("phase")
        status = assessment.get("status")
        remaining_window = max(0, math.ceil(_number(assessment.get("remaining_seconds"), 0)))
        coverage = _percent(assessment.get("coverage"))
        status_text = {"collecting": f"还需约 {remaining_window} 秒", "complete": "观察窗已完成",
                       "insufficient": "有效证据不足", "skipped": "已跳过"}.get(status, "尚未开始")
        if phase == "pre":
            self.pre_assessment_hint.setText(f"辅助前测：{status_text} · 有效覆盖 {coverage}。\n可直接开始练习；不足时不进行模型前后比较。")
        if phase == "post":
            self.assessment_hint.setText(f"独立后测：{status_text} · 有效覆盖 {coverage}。\n现在保存会结束辅助观察；用户自评与主观反馈仍独立记录。")
        elif self._retest_open:
            self.assessment_hint.setText("未采集独立辅助后测（例如采集已暂停）。仍可保存自己的感受，模型比较会保留不足。")
        self.skip_assessment_button.setVisible(self._retest_open and phase == "post" and status == "collecting")
        finished = current.get("status") in ("completed", "cancelled")
        saved_or_empty = current.get("save_status") not in ("pending", "failed")
        self.request_button.setEnabled(active_session and observing and saved_or_empty)
        self.activity_request_button.setEnabled(active_session and observing and saved_or_empty)
        self.activity_request_button.setText("再选择一种方式" if finished else "我想调整一下")
        self.accept_button.setEnabled(active_session and confirming)
        self.reject_button.setEnabled(active_session and confirming)
        self.setup_widget.setEnabled(active_session and ready)
        self.begin_activity_button.setEnabled(active_session and ready)
        self.post_combo.setEnabled(retesting)
        self.helpfulness_combo.setEnabled(retesting)
        self.complete_button.setEnabled(retesting)
        self.cancel_button.setEnabled(confirming or ready or running)
        macro_state = "cooldown" if retesting else state
        if state == "cooldown" and (not current or current.get("status") == "cancelled"):
            macro_state = "observing"
        self.activity_flow.set_state(macro_state)
        self.activity_request_button.setVisible(observing)
        self.confirm_widget.setVisible(confirming)
        self.setup_widget.setVisible(ready)
        self.begin_activity_button.setVisible(ready)
        self.scale_note.setVisible(ready)
        self.activity_options_hint.setVisible(observing or confirming)
        self.context_widget.setVisible(not running)
        self.context_widget.setEnabled(not running)
        self.recommend_toggle.setVisible(not running)
        self.recommendation_cards.setVisible(not running and self.recommend_toggle.isChecked())
        self.recommendation_reason.setVisible(not running)
        self.activity_summary.setVisible(running or finished)
        if current:
            name = ACTIVITIES.get(current.get("activity"), "活动确认")
            need = NEEDS.get((current.get("context") or {}).get("need"), "按当前需要")
            self.activity_summary.setText(f"本次选择  {name}\n{need}\n\n调节前自评  {_text(current.get('pre'))} / 10\n计划时长  {int(current.get('duration', 0)) // 60} 分钟")
        self.feedback_widget.setVisible(retesting)
        self.guide_widget.setVisible(guiding)
        self.process_widget.setVisible(guiding)
        self.process_uncomfortable.setVisible(guiding)
        self.open_retest_button.setVisible(guiding)
        self.open_retest_button.setEnabled(guiding)
        self.back_to_guide_button.setVisible(retesting)
        self.activity_preview.setVisible(not running and not finished)
        self.activity_hint.setVisible(not running and not finished)
        self.progress.setVisible(guiding)
        self.view_feedback_button.setVisible(finished)
        self.activity_name.setVisible(not finished)
        self.countdown.setVisible(not finished)
        if self._retest_open and phase == "post":
            self.countdown.setText(f"{remaining_window // 60:02d}:{remaining_window % 60:02d}" if status == "collecting" else status_text)
            self.countdown.setToolTip("独立辅助后测的剩余观察时间，与练习计划时长分别计时。")
        else:
            self.countdown.setToolTip("本次练习的计划剩余时长。")
        self.breathing.setVisible(guiding and current.get("activity") == "breathing")
        self.breathing.set_running(guiding and current.get("activity") == "breathing")
        self.outcome_card.set_record(current if finished else None, self._session_mode)
        self.retry_save_button.setVisible(current.get("save_status") == "failed")
        self.pre_assessment_hint.setVisible(ready)
        self._sync_action_hierarchy()

    def _activity_choice_changed(self, *_args) -> None:
        # Treat every explicit selector change as a choice, while shielding the
        # system's own default selection. Late recommendations cannot replace it.
        if _args and not self._applying_recommendation_selection:
            self._manual_activity_choice = True
        key = self.activity_combo.currentData()
        if key not in ACTIVITY_LIBRARY:
            return
        self.activity_name.setText(ACTIVITIES[key])
        self.activity_hint.setText(ACTIVITY_HINTS[key])
        self.task_form.setVisible(key == "task_split")
        steps = ACTIVITY_LIBRARY[key].get("steps", [])
        self.activity_preview.setText("练习路径\n" + "\n".join(f"{index+1}. {step['title']}" for index, step in enumerate(steps)))
        selected = next((item for item in self._recommendation_items if item["activity"] == key), None)
        if selected:
            self.recommendation_reason.setText("；".join(selected.get("reasons", [])[:3]))
        self.breathing.setVisible(self.activity_state == "running" and key == "breathing" and not self._retest_open)

    def _open_retest(self):
        if self.activity_state != "running":
            return
        self._retest_open = True
        self.controller.begin_post_assessment()
        self.activity_status.setText("请记录此刻的感受。复测可以缺失，但不能由模型替你填写。")
        self._set_activity_controls()

    def _return_to_guide(self):
        self._retest_open = False
        self.controller.resume_activity_guidance()
        self._set_activity_controls()

    def _context_changed(self, *_args):
        self._manual_activity_choice = False
        seconds = self.available_combo.currentData()
        index = self.duration_combo.findData(seconds)
        if index >= 0 and self.activity_state != "running":
            self.duration_combo.setCurrentIndex(index)
        self._refresh_recommendations()

    def _refresh_recommendations(self, *_args):
        if not hasattr(self.controller, "refresh_recommendations"):
            return
        self.controller.refresh_recommendations(need=self.need_combo.currentData(),
            available_seconds=self.available_combo.currentData(), user_name=self.user_name.text().strip(),
            mode=self.mode_combo.currentData())

    def _regulation_tab_changed(self, index):
        self._sync_action_hierarchy()
        if index:
            self._refresh_recommendations()

    def _invalidate_personal_view(self, *_args):
        if self.session_state in ("running", "paused", "stopping"):
            return
        if (self._recommendation_user_name != self.user_name.text().strip()
                or self._recommendation_mode != self.mode_combo.currentData()):
            self._recommendation_profile_id = None
            self._recommendation_items = []
            self._activity_preferences = {}
            self._pending_followups = []
            self.recommendation_cards.set_items([])
            self.save_preference_button.setEnabled(False)
            self.followup_form.hide()
            self.pending_followup_combo.clear()
            self.feedback_table.setRowCount(0)
            self.personal_stats_label.setText("尚未加载当前档案与来源的反馈")
            self.personal_feedback_scope.setText("档案已改变，加载当前档案后再显示个人反馈。")

    def _on_recommendations(self, payload):
        if payload.get("user_name") != self.user_name.text().strip() or payload.get("mode") != self.mode_combo.currentData():
            return
        self._recommendation_user_name = payload.get("user_name")
        self._recommendation_mode = payload.get("mode")
        self._recommendation_profile_id = payload.get("user_id")
        self._activity_preferences = payload.get("preferences") or {}
        self._recommendation_items = list(payload.get("items") or [])
        selected = self.activity_combo.currentData()
        blocked = {item["activity"] for item in self._recommendation_items if item.get("blocked")}
        for index in range(self.activity_combo.count()):
            item = self.activity_combo.model().item(index)
            if item:
                item.setEnabled(self.activity_combo.itemData(index) not in blocked)
        if self.activity_state != "running" and (not self._manual_activity_choice or selected in blocked):
            candidate = next((item["activity"] for item in self._recommendation_items if not item.get("blocked")), None)
            if candidate:
                self._applying_recommendation_selection = True
                try:
                    self.activity_combo.setCurrentIndex(self.activity_combo.findData(candidate))
                finally:
                    self._applying_recommendation_selection = False
        self.recommendation_cards.set_items(self._recommendation_items, self.activity_combo.currentData())
        self._activity_choice_changed()
        stats = payload.get("stats") or {}
        source = "模拟反馈单独统计" if payload.get("mode") == "simulation" else "仅使用该档案的真实会话反馈"
        self.personal_feedback_scope.setText(f"{payload.get('user_name') or '当前档案'} · {source}\n有帮助与不适分别保留，不用平均分覆盖你的主观反馈。")
        self.personal_stats_label.setText(f"独立活动样本 {stats.get('independent_evidence_count', 0)} 次 · 即时前后测 {stats.get('evidence_count', 0)} 次 · 稍后复测 {stats.get('followup_count', 0)} 次 · 不适 {stats.get('discomfort_count', 0)} 次")
        rows = list(stats.get("per_activity") or [])
        self.feedback_table.setRowCount(len(rows))
        for row, item in enumerate(rows):
            count = int(item.get("independent_evidence_count", item.get("evidence_count", 0)) or 0)
            delta = item.get("effective_mean_delta", item.get("mean_delta"))
            subjective_count = item.get("subjective_count", 0)
            helpful_count = item.get("helpful_count", 0)
            delta_text = "反馈不足" if count < 3 else (f"{delta:+.1f}" if delta is not None else "—")
            values = [item.get("title", ACTIVITIES.get(item.get("activity"), "方法")), item.get("completed_count", 0),
                      count, delta_text, f"{helpful_count} / {subjective_count} 次" if subjective_count else "—", item.get("discomfort_count", 0)]
            for col, value in enumerate(values):
                cell = QTableWidgetItem(str(value))
                cell.setToolTip(f"独立活动 {count} 次；先配对即时与延迟，再跨活动平均。换方法链排除 {item.get('chain_excluded_count', 0)} 次。综合自评变化 {delta if delta is not None else '缺失'}，不代表因果效果。")
                if col:
                    cell.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.feedback_table.setItem(row, col, cell)
        self.save_preference_button.setEnabled(bool(self._recommendation_profile_id))
        self._preference_activity_changed()
        self._pending_followups = list(payload.get("pending_followups") or [])
        previous = (self.pending_followup_combo.currentData() or {}).get("id")
        self.pending_followup_combo.blockSignals(True)
        self.pending_followup_combo.clear()
        for record in self._pending_followups:
            title = ACTIVITIES.get(record.get("activity"), "调节活动")
            due = (record.get("followup") or {}).get("due_at")
            self.pending_followup_combo.addItem(f"{title} · {_time_label(due)}", record)
        for index, record in enumerate(self._pending_followups):
            if record.get("id") == previous:
                self.pending_followup_combo.setCurrentIndex(index)
        self.pending_followup_combo.blockSignals(False)
        self.followup_form.setVisible(bool(self._pending_followups))
        self.followup_empty.setText(f"当前有 {len(self._pending_followups)} 条待复测。可按计划时间填写，也可主动提前记录。" if self._pending_followups else "目前没有待复测记录。结束活动时，可自愿选择稍后再复测一次。")
        self.regulation_tabs.setTabText(2, f"稍后复测 ({len(self._pending_followups)})" if self._pending_followups else "稍后复测")
        self._followup_selection_changed()

    def _choose_recommendation(self, activity):
        if self.activity_state == "running":
            return
        self._manual_activity_choice = True
        self.activity_combo.setCurrentIndex(self.activity_combo.findData(activity))
        self.recommendation_cards.set_items(self._recommendation_items, activity)

    def _save_activity_preference(self):
        self.controller.set_activity_preference(self.pref_activity_combo.currentData(), self.pref_state_combo.currentData())

    def _preference_activity_changed(self, *_args):
        value = self._activity_preferences.get(self.pref_activity_combo.currentData(), "neutral")
        index = self.pref_state_combo.findData(value)
        self.pref_state_combo.setCurrentIndex(index if index >= 0 else 2)

    def _followup_selection_changed(self, *_args):
        record = self.pending_followup_combo.currentData() or {}
        followup = record.get("followup") or {}
        due = followup.get("due_at")
        if not due:
            self.followup_due_label.setText("")
            return
        due_hint = "已到计划复测时间" if record.get("is_due", record.get("due")) else "尚未到计划时间，也可主动提前记录"
        self.followup_due_label.setText(f"计划复测时间：{_time_label(due)}（北京时间）\n{due_hint}。提前填写会保留实际时间；未填写不记为持续改善。")

    def _submit_followup(self):
        record = self.pending_followup_combo.currentData()
        if record:
            self.controller.submit_followup(record["session_id"], record["id"], self.followup_distress_combo.currentData(), self.followup_helpfulness_combo.currentData())
            self.followup_distress_combo.setCurrentIndex(0)

    def _duration_changed(self, *_args) -> None:
        if self.activity_state not in ("running", "active"):
            seconds = self.duration_combo.currentData()
            self.countdown.setText(f"{seconds // 60:02d}:{seconds % 60:02d}")

    def _begin_activity(self) -> None:
        pre = self.pre_combo.currentData()
        if pre is None:
            self._on_status("开始活动前，请选择此刻的困扰度（0–10）。")
            self.pre_combo.setFocus()
            self.pre_combo.showPopup()
            return
        activity = self.activity_combo.currentData()
        duration = self.duration_combo.currentData()
        task = {"blocker": self.task_blocker.text().strip(), "tried": self.task_tried.text().strip(), "next_step": self.task_next.text().strip()}
        method = self.controller.start_activity
        self._retest_open = False
        kwargs = {"task_details": task if activity == "task_split" else None}
        if "context" in inspect.signature(method).parameters:
            kwargs["context"] = {"need": self.need_combo.currentData(), "available_seconds": self.available_combo.currentData()}
        method(activity, pre, duration, **kwargs)

    def _complete_activity(self) -> None:
        helpfulness = self.helpfulness_combo.currentData()
        if not helpfulness:
            self._on_status("请先选择主观帮助程度；不确定时可以选择“暂时无法评价”。")
            self.helpfulness_combo.setFocus()
            self.helpfulness_combo.showPopup()
            return
        self.controller.complete_activity(self.post_combo.currentData(), helpfulness,
            followup_seconds=self.followup_delay_combo.currentData() if self.followup_optin.isChecked() else None,
            next_step=self.next_action_edit.text().strip(), preference=self.completion_preference.currentData(),
            feedback_note=self.feedback_note_edit.text().strip())

    def _on_intervention(self, result: dict) -> None:
        self._latest_assessment = result.get("assessment")
        old_state = self.activity_state
        self.activity_state = str(result.get("state", "observing"))
        self.activity_current = result.get("current") or {}
        if self.activity_state != "running" or old_state != "running":
            self._retest_open = False
        state = self.activity_state
        labels = {
            "observing": "目前没有活动。你可以随时主动请求，不需要模型批准。",
            "confirming": "是否需要一次短时调整？是否接受、选择什么活动，都由你决定。",
            "ready": "已确认需求。请填写调节前自评，再选择活动与计划时长。",
            "running": "活动进行中。可以提前结束并留下反馈，或随时中止。",
            "cooldown": "本次活动或确认已结束。暂时不重复主动打扰，你仍可随时主动请求。",
        }
        self.activity_status.setText(str(result.get("message") or labels.get(state, "当前流程：" + state)))
        if state in ("confirming", "pending", "requested") and old_state not in ("confirming", "pending", "requested"):
            self.tabs.setCurrentIndex(1)
            self.regulation_tabs.setCurrentIndex(0)
            self._manual_activity_choice = False
            self.followup_optin.setChecked(False)
            self.completion_preference.setCurrentIndex(0)
            self.next_action_edit.clear()
            self.feedback_note_edit.clear()
            self._refresh_recommendations()
            self.pre_combo.setCurrentIndex(0)
            self.post_combo.setCurrentIndex(0)
            self.helpfulness_combo.setCurrentIndex(0)
            self.activity_result.setText("活动后请以自己的感受为准。缺少复测时将记录为无法评价。")
        current = self.activity_current
        feedback_role = "savedFeedback" if current.get("status") == "completed" and current.get("save_status", "saved") == "saved" else "smallMuted"
        if self.activity_result.objectName() != feedback_role:
            self.activity_result.setObjectName(feedback_role)
            self.activity_result.style().unpolish(self.activity_result)
            self.activity_result.style().polish(self.activity_result)
        if current.get("activity") in ACTIVITIES:
            activity = current["activity"]
            self.activity_name.setText(ACTIVITIES[activity])
            self.activity_hint.setText(ACTIVITY_HINTS[activity])
            self.breathing.setVisible(activity == "breathing")
        steps = current.get("steps") or ACTIVITY_LIBRARY.get(current.get("activity"), {}).get("steps", [])
        index = min(max(0, int(current.get("step_index", 0))), max(0, len(steps)-1))
        if steps:
            self.guide_step_counter.setText(f"步骤 {index+1} / {len(steps)} · 按自己的节奏")
            self.guide_step_title.setText(steps[index]["title"])
            self.guide_step_body.setText(steps[index]["body"])
        self.next_guide_button.setEnabled(bool(steps) and index < len(steps)-1)
        self.next_guide_button.setText("下一步" if index < len(steps)-1 else "已到最后一步，可随时复测")
        checkins = current.get("checkins") or []
        last = checkins[-1].get("feeling") if checkins else None
        hints = {"comfortable": "已记录：目前合适。你可以继续，也可以按自己的节奏结束。",
                 "no_change": "已记录：暂时没变化。可继续当前步骤，或主动换一种方式。",
                 "uncomfortable": "已记录不适，指导已停止。这种方法不会继续被优先推荐。"}
        self.process_feedback_hint.setText(hints.get(last, "可跳过任何步骤。出现不适时会停止引导，并保存这条反馈。"))
        remaining = _number(result.get("remaining_seconds", current.get("remaining_seconds")))
        total = _number(current.get("duration"), self.duration_combo.currentData()) or 1
        if remaining is not None:
            seconds = max(0, math.ceil(remaining))
            self.countdown.setText(f"{seconds // 60:02d}:{seconds % 60:02d}")
            self.progress.setValue(int(max(0, min(100, (1 - remaining / total) * 100))))
            self.progress.setFormat("计划时长已到，请按当前感受复测" if remaining <= 0 else "活动进行中 %p%")
            if remaining <= 0 and state == "running":
                self.activity_flow.set_state("cooldown")
        if state in ("running", "active") and old_state not in ("running", "active"):
            self.progress.setValue(0)
            self.progress.setFormat("活动进行中 %p%")
            self.countdown.setText(f"{int(total) // 60:02d}:{int(total) % 60:02d}")
        outcome = current.get("outcome", current.get("evaluation", result.get("outcome")))
        if outcome:
            readable = {"improved": "用户自评困扰下降", "unchanged": "用户自评无变化", "worse": "用户自评困扰上升", "discomfort": "用户报告不适", "unavailable": "缺少有效复测，无法评价", "insufficient": "缺少有效复测，无法评价"}
            if current.get("status") != "running":
                saved = {"pending": "正在写入：", "failed": "尚未保存："}.get(current.get("save_status"), "已保存：")
                self.activity_result.setText(saved + readable.get(str(outcome), str(outcome)))
        if current.get("status") in ("completed", "cancelled"):
            self.controller.refresh_history()
        if current.get("status") == "completed":
            self.countdown.setText("已保存")
        elif current.get("status") == "cancelled":
            self.countdown.setText("已停止")
        if self.session_state == "stopped":
            self._set_session_controls()
        else:
            self._set_activity_controls()

    def _on_history(self, sessions: list) -> None:
        selected = self._selected_session()
        selected_id = (selected or {}).get("id", (selected or {}).get("session_id"))
        self._all_history = list(sessions or [])
        # §8.4: a genuinely empty view has an explanation and next action.
        # Filtering to zero keeps the tools needed to clear that filter.
        self.history_tools.setVisible(bool(self._all_history))
        self.history_stats.setVisible(bool(self._all_history))
        self.history_empty_start.setVisible(not self._all_history)
        self.history_content.setMinimumHeight(580 if self._all_history else 0)
        source = self.history_filter.currentData()
        query = self.history_search.text().strip().casefold()
        def source_kind(item):
            mode = item.get("mode")
            return "simulation" if mode in ("simulation", "demo", "replay") else "real" if mode == "real" else "unknown"
        self._history = [item for item in self._all_history
                         if (source == "all" or source == source_kind(item))
                         and (not query or query in f"{item.get('user_name', '')} {item.get('started_at', '')}".casefold())]
        self.history_totals[0].setText(str(len(self._history)))
        self.history_totals[1].setText(str(sum(item.get("intervention_count", 0) for item in self._history)))
        seconds = sum((item.get("coverage") or {}).get("joint_seconds", 0) for item in self._history)
        self.history_totals[2].setText(f"{seconds / 60:.1f} 分钟" if seconds >= 60 else f"{seconds:.0f} 秒")
        self.history_table.setVisible(bool(self._history))
        self.history_empty.setVisible(not self._history)
        self.history_empty_group.setVisible(not self._history)
        self.history_empty.setText("没有匹配的记录\n\n尝试更换来源或搜索内容。" if (source != "all" or query) else "还没有会话记录\n\n完成一次观察与活动后，在这里回看自己的反馈。")
        profile_names = list(dict.fromkeys(str(item["user_name"]) for item in self._all_history if item.get("user_name")))
        completer = QCompleter(profile_names, self.user_name)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.user_name.setCompleter(completer)
        self.history_table.setRowCount(len(self._history))
        select_row = None
        for row, item in enumerate(self._history):
            session_id = item.get("id", item.get("session_id"))
            mode = item.get("mode")
            simulated = mode in ("simulation", "demo", "replay")
            status = item.get("state", item.get("status", "stopped" if item.get("ended_at") else "running"))
            status = {"running": "进行中", "paused": "已暂停", "stopped": "已结束", "completed": "已结束"}.get(status, status)
            duration = _number(item.get("duration", item.get("duration_seconds")))
            if duration is not None:
                status += f" · {int(duration) // 60}分{int(duration) % 60:02d}秒"
            evaluation = {"improved": "自评下降", "unchanged": "自评无变化", "worse": "自评上升", "insufficient": "无有效复测"}.get(item.get("last_evaluation"), "—")
            delta = _number(item.get("last_delta"))
            if delta is not None and delta != 0:
                evaluation += f" {abs(delta):g}"
            # §2.1: explicit discomfort takes precedence over a numeric decline.
            if item.get("latest_loop_state") == "discomfort" or item.get("latest_discomfort"):
                evaluation = "报告不适"
            elif item.get("latest_followup_status") == "completed":
                evaluation = {"improved": "稍后自评下降", "unchanged": "稍后自评无变化",
                              "worse": "稍后自评上升", "insufficient": "稍后复测不足"}.get(item.get("latest_loop_state"), "已稍后复测")
            values = [
                _time_label(item.get("started_at", item.get("start_time", item.get("created_at")))),
                _text(item.get("user_name", item.get("name", item.get("user_id")))),
                "模拟（非实测）" if simulated else "真实采集" if mode == "real" else "来源未核实", status or "已记录",
                _text(item.get("observation_count", item.get("observations_count"))),
                _text(item.get("intervention_count", item.get("interventions_count"))),
                evaluation,
            ]
            for col, value in enumerate(values):
                cell = QTableWidgetItem(str(value))
                cell.setToolTip(str(value))
                if col == 6 and item.get("latest_loop_caption"):
                    cell.setToolTip(str(item["latest_loop_caption"]))
                # §4.7.2: counts align by their right edge for row scanning.
                cell.setTextAlignment((Qt.AlignmentFlag.AlignRight if col in (4, 5)
                                       else Qt.AlignmentFlag.AlignLeft) | Qt.AlignmentFlag.AlignVCenter)
                if col == 0:
                    cell.setData(Qt.ItemDataRole.UserRole, item)
                if simulated:
                    cell.setForeground(QColor(COLORS["warning"]))
                self.history_table.setItem(row, col, cell)
            if session_id == selected_id:
                select_row = row
        if self._history:
            self.history_table.selectRow(select_row if select_row is not None else 0)
        self._history_selected()
        if not self._history:
            self.history_details.setPlainText("还没有本地会话。完成一次“观察—确认—活动—复测”后，可在这里查看记录。")

    def _selected_session(self) -> dict | None:
        row = self.history_table.currentRow() if hasattr(self, "history_table") else -1
        if row < 0:
            return None
        item = self.history_table.item(row, 0)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _history_selected(self) -> None:
        session = self._selected_session()
        self.history_export.setEnabled(session is not None)
        current_active = bool(session and session.get("id", session.get("session_id")) == self.session_id
                              and self.session_state in ("running", "paused", "starting", "stopping", "stop_failed"))
        self.history_delete.setEnabled(session is not None and not current_active)
        self.history_detail_card.setVisible(session is not None)
        self._sync_action_hierarchy()
        if session is not None:
            simulated = session.get("mode") in ("simulation", "demo", "replay")
            consent = session.get("consent") or {}
            lines = [
                f"个人档案：{_text(session.get('user_name'))}",
                "记录来源：" + ("模拟回放（非实测）" if simulated else "真实设备采集" if session.get("mode") == "real" else "来源未核实，不纳入真实反馈统计"),
                f"开始时间：{_time_label(session.get('started_at'))}（北京时间）",
                f"结束时间：{_time_label(session.get('ended_at')) if session.get('ended_at') else '进行中 / 尚未结束'}",
                "转写保存：" + ("已撤回；之前获授权的历史保留" if consent.get("transcript_revoked_at") else "已单独授权" if consent.get("save_transcript", consent.get("transcript")) else "未授权，未保存转写文本"),
                "原始媒体：不长期保存视频与录音",
            ]
            if session.get("latest_loop_caption"):
                lines += ["", "最近一次活动反馈：" + session["latest_loop_caption"]]
            if session.get("last_evaluation"):
                numeric_state = session.get("last_numeric_evaluation", session.get("last_evaluation"))
                evaluation = {"improved": "用户自评困扰下降", "unchanged": "用户自评无变化", "worse": "用户自评困扰上升", "insufficient": "缺少有效复测，无法评价"}.get(numeric_state, "已记录")
                delta = _number(session.get("last_delta"))
                if delta is not None:
                    evaluation += f"（前测减后测 = {delta:g}）"
                feedback = {"helpful": "觉得有帮助", "no_change": "无明显帮助", "worse": "更不舒服", "unknown": "暂时无法评价"}.get(session.get("last_helpfulness"), "未填写")
                lines += ["即时前后自评：" + evaluation, "即时主观帮助程度：" + feedback]
            followup_status = {"pending": "已安排，尚未填写", "completed": "已完成，结果已计入最近活动反馈",
                               "skipped": "缺少有效自评，无法评价持续变化"}.get(session.get("latest_followup_status"))
            if followup_status:
                lines.append("稍后复测：" + followup_status)
            coverage = session.get("coverage") or {}
            if isinstance(coverage, dict) and coverage:
                lines += ["", "有效观察覆盖（区间去重，不把缺失视为中性）："]
                for key, label in (("visual_seconds", "视觉"), ("audio_seconds", "语音"), ("joint_seconds", "联合")):
                    value = _number(coverage.get(key))
                    if value is not None:
                        lines.append(f"  {label}：{value:.1f} 秒")
            interventions = session.get("interventions", session.get("recent_interventions", []))
            if isinstance(interventions, list) and interventions:
                lines += ["", "活动反馈："]
                for record in interventions[-8:]:
                    activity = ACTIVITIES.get(record.get("activity"), "活动确认")
                    pre, post = record.get("pre"), record.get("post")
                    helpfulness = {"helpful": "觉得有帮助", "no_change": "无明显帮助", "worse": "更不舒服", "unknown": "暂时无法评价"}.get(record.get("helpfulness"), "未填写反馈")
                    status = {"completed": "已完成", "cancelled": "已中止", "running": "进行中"}.get(record.get("status"), "已记录")
                    lines.append(f"  {activity} · {status} · 自评 {_text(pre)} → {_text(post, '无复测')} · {helpfulness}")
            lines += ["", "导出所选会话可获得完整观察、自评、反馈与事件时间记录。"]
            self.history_details.setPlainText("\n".join(lines))

    def _export_history(self) -> None:
        session = self._selected_session()
        if not session:
            self._on_status("请先选择要导出的会话。")
            return
        session_id = str(session.get("id", session.get("session_id", "")))
        path, _ = QFileDialog.getSaveFileName(self, "导出会话及完整结构化事件", f"会话_{session_id[:12]}.json", "JSON 完整记录 (*.json)")
        if path:
            if not path.lower().endswith(".json"):
                path += ".json"
            self.controller.export_session(session_id, path)

    def _delete_history(self) -> None:
        session = self._selected_session()
        if not session:
            self._on_status("请先选择要删除的会话。")
            return
        session_id = str(session.get("id", session.get("session_id", "")))
        if session_id == self.session_id and self.session_state in ("running", "paused", "starting", "stopping", "stop_failed"):
            self._on_status("请先结束当前会话，再删除这条记录。")
            return
        answer = QMessageBox.question(self, "删除本地会话", "删除所选会话及其观察、转写、自评和活动记录？\n此操作无法撤销。", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            self.controller.delete_session(session_id)

    def closeEvent(self, event) -> None:
        self.breathing.set_running(False)
        self.controller.shutdown()
        event.accept()
