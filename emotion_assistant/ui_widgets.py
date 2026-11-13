"""Small painted components for the local desktop interface.

These widgets only render observations supplied by the controller.  They never
invent scores, carry a previous score into a missing interval, or infer a state.
"""

from __future__ import annotations

import math
import time

from PySide6.QtCore import QLineF, QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget
from .ui_theme import COLORS, TYPE, SPACE


LABEL_ORDER = ("positive", "neutral", "negative", "uncertain")
SCORE_COLORS = {
    "positive": "#5A9F89",
    "neutral": "#C1D0CB",
    "negative": "#D7A77B",
    "uncertain": "#E5EAE7",
}


def _number(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _distribution(scores) -> dict[str, float] | None:
    if not isinstance(scores, dict):
        return None
    values = {label: _number(scores.get(label, 0.0)) for label in LABEL_ORDER}
    if any(value is None or value < 0 for value in values.values()):
        return None
    total = sum(values.values())
    if total <= 0 or not math.isfinite(total):
        return None
    return {label: value / total for label, value in values.items()}


def _font(size: int, weight=QFont.Weight.Normal) -> QFont:
    # Refactoring UI §4.1: painted labels use the same pixel scale as QSS.
    font = QFont("Microsoft YaHei UI")
    font.setPixelSize(TYPE["meta"] if size <= 9 else TYPE["body"])
    font.setWeight(QFont.Weight.DemiBold if weight >= QFont.Weight.Medium else QFont.Weight.Normal)
    return font


# §5.7: color reinforces a signal already conveyed by line and marker shape.
SERIES_STYLES = {
    "joint": (COLORS["chart_joint"], Qt.PenStyle.SolidLine),
    "visual": (COLORS["chart_visual"], Qt.PenStyle.DashLine),
    "audio": (COLORS["chart_audio"], Qt.PenStyle.DotLine),
}


def _marker(painter, point, series, radius=3):
    if series == "joint":
        painter.drawEllipse(point, radius, radius)
    elif series == "visual":
        painter.drawRect(QRectF(point.x() - radius, point.y() - radius, radius * 2, radius * 2))
    else:
        diamond = QPainterPath()
        diamond.moveTo(point.x(), point.y() - radius - 1)
        diamond.lineTo(point.x() + radius + 1, point.y())
        diamond.lineTo(point.x(), point.y() + radius + 1)
        diamond.lineTo(point.x() - radius - 1, point.y())
        diamond.closeSubpath()
        painter.drawPath(diamond)


class ChartLegend(QWidget):
    """§5.7: named series include actual non-color line/shape keys."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self.setMinimumWidth(260)
        self.setAccessibleName("联合：实线圆点；视觉：虚线方点；语音：点线菱形")

    def sizeHint(self):
        return QSize(276, 28)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(_font(9))
        for index, (series, label) in enumerate((("joint", "联合"), ("visual", "视觉"), ("audio", "语音"))):
            x = index * 88
            color, style = SERIES_STYLES[series]
            painter.setPen(QPen(QColor(color), 2, style))
            painter.drawLine(QLineF(x + 2, 14, x + 29, 14))
            painter.setPen(QPen(QColor(color), 1))
            painter.setBrush(QColor(color))
            _marker(painter, QPointF(x + 16, 14), series)
            painter.setPen(QColor(COLORS["secondary_text"]))
            painter.drawText(QRectF(x + 36, 3, 45, 22), Qt.AlignmentFlag.AlignVCenter, label)


def _elapsed_label(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


class TrendChart(QWidget):
    """A bounded chart on the session clock, with missing evidence left blank."""

    COLORS = {series: style[0] for series, style in SERIES_STYLES.items()}

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.points: list[tuple[float, float | None, str]] = []
        self._samples: list[tuple[float, float | None, str, float | None]] = []
        self._event_samples = {}
        self._clock_origin = time.monotonic()
        self.setMinimumHeight(204)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAccessibleName("最近两分钟的负向线索趋势")

    def sizeHint(self) -> QSize:
        return QSize(640, 230)

    def clear(self) -> None:
        self.points.clear()
        self._samples.clear()
        self._event_samples.clear()
        self._clock_origin = time.monotonic()
        self.update()

    @staticmethod
    def _series(result: dict) -> str:
        # A fusion result may contain only one usable modality.  Its status is
        # more precise than the generic "fusion" modality name.
        for key in ("status", "mode", "source", "reason"):
            status = str(result.get(key, "")).lower()
            if status in ("visual_only", "visual", "vision"):
                return "visual"
            if status in ("audio_only", "audio"):
                return "audio"
            if status == "joint":
                return "joint"
        modality = str(result.get("modality", "")).lower()
        if modality in ("visual", "visual_summary", "vision"):
            return "visual"
        if modality == "audio":
            return "audio"
        return "joint"

    def add_observation(self, result: dict) -> None:
        stamp = next((value for key in ("t_end", "timestamp", "t_start")
                      if (value := _number(result.get(key))) is not None), None)
        if stamp is None:
            stamp = time.monotonic() - self._clock_origin
        if stamp < 0:
            return
        start = _number(result.get("t_start"))
        series = self._series(result)
        raw_scores = result.get("scores") or result.get("fused_scores")
        scores = _distribution(raw_scores)
        score = _number(result.get("negative_score"))
        if score is None and scores is not None:
            score = scores["negative"] * 100
        elif score is not None and 0 <= score <= 1:
            score *= 100
        if score is not None and not 0 <= score <= 100:
            score = None
        if (result.get("valid") is False or result.get("display_score") is False
                or (scores is not None and scores["uncertain"] >= 0.5)
                or (raw_scores is not None and scores is None)
                or (start is not None and (start < 0 or start >= stamp))):
            score = None
        sample = (stamp, score, series, start)
        event_id = result.get("event_id")
        old = self._event_samples.get(event_id) if event_id else None
        if old in self._samples:
            self._samples.remove(old)
        if event_id:
            self._event_samples[event_id] = sample
        self._samples.append(sample)
        # Late inference keeps its recorded time.  It never appears as a fresh
        # point at the right edge of the chart.
        self._samples.sort(key=lambda sample: sample[0])
        self._samples = self._samples[-240:]
        self._event_samples = {key: value for key, value in self._event_samples.items() if value in self._samples}
        self.points = [(stamp, score, kind) for stamp, score, kind, _ in self._samples]
        self.update()

    def _empty_state(self, painter: QPainter, rect: QRectF) -> None:
        middle = rect.center()
        icon_y = middle.y() - 35
        painter.setPen(QPen(QColor("#D6E2DB"), 1.5, Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap))
        for offset, height in ((-19, 10), (0, 18), (19, 13)):
            painter.drawLine(QLineF(middle.x() + offset, icon_y - height,
                                    middle.x() + offset, icon_y + height))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#B6CFC0"))
        painter.drawEllipse(QPointF(middle.x(), icon_y), 4, 4)
        painter.setFont(_font(11, QFont.Weight.Medium))
        painter.setPen(QColor(COLORS["secondary_text"]))
        painter.drawText(QRectF(rect.left(), middle.y() - 3, rect.width(), 25),
                         Qt.AlignmentFlag.AlignCenter, "等待第一条有效观察")
        painter.setFont(_font(9))
        painter.setPen(QColor(COLORS["tertiary_text"]))
        painter.drawText(QRectF(rect.left(), middle.y() + 26, rect.width(), 22),
                         Qt.AlignmentFlag.AlignCenter, "开始会话后，趋势将在这里更新")

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(35, 12, max(self.width() - 48, 1), max(self.height() - 44, 1))
        painter.setFont(_font(8))
        grid_pen = QPen(QColor("#E3EAE5"), 1, Qt.PenStyle.DotLine)
        for tick in (0, 25, 50, 75, 100):
            y = rect.bottom() - tick / 100 * rect.height()
            painter.setPen(grid_pen)
            painter.drawLine(QLineF(rect.left(), y, rect.right(), y))
            painter.setPen(QColor(COLORS["tertiary_text"]))
            painter.drawText(QRectF(0, y - 9, 26, 18),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             str(tick))

        latest = max((point[0] for point in self.points), default=0)
        right = max(20.0, latest)
        left = max(0.0, right - 120.0)
        span = right - left
        for index in range(5):
            fraction = index / 4
            x = rect.left() + fraction * rect.width()
            alignment = (Qt.AlignmentFlag.AlignLeft if index == 0 else
                         Qt.AlignmentFlag.AlignRight if index == 4 else
                         Qt.AlignmentFlag.AlignHCenter)
            width = 48
            label_x = x if index == 0 else x - width if index == 4 else x - width / 2
            painter.setPen(QColor(COLORS["tertiary_text"]))
            painter.drawText(QRectF(label_x, rect.bottom() + 10, width, 19),
                             alignment | Qt.AlignmentFlag.AlignVCenter,
                             _elapsed_label(left + fraction * span))

        visible = [sample for sample in self._samples if left <= sample[0] <= right]
        if not any(score is not None for _, score, _, _ in visible):
            self._empty_state(painter, rect)
            return

        painter.save()
        painter.setClipRect(rect.adjusted(-4, -4, 4, 4))
        for series, color in self.COLORS.items():
            line_pen = QPen(QColor(color), 2.3 if series == "joint" else 1.7,
                                SERIES_STYLES[series][1], Qt.PenCapStyle.RoundCap,
                                Qt.PenJoinStyle.RoundJoin)
            painter.setPen(line_pen)
            previous = None
            for stamp, score, kind, start in visible:
                if kind != series:
                    continue
                if score is None:
                    previous = None
                    continue
                x = rect.left() + (stamp - left) / span * rect.width()
                y = rect.bottom() - score / 100 * rect.height()
                gap_limit = {"visual": 5.0, "audio": 10.0, "joint": 12.0}[series]
                # Where recorded intervals exist, only join contiguous evidence.
                # An utterance after silence must remain a separate observation.
                contiguous = (previous is not None
                              and 0 <= stamp - previous[0] <= gap_limit
                              and (start is None or start <= previous[0] + 0.075))
                if contiguous:
                    painter.setPen(line_pen)
                    painter.drawLine(QLineF(previous[1], previous[2], x, y))
                painter.setBrush(QColor(color))
                painter.setPen(QPen(QColor(color), 1))
                radius = 2.8 if series == "joint" else 2.1
                _marker(painter, QPointF(x, y), series, radius)
                previous = (stamp, x, y)
        painter.restore()


class SignalDistribution(QWidget):
    """A compact four-class distribution; unavailable scores stay unfilled."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._scores: dict[str, float] | None = None
        self.setMinimumHeight(8)
        self.setMaximumHeight(12)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAccessibleName("四类线索分布")

    def sizeHint(self) -> QSize:
        return QSize(180, 10)

    def set_scores(self, scores: dict | None) -> None:
        self._scores = _distribution(scores)
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0, 0, self.width(), self.height())
        outline = QPainterPath()
        outline.addRoundedRect(rect, rect.height() / 2, rect.height() / 2)
        painter.fillPath(outline, QColor("#EDF1EE"))
        if self._scores is None:
            return
        painter.setClipPath(outline)
        x = 0.0
        for label in LABEL_ORDER:
            width = self._scores[label] * rect.width()
            if width > 0:
                painter.fillRect(QRectF(x, 0, width, rect.height()), QColor(SCORE_COLORS[label]))
            x += width


class StepIndicator(QWidget):
    """Four numbered stages of the voluntary adjustment flow."""

    LABELS = ("确认需要", "调节前自评", "执行活动", "复测与反馈")
    STAGES = {"observing": 0, "confirming": 0, "ready": 1,
              "running": 2, "cooldown": 3, "completed": 3}

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._stage = 0
        self.setMinimumHeight(76)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAccessibleName("调节流程")

    def sizeHint(self) -> QSize:
        return QSize(520, 76)

    def set_state(self, state: str) -> None:
        self._stage = self.STAGES.get(str(state), 0)
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width = self.width() / 4
        centers = [(index + 0.5) * width for index in range(4)]
        y, radius = 22.0, 13.0
        for index in range(3):
            color = "#A9C5B7" if index < self._stage else "#E1E8E3"
            painter.setPen(QPen(QColor(color), 1.5, Qt.PenStyle.SolidLine,
                                Qt.PenCapStyle.RoundCap))
            painter.drawLine(QLineF(centers[index] + radius + 8, y,
                                    centers[index + 1] - radius - 8, y))
        for index, label in enumerate(self.LABELS):
            active = index == self._stage
            previous = index < self._stage
            center = QPointF(centers[index], y)
            if active:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor("#E8F1EC"))
                painter.drawEllipse(center, radius + 4, radius + 4)
            fill = COLORS["primary"] if active else "#EAF2ED" if previous else "#F4F6F3"
            border = COLORS["primary"] if active else "#C8DCD0" if previous else "#DDE6DF"
            painter.setBrush(QColor(fill))
            painter.setPen(QPen(QColor(border), 1))
            painter.drawEllipse(center, radius, radius)
            painter.setFont(_font(9, QFont.Weight.DemiBold))
            painter.setPen(QColor("#FFFFFF" if active else COLORS["success"] if previous else COLORS["tertiary_text"]))
            painter.drawText(QRectF(center.x() - radius, y - radius, radius * 2, radius * 2),
                             Qt.AlignmentFlag.AlignCenter, str(index + 1))
            painter.setFont(_font(9, QFont.Weight.DemiBold if active else QFont.Weight.Normal))
            painter.setPen(QColor(COLORS["text"] if active else COLORS["tertiary_text"]))
            painter.drawText(QRectF(index * width, 45, width, 23),
                             Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, label)
