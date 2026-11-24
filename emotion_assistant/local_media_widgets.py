"""Local file analysis: the source frame and its result are primary, sample
coverage is secondary, and export is available after completion. Use the
existing card, button, text and spacing tokens with one primary file action.
"""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from PySide6.QtCore import QIODevice, QSaveFile, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QProgressBar, QScrollArea,
    QSizePolicy, QVBoxLayout, QWidget,
)

from .local_media import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, MediaCancelled, analyze_media
from .fusion_explanation import evidence_text
from .text_file import MAX_TEXT_BYTES, text_polarity_summary
from .content_widgets import ContentReadout
from .paths import MODELS
from .ui_theme import COLORS, RADIUS, SPACE
from .ui_widgets import LABEL_ORDER, SignalDistribution

MODES = {"joint": "表情＋声音", "visual_only": "仅表情", "audio_only": "仅声音",
         "mixed": "不同时段使用不同通道", "unavailable": "缺少有效线索"}

def _annotated_preview(frame, observation):
    height, width = frame.shape[:2]
    raw = QImage(frame.data, width, height, frame.strides[0], QImage.Format.Format_BGR888)
    if width > 960 or height > 640:
        raw = raw.scaled(960, 640, Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
    image = raw.convertToFormat(QImage.Format.Format_RGB32)
    box = observation.get("face_box")
    if box:
        x, y, w, h = box
        sx, sy = image.width() / width, image.height() / height
        painter = QPainter(image)
        painter.setPen(QPen(QColor(COLORS["surface"]), 4))
        painter.drawRect(round(x * sx), round(y * sy), round(w * sx), round(h * sy))
        painter.setPen(QPen(QColor(COLORS["primary"]), 2))
        painter.drawRect(round(x * sx), round(y * sy), round(w * sx), round(h * sy))
        painter.end()
    return image


class _MediaWorker(QThread):
    """The GUI polls a single preview slot; video frames never queue as signals."""

    def __init__(self, path, parent, transcribe, strategy):
        super().__init__(parent)
        self.path = path
        self.transcribe = transcribe
        self.strategy = strategy
        self.summary = None
        self.error = None
        self.was_cancelled = False
        self.first_valid = None
        self._pending = None
        self._stage = None
        self._lock = Lock()

    def run(self):
        try:
            self.summary = analyze_media(self.path, model_dir=MODELS, on_sample=self._publish,
                                         cancelled=self.isInterruptionRequested, transcribe=self.transcribe,
                                         strategy=self.strategy, on_status=self._publish_stage)
        except MediaCancelled:
            self.was_cancelled = True
        except Exception as exc:
            self.error = str(exc) or type(exc).__name__

    def _publish(self, frame, observation, progress):
        if self.isInterruptionRequested():
            raise MediaCancelled()
        sample = (_annotated_preview(frame, observation), observation, progress)
        with self._lock:
            self._pending = sample
            if self.first_valid is None and observation.get("eligible"):
                self.first_valid = sample

    def take_preview(self):
        with self._lock:
            sample, self._pending = self._pending, None
            return sample

    def _publish_stage(self, message):
        with self._lock:
            self._stage = message

    def take_stage(self):
        with self._lock:
            stage, self._stage = self._stage, None
            return stage


def _media_reason(reason):
    from .ui import _reason

    if reason == "low_visual_quality":
        return "画面质量偏低"
    return _reason(reason).rstrip("。")


class LocalMediaPage(QWidget):
    ready_to_close = Signal()

    def __init__(self, parent=None, *, strategy_provider=None):
        super().__init__(parent)
        from .ui import _button, _card

        self._worker = None
        self._path = None
        self._image = None
        self._closing = False
        self._cancel_requested = False
        self.result = None
        self._strategy_provider = strategy_provider
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._refresh_sample)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE["lg"])
        toolbar = QHBoxLayout()
        toolbar.setSpacing(SPACE["sm"])
        self.choose_button = _button("选择本地文件", True)
        self.choose_button.clicked.connect(self._choose)
        toolbar.addWidget(self.choose_button)
        self.transcribe = QCheckBox("转写本次文件语音")
        self.transcribe.setToolTip("让本次英文转写参与发言内容分析。导出的 JSON 包含自动转写和你确认的校对文字。")
        toolbar.addWidget(self.transcribe)
        self.cancel_button = _button("取消检测")
        self.cancel_button.clicked.connect(self.cancel)
        self.cancel_button.hide()
        toolbar.addWidget(self.cancel_button)
        self.filename = QLabel()
        self.filename.setTextFormat(Qt.TextFormat.PlainText)
        self.filename.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.filename.setObjectName("muted")
        toolbar.addWidget(self.filename, 1)
        self.export_button = _button("导出结果 JSON")
        self.export_button.clicked.connect(self._export)
        self.export_button.setEnabled(False)
        self.export_button.hide()
        toolbar.addWidget(self.export_button)
        layout.addLayout(toolbar)

        panels = QHBoxLayout()
        panels.setSpacing(SPACE["lg"])
        source_card, source_layout = _card("文件预览")
        self.preview = QLabel("选择照片、视频、音频或文本")
        self.preview.setTextFormat(Qt.TextFormat.PlainText)
        self.preview.setAccessibleName("本地文件检测画面")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setWordWrap(True)
        self.preview.setMinimumSize(160, 160)
        self.preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.preview.setStyleSheet(f"background:{COLORS['tinted_surface']}; color:{COLORS['secondary_text']};"
                                   f"border-radius:{RADIUS['card']}px;")
        source_layout.addWidget(self.preview, 1)
        self.sample_caption = QLabel()
        self.sample_caption.setObjectName("muted")
        self.sample_caption.setWordWrap(True)
        source_layout.addWidget(self.sample_caption)
        panels.addWidget(source_card, 3)

        result_card, result_layout = _card("检测结果")
        self.status = QLabel("等待选择文件")
        self.status.setObjectName("cardTitle")
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        result_layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setAccessibleName("文件检测进度")
        self.progress.setTextVisible(False)
        self.progress.hide()
        result_layout.addWidget(self.progress)
        self.content_readout = ContentReadout()
        result_layout.addWidget(self.content_readout)
        self.content_intervals = QComboBox()
        self.content_intervals.setAccessibleName("选择要查看的发言时段")
        self.content_intervals.currentIndexChanged.connect(self._show_content_result)
        result_layout.addWidget(self.content_intervals)
        self.review_button = _button("校对英文转写")
        self.review_button.clicked.connect(self._review_transcript)
        result_layout.addWidget(self.review_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.reviewed_intensity = QLabel()
        self.reviewed_intensity.setTextFormat(Qt.TextFormat.PlainText)
        self.reviewed_intensity.setWordWrap(True)
        result_layout.addWidget(self.reviewed_intensity)
        self.cue_caption = QLabel("声画线索")
        self.cue_caption.setObjectName("muted")
        result_layout.addWidget(self.cue_caption)
        self.aggregate = QLabel()
        self.aggregate.setObjectName("scoreValue")
        self.distribution = SignalDistribution()
        self.score_text = QLabel()
        self.score_text.setWordWrap(True)
        self.details = QLabel("照片看表情，音频听声音，视频按时间对齐两路线索。也可单独选择 UTF-8 文本。")
        self.details.setObjectName("muted")
        self.details.setTextFormat(Qt.TextFormat.PlainText)
        self.details.setWordWrap(True)
        for widget in (self.aggregate, self.distribution, self.score_text, self.details):
            result_layout.addWidget(widget)
        self.intervals = QComboBox()
        self.intervals.setAccessibleName("查看分时段识别依据")
        self.intervals.currentIndexChanged.connect(self._show_interval)
        result_layout.addWidget(self.intervals)
        self.interval_detail = QLabel()
        self.interval_detail.setTextFormat(Qt.TextFormat.PlainText)
        self.interval_detail.setObjectName("muted")
        self.interval_detail.setWordWrap(True)
        self.interval_detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        result_layout.addWidget(self.interval_detail)
        result_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(result_card)
        panels.addWidget(scroll, 2)
        layout.addLayout(panels, 1)
        self._reset_scores()

    def _choose(self):
        photos = " ".join("*" + ext for ext in sorted(IMAGE_EXTENSIONS))
        videos = " ".join("*" + ext for ext in sorted(VIDEO_EXTENSIONS))
        audio = " ".join("*" + ext for ext in sorted(AUDIO_EXTENSIONS))
        path, _ = QFileDialog.getOpenFileName(
            self, "选择本地文件", str(self._path.parent) if self._path else "",
            f"支持的文件 ({photos} {videos} {audio} *.txt);;照片 ({photos});;视频 ({videos});;音频 ({audio});;文本 (*.txt)")
        if path:
            self.start_file(path)

    def start_file(self, path):
        if self._worker is not None or self._closing:
            return False
        self._path = Path(path)
        self.result = None
        self._cancel_requested = False
        self.filename.setText(self._path.name)
        self.filename.setToolTip(str(self._path))
        self._clear_preview("正在读取画面…")
        self._reset_scores()
        self.details.clear()
        self.status.setText("正在读取文件…")
        self.choose_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.export_button.hide()
        self.cancel_button.setEnabled(True)
        self.cancel_button.show()
        self.progress.setRange(0, 0)
        self.progress.show()
        transcribe = self.transcribe.isChecked() and self._path.suffix.lower() in AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
        strategy = self._strategy_provider() if self._strategy_provider else None
        self._worker = _MediaWorker(self._path, self, transcribe, strategy)
        self.transcribe.setEnabled(False)
        self._worker.finished.connect(self._finished)
        self._worker.start()
        self._timer.start()
        return True

    def cancel(self):
        if self._worker is not None:
            self._cancel_requested = True
            self._worker.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.status.setText("正在取消检测…")

    def stop_for_close(self):
        if self._worker is None:
            return True
        self._closing = True
        self.cancel()
        return False

    def _clear_preview(self, text):
        self._image = None
        self.preview.setText(text)
        self.sample_caption.clear()

    def _reset_scores(self):
        self.content_readout.set_result(None)
        self.content_intervals.hide()
        self.content_intervals.clear()
        self.review_button.hide()
        self.reviewed_intensity.clear()
        self.reviewed_intensity.hide()
        self.cue_caption.hide()
        self.aggregate.hide()
        self.distribution.set_scores(None)
        self.distribution.hide()
        self.score_text.clear()
        self.score_text.hide()
        self.intervals.hide()
        self.intervals.clear()
        self.interval_detail.clear()
        self.interval_detail.hide()

    def _refresh_sample(self):
        if self._worker is not None and not self._cancel_requested:
            sample = self._worker.take_preview()
            if sample is not None:
                self._show_sample(*sample)
            stage = self._worker.take_stage()
            if stage:
                self.status.setText(stage)

    def _show_sample(self, image, observation, progress):
        from .ui import LABELS

        self._image = image
        self._scale_preview()
        prefix = "照片"
        if self._path.suffix.lower() in VIDEO_EXTENSIONS:
            prefix = f"抽样 {observation['sample_index']} · {observation['sample_time_s']:.2f} 秒"
        scores = observation.get("scores")
        if observation.get("eligible") and scores:
            top = max(scores, key=scores.get)
            frame_label = f"四类线索：{LABELS[top]} {scores[top]:.1%}"
            raw = observation.get("raw_scores")
            if raw:
                raw_top = max(raw, key=raw.get)
                label = "轻蔑" if raw_top == "contempt" else LABELS.get(raw_top, raw_top)
                frame_label = f"八类表情：{label} {raw[raw_top]:.1%}　·　" + frame_label
        else:
            reason = observation.get("rejection_reason") or observation.get("reason")
            if observation.get("valid") and reason == "ok":
                reason = "low_visual_quality"
            frame_label = f"未计入结果 · {_media_reason(reason)}"
        self.sample_caption.setText(f"{prefix}　{frame_label}")
        self.status.setText("正在检测…")
        if progress is not None:
            self.progress.setRange(0, 100)
            self.progress.setValue(round(100 * progress))

    def _scale_preview(self):
        if self._image is not None:
            self.preview.setPixmap(QPixmap.fromImage(self._image).scaled(
                self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale_preview()

    @Slot()
    def _finished(self):
        worker = self._worker
        self._timer.stop()
        self._refresh_sample()
        self._worker = None
        self.choose_button.setEnabled(True)
        self.transcribe.setChecked(False)
        self.transcribe.setEnabled(True)
        self.cancel_button.hide()
        self.progress.hide()
        if worker.was_cancelled or self._cancel_requested:
            self._clear_preview("选择文件以重新检测")
            self.status.setText("检测已取消")
        elif worker.error:
            self._clear_preview("请选择其他文件，或修复文件后重试")
            self.status.setText("检测失败")
            self.details.setText(worker.error)
        else:
            self.result = worker.summary
            if worker.first_valid:
                self._show_sample(*worker.first_valid)
            self._show_summary()
            self.export_button.setEnabled(True)
            self.export_button.show()
        worker.deleteLater()
        if self._closing:
            self._closing = False
            self.ready_to_close.emit()

    def _show_summary(self):
        from .ui import LABELS, NEEDS

        result = self.result
        kind = result["kind"]
        summary = result.get("content_summary")
        show_content = bool(summary) and (kind == "text" or result.get("transcription_status") or
                                          result.get("transcripts"))
        if show_content:
            self.content_intervals.blockSignals(True)
            self.content_intervals.clear()
            self.content_intervals.addItem("全部发言时段", -1)
            for i, row in enumerate(result.get("content_intervals", [])):
                if "text" in row.get("available_modalities", []) or row.get("intensity") is not None:
                    self.content_intervals.addItem(f"发言 {row['t_start']:.1f}–{row['t_end']:.1f} 秒", i)
            self.content_intervals.blockSignals(False)
            self.content_intervals.setVisible(self.content_intervals.count() > 1)
            self._show_content_result(0)
            self.cue_caption.setVisible(kind != "text")
        if kind == "text":
            text = result["text"]
            self._clear_preview(text[:360] + ("…" if len(text) > 360 else ""))
            self.sample_caption.setText(f"文本原文 · {len(text)} 个字符")
            context = result["semantic_context"]
            polarity = text_polarity_summary(context)
            self._completed_status = "文本分析完成"
            self.status.setText(self._completed_status)
            lines = [] if show_content else [polarity or "本段未获得可用的英文极性分数。"]
            if context.get("need_hint") in NEEDS:
                lines.append("文本提及可能的需要：" + NEEDS[context["need_hint"]] + "，需要本人确认。")
            lines.append("结果描述文字表达的倾向。导出包含选择的原文。")
            self.details.setText("\n".join(lines))
            return
        valid, total = result["valid_count"], result["sampled_count"]
        audio_count = result.get("audio_valid_count", 0)
        mode = MODES.get(result.get("mode"), "仅表情")
        name = {"image": "照片", "video": "视频", "audio": "音频"}[kind]
        self._completed_status = f"{name}检测完成 · {mode}"
        self.status.setText(self._completed_status)
        intervals = result.get("fusion_intervals", [])
        decisions = result.get("decision_counts", {})
        accepted = sum(count for k, count in decisions.items() if k != "abstain")
        scores = result["scores"]
        lines = []
        if kind != "audio":
            lines.append(f"画面通过检查 {valid}/{total} 帧")
        if kind != "image":
            audio_state = {"no_audio_track": "没有音轨", "no_speech": "未检出可用人声",
                           "unusable": "人声未通过质量检查", "unavailable": "音轨处理不完整"}
            reason = audio_state.get(result.get("audio_status"))
            lines.append(f"人声通过检查 {audio_count} 段" + (f" · {reason}" if reason else ""))
            coverage = result.get("timeline_coverage")
            if coverage:
                end = coverage["duration_s"]
                length = f" / 文件 {end:.1f} 秒" if end is not None else ""
                lines.append(f"有分数 {coverage['scored_s']:.1f} 秒{length}")
            if not total:
                self._clear_preview("此文件未取得画面\n可以在右侧查看声音结果")
        if scores and (accepted or not decisions):
            top = max(scores, key=scores.get)
            self.aggregate.setText(f"{LABELS[top]}  {scores[top]:.1%}")
            self.distribution.set_scores(scores)
            labels = [f"{LABELS[key]} {scores[key]:.1%}" for key in LABEL_ORDER]
            self.score_text.setText("　".join(labels[:2]) + "\n" + "　".join(labels[2:]))
            for widget in (self.aggregate, self.distribution, self.score_text):
                widget.show()
            if kind == "video" or kind == "audio":
                lines.append("分数按有输出的时段长度汇总，含暂缓时段的模型分数。")
        else:
            lines.append("暂缓判断：当前没有足够可靠的依据。")
        if decisions.get("abstain"):
            lines.append(f"暂缓判断 {decisions['abstain']}/{len(intervals)} 个时段")
        lines.extend(f"{_media_reason(reason)}（{count} 帧）" for reason, count in result["reasons"].items())
        lines.extend(result.get("issues", []))
        for row in result.get("transcripts", []):
            if row.get("valid") and row.get("text"):
                lines.append(f"转写 {row['t_start']:.1f} 秒 · " + row["text"][:160])
                break
        polarity = text_polarity_summary(result.get("semantic_context"))
        if polarity:
            lines.append("按有效转写汇总 · " + polarity)
        elif result.get("transcription_status") == "no_usable_transcript":
            lines.append("本次已请求转写，但没有取得可用文字。")
        self.details.setText("\n".join(lines))
        # A saved JSON contains every interval. The selector initially shows a
        # joint decision if available, otherwise the first scored interval.
        self.intervals.blockSignals(True)
        self.intervals.clear()
        for row in intervals:
            self.intervals.addItem(f"{row['t_start']:.1f}–{row['t_end']:.1f} 秒 · " +
                                   MODES.get(row.get("status"), "缺少有效线索"))
        first_scored = next((i for i, row in enumerate(intervals) if row.get("scores")), 0)
        chosen = next((i for i, row in enumerate(intervals) if row.get("status") == "joint"), first_scored)
        self.intervals.setCurrentIndex(chosen)
        self.intervals.blockSignals(False)
        self.intervals.setVisible(bool(intervals) and kind != "image")
        self._show_interval(chosen)

    def _show_content_result(self, index):
        if index < 0 or not self.result:
            return
        source = self.content_intervals.itemData(index)
        whole_file = source is None or source < 0
        reviewable = (whole_file and self.result.get("kind") in {"audio", "video"}
                      and self.result.get("transcription_status") == "available")
        self.review_button.setVisible(reviewable)
        review = self.result.get("reviewed_transcript")
        self.reviewed_intensity.setVisible(bool(reviewable and review))
        if reviewable and review:
            estimate = review["intensity"]
            value = "英文文字依据不足，暂缓估计"
            if estimate is not None:
                score = round(estimate["value"], 2)
                value = (f"{score:+.2f}" if score else "0.00") + "（−3 至 +3）"
            self.reviewed_intensity.setText("校对后整段英文强度　" + value)
        else:
            self.reviewed_intensity.clear()
        if whole_file:
            self.content_readout.set_result(self.result.get("content_summary"),
                                            automatic_transcript=self.result.get("kind") != "text")
        else:
            row = self.result["content_intervals"][source]
            self.content_readout.set_result(row, f"发言内容 {row['t_start']:.1f}–{row['t_end']:.1f} 秒",
                                            automatic_transcript=True)

    def _review_transcript(self):
        from .english_intensity import predict_english_intensity

        result = self.result
        if not result or result.get("transcription_status") != "available":
            return
        original = " ".join(row["text"].strip() for row in result.get("transcripts", [])
                            if row.get("valid") and not row.get("stale") and row.get("text"))
        if not original:
            return
        previous = result.get("reviewed_transcript") or {}
        edited, accepted = QInputDialog.getMultiLineText(
            self, "校对本次文件的英文转写",
            "对照原音修正漏词和错词，或清空以撤销校对。确认后按整段英文估分，导出时会包含校对文字。",
            previous.get("text", original))
        if not accepted or result is not self.result:
            return
        edited = edited.strip()
        if len(edited) > MAX_TEXT_BYTES or len(edited.encode("utf-8")) > MAX_TEXT_BYTES:
            self.status.setText("校对文字过长，请缩短至 16 KiB 后重试。")
            return
        if not edited or edited == original:
            result.pop("reviewed_transcript", None)
        else:
            result["reviewed_transcript"] = {"source": "user_edited", "text": edited,
                                              "method": "reviewed_english_text_v1",
                                              "intensity": predict_english_intensity(edited)}
        self._show_content_result(self.content_intervals.currentIndex())
        self.status.setText(self._completed_status)

    def _show_interval(self, index):
        rows = (self.result or {}).get("fusion_intervals", [])
        if index < 0 or index >= len(rows):
            self.interval_detail.hide()
            return
        row = rows[index]
        decision = {"abstain": "暂缓判断", "observe_only": "仅观察", "ask": "可询问本人感受"}
        text = decision.get(row.get("decision"), "仅观察")
        if row.get("conflict"):
            text += " · 表情与声音有分歧"
        if row.get("strategy_used") == "single_cue_calibration":
            text += " · 已按单路标注校准"
        explanation = evidence_text(row.get("evidence_explanation"))
        self.interval_detail.setText(text + ("\n" + explanation if explanation else ""))
        self.interval_detail.show()

    def _export(self):
        if self.result is None:
            return
        target, _ = QFileDialog.getSaveFileName(
            self, "导出文件检测结果", f"{self._path.stem}-表情检测.json", "JSON 文件 (*.json)")
        if not target:
            return
        try:
            output = Path(target)
            if not output.suffix:
                output = output.with_suffix(".json")
                if output.exists():
                    raise ValueError("同名 JSON 文件已存在。请选择新名称，或输入含 .json 的全名确认覆盖。")
            if output.resolve() == self._path.resolve() or (
                output.exists() and self._path.exists() and output.samefile(self._path)
            ):
                raise ValueError("请选择新的 JSON 文件名，以保留原始照片或视频。")
            if output.suffix.lower() != ".json":
                raise ValueError("请选择 .json 格式的文件名。")
            data = json.dumps(self.result, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
            file = QSaveFile(str(output))
            if not file.open(QIODevice.OpenModeFlag.WriteOnly):
                raise OSError(file.errorString())
            if file.write(data) != len(data) or not file.commit():
                raise OSError(file.errorString())
            self.status.setText(f"{self._completed_status}\n已导出至 {output.name}")
        except (OSError, ValueError) as exc:
            self.status.setText(f"导出失败：{exc}")
