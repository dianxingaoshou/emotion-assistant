"""Display an utterance-content estimate, its evidence and the optional comparison."""
import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from .ui_theme import SPACE


NAMES = {"text": "文字", "visual": "表情", "audio": "声音"}
POLARITIES = {"positive": "正向", "neutral": "中性", "negative": "负向"}


class ContentReadout(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE["xs"])
        self.headline = QLabel()
        self.headline.setObjectName("cardTitle")
        self.basis = QLabel()
        self.basis.setObjectName("muted")
        self.intensity = QLabel()
        self.intensity.setToolTip("英文发言表达的强度。−3 表示强负向，0 表示中性，+3 表示强正向。")
        self.compare = QPushButton("查看联合试算")
        self.compare.setObjectName("ghost")
        self.compare.setCheckable(True)
        self.comparison = QLabel()
        self.comparison.setObjectName("smallMuted")
        for label in (self.headline, self.basis, self.intensity, self.comparison):
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setWordWrap(True)
        layout.addWidget(self.headline)
        layout.addWidget(self.basis)
        layout.addWidget(self.intensity)
        layout.addWidget(self.compare, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.comparison)
        self.compare.toggled.connect(self._toggle_comparison)
        self.set_result(None)

    def _toggle_comparison(self, expanded):
        self.compare.setText("收起联合试算" if expanded else "查看联合试算")
        self.comparison.setVisible(expanded)

    def set_result(self, result, title="发言内容倾向", *, automatic_transcript=False):
        if not result:
            self.headline.clear()
            self.basis.clear()
            self.intensity.clear()
            self.comparison.clear()
            self.compare.setChecked(False)
            self.hide()
            return
        label = result.get("label")
        scores = result.get("scores") or {}
        available = result.get("available_modalities", [])
        used = result.get("used_modalities", [])
        intensity = (result.get("intensity") or {}).get("value")
        has_intensity = (type(intensity) in (int, float) and math.isfinite(intensity)
                         and -3 <= intensity <= 3)
        if result.get("decision") == "estimate" and label in scores and label in POLARITIES:
            self.headline.setText(f"{title} · {POLARITIES[label]} {scores[label]:.1%}")
            names = "、".join(NAMES[key] for key in ("text", "audio", "visual") if key in used)
            basis = f"本次依据：{names}" if names else ""
            if result.get("status") == "text_preferred":
                basis += "。联合试算供比较。"
        else:
            self.headline.setText(f"{title} · 暂缓判断")
            basis = ("极性分类的依据不足。" if "text" in available or has_intensity
                     else "没有足够的英文文字依据。")
        if result.get("disagreement") and len(available) > 1:
            basis += "　各路线索的最高类别有分歧。"
        if (result.get("observed_seconds") or 0) > 0:
            basis += f"\n可判断 {result.get('covered_seconds', 0):.1f} 秒 / 所列时段 {result['observed_seconds']:.1f} 秒"
        self.basis.setText(basis)
        self.intensity.setVisible("intensity" in result)
        if has_intensity:
            value = round(intensity, 2)
            value_text = f"{value:+.2f}" if value else "0.00"
            title = "英文表达强度 · 自动转写试算" if automatic_transcript else "英文表达强度"
            if automatic_transcript and result["intensity"].get("method") == "asr_word_character_pool":
                title += "（整段词与字符）"
            elif automatic_transcript and "interval_count" in result:
                title += "（按发言时长加权）"
            line = f"{title}　{value_text}（−3 至 +3）"
            seconds = result.get("intensity_covered_seconds")
            if seconds is not None:
                line += f"　依据 {seconds:.1f} 秒发言"
            self.intensity.setText(line)
        else:
            self.intensity.setText("英文表达强度　暂缓估计" if "intensity" in result else "")
        trial_inputs = result.get("candidate_weights") or {}
        if trial_inputs.get("text", 0) > 0 and any(trial_inputs.get(k, 0) > 0 for k in ("audio", "visual")):
            name = result.get("candidate_label")
            candidate = result.get("candidate_estimate_scores") or {}
            weights = result.get("candidate_estimate_weights") or {}
            if result.get("candidate_decision") == "estimate" and name in candidate:
                keys = [k for k in ("text", "audio", "visual") if weights.get(k, 0) > 0]
                sources = "＋".join(NAMES[k] for k in keys)
                share = "、".join(f"{NAMES[k]} {weights[k]:.0%}" for k in keys)
                scope = "有效时段汇总" if "interval_count" in result else "试算"
                details = f"{sources}{scope}：{POLARITIES[name]} {candidate[name]:.1%}\n试算权重：{share}"
            else:
                details = "联合试算暂缓判断，分数未达到输出门槛。"
            self.comparison.setText(details)
            self.compare.show()
            self.comparison.setVisible(self.compare.isChecked())
        else:
            self.compare.hide()
            self.compare.setChecked(False)
            self.comparison.clear()
            self.comparison.hide()
        self.show()
