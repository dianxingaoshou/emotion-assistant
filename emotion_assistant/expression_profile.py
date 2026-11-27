"""Offline measurements of emotion expressed in a Chinese or English passage."""

from .affect_dimensions import predict_affect_dimensions
from .chinese_affect import is_chinese_text
from .chinese_dimensions import predict_chinese_dimensions
from .chinese_emotions import analyze_chinese_emotions
from .fine_emotions import analyze_fine_emotions
from .research_text_review import _language_supported


def english_expression_profile(text):
    if not text or not _language_supported(text):
        return None
    fine = analyze_fine_emotions(text)
    ids = {item["id"] for item in fine["labels"]}
    fine["neutral_conflict"] = "neutral" in ids and len(ids) > 1
    return {"scope": "expressed_text", "language": "en", "fine_emotions": fine,
            "affect_dimensions": predict_affect_dimensions(text)}


def text_expression_profile(text):
    if is_chinese_text(text):
        return {"scope": "expressed_text", "language": "zh",
                "fine_emotions": analyze_chinese_emotions(text),
                "affect_dimensions": predict_chinese_dimensions(text)}
    return english_expression_profile(text)


def format_expression_profile(profile):
    if not profile:
        return ""
    labels = profile["fine_emotions"]
    dimensions = profile["affect_dimensions"]
    if profile.get("language") == "zh":
        lines = []
        if labels["status"] == "ok" and labels["labels"]:
            first = labels["labels"][0]
            lines.append(f"中文表达候选　{first['name_zh']}（模型相对分 {first['score']:.2f}）")
        elif labels["status"] == "model_unavailable":
            lines.append("中文细类模型未就绪。")
        else:
            lines.append("中文表达类别暂缓判断。")
        if dimensions["status"] == "ok":
            axes = dimensions["dimensions"]
            lines.append(f"中文表达维度（1–9，本地字符模型）\n愉悦度　{axes['valence']['score']:.2f}"
                         f"　激活度　{axes['arousal']['score']:.2f}")
        elif dimensions["status"] == "too_long":
            lines.append("维度估计适用于短文本，请选择一次发言。")
        elif dimensions["status"] == "model_unavailable":
            lines.append("中文维度模型未就绪。")
        else:
            lines.append("文字依据不足，暂缓估计愉悦度和激活度。")
        return "\n".join(lines)
    lines = []
    if labels["status"] == "ok":
        visible = labels["labels"][:4]
        items = "、".join(f"{item['name_zh']} {item['score']:.3f}" for item in visible)
        tail = f"（共 {len(labels['labels'])} 类）" if len(labels["labels"]) > len(visible) else ""
        lines.append("文字表达的情绪　" + items + tail + " · 各类独立模型分")
        if labels.get("neutral_conflict"):
            lines.append("中性和具体情绪同时超过阈值，请核对原话。")
    elif labels["status"] == "model_unavailable":
        lines.append("细类模型未就绪。")
    else:
        lines.append("文字情绪细类暂缓判断。")
    if dimensions["status"] == "ok":
        lines.append("英文表达维度（1–5，括号内为各维约 80% 预测区间）")
        for key, name in (("valence", "愉悦度"), ("arousal", "激活度"), ("dominance", "掌控感线索")):
            item = dimensions["dimensions"][key]
            low, high = item["interval"]
            lines.append(f"{name}　{item['score']:.2f}（{low:.2f}–{high:.2f}）")
    elif dimensions["status"] == "too_long":
        lines.append("维度估计适用于短文本，请选择一次发言。")
    elif dimensions["status"] == "model_unavailable":
        lines.append("英文维度模型未就绪。")
    else:
        lines.append("可用英文词不足，暂缓维度估计。")
    return "\n".join(lines)
