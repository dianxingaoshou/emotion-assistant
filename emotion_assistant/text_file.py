"""Analyze a deliberately selected short UTF-8 text without camera or ASR."""

from pathlib import Path

from .semantic_context import interpret_context
from .content_fusion import fuse_content


MAX_TEXT_BYTES = 16384


def analyze_text_file(path):
    path = Path(path)
    with path.open("rb") as stream:
        data = stream.read(MAX_TEXT_BYTES + 1)
    if len(data) > MAX_TEXT_BYTES:
        raise ValueError("文本较长，请每次选择一个片段（上限 16 KiB）。")
    try:
        text = data.decode("utf-8-sig").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("无法读取文本，请另存为 UTF-8 编码的 TXT 文件。") from exc
    if not text or "\x00" in text:
        raise ValueError("文件没有可用的文本内容。")
    context = interpret_context(text)
    content = fuse_content(text=context.get("english_polarity"))
    content.update(covered_seconds=None, observed_seconds=None, candidate_seconds=None, interval_count=0)
    content.update(intensity=context.get("english_intensity"), intensity_covered_seconds=None)
    return {
        "kind": "text",
        "filename": path.name,
        "text": text,
        "semantic_context": context,
        "content_summary": content,
        "content_intervals": [],
        "mode": "text_only",
        "scores": None,
        "valid_count": 0,
        "sampled_count": 0,
        "samples": [],
        "reasons": {},
    }


def text_polarity_summary(context):
    result = (context or {}).get("english_polarity")
    if not result:
        return ""
    names = {"positive": "正向", "neutral": "中性", "negative": "负向"}
    return f"英文文本极性：{names[result['label']]}（模型分数 {result['scores'][result['label']]:.1%}）"
