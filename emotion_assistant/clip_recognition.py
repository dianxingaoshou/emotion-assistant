"""Chinese short-clip polarity, used as a prediction rather than a generated review."""

from .chinese_affect import HAN
from .clip_inputs import clip_inputs
from .local_media import MediaCancelled, analyze_media
from .models import ModelHub
from .semantic_review import POLARITY


def analyze_foundation_media(path, *, model_dir, on_sample, cancelled, transcribe=False,
                            strategy=None, on_status=None, model_hub=None, adaptive=False):
    hub = model_hub or ModelHub(model_dir, enable_asr=transcribe)
    result = analyze_media(path, model_dir=model_dir, on_sample=on_sample, cancelled=cancelled,
                           transcribe=transcribe, strategy=strategy, on_status=on_status, model_hub=hub)
    if result["kind"] in ("video", "audio"):
        if on_status:
            on_status("正在用本地融合模型识别中文片段表达…")
        result["foundation_clip"] = recognize_file_clip(path, result, model_hub=hub,
                                                        cancelled=cancelled, transcribe=transcribe, adaptive=adaptive)
    return result


def recognize_file_clip(path, result, *, model_hub, cancelled, transcribe, adaptive=False):
    from .foundation_embeddings import clip_audio_embedding

    duration = (result.get("timeline_coverage") or {}).get("duration_s")
    if duration is None or not 0 < duration <= 30:
        return {"status": "unsupported_duration", "label": None, "used_modalities": []}
    inputs = clip_inputs(result)
    text = inputs["text"] if transcribe else ""
    if cancelled():
        raise MediaCancelled("已取消文件检测。")
    try:
        speech_vector = clip_audio_embedding(path, model_hub=model_hub, cancelled=cancelled)
    except InterruptedError as exc:
        raise MediaCancelled("已取消文件检测。") from exc
    if cancelled():
        raise MediaCancelled("已取消文件检测。")
    prediction = recognize_cues(text, audio=inputs["audio"], visual=inputs["visual"],
                                speech_vector=speech_vector, adaptive=adaptive)
    if cancelled():
        raise MediaCancelled("已取消文件检测。")
    return prediction


def recognize_cues(text, *, audio=None, visual=None, speech_vector=None, adaptive=False):
    from .foundation_embeddings import embed_text
    from .foundation_fusion import predict_foundation_clip

    text_vector = embed_text(text) if text and HAN.search(text) else None
    return predict_foundation_clip(text, audio=audio, visual=visual,
                                   embedding=text_vector, audio_embedding=speech_vector, adaptive=adaptive)


def format_clip_result(result):
    if not result:
        return ""
    if result["status"] == "unsupported_duration":
        return "中文短片融合适用于时长明确的 30 秒以内片段。"
    if result["status"] == "head_unavailable":
        return "中文短片融合分类器未能加载，请检查项目自带的模型文件。"
    if result["status"] == "service_error":
        return "本次中文融合识别未能完成。"
    label = POLARITY.get(result.get("label"))
    if not label or result.get("label") == "unknown":
        return "中文短片表达倾向　暂缓判断：缺少可用的输入线索。"
    names = {"text": "文字", "audio": "声音", "visual": "表情"}
    sources = "、".join(names.get(key, key) for key in result["used_modalities"])
    details = []
    if "text" in result["used_modalities"]:
        details.append("千问语义向量" if result["foundation_used"] else "本地字符特征")
    if result["audio_embedding_used"]:
        details.append("预训练语音向量")
    method = " · ".join(details) or "可用声画分数"
    scores = result["scores"]
    distribution = "　".join(f"{POLARITY[k]} {scores[k]:.1%}" for k in ("negative", "neutral", "positive"))
    lines = [f"中文短片表达倾向　{label}", f"模型分数　{distribution}", f"参与识别：{sources}　|　{method}"]
    experts = result.get("experts", {})
    titles = {"text": "文字", "audio_embedding": "语音向量", "audio_scores": "声学分数", "visual_scores": "表情"}
    if experts and result.get("version") == "chsims-adaptive-v2":
        lines.append("本段融合权重　" + "　".join(f"{titles[k]} {e['weight']:.1%}" for k, e in experts.items()))
        changes = [f"{titles[k]} {POLARITY[e['raw_label']]}→{POLARITY[e['label']]}"
                   for k, e in experts.items() if e.get("raw_label") in POLARITY and e["raw_label"] != e["label"]]
        if changes:
            lines.append("类别校准　" + "　".join(changes))
    if len({e["label"] for e in experts.values()}) > 1:
        votes = "、".join(f"{titles[k]}偏{POLARITY[e['label']]}" for k, e in experts.items())
        first, second, _ = sorted(scores.values(), reverse=True)
        lines.append(f"线索有分歧：{votes}。前两类相差 {(first - second) * 100:.2f} 个百分点。")
    return "\n".join(lines)
