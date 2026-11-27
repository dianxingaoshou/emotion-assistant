import os
import json
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from emotion_assistant import clip_recognition, foundation_embeddings, foundation_fusion
from emotion_assistant.clip_inputs import clip_inputs


def observed(start, end, category):
    return {"eligible": True, "t_start": start, "t_end": end, "quality": .8,
            "scores": {k: float(category == k) for k in ("negative", "neutral", "positive", "uncertain")}}


def test_clip_inputs_respect_valid_seconds_and_never_fill_a_missing_modality():
    result = {"timeline_coverage": {"duration_s": 5}, "audio_samples": [
        observed(0, 1, "positive"), observed(1, 3, "negative"), observed(1, 3, "negative"),
        dict(observed(3, 4, "neutral"), eligible=False)], "transcripts": [
            {"t_start": 1., "t_end": 2., "valid": True, "text": " 真好 "},
            {"t_start": 0., "t_end": 1., "valid": True, "text": "今天"},
            {"t_start": 2., "t_end": 3., "valid": True, "text": "过期", "stale": True}]}
    x = clip_inputs(result)
    assert x["visual"] is None and x["text"] == "今天 真好"
    assert x["audio"]["scores"]["negative"] == pytest.approx(2/3)
    assert x["audio"]["coverage"] == pytest.approx(3/5)


def test_file_classifier_is_wired_to_vectors_and_respects_transcription_choice(monkeypatch):
    captured = []
    def predict(text, **kwargs):
        captured.append((text, kwargs))
        return {"status": "ok"}
    monkeypatch.setattr(foundation_fusion, "predict_foundation_clip", predict)
    monkeypatch.setattr(foundation_embeddings, "clip_audio_embedding", lambda *a, **kw: [7])
    monkeypatch.setattr(foundation_embeddings, "embed_text", lambda text: [len(text)])
    result = {"kind": "video", "timeline_coverage": {"duration_s": 2},
              "transcripts": [{"t_start": 0, "t_end": 2, "valid": True, "text": "很开心"}]}
    monkeypatch.setattr(clip_recognition, "analyze_media", lambda *a, **kw: dict(result))
    for consent, expected in [(True, ("很开心", [3])), (False, ("", None))]:
        out = clip_recognition.analyze_foundation_media("clip.mp4", model_dir=None, model_hub=object(),
                on_sample=None, cancelled=lambda: False, transcribe=consent)
        text, inputs = captured[-1]
        assert (text, inputs["embedding"]) == expected
        assert inputs["audio_embedding"] == [7] and inputs["visual"] is None
        assert out["foundation_clip"]["status"] == "ok"
    result["timeline_coverage"]["duration_s"] = 31
    assert clip_recognition.recognize_file_clip("long.mp4", result, model_hub=object(),
            cancelled=lambda: False, transcribe=True)["status"] == "unsupported_duration"
    assert len(captured) == 2


def test_live_fusion_drops_revoked_and_outdated_results(monkeypatch):
    from PySide6.QtWidgets import QApplication
    from emotion_assistant.live_semantic_review import LiveSemanticReview

    app = QApplication.instance() or QApplication([])
    panel = LiveSemanticReview()
    submitted = []
    def submit(payload):
        submitted.append(payload)
        return len(submitted)
    monkeypatch.setattr(panel.foundation_queue, "submit", submit)
    try:
        panel.observe("one", "她很开心", 3.)
        assert not submitted
        panel.foundation.setChecked(True)
        assert submitted[-1] == ("她很开心", None, None, False)
        cue = {"quality": .8, "coverage": .6}
        panel.observe_content({"t_start": 1., "t_end": 3., "conflict_segment_id": "one",
                               "conflict_evidence": {"cues": {"audio": cue}, "conflict": False}})
        assert submitted[-1] == ("她很开心", cue, None, False)
        panel.foundation_queue.finished.emit(1, {"status": "insufficient_evidence"})
        assert "正在识别" in panel.foundation_readout.text()
        panel.clear()
        panel.foundation_queue.finished.emit(2, {"status": "insufficient_evidence"})
        assert not panel.foundation.isChecked() and panel.foundation_readout.isHidden()
    finally:
        panel.close()
        app.processEvents()


def test_file_checkbox_selects_classifier_and_exports_the_displayed_result(monkeypatch, tmp_path):
    from PySide6.QtWidgets import QApplication, QFileDialog
    from emotion_assistant import local_media_widgets

    app = QApplication.instance() or QApplication([])
    page = local_media_widgets.LocalMediaPage()
    pred = {"status": "ok", "label": "positive", "used_modalities": ["text"], "foundation_used": True,
            "audio_embedding_used": False, "scores": {"negative": .1, "neutral": .2, "positive": .7}}
    def analyze(path, **options):
        assert options["transcribe"]
        return {"kind": "video", "mode": "unavailable", "foundation_clip": pred,
                "valid_count": 0, "sampled_count": 0, "scores": None, "reasons": {}}
    monkeypatch.setattr(local_media_widgets, "analyze_foundation_media", analyze)
    output = tmp_path / "result.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **kw: (str(output), ""))
    page.show()
    try:
        page.transcribe.setChecked(True)
        page.foundation.setChecked(True)
        assert page.start_file(tmp_path / "发言.mp4")
        deadline = time.monotonic() + 5
        while page._worker is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.005)
        assert page._worker is None and page.clip_readout.isVisible()
        assert "正向" in page.clip_readout.text() and "千问语义向量" in page.clip_readout.text()
        page.export_button.click()
        assert json.loads(output.read_text(encoding="utf-8"))["foundation_clip"] == pred
    finally:
        page.close()
        app.processEvents()


def test_ui_exposes_a_close_disagreement_between_language_and_voice():
    message = clip_recognition.format_clip_result({
        "status": "ok", "label": "negative", "used_modalities": ["text", "audio"],
        "foundation_used": True, "audio_embedding_used": True,
        "experts": {"text": {"label": "negative"}, "audio_embedding": {"label": "positive"}},
        "scores": {"negative": .4024, "neutral": .1969, "positive": .4007}})
    assert "文字偏负向" in message and "语音向量偏正向" in message and "0.17 个百分点" in message


def test_ui_displays_actual_adaptive_weights_and_recalibrated_expert():
    message = clip_recognition.format_clip_result({
        "version": "chsims-adaptive-v2", "status": "ok", "label": "neutral",
        "used_modalities": ["text", "audio"], "foundation_used": True, "audio_embedding_used": True,
        "experts": {"text": {"label": "negative", "raw_label": "negative", "weight": .25},
                    "audio_embedding": {"label": "neutral", "raw_label": "negative", "weight": .75}},
        "scores": {"negative": .24, "neutral": .56, "positive": .20}})
    assert "文字 25.0%" in message and "语音向量 75.0%" in message
    assert "语音向量 负向→中性" in message
