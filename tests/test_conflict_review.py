import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from emotion_assistant import conflict_review as review
from emotion_assistant.content_timeline import iter_content_features, predict_content_interval


def channel(negative, neutral, positive, **rest):
    return {"scores": dict(negative=negative, neutral=neutral, positive=positive), **rest}


def dimensions(valence):
    return {"status": "ok", "dimensions": {"valence": {"score": valence}}}


@pytest.mark.parametrize(("value", "label"), [(4.49, "negative"), (4.5, "neutral"),
                                             (5.5, "neutral"), (5.51, "positive")])
def test_chinese_cue_boundaries_and_missing_mass(value, label):
    ev = review.build_evidence(None, chinese_dimensions=dimensions(value),
                               audio=channel(0.8, 0.1, 0.1, coverage=0.3),
                               visual=channel(0, 0, 0, quality=0))
    assert ev["missing"] == ["visual"] and ev["cues"]["audio"]["coverage"] == .3
    assert ev["baseline_label"] == label and ev["conflict"] == (label != "negative")


def test_chinese_file_and_live_use_identical_cotimed_evidence():
    from emotion_assistant.content_timeline import LiveContentTimeline

    span = {"t_start": 1., "t_end": 3., "segment_id": "seg"}
    tx = {**span, "valid": True, "text": "付出的努力值得了", "semantic_context": {
        "expression_profile": {"language": "zh", "affect_dimensions": dimensions(7)}}}
    av = {**span, "audio_event_id": "a", "per_modality_scores": {
        "visual": channel(.1, .1, .8)["scores"], "audio": channel(.8, .1, .1)["scores"]}}
    file = predict_content_interval(iter_content_features([av], [tx], [av])[0])
    live = LiveContentTimeline()
    live.observe_transcript(tx)
    assert file == live.observe_fusion(av)
    assert list(file["conflict_evidence"]["cues"]) == ["text", "audio", "visual"]
    assert file["conflict_evidence"]["conflict"] and tx["text"] not in repr(live.utterances)
    stale = iter_content_features([av], [dict(tx, stale=True)], [av])[0]
    assert "conflict_evidence" not in stale


def test_quotes_and_requested_modalities_are_validated(monkeypatch):
    evidence = review.build_evidence(None, chinese_dimensions=dimensions(7), audio=channel(.8, .1, .1))
    answer = {"label": "unknown", "quote": "没事", "clarify": "tone"}
    monkeypatch.setattr(review, "request_json", lambda *args: ("ok", answer))
    assert review.review_conflict("我没事", evidence)["question"] == review.QUESTIONS["tone"]
    answer["clarify"] = "expression"
    assert review.review_conflict("我没事", evidence)["status"] == "invalid_response"
    answer.update(clarify="meaning", quote="并不存在")
    assert review.review_conflict("我没事", evidence)["status"] == "invalid_response"
    assert review.review_conflict("我没事", {**evidence, "conflict": False})["status"] == "not_applicable"


def test_late_live_asr_updates_the_selected_segment_without_rewinding_the_readout(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from emotion_assistant.controller import Controller
    from emotion_assistant.ui import MainWindow

    app = QApplication.instance() or QApplication([])
    ctrl = Controller(tmp_path / "late.sqlite")
    monkeypatch.setattr(ctrl, "refresh_devices", lambda: None)
    window = MainWindow(ctrl)
    try:
        window.transcript_consent.setChecked(True)
        ctrl.start_session("late-test", -1, None, {"collection": True, "transcript": True}, "simulation")
        ctrl._simulation_timer.stop()
        sid = ctrl.session_id
        tx_a = {"t_start": 1., "t_end": 2., "segment_id": "a", "event_id": "tx-a", "session_id": sid,
                "valid": True, "text": "真值得", "semantic_context": {"expression_profile": {
                    "language": "zh", "affect_dimensions": dimensions(7)}}}
        tx_b = dict(tx_a, t_start=3., t_end=4., segment_id="b", event_id="tx-b", text="很开心")
        av = {"t_start": 1., "t_end": 2., "segment_id": "a", "session_id": sid, "audio_event_id": "audio-a",
              "per_modality_scores": {"audio": channel(.8, .1, .1)["scores"]}}
        def publish_audio(row):
            ctrl._emit_content(ctrl._content_timeline.observe_fusion(row), row)

        publish_audio(av)
        ctrl._handle_event({"kind": "utterance", "utterance": tx_b})
        assert window._content_end == 4.
        ctrl._handle_event({"kind": "utterance", "utterance": tx_a})
        chooser = window.live_review
        assert chooser.choices.currentData()[0] == "b" and window._content_end == 4.
        chooser.choices.setCurrentIndex(1)
        assert chooser.choices.currentData()[0] == "a"
        assert not chooser.conflict_panel.isHidden()
        record = chooser.conflict_panel._record
        assert record["t_start"] == 1. and record["conflict_segment_id"] == "a"
        assert record["conflict_evidence"]["cues"]["audio"]["label"] == "negative"
        # Duplicate observations must preserve the pinned review. A genuinely revised observation can remove it.
        publish_audio(av)
        assert chooser.conflict_panel._record is record
        publish_audio(dict(av, per_modality_scores={"audio": channel(.1, .1, .8)["scores"]}))
        assert chooser.conflict_panel.isHidden() and window._content_end == 4.
        window._revoke_transcript()
        assert chooser.choices.count() == 0
        publish_audio(av)
        assert chooser.choices.count() == 0 and chooser.conflict_panel.isHidden()
    finally:
        window.close()
        app.processEvents()


def test_panel_drops_late_results_and_exports_only_selected_segment(monkeypatch):
    from PySide6.QtWidgets import QApplication
    from emotion_assistant.local_media_widgets import LocalMediaPage

    app = QApplication.instance() or QApplication([])
    page = LocalMediaPage()
    try:
        ev = review.build_evidence(None, chinese_dimensions=dimensions(7), audio=channel(.8, .1, .1))
        rows = [{"t_start": i, "t_end": i + 1, "conflict_evidence": ev} for i in range(2)]
        page.result = {"kind": "video", "valid_count": 0, "sampled_count": 0, "scores": None, "reasons": {},
                       "transcription_status": "available", "content_summary": {"decision": "abstain"},
                       "content_intervals": rows,
                       "transcripts": [{**r, "text": "没事", "valid": True} for r in rows]}
        page._show_summary()
        panel = page.conflict_panel
        assert panel.isHidden()
        page.content_intervals.setCurrentIndex(1)
        assert not panel.isHidden() and "分歧" in page.content_intervals.currentText()
        ids = iter([1, 2])
        monkeypatch.setattr(panel.queue, "submit", lambda _: next(ids))
        panel.button.click()
        page.content_intervals.setCurrentIndex(2)
        panel.button.click()
        answer = {"status": "ok", "label": "unknown", "quote": "没事", "question": "想表达什么？"}
        panel.queue.finished.emit(1, answer)
        assert not any("conflict_review" in r for r in rows)
        panel.queue.finished.emit(2, answer)
        assert rows[1]["conflict_review"] == answer and "缺少可用分数：表情" in panel.label.text()
        page.content_intervals.setCurrentIndex(0)
        assert panel.isHidden()
    finally:
        page.close()
        app.processEvents()
