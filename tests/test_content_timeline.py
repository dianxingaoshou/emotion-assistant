import json
import os

import pytest

from emotion_assistant.content_timeline import (
    LiveContentTimeline,
    iter_content_features,
    predict_content_interval,
)


def observation(modality, start, end):
    return {
        "t_start": start,
        "t_end": end,
        "source_ids": [f"{modality}{start}"],
        "per_modality_scores": {
            modality: {"negative": 0.1, "positive": 0.65, "neutral": 0.2, "uncertain": 0.05}
        },
        "per_modality_coverage": {modality: 1.0},
        "per_modality_quality": {modality: 0.7},
        "status": modality + "_only",
    }


def transcript(start, end, key="words"):
    return {
        "t_start": start,
        "t_end": end,
        "event_id": key,
        "segment_id": "seg",
        "text": "Sensitive fixture opinion.",
        "valid": True,
        "semantic_context": {
            "english_polarity": {
                "label": "positive",
                "scores": {"positive": 0.82, "negative": 0.06, "neutral": 0.12},
            }
        },
    }


def test_a_transcript_and_two_visual_bins_are_one_utterance_without_cross_gap_lookahead():
    visual = [observation("visual", 0.0, 1.0), observation("visual", 2.0, 3.0)]
    speech = transcript(0.5, 2.5)
    rows = iter_content_features(visual, [speech], [speech])
    assert [(r["t_start"], r["t_end"]) for r in rows] == [(0.0, 0.5), (0.5, 2.5), (2.5, 3.0)]
    assert [bool(r["text"]) for r in rows] == [False, True, False]
    assert rows[1]["visual"]["coverage"] == 0.5 and rows[1]["audio"] is None
    assert rows[1]["source_ids"] == ["visual0.0", "visual2.0", "words"]
    assert "Sensitive" not in str(rows)
    unlinked = iter_content_features(visual, [speech], [speech], can_align=False)
    assert unlinked[1]["visual"] is None and unlinked[1]["text"]


@pytest.mark.parametrize("audio_first", [True, False])
def test_live_revisions_match_file_inputs_and_keep_segments_separate(audio_first):
    audio = observation("audio", 1.0, 3.0)
    audio.update(audio_event_id="audio1", segment_id="seg")
    audio["per_modality_scores"]["visual"] = observation("visual", 1.0, 3.0)["per_modality_scores"]["visual"]
    audio["per_modality_quality"]["visual"] = 0.6
    text = transcript(1.0, 3.0)
    baseline = predict_content_interval(iter_content_features([audio], [text], [audio])[0])
    live = LiveContentTimeline()
    if audio_first:
        live.observe_fusion(audio)
        joined = live.observe_transcript(text)
    else:
        live.observe_transcript(text)
        joined = live.observe_fusion(audio)
    assert joined == baseline
    assert set(joined["available_modalities"]) == {"audio", "visual", "text"}
    assert live.observe_transcript(text) is None
    assert live.observe_transcript(dict(text, segment_id="wrong")) is None
    assert live.last_display == baseline
    other = dict(audio, t_start=3.0, t_end=5.0, segment_id="next", audio_event_id="audio2")
    latest = live.observe_fusion(other)
    assert latest["t_start"] == 3.0 and "text" not in latest["available_modalities"]
    assert live.observe_transcript(text) is None
    late = dict(text, t_start=3.0, t_end=5.0, segment_id="wrong")
    assert live.observe_transcript(late) is None
    live.reset(minimum_start=6.0)
    assert live.observe_transcript(text) is None
    assert "Sensitive" not in repr(live.utterances)


def test_overlapping_speech_is_abstained_and_does_not_double_count_time():
    from emotion_assistant.content_fusion import summarize_content

    first, second = transcript(1, 3), transcript(2, 4)
    third = transcript(4, 5)
    rows = iter_content_features([], [first, second, third], [first, second, third])
    results = [predict_content_interval(row) for row in rows]
    assert results[0]["timing_issue"] == "overlapping_speech"
    assert results[0]["available_modalities"] == []
    assert results[1]["used_modalities"] == ["text"]
    summary = summarize_content(results)
    assert (summary["observed_seconds"], summary["covered_seconds"]) == (4.0, 1.0)


def test_intensity_uses_its_own_scored_duration_and_matches_live_without_polarity():
    from emotion_assistant.content_fusion import summarize_content

    speech = [transcript(0., 1., "a"), transcript(1., 4., "b"), transcript(5., 6., "c")]
    visual = observation("visual", 4., 5.)
    for utterance, value in zip(speech, (0., -2., float("nan")), strict=True):
        utterance["semantic_context"] = {"english_intensity": {
            "value": value, "model_version": "fixture", "used_modalities": ["text"]}}
    rows = [predict_content_interval(row) for row in iter_content_features([visual], speech)]
    live = LiveContentTimeline()
    for index, utterance in zip((0, 1, 3), speech, strict=True):
        assert live.observe_transcript(utterance) == rows[index]
    assert all(row["decision"] == "abstain" for row in rows)
    assert rows[0]["intensity"]["value"] == 0. and rows[0]["source_ids"] == ["a"]
    assert rows[2]["intensity"] is None and rows[3]["intensity"] is None
    summary = summarize_content(rows)
    assert summary["covered_seconds"] == 0. and summary["intensity_covered_seconds"] == 4.
    assert summary["intensity"]["value"] == pytest.approx(-1.5)
    assert summary["observed_seconds"] == 6.
    assert "Sensitive" not in repr(live.utterances)


def test_controller_keeps_content_ephemeral_and_revoke_clears_derived_scores(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from emotion_assistant.controller import Controller, simulation_observation

    app = QApplication.instance() or QApplication([])
    ctrl = Controller(tmp_path / "isolated.sqlite")
    events, cue_events = [], []
    ctrl.content_ready.connect(events.append)
    ctrl.fusion_ready.connect(cue_events.append)
    try:
        ctrl.start_session("fixture", -1, None, {"collection": True, "transcript": True}, "simulation")
        ctrl._simulation_timer.stop()
        sid = ctrl.session_id
        obs = simulation_observation(sid, "visual", 1.0, 3.0, 0.2)
        ctrl._handle_event({"kind": "observation", "observation": obs})
        tx = dict(transcript(1.0, 3.0), session_id=sid)
        tx["semantic_context"]["english_intensity"] = {
            "value": 0.0, "model_version": "fixture", "used_modalities": ["text"]}
        ctrl._handle_event({"kind": "utterance", "utterance": tx})
        ac = simulation_observation(sid, "audio", 1.0, 3.0, 0.2)
        ac["segment_id"] = "seg"
        ctrl._handle_event({"kind": "observation", "observation": ac})
        assert set(events[-1]["available_modalities"]) == {"visual", "audio", "text"}
        assert events[-1]["intensity"]["value"] == 0.
        assert events[-1]["session_id"] == sid and "Sensitive" not in str(events)
        assert all("content_summary" not in row and "candidate_scores" not in row
                   and "intensity" not in row for row in cue_events)
        # Queue behind the accepted write before checking what the authorized export contains.
        authorized = ctrl.writer.submit(ctrl.store.export_session, sid, tmp_path / "authorized.json").result(5)
        saved = json.loads(authorized.read_text("utf-8"))
        assert saved["utterances"][0]["semantic_context"]["english_intensity"]["value"] == 0.
        ctrl.pause_session(True)
        assert events[-1] == {}
        ctrl._handle_event({"kind": "utterance", "utterance": tx})
        assert events[-1] == {}
        ctrl.pause_session(False)
        ctrl.set_transcript_allowed(False).result(timeout=5)
        ctrl._handle_event({"kind": "utterance", "utterance": tx})
        assert events[-1] == {} and not ctrl._content_timeline.utterances
        history = json.loads(ctrl.store.export_session(sid, tmp_path / "result.json").read_text("utf-8"))
        assert "candidate_scores" not in str(history) and "english_polarity" not in str(history)
        assert "english_intensity" not in str(history)
        assert not ctrl.transcript_allowed
    finally:
        ctrl.shutdown()
        app.processEvents()
