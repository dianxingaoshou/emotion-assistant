"""Decode real timestamped containers and exercise file fusion without neural weights."""
from fractions import Fraction
import json
import wave

import av
import numpy as np
import pytest

from emotion_assistant import file_tracks
from emotion_assistant.local_media import analyze_media
from emotion_assistant.models import ModelHub


class FixtureHub:
    def __init__(self, visual=True, acoustic=True):
        self.visual = visual
        self.acoustic = acoustic
        self.asr_loads = 0

    def load_visual(self):
        return {"visual": {"ready": self.visual, "reason": "fixture missing visual weights"}}

    def load_audio_emotion(self):
        return {"audio": {"ready": self.acoustic, "reason": "fixture missing acoustic weights"}}

    def load_asr(self):
        self.asr_loads += 1
        return {"asr": {"ready": True}}

    def transcribe_audio(self, samples, sid, start, end, asr_allowed=False):
        assert asr_allowed
        return {"text": "fixture", "valid": True, "t_start": start, "t_end": end}

    def observation(self, modality, sid, start, end):
        obs = ModelHub._observation(modality, sid, start, end)
        obs.update(model_version="fixture", quality=.8, valid=True, reason="ok",
                   scores={"positive": .8, "negative": .1, "neutral": .1, "uncertain": 0.})
        return obs

    def analyze_frame(self, frame, sid, start, end):
        return self.observation("visual", sid, start, end)

    def analyze_audio_emotion(self, samples, sid, start, end, metrics):
        assert len(samples) / 16000 == pytest.approx(end - start, abs=1/16000)
        obs = self.observation("audio", sid, start, end)
        obs["quality_metrics"] = metrics
        if not self.acoustic:
            obs.update(valid=False, scores=None, reason="model_unavailable: fixture")
        return obs


class SpeechDetector:
    def __init__(self, _path):
        pass

    def reset(self):
        pass

    def __call__(self, block):
        return float(np.max(np.abs(block)) > .01)


def make_av(path):
    """Six visual seconds and three speech seconds starting at PTS +2."""
    with av.open(str(path), "w") as out:
        video = out.add_stream("ffv1", rate=4)
        video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
        audio = out.add_stream("pcm_s16le", rate=16000)
        audio.layout = "mono"
        for index in range(24):
            frame = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), np.uint8), format="bgr24")
            frame.pts, frame.time_base = index, Fraction(1, 4)
            out.mux(video.encode(frame))
            if 8 <= index < 20:
                sound = av.AudioFrame.from_ndarray(np.full((1, 4000), 4000, np.int16), format="s16", layout="mono")
                sound.pts, sound.time_base, sound.sample_rate = index * 4000, Fraction(1, 16000), 16000
                out.mux(audio.encode(sound))
        out.mux(video.encode(None))
        out.mux(audio.encode(None))


def run(path, hub, **kwargs):
    return analyze_media(path, model_dir=path.parent, model_hub=hub, on_sample=lambda *_: None,
                         cancelled=lambda: False, **kwargs)


def test_video_joins_only_overlapping_stream_pts_and_counts_time_once(tmp_path, monkeypatch):
    monkeypatch.setattr(file_tracks, "SileroVAD", SpeechDetector)
    path = tmp_path / "音轨延迟.mkv"
    make_av(path)
    hub = FixtureHub()
    result = run(path, hub)
    assert result["mode"] == "mixed" and result["audio_valid_count"] == 1
    rows = result["fusion_intervals"]
    joint = [r for r in rows if r["status"] == "joint"]
    assert len(joint) == 1
    assert (joint[0]["t_start"], joint[0]["t_end"]) == pytest.approx((2., 5.))
    assert sum(r["t_end"]-r["t_start"] for r in rows) == pytest.approx(6.)
    assert result["timeline_coverage"] == pytest.approx({"duration_s": 6., "scored_s": 6., "accepted_s": 6.})
    assert all(a["t_end"] <= b["t_start"] + 1e-6 for a, b in zip(rows, rows[1:]))
    assert hub.asr_loads == 0 and result["transcripts"] == []
    json.dumps(result, allow_nan=False)


def test_missing_visual_model_keeps_video_speech(tmp_path, monkeypatch):
    monkeypatch.setattr(file_tracks, "SileroVAD", SpeechDetector)
    path = tmp_path / "without-visual.mkv"
    make_av(path)
    result = run(path, FixtureHub(visual=False))
    assert result["used_modalities"] == ["audio"] and result["scores"]
    assert result["timeline_coverage"]["scored_s"] == pytest.approx(3.)
    assert result["audio_status"] == "ok" and result["issues"]


def test_file_expression_uses_only_authorized_overlapping_utterance(tmp_path, monkeypatch):
    monkeypatch.setattr(file_tracks, "SileroVAD", SpeechDetector)
    path = tmp_path / "delayed-utterance.mkv"
    make_av(path)
    hub = FixtureHub()

    def transcribe(samples, sid, start, end, asr_allowed):
        assert asr_allowed
        return {"text": "A good film.", "t_start": start, "t_end": end, "valid": True,
                "semantic_context": {
                    "english_polarity": {"scores": {"positive": .8, "neutral": .1, "negative": .1}},
                    "english_intensity": {"value": .65, "model_version": "fixture", "used_modalities": ["text"]},
                }}

    monkeypatch.setattr(hub, "transcribe_audio", transcribe)
    denied = run(path, hub, transcribe=False)
    assert hub.asr_loads == 0
    assert denied["content_summary"]["decision"] == "abstain"
    assert denied["content_summary"]["intensity"] is None
    assert "text" not in denied["content_summary"]["available_modalities"]
    result = run(path, hub, transcribe=True)
    rows = result["content_intervals"]
    spoken = [row for row in rows if "text" in row["available_modalities"]]
    assert len(spoken) == 1
    assert (spoken[0]["t_start"], spoken[0]["t_end"]) == pytest.approx((2., 5.))
    assert set(spoken[0]["available_modalities"]) == {"visual", "text", "audio"}
    assert result["content_summary"]["covered_seconds"] == pytest.approx(3.)
    assert result["content_summary"]["intensity_covered_seconds"] == pytest.approx(3.)
    assert result["content_summary"]["intensity"]["value"] == pytest.approx(.65)
    assert [r["intensity"]["value"] for r in rows if r["intensity"] is not None] == [.65]
    assert result["content_summary"]["observed_seconds"] == pytest.approx(6.)
    assert denied["scores"] == result["scores"]


def test_default_file_english_intensity_uses_same_transcript_and_keeps_segment_audit(tmp_path, monkeypatch):
    from emotion_assistant import file_asr_intensity
    from emotion_assistant.english_intensity import predict_english_intensity
    from emotion_assistant.local_media import MediaCancelled

    monkeypatch.setattr(file_tracks, "SileroVAD", SpeechDetector)
    path = tmp_path / "english-audio.mkv"
    make_av(path)
    hub = FixtureHub()
    text = "This is a wonderful film"
    segment_score = predict_english_intensity(text)
    calls = []

    def transcribe(samples, sid, start, end, asr_allowed):
        assert asr_allowed is True
        calls.append(sid)
        return {"text": text, "session_id": sid, "valid": True, "asr_language": "en",
                "t_start": start, "t_end": end, "semantic_context": {"english_intensity": segment_score}}

    monkeypatch.setattr(hub, "transcribe_audio", transcribe)
    legacy = run(path, hub, transcribe=True, refine_file_intensity=False)
    current = run(path, hub, transcribe=True)
    assert len(calls) == 2
    assert legacy["content_summary"]["intensity"] == segment_score
    summary = current["content_summary"]
    expected = file_asr_intensity.estimate_file_intensity(current["transcripts"], source_id=calls[-1], duration=3.)
    assert summary["intensity"] == expected["intensity"]
    assert summary["intensity"]["method"] == "asr_word_character_pool" and "model_version" in summary["intensity"]
    assert summary["intensity_covered_seconds"] == expected["covered_seconds"] == pytest.approx(3.)
    assert summary["segment_weighted_intensity"] == legacy["content_summary"]["intensity"]
    assert [r["intensity"] for r in current["content_intervals"]] == [r["intensity"] for r in legacy["content_intervals"]]
    assert current["scores"] == legacy["scores"]
    forbidden = run(path, hub, transcribe="no")
    assert len(calls) == 2 and forbidden["transcripts"] == [] and forbidden["content_summary"]["intensity"] is None

    finished = []
    def cancel_during_estimate(*args, **kwargs):
        finished.append(True)
        return expected
    monkeypatch.setattr(file_asr_intensity, "estimate_file_intensity", cancel_during_estimate)
    with pytest.raises(MediaCancelled):
        analyze_media(path, model_dir=path.parent, model_hub=hub, transcribe=True,
                      on_sample=lambda *_: None, cancelled=lambda: bool(finished))


def test_interrupted_visual_stream_keeps_observed_frames(tmp_path, monkeypatch):
    from emotion_assistant import local_media
    path = tmp_path / "部分可用.mkv"
    make_av(path)
    def partial_frames(path, track, origin, check, limits, max_samples, interval):
        limits["end"] = .5
        yield np.zeros((48, 64, 3), dtype=np.uint8), 0, 0., 0.
        raise ValueError("decode interrupted")
    monkeypatch.setattr(local_media, "video_frames", partial_frames)
    monkeypatch.setattr(file_tracks, "SileroVAD", SpeechDetector)
    result = run(path, FixtureHub())
    assert result["valid_count"] == 1 and "visual" in result["used_modalities"]
    assert any("保留前面 1 帧" in issue for issue in result["issues"])


def test_vad_load_failure_keeps_video_picture(tmp_path, monkeypatch):
    def missing(_):
        raise RuntimeError("missing")
    monkeypatch.setattr(file_tracks, "SileroVAD", missing)
    path = tmp_path / "without-vad.mkv"
    make_av(path)
    result = run(path, FixtureHub())
    assert result["used_modalities"] == ["visual"] and result["scores"]
    assert result["audio_status"] == "unavailable" and result["issues"]


def test_variable_video_clock_follows_presentation_pts(tmp_path):
    path = tmp_path / "varying-pts.mkv"
    stamps_ms = [0, 100, 520, 700, 1000, 1730, 2200]
    with av.open(str(path), "w") as out:
        stream = out.add_stream("ffv1", rate=1000)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for ms in stamps_ms:
            image = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), np.uint8), format="bgr24")
            image.pts, image.time_base = ms, Fraction(1, 1000)
            out.mux(stream.encode(image))
        out.mux(stream.encode(None))
    result = run(path, FixtureHub())
    assert [s["sample_time_s"] for s in result["samples"]] == pytest.approx([0., .52, 1., 1.73, 2.2])
    assert all(s["t_end"] - s["t_start"] <= .5000001 for s in result["samples"])


def test_long_file_retains_samples_on_both_sides_of_a_short_sampling_gap(tmp_path, monkeypatch):
    # Every 10-second window has a short gap before the next; typical decoded
    # blocks straddle both. The synthetic PCM value encodes its source index.
    rate = file_tracks.RATE
    track = file_tracks.Track(0, 0., 120.1)
    windows = file_tracks.audio_windows(track, 0.)
    count = round(track.duration * rate)

    def pcm(*_):
        for start in range(0, count, 4096):
            yield start / rate, np.arange(start, min(start + 4096, count), dtype=np.float32)

    monkeypatch.setattr(file_tracks, "SileroVAD", SpeechDetector)
    monkeypatch.setattr(file_tracks, "pcm_parts", pcm)
    pieces = file_tracks.read_speech(tmp_path / "fixture.wav", track, 0., tmp_path, lambda: None, windows)
    actual = np.concatenate([samples for samples, *_ in pieces])
    expected = np.concatenate([np.arange(round(start * rate), round(end * rate), dtype=np.float32)
                               for start, end in windows])
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("value", [0, 4000])
def test_wav_without_visual_only_transcribes_when_requested(tmp_path, monkeypatch, value):
    monkeypatch.setattr(file_tracks, "SileroVAD", SpeechDetector)
    path = tmp_path / "声音.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", ""))
        wav.writeframes(np.full(32123, value, dtype="<i2").tobytes())
    hub = FixtureHub(acoustic=False)
    result = run(path, hub, transcribe=True)
    assert result["scores"] is None and result["mode"] == "unavailable"
    assert result["audio_status"] == ("no_speech" if value == 0 else "unavailable")
    assert len(result["transcripts"]) == int(bool(value))
    if value:
        assert result["transcripts"][0]["t_end"] == pytest.approx(32123/16000, abs=1/16000)
