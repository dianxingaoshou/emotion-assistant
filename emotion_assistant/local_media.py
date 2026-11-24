"""Analyze local media on its presentation timeline with the live fusion policy."""
from __future__ import annotations

from collections import Counter
from contextlib import closing
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from .models import CANONICAL, ModelHub
from .core import FusionEngine, interval_duration, interval_union
from .content_fusion import summarize_content
from .content_timeline import iter_content_features, predict_content_interval
from .file_tracks import audio_windows, media_info, read_speech, video_frames
from .paths import DATA


IMAGE_EXTENSIONS = frozenset({".bmp", ".jfif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
VIDEO_EXTENSIONS = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})
AUDIO_EXTENSIONS = frozenset({".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"})
MAX_VIDEO_SAMPLES = 240
SAMPLE_INTERVAL_S = 0.5
MAX_INFERENCE_EDGE = 1600


class MediaCancelled(Exception):
    """The user interrupted this import."""


def _check_cancelled(cancelled):
    if cancelled():
        raise MediaCancelled("已取消文件检测。")


def _photo_frames(path):
    try:
        # imdecode honors EXIF orientation. numpy opens Unicode Windows paths.
        encoded = np.fromfile(path, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    except (OSError, cv2.error) as exc:
        raise ValueError("无法读取照片，请确认文件完整且可访问。") from exc
    if frame is None:
        raise ValueError("无法解码照片，请确认文件完整，或另存为 JPEG、PNG 后重试。")
    yield frame, 0, 0.0, 1.0



def _analyze_frame(hub, frame, source_id, stamp):
    height, width = frame.shape[:2]
    scale = min(1.0, MAX_INFERENCE_EDGE / max(width, height))
    work = frame
    if scale < 1:
        work = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))),
                          interpolation=cv2.INTER_AREA)
    observation = hub.analyze_frame(work, source_id, stamp, stamp + 0.001)
    work_height, work_width = work.shape[:2]
    observation["inference_size"] = [work_width, work_height]
    observation["source_size"] = [width, height]
    box = observation.get("face_box")
    if scale < 1 and box:
        x, y, w, h = box
        left, top = round(x * width / work_width), round(y * height / work_height)
        right, bottom = round((x + w) * width / work_width), round((y + h) * height / work_height)
        observation["face_box"] = [left, top, right - left, bottom - top]
    return observation


def _visual_pass(frames, hub, source_id, on_sample, check, min_quality, observations):
    with closing(frames):
        for frame, frame_index, stamp, progress in frames:
            check()
            obs = _analyze_frame(hub, frame, source_id, stamp)
            check()
            reason = obs["reason"]
            if reason.startswith(("inference_error", "model_unavailable")):
                raise ValueError(f"本地表情模型未能完成检测。 {reason}")
            eligible = bool(obs["valid"] and obs["quality"] >= min_quality and obs.get("scores"))
            obs.update(sample_index=len(observations) + 1, frame_index=frame_index, sample_time_s=stamp,
                       eligible=eligible, rejection_reason=None if eligible else
                       ("low_visual_quality" if obs["valid"] else reason))
            observations.append(obs)
            on_sample(frame, obs, progress)


def _audio_pass(path, track, origin, hub, source_id, model_dir, check, transcribe, issues, min_quality):
    import av
    audio, transcripts = [], []
    if track is None:
        return audio, transcripts, "no_audio_track", []
    windows = audio_windows(track, origin)
    acoustic_ready = asr_ready = None
    try:
        for samples, start, end, metrics in read_speech(path, track, origin, model_dir, check, windows):
            check()
            if end - start >= 2 and acoustic_ready is None:
                state = hub.load_audio_emotion()["audio"]
                acoustic_ready = state["ready"]
                if not acoustic_ready:
                    issues.append(f"声学模型未就绪：{state['reason']}")
            obs = hub.analyze_audio_emotion(samples, source_id, start, end, metrics)
            obs["eligible"] = bool(obs["valid"] and obs["quality"] >= min_quality and obs.get("scores"))
            obs["rejection_reason"] = None if obs["eligible"] else (
                "low_audio_quality" if obs["valid"] else obs["reason"])
            audio.append(obs)
            if transcribe:
                if asr_ready is None:
                    state = hub.load_asr()["asr"]
                    asr_ready = state["ready"]
                    if not asr_ready:
                        issues.append(f"转写模型未就绪：{state['reason']}")
                check()
                if asr_ready:
                    transcript = hub.transcribe_audio(samples, source_id, start, end, asr_allowed=True)
                    obs["segment_id"] = obs["event_id"]
                    transcript["segment_id"] = obs["segment_id"]
                    transcripts.append(transcript)
    except MediaCancelled:
        raise
    except (av.FFmpegError, OSError, ValueError, RuntimeError) as exc:
        issues.append(f"音轨处理不完整，保留可用证据：{exc}")
        return audio, transcripts, "unavailable", windows
    if acoustic_ready is False:
        status = "unavailable"
    elif any(s["eligible"] for s in audio):
        status = "ok"
    else:
        status = "unusable" if audio else "no_speech"
    return audio, transcripts, status, windows


def _uncovered(start, end, occupied):
    for left, right in occupied:
        if right <= start:
            continue
        if left >= end:
            break
        if left > start:
            yield start, left
        start = max(start, right)
    if start < end:
        yield start, end


def _file_intervals(engine, visual, audio, can_align):
    # Only co-timed evidence may contribute to the same prediction. Fill the
    # remaining sampled visual bins once, never counting spoken time twice.
    good_audio = [obs for obs in audio if obs["eligible"]]
    spans = interval_union((obs["t_start"], obs["t_end"]) for obs in good_audio)
    events = [(obs["t_start"], obs["t_end"], obs) for obs in good_audio]
    for obs in visual:
        for start, end in _uncovered(obs["t_start"], obs["t_end"], spans):
            events.append((start, end, None))
    # Preserve abstentions for audio-only files where every speech segment failed.
    if not events:
        events = [(obs["t_start"], obs["t_end"], obs) for obs in audio]
    intervals = []
    cursor = 0
    for start, end, acoustic in sorted(events, key=lambda event: (event[0], event[1])):
        while cursor < len(visual) and visual[cursor]["t_start"] < end:
            engine.add_visual(visual[cursor])
            cursor += 1
        if acoustic is not None:
            if not can_align:
                # Keep observed scores for this channel, but prevent temporal
                # fusion when the container had no common timestamp origin.
                acoustic = dict(acoustic, session_id=acoustic["session_id"] + ":unlinked")
            row = engine.fuse_audio(acoustic)
        else:
            row = engine.visual_summary(start, end)
        intervals.append(row)
    return intervals


def _mean_scores(items, duration_weighted=False):
    total = 0.
    mass = dict.fromkeys(CANONICAL, 0.)
    for row in items:
        if not row.get("scores"):
            continue
        weight = row["t_end"] - row["t_start"] if duration_weighted else 1.
        total += weight
        for key in mass:
            mass[key] += weight * row["scores"][key]
    return {k: v / total for k, v in mass.items()} if total else None


def analyze_media(path, *, model_dir, on_sample, cancelled, model_hub=None, transcribe=False,
                  strategy=None, on_status=None):
    """Decode on-device. Scores use disjoint scored intervals, not whole-track votes."""
    def check():
        _check_cancelled(cancelled)

    check()
    path = Path(path)
    if not path.is_file():
        raise ValueError("找不到所选文件，请重新选择。")
    suffix = path.suffix.lower()
    if suffix == ".txt":
        from .text_file import analyze_text_file
        return analyze_text_file(path)
    if suffix not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS:
        raise ValueError("暂不支持此文件类型，请选择支持的照片、音频或视频。")
    kind = "image" if suffix in IMAGE_EXTENSIONS else "audio" if suffix in AUDIO_EXTENSIONS else "video"
    hub = model_hub or ModelHub(model_dir, enable_asr=transcribe)
    engine = FusionEngine.from_project(DATA, strategy=strategy)
    issues, visual = [], []
    source_id = str(uuid4())
    tracks, origin, limits = {}, 0., {}
    if kind != "image":
        try:
            tracks, origin = media_info(path, kind == "video")
        except (OSError, ValueError) as exc:
            raise ValueError(f"无法读取所选媒体文件：{exc}") from exc
    if kind == "image" or "video" in tracks:
        try:
            status = hub.load_visual()
            missing = [s["reason"] for s in status.values() if not s["ready"]]
            if missing:
                raise ValueError("本地视觉模型未就绪：" + " / ".join(missing))
            frames = (_photo_frames(path) if kind == "image" else video_frames(
                path, tracks["video"], origin, check, limits, MAX_VIDEO_SAMPLES, SAMPLE_INTERVAL_S))
            _visual_pass(frames, hub, source_id, on_sample, check, engine.min_quality, visual)
        except (OSError, ValueError, RuntimeError) as exc:
            if not visual and "audio" not in tracks:
                raise
            issues.append(f"画面处理中断，保留前面 {len(visual)} 帧：{exc}" if visual else f"视觉通道不可用：{exc}")
    for i, obs in enumerate(visual):
        if kind == "video":
            next_frame = visual[i+1]["t_start"] if i+1 < len(visual) else limits["end"]
            obs["t_end"] = min(obs["t_start"] + SAMPLE_INTERVAL_S, next_frame)
    audio, transcripts, audio_status, windows = [], [], "no_audio_track", []
    if "audio" in tracks:
        if on_status:
            on_status("正在识别声音并按时间对齐画面…")
        audio, transcripts, audio_status, windows = _audio_pass(
            path, tracks["audio"], origin, hub, source_id, model_dir, check, transcribe, issues, engine.min_quality)
    can_align = all(t.start is not None for t in tracks.values())
    if not can_align and len(tracks) > 1:
        issues.append("轨道缺少共同时间戳，重叠区间只使用有效声音。")
    check()
    intervals = _file_intervals(engine, visual, audio, can_align)
    content = [predict_content_interval(row) for row in iter_content_features(
        intervals, transcripts, audio, can_align=can_align)]
    statuses = {r["status"] for r in intervals if r.get("scores")}
    mode = "mixed" if len(statuses) > 1 else next(iter(statuses), "unavailable")
    used = sorted({m for r in intervals for m in r["modalities"]})
    valid_visual = [obs for obs in visual if obs["eligible"]]
    result = {"kind": kind, "filename": path.name, "sampled_count": len(visual),
            "valid_count": len(valid_visual), "samples": visual,
            "reasons": dict(Counter(o["rejection_reason"] for o in visual if not o["eligible"])),
            "visual_scores": _mean_scores(valid_visual), "scores": _mean_scores(intervals, True),
            "mode": mode, "used_modalities": used, "fusion_intervals": intervals,
            "content_intervals": content, "content_summary": summarize_content(content),
            "decision_counts": dict(Counter(r["decision"] for r in intervals)),
            "audio_status": audio_status, "audio_samples": audio, "audio_valid_count": sum(s["eligible"] for s in audio),
            "audio_sampling_windows_s": windows, "issues": issues, "transcripts": transcripts}
    if kind == "image":
        # A still image has no measured time support.
        result["content_intervals"] = []
        result["content_summary"].update(covered_seconds=None, observed_seconds=None, candidate_seconds=None,
                                         intensity_covered_seconds=None, interval_count=0)
    if kind != "image":
        scored_seconds = interval_duration((r["t_start"], r["t_end"]) for r in intervals if r.get("scores"))
        accepted_seconds = interval_duration((r["t_start"], r["t_end"]) for r in intervals
                                             if r.get("scores") and r["decision"] != "abstain")
        ends = []
        for name, track in tracks.items():
            if name == "video" and visual and limits.get("end") is not None:
                ends.append(limits.get("declared_end", limits["end"]))
            elif track.start is not None and track.duration is not None:
                ends.append(track.start - origin + track.duration)
        duration = max(ends) if len(ends) == len(tracks) and can_align else None
        result["timeline_coverage"] = {"duration_s": duration, "scored_s": scored_seconds,
                                      "accepted_s": accepted_seconds}
    if transcribe and kind != "image":
        from .english_polarity import predict_english_polarity
        utterances = [row["text"] for row in transcripts if row.get("valid") and row.get("text")]
        result["transcription_status"] = "available" if utterances else "no_usable_transcript"
        if utterances:
            polarity = predict_english_polarity(" ".join(utterances))
            if polarity:
                result["semantic_context"] = {"english_polarity": polarity}
    return result
