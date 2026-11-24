"""Stream local media with a shared, presentation-timestamp-based file clock."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .capture import SileroVAD, SpeechSegmenter


RATE = 16000
MAX_AUDIO_SECONDS = 120
LONG_FILE_WINDOWS = 12


def open_media(path):
    import av
    return av.open(str(path), options={"protocol_whitelist": "file"})


@dataclass(frozen=True)
class Track:
    index: int
    start: float | None
    duration: float | None


def media_info(path, want_video):
    import av
    tracks = {}
    with open_media(path) as source:
        container_duration = source.duration / av.time_base if source.duration else None
        container_start = source.start_time / av.time_base if source.start_time is not None else None
        for name in (("video", "audio") if want_video else ("audio",)):
            streams = getattr(source.streams, name)
            if not streams:
                continue
            stream = streams[0]
            start = float(stream.start_time * stream.time_base) if stream.start_time is not None else None
            if start is None:
                # Probe independent readers so checking a stream cannot consume another.
                try:
                    with open_media(path) as probe:
                        first = next(probe.decode(probe.streams[stream.index]), None)
                        if first is not None:
                            start = first.time
                except av.FFmpegError:
                    pass
            duration = float(stream.duration * stream.time_base) if stream.duration else None
            if not duration and container_duration and container_start is not None and start is not None:
                duration = max(0.0, container_start + container_duration - start)
            tracks[name] = Track(stream.index, start, duration)
    if not tracks:
        raise ValueError("文件中没有支持的音轨或画面。")
    origin = min((t.start for t in tracks.values() if t.start is not None), default=0.)
    return tracks, origin


def video_frames(path, track, origin, check, limits, max_samples, period):
    with open_media(path) as media:
        stream = media.streams[track.index]
        rate = float(stream.average_rate or stream.base_rate or 0.)
        duration = track.duration
        if not duration:
            with open_media(path) as probe:
                last = None
                for frame in probe.decode(probe.streams[track.index]):
                    check()
                    last = frame.time
                if last is not None and track.start is not None:
                    duration = last - track.start + (1 / rate if rate > 0 else period)
        interval = max(period, (duration or 0.) / (max_samples - 1))
        offset = (track.start - origin) if track.start is not None else 0.
        next_time = offset
        last = None
        count = 0
        last_sample = None
        for index, frame in enumerate(media.decode(stream)):
            check()
            if frame.time is not None:
                stamp = max(0., frame.time - origin)
            elif rate > 0:
                stamp = offset if last is None else last + 1 / rate
            else:
                raise ValueError("视频缺少可用的时间戳和帧率。")
            if last is not None and stamp < last - 1e-5:
                raise ValueError("视频时间戳倒退，无法可靠对齐。")
            last = stamp
            frame_duration = getattr(frame, "duration", None)
            span = float(frame_duration * frame.time_base) if frame_duration and frame.time_base else 0.
            limits["end"] = stamp + (span if span > 0 else (1 / rate if rate > 0 else .001))
            if stamp + 1e-7 < next_time or count >= max_samples:
                continue
            image = frame.to_ndarray(format="bgr24")
            if frame.rotation in (-90, 90, -180, 180):
                image = np.ascontiguousarray(np.rot90(image, (frame.rotation // 90) % 4))
            count += 1
            last_sample = stamp
            next_time = offset + (math.floor((stamp - offset + 1e-7) / interval) + 1) * interval
            yield image, index, stamp, min(1., (stamp - offset) / duration) if duration else None
        if count == 0:
            raise ValueError("视频中没有可解码的画面。")
        # Frame counts in edited MP4s include samples outside the edit. Use the
        # stream's own time base to report appreciable early EOF without losing
        # the frames that were successfully decoded.
        if stream.duration and track.start is not None:
            declared_end = offset + float(stream.duration * stream.time_base)
            if declared_end - limits["end"] > max(0.5, (declared_end - offset) * .05):
                limits["declared_end"] = declared_end
                raise ValueError(f"视频流标记至 {declared_end:.1f} 秒，最后可解码画面在 {limits['end']:.1f} 秒。")
        if count == max_samples and last - last_sample > interval * 1.5:
            raise ValueError("视频实际时长超出时间索引，抽样未覆盖结尾。")


def audio_windows(track, origin):
    start = track.start - origin if track.start is not None else 0.
    duration = track.duration
    if duration is None:
        return [(start, start + MAX_AUDIO_SECONDS)]
    if duration <= MAX_AUDIO_SECONDS:
        return [(start, start + duration)]
    width = MAX_AUDIO_SECONDS / LONG_FILE_WINDOWS
    return [(float(t), float(t + width))
            for t in np.linspace(start, start + duration - width, LONG_FILE_WINDOWS)]


def pcm_parts(path, track, origin, check):
    """Yield mono 16 kHz PCM with PTS. Bound memory to one compressed/decoded frame."""
    import av
    with open_media(path) as media:
        stream = media.streams[track.index]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=RATE)
        next_stamp = (track.start - origin) if track.start is not None else 0.
        decoded = False
        for packet in media.demux(stream):
            check()
            for source in packet.decode():
                decoded = True
                for frame in resampler.resample(source):
                    stamp = frame.time - origin if frame.time is not None else next_stamp
                    pcm = frame.to_ndarray().reshape(-1)
                    next_stamp = stamp + len(pcm) / RATE
                    yield stamp, pcm
        if not decoded:
            raise ValueError("音轨没有可解码的采样。")
        for frame in resampler.resample(None):
            stamp = frame.time - origin if frame.time is not None else next_stamp
            pcm = frame.to_ndarray().reshape(-1)
            next_stamp = stamp + len(pcm) / RATE
            yield stamp, pcm


def speech_segments(parts, vad, check):
    """Respect discontinuities; VAD padding for the last 32ms never enters inference."""
    endpoint = SpeechSegmenter()
    pending = np.empty(0, dtype=np.float32)
    pending_start = expected = None

    def feed(block, start):
        prob = vad(block if len(block) == 512 else np.pad(block, (0, 512 - len(block))))
        segment = endpoint.feed(block, start, prob)
        if segment is not None:
            yield (*segment, endpoint.last_metrics)

    def finish():
        if len(pending):
            yield from feed(pending, pending_start)
        segment = endpoint.finish()
        if segment is not None:
            yield (*segment, endpoint.last_metrics)

    for stamp, pcm in parts:
        check()
        if expected is not None and abs(stamp - expected) > 2 / RATE:
            yield from finish()
            endpoint.reset()
            vad.reset()
            pending = np.empty(0, dtype=np.float32)
        if not len(pending):
            pending_start = stamp
        pcm = np.concatenate((pending, pcm)) if len(pending) else pcm
        expected = pending_start + len(pcm) / RATE
        whole = len(pcm) // 512 * 512
        for i in range(0, whole, 512):
            check()
            yield from feed(pcm[i:i+512], pending_start + i / RATE)
        pending = pcm[whole:].copy()
        pending_start += whole / RATE
    yield from finish()


def read_speech(path, track, origin, model_dir, check, windows):
    def selected_parts():
        idx = 0
        emitted_end = 0.
        for stamp, samples in pcm_parts(path, track, origin, check):
            right = stamp + len(samples) / RATE
            # A single decoded block can cover both sides of a short sampling gap.
            while idx < len(windows):
                start, end = windows[idx]
                if right <= start:
                    break
                i = max(0, round((max(start, emitted_end) - stamp) * RATE))
                j = min(len(samples), round((end - stamp) * RATE))
                if j > i:
                    block_start = max(0., stamp + i / RATE)
                    emitted_end = block_start + (j - i) / RATE
                    yield block_start, samples[i:j]
                if right < end:
                    break
                idx += 1
            if idx == len(windows):
                return
    try:
        vad = SileroVAD(Path(model_dir) / "vad" / "silero_vad.onnx")
    except Exception as exc:
        # ONNX Runtime exposes platform-specific exception types on load failure.
        raise ValueError(f"本地语音活动检测模型未就绪（{type(exc).__name__}）") from exc
    yield from speech_segments(selected_parts(), vad, check)
