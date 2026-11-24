"""Exercise real file decoding and sampling with deterministic model observations."""
import json
from threading import Event
import time

import cv2
import numpy as np
import pytest

from emotion_assistant import local_media
from emotion_assistant.models import ModelHub


@pytest.fixture
def analyze(monkeypatch, tmp_path):
    def classify(self, frame, session_id, t_start, t_end):
        assert frame.dtype == np.uint8 and frame.shape == (48, 64, 3)
        assert 0 <= t_start < t_end
        value = frame.mean()
        obs = ModelHub._observation("visual", session_id, t_start, t_end)
        if value < 30:
            obs["reason"] = "no_face"
            return obs
        positive = 0.9 if value < 180 else 0.1
        obs.update(
            valid=True, reason="ok", quality=0.1 if value < 90 else 0.8,
            scores={"positive": positive, "neutral": 1 - positive, "negative": 0.0, "uncertain": 0.0},
            face_box=[10, 8, 24, 30],
        )
        return obs

    monkeypatch.setattr(ModelHub, "load_visual", lambda self: {
        "face": {"ready": True}, "visual": {"ready": True},
    })
    monkeypatch.setattr(ModelHub, "analyze_frame", classify)

    def run(path, on_sample=lambda *_: None, cancelled=lambda: False):
        return local_media.analyze_media(path, model_dir=tmp_path, on_sample=on_sample, cancelled=cancelled)

    return run


def write_photo(path, intensity):
    frame = np.full((48, 64, 3), intensity, np.uint8)
    ok, data = cv2.imencode(".png", frame)
    assert ok
    path.write_bytes(data.tobytes())


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "fixture.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8, (64, 48))
    assert writer.isOpened()
    try:
        # One second each of no face, low quality, positive and neutral.
        for intensity in (0, 60, 140, 220):
            for _ in range(8):
                writer.write(np.full((48, 64, 3), intensity, np.uint8))
    finally:
        writer.release()
    return path.rename(tmp_path / "四段表情.avi")


def test_photo_with_unicode_path_returns_box_and_serializable_result(tmp_path, analyze):
    path = tmp_path / "中文 照片.PNG"
    write_photo(path, 140)
    shown = []
    result = analyze(path, on_sample=lambda frame, obs, progress: shown.append(obs))
    assert result["kind"] == "image" and result["filename"] == path.name
    assert result["valid_count"] == result["sampled_count"] == len(shown) == 1
    assert shown[0]["face_box"] == [10, 8, 24, 30]
    assert result["visual_scores"]["positive"] == pytest.approx(0.9)
    assert result["scores"] == pytest.approx(result["fusion_intervals"][0]["scores"])
    assert json.loads(json.dumps(result, allow_nan=False))["samples"][0]["sample_index"] == 1


def test_video_uses_only_eligible_sample_scores_and_preserves_timing(video, analyze):
    seen = []

    def collect(frame, observation, progress):
        seen.append((round(frame.mean()), observation["sample_time_s"], observation["eligible"]))

    result = analyze(video, on_sample=collect)
    assert result["kind"] == "video" and result["filename"] == video.name
    assert result["sampled_count"] == len(seen) == len(result["samples"])
    times = [time for _, time, _ in seen]
    assert times == sorted(set(times)) and times[0] == pytest.approx(0) and times[-1] >= 3
    for value, _, eligible in seen:
        assert eligible == (value >= 90)
    useful = [value for value, _, _ in seen if value >= 90]
    assert len(useful) == result["valid_count"] and min(useful) < 180 <= max(useful)
    assert result["reasons"]["no_face"] == sum(value < 30 for value, _, _ in seen)
    assert sum(result["reasons"].values()) == len(seen) - len(useful)
    expected = sum(0.9 if value < 180 else 0.1 for value in useful) / len(useful)
    assert result["visual_scores"]["positive"] == pytest.approx(expected)
    assert result["mode"] == "visual_only" and result["audio_status"] == "no_audio_track"
    json.dumps(result, allow_nan=False)


def test_no_usable_photo_returns_no_summary_scores(tmp_path, analyze):
    path = tmp_path / "空白.png"
    write_photo(path, 0)
    result = analyze(path)
    assert result["sampled_count"] == 1 and result["valid_count"] == 0
    assert result["scores"] is None and result["reasons"] == {"no_face": 1}


def test_large_photo_face_box_uses_original_pixel_coordinates(tmp_path, monkeypatch, analyze):
    # Exercise the production downscaling path with a small image.
    monkeypatch.setattr(local_media, "MAX_INFERENCE_EDGE", 64)
    path = tmp_path / "原图.png"
    image = np.full((96, 128, 3), 140, np.uint8)
    ok, data = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(data.tobytes())
    previews = []
    result = analyze(path, on_sample=lambda frame, obs, _: previews.append((frame.shape, obs["face_box"])))
    assert previews == [((96, 128, 3), [20, 16, 48, 60])]
    assert result["samples"][0]["source_size"] == [128, 96]
    assert result["samples"][0]["inference_size"] == [64, 48]


def test_cancel_after_first_video_frame_does_not_complete(video, analyze):
    seen = []
    with pytest.raises(local_media.MediaCancelled):
        analyze(video, on_sample=lambda *_: seen.append(True), cancelled=lambda: bool(seen))
    assert len(seen) == 1


@pytest.mark.parametrize("name", ["损坏.png", "损坏.avi", "笔记.pdf"])
def test_damaged_or_unsupported_files_fail_before_a_preview(tmp_path, analyze, name):
    path = tmp_path / name
    path.write_bytes(b"not media")
    with pytest.raises((OSError, ValueError, RuntimeError)):
        analyze(path, on_sample=lambda *_: pytest.fail("invalid file produced a sample"))


@pytest.mark.usefixtures("analyze")
def test_file_picker_export_and_recovery_after_bad_file(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QFileDialog
    from emotion_assistant.local_media_widgets import LocalMediaPage

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    page = LocalMediaPage()
    photo = tmp_path / "选取的照片.png"
    write_photo(photo, 140)

    def wait_until_finished():
        deadline = time.monotonic() + 8
        while page._worker is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        app.processEvents()
        assert page._worker is None

    page.show()
    try:
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(photo), ""))
        page.choose_button.click()
        wait_until_finished()
        assert page.result["filename"] == photo.name and page.result["valid_count"] == 1
        assert page.export_button.isVisible() and not page.preview.pixmap().isNull()
        original = photo.read_bytes()
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(photo), ""))
        page.export_button.click()
        assert photo.read_bytes() == original and "导出失败" in page.status.text()
        output = tmp_path / "识别结果.json"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(output), ""))
        page.export_button.click()
        exported = json.loads(output.read_text(encoding="utf-8"))
        assert exported["scores"] == page.result["scores"]
        assert exported["samples"][0]["face_box"] == [10, 8, 24, 30]

        broken = tmp_path / "损坏.png"
        broken.write_bytes(b"not an image")
        assert page.start_file(broken)
        wait_until_finished()
        assert page.result is None and not page.export_button.isVisible()
        assert "检测失败" in page.status.text() and page.preview.pixmap().isNull()

        write_photo(photo, 0)
        assert page.start_file(photo)
        wait_until_finished()
        assert page.result["valid_count"] == 0 and page.result["scores"] is None
        assert page.export_button.isVisible()
    finally:
        page.stop_for_close()
        wait_until_finished()
        page.close()
        app.processEvents()


def test_closing_active_import_waits_for_worker_and_discards_partial_results(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from emotion_assistant import local_media_widgets

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    page = local_media_widgets.LocalMediaPage()
    path = tmp_path / "处理中.png"
    write_photo(path, 140)
    started = Event()
    ready = []
    page.ready_to_close.connect(lambda: ready.append(True))

    def delayed_analysis(path, *, model_dir, on_sample, cancelled, **options):
        started.set()
        deadline = time.monotonic() + 5
        while not cancelled() and time.monotonic() < deadline:
            time.sleep(0.005)
        if cancelled():
            raise local_media.MediaCancelled()
        raise RuntimeError("worker did not receive cancellation")

    monkeypatch.setattr(local_media_widgets, "analyze_media", delayed_analysis)
    try:
        assert page.start_file(path)
        assert started.wait(3)
        assert not page.stop_for_close() and not page.start_file(path)
    finally:
        page.cancel()
        deadline = time.monotonic() + 6
        while page._worker is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        page.close()
        app.processEvents()
    assert ready == [True] and page.stop_for_close()
    assert page.result is None and "已取消" in page.status.text()
