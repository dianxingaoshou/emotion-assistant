import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.mark.parametrize(("enabled", "index", "expected"), [
    (False, 1, "original"), (True, 0, "v1"), (True, 1, "v2"),
])
def test_file_selector_routes_policy_without_enabling_transcription(monkeypatch, tmp_path, enabled, index, expected):
    from PySide6.QtWidgets import QApplication
    from emotion_assistant import local_media_widgets

    app = QApplication.instance() or QApplication([])
    page = local_media_widgets.LocalMediaPage()
    calls = []
    stub = {"kind": "video", "mode": "unavailable", "valid_count": 0,
            "sampled_count": 0, "scores": None, "reasons": {}}

    def original(path, **options):
        assert "adaptive" not in options and not options["transcribe"]
        calls.append("original")
        return stub

    def classify(path, *, adaptive, **options):
        assert not options["transcribe"]
        calls.append("v2" if adaptive else "v1")
        return stub

    monkeypatch.setattr(local_media_widgets, "analyze_media", original)
    monkeypatch.setattr(local_media_widgets, "analyze_foundation_media", classify)
    page.show()
    try:
        assert page.foundation_policy.isHidden() and page.foundation_policy.currentData() is False
        page.foundation.setChecked(enabled)
        assert page.foundation_policy.isHidden() != enabled
        page.foundation_policy.setCurrentIndex(index)
        assert page.start_file(tmp_path / "输入.mp4") and not page.foundation_policy.isEnabled()
        deadline = time.monotonic() + 5
        while page._worker is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.005)
        assert page._worker is None and page.foundation_policy.isEnabled()
        assert calls == [expected] and page.result == stub
    finally:
        page.close()
        app.processEvents()
