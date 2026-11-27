"""Exercise the desktop file page using the publisher's Mandarin audio example."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MODELSCOPE_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication, QFileDialog

from emotion_assistant.local_media_widgets import LocalMediaPage
from model_smoke import SAMPLES


DEMO_PAGE = "https://fun-audio-llm.github.io/"
PUBLISHER_TRANSCRIPT = "开放时间早上9点至下午5点。"

def wait_for(app, predicate, seconds=180):
    deadline = time.monotonic() + seconds
    while not predicate():
        app.processEvents()
        if time.monotonic() > deadline:
            raise TimeoutError("Desktop verification timed out")
        time.sleep(.02)
    app.processEvents()


def analyze_file(page, app, sample, transcribe):
    page.transcribe.setChecked(transcribe)
    with patch.object(QFileDialog, "getOpenFileName", return_value=(str(sample), "")):
        page.choose_button.click()
    wait_for(app, lambda: page._worker is None)
    assert page.result and page.result["kind"] == "audio", page.status.text()
    return page.result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=ROOT / ".runtime/test_assets/sensevoice_zh.mp3")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/mandarin-file-recognition.json")
    parser.add_argument("--with-ollama", action="store_true", help="Also press the local Qwen review button")
    args = parser.parse_args()
    spec = SAMPLES["sensevoice_zh.mp3"]
    if not args.sample.is_file():
        parser.error("Missing public example. Obtain example/zh.mp3 from " + spec["url"])
    digest = hashlib.sha256(args.sample.read_bytes()).hexdigest()
    if digest != spec["sha256"]:
        parser.error("The sample does not match the pinned publisher file")

    app = QApplication.instance() or QApplication([])
    page = LocalMediaPage()
    page.show()
    try:
        off = analyze_file(page, app, args.sample, False)
        assert not off["transcripts"] and not off.get("semantic_context", {}).get("expression_profile")
        analyzed = analyze_file(page, app, args.sample, True)
        assert analyzed["transcription_status"] == "available", page.status.text()
        transcripts = [r for r in analyzed["transcripts"] if r.get("valid") and not r.get("stale")]
        assert transcripts and all(r["text"].strip() for r in transcripts)
        print("SenseVoice utterances:", [r["text"] for r in transcripts], flush=True)
        whole, text, origin = page._semantic_target()
        assert origin == "asr" and whole is analyzed
        assert text == PUBLISHER_TRANSCRIPT, (text, PUBLISHER_TRANSCRIPT)
        profile = whole["semantic_context"]["expression_profile"]
        assert profile["language"] == "zh" and profile["scope"] == "expressed_text"
        assert profile["affect_dimensions"]["status"] == "ok" and set(profile["affect_dimensions"]["dimensions"]) == {"valence", "arousal"}
        assert profile["fine_emotions"]["status"] in {"ok", "below_threshold"}, profile["fine_emotions"]
        whole_readout = page.profile_readout.text()
        assert page.profile_readout.isVisible() and "愉悦度" in whole_readout and "激活度" in whole_readout
        review = None
        if args.with_ollama:
            page.semantic_button.click()
            wait_for(app, lambda: page._review_request is None)
            review = analyzed.get("semantic_review") or {}
            assert review.get("status") == "ok" and review["text_source"] == origin, page.semantic_readout.text()
            assert review["chinese_dimensions"]["status"] == "ok" and review["chinese_dimensions"]["evidence"] in text

        checked = []
        for i in range(page.content_intervals.count()):
            page.content_intervals.setCurrentIndex(i)
            target = page._semantic_target()
            if target and target[2] == "asr_segment":
                row = page._utterance_for(target[0])
                segment_profile = row["semantic_context"]["expression_profile"]
                assert segment_profile["language"] == "zh" and page.profile_readout.isVisible()
                checked.append({"t_start": row["t_start"], "t_end": row["t_end"], "text": row["text"],
                                "emotion_status": segment_profile["fine_emotions"]["status"],
                                "dimension_status": segment_profile["affect_dimensions"]["status"]})
        assert len(checked) == len(transcripts)
        with TemporaryDirectory() as folder:
            destination = Path(folder) / "export.json"
            with patch.object(QFileDialog, "getSaveFileName", return_value=(str(destination), "")):
                page.export_button.click()
            exported = json.loads(destination.read_text(encoding="utf8"))
        assert exported == json.loads(json.dumps(analyzed, ensure_ascii=False, allow_nan=False))
        report = {"scope": "One publisher-provided Mandarin ASR demo recording, compared with the publisher's own output with inverse text normalization. No independent ASR benchmark, human emotion labels, or user outcome measurements.",
                  "sample_url": spec["url"], "sample_sha256": digest,
                  "publisher_demo_url": DEMO_PAGE, "matches_publisher_displayed_transcript": text == PUBLISHER_TRANSCRIPT,
                  "asr_model": "SenseVoiceSmall", "asr_disabled_utterances": len(off["transcripts"]),
                  "asr_enabled_utterances": len(transcripts), "selected_asr_segments": checked,
                  "whole_transcript": text, "automatic_expression_profile": profile,
                  "file_page_readout": whole_readout, "qwen_review": review,
                  "json_export_matches_page_result": True}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
        print("Verified", len(checked), "ASR segment(s) and the export at", args.output)
    finally:
        if page._worker is not None:
            page._worker.requestInterruption()
            wait_for(app, lambda: page._worker is None)
        page.stop_for_close()
        page.close()


if __name__ == "__main__":
    main()
