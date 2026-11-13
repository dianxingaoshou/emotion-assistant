"""Read-only environment/device inventory. Does not open a camera/microphone."""
from __future__ import annotations
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emotion_assistant.paths import configure_environment, MODELS, REPORTS


def main():
    configure_environment()
    packages = {}
    for name in ("PySide6", "numpy", "opencv-python-headless", "sounddevice", "soxr",
                 "onnxruntime", "torch", "torchaudio", "funasr"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {"python": sys.version, "executable": sys.executable,
              "isolated": sys.prefix != sys.base_prefix, "platform": platform.platform(),
              "project": str(ROOT), "packages": packages, "cameras": [], "microphones": [],
              "device_capture_tested": False, "models": {}}
    try:
        import psutil
        report["hardware"] = {"processor": platform.processor(), "logical_cpus": psutil.cpu_count(),
                              "memory_bytes": psutil.virtual_memory().total}
    except ImportError:
        pass
    try:
        import torch
        report["inference"] = {"device": "cpu", "torch_runtime_version": torch.__version__,
                               "cuda_runtime": torch.version.cuda}
    except ImportError:
        pass
    try:
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtMultimedia import QMediaDevices
        app = QCoreApplication.instance() or QCoreApplication([])
        report["cameras"] = [device.description() for device in QMediaDevices.videoInputs()]
    except Exception as exc:
        report["camera_error"] = type(exc).__name__
    try:
        import sounddevice as sd
        report["microphones"] = [{"index": index, "name": item["name"],
                                   "default_samplerate": item["default_samplerate"]}
                                  for index, item in enumerate(sd.query_devices())
                                  if item["max_input_channels"] > 0]
    except Exception as exc:
        report["microphone_error"] = type(exc).__name__
    for relative in ("visual/face_detection_yunet_2023mar.onnx", "visual/enet_b0_8_best_afew.onnx",
                     "vad/silero_vad.onnx", "emotion2vec_plus_seed/model.pt", "SenseVoiceSmall/model.pt"):
        path = MODELS / relative
        report["models"][relative] = {"present": path.is_file(),
                                     "bytes": path.stat().st_size if path.is_file() else 0}
    target = REPORTS / "environment.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["isolated"] and all(packages.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
