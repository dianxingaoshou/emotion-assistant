"""Download pinned, official model assets into this project's models directory.

Only this explicit setup command uses the network; ModelHub never downloads.
Interrupted downloads keep a .part file and may be resumed on the next run.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
EMOTIEFF_REV = "520a051c64cd191521e5934655314e769a319684"
YUNET_REV = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
EMOTION_REV = "7be9a30d52bdc793cc24e6f7d25358d6648927c1"
ASR_REV = "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7"
VAD_REV = "6478567951ae5c9979ad7b234185b5515f4be7a1"


def asset(group, path, url, sha256, revision):
    return dict(group=group, path=path, url=url, sha256=sha256, revision=revision)


def hf(group, repo, revision, filename, sha256):
    folder = "emotion2vec_plus_seed" if group == "audio" else "SenseVoiceSmall"
    spec = asset(group, f"{folder}/{filename}",
                 f"https://huggingface.co/{repo}/resolve/{revision}/{filename}", sha256, revision)
    # The authors also publish these identical parameters on ModelScope.
    # Its branch may move: accept it ONLY if the pinned HF SHA256 matches.
    if filename == "model.pt":
        spec["official_mirror"] = f"https://modelscope.cn/models/iic/{folder}/resolve/master/{filename}"
    return spec


ASSETS = [
    asset("visual", "visual/face_detection_yunet_2023mar.onnx",
          f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{YUNET_REV}/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
          "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4", YUNET_REV),
    asset("visual", "visual/enet_b0_8_best_afew.onnx",
          f"https://raw.githubusercontent.com/sb-ai-lab/EmotiEffLib/{EMOTIEFF_REV}/models/affectnet_emotions/onnx/enet_b0_8_best_afew.onnx",
          "7aa2ea31c1311f4f8aa9d3fdb085d418dd4e7a48c4b9ed41df8c044f91d0213f", EMOTIEFF_REV),
    asset("audio", "vad/silero_vad.onnx",
          f"https://raw.githubusercontent.com/snakers4/silero-vad/{VAD_REV}/src/silero_vad/data/silero_vad.onnx",
          "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f", VAD_REV),
    hf("audio", "emotion2vec/emotion2vec_plus_seed", EMOTION_REV, "model.pt",
       "327e861049c7105a3f8589edfd16543a7cdc0c1926f1b64de975d98e0650cdd3"),
    hf("audio", "emotion2vec/emotion2vec_plus_seed", EMOTION_REV, "config.yaml",
       "50674ec187406b4f1208925594ceaf1f28f16a22333b5f2ac5b3d0cbffdd670e"),
    hf("audio", "emotion2vec/emotion2vec_plus_seed", EMOTION_REV, "configuration.json",
       "8b6a745213025c7d4565f9f074cfef1b5cd5ef76e38e2a8f7f8a3db271e735b2"),
    hf("audio", "emotion2vec/emotion2vec_plus_seed", EMOTION_REV, "tokens.txt",
       "654625c18ef0f0583441a3b811c20904f2cf30febb716f32e14810b8e6bb715c"),
    hf("asr", "FunAudioLLM/SenseVoiceSmall", ASR_REV, "model.pt",
       "833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea"),
    hf("asr", "FunAudioLLM/SenseVoiceSmall", ASR_REV, "config.yaml",
       "f71e239ba36705564b5bf2d2ffd07eece07b8e3f2bbf6d2c99d8df856339ac19"),
    hf("asr", "FunAudioLLM/SenseVoiceSmall", ASR_REV, "configuration.json",
       "02810a7f8e9e8aee10370a265f7e799728ce25b4c00cdbf4602b303ee395a38e"),
    hf("asr", "FunAudioLLM/SenseVoiceSmall", ASR_REV, "am.mvn",
       "29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5"),
    hf("asr", "FunAudioLLM/SenseVoiceSmall", ASR_REV, "chn_jpn_yue_eng_ko_spectok.bpe.model",
       "aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8"),
]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(spec, model_dir):
    target = model_dir / spec["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = spec["sha256"]
    if target.is_file():
        actual = sha256_file(target)
        if expected and actual != expected:
            raise ValueError(f"Checksum mismatch for existing {target}; move it aside before retrying")
        print(f"Verified {spec['path']}", flush=True)
        return {**spec, "sha256": actual, "bytes": target.stat().st_size}
    partial = target.with_name(target.name + ".part")
    if partial.is_file() and expected and sha256_file(partial) == expected:
        os.replace(partial, target)
        print(f"Verified completed partial {spec['path']}", flush=True)
        return {**spec, "sha256": expected, "bytes": target.stat().st_size}
    for attempt in range(1, 6):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "emotion-assistant-phase1/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            print(f"Downloading {spec['path']} (resume {offset:,} bytes)", flush=True)
            source = spec.get("official_mirror", spec["url"]) if attempt <= 3 else spec["url"]
            req = urllib.request.Request(source, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as response:
                resumed = response.status == 206 and offset > 0
                with partial.open("ab" if resumed else "wb") as stream:
                    last_log = time.monotonic()
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        if time.monotonic() - last_log > 15:
                            print(f"  {spec['path']}: {stream.tell() / 2**20:.1f} MiB", flush=True)
                            last_log = time.monotonic()
            actual = sha256_file(partial)
            if expected and actual != expected:
                # Never publish a corrupt or truncated weight as a valid model.
                raise ValueError(f"SHA256 mismatch: {spec['path']}: got {actual}, expected {expected}")
            os.replace(partial, target)
            print(f"Downloaded and verified {spec['path']}", flush=True)
            return {**spec, "sha256": actual, "bytes": target.stat().st_size, "download_url": source}
        except (OSError, urllib.error.URLError) as exc:
            if attempt == 5:
                raise RuntimeError(f"Download failed: {spec['path']}: {exc}") from exc
            print(f"Retry {attempt}/5 for {spec['path']}: {exc}", flush=True)
            time.sleep(min(2 * attempt, 8))
    raise RuntimeError("Unreachable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual", action="store_true", help="YuNet + EmotiEffLib ONNX")
    parser.add_argument("--audio", action="store_true", help="emotion2vec+ seed + Silero VAD")
    parser.add_argument("--asr", action="store_true", help="optional SenseVoiceSmall")
    parser.add_argument("--all", action="store_true", help="download all models, about 2 GiB")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    groups = {g for g in ("visual", "audio", "asr") if getattr(args, g)}
    if args.all:
        groups = {"visual", "audio", "asr"}
    elif not groups:
        groups = {"visual", "audio"}
    args.model_dir = args.model_dir.resolve()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    selected = [spec for spec in ASSETS if spec["group"] in groups]
    manifest_file = args.model_dir / "manifest.json"
    previous = json.loads(manifest_file.read_text("utf-8")) if manifest_file.exists() else {}
    records = {record["path"]: record for record in previous.get("assets", [])}
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(args.workers, 6))) as pool:
        futures = {pool.submit(download_one, spec, args.model_dir): spec for spec in selected}
        for future in concurrent.futures.as_completed(futures):
            spec = futures[future]
            try:
                records[spec["path"]] = future.result()
            except Exception as exc:
                failures.append(f"{spec['path']}: {exc}")
                print(f"ERROR: {failures[-1]}", flush=True)
            temp = manifest_file.with_suffix(".json.tmp")
            temp.write_text(json.dumps({"schema_version": 1, "assets": list(records.values())},
                                       ensure_ascii=False, indent=2), "utf-8")
            os.replace(temp, manifest_file)
    if failures:
        raise SystemExit("Some downloads failed. Re-run the same command to resume.\n" + "\n".join(failures))
    print(f"Done. Offline model manifest: {manifest_file}")


if __name__ == "__main__":
    main()
