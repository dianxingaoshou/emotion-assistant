"""Extract visual-only Omni representations from faces in the masked CH-SIMS frames."""
import argparse
import inspect
import io
import json
import logging
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.chsims_teacher_data import CACHE, SPLITS, model_frame, ranked_ids, read_rows, sha_file, write_atomic


def face_crops(video, observe):
    import cv2

    crops, sizes = [], []
    for stamp, rgb in enumerate(video):
        bgr = cv2.resize(rgb[:, :, ::-1], (448, 448), interpolation=cv2.INTER_LINEAR)
        observation = observe(bgr, "face-teacher", stamp, stamp + .5)
        if observation["reason"].startswith(("model_unavailable", "inference_error")):
            raise RuntimeError(observation["reason"])
        box = observation.get("face_box")
        if not box:
            continue
        x, y, width, height = box
        if min(width, height) < 24:
            continue
        center_x, center_y = x + width / 2, y + height / 2
        radius = .68 * max(width, height)
        left, right = max(0, round(center_x - radius)), min(448, round(center_x + radius))
        top, bottom = max(0, round(center_y - radius)), min(448, round(center_y + radius))
        # Resize from pixels in the already masked frame. No original subtitles
        # or other modalities are added by the crop.
        crops.append(model_frame(bgr[top:bottom, left:right]))
        sizes.append(min(width, height) / 2)
    return np.asarray(crops, dtype=np.uint8).reshape(-1, 224, 224, 3), sizes


def source_identity(cache):
    from emotion_assistant.models import ModelHub
    from emotion_assistant.paths import MODELS
    from scripts import chsims_omni_teacher

    model_dir = cache.parent / "omni-3b"
    return {"extractor_sha256": sha_file(Path(__file__)),
            "omni_sha256": sha_file(Path(chsims_omni_teacher.__file__)),
            "receipt_sha256": sha_file(model_dir / "verified-model.json"),
            "visual_adapter_sha256": sha_file(Path(inspect.getfile(ModelHub))),
            "assets_sha256": sha_file(MODELS / "manifest.json")}


def score(cache, keys, batch_size):
    root = Path(__file__).resolve().parents[1]
    sys.path.append(str(root / ".venv/Lib/site-packages"))
    import cv2
    from emotion_assistant.models import ModelHub
    from emotion_assistant.paths import MODELS, configure_environment
    from scripts.chsims_omni_teacher import OmniTeacher

    configure_environment()
    cv2.setNumThreads(1)
    hub = ModelHub(MODELS, enable_asr=False)
    status = hub.load_visual()
    if not all(model["ready"] for model in status.values()):
        raise RuntimeError(f"Face detection setup failed: {status}")
    logging.getLogger().addFilter(
        lambda record: not record.getMessage().startswith("System prompt modified, audio output"))
    teacher = OmniTeacher(cache.parent / "omni-3b")
    identity = source_identity(cache)
    started = perf_counter()
    for start in range(0, len(keys), batch_size):
        groups = []
        for key in keys[start:start + batch_size]:
            media = cache / "media" / f"{key}.npz"
            destination = cache / "face" / f"{key}.npz"
            provenance = {"source": sha_file(media), "extractor": identity}
            if destination.is_file():
                with np.load(destination, allow_pickle=False) as existing:
                    if json.loads(existing["info"].item())["provenance"] == provenance:
                        continue
            with np.load(media, allow_pickle=False) as source:
                video = source["video"]
            faces, sizes = face_crops(video, hub.analyze_frame)
            info = {"provenance": provenance, "frames": len(video), "face_frames": len(faces),
                    "mean_face_size": float(np.mean(sizes)) if sizes else 0.0}
            groups.append((destination, info, faces))
        items = [{"video": faces} for _, _, faces in groups if len(faces)]
        predictions, pooled = teacher.score("visual", items, capture=True) if items else ([], [])
        encoded = iter(zip(predictions, pooled, strict=True))
        for path, info, faces in groups:
            prediction, blocks = next(encoded) if len(faces) else (None, {})
            info["prediction"] = prediction
            buffer = io.BytesIO()
            np.savez(buffer, info=json.dumps(info), **blocks)
            write_atomic(path, buffer.getvalue())
        done = min(start + batch_size, len(keys))
        if done % 100 < batch_size or done == len(keys):
            print(f"Face features {done}/{len(keys)} in {perf_counter() - started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Use a positive batch size and limit")
    splits = SPLITS if args.split == "all" else (args.split,)
    rows = read_rows(args.data)
    keys = [key for split in splits for key in ranked_ids(rows, split)]
    score(args.cache, keys[:args.limit], args.batch_size)


if __name__ == "__main__":
    main()
