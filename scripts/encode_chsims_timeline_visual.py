"""Encode frames within each surviving interval using the pinned Omni vision tower."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.chsims_omni_teacher import MODEL, REVISION
from scripts.chsims_teacher_data import CACHE, FRAME_PERIOD, SPLITS, ranked_ids, read_rows, sha_file
from scripts.prepare_chsims_teacher_gaps import PREPARED, save_archive, with_cache
from scripts.prepare_chsims_teacher_timeline import DIRECTORY, remaining_spans


VISUAL = f"{DIRECTORY}/vision"


class VisionTower:
    def __init__(self, model_dir):
        import torch
        import transformers
        from accelerate import init_empty_weights
        from safetensors import safe_open
        from transformers import Qwen2_5OmniConfig, Qwen2_5OmniProcessor
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniVisionEncoder

        receipt = json.loads((model_dir / "verified-model.json").read_text("utf-8-sig"))
        if (receipt["model"], receipt["revision"]) != (MODEL, REVISION):
            raise ValueError("Use the pinned Omni checkpoint")
        self.torch, self.device = torch, torch.device("cuda:0")
        torch.set_num_threads(2)
        processor = Qwen2_5OmniProcessor.from_pretrained(model_dir, local_files_only=True)
        self.processor = processor.video_processor
        config = Qwen2_5OmniConfig.from_pretrained(model_dir, local_files_only=True).thinker_config
        config._attn_implementation = "sdpa"
        with init_empty_weights():
            self.model = Qwen2_5OmniVisionEncoder._from_config(config.vision_config)
        index = json.loads((model_dir / "model.safetensors.index.json").read_text("utf8"))["weight_map"]
        prefix = "thinker.visual."
        needed = {path for key, path in index.items() if key.startswith(prefix)}
        verified = {}
        weights = {}
        for shard in sorted(needed):
            checksum = sha_file(model_dir / shard)
            if checksum != receipt["sha256"][shard]:
                raise ValueError("The released vision tower weights failed SHA-256 verification")
            verified[shard] = checksum
            with safe_open(model_dir / shard, framework="pt") as archive:
                for name in archive.keys():
                    if name.startswith(prefix):
                        weights[name.removeprefix(prefix)] = archive.get_tensor(name).to(self.device, dtype=torch.bfloat16)
        self.model.load_state_dict(weights, strict=True, assign=True)
        self.model.to(self.device).eval()
        self.identity = {"code": sha_file(Path(__file__)), "model": MODEL, "revision": REVISION, "weights": verified,
                         "torch": torch.__version__, "transformers": transformers.__version__}

    def encode(self, clips):
        torch = self.torch
        features = self.processor(videos=clips, fps=1 / FRAME_PERIOD, do_sample_frames=False,
                                  size={"shortest_edge": 224**2, "longest_edge": 224**2}, return_tensors="pt")
        patches = features["pixel_values_videos"].to(self.device, dtype=self.model.dtype)
        grids = features["video_grid_thw"].to(self.device)
        lengths = (grids.prod(1) // self.processor.merge_size**2).tolist()
        with torch.inference_mode():
            values = self.model(patches, grid_thw=grids).float()
            pooled = torch.stack([torch.cat((item.mean(0), item.std(0, unbiased=False)))
                                  for item in values.split(lengths)]).cpu().numpy()
        if not np.isfinite(pooled).all():
            raise ValueError("The Omni vision encoder returned invalid window features")
        return pooled


def encode(cache, keys, batch_size):
    model = VisionTower(CACHE / "omni-3b")
    started = perf_counter()
    # Confirm pose/timing windows use the same feature definition as the original clip encoder.
    clips, frozen = [], []
    for key in keys[:8]:
        with np.load(cache / PREPARED / f"{key}.npz", allow_pickle=False) as entry:
            video = entry["video"]
        if len(video):
            with np.load(cache / "missing-evidence/gap-teacher" / f"{key}.npz", allow_pickle=False) as original:
                frozen.append(original["visual_frames"])
            clips.append(video)
    if clips:
        old = np.stack(frozen)
        deviation = np.linalg.norm(model.encode(clips) - old, axis=1) / np.linalg.norm(old, axis=1)
        if deviation.max() > .035:
            raise ValueError("Window vision features differ from the pinned Omni encoder")
        print(f"Omni vision feature replay on {len(clips)} clips: relative L2 {deviation.max():.5f}", flush=True)
    for start in range(0, len(keys), batch_size):
        pending, clips = [], []
        for key in keys[start:start + batch_size]:
            src = cache / PREPARED / f"{key}.npz"
            dest = cache / VISUAL / f"{key}.npz"
            receipt = {"input": sha_file(src), "encoder": model.identity}
            if with_cache(dest, receipt, "provenance"):
                continue
            with np.load(src, allow_pickle=False) as entry:
                info = json.loads(entry["info"].item())
                video, times = entry["video"], np.asarray(info["frame_times"])
            seen, coverage = [], []
            for left, right in remaining_spans(info["duration"], info["gap"]):
                keep = (times >= left) & (times < right)
                if keep.any():
                    clips.append(video[keep])
                    covered = np.minimum(np.diff(np.r_[times[keep], right]), FRAME_PERIOD).sum()
                    coverage.append(float(np.clip(covered / (right - left), 0., 1.)))
                else:
                    coverage.append(0.)
                seen.append(bool(keep.any()))
            pending.append((dest, {"provenance": receipt, "coverage": coverage}, seen))
        vectors = iter(model.encode(clips) if clips else ())
        for path, info, seen in pending:
            blocks = {f"window-{i}": next(vectors) for i, present in enumerate(seen) if present}
            save_archive(path, info, **blocks)
        done = min(start + batch_size, len(keys))
        if done % 128 < batch_size or done == len(keys):
            print(f"Temporal vision {done}/{len(keys)} in {perf_counter() - started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--data", type=Path)
    inputs.add_argument("--cohort", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Use a positive batch size and limit")
    if args.data is not None:
        rows = read_rows(args.data)
        keys = [key for split in SPLITS for key in ranked_ids(rows, split)]
    else:
        keys = [row["id"] for row in json.loads(args.cohort.read_text("utf8"))["rows"]]
    encode(args.cache, keys[:args.limit], args.batch_size)


if __name__ == "__main__":
    main()
