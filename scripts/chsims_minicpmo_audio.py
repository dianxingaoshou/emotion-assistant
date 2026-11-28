"""Read the native audio encoder from the MiniCPM-o 4.5 multimodal checkpoint."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.chsims_teacher_data import CACHE, SPLITS, ranked_ids, read_rows, sha_file, write_atomic


MODEL = "OpenBMB/MiniCPM-o-4_5"
GAP_AUDIO = "missing-evidence/gap-minicpmo"
ASSETS = {
    "config.json": "b0c03cf9f0cfc1178ade50e4ba62727ca7a77aa6a6a71eda6668d8f289e96c1d",
    "model.safetensors.index.json": "e578de05a95804bb15237a6fd7c236414e0160cd76751da82c9f7f0d134596e7",
    "model-00001-of-00004.safetensors": "30c40b9a10386c1bc404568d8829e5aada2e3501d9de6fb46ff80451aff7e077",
}


class MiniCPMAudio:
    def __init__(self, root):
        import torch
        from accelerate import init_empty_weights
        from safetensors import safe_open
        from transformers import WhisperConfig, WhisperFeatureExtractor
        from transformers.models.whisper.modeling_whisper import WhisperEncoder

        self.torch, self.device = torch, torch.device("cuda:0")
        torch.set_num_threads(2)
        for file, digest in ASSETS.items():
            if sha_file(root / file) != digest:
                raise ValueError(f"MiniCPM-o checkpoint checksum failed: {file}")
        data = json.loads((root / "config.json").read_text("utf-8"))
        if data["model_type"] != "minicpmo" or not data.get("init_audio") or not data.get("init_vision"):
            raise ValueError("Expected the full MiniCPM-o checkpoint metadata")
        cfg = WhisperConfig(**data["audio_config"])
        cfg._attn_implementation = "sdpa"
        with init_empty_weights():
            self.encoder = WhisperEncoder(cfg)
        index = json.loads((root / "model.safetensors.index.json").read_text("utf-8"))["weight_map"]
        shards = {index[key] for key in index if key.startswith("apm.")}
        if len(shards) != 1 or not shards.issubset(ASSETS):
            raise ValueError("Audio tensors differ from the verified release")
        with safe_open(root / shards.pop(), framework="pt") as reader:
            values = {key[4:]: reader.get_tensor(key).to(self.device, dtype=torch.bfloat16)
                      for key in reader.keys() if key.startswith("apm.")}
        self.encoder.load_state_dict(values, strict=True, assign=True)
        self.encoder.to(self.device).eval().requires_grad_(False)
        self.extractor = WhisperFeatureExtractor(feature_size=cfg.num_mel_bins, dither=0.)
        self.identity = {"model": MODEL, "assets": ASSETS, "code": sha_file(Path(__file__)),
                         "native_modalities": ["text", "audio", "video"], "torch": torch.__version__}

    def encode(self, waves):
        from torch.nn.functional import gelu

        torch = self.torch
        # Whisper has 30 seconds of positional embeddings. Longer clips are
        # split and pooled over valid acoustic frames.
        parts = [(index, wave[start:start + 480000]) for index, wave in enumerate(waves)
                 for start in range(0, len(wave), 480000)]
        audio = [samples for _, samples in parts]
        max_length = ((max(map(len, audio)) + 159) // 160) * 160
        result = self.extractor(audio, sampling_rate=16000, padding="max_length", max_length=max_length,
                                return_attention_mask=True, return_tensors="pt", device=str(self.device))
        mel = result.input_features.to(self.device, dtype=torch.bfloat16)
        lengths = ((result.attention_mask.sum(1) + 1) // 2).to(self.device)
        encoder = self.encoder
        with torch.inference_mode():
            x = gelu(encoder.conv2(gelu(encoder.conv1(mel)))).transpose(1, 2)
            positions = torch.arange(x.shape[1], device=self.device)
            x = x + encoder.embed_positions(positions)
            keep = positions[None, :] < lengths[:, None]
            mask = torch.zeros((len(audio), 1, 1, len(positions)), dtype=x.dtype, device=self.device)
            mask.masked_fill_(~keep[:, None, None, :], -torch.inf)
            mask = mask.expand(-1, -1, len(positions), -1)
            states = {}
            for n, layer in enumerate(encoder.layers, 1):
                x = layer(x, attention_mask=mask, layer_head_mask=None)[0]
                if n == 12:
                    states["mini_middle"] = x.float().cpu().numpy()
            states["mini_native"] = encoder.layer_norm(x).float().cpu().numpy()
        owners = [index for index, _ in parts]
        sizes = lengths.tolist()
        records = []
        for index in range(len(waves)):
            record = {}
            for name, array in states.items():
                points = np.concatenate([array[j, :sizes[j]] for j, owner in enumerate(owners) if owner == index])
                vector = np.concatenate((points.mean(0), points.std(0)))
                if not np.isfinite(vector).all():
                    raise ValueError("MiniCPM-o returned non-finite audio representations")
                record[name] = vector
            records.append(record)
        return records


def extract(model, cache, keys, batch_size, *, input_folder="media", output_folder="minicpmo"):
    start = perf_counter()
    for offset in range(0, len(keys), batch_size):
        pending = []
        for clip in keys[offset:offset + batch_size]:
            with np.load(cache / input_folder / f"{clip}.npz", allow_pickle=False) as media:
                wave = media["pcm"]
            info = {"model": model.identity, "audio_sha256": hashlib.sha256(np.ascontiguousarray(wave)).hexdigest()}
            dst = cache / output_folder / f"{clip}.npz"
            if dst.is_file():
                with np.load(dst, allow_pickle=False) as archive:
                    if json.loads(archive["info"].item()) == info:
                        continue
            pending.append((dst, info, wave))
        active = [wave for _, _, wave in pending if len(wave) >= 400]
        vectors = iter(model.encode(active) if active else ())
        for dst, info, wave in pending:
            data = next(vectors) if len(wave) >= 400 else {}
            buf = io.BytesIO()
            np.savez(buf, info=json.dumps(info), **data)
            write_atomic(dst, buf.getvalue())
        done = min(offset + batch_size, len(keys))
        if done % 128 < batch_size or done == len(keys):
            print(f"MiniCPM-o audio {done}/{len(keys)} in {perf_counter() - start:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--audio-gap", action="store_true", help="Encode the prepared, locally interrupted audio tracks")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Use a positive batch size")
    rows = read_rows(args.data)
    splits = SPLITS if args.split == "all" else (args.split,)
    keys = [key for split in splits for key in ranked_ids(rows, split)]
    model = MiniCPMAudio(CACHE / "minicpm-o-4_5")
    if args.audio_gap:
        from scripts.prepare_chsims_teacher_gaps import PREPARED

        extract(model, args.cache, keys, args.batch_size, input_folder=PREPARED, output_folder=GAP_AUDIO)
    else:
        extract(model, args.cache, keys, args.batch_size)


if __name__ == "__main__":
    main()
