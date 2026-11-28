"""Cache the audio tower input to its final Transformer block without running the LLM."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.chsims_omni_teacher import MODEL, REVISION
from scripts.chsims_teacher_data import CACHE, SPLITS, ranked_ids, read_rows, sha_file, write_atomic


PREFIX = "thinker.audio_tower."
EARLY_LAYER = 15


def verified_audio_assets(model_dir):
    receipt = json.loads((model_dir / "verified-model.json").read_text("utf-8-sig"))
    if (receipt["model"], receipt["revision"]) != (MODEL, REVISION):
        raise ValueError("Expected the pinned Qwen2.5-Omni-3B model")
    index = model_dir / "model.safetensors.index.json"
    files = receipt["sha256"]
    if sha_file(index) != files[index.name]:
        raise ValueError(f"Checkpoint index checksum differs: {index}")
    weight_map = json.loads(index.read_text("utf-8"))["weight_map"]
    shard_names = {shard for name, shard in weight_map.items() if name.startswith(PREFIX)}
    for name in sorted(shard_names | {"config.json", "preprocessor_config.json"}):
        if sha_file(model_dir / name) != files[name]:
            raise ValueError(f"Checkpoint checksum differs: {name}")
    return receipt, sorted(shard_names)


class AudioTower:
    def __init__(self, model_dir):
        import torch
        from accelerate import init_empty_weights
        from safetensors import safe_open
        from transformers import Qwen2_5OmniConfig, WhisperFeatureExtractor
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoder

        receipt, shards = verified_audio_assets(model_dir)
        if not torch.cuda.is_available():
            raise RuntimeError("Run the experiment in the isolated CUDA environment")
        self.device, self.torch = torch.device("cuda:0"), torch
        torch.set_num_threads(2)
        self.extractor = WhisperFeatureExtractor.from_pretrained(model_dir, local_files_only=True)
        cfg = Qwen2_5OmniConfig.from_pretrained(model_dir, local_files_only=True).thinker_config.audio_config
        cfg._attn_implementation = "sdpa"
        with init_empty_weights():
            self.encoder = Qwen2_5OmniAudioEncoder(cfg)
        loaded = set()
        for shard in shards:
            with safe_open(model_dir / shard, framework="pt") as handle:
                tensors = {key[len(PREFIX):]: handle.get_tensor(key).to(self.device, dtype=torch.bfloat16)
                           for key in handle.keys() if key.startswith(PREFIX)}
                self.encoder.load_state_dict(tensors, strict=False, assign=True)
                loaded.update(tensors)
        if loaded != set(self.encoder.state_dict()) or any(p.is_meta for p in self.encoder.parameters()):
            raise ValueError("The audio tower did not load exactly from the checkpoint")
        self.encoder.to(self.device).eval().requires_grad_(False)
        self.identity = {"model": MODEL, "revision": REVISION, "weights": {key: receipt["sha256"][key] for key in shards},
                         "code": sha_file(Path(__file__)), "early_layer": EARLY_LAYER,
                         "torch": torch.__version__, "device": torch.cuda.get_device_name(0)}

    def extract(self, waves):
        torch = self.torch
        hop = self.extractor.hop_length
        max_length = ((max(map(len, waves)) + hop - 1) // hop) * hop
        values = self.extractor(waves, sampling_rate=16000, padding="max_length", max_length=max_length,
                                return_attention_mask=True, return_tensors="pt", device=str(self.device))
        valid = values.attention_mask.to(device=self.device, dtype=torch.bool)
        features = values.input_features.to(self.device, dtype=torch.bfloat16)
        lengths = valid.sum(1)
        after_cnn, after_pool = self.encoder._get_feat_extract_output_lengths(lengths)
        intermediate = {}

        def early(_, _args, out):
            intermediate["early"] = out[0]

        def tail(_, args, kwargs):
            intermediate["tail"] = args[0]
            spans = kwargs["cu_seqlens"]
            intermediate["chunks"] = (spans[1:] - spans[:-1]).tolist()

        first = self.encoder.layers[EARLY_LAYER].register_forward_hook(early)
        last = self.encoder.layers[-1].register_forward_pre_hook(tail, with_kwargs=True)
        try:
            with torch.inference_mode():
                pack = features.transpose(1, 2)[valid].T
                token_output = self.encoder(pack, feature_lens=lengths, aftercnn_lens=after_cnn).last_hidden_state
        finally:
            first.remove()
            last.remove()
        tail_array = intermediate["tail"].to(torch.float16).cpu().numpy()
        early_array = intermediate["early"].float().cpu().numpy()
        token_array = token_output.float().cpu().numpy()
        samples, cursor, tokens_at, chunk_at = [], 0, 0, 0
        for size, next_size in zip(after_cnn.tolist(), after_pool.tolist(), strict=True):
            end = cursor + size
            frame = tail_array[cursor:end]
            middle = early_array[cursor:end]
            pooled = token_array[tokens_at:tokens_at + next_size]
            n_chunks = (size + self.encoder.n_window - 1) // self.encoder.n_window
            chunks = np.array(intermediate["chunks"][chunk_at:chunk_at + n_chunks], np.int16)
            if chunks.sum() != size or not all(np.isfinite(value).all() for value in (frame, middle, pooled)):
                raise ValueError("Invalid audio token lengths or values")
            samples.append({"tokens": frame, "chunks": chunks,
                            "middle": np.concatenate((middle.mean(0), middle.std(0))),
                            "native": np.concatenate((pooled.mean(0), pooled.std(0)))})
            cursor, tokens_at, chunk_at = end, tokens_at + next_size, chunk_at + n_chunks
        return samples


def encode_batch(tower, items):
    pending = []
    for wave, destination in items:
        audio = np.ascontiguousarray(wave, dtype=np.float32)
        stamp = hashlib.sha256(audio).hexdigest()
        provenance = {"audio_sha256": stamp, "encoder": tower.identity}
        if destination.is_file():
            with np.load(destination, allow_pickle=False) as archive:
                if json.loads(archive["info"].item()) == provenance:
                    continue
        pending.append((audio, destination, provenance))
    supported = [entry for entry in pending if len(entry[0]) >= 400]
    pooled = iter(tower.extract([x[0] for x in supported]) if supported else ())
    for audio, destination, provenance in pending:
        values = next(pooled) if len(audio) >= 400 else {}
        stream = io.BytesIO()
        np.savez(stream, info=json.dumps(provenance), **values)
        write_atomic(destination, stream.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Use positive sizes")
    rows = read_rows(args.data)
    splits = SPLITS if args.split == "all" else (args.split,)
    keys = [key for split in splits for key in ranked_ids(rows, split)][:args.limit]
    tower = AudioTower(CACHE / "omni-3b")
    start = perf_counter()
    for offset in range(0, len(keys), args.batch_size):
        current = []
        for key in keys[offset:offset + args.batch_size]:
            with np.load(args.cache / "media" / f"{key}.npz", allow_pickle=False) as archive:
                current.append((archive["pcm"], args.cache / "audio-tokens" / f"{key}.npz"))
        encode_batch(tower, current)
        done = min(offset + args.batch_size, len(keys))
        if done % 128 < args.batch_size or done == len(keys):
            print(f"Audio tokens {done}/{len(keys)} in {perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
