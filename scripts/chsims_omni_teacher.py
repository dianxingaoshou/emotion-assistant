"""Local Qwen2.5-Omni next-token scoring for the four CH-SIMS evidence conditions."""
import json
from pathlib import Path
import unicodedata

import numpy as np

from scripts.chsims_teacher_data import FRAME_PERIOD, LABELS, VIEWS, sha_file


MODEL = "Qwen/Qwen2.5-Omni-3B"
REVISION = "f61daeb0493a6e98cec1a58ddeec8f13c1121cc0"
CHOICES = (*LABELS, "abstain")
QUESTION = {
    "text": "根据给出的自动转写，只判断这句台词文字表达的感情色彩。",
    "audio": "根据音频判断说话人声音传达的感情色彩，以语调、重音、语速、音量和语气为依据。",
    "visual": "根据静音画面，判断主要说话人的表情和动作表现出的感情色彩。",
    "full": "综合当前给出的台词、声音和画面，判断主要说话人整段表达的感情色彩。",
}
OPTIONS = ("A 负向，B 中性，C 正向，D 现有证据不足。平静、客观陈述属于中性。"
           "可用线索无法支持判断时选D。只输出一个选项字母。")


class OmniTeacher:
    def __init__(self, model_dir):
        import torch
        import transformers
        from accelerate import init_empty_weights
        from safetensors import safe_open
        from transformers import Qwen2_5OmniConfig, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        receipt = json.loads((model_dir / "verified-model.json").read_text("utf-8-sig"))
        if (receipt["model"], receipt["revision"]) != (MODEL, REVISION):
            raise ValueError("Expected the pinned Qwen2.5-Omni model receipt")
        for filename, digest in receipt["sha256"].items():
            if sha_file(model_dir / filename) != digest:
                raise ValueError(f"Teacher checkpoint checksum mismatch: {filename}")
        if not torch.cuda.is_available():
            raise RuntimeError("Use the isolated CUDA PyTorch environment for teacher scoring")
        self.torch, self.device = torch, torch.device("cuda:0")
        torch.set_num_threads(2)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_dir, local_files_only=True)
        tokenizer = self.processor.tokenizer
        self.choice_ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in "ABCD"]
        if not all(len(ids) == 1 for ids in self.choice_ids):
            raise ValueError("Each option must be one token")
        self.choice_ids = [ids[0] for ids in self.choice_ids]
        config = Qwen2_5OmniConfig.from_pretrained(model_dir, local_files_only=True).thinker_config
        config._attn_implementation = "sdpa"
        with init_empty_weights():
            self.model = Qwen2_5OmniThinkerForConditionalGeneration(config)
        index = json.loads((model_dir / "model.safetensors.index.json").read_text("utf8"))["weight_map"]
        model_keys = set(self.model.state_dict())
        loaded = set()
        for shard in sorted({shard for name, shard in index.items() if name.startswith("thinker.")}):
            with safe_open(model_dir / shard, framework="pt") as tensors:
                weights = {name[8:]: tensors.get_tensor(name).to(self.device, dtype=torch.bfloat16)
                           for name in tensors.keys() if name.startswith("thinker.")}
                self.model.load_state_dict(weights, strict=False, assign=True)
                loaded.update(weights)
            del weights
        self.model.tie_weights()
        uninitialized = [name for name, value in self.model.state_dict().items() if value.is_meta]
        if loaded - model_keys or uninitialized:
            raise ValueError(f"Incompatible checkpoint. Unexpected {sorted(loaded - model_keys)}, missing {uninitialized}")
        # The Transformers 4.57 Omni head otherwise scores every prompt token.
        self.features = None
        self.model.lm_head.register_forward_pre_hook(self._last_token)
        self.model.audio_tower.register_forward_hook(self._audio_tokens)
        self.model.visual.register_forward_hook(self._video_tokens)
        self.model.to(self.device).eval()
        torch.cuda.synchronize()
        self.identity = {**receipt, "implementation_sha256": sha_file(Path(__file__)), "dtype": "bfloat16",
                         "torch": torch.__version__, "transformers": transformers.__version__,
                         "gpu": torch.cuda.get_device_name(0), "native_audio_and_video": True}

    def _last_token(self, _module, args):
        token = args[0][:, -1:]
        if self.features is not None:
            self.features["context"] = token[:, 0]
        return (token,)

    def _audio_tokens(self, _module, _args, output):
        if self.features is not None:
            self.features["acoustic"] = output.last_hidden_state

    def _video_tokens(self, _module, _args, output):
        if self.features is not None:
            self.features["frames"] = output

    def _pool_features(self, inputs, count):
        lengths = {}
        if "acoustic" in self.features:
            valid = inputs["feature_attention_mask"].sum(dim=-1)
            lengths["acoustic"] = self.model.audio_tower._get_feat_extract_output_lengths(valid)[1].tolist()
        if "frames" in self.features:
            merge_size = self.model.spatial_merge_size
            lengths["frames"] = (inputs["video_grid_thw"].prod(dim=1) // merge_size**2).tolist()
        pooled = {"context": self.features["context"].float().cpu().numpy()}
        for name, sizes in lengths.items():
            tokens = self.features[name].float()
            if len(sizes) != count or min(sizes) <= 0 or sum(sizes) != tokens.shape[0]:
                raise ValueError(f"Cannot assign {name} tokens to individual clips")
            pooled[name] = self.torch.stack([self.torch.cat((part.mean(0), part.std(0, unbiased=False)))
                                             for part in tokens.split(sizes)]).cpu().numpy()
        if not all(np.isfinite(array).all() for array in pooled.values()):
            raise ValueError("Teacher returned invalid representations")
        return [{name: array[i] for name, array in pooled.items()} for i in range(count)]

    def score(self, view, items, *, capture=False):
        torch = self.torch
        # Batch examples with the same evidence set to keep audio/visual tokens aligned.
        keys = set(items[0])
        if not keys or not keys.issubset(VIEWS[view]) or any(set(item) != keys for item in items):
            raise ValueError("A batch must use one evidence subset")
        prompts = []
        for item in items:
            question = QUESTION[view] + OPTIONS
            if "text" in item:
                transcript = unicodedata.normalize("NFC", item["text"]).replace("<", "＜").replace(">", "＞")
                question += f"\n自动转写：『{transcript}』"
            content = [{"type": kind} for kind in ("video", "audio") if kind in item]
            content.append({"type": "text", "text": question})
            chat = [{"role": "system", "content": [{"type": "text", "text": "你评估中文短片中可观察的表达。"}]},
                    {"role": "user", "content": content}]
            prompts.append(self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))

        options = {}
        if "audio" in keys:
            waveforms = [item["audio"] for item in items]
            hop = self.processor.feature_extractor.hop_length
            longest = max(map(len, waveforms))
            options["audio"] = waveforms
            options["audio_kwargs"] = {"max_length": ((longest + hop - 1) // hop) * hop}
        if "video" in keys:
            options["videos"] = [item["video"] for item in items]
            options["videos_kwargs"] = {"fps": 1 / FRAME_PERIOD, "do_sample_frames": False,
                                         "size": {"shortest_edge": 224**2, "longest_edge": 224**2}}
        inputs = self.processor(text=prompts, padding=True, return_tensors="pt", **options)
        if ("input_features" in inputs) != ("audio" in keys) or (
                "pixel_values_videos" in inputs) != ("video" in keys):
            raise ValueError("Encoded modalities differ from the requested inputs")
        inputs = {name: value.to(self.device, dtype=self.model.dtype) if value.is_floating_point()
                  else value.to(self.device) for name, value in inputs.items()}
        self.features = {} if capture else None
        try:
            with torch.inference_mode():
                logits = self.model(**inputs, use_cache=False).logits[:, -1].float()
                selected = logits[:, self.choice_ids]
                distribution = selected.softmax(-1).cpu().numpy()
                choice_mass = (selected.logsumexp(-1) - logits.logsumexp(-1)).exp().cpu().tolist()
                pooled = self._pool_features(inputs, len(items)) if capture else None
        finally:
            self.features = None
        if not np.isfinite(distribution).all() or not np.isfinite(choice_mass).all():
            raise ValueError("Teacher returned nonfinite scores")
        predictions = [{"label": CHOICES[int(np.argmax(values))], "scores": dict(zip(CHOICES, values.tolist())),
                        "options_mass": mass, "seen": sorted(keys)}
                       for values, mass in zip(distribution, choice_mass, strict=True)]
        return (predictions, pooled) if capture else predictions
