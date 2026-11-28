"""Fine-tune the final Qwen2.5-Omni audio block and compare matched audio baselines."""
import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1] / ".venv/Lib/site-packages"))
import numpy as np
from scipy.special import softmax
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from torch.nn import functional as F

from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_audio_prior import audio_logits
from scripts.chsims_teacher_cross_source import group_ids, labels, metrics, mix
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file, write_atomic
from scripts.chsims_teacher_supervision import compare, numbers, read_blocks, read_full_report, temperature
from scripts.chsims_omni_audio_encoder import AudioTower


MODES = ("desktop_matched", "qwen_frozen", "qwen_lora")
EPOCHS = (3, 6, 10)
BATCH_SIZE = 32
SEED = 20261128
NEUTRAL = LABELS.index("neutral")


@dataclass
class Clip:
    tokens: np.ndarray
    chunks: tuple
    context: np.ndarray
    acoustic: np.ndarray
    desktop: np.ndarray
    available: bool


def expert_features(info, speech):
    observed = info.get("audio")
    if observed:
        cue = observed["scores"]
        scores = [cue[label] for label in (*LABELS, "uncertain")]
        quality = [observed["quality"], observed["coverage"], 1.]
    else:
        scores, quality = [0.] * 4, [0.] * 3
    return np.concatenate((speech if speech.size else np.zeros(768, np.float32),
                           scores, quality, [float(bool(speech.size))])).astype(np.float32)


def read_clip(cache, key):
    media = cache / "media" / f"{key}.npz"
    with np.load(media, allow_pickle=False) as inp:
        pcm = inp["pcm"]
        desktop = expert_features(json.loads(inp["info"].item()), inp["speech"])
    with np.load(cache / "audio-tokens" / f"{key}.npz", allow_pickle=False) as entry:
        receipt = json.loads(entry["info"].item())
        if hashlib.sha256(np.ascontiguousarray(pcm)).hexdigest() != receipt["audio_sha256"]:
            raise ValueError(f"Audio token cache changed for {key}")
        available = "tokens" in entry
        if not available:
            return Clip(np.empty((0, 1280), np.float16), (), np.zeros(2048, np.float32),
                        np.zeros(2560, np.float32), desktop, False)
        tokens, chunks, acoustic = entry["tokens"], tuple(int(x) for x in entry["chunks"]), entry["middle"]
        if not chunks or max(chunks) > 100 or min(chunks) < 1 or sum(chunks) != len(tokens):
            raise ValueError(f"Unexpected encoder sequence at {key}")
    with np.load(cache / "representations" / f"{key}.npz", allow_pickle=False) as entry:
        context = entry["audio_context"]
    return Clip(tokens, chunks, context, acoustic, desktop, True)


def continuous_labels(data):
    with (data / "meta.csv").open(encoding="utf-8-sig", newline="") as stream:
        return {f"{row['video_id']}/{row['clip_id']}": float(row["label_A"]) for row in csv.DictReader(stream)}


class RankAdapter(nn.Module):
    def __init__(self, width, rank=8):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, features):
        return 2. * self.up(self.down(features))


class AudioHead(nn.Module):
    def __init__(self, tail, mode):
        super().__init__()
        self.mode = mode
        self.tail = tail
        width = tail.embed_dim
        self.adapter = nn.ModuleDict({name: RankAdapter(width) for name in ("q", "v", "o")}) \
            if mode == "qwen_lora" else nn.ModuleDict()
        if mode == "desktop_matched":
            self.input_norm = nn.LayerNorm(776)
        else:
            self.input_norm = nn.LayerNorm(2 * width)
            self.middle_norm = nn.LayerNorm(2 * width)
            self.context_norm = nn.LayerNorm(2048)
        self.head = nn.Sequential(nn.Linear(776 if mode == "desktop_matched" else 4 * width + 2048, 96),
                                  nn.GELU(), nn.Dropout(.18), nn.Linear(96, 4))

    def projected(self, features, layer, name):
        values = layer(features)
        if name in self.adapter:
            values = values + self.adapter[name](features)
        return values

    def tail_features(self, chunks, sizes, owner, n):
        block = self.tail
        attention = block.self_attn
        values = block.self_attn_layer_norm(chunks)

        def reshape(v):
            return v.unflatten(-1, (attention.num_heads, attention.head_dim)).transpose(1, 2)

        query = reshape(self.projected(values, attention.q_proj, "q"))
        key = reshape(attention.k_proj(values))
        value = reshape(self.projected(values, attention.v_proj, "v"))
        keep = torch.arange(chunks.shape[1], device=chunks.device)[None, :] < sizes[:, None]
        attention_result = F.scaled_dot_product_attention(query, key, value, attn_mask=keep[:, None, None, :])
        output = chunks + self.projected(attention_result.transpose(1, 2).flatten(2), attention.out_proj, "o")
        value = block.activation_fn(block.fc1(block.final_layer_norm(output)))
        output = output + block.fc2(value)
        # Each chunk attends locally. Pool original frames only, with the source
        # boundary preserved even when different clips share a batch.
        values = output.float() * keep[:, :, None]
        mass = values.new_zeros(n).index_add(0, owner, sizes.float()).clamp_min(1.)[:, None]
        mean = values.new_zeros((n, block.embed_dim)).index_add(0, owner, values.sum(1)) / mass
        square = torch.zeros_like(mean).index_add(0, owner, values.square().sum(1)) / mass
        return torch.cat((mean, (square - mean.square()).clamp_min(1e-5).sqrt()), dim=-1)

    def forward(self, chunks, sizes, owner, middle, context, desktop):
        if self.mode == "desktop_matched":
            features = self.input_norm(desktop)
        else:
            pulse = self.tail_features(chunks, sizes, owner, len(context))
            features = torch.cat((self.input_norm(pulse), self.middle_norm(middle), self.context_norm(context)), -1)
        result = self.head(features).float()
        return result[:, :3], result[:, 3].tanh()


def batch_arrays(clips, device, desktop_only):
    desktop = torch.from_numpy(np.stack([clip.desktop for clip in clips])).to(device)
    if desktop_only:
        return None, None, None, None, None, desktop
    chunks, sizes, owner = [], [], []
    for index, clip in enumerate(clips):
        offset = 0
        for length in clip.chunks:
            chunks.append(clip.tokens[offset:offset + length])
            sizes.append(length)
            owner.append(index)
            offset += length
    packed = np.zeros((len(chunks), max(sizes), chunks[0].shape[1]), np.float16)
    for i, x in enumerate(chunks):
        packed[i, :len(x)] = x
    return (torch.from_numpy(packed).to(device, dtype=torch.bfloat16), torch.tensor(sizes, device=device),
            torch.tensor(owner, device=device), torch.from_numpy(np.stack([c.acoustic for c in clips])).to(device),
            torch.from_numpy(np.stack([c.context for c in clips])).to(device), desktop)


def supported(clips, mode):
    if mode == "desktop_matched":
        return np.array([bool(c.desktop[-1] or c.desktop[-2]) for c in clips])
    return np.array([c.available for c in clips])


def batches(indices, batch_size=BATCH_SIZE):
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def model_for(tower, mode):
    return AudioHead(tower.encoder.layers[-1], mode).to(tower.device)


def predict(model, clips, indices, device):
    guess = np.full((len(indices), len(LABELS)), np.nan)
    observed = np.flatnonzero(supported(clips, model.mode)[indices])
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in batches(observed):
            values = batch_arrays([clips[i] for i in indices[batch]], device, model.mode == "desktop_matched")
            guess[batch] = model(*values)[0].cpu().numpy()
    return guess


def step_weights(truth, sources):
    _, groups, group_count = np.unique(sources, return_inverse=True, return_counts=True)
    group_weight = group_count[groups].astype(float) ** -.3
    class_weight = np.bincount(truth, minlength=3)[truth].astype(float) ** -.25
    combined = group_weight * class_weight
    return (combined / combined.mean()).astype(np.float32)


def fit_head(tower, mode, clips, y, intensity, sources, candidates, epochs, validation=None):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    indices = candidates[supported(clips, mode)[candidates]]
    model = model_for(tower, mode)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1.5e-4, weight_decay=.035, fused=True)
    weights = np.zeros(len(clips), np.float32)
    weights[indices] = step_weights(y[indices], sources[indices])
    results = {}
    for epoch in range(1, epochs + 1):
        model.train()
        model.tail.eval()
        for batch in batches(rng.permutation(indices)):
            values = batch_arrays([clips[i] for i in batch], tower.device, mode == "desktop_matched")
            target = torch.from_numpy(y[batch]).to(tower.device, dtype=torch.long)
            valence = torch.from_numpy(intensity[batch]).to(tower.device, dtype=torch.float32)
            mass = torch.from_numpy(weights[batch]).to(tower.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction, auxiliary = model(*values)
                primary = F.cross_entropy(prediction, target, reduction="none")
                loss = ((primary + .45 * F.smooth_l1_loss(auxiliary, valence, reduction="none", beta=.2))
                        * mass).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.)
            optimizer.step()
        if validation is not None and epoch in EPOCHS:
            examples, selected = validation
            results[epoch] = predict(model, examples, selected, tower.device)
    return model, results


def select_calibration(validation, y, sources, baseline):
    groups = [sources == source for source in sorted(set(sources))]
    baseline_stats = metrics(y, baseline)
    min_neutral = baseline_stats["confusion"][NEUTRAL][NEUTRAL]
    options = {}
    for epoch, logits in validation.items():
        present = np.isfinite(logits[:, 0])
        scale = temperature(logits[present], y[present])
        for offset in (-.35, 0., .35, .7, 1.05):
            p = calibrated(logits, scale, offset)
            stats = metrics(y, p)
            balanced_sources = np.mean([metrics(y, p, mask)["accuracy"] for mask in groups])
            eligible = stats["confusion"][NEUTRAL][NEUTRAL] >= min_neutral
            rank = (eligible, .50 * stats["macro_f1"] + .25 * stats["accuracy"] + .25 * balanced_sources,
                    -abs(offset), -epoch)
            options[epoch, offset] = (rank, scale, stats)
    (epoch, offset), (_, scale, stats) = max(options.items(), key=lambda item: item[1][0])
    return {"epochs": epoch, "offset": offset, "temperature": scale, "validation": stats,
            "minimum_neutral_validation_hits": min_neutral}


def calibrated(logits, scale, offset):
    scores = logits / scale
    scores[:, NEUTRAL] += offset
    return softmax(scores, axis=1)


def reference_logits(cache, report, head):
    desktop = read_blocks(cache, report)["audio"]["baseline"]
    return audio_logits(desktop, head, np.arange(len(report["rows"])))


def fit(data, cache):
    train, valid = (read_full_report(cache, split) for split in ("train", "valid"))
    if train["models"] != valid["models"] or train["meta_sha256"] != sha_file(data / "meta.csv"):
        raise ValueError("Training/validation releases differ")
    tr, va = (r["rows"] for r in (train, valid))
    y, vy = (numbers(rows, "audio") for rows in (tr, va))
    sources, validation_sources = map(group_ids, (tr, va))
    valence = continuous_labels(data)
    intensity = np.array([valence[row["id"]] for row in tr], np.float32)
    np.testing.assert_array_equal(np.sign(intensity).astype(int), y - 1)
    clips, validation = ([read_clip(cache, row["id"]) for row in rows] for rows in (tr, va))
    tower = AudioTower(CACHE / "omni-3b")
    legacy = json.loads((cache / "teacher-supervision.json").read_text("utf-8"))["heads"]["audio"]
    original_desktop = legacy["desktop_calibrated"]
    baseline = softmax(reference_logits(cache, valid, original_desktop) / original_desktop["temperature"], axis=1)
    folds = list(GroupKFold(n_splits=3).split(vy, groups=validation_sources))
    frozen, probabilities = {}, {"desktop_calibrated": baseline}
    path = cache / "audio-lora"
    path.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        start = perf_counter()
        grouped = {epoch: np.full((len(validation), 3), np.nan) for epoch in EPOCHS}
        for fold, (_, evaluate_index) in enumerate(folds, 1):
            allowed = np.flatnonzero(~np.isin(sources, validation_sources[evaluate_index]))
            _, results = fit_head(tower, mode, clips, y, intensity, sources, allowed,
                                  max(EPOCHS), (validation, evaluate_index))
            for epoch, result in results.items():
                grouped[epoch][evaluate_index] = result
            print(f"{mode} fold {fold} completed in {perf_counter() - start:.1f}s", flush=True)
        method = select_calibration(grouped, vy, validation_sources, baseline)
        probabilities[mode] = calibrated(grouped[method["epochs"]], method["temperature"], method["offset"])
        model, _ = fit_head(tower, mode, clips, y, intensity, sources, np.arange(len(clips)), method["epochs"])
        from safetensors.torch import save

        lightweight = {name: v.cpu() for name, v in model.state_dict().items() if not name.startswith("tail.")}
        weight_path = path / f"{mode}.safetensors"
        write_atomic(weight_path, save(lightweight))
        method["weights_sha256"] = sha_file(weight_path)
        method["adapter_parameters"] = sum(p.numel() for p in model.adapter.parameters())
        if model.adapter:
            update = sum(float(part.up.weight.detach().norm().cpu()) for part in model.adapter.values())
            if not np.isfinite(update) or update <= 0:
                raise RuntimeError("LoRA did not update the audio encoder")
            method["adapter_up_norm"] = update
        frozen[mode] = method
        print(mode, method, flush=True)
    # Require a positive contribution from Omni in each fusion arm. Keep the
    # plain desktop result as a separate matched control.
    blend = []
    for desk in ("desktop_matched", "desktop_calibrated"):
        for weight in (.25, .5, .75):
            policy = {"qwen_lora": weight, desk: 1 - weight}
            result = metrics(vy, mix(probabilities, policy))
            supported_neutral = (result["confusion"][NEUTRAL][NEUTRAL]
                                 >= frozen["qwen_lora"]["minimum_neutral_validation_hits"])
            blend.append(((supported_neutral, .6 * result["macro_f1"] + .4 * result["accuracy"]), policy))
    selected = max(blend, key=lambda row: row[0])[1]
    receipt = {"model": tower.identity, "source": sha_file(Path(__file__)), "arms": frozen,
               "development": {s: sha_file(cache / f"evaluation-{s}.json") for s in ("train", "valid")},
               "test_protocol": {"groups": ["official", "unseen_sources", "audio_text_disagreement"],
                                 "primary_metrics": ["accuracy", "macro_f1", "neutral_recall"]},
               "meta_sha256": train["meta_sha256"], "split_sizes": [len(tr), len(va)],
               "blend": selected, "blend_validation": metrics(vy, mix(probabilities, selected))}
    save_json(path / "model.json", receipt)
    print("Frozen", selected, sha_file(path / "model.json"), flush=True)


def custom_predictions(cache, clips, tower):
    from safetensors.torch import load_file

    path = cache / "audio-lora"
    record = json.loads((path / "model.json").read_text("utf-8"))
    if record["source"] != sha_file(Path(__file__)) or record["model"] != tower.identity:
        raise ValueError("Audio adaptation provenance differs")
    values = {}
    for mode, model_settings in record["arms"].items():
        weights = path / f"{mode}.safetensors"
        if sha_file(weights) != model_settings["weights_sha256"]:
            raise ValueError(f"Audio adaptation weights changed for {mode}")
        model = model_for(tower, mode)
        incompatible = model.load_state_dict(load_file(weights), strict=False)
        if incompatible.unexpected_keys or any(not name.startswith("tail.") for name in incompatible.missing_keys):
            raise ValueError(f"Incompatible adapted weights for {mode}")
        logits = predict(model, clips, np.arange(len(clips)), tower.device)
        values[mode] = calibrated(logits, model_settings["temperature"], model_settings["offset"])
    return values, record


def evaluate(cache, output):
    test = read_full_report(cache, "test")
    rows = test["rows"]
    tower = AudioTower(CACHE / "omni-3b")
    clips = [read_clip(cache, row["id"]) for row in rows]
    predictions, model = custom_predictions(cache, clips, tower)
    test_ids = {row["id"] for row in rows}
    known = set()
    for split, digest in model["development"].items():
        history = cache / f"evaluation-{split}.json"
        if sha_file(history) != digest:
            raise ValueError("The development reports changed")
        fit_rows = read_full_report(cache, split)["rows"]
        if test_ids & {row["id"] for row in fit_rows}:
            raise ValueError("Test and development IDs overlap")
        known.update(group_ids(fit_rows))
    if test["meta_sha256"] != model["meta_sha256"]:
        raise ValueError("The test release changed")
    old = json.loads((cache / "teacher-supervision.json").read_text("utf-8"))["heads"]["audio"]
    desktop = read_blocks(cache, test)["audio"]["baseline"]
    indices = np.arange(len(rows))
    for name in ("desktop_calibrated", "desktop_acoustic"):
        predictions[name] = softmax(audio_logits(desktop, old[name], indices) / old[name]["temperature"], axis=1)
    predictions["lora_and_desktop"] = mix(predictions, model["blend"])
    prior = json.loads((cache / "audio-prior-test-predictions.json").read_text("utf-8"))["predictions"]
    named = {name: labels(value) for name, value in predictions.items()}
    named["qwen_prior"] = [prior[row["id"]]["omni_prior"] for row in rows]
    y = numbers(rows, "audio")
    selected = {"official": indices, "unseen_sources": np.flatnonzero(~np.isin(group_ids(rows), list(known))),
                "audio_text_disagreement": np.flatnonzero([r["truth"]["audio"] != r["truth"]["text"] for r in rows])}
    report = {"checkpoint": model, "test_ids_sha256": test["selected_ids_sha256"], "groups": {}}
    for name, members in selected.items():
        comparisons = {arm: {reference: compare(rows, "audio", named[reference], named[arm], members)
                             for reference in ("desktop_calibrated", "desktop_matched", "desktop_acoustic",
                                               "qwen_prior", "qwen_frozen")}
                       for arm in ("qwen_lora", "lora_and_desktop")}
        report["groups"][name] = {"comparisons": comparisons,
                                   "metrics": {arm: metrics(y, p, members) for arm, p in predictions.items()}}
        print(name, [(arm, stats["correct"], round(stats["macro_f1"], 4),
                      stats["predicted_neutral"]) for arm, stats in report["groups"][name]["metrics"].items()],
              flush=True)
    save_json(output, report)
    save_json(cache / "audio-lora" / "test-predictions.json",
              {"ids": [row["id"] for row in rows], "labels": named})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("fit", "evaluate"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=4, user_api="blas"):
        if args.step == "fit":
            if args.data is None:
                parser.error("fit requires --data")
            fit(args.data, args.cache)
        else:
            evaluate(args.cache, args.output or args.cache / "audio-lora" / "results.json")


if __name__ == "__main__":
    main()
