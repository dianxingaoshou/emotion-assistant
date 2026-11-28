"""Prepare real clips, run Qwen2.5-Omni and the desktop baseline, and score paired results."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.chsims_teacher_data import (
    CACHE, LABELS, SEED, SPLITS, VIEWS, evidence_for, make_preparer, media_path, ranked_ids, read_rows, sha_file,
    write_atomic,
)


def save_json(path, result):
    write_atomic(path, (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf8"))


def prepared(cache, key):
    with np.load(cache / "media" / f"{key}.npz", allow_pickle=False) as asset:
        return json.loads(asset["info"].item()), asset["pcm"], asset["video"], asset["speech"]


def prepare(cache, data, keys, *, device="cpu"):
    started = perf_counter()
    extract = make_preparer(cache, device=device)
    for index, key in enumerate(keys, 1):
        extract(media_path(data, key), cache / "media" / f"{key}.npz")
        if index % 24 == 0 or index == len(keys):
            print(f"Prepared {index}/{len(keys)} in {perf_counter() - started:.1f}s", flush=True)


def score_teacher(cache, keys, batch_size, *, representations=False):
    from scripts.chsims_omni_teacher import OmniTeacher

    started = perf_counter()
    model_dir = cache / "omni-3b"
    teacher = OmniTeacher(model_dir if model_dir.is_dir() else CACHE / "omni-3b")
    print(f"Teacher ready in {perf_counter() - started:.1f}s on {teacher.identity['gpu']}", flush=True)
    identity = {**teacher.identity, "representations": representations}
    save_json(cache / "teacher-model.json", identity)
    for start in range(0, len(keys), batch_size):
        batches, outputs, features = defaultdict(list), {}, defaultdict(dict)
        for key in keys[start:start + batch_size]:
            path = cache / "teacher" / f"{key}.json"
            feature_path = cache / "media" / f"{key}.npz"
            representation_path = cache / "representations" / f"{key}.npz"
            signature = sha_file(feature_path)
            if path.is_file():
                previous = json.loads(path.read_text("utf8"))
                if (previous["media_sha256"] == signature and previous["model"] == identity
                        and (not representations or representation_path.is_file()
                             and previous.get("representations_sha256") == sha_file(representation_path))):
                    continue
            info, pcm, video, _ = prepared(cache, key)
            outputs[key] = {"media_sha256": signature, "model": identity, "predictions": {}}
            for view in VIEWS:
                evidence = evidence_for(view, info["text"], pcm, video)
                if evidence:
                    batches[(view, tuple(evidence))].append((key, evidence))
                else:
                    outputs[key]["predictions"][view] = {"label": "abstain", "scores": None, "seen": []}
        for (view, _), items in batches.items():
            scored = teacher.score(view, [data for _, data in items], capture=representations)
            scores, vectors = scored if representations else (scored, [{}] * len(items))
            for (key, _), result, blocks in zip(items, scores, vectors, strict=True):
                outputs[key]["predictions"][view] = result
                features[key].update({f"{view}_{name}": block for name, block in blocks.items()})
        for key, result in outputs.items():
            if representations:
                buffer = io.BytesIO()
                np.savez(buffer, **features[key])
                payload = buffer.getvalue()
                write_atomic(cache / "representations" / f"{key}.npz", payload)
                result["representations_sha256"] = hashlib.sha256(payload).hexdigest()
            save_json(cache / "teacher" / f"{key}.json", result)
        done = min(start + batch_size, len(keys))
        if done % 24 < batch_size or done == len(keys):
            print(f"Teacher {done}/{len(keys)} in {perf_counter() - started:.1f}s", flush=True)


def score_baseline(cache, keys):
    from emotion_assistant.chinese_affect import HAN
    from emotion_assistant.foundation_embeddings import TEXT_DIGEST, embed_texts
    from emotion_assistant.foundation_fusion import ASSET, predict_foundation_clip

    identity = {"name": "desktop fixed weight v1", "classifier_sha256": sha_file(ASSET),
                "embedding_digest": TEXT_DIGEST, "implementation_sha256": sha_file(Path(__file__))}
    save_json(cache / "baseline-model.json", identity)
    for start in range(0, len(keys), 24):
        pending, chunks = [], {}
        for key in keys[start:start + 24]:
            destination = cache / "baseline" / f"{key}.json"
            signature = sha_file(cache / "media" / f"{key}.npz")
            if destination.is_file():
                previous = json.loads(destination.read_text("utf8"))
                if previous["media_sha256"] == signature and previous["model"] == identity:
                    continue
            info, _, _, speech = prepared(cache, key)
            chunks[key] = (signature, info, speech if speech.size else None)
            if HAN.search(info["text"]):
                pending.append((key, info["text"]))
        embeddings = embed_texts([text for _, text in pending]) if pending else []
        text_vectors = dict(zip((key for key, _ in pending), embeddings, strict=True))
        for key, (signature, info, speech) in chunks.items():
            predictions = {}
            for view, channels in VIEWS.items():
                text = info["text"] if "text" in channels else ""
                audio = "audio" in channels
                result = predict_foundation_clip(text, audio=info["audio"] if audio else None,
                                                 visual=info["visual"] if "video" in channels else None,
                                                 embedding=text_vectors.get(key) if text else None,
                                                 audio_embedding=speech if audio else None)
                if result["status"] not in ("ok", "insufficient_evidence"):
                    raise RuntimeError(f"Unexpected baseline status for {key}: {result['status']}")
                predictions[view] = {"label": result["label"] if result["label"] in LABELS else "abstain",
                                     "seen": result["used_modalities"], "scores": result["scores"]}
            save_json(cache / "baseline" / f"{key}.json", {"media_sha256": signature, "model": identity,
                                                         "predictions": predictions})
        print(f"Baseline {min(start + 24, len(keys))}/{len(keys)}", flush=True)


CLASSES = (*LABELS, "abstain")


def scores_from_confusion(matrix):
    true = matrix.sum(axis=-1)
    predicted = matrix.sum(axis=-2)[..., :3]
    hits = np.diagonal(matrix[..., :3], axis1=-2, axis2=-1)
    denominator = true + predicted
    f1 = np.divide(2 * hits, denominator, out=np.zeros(denominator.shape, float), where=denominator > 0)
    return np.stack((hits.sum(axis=-1) / true.sum(axis=-1), f1.mean(axis=-1)), axis=-1)


def paired_metrics(rows, view):
    if not rows:
        return None
    sources = sorted({key.split("/")[0] for key, *_ in rows})
    source_index = {source: idx for idx, source in enumerate(sources)}
    # Preserve dependence between clips from the same source during resampling.
    matrices = np.zeros((len(sources), 2, 3, 4), dtype=int)
    corrected = regressed = 0
    for key, truth, baseline, teacher in rows:
        label = truth[view]
        actual, first, second = LABELS.index(label), CLASSES.index(baseline[view]["label"]), CLASSES.index(teacher[view]["label"])
        block = matrices[source_index[key.split("/")[0]]]
        block[0, actual, first] += 1
        block[1, actual, second] += 1
        corrected += first != actual and second == actual
        regressed += first == actual and second != actual
    total = matrices.sum(axis=0)
    scores = scores_from_confusion(total)
    result = {"n": len(rows), "sources": len(sources), "corrected": corrected, "regressed": regressed}
    for idx, method in enumerate(("baseline", "teacher")):
        mat = total[idx]
        result[method] = {"accuracy": float(scores[idx, 0]), "macro_f1": float(scores[idx, 1]),
                          "correct": int(mat.trace()), "abstain": int(mat[:, -1].sum()), "confusion": mat.tolist(),
                          "recall": {label: {"hits": int(mat[i, i]), "total": int(mat[i].sum())}
                                     for i, label in enumerate(LABELS)}}
    rng = np.random.default_rng(813)
    samples = rng.multinomial(len(sources), [1 / len(sources)] * len(sources), size=2000)
    resampled = scores_from_confusion((samples @ matrices.reshape(len(sources), -1)).reshape(-1, 2, 3, 4))
    intervals = np.quantile(resampled[:, 1] - resampled[:, 0], [.025, .975], axis=0).T
    result["teacher_minus_baseline_95_source_bootstrap"] = dict(zip(("accuracy", "macro_f1"), intervals.tolist()))
    return result


def evaluate(cache, keys, truth, data, split, output):
    identities = {name: json.loads((cache / f"{name}-model.json").read_text("utf8"))
                  for name in ("baseline", "teacher")}
    inputs, rows = Counter(), []
    for key in keys:
        hash_media = sha_file(cache / "media" / f"{key}.npz")
        predictions = []
        for name, identity in identities.items():
            record = json.loads((cache / name / f"{key}.json").read_text("utf8"))
            if record["media_sha256"] != hash_media or record["model"] != identity:
                raise ValueError(f"Inconsistent {name} cache for {key}")
            predictions.append(record["predictions"])
        _, teacher = predictions
        for view, allowed in VIEWS.items():
            seen = teacher[view]["seen"]
            if set(seen) - set(allowed):
                raise ValueError(f"Teacher received masked evidence for {key} {view}")
            inputs[view] += bool(seen)
        inputs["full_three"] += len(teacher["full"]["seen"]) == 3
        rows.append((key, truth[key], *predictions))
    differ = [row for row in rows if row[1]["audio"] != row[1]["text"]]
    independent = [row for row in rows if len(row[3]["full"]["seen"]) == 3]
    report = {"split": split, "meta_sha256": sha_file(data / "meta.csv"), "selection_seed": SEED,
              "selected_ids_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
              "split_size": len(truth), "used_teacher_inputs": dict(inputs), "models": identities,
              "results": {view: paired_metrics(rows, view) for view in VIEWS},
              "full_three_present_results": paired_metrics(independent, "full"),
              "audio_vs_text_annotation_disagreement": paired_metrics(differ, "audio"),
              "audio_prediction_matches_text_on_disagreement": {
                  name: sum(row[i]["audio"]["label"] == row[1]["text"] for row in differ)
                  for i, name in ((2, "baseline"), (3, "teacher"))},
              "rows": [{"id": key, "truth": labels, "baseline": baseline, "teacher": teacher}
                       for key, labels, baseline, teacher in rows]}
    save_json(output, report)
    for view, result in report["results"].items():
        print(view, {name: (result[name]["correct"], round(result[name]["macro_f1"], 4))
                     for name in identities}, f"/ {result['n']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("prepare", "teacher", "baseline", "evaluate"))
    parser.add_argument("--data", required=True, type=Path, help="Directory with meta.csv and Raw")
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prepare-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--representations", action="store_true", help="Also cache frozen Omni features")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = read_rows(args.data)
    keys = ranked_ids(rows, args.split)
    end = args.start + args.limit if args.limit is not None else len(keys)
    if not 0 <= args.start < end or args.start >= len(keys) or args.batch_size < 1:
        parser.error("Select a nonempty interval and positive batch size")
    keys = keys[args.start:end]
    if args.step == "prepare":
        prepare(args.cache, args.data, keys, device=args.prepare_device)
    elif args.step == "teacher":
        score_teacher(args.cache, keys, args.batch_size, representations=args.representations)
    elif args.step == "baseline":
        score_baseline(args.cache, keys)
    else:
        output = args.output or args.cache / f"evaluation-{args.split}.json"
        evaluate(args.cache, keys, rows[args.split], args.data, args.split, output)


if __name__ == "__main__":
    main()
