"""Train equal-width clip heads and freeze all choices using CH-SIMS train/valid only."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
from importlib.metadata import version

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emotion_assistant import foundation_fusion as fusion
from emotion_assistant.foundation_embeddings import AUDIO_SECONDS, TEXT_MODEL, TEXT_DIGEST, TEXT_PREFIX
from benchmark_chsims_conflict import EXTRACT_MODULES, code_signature
from extract_foundation_embeddings import signature

DATA = ROOT / ".runtime" / "chsims-foundation"
FROZEN_MANIFESTS = {
    "raw-cohort.json": "6d74a549d8acfb061729ccd908f05da3e7280e7dc703294ff46e5f0ec971238c",
    "audit-cohort.json": "36ad5a041e0a736562704e6f61797d7e58f9d3c8823173f55a87669c64b0a085",
}
SCENARIOS = {
    "observed": (True, True, True), "hide_text": (False, True, True),
    "hide_audio": (True, False, True), "hide_visual": (True, True, False),
    "only_visual": (False, False, True),
}
PROTOCOL = {
    "version": "chsims-linear-missing-modalities-v1",
    "classes": list(fusion.LABELS),
    "regularization_C": [0.3, 3., 30., 300.],
    "inverse_class_frequency_exponent": [0., 0.5, 1.],
    "head_selection": "maximize mean(accuracy, 3-class macro-F1) on validation; ties: positive recall, then grid order",
    "numeric_precision": "float64 fit; frozen coefficients and predictions use float32 export with float64 arithmetic",
    "solver": {"name": "lbfgs", "max_iterations": 3000, "tolerance": 1e-6},
    "text_head": "2048 inputs: signed normalized 1-3 character n-grams (1024) + either another hash seed (1024) or normalized frozen Qwen3-Embedding-0.6B (1024)",
    "audio_head": "normalized 768-dimensional frozen emotion2vec+ embedding",
    "training_text": "official train captions; select text heads on official validation captions",
    "validation_fusion": "fixed raw validation cohort, file-entry ASR and exact training encoders",
    "audio_visual_weight_grid": "with audio head: [1, s, v], s and v in {0.125, 0.5, 1.5}; without: [0,1,v] same v grid",
    "text_weight_grid": [0.25, 0.5, 1., 2., 4., 8.],
    "fusion_selection": "audio/visual weights fixed using validation without text; then select text weight on mean objective over observed, hide_audio and hide_visual; same AV weights in matched Qwen ablation",
    "decision": "maximum weighted mean class score; omit absent blocks, abstain when no usable inputs",
    "test": "same fixed 48 hash-selected official test videos, no fitting or threshold selection",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def frozen_cohort(data, name):
    path = data / name
    if file_sha(path) != FROZEN_MANIFESTS[name]:
        raise ValueError(f"Manifest differs from the prespecified cohort: {name}")
    return read_json(path)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def embeddings(path, dim, *, expected=None):
    mode = path.stem.removeprefix("embeddings-")
    with np.load(path, allow_pickle=False) as data:
        if data["identity"].item() != signature(mode):
            raise ValueError(f"Embedding cache uses a different encoder or extraction code: {path.name}")
        ids = data["ids"].tolist()
        vectors = data["values"]
        checksums = dict(zip(ids, data["checksums"].tolist(), strict=True))
    if len(ids) != len(set(ids)) or vectors.shape != (len(ids), dim) or not np.isfinite(vectors).all():
        raise ValueError(f"Invalid embedding file: {path.name}")
    if expected is not None and any(checksums.get(k) != sha for k, sha in expected.items()):
        raise ValueError(f"Embedding cache is incomplete or mismatches the requested input: {path.name}")
    return {key: value for key, value in zip(ids, vectors) if np.linalg.norm(value) > 1e-10}


def observation_versions():
    return {"production_code": code_signature((*EXTRACT_MODULES, "clip_inputs.py")),
            "models_manifest": file_sha(ROOT / "models/manifest.json")}


def checksums_for_raw(data, ids):
    text, audio = {}, {}
    versions = observation_versions()
    for key in ids:
        obs = read_json(data / "observations" / (key + ".json"))
        if obs["id"] != key or obs.get("versions") != versions:
            raise ValueError(f"Unexpected ID or extraction version for cached clip {key}")
        text[key] = hashlib.sha256(obs["text"].encode()).hexdigest()
        audio[key] = obs["sha256"]
    return text, audio


def read_rows(data):
    rows = [json.loads(line) for line in (data / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate video/clip ID")
    return rows


def dataset(data, rows, ids, text_emb, audio_emb, *, raw):
    result = []
    for row in rows:
        key = row["id"]
        if key not in ids:
            continue
        obs = read_json(data / "observations" / (key + ".json")) if raw else {}
        text = obs.get("text", "") if raw else row["text"]
        result.append({
            "id": key, "label": row["label"], "text": text,
            "audio": obs.get("audio"), "visual": obs.get("visual"),
            "embedding": text_emb.get(key), "audio_embedding": audio_emb.get(key),
        })
    if {r["id"] for r in result} != set(ids):
        raise ValueError("A requested cohort member is missing")
    return result


def metrics(actual, predictions):
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predictions, dtype=int)
    matrix = np.zeros((3, 4), dtype=int)  # last column = abstention
    np.add.at(matrix, (actual, np.where(predicted < 0, 3, predicted)), 1)
    supported = matrix.sum(axis=1)
    called = matrix[:, :3].sum(axis=0)
    tp = np.diag(matrix[:, :3])
    f1 = 2 * tp / np.maximum(1, supported + called)
    pos = int(supported[2])
    correct, valid, n = int(tp.sum()), int(called.sum()), len(actual)
    return {"n": n, "correct": correct, "wrong": valid - correct, "abstain": n - valid,
            "accuracy": correct/n if n else None, "macro_f1": float(f1.mean()) if n else None,
            "positive_hits": int(tp[2]), "positive_support": pos,
            "positive_recall": float(tp[2]/pos) if pos else None,
            "coverage": valid/n if n else None, "confusion_gold_by_prediction": matrix.tolist()}


def rank(gold, predicted):
    m = metrics(gold, predicted)
    return ((m["accuracy"] + m["macro_f1"])/2, m["positive_recall"] or 0.)


def target(rows):
    return np.array([fusion.LABELS.index(r["label"]) for r in rows], dtype=int)


def inputs(rows, key):
    vectors, indices = [], []
    for i, row in enumerate(rows):
        vector = (fusion.unit(row["audio_embedding"], fusion.AUDIO_DIM) if key == "audio" else
                  fusion.text_input(row["text"], row["embedding"], key == "qwen"))
        if vector is not None:
            vectors.append(vector)
            indices.append(i)
    if not vectors:
        raise ValueError("No usable rows for " + key)
    return np.array(vectors), np.array(indices, dtype=int)


def fit_head(key, train, valid):
    from sklearn.linear_model import LogisticRegression

    X, selected = inputs(train, key)
    V, scored = inputs(valid, key)
    y, held = target(train)[selected], target(valid)
    counts = np.bincount(y, minlength=3)
    if counts.min() < 2:
        raise ValueError("Insufficient training examples per class for " + key)
    center = X.mean(axis=0)
    def normalize(matrix):
        shifted = matrix - center
        return shifted / np.maximum(np.linalg.norm(shifted, axis=1, keepdims=True), 1e-10)
    X, V = normalize(X), normalize(V)
    best, winner = (-1., -1.), None
    for C, exponent in itertools.product(PROTOCOL["regularization_C"], PROTOCOL["inverse_class_frequency_exponent"]):
        cw = (len(y)/(3*counts))**exponent
        solver = PROTOCOL["solver"]
        model = LogisticRegression(C=C, solver=solver["name"], max_iter=solver["max_iterations"],
                                   class_weight=dict(enumerate(cw)), tol=solver["tolerance"])
        model.fit(X, y)
        if np.max(model.n_iter_) >= model.max_iter:
            raise RuntimeError(f"Validation candidate for {key} reached the optimizer iteration limit")
        pred = np.full(len(held), -1, dtype=int)
        pred[scored] = model.predict(V)
        quality = rank(held, pred)
        if quality > best:
            best = quality
            winner = (model, C, exponent, pred)
    model, C, exponent, pred = winner
    params = {key + "_coef": model.coef_.astype(np.float32), key + "_bias": model.intercept_.astype(np.float32),
              key + "_center": center.astype(np.float32)}
    return params, {"train_available": len(y), "train_class_counts": counts.tolist(),
                    "validation_available": len(scored), "C": C, "class_exponent": exponent,
                    "optimizer_iterations": int(np.max(model.n_iter_)),
                    "validation": metrics(held, pred)}


def components(rows, model, variant):
    values = np.zeros((len(rows), 4, 3))
    availability = np.zeros((len(rows), 4))
    for i, row in enumerate(rows):
        use_qwen = variant == "foundation" and row["embedding"] is not None
        x = fusion.text_input(row["text"], row["embedding"], use_qwen)
        tx = fusion.learned_scores(model, "qwen" if use_qwen else "base", x)
        ax = (fusion.learned_scores(model, "audio", fusion.unit(row["audio_embedding"], fusion.AUDIO_DIM))
              if variant != "base" else None)
        a, qa = fusion.cue_scores(row["audio"])
        v, qv = fusion.cue_scores(row["visual"])
        for j, (vector, quality) in enumerate(((tx, 1.), (ax, 1.), (a, qa), (v, qv))):
            if vector is not None:
                values[i, j], availability[i, j] = vector, quality
    return values, availability


def score_blocks(blocks, weights, mask):
    tx, au, vi = mask
    values, availability = blocks
    weights = np.asarray(weights) * (tx, au, au, vi) * availability
    return np.where(weights.sum(axis=1) > 0, (values * weights[:,:,None]).sum(axis=1).argmax(axis=1), -1)


def select_weights(rows, model):
    gold = target(rows)
    av_weights = {}
    result = {}
    for variant in ("base", "no_qwen", "foundation"):
        blocks = components(rows, model, variant)
        kind = "old" if variant == "base" else "encoder"
        if kind not in av_weights:
            options = [(1., s, v) for s, v in itertools.product((.125,.5,1.5), repeat=2)]
            if kind == "old":
                options = [(0., 1., v) for v in (.125,.5,1.5)]
            av_weights[kind] = max(options, key=lambda w: rank(
                gold, score_blocks(blocks, (0., *w), SCENARIOS["hide_text"])))
        def performance(t):
            rewards = [rank(gold, score_blocks(blocks, (t,*av_weights[kind]), SCENARIOS[s]))
                       for s in ("observed", "hide_audio", "hide_visual")]
            return tuple(np.mean(rewards, axis=0))
        text_weight = max(PROTOCOL["text_weight_grid"], key=performance)
        w = np.asarray((text_weight,*av_weights[kind]), dtype=np.float32)
        model[variant + "_weights"] = w
        result[variant] = {"weights": w.tolist(), "scenarios": {
            name: metrics(gold, score_blocks(blocks, w, mask)) for name, mask in SCENARIOS.items()}}
    return result


def train(data, output, record, raw_dir):
    rows = read_rows(data)
    raw = frozen_cohort(data, "raw-cohort.json")
    expected = {r["id"]: hashlib.sha256(r["text"].encode()).hexdigest()
                for r in rows if r["split"] in ("train", "valid")}
    raw_text, raw_audio = checksums_for_raw(data, raw["valid"])
    raw_audio.update({key: file_sha(raw_dir / (key + ".mp4")) for key in raw["train"]})
    text = embeddings(data / "embeddings-text.npz", fusion.TEXT_DIM, expected=expected)
    asr = embeddings(data / "embeddings-text-asr.npz", fusion.TEXT_DIM, expected=raw_text)
    acoustic = embeddings(data / "embeddings-audio.npz", fusion.AUDIO_DIM, expected=raw_audio)
    text_rows, raw_rows = {}, {}
    for split in ("train", "valid"):
        ids = {r["id"] for r in rows if r["split"] == split}
        if not set(raw[split]) <= ids:
            raise ValueError("Raw subset leaks out of official " + split + " split")
        text_rows[split] = dataset(data, rows, ids, text, {}, raw=False)
        if split == "valid":
            raw_rows[split] = dataset(data, rows, set(raw[split]), asr, acoustic, raw=True)
        else:
            # Training the acoustic head requires only real MP4 embeddings and labels.
            train_ids = set(raw["train"])
            raw_rows[split] = [{"id": r["id"], "label": r["label"], "audio_embedding": acoustic.get(r["id"])}
                               for r in rows if r["id"] in train_ids]
    if any(r["text"] and fusion.HAN.search(r["text"]) and r["embedding"] is None for r in raw_rows["valid"]):
        raise ValueError("Validation ASR embeddings incomplete")
    model = {"version": np.array(fusion.VERSION), "labels": np.asarray(fusion.LABELS)}
    heads = {}
    for key in ("base", "qwen", "audio"):
        split = raw_rows if key == "audio" else text_rows
        params, heads[key] = fit_head(key, split["train"], split["valid"])
        model.update(params)
        print("Fitted", key, heads[key]["train_available"], flush=True)
    scores = select_weights(raw_rows["valid"], model)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **model)
    from emotion_assistant.models import VERSIONS
    write_json(record, {"protocol": PROTOCOL, "model_sha256": file_sha(output),
                        "encoders": {"text": TEXT_MODEL, "ollama_manifest_digest": TEXT_DIGEST, "instruction": TEXT_PREFIX,
                                     "audio": VERSIONS["audio"], "max_audio_seconds": AUDIO_SECONDS},
                        "extractor_signatures": {key: signature(key) for key in ("text", "text-asr", "audio")},
                        "file_entry_extractor": observation_versions(),
                        "fit_library_versions": {p: version(p) for p in ("scikit-learn","numpy","scipy")},
                        "raw_counts": {k: len(raw[k]) for k in ("train","valid")}, "heads": heads,
                        "validation_fusion": scores})


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--raw", type=Path, default=ROOT / ".runtime/chsims/Raw")
    parser.add_argument("--output", type=Path, default=fusion.ASSET)
    parser.add_argument("--record", type=Path, default=DATA / "fit.json")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        train(args.data, args.output, args.record, args.raw)
