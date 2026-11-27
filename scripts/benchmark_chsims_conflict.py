"""Paired, label-blind Qwen conflict review on the same original CH-SIMS test videos."""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from math import comb
import os
from pathlib import Path
import random
import sys
import time
import urllib.request

from chsims_samples import CACHE, ROOT, SAMPLE_SEED, cohort

sys.path.insert(0, str(ROOT))
LABELS = ("negative", "neutral", "positive")
CONDITIONS = ("observed", "qwen_text_only", "hide_audio", "hide_visual", "hide_text")
EXTRACT_MODULES = ("local_media.py", "content_timeline.py", "conflict_review.py", "models.py",
                   "chinese_dimensions.py", "chinese_affect.py", "chinese_emotions.py",
                   "core.py", "capture.py", "file_tracks.py", "native_expression.py", "fusion_policy.py",
                   "english_polarity.py", "expression_profile.py", "content_fusion.py",
                   "affect_dimensions.py", "semantic_context.py", "single_cue.py", "calibration.py")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def code_signature(names):
    return digest(b"".join(name.encode() + b"\0" + (ROOT / "emotion_assistant" / name).read_bytes().replace(
        b"\r\n", b"\n") + b"\0" for name in names))


def extraction_identity():
    models_dir = ROOT / "models"
    manifest = json.loads((models_dir / "manifest.json").read_text(encoding="utf8"))
    files = {}
    for item in manifest["assets"]:
        path = models_dir / item["path"]
        if sha_file(path) != item["sha256"]:
            raise ValueError(f"Local model differs from its manifest: {item['path']}")
        files["models/" + item["path"]] = item["sha256"]
    for path in sorted((ROOT / "emotion_assistant/assets").iterdir()):
        if path.suffix in (".npz", ".gz", ".json") and path.name != "design-palettes.json":
            files[path.name] = (digest(path.read_bytes().replace(b"\r\n", b"\n"))
                                if path.suffix == ".json" else sha_file(path))
    for name in ("fusion-policy.json", "calibration.json"):
        path = ROOT / "data" / name
        files["data/" + name] = digest(path.read_bytes().replace(b"\r\n", b"\n")) if path.exists() else None
    versions = {}
    for package in ("numpy", "torch", "onnxruntime", "funasr", "av", "opencv-python", "opencv-python-headless"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return {"source_sha256_lf": code_signature(EXTRACT_MODULES), "assets": files, "libraries": versions}


def check_ollama_manifest(identity):
    model_dir = Path(os.environ.get("OLLAMA_MODELS", ROOT / "models/ollama"))
    model_name, tag = identity["name"].split(":")
    manifest = model_dir / "manifests/registry.ollama.ai/library" / model_name / tag
    if sha_file(manifest) != identity["digest"]:
        raise ValueError("The recorded Qwen digest and local model manifest differ")
    spec = json.loads(manifest.read_text(encoding="utf8"))
    for part in [spec["config"], *spec["layers"]]:
        checksum = part["digest"].removeprefix("sha256:")
        if sha_file(model_dir / "blobs" / f"sha256-{checksum}") != checksum:
            raise ValueError("A local Ollama model blob differs from the official manifest")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temp.replace(path)


def extract(data, ids, signature):
    from emotion_assistant.paths import MODELS, configure_environment
    configure_environment()
    import cv2
    import numpy as np
    import torch
    from emotion_assistant.models import ModelHub
    from emotion_assistant.local_media import analyze_media

    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    hub = ModelHub(MODELS, enable_asr=True)
    ready = {**hub.load_visual(), **hub.load_audio()}
    if not all(item["ready"] for item in ready.values()):
        raise RuntimeError(str(ready))
    for index, key in enumerate(ids, 1):
        path = data / "Raw" / f"{key}.mp4"
        checksum = sha_file(path)
        out = data / "observations" / f"{key}.json"
        if out.exists():
            cached = json.loads(out.read_text(encoding="utf8"))
            if (cached["sha256"], cached["code_sha256"]) == (checksum, signature) and "av" in cached:
                continue
        seed = int(checksum[:8], 16)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        result = analyze_media(path, model_dir=MODELS, model_hub=hub, transcribe=True,
                               on_sample=lambda *_: None, cancelled=lambda: False)
        if any(str(t.get("reason", "")).startswith("inference_error") for t in result["transcripts"]):
            raise RuntimeError(f"ASR error for {key}")
        utterances = {(t["segment_id"], t["t_start"], t["t_end"]): t for t in result["transcripts"]
                      if t.get("valid") and not t.get("stale") and t.get("text")}
        windows = []
        for row in result["content_intervals"]:
            if not row.get("conflict_evidence"):
                continue
            anchor = (row["conflict_segment_id"], row["t_start"], row["t_end"])
            utterance = utterances.get(anchor)
            if utterance is None:
                raise ValueError(f"Missing exact production transcript join for {key}")
            windows.append({"duration": row["t_end"] - row["t_start"], "text": utterance["text"],
                            "evidence": row["conflict_evidence"]})
        av = {"audio_detected": result["audio_valid_count"] > 0, "visual_detected": result["valid_count"] > 0,
              "accepted_seconds": result.get("timeline_coverage", {}).get("accepted_s", 0)}
        save(out, {"sha256": checksum, "code_sha256": signature, "windows": windows, "av": av,
                   "legacy": result["content_summary"]["label"], "available_asr": bool(utterances)})
        print(f"Extracted {index}/{len(ids)} {key}: {len(windows)} utterances", flush=True)


def masked(window, condition):
    # Mask after feature extraction. In hide_audio the identical decoded text stays available.
    raw = window["evidence"]
    if condition == "observed":
        return window["text"], raw
    if condition == "qwen_text_only":
        # Preserve the selection made from the full cues. The model gets only the available text cue.
        cues = {k: v for k, v in raw["cues"].items() if k == "text"}
        return window["text"], {**raw, "cues": cues, "missing": sorted({"audio", "visual"} | set(raw["missing"]))}
    hide = condition.removeprefix("hide_")
    cues = {name: cue for name, cue in raw["cues"].items() if name != hide}
    evidence = {**raw, "cues": cues, "missing": sorted(set(raw["missing"]) | {hide}),
                "baseline_label": (cues.get("text") or {}).get("label"),
                "conflict": len({cue["label"] for cue in cues.values()}) > 1}
    return ("" if hide == "text" else window["text"]), evidence


def vote(weighted):
    seconds = Counter()
    for label, duration in weighted:
        if label in LABELS:
            seconds[label] += duration
    best = seconds.most_common(2)
    if not best or (len(best) > 1 and abs(best[0][1] - best[1][1]) < 1e-8):
        return None
    return best[0][0]


def channel_vote(cues):
    return vote([(cue["label"], cue.get("quality", 1.) * cue.get("coverage", 1.))
                 for cue in cues.values()])


def binary_test(first, second):
    n = first + second
    return min(1., 2 * sum(comb(n, k) for k in range(min(first, second) + 1)) / (2 ** n))


def metrics(rows, field):
    hits = sum(row[field] == row["gold"] for row in rows)
    valid = sum(row[field] in LABELS for row in rows)
    f1s = {}
    for label in LABELS:
        tp = sum(r["gold"] == r[field] == label for r in rows)
        gold = sum(r["gold"] == label for r in rows)
        predicted = sum(r[field] == label for r in rows)
        f1s[label] = 2 * tp / (gold + predicted) if gold + predicted else 0.
    return {"n": len(rows), "correct": hits, "wrong": valid - hits, "abstain": len(rows) - valid,
            "coverage": round(valid / len(rows), 4) if rows else None,
            "accuracy": round(hits / len(rows), 4) if rows else None,
            "accepted_accuracy": round(hits / valid, 4) if valid else None,
            "macro_f1": round(sum(f1s.values()) / 3, 4) if rows else None}


def paired(rows):
    before_only = sum(r["before"] == r["gold"] != r["after"] for r in rows)
    after_only = sum(r["after"] == r["gold"] != r["before"] for r in rows)
    common = [r for r in rows if r["before"] in LABELS and r["after"] in LABELS]
    by_source = {}
    for row in rows:
        delta = int(row["after"] == row["gold"]) - int(row["before"] == row["gold"])
        by_source.setdefault(row["id"].split("/")[0], []).append(delta)
    groups = list(by_source.values())
    rng = random.Random(20261127)
    intervals = []
    if len(groups) > 1:
        for _ in range(2000):
            sampled = rng.choices(groups, k=len(groups))
            intervals.append(sum(map(sum, sampled)) / sum(map(len, sampled)))
        intervals.sort()
    return {"before": metrics(rows, "before"), "after": metrics(rows, "after"),
            "before_only_correct": before_only, "after_only_correct": after_only,
            "mcnemar_exact_p": round(binary_test(before_only, after_only), 6),
            "source_bootstrap_delta_accuracy_95": [round(intervals[i], 4) for i in (49, 1949)] if intervals else None,
            "common_before": metrics(common, "before"), "common_after": metrics(common, "after")}


def review_all(data, selected, replay, extraction, extraction_key):
    from emotion_assistant.conflict_review import review_conflict, VERSION
    from emotion_assistant.semantic_review import MODEL
    from emotion_assistant.models import VERSIONS

    signature = code_signature(("conflict_review.py", "local_qwen.py"))
    identity_file = data / "qwen-identity.json"
    if replay:
        identity = json.loads(identity_file.read_text(encoding="utf8"))
    else:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=6) as response:
            models = json.load(response)["models"]
        model = next((item for item in models if item["name"] == MODEL), None)
        if not model:
            raise RuntimeError(f"Local model {MODEL} is unavailable")
        identity = {"name": MODEL, "digest": model["digest"],
                    "quantization": model["details"]["quantization_level"]}
    check_ollama_manifest(identity)
    if not replay:
        save(identity_file, identity)
    outcomes = {c: [] for c in CONDITIONS}
    unique_inferences = set()
    for idx, (key, row) in enumerate(selected.items(), 1):
        local = json.loads((data / "observations" / f"{key}.json").read_text(encoding="utf8"))
        if local["sha256"] != sha_file(data / "Raw" / f"{key}.mp4") or local["code_sha256"] != extraction_key:
            raise ValueError(f"Cached observation no longer matches video, software or model assets: {key}")
        for condition in CONDITIONS:
            before, after, simple_vote = [], [], []
            trace = []
            statuses = Counter()
            all_three = False
            for window in local["windows"]:
                text, evidence = masked(window, condition)
                baseline = evidence["baseline_label"]
                before.append((baseline, window["duration"]))
                simple_vote.append((channel_vote(evidence["cues"]), window["duration"]))
                candidate = baseline
                if not text or not evidence["conflict"]:
                    statuses["no_request"] += 1
                else:
                    cache_key = digest(json.dumps([signature, identity, text, evidence["cues"]],
                                                  ensure_ascii=False, sort_keys=True).encode())
                    folder = data / "qwen" / f"{cache_key}.json"
                    unique_inferences.add(cache_key)
                    if folder.exists():
                        response = json.loads(folder.read_text(encoding="utf8"))
                    elif replay:
                        raise FileNotFoundError(f"Missing cached review for {key} / {condition}")
                    else:
                        start = time.monotonic()
                        result = review_conflict(text, evidence)
                        response = {"label": result["label"], "status": result["status"],
                                    "seconds": round(time.monotonic() - start, 3)}
                        save(folder, response)
                    statuses[response["status"]] += 1
                    if response["status"] == "ok":
                        candidate = response["label"]
                        if candidate == "unknown":
                            statuses["model_abstain"] += 1
                all_three |= set(evidence["cues"]) == {"text", "audio", "visual"}
                after.append((candidate, window["duration"]))
                if condition == "observed":
                    trace.append({"duration_s": round(window["duration"], 4),
                                  "cues": {k: v["label"] for k, v in evidence["cues"].items()},
                                  "triggered": bool(text and evidence["conflict"]), "candidate": candidate})
            outcomes[condition].append({"id": key, "sha256": local["sha256"], "gold": row[-2].lower(),
                                        "human_channel_disagreement": len({(float(x) > 0) - (float(x) < 0)
                                                                           for x in row[4:7]}) > 1,
                                        "legacy": local["legacy"] if local["legacy"] in LABELS else None,
                                        "asr": local["available_asr"], "av": local["av"],
                                        "windows": len(before), "all_three": all_three, "cue_vote": vote(simple_vote),
                                        "before": vote(before), "after": vote(after), "statuses": dict(statuses)})
            if condition == "observed":
                outcomes[condition][-1]["utterance_trace"] = trace
        print(f"Evaluated {idx}/{len(selected)} {key}", flush=True)
    reports = {}
    for condition, rows in outcomes.items():
        triggers = [r for r in rows if any(k != "no_request" for k in r["statuses"])]
        no_triggers = [r for r in rows if r not in triggers]
        reports[condition] = {
            **paired(rows), "triggered_clips": len(triggers), "triggered_comparison": paired(triggers),
            "no_trigger_comparison": paired(no_triggers), "quality_coverage_modal_vote": metrics(rows, "cue_vote"),
            "human_disagreement": paired([r for r in rows if r["human_channel_disagreement"]]),
            "complete_available_cues": sum(r["all_three"] for r in rows),
            "utterance_statuses": dict(sum((Counter(r["statuses"]) for r in rows), Counter())),
        }
    observed = outcomes["observed"]
    text_only = {row["id"]: row for row in outcomes["qwen_text_only"]}
    direct = [{**row, "before": text_only[row["id"]]["after"]} for row in observed]
    selected_windows = [r for r in direct if any(status != "no_request" for status in r["statuses"])]
    from csv import reader
    with (data / "label.csv").open(encoding="utf-8-sig", newline="") as stream:
        train_counts = Counter(r[-2].lower() for r in reader(stream) if r and r[-1] == "train")
    majority = train_counts.most_common(1)[0][0]
    av_coverage = {"audio_detected": sum(r["av"]["audio_detected"] for r in observed),
                   "visual_detected": sum(r["av"]["visual_detected"] for r in observed),
                   "either_detected": sum(r["av"]["audio_detected"] or r["av"]["visual_detected"] for r in observed),
                   "any_accepted_av_seconds": sum(r["av"]["accepted_seconds"] > 0 for r in observed)}
    public_rows = {"observed": observed}
    for condition in CONDITIONS[1:]:
        public_rows[condition] = [
            {field: row[field] for field in ("id", "before", "after", "cue_vote", "statuses")}
            for row in outcomes[condition]
        ]
    return {"dataset": "original CH-SIMS v1, official test", "seed": SAMPLE_SEED,
            "label_csv_sha256": sha_file(data / "label.csv"), "test_population": 457, "sample_n": len(selected),
            "sources": len({r[0] for r in selected.values()}), "targets": dict(Counter(r["gold"] for r in observed)),
            "asr_clips": sum(r["asr"] for r in observed), "aligned_utterances": sum(r["windows"] for r in observed),
            "source_models": VERSIONS, "extraction": extraction, "extraction_fingerprint": extraction_key,
            "review_model": identity, "review_version": VERSION,
            "review_code_sha256": signature, "unique_model_requests": len(unique_inferences),
            "independent_av_coverage_when_hiding_text": av_coverage,
            "qwen_identical_triggers_text_only_vs_multimodal": {
                "all_clips": paired(direct), "triggered_clips": paired(selected_windows)},
            "train_majority": {"label": majority, **metrics([dict(r, majority=majority) for r in observed], "majority")},
            "legacy_file_content_summary": metrics(observed, "legacy"),
            "conditions": reports, "clips": public_rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=CACHE)
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--replay", action="store_true", help="Recompute from private, validated review caches")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "chsims-conflict.json")
    args = parser.parse_args()
    clips = cohort(args.data / "label.csv", args.limit)
    source = extraction_identity()
    fingerprint = digest(json.dumps(source, sort_keys=True).encode())
    if not args.replay:
        extract(args.data, clips, fingerprint)
    if not args.extract_only:
        save(args.output, review_all(args.data, clips, args.replay, source, fingerprint))
