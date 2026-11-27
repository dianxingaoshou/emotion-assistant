"""Evaluate a frozen deployment artifact on the two preselected, disjoint video cohorts."""

import argparse
from collections import Counter
import csv
from math import comb
from pathlib import Path
import random
import unicodedata

import numpy as np

from fit_chsims_foundation import (DATA, ROOT, SCENARIOS, checksums_for_raw, dataset, embeddings, file_sha,
                                   frozen_cohort, fusion, metrics, read_json, read_rows, target, write_json)


def numbers(labels):
    return np.array([fusion.LABELS.index(x) if x in fusion.LABELS else -1 for x in labels], dtype=int)


def compare(rows, first, second):
    gold = target(rows)
    a, b = numbers(first), numbers(second)
    wins = int(np.sum((b == gold) & (a != gold)))
    losses = int(np.sum((a == gold) & (b != gold)))
    discordant = wins + losses
    p = min(1., 2 * sum(comb(discordant, i) for i in range(min(wins, losses)+1))/2**discordant)
    clusters = {}
    for i, r in enumerate(rows):
        clusters.setdefault(r["id"].split("/")[0], []).append(i)
    groups = list(clusters.values())
    gaps = []
    rng = random.Random(20261127)
    for _ in range(2000):
        ids = [i for group in rng.choices(groups, k=len(groups)) for i in group]
        m1 = metrics(gold[ids], a[ids])
        m2 = metrics(gold[ids], b[ids])
        gaps.append([m2[k]-m1[k] if m1[k] is not None else np.nan for k in ("accuracy","macro_f1","positive_recall")])
    bounds = np.nanpercentile(gaps, (2.5,97.5), axis=0).T
    return {"corrected": wins, "new_errors": losses, "mcnemar_exact_p": p,
            "delta_source_bootstrap_95": dict(zip(("accuracy","macro_f1","positive_recall"), bounds.tolist()))}


def duplicate_sensitivity(label_file, groups):
    """Official test captions enter this post-prediction audit only, never the video classifier."""
    def normalized(caption):
        return "".join(ch for ch in unicodedata.normalize("NFKC", caption).casefold() if ch.isalnum())
    with label_file.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    known = {normalized(r[2]) for r in rows if r[-1] == "train"} - {""}
    collisions = {f"{r[0]}/{r[1]}" for r in rows if r[-1] == "test" and normalized(r[2]) in known}
    summary = {}
    for name, outcome in groups.items():
        kept = [r for r in outcome["clips"] if r["id"] not in collisions]
        summary[name] = {
            "excluded_ids": sorted(collisions & {r["id"] for r in outcome["clips"]}),
            "observed": {v: metrics(numbers([r["gold"] for r in kept]),
                                    numbers([r["predictions"]["observed"][v] for r in kept]))
                         for v in ("base","no_qwen","foundation")},
        }
    return summary


def run_group(rows, model, legacy=None):
    gold = target(rows)
    counts = Counter(r["label"] for r in rows)
    full_trace = []
    result = {}
    trace = {r["id"]: {"id": r["id"], "gold": r["label"], "predictions": {}} for r in rows}
    fixed_weight_control = {**model, "no_qwen_weights": model["foundation_weights"]}
    for scenario, (tx, au, vi) in SCENARIOS.items():
        pred = {k: [] for k in ("base", "no_qwen","foundation")}
        if scenario == "observed":
            pred["fixed_weights_no_qwen"] = []
        for row in rows:
            answers = {}
            for variant in pred:
                fixed_control = variant == "fixed_weights_no_qwen"
                ans = fusion.predict_with_head(
                    fixed_weight_control if fixed_control else model,
                    row["text"] if tx else "", audio=row["audio"] if au else None,
                    visual=row["visual"] if vi else None, embedding=row["embedding"] if tx else None,
                    audio_embedding=row["audio_embedding"] if au else None, variant="no_qwen" if fixed_control else variant)
                answers[variant] = ans["label"]
                pred[variant].append(ans["label"])
                if scenario == "observed" and variant == "foundation":
                    full_trace.append({"foundation_used": ans["foundation_used"], "experts": list(ans["experts"])})
                    trace[row["id"]]["observed_foundation"] = {
                        "scores": ans["scores"], "experts": ans["experts"],
                        "used_modalities": ans["used_modalities"], "qwen_used": ans["foundation_used"],
                    }
            trace[row["id"]]["predictions"][scenario] = answers
        if legacy and scenario in legacy:
            old = {r["id"]: r for r in legacy[scenario]}
            if set(old) != set(trace):
                raise ValueError("The published baseline uses different clips")
            for method, field in (("published_vote","cue_vote"), ("published_qwen","after")):
                pred[method] = [old[r["id"]][field] for r in rows]
        result[scenario] = {"metrics": {k: metrics(gold, numbers(v)) for k,v in pred.items()},
                            "source_paired_deltas": {
                                k: compare(rows, pred[k], pred["foundation"])
                                for k in ("no_qwen", "published_vote", "fixed_weights_no_qwen") if k in pred}}
    source_ids = {r["id"].split("/")[0] for r in rows}
    return {"n": len(rows), "sources": len(source_ids), "gold": counts,
            "qwen_text_embeddings_used": sum(r["foundation_used"] for r in full_trace),
            "audio_embeddings_used": sum("audio_embedding" in r["experts"] for r in full_trace),
            "conditions": result, "clips": list(trace.values())}


def evaluate(data, head, fit_record, output, raw_dir):
    fit = read_json(fit_record)
    if file_sha(head) != fit["model_sha256"]:
        raise ValueError("Frozen model differs from the validation record")
    with np.load(head, allow_pickle=False) as archive:
        model = dict(archive)
    main_rows = read_rows(data)
    # Training and primary/audit evaluation consume separate, explicitly enumerated manifests.
    import json
    audit_rows = [json.loads(line) for line in (data / "audit-rows.jsonl").read_text(encoding="utf-8").splitlines()]
    main = frozen_cohort(data, "raw-cohort.json")
    audit = frozen_cohort(data, "audit-cohort.json")["audit"]
    if len(main["test"]) != 48 or len(audit) != 48 or set(main["test"]) & set(audit):
        raise ValueError("Expected two fixed disjoint sets of 48 videos")
    used = set(main["train"]) | set(main["valid"])
    if used & (set(main["test"]) | set(audit)):
        raise ValueError("Unexpected overlap with training/selection videos")
    old = read_json(ROOT / "docs/results/chsims-conflict.json")
    if {(r["id"],r["gold"]) for r in old["clips"]["observed"]} != {
            (r["id"],r["label"]) for r in main_rows if r["id"] in main["test"]}:
        raise ValueError("Published cohort labels do not match")
    files = {"base_text": data / "embeddings-text.npz", "asr": data / "embeddings-text-asr.npz",
             "audio": data / "embeddings-audio.npz"}
    wanted_text, wanted_audio = checksums_for_raw(data, [*main["test"], *audit])
    if any(file_sha(raw_dir / (key + ".mp4")) != value for key, value in wanted_audio.items()):
        raise ValueError("The current raw video differs from the feature cache")
    if any(wanted_audio[r["id"]] != r["sha256"] for r in old["clips"]["observed"]):
        raise ValueError("A primary video differs from the published comparison")
    text_emb = embeddings(files["asr"], fusion.TEXT_DIM, expected=wanted_text)
    acoustic = embeddings(files["audio"], fusion.AUDIO_DIM, expected=wanted_audio)
    public = {"schema": "chsims-foundation-paired-v1","labels": list(fusion.LABELS),
              "confusion_columns": [*fusion.LABELS,"abstain"],"fit":fit,
              "manifest_sha256": {p: file_sha(data/p) for p in ("raw-cohort.json","audit-cohort.json")},
              "features_sha256": {k:file_sha(p) for k,p in files.items()},
              "verified_original_primary_video_hashes": len(main["test"])}
    for name, rows, ids, baselines in (("primary48", main_rows, main["test"], old["clips"]),
                                       ("audit48", audit_rows, audit, None)):
        sample = dataset(data, rows, ids, text_emb, acoustic, raw=True)
        if any(r["text"] and fusion.HAN.search(r["text"]) and r["embedding"] is None for r in sample):
            raise ValueError("ASR embedding missing for a transcribed evaluation clip")
        public[name] = run_group(sample, model, baselines)
    compared = public["primary48"]["conditions"]["observed"]["metrics"]
    current, old_vote, matched = compared["foundation"], compared["published_vote"], compared["no_qwen"]
    measures = ("accuracy", "macro_f1", "positive_recall")
    public["primary_release_checks"] = {
        "accuracy_above_published_vote": current["accuracy"] > old_vote["accuracy"],
        "macro_f1_above_published_vote": current["macro_f1"] > old_vote["macro_f1"],
        "positive_recall_at_least_published_vote": current["positive_recall"] >= old_vote["positive_recall"],
        "one_or_more_metrics_above_matched_no_qwen": any(current[k] > matched[k] for k in measures),
        "delta_vs_matched_no_qwen": {k: current[k] - matched[k] for k in measures},
    }
    public["caption_duplication_sensitivity"] = duplicate_sensitivity(
        raw_dir.parent / "label.csv", {k: public[k] for k in ("primary48", "audit48")})
    write_json(output, public)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--head", type=Path, default=fusion.ASSET)
    parser.add_argument("--raw", type=Path, default=ROOT / ".runtime/chsims/Raw")
    parser.add_argument("--fit-record", type=Path, default=DATA/"fit.json")
    parser.add_argument("--output", type=Path, default=ROOT/"docs/results/chsims-foundation-fusion.json")
    args = parser.parse_args()
    evaluate(args.data, args.head, args.fit_record, args.output, args.raw)
