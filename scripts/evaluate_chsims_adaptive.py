"""Score the frozen adaptive fusion and its matched ablation on the two historical CH-SIMS cohorts."""

import argparse
from collections import Counter
import csv
from math import comb
from pathlib import Path

import numpy as np

from chsims_samples import LABEL_SHA256
from evaluate_chsims_foundation import numbers
from fit_chsims_foundation import (
    DATA, ROOT, SCENARIOS, checksums_for_raw, dataset, embeddings, file_sha,
    frozen_cohort, fusion, metrics, read_json, write_json,
)

PUBLIC = ROOT / "docs/results"
VARIANTS = ("v1_no_qwen", "v1_foundation", "no_qwen", "foundation")
MEASURES = ("accuracy", "macro_f1", "positive_recall", "neutral_recall")
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261127


def release_checks(scores):
    new, control = scores["foundation"], scores["no_qwen"]
    checks = {
        "accuracy_strictly_above_matched_no_qwen": new["accuracy"] > control["accuracy"],
        "macro_f1_at_least_matched_no_qwen": new["macro_f1"] >= control["macro_f1"],
        "positive_recall_at_least_matched_no_qwen": new["positive_recall"] >= control["positive_recall"],
    }
    return {**checks, "metrics_passed": all(checks.values())}


def all_metrics(gold, predicted):
    scores = metrics(gold, predicted)
    neutral = scores["confusion_gold_by_prediction"][fusion.LABELS.index("neutral")]
    hits, support = neutral[fusion.LABELS.index("neutral")], sum(neutral)
    return {**scores, "neutral_hits": hits, "neutral_support": support,
            "neutral_recall": hits / support if support else None}


def source_bootstrap_weights(ids):
    """Sample source videos, retaining all of each sampled source's clips."""
    sources, inverse = np.unique([key.split("/")[0] for key in ids], return_inverse=True)
    count = len(sources)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.multinomial(count, np.full(count, 1 / count), size=BOOTSTRAP_REPLICATES)
    return draws[:, inverse].astype(float)


def bootstrap_measures(gold, pred, weights):
    correct = gold == pred
    accuracy = (weights @ correct) / weights.sum(axis=1)
    f1 = []
    recalls = {}
    for index, name in enumerate(fusion.LABELS):
        actual, forecast = gold == index, pred == index
        support = weights @ actual
        hits = weights @ (actual & forecast)
        f1.append(2 * hits / np.maximum(1, support + weights @ forecast))
        recalls[name] = np.divide(hits, support, out=np.full_like(support, np.nan), where=support > 0)
    return np.column_stack((accuracy, np.mean(f1, axis=0), recalls["positive"], recalls["neutral"]))


def paired_difference(ids, gold, before, after, base_scores, new_scores, boot_delta):
    fixed = [key for key, y, a, b in zip(ids, gold, before, after) if a != y and b == y]
    broken = [key for key, y, a, b in zip(ids, gold, before, after) if a == y and b != y]
    n = len(fixed) + len(broken)
    tail = sum(comb(n, k) for k in range(min(len(fixed), len(broken)) + 1))
    ci = np.nanpercentile(boot_delta, (2.5, 97.5), axis=0).T
    return {"delta": {k: new_scores[k] - base_scores[k] for k in MEASURES},
            "corrected": len(fixed), "new_errors": len(broken), "corrected_ids": fixed,
            "new_error_ids": broken, "mcnemar_exact_p": min(1., 2 * tail / 2**n),
            "delta_source_bootstrap_95": dict(zip(MEASURES, ci.tolist()))}


def trace_of(result):
    return {"scores": result["scores"], "experts": result["experts"],
            "used_modalities": result["used_modalities"], "qwen_used": result["foundation_used"]}


def score_group(rows, previous, head, params):
    ids = [r["id"] for r in rows]
    old = {r["id"]: r for r in previous["clips"]}
    if len(ids) != len(set(ids)) or set(ids) != set(old):
        raise ValueError("Historical clip IDs do not match the frozen cohort")
    gold = numbers([r["label"] for r in rows])
    traces = [{"id": r["id"], "gold": r["label"], "predictions": {}} for r in rows]
    conditions = {}
    resamples = source_bootstrap_weights(ids)
    comparisons = (("no_qwen", "foundation"), ("v1_foundation", "foundation"),
                   ("v1_no_qwen", "no_qwen"), ("v1_no_qwen", "foundation"))
    for condition, (has_text, has_audio, has_visual) in SCENARIOS.items():
        for row, trace in zip(rows, traces):
            old_record = old[row["id"]]
            if row["label"] != old_record["gold"]:
                raise ValueError("Historical label changed")
            previous_predictions = old_record["predictions"][condition]
            labels = {"v1_" + v: previous_predictions[v] for v in ("no_qwen", "foundation")}
            for variant in ("no_qwen", "foundation"):
                answer = fusion.predict_with_head(
                    head, row["text"] if has_text else "", audio=row["audio"] if has_audio else None,
                    visual=row["visual"] if has_visual else None,
                    embedding=row["embedding"] if variant == "foundation" and has_text else None,
                    audio_embedding=row["audio_embedding"] if has_audio else None,
                    variant=variant, adaptive=params)
                labels[variant] = answer["label"]
                if answer["label"] is not None and answer["label"] not in fusion.LABELS:
                    raise ValueError("Invalid output class")
                if answer["foundation_used"] and (not has_text or variant == "no_qwen"):
                    raise ValueError("A blinded text encoder contributed to a prediction")
                if condition == "observed":
                    trace["observed_" + variant] = trace_of(answer)
            trace["predictions"][condition] = labels
        pred = {v: numbers([r["predictions"][condition][v] for r in traces]) for v in VARIANTS}
        scores = {v: all_metrics(gold, x) for v, x in pred.items()}
        for v in ("no_qwen", "foundation"):
            historical = previous["conditions"][condition]["metrics"][v]
            if any(scores["v1_" + v][k] != historical[k] for k in historical):
                raise ValueError("Historical predictions do not reproduce the recorded scores")
        estimates = {v: bootstrap_measures(gold, x, resamples) for v, x in pred.items()}
        pairs = {f"{a}_to_{b}": paired_difference(ids, gold, pred[a], pred[b], scores[a], scores[b],
                                                   estimates[b] - estimates[a]) for a, b in comparisons}
        conditions[condition] = {"metrics": scores, "source_paired_deltas": pairs}
    old_errors = []
    neutrals = []
    for row in traces:
        p = row["predictions"]["observed"]
        record = {"id": row["id"], "gold": row["gold"], **p}
        if p["v1_no_qwen"] == row["gold"] and p["v1_foundation"] != row["gold"]:
            old_errors.append(record)
        if row["gold"] == "neutral":
            neutrals.append(record)
    qwen_used = sum(r["observed_foundation"]["qwen_used"] for r in traces)
    audio_used = sum("audio_embedding" in r["observed_foundation"]["experts"] for r in traces)
    return {"n": len(rows), "sources": len({k.split("/")[0] for k in ids}),
            "gold": Counter(r["label"] for r in rows), "qwen_text_embeddings_used": qwen_used,
            "audio_embeddings_used": audio_used, "conditions": conditions, "clips": traces,
            "v1_qwen_new_errors_revisited": old_errors, "neutral_clips": neutrals}


def evaluate(data, raw, record, output, params_path):
    fit = read_json(record)
    historical = read_json(PUBLIC / "chsims-foundation-fusion.json")
    if file_sha(params_path) != fit["model_sha256"] or file_sha(fusion.ASSET) != fit["head_sha256"]:
        raise ValueError("Fusion parameters or base heads differ from the training freeze")
    if fit["head_sha256"] != historical["fit"]["model_sha256"]:
        raise ValueError("Base heads differ from the fixed historical comparison")
    params = read_json(params_path)
    with np.load(fusion.ASSET, allow_pickle=False) as archive:
        head = dict(archive)
    main = frozen_cohort(data, "raw-cohort.json")
    cohorts = {"primary48": main["test"], "audit48": frozen_cohort(data, "audit-cohort.json")["audit"]}
    all_ids = set(cohorts["primary48"]) | set(cohorts["audit48"])
    if (len(all_ids) != 96 or any(len(ids) != 48 for ids in cohorts.values())
            or all_ids.intersection(main["train"]) or all_ids.intersection(main["valid"])):
        raise ValueError("Unexpected cohort size, duplicate or overlap with train/valid")
    label_path = raw.parent / "label.csv"
    if file_sha(label_path) != LABEL_SHA256:
        raise ValueError("Official labels differ from the CH-SIMS v1 fingerprint")
    with label_path.open(encoding="utf-8-sig", newline="") as stream:
        official = {f"{r[0]}/{r[1]}": r[-2].lower() for r in csv.reader(stream) if r[-1] == "test"}
    if not all_ids <= set(official):
        raise ValueError("A selected clip is outside the official test set")
    text_sha, audio_sha = checksums_for_raw(data, all_ids)
    if any(file_sha(raw / (key + ".mp4")) != sha for key, sha in audio_sha.items()):
        raise ValueError("Original MP4 differs from the feature cache")
    vectors = {}
    for key, suffix, dim, checks in (("asr", "text-asr", fusion.TEXT_DIM, text_sha),
                                     ("audio", "audio", fusion.AUDIO_DIM, audio_sha)):
        file = data / f"embeddings-{suffix}.npz"
        if file_sha(file) != historical["features_sha256"][key]:
            raise ValueError("Input representation differs from the original paired comparison")
        vectors[key] = embeddings(file, dim, expected=checks)
    result = {"schema": "chsims-paired-adaptive-v2", "labels": list(fusion.LABELS),
              "confusion_columns": [*fusion.LABELS, "abstain"], "fit": fit,
              "v1_results_sha256": file_sha(PUBLIC / "chsims-foundation-fusion.json"),
              "official_labels_sha256": LABEL_SHA256,
              "original_mp4_sha256": dict(sorted(audio_sha.items())),
              "uncertainty": {"resample_unit": "source video", "paired_replicates": BOOTSTRAP_REPLICATES,
                              "seed": BOOTSTRAP_SEED, "interval": "percentile, 95%"},
              "manifest_sha256": historical["manifest_sha256"]}
    for name, ids in cohorts.items():
        rows = [{"id": r["id"], "label": official[r["id"]]} for r in historical[name]["clips"]]
        sample = dataset(data, rows, ids, vectors["asr"], vectors["audio"], raw=True)
        result[name] = score_group(sample, historical[name], head, params)
    result["primary_release_checks"] = release_checks(result["primary48"]["conditions"]["observed"]["metrics"])
    write_json(output, result)
    return result


if __name__ == "__main__":
    from emotion_assistant.adaptive_clip_fusion import ASSET
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--raw", type=Path, default=ROOT / ".runtime/chsims/Raw")
    parser.add_argument("--fit-record", type=Path, default=DATA / "adaptive-fit.json")
    parser.add_argument("--params", type=Path, default=ASSET)
    parser.add_argument("--output", type=Path, default=PUBLIC / "chsims-adaptive-fusion.json")
    args = parser.parse_args()
    result = evaluate(args.data, args.raw, args.fit_record, args.output, args.params)
    print({k: result["primary48"]["conditions"]["observed"]["metrics"][k]["correct"] for k in VARIANTS})
    print(result["primary_release_checks"])
