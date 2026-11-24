"""Select an utterance content rule on development sources, then freeze and audit it."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emotion_assistant.content_fusion import AV_CAP, CLASSES, MODEL_VERSION, _fuse, summarize_content


def predictions(clip, *, development=False, mask=()):
    inputs = clip["intervals"]
    arms = {"text": [], "joint_candidate": [], "av_proxy": [], "av_during_text": []}
    has_joint = False
    for item in inputs:
        text = item["dev_text" if development else "text"] if "text" not in mask else None
        audio = item["audio"] if "audio" not in mask else None
        visual = item["visual"] if "visual" not in mask else None
        time = {"t_start": item["t_start"], "t_end": item["t_end"]}
        text_arm = _fuse(text, None, None, "text_first")
        joint = _fuse(text, visual, audio, "blend")
        av = _fuse(None, visual, audio, "blend")
        arms["text"].append(dict(text_arm, **time))
        arms["joint_candidate"].append(dict(joint, **time))
        arms["av_proxy"].append(dict(av, **time))
        if text_arm["scores"] is not None:
            arms["av_during_text"].append(dict(av, **time))
        has_joint |= len(joint["available_modalities"]) == 3
    text = summarize_content(arms["text"])
    joint = summarize_content(arms["joint_candidate"])
    av = summarize_content(arms["av_proxy"])
    proxy = av["candidate_scores"]
    matched_proxy = summarize_content(arms["av_during_text"])["candidate_scores"]
    truth = clip["label"]
    errors = {
        name: sum((scores[k] - (k == truth)) ** 2 for k in CLASSES) if scores else None
        for name, scores in (("text", text["scores"]), ("joint_candidate", joint["scores"]))
    }
    return {
        "id": clip["id"],
        "source": clip["source_id"],
        "truth": clip["label"],
        "text": text["label"],
        "joint_candidate": joint["label"],
        "av_proxy": max(proxy, key=proxy.get) if proxy else None,
        "av_during_text": max(matched_proxy, key=matched_proxy.get) if matched_proxy else None,
        "brier": errors,
        "has_joint": has_joint,
        "n_intervals": len(inputs),
    }


def metrics(rows, key, *, equal_source=False):
    matrix = np.zeros((3, 4), dtype=float)
    group_sizes = Counter(r["source"] for r in rows)
    for r in rows:
        prediction = r[key]
        col = CLASSES.index(prediction) if prediction is not None else 3
        matrix[CLASSES.index(r["truth"]), col] += 1 / group_sizes[r["source"]] if equal_source else 1.0
    hits = matrix[:, :3].diagonal()
    den = matrix.sum(axis=1) + matrix[:, :3].sum(axis=0)
    count, scored = matrix.sum(), matrix[:, :3].sum()
    return {
        "clips": len(rows),
        "sources": len(group_sizes),
        "coverage": scored / count if count else None,
        "accuracy_all": hits.sum() / count if count else None,
        "accuracy_scored": hits.sum() / scored if scored else None,
        "macro_f1_all": np.divide(2 * hits, den, out=np.zeros(3), where=den > 0).mean() if count else None,
        "confusion_truth_negative_neutral_positive_by_pred_including_abstain": matrix.tolist(),
    }


def paired(rows, other):
    """Clip accuracy counts and paired 95% percentile interval resampling source videos."""
    if not rows:
        return None
    groups = {}
    counts = Counter()
    for r in rows:
        delta = int(r["joint_candidate"] == r["truth"]) - int(r[other] == r["truth"])
        counts[delta] += 1
        pair = groups.setdefault(r["source"], [0, 0])
        pair[0] += 1
        pair[1] += delta
    pairs = np.array(list(groups.values()))
    rng = np.random.default_rng(20261123)
    draws = pairs[rng.integers(0, len(groups), size=(4000, len(groups)))].sum(axis=1)
    return {
        "joint_wins": counts[1],
        "joint_losses": counts[-1],
        "same_correctness": counts[0],
        "accuracy_difference_95pct_source_bootstrap": np.quantile(
            draws[:, 1] / draws[:, 0], [0.025, 0.975]
        ).tolist(),
    }


def evaluate(rows, policy):
    report = {}
    for row in rows:
        row["deployed"] = row["joint_candidate"] if policy == "blend" else row["text"]
        row["train_majority"] = "positive"
    for key in ("av_proxy", "av_during_text", "text", "joint_candidate", "deployed", "train_majority"):
        report[key] = metrics(rows, key)
    report["paired_candidate_vs_text"] = paired(rows, "text")
    report["paired_candidate_vs_av_proxy"] = paired(rows, "av_proxy")
    matched = [row["brier"] for row in rows if all(v is not None for v in row["brier"].values())]
    report["paired_brier_on_common_scored"] = {
        "n": len(matched),
        **{
            k: sum(row[k] for row in matched) / len(matched) if matched else None
            for k in ("text", "joint_candidate")
        },
    }
    return report


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=ROOT / "reports/content-features.jsonl")
    p.add_argument("--out", type=Path, default=ROOT / "reports/content-fusion.json")
    p.add_argument("--write-policy", type=Path, help="Save the development-selected frozen deployment rule")
    args = p.parse_args()
    feature_bytes = args.features.read_bytes()
    name = args.features
    if name.suffix == ".gz":
        feature_bytes = gzip.decompress(feature_bytes)
        name = name.with_suffix("")
    metadata = json.loads(name.with_suffix(".metadata.json").read_text("utf-8"))
    clips = [json.loads(line) for line in feature_bytes.decode("utf-8").splitlines()]
    dev = [c for c in clips if c["partition"] == "development"]
    held = [c for c in clips if c["partition"] == "holdout"]
    dev_sources = {c["source_id"] for c in dev}
    held_sources = {c["source_id"] for c in held}
    if len(clips) != 100 or len({c["id"] for c in clips}) != 100 or len(dev) != 57 or len(held) != 43:
        raise ValueError("Unexpected attachment 1 partition")
    if (
        dev_sources & held_sources
        or any(c["base_split"] in ("train", "valid") for c in held)
        or metadata["surrogate_excludes_raw_sources"] != len(dev_sources | held_sources)
    ):
        raise ValueError(
            "Development, base text fit and holdout sources must follow the frozen disjoint split"
        )
    development = [predictions(c, development=True) for c in dev]
    baseline = metrics(development, "text", equal_source=True)
    candidate = metrics(development, "joint_candidate", equal_source=True)
    # Fixed before matched feature extraction: one candidate, no weight search.
    deploy_joint = (
        candidate["macro_f1_all"] >= baseline["macro_f1_all"] + 0.01
        and candidate["accuracy_all"] + 1e-12 >= baseline["accuracy_all"]
        and candidate["coverage"] + 1e-12 >= baseline["coverage"]
    )
    policy = "blend" if deploy_joint else "text_first"
    freeze = {
        "model_version": MODEL_VERSION,
        "av_cap": AV_CAP,
        "policy": policy,
        "english_classifier_sha256": metadata["text_artifact_sha256"],
        "development_source_ids": sorted(dev_sources),
        "surrogate_text_model": metadata["surrogate_text_model"],
        "rule": "Select blend only for source-balanced development macro-F1 improvement >= .01 with no accuracy or coverage loss.",
    }
    if args.write_policy:
        write(args.write_policy, freeze)
    # Policy frozen above. No score, threshold or hypothesis is fitted to the next 43 clips.
    held_rows = [predictions(c) for c in held]
    primary = evaluate(held_rows, policy)
    native_joint = [r for r in held_rows if r["has_joint"]]
    stress = {}
    for missing in (("audio",), ("visual",), ("text",), ("audio", "visual"), ("text", "audio", "visual")):
        rows = [predictions(c, mask=missing) for c in held]
        stress["drop_" + "_".join(missing)] = evaluate(rows, policy)
    for clip, row in zip(held, held_rows, strict=True):
        one = clip["clip_text"]
        row["whole_asr_text"] = one["label"] if one else None
        av = clip["clip_av"]
        row["old_clip_av"] = max(av, key=av.get) if av else None
        if row["old_clip_av"] == "uncertain":
            row["old_clip_av"] = None
    output = {
        "protocol": "aligned-utterance-content-audit-v1",
        "model": freeze,
        "features_sha256": hashlib.sha256(feature_bytes).hexdigest(),
        "feature_metadata": metadata,
        "class_order": list(CLASSES),
        "target": "Official positive/neutral/negative expressed sentiment of the entire clip, based on actual ASR utterance timelines. No individual-window ground truth or reference transcripts as inputs.",
        "splits": "57 clips, 18 base-train/validation-overlapping sources form development only. Development text is scored by a shadow base trained without any of the 37 attachment-1 source IDs and normalized reference duplicates. 43 clips from 19 other sources are held out from both policy selection and frozen English base train/validation. This cohort was previously assessed for other systems and is not a newly collected blind cohort.",
        "candidate": "Linear late fusion on aligned utterances. Reject dominant unmapped or invalid A/V data, renormalize known mass; each usable cue adds <=12.5% auxiliary weight scaled by signal quality, coverage and mapped mass. Text required for an endorsed semantic label. Disagreement is reported. Aggregate by actual nonoverlapping duration on text-supported scored intervals, abstain below .40.",
        "metrics": "Accuracy and three-class macro-F1 include abstentions as misses; conditional accuracy and coverage reported separately. Split selection weights every source equally. Holdout result is clip-weighted. Masks rerun content fusion after omitting the indicated extracted channels; audio removal preserves cached ASR only to simulate a provided transcript. A/V proxy is an exploratory semantic comparison, never an endorsed interpretation of someone's subjective mood.",
        "selection": {
            "development_source_balanced_text": baseline,
            "development_source_balanced_candidate": candidate,
            "development_clip_weighted": evaluate(development, policy),
        },
        "holdout": primary,
        "holdout_natively_three_channels_in_same_window": evaluate(native_joint, policy),
        "holdout_whole_asr_text_prior_pipeline": metrics(held_rows, "whole_asr_text"),
        "holdout_old_file_av_mapped_prior_pipeline": metrics(held_rows, "old_clip_av"),
        "holdout_channel_mask_sensitivity": stress,
        "clips": held_rows,
    }
    write(args.out, output)
    print(
        "deployment",
        policy,
        "development",
        len(dev),
        "holdout",
        len(held),
        "complete cases",
        len(native_joint),
    )
    for key in ("av_proxy", "text", "joint_candidate", "deployed", "train_majority"):
        print(key, {k: primary[key][k] for k in ("coverage", "accuracy_all", "macro_f1_all")})
    print("paired", primary["paired_candidate_vs_text"])


if __name__ == "__main__":
    main()
