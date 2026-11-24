"""Paired wording and aggregation interventions on the original English video holdout."""

import argparse
from collections import Counter
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fit_english_intensity import regression_metrics
from emotion_assistant.english_intensity import ASSET, predict_segments_with_model


WORDS = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
ARMS = ("asr_duration", "asr_whole", "asr_restore_deletions",
        "reference_without_deletions", "reference_duration_proxy", "reference_whole")


@dataclass(frozen=True)
class Edit:
    operation: str
    reference: str | None
    hypothesis: str | None
    segment: int | None


def words(text):
    # Keep digits and non-English output for edit counts. Regression still uses
    # its own frozen English tokenizer and abstention rules.
    return WORDS.findall(text.lower().replace("’", "'"))


def align(reference, segments):
    """Minimum word Levenshtein distance, with reproducible diagonal/deletion/insertion ties."""
    ref = words(reference)
    hyp = [(w, i) for i, segment in enumerate(segments) for w in words(segment["text"])]
    n, m = len(ref), len(hyp)
    costs = [list(range(m + 1))]
    for i, r in enumerate(ref, 1):
        prev = costs[-1]
        row = [i]
        for j, (h, _) in enumerate(hyp, 1):
            row.append(min(prev[j - 1] + (r != h), prev[j] + 1, row[-1] + 1))
        costs.append(row)
    path = []
    i, j = n, m
    while i or j:
        if i and j and costs[i][j] == costs[i-1][j-1] + (ref[i-1] != hyp[j-1][0]):
            h, segment = hyp[j-1]
            path.append(Edit("match" if ref[i-1] == h else "substitution", ref[i-1], h, segment))
            i, j = i-1, j-1
        elif i and costs[i][j] == costs[i-1][j] + 1:
            path.append(Edit("deletion", ref[i-1], None, None))
            i -= 1
        else:
            h, segment = hyp[j-1]
            path.append(Edit("insertion", None, h, segment))
            j -= 1
    return path[::-1]


def reference_groups(path, n_segments):
    """Proxy boundaries only. Deletions go to the nearest ASR word, tie to preceding word."""
    bins = [[] for _ in range(n_segments)]
    next_at = [None] * len(path)
    following = None
    for i in reversed(range(len(path))):
        if path[i].hypothesis is not None:
            following = i
        next_at[i] = following
    previous = None
    for i, op in enumerate(path):
        if op.hypothesis is not None:
            previous = i
        if op.reference is not None:
            right = next_at[i]
            chosen = previous if right is None or (previous is not None and i - previous <= right - i) else right
            if chosen is not None:
                bins[path[chosen].segment].append(op.reference)
    return [" ".join(group) for group in bins]


def value(text, model):
    estimate = predict_segments_with_model([text], model)
    return estimate["value"] if estimate is not None else None


def by_duration(texts, segments, model):
    weighted = duration = 0.
    for text, segment in zip(texts, segments, strict=True):
        v = value(text, model)
        span = segment["end"] - segment["start"]
        if span <= 0:
            raise ValueError("Expected a positive utterance duration")
        if v is not None:
            weighted += v * span
            duration += span
    return weighted / duration if duration else None


def analyze(row, model):
    reference, segments = row["official_text"], row["segments"]
    path = align(reference, segments)
    grouped = reference_groups(path, len(segments))
    parts = [s["text"] for s in segments]
    hypothesis = " ".join(parts)
    # Insert only missing reference words into the hypothesis. Keep its actual
    # substitutions, insertions and order. Conversely, remove only D from the
    # reference in a second oracle; this exposes path-dependence of these effects.
    restored = " ".join(op.hypothesis if op.hypothesis is not None else op.reference for op in path)
    ref_minus_d = " ".join(op.reference for op in path if op.reference and op.operation != "deletion")
    outcome = {"asr_duration": by_duration(parts, segments, model), "asr_whole": value(hypothesis, model),
               "asr_restore_deletions": value(restored, model),
               "reference_without_deletions": value(ref_minus_d, model),
               "reference_duration_proxy": by_duration(grouped, segments, model),
               "reference_whole": value(reference, model)}
    for sentence, baseline in ((hypothesis, "asr_whole"), (reference, "reference_whole")):
        normalized = value(" ".join(words(sentence)), model)
        if not equal(normalized, outcome[baseline]):
            raise ValueError(f"Lexical normalization changed frozen model prediction: {row['id']} {baseline}")
    return {"id": row["id"], "source": row["source"], "target": row["target"],
            "segments": len(segments), "edit_counts": dict(Counter(op.operation for op in path)), **outcome}


def equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(a-b) <= 1e-9


def metrics(rows, key):
    scored = [r for r in rows if r[key] is not None]
    return {**regression_metrics([r["target"] for r in scored], [r[key] for r in scored]),
            "total": len(rows), "abstained": len(rows) - len(scored)}


class SourceBootstrap:
    """Resample source clusters, reusing identical resampled observations across paired arms."""
    def __init__(self, rows):
        self.y = np.asarray([r["target"] for r in rows])
        self.rows = rows
        groups, inverse = np.unique([r["source"] for r in rows], return_inverse=True)
        rng = np.random.default_rng(20261124)
        choices = rng.multinomial(len(groups), np.ones(len(groups)) / len(groups), size=4000)
        self.weight = choices[:, inverse]
        self.n = self.weight.sum(axis=1)
        self.mean_y = (self.weight * self.y).sum(axis=1) / self.n
        self.var_y = (self.weight * (self.y - self.mean_y[:, None])**2).sum(axis=1)
        self.results = {}

    def sample(self, key):
        if key not in self.results:
            p = np.array([row[key] for row in self.rows])
            w = self.weight
            centered = p - ((w * p).sum(axis=1) / self.n)[:, None]
            cov = (w * (self.y - self.mean_y[:, None]) * centered).sum(axis=1)
            denom = np.sqrt(self.var_y * (w * centered**2).sum(axis=1))
            pearson = np.divide(cov, denom, out=np.full_like(cov, np.nan), where=denom > 0)
            mae = (w * np.abs(self.y - p)).sum(axis=1) / self.n
            self.results[key] = mae, pearson
        return self.results[key]

    def compare(self, before, after):
        weights = Counter({before: -1})
        weights[after] += 1
        return self.linear_contrast(weights)

    def linear_contrast(self, coefficients):
        scores = {key: metrics(self.rows, key) for key in coefficients}
        differences = {}
        for i, k in enumerate(("mae", "pearson")):
            deltas = sum(c * self.sample(key)[i] for key, c in coefficients.items())
            finite = deltas[np.isfinite(deltas)]
            differences[k + "_delta"] = (
                sum(c * scores[key][k] for key, c in coefficients.items())
                if all(s[k] is not None for s in scores.values()) else None)
            differences[k + "_delta_95pct"] = np.quantile(finite, [.025, .975]).tolist() if len(finite) else None
        return differences


def summarize(rows):
    common = [r for r in rows if all(r[k] is not None for k in ARMS)]
    boot = SourceBootstrap(common)
    counts = Counter()
    for r in rows:
        counts.update(r["edit_counts"])
    insert, sub, delete = (counts[k] for k in ("insertion", "substitution", "deletion"))
    total = counts["match"] + sub + delete
    groups = {}
    for name, bucket in (("none", [r for r in rows if r["segments"] == 0]),
                         ("one", [r for r in rows if r["segments"] == 1]),
                         ("multiple", [r for r in rows if r["segments"] >= 2])):
        groups[name] = {"clips": len(bucket), **{k: metrics(bucket, k) for k in
                       ("asr_duration", "asr_whole", "reference_whole")}}
    return {
        "clips": len(rows), "sources": len({r["source"] for r in rows}),
        "word_edits": {**counts, "reference_words": total, "wer": (insert + sub + delete) / total},
        "all": {k: metrics(rows, k) for k in ARMS}, "common_count": len(common),
        "common_sources": len({r["source"] for r in common}), "common": {k: metrics(common, k) for k in ARMS},
        "wording_change_minus_pooling_change": boot.linear_contrast(
            {"reference_whole": 1, "asr_whole": -2, "asr_duration": 1}),
        "paired_common_changes": {f"{a}_to_{b}": boot.compare(a, b) for a, b in (
            ("asr_duration", "asr_whole"), ("asr_whole", "reference_whole"),
            ("asr_whole", "asr_restore_deletions"), ("asr_restore_deletions", "reference_whole"),
            ("reference_without_deletions", "reference_whole"),
            ("asr_duration", "reference_duration_proxy"), ("reference_duration_proxy", "reference_whole"))},
        "utterance_count_groups": groups,
    }


def holdout_sensitivity(rows):
    """Check the main comparison against complete-case selection and single influential sources."""
    keys = ("asr_duration", "asr_whole", "reference_whole")
    common = [r for r in rows if all(r[k] is not None for k in keys)]
    sources = sorted({r["source"] for r in common})
    leave_out = {"mae": [], "pearson": []}
    for source in sources:
        subset = [r for r in common if r["source"] != source]
        duration, whole, reference = [metrics(subset, k) for k in keys]
        for k, changes in leave_out.items():
            changes.append((reference[k] - whole[k]) - (whole[k] - duration[k]))

    fallback = [
        {**r, "asr_whole_if_available": r["asr_whole"] if r["asr_whole"] is not None else r["asr_duration"]}
        for r in rows if r["asr_duration"] is not None and r["reference_whole"] is not None
    ]
    return {
        "three_arm_common_n": len(common),
        "whole_with_legacy_fallback": {
            "fallback_n": sum(r["asr_whole"] is None for r in fallback),
            "paired": {k: metrics(fallback, k) for k in ("asr_duration", "asr_whole_if_available", "reference_whole")},
        },
        "leave_one_source_out_wording_change_minus_pooling_change": {
            "source_count": len(sources),
            "mae_range": [min(leave_out["mae"]), max(leave_out["mae"])],
            "pearson_range": [min(leave_out["pearson"]), max(leave_out["pearson"])],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", type=Path, default=ROOT / ".runtime/asr-intensity-study")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/asr-attribution.json")
    args = parser.parse_args()
    snapshot = (args.private / "utterances.jsonl").read_bytes()
    meta = json.loads((args.private / "metadata.json").read_text("utf-8"))
    if meta["private_snapshot_sha256"] != hashlib.sha256(snapshot).hexdigest():
        raise ValueError("Snapshot differs from the recorded extraction")
    rows = [json.loads(line) for line in snapshot.splitlines()]
    with gzip.open(ASSET, "rt", encoding="utf-8") as stream:
        frozen = json.load(stream)
    shadow = json.loads((args.private / "shadow.json").read_text("utf-8"))
    archived = {r["id"]: r for r in json.loads((ROOT / "docs/results/file-intensity.json").read_text("utf-8"))["clips"]}
    by_split = {}
    for split, model, count in (("development", shadow, 57), ("holdout", frozen, 43)):
        selected = [r for r in rows if r["split"] == split]
        if len(selected) != count or len({r["id"] for r in selected}) != count:
            raise ValueError("Unexpected official partition")
        result = []
        for r in selected:
            out = analyze(r, model)
            if split == "holdout":
                old = archived[r["id"]]
                if old["target"] != r["target"] or not equal(old["official_text"], out["reference_whole"]):
                    raise ValueError("Historical source label / official text model differs")
                if not equal(out["asr_duration"], r["original_file_value"]):
                    raise ValueError("Reconstructed utterance-weighted intensity differs from the file importer")
                out["archived_file"] = old["file"]
            result.append(out)
        by_split[split] = result
    held = by_split["holdout"]
    if set(archived) != {r["id"] for r in held} or (
        {r["source"] for r in held} & {r["source"] for r in by_split["development"]}
    ):
        raise ValueError("Unexpected overlap or mismatch in evaluation source groups")
    same = [r for r in held if equal(r["asr_duration"], r["archived_file"])]
    common_archive = [r for r in held if r["archived_file"] is not None and r["asr_duration"] is not None]
    doc = {
        "protocol": "attachment1-asr-wording-vs-aggregation-v1",
        "design": "Use the frozen 43-clip holdout and one seeded actual production ASR extraction. "
                  "Vary only the regression pooling for ASR duration vs whole; vary only transcript for ASR vs reference whole. "
                  "Both are observational sensitivity contrasts, not independent causal shares.",
        "alignment": "Minimum edit distance on lower-case word/alphanumeric tokens with contractions. Ties prefer diagonal, "
                     "then deletion, then insertion. WER pools words including untranscribed clips. Oracles restore only "
                     "deletions in ASR or remove those words in the reference; no reference input to the actual pipeline. "
                     "Reference duration proxy groups official tokens by aligned hypothesis segment, assigning deletions to "
                     "nearest hypothesis token in edit path, tie previous. Official words have no actual timestamps.",
        "comparisons": "Fixed common denominator across word/aggregation contrasts. Signed changes = after minus before, "
                       "4000 paired, source-video cluster bootstrap percentile intervals, seed 20261124. "
                       "Subgroup metrics use the scored clips in each arm.",
        "extraction": meta,
        "historical_recheck": {
            "archived": metrics(held, "archived_file"), "seeded_baseline": metrics(held, "asr_duration"),
            "paired_n": len(common_archive), "same_within_1e_9_including_both_null": len(same),
            "changed": len(held) - len(same), "changed_availability": sum(
                (r["asr_duration"] is None) != (r["archived_file"] is None) for r in held),
            "max_abs_prediction_difference_on_paired": max(abs(r["archived_file"] - r["asr_duration"]) for r in common_archive),
        },
        **{split: summarize(rs) for split, rs in by_split.items()},
        "holdout_sensitivity": holdout_sensitivity(held),
        "holdout_rows": held,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n", "utf-8", newline="\n")
    print(json.dumps({key: doc["holdout"][key] for key in ("word_edits", "common")}, indent=2))


if __name__ == "__main__":
    main()
