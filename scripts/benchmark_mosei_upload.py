"""Replay every labeled raw clip in competition E attachment 1 through the file importer.

Pass --data as the dataset root or the path to label-100.xlsx. The checked-in
baseline consists of outputs captured before the upgrade, keyed by source SHA.
The raw files and transcripts stay outside the repository.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LABELS = ("negative", "neutral", "positive")
REFERENCE = ROOT / "docs/results/mosei-upload.json"
OLD_SHA = "65dfea7fc0360bdd3f0cbed9772e30cfdc3a112219165b305c0c188b84a1caf3"


def audit_id(source_id, clip_id):
    source = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:16]
    clip = hashlib.sha256(f"{source_id}/{clip_id}".encode("utf-8")).hexdigest()[:16]
    return f"{source}/{clip}"


def attachment_rows(data):
    books = [data] if data.is_file() else list(data.rglob("label-100.xlsx"))
    if len(books) != 1:
        raise ValueError("Expected one label-100.xlsx under --data.")
    book = books[0]
    with ZipFile(book) as source:
        strings = ["".join(node.itertext()) for node in ET.fromstring(source.read("xl/sharedStrings.xml"))]
        sheet = ET.fromstring(source.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.findall(".//{*}sheetData/{*}row"):
        values = []
        for cell in row.findall("{*}c"):
            value = cell.findtext("{*}v", default="")
            values.append(strings[int(value)] if cell.get("t") == "s" else value)
        rows.append(values)
    samples = []
    for values in rows[1:]:
        row = dict(zip(rows[0], values, strict=True))
        score = float(row["label"])
        label = "neutral" if score == 0 else "positive" if score > 0 else "negative"
        if label != row["annotation"].strip().lower():
            raise ValueError("MOSEI polarity and intensity disagree")
        source_id, clip_id = row["video_id"], row["clip_id"]
        path = book.parent / source_id / (clip_id + ".mp4")
        samples.append((audit_id(source_id, clip_id), path, label, row["text"]))
    if len(samples) != 100 or len({r[0] for r in samples}) != len(samples):
        raise ValueError("Expected 100 distinct raw clips in attachment 1")
    return sorted(samples)


def clips(data):
    return [(identifier, path, label) for identifier, path, label, _ in attachment_rows(data)]


def predicted(scores):
    return max(scores, key=scores.get) if scores else None


def metrics(rows, key):
    counts = Counter((row["label"], row[key]) for row in rows)
    correct = sum(counts[t, t] for t in LABELS)
    scored = sum(row[key] is not None for row in rows)
    # An unavailable or unmapped prediction counts as a miss for its true class.
    f1 = []
    for target in LABELS:
        truth = sum(n for (y, _), n in counts.items() if y == target)
        called = sum(n for (_, p), n in counts.items() if p == target)
        f1.append(2 * counts[target, target] / (truth + called) if truth + called else 0.0)
    return {"n": len(rows), "scored": scored,
            "mapped_argmax": sum(row[key] in LABELS for row in rows), "correct": correct,
            "unmapped_argmax": sum(row[key] == "uncertain" for row in rows),
            "accuracy_all": correct / len(rows) if rows else None,
            "accuracy_scored": correct / scored if scored else None,
            "macro_f1_all": sum(f1) / len(f1) if rows else None}


def paired_accuracy_interval(rows, before, after):
    import numpy as np
    groups = {}
    for row in rows:
        values = groups.setdefault(row["id"].split("/")[0], [0, 0])
        values[0] += 1
        values[1] += int(row[after] == row["label"]) - int(row[before] == row["label"])
    if not groups:
        return None
    pairs = np.asarray(list(groups.values()))
    rng = np.random.default_rng(20261123)
    draws = pairs[rng.integers(0, len(pairs), size=(2000, len(pairs)))].sum(axis=1)
    return np.quantile(draws[:, 1] / draws[:, 0], [.025, .975]).tolist()


def input_groups(rows):
    return {name: metrics(group, "new_file") for name, group in (
        ("no_scored_visual_frame", [r for r in rows if r["same_frames_visual"] is None]),
        ("no_acoustic_contribution", [r for r in rows if "audio" not in r["used_modalities"]]),
        ("both_contribute", [r for r in rows if set(r["used_modalities"]) == {"visual", "audio"}]),
    )}


def constant_positive(rows):
    return metrics([dict(row, constant="positive") for row in rows], "constant")


def runtime_configuration():
    from emotion_assistant.core import FusionEngine
    from emotion_assistant.models import VERSIONS
    from emotion_assistant.paths import DATA
    return {"fusion": FusionEngine.from_project(DATA).config,
            "model_versions": {key: VERSIONS[key] for key in ("face", "visual", "audio")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=REFERENCE)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/mosei-upload.json")
    args = parser.parse_args()
    from emotion_assistant.paths import MODELS, configure_environment
    configure_environment()
    import cv2
    from emotion_assistant.local_media import analyze_media
    from emotion_assistant.models import ModelHub

    archive = json.loads(args.baseline.read_text("utf-8"))
    if archive["baseline_file_module_sha256"] != OLD_SHA:
        raise ValueError("Wrong pre-upgrade baseline")
    baseline = {r["id"]: r for r in archive["clips"]}
    data = clips(args.data)
    if set(baseline) != {k for k, _, _ in data}:
        raise ValueError("Archive and labeled clips differ")
    cv2.setNumThreads(2)
    hub = ModelHub(MODELS, enable_asr=False)
    rows = []
    for counter, (identifier, path, label) in enumerate(data, start=1):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        old = baseline[identifier]
        if old["sha256"] != digest or old["label"] != label:
            raise ValueError("Source/label mismatch for " + identifier)
        row = {k: old[k] for k in ("id", "sha256", "label", "old_file", "old_status")}
        try:
            r = analyze_media(path, model_dir=MODELS, model_hub=hub, on_sample=lambda *_: None,
                              cancelled=lambda: False)
            row.update(new_file=predicted(r["scores"]), same_frames_visual=predicted(r["visual_scores"]),
                       audio_status=r["audio_status"], used_modalities=r["used_modalities"])
        except (OSError, ValueError, RuntimeError) as error:
            row.update(new_file=None, same_frames_visual=None, used_modalities=[], error=str(error))
        rows.append(row)
        if counter % 10 == 0:
            print(f"Raw E clips {counter}/{len(data)}", flush=True)
    both = [r for r in rows if r["old_file"] is not None and r["new_file"] is not None]
    both_same_frames = [r for r in rows if r["same_frames_visual"] is not None and r["new_file"] is not None]
    report = {
        "dataset": "CMU-MOSEI competition E, attachment 1, all 100 labeled raw videos",
        "scope": "Frozen production importer without ASR; three-way whole-clip sentiment targets. "
                 "Media and transcripts are not distributed; no model or threshold fitted to these clips.",
        "metric": "Top canonical score from file summary; missing and unmapped count as wrong. "
                  "Coverage = clips returning any scores. Not prompt accuracy, self-reported mood, or treatment effect.",
        "baseline_file_module_sha256": OLD_SHA,
        "runtime_configuration": runtime_configuration(),
        "n_source_videos": len({r["id"].split("/")[0] for r in rows}),
        "label_counts": dict(Counter(r["label"] for r in rows)),
        "always_positive_comparison": constant_positive(rows),
        "old_status_counts": dict(Counter(r["old_status"] for r in rows)),
        "new_input_modes": dict(Counter("+".join(sorted(r["used_modalities"])) or "none" for r in rows)),
        "new_errors": sum("error" in r for r in rows),
        "new_by_observed_input": input_groups(rows),
        "all_clips": {k: metrics(rows, k) for k in ("old_file", "same_frames_visual", "new_file")},
        "with_both_old_and_new_scores": {k: metrics(both, k) for k in ("old_file", "new_file")},
        "with_both_current_visual_and_new_scores": {k: metrics(both_same_frames, k) for k in ("same_frames_visual", "new_file")},
        "accuracy_delta_95pct_source_cluster_bootstrap": {
            "method": "2000 paired percentile draws of source-video IDs, seed=20261123; baseline-relative, clip weighted",
            "new_minus_old_file": paired_accuracy_interval(rows, "old_file", "new_file"),
            "new_minus_same_frames_visual": paired_accuracy_interval(rows, "same_frames_visual", "new_file")},
        "clips": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", "utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "clips"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
