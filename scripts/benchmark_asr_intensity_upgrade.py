"""Select an actual-ASR short-file policy on source-disjoint development and then evaluate it once."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from benchmark_asr_attribution import SourceBootstrap, align, by_duration, metrics, value
from benchmark_mosei_upload import ROOT
from extract_asr_intensity import PRIVATE
from emotion_assistant.english_intensity import _bundled
from emotion_assistant.file_asr_intensity import english_speech, estimate_file_intensity
from extract_asr_context import select_words
from emotion_assistant.asr_word_intensity import ASSET as CHAR_ASSET, robust_value


# Fixed before evaluating the candidates. Input words are from local speech recognition in all arms.
METHODS = ("pooled", "context", "longer")
ROBUST = ("char_pooled", "char_two_views")
RULE = {"min_development_pearson_gain": .02, "max_development_mae_increase": .005,
        "coverage": "Retain every development clip scored by the old policy.",
        "order": list(METHODS), "rank": "Highest Pearson on same old-covered development clips; ties by order."}
POLICY = PRIVATE / "selected-policy.json"
REPORT = ROOT / "reports/asr-intensity-upgrade.json"
FILE_RESULTS = ROOT / "reports/asr-file-end-to-end.json"


def read_rows(path):
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_rows(raw, retries, model, methods, char_model=None):
    result = []
    for row in raw:
        seg = row["segments"]
        retry = retries[row["id"]]
        if retry["media_sha256"] != row["media_sha256"]:
            raise ValueError("Context speech media hash mismatch")
        first = [s["text"] for s in seg]
        before = by_duration(first, seg, model)
        old_words = " ".join(first)
        out = {k: row[k] for k in ("id", "source", "target")}
        out.update(before=before, median=0., methods={}, word_edits={}, timing={})
        def edits(text):
            return dict(Counter(x.operation for x in align(row["official_text"], [{"text": text}])))
        out["word_edits"]["before"] = edits(old_words)
        for mode in methods:
            chosen, text, pred = "old_duration", old_words, before
            eligible = english_speech(seg)
            attempted = mode in ("context", "longer", "char_two_views") and eligible
            if mode in ROBUST and eligible:
                one = robust_value(old_words, model, char_model)
                if one is not None and mode == "char_pooled":
                    chosen, pred = "first_pass_char_word", one
                if one is not None and mode == "char_two_views" and retry["text"]:
                    two = robust_value(retry["text"], model, char_model)
                    if two is not None:
                        chosen, pred = "two_view_char_word", (one + two) / 2
            elif eligible and (mode == "pooled" or (attempted and retry["text"])):
                candidate, path = select_words(seg, retry["text"], mode)
                score = value(candidate, model)
                if score is not None:
                    chosen, text, pred = path, candidate, score
            out[mode] = pred
            out["methods"][mode] = chosen
            out["word_edits"][mode] = edits(text)
            if chosen == "two_view_char_word":
                out["word_edits"]["second_view"] = edits(retry["text"])
            out["timing"][mode] = {k: retry[k] for k in ("wall_seconds", "cpu_seconds")} if attempted else None
        result.append(out)
    return result


def select_dev(rows, methods):
    common = [r for r in rows if r["before"] is not None]
    old = metrics(common, "before")
    records, eligible = {}, []
    for mode in methods:
        score = metrics(common, mode)
        keep = (score["n"] == len(common) and score["pearson"] is not None and
                score["pearson"] >= old["pearson"] + RULE["min_development_pearson_gain"] and
                score["mae"] <= old["mae"] + RULE["max_development_mae_increase"])
        attempts = [r["timing"][mode] for r in rows if r["timing"][mode]]
        records[mode] = {"same_old_covered": score, "all": metrics(rows, mode), "eligible": keep,
                         "first_or_selected_transcript_words": edit_counts(rows, mode),
                         "second_view_on_actual_two_view_clips": edit_counts(
                             [r for r in rows if "second_view" in r["word_edits"]], "second_view")
                         if mode == "char_two_views" and any("second_view" in r["word_edits"] for r in rows) else None,
                         "extra_recognition_attempts": len(attempts), "cpu_seconds_total": sum(a["cpu_seconds"] for a in attempts),
                         "wall_seconds_median_per_retry": float(np.median([a["wall_seconds"] for a in attempts])) if attempts else None}
        if keep:
            eligible.append(mode)
    selected = max(eligible, key=lambda mode: records[mode]["same_old_covered"]["pearson"]) if eligible else "before"
    return {"clips": len(rows), "sources": len({r["source"] for r in rows}), "before": old,
            "before_transcript_words": edit_counts(rows, "before"),
            "rule": {**RULE, "order": list(methods)}, "candidates": records, "selected": selected}


def edit_counts(rows, key):
    counts = Counter()
    for row in rows:
        counts.update(row["word_edits"][key])
    ref = sum(counts[k] for k in ("match", "deletion", "substitution"))
    return {**{k: counts[k] for k in ("match", "deletion", "substitution", "insertion")},
            "reference_words": ref, "wer": sum(counts[k] for k in ("deletion", "substitution", "insertion")) / ref}


def summarize(rows, selected):
    common = [r for r in rows if r["before"] is not None and r[selected] is not None]
    old_covered = [r for r in rows if r["archive"] is not None]
    common_archive = [r for r in old_covered if r[selected] is not None and r["before"] is not None]
    attempts = [r["timing"][selected] for r in rows if selected != "before" and r["timing"][selected]]
    bootstrap = SourceBootstrap(common)
    return {
        "clips": len(rows), "sources": len({r["source"] for r in rows}), "common_count": len(common),
        "common_sources": len({r["source"] for r in common}), "before": metrics(rows, "before"),
        "after": metrics(rows, selected), "on_same_clips": {
            "before": metrics(common, "before"), "after": metrics(common, selected),
            "train_median": metrics(common, "median"), "paired_delta": bootstrap.compare("before", selected)},
        "original_archive_intersection": {"archived_scored": len(old_covered),
            "archive": metrics(common_archive, "archive"), "fresh": metrics(common_archive, "before"),
            "after": metrics(common_archive, selected)},
        "rescued": sum(r["before"] is None and r[selected] is not None for r in rows),
        "lost": sum(r["before"] is not None and r[selected] is None for r in rows),
        "transcripts": {"before": edit_counts(rows, "before"), "after_first_or_selected_view": edit_counts(rows, selected),
                        "second_view_on_actual_two_view_clips": edit_counts(
                            [r for r in rows if "second_view" in r["word_edits"]], "second_view")
                        if any("second_view" in r["word_edits"] for r in rows) else None},
        "method_counts": dict(Counter(r["methods"].get(selected, "old_duration") for r in rows)),
        "timing": {"retry_attempts": len(attempts), **{
            k: {"total": sum(a[k] for a in attempts), "median_per_retry": float(np.median([a[k] for a in attempts]))
                if attempts else None, "p95_per_retry": float(np.quantile([a[k] for a in attempts], .95)) if attempts else None}
            for k in ("wall_seconds", "cpu_seconds")}},
        "clips_paired": [{"id": r["id"], "target": r["target"], "archived": r["archive"],
                          "before": r["before"], "after": r[selected], "path": r["methods"].get(selected, "old_duration")}
                         for r in rows]}


def time_deployed_scorer(raw, rows, repeats=100):
    # Reuse real ASR strings and spans, warming the packaged models before measurement.
    # All 100 audio tracks were checked to be <= 30 s. Duration participates only in eligibility.
    inputs = [[{"text": s["text"], "valid": True, "t_start": s["start"], "t_end": s["end"], "session_id": r["id"]}
               for s in r["segments"]] for r in raw]
    def score_pass():
        for segments, row in zip(inputs, rows, strict=True):
            yield estimate_file_intensity(segments, source_id=row["id"], duration=30.)
    for estimate, row in zip(score_pass(), rows, strict=True):
        expected = row["char_pooled"] if row["methods"]["char_pooled"] != "old_duration" else None
        if (estimate is None) != (expected is None) or (estimate and abs(expected - estimate["intensity"]["value"]) > 1e-12):
            raise ValueError("Packaged scoring path disagrees with the frozen policy")
    wall, cpu = time.perf_counter(), time.process_time()
    for _ in range(repeats):
        list(score_pass())
    n = len(rows) * repeats
    return {"files_per_repeat": len(rows), "repeats": repeats, "cpu_ms_per_file": 1000*(time.process_time() - cpu)/n,
            "wall_ms_per_file": 1000*(time.perf_counter() - wall)/n,
            "scope": "Warm additional file scorer only, measured locally after first ASR; includes eligibility and pooling. No decode/model loading/UI."}


def has_non_english_tag(tags):
    # The file benchmark serializes a missing recognizer tag as "unreported".
    return any(tag not in (None, "unreported", "en") for tag in tags)


def audit_production_gates(report_path, file_results):
    """Cross-check the frozen development selection and independent real-file acceptance run."""
    report = json.loads(report_path.read_text("utf-8"))
    gates = json.loads((PRIVATE / "development-language-gates.json").read_text("utf-8"))
    dev = read_rows(PRIVATE / "development.jsonl")
    if set(gates) != {r["id"] for r in dev}:
        raise ValueError("Development language gate coverage mismatch")
    dev_disagreements = [r["id"] for r in dev if gates[r["id"]] != english_speech(r["segments"])]
    if dev_disagreements:
        raise ValueError("Production and selection eligibility differ on development: recompute eligibility audit.")
    acceptance = json.loads(file_results.read_text("utf-8"))
    actual = {r["id"]: r for r in acceptance["clips"]}
    proxy = report["holdout"]["clips_paired"]
    if set(actual) != {r["id"] for r in proxy}:
        raise ValueError("Production and selection holdouts differ")
    def same(a, b):
        return (a is None and b is None) or (a is not None and b is not None and abs(a-b) < 1e-10)
    if any(not same(actual[r["id"]]["first_pass"], r["before"]) for r in proxy):
        raise ValueError("Paired file baseline mismatch")
    changed = [r for r in proxy if not same(r["after"], actual[r["id"]]["file"])]
    if any(r["path"] != "first_pass_char_word" or not same(r["before"], actual[r["id"]]["file"]) for r in changed):
        raise ValueError("Unexpected end-to-end difference; inspect the source files")
    actual_counts = Counter(row["method"] for row in actual.values())
    if actual_counts != acceptance["method_counts"]:
        raise ValueError("Stored per-file methods disagree with the end-to-end totals")
    for row in actual.values():
        if row["method"] in ("abstained", "segment_weighted") and not same(row["file"], row["first_pass"]):
            raise ValueError("Published fallback disagrees with its first-pass value")
        if row["method"] == "asr_word_character_pool" and has_non_english_tag(row["asr_languages"]):
            raise ValueError("Published word/character estimate violates its ASR language gate")
    for row in changed:
        actual_row = actual[row["id"]]
        if actual_row["method"] != "segment_weighted" or not has_non_english_tag(actual_row["asr_languages"]):
            raise ValueError("Language fallback not confirmed by an observed first-pass tag")
    report["holdout_input"] = "Transcript-cache replay, valid text and timing only; legacy snapshot did not store ASR language tags. Use production_gate_audit for final application performance."
    report["production_gate_audit"] = {
        "development": {"clips": len(gates), "eligible": sum(gates.values()), "fallback": len(gates) - sum(gates.values()),
                        "eligibility_mismatches": dev_disagreements, "selection_and_threshold_unchanged": True},
        "holdout": {"clips": len(proxy), "legacy_value_mismatches": 0,
                    "actual_default": acceptance["default_on_paired"], "actual_legacy": acceptance["first_pass_on_paired"],
                    "paired_delta": acceptance["default_minus_first_pass"],
                    "actual_method_counts": dict(actual_counts), "word_character_used": actual_counts["asr_word_character_pool"],
                    "extra_language_fallbacks": [{"id": r["id"], "first_pass_asr_languages": actual[r["id"]]["asr_languages"]} for r in changed],
                    "lost": acceptance["lost"],
                    "gained": acceptance["gained"], "end_to_end_evidence": file_results.name,
                    "end_to_end_sha256": digest(file_results)}}
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", "utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("select", "evaluate", "audit"), required=True)
    parser.add_argument("--out", type=Path, default=REPORT, help="Replay report to write (evaluate) or verify and update (audit).")
    parser.add_argument("--file-results", type=Path, default=FILE_RESULTS, help="Paired real-media output of benchmark_file_intensity.py --current.")
    args = parser.parse_args()
    if args.stage == "audit":
        audit_production_gates(args.out, args.file_results)
        return
    cache = PRIVATE / "english-context.jsonl"
    if args.stage == "select" and not cache.exists():
        cache = cache.with_suffix(".partial")
    retries = {r["id"]: r for r in read_rows(cache)}
    dev = read_rows(PRIVATE / "development.jsonl")
    model = json.loads((PRIVATE / "shadow.json").read_text("utf-8"))
    first_round = select_dev(paired_rows(dev, retries, model, METHODS), METHODS)
    dev_result = dict(first_round)
    char_model = None
    if first_round["selected"] == "before":
        with gzip.open(PRIVATE / "asr-character-intensity.json.gz", "rt", encoding="utf-8") as src:
            char_model = json.load(src)
        dev_result = select_dev(paired_rows(dev, retries, model, ROBUST, char_model), ROBUST)
        dev_result["first_round"] = first_round
        dev_result["character_training"] = json.loads((PRIVATE / "asr-character-training.json").read_text("utf-8"))
    dev_result["shadow_version"] = model["model_version"]
    dev_result["development_sha256"] = digest(PRIVATE / "development.jsonl")
    if args.stage == "select":
        POLICY.write_text(json.dumps(dev_result, indent=2, allow_nan=False) + "\n", "utf-8")
        print(json.dumps(dev_result, indent=2, allow_nan=False))
        return
    if not POLICY.exists() or json.loads(POLICY.read_text("utf-8")) != dev_result:
        raise ValueError("Run development selection and freeze the decision before evaluation.")
    model = _bundled()
    selected = dev_result["selected"]
    held = [r for r in read_rows(PRIVATE / "utterances.jsonl") if r["split"] == "holdout"]
    rows = paired_rows(held, retries, model, () if selected == "before" else (selected,), char_model)
    archive = {r["id"]: r for r in json.loads((ROOT / "docs/results/file-intensity.json").read_text("utf-8"))["clips"]}
    if {r["id"] for r in rows} != set(archive):
        raise ValueError("Original holdout changed.")
    for r in rows:
        if r["target"] != archive[r["id"]]["target"]:
            raise ValueError("Original labels changed.")
        r["archive"] = archive[r["id"]]["file"]
    report = {"protocol": "short-english-file-asr-intensity-v1", "development": dev_result,
              "model_version": model["model_version"], "selected": selected,
              "private_retries_sha256": digest(cache), "private_baseline_sha256": digest(PRIVATE / "utterances.jsonl"),
              "bootstrap": {"draws": 4000, "seed": 20261124, "unit": "source video; paired percentile intervals"},
              "runtime_inputs": "Selected char_pooled uses only first-pass English ASR on tracks <=30s. No second recognition, label, reference or raw waveform.",
              "word_model_role": "Previously deployed English word regressor for holdout and product; all-source-disjoint shadow only on development.",
              "holdout": summarize(rows, selected)}
    if selected == "char_pooled":
        report["asset"] = {"gzip_bytes": CHAR_ASSET.stat().st_size, "sha256": digest(CHAR_ASSET)}
        report["holdout"]["timing"]["incremental_file_scoring"] = time_deployed_scorer(held, rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", "utf-8", newline="\n")
    print(json.dumps({k: v for k, v in report["holdout"].items() if k != "clips_paired"}, indent=2))


if __name__ == "__main__":
    main()
