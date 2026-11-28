"""Replay local character-based desktop controls on the frozen teacher test clips."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotion_assistant.foundation_fusion import ASSET, LABELS, bundled_head, predict_with_head
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_teacher_cross_source import group_ids
from scripts.chsims_teacher_data import CACHE, sha_file
from scripts.chsims_teacher_robustness import CHANNEL
from scripts.chsims_teacher_supervision import compare, read_full_report
from scripts.chsims_teacher_timeline import DIRECTORY
from scripts.evaluate_chsims_teacher_refinement import CASES, SUMMARY, condition_metrics, desktop_case_inputs, input_pair


CHARACTER = "desktop_character_v1"
CHARACTER_FIXED = "desktop_character_fixed_weights_v1"
BASE = "desktop_base_v1"
FROZEN_TEACHER = "neutral-guard-test-predictions.json"


def replay(cache, rows):
    head = bundled_head()
    if head is None:
        raise ValueError("The frozen desktop v1 head is unavailable")
    matched = {**head, "no_qwen_weights": head["foundation_weights"]}
    arms = {CHARACTER: (head, "no_qwen"), CHARACTER_FIXED: (matched, "no_qwen"), BASE: (head, "base")}
    result = {name: {method: [] for method in arms} for name in CASES}
    for row in rows:
        clip = row["id"]
        (clean, speech), (gap, gap_speech) = input_pair(cache, clip)
        source = cache / "media" / f"{clip}.npz"
        if gap["provenance"]["input"] != sha_file(source):
            raise ValueError(f"Gap input no longer matches the clean clip {clip}")
        for name, evidence in desktop_case_inputs(clean, speech, gap, gap_speech):
            for method, (classifier, variant) in arms.items():
                prediction = predict_with_head(classifier, **evidence, variant=variant)
                if prediction["status"] not in ("ok", "insufficient_evidence") or prediction["foundation_used"]:
                    raise ValueError(f"Cannot replay a non-LLM desktop result for {clip}/{name}")
                if method == BASE and prediction["audio_embedding_used"]:
                    raise ValueError("The base control must disable the dedicated audio embedding head")
                result[name][method].append(prediction["label"] or "abstain")
    return result


def run(cache, output, external=None):
    if external:
        report = json.loads((external / "report.json").read_text("utf-8"))
        manifest = json.loads((external / "cohort.json").read_text("utf-8"))
        expected = [(row["id"], row["truth"]) for row in manifest["rows"]]
        if expected != [(row["id"], row["truth"]) for row in report["rows"]] or (
                manifest["labels_sha256"] != report["meta_sha256"]):
            raise ValueError("CMMA predictions and the cohort manifest disagree")
    else:
        report = read_full_report(cache, "test")
    rows = report["rows"]
    ids = [row["id"] for row in rows]
    fingerprint = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    if fingerprint != report["selected_ids_sha256"] or sha_file(ASSET) != report["models"]["baseline"]["classifier_sha256"]:
        raise ValueError("Expected the frozen desktop v1 weights and the original clip order")
    data = external or cache
    comparison = data / DIRECTORY / FROZEN_TEACHER
    frozen = json.loads(comparison.read_text("utf-8"))
    if frozen["ids"] != ids or set(frozen["cases"]) != set(CASES):
        raise ValueError("The desktop and teacher comparisons must use identical test clips and conditions")
    predictions = replay(data, rows)

    groups = {"external" if external else "official": np.arange(len(rows))}
    if not external:
        development = [row for split in ("train", "valid") for row in read_full_report(cache, split)["rows"]]
        if set(ids) & {row["id"] for row in development}:
            raise ValueError("Development and test clips overlap")
        groups["unseen_sources"] = np.flatnonzero(~np.isin(group_ids(rows), np.unique(group_ids(development))))

    cases = {}
    for name, arms in predictions.items():
        published = frozen["cases"][name]
        methods = {**arms, "desktop_fixed_v1": published["desktop_fixed_v1"], "teacher": published["teacher"]}
        _, mask = CASES[name]
        if not mask & CHANNEL["text"] and methods[CHARACTER_FIXED] != methods["desktop_fixed_v1"]:
            raise ValueError(f"The matched control differs without text in {name}")
        if name == "no_evidence" and any(set(pred) != {"abstain"} for pred in methods.values()):
            raise ValueError("Empty evidence must produce only abstentions")
        cases[name] = {group: condition_metrics(rows, methods, indices) for group, indices in groups.items()}
        for group, indices in groups.items():
            isolated = compare(rows, "full", methods[CHARACTER_FIXED], methods["desktop_fixed_v1"], indices)
            cases[name][group]["qwen_text_fixed_weight_comparison"] = {key: isolated[key] for key in (
                "corrected", "regressed", "candidate_minus_reference_95_source_bootstrap")}

    summaries = {section: {group: {"metrics": {arm: {
        metric: float(np.mean([cases[case][group]["metrics"][arm][metric] for case in names]))
        for metric in ("accuracy", "macro_f1", "coverage")} for arm in cases["complete"][group]["metrics"]}}
        for group in groups} for section, names in SUMMARY.items()}
    save_json(output, {"test_ids_sha256": fingerprint, "metadata_sha256": report["meta_sha256"],
                       "desktop_model_sha256": sha_file(ASSET), "teacher_predictions_sha256": sha_file(comparison),
                       "classes": list(LABELS), "cohorts": {
                           group: {"n": len(indices), "groups": len(set(group_ids(rows)[indices]))}
                           for group, indices in groups.items()},
                       "arms": {CHARACTER: "v1 character n-grams, emotion2vec+ embedding and desktop cues",
                                CHARACTER_FIXED: "v1 character n-grams and original Qwen-v1 fusion weights",
                                BASE: "v1 character n-grams and desktop cues without the speech embedding head",
                                "desktop_fixed_v1": "frozen v1 with Qwen text embeddings",
                                "teacher": "frozen neutral-guard teacher"},
                       "cases": cases, "averages": summaries})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    parser.add_argument("--external", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2, user_api="blas"):
        run(args.cache, args.output, args.external)


if __name__ == "__main__":
    main()
