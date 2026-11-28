"""Evaluate frozen CH-SIMS teachers and replay the original desktop on the same inputs."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotion_assistant.chinese_affect import HAN
from emotion_assistant.foundation_embeddings import TEXT_DIGEST, embed_texts
from emotion_assistant.foundation_fusion import ASSET, cue_scores, predict_foundation_clip
from scripts import chsims_teacher_refinement as refined
from scripts import chsims_teacher_robustness as previous
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_teacher_cross_source import group_ids, labels
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file
from scripts.chsims_teacher_supervision import compare, numbers, read_full_report


WHOLE = {case: ("complete", mask) for case, mask in previous.WHOLE.items()}
PARTIAL = {case: (case, previous.WHOLE["complete"]) for case in previous.PARTIAL}
CASES = {**WHOLE, **PARTIAL, "audio_only_with_gap": ("audio_gap", previous.WHOLE["audio_only"])}
SUMMARY = {"one_missing": ("no_text", "no_audio", "no_visual"),
           "one_remaining": ("text_only", "audio_only", "visual_only"), "partial": tuple(PARTIAL)}
SUMMARY["missing"] = sum(SUMMARY.values(), ())


def input_pair(cache, clip):
    bundles = []
    for folder in ("media", previous.PREPARED):
        with np.load(cache / folder / f"{clip}.npz", allow_pickle=False) as raw:
            bundles.append((json.loads(raw["info"].item()), raw["speech"]))
    return bundles


def fixed_desktop(cache, test):
    baseline = test["models"]["baseline"]
    if sha_file(ASSET) != baseline["classifier_sha256"] or TEXT_DIGEST != baseline["embedding_digest"]:
        raise ValueError("The original desktop must use the pinned v1 assets")
    rows = test["rows"]
    results = {case: [] for case in CASES}
    audio_only = {case: [] for case in ("audio_only", "audio_only_with_gap")}
    weights = previous.CHANNEL
    for start in range(0, len(rows), 10):
        batch = [input_pair(cache, row["id"]) for row in rows[start:start + 10]]
        utterances = sorted({text for (clean, _), (gap, _) in batch
                             for text in (clean["text"], gap["text"], gap["joint_text"]) if HAN.search(text)})
        vectors = embed_texts(utterances) if utterances else []
        embeddings = dict(zip(utterances, vectors, strict=True))
        for (clean, speech), (gap, gap_speech) in batch:
            for case, (state, mask) in CASES.items():
                dropped = previous.PARTIAL.get(state, ())
                text = gap["joint_text"] if "joint_text" in dropped else (gap if "text" in dropped else clean)["text"]
                acoustic = (gap if "audio" in dropped else clean)["audio"]
                visual = (gap if "visual" in dropped else clean)["visual"]
                used_speech = gap_speech if "audio" in dropped else speech
                active = {name: bool(mask & bit) for name, bit in weights.items()}
                result = predict_foundation_clip(text if active["text"] else "",
                    embedding=embeddings.get(text) if active["text"] else None,
                    audio=acoustic if active["audio"] else None,
                    audio_embedding=used_speech if active["audio"] and used_speech.size else None,
                    visual=visual if active["visual"] else None)
                if result["status"] not in ("ok", "insufficient_evidence"):
                    raise RuntimeError(f"Cannot replay the original desktop: {result['status']}")
                results[case].append(result["label"] or "abstain")
                if case in audio_only:
                    scores, _ = cue_scores(acoustic)
                    audio_only[case].append(LABELS[scores.argmax()] if scores is not None else "abstain")
    for view, condition in (("full", "complete"), ("audio", "audio_only"), ("visual", "visual_only"),
                            ("text", "text_only")):
        if results[condition] != [row["baseline"][view]["label"] for row in rows]:
            raise ValueError(f"Original desktop replay differs from its cached {view} decisions")
    return results, audio_only


def condition_metrics(rows, predictions, indices, label="full"):
    values, contrasts = {}, {}
    for arm, choices in predictions.items():
        if arm == "teacher":
            continue
        pair = compare(rows, label, choices, predictions["teacher"], indices)
        values[arm], values["teacher"] = pair["reference"], pair["candidate"]
        contrasts[arm] = {key: pair[key] for key in (
            "corrected", "regressed", "candidate_minus_reference_95_source_bootstrap")}
    for scores in values.values():
        scores["wrong"] = len(indices) - scores["correct"] - scores["abstain"]
        scores["coverage"] = 1. - scores["abstain"] / len(indices)
    return {"n": len(indices), "metrics": values, "teacher_vs": contrasts}


def model_config(cache, test):
    model_path = cache / refined.DIRECTORY / "heads.json"
    prior_path = cache / "missing-evidence/fusion-heads.json"
    model, prior = (json.loads(path.read_text("utf8")) for path in (model_path, prior_path))
    if (model["implementation"] != sha_file(Path(refined.__file__)) or model["prior"] != sha_file(prior_path)
            or model["metadata"] != test["meta_sha256"] or prior["models"] != test["models"]):
        raise ValueError("The test requires the frozen teacher, original reference and matching release")
    for arms in model["heads"].values():
        for head in arms.values():
            if head["sha256"] != sha_file(cache / refined.DIRECTORY / head["artifact"]):
                raise ValueError(f"Classification head changed: {head['artifact']}")
    return model_path, model, prior


def evaluate(cache, output):
    test = read_full_report(cache, "test")
    model_path, model, prior = model_config(cache, test)
    rows = test["rows"]
    ids = {row["id"] for row in rows}
    sources = set()
    for phase, checksum in model["splits"].items():
        if checksum != sha_file(cache / f"evaluation-{phase}.json"):
            raise ValueError(f"The {phase} set changed after fitting")
        dev = read_full_report(cache, phase)["rows"]
        if ids & {row["id"] for row in dev}:
            raise ValueError(f"The {phase} test IDs overlap")
        sources.update(group_ids(dev))
    variants, _, extraction = previous.inputs(cache, test)
    if extraction != model["extractors"]:
        raise ValueError("The test must use the same gap extractors as development")
    mini = refined.mini_variants(cache, rows, variants)
    fixed, emotion2vec = fixed_desktop(cache, test)
    groups = {"official": np.arange(len(rows)),
              "unseen_sources": np.flatnonzero(~np.isin(group_ids(rows), list(sources)))}
    intact = np.flatnonzero(previous.observed_mask(variants["complete"]) == 7)
    configs = {arm: {mask: value[arm] for mask, value in model["heads"].items()} for arm in refined.FEATURES}
    cases, predictions = {}, {}
    for condition, (variant, mask) in CASES.items():
        blocks = variants[variant]
        old, available = previous.robust_predict(blocks, mask, prior)
        qwen = refined.new_predict(cache, blocks, mask, configs["qwen"], old["teacher"])
        alternative = refined.new_predict(cache, mini[variant], mask, configs["mini_hybrid"], None)
        balanced, teacher = qwen.copy(), old["teacher"].copy()
        for code, gate in model["audio_blends"].items():
            selected = available == int(code)
            if not selected.any():
                continue
            if gate["mini_share"]:
                balanced[selected] = refined.blend_audio_encoders(qwen[selected], alternative[selected],
                                                                  gate["mini_share"], gate["when_uncertain"])
            teacher[selected] = refined.accept_corrections(old["teacher"][selected], balanced[selected],
                                                           alternative[selected], model["corrections"][code])
        methods = {"teacher_previous": old["teacher"], "desktop_previous": old["desktop"],
                   "desktop_refit": refined.new_predict(cache, blocks, mask, configs["desktop"], None),
                   "minicpm_audio_hybrid": alternative, "qwen_refit": qwen, "teacher_balanced": balanced,
                   "teacher": teacher}
        decisions = {"desktop_fixed_v1": fixed[condition], **{name: labels(p) for name, p in methods.items()}}
        if condition in emotion2vec:
            decisions["emotion2vec_scores"] = emotion2vec[condition]
        if mask == 0 and any(pred != "abstain" for values in decisions.values() for pred in values):
            raise ValueError("An empty input produced an emotion prediction")
        predictions[condition] = decisions
        cohort = {**groups, "all_three_present": intact} if condition == "complete" else groups
        cases[condition] = {name: condition_metrics(rows, decisions, indices) for name, indices in cohort.items()
                            if len(indices)}
        if condition in emotion2vec:
            cases[condition]["audio_annotation"] = {name: condition_metrics(rows, decisions, indices, label="audio")
                                                    for name, indices in groups.items() if len(indices)}
        summary = cases[condition]["official"]["metrics"]
        print(f"{condition}: " + " / ".join(f"{name} {v['correct']}, {v['macro_f1']:.4f}"
                                              for name, v in summary.items()), flush=True)
    averages = {}
    truth = numbers(rows, "full")
    aggregate_arms = tuple(cases["complete"]["official"]["metrics"])
    for summary, conditions in SUMMARY.items():
        averages[summary] = {}
        for name, indices in groups.items():
            if not len(indices):
                continue
            averages[summary][name] = {
                "metrics": {arm: {metric: float(np.mean([cases[case][name]["metrics"][arm][metric]
                                                         for case in conditions]))
                                  for metric in ("accuracy", "macro_f1", "coverage")} for arm in aggregate_arms},
                "teacher_delta_95_source_bootstrap": {arm: previous.average_delta(rows, truth, predictions,
                                                                                  conditions, indices, arm)
                                                       for arm in aggregate_arms if arm != "teacher"},
            }
    report = {"test_ids_sha256": test["selected_ids_sha256"], "metadata_sha256": test["meta_sha256"],
              "model_sha256": sha_file(model_path), "previous_model_sha256": model["prior"],
              "sizes": {k: len(v) for k, v in groups.items()} | {"all_three_present": len(intact)},
              "gap_fraction": previous.FRACTION, "validation": model["heads"], "audio_blends": model["audio_blends"],
              "corrections": model["corrections"], "cases": cases, "averages": averages}
    save_json(output, report)
    save_json(cache / refined.DIRECTORY / "test-predictions.json", {"ids": [row["id"] for row in rows],
                                                                    "cases": predictions})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=6, user_api="blas"):
        evaluate(args.cache, args.output or args.cache / refined.DIRECTORY / "results.json")


if __name__ == "__main__":
    main()
