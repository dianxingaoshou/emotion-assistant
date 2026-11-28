"""Replay the original desktop audio baselines beside the frozen missing-evidence teacher."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotion_assistant.foundation_fusion import ASSET, cue_scores, predict_foundation_clip
from emotion_assistant.models import VERSIONS
from scripts import chsims_teacher_robustness as study
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_teacher_cross_source import group_ids, labels
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file
from scripts.chsims_teacher_supervision import compare, read_full_report


def original_audio(cache, folder, rows):
    shape = (len(rows), len(LABELS))
    direct, fixed = np.full(shape, np.nan), np.full(shape, np.nan)
    for index, row in enumerate(rows):
        with np.load(cache / folder / f"{row['id']}.npz", allow_pickle=False) as evidence:
            cue = json.loads(evidence["info"].item())["audio"]
            speech = evidence["speech"]
        probabilities, _ = cue_scores(cue)
        if probabilities is not None:
            direct[index] = probabilities
        decision = predict_foundation_clip("", audio=cue, audio_embedding=speech if speech.size else None)
        if decision["status"] == "ok" and decision["used_modalities"] == ["audio"]:
            fixed[index] = [decision["scores"][label] for label in LABELS]
        elif decision["status"] != "insufficient_evidence":
            raise ValueError(f"Unexpected fixed v1 audio result for {row['id']}")
    return {"emotion2vec_scores": direct, "desktop_fixed_v1": fixed}


def evaluate(cache, output):
    path = cache / "missing-evidence/fusion-heads.json"
    model = json.loads(path.read_text("utf-8"))
    test = read_full_report(cache, "test")
    if (model["implementation"] != sha_file(Path(study.__file__))
            or model["metadata"] != test["meta_sha256"] or model["models"] != test["models"]
            or test["models"]["baseline"]["classifier_sha256"] != sha_file(ASSET)):
        raise ValueError("Use the matching original desktop asset and frozen missing-evidence teacher")
    rows = test["rows"]
    test_ids = {row["id"] for row in rows}
    sources = set()
    for phase in ("train", "valid"):
        if model["splits"][phase] != sha_file(cache / f"evaluation-{phase}.json"):
            raise ValueError(f"The {phase} report changed after fitting")
        development = read_full_report(cache, phase)["rows"]
        if test_ids & {row["id"] for row in development}:
            raise ValueError(f"The test IDs overlap with {phase}")
        sources.update(group_ids(development))
    groups = {"official": np.arange(len(rows)),
              "unseen_sources": np.flatnonzero(~np.isin(group_ids(rows), list(sources)))}
    variants, _, extraction = study.inputs(cache, test)
    if extraction != model["gap_extractors"]:
        raise ValueError("Gap extraction differs from the frozen experiment")

    cases = {}
    for case, state, folder in (("audio_only", "complete", "media"),
                                ("audio_only_with_gap", "audio_gap", study.PREPARED)):
        predictions, _ = study.robust_predict(variants[state], study.CHANNEL["audio"], model)
        scores = original_audio(cache, folder, rows)
        scores.update({arm: predictions[arm] for arm in ("desktop", "teacher")})
        choices = {arm: labels(values) for arm, values in scores.items()}
        if case == "audio_only" and choices["desktop_fixed_v1"] != [row["baseline"]["audio"]["label"] for row in rows]:
            raise ValueError("Replayed original desktop decisions differ from the stored audio benchmark")
        shared = np.logical_and.reduce([np.isfinite(values[:, 0]) for values in scores.values()])
        cases[case] = {}
        for group, indices in {**groups, "shared_coverage": np.flatnonzero(shared)}.items():
            if not len(indices):
                continue
            cases[case][group] = {}
            for view in ("full", "audio"):
                pairs = {arm: compare(rows, view, choices[arm], choices["teacher"], indices)
                         for arm in ("emotion2vec_scores", "desktop_fixed_v1", "desktop")}
                metrics = {arm: pair["reference"] for arm, pair in pairs.items()}
                metrics["teacher"] = pairs["desktop"]["candidate"]
                difference_keys = ("corrected", "regressed", "candidate_minus_reference_95_source_bootstrap")
                cases[case][group][view] = {
                    "n": len(indices),
                    "metrics": metrics,
                    "teacher_vs_baselines": {arm: {key: pair[key] for key in difference_keys}
                                             for arm, pair in pairs.items()},
                }
    save_json(output, {"test_ids_sha256": test["selected_ids_sha256"], "meta_sha256": test["meta_sha256"],
                       "teacher_head_sha256": sha_file(path), "desktop_fixed_asset_sha256": sha_file(ASSET),
                       "extractor": VERSIONS["audio"], "audio_gap_fraction": study.FRACTION, "cases": cases})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=5, user_api="blas"):
        evaluate(args.cache, args.output or args.cache / "missing-evidence/original-audio-comparison.json")


if __name__ == "__main__":
    main()
