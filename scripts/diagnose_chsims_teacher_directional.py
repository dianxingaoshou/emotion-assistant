"""Audit the preceding teacher's negative-to-neutral errors on held-out CH-SIMS sources."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import chsims_teacher_directional as direction
from scripts import chsims_teacher_directional_pair as pair
from scripts import chsims_teacher_timeline as timeline
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_teacher_cross_source import group_ids, labels
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file
from scripts.chsims_teacher_supervision import numbers, read_full_report


def analyze(cache, output, with_pair=False):
    previous_path, model, original = direction.frozen_inputs(cache)
    reports = {split: read_full_report(cache, split) for split in ("train", "valid", "test")}
    known = set(np.concatenate([group_ids(reports[part]["rows"]) for part in ("train", "valid")]))
    test = reports["test"]
    all_sources = group_ids(test["rows"])
    held = np.flatnonzero(~np.isin(all_sources, list(known)))
    rows = [test["rows"][i] for i in held]
    data = timeline.read_data(cache, {**test, "rows": rows})
    scenario = "synchronous_gap"
    scores = timeline.score_examples(data, model["specialists"], original["heads"], np.arange(len(rows)))[scenario]
    rule = model["protected"][scenario]
    previous, _, active, local, _, _ = direction.prior_timeline(scores, data["time"], scenario, rule)
    no_offset, reference, _, _, _, _ = direction.prior_timeline(scores, data["time"], scenario,
                                                                {**rule, "neutral_offset": 0.})
    frozen = json.loads((cache / timeline.DIRECTORY / "test-predictions.json").read_text("utf8"))
    if frozen["ids"] != [row["id"] for row in test["rows"]]:
        raise ValueError("The preceding comparison used different CH-SIMS test IDs")
    runs = frozen["cases"][scenario]
    before, after, desktop = (np.asarray(runs[arm])[held] for arm in ("teacher_previous", "teacher", "desktop_fixed_v1"))
    if labels(previous[active]) != after[active].tolist():
        raise ValueError("The replay differs from the preceding timeline teacher")
    truth = numbers(rows, "full")
    negative_truth = truth == direction.NEGATIVE
    new_error = negative_truth & (after == "neutral") & (before != "neutral")
    corrected = negative_truth & (before == "neutral") & (after != "neutral")
    attention = new_error & (before == "negative")
    local_branches = direction.local_channel_scores(scores["windows"], data["time"]["windows"])
    probabilities = {"global_text": scores["text"], "global_audio": scores["audio"], "time_fusion": local,
                     **dict(zip(("local_text", "local_audio", "local_video"), local_branches, strict=True))}
    counts = {channel: {label: labels(value[attention]).count(label) for label in (*LABELS, "abstain")}
              for channel, value in probabilities.items()}
    before_margin = direction.log_ratios(reference[attention])[:, 0]
    adjusted_margin = direction.log_ratios(no_offset[attention])[:, 0]
    result = {"n": len(rows), "sources": len(set(all_sources[held])), "test_ids_sha256": test["selected_ids_sha256"],
              "timeline_model_sha256": sha_file(previous_path),
              "negative_as_neutral_before": int((negative_truth & (before == "neutral")).sum()),
              "negative_as_neutral_after": int((negative_truth & (after == "neutral")).sum()),
              "new_errors_by_before_prediction": {label: int((new_error & (before == label)).sum()) for label in LABELS},
              "corrected_negative_as_neutral": int(corrected.sum()),
              "newly_neutral_previous_negative_desktop_negative": int((attention & (desktop == "negative")).sum()),
              "newly_neutral_previous_negative_cue_winners": counts,
              "log_negative_to_neutral_median_before": float(np.median(before_margin)) if attention.any() else None,
              "log_negative_to_neutral_median_after_residual_without_offset": (
                  float(np.median(adjusted_margin)) if attention.any() else None),
              "negative_recovered_by_zeroing_neutral_offset": labels(no_offset[attention]).count("negative")}
    if with_pair:
        policy = pair.load(cache, previous_path)["policies"][scenario]
        desktop_scores = direction.desktop_gap_scores(cache, rows, data["time"])
        evidence = direction.describe_evidence(scores, data["time"], scenario, rule, desktop_scores)
        recovered = pair.predict(evidence, policy)
        paired = json.loads((cache / timeline.DIRECTORY / "pair-test-predictions.json").read_text("utf8"))
        choices = np.asarray(paired["cases"][scenario]["teacher"])[held]
        if frozen["ids"] != paired["ids"] or choices[active].tolist() != labels(recovered[active]):
            raise ValueError("The new pair inference differs from the frozen evaluation")
        gain = negative_truth & (after != "negative") & (choices == "negative")
        loss = negative_truth & (after == "negative") & (choices != "negative")
        replay = {"model_sha256": sha_file(cache / timeline.DIRECTORY / pair.MODEL),
                  "paired_negative_gained": int(gain.sum()), "paired_negative_lost": int(loss.sum())}
        if policy["rule"] is not None and attention.any():
            posterior = pair.pair_posterior(evidence, policy["rule"])
            votes = direction.support_counts(evidence["support"], np.full(len(rows), direction.NEGATIVE))
            replay.update({"earlier_seven_passing_support": int((attention & (votes >= policy["required_votes"])).sum()),
                           "earlier_seven_passing_posterior": int((attention & (posterior >= policy["recover_at"])).sum()),
                           "earlier_seven_posterior_median": float(np.median(posterior[attention])),
                           "negative_posterior_threshold": policy["recover_at"]})
        result["pair_recheck"] = replay
    save_json(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--with-pair", action="store_true", help="also diagnose the newly frozen pair review")
    args = parser.parse_args()
    with threadpool_limits(limits=5, user_api="blas"):
        analyze(args.cache, args.output, args.with_pair)


if __name__ == "__main__":
    main()
