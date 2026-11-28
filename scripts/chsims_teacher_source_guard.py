"""Select a neutral safeguard for window-based negative recovery using development sources."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import chsims_teacher_directional as directional
from scripts import chsims_teacher_directional_pair as pair
from scripts import chsims_teacher_source_review as review
from scripts import chsims_teacher_timeline as timeline
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_teacher_cross_source import metrics
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file


MODEL = "source-guard.json"
NEGATIVE, NEUTRAL, POSITIVE = review.NEGATIVE, review.NEUTRAL, review.POSITIVE


def neutral_checks(batch):
    cues = review.conflict_evidence(batch)
    earlier = cues["reference"]
    window = cues["support"][:, 1:4]
    neutral_lead = window[:, :, NEUTRAL] - np.maximum(window[:, :, NEGATIVE], window[:, :, POSITIVE])
    return {"earlier_negative": np.isfinite(earlier[:, NEGATIVE]) & (earlier.argmax(1) == NEGATIVE),
            "earlier_margin": earlier[:, NEGATIVE] - earlier[:, NEUTRAL],
            "strongest_window_neutral": np.maximum(0., np.nan_to_num(neutral_lead).max(1))}


def protect(previous, proposed, checks, settings):
    if settings is None:
        return previous
    change = (previous.argmax(1) == NEUTRAL) & (proposed.argmax(1) == NEGATIVE)
    accepted = (change & checks["earlier_negative"] & (checks["earlier_margin"] >= settings["minimum_anchor_margin"])
                & (checks["strongest_window_neutral"] <= settings["maximum_neutral_lead"]))
    return np.where(accepted[:, None], proposed, previous)


def predict(batch, previous, proposed, chosen):
    return protect(previous, proposed, neutral_checks(batch), chosen["policy"])


def select(development, checkpoint):
    rule = checkpoint["preceding"]["policies"][review.CONDITION]
    records = {}
    for phase, batch in development.items():
        reference = pair.predict(batch, rule)
        proposed, _ = review.predict(batch, reference, checkpoint["review"])
        _, group, count = np.unique(batch["sources"], return_inverse=True, return_counts=True)
        records[phase] = (batch["truth"], reference, proposed, neutral_checks(batch), group, count)
    before = {phase: metrics(values[0], values[1]) for phase, values in records.items()}
    floors = {phase: np.diag(np.asarray(score["confusion"])[:, :len(LABELS)]) for phase, score in before.items()}
    winner = {"policy": None, "before": before, "after": before}
    best = (0., 0., 0.)
    for margin in (0., .025, .05, .10, .15, .20, .25):
        for ceiling in (.15, .20, .30, .45, .60, 1.):
            policy = {"minimum_anchor_margin": margin, "maximum_neutral_lead": ceiling}
            updated, by_source = {}, {}
            for phase, (truth, previous, proposed, checks, group, count) in records.items():
                protected = protect(previous, proposed, checks, policy)
                after = metrics(truth, protected)
                hits = np.diag(np.asarray(after["confusion"])[:, :len(LABELS)])
                if np.any(hits < floors[phase]) or hits[NEGATIVE] <= floors[phase][NEGATIVE]:
                    break
                delta = ((np.isfinite(protected[:, 0]) & (protected.argmax(1) == truth)).astype(int) -
                         (np.isfinite(previous[:, 0]) & (previous.argmax(1) == truth)))
                by_source[phase] = float(np.mean(np.bincount(group, weights=delta) / count))
                updated[phase] = after
            if len(updated) != len(records) or updated["valid"]["macro_f1"] < before["valid"]["macro_f1"]:
                continue
            accuracy = updated["valid"]["accuracy"] - before["valid"]["accuracy"]
            macro = updated["valid"]["macro_f1"] - before["valid"]["macro_f1"]
            score = .35 * accuracy + .25 * macro + .25 * by_source["valid"] + .15 * by_source["train"]
            rank = (score, by_source["train"], -ceiling)
            if rank > best:
                best = rank
                winner = {"policy": policy, "before": before, "after": updated, "source_accuracy_delta": by_source}
    return winner


def fit(cache):
    previous, timeline_checkpoint, original = directional.frozen_inputs(cache)
    checkpoint = review.load(cache, previous)
    provenance = {"timeline": sha_file(previous), "pair": checkpoint["previous_model"], "splits": checkpoint["splits"]}
    for split, expected in provenance["splits"].items():
        if sha_file(cache / f"evaluation-{split}.json") != expected:
            raise ValueError(f"The frozen {split} labels changed")
    development = review.development(cache, timeline_checkpoint, original, provenance)
    chosen = select(development, checkpoint)
    print("Neutral protection", chosen["policy"], [(phase, chosen["before"][phase]["correct"], count["correct"])
                                  for phase, count in chosen["after"].items()], flush=True)
    save_json(cache / timeline.DIRECTORY / MODEL,
              {"implementation": sha_file(Path(__file__)), "source_model": sha_file(cache / timeline.DIRECTORY / review.MODEL),
               "guard": chosen})


def load(cache, timeline_path):
    calibration = review.load(cache, timeline_path)
    stored = json.loads((cache / timeline.DIRECTORY / MODEL).read_text("utf8"))
    if (stored["implementation"] != sha_file(Path(__file__)) or
            stored["source_model"] != sha_file(cache / timeline.DIRECTORY / review.MODEL)):
        raise ValueError("Fit the neutral safeguard on the frozen development inputs first")
    return {**calibration, "kind": "source_guard", "guard": stored["guard"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    args = parser.parse_args()
    with threadpool_limits(limits=5, user_api="blas"):
        fit(args.cache)


if __name__ == "__main__":
    main()
