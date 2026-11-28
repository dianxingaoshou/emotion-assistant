"""Use calibrated Omni evidence to make selective corrections to desktop speech judgments."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_audio_prior import audio_logits, collection_ids, fit_audio_head, held_out_collections
from scripts.chsims_teacher_cross_source import group_ids, labels, metrics, prior_adjustment
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file
from scripts.chsims_teacher_supervision import compare, numbers, read_blocks, read_full_report, source_folds, temperature


NEUTRAL = LABELS.index("neutral")


def review(desktop, omni, threshold, max_desktop, neutral_threshold, *, protect_neutral=False):
    """The classifier's uncertainty controls an override, independently of its neutral label."""
    available = np.isfinite(desktop[:, 0])
    can_review = np.isfinite(omni[:, 0])
    proposal = np.where(can_review[:, None], omni, -np.inf).argmax(axis=1)
    current = np.where(available[:, None], desktop, -np.inf).argmax(axis=1)
    confidence = np.where(can_review, omni[np.arange(len(omni)), proposal], 0.)
    desktop_confidence = np.where(available, desktop[np.arange(len(desktop)), current], 0.)
    required = np.where(proposal == NEUTRAL, neutral_threshold, threshold)
    accept = can_review & (confidence >= required) & (~available | (desktop_confidence <= max_desktop))
    if protect_neutral:
        accept &= ~available | (current != NEUTRAL)
    output = desktop.copy()
    output[accept] = omni[accept]
    return output


def choose_gate(y, groups, desktop, omni, *, protect_neutral):
    selectors = [groups == group for group in sorted(set(groups))]
    desk = metrics(y, desktop)
    min_hits = desk["confusion"][NEUTRAL][NEUTRAL]
    baseline = np.array([metrics(y, desktop, subset)["accuracy"] for subset in selectors])
    settings = [(q, d, n) for q in (.55, .65, .75, .85, .9)
                for d in (.5, .65, .8, 1.) for n in (.55, .65, .75, .85)]
    settings.append((1.01, 0., 1.01))

    def rank(choice):
        prediction = review(desktop, omni, *choice, protect_neutral=protect_neutral)
        stats = metrics(y, prediction)
        changes = np.array([metrics(y, prediction, subset)["accuracy"] for subset in selectors]) - baseline
        eligible = stats["confusion"][NEUTRAL][NEUTRAL] >= min_hits
        return (eligible, .65 * float(np.mean(changes)) + .35 * float(np.quantile(changes, .25)),
                stats["macro_f1"], stats["accuracy"], choice[0])

    return max(settings, key=rank)


def training_targets(rows, tr, truth, heads, thresholds, scales):
    probabilities = np.full((len(rows), 3), np.nan)
    targets = [None] * len(rows)
    for fold, (ti, vi) in enumerate(source_folds(rows)):
        raw = {arm: audio_logits(tr[arm], fit_audio_head(tr[arm], truth, head, ti), vi)
               for arm, head in heads.items()}
        active = np.isfinite(tr["teacher"]["scores"][ti, 0])
        raw["teacher"] += prior_adjustment(truth[ti[active]], heads["teacher"]["balance_exponent"], 1.)
        base = softmax(raw["baseline"] / scales["baseline"], axis=1)
        teacher = softmax(raw["teacher"] / scales["teacher"], axis=1)
        probabilities[vi] = review(base, teacher, *thresholds, protect_neutral=True)
        for i in vi:
            scores = probabilities[i]
            targets[i] = {"id": rows[i]["id"], "fold": fold,
                          "probabilities": scores.tolist() if np.isfinite(scores[0]) else None}
    return {"split": "train", "classes": LABELS, "groups": "video_id", "rows": targets,
            "metrics": metrics(truth, probabilities)}


def fit(cache):
    train, valid = (read_full_report(cache, s) for s in ("train", "valid"))
    original_path = cache / "teacher-supervision.json"
    old = json.loads(original_path.read_text("utf-8"))
    if (old["extractors"] != train["models"] or old["extractors"] != valid["models"]
            or old["meta_sha256"] != train["meta_sha256"] or old["meta_sha256"] != valid["meta_sha256"]):
        raise ValueError("Model and development evidence use different releases")
    if {r["id"] for r in train["rows"]} & {r["id"] for r in valid["rows"]}:
        raise ValueError("Training and validation IDs overlap")
    tr, va = (read_blocks(cache, r)["audio"] for r in (train, valid))
    yt, yv = (numbers(r["rows"], "audio") for r in (train, valid))
    training_groups, valid_groups = (collection_ids(r["rows"]) for r in (train, valid))
    heads = {"baseline": old["heads"]["audio"]["desktop_calibrated"],
             "teacher": old["heads"]["audio"]["omni_selected"]}
    logits = {arm: np.full((len(yv), 3), np.nan) for arm in heads}
    for _, ti, vi in held_out_collections(training_groups, valid_groups):
        for arm, head in heads.items():
            fitted = fit_audio_head(tr[arm], yt, head, ti)
            logits[arm][vi] = audio_logits(va[arm], fitted, vi)
        active = np.isfinite(tr["teacher"]["scores"][ti, 0])
        logits["teacher"][vi] += prior_adjustment(yt[ti[active]], heads["teacher"]["balance_exponent"], 1.)
    temperature_scales = {}
    for arm, values in logits.items():
        active = np.isfinite(values[:, 0])
        temperature_scales[arm] = temperature(values[active], yv[active])
    # Preserve the calibrated desktop decision distribution at deployment.
    temperature_scales["baseline"] = heads["baseline"]["temperature"]
    desktop = softmax(logits["baseline"] / temperature_scales["baseline"], axis=1)
    omni = softmax(logits["teacher"] / temperature_scales["teacher"], axis=1)
    selected = choose_gate(yv, valid_groups, desktop, omni, protect_neutral=False)
    protected = choose_gate(yv, valid_groups, desktop, omni, protect_neutral=True)
    outputs = {"review": metrics(yv, review(desktop, omni, *selected)),
               "protected": metrics(yv, review(desktop, omni, *protected, protect_neutral=True))}
    targets_path = cache / "audio-review-training-targets.json"
    save_json(targets_path, training_targets(train["rows"], tr, yt, heads, protected, temperature_scales))
    save_json(cache / "audio-review-gate.json", {
        "script_sha256": sha_file(Path(__file__)), "checkpoint_sha256": sha_file(original_path),
        "development": {s: sha_file(cache / f"evaluation-{s}.json") for s in ("train", "valid")},
        "temperature": temperature_scales, "thresholds": selected, "protected_thresholds": protected,
        "targets_sha256": sha_file(targets_path),
        "prior": prior_adjustment(yt[np.isfinite(tr["teacher"]["scores"][:, 0])],
                                  heads["teacher"]["balance_exponent"], 1.).tolist(),
        "validation": {"desktop": metrics(yv, desktop), "omni": metrics(yv, omni), **outputs}})
    print("Frozen audio review", selected, protected, {k: v["correct"] for k, v in outputs.items()}, flush=True)


def from_blocks(cache, desktop, omni):
    checkpoint = cache / "teacher-supervision.json"
    config = json.loads((cache / "audio-review-gate.json").read_text("utf-8"))
    if config["script_sha256"] != sha_file(Path(__file__)) or config["checkpoint_sha256"] != sha_file(checkpoint):
        raise ValueError("The audio review requires the matching frozen heads")
    heads = json.loads(checkpoint.read_text("utf-8"))["heads"]["audio"]
    idx = np.arange(len(omni["context"]))
    scores = audio_logits(desktop, heads["desktop_calibrated"], idx)
    base = softmax(scores / config["temperature"]["baseline"], axis=1)
    alternative = audio_logits(omni, heads["omni_selected"], idx) + config["prior"]
    teacher = softmax(alternative / config["temperature"]["teacher"], axis=1)
    return {"review": review(base, teacher, *config["thresholds"]),
            "protected": review(base, teacher, *config["protected_thresholds"], protect_neutral=True)}, config


def evaluate(cache, output):
    test = read_full_report(cache, "test")
    old = json.loads((cache / "teacher-supervision.json").read_text("utf-8"))
    if old["extractors"] != test["models"] or old["meta_sha256"] != test["meta_sha256"]:
        raise ValueError("Model and test evidence use different releases")
    rows = test["rows"]
    blocks = read_blocks(cache, test)["audio"]
    scores, method = from_blocks(cache, blocks["baseline"], blocks["teacher"])
    desktop = old["heads"]["audio"]["desktop_calibrated"]
    output_labels = {key: labels(value) for key, value in scores.items()}
    original_labels = labels(softmax(audio_logits(blocks["baseline"], desktop, np.arange(len(rows))) /
                                      desktop["temperature"], axis=1))
    known = set()
    test_ids = {r["id"] for r in rows}
    for split, digest in method["development"].items():
        if digest != sha_file(cache / f"evaluation-{split}.json"):
            raise ValueError("Development evidence changed after calibration")
        development = read_full_report(cache, split)["rows"]
        if test_ids & {row["id"] for row in development}:
            raise ValueError(f"Test and {split} IDs overlap")
        known.update(group_ids(development))
    groups = {"official": np.arange(len(rows)), "unseen_sources": np.flatnonzero(~np.isin(group_ids(rows), list(known))),
              "audio_text_disagreement": np.flatnonzero([r["truth"]["audio"] != r["truth"]["text"] for r in rows])}
    result = {"method": method, "test_ids_sha256": test["selected_ids_sha256"], "groups": {}}
    for name, selected in groups.items():
        result["groups"][name] = {key: compare(rows, "audio", original_labels, guesses, selected)
                                  for key, guesses in output_labels.items()}
        for key, pair in result["groups"][name].items():
            print(name, key, pair["reference"]["correct"], pair["candidate"]["correct"],
                  pair["candidate"]["macro_f1"], pair["corrected"], pair["regressed"], flush=True)
    save_json(output, result)
    save_json(cache / "audio-review-gate-test-predictions.json",
              {"ids": [r["id"] for r in rows], "labels": output_labels})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("fit", "evaluate"))
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=4, user_api="blas"):
        if args.step == "fit":
            fit(args.cache)
        else:
            evaluate(args.cache, args.output or args.cache / "audio-review-gate-results.json")


if __name__ == "__main__":
    main()
