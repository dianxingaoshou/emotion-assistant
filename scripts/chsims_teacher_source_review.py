"""Resolve neutral whole-track versus negative retained-window evidence using development sources."""

import argparse
import io
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, softmax
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import chsims_teacher_directional as directional
from scripts import chsims_teacher_directional_pair as pair
from scripts import chsims_teacher_timeline as timeline
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_teacher_cross_source import metrics
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file, write_atomic


MODEL = "source-review.json"
DEVELOPMENT = "source-review-development.npz"
CONDITION = "synchronous_gap"
NEGATIVE, NEUTRAL, POSITIVE = directional.NEGATIVE, directional.NEUTRAL, directional.POSITIVE
TIMELINE_CUES = ("previous", "reference", "time", "desktop", "global_text", "global_audio",
                 "window_text", "window_audio", "window_visual")


def conflict_evidence(batch):
    # The preceding teacher's feature layout is frozen and checked with its implementation digest.
    design = batch["design"]
    width = 2 * len(TIMELINE_CUES)
    if design.shape[1] != width + len(TIMELINE_CUES[3:]) + 7:
        raise ValueError("Expected the frozen synchronous-gap feature layout")
    ratios = dict(zip(TIMELINE_CUES, design[:, :width].reshape(-1, len(TIMELINE_CUES), 2).transpose(1, 0, 2),
                      strict=True))
    available = dict(zip(TIMELINE_CUES[3:], design[:, width:width + len(TIMELINE_CUES[3:])].T.astype(bool),
                         strict=True))

    def cue(name):
        log_neg, log_pos = ratios[name].T
        probability = softmax(np.column_stack((log_neg, np.zeros(len(log_neg)), log_pos)), axis=1)
        return np.where(available[name][:, None], probability, np.nan)

    reference, local, desktop, audio_window, visual_window = batch["support"].transpose(1, 0, 2)
    global_audio, text_window = cue("global_audio"), cue("window_text")
    present = np.isfinite(global_audio[:, 0]) & np.isfinite(local[:, 0])
    conflict = batch["active"] & present & (global_audio.argmax(1) == NEUTRAL) & (local.argmax(1) == NEGATIVE)
    pool = np.stack((local, text_window, audio_window, visual_window, reference, desktop, global_audio), axis=1)
    margin = np.nan_to_num(pool[:, :, NEGATIVE] - pool[:, :, NEUTRAL])
    local_ratio = local[:, NEGATIVE] / np.maximum(local[:, NEGATIVE] + local[:, NEUTRAL], 1e-9)
    reference_ratio = reference[:, NEGATIVE] / np.maximum(reference[:, NEGATIVE] + reference[:, NEUTRAL], 1e-9)
    return {"conflict": conflict, "support": pool, "features": margin, "reference": reference,
            "desktop": desktop, "local_ratio": local_ratio, "reference_ratio": reference_ratio}


def fit_calibration(data, truth, sources, members, penalty):
    eligible = members[data["conflict"][members] & (truth[members] != POSITIVE)]
    y = truth[eligible] == NEGATIVE
    if len(y) < 12 or min(y.sum(), (~y).sum()) < 4:
        return None
    values = data["features"][eligible]
    _, inverse, count = np.unique(sources[eligible], return_inverse=True, return_counts=True)
    weights = count[inverse] ** -.5
    weights /= weights.sum()

    def objective(coef):
        logits = coef[0] + values @ coef[1:]
        error = (expit(logits) - y) * weights
        loss = ((np.logaddexp(0., logits) - y * logits) * weights).sum()
        return loss + penalty * (coef[1:] @ coef[1:]), np.r_[error.sum(), values.T @ error + 2 * penalty * coef[1:]]

    solution = minimize(objective, np.zeros(values.shape[1] + 1), jac=True, method="L-BFGS-B",
                        bounds=[(None, None), *[(0., None)] * values.shape[1]])
    if not solution.success:
        raise RuntimeError("Could not calibrate the conflict subset")
    return {"coefficient": solution.x.tolist(), "training_conflicts_for_fit": len(y)}


def confidence(data, method, calibrated=None):
    if method == "time":
        return data["local_ratio"]
    if method == "time_reference":
        return .6 * data["local_ratio"] + .4 * data["reference_ratio"]
    parameters = np.asarray(calibrated["coefficient"])
    return expit(parameters[0] + data["features"] @ parameters[1:])


def review(data, previous, posterior, policy):
    current = previous.argmax(1)
    anchor_support = np.isfinite(data["reference"][:, 0]) & (data["reference"].argmax(1) == NEGATIVE)
    if policy["anchor"] == "reference_or_desktop":
        desktop = data["desktop"]
        anchor_support |= np.isfinite(desktop[:, 0]) & (desktop.argmax(1) == NEGATIVE)
    elif policy["anchor"] == "none":
        anchor_support[:] = True
    eligible = data["conflict"] & ((current != NEUTRAL) | anchor_support)
    votes = directional.support_counts(data["support"][:, 1:4], np.full(len(current), NEGATIVE))
    eligible &= (current != NEUTRAL) | (votes >= policy["window_support"])
    return pair.review_pair(previous, posterior, data["support"], eligible,
                            policy["recover_at"], policy["neutral_at"], required_votes=1)


def predict(batch, previous, settings):
    data = conflict_evidence(batch)
    chosen = settings["policy"]
    if chosen is None:
        return previous, data["conflict"]
    posterior = confidence(data, chosen["method"], settings["calibration"])
    return review(data, previous, posterior, chosen), data["conflict"]


def development(cache, timeline_checkpoint, original, provenance):
    archive = cache / timeline.DIRECTORY / DEVELOPMENT
    if archive.is_file():
        with np.load(archive, allow_pickle=False) as saved:
            if json.loads(saved["provenance"].item()) == provenance:
                return {phase: {name.removeprefix(phase + "_"): saved[name] for name in saved.files
                                if name.startswith(phase + "_")} for phase in ("train", "valid")}
    batches, truth, sources = directional.development_data(cache, timeline_checkpoint, original)
    record = {phase: {**batches[phase][CONDITION], "truth": truth[phase], "sources": sources[phase]}
              for phase in ("train", "valid")}
    encoded = io.BytesIO()
    np.savez_compressed(encoded, provenance=json.dumps(provenance),
                        **{f"{phase}_{name}": value for phase, row in record.items() for name, value in row.items()})
    write_atomic(archive, encoded.getvalue())
    return record


def conflict_counts(batch, features):
    keep = features["conflict"]
    return {"n": int(keep.sum()), "sources": len(set(batch["sources"][keep])),
            "truth": dict(zip(LABELS, np.bincount(batch["truth"][keep], minlength=3).tolist(), strict=True))}


def select(train, valid, pair_policy):
    data, validation = conflict_evidence(train), conflict_evidence(valid)
    previous = pair.predict(valid, pair_policy)
    targets, validation_sources = valid["truth"], valid["sources"]
    before = metrics(targets, previous)
    old_hits = np.diag(np.asarray(before["confusion"])[:, :len(LABELS)])
    best = {"policy": None, "calibration": None, "previous_validation": before, "validation": before,
            "changes": 0, "training_conflict": conflict_counts(train, data),
            "validation_conflict": conflict_counts(valid, validation)}
    base_correct = np.isfinite(previous[:, 0]) & (previous.argmax(1) == targets)
    _, group, group_size = np.unique(validation_sources, return_inverse=True, return_counts=True)
    sources = np.concatenate((train["sources"], validation_sources))
    score_options = [(name, None, confidence(validation, name), None) for name in ("time", "time_reference")]
    for penalty in (.015, .08):
        held_scores = np.full(len(targets), np.nan)
        for _, held in GroupKFold(4).split(sources, groups=sources):
            withheld = set(sources[held])
            train_members = np.flatnonzero(~np.isin(train["sources"], list(withheld)))
            val_members = np.isin(validation_sources, list(withheld))
            fitted = fit_calibration(data, train["truth"], train["sources"], train_members, penalty)
            if fitted is not None:
                held_scores[val_members] = confidence(validation, "calibrated", fitted)[val_members]
        final = fit_calibration(data, train["truth"], train["sources"], np.arange(len(train["truth"])), penalty)
        if final is not None and np.isfinite(held_scores[validation["conflict"]]).all():
            score_options.append(("calibrated", penalty, held_scores, final))

    best_rank = (0., 0., 0)
    for method, penalty, posterior, calibration in score_options:
        for anchor in ("none", "reference", "reference_or_desktop"):
            for support in (1, 2):
                for recover in (.50, .55, .60, .65, .70, .75, .80, .85, .90, .95):
                    for neutral in (0., .35, .44, .49):
                        policy = {"method": method, "penalty": penalty, "anchor": anchor, "window_support": support,
                                  "recover_at": recover, "neutral_at": neutral}
                        predicted = review(validation, previous, posterior, policy)
                        measured = metrics(targets, predicted)
                        recall = np.diag(np.asarray(measured["confusion"])[:, :len(LABELS)])
                        if (recall[NEGATIVE] <= old_hits[NEGATIVE] or np.any(recall < old_hits) or
                                measured["macro_f1"] < before["macro_f1"]):
                            continue
                        labels = predicted.argmax(1)
                        good = np.isfinite(predicted[:, 0]) & (labels == targets)
                        gains = np.bincount(group, weights=good.astype(int) - base_correct) / group_size
                        changes = int(np.count_nonzero(labels != previous.argmax(1)))
                        quality = (.4 * (measured["accuracy"] - before["accuracy"]) +
                                   .3 * (measured["macro_f1"] - before["macro_f1"]) + .3 * gains.mean())
                        rank = (quality, float(gains.mean()), -changes)
                        if rank > best_rank:
                            best_rank = rank
                            best.update(policy=policy, calibration=calibration, validation=measured, changes=changes,
                                        source_accuracy_delta=float(gains.mean()))
    return best


def fit(cache):
    timeline_path, timeline_checkpoint, original = directional.frozen_inputs(cache)
    preceding = pair.load(cache, timeline_path)
    provenance = {"timeline": sha_file(timeline_path), "pair": sha_file(cache / timeline.DIRECTORY / pair.MODEL),
                  "splits": timeline_checkpoint["splits"]}
    for split, expected in provenance["splits"].items():
        if sha_file(cache / f"evaluation-{split}.json") != expected:
            raise ValueError(f"The frozen {split} labels changed")
    records = development(cache, timeline_checkpoint, original, provenance)
    chosen = select(records["train"], records["valid"], preceding["policies"][CONDITION])
    print("Training and validation conflicts", chosen["training_conflict"], chosen["validation_conflict"], flush=True)
    print("Selected policy", chosen["policy"], "correct", chosen["previous_validation"]["correct"], "->",
          chosen["validation"]["correct"], "Macro-F1", chosen["validation"]["macro_f1"], flush=True)
    save_json(cache / timeline.DIRECTORY / MODEL,
              {"kind": "source_review", "implementation": sha_file(Path(__file__)),
               "splits": provenance["splits"], "previous_model": provenance["pair"], "review": chosen})


def load(cache, timeline_path):
    path = cache / timeline.DIRECTORY / MODEL
    result = json.loads(path.read_text("utf8"))
    previous = pair.load(cache, timeline_path)
    if (result["implementation"] != sha_file(Path(__file__)) or
            result["previous_model"] != sha_file(cache / timeline.DIRECTORY / pair.MODEL)):
        raise ValueError("Fit the source review with the frozen pair model and current implementation")
    return {**result, "preceding": previous}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    args = parser.parse_args()
    with threadpool_limits(limits=5, user_api="blas"):
        fit(args.cache)


if __name__ == "__main__":
    main()
