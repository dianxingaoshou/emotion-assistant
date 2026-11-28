"""Add modality-specific, source-held-out context to the MiniCPM-o development batches."""

import argparse
import io
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import chsims_teacher_directional as directional
from scripts import chsims_teacher_minicpm_review as mini_review
from scripts import chsims_teacher_robustness as robust
from scripts import chsims_teacher_timeline as timeline
from scripts.chsims_teacher_cross_source import group_ids
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file, write_atomic
from scripts.chsims_teacher_supervision import numbers, read_full_report


DEVELOPMENT = "minicpm-context-development.npz"
SPLITS = ("train", "valid")


def cues(scored, time, case):
    """Return available text alone, or separate retained-window text, voice and face scores."""
    if case == "no_visual":
        text = np.where(time["clean"][:, :1] > 0., scored["text"], np.nan)
        return {"context": text[:, None, :]}
    if case == "synchronous_gap":
        return {"context": np.stack(directional.local_channel_scores(scored["windows"], time["windows"]), axis=1)}
    raise ValueError(f"Unsupported context condition: {case}")


def original_development(cache):
    reference, teacher, _ = directional.frozen_inputs(cache)
    state = mini_review.load(cache, reference)["audio_review"]
    protocol = {key: state[key] for key in ("guard", "refinement", "implementation", "checkpoint", "splits")}
    reports = {phase: read_full_report(cache, phase) for phase in SPLITS}
    for phase in SPLITS:
        if sha_file(cache / f"evaluation-{phase}.json") != protocol["splits"][phase]:
            raise ValueError(f"The frozen {phase} report changed")
    archive = cache / timeline.DIRECTORY / mini_review.DEVELOPMENT
    with np.load(archive, allow_pickle=False) as original:
        if json.loads(original["provenance"].item()) != protocol:
            raise ValueError("The frozen MiniCPM-o predictions must match their development inputs")
        records = {}
        for phase, report in reports.items():
            rows = report["rows"]
            expected = {"source": group_ids(rows), "target": numbers(rows, "full"),
                        "audio_target": numbers(rows, "audio")}
            records[phase] = {}
            for case in timeline.CASES:
                prefix = f"{phase}/{case}/"
                batch = {name.removeprefix(prefix): original[name] for name in original.files if name.startswith(prefix)}
                if (any(len(value) != len(rows) for value in batch.values()) or
                        any(not np.array_equal(batch[key], value) for key, value in expected.items())):
                    raise ValueError(f"MiniCPM-o and the frozen {phase} {case} reports use different row ordering")
                records[phase][case] = batch
    provenance = {"implementation": sha_file(Path(__file__)), "original_development": sha_file(archive),
                  "original_protocol": protocol}
    return reports, teacher, records, provenance


def held_context(data, specialties, indices):
    complete = data["variants"]["complete"]
    time = data["time"]
    text = timeline.predict(specialties["text"], {name: complete[name][indices] for name in timeline.TEXT_FEATURES}, None)
    short = {key: value[indices] for key, value in time["windows"].items()}
    windows = {}
    for name, features in timeline.SPECIALISTS.items():
        if name.startswith("window_"):
            windows[name] = np.stack([timeline.predict(specialties[name], {name: short[name][:, index] for name in features},
                                                        None) for index in range(short["quality"].shape[1])], axis=1)
    return {"no_visual": cues({"text": text}, {"clean": time["clean"][indices]}, "no_visual"),
            "synchronous_gap": cues({"windows": windows}, {"windows": short}, "synchronous_gap")}


def build(cache):
    reports, teacher, records, provenance = original_development(cache)
    data = {phase: timeline.read_data(cache, doc) for phase, doc in reports.items()}
    for record in data.values():
        if record["receipt"] != teacher["extractors"] or record["time"]["provenance"] != teacher["timeline"]:
            raise ValueError("Development context evidence differs from the frozen teacher")
    training_rows = reports["train"]["rows"]
    assignment = robust.case_assignment(training_rows)
    sources = {phase: records[phase]["no_visual"]["source"] for phase in SPLITS}
    combined = np.concatenate(tuple(sources.values()))
    contexts = {phase: {case: np.full((len(sources[phase]), count, len(LABELS)), np.nan)
                        for case, count in (("no_visual", 1), ("synchronous_gap", 3))} for phase in SPLITS}
    seen = {phase: np.zeros(len(sources[phase]), dtype=bool) for phase in SPLITS}
    for fold, (_, held) in enumerate(GroupKFold(4).split(combined, groups=combined), start=1):
        withheld = np.unique(combined[held])
        training = np.flatnonzero(~np.isin(sources["train"], withheld))
        specialties = timeline.fit_specialists(data["train"], training_rows, sources["train"], training, assignment)
        for phase in SPLITS:
            indices = np.flatnonzero(np.isin(sources[phase], withheld))
            if seen[phase][indices].any():
                raise ValueError("A source was assigned to multiple held-out folds")
            seen[phase][indices] = True
            for case, result in held_context(data[phase], specialties, indices).items():
                contexts[phase][case][indices] = result["context"]
        print(f"Scored text and retained windows without held sources {fold}/4", flush=True)
    if any(not values.all() for values in seen.values()):
        raise ValueError("The held-out folds did not cover every development clip")

    payload = {f"{phase}/ids": np.array([row["id"] for row in reports[phase]["rows"]]) for phase in SPLITS}
    payload.update({f"{phase}/{case}": values for phase, entries in contexts.items() for case, values in entries.items()})
    buffer = io.BytesIO()
    np.savez_compressed(buffer, provenance=json.dumps(provenance), **payload)
    write_atomic(cache / timeline.DIRECTORY / DEVELOPMENT, buffer.getvalue())


def load_development(cache):
    reports, _, records, provenance = original_development(cache)
    with np.load(cache / timeline.DIRECTORY / DEVELOPMENT, allow_pickle=False) as archive:
        if json.loads(archive["provenance"].item()) != provenance:
            raise ValueError("Rebuild the context evidence for the current frozen development inputs")
        for phase in SPLITS:
            ids = [row["id"] for row in reports[phase]["rows"]]
            if archive[f"{phase}/ids"].tolist() != ids:
                raise ValueError(f"The {phase} context evidence uses a different clip order")
            for case, count in (("no_visual", 1), ("synchronous_gap", 3)):
                context = archive[f"{phase}/{case}"]
                if context.shape != (len(ids), count, len(LABELS)):
                    raise ValueError(f"Invalid {phase} {case} context shape")
                records[phase][case]["context"] = context
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    args = parser.parse_args()
    with threadpool_limits(limits=5, user_api="blas"):
        build(args.cache)
        for phase, cases in load_development(args.cache).items():
            for case, record in cases.items():
                print(f"{phase} {case}: {len(record['context'])} clips, available cues "
                      f"{np.isfinite(record['context'][:, :, 0]).sum(0).tolist()}", flush=True)


if __name__ == "__main__":
    main()
