"""Recreate the audio specialist's predictions without training on the held source."""

import argparse
import io
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import chsims_teacher_context_evidence as context
from scripts import chsims_teacher_robustness as robust
from scripts import chsims_teacher_timeline as timeline
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file, write_atomic
from scripts.chsims_teacher_supervision import numbers


DEVELOPMENT = "source-calibration-audio-development.npz"


def inputs(cache):
    reports, teacher, records, provenance = context.original_development(cache)
    provenance = {**provenance, "audio_implementation": sha_file(Path(__file__)),
                  "teacher": sha_file(cache / timeline.DIRECTORY / "timeline-fusion.json")}
    return reports, teacher, records, provenance


def build(cache):
    reports, teacher, records, provenance = inputs(cache)
    variants = {}
    for split, report in reports.items():
        variants[split], _, receipt = robust.inputs(cache, report)
        if receipt != teacher["extractors"]:
            raise ValueError(f"The {split} audio inputs differ from the frozen teacher")
    train_rows = reports["train"]["rows"]
    assignment = robust.case_assignment(train_rows)
    target = numbers(train_rows, "audio")
    sources = {split: cases["no_visual"]["source"] for split, cases in records.items()}
    combined = np.concatenate(tuple(sources.values()))
    scores = {split: np.full((len(rows), len(LABELS)), np.nan) for split, rows in sources.items()}
    seen = {split: np.zeros(len(rows), dtype=bool) for split, rows in sources.items()}
    for fold, (_, held) in enumerate(GroupKFold(4).split(combined, groups=combined), start=1):
        withheld = np.unique(combined[held])
        training = np.flatnonzero(~np.isin(sources["train"], withheld))
        head = robust.fit_head(variants["train"], timeline.AUDIO_FEATURES, robust.CHANNEL["audio"], target,
                               sources["train"], training, assignment,
                               {"C": .75, "temperature": 1., "neutral_shift": 0.})
        for split, group in sources.items():
            indices = np.flatnonzero(np.isin(group, withheld))
            if seen[split][indices].any():
                raise ValueError("Audio predictions must come from exactly one held source fold")
            blocks = {name: variants[split]["complete"][name][indices] for name in timeline.AUDIO_FEATURES}
            scores[split][indices] = timeline.predict(head, blocks, None)
            seen[split][indices] = True
        print(f"Scored acoustic evidence in held source fold {fold}/4", flush=True)
    if any(not observed.all() for observed in seen.values()):
        raise ValueError("The held source folds missed some development clips")
    values = {f"{split}/audio": array for split, array in scores.items()}
    values.update({f"{split}/ids": np.array([row["id"] for row in report["rows"]])
                   for split, report in reports.items()})
    result = io.BytesIO()
    np.savez_compressed(result, provenance=json.dumps(provenance), **values)
    write_atomic(cache / timeline.DIRECTORY / DEVELOPMENT, result.getvalue())


def load(cache):
    reports, _, _, provenance = inputs(cache)
    values = {}
    with np.load(cache / timeline.DIRECTORY / DEVELOPMENT, allow_pickle=False) as archive:
        if json.loads(archive["provenance"].item()) != provenance:
            raise ValueError("Rebuild audio development predictions for the frozen inputs")
        for split, report in reports.items():
            ids = [row["id"] for row in report["rows"]]
            audio = archive[f"{split}/audio"]
            if archive[f"{split}/ids"].tolist() != ids or audio.shape != (len(ids), len(LABELS)):
                raise ValueError(f"Audio predictions do not match {split} rows")
            values[split] = audio
    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE / "full")
    with threadpool_limits(limits=5, user_api="blas"):
        build(parser.parse_args().cache)
