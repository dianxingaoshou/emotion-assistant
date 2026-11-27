"""Explain the frozen v1 errors. Official test captions enter a separate post-hoc control only."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import unicodedata

import numpy as np

from fit_chsims_foundation import (
    DATA, ROOT, checksums_for_raw, components, dataset, embeddings, file_sha, frozen_cohort,
    fusion, read_json, read_rows, write_json,
)
from chsims_samples import LABEL_SHA256
from emotion_assistant.foundation_embeddings import embed_texts


V1 = ROOT / "emotion_assistant/assets/chsims-foundation-v1.npz"
FROZEN_SHA = "f7c5ae8fe83399abe95a53099a5087c16aed6f47048eae2d713a2e7c78a23393"


def normalized(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (x != y)))
        previous = current
    return previous[-1]


def distribution(values):
    if values is None or not np.any(values):
        return None
    return {"label": fusion.LABELS[int(np.argmax(values))],
            "scores": dict(zip(fusion.LABELS, values.tolist(), strict=True))}


def mix(values, weights):
    mass = weights.sum()
    return distribution((values * weights[:, None]).sum(axis=0) / mass) if mass else None


def diagnose_group(rows, model, original, *, include_neutral):
    qwen_blocks, coverage = components(rows, model, "foundation")
    char_blocks, char_coverage = components(rows, model, "no_qwen")
    if not np.array_equal(coverage, char_coverage):
        raise ValueError("The v1 paired cohorts require observed Chinese and the same available channels")
    prior = {r["id"]: r for r in original["clips"]}
    cases = []
    by_conflict = {k: Counter() for k in ("conflict", "agree", "missing")}
    for i, row in enumerate(rows):
        weights = coverage[i] * model["foundation_weights"]
        control = coverage[i] * model["no_qwen_weights"]
        q, c = qwen_blocks[i], char_blocks[i]
        av = mix(q[1:], weights[1:])
        text = distribution(q[0])
        four = {"char_low": mix(c, control), "char_high": mix(c, weights),
                "qwen_low": mix(q, control), "qwen_high": mix(q, weights)}
        gold = row["label"]
        saved = prior[row["id"]]
        actual = saved["predictions"]["observed"]
        for k, v in (("char_low", "no_qwen"), ("char_high", "fixed_weights_no_qwen"), ("qwen_high", "foundation")):
            if four[k]["label"] != actual[v]:
                raise ValueError(f"Different v1 decision for {row['id']}")
        if not all(np.isclose(four["qwen_high"]["scores"][k], v, atol=1e-10, rtol=0)
                   for k, v in saved["observed_foundation"]["scores"].items()):
            raise ValueError("Different original class score")
        conflict = "missing" if av is None or text is None else ("conflict" if av["label"] != text["label"] else "agree")
        before, after = four["char_low"]["label"] == gold, four["qwen_high"]["label"] == gold
        outcomes = by_conflict[conflict]
        outcomes["total"] += 1
        outcomes["new_errors"] += int(before and not after)
        outcomes["corrected"] += int(not before and after)
        outcomes["unchanged_correctness"] += int(before == after)
        if (include_neutral and gold == "neutral") or (before and not after):
            cases.append({
                "id": row["id"], "gold": gold, "new_error": bool(before and not after),
                "text_av": conflict, "without_text": av, "factorial": four,
                "experts": {k: distribution(q[j]) for j, k in enumerate(fusion.EXPERTS)},
                "character_expert": distribution(c[0]),
                "qwen_normalized_weights": dict(zip(fusion.EXPERTS, (weights/weights.sum()).tolist(), strict=True)),
            })
    return {"text_av_pairing": by_conflict, "cases": cases}


def add_caption_control(groups, all_rows, model, annotations):
    cases = [case for group in groups.values() for case in group["cases"]]
    corrected = [{**all_rows[case["id"]], "text": annotations[case["id"]][2]} for case in cases]
    vectors = embed_texts([r["text"] for r in corrected])
    for row, vector in zip(corrected, vectors, strict=True):
        row["embedding"] = vector
    values, available = components(corrected, model, "foundation")
    for i, (case, row) in enumerate(zip(cases, corrected, strict=True)):
        reference = normalized(row["text"])
        original_text = all_rows[case["id"]]["text"]
        asr = normalized(original_text)
        fields = annotations[case["id"]]
        if fields[7].lower() != case["gold"]:
            raise ValueError("Official multimodal label differs from paired study")
        case["official_annotations"] = dict(zip(("overall", "text", "audio", "visual"),
                                                map(float, fields[3:7]), strict=True))
        case["caption_control"] = {
            "exact_equal": row["text"] == original_text, "normalized_equal": reference == asr,
            "character_edits": edit_distance(reference, asr),
            "reference_characters": len(reference), "qwen_text": distribution(values[i, 0]),
            "fused": mix(values[i], available[i] * model["foundation_weights"]),
        }


def diagnose(data, label_file, output):
    study = read_json(ROOT / "docs/results/chsims-foundation-fusion.json")
    if study["fit"]["model_sha256"] != FROZEN_SHA or file_sha(V1) != FROZEN_SHA:
        raise ValueError("Expected the frozen first-generation CH-SIMS classifier")
    if file_sha(label_file) != LABEL_SHA256:
        raise ValueError("Unexpected official labels file")
    with np.load(V1, allow_pickle=False) as archive:
        model = dict(archive)
    with label_file.open(encoding="utf-8-sig", newline="") as f:
        annotations = {f"{r[0]}/{r[1]}": r for r in csv.reader(f) if r[-1] == "test"}
    primary = frozen_cohort(data, "raw-cohort.json")["test"]
    audit = frozen_cohort(data, "audit-cohort.json")["audit"]
    text_sha, audio_sha = checksums_for_raw(data, [*primary, *audit])
    text = embeddings(data / "embeddings-text-asr.npz", fusion.TEXT_DIM, expected=text_sha)
    audio = embeddings(data / "embeddings-audio.npz", fusion.AUDIO_DIM, expected=audio_sha)
    audit_rows = [json.loads(line) for line in (data / "audit-rows.jsonl").read_text("utf8").splitlines()]
    groups, all_rows = {}, {}
    for name, ids, source in (("primary48", primary, read_rows(data)), ("audit48", audit, audit_rows)):
        rows = dataset(data, source, ids, text, audio, raw=True)
        all_rows.update((row["id"], row) for row in rows)
        groups[name] = diagnose_group(rows, model, study[name], include_neutral=name == "audit48")
    add_caption_control(groups, all_rows, model, annotations)
    write_json(output, {"version": "chsims-foundation-v1-error-analysis", "head_sha256": FROZEN_SHA,
                        "class_order": list(fusion.LABELS), "text_coefficients": {"low": .25, "high": 1.},
                        "source_study_sha256": file_sha(ROOT / "docs/results/chsims-foundation-fusion.json"),
                        "caption_control": "Post-prediction only. Replace ASR with official caption, re-embed locally; "
                                           "retain the original AV values, classifiers and fusion weights.",
                        **groups})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--labels", type=Path, default=ROOT / ".runtime/chsims/label.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/chsims-foundation-errors.json")
    args = parser.parse_args()
    diagnose(args.data, args.labels, args.output)
