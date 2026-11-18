"""Fit and evaluate frozen observable-cue fusion on disjoint labeled groups."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotion_assistant.calibration import (Calibration, FEATURES, GROUPS, LABELS, MAPPING_VERSION,
                                          MIN_FIT, MIN_GROUP, TARGET, fingerprint, gate_features, read_json)
from emotion_assistant.fusion_policy import FusionConfig, FusionPolicy, divergence_of


def number(value, low=0, high=1):
    if (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
            or not low <= value <= high):
        raise ValueError(f"expected a finite number in [{low}, {high}]")
    return float(value)


def load_rows(path):
    """One independent interval per sample; split entire subjects/groups beforehand."""
    splits = defaultdict(list)
    seen, group_split = set(), {}
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = read_json(line)
            split, sample, group = row["split"], row["sample_id"], row["group_id"]
            if split not in ("fit", "policy", "test") or row["target"] != TARGET:
                raise ValueError("invalid split or annotation target")
            if not all(isinstance(v, str) and v.strip() for v in (sample, group)):
                raise ValueError("nonempty sample and group IDs required")
            if sample in seen or group_split.setdefault(group, split) != split:
                raise ValueError("sample repeated or group crosses splits")
            seen.add(sample)
            label = row["label"]
            if label not in ("positive", "neutral", "negative", "unmapped"):
                raise ValueError("requires an external-cue label")
            modalities = row["modalities"]
            if not isinstance(modalities, dict) or not modalities or set(modalities) - {"visual", "audio"}:
                raise ValueError("expected at least one eligible modality")
            branches = {}
            for key, source in modalities.items():
                scores = source["scores"]
                if set(scores) != set(LABELS):
                    raise ValueError("requires all four mapped probabilities")
                scores = {k: number(scores[k]) for k in LABELS}
                total = sum(scores.values())
                if abs(total - 1) > .0001:
                    raise ValueError("mapped probabilities must sum to one")
                quality, coverage = number(source["quality"]), number(source["coverage"])
                if quality * coverage < FusionConfig().min_quality:
                    raise ValueError("use outputs after the production eligibility gate")
                version = source["model_version"]
                if not isinstance(version, str) or not version.strip():
                    raise ValueError("model version required")
                branch = {"scores": {k: v / total for k, v in scores.items()}, "quality": quality,
                          "coverage": coverage, "versions": [version],
                          "age": number(source["age_seconds"], 0, math.inf)}
                if key == "audio":
                    branch["duration"] = number(source["speech_seconds"], 0, math.inf)
                    if coverage != 1 or branch["duration"] <= 0:
                        raise ValueError("audio requires full segment coverage and positive duration")
                branches[key] = branch
            splits[split].append((branches, "uncertain" if label == "unmapped" else label))
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"line {line_no}: {exc}") from exc
    if set(splits) != {"fit", "policy", "test"}:
        raise ValueError("provide three nonempty group-disjoint splits: fit, policy, test")
    return splits, dict(Counter(group_split.values()))


def fit_branch(rows, key):
    examples = [(ctx, label) for ctx, label in rows if key in ctx]
    counts = Counter(label for _, label in examples)
    if len(examples) < MIN_FIT or any(counts[label] < 2 for label in LABELS):
        return None
    versions = {ctx[key]["versions"][0] for ctx, _ in examples}
    if len(versions) != 1:
        raise ValueError(f"fit contains different {key} model versions; fit each deployment separately")
    X, y, probabilities = [], [], []
    for ctx, label in examples:
        branch = ctx[key]
        X.append(gate_features(branch, divergence_of(ctx) or 0, len(ctx) == 2,
                               ctx.get("audio", {}).get("duration", 0)))
        probabilities.append([branch["scores"][k] for k in LABELS])
        y.append(LABELS.index(label))
    X, y = np.asarray(X), np.asarray(y)
    logs = np.log(np.clip(probabilities, 1e-12, 1))

    def class_nll(log_t):
        logits = logs / np.exp(log_t)
        return float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(y)), y]))

    temperature = minimize_scalar(class_nll, bounds=(math.log(.1), math.log(10)), method="bounded")
    design = np.column_stack((np.ones(len(y)), X))
    correct = (logs.argmax(axis=1) == y).astype(float)

    def binary_loss(params):
        log_odds = design @ params
        # Ridge also bounds the intercept if every fitting sample was correct.
        loss = np.logaddexp(0, log_odds).sum() - correct @ log_odds + .5 * (params @ params)
        return loss, design.T @ (expit(log_odds) - correct) + params

    gate = minimize(binary_loss, np.zeros(design.shape[1]), method="L-BFGS-B", jac=True)
    if not temperature.success or not gate.success:
        raise ValueError(f"optimization failed for {key}")
    return {"model_version": versions.pop(), "temperature": float(np.exp(temperature.x)),
            "intercept": float(gate.x[0]), "coefficients": gate.x[1:].tolist(), "n_fit": len(y)}


def sealed(branches, groups):
    document = {"schema": 1, "target": TARGET, "mapping_version": MAPPING_VERSION,
                "features": list(FEATURES), "branches": branches, "groups": groups}
    return {**document, "version": fingerprint(document)}


def choose_threshold(items, max_policy_error):
    """Largest validation coverage meeting the target empirical error; keep confidence ties intact."""
    errors = accepted = 0
    chosen = None
    buckets = defaultdict(list)
    for conf, correct in items:
        buckets[conf].append(correct)
    for conf in sorted(buckets, reverse=True):
        group = buckets[conf]
        accepted += len(group)
        errors += len(group) - sum(group)
        if accepted >= MIN_GROUP and errors / accepted <= max_policy_error:
            chosen = conf
    return chosen


def fit_groups(branches, policy_rows, max_policy_error):
    # Temporary acceptance at any confidence permits estimating group cutoffs
    # only on the policy split. The test split is never used for this decision.
    provisional = sealed(branches, {g: {"n_policy": MIN_GROUP, "threshold": 0} for g in GROUPS})
    policy = FusionPolicy(FusionConfig(strategy="calibrated_fusion"), Calibration.from_document(provisional))
    samples, counts = defaultdict(list), Counter()
    for inputs, label in policy_rows:
        item = policy.evaluate(inputs)
        if item["calibration_status"] != "fitted":
            continue
        group = item["status"]
        counts[group] += 1
        if item["decision"] != "abstain":
            scores = item["scores"]
            top = max(scores, key=scores.get)
            samples[group].append((scores[top], top == label))
    groups = {g: {"n_policy": n, "threshold": choose_threshold(samples[g], max_policy_error),
                  "target_empirical_error": max_policy_error}
              for g, n in counts.items() if n >= MIN_GROUP}
    if not groups:
        raise ValueError(f"need at least {MIN_GROUP} policy rows in a supported combination")
    return sealed(branches, groups)


def metrics(records):
    n = len(records)
    if not n:
        return {"n": 0}
    log_loss, brier = 0.0, 0.0
    bins, guesses, actual = [[] for _ in range(10)], [], []
    selective = []
    for item, label in records:
        scores = item["scores"]
        top = max(scores, key=scores.get)
        confidence, correct = scores[top], top == label
        bins[min(9, int(confidence * 10))].append((confidence, correct))
        guesses.append(top)
        actual.append(label)
        log_loss -= math.log(max(1e-12, scores[label]))
        brier += sum((p - (k == label)) ** 2 for k, p in scores.items())
        if item["decision"] != "abstain":
            selective.append(correct)
    f1 = []
    for k in LABELS:
        tp = sum(p == t == k for p, t in zip(guesses, actual))
        total = guesses.count(k) + actual.count(k)
        f1.append(2 * tp / total if total else 0.0)
    reliability = [{"bin": i, "n": len(bucket), "confidence": sum(p for p, _ in bucket) / len(bucket),
                    "observed_accuracy": sum(correct for _, correct in bucket) / len(bucket)}
                   for i, bucket in enumerate(bins) if bucket]
    pairs = sorted((p, int(good)) for bucket in bins for p, good in bucket)
    curve = []
    for floor in (0, .4, .5, .6, .7, .8, .9, .95):
        accepted = [good for p, good in pairs if p >= floor]
        curve.append({"minimum_confidence": floor, "coverage": len(accepted) / n,
                      "error": 1 - sum(accepted) / len(accepted) if accepted else None})
    return {"n": n, "class_support": dict(Counter(actual)), "macro_f1": sum(f1) / len(f1),
            "log_loss": log_loss / n, "brier_multiclass": brier / n, "reliability_bins": reliability,
            "confidence_risk_coverage": curve, "policy_coverage": len(selective) / n,
            "policy_accepted_error": 1 - sum(selective) / len(selective) if selective else None,
            "calibration_status_counts": dict(Counter(item["calibration_status"] for item, _ in records))}


def evaluate(policy, rows):
    all_rows = [(policy.evaluate(ctx), label) for ctx, label in rows]
    return {"overall": metrics(all_rows), "by_combination": {
        g: metrics([(r, y) for r, y in all_rows if r["status"] == g]) for g in GROUPS}}


def reliability_svg(report):
    # Vector rendering keeps the offline CLI usable with the base dependencies.
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 580 365">',
           '<rect width="580" height="365" fill="white"/>',
           '<g font-family="sans-serif" fill="#20292b" font-size="13">',
           '<text x="45" y="24">Held-out observable cues: top-label reliability</text>',
           '<path d="M55 45 V300 H310" fill="none" stroke="#344"/>',
           '<path d="M55 300 L310 45" stroke="#aaa" stroke-dasharray="4,4"/>',
           '<text x="90" y="350">Mean predicted confidence</text>',
           '<text transform="translate(16 230) rotate(-90)">Observed accuracy</text>']
    for t in (0, .25, .5, .75, 1):
        svg.append(f'<text x="{55+255*t-7}" y="320">{t:g}</text>')
        svg.append(f'<text x="22" y="{304-255*t}">{t:g}</text>')
    for i, (name, color) in enumerate((("quality_v2", "#24637a"), ("calibrated_fusion", "#ae491d"))):
        series = report["policies"][name]["overall"]["reliability_bins"]
        points = [(55+255*b["confidence"], 300-255*b["observed_accuracy"]) for b in series]
        path = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        svg.append(f'<polyline points="{path}" fill="none" stroke="{color}"/>')
        svg.extend(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>' for x, y in points)
        svg.append(f'<text x="345" y="{80+25*i}" fill="{color}">{name}</text>')
    return "".join(svg) + "</g></svg>"


def run(dataset, out, max_policy_error=.25):
    max_policy_error = number(max_policy_error)
    rows, group_counts = load_rows(dataset)
    branches = {k: fit for k in ("visual", "audio") if (fit := fit_branch(rows["fit"], k))}
    if not branches:
        raise ValueError(f"need {MIN_FIT} fitting rows with at least two annotations per class in a modality")
    policies = {"quality_v2": evaluate(FusionPolicy(FusionConfig()), rows["test"])}
    for variant in ("temperature_only", "gate_only", "calibrated_fusion"):
        fits = {k: dict(v) for k, v in branches.items()}
        for fit in fits.values():
            if variant == "temperature_only":
                fit.update(intercept=0, coefficients=[0.] * len(FEATURES))
            if variant == "gate_only":
                fit["temperature"] = 1.0
        artifact = fit_groups(fits, rows["policy"], max_policy_error)
        policy = FusionPolicy(FusionConfig(strategy="calibrated_fusion"), Calibration.from_document(artifact))
        policies[variant] = evaluate(policy, rows["test"])
    report = {"target": TARGET, "splits": {k: len(v) for k, v in rows.items()},
              "independent_groups": group_counts, "policies": policies}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    outputs = {"calibration.json": json.dumps(artifact, indent=2), "evaluation.json": json.dumps(report, indent=2),
               "reliability.svg": reliability_svg(report)}
    for name, content in outputs.items():
        with NamedTemporaryFile("w", encoding="utf-8", dir=out, delete=False) as temp:
            temp.write(content)
        Path(temp.name).replace(out / name)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-policy-error", type=float, default=.25)
    args = parser.parse_args()
    try:
        report = run(args.dataset, args.out, args.max_policy_error)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")
    print(f"Evaluated {report['splits']['test']} held-out cue labels; wrote results to {args.out}")
