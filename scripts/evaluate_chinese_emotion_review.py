"""Paired CPED check of one optional Qwen call vs. the desktop classifier, using only utterance text."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import inspect
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emotion_assistant import chinese_emotion_review as review
from emotion_assistant import chinese_emotions as desktop
from scripts.calibrate_chinese_affect import MODEL_DIGEST, digest, verify_local_model
from scripts.evaluate_chinese_emotions import CLASSES, CPED_COMMIT, load_source, metrics, read_split


SALT = "paired-cped-qwen-v1|"
COHORT_DIGEST = "a308fe3b5360c569a6b3cff64d856fcd577764ad11ccabdb265341b003f99294"


def select_cohort(cache):
    translation = desktop._bundled_model()["translation"]
    seen = set()
    for name in ("train", "valid"):
        texts, _, _ = read_split(load_source(cache, name + "_split.csv", False))
        seen.update(desktop.normalized_text(text, translation) for text in texts)
    texts, gold, _ = read_split(load_source(cache, "test_split.csv", False))
    choices = defaultdict(list)
    # Keep the first occurrence of each novel normalized test string, regardless of reference label.
    for text, target in zip(texts, gold):
        normalized = desktop.normalized_text(text, translation)
        if normalized in seen:
            continue
        seen.add(normalized)
        ident = digest((SALT + normalized).encode())
        choices[CLASSES[target]].append((ident, text, int(target)))
    cohort = []
    for label in CLASSES:
        ordered = sorted(choices[label])
        if len(ordered) < 10:
            raise ValueError(f"Fewer than ten novel lines for {label}")
        cohort.extend(ordered[:10])
    return sorted(cohort)


def get_ratings(cohort, cache, workers, offline):
    if not offline:
        verify_local_model()
    fingerprint = digest((review.PROMPT + json.dumps(review.SCHEMA, sort_keys=True)
                          + inspect.getsource(review.review_chinese_emotion)
                          + inspect.getsource(review.validated_result)).encode())
    identity = {"model": MODEL_DIGEST, "prompt_and_rater": fingerprint, "cped": CPED_COMMIT}
    ratings = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item["identity"] != identity:
                raise ValueError("Cached rater provenance differs; use another cache file")
            ratings[item["id"]] = item["result"]
    pending = [r for r in cohort if r[0] not in ratings]
    if pending and offline:
        raise ValueError("Missing local inference results")

    def query(row):
        result = review.review_chinese_emotion(row[1])
        if result["status"] == "service_error":
            raise RuntimeError("Local service unavailable; retry to resume from completed responses")
        return row[0], {k: result[k] for k in ("status", "label")}

    cache.parent.mkdir(exist_ok=True, parents=True)
    if pending:
        with cache.open("a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=workers) as pool:
            for n, (identifier, result) in enumerate(pool.map(query, pending), 1):
                out.write(json.dumps({"id": identifier, "identity": identity, "result": result}) + "\n")
                out.flush()
                ratings[identifier] = result
                if n % 25 == 0:
                    print(f"{n}/{len(pending)} responses obtained", flush=True)
    return ratings, fingerprint


def evaluate(cohort, ratings, fingerprint, group_digest):
    model = desktop._bundled_model()
    forced, gated, qwen, statuses = [], [], [], []
    for identifier, text, _ in cohort:
        _, scores = desktop.score_text(text, model)
        label = None if scores is None else desktop.select_label(scores, model["threshold"])
        forced.append(-1 if scores is None else int(np.argmax(scores)))
        gated.append(-1 if label is None else label)
        result = ratings[identifier]
        statuses.append(result["status"])
        qwen.append(CLASSES.index(result["label"]) if result["status"] == "ok" else -1)
    gold = np.asarray([row[2] for row in cohort])
    gated, forced, qwen = (np.asarray(a) for a in (gated, forced, qwen))
    q_hit, g_hit, f_hit = (v == gold for v in (qwen, gated, forced))
    def paired(other_hit):
        only_qwen = int(np.sum(q_hit & ~other_hit))
        only_other = int(np.sum(~q_hit & other_hit))
        discordant = only_qwen + only_other
        return {"only_qwen_correct": only_qwen, "only_desktop_correct": only_other,
                "both_correct": int(np.sum(q_hit & other_hit)),
                "exploratory_mcnemar_exact_two_sided_p": float(binomtest(only_qwen, discordant).pvalue) if discordant else 1.0}

    return {"source": "https://github.com/scutcyr/CPED", "data_commit": CPED_COMMIT,
            "task": "One dominant expression in a Chinese utterance. Gold labels are used for stratification and scoring only.",
            "sampling": "10 per CPED class, lowest SHA-256(salt + normalized text) within each gold class; "
                        "exclude any training/development text and repeat normalized test strings before sampling. No length or correctness filter.",
            "salt": SALT, "n": len(gold), "cohort_sha256": group_digest, "cohort_sha256_ids": [x[0] for x in cohort],
            "qwen_digest": MODEL_DIGEST, "rater_version": review.VERSION, "rater_fingerprint": fingerprint,
            "classifier_sha256": desktop.ASSET_SHA256, "qwen_status": dict(Counter(statuses)),
            "methods": {"desktop_gated": metrics(gold, gated), "desktop_top1_on_eligible": metrics(gold, forced),
                        "qwen_quote_review": metrics(gold, qwen)},
            "qwen_vs_gated": paired(g_hit), "qwen_vs_forced": paired(f_hit),
            "limitations": "Class-balanced exploratory check of public scripted dialogue; 10 per class, pretraining overlap unknown. "
                           "Research-only 13-class Qwen rater. The desktop uses its offline CPED classifier for this task and offers Qwen for semantic and dimensional review."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / ".runtime/cped")
    parser.add_argument("--cache", type=Path, default=ROOT / ".runtime/cped/qwen-cped-review.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/cped-qwen-paired.json")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    cohort = select_cohort(args.data)
    fingerprint = digest("\n".join(row[0] for row in cohort).encode())
    print("Cohort SHA-256", fingerprint, flush=True)
    if args.prepare_only:
        sys.exit(0)
    if fingerprint != COHORT_DIGEST:
        raise ValueError("Evaluation cohort differs from frozen selection")
    ratings, rater_id = get_ratings(cohort, args.cache, args.workers, args.offline)
    report = evaluate(cohort, ratings, rater_id, fingerprint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, result in report["methods"].items():
        print(name, {k: v for k, v in result.items() if k != "per_class"})
