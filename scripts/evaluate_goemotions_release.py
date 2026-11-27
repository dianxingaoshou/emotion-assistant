"""Fresh, unstratified GoEmotions comparison of the shipped English classifiers and opt-in Qwen."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emotion_assistant import english_polarity, research_text_review, semantic_review
from scripts.evaluate_goemotions import (
    ABSTAIN, CLASSES, COMMIT, SHA256, group_split, load_files, metrics, ollama_metadata,
    paired, select_sample, sha256,
)
from scripts.evaluate_goemotions_linear import TRAIN_SHA256, load_train
from scripts.evaluate_fine_emotions import DEV_SHA256, load_development

SEED = "goemotions-release-240-v1"
SIZE = 240
PINNED_MODEL = "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41"
PINNED_REVIEW = "9653812ca5de83d11e4b5a11f62f8662550390724105b14b2ff6f30a274d01e0"
PINNED_IDS = "eeff649a39636599b118dba44eb2d60ef15806ad500d872945e29fb78acc1643"
METHODS = ("legacy", "in_domain_gated", "in_domain_top1", "qwen")


def normalized(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def cohort(directory, download):
    files = load_files(directory, download)
    excluded = set()
    for split in (load_train(directory, download), load_development(directory, download)):
        excluded.update(normalized(line.split("\t")[0]) for line in split.splitlines())
    counts = Counter(normalized(line.split("\t")[0]) for line in files["test.tsv"].splitlines())
    old, _ = select_sample(files, 80)
    old_ids = {r[1] for r in old}
    old_text = {normalized(r[3]) for r in old}
    by_class, split_counts = group_split(files, "test.tsv")
    pool = []
    for rows in by_class.values():
        for _, identifier, truth, text in rows:
            key = normalized(text)
            if key and key not in excluded and key not in old_text and identifier not in old_ids and counts[key] == 1:
                rank = sha256(f"{SEED}|{identifier}".encode())
                pool.append((rank, identifier, truth, text))
    selected = sorted(pool)[:SIZE]
    if len(selected) != SIZE:
        raise ValueError("Not enough eligible comments")
    fingerprint = sha256("\n".join(r[1] for r in selected).encode())
    selection = {"official_test_rows": split_counts["official_count"],
                 "single_polarity_rows": sum(split_counts["eligible_by_class"].values()),
                 "previous_cohort_ids_excluded": len(old_ids), "remaining_pool": len(pool),
                 "pool_by_class": dict(Counter(r[2] for r in pool)), "selected": SIZE,
                 "selected_by_class": dict(Counter(r[2] for r in selected)),
                 "seed": SEED, "ordered_ids_sha256": fingerprint}
    return selected, selection


def qwen_decisions(selected, cache, identity):
    model = ollama_metadata()
    if model["manifest_digest"] != PINNED_MODEL:
        raise ValueError("The local Qwen manifest differs from the frozen model")
    identity = dict(identity, ollama=model["ollama"])
    labels = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if (record["identity"] != identity or record["qwen"] not in (*CLASSES, ABSTAIN)
                    or (record["id"] in labels and record["qwen"] != labels[record["id"]])):
                raise ValueError("Invalid or incompatible cache")
            labels[record["id"]] = record["qwen"]
    selected_ids = {row[1] for row in selected}
    if labels.keys() - selected_ids:
        raise ValueError("Unexpected cached ID")

    def predict(row):
        result = semantic_review.review_text(row[3])
        if result["status"] in {"unavailable", "service_error", "model_missing"}:
            raise RuntimeError("Local service unavailable. Re-run to resume.")
        label = result.get("content_polarity") if result["status"] == "ok" else None
        return row[1], label if label in CLASSES else ABSTAIN

    pending = [row for row in selected if row[1] not in labels]
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=3) as pool:
        for identifier, label in pool.map(predict, pending):
            labels[identifier] = label
            out.write(json.dumps({"id": identifier, "qwen": label, "identity": identity}) + "\n")
            out.flush()
            if len(labels) % 30 == 0:
                print(f"{len(labels)}/{SIZE} reviews", flush=True)
    return labels, model["ollama"]


def evaluate(selected, labels):
    reference = research_text_review.bundled_model()
    if reference is None:
        raise ValueError("Missing in-domain model")
    rows = []
    for _, identifier, truth, text in selected:
        legacy = english_polarity.predict_english_polarity(text)
        rows.append({"id": identifier, "truth": truth, "legacy": legacy["label"] if legacy else ABSTAIN,
                     "in_domain_gated": research_text_review.predict_with_model(text, reference) or ABSTAIN,
                     "in_domain_top1": research_text_review.predict_with_model(text, reference, gate=False),
                     "qwen": labels[identifier]})
    return rows


def summarize(rows, method):
    summary = metrics(rows, method)
    confusion = summary["confusion_true_by_predicted"]
    for name, item in summary["per_class"].items():
        predicted = sum(row[name] for row in confusion.values())
        item["precision"] = item["correct"] / predicted if predicted else 0.0
        item["recall"] = item["correct"] / item["support"] if item["support"] else 0.0
    for metric in ("precision", "recall"):
        summary["macro_" + metric] = sum(item[metric] for item in summary["per_class"].values()) / len(CLASSES)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / ".runtime/goemotions" / COMMIT)
    p.add_argument("--download", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--cache", type=Path, default=ROOT / ".runtime/goemotions/release-240-v1.jsonl")
    p.add_argument("--replay", type=Path, help="Read published Qwen decisions; use current offline models and official gold")
    p.add_argument("--output", type=Path, default=ROOT / "reports/goemotions-release.json")
    args = p.parse_args()
    if args.output.suffix.lower() != ".json" or args.cache.suffix.lower() != ".jsonl":
        p.error("Use separate JSON output and JSONL cache files")
    sources = [args.data_dir / file for file in (*SHA256, "dev.tsv", "train.tsv")]
    if args.replay:
        sources.append(args.replay)
    if args.output.resolve() in {path.resolve() for path in sources}:
        p.error("Choose report output separate from inputs")
    selected, selection = cohort(args.data_dir, args.download)
    print(selection, flush=True)
    if args.prepare_only:
        return
    if selection["ordered_ids_sha256"] != PINNED_IDS:
        raise ValueError("Fingerprint differs from the pre-inference cohort")
    source_hash = sha256(Path(semantic_review.__file__).read_bytes().replace(b"\r\n", b"\n"))
    if source_hash != PINNED_REVIEW:
        raise ValueError("Reviewer differs from frozen code")
    identity = {"ordered_ids_sha256": PINNED_IDS, "manifest": PINNED_MODEL, "review_source_sha256": source_hash}
    if args.replay:
        saved = json.loads(args.replay.read_text(encoding="utf8"))
        saved_ids_gold = [(r["id"], r["truth"]) for r in saved["rows"]]
        if saved["identity"] != identity or saved_ids_gold != [(r[1], r[2]) for r in selected]:
            raise ValueError("Published model or cohort does not match")
        qwen = {row["id"]: row["qwen"] for row in saved["rows"]}
        if any(label not in (*CLASSES, ABSTAIN) for label in qwen.values()):
            raise ValueError("Unrecognized published label")
        version = saved["ollama"]
    else:
        qwen, version = qwen_decisions(selected, args.cache, identity)
    rows = evaluate(selected, qwen)
    if args.replay and rows != saved["rows"]:
        raise ValueError("Published offline predictions differ from the installed models")
    report = {"dataset": {"name": "GoEmotions", "commit": COMMIT,
                          "sha256": {**SHA256, "dev.tsv": DEV_SHA256, "train.tsv": TRAIN_SHA256}},
              "selection": selection, "identity": identity, "ollama": version,
              "sources": {"legacy_sha256": sha256(english_polarity.ASSET.read_bytes()),
                          "in_domain_sha256": research_text_review.MODEL_SHA256},
              "protocol": "Fixed lowest SHA256(seed|ID) across eligible test comments without class strata. "
                          "Exclude all prior pilot and confirmation IDs, normalized training/development texts and any "
                          "text repeated anywhere in official test; NFKC, casefold and alphanumeric normalization. "
                          "Three polarity labels only. All abstentions stay in denominator. Paired unadjusted two-sided "
                          "exact McNemar tests, primary reference legacy; domain-trained references also reported.",
              "metrics": {m: summarize(rows, m) for m in METHODS},
              "paired_qwen_vs": {m: paired([dict(row, baseline=row[m], review=row["qwen"]) for row in rows])
                                 for m in METHODS[:-1]}, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for method, scores in report["metrics"].items():
        print(method, scores["correct"], scores["incorrect"], scores["abstained"], scores["macro_f1_all"])


if __name__ == "__main__":
    main()
