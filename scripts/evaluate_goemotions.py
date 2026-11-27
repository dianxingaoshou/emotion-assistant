"""Paired, label-stratified test of desktop classifiers on published GoEmotions comments."""

import argparse
from collections import Counter
import hashlib
from http.client import HTTPConnection
import json
from math import comb
from pathlib import Path
from statistics import median
import sys
from time import monotonic
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emotion_assistant import english_polarity, semantic_review

COMMIT = "7951440944924ac61b6e2f9a2a3c715834005c40"
BASE_URL = f"https://raw.githubusercontent.com/google-research/google-research/{COMMIT}/goemotions/data"
SHA256 = {
    "test.tsv": "0587b2dd8b27b97352adbfc3fb083d46005c8946657fdc2b1ca8b1cc7f1f8be4",
    "emotions.txt": "45c3ef86782d2a4d7fedcd6d8c111aa0d0e94720689bd164fac94fefb4495a89",
    "sentiment_mapping.json": "801f6f028082141bd38541a4b14d65b33bdc55d29c7b2aaa029fefecf206f76b",
}
CLASSES = english_polarity.CLASSES
SEED = "goemotions-text-review-v1"
ABSTAIN = "abstain"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_files(directory, download):
    contents = {}
    for name, digest in SHA256.items():
        path = directory / name
        if path.is_file():
            data = path.read_bytes()
        elif download:
            with urlopen(f"{BASE_URL}/{name}", timeout=45) as response:
                data = response.read()
        else:
            raise ValueError(f"Missing {path}. Use --download to cache the pinned official files.")
        if sha256(data) != digest:
            raise ValueError(f"SHA-256 mismatch: {name}")
        if not path.is_file():
            directory.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        contents[name] = data.decode("utf-8")
    return contents


def group_split(files, split_filename):
    """Collapse the official fine labels and keep only one determinate polarity per comment."""
    emotions = files["emotions.txt"].splitlines()
    mapping = {"neutral": "neutral"}
    for group, members in json.loads(files["sentiment_mapping.json"]).items():
        mapping.update(dict.fromkeys(members, group))
    if set(emotions) != set(mapping) or set(mapping.values()) != {*CLASSES, "ambiguous"}:
        raise ValueError("Incomplete official taxonomy")

    grouped = {label: [] for label in CLASSES}
    excluded = dict.fromkeys(("missing", "ambiguous", "cross_polarity"), 0)
    ids = set()
    lines = files[split_filename].splitlines()
    for line in lines:
        text, labels, comment_id = line.split("\t")
        if not comment_id or comment_id in ids:
            raise ValueError("Missing or duplicate official comment ID")
        ids.add(comment_id)
        if not text.strip() or not labels.strip():
            excluded["missing"] += 1
            continue
        polarities = {mapping[emotions[int(index)]] for index in labels.split(",")}
        if "ambiguous" in polarities:
            excluded["ambiguous"] += 1
        elif len(polarities) > 1:
            excluded["cross_polarity"] += 1
        else:
            truth = polarities.pop()
            key = sha256(f"{SEED}|{comment_id}".encode("utf-8"))
            grouped[truth].append((key, comment_id, truth, text))
    return grouped, {
        "official_count": len(lines),
        "eligible_by_class": {label: len(rows) for label, rows in grouped.items()},
        "excluded": excluded,
    }


def select_sample(files, per_class, offset_per_class=0):
    grouped, counts = group_split(files, "test.tsv")
    stop = offset_per_class + per_class
    if offset_per_class < 0 or per_class < 1 or stop > min(map(len, grouped.values())):
        raise ValueError("Choose a nonnegative offset and a positive count within available class support")
    selected = sorted(row for rows in grouped.values() for row in sorted(rows)[offset_per_class:stop])
    selection = {
        "official_test_count": counts["official_count"],
        "eligible_by_class": counts["eligible_by_class"],
        "excluded": counts["excluded"],
        "per_class": per_class,
        "offset_per_class": offset_per_class,
        "sample_count": len(selected),
        "seed": SEED,
        "ordered_ids_sha256": sha256("\n".join(row[1] for row in selected).encode("utf-8")),
    }
    return selected, selection


def metrics(outcomes, method):
    confusion = {truth: dict.fromkeys((*CLASSES, ABSTAIN), 0) for truth in CLASSES}
    for row in outcomes:
        confusion[row["truth"]][row[method]] += 1
    sample_count = len(outcomes)
    per_class = {}
    for truth in CLASSES:
        counts = confusion[truth]
        support = sum(counts.values())
        predicted = sum(values[truth] for values in confusion.values())
        correct = counts[truth]
        per_class[truth] = {
            "support": support,
            "correct": correct,
            "incorrect": support - correct - counts[ABSTAIN],
            "abstained": counts[ABSTAIN],
            "f1": 2 * correct / (support + predicted) if support + predicted else 0.0,
        }
    correct = sum(row["correct"] for row in per_class.values())
    abstained = sum(row["abstained"] for row in per_class.values())
    classified = sample_count - abstained
    return {
        "correct": correct, "incorrect": classified - correct, "abstained": abstained,
        "coverage": classified / sample_count, "accuracy_all": correct / sample_count,
        "accuracy_classified": correct / classified if classified else None,
        "macro_f1_all": sum(row["f1"] for row in per_class.values()) / len(CLASSES),
        "per_class": per_class, "confusion_true_by_predicted": confusion,
    }


def paired(outcomes):
    counts = dict.fromkeys(("both_correct", "baseline_only", "review_only", "neither_correct"), 0)
    for row in outcomes:
        base = row["baseline"] == row["truth"]
        review = row["review"] == row["truth"]
        key = {(True, True): "both_correct", (True, False): "baseline_only",
               (False, True): "review_only", (False, False): "neither_correct"}[base, review]
        counts[key] += 1
    up, down = counts["review_only"], counts["baseline_only"]
    discordant = up + down
    tail = sum(comb(discordant, i) for i in range(min(up, down) + 1))
    counts["accuracy_delta"] = (up - down) / len(outcomes)
    counts["exact_mcnemar_two_sided_p"] = min(1.0, 2 * tail / 2**discordant)
    return counts


def ollama_metadata():
    connection = HTTPConnection("127.0.0.1", 11434, timeout=10)
    try:
        connection.request("GET", "/api/tags")
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError("Could not read model identity from local Ollama")
        models = json.load(response)["models"]
        metadata = next((row for row in models if row["name"] == semantic_review.MODEL), None)
        if metadata is None:
            raise ValueError(f"Local Ollama does not have {semantic_review.MODEL}")
        connection.request("GET", "/api/version")
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError("Could not read local Ollama version")
        return {"model": metadata["name"], "manifest_digest": metadata["digest"],
                "quantization": metadata["details"]["quantization_level"],
                "ollama": json.load(response)["version"]}
    finally:
        connection.close()


def evaluate(selected, selection, predictions_output=None):
    model = ollama_metadata()
    model["review_version"] = semantic_review.REVIEW_VERSION
    model["review_source_sha256"] = sha256(Path(semantic_review.__file__).read_bytes().replace(b"\r\n", b"\n"))
    outcomes, timings = [], []
    statuses = Counter()
    for index, (_, comment_id, truth, text) in enumerate(selected, 1):
        baseline = english_polarity.predict_english_polarity(text)
        start = monotonic()
        result = semantic_review.review_text(text)
        timings.append(monotonic() - start)
        status = result["status"]
        if status in {"unavailable", "service_error", "model_missing"}:
            raise RuntimeError(f"Local Ollama failed on item {index} ({status}); incomplete run")
        statuses[status] += 1
        label = result.get("content_polarity") if status == "ok" else None
        outcomes.append({"id": comment_id, "truth": truth, "baseline": baseline["label"] if baseline else ABSTAIN,
                         "review": label if label in CLASSES else ABSTAIN})
        if index % 10 == 0:
            print(f"Reviewed {index}/{len(selected)}", flush=True)
    if predictions_output is not None:
        predictions_output.parent.mkdir(parents=True, exist_ok=True)
        predictions_output.write_text(json.dumps(outcomes, indent=2) + "\n", encoding="utf-8")
    return {
        "dataset": {"repository": "google-research/google-research/goemotions", "commit": COMMIT,
                    "sha256": SHA256, "split": "test"},
        "task": "expressed content polarity", "selection": selection,
        "model": model, "baseline_asset_sha256": sha256(english_polarity.ASSET.read_bytes()),
        "baseline": metrics(outcomes, "baseline"), "semantic_review": metrics(outcomes, "review"),
        "paired": paired(outcomes), "review_status_counts": dict(statuses),
        "review_latency_seconds_median": round(median(timings), 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".runtime/goemotions" / COMMIT)
    parser.add_argument("--download", action="store_true", help="Fetch missing files from the pinned Google Research repository")
    parser.add_argument("--prepare-only", action="store_true", help="Show verified selection counts without model inference")
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--offset-per-class", type=int, default=0,
                        help="Skip this many hash-ranked items within each eligible class")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/semantic-review/goemotions.json")
    parser.add_argument("--predictions-output", type=Path,
                        help="Optional separate JSON of official IDs and decisions without source text")
    args = parser.parse_args()
    outputs = [args.output] if args.predictions_output is None else [args.output, args.predictions_output]
    targets = {path.resolve() for path in outputs}
    if (len(targets) != len(outputs) or any(path.suffix.lower() != ".json" for path in outputs)
            or targets & {(args.data_dir / name).resolve() for name in SHA256}):
        parser.error("Choose a separate JSON output file")
    try:
        files = load_files(args.data_dir, args.download)
        selected, selection = select_sample(files, args.per_class, args.offset_per_class)
        if args.prepare_only:
            print(json.dumps(selection, indent=2))
            return
        result = evaluate(selected, selection, args.predictions_output)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
