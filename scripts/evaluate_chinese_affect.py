"""Compare frozen local Chinese VA ratings with the authors' CVAS sentence annotations."""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
from statistics import correlation, mean
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emotion_assistant.chinese_affect import AXES, VERSION, rate_chinese_expression

COMMIT = "eaf0ca0fd6ec9e48e701231eee490d58bd074c12"
URL = f"https://raw.githubusercontent.com/NYCU-NLP/Chinese-EmoBank/{COMMIT}/ChineseEmoBank/CVAS_SD/CVAS_all.csv"
DIGEST = "d7b78e2ed06fccbee57d8ec88ac29389c1a9356dfd93b7ac3878c89d89eea3d9"


def errors(rows, axis, prediction):
    if not rows:
        return {"n": 0, "mae": None, "pearson": None}
    truth = [r[axis] for r in rows]
    preds = [prediction(r) for r in rows]
    r = correlation(truth, preds) if len(set(truth)) > 1 and len(set(preds)) > 1 else None
    return {"n": len(rows), "mae": round(mean(abs(a-b) for a, b in zip(truth, preds)), 4),
            "pearson": round(r, 4) if r is not None else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / ".runtime/chinese-emobank/CVAS_all.csv")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/chinese-cvas.json")
    args = parser.parse_args()
    if args.output.resolve() == args.data.resolve() or args.output.suffix.lower() != ".json":
        parser.error("Choose a JSON output distinct from the source CSV")
    if args.download and not args.data.exists():
        with urlopen(URL, timeout=45) as stream:
            raw = stream.read()
        if hashlib.sha256(raw).hexdigest() != DIGEST:
            raise ValueError("Unexpected download digest")
        args.data.parent.mkdir(parents=True, exist_ok=True)
        args.data.write_bytes(raw)
    raw = args.data.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DIGEST:
        raise ValueError("Unexpected CVAS digest")
    rows = []
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter="\t"):
        text = row["Text"].strip()
        key = hashlib.sha256(("zh-va-v1|" + text).encode()).hexdigest()
        rows.append({"id": key, "text": text, "valence": float(row["Valence_Mean"]),
                     "arousal": float(row["Arousal_Mean"])})
    rows.sort(key=lambda row: row["id"])
    sample = rows[:96]
    reference = {axis: mean(x[axis] for x in rows[96:]) for axis in AXES}
    statuses = Counter()
    for row in sample:
        result = rate_chinese_expression(row["text"])
        statuses[result["status"]] += 1
        if result["status"] == "service_error":
            raise RuntimeError("Local Ollama unavailable; run scripts/start_text_review.ps1")
        row["result"] = result
    valid = [row for row in sample if row["result"]["status"] == "ok"]
    metrics = {}
    for axis in AXES:
        prior = reference[axis]
        metrics[axis] = {
            "qwen_accepted": errors(valid, axis, lambda row: row["result"][axis]),
            "constant_prior_same_accepted": errors(valid, axis, lambda row: prior),
            "qwen_with_prior_for_missing": errors(sample, axis, lambda row: row["result"].get(axis, prior)),
            "constant_prior_all": errors(sample, axis, lambda row: prior),
            "complement_mean": round(prior, 4),
        }
    report = {"version": VERSION, "source": URL, "source_sha256": DIGEST, "source_rows": len(rows),
              "selection": "First 96 distinct sentences sorted by SHA256(zh-va-v1|original stripped text); no task fitting.",
              "ids_sha256": hashlib.sha256("\n".join(row["id"] for row in sample).encode()).hexdigest(),
              "scale": [1, 9], "attempted": len(sample), "coverage": len(valid) / len(sample),
              "status_counts": dict(statuses), "metrics": metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
