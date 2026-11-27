"""Replay the held-out RAVDESS actors at the desktop's normal local-video entry point."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import correlation, mean, median
import sys
from xml.etree import ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench_real_ravdess_data import CACHE, specimens, split_for
from ravdess_rater_labels import DIGEST, NS, URL, av_ratings

LABELS = {1: "neutral", 2: "calm", 3: "happiness", 4: "sadness", 5: "anger", 6: "fear",
          7: "disgust", 8: "surprise", 9: "none_of_the_above"}
MODES = {"visual": ("VO", "02"), "audio": ("AO", "03")}


def score(rows, modality):
    scored = [row for row in rows if row[modality]["human_label"] is not None]
    decided = [row[modality] for row in scored if row[modality]["top_two"]]
    first = sum(item["human_label"] == item["top_two"][0] for item in decided)
    two = sum(item["human_label"] in item["top_two"] for item in decided)
    right = [item["margin"] for item in decided if item["human_label"] == item["top_two"][0]]
    wrong = [item["margin"] for item in decided if item["human_label"] != item["top_two"][0]]
    separation = (sum((a > b) + .5 * (a == b) for a in right for b in wrong) / (len(right) * len(wrong))
                  if right and wrong else None)
    intensity_pairs = [(row[modality]["top1_score"], row[modality]["human_intensity_mean"])
                       for row in rows if row[modality]["top1_score"] is not None]
    top_scores = [x for x, _ in intensity_pairs]
    intensities = [y for _, y in intensity_pairs]
    return {"clips": len(rows), "tied_human_vote": len(rows) - len(scored),
            "with_reference": len(scored), "with_output_and_reference": len(decided),
            "abstained_with_reference": len(scored) - len(decided),
            "top1_correct": first, "top2_contains_reference": two,
            "accuracy_all_with_reference": first / len(scored),
            "top2_recall_all_with_reference": two / len(scored),
            "median_gap_when_top1_correct": median(right) if right else None,
            "median_gap_when_top1_incorrect": median(wrong) if wrong else None,
            "gap_auc_for_top1_correct_on_outputs": separation,
            "intensity_pairs": len(intensity_pairs),
            "top_score_vs_perceived_intensity_pearson": (correlation(top_scores, intensities)
                if len(set(top_scores)) > 1 and len(set(intensities)) > 1 else None),
            "reference_counts": dict(Counter(r[modality]["human_label"] for r in scored))}


def evaluate(cache, workbook):
    from emotion_assistant.paths import MODELS, configure_environment
    configure_environment()
    import cv2
    from emotion_assistant.local_media import analyze_media
    from emotion_assistant.models import ModelHub, VERSIONS

    if sha256(workbook.read_bytes()).hexdigest() != DIGEST:
        raise ValueError("Unexpected official RAVDESS S5 workbook")
    with ZipFile(workbook) as z:
        shared = ["".join(item.itertext()) for item in ET.fromstring(z.read("xl/sharedStrings.xml"))]
        ratings = {key: av_ratings(z, shared, "sheet3", code) for key, (code, _) in MODES.items()}
        intensity = {key: av_ratings(z, shared, "sheet5", code) for key, (code, _) in MODES.items()}
    manifest = {entry["id"]: entry["sha256"] for entry in json.loads((cache / "members.json").read_text("utf-8"))}
    actors = [i for i in range(1, 25) if split_for(i) == "test"]
    rows = []
    cv2.setNumThreads(2)
    hub = ModelHub(MODELS, enable_asr=False)
    for actor in actors:
        for clip in specimens(actor):
            path = cache / (clip + ".mp4")
            if sha256(path.read_bytes()).hexdigest() != manifest[clip]:
                raise ValueError(f"Clip hash mismatch: {clip}")
            result = analyze_media(path, model_dir=MODELS, model_hub=hub, transcribe=False,
                                   on_sample=lambda *_: None, cancelled=lambda: False)
            if result["issues"]:
                raise ValueError(f"Model or decoding errors on {clip}: {result['issues']}")
            record = {"clip": clip, "media_sha256": manifest[clip]}
            for modality, (_, prefix) in MODES.items():
                _, votes = ratings[modality][prefix + clip[2:]]
                counts = Counter(LABELS[v] for v in votes)
                winner = [key for key in counts if counts[key] == max(counts.values())]
                profile = result["expression_profiles"][modality]
                _, ratings_1_to_5 = intensity[modality][prefix + clip[2:]]
                if len(ratings_1_to_5) != 10 or any(r not in range(1, 6) for r in ratings_1_to_5):
                    raise ValueError("Malformed official perceived intensity ratings")
                record[modality] = {"human_label": winner[0] if len(winner) == 1 else None,
                                    "votes": dict(counts), "status": profile["status"],
                                    "top_two": profile["top_two"], "margin": profile["margin"],
                                    "top1_score": (max(profile["scores"].values()) if profile["top_two"] else None),
                                    "human_intensity_mean": mean(ratings_1_to_5),
                                    "eligible_observations": profile["eligible_observations"]}
            rows.append(record)
        print(f"Actor {actor:02d} complete ({len(rows)} files)", flush=True)
    return {"source": "https://zenodo.org/records/1188976", "human_ratings_source": URL,
            "workbook_sha256": DIGEST, "actors": actors, "model_versions": VERSIONS,
            "method": "Local AV file path, separate audio and visual distributions. Duration × input quality among eligible observations. "
                      "Each native ranking compared to the same take's unique plurality of 10 original audio-only or video-only votes. "
                      "Check top score versus perceived intensity (ten modality-specific 1-5 ratings) for all clips with an output. "
                      "Full model vocabularies retained. Tied votes excluded per modality. "
                      "8 clips per held-out actor, normal intensity, acted neutral/happy/angry/surprised. "
                      "No fitting or threshold selection; upstream training overlap unknown.",
            "metrics": {key: score(rows, key) for key in MODES}, "clips": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--workbook", type=Path, default=ROOT / ".runtime/study-sources/ravdess-s5.xlsx")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/expression-profiles.json")
    args = parser.parse_args()
    report = evaluate(args.cache, args.workbook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"]))
