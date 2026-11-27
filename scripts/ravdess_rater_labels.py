"""Prepare four-class reference labels from the authors' ten audiovisual ratings.

Uses the original PLOS supplement, not the model predictions or the actor's
intended expression. No third-party Excel dependency is required.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import urllib.request
import xml.etree.ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
URL = ("https://journals.plos.org/plosone/article/file?type=supplementary"
       "&id=info:doi/10.1371/journal.pone.0196391.s008")
DIGEST = "b304b75a1a1a8928be557b0fe6a248c40db193e95e4d5b02073dce8dd034fcee"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
# IDs verified against correct responses in the S5 WORKBOOK. Its landing-page
# caption reverses 7 and 8. See validate_codebook and the experiment note.
# Calm and neutral are grouped for this cue task;
# surprise and an explicit "none of the above" are outside the first 3 classes.
CUE = {1: "neutral", 2: "neutral", 3: "positive", 4: "negative",
       5: "negative", 6: "negative", 7: "negative", 8: "unmapped", 9: "unmapped"}
RATING_COLS = tuple("PQRSTUVWXY")


def adjudicate(ratings):
    if len(ratings) != 10 or not all(v in CUE for v in ratings):
        raise ValueError("Expected ten categorical S5 ratings in 1..9")
    counts = Counter(CUE[v] for v in ratings)
    top = max(counts.values())
    winners = [k for k, v in counts.items() if v == top]
    return (winners[0] if len(winners) == 1 else None), top, dict(sorted(counts.items()))


def av_ratings(z, shared, sheet, modality="AV"):
    prefix, extension = {"AV": ("01", "mp4"), "VO": ("02", "mp4"), "AO": ("03", "wav")}[modality]
    records = {}
    with z.open(f"xl/worksheets/{sheet}.xml") as stream:
        for _, row in ET.iterparse(stream, events=("end",)):
            if row.tag != NS + "row":
                continue
            cells = {}
            for cell in row.findall(NS + "c"):
                v = cell.findtext(NS + "v")
                if v is not None:
                    col = cell.attrib["r"].rstrip("0123456789")
                    cells[col] = shared[int(v)] if cell.get("t") == "s" else v
            row.clear()
            if cells.get("B") != modality or cells.get("C") != "Speech":
                continue
            name = cells["A"]
            if not re.fullmatch(rf"{prefix}-01-(?:[0-9]{{2}}-){{4}}[0-9]{{2}}\.{extension}", name) or name[:-4] in records:
                raise ValueError(f"Duplicate or invalid {modality} stimulus: {name}")
            records[name[:-4]] = (cells["D"], [int(cells[k]) for k in RATING_COLS])
    return records


def validate_codebook(category, correct):
    if category.keys() != correct.keys():
        raise ValueError("Mismatched S5 sheets")
    matched = {"Disgust": Counter(), "Surprise": Counter()}
    for stem, (intended, codes) in category.items():
        if intended in matched:
            matched[intended].update(v for v, hit in zip(codes, correct[stem][1], strict=True) if hit == 1)
    if matched != {"Disgust": {7: 1707}, "Surprise": {8: 1607}}:
        raise ValueError(f"S5 codebook check failed: {matched}")


def prepare(workbook):
    if sha256(workbook.read_bytes()).hexdigest() != DIGEST:
        raise ValueError("Unexpected S5 file SHA256; verify the source/version before use.")
    labels, agreement = {}, {}
    with ZipFile(workbook) as z:
        shared = ["".join(si.itertext()) for si in ET.fromstring(z.read("xl/sharedStrings.xml"))]
        category = av_ratings(z, shared, "sheet3")  # Speech_EmoCategory
        correct = av_ratings(z, shared, "sheet1")  # Speech_RawAccuracy
    validate_codebook(category, correct)
    for stem, (_, votes) in category.items():
        winner, top, counts = adjudicate(votes)
        if winner:
            labels[stem] = winner
        agreement[stem] = {"votes": counts, "winning_votes": top, "label": winner}
    if len(agreement) != 1440:
        raise ValueError(f"Expected 1440 paired speech stimuli, found {len(agreement)}")
    return labels, agreement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=ROOT / ".runtime/study-sources/ravdess-s5.xlsx")
    parser.add_argument("--download", action="store_true", help="download the pinned supplement if absent")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/ravdess")
    args = parser.parse_args()
    if args.download and not args.workbook.exists():
        with urllib.request.urlopen(URL, timeout=120) as resp:
            data = resp.read()
        if sha256(data).hexdigest() != DIGEST:
            raise ValueError("Downloaded S5 is not the pinned workbook")
        args.workbook.parent.mkdir(parents=True, exist_ok=True)
        args.workbook.write_bytes(data)
    labels, agreement = prepare(args.workbook)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, obj in (("rater-labels", labels), ("rater-votes", agreement)):
        (args.out / (name + ".json")).write_text(
            json.dumps(obj, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"{len(agreement)} audiovisual stimuli, {len(labels)} unique pluralities, "
          f"{len(agreement)-len(labels)} ties")


if __name__ == "__main__":
    main()
