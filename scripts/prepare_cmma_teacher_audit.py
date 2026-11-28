"""Fetch a fixed sample of the authors' CMMA test videos for external teacher evaluation."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gc
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import sys
from urllib.request import Request, urlopen
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_chsims_teacher import save_json
from scripts.chsims_teacher_data import CACHE, LABELS, sha_file, write_atomic


URL = "https://drive.usercontent.google.com/download?id=19NgqpYLPa3bLFm4YUjuCT2sbmQemziG-&export=download&confirm=t"
LENGTH = 12_661_638_986
SEED = "cmma-teacher-timeline-test-01"
PER_CLASS = 40


def fetch_range(start, end):
    request = Request(f"{URL}&chunk={start}", headers={"Range": f"bytes={start}-{end - 1}"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=25) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end - 1}/{LENGTH}":
                    raise ValueError("The CMMA archive does not match the expected range")
                content = response.read(end - start + 1)
            if len(content) != end - start:
                raise OSError("Incomplete CMMA range")
            return content
        except OSError:
            if attempt == 2:
                raise


class ArchiveIndex(io.RawIOBase):
    def __init__(self):
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (0, self.position, LENGTH)[whence]
        return self.position

    def read(self, count=-1):
        end = LENGTH if count < 0 else min(LENGTH, self.position + count)
        if end <= self.position:
            return b""
        content = fetch_range(self.position, end)
        self.position = end
        return content


def read_member(entry):
    start = entry.header_offset
    payload = fetch_range(start, min(LENGTH, start + entry.compress_size + 4096))
    signature, name, extra = struct.unpack_from("<I22xHH", payload)
    if signature != 0x04034B50 or payload[30:30 + name].decode("utf8") != entry.filename:
        raise ValueError(f"Unexpected CMMA member header {entry.filename}")
    offset = 30 + name + extra
    packed = payload[offset:offset + entry.compress_size]
    if len(packed) != entry.compress_size:
        packed = fetch_range(start + offset, start + offset + entry.compress_size)
    if entry.compress_type == ZIP_DEFLATED:
        data = zlib.decompress(packed, -zlib.MAX_WBITS)
    elif entry.compress_type == ZIP_STORED:
        data = packed
    else:
        raise ValueError("Unsupported CMMA member compression type")
    if len(data) != entry.file_size or zlib.crc32(data) != entry.CRC:
        raise ValueError(f"Corrupt CMMA member {entry.filename}")
    return data


def seconds(value):
    hour, minute, second = value.replace(",", ".").split(":")
    return 3600 * int(hour) + 60 * int(minute) + float(second)


def choose(clips, annotations, *, excluded=(), per_class=PER_CLASS, seed=SEED, all_conversations=False):
    used = set(excluded)
    pools = {label: [] for label in LABELS}
    for item in csv.DictReader(io.StringIO(annotations.decode("gb18030"))):
        conversation, utterance, label = (item[k] for k in ("CoversationID", "UtteranceID", "Sentiment"))
        if not re.fullmatch(r"C_\d+", conversation) or not re.fullmatch(rf"{conversation}_U_\d+", utterance):
            raise ValueError("Unexpected CMMA identifier")
        path = f"CMMA/test/video/{conversation}/{utterance}.mp4"
        duration = seconds(item["EndTime"]) - seconds(item["StartTime"])
        if conversation not in used and path in clips and label in pools and 1.8 <= duration <= 20.:
            order = hashlib.sha256(f"{seed}|{utterance}".encode("ascii")).digest()
            row = {"id": f"{conversation}/{utterance}", "member": path, "truth": {"full": label}}
            pools[label].append((order, conversation, row))
    selected = []
    if all_conversations:
        for _, conversation, row in sorted(candidate for group in pools.values() for candidate in group):
            if conversation not in used:
                selected.append(row)
                used.add(conversation)
        return sorted(selected, key=lambda row: row["id"])
    for label, candidates in pools.items():
        count = 0
        for _, conversation, row in sorted(candidates):
            if conversation in used:
                continue
            used.add(conversation)
            selected.append(row)
            count += 1
            if count == per_class:
                break
        if count != per_class:
            raise ValueError(f"Insufficient distinct CMMA conversations for {label}")
    return sorted(selected, key=lambda row: row["id"])


def download(cache, *, exclude_cohorts=(), per_class=PER_CLASS, seed=SEED, all_conversations=False):
    previous_rows = []
    for previous in exclude_cohorts:
        if previous.resolve() == (cache / "cohort.json").resolve():
            raise ValueError("Write the new cohort to a different cache")
        previous_rows.extend(json.loads(previous.read_text("utf8"))["rows"])
    excluded = {row["id"].split("/")[0] for row in previous_rows}
    with ZipFile(ArchiveIndex()) as archive:
        index = {entry.filename: entry for entry in archive.infolist()}
    annotations = read_member(index["CMMA/test/text/test.csv"])
    rows = choose(index, annotations, excluded=excluded, per_class=per_class, seed=seed,
                  all_conversations=all_conversations)
    if not rows:
        raise ValueError("No CMMA videos are eligible for the selected cohort")

    def fetch(row):
        member = index[row["member"]]
        path = cache / "raw" / f"{row['id']}.mp4"
        if not path.is_file() or path.stat().st_size != member.file_size or zlib.crc32(path.read_bytes()) != member.CRC:
            write_atomic(path, read_member(member))
        return {**row, "raw_sha256": sha_file(path)}

    with ThreadPoolExecutor(max_workers=12) as pool:
        downloaded = []
        for count, item in enumerate(pool.map(fetch, rows), 1):
            downloaded.append(item)
            if count % 20 == 0:
                print(f"CMMA test video {count}/{len(rows)}", flush=True)
    hashes = {item["raw_sha256"] for item in downloaded}
    if len(hashes) != len(rows) or hashes & {row["raw_sha256"] for row in previous_rows}:
        raise ValueError("The frozen CMMA selection contains duplicate media")
    cohort = {"source": "CMMA official test", "selection": seed,
              "labels_sha256": hashlib.sha256(annotations).hexdigest(), "rows": downloaded}
    if exclude_cohorts:
        fingerprints = sorted({sha_file(file) for file in exclude_cohorts})
        if len(fingerprints) == 1:
            cohort["excluded_cohort_sha256"] = fingerprints[0]
        else:
            cohort["excluded_cohorts_sha256"] = fingerprints
        cohort["excluded_conversations"] = len(excluded)
    if all_conversations:
        cohort["one_per_available_conversation"] = True
    save_json(cache / "cohort.json", cohort)


def prepare_encoders(cache, rows):
    import torch
    from scripts import benchmark_chsims_teacher as anchor
    from scripts import chsims_face_teacher as faces
    from scripts import prepare_chsims_teacher_gaps as gaps
    from scripts import prepare_chsims_teacher_timeline as timeline
    from scripts.chsims_minicpmo_audio import GAP_AUDIO, MiniCPMAudio, extract
    from scripts.chsims_teacher_data import make_preparer

    def clear():
        gc.collect()
        torch.cuda.empty_cache()

    keys = [row["id"] for row in rows]
    prepare = make_preparer(cache, device="cuda")
    for row in rows:
        source = cache / "raw" / f"{row['id']}.mp4"
        if sha_file(source) != row["raw_sha256"]:
            raise ValueError(f"The CMMA original changed for {row['id']}")
        prepare(source, cache / "media" / f"{row['id']}.npz")
    del prepare
    clear()
    gaps.prepare(cache, keys, "cuda")
    timeline.prepare(cache, keys, "cuda")
    clear()
    teacher = MiniCPMAudio(CACHE / "minicpm-o-4_5")
    extract(teacher, cache, keys, 8)
    extract(teacher, cache, keys, 8, input_folder=gaps.PREPARED, output_folder=GAP_AUDIO)
    del teacher
    clear()
    anchor.score_teacher(cache, keys, 4, representations=True)
    clear()
    faces.score(cache, keys, 4)
    clear()
    gaps.encode(cache, keys, 4)


def prepare_desktop(cache, cohort):
    from scripts import benchmark_chsims_teacher as anchor
    from scripts import prepare_chsims_teacher_gaps as gaps
    from scripts import prepare_chsims_teacher_timeline as timeline

    rows = cohort["rows"]
    keys = [row["id"] for row in rows]
    anchor.score_baseline(cache, keys)
    gaps.score_desktop(cache, keys)
    timeline.embed(cache, keys)
    models = {family: json.loads((cache / f"{family}-model.json").read_text("utf8"))
              for family in ("baseline", "teacher")}
    evidence = []
    for row in rows:
        predictions = {}
        for family, identity in models.items():
            source = json.loads((cache / family / f"{row['id']}.json").read_text("utf8"))
            if source["model"] != identity or source["media_sha256"] != sha_file(cache / "media" / f"{row['id']}.npz"):
                raise ValueError(f"Inconsistent CMMA prediction for {row['id']}")
            predictions[family] = source["predictions"]
        evidence.append({"id": row["id"], "truth": row["truth"], **predictions})
    save_json(cache / "report.json", {"models": models, "rows": evidence, "meta_sha256": cohort["labels_sha256"],
                                     "selected_ids_sha256": hashlib.sha256("\n".join(keys).encode("utf8")).hexdigest()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("download", "gpu", "desktop"), nargs="?", default="download")
    parser.add_argument("--cache", type=Path, default=CACHE / "cmma-timeline")
    parser.add_argument("--exclude-cohort", type=Path, action="append", default=[],
                        help="exclude all conversations from this earlier manifest (repeatable)")
    parser.add_argument("--per-class", type=int, default=PER_CLASS)
    parser.add_argument("--all-conversations", action="store_true", help="choose one qualifying video per conversation")
    parser.add_argument("--seed", default=SEED)
    args = parser.parse_args()
    if args.step == "download":
        if args.per_class < 1:
            parser.error("--per-class must be positive")
        download(args.cache, exclude_cohorts=args.exclude_cohort, per_class=args.per_class, seed=args.seed,
                 all_conversations=args.all_conversations)
    else:
        cohort = json.loads((args.cache / "cohort.json").read_text("utf8"))
        if args.step == "gpu":
            prepare_encoders(args.cache, cohort["rows"])
        else:
            prepare_desktop(args.cache, cohort)


if __name__ == "__main__":
    main()
