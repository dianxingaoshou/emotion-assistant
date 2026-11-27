"""Select the official CH-SIMS v1 test cohort and fetch just its MP4 archive members."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
from pathlib import Path
import re
import struct
import tempfile
import time
import urllib.request
import zipfile
import zlib


ROOT = Path(__file__).resolve().parents[1]
LABEL_URL = "https://drive.usercontent.google.com/download?id=1R0WUJui1utRBi-gtJK-2MxGiZrwKcDmv&export=download&confirm=t"
VIDEO_URL = "https://drive.usercontent.google.com/download?id=1PW6a-4zWMocrBGQERpRexfj0D2ydwq9u&export=download&confirm=t"
SAMPLE_SEED = "chsims-conflict-v1"
LABEL_SHA256 = "c7cdb97d9270b8cd975fd01cbf9b79e3526162be73634a50e0c622acdf16ae15"
CACHE = ROOT / ".runtime" / "chsims"


def write_atomic(destination, data):
    """Publish a complete file even with another preparer running in this cache."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    stream = tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False)
    temp = Path(stream.name)
    try:
        with stream:
            stream.write(data)
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)


def cohort(labels, limit=48):
    if hashlib.sha256(labels.read_bytes()).hexdigest() != LABEL_SHA256:
        raise ValueError("label.csv differs from the pinned official (headerless) CH-SIMS v1 labels")
    with labels.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.reader(stream) if row and row[-1] == "test"]
    ids = [f"{row[0]}/{row[1]}" for row in rows]
    if len(set(ids)) != 457 or len(rows) != 457 or not 1 <= limit <= len(rows):
        raise ValueError("Expected the original 457 distinct CH-SIMS v1 test clips")
    if not all(re.fullmatch(r"video_[0-9]{4}/[0-9]{4}", key) for key in ids):
        raise ValueError("Unexpected official video ID")
    ranked = sorted(zip(ids, rows), key=lambda pair: hashlib.sha256(
        f"{SAMPLE_SEED}|{pair[0]}".encode()).digest())
    return dict(ranked[:limit])


def read_range(left, right):
    for attempt in range(4):
        try:
            # Vary the request URL on retries: some shared caches keep a partial
            # response for a different Range of this same multi-gigabyte file.
            url = VIDEO_URL if not attempt else f"{VIDEO_URL}&slice={left}-{right}&retry={attempt}"
            request = urllib.request.Request(url, headers={"Range": f"bytes={left}-{right}"})
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 206 or not response.headers.get("Content-Range", "").startswith(
                        f"bytes {left}-{right}/"):
                    raise ValueError("Official host did not return the requested byte range")
                data = response.read(right - left + 2)
                if len(data) != right - left + 1:
                    raise ValueError("Incomplete official archive range")
                return data
        except (OSError, ValueError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


class ArchiveTail(io.BytesIO):
    """Provide zipfile the trailing central directory with logical archive offsets."""
    def __init__(self, size):
        self.offset = max(0, size - 2**20)
        super().__init__(read_range(self.offset, size - 1))

    def seek(self, pos, whence=0):
        super().seek(pos - self.offset if whence == 0 else pos, whence)
        return self.tell()

    def tell(self):
        return super().tell() + self.offset


def fetch_selected(data, limit):
    data.mkdir(parents=True, exist_ok=True)
    labels = data / "label.csv"
    if not labels.exists():
        with urllib.request.urlopen(LABEL_URL, timeout=60) as response:
            write_atomic(labels, response.read(2**20))
    selected = cohort(labels, limit)
    fetch_members(data, selected)


def fetch_members(data, selected, *, workers=5):
    """Fetch the precise official MP4 members requested by a predeclared split."""
    if not selected or any(not re.fullmatch(r"video_[0-9]{4}/[0-9]{4}", key) for key in selected):
        raise ValueError("Expected official video/clip IDs")
    request = urllib.request.Request(VIDEO_URL, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        match = re.fullmatch(r"bytes 0-0/([0-9]+)", response.headers.get("Content-Range", ""))
        if response.status != 206 or not match:
            raise ValueError("Expected archive size in the official host's Content-Range")
        size = int(match.group(1))
    with zipfile.ZipFile(ArchiveTail(size)) as archive:
        videos = {"/".join(Path(item.filename).with_suffix("").parts[-2:]): item
                  for item in archive.infolist() if item.filename.lower().endswith(".mp4")}
    if not set(selected) <= videos.keys():
        raise ValueError("Missing selected clips in official archive")

    def fetch(key):
        item = videos[key]
        destination = data / "Raw" / f"{key}.mp4"
        if destination.is_file() and zlib.crc32(destination.read_bytes()) == item.CRC:
            return key
        start = item.header_offset
        blob = read_range(start, min(size - 1, start + item.compress_size + 4095))
        if blob[:4] != b"PK\x03\x04":
            raise ValueError(f"Invalid ZIP header for {key}")
        name_len, extra_len = struct.unpack_from("<HH", blob, 26)
        offset = 30 + name_len + extra_len
        compressed = blob[offset:offset + item.compress_size]
        if item.compress_type == zipfile.ZIP_DEFLATED:
            raw = zlib.decompress(compressed, -15)
        elif item.compress_type == zipfile.ZIP_STORED:
            raw = compressed
        else:
            raise ValueError("Unsupported official ZIP compression")
        if len(raw) != item.file_size or zlib.crc32(raw) != item.CRC:
            raise ValueError(f"Official ZIP member integrity failure: {key}")
        # Independent split jobs may finish the same already-verified clip.
        write_atomic(destination, raw)
        return key

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, key in enumerate(pool.map(fetch, selected), 1):
            print(f"{i}/{len(selected)} {key}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=CACHE)
    parser.add_argument("--limit", type=int, default=48)
    args = parser.parse_args()
    fetch_selected(args.data, args.limit)
