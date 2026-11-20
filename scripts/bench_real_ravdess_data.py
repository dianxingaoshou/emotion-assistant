"""Fetch a fixed RAVDESS subset from the authors' versioned Zenodo record.

Only the teaching/research (CC BY-NC-SA 4.0) speech subset is downloaded.
Each member's uncompressed size and ZIP CRC are verified before caching.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import io
import json
from pathlib import Path
import struct
import time
import zipfile
import zlib

import requests

ROOT = Path(__file__).resolve().parents[1]
RECORD = "https://zenodo.org/records/1188976/files/"
CACHE = ROOT / ".runtime" / "ravdess"
LABEL = {"01": "neutral", "03": "positive", "05": "negative", "08": "unmapped"}


def specimens(actor):
    return [f"01-01-{emotion}-01-{statement:02d}-01-{actor:02d}"
            for emotion in LABEL for statement in (1, 2)]


def split_for(actor):
    # Both sexes in all splits, eight independent actors per split.
    return ("fit", "policy", "test")[((actor - 1) // 2) % 3]


def fetch(session, url, interval):
    for attempt in range(6):
        try:
            with session.get(url, headers={"Range": "bytes=" + interval}, stream=True,
                             timeout=(15, 90)) as response:
                if response.status_code == 429:
                    time.sleep(min(90, int(response.headers.get("retry-after", 60))))
                    continue
                response.raise_for_status()
                bounds = response.headers.get("Content-Range", "")
                if response.status_code != 206 or not bounds.startswith("bytes "):
                    raise ValueError("server did not honor byte range")
                start, end = map(int, bounds.split(" ")[1].split("/")[0].split("-"))
                size = int(bounds.split("/")[-1])
                if interval.startswith("-"):
                    expected = (max(0, size - int(interval[1:])), size - 1)
                else:
                    expected = tuple(map(int, interval.split("-")))
                if (start, end) != expected:
                    raise ValueError(f"unexpected range: {bounds} for {interval}")
                body = response.content
                if len(body) != end - start + 1:
                    raise IOError("incomplete range")
                return body, size
        except (requests.RequestException, OSError, ValueError):
            if attempt == 5:
                raise
            time.sleep(min(22, 2 ** attempt))
    raise IOError("repeated rate limit")


def get_actor(actor, dest):
    names = specimens(actor)
    dest.mkdir(parents=True, exist_ok=True)
    url = RECORD + f"Video_Speech_Actor_{actor:02d}.zip?download=1"
    with closing(requests.Session()) as session:
        tail, size = fetch(session, url, "-65536")
        # Avoid creating a 500MB temporary file even on non-sparse filesystems:
        # zipfile supports a rebased central directory (negative concat offset).
        with zipfile.ZipFile(io.BytesIO(tail)) as central:
            entries = {Path(i.filename).stem: i for i in central.infolist()}
        result = []
        for name in names:
            entry = entries[name]
            path = dest / (name + ".mp4")
            if not path.exists() or path.stat().st_size != entry.file_size or zlib.crc32(path.read_bytes()) != entry.CRC:
                start = entry.header_offset + size - len(tail)
                end = min(size - 1, start + 30 + 65535 + entry.compress_size + 512)
                body, _ = fetch(session, url, f"{start}-{end}")
                if body[:4] != b"PK\x03\x04":
                    raise ValueError(f"{name}: invalid ZIP local header")
                n, extra = struct.unpack_from("<HH", body, 26)
                member = body[30+n+extra:30+n+extra+entry.compress_size]
                if entry.compress_type == zipfile.ZIP_DEFLATED:
                    member = zlib.decompress(member, -15)
                elif entry.compress_type != zipfile.ZIP_STORED:
                    raise ValueError("unsupported compression")
                if len(member) != entry.file_size or zlib.crc32(member) != entry.CRC:
                    raise ValueError(f"{name}: ZIP integrity check failed")
                tmp = path.with_suffix(".part")
                tmp.write_bytes(member)
                tmp.replace(path)
            result.append({"id": name, "crc32": f"{entry.CRC:08x}", "size": entry.file_size,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        print(f"Actor {actor:02d}: verified {len(result)} clips", flush=True)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    with ThreadPoolExecutor(max_workers=max(1, min(8, args.workers))) as pool:
        rows = [r for batch in pool.map(lambda a: get_actor(a, args.cache), range(1, 25)) for r in batch]
    (args.cache / "members.json").write_text(json.dumps(rows, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
