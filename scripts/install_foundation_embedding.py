"""Install the pinned text encoder in the portable project Ollama store."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest, sha256
import json
from pathlib import Path
import sys
import tempfile
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emotion_assistant.foundation_embeddings import TEXT_DIGEST

REGISTRY = "https://registry.ollama.ai/v2/library/qwen3-embedding/"
BLOCK_SIZE = 4 * 1024 * 1024


def checksum(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def install_blob(entry, scratch, store):
    digest = entry["digest"].removeprefix("sha256:")
    size = entry["size"]
    url = REGISTRY + "blobs/" + entry["digest"]
    target = store / "blobs" / ("sha256-" + digest)
    if target.exists():
        if target.stat().st_size != size or checksum(target) != digest:
            raise ValueError(f"Unexpected existing model blob {target.name}")
        return

    def block(i):
        left = BLOCK_SIZE * i
        right = min(left + BLOCK_SIZE, size) - 1
        path = scratch / f"{digest}-{i}"
        for attempt in range(4):
            try:
                request = Request(url, headers={"Range": f"bytes={left}-{right}"})
                with urlopen(request, timeout=60) as response:
                    expected = f"bytes {left}-{right}/{size}"
                    if response.headers.get("Content-Range") != expected and not (
                        response.status == 200 and size <= BLOCK_SIZE):
                        raise ValueError("Unexpected model archive range")
                    data = response.read(right - left + 2)
                if len(data) != right - left + 1:
                    raise ValueError("Incomplete model archive range")
                path.write_bytes(data)
                return path
            except (OSError, ValueError):
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)

    target.parent.mkdir(parents=True, exist_ok=True)
    assembled = scratch / digest
    blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    with ThreadPoolExecutor(max_workers=6) as pool, assembled.open("wb") as out:
        for path in pool.map(block, range(blocks)):
            out.write(path.read_bytes())
    if assembled.stat().st_size != size or checksum(assembled) != digest:
        raise ValueError("Downloaded model blob failed SHA256 verification")
    assembled.replace(target)


def main():
    store = ROOT / "models/ollama"
    cache = ROOT / ".runtime"
    cache.mkdir(exist_ok=True)
    with urlopen(REGISTRY + "manifests/0.6b", timeout=30) as response:
        raw = response.read(16385)
    if sha256(raw).hexdigest() != TEXT_DIGEST:
        raise ValueError("The registry manifest differs from the trained classifier")
    manifest = json.loads(raw)
    with tempfile.TemporaryDirectory(dir=cache, prefix="embedding-download-") as directory:
        scratch = Path(directory)
        for entry in (manifest["config"], *manifest["layers"]):
            install_blob(entry, scratch, store)
        target = store / "manifests/registry.ollama.ai/library/qwen3-embedding/0.6b"
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = scratch / "manifest.json"
        staged.write_bytes(raw)
        staged.replace(target)
    print("Verified Qwen3-Embedding-0.6B in the portable project model store.")


if __name__ == "__main__":
    main()
