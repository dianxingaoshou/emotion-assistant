"""Frozen language and acoustic representations for supervised Chinese clip polarity."""

import json
from http.client import HTTPConnection, HTTPException

import numpy as np


TEXT_MODEL = "qwen3-embedding:0.6b"
TEXT_DIGEST = "ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d"
TEXT_DIM = 1024
AUDIO_DIM = 768
AUDIO_SECONDS = 30
TEXT_LIMIT = 1024
TEXT_PREFIX = ("Instruct: Represent the sentiment expressed by this Chinese utterance, "
               "including negation, praise, criticism and neutral statements.\nQuery: ")


def _service(method, endpoint, data=None):
    conn = HTTPConnection("127.0.0.1", 11434, timeout=45)
    try:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8") if data is not None else None
        conn.request(method, endpoint, body, {"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f"Local embedding service returned HTTP {response.status}")
        limit = 2 * 1024 * 1024
        raw = response.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("Unexpectedly large local embedding response")
        return json.loads(raw)
    finally:
        conn.close()


def embed_texts(texts):
    """Batched strict research entry. The optional desktop wrapper handles missing services."""
    if not 1 <= len(texts) <= 32 or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("Supply one to 32 nonempty utterances")
    tags = _service("GET", "/api/tags")
    if not any(m.get("name") == TEXT_MODEL and m.get("digest") == TEXT_DIGEST
               for m in tags.get("models", [])):
        raise RuntimeError("The pinned Qwen3-Embedding-0.6B model is unavailable")
    response = _service("POST", "/api/embed", {
        "model": TEXT_MODEL, "input": [TEXT_PREFIX + t.strip()[:TEXT_LIMIT] for t in texts],
        "truncate": False,
    })
    vectors = np.asarray(response["embeddings"], dtype=np.float32)
    if (response.get("model") != TEXT_MODEL or vectors.shape != (len(texts), TEXT_DIM)
            or not np.isfinite(vectors).all()):
        raise ValueError("Invalid local language embedding")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms < 1e-6):
        raise ValueError("Zero local language embedding")
    return vectors / norms


def embed_text(text):
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        return embed_texts([text])[0]
    except (OSError, HTTPException, RuntimeError, ValueError, TypeError, KeyError):
        return None


def clip_audio_embedding(path, *, model_hub, cancelled=lambda: False):
    """Pool valid VAD utterances in the first 30 seconds. No text or ASR input."""
    import av
    from .file_tracks import media_info, read_speech

    def check():
        if cancelled():
            raise InterruptedError("Foundation audio feature extraction cancelled")

    check()
    try:
        tracks, origin = media_info(path, want_video=False)
        track = tracks.get("audio")
        if track is None:
            return None
        start = (track.start - origin) if track.start is not None else 0.
        end = start + min(AUDIO_SECONDS, track.duration if track.duration is not None else AUDIO_SECONDS)
        total = np.zeros(AUDIO_DIM, dtype=np.float64)
        duration = 0.
        for samples, left, right, _ in read_speech(path, track, origin, model_hub.model_dir, check, [(start, end)]):
            check()
            try:
                feature = model_hub.audio_representation(samples, left, right)
            except InterruptedError:
                raise
            except (KeyError, IndexError, TypeError, ValueError, RuntimeError, OSError):
                # A malformed output from one utterance leaves later speech available.
                continue
            if feature is not None:
                total += (right - left) * feature
                duration += right - left
        if duration:
            return (total / duration).astype(np.float32)
    except InterruptedError:
        raise
    except (av.FFmpegError, OSError, RuntimeError, ValueError):
        return None
    return None
