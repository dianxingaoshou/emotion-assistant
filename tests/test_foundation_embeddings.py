import numpy as np
import pytest

from emotion_assistant import foundation_embeddings as enc
from emotion_assistant import file_tracks


def test_fixed_local_model_identity_and_normalization(monkeypatch):
    requests = []
    def fake_service(method, endpoint, payload=None):
        requests.append((endpoint, payload))
        if method == "GET":
            return {"models": [{"name": enc.TEXT_MODEL, "digest": enc.TEXT_DIGEST}]}
        return {"model": enc.TEXT_MODEL, "embeddings": [[1.] * enc.TEXT_DIM for _ in payload["input"]]}
    monkeypatch.setattr(enc, "_service", fake_service)
    vec = enc.embed_text("好" * (enc.TEXT_LIMIT + 20))
    assert vec.shape == (enc.TEXT_DIM,) and abs(np.linalg.norm(vec) - 1) < 1e-5
    assert requests[1][1]["input"] == [enc.TEXT_PREFIX + "好" * enc.TEXT_LIMIT]
    assert requests[1][1]["truncate"] is False
    monkeypatch.setattr(enc, "_service", lambda *args: {"models": [{"name": enc.TEXT_MODEL, "digest": "other"}]})
    assert enc.embed_text("同名但权重改变") is None
    assert enc.embed_text("  ") is None


def test_unavailable_and_bad_dimensions_fall_back(monkeypatch):
    def offline(*args):
        raise ConnectionError("offline")
    monkeypatch.setattr(enc, "_service", offline)
    assert enc.embed_text("消息") is None
    monkeypatch.setattr(enc, "_service", lambda method, *args:
        {"models": [{"name": enc.TEXT_MODEL, "digest": enc.TEXT_DIGEST}]} if method == "GET"
        else {"model": enc.TEXT_MODEL, "embeddings": [[4.0]]})
    assert enc.embed_text("消息") is None


def test_file_audio_averages_by_duration_and_stops_at_30_seconds(monkeypatch):
    class Hub:
        model_dir = object()
        @staticmethod
        def audio_representation(samples, *args):
            if samples == -1:
                raise IndexError("Empty per-utterance model output")
            return np.full(enc.AUDIO_DIM, samples, dtype=np.float32) if samples else None
    monkeypatch.setattr(file_tracks, "media_info", lambda *a, **k: (
        {"audio": file_tracks.Track(0, 12., 9999.)}, 12.))
    def read(path, track, origin, model_dir, check, windows):
        assert windows == [(0., 30.)]
        yield -1, 0., 0.25, {}
        yield 1, 0., 1., {}
        yield 0, 1., 5., {}
        yield 3, 5., 8., {}
    monkeypatch.setattr(file_tracks, "read_speech", read)
    assert np.all(enc.clip_audio_embedding("mock.mp4", model_hub=Hub()) == 2.5)
    with pytest.raises(InterruptedError):
        enc.clip_audio_embedding("mock.mp4", model_hub=Hub(), cancelled=lambda: True)
