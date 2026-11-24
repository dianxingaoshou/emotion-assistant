"""Score the first-pass transcript of a short English audio/video file as one utterance."""
import math

from .asr_word_intensity import bundled_char, robust_value
from .core import interval_duration
from .english_intensity import _bundled
from .english_polarity import tokens

METHOD = "asr_word_character_pool"


def english_speech(segments):
    words = []
    for segment in segments:
        text = segment.get("text", "").strip()
        if not text:
            continue
        if segment.get("asr_language") not in (None, "en"):
            return False
        if any(c.isalpha() and not c.isascii() for c in text):
            return False
        words.extend(tokens(text))
    return len(words) >= 2


def estimate_file_intensity(segments, *, source_id, duration):
    """Only score consented file transcripts. The caller retains its old summary on any abstention."""
    if not source_id or duration is None or not math.isfinite(duration) or not .2 <= duration <= 30:
        return None
    current = []
    for row in segments:
        if (row.get("session_id") != source_id or row.get("valid") is not True or row.get("stale")
                or not isinstance(row.get("text"), str) or not row["text"].strip()):
            continue
        start, end = row.get("t_start"), row.get("t_end")
        if not all(isinstance(t, (int, float)) and math.isfinite(t) for t in (start, end)) or not 0 <= start < end:
            return None
        current.append(row)
    if not english_speech(current):
        return None
    current.sort(key=lambda row: row["t_start"])
    word_model, char_model = _bundled(), bundled_char()
    if word_model is None or char_model is None:
        return None
    estimate = robust_value(" ".join(row["text"] for row in current), word_model, char_model)
    if estimate is None:
        return None
    return {"method": METHOD, "coverage_kind": "speech_segments",
            "covered_seconds": interval_duration((r["t_start"], r["t_end"]) for r in current),
            "intensity": {"value": estimate, "scale": [-3, 3], "method": METHOD, "used_modalities": ["text"],
                          "model_version": word_model["model_version"] + "+" + char_model["model_version"]}}
