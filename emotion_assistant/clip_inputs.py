"""Prepare clip-level inputs from the desktop importer's timestamped observations."""

import math

from .models import CANONICAL


def _bounds(row):
    start, end = row.get("t_start"), row.get("t_end")
    if (type(start) in (int, float) and type(end) in (int, float)
            and math.isfinite(end) and 0 <= start < end):
        return start, end
    return None


def _channel(rows, duration):
    mass = dict.fromkeys(CANONICAL, 0.)
    total = quality_seconds = covered_until = 0.
    for (start, end), row in sorted(((_bounds(r), r) for r in rows if r.get("eligible") and _bounds(r)),
                                    key=lambda item: item[0]):
        scores, quality = row.get("scores"), row.get("quality")
        if (not isinstance(scores, dict) or type(quality) not in (int, float) or not 0 < quality <= 1
                or not all(type(scores.get(k)) in (float, int) and 0 <= scores[k] <= 1 for k in CANONICAL)
                or abs(sum(scores[k] for k in CANONICAL) - 1) > 0.025):
            continue
        seconds = min(duration, end) - max(start, covered_until)
        if seconds <= 0:
            continue
        covered_until = end
        total += seconds
        quality_seconds += quality * seconds
        for name in mass:
            mass[name] += seconds * scores[name]
    if not total:
        return None
    return {"scores": {k: v / total for k, v in mass.items()},
            "quality": quality_seconds / total, "coverage": min(1., total / duration)}


def clip_inputs(result):
    """Average each available sensor on its own clock. Text always comes from eligible ASR."""
    audio = result.get("audio_samples", ())
    visual = result.get("samples", ())
    transcripts = [(span, row["text"].strip()) for row in result.get("transcripts", ())
                   if row.get("valid") and not row.get("stale") and isinstance(row.get("text"), str)
                   and row["text"].strip() and (span := _bounds(row))]
    duration = (result.get("timeline_coverage") or {}).get("duration_s")
    if type(duration) not in (float, int) or not math.isfinite(duration) or duration <= 0:
        duration = max((span[1] for rows in (audio, visual) for row in rows if (span := _bounds(row))), default=0.)
    return {"text": " ".join(s for _, s in sorted(transcripts, key=lambda t: t[0])),
            "audio": _channel(audio, duration), "visual": _channel(visual, duration)}
