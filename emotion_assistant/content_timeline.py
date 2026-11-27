"""Assemble expressed-content inputs on the original speech clock.

One ASR answer describes one VAD utterance. Do not spread it over visual bins
or borrow an acoustic score produced from a different speech segment.
"""

from collections import OrderedDict
import math

from .conflict_review import build_evidence


def _span(row):
    try:
        start, end = float(row["t_start"]), float(row["t_end"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return (start, end) if math.isfinite(start) and math.isfinite(end) and 0 <= start < end else None


def _overlaps(left, right):
    return left[0] < right[1] and right[0] < left[1]


def _segment_matches(left, right):
    # Older observations have no segment ID. Exact timestamps are still required.
    a, b = left.get("segment_id"), right.get("segment_id")
    return a is None or b is None or a == b


def _text_intensity(context):
    record = context.get("english_intensity")
    if not isinstance(record, dict):
        return None
    value = record.get("value")
    version = record.get("model_version")
    if type(value) not in (int, float) or not -3 <= value <= 3 or not isinstance(version, str) or not version:
        return None
    return {"value": float(value), "model_version": version, "used_modalities": ["text"]}


def _features(span, visual_rows, joint=None, utterance=None):
    start, end = span
    channels = {}
    source_ids = set()
    for name in ("audio", "visual"):
        mass, weight, quality = {}, 0.0, 0.0
        rows = (joint,) if joint is not None else (visual_rows if name == "visual" else ())
        for row in rows:
            left, right = _span(row)
            overlap = min(end, right) - max(start, left)
            if overlap <= 0:
                continue
            scores = (row.get("per_modality_output_scores") or {}).get(name)
            if not scores:
                scores = (row.get("per_modality_scores") or {}).get(name)
            if not scores:
                continue
            fraction = (row.get("per_modality_coverage") or {}).get(name, 1.0)
            seconds = overlap * min(1.0, max(0.0, fraction))
            if seconds <= 0:
                continue
            weight += seconds
            quality += seconds * (row.get("per_modality_quality") or {}).get(name, 1.0)
            for label, value in scores.items():
                mass[label] = mass.get(label, 0.0) + value * seconds
            source_ids.update(row.get("source_ids") or ())
        if weight:
            channels[name] = {
                "scores": {label: value / weight for label, value in mass.items()},
                "quality": quality / weight,
                "coverage": min(1.0, weight / (end - start)),
            }
    context = (utterance.get("semantic_context") or {}) if utterance and utterance.get("valid") else {}
    text = context.get("english_polarity")
    intensity = _text_intensity(context)
    if (text or intensity) and utterance.get("event_id"):
        source_ids.add(utterance["event_id"])
    features = {
        "t_start": start,
        "t_end": end,
        "source_ids": sorted(source_ids),
        "text": text,
        "audio": channels.get("audio"),
        "visual": channels.get("visual"),
    }
    if intensity is not None:
        features["intensity"] = intensity
    if utterance and utterance.get("valid") and not utterance.get("stale"):
        profile = context.get("expression_profile") or {}
        features["conflict_segment_id"] = utterance.get("segment_id") or utterance.get("event_id") or str(end)
        features["conflict_evidence"] = build_evidence(
            None, english_polarity=text, chinese_dimensions=profile.get("affect_dimensions")
            if profile.get("language") == "zh" else None, **channels)
    return features


def iter_content_features(fusion_intervals, utterances, audio_samples=(), *, can_align=True):
    """Return disjoint utterance windows and the remaining observed visual bins."""
    visual, acoustic = {}, {}
    for row in fusion_intervals:
        span = _span(row)
        if not span:
            continue
        if "audio" in (row.get("per_modality_scores") or {}):
            acoustic[span] = row
        elif "visual" in (row.get("per_modality_scores") or {}) or row.get("status") != "audio_only":
            visual[span] = row
    speech = {span: row for row in audio_samples if (span := _span(row))}
    for span, row in acoustic.items():
        speech.setdefault(span, row)
    texts = {}
    for row in utterances:
        span = _span(row)
        if not span or not row.get("valid") or row.get("stale"):
            continue
        if span in speech:
            if not _segment_matches(speech[span], row):
                continue
        elif any(_overlaps(other, span) for other in speech):
            continue
        else:
            speech[span] = row
        texts[span] = row
    anchors = []
    ambiguous = set()
    for span in sorted(speech):
        if anchors and span[0] < anchors[-1][1]:
            start, end = anchors.pop()
            span = start, max(end, span[1])
            ambiguous.add(span)
        anchors.append(span)
    visual_rows = list(visual.values())
    aligned_visual = visual_rows if can_align else ()
    results = []
    for span in anchors:
        if span in ambiguous:
            row = _features(span, aligned_visual)
            row["timing_issue"] = "overlapping_speech"
        else:
            row = _features(span, aligned_visual, acoustic.get(span), texts.get(span))
        results.append(row)
    for (left, right), row in sorted(visual.items()):
        for start, end in anchors:
            if end <= left:
                continue
            if start >= right:
                break
            if left < start:
                results.append(_features((left, start), [row]))
            left = max(left, end)
            if left >= right:
                break
        if left < right:
            results.append(_features((left, right), [row]))
    return sorted(results, key=lambda row: (row["t_start"], row["t_end"]))


def predict_content_interval(row):
    from .content_fusion import fuse_content

    scores = fuse_content(text=row["text"], visual=row["visual"], audio=row["audio"])
    result = {
        **scores,
        "t_start": row["t_start"],
        "t_end": row["t_end"],
        "source_ids": row["source_ids"],
        "intensity": row.get("intensity"),
    }
    if row.get("timing_issue"):
        result["timing_issue"] = row["timing_issue"]
    if "conflict_evidence" in row:
        result["conflict_evidence"] = row["conflict_evidence"]
        result["conflict_segment_id"] = row["conflict_segment_id"]
    return result


class LiveContentTimeline:
    """Ephemeral, bounded joins. Delayed ASR only revises its own utterance."""

    def __init__(self):
        self.reset()

    def reset(self, minimum_start=0.0):
        self.visual = OrderedDict()
        self.speech = OrderedDict()
        self.utterances = OrderedDict()
        self.last_end = minimum_start
        self.minimum_start = minimum_start
        self.last_display = None

    def _remember(self, records, span, value):
        if not span or span[0] < self.minimum_start or span[1] < self.last_end - 60.0:
            return False
        self.last_end = max(self.last_end, span[1])
        records[span] = value
        records.move_to_end(span)
        for cache in (self.visual, self.speech, self.utterances):
            for key in [key for key in cache if key[1] < self.last_end - 60.0]:
                del cache[key]
            while len(cache) > 256:
                cache.popitem(last=False)
        return True

    def observe_fusion(self, row):
        target = self.speech if row.get("audio_event_id") else self.visual
        if not self._remember(target, _span(row), row):
            return None
        return self._current()

    def observe_transcript(self, row):
        if not row.get("valid") or row.get("stale"):
            return None
        span = _span(row)
        if span is None:
            return None
        known = self.speech.get(span)
        if known and not _segment_matches(known, row):
            return None
        if not known and any(_overlaps(other, span) for other in self.speech):
            return None
        # Keep only derived estimates, never the transcript or quoted phrases.
        safe = {k: row.get(k) for k in ("event_id", "segment_id", "valid")}
        context = row.get("semantic_context") or {}
        safe["semantic_context"] = {
            "english_polarity": context.get("english_polarity"),
            "english_intensity": _text_intensity(context),
        }
        profile = context.get("expression_profile") or {}
        if profile.get("language") == "zh":
            safe["semantic_context"]["expression_profile"] = {
                "language": "zh", "affect_dimensions": profile.get("affect_dimensions")}
        if not self._remember(self.utterances, span, safe):
            return None
        return self._current()

    def _speech_features(self, span):
        acoustic = self.speech.get(span)
        text = self.utterances.get(span)
        if any(s != span and _overlaps(s, span) for s in self.speech):
            inputs = _features(span, self.visual.values())
            if acoustic:
                inputs["timing_issue"] = "overlapping_speech"
            return inputs
        if text and acoustic and not _segment_matches(text, acoustic):
            text = None
        joint = acoustic if acoustic and "audio" in (acoustic.get("per_modality_scores") or {}) else None
        return _features(span, self.visual.values(), joint, text)

    def historical_reviews(self, event):
        """Revise affected utterances in the selector without rolling back the live readout."""
        changed = _span(event)
        latest = _span(self.last_display) if self.last_display else None
        if changed:
            for span in self.utterances:
                if span != latest and _overlaps(changed, span):
                    yield predict_content_interval(self._speech_features(span))

    def _current(self):
        speech = self.speech.keys() | self.utterances.keys()
        latest = max(speech, key=lambda span: (span[1], span[0])) if speech else None
        if latest and latest[1] >= self.last_end - 8.0:
            inputs = self._speech_features(latest)
        elif self.visual:
            latest = max(self.visual, key=lambda span: (span[1], span[0]))
            inputs = _features(latest, [self.visual[latest]])
        else:
            return None
        displayed = predict_content_interval(inputs)
        if self.last_display == displayed:
            return None
        self.last_display = displayed
        return displayed
