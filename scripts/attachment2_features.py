"""Read official aligned_50 features and place word gaps on their shared input axis."""

from dataclasses import dataclass
import hashlib
import math
import pickle

import numpy as np


VOCAB_SHA256 = "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"
CHANNELS = "TAV"
POSITIONS = ("start", "middle", "end")


class ArrayReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in {("numpy", "ndarray"), ("numpy", "dtype")}:
            return getattr(np, name)
        if (
            module in {"numpy.core.multiarray", "numpy._core.multiarray"}
            and name == "_reconstruct"
        ):
            from numpy._core.multiarray import _reconstruct

            return _reconstruct
        raise pickle.UnpicklingError(
            f"Unexpected global in attachment: {module}.{name}"
        )


@dataclass
class Split:
    ids: list
    sources: np.ndarray
    words: list
    slots: list
    tracks: dict
    observed: dict
    classes: np.ndarray
    intensities: np.ndarray


def word_coordinates(ids, attention, vocab):
    """Unite ## continuations and apostrophe contractions, preserving aligned BERT slots."""
    words, starts = [], []
    join_next = False
    for j in np.flatnonzero(attention):
        token = vocab[ids[j]]
        if token in ("[CLS]", "[PAD]", "[SEP]"):
            continue
        if token == "[UNK]":
            starts.append(int(j))
            words.append(token)
            join_next = False
        elif token.startswith("##") and words:
            if words[-1] != "[UNK]":
                words[-1] += token[2:]
        elif token == "'" and words:
            join_next = words[-1] != "[UNK]"
        elif any(char.isalnum() for char in token):
            if join_next:
                words[-1] += "'" + token
            else:
                starts.append(int(j))
                words.append(token)
            join_next = False
        else:
            join_next = False
    # Interior punctuation belongs to the preceding word, leading punctuation to the first.
    content = [
        int(j)
        for j in np.flatnonzero(attention)
        if vocab[ids[j]] not in ("[CLS]", "[PAD]", "[SEP]")
    ]
    if words:
        starts[0] = content[0]
        starts.append(content[-1] + 1)
    return words, starts


def load_splits(aligned, vocab_path):
    vocab_bytes = vocab_path.read_bytes()
    if hashlib.sha256(vocab_bytes).hexdigest() != VOCAB_SHA256:
        raise ValueError("Expected original bert-base-uncased vocab.txt")
    vocab = vocab_bytes.decode("utf8").splitlines()
    with aligned.open("rb") as handle:
        raw = ArrayReader(handle).load()
    result = {}
    for stage, count in (("train", 3395), ("valid", 728), ("test", 727)):
        rows = raw.pop(stage)
        ids = list(rows["id"])
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError(f"Wrong number or duplicate {stage} clips")
        tokens = rows["text_bert"]
        if tokens.shape != (count, 3, 50):
            raise ValueError("Expected aligned BERT coordinates")
        coordinates = [word_coordinates(item[0], item[1], vocab) for item in tokens]
        words, slots = map(list, zip(*coordinates))
        tracks, observed = {}, {}
        for key, field, dims in (("A", "audio", 74), ("V", "vision", 35)):
            series = rows[field].astype(np.float32)
            if series.shape != (count, 50, dims):
                raise ValueError("Expected official aligned audio/vision dimensions")
            usable = np.isfinite(series).all(axis=2) & (series != 0).any(axis=2)
            usable &= (tokens[:, 1, :] == 1) & ~np.isin(tokens[:, 0, :], [0, 101, 102])
            series[~usable] = 0
            tracks[key], observed[key] = series, usable
        classes = np.asarray(rows["classification_labels"])
        intensities = np.asarray(rows["regression_labels"])
        if (
            not np.isfinite(intensities).all()
            or np.any(np.abs(intensities) > 3)
            or not np.array_equal(classes, np.sign(intensities) + 1)
        ):
            raise ValueError("Inconsistent official sentiment targets")
        sources = np.asarray([key.rsplit("$_$", 1)[0] for key in ids])
        if any("$_$" not in key for key in ids):
            raise ValueError("Unexpected source/clip identifier")
        result[stage] = Split(
            ids,
            sources,
            words,
            slots,
            tracks,
            observed,
            classes.astype(int),
            intensities,
        )
    stages = list(result)
    for i, left in enumerate(stages):
        for right in stages[i + 1 :]:
            if set(result[left].sources) & set(result[right].sources):
                raise ValueError(f"Source overlap between {left} and {right}")
    return result


@dataclass(frozen=True)
class Gap:
    name: str
    channels: str = ""
    fraction: float = 0.0
    position: str = "start"
    staggered: bool = False


def conditions():
    cases = [Gap("complete")]
    for channels in ("T", "A", "V", "TA", "TV", "AV", "TAV"):
        for fraction in (0.25, 0.50):
            for position in POSITIONS:
                name = f"{channels}_{int(100 * fraction)}_{position}"
                cases.append(Gap(name, channels, fraction, position))
    for fraction in (0.25, 0.50):
        cases.append(
            Gap(f"TAV_{int(100 * fraction)}_staggered", "TAV", fraction, staggered=True)
        )
    for channels in ("T", "AV", "TAV"):
        cases.append(Gap(f"{channels}_100", channels, 1.0))
    return cases


def bounds(length, fraction, position):
    count = min(length, math.ceil(fraction * length))
    start = {"start": 0, "middle": (length - count) // 2, "end": length - count}[
        position
    ]
    return start, start + count


def lexical_terms(words, start, end):
    """No bigram spans a removed block or an unknown token."""
    result = []
    runs = (words,) if start == end else (words[:start], words[end:])
    for run in runs:
        previous = None
        for word in run:
            if word == "[UNK]":
                previous = None
                continue
            result.append(word)
            if previous is not None:
                result.append(previous + " " + word)
            previous = word
    return result


def masked_inputs(split, gap):
    docs, text_count, text_removed = [], 0, 0
    keep = {k: mask.copy() for k, mask in split.observed.items()}
    for i, (words, slots) in enumerate(zip(split.words, split.slots, strict=True)):
        text_bounds = (0, 0)
        for j, channel in enumerate(CHANNELS):
            if channel not in gap.channels:
                continue
            position = POSITIONS[j] if gap.staggered else gap.position
            start, end = bounds(len(words), gap.fraction, position)
            if channel == "T":
                text_bounds = (start, end)
            elif start < end:
                keep[channel][i, slots[start] : slots[end]] = False
            if not slots and gap.fraction == 1.0 and channel in keep:
                keep[channel][i] = False
        docs.append(lexical_terms(words, *text_bounds))
        text_count += sum(w != "[UNK]" for w in words)
        text_removed += sum(w != "[UNK]" for w in words[slice(*text_bounds)])
    pooled, available = {}, {}
    ratios = {"T": text_removed / max(1, text_count)}
    for k, series in split.tracks.items():
        seen = keep[k]
        count = seen.sum(axis=1)
        den = np.maximum(count, 1)[:, None]
        mean = np.einsum("ntd,nt->nd", series, seen) / den
        squares = np.einsum("ntd,ntd,nt->nd", series, series, seen) / den
        std = np.sqrt(np.maximum(0, squares - mean**2))
        pooled[k] = np.column_stack((mean, std))
        available[k] = count > 0
        ratios[k] = 1 - count.sum() / max(1, split.observed[k].sum())
    return docs, pooled, available, ratios
