from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

from scripts.benchmark_asr_attribution import SourceBootstrap, align, analyze, reference_groups, value
from emotion_assistant.english_intensity import _bundled


def segment(text, start=0, end=1):
    return {"text": text, "start": start, "end": end}


def test_edit_alignment_preserves_insertions_and_substitutions_when_restoring_deletions():
    segments = [segment("I really love", 0, 1), segment("this new place today", 1, 4)]
    official = "I absolutely really love this old place"
    path = align(official, segments)
    counts = Counter(p.operation for p in path)
    assert counts == {"match": 5, "deletion": 1, "insertion": 1, "substitution": 1}
    assert reference_groups(path, 2) == ["i absolutely really love", "this old place"]
    model = _bundled()
    out = analyze({"id": "example", "source": "example", "segments": segments,
                   "official_text": official, "target": 0}, model)
    assert out["asr_restore_deletions"] == value("I absolutely really love this new place today", model)
    assert out["reference_without_deletions"] == value("I really love this old place", model)
    assert out["asr_duration"] == pytest.approx(
        (value(segments[0]["text"], model) + 3 * value(segments[1]["text"], model)) / 4)


def test_reference_timing_proxy_handles_terminal_and_tied_missing_words():
    parts = [segment("love"), segment("movies")]
    assert reference_groups(align("We love these movies today", parts), 2) == ["we love these", "movies today"]
    no_speech = align("very moving", [])
    assert reference_groups(no_speech, 0) == []
    assert [p.operation for p in no_speech] == ["deletion", "deletion"]


def test_unchanged_selected_policy_has_zero_paired_deltas():
    rows = [{"source": "a", "target": 0, "before": 1.0},
            {"source": "b", "target": 2, "before": 1.5}]
    delta = SourceBootstrap(rows).compare("before", "before")
    assert delta == {"mae_delta": 0, "pearson_delta": 0, "mae_delta_95pct": [0, 0], "pearson_delta_95pct": [0, 0]}


def test_audit_distinguishes_missing_and_mixed_language_tags(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from benchmark_asr_intensity_upgrade import has_non_english_tag

    assert not has_non_english_tag(["en", "unreported", None])
    assert has_non_english_tag(["unreported", "ja"])


def test_published_end_to_end_sha_matches_git_checkout_bytes():
    results = Path(__file__).resolve().parents[1] / "docs/results"
    audit = json.loads((results / "asr-intensity-upgrade.json").read_text("utf-8"))["production_gate_audit"]["holdout"]
    evidence = (results / audit["end_to_end_evidence"]).read_bytes()
    assert b"\r" not in evidence
    assert hashlib.sha256(evidence).hexdigest() == audit["end_to_end_sha256"]
