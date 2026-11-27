"""Recount the published paired outcomes without downloading anyone's video."""
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
bench = importlib.import_module("benchmark_chsims_conflict")


def test_zero_coverage_and_explicit_abstention():
    rows = [{"gold": "neutral", "before": "neutral", "after": None, "id": "a/1"},
            {"gold": "negative", "before": None, "after": None, "id": "b/2"}]
    pair = bench.paired(rows)
    assert pair["after"]["abstain"] == 2
    assert pair["after"]["accepted_accuracy"] is None
    assert pair["before_only_correct"] == 1
    assert pair["common_before"]["n"] == 0
    assert bench.vote([("unknown", 10), ("positive", 1), ("negative", 1)]) is None


def test_text_only_ablation_preserves_the_original_gate_without_sound_or_face():
    window = {"text": "测试句", "evidence": {"conflict": True, "missing": [], "baseline_label": "neutral",
              "cues": {"text": {"label": "neutral"}, "audio": {"label": "negative"},
                       "visual": {"label": "negative"}}}}
    text, selected = bench.masked(window, "qwen_text_only")
    assert text == window["text"] and selected["conflict"]
    assert set(selected["cues"]) == {"text"}
    text, hidden = bench.masked(window, "hide_text")
    assert text == "" and not hidden["conflict"]
    assert window["evidence"]["conflict"] and "audio" in window["evidence"]["cues"]


def test_published_clip_counts_and_same_cohort():
    report = json.loads((ROOT / "docs/results/chsims-conflict.json").read_text(encoding="utf8"))
    originals = {r["id"]: r for r in report["clips"]["observed"]}
    assert len(originals) == report["sample_n"] == 48
    assert report["test_population"] == 457
    for condition in bench.CONDITIONS:
        rows = report["clips"][condition]
        assert len(rows) == len(originals) and {r["id"] for r in rows} == set(originals)
        with_gold = [{**r, "gold": originals[r["id"]]["gold"]} for r in rows]
        for field in ("before", "after"):
            assert bench.metrics(with_gold, field) == report["conditions"][condition][field]
        if condition == "hide_text":
            assert all(not (set(r["statuses"]) - {"no_request"}) for r in rows)
