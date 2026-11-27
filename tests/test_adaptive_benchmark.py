import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_chsims_adaptive import (
    MEASURES, SCENARIOS, VARIANTS, all_metrics, bootstrap_measures, file_sha,
    numbers, release_checks, source_bootstrap_weights,
)


def test_release_decision_requires_accuracy_gain_and_no_f1_or_positive_recall_loss():
    matched = dict(accuracy=.6, macro_f1=.5, positive_recall=.4)

    def passes(**overrides):
        return release_checks({"no_qwen": matched, "foundation": {**matched, **overrides}})["metrics_passed"]

    assert passes(accuracy=.61)
    assert not passes(macro_f1=.51, positive_recall=.45)
    assert not passes(accuracy=.61, macro_f1=.49)
    assert not passes(accuracy=.61, positive_recall=.39)


def test_cluster_bootstrap_keeps_siblings_and_counts_abstentions():
    ids = ("s1/a", "s1/b", "s2/a", "s3/a")
    weights = source_bootstrap_weights(ids)
    np.testing.assert_array_equal(weights[:, 0], weights[:, 1])
    gold = np.array([0, 1, 1, 2])
    pred = np.array([0, -1, 2, 2])
    one = np.ones((1, 4))
    expected = all_metrics(gold, pred)
    assert expected["neutral_hits"] == 0 and expected["neutral_support"] == 2
    assert expected["accuracy"] == .5 and expected["abstain"] == 1
    assert bootstrap_measures(gold, pred, one)[0] == pytest.approx([expected[k] for k in MEASURES])


def test_paired_artifact_cohorts_predictions_and_freeze():
    from emotion_assistant import adaptive_clip_fusion, foundation_fusion

    previous = ROOT / "docs/results/chsims-foundation-fusion.json"
    study = json.loads((ROOT / "docs/results/chsims-adaptive-fusion.json").read_text(encoding="utf-8"))
    old = json.loads(previous.read_text(encoding="utf-8"))
    assert file_sha(previous) == study["v1_results_sha256"]
    assert file_sha(adaptive_clip_fusion.ASSET) == study["fit"]["model_sha256"]
    assert file_sha(foundation_fusion.ASSET) == study["fit"]["head_sha256"]
    all_ids = set()
    for cohort in ("primary48", "audit48"):
        rows = study[cohort]["clips"]
        ids = [r["id"] for r in rows]
        old_rows = {r["id"]: r for r in old[cohort]["clips"]}
        assert len(ids) == len(set(ids)) == 48 and set(ids) == set(old_rows)
        assert not all_ids.intersection(ids)
        all_ids.update(ids)
        gold = numbers([r["gold"] for r in rows])
        for mask in SCENARIOS:
            condition = study[cohort]["conditions"][mask]
            for key in VARIANTS:
                labels = [r["predictions"][mask][key] for r in rows]
                if key.startswith("v1_"):
                    assert labels == [old_rows[k]["predictions"][mask][key[3:]] for k in ids]
                assert condition["metrics"][key] == all_metrics(gold, numbers(labels))
            compared = condition["source_paired_deltas"]["no_qwen_to_foundation"]
            pairs = [r["predictions"][mask] for r in rows]
            assert compared["corrected_ids"] == [
                k for k, p in zip(ids, pairs) if p["no_qwen"] != old_rows[k]["gold"] == p["foundation"]]
            assert compared["new_error_ids"] == [
                k for k, p in zip(ids, pairs) if p["no_qwen"] == old_rows[k]["gold"] != p["foundation"]]
    assert all_ids == set(study["original_mp4_sha256"])
    assert len(study["audit48"]["neutral_clips"]) == 9
    assert study["primary_release_checks"] == release_checks(study["primary48"]["conditions"]["observed"]["metrics"])
