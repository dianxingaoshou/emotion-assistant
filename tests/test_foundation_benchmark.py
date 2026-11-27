import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_chsims_foundation import numbers
from fit_chsims_foundation import SCENARIOS, file_sha, fusion, metrics


def test_abstention_remains_in_positive_recall_denominator():
    m = metrics(np.array([2,2,0]), np.array([2,-1,0]))
    assert (m["positive_hits"], m["positive_support"], m["positive_recall"]) == (1,2,.5)
    assert (m["correct"],m["wrong"],m["abstain"]) == (2,0,1)


def test_frozen_results_match_published_decisions_and_head():
    result = json.loads((ROOT / "docs/results/chsims-foundation-fusion.json").read_text(encoding="utf8"))
    old = json.loads((ROOT / "docs/results/chsims-conflict.json").read_text(encoding="utf8"))
    primary_ids = {r["id"] for r in result["primary48"]["clips"]}
    audit_ids = {r["id"] for r in result["audit48"]["clips"]}
    assert len(primary_ids) == len(audit_ids) == 48 and primary_ids.isdisjoint(audit_ids)
    assert primary_ids == {r["id"] for r in old["clips"]["observed"]}
    assert file_sha(fusion.ASSET) == result["fit"]["model_sha256"]
    assert fusion.bundled_head() is not None
    for name in ("primary48","audit48"):
        rows = result[name]["clips"]
        y = numbers([r["gold"] for r in rows])
        for condition in SCENARIOS:
            for variant in rows[0]["predictions"][condition]:
                p = numbers([r["predictions"][condition][variant] for r in rows])
                assert result[name]["conditions"][condition]["metrics"][variant] == metrics(y,p)
