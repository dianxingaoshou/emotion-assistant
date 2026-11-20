"""Dataset-independent invariants of the paired public benchmark."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bench_real_ravdess_data import specimens, split_for
from bench_real_ravdess_eval import evaluate, stats


def test_holdout_has_disjoint_balanced_actors_and_paired_media():
    groups = {s:{a for a in range(1,25) if split_for(a)==s} for s in ("fit","policy","test")}
    assert all(len(g)==8 and sum(a%2 for a in g)==4 for g in groups.values())
    assert len(set.union(*groups.values()))==sum(map(len,groups.values()))
    for g in groups.values():
        names = [n for a in g for n in specimens(a)]
        assert len(set(names))==64
        assert Counter(n.split("-")[2] for n in names) == dict.fromkeys(("01","03","05","08"),16)
        assert all(n.startswith("01-01-") for n in names)


def test_fixed_coverage_scores_do_not_use_correctness_for_ranking():
    # The two confident errors come first. Exactly floor(.7*4)=2 clips scored.
    p = np.array([[0,1,0,0],[1,0,0,0],[.1,.1,.7,.1],[.1,.1,.1,.7]])
    y = np.arange(4)
    f1, brier, risk = stats(p, y, np.arange(4))
    assert f1 == pytest.approx(.5)
    assert brier == pytest.approx(1.06)
    assert risk == 1


def test_incomplete_data_refuses_to_report_numbers():
    with pytest.raises(ValueError, match="192-clip"):
        evaluate([], {}, {})
