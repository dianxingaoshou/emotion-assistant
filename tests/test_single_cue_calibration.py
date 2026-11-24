import json

import pytest

from emotion_assistant.core import FusionEngine
from emotion_assistant.fusion_policy import FusionConfig, FusionPolicy
from emotion_assistant.models import VERSIONS


def branch(key):
    return {"scores": {"positive": .07, "neutral": .28, "negative": .58, "uncertain": .07},
            "quality": .8, "coverage": 1., "age": .1, "duration": 2., "versions": [VERSIONS[key]]}


def test_project_and_legacy_fusion_cover_missing_inputs_with_the_same_contract(tmp_path):
    production = FusionEngine.from_project(tmp_path).policy
    legacy = FusionPolicy(FusionConfig())
    ctx = {"visual": branch("visual"), "audio": branch("audio")}
    assert production.evaluate(ctx)["scores"] == legacy.evaluate(ctx)["scores"]
    assert production.evaluate({})["decision"] == "abstain"
    assert not production.evaluate({}).get("scores")
    for key, value in ctx.items():
        single = production.evaluate({key: value})
        assert single["available_modalities"] == [key]
        assert single["missing_modalities"] == [other for other in ctx if other != key]
        assert single["missing_modality_calibration"] == "applied"
        assert single["per_modality_scores"][key] == value["scores"]
        assert single["scores"] == single["per_modality_output_scores"][key]
        assert sum(single["scores"].values()) == pytest.approx(1)
        assert legacy.evaluate({key: value})["scores"] != single["scores"]
    changed = {**ctx["visual"], "versions": ["future-encoder"]}
    out = production.evaluate({"visual": changed})
    assert out["scores"] == changed["scores"]
    assert out["missing_modality_calibration"] == "version_mismatch"


def test_explicit_opt_out_restores_legacy_scoring_and_version(tmp_path):
    (tmp_path / "fusion-policy.json").write_text(json.dumps({"adapt_single_modality": False}))
    previous = FusionPolicy(FusionConfig())
    configured = FusionEngine.from_project(tmp_path).policy
    assert configured.basis_hash == previous.basis_hash
    assert configured.evaluate({"audio": branch("audio")}) == previous.evaluate({"audio": branch("audio")})
