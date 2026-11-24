"""Decision, provenance and evaluation contracts on synthetic external-cue labels."""
import json
import math

import pytest

from emotion_assistant.calibration import Calibration, FEATURES, LABELS, TARGET
from emotion_assistant.core import FusionEngine
from emotion_assistant.evidence import SustainedEvidenceGate
from emotion_assistant.fusion_policy import FusionConfig
from scripts.calibrate_fusion import choose_threshold, load_rows, run, sealed


def probs(top, high=.84):
    return {k: high if k == top else (1 - high) / 3 for k in LABELS}


def obs(modality, top="negative", quality=.8, end=2):
    return {"event_id": modality, "session_id": "s", "modality": modality, "model_version": modality + "-1",
            "valid": True, "scores": probs(top), "t_start": 0., "t_end": end, "quality": quality,
            "raw_scores": {"native1": .88, "native2": .12}, "end_to_end_ms": 400}


def document():
    branches = {key: {"model_version": key + "-1", "temperature": 1.0, "n_fit": 32,
                      "coefficients": [0.] * len(FEATURES), "intercept": intercept}
                for key, intercept in (("visual", -2.), ("audio", 2.))}
    groups = {g: {"threshold": threshold, "n_policy": 12}
              for g, threshold in (("visual_only", None), ("audio_only", .85), ("joint", .5))}
    return sealed(branches, groups)


def write_json(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


def configured(tmp_path, doc=None, baseline=None):
    write_json(tmp_path / "fusion-policy.json", {"strategy": "calibrated_fusion"})
    if doc:
        write_json(tmp_path / "calibration.json", doc)
    return FusionEngine.from_project(tmp_path, baseline_version=baseline)


def test_model_fit_is_group_and_version_specific(tmp_path):
    engine = configured(tmp_path, document())
    engine.add_visual(obs("visual"))
    visual = engine.visual_summary(0, 2)
    assert visual["valid"] and visual["decision"] == "abstain" and visual["decision_threshold"] is None
    item = obs("audio")
    joint = engine.fuse_audio(item)
    assert joint["decision"] == "ask" and joint["status"] == "joint"
    assert joint["weights"]["audio"] > .8 and joint["weights"]["visual"] < .2
    assert joint["per_modality_quality"] == {"audio": .8, "visual": .8}
    assert joint["predictive_uncertainty"]["visual"]["top1"] == pytest.approx(.88)
    assert joint["unmapped_mass"] == pytest.approx((1 - .84) / 3)
    assert joint["per_modality_age_seconds"]["visual"] == pytest.approx(.4)
    assert joint["input_quality"] == pytest.approx(.8)
    single = FusionEngine.from_project(tmp_path).fuse_audio(item)
    assert single["decision"] == "abstain" and single["decision_threshold"] == .85
    item["model_version"] = "audio-2"
    revised = engine.fuse_audio(item)
    assert revised["strategy_used"] == "quality_v2" and revised["calibration_status"] == "version_mismatch"
    assert revised["decision_reasons"][0] == "fallback_version_mismatch"
    assert revised["weights"]["audio"] == .5 and revised["reliability"] == {}


def test_unfitted_invalid_config_and_artifact_fall_back(tmp_path):
    audio = obs("audio")
    baseline = FusionEngine().fuse_audio(audio)
    assert baseline["calibration_status"] == "disabled"
    missing = configured(tmp_path).fuse_audio(audio)
    assert missing["calibration_status"] == "unfitted" and missing["scores"] == baseline["scores"]
    damaged = document()
    damaged["branches"]["audio"]["temperature"] = 9.0
    assert configured(tmp_path, damaged).fuse_audio(audio)["calibration_status"] == "invalid_artifact"
    write_json(tmp_path / "fusion-policy.json", {"strategy": "unknown"})
    result = FusionEngine.from_project(tmp_path).fuse_audio(audio)
    assert result["strategy_used"] == "quality_v2" and "invalid_policy_configuration" in result["decision_reasons"]
    assert FusionEngine().visual_summary(0, 1)["decision"] == "abstain"


def test_real_config_drives_conflict_and_fingerprint(tmp_path):
    first = FusionEngine()
    strict = FusionEngine(config=FusionConfig(conflict_js_threshold=.99))
    for engine in (first, strict):
        engine.add_visual(obs("visual", "neutral"))
    assert first.fuse_audio(obs("audio"))["conflict"] and not strict.fuse_audio(obs("audio"))["conflict"]
    assert first.config_hash != strict.config_hash
    extracted = strict.config
    extracted["conflict_js_threshold"] = .01
    assert strict.config["conflict_js_threshold"] == .99
    a = configured(tmp_path, document(), baseline="first")
    b = configured(tmp_path, document(), baseline="second")
    assert a.config_hash != b.config_hash and a.policy.basis_hash == b.policy.basis_hash


def test_partial_coverage_intrinsic_quality_and_unknown_models():
    core = FusionEngine()
    core.add_visual(obs("visual", quality=.8, end=1))
    summary = core.visual_summary(0, 2)
    assert summary["input_quality"] == pytest.approx(.4) and summary["per_modality_quality"]["visual"] == .8
    assert summary["per_modality_coverage"]["visual"] == .5
    artifact = Calibration.from_document(document())
    engine = FusionEngine(config=FusionConfig(strategy="calibrated_fusion"), calibration=artifact)
    engine.add_visual(obs("visual"))
    engine.add_visual({**obs("visual"), "event_id": "unknown-version", "model_version": None})
    assert engine.fuse_audio(obs("audio"))["calibration_status"] == "version_mismatch"


def test_counterfactual_keeps_target_and_replays_actual_single_channel_policy(tmp_path):
    engine = configured(tmp_path, document())
    engine.add_visual(obs("visual", "positive"))
    acoustic = obs("audio", "negative")
    joint = engine.fuse_audio(acoustic)
    explanation = joint["evidence_explanation"]
    label = max(joint["scores"], key=joint["scores"].get)
    assert joint["conflict"] and explanation["target"] == label == "negative"
    visual = engine.visual_summary(0, 2)
    audio = configured(tmp_path, document()).fuse_audio(acoustic)
    for removed, remaining in (("audio", visual), ("visual", audio)):
        without = explanation["without"][removed]
        assert without["target_score"] == pytest.approx(remaining["scores"][label])
        assert without["score_difference"] == pytest.approx(joint["scores"][label] - remaining["scores"][label])
        assert without["decision"] == remaining["decision"]
    assert visual["evidence_explanation"]["without"]["visual"]["target_score"] is None


def test_rejecting_policy_cannot_accumulate_prompt_evidence(tmp_path):
    engine = configured(tmp_path, document())
    engine.add_visual(obs("visual"))
    result = engine.visual_summary(0, 2)
    gate = SustainedEvidenceGate(required_seconds=1, settle_seconds=0)
    gate.observe(result, 2)
    assert gate.diagnostics(2)["effective_seconds"] == 0
    assert result["scores"]["negative"] > .7 and result["decision"] == "abstain"


def test_zero_weight_and_unrepresented_group_degrade_without_fabricating_reliability(tmp_path):
    zero = FusionEngine(min_quality=0)
    zero.add_visual(obs("visual", quality=0))
    assert not zero.fuse_audio(obs("audio", quality=0))["valid"]
    original = document()
    single = sealed(original["branches"], {"joint": original["groups"]["joint"]})
    result = configured(tmp_path, single).fuse_audio(obs("audio"))
    assert result["calibration_status"] == "missing_modality" and not result["reliability"]
    assert result["per_modality_output_scores"] == result["per_modality_scores"]
    (tmp_path / "fusion-policy.json").write_text('{"strategy":"quality_v2","strategy":"calibrated_fusion"}')
    assert "invalid_policy_configuration" in FusionEngine.from_project(tmp_path).fuse_audio(obs("audio"))["decision_reasons"]


def test_output_distribution_explains_applied_fusion_and_preserves_assessment_scale(tmp_path):
    original = document()
    original["branches"]["visual"]["temperature"] = 3.
    changed = sealed(original["branches"], original["groups"])
    engine = configured(tmp_path, changed)
    engine.add_visual(obs("visual"))
    item = engine.fuse_audio(obs("audio"))
    assert item["per_modality_scores"]["visual"]["negative"] == pytest.approx(.84)
    assert item["per_modality_output_scores"]["visual"]["negative"] < .84
    for label in LABELS:
        reproduced = sum(item["weights"][k] * scores[label] for k, scores in item["per_modality_output_scores"].items())
        assert item["scores"][label] == pytest.approx(reproduced)


def test_group_threshold_keeps_ties_and_empty_acceptance():
    assert choose_threshold([(.8, True)] * 10 + [(.7, False)] * 2, max_policy_error=.1) is None
    assert choose_threshold([(.8, True)] * 12 + [(.7, False)] * 3, max_policy_error=.1) == .8


def synthetic_rows():
    rows = []
    for split in ("fit", "policy", "test"):
        for group, keys in enumerate((("visual",), ("audio",), ("visual", "audio"))):
            for i in range(32):
                label = LABELS[i % len(LABELS)]
                modalities = {}
                for k in keys:
                    predicted = "neutral" if i % 7 == 0 and k == "visual" else label
                    modalities[k] = {"model_version": k + "-1", "scores": probs(predicted),
                                     "quality": .8, "coverage": 1., "age_seconds": .1}
                    if k == "audio":
                        modalities[k]["speech_seconds"] = 2.0
                rows.append({"sample_id": f"{split}-{group}-{i}", "group_id": f"{split}-subject-{i%3}",
                             "target": TARGET, "label": "unmapped" if label == "uncertain" else label,
                             "split": split, "modalities": modalities})
    return rows


def test_offline_fit_all_ablation_models_and_holdout(tmp_path):
    dataset = tmp_path / "cues.jsonl"
    dataset.write_text("\n".join(json.dumps(x) for x in synthetic_rows()), encoding="utf-8")
    report = run(dataset, tmp_path)
    assert report["splits"] == {"fit": 96, "policy": 96, "test": 96}
    assert report["independent_groups"] == {"fit": 3, "policy": 3, "test": 3}
    assert set(report["policies"]) == {"quality_v2", "temperature_only", "gate_only", "calibrated_fusion"}
    for item in report["policies"].values():
        overall = item["overall"]
        assert overall["n"] == 96 and sum(b["n"] for b in overall["reliability_bins"]) == 96
        assert 0 <= overall["macro_f1"] <= 1 and math.isfinite(overall["log_loss"])
        assert 0 <= overall["brier_multiclass"] <= 2
        assert all(item["by_combination"][g]["n"] == 32 for g in ("visual_only", "audio_only", "joint"))
    assert (tmp_path / "reliability.svg").is_file()
    assert Calibration.load(tmp_path / "calibration.json").status == "fitted"
    write_json(tmp_path / "fusion-policy.json", {"strategy": "calibrated_fusion"})
    assert FusionEngine.from_project(tmp_path).fuse_audio(obs("audio"))["calibration_status"] == "fitted"


@pytest.mark.parametrize("bad", ("group_leak", "duplicate", "subjective", "model_probabilities"))
def test_bad_inputs_refused(tmp_path, bad):
    rows = synthetic_rows()
    if bad == "group_leak":
        rows[-1]["group_id"] = rows[0]["group_id"]
    elif bad == "duplicate":
        rows[-1]["sample_id"] = rows[0]["sample_id"]
    elif bad == "subjective":
        rows[0]["target"] = "distress_after_regulation"
    else:
        rows[0]["modalities"]["visual"]["scores"]["negative"] = -1.
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
    with pytest.raises(ValueError):
        load_rows(dataset)
