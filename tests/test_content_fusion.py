import pytest
import hashlib
import json

from emotion_assistant.content_fusion import _fuse, fuse_content, summarize_content


def test_policy_file_binds_enabled_candidate_to_the_validated_text_model(tmp_path, monkeypatch):
    from emotion_assistant import content_fusion as module

    artifact = tmp_path / "policy.json"
    english = tmp_path / "model.gz"
    english.write_bytes(b"frozen text model")
    monkeypatch.setattr(module, "ASSET", artifact)
    monkeypatch.setattr(module, "ENGLISH_MODEL", english)
    artifact.write_text(
        json.dumps(
            {
                "model_version": module.MODEL_VERSION,
                "policy": "blend",
                "av_cap": module.AV_CAP,
                "english_classifier_sha256": hashlib.sha256(english.read_bytes()).hexdigest(),
            }
        )
    )
    try:
        module.deployed_policy.cache_clear()
        assert module.deployed_policy() == "blend"
        english.write_bytes(b"replaced")
        module.deployed_policy.cache_clear()
        assert module.deployed_policy() == "text_first"
    finally:
        module.deployed_policy.cache_clear()


def scored(negative, neutral, positive, **extra):
    return {"scores": dict(negative=negative, neutral=neutral, positive=positive, **extra)}


def test_three_way_candidate_and_text_policy_have_distinct_contributions():
    text = scored(0.08, 0.12, 0.8)
    visual = scored(0.75, 0.05, 0.1, uncertain=0.1)
    audio = scored(0.12, 0.55, 0.23, uncertain=0.1)
    joint = _fuse(text, visual, audio, "blend")
    guarded = _fuse(text, visual, audio, "text_first")
    assert joint["available_modalities"] == ["text", "audio", "visual"]
    assert joint["status"] == "multimodal" and joint["disagreement"]
    assert guarded["status"] == "text_preferred" and guarded["weights"] == {"text": 1.0}
    assert joint["scores"] == guarded["candidate_scores"]
    assert joint["scores"] != guarded["scores"]
    assert sum(joint["weights"].values()) == pytest.approx(1.0)
    assert joint["weights"]["text"] >= 0.75
    weak = _fuse(text, dict(visual, coverage=0.2), audio, "blend")
    assert weak["weights"]["visual"] < joint["weights"]["visual"]


def test_losing_text_preserves_proxy_but_never_infers_spoken_meaning():
    proxy = fuse_content(visual=scored(0.8, 0.1, 0.1))
    assert proxy["status"] == "no_text" and proxy["decision"] == "abstain"
    assert proxy["scores"] is proxy["label"] is None
    assert proxy["candidate_label"] is proxy["candidate_estimate_scores"] is None
    assert proxy["candidate_decision"] == "abstain" and proxy["candidate_estimate_weights"] == {}
    assert proxy["weights"] == {} and proxy["candidate_weights"] == {"visual": 1.0}
    for invalid in (
        scored(0.05, 0.05, 0.1, uncertain=0.8),
        scored(float("nan"), 0, 1),
        {"scores": {"neutral": 1.0}},
    ):
        assert fuse_content(audio=invalid)["available_modalities"] == []


def test_summary_weights_by_time_without_extending_text_to_other_frames():
    transcript = _fuse(scored(0.05, 0.15, 0.8), None, None, "text_first")
    quiet = _fuse(None, scored(0.8, 0.1, 0.1), None, "text_first")
    rows = [dict(transcript, t_start=1.0, t_end=3.0), dict(quiet, t_start=3.0, t_end=8.0)]
    summary = summarize_content(rows)
    assert (
        summary["observed_seconds"] == 7 and summary["covered_seconds"] == summary["candidate_seconds"] == 2
    )
    assert summary["scores"] == transcript["scores"] == summary["candidate_scores"]
    assert summary["used_modalities"] == ["text"] and "visual" in summary["available_modalities"]
    with pytest.raises(ValueError, match="overlapping"):
        summarize_content([rows[0], dict(rows[1], t_start=2.0)])


def test_clip_disagreement_can_leave_no_qualified_aggregate():
    rows = [
        dict(_fuse(scored(*p), None, None, "text_first"), t_start=i, t_end=i + 1)
        for i, p in enumerate(((0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8)))
    ]
    result = summarize_content(rows)
    assert result["status"] == "low_confidence" and result["scores"] is None
    assert result["candidate_label"] is None and result["candidate_decision"] == "abstain"
    assert result["candidate_estimate_scores"] is None and result["candidate_estimate_weights"] == {}


def test_diagnostic_raw_average_is_distinct_from_the_actual_gated_joint_candidate():
    weak = scored(0.41, 0.31, 0.28)
    contra = scored(0.15, 0.35, 0.50)
    decisive = scored(0.1, 0.1, 0.8)

    def replay(policy):
        rows = [
            dict(_fuse(weak, contra, contra, policy), t_start=0, t_end=4),
            dict(_fuse(decisive, None, None, policy), t_start=4, t_end=5),
        ]
        assert rows[0]["candidate_scores"] and rows[0]["candidate_label"] is None
        return summarize_content(rows)

    guarded, benchmark = replay("text_first"), replay("blend")
    assert guarded["candidate_scores"] != guarded["candidate_estimate_scores"]
    assert guarded["candidate_estimate_scores"] == benchmark["scores"] == decisive["scores"]
    assert guarded["candidate_estimate_weights"] == benchmark["weights"] == {"text": 1.0}
    assert guarded["candidate_estimate_seconds"] == benchmark["covered_seconds"] == 1.0
    assert guarded["candidate_seconds"] == 5.0
    assert guarded["candidate_label"] == benchmark["label"] == "positive"
    assert guarded["candidate_decision"] == benchmark["decision"] == "estimate"
