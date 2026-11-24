from scripts.extract_content_features import align_features, strip_private_data


def test_extraction_keeps_only_model_evidence_and_uses_same_utterance_for_both_text_models():
    primary = {"label": "positive", "scores": {"positive": .8, "negative": .1, "neutral": .1}}
    alternate = {"label": "neutral", "scores": {"positive": .2, "negative": .1, "neutral": .7}}
    transcript = {
        "valid": True, "text": "Example raw speech", "event_id": "arbitrary-run-id",
        "t_start": 1., "t_end": 3., "semantic_context": {"english_polarity": primary},
    }
    result = {"fusion_intervals": [], "audio_samples": [{"event_id": "speech", "t_start": 1., "t_end": 3.}],
              "transcripts": [transcript], "semantic_context": {}, "scores": None,
              "timeline_coverage": {"duration_s": 4.}}
    raw = strip_private_data(result, "clip", None)
    assert "Example raw speech" not in repr(raw)
    row = align_features(raw, False)["intervals"][0]
    assert row["text"] == primary and "dev_text" not in row and "source_ids" not in row
    raw["transcripts"][0]["dev_prediction"] = alternate
    dev = align_features(raw, True)["intervals"][0]
    assert (dev["t_start"], dev["t_end"]) == (1., 3.)
    assert dev["text"] == primary and dev["dev_text"] == alternate
