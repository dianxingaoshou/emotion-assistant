"""Keep held-out video inputs independent of the official test captions."""
import csv
import json

from scripts import prepare_chsims_foundation as prepare


def test_audit_is_disjoint_and_test_never_uses_reference_caption(tmp_path, monkeypatch):
    primary_id, audit_id = "video_0001/0001", "video_0001/0002"
    train_id = "video_0002/0001"
    with (tmp_path / "label.csv").open("w", encoding="utf8", newline="") as stream:
        csv.writer(stream).writerows([
            [*key.split("/"), "official caption", "Positive", split]
            for key, split in ((train_id, "train"), (primary_id, "test"), (audit_id, "test"))
        ])
    monkeypatch.setattr(prepare, "cohort", lambda _: {primary_id: None})
    dest = tmp_path / "prepared"
    observed = dest / "observations" / f"{primary_id}.json"
    observed.parent.mkdir(parents=True)
    observed.write_text(json.dumps({"text": "actual ASR"}), encoding="utf8")
    main, main_ids = prepare.selected_rows(tmp_path, dest)
    audit, audit_ids = prepare.selected_rows(tmp_path, dest, audit_only=True)
    assert main_ids["test"] == [primary_id] and audit_ids["audit"] == [audit_id]
    assert [(row["split"], row["text"], row["text_source"]) for row in main] == [
        ("train", "official caption", "official_caption"), ("test", "actual ASR", "desktop_asr")
    ]
    assert audit[0]["text"] == "" and audit[0]["text_source"] == "desktop_asr"
