import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest

from emotion_assistant.storage import Store


def observation(event_id="v1", modality="visual", start=0, end=3, quality=.8):
    return {"event_id": event_id, "modality": modality, "t_start": start,
            "t_end": end, "quality": quality, "valid": True,
            "scores": {"positive": .1, "neutral": .1, "negative": .6, "uncertain": .2}}


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.store = Store(self.path / "test.sqlite3")
        self.user_id = self.store.create_user("测试用户")
        self.session_id = self.store.start_session(self.user_id, {"save_transcript": False}, "demo")

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def exported(self, session_id=None):
        path = self.store.export_session(session_id or self.session_id, self.path / "export.json")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_profile_session_and_utc_stamps(self):
        self.assertEqual(self.store.list_users(), [{"id": self.user_id, "name": "测试用户"}])
        self.assertEqual(len(self.store.list_sessions(self.user_id)), 1)
        self.store.finish_session(self.session_id)
        session = self.store.list_sessions()[0]
        self.assertTrue(session["started_at"].endswith("+00:00"))
        self.assertTrue(session["ended_at"].endswith("+00:00"))

    def test_transcript_is_removed_even_if_caller_attempts_to_store_it(self):
        self.store.save_utterance(self.session_id, {"id": "u", "text": "不应保存的转写", "t_start": 0})
        self.store.append_event(self.session_id, "asr", {"nested": {"transcript": "敏感内容"}, "valid": True})
        result = self.exported()
        self.assertNotIn("text", result["utterances"][0])
        self.assertEqual(result["events"][0]["payload"], {"nested": {}, "valid": True})
        self.assertNotIn("敏感内容", json.dumps(result, ensure_ascii=False))

    def test_sqlite_trigger_blocks_bypass_without_opt_in(self):
        with closing(sqlite3.connect(self.path / "test.sqlite3")) as database:
            with self.assertRaises(sqlite3.IntegrityError):
                database.execute("INSERT INTO utterances VALUES (?, ?, ?)",
                                 (self.session_id, "bypass", '{"text":"private"}'))

    def test_explicit_consent_preserves_transcript(self):
        session_id = self.store.start_session(self.user_id, {"save_transcript": True}, "real")
        self.store.save_utterance(session_id, {"id": "u", "text": "授权记录"})
        self.assertEqual(self.exported(session_id)["utterances"][0]["text"], "授权记录")

    def test_utterance_cannot_cross_session_boundary(self):
        with self.assertRaises(ValueError):
            self.store.save_utterance(self.session_id, {"id": "u", "session_id": "old-session", "text": "old"})
        self.assertEqual(self.exported()["utterances"], [])

    def test_denial_overrides_compatibility_alias(self):
        session_id = self.store.start_session(self.user_id, {"save_transcript": False, "transcript": True}, "real")
        self.store.save_utterance(session_id, {"id": "u", "text": "不保存"})
        self.assertNotIn("text", self.exported(session_id)["utterances"][0])

    def test_deduplication_union_coverage_and_upsert_intervention(self):
        self.store.save_observation(self.session_id, observation())
        self.store.save_observation(self.session_id, observation())
        self.store.save_observation(self.session_id, observation("v2", start=2, end=5))
        self.store.save_observation(self.session_id, observation("low", start=10, end=20, quality=.1))
        self.store.save_observation(self.session_id, observation("a1", "audio", 1, 4))
        self.store.save_intervention(self.session_id, {"id": "i", "status": "running", "pre": 7})
        self.store.save_intervention(self.session_id, {"id": "i", "status": "completed", "pre": 7, "post": 4, "delta": 3})
        self.store.save_self_report(self.session_id, "pre", 7, "i")
        result = self.exported()
        self.assertEqual(len(result["observations"]), 4)
        self.assertEqual(result["coverage"]["visual_seconds"], 5)
        self.assertEqual(result["coverage"]["audio_seconds"], 3)
        self.assertEqual(result["coverage"]["joint_seconds"], 3)
        self.assertEqual(result["coverage"]["observed_seconds"], 5)
        self.assertEqual(len(result["interventions"]), 1)
        self.assertEqual(result["interventions"][0]["delta"], 3)
        summary = self.store.list_sessions()[0]
        self.assertEqual(summary["observation_count"], 4)
        self.assertEqual(summary["raw_observation_count"], 4)
        self.assertEqual(summary["intervention_count"], 1)
        self.assertEqual(summary["last_delta"], 3)
        self.assertGreaterEqual(summary["duration_seconds"], 0)
        self.assertEqual(summary["coverage"]["joint_seconds"], 3)

    def test_deletion_cascades_every_session_table_but_keeps_profile(self):
        self.store.save_observation(self.session_id, observation())
        self.store.save_utterance(self.session_id, {"id": "u"})
        self.store.save_intervention(self.session_id, {"id": "i"})
        self.store.save_self_report(self.session_id, "pre", 4)
        self.store.append_event(self.session_id, "test", {"valid": True})
        self.store.delete_session(self.session_id)
        self.assertEqual(self.store.list_sessions(), [])
        self.assertEqual(len(self.store.list_users()), 1)
        with closing(sqlite3.connect(self.path / "test.sqlite3")) as database:
            for table in ("observations", "utterances", "interventions", "self_reports", "events"):
                self.assertEqual(database.execute(f"SELECT count(*) FROM {table}").fetchone()[0], 0)

    def test_concurrent_worker_writes(self):
        errors = []
        def writer(index):
            try:
                for event in range(10):
                    self.store.save_observation(self.session_id, observation(f"{index}-{event}"))
            except Exception as error:
                errors.append(error)
        threads = [threading.Thread(target=writer, args=(index,)) for index in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(self.exported()["observations"]), 30)

    def test_user_input_validation(self):
        with self.assertRaises(ValueError):
            self.store.create_user(" ")
        with self.assertRaises(ValueError):
            self.store.save_self_report(self.session_id, "pre", 12)
        with self.assertRaises(ValueError):
            self.store.start_session("nonexistent", {}, "test")


if __name__ == "__main__":
    unittest.main()
