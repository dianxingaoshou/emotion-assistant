import unittest

from emotion_assistant.core import FusionEngine, InterventionFlow, interval_duration, interval_union


def observation(event_id="v1", modality="visual", start=0, end=2, quality=0.8,
                valid=True, scores=None, session_id="s1", processed_at=2):
    return {"session_id": session_id, "event_id": event_id, "modality": modality,
            "t_start": start, "t_end": end, "processed_at": processed_at,
            "quality": quality, "valid": valid, "reason": "", "raw_scores": {},
            "model_version": "test", "latency_ms": 10,
            "scores": scores or {"positive": .1, "neutral": .1, "negative": .6, "uncertain": .2}}


class FusionTests(unittest.TestCase):
    def test_visual_without_voice_is_visual_only(self):
        engine = FusionEngine()
        engine.add_visual(observation())
        result = engine.visual_summary(0, 2)
        self.assertEqual(result["status"], "visual_only")
        self.assertEqual(result["weights"]["audio"], 0)
        self.assertAlmostEqual(result["scores"]["uncertain"], .2)

    def test_unknown_mass_is_not_discarded(self):
        engine = FusionEngine()
        scores = {"positive": .05, "neutral": .05, "negative": .1, "uncertain": .8}
        engine.add_visual(observation(scores=scores))
        result = engine.fuse_audio(observation("a1", "audio", scores=scores))
        self.assertAlmostEqual(result["scores"]["uncertain"], .8)
        self.assertAlmostEqual(sum(result["scores"].values()), 1)

    def test_quality_changes_joint_output(self):
        engine = FusionEngine()
        visual = {"positive": .1, "neutral": .1, "negative": .7, "uncertain": .1}
        audio = {"positive": .7, "neutral": .1, "negative": .1, "uncertain": .1}
        engine.add_visual(observation(quality=.2, scores=visual))
        result = engine.fuse_audio(observation("a1", "audio", quality=.8, scores=audio))
        self.assertEqual(result["status"], "joint")
        self.assertAlmostEqual(result["weights"]["visual"], .2)
        self.assertAlmostEqual(result["scores"]["negative"], .22)
        self.assertTrue(result["conflict"])

    def test_low_quality_is_rejected_not_renormalized_to_trust(self):
        engine = FusionEngine()
        engine.add_visual(observation(quality=.1))
        result = engine.fuse_audio(observation("a1", "audio", quality=.1))
        self.assertFalse(result["valid"])
        self.assertIsNone(result["scores"])
        self.assertEqual(result["status"], "unavailable")

    def test_audio_does_not_persist_into_visual_summary(self):
        engine = FusionEngine()
        result = engine.fuse_audio(observation("a1", "audio"))
        self.assertEqual(result["status"], "audio_only")
        self.assertFalse(engine.visual_summary(0, 2)["valid"])

    def test_late_audio_aligns_to_capture_time_not_latest_frame(self):
        engine = FusionEngine()
        engine.add_visual(observation("old", start=0, end=2))
        engine.add_visual(observation("latest", start=10, end=12))
        result = engine.fuse_audio(observation("a1", "audio", processed_at=12))
        self.assertEqual(set(result["source_ids"]), {"old", "a1"})
        missing = engine.fuse_audio(observation("a2", "audio", start=3, end=5))
        self.assertEqual(missing["status"], "audio_only")

    def test_boundary_touch_and_other_session_do_not_count_as_overlap(self):
        engine = FusionEngine()
        engine.add_visual(observation())
        self.assertEqual(engine.fuse_audio(observation("a", "audio", start=2, end=4))["status"], "audio_only")
        self.assertEqual(engine.fuse_audio(observation("a", "audio", session_id="s2"))["status"], "audio_only")

    def test_partial_interval_quality_and_deduplication(self):
        engine = FusionEngine()
        engine.add_visual(observation(end=1))
        engine.add_visual(observation(end=1))
        summary = engine.visual_summary(0, 2)
        self.assertAlmostEqual(summary["quality"], .4)
        self.assertAlmostEqual(summary["coverage"], .5)
        self.assertEqual(summary["source_ids"], ["v1"])
        engine.add_visual(observation("v2", end=1))
        self.assertAlmostEqual(engine.visual_summary(0, 2)["quality"], .4)
        self.assertAlmostEqual(engine.visual_summary(0, 2)["covered_seconds"], 1)

    def test_history_is_bounded(self):
        engine = FusionEngine(history_seconds=3, max_history=2)
        for index in range(5):
            engine.add_visual(observation(str(index), start=index, end=index + 1))
        self.assertFalse(engine.visual_summary(0, 1)["valid"])
        self.assertEqual(set(engine.visual_summary(3, 5)["source_ids"]), {"3", "4"})

    def test_half_second_spans_keep_camera_gaps_and_clip_audio_overlap(self):
        engine = FusionEngine()
        engine.add_visual(observation("first", start=0, end=.5))
        engine.add_visual(observation("first", start=0, end=.5))
        engine.add_visual(observation("after-gap", start=1, end=1.5))
        engine.add_visual(observation("invalid", start=1.5, end=2, valid=False))
        result = engine.fuse_audio(observation("a1", "audio", start=.25, end=1.75))
        self.assertEqual(result["status"], "joint")
        self.assertAlmostEqual(result["visual_coverage"], .5)
        self.assertAlmostEqual(result["weights"]["visual"], 1 / 3)
        self.assertEqual(set(result["source_ids"]), {"first", "after-gap", "a1"})
        # A request lying entirely inside the missing camera interval remains blank.
        self.assertFalse(engine.visual_summary(.5, 1)["valid"])

    def test_invalid_distribution_and_zero_interval(self):
        engine = FusionEngine()
        item = observation("a", "audio")
        item["scores"] = {"negative": float("nan")}
        self.assertFalse(engine.fuse_audio(item)["valid"])
        item["t_end"] = 0
        self.assertEqual(engine.fuse_audio(item)["reason"], "invalid_audio_interval")


class InterventionTests(unittest.TestCase):
    def test_confirmation_and_complete_loop(self):
        flow = InterventionFlow()
        self.assertEqual(flow.state, "observing")
        with self.assertRaises(ValueError):
            flow.start("break", 5)
        self.assertTrue(flow.request())
        self.assertFalse(flow.request())
        flow.confirm(True)
        record = flow.start("break", 7, 60)
        self.assertEqual(flow.state, "running")
        result = flow.complete(4, "helpful")
        self.assertEqual(result["id"], record["id"])
        self.assertEqual(result["delta"], 3)
        self.assertEqual(result["evaluation"], "improved")
        self.assertEqual(flow.state, "cooldown")
        self.assertTrue(flow.request(), "voluntary requests must work during cooldown")

    def test_decline_missing_retest_and_cancellation(self):
        flow = InterventionFlow()
        flow.request()
        flow.confirm(False)
        self.assertEqual(flow.state, "cooldown")
        flow.request()
        flow.confirm(True)
        flow.start("breathing", 5)
        result = flow.complete(None, "unknown")
        self.assertIsNone(result["delta"])
        self.assertEqual(result["evaluation"], "insufficient")
        flow.request()
        result = flow.cancel()
        self.assertEqual(result["status"], "cancelled")
        self.assertIsNone(result["delta"])

    def test_worse_and_state_copies(self):
        flow = InterventionFlow()
        flow.request()
        flow.confirm(True)
        flow.start("break", 3)
        current = flow.current
        current["pre"] = 10
        self.assertEqual(flow.complete(5, "uncomfortable")["delta"], -2)
        self.assertEqual(flow.current["evaluation"], "worse")
        with self.assertRaises(ValueError):
            flow.complete(1, "helpful")

    def test_distress_range(self):
        flow = InterventionFlow()
        flow.request()
        flow.confirm(True)
        for invalid in (-1, 11, True, "5"):
            with self.assertRaises(ValueError):
                flow.start("break", invalid)


class IntervalTests(unittest.TestCase):
    def test_union_does_not_count_rolling_windows_twice(self):
        intervals = [(0, 2), (1, 3), (3, 4), (7, 8), (9, 8), (1, 1)]
        self.assertEqual(interval_union(intervals), [[0, 4], [7, 8]])
        self.assertEqual(interval_duration(intervals), 5)


if __name__ == "__main__":
    unittest.main()
