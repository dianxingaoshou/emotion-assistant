import unittest
from unittest.mock import patch

from emotion_assistant.assessment import AssessmentWindowCollector, compare_assessments
from emotion_assistant.core import FusionEngine


def observation(identity="x", start=0, end=30, **extra):
    return {"event_id": identity, "session_id": "s", "modality": "visual_summary", "status": "visual_only",
            "t_start": start, "t_end": end, "valid": True, "quality": .8,
            "scores": {"positive": .05, "neutral": .1, "negative": .8, "uncertain": .05},
            "source_ids": [identity], "modalities": ["visual"], "model_versions": {"visual": ["fixture-v1"]},
            "fusion_config_hash": "fixed-config", **extra}


class AssessmentTests(unittest.TestCase):
    def collect(self, phase, start, **changes):
        collector = AssessmentWindowCollector()
        collector.begin("s", "activity", phase, start)
        collector.add(observation(phase, start, start+30, **changes))
        return collector.finish(start+30)

    def test_independent_windows_comparable(self):
        before, after = self.collect("pre", 0), self.collect("post", 60)
        result = compare_assessments(before, after)
        self.assertTrue(result["comparable"])
        self.assertEqual(result["model_delta"]["negative"], 0)
        self.assertEqual(before["covered_seconds"], 30)
        self.assertEqual(before["coverage"], 1)
        self.assertEqual(before["source_ids"], ["pre"])

    def test_window_must_finish_or_be_explicitly_skipped(self):
        collector = AssessmentWindowCollector()
        collector.begin("s", "activity", "pre", 0)
        collector.add(observation())
        self.assertEqual(collector.finish(10)["status"], "collecting")
        result = collector.finish(10, skip=True)
        self.assertEqual(result["status"], "skipped")
        self.assertIn("user_skipped", result["exclusion_reasons"])

    def test_during_never_becomes_post(self):
        collector = AssessmentWindowCollector()
        collector.begin("s", "activity", "post", 100)
        self.assertFalse(collector.add(observation("during", 40, 70)))
        result = collector.finish(130)
        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(result["source_ids"], [])
        self.assertIn("outside_window", result["exclusion_reasons"])

    def test_cross_boundary_classification_not_clipped_into_post(self):
        collector = AssessmentWindowCollector()
        collector.begin("s", "activity", "post", 100)
        self.assertFalse(collector.add(observation("crossing", 95, 105)))
        result = collector.finish(130)
        self.assertEqual(result["status"], "insufficient")
        self.assertIn("crosses_window_boundary", result["exclusion_reasons"])

    def test_overlap_not_double_coverage(self):
        collector = AssessmentWindowCollector()
        collector.begin("s", "activity", "pre", 0)
        collector.add(observation("a", 0, 10))
        collector.add(observation("b", 5, 15))
        result = collector.finish(30)
        self.assertEqual(result["covered_seconds"], 15)
        self.assertEqual(result["coverage"], .5)

    def test_missing_coverage_explicit(self):
        collector = AssessmentWindowCollector()
        collector.begin("s", "activity", "pre", 0)
        collector.add(observation("a", 0, 10))
        result = collector.finish(30)
        self.assertEqual(result["status"], "insufficient")
        self.assertIn("insufficient_valid_coverage", result["exclusion_reasons"])

    def test_conflict_stale_playback_and_session_exclusions(self):
        collector = AssessmentWindowCollector()
        collector.begin("s", "activity", "pre", 0)
        for change in ({"conflict": True}, {"stale": True}, {"system_playback": True}, {"session_id": "other"}):
            self.assertFalse(collector.add(observation(**change)))
        self.assertEqual(collector.finish(30)["covered_seconds"], 0)

    def test_revised_invalid_event_removes_prior_evidence(self):
        collector = AssessmentWindowCollector()
        collector.begin("s", "activity", "pre", 0)
        collector.add(observation("a"))
        collector.add(observation("a", conflict=True))
        self.assertEqual(collector.finish(30)["covered_seconds"], 0)

    def test_model_or_configuration_changes_make_window_insufficient(self):
        for change in ({"model_versions": {"visual": ["new"]}}, {"fusion_config_hash": "new"}):
            collector = AssessmentWindowCollector()
            collector.begin("s", "activity", "pre", 0)
            collector.add(observation("a", 0, 15))
            collector.add(observation("b", 15, 30, **change))
            self.assertEqual(collector.finish(30)["status"], "insufficient")

    def test_comparison_reports_modality_version_and_config_mismatch(self):
        before = self.collect("pre", 0)
        for change, reason in (({"modalities": ["audio"], "model_versions": {"audio": ["fixture-v1"]}}, "different_modalities"),
                               ({"model_versions": {"visual": ["new"]}}, "different_model_versions"),
                               ({"fusion_config_hash": "new"}, "different_fusion_configuration")):
            result = compare_assessments(before, self.collect("post", 60, **change))
            self.assertFalse(result["comparable"])
            self.assertIn(reason, result["reasons"])

    def test_skipped_or_overlapping_windows_never_compared(self):
        before, after = self.collect("pre", 0), self.collect("post", 10)
        self.assertIn("windows_overlap", compare_assessments(before, after)["reasons"])
        after["status"] = "skipped"
        self.assertIn("post_window_incomplete", compare_assessments(before, after)["reasons"])

    def test_partial_joint_modality_coverage_checked_independently(self):
        collector = AssessmentWindowCollector()
        collector.begin("s", "activity", "pre", 0)
        collector.add(observation(modalities=["visual", "audio"], model_versions={"visual": ["v"], "audio": ["a"]},
                                  modality_intervals={"visual": [[0, 5]], "audio": [[0, 30]]}))
        result = collector.finish(30)
        self.assertEqual(result["status"], "insufficient")
        self.assertAlmostEqual(result["coverage_by_modality"]["visual"], 1/6)

    def test_invalid_policy_and_phase(self):
        for kwargs in ({"window_seconds": 0}, {"min_coverage": 0}, {"min_quality": 2}):
            with self.assertRaises(ValueError):
                AssessmentWindowCollector(**kwargs)
        with self.assertRaises(ValueError):
            AssessmentWindowCollector().begin("s", "a", "during", 0)

    def test_settle_allows_only_captured_window_results_and_freezes_once(self):
        collector = AssessmentWindowCollector()
        collector.begin("s", "activity", "post", 100)
        self.assertEqual(collector.snapshot(130)["lifecycle"], "settling")
        self.assertFalse(collector.add(observation("during", 95, 105), received_at=131))
        self.assertTrue(collector.add(observation("late", 100, 130), received_at=132))
        with patch.object(collector, "_finalize", wraps=collector._finalize) as finalize:
            first = collector.snapshot(133)
            self.assertEqual((first["status"], first["lifecycle"]), ("complete", "finalized"))
            self.assertFalse(collector.add(observation("late", 100, 130, conflict=True), received_at=134))
            self.assertEqual(collector.finish(140, skip=True), first)
            first["source_ids"].clear()
            self.assertEqual(collector.snapshot(150)["source_ids"], ["late"])
            finalize.assert_called_once()

    def test_deadline_on_receive_and_early_skip_are_terminal(self):
        collector = AssessmentWindowCollector()
        collector.begin("s", "a", "pre", 0)
        self.assertFalse(collector.add(observation(), received_at=33))
        self.assertEqual(collector.snapshot(34)["status"], "insufficient")
        self.assertEqual(collector.snapshot(100)["finalized_at"], 33)
        collector.begin("s", "a", "post", 100)
        skipped = collector.finish(103, skip=True)
        self.assertFalse(collector.add(observation("late", 100, 115), received_at=131))
        self.assertEqual(collector.finish(150), skipped)

    @staticmethod
    def scores(negative):
        return {"positive": 0., "neutral": 1-negative, "negative": negative, "uncertain": 0.}

    def joint(self, identity, start, end, visual_quality=.8, audio_quality=.8):
        return observation(identity, start, end, status="joint", modalities=["audio", "visual"],
                           scores=self.scores(.6), weights={"audio": .5, "visual": .5},
                           model_versions={"visual": ["fixture-v1"], "audio": ["audio-v1"]},
                           per_modality_scores={"visual": self.scores(.8), "audio": self.scores(.4)},
                           per_modality_quality={"visual": visual_quality, "audio": audio_quality})

    def test_constant_sources_and_different_joint_mix_make_zero_protocol_delta(self):
        collector = AssessmentWindowCollector()
        collector.begin("s", "activity", "pre", 0)
        collector.add(self.joint("first-joint", 0, 15))
        collector.add(observation("first-visual", 15, 30, scores=self.scores(.8)))
        before = collector.finish(33)
        collector.begin("s", "activity", "post", 60, comparison_protocol=before["comparison_protocol"])
        collector.add(self.joint("other-joint", 60, 90))
        after = collector.finish(93)
        result = compare_assessments(before, after)
        self.assertAlmostEqual(before["scores"]["negative"]-after["scores"]["negative"], .1)
        self.assertTrue(result["comparable"])
        self.assertAlmostEqual(result["model_delta"]["negative"], 0)
        self.assertEqual(before["modality_mix"]["joint"], {"seconds": 15., "fraction": .5})
        self.assertEqual(after["modality_mix"]["joint"]["fraction"], 1)
        self.assertAlmostEqual(before["weight_summary"]["audio"], .25)
        self.assertEqual(result["per_modality_delta"]["audio"]["negative"], 0.)

    def test_quality_mismatch_reports_only_the_matching_modality(self):
        c = AssessmentWindowCollector()
        c.begin("s", "activity", "pre", 0)
        c.add(self.joint("pre", 0, 30))
        before = c.finish(33)
        c.begin("s", "activity", "post", 100)
        c.add(self.joint("post", 100, 130, audio_quality=.4))
        result = compare_assessments(before, c.finish(133))
        self.assertEqual(result["scope"], "per_modality_only")
        self.assertEqual(set(result["per_modality_delta"]), {"visual"})
        self.assertIn("different_or_missing_quality", result["modality_exclusions"]["audio"])
        self.assertIsNone(result["model_delta"])

    def test_fusion_engine_payloads_support_matched_composition_protocol(self):
        engine = FusionEngine()
        collector = AssessmentWindowCollector()
        windows = []
        for phase, start, joint_duration in (("pre", 0, 15), ("post", 60, 30)):
            collector.begin("s", "activity", phase, start)
            visual = {"event_id": phase+"-visual", "modality": "visual", "valid": True, "session_id": "s",
                      "quality": .8, "scores": self.scores(.8), "t_start": start, "t_end": start+30,
                      "model_version": "visual-fixture"}
            engine.add_visual(visual)
            joint = engine.fuse_audio({**visual, "event_id": phase+"-audio", "modality": "audio",
                                       "model_version": "audio-fixture", "scores": self.scores(.4),
                                       "t_end": start+joint_duration})
            self.assertTrue(collector.add(joint))
            if joint_duration < 30:
                self.assertTrue(collector.add(engine.visual_summary(start+joint_duration, start+30)))
            windows.append(collector.finish(start+33))
        result = compare_assessments(*windows)
        self.assertEqual(result["scope"], "protocol")
        self.assertAlmostEqual(result["model_delta"]["negative"], 0.)
        self.assertAlmostEqual(windows[0]["scores"]["negative"]-windows[1]["scores"]["negative"], .1)

    def test_different_modalities_report_only_shared_visual(self):
        c = AssessmentWindowCollector()
        c.begin("s", "activity", "pre", 0)
        c.add(self.joint("pre", 0, 30))
        result = compare_assessments(c.finish(33), self.collect("post", 100))
        self.assertEqual(result["scope"], "per_modality_only")
        self.assertEqual(set(result["per_modality_delta"]), {"visual"})
        self.assertIsNone(result["model_delta"])

    def test_legacy_joint_without_source_scores_cannot_synthesize_modality_comparison(self):
        c = AssessmentWindowCollector()
        windows = []
        for phase, start in (("pre", 0), ("post", 60)):
            c.begin("s", "activity", phase, start)
            incoming = self.joint(phase, start, start+30)
            incoming.pop("per_modality_scores")
            c.add(incoming)
            windows.append(c.finish(start+33))
        self.assertFalse(compare_assessments(*windows)["comparable"])
        self.assertEqual(windows[0]["per_modality_scores"], {})


if __name__ == "__main__":
    unittest.main()
