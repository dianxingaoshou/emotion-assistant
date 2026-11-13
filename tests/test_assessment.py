import unittest

from emotion_assistant.assessment import AssessmentWindowCollector, compare_assessments


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


if __name__ == "__main__":
    unittest.main()
