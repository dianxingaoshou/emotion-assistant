import unittest

from emotion_assistant.core import FusionEngine, InterventionFlow
from emotion_assistant.evidence import SustainedEvidenceGate, jensen_shannon_divergence, raw_prediction_diagnostics


def observation(identity="x", start=0, end=1, session="s", scores=None, **extra):
    return {"event_id": identity, "session_id": session, "modality": "visual_summary", "status": "visual_only",
            "t_start": start, "t_end": end, "valid": True, "quality": .8,
            "scores": scores or {"positive": .05, "neutral": .1, "negative": .8, "uncertain": .05},
            "raw_scores": {}, "model_version": "fixture", **extra}


class DiagnosticsTests(unittest.TestCase):
    def test_uniform_raw_does_not_become_negative_evidence(self):
        raw = {str(i): .134 if i < 5 else .11 for i in range(8)}
        result = raw_prediction_diagnostics(raw)
        self.assertTrue(result["near_uniform"])
        self.assertFalse(result["decision_eligible"])
        self.assertGreater(result["normalized_entropy"], .98)

    def test_peaked_distribution_retains_diagnostics(self):
        result = raw_prediction_diagnostics({"one": .9, "two": .1})
        self.assertTrue(result["decision_eligible"])
        self.assertAlmostEqual(result["top1"], .9)
        self.assertAlmostEqual(result["margin"], .8)

    def test_missing_raw_explicit_and_compatible(self):
        result = raw_prediction_diagnostics({})
        self.assertFalse(result["available"])
        self.assertIsNone(result["normalized_entropy"])
        self.assertTrue(result["decision_eligible"])

    def test_invalid_raw_is_rejected(self):
        for raw in ({"a": float("nan"), "b": .3}, {"a": -1, "b": 1}, {"a": True, "b": .3}, [1, 2]):
            self.assertFalse(raw_prediction_diagnostics(raw)["decision_eligible"])

    def test_js_properties(self):
        self.assertAlmostEqual(jensen_shannon_divergence({"a": 1}, {"a": 1}), 0)
        self.assertAlmostEqual(jensen_shannon_divergence({"a": 1}, {"b": 1}), 1)
        self.assertIsNone(jensen_shannon_divergence({"a": -1}, {"a": 1}))


class GateTests(unittest.TestCase):
    def test_twenty_effective_seconds_and_settlement(self):
        gate = SustainedEvidenceGate()
        for index in range(40):
            self.assertFalse(gate.observe(observation(str(index), index/2, (index+1)/2), (index+1)/2))
        self.assertFalse(gate.ready(22.9))
        self.assertTrue(gate.ready(23))
        self.assertAlmostEqual(gate.diagnostics(23)["effective_seconds"], 20)

    def test_sparse_bursts_not_counted_as_full_span(self):
        gate = SustainedEvidenceGate()
        for index in range(11):
            gate.observe(observation(str(index), index*2, index*2+.5), index*2+.5)
        result = gate.diagnostics(23.5)
        self.assertFalse(result["ready"])
        self.assertAlmostEqual(result["effective_seconds"], 5.5)
        self.assertLess(result["coverage_density"], .3)

    def test_overlap_and_duplicate_union(self):
        gate = SustainedEvidenceGate(settle_seconds=0)
        for i in range(30):
            gate.observe(observation("same", 0, 10), 10)
        gate.observe(observation("overlap", 5, 15), 15)
        self.assertAlmostEqual(gate.diagnostics(15)["effective_seconds"], 15)
        self.assertFalse(gate.ready(15))

    def test_late_joint_contradiction_revises_visual_coverage(self):
        gate = SustainedEvidenceGate()
        for i in range(20):
            gate.observe(observation(str(i), i, i+1), i+1)
        gate.observe(observation("late", 15, 20, modality="fusion", status="joint", conflict=True), 22)
        self.assertFalse(gate.ready(23))
        self.assertEqual(gate.diagnostics(23)["effective_seconds"], 0)

    def test_same_audio_event_revision_replaces_prior(self):
        gate = SustainedEvidenceGate(settle_seconds=0)
        gate.observe(observation("first", 0, 20, audio_event_id="audio", modality="fusion", status="joint"), 20)
        self.assertTrue(gate.ready(20))
        gate.observe(observation("revised", 0, 20, audio_event_id="audio", modality="fusion", status="joint", conflict=True), 20)
        self.assertFalse(gate.ready(20))

    def test_neutral_resets_latest_chain(self):
        gate = SustainedEvidenceGate(settle_seconds=0)
        gate.observe(observation("long", 0, 20), 20)
        neutral = {"positive": .05, "neutral": .85, "negative": .05, "uncertain": .05}
        gate.observe(observation("neutral", 20, 21, scores=neutral), 21)
        self.assertFalse(gate.ready(21))

    def test_future_stale_unknown_session_and_expired_cannot_prompt(self):
        gate = SustainedEvidenceGate(settle_seconds=0)
        self.assertFalse(gate.observe(observation("future", 0, 20), 19))
        self.assertFalse(gate.observe(observation("stale", 0, 20, stale=True), 20))
        self.assertFalse(gate.observe(observation("unknown", 0, 20, session=""), 20))
        self.assertFalse(gate.observe(observation("expired", 0, 20), 31))
        self.assertEqual(gate.diagnostics(31)["effective_seconds"], 20)
        self.assertEqual(gate.diagnostics(31)["reason"], "stale_evidence")

    def test_aged_contradiction_still_revises_retained_history(self):
        gate = SustainedEvidenceGate()
        for i in range(20):
            gate.observe(observation(str(i), i, i+1), i+1)
        # The contradictory interval ended 12 seconds ago, but is still inside
        # the ledger. It must revise that prefix even though it cannot prompt.
        gate.observe(observation("late", 0, 10, modality="fusion", status="joint", conflict=True), 22)
        self.assertFalse(gate.ready(23))
        self.assertEqual(gate.diagnostics(23)["effective_seconds"], 10)

    def test_new_session_resets_previous_coverage(self):
        gate = SustainedEvidenceGate(settle_seconds=0)
        gate.observe(observation("s1", 0, 15), 15)
        gate.observe(observation("s2", 0, 10, session="s2"), 15)
        self.assertEqual(gate.session_id, "s2")
        self.assertEqual(gate.diagnostics(15)["effective_seconds"], 10)

    def test_invalid_policy(self):
        for kwargs in ({"required_seconds": 0}, {"min_density": 2}, {"settle_seconds": float("nan")}, {"max_gap": -1}):
            with self.assertRaises(ValueError):
                SustainedEvidenceGate(**kwargs)


class FusionUpgradeTests(unittest.TestCase):
    def test_near_uniform_visual_is_not_fused(self):
        engine = FusionEngine()
        engine.add_visual(observation("v", 0, 2, modality="visual", raw_scores={str(i): .125 for i in range(8)}))
        self.assertFalse(engine.visual_summary(0, 2)["valid"])

    def test_neutral_negative_conflict_and_metadata(self):
        engine = FusionEngine()
        engine.add_visual(observation("v", 0, 2, modality="visual", scores={"positive": .02, "neutral": .94, "negative": .02, "uncertain": .02}))
        result = engine.fuse_audio(observation("a", 0, 2, modality="audio"))
        self.assertTrue(result["conflict"])
        self.assertGreater(result["conflict_diagnostics"]["jensen_shannon"], .25)
        self.assertEqual(result["modalities"], ["audio", "visual"])
        self.assertEqual(result["model_versions"], {"audio": ["fixture"], "visual": ["fixture"]})
        self.assertEqual(set(result["source_ids"]), {"v", "a"})
        self.assertEqual(result["fusion_config_hash"], engine.config_hash)

    def test_discomfort_overrides_flow_evaluation_but_retains_numeric(self):
        flow = InterventionFlow()
        flow.request(); flow.confirm(True); flow.start("break", 7)
        result = flow.complete(3, "worse")
        self.assertEqual(result["evaluation"], "discomfort")
        self.assertEqual(result["numeric_evaluation"], "improved")
        self.assertEqual(result["delta"], 4)

    def test_cancel_preserves_process_discomfort(self):
        flow = InterventionFlow()
        flow.request(); flow.confirm(True); flow.start("break", 7)
        flow.checkin("uncomfortable")
        result = flow.cancel()
        self.assertEqual(result["evaluation"], "discomfort")
        self.assertIsNone(result["post"])
