"""Fast refusal/consent checks; real neural smoke is an explicit separate script."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from emotion_assistant.models import ModelHub, _canonical


class CallRecorder:
    """Only records ASR calls, never simulates emotion classification."""
    def __init__(self):
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        return [{"text": "<|zh|>test consent"}]


class ModelBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.hub = ModelHub(Path(self.directory.name))

    def audio(self, samples, **kwargs):
        return self.hub.analyze_audio(samples, "test", 0, len(samples)/16000, **kwargs)

    def assert_refused(self, result, reason):
        observation, utterance = result
        self.assertFalse(observation["valid"])
        self.assertIsNone(observation["scores"])
        self.assertEqual(observation["raw_scores"], {})
        self.assertEqual(observation["reason"], reason)
        self.assertIsNone(utterance)

    def test_missing_model_never_fills_a_neutral_score(self):
        self.assertFalse(self.hub.load_audio()["audio"]["ready"])
        result, utterance = self.audio(np.full(48000, .03, np.float32))
        self.assertFalse(result["valid"])
        self.assertIsNone(result["scores"])
        self.assertTrue(result["reason"].startswith("model_unavailable"))
        self.assertIsNone(utterance)
        visual = self.hub.analyze_frame(np.zeros((100, 100, 3), np.uint8), "test", 0, .5)
        self.assertFalse(visual["valid"])
        self.assertIsNone(visual["scores"])

    def test_silence_short_speech_and_clipping_refuse(self):
        self.assert_refused(self.audio(np.zeros(48000, np.float32)), "audio_silent_or_too_quiet")
        self.assert_refused(self.audio(np.full(16000, .03, np.float32)), "insufficient_speech_duration")
        self.assert_refused(self.audio(np.ones(48000, np.float32)), "audio_clipped")

    def test_invalid_samples_and_timestamp_mismatch_refuse(self):
        self.assert_refused(self.audio(np.full(48000, np.nan, np.float32)), "invalid_mono_audio")
        observation, _ = self.hub.analyze_audio(np.full(48000, .03, np.float32), "test", 0, 8)
        self.assertFalse(observation["valid"])
        self.assertEqual(observation["reason"], "audio_timestamp_length_mismatch")

    def test_asr_generate_requires_current_consent(self):
        recorder = CallRecorder()
        self.hub._asr_model = recorder
        # Acoustic model remains unavailable; this isolates the ASR call boundary.
        samples = np.full(48000, .03, np.float32)
        _, utterance = self.audio(samples, asr_allowed=False)
        self.assertEqual(recorder.calls, 0)
        self.assertIsNone(utterance)
        _, utterance = self.audio(samples, asr_allowed=True)
        self.assertEqual(recorder.calls, 1)
        self.assertEqual(utterance["text"], "test consent")
        self.hub.enable_asr = False
        self.audio(samples, asr_allowed=True)
        self.assertEqual(recorder.calls, 1)

    def test_uncertain_probability_mass_is_preserved(self):
        result = _canonical({"happiness": .3, "neutral": .1, "sad": .1,
                             "surprised": .2, "other": .1, "<unk>": .2})
        self.assertAlmostEqual(result["uncertain"], .5)
        self.assertAlmostEqual(result["positive"], .3)
        self.assertAlmostEqual(sum(result.values()), 1)


if __name__ == "__main__":
    unittest.main()
