"""Device-free proofs that acoustic output never waits for optional text."""
from pathlib import Path
import queue
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from emotion_assistant.capture import CaptureRuntime, inference_process
from emotion_assistant.models import ModelHub


class AcousticModel:
    def __init__(self, values=None):
        self.calls = 0
        self.values = values or [.9, .04, .03, .03]

    def generate(self, **kwargs):
        self.calls += 1
        return [{"labels": ["sad", "happy", "neutral", "other"], "scores": self.values}]


class TextModel:
    calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        return [{"text": "<|zh|><|SAD|><|Speech|>可选的转写文本"}]


class AudioAdapterTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.hub = ModelHub(Path(self.scratch.name))
        self.hub._audio_model = AcousticModel()
        self.hub._asr_model = TextModel()
        self.samples = np.full(48000, .03, np.float32)

    def test_acoustic_api_never_calls_asr(self):
        def forbidden(**kwargs):
            raise AssertionError("ASR must not be part of acoustic inference")
        self.hub._asr_model.generate = forbidden
        observation = self.hub.analyze_audio_emotion(self.samples, "s", 0, 3)
        self.assertTrue(observation["valid"])
        self.assertEqual(self.hub._audio_model.calls, 1)
        self.assertIn("prediction_diagnostics", observation)

    def test_near_uniform_raw_is_retained_but_cannot_be_negative_evidence(self):
        self.hub._audio_model = AcousticModel([.25, .25, .25, .25])
        observation = self.hub.analyze_audio_emotion(self.samples, "s", 0, 3)
        self.assertFalse(observation["valid"])
        self.assertIsNone(observation["scores"])
        self.assertEqual(len(observation["raw_scores"]), 4)
        self.assertTrue(observation["prediction_diagnostics"]["near_uniform"])
        self.assertEqual(observation["reason"], "prediction_near_uniform")

    def test_clear_face_with_uniform_model_output_still_refuses_decision(self):
        face = np.array([[40, 40, 200, 200, 100, 100, 180, 100,
                          140, 140, 110, 180, 170, 180, .99]], np.float32)
        class Detector:
            def setInputSize(self, size):
                pass
            def detect(self, frame):
                return None, face
        class VisualModel:
            def get_inputs(self):
                return [SimpleNamespace(name="input")]
            def run(self, *args):
                return [np.zeros((1, 8), np.float32)]
        self.hub._detector, self.hub._visual_model = Detector(), VisualModel()
        frame = np.random.default_rng(7).integers(50, 200, (300, 300, 3), dtype=np.uint8)
        observation = self.hub.analyze_frame(frame, "s", 0, .5)
        self.assertEqual(observation["reason"], "prediction_near_uniform")
        self.assertGreater(observation["quality"], .9)
        self.assertFalse(observation["valid"])
        self.assertIsNone(observation["scores"])
        self.assertEqual(len(observation["raw_logits"]), 8)

    def test_duration_mismatch_does_not_call_either_model(self):
        observation = self.hub.analyze_audio_emotion(self.samples, "s", 0, 8)
        text = self.hub.transcribe_audio(self.samples, "s", 0, 8, True)
        self.assertEqual(observation["reason"], "audio_timestamp_length_mismatch")
        self.assertFalse(text["valid"])
        self.assertEqual(text["reason"], "audio_timestamp_length_mismatch")
        self.assertEqual(self.hub._audio_model.calls, 0)
        self.assertEqual(self.hub._asr_model.calls, 0)

    def test_no_consent_silence_and_clipping_never_call_asr(self):
        self.assertEqual(self.hub.transcribe_audio(self.samples, "s", 0, 3, False)["reason"], "consent_not_granted")
        self.assertFalse(self.hub.transcribe_audio(np.zeros_like(self.samples), "s", 0, 3, True)["valid"])
        self.assertFalse(self.hub.transcribe_audio(np.ones_like(self.samples), "s", 0, 3, True)["valid"])
        self.assertEqual(self.hub._asr_model.calls, 0)

    def test_asr_text_has_no_extra_emotion_vote(self):
        text = self.hub.transcribe_audio(self.samples, "s", 0, 3, True)
        self.assertEqual(text["text"], "可选的转写文本")
        self.assertNotIn("scores", text)
        self.assertNotIn("raw_scores", text)
        self.assertFalse(text["is_current_concern"])

    def test_legacy_api_still_returns_observation_and_text(self):
        observation, text = self.hub.analyze_audio(self.samples, "s", 0, 3, True)
        self.assertTrue(observation["valid"])
        self.assertTrue(text["valid"])
        self.assertEqual(self.hub._asr_model.calls, 1)

    def test_acoustic_loader_does_not_load_text_model(self):
        self.hub._audio_model = self.hub._asr_model = None
        calls = []
        self.hub._load_funasr = lambda folder, required: calls.append(folder) or object()
        self.hub.load_audio_emotion()
        self.assertEqual(calls, ["emotion2vec_plus_seed"])
        self.assertIsNone(self.hub._asr_model)


def fake_runtime():
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    runtime.session_id = "s"
    runtime.origin = time.monotonic() - 3
    runtime.generation = 0
    runtime.asr_allowed = True
    runtime._consent_value = SimpleNamespace(value=0)
    runtime._generation_value = SimpleNamespace(value=0)
    runtime._asr_permission = threading.Event()
    runtime._asr_permission.set()
    runtime.stop_event, runtime.pause_event = threading.Event(), threading.Event()
    runtime.jobs = {"audio": queue.Queue(2), "asr": queue.Queue(2)}
    runtime.processes = {}
    runtime.microphone = 0
    runtime.events = []
    runtime.on_event = runtime.events.append
    return runtime


class PipelineTests(unittest.TestCase):
    def test_runtime_starts_independent_workers_with_two_pending_audio_jobs_each(self):
        runtime = CaptureRuntime("s", -1, 0, Path("."), lambda frame: None,
                                 lambda event: None, asr_allowed=True)
        starts = []
        try:
            with patch.object(runtime, "_spawn", side_effect=lambda name, target, args: starts.append((name, target, args))), \
                    patch("emotion_assistant.capture.threading.Thread"):
                runtime.start()
            self.assertEqual([name for name, _, _ in starts], ["audio", "asr", "microphone"])
            workers = {name: args for name, _, args in starts if name in ("audio", "asr")}
            self.assertIsNot(workers["audio"][4], workers["asr"][4])
            for stage in ("audio", "asr"):
                runtime.jobs[stage].put_nowait("first")
                runtime.jobs[stage].put_nowait("second")
                with self.assertRaises(queue.Full):
                    runtime.jobs[stage].put_nowait("third")
        finally:
            runtime.thread = None
            runtime.stop()

    def test_pcm_snapshots_are_independent_and_immutable(self):
        runtime = fake_runtime()
        samples = np.full(16000, .04, np.float32)
        runtime._queue_segment(samples, 0, 1, set())
        acoustic, text = runtime.jobs["audio"].get_nowait(), runtime.jobs["asr"].get_nowait()
        samples[:] = 0
        self.assertTrue(np.all(acoustic[1] == .04))
        self.assertTrue(np.all(text[1] == .04))
        self.assertFalse(acoustic[1].flags.writeable)
        self.assertFalse(text[1].flags.writeable)
        self.assertFalse(np.shares_memory(acoustic[1], text[1]))

    def test_asr_overflow_does_not_drop_acoustic_job(self):
        runtime = fake_runtime()
        runtime.jobs["asr"].put_nowait("first")
        runtime.jobs["asr"].put_nowait("second")
        runtime._queue_segment(np.zeros(16000, np.float32), 0, 1, set())
        self.assertEqual(runtime.jobs["audio"].qsize(), 1)
        self.assertEqual(runtime.jobs["asr"].qsize(), 2)
        self.assertEqual(runtime.events[0]["stage"], "asr")
        self.assertEqual(runtime.events[0]["reason"], "queue_full")
        self.assertEqual(runtime.events[0]["dropped_count"], 1)

    def test_revoke_then_reenable_rejects_old_inflight_text(self):
        runtime = fake_runtime()
        runtime.processes["asr"] = object()
        old = {"kind": "utterance", "generation": 0, "consent_generation": 0,
               "utterance": {"text": "old", "t_end": 1}}
        runtime.set_asr_allowed(False)
        runtime._publish_result(old)
        runtime.set_asr_allowed(True)
        runtime._publish_result(old)
        self.assertFalse(any(event["kind"] == "utterance" for event in runtime.events))
        fresh = {**old, "consent_generation": 2, "utterance": {"text": "fresh", "t_end": 2}}
        runtime._publish_result(fresh)
        self.assertEqual(runtime.events[-1]["utterance"]["text"], "fresh")

    def test_pause_and_stop_reject_inflight_output(self):
        runtime = fake_runtime()
        event = {"kind": "observation", "generation": 0, "observation": {"t_end": 2}}
        runtime.pause_event.set()
        runtime._publish_result(event)
        runtime.pause_event.clear()
        runtime.generation = 1
        runtime._publish_result(event)
        runtime.generation = 0
        runtime.stop_event.set()
        runtime._publish_result(event)
        self.assertEqual(runtime.events, [])

    def test_no_consent_never_enqueues_text(self):
        runtime = fake_runtime()
        runtime.asr_allowed = False
        runtime._queue_segment(np.ones(16000, np.float32), 0, 1, set())
        self.assertEqual(runtime.jobs["audio"].qsize(), 1)
        self.assertTrue(runtime.jobs["asr"].empty())

    def test_new_consent_does_not_transcribe_audio_captured_before_it(self):
        runtime = fake_runtime()
        runtime.processes["asr"] = object()
        runtime.set_asr_allowed(False)
        runtime.set_asr_allowed(True)
        runtime._queue_segment(np.zeros(16000, np.float32), 0, 1, set())
        self.assertEqual(runtime.jobs["audio"].qsize(), 1)
        self.assertTrue(runtime.jobs["asr"].empty())

    def test_worker_ignores_jobs_from_old_session_generation(self):
        results, jobs, stop = queue.Queue(), queue.Queue(), threading.Event()
        jobs.put(("old", np.zeros(16000, np.float32), 0, 1, 0))
        jobs.put(None)
        class FakeHub:
            def __init__(self, *args, **kwargs):
                pass
            def load_audio_emotion(self):
                return {"audio": {"ready": True}}
            def analyze_audio_emotion(self, *args):
                raise AssertionError("pre-pause work must never be inferred")
        with patch("emotion_assistant.models.ModelHub", FakeHub):
            inference_process("audio", ".", "s", time.monotonic(), jobs, results, stop,
                              False, generation_value=SimpleNamespace(value=1))
        observed = []
        while not results.empty():
            observed.append(results.get_nowait()["kind"])
        self.assertEqual(observed, ["loaded"])

    def test_slow_asr_worker_cannot_hold_acoustic_publication(self):
        stop, permission, asr_entered, asr_release = [threading.Event() for _ in range(4)]
        permission.set()
        jobs_audio, jobs_asr, results = queue.Queue(2), queue.Queue(2), queue.Queue(64)
        origin = time.monotonic() - 3
        snapshot = np.zeros(48000, np.float32)
        job = ("job", snapshot, 0, 3, 0, time.monotonic()-origin, 0)
        jobs_audio.put(job)
        jobs_asr.put(job)

        class FakeHub:
            def __init__(self, *args, **kwargs):
                pass
            def load_audio_emotion(self):
                return {"audio": {"ready": True}}
            def load_asr(self):
                return {"asr": {"ready": True}}
            def analyze_audio_emotion(self, *args):
                return {"valid": True, "t_end": 3, "modality": "audio"}
            def transcribe_audio(self, *args, **kwargs):
                asr_entered.set()
                asr_release.wait(3)
                return {"text": "late", "t_end": 3, "valid": True}

        with patch("emotion_assistant.models.ModelHub", FakeHub):
            threads = [threading.Thread(target=inference_process, args=(
                stage, ".", "s", origin, jobs, results, stop, True, permission,
                SimpleNamespace(value=0), SimpleNamespace(value=0)))
                for stage, jobs in (("asr", jobs_asr), ("audio", jobs_audio))]
            try:
                threads[0].start()
                self.assertTrue(asr_entered.wait(1))
                threads[1].start()
                observed = []
                deadline = time.monotonic()+1
                while time.monotonic() < deadline:
                    event = results.get(timeout=1)
                    observed.append(event)
                    if event["kind"] == "observation":
                        break
                self.assertTrue(any(event["kind"] == "observation" for event in observed))
                self.assertFalse(any(event["kind"] == "utterance" for event in observed))
                audio = next(event["observation"] for event in observed if event["kind"] == "observation")
                self.assertGreaterEqual(audio["queue_wait_ms"], 0)
                self.assertGreaterEqual(audio["end_to_end_ms"], audio["inference_ms"])
                self.assertIn("inference_finished", audio["timing"])
            finally:
                stop.set()
                asr_release.set()
                for thread in threads:
                    if thread.ident:
                        thread.join(2)
                self.assertTrue(all(not thread.is_alive() for thread in threads))


if __name__ == "__main__":
    unittest.main()
