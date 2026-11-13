"""Device-free checks for endpointing, bounded queues and runtime lifecycle."""
import queue
import threading
import time
import unittest

import numpy as np

from emotion_assistant.capture import CaptureRuntime, SpeechSegmenter, _offer


class SpeechSegmenterTests(unittest.TestCase):
    def setUp(self):
        self.block = np.ones(512, np.float32)

    def test_silence_is_not_an_utterance_and_preroll_stays_bounded(self):
        segmenter = SpeechSegmenter()
        for index in range(100):
            self.assertIsNone(segmenter.feed(self.block, index * .032, .1))
        self.assertEqual(len(segmenter.preroll), 6)
        self.assertIsNone(segmenter.start)

    def test_endpoint_includes_bounded_preroll_and_keeps_interval_sample_length(self):
        segmenter = SpeechSegmenter()
        result = None
        for index in range(6 + 3 + 19):
            probability = .8 if 6 <= index < 9 else .1
            result = segmenter.feed(self.block, index * .032, probability)
        self.assertIsNotNone(result)
        samples, start, end = result
        self.assertAlmostEqual(start, 0)
        self.assertAlmostEqual(end, 28 * .032)
        self.assertAlmostEqual(len(samples) / 16000, end - start)
        self.assertIsNone(segmenter.start)

    def test_maximum_duration_splits_continuous_speech(self):
        segmenter = SpeechSegmenter(maximum=1)
        result = None
        for index in range(32):
            result = segmenter.feed(self.block, index * .032, .9)
        self.assertIsNotNone(result)
        samples, start, end = result
        self.assertLessEqual(end - start, 1.032)
        self.assertAlmostEqual(len(samples) / 16000, end - start)
        self.assertIsNone(segmenter.feed(self.block, 32 * .032, .9))
        self.assertAlmostEqual(segmenter.start, 32 * .032)

    def test_copies_input_and_reset_discards_active_speech(self):
        segmenter = SpeechSegmenter()
        segmenter.feed(self.block, 0, .9)
        self.block[:] = 0
        self.assertTrue(np.all(segmenter.blocks[0] == 1))
        segmenter.reset()
        self.assertEqual(segmenter.blocks, [])
        self.assertIsNone(segmenter.start)

    def test_timestamp_gap_never_stitches_disconnected_speech(self):
        segmenter = SpeechSegmenter()
        segmenter.feed(self.block, 0, .9)
        segmenter.feed(self.block, 3, .9)
        result = None
        for index in range(19):
            result = segmenter.feed(self.block, 3.032 + index * .032, .1)
        self.assertIsNotNone(result)
        samples, start, end = result
        self.assertGreaterEqual(start, 3, "a dropped interval must reset active speech")
        self.assertAlmostEqual(len(samples) / 16000, end - start)


class QueueTests(unittest.TestCase):
    def test_full_audio_queue_reports_loss_without_destroying_old_segment(self):
        target = queue.Queue(maxsize=1)
        self.assertTrue(_offer(target, "first"))
        self.assertFalse(_offer(target, "second"))
        self.assertEqual(target.get_nowait(), "first")

    def test_latest_mode_replaces_pending_old_video(self):
        target = queue.Queue(maxsize=1)
        target.put_nowait("old")
        self.assertTrue(_offer(target, "new", latest=True))
        self.assertEqual(target.get_nowait(), "new")


def fake_runtime():
    """Use only Python queues/events; never initialize devices or processes."""
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    runtime.session_id = "test"
    runtime.origin = time.monotonic()
    runtime.accept_after = 0.0
    runtime.generation = 0
    runtime.stop_event = threading.Event()
    runtime.pause_event = threading.Event()
    runtime.frames = queue.Queue(2)
    runtime.segments = queue.Queue(3)
    runtime.results = queue.Queue(64)
    runtime.jobs = {"visual": queue.Queue(1), "audio": queue.Queue(3)}
    runtime.processes = {}
    runtime.thread = None
    return runtime


class RuntimeTests(unittest.TestCase):
    def test_pause_invalidates_generation_and_drains_waiting_capture(self):
        runtime = fake_runtime()
        for target in (runtime.frames, runtime.segments, *runtime.jobs.values()):
            target.put_nowait("pending")
        runtime.pause(True)
        self.assertTrue(runtime.pause_event.is_set())
        self.assertEqual(runtime.generation, 1)
        self.assertTrue(all(target.empty() for target in
                            (runtime.frames, runtime.segments, *runtime.jobs.values())))
        runtime.pause(False)
        self.assertFalse(runtime.pause_event.is_set())
        self.assertEqual(runtime.generation, 2)

    def test_monitor_ignores_pre_pause_results_and_shows_latest_frame(self):
        runtime = fake_runtime()
        runtime.generation = 2
        events, frames = [], []
        runtime.on_event = events.append
        runtime.on_frame = frames.append
        runtime.results.put_nowait({"kind": "observation", "generation": 1, "observation": {"id": "stale"}})
        runtime.results.put_nowait({"kind": "observation", "generation": 2, "observation": {"id": "current"}})
        runtime.frames.put_nowait((1, "old-frame"))
        runtime.frames.put_nowait((2, "new-frame"))
        class OneIteration:
            stopped = False
            def is_set(self):
                return self.stopped
            def wait(self, timeout):
                self.stopped = True
        runtime.stop_event = OneIteration()
        runtime._monitor()
        self.assertEqual([event["observation"]["id"] for event in events], ["current"])
        self.assertEqual(frames, ["new-frame"])
        job = runtime.jobs["visual"].get_nowait()
        self.assertEqual(job[1:5], ("new-frame", 1.5, 2, 2))
        self.assertIsInstance(job[5], float)
        self.assertGreaterEqual(job[5], 0, "visual queue wait must use the actual enqueue time")

    def test_stop_joins_monitor_before_closing_process_handles(self):
        runtime = fake_runtime()
        order = []
        class FakeProcess:
            def join(self, timeout):
                order.append("process_join")
            def is_alive(self):
                return False
            def close(self):
                order.append("process_close")
        class FakeThread:
            def join(self, timeout):
                order.append("thread_join")
        class CloseableQueue(queue.Queue):
            def cancel_join_thread(self):
                pass
            def close(self):
                order.append("queue_close")
        runtime.processes = {"visual": FakeProcess()}
        runtime.thread = FakeThread()
        runtime.frames, runtime.segments, runtime.results = [CloseableQueue() for _ in range(3)]
        runtime.jobs = {"visual": CloseableQueue(), "audio": CloseableQueue()}
        runtime.stop()
        self.assertLess(order.index("thread_join"), order.index("process_close"),
                        "monitor may still call is_alive(); closing its handles first races")
        self.assertLess(order.index("thread_join"), order.index("queue_close"))

    def test_monitor_rejects_pre_resume_frame_and_straddling_audio(self):
        runtime = fake_runtime()
        runtime.accept_after = 5
        frames = []
        runtime.on_event = lambda event: None
        runtime.on_frame = frames.append
        runtime.frames.put_nowait((4.9, "pre-resume"))
        runtime.segments.put_nowait((np.zeros(16000, np.float32), 4.5, 5.5))
        class OneIteration:
            stopped = False
            def is_set(self):
                return self.stopped
            def wait(self, timeout):
                self.stopped = True
        runtime.stop_event = OneIteration()
        runtime._monitor()
        self.assertEqual(frames, [])
        self.assertTrue(runtime.jobs["visual"].empty())
        self.assertTrue(runtime.jobs["audio"].empty())

    def test_first_post_resume_visual_span_does_not_cover_paused_time(self):
        runtime = fake_runtime()
        runtime.accept_after = 5
        runtime.on_event = lambda event: None
        runtime.on_frame = lambda frame: None
        runtime.frames.put_nowait((5.2, "resumed-frame"))
        class OneIteration:
            stopped = False
            def is_set(self):
                return self.stopped
            def wait(self, timeout):
                self.stopped = True
        runtime.stop_event = OneIteration()
        runtime._monitor()
        job = runtime.jobs["visual"].get_nowait()
        self.assertGreaterEqual(job[2], 5)
        self.assertEqual(job[3], 5.2)


if __name__ == "__main__":
    unittest.main()
