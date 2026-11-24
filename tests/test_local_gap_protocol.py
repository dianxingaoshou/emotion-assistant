import numpy as np

from scripts.benchmark_local_gaps import RATE, acoustic_chunks, mask_visual


def test_centre_mask_never_reuses_hidden_visual_evidence_or_input_samples():
    visual = [{"sample_time_s":i+.5, "t_start":i, "t_end":i+1} for i in range(3)]
    kept = mask_visual(visual, .75, 2.25)
    assert [(row["t_start"],row["t_end"]) for row in kept] == [(0,.75),(2.25,3)]
    assert [(row["t_start"],row["t_end"]) for row in visual] == [(0,1),(1,2),(2,3)]
    wave = np.arange(12)
    calls = []

    class Hub:
        def analyze_audio_emotion(self, samples, identifier, start, end):
            calls.append((samples.tolist(),start,end))
            return {"reason":"insufficient_speech_duration","valid":False}

    cache = {}
    spans = [(0,3),(9,12)]
    acoustic_chunks(wave,spans,cache,Hub(),"clip")
    acoustic_chunks(wave,spans,cache,Hub(),"clip")
    assert calls == [([0,1,2],0,3/RATE),([9,10,11],9/RATE,12/RATE)]
