"""50 user-patch regressions adapted to the real package. Four AST-only controller callback cases are superseded by actual Qt/DB integration tests in test_review_controller.py. No real hardware or neural weights are used here."""
from pathlib import Path
import math
import unittest
from emotion_assistant.evidence import distribution_diagnostics, js_divergence, PromptPolicy, SustainedEvidenceGate
from emotion_assistant.core import FusionEngine, InterventionFlow, interval_duration
from emotion_assistant.regulation import summarize_feedback, recommend_activities, evaluate_outcome
from emotion_assistant.models import ModelHub, _canonical, VISUAL_LABELS

ROOT=Path(__file__).resolve().parents[1]

def observation(modality='visual', start=0, end=4, sid='s', eid='v', scores=None, raw=None, quality=.9):
    return {'event_id':eid,'session_id':sid,'modality':modality,'t_start':start,'t_end':end,
            'scores':scores or {'positive':.02,'neutral':.06,'negative':.9,'uncertain':.02},
            'raw_scores':raw or {},'quality':quality,'valid':True,'model_version':'test-model-v1'}

def summary(start,end,eid=None,negative=.9,status='visual_only',sid='s'):
    item=observation(start=start,end=end,sid=sid,eid=eid or f'v{start}:{end}',
                     scores={'positive':.01,'neutral':.97-negative,'negative':negative,'uncertain':.02})
    item.update(modality='fusion' if status in ('joint','audio_only') else 'visual_summary',
                status=status,conflict=False)
    if status in ('joint','audio_only'):item['audio_event_id']=eid or f'a{start}:{end}'
    return item

def entry(eid='a',pre=8,post=2,**kwargs):
    return dict(id=eid,session_id='s',mode='real',activity='break',status='completed',
                pre=pre,post=post,helpfulness='helpful',**kwargs)

def stat(history,activity='break',**kwargs):
    return next(v for v in summarize_feedback(history,**kwargs)['per_activity'] if v['activity']==activity)


class DiagnosticsTests(unittest.TestCase):
    def test_uniform_raw_does_not_mean_negative_evidence(self):
        raw=dict.fromkeys(VISUAL_LABELS,1/8)
        self.assertEqual(_canonical(raw)['negative'],.625)
        self.assertTrue(distribution_diagnostics(raw)['near_uniform'])
    def test_near_uniform_false_high_negative(self):
        raw={k:.134 if k in ('anger','contempt','disgust','fear','sadness') else .11 for k in VISUAL_LABELS}
        self.assertAlmostEqual(_canonical(raw)['negative'],.67)
        self.assertTrue(distribution_diagnostics(raw)['near_uniform'])
    def test_peaked_raw_not_rejected(self):
        self.assertFalse(distribution_diagnostics({'happy':.9,'neutral':.1})['near_uniform'])
    def test_missing_raw_explicit(self):
        self.assertFalse(distribution_diagnostics({})['available'])
    def test_invalid_raw_explicit(self):
        self.assertFalse(distribution_diagnostics({'x':math.nan,'y':1})['available'])
    def test_js_identical(self):
        self.assertAlmostEqual(js_divergence({'x':.2,'y':.8},{'x':.2,'y':.8}),0)
    def test_js_disjoint(self):
        self.assertAlmostEqual(js_divergence({'x':1},{'y':1}),1)
    def test_js_invalid(self):
        with self.assertRaises(ValueError):js_divergence({'x':-1},{'x':1})

class FusionTests(unittest.TestCase):
    def test_uniform_visual_rejected(self):
        f=FusionEngine();f.add_visual(observation(raw=dict.fromkeys(VISUAL_LABELS,1/8)))
        self.assertFalse(f.visual_summary(0,4)['valid'])
    def test_neutral_negative_conflict(self):
        f=FusionEngine();f.add_visual(observation(scores={'neutral':.9,'negative':.08,'positive':.01,'uncertain':.01}))
        out=f.fuse_audio(observation('audio',eid='a'))
        self.assertTrue(out['conflict']);self.assertGreater(out['disagreement_js'],.18)
    def test_aligned_agreement(self):
        f=FusionEngine();f.add_visual(observation());out=f.fuse_audio(observation('audio',eid='a'))
        self.assertEqual(out['status'],'joint');self.assertFalse(out['conflict'])
    def test_session_isolation(self):
        f=FusionEngine();f.add_visual(observation());out=f.fuse_audio(observation('audio',sid='other',eid='a'))
        self.assertEqual(out['status'],'audio_only')
    def test_duplicate_not_double_duration(self):
        f=FusionEngine();f.add_visual(observation());f.add_visual(observation())
        self.assertEqual(f.visual_summary(0,4)['covered_seconds'],4)
    def test_overlap_not_double_duration(self):
        f=FusionEngine();f.add_visual(observation(end=3));f.add_visual(observation(start=2,end=4,eid='v2'))
        self.assertEqual(f.visual_summary(0,4)['covered_seconds'],4)
    def test_missing_audio_not_neutral(self):
        f=FusionEngine();f.add_visual(observation());a=observation('audio',eid='a');a.update(valid=False,scores=None)
        self.assertEqual(f.fuse_audio(a)['status'],'visual_only')
    def test_versions_retained(self):
        f=FusionEngine();f.add_visual(observation());out=f.fuse_audio(observation('audio',eid='a'))
        self.assertEqual(set(out['model_versions']),{'visual','audio'})

class PromptTests(unittest.TestCase):
    def gate(self,**kwargs):return SustainedEvidenceGate(PromptPolicy(settle_seconds=0,**kwargs))
    def test_sparse_bursts_do_not_trigger(self):
        gate=self.gate()
        for i in range(11):out=gate.update(summary(i*2,i*2+.5),now=i*2+.5)
        self.assertAlmostEqual(out['negative_seconds'],5.5);self.assertFalse(out['should_prompt'])
    def test_continuous_twenty_seconds_triggers(self):
        gate=self.gate()
        for i in range(40):out=gate.update(summary(i*.5,i*.5+.5),now=i*.5+.5)
        self.assertTrue(out['should_prompt']);self.assertAlmostEqual(out['negative_seconds'],20)
    def test_settle_delay_waits_for_late_modalities(self):
        gate=SustainedEvidenceGate()
        for i in range(40):out=gate.update(summary(i*.5,i*.5+.5),now=i*.5+.5)
        self.assertFalse(out['should_prompt'])
        for i in range(40,46):out=gate.update(summary(i*.5,i*.5+.5),now=i*.5+.5)
        self.assertTrue(out['should_prompt'])
    def test_same_event_never_counts_twice(self):
        gate=self.gate()
        for i in range(10):out=gate.update(summary(0,4,eid='same'),now=4)
        self.assertEqual(out['negative_seconds'],4);self.assertFalse(out['should_prompt'])
    def test_neutral_resets_chain(self):
        gate=self.gate();gate.update(summary(0,15),15)
        gate.update(summary(15,16,negative=.1),16)
        out=gate.update(summary(16,25),25)
        self.assertEqual(out['negative_seconds'],9);self.assertFalse(out['should_prompt'])
    def test_late_joint_contradiction_revises_overlap(self):
        gate=self.gate();gate.update(summary(0,19),19)
        gate.update(summary(4,17,eid='audio',negative=.1,status='joint'),19)
        out=gate.update(summary(19,25),25)
        self.assertEqual(out['negative_seconds'],8);self.assertFalse(out['should_prompt'])
    def test_stale_result_cannot_prompt(self):
        gate=self.gate();out=gate.update(summary(0,30),now=60)
        self.assertFalse(out['should_prompt'])
    def test_future_interval_rejected(self):
        self.assertFalse(self.gate().update(summary(0,30),now=10)['should_prompt'])
    def test_session_change_resets_history(self):
        gate=self.gate();gate.update(summary(0,19),19)
        out=gate.update(summary(19,21,sid='different'),21)
        self.assertEqual(out['negative_seconds'],2)
    def test_empty_session_rejected(self):
        self.assertEqual(self.gate().update(summary(0,1,sid=''),1)['reason'],'missing_session_id')
    def test_invalid_policy(self):
        with self.assertRaises(ValueError):PromptPolicy(settle_seconds=float('nan'))
    def test_union_reference(self):
        self.assertEqual(interval_duration([(0,2),(1,3),(4,5)]),4)

class FeedbackTests(unittest.TestCase):
    def test_disjoint_immediate_delayed_count_union(self):
        a=entry();b=entry('b',pre=5,post=None,followup={'status':'completed','distress':5})
        s=stat([a,b]);self.assertEqual(s['feedback_count'],2);self.assertEqual(s['feedback_mean_delta'],3)
    def test_same_activity_two_ratings_one_count(self):
        s=stat([entry(followup={'status':'completed','distress':5})])
        self.assertEqual(s['feedback_count'],1);self.assertAlmostEqual(s['feedback_mean_delta'],5.1)
    def test_missing_mode_not_real(self):
        record=entry();record.pop('mode');self.assertEqual(summarize_feedback([record])['total_count'],0)
    def test_simulation_not_real(self):
        record=entry();record['mode']='simulation';self.assertEqual(summarize_feedback([record])['total_count'],0)
    def test_replay_not_real(self):
        record=entry();record['mode']='replay';self.assertEqual(summarize_feedback([record])['total_count'],0)
    def test_explicit_helpfulness_without_numeric_post(self):
        s=stat([entry(post=None)]);self.assertEqual(s['subjective_count'],1)
        self.assertEqual(s['helpful_rate'],1);self.assertEqual(s['ranking_subjective_count'],1)
    def test_chain_not_attributed_to_last_method(self):
        a=entry();a['status']='cancelled'
        b=entry('b',root_intervention_id='a',previous_intervention_id='a')
        s=stat([a,b]);self.assertEqual(s['feedback_count'],0);self.assertEqual(s['excluded_chain_count'],1)
    def test_newer_revision_wins(self):
        old=entry(updated_at='2026-01-01T00:00:00+00:00')
        new=entry(updated_at='2026-01-02T00:00:00+00:00');new['status']='cancelled'
        self.assertEqual(summarize_feedback([old,new])['completed_count'],0)
    def test_discomfort_blocks_across_contexts(self):
        h=[entry(context={'need':'fatigue'},discomfort=True)]
        item=next(x for x in recommend_activities(h,need='tense') if x['activity']=='break')
        self.assertTrue(item['blocked'])
    def test_same_need_preferred_for_feedback(self):
        h=[entry(context={'need':'tense'})]+[entry(str(i),context={'need':'fatigue'}) for i in range(3)]
        item=next(x for x in recommend_activities(h,need='tense') if x['activity']=='break')
        self.assertEqual(item['personalization_scope'],'same_need');self.assertEqual(item['rank_feedback_count'],1)
    def test_discomfort_overrides_flow_evaluation(self):
        flow=InterventionFlow();flow.request();flow.confirm(True);flow.start('break',8,60)
        record=flow.complete(2,'worse')
        self.assertEqual(record['evaluation'],'discomfort');self.assertEqual(record['numeric_evaluation'],'improved')
    def test_process_discomfort_remains_visible(self):
        flow=InterventionFlow();flow.request();flow.confirm(True);flow.start('break',8,60)
        flow.checkin('uncomfortable');self.assertEqual(flow.complete(2,'helpful')['evaluation'],'discomfort')
    def test_cancel_preserves_discomfort_priority(self):
        flow=InterventionFlow();flow.request();flow.confirm(True);flow.start('break',8,60)
        result=flow.cancel(reason='uncomfortable')
        self.assertEqual(result['evaluation'],'discomfort');self.assertIsNone(result['delta'])
    def test_malformed_legacy_context_does_not_crash(self):
        result=recommend_activities([entry(context='legacy-text')],need='tense')
        self.assertEqual(len(result),8)
    def test_numeric_drop_and_helpfulness_separate(self):
        self.assertEqual(evaluate_outcome(8,2,'no_change')['state'],'improved')
    def test_invalid_rating_does_not_invent_improvement(self):
        self.assertEqual(evaluate_outcome(True,2,'unknown')['state'],'insufficient')


try:
    import numpy as np
except ImportError:
    np=None

@unittest.skipIf(np is None,'NumPy not installed; model adapter mock tests skipped')
class AudioAdapterTests(unittest.TestCase):
    def hub(self):return ModelHub(ROOT/'no_weights_needed')
    def test_acoustic_api_never_calls_asr(self):
        hub=self.hub()
        class Emotion:
            def generate(self,**kwargs):return [{'labels':['happy','neutral','sad'],'scores':[.8,.1,.1]}]
        class ASR:
            def generate(self,**kwargs):raise AssertionError('ASR entered the acoustic critical path')
        hub._audio_model,hub._asr_model=Emotion(),ASR()
        result=hub.analyze_audio_emotion(np.full(32000,.03,dtype=np.float32),'s',0,2)
        self.assertTrue(result['valid'])
    def test_no_consent_never_calls_asr(self):
        hub=self.hub()
        class ASR:
            def generate(self,**kwargs):raise AssertionError('consent violation')
        hub._asr_model=ASR()
        self.assertEqual(hub.transcribe_audio(np.ones(32000),'s',0,2,False)['reason'],'consent_not_granted')
    def test_asr_tags_not_extra_emotion_votes(self):
        hub=self.hub()
        class ASR:
            def generate(self,**kwargs):return [{'text':'<|zh|><|HAPPY|>测试内容'}]
        hub._asr_model=ASR();out=hub.transcribe_audio(np.full(32000,.03),'s',0,2,True)
        self.assertTrue(out['valid']);self.assertEqual(out['text'],'测试内容');self.assertNotIn('scores',out)
    def test_asr_rejects_silence(self):
        hub=self.hub();hub._asr_model=object()
        self.assertFalse(hub.transcribe_audio(np.zeros(32000),'s',0,2,True)['valid'])
    def test_acoustic_rejects_duration_mismatch(self):
        self.assertEqual(self.hub().analyze_audio_emotion(np.full(32000,.03),'s',0,5)['reason'],'audio_timestamp_length_mismatch')
    def test_legacy_signature_still_returns_pair(self):
        result=self.hub().analyze_audio(np.full(32000,.03),'s',0,2,False)
        self.assertEqual(len(result),2);self.assertIsNone(result[1]);self.assertIn('wrapper_latency_ms',result[0])

if __name__=='__main__':unittest.main(verbosity=2)
