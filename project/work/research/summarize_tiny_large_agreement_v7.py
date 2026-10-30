"""Join completed cached tiny/large results without changing any screen flag."""
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path

STORE=Path(r'E:\gpu_training_scratchpad')
AREA=STORE/'analysis/CPU_validation_v7'
LARGE=STORE/'analysis/tiny_large_review_v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def readl(path):return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line]
def category(wer):return 'high_agreement' if wer<=.10 else 'moderate_agreement' if wer<=.30 else 'low_agreement_high_disagreement'


def main():
    old_path=AREA/'tiny_fullclip_window_sensitivity_summary_v7.json'
    old=json.loads(old_path.read_text())
    closed_path=LARGE/'actual_CLOSED0.json';closed=json.loads(closed_path.read_text())
    assert closed['actual_exit_code']==0 and closed['samples']==36
    ledger=LARGE/'large_v3_ledger.jsonl';assert sha(ledger)==closed['ledger_SHA']
    large={r['segment_id']:r for r in readl(ledger)};assert len(large)==36
    tiny={r['segment_id']:r for r in readl(AREA/'tiny_LID_per_clip_ledger_v7.jsonl')}
    random24=readl(AREA/'random24_full_large_v3_manifest_v7.jsonl')
    joined=[]
    for row in random24:
        sid=row['segment_id'];l=large[sid];t=tiny[sid]
        assert l['source_transcript']==row['transcript']==t['transcript']
        assert not l['accepted_TRAIN'] and not l['source_label_corrected']
        w=l['source_vs_large_v3']['spoken_diagnostic']['wer']
        joined.append({'segment_id':sid,'source_transcript':l['source_transcript'],
            'full_large_v3_transcript':l['large_v3_transcript'],'source_vs_full_large_v3':l['source_vs_large_v3'],
            'large_agreement_category':category(w),'tiny_original_screening_category':t['screening_category'],
            'tiny_full_clip':{k:v for k,v in t['language_observations'][0].items() if k!='raw_language_probabilities'},
            'tiny_window_only_alarm':t['language_observations'][0]['Turkish_70pass'] and not t['retains_language_screen_eligibility'],
            'source_or_detector_correctness_established':False,'flags_changed':False})
    window_alarms=[r for r in joined if r['tiny_window_only_alarm']]
    assert len(window_alarms)==2
    supplementary=[]
    for case in old['worst12_supplement_not_in_random_denominator']['confident_nontr_cases']:
        l=large[case['segment_id']]
        supplementary.append({**case,'full_large_v3_transcript':l['large_v3_transcript'],
                              'source_vs_full_large_v3':l['source_vs_large_v3'],
                              'large_agreement_category':category(l['source_vs_large_v3']['spoken_diagnostic']['wer'])})
    report={'kind':'cached_tiny_window_alarms_and_full_large_v3_agreement_v7',
        'completed_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'scope':'Random24 pinned before predictions; cached fullFP16 large-v3 beam5 outputs plus cached tiny203LID observations',
        'random64_tiny_screen_summary':{k:v for k,v in old['random64'].items() if k!='uncertain_under70_cases'},
        'random24_large_agreement_counts':dict(collections.Counter(r['large_agreement_category'] for r in joined)),
        'random24_tiny_window_only_alarm_cases_with_large':window_alarms,
        'random24_all_cached_joins':joined,'supplemental_confident_nontr_cases_with_large':supplementary,
        'interpretation':[
            'Both preselected random24 short-window alarms have high full-clip source/large agreement:spoken WER0.060606 and0.05. This is consistent with short-window sensitivity, but does not prove the tiny alarms false or establish audio truth.',
            'The supplemental Swedish-window alarm df2755...00075 has full-clip tinyP(TR)0.98150 and a mainly Turkish large transcript with spoken WER0.133333/CER0.014925. It needs listening; a confidence threshold can produce an alarm even with substantial full-clip Turkish agreement.',
            'For0f8748...00221, tiny predicts strong Arabic in0-8s and Turkish in8-16s, while large outputs only Arabic script. Source includes Latin transliteration plus Turkish. WER1.0 combines script/rendering and coverage differences and cannot by itself diagnose a bad source transcript.',
            'No threshold or actual tiny flag is changed by this comparison; requested70% screening remains recorded separately from transcript agreement.'],
        'limitations':old['limitations']+[
            'Large-v3 is another model output, not groundtruth. Agreement and disagreement require human listening for correctness judgments.',
            'Whole-clip large transcription does not validate the exact language of every short window; per-window large inference was not requested or run.',
            'High/moderate/low agreement uses the declared spoken-diagnostic WER bands<=.10,<=.30,>.30; these are not good/bad audio labels.'],
        'provenance':{'tiny_cached_summary':str(old_path),'tiny_cached_summary_SHA':sha(old_path),
            'large_closure':str(closed_path),'large_closure_SHA':sha(closed_path),'large_ledger':str(ledger),'large_ledger_SHA':sha(ledger),
            'ROOT_large_actual_session':closed['owned_session'],'ROOT_large_actual_PID':closed['actual_worker_PID'],
            'ROOT_large_actual_tool_chunk':closed['actual_tool_chunk'],'model_family':'full original Whisper-large-v3 FP16',
            'model_SHA':closed['report']['model_sha256'],'large_GPU_was_owned_exclusively_by_ROOT':True},
        'cached_only':True,'audio_read':False,'model_loaded':False,'original_tiny_flags_changed':0,
        'transcripts_corrected':0,'new_TRAIN_hours_accepted':0}
    path=AREA/'tiny_large_window_sensitivity_summary_v7.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(path),'random24_agreement_counts':report['random24_large_agreement_counts'],
                     'two_window_alarms':[{'segment_id':r['segment_id'],'spoken_WER':r['source_vs_full_large_v3']['spoken_diagnostic']['wer'],
                                          'CER':r['source_vs_full_large_v3']['spoken_diagnostic']['cer']} for r in window_alarms],
                     'flags_changed':0,'audio_read':False}))


if __name__=='__main__':main()
