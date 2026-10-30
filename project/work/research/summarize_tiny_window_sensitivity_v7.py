"""Cached-only review of completed language screening; no audio/model imports."""
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

STORE=Path(r'E:\gpu_training_scratchpad')
AREA=STORE/'analysis/CPU_validation_v7'
sys.path.insert(0,str(STORE/'scratch/CPU_validation_v2_env/Lib/site-packages'))
import psutil


def readl(path):return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    ledger=AREA/'tiny_LID_per_clip_ledger_v7.jsonl'
    rows=readl(ledger);assert len(rows)==76
    metadata={r['segment_id']:r for r in readl(AREA/'random64_tiny_language_manifest_v7.jsonl')}
    random24_ids={r['segment_id'] for r in readl(AREA/'random24_full_large_v3_manifest_v7.jsonl')}
    random64=[r for r in rows if r['random64_member']]
    supplemental=[r for r in rows if not r['random64_member']]
    assert len(random64)==64 and len(supplemental)==12
    observations=[o for r in rows for o in r['language_observations']]
    for o in observations:
        probs=list(o['raw_language_probabilities'].values())
        assert all(0<=p<=1 for p in probs) and abs(sum(probs)-1)<1e-4

    def full_pass(row):return row['language_observations'][0]['Turkish_70pass']
    def display(row):
        observed=row['language_observations'];full=observed[0]
        fields=('scope','start_s_in_clip','end_s_in_clip','detected_language','detected_language_probability',
                'Turkish_probability','energy_speech_candidate_fraction','low_energy_window_sensitivity',
                'confident_nontr','uncertain_Turkish_below70')
        def compact(value):return {k:value[k] for k in fields}
        return {'segment_id':row['segment_id'],'recording_id':row['recording_id'],'shard':row['shard'],
                'start_s':row['start_s'],'end_s':row['end_s'],'duration_s':row['duration_s'],
                'source_transcript':row['transcript'],'QC_group':row['QC_group'],
                'preselected_random24':row['segment_id'] in random24_ids,
                'full_clip_only_TR70_pass':full_pass(row),'strict_full_and_windows_TR70_pass':row['retains_language_screen_eligibility'],
                'flagged_only_because_window_failed':full_pass(row) and not row['retains_language_screen_eligibility'],
                'screening_category':row['screening_category'],'full_clip':compact(full),
                'failing_windows':[compact(o) for o in observed[1:] if not o['Turkish_70pass']],
                'all_window_top_languages':[o['detected_language'] for o in observed[1:]],
                'source_metadata_LID_if_random':metadata.get(row['segment_id'],{}).get('lid'),
                'model_probabilities_are_uncalibrated':True,'source_correctness_established':False}

    uncertain=[display(r) for r in random64 if r['screening_category']=='quarantine_uncertain_under70']
    supplemental_nontr=[display(r) for r in supplemental if r['any_confident_nontr']]
    selected_review=[r for r in random64 if r['segment_id'] in random24_ids]
    counts=lambda values:dict(collections.Counter(r['screening_category'] for r in values))
    report={'kind':'cached_tiny_fullclip_vs_window_sensitivity_v7','completed_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'scope':'Completed64random+12separateworst; cached probabilities/text only; no new audio/inference',
            'random64':{'declared':64,'full_clip_only_TR70_pass':sum(full_pass(r) for r in random64),
                        'full_clip_only_does_not_pass':sum(not full_pass(r) for r in random64),
                        'strict_full_plus_windows_TR70_pass':sum(r['retains_language_screen_eligibility'] for r in random64),
                        'additional_window_only_quarantines':sum(full_pass(r) and not r['retains_language_screen_eligibility'] for r in random64),
                        'screen_counts':counts(random64),'uncertain_under70_cases':uncertain},
            'random24_preselected_before_LID':{'clips':len(selected_review),'screen_counts':counts(selected_review),
                        'all24_preserved_for_large_regardless_of_LID':True,
                        'uncertain_ids':[r['segment_id'] for r in selected_review if not r['retains_language_screen_eligibility']]},
            'worst12_supplement_not_in_random_denominator':{'screen_counts':counts(supplemental),'confident_nontr_cases':supplemental_nontr,
                        'uncertain_cases':[display(r) for r in supplemental if r['screening_category']=='quarantine_uncertain_under70']},
            'all76':{'full_clip_observations':76,'short_disjoint_window_observations':len(observations)-76,
                     'probability_vectors_checked_for_finite_range_and_sum1':len(observations),
                     'own_RSS_max_at_sample_boundaries':max(r['own_RSS_bytes'] for r in rows)},
            'limitations':['The70% threshold is the requested screening rule; probabilities are not calibrated correctness guarantees.',
                           'A full-clip pass with a short-window failure is additional sensitivity, not proof the source is wrong or that the window is non-Turkish.',
                           'Whisper pads short language inputs; short duration, names, noise, silence and code-switching can change confidence. No human-listening judgment has been assigned here.',
                           'For clips>=8s, windows are disjoint4s blocks with remaining tails>=2s. Full clip always tested; very short switches and skipped<2s tails can be missed.',
                           'Energy fractions are heuristic speech-candidate indicators, not VAD or forced-alignment certification.',
                           'Random64 is stratified across11shards and3duration bins within the declared bounded-member population; it is not a simple random sample of all raw audio.',
                           'The supplemental12 were deliberately selected from historical small-model worst cases and cannot estimate prevalence.',
                           'No source transcript was corrected, no clip admitted to TRAIN, and no heldout/rights/audio-dedup barrier was removed.'],
            'inputs':{'ledger':str(ledger),'ledger_SHA':sha(ledger),
                      'random64_manifest_SHA':sha(AREA/'random64_tiny_language_manifest_v7.jsonl'),
                      'random24_manifest_SHA':sha(AREA/'random24_full_large_v3_manifest_v7.jsonl')},
            'CPU_only':True,'audio_read':False,'model_loaded':False,'accepted_TRAIN_hours':0}
    destination=AREA/'tiny_fullclip_window_sensitivity_summary_v7.json'
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    examples=uncertain+supplemental_nontr
    (AREA/'tiny_language_review_examples_v7.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in examples),encoding='utf-8')
    print(json.dumps({'report':str(destination),'random64':{k:v for k,v in report['random64'].items() if k!='uncertain_under70_cases'},
                      'random24':report['random24_preselected_before_LID'],
                      'supplemental_counts':counts(supplemental),'all76':report['all76'],
                      '8random_uncertain_ids':[r['segment_id'] for r in uncertain],
                      'supplemental_nontr_ids':[r['segment_id'] for r in supplemental_nontr]},ensure_ascii=False))


if __name__=='__main__':main()
