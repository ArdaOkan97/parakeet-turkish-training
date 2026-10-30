"""Validate a closed screen and derive a configurable language-only manifest.

Streams scores; does not run ASR, alter transcripts, or admit training data.
"""
import pathlib,json,hashlib,argparse,collections,datetime,math
AREA=pathlib.Path(r'E:\gpu_training_scratchpad\analysis\whisper_tiny_GPU_batch_v1')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ledger',default='staged_full_v11_duration_ordered_ledger.jsonl');ap.add_argument('--prefix',default='screen90_v11');ap.add_argument('--cutoff',type=float,default=.9);args=ap.parse_args();assert 0<args.cutoff<=1
    ledger=(AREA/args.ledger).resolve();assert ledger.parent==AREA and '/' not in args.prefix and '\\' not in args.prefix
    policy=json.loads((AREA/'final_screening_policy_v11.json').read_text());langs=policy['raw_probability_language_order'];tr=langs.index('tr')
    source=AREA/'staged_bounded_manifest.jsonl';expected={r['segment_id']:r for r in map(json.loads,source.read_text(encoding='utf-8').splitlines())};seen=set();counts=collections.Counter();hours=collections.Counter();byshard={};windows=0;rescored=0;changed=0;voiced=0.;speech=collections.Counter();digest=hashlib.sha256();readbytes=0;quarantines=[]
    outputs={name:AREA/(args.prefix+'_'+name+'.jsonl') for name in ['language_screened_candidates','deferred_candidates']}
    with outputs['language_screened_candidates'].open('x',encoding='utf-8') as accepted,outputs['deferred_candidates'].open('x',encoding='utf-8') as deferred,ledger.open('rb') as f:
        for raw in f:
            digest.update(raw);readbytes+=len(raw);r=json.loads(raw);sid=r['segment_id'];assert sid in expected and sid not in seen;seen.add(sid);src=expected[sid];assert r['new_TRAIN_admitted'] is False
            assert r['recording_id']==src['recording_id'] and r['shard']==src['shard']
            if r.get('source_decode_quarantined'):
                decision='source_decode_quarantine';quarantines.append({'segment_id':sid,'reason':r['decode_proof']['reason']})
            else:
                assert r['decode_proof']['timestamp_coverage_fraction']>=.995 and len(r['decode_proof']['PCM_f32le_sha256'])==64
                assert abs(r['duration_s']-src['duration_s'])<1/16000+.00001
                assert r['start_s']==src['start_s']
                prev=0;total=0
                for a,b in r['speech_ranges_samples']:
                    assert 0<=prev<=a<b<=round(r['duration_s']*16000);total+=b-a;prev=b
                assert abs(r['speech_seconds']-total/16000)<1e-6
                bucket=collections.Counter();window_speech=0.;previous_end=0.
                for w in r['windows']:
                    assert previous_end<=w['start_s']<w['end_s']<=r['duration_s']+1e-7 and 0<w['speech_s']<=w['end_s']-w['start_s']+1e-7;previous_end=w['end_s'];window_speech+=w['speech_s']
                    p=w['language_probabilities'];assert len(p)==len(langs) and all(math.isfinite(x) and 0<=x<=1 for x in p) and abs(sum(p)-1)<.00001
                    top=max(range(len(p)),key=p.__getitem__);assert langs[top]==w['top_language'] and p[tr]==w['Turkish_probability']
                    cat=('tr' if top==tr else 'other') if p[top]>=.7 and w['speech_s']>=.25 else 'uncertain';assert cat==w['category'];bucket[cat]+=w['speech_s'];windows+=1
                    rescored+=int(w['score_path']=='canonical_HF_FP32_borderline_rescore');changed+=int(w.get('confidence70_category_changed',False))
                assert abs(window_speech-r['speech_seconds'])<1e-6
                lo=r['estimated_Turkish_share_lower'];hi=r['estimated_Turkish_share_upper'];n=r['speech_seconds'];voiced+=n;speech.update(bucket)
                if n:
                    assert abs(lo-bucket['tr']/n)<1e-7 and abs(hi-(bucket['tr']+bucket['uncertain'])/n)<1e-7
                    decision='pass' if lo>=args.cutoff else 'below_cutoff' if hi<args.cutoff else 'uncertain_review'
                else:assert lo is None and hi is None;decision='no_detected_speech'
            counts[decision]+=1;hours[decision]+=src['duration_s']/3600;shard=src['shard'];byshard.setdefault(shard,collections.Counter());byshard[shard][decision]+=1
            out=dict(src,language_screening_decision=decision,estimated_Turkish_share_lower=r.get('estimated_Turkish_share_lower'),estimated_Turkish_share_upper=r.get('estimated_Turkish_share_upper'),cutoff=args.cutoff,new_TRAIN_admitted=False)
            (accepted if decision=='pass' else deferred).write(json.dumps(out,ensure_ascii=False,separators=(',',':'))+'\n')
    assert seen==expected.keys(),(len(seen),len(expected));assert sum(counts.values())==len(expected)
    mechanical=json.loads((AREA/'staged_manifest_receipt.json').read_text())
    summary=dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),kind='Validated complete language-only candidate screen',cutoff=args.cutoff,unique_processed_clips=len(seen),counts=dict(counts),clip_duration_hours_by_decision=dict(hours),processed_clip_duration_hours=sum(hours.values()),by_shard={k:dict(v) for k,v in byshard.items()},speech_seconds=voiced,estimated_speech_seconds_by_language_bucket=dict(speech),windows=windows,canonical_FP32_rescored_windows=rescored,FP32_confidence70_category_changes=changed,ledger_path=str(ledger),ledger_bytes=readbytes,ledger_SHA256=digest.hexdigest(),outputs={k:str(v) for k,v in outputs.items()},mechanical_input_accounting=mechanical,source_decode_quarantines=quarantines,validated=['exact pinned ID coverage,no duplicates','timestamp PCM coverage and hashes','duration fractions and VAD/window accounting','full raw probability-vector dimensions,sum,top-language categories','no new TRAIN admission'],limits=['Tiny language labels are estimates,not human ground truth','4s windows can miss language switches within a window','Model confidence is not a calibrated Turkish-duration probability','Language eligibility alone does not verify transcripts,rights,deduplication,or heldout separation'],new_TRAIN_admitted=0)
    p=AREA/(args.prefix+'_summary.json');assert not p.exists();p.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:summary[k] for k in ['unique_processed_clips','counts','clip_duration_hours_by_decision','windows','canonical_FP32_rescored_windows','FP32_confidence70_category_changes','ledger_SHA256']},ensure_ascii=False))

if __name__=='__main__':main()
