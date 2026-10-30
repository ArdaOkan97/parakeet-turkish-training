"""Compare codec worker counts after the completed full screening boundary."""
import pathlib,json,functools
import whisper_tiny_gpu_screen_v11 as final
runner=final.runner
original_run_stream=runner.run_stream

def compare(engine,vad,rows,spec,guard,prefix,cutoff=.9,**kwargs):
    # Compile both VAD group shapes before comparing worker variants.
    for size in [64,128]:
        guard.check();silence=engine.np.zeros(320000,dtype=engine.np.float32);assert vad([silence]*size)==[[]]*size
    offset_rows=sorted(rows,key=lambda r:(int(r['duration_s']/2),r['shard'],spec['members'][r['shard'],r['recording_id']][0],r['start_s']))
    variants=[('workers4_group64',4,64,rows),('workers8_group64',8,64,rows),('workers8_group64_offset',8,64,offset_rows),('workers8_group128_offset',8,128,offset_rows)]
    results=[];ledgers=[]
    for name,n,group,ordered in variants:
        metrics=original_run_stream(engine,vad,ordered,spec,guard,prefix+'_'+name,cutoff,workers=n,group=group);metrics.update(variant=name,VAD_group=group,OS_cache_not_flushed=True,unique_clips_per_phase=len(rows));results.append(metrics);ledgers.append(runner.AREA/(prefix+'_'+name+'_ledger.jsonl'))
    baseline={x['segment_id']:x for x in map(json.loads,ledgers[0].read_text(encoding='utf-8').splitlines())};checks=[]
    for path in ledgers[1:]:
        candidate={x['segment_id']:x for x in map(json.loads,path.read_text(encoding='utf-8').splitlines())};assert baseline.keys()==candidate.keys();PCM=0;decision=0
        for sid in baseline:
            x=baseline[sid];y=candidate[sid]
            assert x.get('source_decode_quarantined',False)==y.get('source_decode_quarantined',False)
            if not x.get('source_decode_quarantined'):assert x['decode_proof']['PCM_f32le_sha256']==y['decode_proof']['PCM_f32le_sha256'];PCM+=1
            decision+=int(x['language_screening_decision']==y['language_screening_decision'])
        checks.append(dict(ledger=str(path),PCM_hash_matches=PCM,identical_screening_decisions=decision,total=len(baseline)))
    report=dict(results=results,checks=checks,all_CPU_children_closed=True,selection_requires_score_stability=True)
    runner.common.write('final_workers_v12_comparison.json',report);return report

if __name__=='__main__':
    runner.TinyEngine=final.ThresholdCheckedEngine;runner.analyze=final.annotated_analyze;runner.run_stream=compare;runner.__file__=__file__
    runner.main()
