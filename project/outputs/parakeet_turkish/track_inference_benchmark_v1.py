"""Publish the measured inference benchmark to the existing LOCAL project."""
import hashlib
import json
from pathlib import Path
from track_experiment import trackio

OUT=Path(__file__).resolve().parent
source=OUT/'asr_inference_throughput_v1.json'
report=json.loads(source.read_text(encoding='utf8'))
assert report['integrity_passed'] and report['required_cases_completed']
assert set(report['cases']) == {'1','4','8'}
trackio.init(project='parakeet-turkish',name='inference_selected1871_B1_B4_B8',
             group='Inference batch benchmarks',
             config={'benchmark_type':'inference','x_axis':'physical batch size',
                     'source_path':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                     'checkpoint_step':1871,'checkpoint_sha256':report['checkpoint_sha256'],
                     'utterances':16,'unpadded_audio_seconds':report['unpadded_audio_seconds_per_pass'],
                     'timing_scope':report['timing_scope'],'precision':report['precision'],
                     'quality_scope':'8TR+8EN fixed DEV clips; batch transcripts differ, no production-quality-equivalence claim'})
try:
    for size in (1,4,8):
        case=report['cases'][str(size)]
        trackio.log({'inference/audio_minutes_per_wall_minute':case['audio_minutes_per_wall_minute'],
                     'inference/speedup_vs_batch1':case.get('speedup_vs_b1',1),
                     'inference/median_wall_seconds':case['median_seconds_per_pass'],
                     'inference/peak_allocated_vram_gb':case['peak_allocated_bytes']/1e9,
                     'inference/peak_reserved_vram_gb':case['peak_reserved_bytes']/1e9,
                     'inference/changed_transcripts_vs_batch1':case.get('versus_b1',{}).get('changed_utterances',0),
                     'inference/tr_DEV_subset_wer_percent':case['development_scores']['tr']['wer']*100,
                     'inference/en_DEV_subset_wer_percent':case['development_scores']['en']['wer']*100},step=size)
finally:
    trackio.finish()
print(json.dumps({'published':True,'project':'parakeet-turkish','run':'inference_selected1871_B1_B4_B8','steps':[1,4,8]}))
