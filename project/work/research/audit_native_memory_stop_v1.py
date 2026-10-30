"""Small read-only audit of the first native training memory stop."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/parakeet_turkish'
LOOP=OUT/'runs/adaptive_20261011_native_data_01'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
state=read(LOOP/'loop_status.json');trial=state['trials'][0]
observations=LOOP/(trial['run_id']+'_memory_observations.jsonl')
rows=[json.loads(line) for line in observations.read_text(encoding='utf-8').splitlines()]
hits=[r for r in rows if r['worker_self'].get('available') and r['worker_self']['sample']['calling_process_commit_memory']['ullAvailPageFile_bytes']<2e9]
assert len(hits)==1
hit=hits[0];self_sample=hit['worker_self']['sample'];glob=self_sample['system_memory'];own=self_sample['calling_process_commit_memory']
assert glob['commit_available_bytes']==own['ullAvailPageFile_bytes']
assert glob['commit_fraction']>=.95 and glob['commit_available_bytes']<2e9
directory=OUT/'runs'/trial['run_id'];history=read(directory/'training_history.json');selection=read(directory/'selection.json')
cache=read(Path(trial['cache_provenance_path']))['cache_reclamation']
assert selection['completed_steps']==len(history)==63 and selection['selected_step']==0
assert not history[-1]['validation']['retention_gate_passed']
assert cache['eligible_update_calls']==63 and cache['failed_reclamations']==0
record={
 'kind':'native_training_memory_stop_independent_review_v1','CPU_only':True,'passed':True,
 'run_id':trial['run_id'],'stop_reason':trial['cooperative_stop_reason'],
 'source_sha256':{str(p):sha(p) for p in (LOOP/'loop_status.json',observations,directory/'training_history.json',directory/'selection.json',Path(trial['cache_provenance_path']))},
 'trigger':{'controller_requested_at_utc':trial['cooperative_stop_requested_at_utc'],
  'self_sample_at_utc':self_sample['observed_at_utc'],'phase':self_sample['phase'],'worker_PID':own['pid'],
  'global_commit_used_bytes':glob['commit_used_bytes'],'global_commit_limit_bytes':glob['commit_limit_bytes'],
  'global_commit_fraction':glob['commit_fraction'],'global_headroom_bytes':glob['commit_available_bytes'],
  'worker_private_bytes':self_sample['worker_private_memory']['private_bytes'],
  'self_process_headroom_bytes':own['ullAvailPageFile_bytes'],
  'controller_one_second_later':hit['memory']['system_memory'],
  'genuine_short_global_commit_spike':True,'per_process_quota_misattribution':False},
 'cache':{'eligible_updates':cache['eligible_update_calls'],'releases':cache['reclamations'],
  'cleanup_seconds':cache['cleanup_seconds_in_training_timing'],
  'largest_reserved_before_release':max(cache['events'],key=lambda e:e['before']['reserved_bytes']),
  'update62':next(e for e in cache['events'] if e['eligible_update_call']==62),
  'intra_update_peak_can_precede_optimizer_end_release':True},
 'quality':{'completed_step':63,'selected_step':0,'selected_WER':selection['turkish_validation_wer'],
  'latest_validation':history[-1]['validation'],'latest_not_selected_quality_winner':True},
 'rates':{'total_audio_seconds':selection['audio_seen_seconds'],'TR_audio_seconds':history[-1]['turkish_audio_seen_seconds'],
  'update_only_wall_seconds':selection['train_wall_seconds'],
  'combined_audio_minutes_per_wall_minute':selection['audio_seen_seconds']/selection['train_wall_seconds'],
  'TR_audio_minutes_per_wall_minute':history[-1]['turkish_audio_seen_seconds']/selection['train_wall_seconds'],
  'peak_allocated_GPU_GB':history[-1]['peak_allocated_vram_gb']},
 'next_plan':{'hypothesis_not_yet_validated':'6GB allocator cap and3GB optimizer-end cache threshold reduce transient WDDM backing commitment',
  'resume_latest_step':63,'retain_qualifying_best_step':0,'retain_official_native_loss':True,
  'same_model_data_optimizer_sampler_RNG_if_exact_resume':True,
  'no_TRAIN_row_drops_or_crop':True,'critical95percent_and2GB_guards_retained':True,
  'required_before_launch':'Root owned exit receipt, immutable state provenance and bounded actual matched long-row feasibility proof; OOM remains an honest failure requiring review'}
}
target=OUT/'native_training_memory_stop_independent_review_v1.json'
with target.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2,ensure_ascii=False)
print(json.dumps({'passed':True,'report':str(target),'genuine_global_spike':True,'completed':63,'selected':0}))
