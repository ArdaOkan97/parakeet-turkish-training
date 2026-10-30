from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUTPUT=ROOT/'outputs/parakeet_turkish'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def ref(path):return {'path':str(Path(path).resolve()),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}
loop=OUTPUT/'runs/adaptive_20261010_continuation_03';status=read(loop/'loop_status.json');trial=status['trials'][0]
observations=trial['watchdog_observations'];trigger=[o for o in observations if o['system_memory']['commit_fraction']>=.95][0]
index=observations.index(trigger);memory=read(trial['memory_guard_provenance_path']);cache=read(trial['cache_provenance_path'])
run=OUTPUT/'runs'/trial['run_id'];selection=read(run/'selection.json');history=read(run/'training_history.json')
before=history[1820];after=history[-1];audio=after['audio_seen_seconds']-before['audio_seen_seconds'];wall=after['train_wall_seconds']-before['train_wall_seconds']
paired=read(OUTPUT/'paired_windows_commit_observation_v1.json')
report={'schema':1,'kind':'continuation03_resource_stop_api_scope_review_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
 'loop':ref(loop/'loop_status.json'),'selection':ref(run/'selection.json'),'history':ref(run/'training_history.json'),
 'memory_provenance':ref(trial['memory_guard_provenance_path']),'cache_provenance':ref(trial['cache_provenance_path']),
 'actual_status':status['status'],'actual_worker_exit':trial['exit_code'],'initial_step':1821,'completed_step':selection['completed_steps'],
 'selected_step':selection['selected_step'],'selected_DEV_wer':selection['turkish_validation_wer'],'selected_accepted':selection['accepted'],
 'actual_stop_reason':status['resource_stop_reason'],'stop_requested_at_utc':trial['cooperative_stop_requested_at_utc'],
 'trigger':trigger,'near_trigger_observations':observations[max(0,index-2):index+3],
 'worker_identity':{'actual_worker_pid':trial['actual_worker_pid'],'launcher_pid':trial['launcher_pid'],
   'identity_file':ref(trial['worker_identity_path']),'private_scope':'Actual interpreter; identity/run/arguments and owned-parent creation validated'},
 'new_training':{'updates':selection['completed_steps']-1821,'audio_seen_seconds':audio,'update_train_wall_seconds':wall,
   'audio_min_per_update_wall_min':audio/wall,'cache_eligible_calls':cache['cache_reclamation']['eligible_update_calls'],
   'cache_reclamations':cache['cache_reclamation']['reclamations'],'cache_failed_reclamations':cache['cache_reclamation']['failed_reclamations'],
   'allocator_cap_bytes':memory['cap_bytes'],'cleanup_threshold_bytes':memory['cleanup_threshold_bytes']},
 'paired_current_host_observation':ref(OUTPUT/'paired_windows_commit_observation_v1.json'),'paired_summary':paired['summary'],
 'primary_sources':paired['primary_sources'],
 'documented_semantics':{'GlobalMemoryStatusEx_ullTotalPageFile':'Smaller system/current-process commit limit',
   'GlobalMemoryStatusEx_ullAvailPageFile':'Current-process available commit, equal or lower than system available commit',
   'GetPerformanceInfo_CommitTotal':'System-wide committed pages','GetPerformanceInfo_CommitLimit':'System-wide current commit limit pages; multiply both byPageSize'},
 'conclusions':['Existing helper incorrectly labeled a possibly process-constrained derived fraction as system commit.',
   'Current8 paired readings agree exactly; the historical03 GetPerformanceInfo system reading was not recorded, so a false historical stop is unproven.',
   'Actual worker private bytes are real process commit size, not a measurement of paging traffic or proof of a particular job quota.',
   'Use new versioned GetPerformanceInfo system guard plus separately attributed fresh actual-worker self process headroom; do not reinterpret controller caller quota as worker quota.',
   'Keep emergency95%/2GB and sustained90% rules. Preserve1871 best and root coordinate a fresh data-pool warmstart only after next runtime/data preflight.'],
 'new_guard':ref(OUTPUT/'resource_watchdog_v2.py'),'new_guard_CPU_verification':ref(OUTPUT/'resource_watchdog_v2_cpu_verification.json'),
 'CPU_only':True,'frozen_sources_modified':False,'GPU_launched':False}
target=OUTPUT/'continuation03_resource_api_review_v1.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'saved':str(target),'trigger_fraction':trigger['system_memory']['commit_fraction'],'worker_private_bytes':trigger['worker_private_memory']['private_bytes'],'new_audio_min_per_wall_min':audio/wall}))
