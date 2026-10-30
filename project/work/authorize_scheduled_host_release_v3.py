from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, os, sys

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(HERE))
from resource_watchdog_v2 import sample_memory
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,v:Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
state_path=HERE/'experiment_queue_state.json'
state=read(state_path)
assert state['queue_id']=='20261012_row_schedule_memory_02' and state['status']=='repair_required'
assert state['owned_controller_closure_observed'] is True
assert state['last_actual_waitpid']['exit_code']==1 and state['owned_child_pid'] is None
loop=Path(state['current_controller_loop'])
status=read(loop/'loop_status.json')
assert status['status']=='resource_review_required' and status['child_exit_code']==0
assert status['completed_steps']==7009
native=Path(status['paths']['native']).with_suffix('.completion.json')
completion=read(native)
assert completion['status']=='complete' and completion['sources_unchanged'] is True and not completion.get('error')
controller=HERE/'run_TSC_bpe_row_schedule_memory_v3.py'
approval_path=HERE/'TSC_BPE_row_schedule_startup_host_release_ROOT_approval_v3.json'
assert not approval_path.exists()
memory=sample_memory(os.getpid(),'ROOT_changed_startup_host_release_preflight')['system_memory']
assert memory['available'] is True
projected=memory['commit_used_bytes']+12_176_424_960
assert projected/memory['commit_limit_bytes']<.95 and memory['commit_limit_bytes']-projected>2_000_000_000
approval={'kind':'ROOT_actual_CLOSED_resource_stop_startup_host_release_approval_v3','passed':True,
 'created_at_utc':datetime.now(timezone.utc).isoformat(),'root_owned_queue_session_id':29031,
 'root_owned_queue_exit_code':1,'root_owned_exit_tool_chunk':'37f468',
 'root_owned_exit_observed_at_utc':'2026-10-12T04:53:44Z',
 'old_actual_updates':9,'old_child_exit_code':0,'native_complete_and_sources_unchanged':True,
 'native_completion_sha256':sha(native),'allow_next_startup_host_release':True,
 'next_controller_sha256':sha(controller),'end_cache_threshold_bytes':3_000_000_000,
 'cap_bytes':8_000_000_000,'unchanged_guard_thresholds':True,
 'guards':{'moderate_fraction':.9,'consecutive_fresh':3,'critical_fraction':.95,'headroom_bytes':2_000_000_000},
 'fresh_baseline_sample':memory,'historical_peak_projection':projected/memory['commit_limit_bytes'],
 'scope':'Fresh changed storage-lifetime policy; equivalent row values/labels and live full state required. Actual relief is measured in smoke; no unchanged retry or no-recurrence claim. Champion7000 preserved.'}
write(approval_path,approval)
archive=ROOT/'work/experiment_queues/20261012_row_schedule_memory_02/closed_state_after_resource_v3.json'
assert state_path.resolve().is_relative_to(ROOT) and archive.resolve().is_relative_to(ROOT) and not archive.exists()
os.replace(state_path,archive)
old=read(HERE/'TSC_BPE_row_schedule_memory_queue_manifest_v2.json')
old.update(kind='ROOT_authorized_TSC_row_schedule_memory_queue_v3',queue_id='20261012_row_schedule_memory_03',trial_namespace='memory_03')
old['predecessor_closed_state']={'path':str(archive),'sha256':sha(archive),
 'ROOT_owned_exec_closed_exit_code':1,'ROOT_owned_exec_session_id':29031,'ROOT_owned_exec_exit_tool_chunk':'37f468'}
old['changed_memory_policy_approval']={'path':str(approval_path),'sha256':sha(approval_path)}
names=['run_TSC_row_schedule_memory_queue_v3.py','run_TSC_bpe_row_schedule_memory_v3.py',
 'record_TSC_BPE_row_schedule_source_v1.py','train_lora_batched_tsc_bpe_row_schedule_memory_v3.py',
 'tsc_bpe_row_schedule_memory_v3.py','bpe_newrows_scheduled_acquisition_v1.py',
 'native_tdt_checkpointed_v4.py','metrics.py']
names.extend(sys.argv[1:])
old['source_sha256']={name:sha(HERE/name) for name in names}
manifest=HERE/'TSC_BPE_row_schedule_memory_queue_manifest_v3.json'
assert not manifest.exists()
write(manifest,old)
print(json.dumps({'passed':True,'approval_sha256':sha(approval_path),'manifest':str(manifest),'manifest_sha256':sha(manifest),'predecessor_archived':str(archive),'projected_fraction':projected/memory['commit_limit_bytes']}))
