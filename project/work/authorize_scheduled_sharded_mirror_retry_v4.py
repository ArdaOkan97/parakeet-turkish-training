from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, os, sys
ROOT=Path(__file__).resolve().parents[1]; HERE=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(HERE))
from resource_watchdog_v2 import sample_memory
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
target=HERE/'run_TSC_row_schedule_sharded_mirror_queue_v4.py'
assert not target.exists()
text=(HERE/'run_TSC_row_schedule_memory_queue_v3.py').read_text(encoding='utf-8')
text=text.replace('ROOT_authorized_TSC_row_schedule_memory_queue_v3','ROOT_authorized_TSC_row_schedule_sharded_mirror_queue_v4')
text=text.replace('== 29031','== 42301').replace("== '37f468'","== '28a6d4'")
text=text.replace("'compact_training_row_objects_v1.py',", "'compact_training_row_objects_v1.py', 'track_continuation_v9.py',")
text=text.replace('ROOT_actual_CLOSED_resource_stop_startup_host_release_approval_v3','ROOT_actual_CLOSED_resource_stop_sharded_mirror_approval_v4')
text=text.replace("approval['allow_next_startup_host_release']", "approval['allow_next_sharded_mirror_retry']")
needle="            'Changed host lifetime policy and actual successful native save are required')"
assert text.count(needle)==1
text=text.replace(needle,needle+"""
    mirror=read(Path(approval['new_mirror_identity_path']))
    require(mirror['status']=='running' and mirror['pid']==approval['new_mirror_pid']
            and mirror['script_sha256']==sha(HERE/'track_continuation_v9.py')
            and mirror['runstate_identity']['project']=='parakeet-turkish'
            and approval['old_mirror_actual_CLOSED_exit_code']==0,
            'The changed sole-mirror policy must actually be running after observed old closure')
""")
target.write_text(text,encoding='utf-8')
state_path=HERE/'experiment_queue_state.json';state=read(state_path)
assert state['queue_id']=='20261012_row_schedule_memory_03' and state['status']=='repair_required'
assert state['owned_child_pid'] is None and state['owned_controller_closure_observed'] is True
assert state['last_actual_waitpid']['exit_code']==1
status=read(Path(state['current_controller_loop'])/'loop_status.json')
assert status['completed_steps']==7011 and status['child_exit_code']==0 and status['stop_category']=='resource'
native=Path(status['paths']['native']).with_suffix('.completion.json');n=read(native)
assert n['status']=='complete' and n['sources_unchanged'] and not n.get('error')
identity=ROOT/'work/trackio_continuation_mirror_v9_01_identity.json';mirror=read(identity)
assert mirror['status']=='running' and mirror['script_sha256']==sha(HERE/'track_continuation_v9.py')
migration=read(ROOT/'work/trackio_continuation_mirror_state_v9_migration_receipt.json')
assert migration['passed'] and migration['source_state_unchanged'] and migration['runs_count']==39
assert read(HERE/'Trackio_V8_actual_CLOSED_ROOT_handoff_v1.json')['exit_code']==0
mem=sample_memory(os.getpid(),'ROOT_sharded_mirror_resource_retry_preflight')['system_memory']
projection=mem['commit_used_bytes']+12_176_424_960
assert mem['available'] and projection/mem['commit_limit_bytes']<.95 and mem['commit_limit_bytes']-projection>2_000_000_000
approval={'kind':'ROOT_actual_CLOSED_resource_stop_sharded_mirror_approval_v4','passed':True,
 'created_at_utc':datetime.now(timezone.utc).isoformat(),'native_complete_and_sources_unchanged':True,
 'native_completion_sha256':sha(native),'old_actual_updates':11,'root_owned_queue_session_id':42301,
 'root_owned_exit_code':1,'root_owned_exit_tool_chunk':'28a6d4','allow_next_sharded_mirror_retry':True,
 'next_controller_sha256':sha(HERE/'run_TSC_bpe_row_schedule_memory_v3.py'),
 'end_cache_threshold_bytes':3_000_000_000,'cap_bytes':8_000_000_000,'unchanged_guard_thresholds':True,
 'new_mirror_identity_path':str(identity),'new_mirror_pid':mirror['pid'],'new_mirror_owned_session':12608,
 'new_mirror_actual_live_tool_chunk':'5393ff','old_mirror_actual_CLOSED_exit_code':0,
 'old_mirror_owned_exit_tool_chunk':'53f2db','old_mirror_handoff':{'path':str(HERE/'Trackio_V8_actual_CLOSED_ROOT_handoff_v1.json'),'sha256':sha(HERE/'Trackio_V8_actual_CLOSED_ROOT_handoff_v1.json')},
 'resource_audit':{'path':str(HERE/'TSC_row_schedule_memory03_N11_resource_DEV_JSON_audit_v1.json'),'sha256':sha(HERE/'TSC_row_schedule_memory03_N11_resource_DEV_JSON_audit_v1.json')},
 'fresh_memory':mem,'historical_peak_projection_fraction':projection/mem['commit_limit_bytes'],
 'changed_policy':'Sole old V8 metricmirror observedCLOSED0; new V9 preserves all39 exact historicalrun states on disk and keeps only active map/mtime-cached currenthistory in RAM. Same frozenV3 training/source/RNG/B8x2/cap/native/schedule/guards; no unchanged resource retry.',
 'objective_complete':False}
ap=HERE/'TSC_BPE_row_schedule_sharded_mirror_ROOT_approval_v4.json';assert not ap.exists();write(ap,approval)
archive=ROOT/'work/experiment_queues/20261012_row_schedule_memory_03/closed_state_before_sharded_mirror_v4.json'
assert state_path.resolve().is_relative_to(ROOT) and archive.resolve().is_relative_to(ROOT) and not archive.exists()
os.replace(state_path,archive)
manifest=read(HERE/'TSC_BPE_row_schedule_memory_queue_manifest_v3.json')
manifest.update(kind='ROOT_authorized_TSC_row_schedule_sharded_mirror_queue_v4',queue_id='20261012_row_schedule_sharded_04',trial_namespace='sharded_04')
manifest['predecessor_closed_state']={'path':str(archive),'sha256':sha(archive),'ROOT_owned_exec_closed_exit_code':1,'ROOT_owned_exec_session_id':42301,'ROOT_owned_exec_exit_tool_chunk':'28a6d4'}
manifest['changed_memory_policy_approval']={'path':str(ap),'sha256':sha(ap)}
names=set(manifest['source_sha256']);names.remove('run_TSC_row_schedule_memory_queue_v3.py');names.update([target.name,'track_continuation_v9.py'])
manifest['source_sha256']={name:sha(HERE/name) for name in sorted(names)}
mp=HERE/'TSC_BPE_row_schedule_sharded_mirror_queue_manifest_v4.json';assert not mp.exists();write(mp,manifest)
planp=HERE/'experiment_plan.json';plan=read(planp);i=next(x for x in plan['ordered_experiments'] if x['id']=='03')
i.update(status='queued_in_local_process',queue_executor=target.name,actual_memory03_updates=11,
 actual_memory03_review='TSC_row_schedule_memory03_N11_resource_DEV_JSON_audit_v1.json',
 run_id='adaptive_20261011_hf_TSC_BPE512_row_schedule_sharded_04_main',
 smoke_run_id='adaptive_20261011_hf_TSC_BPE512_row_schedule_sharded_04_smoke',
 current_family_smoke_passed=False,actual_GPU_trial_completed=False,
 resource_retry_change='V9 streamed migration39 exactrun metric states, inactive on-disk shards and one active in-RAM map; oldV8 actualCLOSED0/53f2db. Training V3 compaction and schedule unchanged.')
write(planp,plan)
print(json.dumps({'passed':True,'manifest':str(mp),'manifest_sha256':sha(mp),'new_queue_sha256':sha(target),'baseline_commit_bytes':mem['commit_used_bytes'],'projection':projection/mem['commit_limit_bytes']}))
