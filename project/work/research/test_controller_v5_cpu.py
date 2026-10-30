"""Narrow controller guards and lifecycle checks; no GPU or model load."""
import copy
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'outputs/parakeet_turkish';sys.path.insert(0,str(OUTPUT))
import run_experiment_loop_continuation_v5 as m

passed=[]
def check(value,name):
 if not value:raise AssertionError(name)
 passed.append(name)
def reject(fn,name):
 try:fn()
 except (RuntimeError,KeyError,ValueError):passed.append(name);return
 raise AssertionError('not rejected: '+name)
tmp=Path(tempfile.mkdtemp(prefix='controller4_cpu_',dir=ROOT/'work/research'))
config=json.loads((OUTPUT/'runs/adaptive_20261010_continuation_03_resume_B8/run_config.json').read_text())
args=m.parser().parse_args(['--train-only']);args.strong_threshold=.1193939393939394
for invalid in (float('nan'),float('inf'),0.,-1.,3.1):
 bad=copy.copy(args);bad.window_hours=invalid
 reject(lambda:m.preflight(bad),'invalid_window_'+str(invalid))
bad=copy.copy(args);bad.train_only=False
reject(lambda:m.preflight(bad),'TEST_export_not_enabled_by_this_worker')
argv=m.training_argv(args,config,'cpu_rank8',tmp/'start.pt',8,250,1000,tmp/'stop')
check(argv[argv.index('--batch-size')+1]=='8' and argv[argv.index('--accumulation')+1]=='2','physical_B8_effective16')
check('--resume-exact' not in argv and '--resume-weights' in argv,'changed_data_fresh_optimizer')
check(argv[argv.index('--warmup-steps')+1]=='25','fresh_warmup_25_updates')
check(argv[argv.index('--steps')+1]=='250','short_screen_update_cap')
r16=m.training_argv(args,config,'cpu_rank16',tmp/'start16.pt',16,250,1000,tmp/'stop16')
check(r16[r16.index('--alpha')+1]=='32.0' and r16[r16.index('--rank')+1]=='16','rank16_scaling2')
log=tmp/'owned.log';log.write_text('public v3 PCM/input verification 5000/74883\n')
check(m.log_phase(log,[])['phase']=='actual_training_input_verification','input_audit_prepublication_phase')
log.write_text('acoustic cache 1300/74883\n')
phase=m.log_phase(log,[])
check(phase['cache_processed']==1300 and phase['optimizer_updates_started'] is False,'cache_count_is_not_optimizer_step')
log.write_text('{"step": 10, "loss": 1.0}\n')
check(m.log_phase(log,[{'step':10}])['phase']=='training_updates','training_phase_separate')
guard=m.ProgressGuard(0)
history=[{'step':i,'current_reserved_vram_gb':3,'audio_seen_seconds':i*100.,'train_wall_seconds':i*2.} for i in range(1,61)]
check(guard.observe(history,1)[1]['audio_min_per_wall_min_last50']==50,'rate_uses_update_wall_not_DEV')
guard.observe(history,2);check(guard.rate_streak==0 and guard.last_rate_step==60,'duplicate_history_does_not_advance_rate_streak')
check(guard.observe(history,602)[0]=='training_publication_stall_600seconds','bounded_training_stall')
guard=m.ProgressGuard(0);history[-1]['current_reserved_vram_gb']=12.1
guard.observe(history,1);guard.observe(history,6)
check(guard.observe(history,11)[0]=='reserved_above12GB_three_observations','reservation_safeguard')
sources=m.source_hashes();start=m.now();pid=123;identity={'kind':'owned_training_worker_identity_v1','expected_run_id':'cpu_rank8',
 'worker_arguments_sha256':m.argument_sha(argv),'pid':pid,'ppid':99,'started_at_utc':start,
 'process_creation_identity':{'available':True,'pid':pid,'creation_unix_seconds':m.stamp(start).timestamp()}}
names={'memory_launcher_sha256':m.WRAPPER,'cache_launcher_sha256':'launch_with_cache_reclaim_v5.py','cache_helper_sha256':'cuda_cache_reclaim_v1.py',
 'resource_helper_sha256':'resource_watchdog_v2.py','resource_telemetry_sha256':'paired_windows_commit_telemetry_v1.py',
 'frozen_worker_sha256':'train_lora_batched_data_v4.py','data_policy_sha256':'public_turkish_data_policy_v3.py'}
identity.update({key:sources[name] for key,name in names.items()})
check(m.validate_identity(identity,'cpu_rank8',argv,99,start,sources)==identity,'actual_owned_child_identity')
bad=copy.deepcopy(identity);bad['ppid']=98
reject(lambda:m.validate_identity(bad,'cpu_rank8',argv,99,start,sources),'other_process_rejected')
bad=copy.deepcopy(identity);bad['worker_arguments_sha256']='0'*64
reject(lambda:m.validate_identity(bad,'cpu_rank8',argv,99,start,sources),'different_worker_arguments_rejected')
ip=tmp/'identity.json';ip.write_text(json.dumps(identity));live=tmp/'live.json'
payload={'kind':'owned_worker_live_memory_v1','pid':pid,'run_id':'cpu_rank8','worker_identity_sha256':m.sha(ip),
 'process_creation_identity':identity['process_creation_identity'],**{key:identity[key] for key in names},
 'sample':{'observed_at_utc':m.now(),'calling_process_commit_memory':{'available':True,'pid':pid,'ullAvailPageFile_bytes':1_999_999_999}}}
live.write_text(json.dumps(payload))
check(m.self_memory(live,identity,ip)[0]=='actual_worker_process_commit_headroom_below2GB','actual_self_quota_guard')
payload['sample']['calling_process_commit_memory']['pid']=99;live.write_text(json.dumps(payload))
check(m.self_memory(live,identity,ip)[1]['available'] is False,'launcher_quota_not_worker_quota')
payload['sample']['observed_at_utc']=(datetime.now(timezone.utc)-timedelta(seconds=16)).isoformat();live.write_text(json.dumps(payload))
check(m.self_memory(live,identity,ip)[1]['reason']=='worker_self_sample_stale','stale_self_not_fresh')
prepared={'source_config':config,'source_selection':{},'source_provenance':{},'source_checkpoint':{'path':str(tmp/'source.pt')},
 'runtime':sources,'input_sha256':{},'receipt':{},'failed_dispatch_predecessor':{},'public_data':{'clips':74883},'screen_updates':250,
 'strong_DEV_threshold':.1193939393939394,'baseline_provenance':{}}
def lifecycle(kind):
 local=copy.copy(args);local.loop_id='lifecycle_'+kind;awake=[]
 def fake_freeze(source,target,*unused):target.parent.mkdir(parents=True,exist_ok=True);target.write_text('CPU');return {'path':str(target)}
 def stage(a,state,*unused):
  if kind=='exception':raise RuntimeError('planned CPU failure')
  if kind=='resource':state.update(resource_review_required=True,status='resource_review_required')
  return {'requested_screen_exposure_completed':False}
 with (patch.object(m,'OUTPUT',tmp),patch.object(m,'preflight',return_value=prepared),patch.object(m,'freeze_trainables',side_effect=fake_freeze),
      patch.object(m,'run_stage',side_effect=stage),patch.object(m.prior,'keep_awake',side_effect=lambda flag:awake.append(flag) or {'enabled':flag})):
  try:code=m.main(local)
  except RuntimeError:code='exception'
 state=json.loads((tmp/'runs'/local.loop_id/'loop_status.json').read_text())
 check(awake==[True,False],'awake_finally_'+kind)
 check(state['objective_complete'] is False and state['objective_incomplete'] is True,'persistent_objective_incomplete_'+kind)
 check(code=={'normal':0,'resource':1,'exception':'exception'}[kind],'lifecycle_exit_'+kind)
(tmp/'runs').mkdir()
for kind in ('normal','resource','exception'):lifecycle(kind)
result={'schema':1,'passed':True,'kind':'continuation_v5_narrow_CPU_guards','checks':passed,'check_count':len(passed),
 'controller_sha256':m.sha(OUTPUT/'run_experiment_loop_continuation_v5.py'),'CPU_only':True,'GPU_or_model_loaded':False,
 'scope':'Argument/layout/cache-phase, fresh system/self scope, rate/publication safeguards, awake/failure lifecycle; root owns actual GPU launch.'}
(OUTPUT/'continuation_controller_v5_cpu_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'passed':True,'checks':len(passed),'controller_sha256':result['controller_sha256']}))
