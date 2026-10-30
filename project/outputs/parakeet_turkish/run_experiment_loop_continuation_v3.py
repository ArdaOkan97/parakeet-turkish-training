"""New, explicitly renewed finite B8 loop with exact state fork and memory guards.

The prior loop deadline and all its artifacts remain immutable. This controller
starts only after a verified parent-owned prior process exit and a real scoped
cache-hook GPU validation. It never changes the frozen trainer or selects on TEST.
"""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from atomic_io_retry_v1 import read_json as read, write_json as write

OUTPUT=Path(__file__).resolve().parent
ROOT=OUTPUT.parents[1]
PRIOR_LOOP='adaptive_20261010_continuation_02'
SOURCE_RUN=PRIOR_LOOP+'_resume_B8'
SOURCE_STEP=1821
SOURCE_BEST_STEP=1821
BASELINE_LOOP='adaptive_20261010_recovery_b8_01'
CACHE_SOURCE_RUN=BASELINE_LOOP+'_b8_quality'
EXIT_RECEIPT_SHA256='54aeab34e637e4aed5a572a80d1f3c6479069f9ce03644e14926300ee4a581b6'
MEMORY_POLICY_SHA256='c6b1089828b04ae3a4fed822dd15934d87aa0f443c0b165aa434f66f19736190'
FROZEN={
 'run_experiment_loop_continuation_v2.py':'7c0f689c6886f8cbd221acd7e8d6b0e9ab9b9b0157ffe25150bbe1bbe39e8f58',
 'launch_with_memory_guard_v3.py':'28e51998bbb481f89a1106ad55fa6579f6ce7475826a8229d615ff973dccc2d1',
 'profile_cuda_cache_reclaim_v4.py':'1fa7ecdba6683644889c5389e58800a72bb65deb343e0d0616d19d3d57721a0b',
 'resource_watchdog_v1.py':'cc4ca1f11fdef19ac38d2aa878334451a100fd79f9a2e25a5afb4ceef89f0d7c',
 'train_lora_batched.py':'b32aa1dc09d5ff7dd85be9c309e8d69a095b8cf859b5498136a27d5d5af7cd9d',
 'run_experiment_loop_recovery_v2.py':'c16d0c61e0c4aa48d4bb93ef5ef8b8192032581d264a3be8fee369f4077a1213',
 'launch_with_io_retry_v1.py':'98032d4f56e56c6441b5ab3551cbd52f0b6681d9b2f312fb307cbc9b285ef00f',
 'atomic_io_retry_v1.py':'2d3fe379d51021fd15679e562c71286f7f9501180d28dde77893e8f72c80a1b0',
 'cuda_cache_reclaim_v1.py':'99ec5974fcd4714fa6fa341eb5cde70dc817085c997bd564c2b465aaaf98a543',
 'launch_with_cache_reclaim_v1.py':'db1ad326f26a89c5163a34a839cc781e457793b83d625a7c99b8366db9301459',
 'launch_with_memory_guard_v1.py':'0a7edea72019baeb951a226f12388d12e330ced36c14cfdc0ce4721f9703d75d',
 'launch_with_memory_guard_v2.py':'f7f4d134f2a583146d376ccc25d169e7e9ea90c6cbf9b84c2eb2c44284b47702',
 'benchmark_strong_tr_test_v2.py':'91012f94ba80bd1c0bbee7f06855d3120d5c982e461184b30968c9fc0a4c9580'}
IDENTITY_KEYS=('epochs','steps','batch_size','accumulation','replay_fraction','bucket_size','seed','max_tokens',
 'head_lr','lora_lr','new_token_lr_multiplier','replay_kl','duration_kl','feature_distillation','rank','alpha',
 'last_n_layers','chunk_size','tdt_implementation','warmup_steps','schedule','min_lr_ratio','eval_per_epoch',
 'eval_epoch_fraction','min_improvement','no_encoder_checkpoint','phonetic_sh_donors','train_limit',
 'early_stop_patience','max_retention_failures','target_turkish_wer')
CHALLENGERS=(
 {'name':'higher_lora','head_lr':5e-5,'lora_lr':6e-4,'new_token_lr_multiplier':6.},
 {'name':'higher_heads','head_lr':1e-4,'lora_lr':3e-4,'new_token_lr_multiplier':3.})

def now():return datetime.now(timezone.utc).isoformat()
def require(ok,message):
 if not ok:raise RuntimeError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def scoped(path,root=ROOT):
 path=Path(path).resolve();require(path.is_relative_to(root.resolve()),'Path leaves authorized workspace: '+str(path));return path
def safe_id(value):
 require(isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}',value),'Unsafe run/loop ID');return value
def stamp(value):
 value=datetime.fromisoformat(value);require(value.tzinfo is not None,'UTC timestamp needs timezone');return value
def current_sources():
 sources={name:sha(OUTPUT/name) for name in FROZEN}
 require(sources==FROZEN,'Frozen runtime/cache/previous controller source changed')
 for name in ('fork_training_state_v3.py','resource_watchdog_v1.py','compare_strong_test_reuse_v1.py','build_cache_launch_validation_v1.py','profile_cuda_cache_reclaim_v3.py'):
  require((OUTPUT/name).is_file(),'Required versioned helper unavailable: '+name);sources[name]=sha(OUTPUT/name)
 sources[Path(__file__).name]=sha(__file__)
 sources['finalize_lora.py']=sha(OUTPUT/'finalize_lora.py')
 original=read(OUTPUT/'runs'/SOURCE_RUN/'run_config.json',{})
 for name,digest in original.get('code_sha256',{}).items():
  require(sha(scoped(OUTPUT/name,OUTPUT))==digest,'Original training helper changed: '+name)
  sources[name]=digest
 return sources

def system_memory():
 from resource_watchdog_v1 import sample_memory
 return sample_memory()['system_memory']

def inferred_worker_phase(log,history,start_step):
 """Read a bounded log tail; phase is diagnostic inference, never a gate."""
 phase='startup_or_unknown';published=history[-1]['step'] if history else start_step
 evidence={'inferred_from_bounded_log_tail':True,'last_published_step':published,
  'history_publication_only_not_durable_checkpoint':True}
 try:
  with Path(log).open('rb') as stream:
   stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-65536));text=stream.read().decode('utf-8',errors='replace')
  lines=[line.strip() for line in text.splitlines() if line.strip()]
  evidence['log_mtime_utc']=datetime.fromtimestamp(Path(log).stat().st_mtime,timezone.utc).isoformat()
  for line in reversed(lines):
   if ' validation ' in line or 'validation tr:' in line:phase='development_evaluation';break
   if line.startswith('dev step'):phase='development_complete_or_checkpoint';break
   if line.startswith('{"step"'):phase='training_updates';break
   if line.startswith('{"accepted"'):phase='final_checkpoint_or_exit';break
   if 'cache' in line.lower() or 'prepar' in line.lower():phase='startup_or_cache';break
 except OSError as error:evidence.update(log_available=False,error_type=type(error).__name__)
 evidence['phase']=phase
 return phase,evidence

def evidence_file(entry):
 path=scoped(entry['path']);require(path.is_file() and sha(path)==entry['sha256'],'Measured evidence file changed: '+str(path));return path

def validate_cache_gpu(path, config):
 # Historical907 13GB/6GB causal and production-hook evidence remains historical.
 # The new9GB/4GB policy requires its own measured validation below.
 import run_experiment_loop_continuation_v2 as old
 return old.validate_cache_gpu(path, config)


def validate_entry(args):
 safe_id(args.loop_id);require(args.source_run_id==SOURCE_RUN,'This continuation forks the reviewed full1821 B8 source')
 require(0<args.window_hours<=3 and 0<args.final_reserve_seconds<=1800 and 30<=args.stop_grace_seconds<=300,'Bounded renewed window/reserve/grace required')
 require(all(math.isfinite(v) and v>0 for v in (args.continuation_seconds,args.challenger_seconds)), 'Stage quotas must be finite and positive')
 require(args.poll_seconds==5 and args.reserved_limit_gb==12 and args.reserved_observations==3
   and args.commit_limit_fraction==.9 and args.cache_threshold_gb==4
    and args.commit_critical_fraction==.95 and args.commit_headroom_bytes==2_000_000_000 and args.commit_observations==3,'Reviewed memory watchdog/cache thresholds required')
 require(0<args.regression_ratio<=.5 and args.rate_window_updates>=50 and args.regression_observations>=3,'Conservative throughput-regression guard required')
 require(args.publication_stall_seconds==600,'Reviewed all-phase publication-stall allowance is600seconds')
 prior_dir=OUTPUT/'runs'/PRIOR_LOOP
 audit_path=OUTPUT/'runs'/BASELINE_LOOP/'adaptive_results_verification.json'
 receipt_path=scoped(args.prior_exit_receipt);receipt=read(receipt_path,{})
 require(sha(receipt_path)==EXIT_RECEIPT_SHA256 and receipt.get('schema')==1
   and receipt.get('kind')=='root_owned_resource_review_exit_receipt_v1'
   and receipt.get('predecessor_loop_id')==PRIOR_LOOP and receipt.get('predecessor_run_id')==SOURCE_RUN
   and receipt.get('root_observed_owned_session_completed') is True and receipt.get('root_owned_exec_session')==3634
   and receipt.get('root_observed_exit_code')==1 and receipt.get('child_exit_code')==0
   and receipt.get('child_durable_step')==SOURCE_STEP and receipt.get('child_selected_step')==SOURCE_BEST_STEP
   and receipt.get('predecessor_status')=='resource_review_required'
   and receipt.get('predecessor_training_checkpoint_review_passed') is True
   and receipt.get('predecessor_audit_available') is False,
   'Parent-owned3634 stopped-process/full-state review receipt required; no old export or TEST claim')
 require(stamp(receipt['created_at_utc'])<=datetime.now(timezone.utc),'Process exit receipt timestamp is invalid')
 for path_key,sha_key in [('predecessor_status_path','predecessor_status_sha256'),
     ('predecessor_controller_path','predecessor_controller_sha256'),
     ('child_run_config_path','child_run_config_sha256'),('child_training_history_path','child_training_history_sha256'),
     ('child_selection_path','child_selection_sha256'),('source_latest_path','source_latest_sha256'),
     ('source_best_path','source_best_sha256'),('resource_review_path','resource_review_sha256'),
     ('prior_audited_receipt_path','prior_audited_receipt_sha256')]:
  require(sha(scoped(receipt[path_key]))==receipt[sha_key],'Reviewed predecessor file changed: '+path_key)
 require(Path(receipt['predecessor_status_path']).resolve()==(prior_dir/'loop_status.json').resolve(),
   'Predecessor controller path/receipt differs')
 prior_status=read(prior_dir/'loop_status.json',{})
 require(prior_status.get('status')=='resource_review_required' and prior_status.get('resource_stop_reason')=='system_commit_at_or_above95percent'
   and prior_status.get('deadline_at_utc')==receipt['predecessor_deadline_at_utc'],
   'Reviewed predecessor stop/deadline changed')
 old_audited_receipt=read(scoped(receipt['prior_audited_receipt_path']),{})
 require(Path(old_audited_receipt['predecessor_audit_path']).resolve()==audit_path.resolve()
   and sha(audit_path)==old_audited_receipt['predecessor_audit_sha256']
   and old_audited_receipt.get('predecessor_audit_passed') is True,
   'Strong TEST baseline audit must remain bound to the original completed recovery')
 for entry in receipt['runtime_provenance'].values():evidence_file(entry)
 for name,digest in receipt['prior_hf_files_sha256'].items():
  require(sha(scoped(Path(receipt['immutable_prior_candidate'])/name))==digest,'Prior published immutable HF candidate changed')
 config=read(OUTPUT/'runs'/SOURCE_RUN/'run_config.json',{})
 identity=config.get('config_identity',{}).get('training_arguments',{})
 require(set(identity)==set(IDENTITY_KEYS) and identity['batch_size']==8 and identity['accumulation']==2
   and identity['epochs']==3 and identity['steps'] is None and identity['warmup_steps']==25
   and identity['tdt_implementation']=='v2' and identity['chunk_size']==16
   and identity['no_encoder_checkpoint'] is False,'Source exact configuration must be the reviewed B8/3epochs/9611updates')
 history=read(OUTPUT/'runs'/SOURCE_RUN/'training_history.json',[])
 selection=read(OUTPUT/'runs'/SOURCE_RUN/'training_selection.json',None) or read(OUTPUT/'runs'/SOURCE_RUN/'selection.json',{})
 require(len(history)==SOURCE_STEP and history[-1]['step']==SOURCE_STEP and selection.get('completed_steps')==SOURCE_STEP
   and selection.get('selected_step')==SOURCE_BEST_STEP,'Source exact continuation needs full1821/latest and1821/best')
 require(config['steps']==9611 and selection.get('accepted') is True,'Source best DEV-selected state must qualify')
 loop_dir=OUTPUT/'runs'/args.loop_id
 require(loop_dir.exists() and {p.name for p in loop_dir.iterdir()}=={'fork_snapshot','fork_snapshot_independent_verification.json','fork_initial_states_archive.json'},
   'Renewed loop may contain only its three owned immutable fork prerequisites before launch')
 fork_path=loop_dir/'fork_snapshot/fork_manifest.json';fork=read(fork_path,{})
 require(fork.get('passed') is True and fork.get('status')=='complete' and fork.get('kind')=='exact_training_state_run_id_fork_v3'
   and fork.get('source_run_id')==SOURCE_RUN and fork.get('target_run_id')==args.loop_id+'_resume_B8'
   and fork.get('source_completed_step')==SOURCE_STEP and fork.get('inherited_selected_step')==SOURCE_BEST_STEP
   and fork.get('only_changed_path')=='run_id' and fork.get('inherited_config_identity')==config['config_identity'],
   'Pre-created exact1821/1821 fork manifest differs from source/target/config')
 require(fork.get('predecessor_exit_receipt_sha256')==EXIT_RECEIPT_SHA256
   and fork.get('source_review_kind')=='resource_stop_full_state_DEV_review','Fork stopped-predecessor receipt/review linkage changed')
 require(sha(scoped(fork['fork_helper_path'],OUTPUT))==fork['fork_helper_sha256'],'Actual completed fork helper source changed')
 archive_path=loop_dir/'fork_initial_states_archive.json';archive=read(archive_path,{})
 require(archive.get('passed') is True and archive.get('status')=='complete'
   and archive.get('fork_manifest_sha256')==sha(fork_path),'Immutable initial fork archive must bind manifest')
 independent_path=loop_dir/'fork_snapshot_independent_verification.json'
 require(archive.get('independent_fork_verification_sha256')==sha(independent_path)
   and read(independent_path,{}).get('passed') is True,'Exact fork independent verification changed or failed')
 for role in ('latest','best'):
  for collection in ('source_checkpoints','fork_checkpoints'):
   value=fork[collection][role];path=scoped(value['path'],ROOT/'work')
   require(sha(path)==value['sha256'] and value['step']==(SOURCE_STEP if role=='latest' else SOURCE_BEST_STEP),'Fork/source checkpoint bytes/step changed')
  value=archive['states'][role];path=scoped(value['archive_path'],loop_dir/'fork_snapshot')
  require(sha(path)==value['archive_sha256']==fork['fork_checkpoints'][role]['sha256'],'Immutable fork archive bytes changed')
 for source,value in fork['snapshot_files'].items():
  snapshot_path=scoped(value['snapshot_path'],loop_dir/'fork_snapshot')
  require(snapshot_path.stat().st_size==value['size_bytes'] and sha(snapshot_path)==value['sha256'],'Immutable snapshot copy changed')
 for value in fork['inherited_development_reuse'].values():
  require(value.get('evaluation_reused_without_new_inference') is True,'Inherited DEV must be labeled exact reuse')
  for source,copied in value['files'].items():
   require(sha(scoped(source))==sha(scoped(copied['snapshot_path'],OUTPUT/'results'))==copied['sha256'],
     'Copied inherited DEV source/target changed')
 validation=validate_cache_gpu(args.cache_validation,config)
 require(sha(scoped(args.memory_policy_validation))==MEMORY_POLICY_SHA256,'Actual reviewed1821 9GB/4GB launch proof changed')
 policy_manifest=read(args.memory_policy_validation,{})
 independent_path=evidence_file(policy_manifest['independent_verification']);independent=read(independent_path,{})
 require(independent.get('passed') is True and independent.get('kind')=='actual_memory_policy_CPU_verification_v1'
   and independent.get('CUDA_initialized') is False and independent['probe']==policy_manifest['probe']
   and independent['capture']==policy_manifest['actual_raw_tensor_capture']
   and independent['builder_source_sha256']==policy_manifest['builder_source_sha256']==sha(OUTPUT/'build_memory_policy_launch_validation_v1.py'),
   'Actual raw tensor/RNG numerical recomputation must remain bound to the measured policy proof')
 policy_validation=validate_memory_policy(args.memory_policy_validation,config)
 if not args.train_only:
  audit_worker=scoped(OUTPUT/args.audit_worker,OUTPUT)
  require(audit_worker.is_file(),'New continuation final auditor is required before launch')
 baseline=read(OUTPUT/'results/strong_whisper_large_v3_greedy_dev_verified/summary.json',{})
 require(baseline.get('status')=='complete' and baseline.get('metrics',{}).get('utterances')==50,'Verified fixed Turkish DEV reference unavailable')
 require(identity['target_turkish_wer']==baseline['metrics']['wer']+.02,'Exact target is pinned strong DEV WER plus2percentage points')
 sources=current_sources()
 for name,digest in config.get('code_sha256',{}).items():require(sha(OUTPUT/name)==digest,'Source training helper changed: '+name)
 for path,digest in config.get('input_sha256',{}).items():require(sha(scoped(path))==digest,'Source dataset identity changed')
 return {'passed':True,'checked_at_utc':now(),'scope':'CPU file/source/receipt preflight; no GPU launch',
   'source_sha256':sources,'prior_status_sha256':sha(prior_dir/'loop_status.json'),
   'prior_audit_path':str(audit_path),'prior_audit_sha256':sha(audit_path),
  'prior_resource_review_path':receipt['resource_review_path'],'prior_resource_review_sha256':receipt['resource_review_sha256'],
  'prior_runtime_cache_evidence':historical_cache_evidence(Path(receipt['runtime_provenance']['cache_provenance_path']['path']),
    Path(receipt['runtime_provenance']['io_provenance_path']['path']),SOURCE_RUN,1773,
    Path(receipt['runtime_provenance']['memory_guard_provenance_path']['path'])),
   'prior_exit_receipt_path':str(receipt_path),'prior_exit_receipt_sha256':sha(receipt_path),
   'cache_validation':validation,'memory_policy_validation':policy_validation,'fork_manifest_path':str(fork_path),'fork_manifest_sha256':sha(fork_path),
   'initial_fork_archive_path':str(archive_path),'initial_fork_archive_sha256':sha(archive_path),
   'old_deadline_at_utc':receipt['predecessor_deadline_at_utc'],'renewed_window_hours':args.window_hours,
   'source_config':config,'source_selection':selection,'challengers':list(CHALLENGERS),
   'first_stage_policy':'New run-ID fork only; all nested weights/Adam/sampler/RNG/history identical through1821',
   'final_audit_worker_sha256':None if args.train_only else sha(OUTPUT/args.audit_worker)}

def training_argv(config,run_id,*,resume_exact=None,resume_weights=None,changes=None,max_wall_seconds=1,stop_file=None):
 require(bool(resume_exact)!=bool(resume_weights),'One exact-state or fresh-optimizer warm start is required')
 identity={**config['config_identity']['training_arguments'],**(changes or {})}
 require(identity['batch_size']==8 and identity['accumulation']==2,'Every successor remains physicalB8/effective16; no silent fallback')
 result=['--run-id',safe_id(run_id)]
 for key in IDENTITY_KEYS:
  value=identity[key]
  if value is None or value is False:continue
  result.append('--'+key.replace('_','-'))
  if value is not True:result.append(str(value))
 for key in ('tr_manifest','replay_manifest','supplement'):
  if config.get(key):result.extend(['--'+key.replace('_','-'),str(config[key])])
 result.extend(['--resume-exact' if resume_exact else '--resume-weights',str(resume_exact or resume_weights),
  '--max-wall-seconds',str(max(1.,max_wall_seconds)),'--stop-file',str(stop_file),'--log-every',str(config.get('log_every',10))])
 return result

def normalized_tail(history,start_step):
 require(isinstance(history,list),'History must be a list')
 if not history:return None
 require(all(r.get('step')==i for i,r in enumerate(history,1)),'History step order/coverage invalid')
 row=history[-1]
 return {'completed_updates':row['step'],'inherited_completed_updates':start_step,'new_completed_updates':max(0,row['step']-start_step),
  'history_publication_only_not_durable_checkpoint':True,'current':row,
  'new_training_records':[r for r in history if r['step']>start_step]}

class Watchdog:
 def __init__(self,args,start_step):
  from resource_watchdog_v1 import ResourceGuard
  self.resource=ResourceGuard()
  self.args=args;self.start_step=start_step;self.reserved_streak=0;self.rate_streak=0;self.last_rate_step=None;self.rate_reference=None
 def observe(self,history,memory,worker_private_memory=None,*,observed_monotonic=None):
  reason,observations=self.resource.observe(memory,worker_private_memory,observed_monotonic=observed_monotonic)
  tail=normalized_tail(history,self.start_step)
  if not tail or not tail['new_completed_updates']:return reason,observations
  row=tail['current'];observations.update(completed_updates=row['step'],new_completed_updates=tail['new_completed_updates'],
   reserved_vram_gb=row.get('current_reserved_vram_gb'))
  reserved=row.get('current_reserved_vram_gb')
  require(isinstance(reserved,(float,int)) and math.isfinite(reserved),'Current reservation must be finite')
  self.reserved_streak=self.reserved_streak+1 if reserved>self.args.reserved_limit_gb else 0
  observations['reserved_observation_streak']=self.reserved_streak
  if self.reserved_streak>=self.args.reserved_observations:reason=reason or 'reserved_above12GB_three_observations'
  records=tail['new_training_records'];width=self.args.rate_window_updates
  if len(records)>width and row['step']!=self.last_rate_step:
   previous=records[-width-1];audio=row['audio_seen_seconds']-previous['audio_seen_seconds'];wall=row['train_wall_seconds']-previous['train_wall_seconds']
   require(audio>0 and wall>0,'Measured new update audio/time interval must be positive')
   rate=audio/wall;observations['recent_audio_min_per_wall_min']=rate
   if self.rate_reference is None:self.rate_reference=rate
   self.rate_streak=self.rate_streak+1 if rate<self.args.regression_ratio*self.rate_reference else 0
   self.last_rate_step=row['step'];observations.update(rate_reference=self.rate_reference,rate_regression_streak=self.rate_streak)
   if self.rate_streak>=self.args.regression_observations:reason=reason or 'sustained_training_rate_regression'
  return reason,observations

class PublicationWatchdog:
 def __init__(self,start_step,started_at,limit=600.):
  self.latest_step=start_step;self.latest_progress_at=started_at;self.limit=limit
 def observe(self,completed_step,current):
  if completed_step>self.latest_step:self.latest_step=completed_step;self.latest_progress_at=current
  return 'training_publication_stall_600seconds' if current-self.latest_progress_at>=self.limit else None

def request_stop(path,reason):
 require(not path.exists(),'Owned stop path must be fresh at request')
 path.write_text('Renewed continuation controller requests cooperative DEV/checkpoint stop: '+reason+'\n',encoding='utf-8')
def terminate_owned(child):
 if child is not None and child.poll() is None:
  child.terminate()
  try:child.wait(timeout=30)
  except subprocess.TimeoutExpired:child.kill();child.wait(timeout=30)

def historical_cache_evidence(path,io_path,run_id,start_step,memory_path=None):
 import run_experiment_loop_continuation_v2 as old
 return old.cache_evidence(path,io_path,run_id,start_step,memory_path)


def argument_sha(arguments):
 values=list(arguments)
 if values[:1]==['--']:values=values[1:]
 return hashlib.sha256(json.dumps(values,ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()

def validate_worker_identity(identity,run_id,arguments,launcher_pid=None,started_at=None):
 require(identity.get('schema')==1 and identity.get('kind')=='owned_training_worker_identity_v1'
   and identity.get('expected_run_id')==run_id and identity.get('worker_arguments_sha256')==argument_sha(arguments),
   'Actual worker identity must bind owned run and canonical worker arguments')
 for key,name in [('memory_launcher_sha256','launch_with_memory_guard_v3.py'),
   ('cache_launcher_sha256','launch_with_cache_reclaim_v1.py'),('cache_helper_sha256','cuda_cache_reclaim_v1.py'),
   ('resource_helper_sha256','resource_watchdog_v1.py'),('frozen_worker_sha256','train_lora_batched.py')]:
  require(identity.get(key)==FROZEN[name],'Self-reported worker source differs: '+name)
 pid=identity.get('pid');ppid=identity.get('ppid')
 require(type(pid) is int and pid>0 and type(ppid) is int and ppid>0,'Actual worker PID/PPID must be positive integers')
 if launcher_pid is not None:require(pid==launcher_pid or ppid==launcher_pid,'Actual worker must belong to owned interpreter/venv redirector')
 when=stamp(identity['started_at_utc']);require(when<=datetime.now(timezone.utc),'Worker identity timestamp is in the future')
 if started_at is not None:require(when>=stamp(started_at)-timedelta(seconds=5),'Identity predates this child')
 creation=identity['process_creation_identity'];require(creation.get('pid')==pid,'Creation identity PID differs')
 if creation.get('available'):
  seconds=creation.get('creation_unix_seconds')
  require(isinstance(seconds,(float,int)) and math.isfinite(seconds) and seconds<=when.timestamp()+1,'Invalid process creation time')
  if started_at is not None:require(seconds>=stamp(started_at).timestamp()-5,'PID creation predates owned launch')
 return identity

def read_worker_identity(path,run_id,arguments,launcher_pid,started_at):
 if not path.exists():return None
 return validate_worker_identity(read(path,{}),run_id,arguments,launcher_pid,started_at)

def cache_evidence(path,io_path,run_id,start_step,memory_path=None):
 evidence=read(path,{});io=read(io_path,{})
 require(evidence.get('status')=='complete' and evidence.get('active_process_attachment') is False
   and evidence.get('frozen_worker_sha256')==FROZEN['train_lora_batched.py'],'Cache launcher must complete unchanged frozen worker')
 for key,name in [('launcher_sha256','launch_with_cache_reclaim_v1.py'),('cache_helper_sha256','cuda_cache_reclaim_v1.py'),
   ('frozen_io_launcher_sha256','launch_with_io_retry_v1.py'),('frozen_io_helper_sha256','atomic_io_retry_v1.py')]:
  require(evidence.get(key)==evidence.get(key+'_after')==FROZEN[name],'Cache/IO source changed: '+name)
 cache=evidence['cache_reclamation']
 require(cache['worker_sha256']==cache['worker_sha256_after']==FROZEN['train_lora_batched.py']
   and cache['caller_function']=='run' and cache['caller_line']==539 and cache['threshold_reserved_bytes']==4_000_000_000
   and cache['failed_reclamations']==0,'Exact optimizer-end scope and4GB cleanup required')
 require(io.get('status')=='complete' and io.get('training_semantics_changed') is False,'Nested IO adapter changed math or failed')
 arguments=io.get('arguments',[])
 require(arguments.count('--run-id')==1 and arguments[arguments.index('--run-id')+1]==run_id,'Nested IO run differs')
 require(type(cache['eligible_update_calls']) is int and cache['eligible_update_calls']>=0
   and type(cache['reclamations']) is int and 0<=cache['reclamations']<=cache['eligible_update_calls'],'Invalid update/reclamation counts')
 compact={key:cache[key] for key in ('caller_function','caller_line','threshold_reserved_bytes','eligible_update_calls',
   'reclamations','failed_reclamations','worker_sha256','worker_sha256_after','cleanup_seconds_in_training_timing')}
 require(memory_path is not None,'Actual9GB memory and worker identity required');memory=read(memory_path,{})
 require(memory.get('status')=='complete' and memory.get('kind')=='memory_guard_training_launcher_v3'
   and memory.get('cleanup_threshold_bytes')==4_000_000_000 and memory.get('cap_bytes')==9_000_000_000
   and memory.get('allocator_cap_applied') is True and all(memory.get(k) is False for k in
   ('training_math_changed','backend_flags_changed','RNG_sampler_optimizer_changed','global_driver_settings_changed')),
   'Actual9GB/4GB policy must apply without changing math/RNG/settings')
 require(15e9<=memory['physical_cuda_device_bytes']<=18e9
   and memory['per_process_allocator_fraction']==9e9/memory['physical_cuda_device_bytes'],'Exact9GB fraction required')
 for key,name in [('memory_launcher_sha256','launch_with_memory_guard_v3.py'),
   ('cache_launcher_sha256','launch_with_cache_reclaim_v1.py'),('cache_helper_sha256','cuda_cache_reclaim_v1.py'),
   ('resource_helper_sha256','resource_watchdog_v1.py'),('frozen_worker_sha256','train_lora_batched.py')]:
  require(memory.get(key)==memory.get(key+'_after')==FROZEN[name],'Memory/telemetry source changed: '+name)
 require(Path(memory['cache_provenance_path']).resolve()==Path(path).resolve()
   and Path(memory['io_provenance_path']).resolve()==Path(io_path).resolve(),'Evidence linkage differs')
 # The new wrapper stores the canonical argument hash, avoiding raw command
 # duplication. The frozen I/O evidence supplies the exact argument list.
 require(memory['worker_arguments_sha256']==argument_sha(arguments),'Memory worker argument hash differs')
 identity_path=scoped(memory['worker_identity_path']);require(sha(identity_path)==memory['worker_identity_sha256'],'Worker identity changed')
 identity=validate_worker_identity(read(identity_path,{}),run_id,arguments)
 require(memory['worker_pid']==identity['pid'] and memory['worker_ppid']==identity['ppid']
   and memory['run_id']==run_id and memory['worker_arguments_sha256']==argument_sha(arguments),'Memory attribution differs')
 telemetry=memory['owned_update_memory_telemetry']
 require(telemetry['eligible_update_calls']==cache['eligible_update_calls'] and telemetry.get('inside_original_training_timing') is True
   and telemetry['minimum_sampling_interval_seconds']==5 and telemetry['sample_capacity']==128,'Telemetry optimizer-end scope differs')
 for sample in telemetry['samples']:
  if sample.get('status')!='telemetry_failed':require(sample['worker_private_memory'].get('pid')==identity['pid'],'Telemetry refers to launcher instead of worker')
 return {'cache_provenance_path':str(path),'cache_provenance_sha256':sha(path),'io_provenance_path':str(io_path),
  'io_provenance_sha256':sha(io_path),'cache':compact,'inherited_completed_step':start_step,
  'memory_guard_provenance_path':str(memory_path),'memory_guard_provenance_sha256':sha(memory_path),
  'allocator_cap_bytes':memory['cap_bytes'],'per_process_allocator_fraction':memory['per_process_allocator_fraction'],
  'worker_identity_path':str(identity_path),'worker_identity_sha256':sha(identity_path),'actual_worker_pid':identity['pid'],
  'owned_update_memory_telemetry':{k:telemetry[k] for k in ('eligible_update_calls','successful_samples','failed_samples','samples_omitted')}}

def validate_memory_policy(path,config):
 manifest=read(scoped(path),{})
 require(manifest.get('schema')==1 and manifest.get('kind')=='memory_guard_policy_launch_validation_v1','New9GB/4GB policy manifest required')
 probe_path=evidence_file(manifest['probe']);probe=read(probe_path,{})
 worker=evidence_file(manifest['probe_worker']);wrapper=evidence_file(manifest['production_wrapper'])
 require(wrapper==OUTPUT/'launch_with_memory_guard_v3.py' and sha(wrapper)==FROZEN[wrapper.name],'Wrong production9GB/4GB wrapper')
 require(probe.get('status')=='complete' and probe.get('passed') is True and probe.get('solo_gpu') is True
   and probe.get('probe_version')==4 and probe.get('matched_sources_equal') is True,'New TRAIN-only source1821 policy probe must complete')
 identity=probe['identity']
 require(identity['source_step']==SOURCE_STEP and identity['source_run_id']==SOURCE_RUN and identity['source_sha256']==config['code_sha256']
   and identity['probe_sha256']==sha(worker)==probe['probe_sha256_after'] and probe.get('original_run_mutated') is False,'New policy source identity differs')
 require(sha(scoped(identity['checkpoint_path']))==identity['checkpoint_sha256']==probe['checkpoint_sha256_after'],'Policy source checkpoint changed')
 require(probe['branch_policies']=={'control':{'allocator_hard_cap_gb':13.,'cleanup_threshold_gb':6.},
   'reclaim':{'allocator_hard_cap_gb':9.,'cleanup_threshold_gb':4.}},'Actual old13/6 vs new9/4 comparison required')
 from resource_watchdog_v1 import POLICY,ResourceGuard
 require(probe['resource_policy']==POLICY and probe['maximum_reserved_gb_guard']==12
   and probe['allocator_hard_cap_gb']==9 and probe['cleanup_threshold_gb']==4,
   'Probe critical/sustained safety limits differ')
 require(identity['resource_helper_sha256']==FROZEN['resource_watchdog_v1.py']
   and identity['memory_launcher_v3_sha256']==FROZEN['launch_with_memory_guard_v3.py']
   and identity['helper_sha256']==FROZEN['cuda_cache_reclaim_v1.py']
   and identity['launcher_sha256']==FROZEN['launch_with_cache_reclaim_v1.py'],'Policy measured runtime source differs')
 control=probe['branches']['control'];reclaim=probe['branches']['reclaim'];count=probe['shared_prefix_updates']
 require(type(count) is int and count>=2 and len(control['updates'])==control['completed_updates']==count
   and len(reclaim['updates'])==reclaim['completed_updates']>=100 and probe.get('intervention_requested_updates_completed') is True
   and reclaim.get('stop_reason') is None,'At least100 varied9GB/4GB updates plus measured prefix required')
 for a,b in zip(control['updates'],reclaim['updates']):
  require(a['source_ids']==b['source_ids'] and set(a['component_losses'])==set(b['component_losses']),'Policy A/B sampled sources/objective differ')
 guard=ResourceGuard()
 require(type(probe['actual_worker_pid']) is int and probe['actual_worker_pid']>0,'Actual probe self PID required')
 for row in reclaim['updates']:
  require(all(isinstance(x,(float,int)) and math.isfinite(x) for x in row['component_losses'].values())
   and math.isfinite(row['gradient_norm_before_clip']) and row['audio_seconds']>0 and row['training_seconds_including_stats_cleanup']>0,
   'Policy objective/gradient/time must be finite')
  memory=row['system'];fraction=memory['commit_fraction']
  available=memory.get('commit_available_bytes',memory['commit_limit_bytes']-memory.get('committed_bytes',memory.get('commit_used_bytes',0)))
  require(memory.get('available') is True and 0<=fraction<.95 and available>=2_000_000_000
   and row['allocator']['reserved_bytes.all.current']<=9_000_000_000
   and row['pre_cleanup_allocator']['reserved_bytes.all.current']<=9_000_000_000,'New branch exceeds critical memory bounds')
  for key in ('before_owned_memory_telemetry','owned_memory_telemetry'):
   sample=row[key];private=sample['worker_private_memory']
   require(private.get('pid')==probe['actual_worker_pid'],'Probe process-private observation must sample the actual interpreter')
   # Replay at the recorded observation time; live freshness remains enforced
   # by the unmodified guard during the actual probe and the production loop.
   from unittest.mock import patch
   with patch('resource_watchdog_v1._timestamp',return_value=sample['observed_at_utc']):
    reason,_=guard.observe(sample['system_memory'],private,observed_monotonic=sample['observed_monotonic'])
   require(reason is None,'Actual new policy telemetry replay trips the critical/sustained guard')
  require(row['moderate_commit_streak_after_observation']==guard.moderate_streak,'Policy declared fresh-observation streak differs')
 for sample in reclaim['capture_memory_events']:
  memory=sample['system_memory']
  require(memory.get('available') is True and memory['commit_fraction']<.95 and memory['commit_available_bytes']>=2_000_000_000,
   'Capture phase must retain critical global memory headroom')
 require(set(probe['comparisons'])=={'1',str(count)},'First and last matched tensors required')
 limits={'gradient_relative_l2':.0005,'gradient_min_cosine':.9999995,'component_absolute_delta':.0001,'weights_relative_l2':.000001}
 maximum=max(abs(a['component_losses'][k]-b['component_losses'][k])
   for a,b in zip(control['updates'],reclaim['updates']) for k in a['component_losses'])
 require(maximum==probe['max_component_absolute_delta'],'Policy declared component error differs')
 for value in probe['comparisons'].values():
  require(value['sampler_rng_equal'] is True and value['python_cpu_cuda_rng_equal'] is True
   and value['quality_or_numerical_equivalence_waiver'] is False,'Policy sampler/RNG changed or numerical limits waived')
  for category in ('gradients','weights'):
   require(all(math.isfinite(value[category][k]) for k in ('max_absolute_delta','relative_l2','cosine'))
    and value[category]['relative_l2']>=0 and value[category]['max_absolute_delta']>=0 and -1<=value[category]['cosine']<=1.000000000001,
    'Invalid measured tensor difference')
 gate=manifest['strict_numerical_gate']
 passed=maximum<limits['component_absolute_delta'] and all(v['gradients']['relative_l2']<limits['gradient_relative_l2']
   and v['gradients']['cosine']>limits['gradient_min_cosine'] and v['weights']['relative_l2']<limits['weights_relative_l2'] for v in probe['comparisons'].values())
 require(gate.get('thresholds')==limits and gate.get('waived') is False and gate.get('passed')==passed,'Policy strict gate differs from measured tensors/components')
 evidence_file({'path':probe['full_capture_path'],'sha256':probe['full_capture_sha256']})
 return {'validation_path':str(scoped(path)),'validation_sha256':sha(path),'probe_path':str(probe_path),'probe_sha256':sha(probe_path),
  'production_wrapper_sha256':sha(wrapper),'source_step':SOURCE_STEP,'allocator_cap_bytes':9_000_000_000,
  'cleanup_threshold_bytes':4_000_000_000,'intervention_completed_updates':reclaim['completed_updates'],
  'strict_numerical_gate':{**gate,'recomputed_max_component_absolute_delta':maximum,
   'scope':'Cache-policy global gradients/trainable weights/components only; encoder features unmeasured',
   'historical_full_BF16_original_vs_v2_equivalence_gate_passed':False},
  'production_hook_evidence_scope':'Historical frozen run:539 execution; new9/4 production evidence recorded per resumed trial'}

def remaining_for_child(deadline):
 remaining=deadline-time.monotonic()
 require(math.isfinite(remaining) and remaining>0,'Renewed deadline expired; do not start another final child')
 return remaining

def freeze_stage_input(source,run_id,loop_dir):
 source=scoped(source);target=scoped(loop_dir/'trial_inputs'/(run_id+'_start_weights.pt'),loop_dir)
 require(not target.exists(),'Preserve immutable per-trial start weights')
 target.parent.mkdir(exist_ok=True);digest=sha(source);shutil.copyfile(source,target)
 require(sha(source)==sha(target)==digest,'Warm-start input changed while snapshotting')
 return target,{'original_source_path':str(source),'original_source_sha256':digest,
   'immutable_input_path':str(target),'immutable_input_sha256':digest}

def run_child(command,log,timeout):
 with log.open('xb') as stream:
  result=subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTHONUTF8='1',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1'),
   stdout=stream,stderr=subprocess.STDOUT,timeout=timeout,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 return result.returncode

def main(args):
 preflight=validate_entry(args)
 if args.validate_only:
  target=scoped(args.preflight_output or OUTPUT/'continuation_v3_preflight.json')
  require(not target.exists(),'Preflight output already exists; preserve reviewed evidence')
  write(target,{k:v for k,v in preflight.items() if k not in ['source_config','source_selection']})
  print(json.dumps({'passed':True,'gpu_used':False,'preflight':str(target)}));return 0
 memory=system_memory();require(memory.get('available') and memory['commit_fraction']<=.9
   and memory['commit_available_bytes']>=2_000_000_000,'System commit pressure must be resolved before any new GPU child')
 loop_dir=OUTPUT/'runs'/args.loop_id
 start=time.monotonic();start_utc=datetime.now(timezone.utc);deadline=start+args.window_hours*3600
 training_deadline=deadline-args.final_reserve_seconds-args.stop_grace_seconds
 state={'schema':1,'kind':'renewed_finite_B8_continuation_v3','loop_id':args.loop_id,'status':'validating_preserved_fork',
  'started_at_utc':start_utc.isoformat(),'deadline_at_utc':(start_utc+timedelta(hours=args.window_hours)).isoformat(),
  'window_hours':args.window_hours,'renewal_authorization':'Explicit user request to fix the transient memory stop, resume, and continue until competitive Turkish DEV WER with retention',
  'prior_loop_id':PRIOR_LOOP,'prior_deadline_at_utc':preflight['old_deadline_at_utc'],'prior_exit_receipt':preflight['prior_exit_receipt_path'],
  'preflight':{k:v for k,v in preflight.items() if k not in ['source_config','source_selection']},
  'selection_split':'FLEURS validation only','heldout_policy':'No TEST parsing during trials or candidate selection; final frozen winner only. Existing fixed Whisper TEST reused read-only for final reporting.',
  'runtime':{'batch_size':8,'accumulation':2,'effective_batch':16,'threshold_reserved_gb':4,'allocator_cap_gb':9,
    'memory_guard_launcher_sha256':FROZEN['launch_with_memory_guard_v3.py'],'strict_BF16_numeric_equivalence_passed':False},
  'watchdog_policy':{'reserved_limit_gb':12,'reserved_observations':3,'commit_limit_fraction':.9,'commit_critical_fraction':.95,'commit_headroom_bytes':2_000_000_000,
    'commit_observations':3,'commit_min_interval_seconds':5,'commit_telemetry_max_age_seconds':15,'memory_observations_phase_tagged':True,'rate_ratio':args.regression_ratio,
    'rate_window_updates':args.rate_window_updates,'rate_observations':args.regression_observations,
    'publication_stall_seconds':args.publication_stall_seconds,'resource_stop_blocks_all_further_GPU':True},
  'stage_plan':{'exact_continuation_cap_seconds':args.continuation_seconds,'challenger_cap_seconds':args.challenger_seconds,
    'challengers':list(CHALLENGERS),'winning_extension_remaining_training_budget':True,'minimum_fresh_TR_epochs_for_comparison':.25},
  'final_reserve_seconds':args.final_reserve_seconds,'stop_grace_seconds':args.stop_grace_seconds,'trials':[],
  'requires_resource_review':False,'warnings':[],
  'objective_complete':False,'objective_status':'incomplete',
  'next_action':'Continue DEV-only adaptive training toward the fixed strong DEV threshold with all retention gates'}
 status_path=loop_dir/'loop_status.json';write(status_path,state)
 source_config=preflight['source_config']
 import run_experiment_loop_recovery_v2 as prior_utils
 source_selection,source_provenance=prior_utils.validate_selection(SOURCE_RUN)
 best={'run_id':SOURCE_RUN,'wer':source_selection['turkish_validation_wer'],'selected_step':source_selection['selected_step'],
  'weights':source_selection['weights'],'provenance':source_provenance,'selected_configuration':{'batch_size':8,'accumulation':2,
    'tdt_implementation':'v2','chunk_size':16,'encoder_checkpointing':True},'parameters':{k:source_config[k] for k in ['head_lr','lora_lr','new_token_lr_multiplier']},'trial':'prior_verified_B8'}
 state['best_development_candidate']=dict(best)
 awake_requested=False
 try:
  state['awake_request']=prior_utils.keep_awake(True)
  awake_requested=bool(state['awake_request'].get('enabled'))
  write(status_path,state)
  resumed_id=safe_id(args.loop_id+'_resume_B8')
  fork_path=Path(preflight['fork_manifest_path']);fork=read(fork_path,{})
  require(sha(fork_path)==preflight['fork_manifest_sha256'],'Pre-created immutable fork manifest changed')
  require(fork.get('passed') is True and fork.get('status')=='complete' and fork.get('source_run_id')==SOURCE_RUN
    and fork.get('target_run_id')==resumed_id and fork.get('only_changed_path')=='run_id'
    and fork.get('source_completed_step')==SOURCE_STEP and fork.get('inherited_selected_step')==SOURCE_BEST_STEP,'Exact fork manifest incomplete/wrong-run')
  require(fork.get('inherited_config_identity')==source_config['config_identity'],'Fork nested config identity changed')
  archive_path=Path(preflight['initial_fork_archive_path']);archive=read(archive_path,{})
  require(sha(archive_path)==preflight['initial_fork_archive_sha256'],'Initial fork archive manifest changed')
  latest=scoped(archive['states']['latest']['archive_path'],loop_dir/'fork_snapshot')
  require(sha(latest)==fork['fork_checkpoints']['latest']['sha256'],'Fork latest bytes changed')
  state.update(status='running',fork_manifest_path=str(fork_path),fork_manifest_sha256=sha(fork_path),
    initial_fork_archive_path=str(archive_path),initial_fork_archive_sha256=sha(archive_path))
  write(status_path,state)

  def stage(name,seconds,*,exact=None,epochs=None,parameters=None):
   nonlocal best
   require(not state['requires_resource_review'],'Resource stop forbids blind GPU restart')
   require(current_sources()==preflight['source_sha256'],'Reviewed runtime/controller sources changed before next GPU stage')
   available=training_deadline-time.monotonic()
   if available<300:return None
   run_id=resumed_id if exact else safe_id(args.loop_id+'_'+name)
   stop=ROOT/'work'/(run_id+'.stop')
   require(not stop.exists() and (exact or not (OUTPUT/'runs'/run_id).exists()),'Stage/owned stop ID already exists')
   source_changes={} if exact else {**(parameters or best['parameters']),'epochs':epochs,'steps':None}
   start_step=SOURCE_STEP if exact else 0
   quota=min(seconds,available);stage_started=time.monotonic();stage_deadline=stage_started+quota
   if exact:
    start_input=exact;input_identity={'original_source_path':fork['fork_checkpoints']['latest']['path'],
      'original_source_sha256':fork['fork_checkpoints']['latest']['sha256'],
      'immutable_input_path':str(exact),'immutable_input_sha256':sha(exact)}
   else:start_input,input_identity=freeze_stage_input(best['weights'],run_id,loop_dir)
   argv=training_argv(source_config,run_id,resume_exact=exact,resume_weights=None if exact else start_input,
     changes=source_changes,max_wall_seconds=quota,stop_file=stop)
   io_path=loop_dir/(run_id+'_io_provenance.json');cache_path=loop_dir/(run_id+'_cache_provenance.json')
   memory_path=loop_dir/(run_id+'_memory_guard_provenance.json')
   identity_path=loop_dir/(run_id+'_worker_identity.json')
   command=[sys.executable,'-X','utf8','-u',str(OUTPUT/'launch_with_memory_guard_v3.py'),
     '--provenance',str(io_path),'--cache-provenance',str(cache_path),'--memory-provenance',str(memory_path),
     '--worker-identity',str(identity_path),'--cap-gb','9','--threshold-gb','4','--',*argv]
   entry={'name':name,'run_id':run_id,'status':'running','command':command,'started_at_utc':now(),'quota_seconds':quota,
     'initial_step':start_step,'inherited_completed_step':start_step,'start_weights':str(start_input),
     'start_weights_sha256':sha(start_input),'start_weights_provenance':input_identity,
     'cache_provenance_path':str(cache_path),'io_provenance_path':str(io_path),'memory_guard_provenance_path':str(memory_path),
     'optimizer_policy':'exact1821optimizer/sampler/RNG continuation' if exact else 'freshAdam DEV-best warmstart',
     'worker_identity_path':str(identity_path),'worker_arguments_sha256':argument_sha(argv),
     'parameters':{k:source_changes.get(k,source_config[k]) for k in ['head_lr','lora_lr','new_token_lr_multiplier']},
     'cache_launcher_sha256':FROZEN['launch_with_cache_reclaim_v1.py'],'cache_helper_sha256':FROZEN['cuda_cache_reclaim_v1.py'],
     'minimum_fresh_comparison_epochs':None if exact else .25,'watchdog_observations':[]}
   state['trials'].append(entry);state['active_run_id']=run_id;write(status_path,state)
   child=None;stop_requested=None;stop_requested_at=None;watch=Watchdog(args,start_step)
   publication=PublicationWatchdog(start_step,stage_started,args.publication_stall_seconds)
   log=loop_dir/(run_id+'.log');entry['log']=str(log)
   try:
    with log.open('xb') as stream:
     child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONUTF8='1',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1'),
       stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
     entry['pid']=child.pid
     while child.poll() is None:
      history=read(OUTPUT/'runs'/run_id/'training_history.json',[])
      from resource_watchdog_v1 import sample_memory
      phase,phase_evidence=inferred_worker_phase(log,history,start_step)
      worker_identity=read_worker_identity(identity_path,run_id,argv,child.pid,entry['started_at_utc'])
      actual_pid=worker_identity['pid'] if worker_identity is not None else None
      if worker_identity is not None:
       if entry.get('worker_identity_sha256'):require(sha(identity_path)==entry['worker_identity_sha256'],'Owned worker identity changed during child execution')
       entry.update(actual_worker_pid=actual_pid,worker_identity_sha256=sha(identity_path),launcher_pid=child.pid)
      snapshot=sample_memory(actual_pid if actual_pid else child.pid,phase)
      snapshot['worker_private_memory'].update(attribution='validated_self_reported_frozen_worker' if actual_pid else 'venv_launcher_only_actual_worker_unresolved',launcher_pid=child.pid)
      reason,observation=watch.observe(history,snapshot['system_memory'],snapshot['worker_private_memory'],
        observed_monotonic=snapshot['observed_monotonic'])
      observation['worker_phase']=phase_evidence
      observation['at_utc']=now();entry['watchdog_observations'].append(observation)
      entry['watchdog_observations']=entry['watchdog_observations'][-120:]
      tail=normalized_tail(history,start_step)
      if tail:
       entry.update(completed_updates=tail['completed_updates'],new_completed_updates=tail['new_completed_updates'],
         latest_training_epoch=tail['current'].get('turkish_epoch'),history_publication_mtime_utc=datetime.fromtimestamp((OUTPUT/'runs'/run_id/'training_history.json').stat().st_mtime,timezone.utc).isoformat())
       dev=[r for r in history if 'validation' in r]
       if dev:entry.update(latest_validation_step=dev[-1]['step'],latest_validation=dev[-1]['validation'])
      current=time.monotonic()
      reason=reason or publication.observe(tail['completed_updates'] if tail else start_step,current)
      if reason and not stop_requested:
       stop_requested=reason;state['requires_resource_review']=True;state['resource_stop_reason']=reason
      if current>=stage_deadline and not stop_requested:stop_requested='stage_wall_quota'
      if stop_requested and stop_requested_at is None:
       request_stop(stop,stop_requested);stop_requested_at=current;entry.update(cooperative_stop_reason=stop_requested,cooperative_stop_requested_at_utc=now())
      if stop_requested_at is not None and current-stop_requested_at>=args.stop_grace_seconds:
       entry['forced_stop']='Owned worker exceeded cooperative DEV/checkpoint grace';state['requires_resource_review']=True
       terminate_owned(child);break
      state.update(updated_at_utc=now(),remaining_wall_seconds=max(0,deadline-current));write(status_path,state)
      time.sleep(args.poll_seconds)
     entry['exit_code']=child.returncode
   finally:
    terminate_owned(child);entry.update(finished_at_utc=now(),wall_seconds=time.monotonic()-stage_started)
    state.pop('active_run_id',None)
   final=read(OUTPUT/'runs'/run_id/'selection.json',{});entry['selection']=final
   entry['durable_completed_updates']=final.get('completed_steps');entry['completed_steps']=final.get('completed_steps')
   if isinstance(final.get('completed_steps'),int):
    entry.update(completed_updates=final['completed_steps'],new_completed_updates=final['completed_steps']-start_step)
   entry['status']='complete' if entry['exit_code']==0 else 'failed'
   if entry['exit_code']!=0:
    state['requires_resource_review']=True;state['resource_stop_reason']=state.get('resource_stop_reason','worker_failure_requires_review')
   if entry['exit_code']==0:
    entry['cache_evidence']=cache_evidence(cache_path,io_path,run_id,start_step,memory_path)
    delta_updates=final.get('completed_steps',start_step)-start_step
    require(entry['cache_evidence']['cache']['eligible_update_calls']==delta_updates,'Cache scope must cover every and only NEW optimizer update')
    if final.get('accepted'):
     selected,provenance=prior_utils.validate_selection(run_id)
     actual_epochs=selected.get('turkish_epochs_completed',0)
     adequate=bool(exact) or actual_epochs>=.25 or stop_requested in ('stage_wall_quota','system_commit_above90percent_three_fresh_observations','system_commit_at_or_above95percent','system_commit_headroom_below2GB','reserved_above12GB_three_observations','sustained_training_rate_regression')
     entry.update(actual_turkish_epochs=actual_epochs,minimum_exposure_met=bool(exact) or actual_epochs>=.25,
        comparison_exception=None if adequate and (exact or actual_epochs>=.25) else stop_requested)
     require(adequate,'Fresh stage lacks quarter-epoch DEV comparison exposure and no deadline/resource exception')
     entry['selection_provenance']=provenance
     if selected['turkish_validation_wer']<best['wer'] or (exact and selected['turkish_validation_wer']==best['wer']):
      best={'run_id':run_id,'wer':selected['turkish_validation_wer'],'selected_step':selected['selected_step'],'weights':selected['weights'],
       'provenance':provenance,'parameters':entry['parameters'],'selected_configuration':{'batch_size':8,'accumulation':2,
         'tdt_implementation':'v2','chunk_size':16,'encoder_checkpointing':True},'trial':name}
   state['best_development_candidate']=dict(best);write(status_path,state)
   print(json.dumps({'stage':name,'run_id':run_id,'status':entry['status'],'new_completed_updates':entry.get('new_completed_updates'),
     'best_DEV_wer':best['wer'],'requires_resource_review':state['requires_resource_review']}),flush=True)
   return entry

  stage('exact_continuation',args.continuation_seconds,exact=latest)
  if not state['requires_resource_review'] and best['wer']>source_config['target_turkish_wer']:
   for challenger in CHALLENGERS:
    if training_deadline-time.monotonic()<900:break
    stage(challenger['name'],args.challenger_seconds,epochs=.5,parameters={k:challenger[k] for k in ['head_lr','lora_lr','new_token_lr_multiplier']})
    if state['requires_resource_review'] or best['wer']<=source_config['target_turkish_wer']:break
  if not state['requires_resource_review'] and best['wer']>source_config['target_turkish_wer'] and training_deadline-time.monotonic()>=900:
   stage('winning_extension',training_deadline-time.monotonic(),epochs=3.,parameters=best['parameters'])
  frozen={**best,'frozen_at_utc':now(),'selection_split':'FLEURS validation only','competitive_threshold_wer':source_config['target_turkish_wer'],
    'baseline_provenance':read(OUTPUT/'runs'/PRIOR_LOOP/'frozen_final_selection.json',{})['baseline_provenance'],
    'loop_id':args.loop_id,'prior_loop_id':PRIOR_LOOP,'inherited_history_provenance':{'fork_manifest_path':str(fork_path),'sha256':sha(fork_path),
      'source_run_id':SOURCE_RUN,'inherited_completed_step':SOURCE_STEP,'inherited_selected_step':SOURCE_BEST_STEP}}
  freeze_path=loop_dir/'frozen_final_selection.json';write(freeze_path,frozen)
  state.update(status='resource_review_required' if state['requires_resource_review'] else 'development_selection_frozen',
    frozen_selection_sha256=sha(freeze_path),best_development_candidate=dict(best))
  write(status_path,state)
  if not state['requires_resource_review'] and not args.train_only:
   require(current_sources()==preflight['source_sha256'],'Runtime/controller sources changed during renewed training')
   require(best['run_id']!=SOURCE_RUN,'Reviewed predecessor has no export/TEST; successful new exact fork must own finalization')
   state['status']='finalizing';write(status_path,state)
   command=[sys.executable,'-X','utf8','-u',str(OUTPUT/'launch_with_io_retry_v1.py'),'--worker','finalize_lora.py',
    '--provenance',str(loop_dir/'finalizer_io_provenance.json'),'--expected-worker-sha',sha(OUTPUT/'finalize_lora.py'),'--','--run-id',best['run_id']]
   require(run_child(command,loop_dir/'finalization.log',remaining_for_child(deadline))==0,'New-run frozen finalizer failed')
   completed_selection=read(OUTPUT/'runs'/best['run_id']/'selection.json',{})
   state['final_selection']={'path':str(OUTPUT/'runs'/best['run_id']/'selection.json'),
     'sha256':sha(OUTPUT/'runs'/best['run_id']/'selection.json'),'run_id':best['run_id'],
     'selected_step':completed_selection.get('selected_step'),'heldout_tests_evaluated':completed_selection.get('heldout_tests_evaluated')}
   merged_gate=completed_selection.get('merged_development_gate',{})
   require(isinstance(merged_gate.get('turkish_wer'),(int,float)) and math.isfinite(merged_gate['turkish_wer']),
     'Final merged winner must contain measured Turkish DEV gate')
   state['competitive_on_development']=merged_gate['turkish_wer']<=source_config['target_turkish_wer']
   state['merged_development_gate']=merged_gate
   state['status']='comparing_reused_strong_TEST';write(status_path,state)
   comparison=loop_dir/'strong_test_comparison.json'
   command=[sys.executable,'-X','utf8','-u',str(OUTPUT/'compare_strong_test_reuse_v1.py'),'--frozen-selection',str(freeze_path),
    '--source-verification',preflight['prior_audit_path'],'--comparison-output',str(comparison)]
   require(run_child(command,loop_dir/'strong_test_comparison.log',remaining_for_child(deadline))==0,'Read-only measured strong TEST comparison failed')
   state.update(status='verifying_final_results',strong_test_comparison_path=str(comparison),strong_test_comparison_sha256=sha(comparison));write(status_path,state)
   require(sha(OUTPUT/args.audit_worker)==preflight['final_audit_worker_sha256'],'Reviewed final auditor source changed during training')
   command=[sys.executable,'-X','utf8','-u',str(OUTPUT/args.audit_worker),'--loop-dir',str(loop_dir),'--frozen-selection',str(freeze_path),
     '--strong-test-comparison',str(comparison),'--output-dir',str(loop_dir)]
   require(run_child(command,loop_dir/'final_audit.log',remaining_for_child(deadline))==0,'New continuation final CPU audit failed')
   audit_path=loop_dir/'adaptive_results_verification.json';audit=read(audit_path,{})
   require(audit.get('passed') is True and audit.get('run_id')==best['run_id'] and audit.get('frozen_selection_sha256')==sha(freeze_path),'Final continuation audit identity/measurement mismatch')
   state.update(status='complete',final_audit_path=str(audit_path),final_audit_sha256=sha(audit_path))
  measured_goal=bool(state.get('status')=='complete' and state.get('competitive_on_development') is True
    and state.get('merged_development_gate',{}).get('retention_gate_passed') is True)
  state.update(objective_complete=measured_goal,objective_status='complete' if measured_goal else 'incomplete',
    next_action='Competitive measured DEV and retention verified' if measured_goal else
      ('Review resource stop then recover exact state; quality objective remains active' if state['requires_resource_review'] else
       'Renew adaptive experiment interval from DEV best; quality objective remains active'))
  state.update(finished_at_utc=now(),total_wall_seconds=time.monotonic()-start)
  write(status_path,state);write(OUTPUT/(args.loop_id+'_report.json'),state)
  return 2 if state['requires_resource_review'] else 0
 except BaseException as exception:
  state.update(status='failed',failure_type=type(exception).__name__,failure_message=str(exception),finished_at_utc=now())
  write(status_path,state);raise
 finally:
  if awake_requested:
   try:state['awake_clear']=prior_utils.keep_awake(False)
   except Exception as exception:
    state['warnings'].append({'kind':'awake_clear_failed','message':str(exception),
      'scope':'Thread-local request also ends when the controller process exits'})
  write(status_path,state);write(OUTPUT/(args.loop_id+'_report.json'),state)

def parser():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--loop-id',default='adaptive_20261010_continuation_03')
 p.add_argument('--source-run-id',default=SOURCE_RUN)
 p.add_argument('--prior-exit-receipt',type=Path,required=True)
 p.add_argument('--cache-validation',type=Path,required=True)
 p.add_argument('--memory-policy-validation',type=Path,required=True)
 p.add_argument('--window-hours',type=float,default=3.)
 p.add_argument('--continuation-seconds',type=float,default=4500.)
 p.add_argument('--challenger-seconds',type=float,default=1500.)
 p.add_argument('--final-reserve-seconds',type=float,default=900.)
 p.add_argument('--stop-grace-seconds',type=float,default=180.)
 p.add_argument('--poll-seconds',type=float,default=5.)
 p.add_argument('--cache-threshold-gb',type=float,default=4.)
 p.add_argument('--reserved-limit-gb',type=float,default=12.)
 p.add_argument('--reserved-observations',type=int,default=3)
 p.add_argument('--commit-limit-fraction',type=float,default=.9)
 p.add_argument('--commit-critical-fraction',type=float,default=.95)
 p.add_argument('--commit-headroom-bytes',type=int,default=2_000_000_000)
 p.add_argument('--commit-observations',type=int,default=3)
 p.add_argument('--regression-ratio',type=float,default=.5)
 p.add_argument('--rate-window-updates',type=int,default=100)
 p.add_argument('--regression-observations',type=int,default=3)
 p.add_argument('--publication-stall-seconds',type=float,default=600.)
 p.add_argument('--audit-worker',default='verify_continued_loop_results_v3.py')
 p.add_argument('--train-only',action='store_true')
 p.add_argument('--validate-only',action='store_true')
 p.add_argument('--preflight-output',type=Path)
 return p

if __name__=='__main__':raise SystemExit(main(parser().parse_args()))
