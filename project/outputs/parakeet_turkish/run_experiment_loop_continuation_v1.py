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
PRIOR_LOOP='adaptive_20261010_recovery_b8_01'
SOURCE_RUN=PRIOR_LOOP+'_b8_quality'
EXIT_RECEIPT_SHA256='acd9e58ee75aca9c03e46a9722d9e77a9aca6418353251f5562491e2e98f8e1b'
FROZEN={
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
 for name in ('fork_training_state_v1.py','compare_strong_test_reuse_v1.py','build_cache_launch_validation_v1.py','profile_cuda_cache_reclaim_v3.py'):
  require((OUTPUT/name).is_file(),'Required versioned helper unavailable: '+name);sources[name]=sha(OUTPUT/name)
 sources[Path(__file__).name]=sha(__file__)
 sources['finalize_lora.py']=sha(OUTPUT/'finalize_lora.py')
 original=read(OUTPUT/'runs'/SOURCE_RUN/'run_config.json',{})
 for name,digest in original.get('code_sha256',{}).items():
  require(sha(scoped(OUTPUT/name,OUTPUT))==digest,'Original training helper changed: '+name)
  sources[name]=digest
 return sources

def system_memory():
 """Win32 physical and commit status, without GPU or process enumeration."""
 if os.name!='nt':return {'available':False,'reason':'Windows native telemetry required'}
 class MEMORYSTATUSEX(ctypes.Structure):
  _fields_=[('length',wintypes.DWORD),('load',wintypes.DWORD)]+[(n,ctypes.c_ulonglong) for n in
    ('total_phys','avail_phys','total_commit','avail_commit','total_virtual','avail_virtual','avail_extended')]
 value=MEMORYSTATUSEX();value.length=ctypes.sizeof(value)
 function=ctypes.WinDLL('kernel32',use_last_error=True).GlobalMemoryStatusEx
 function.argtypes=[ctypes.POINTER(MEMORYSTATUSEX)];function.restype=wintypes.BOOL
 if not function(ctypes.byref(value)):return {'available':False,'error':ctypes.get_last_error()}
 used=value.total_commit-value.avail_commit
 return {'available':True,'observed_at_utc':now(),'physical_load_percent':value.load,
  'physical_available_bytes':value.avail_phys,'commit_used_bytes':used,'commit_limit_bytes':value.total_commit,
  'commit_available_bytes':value.avail_commit,'commit_fraction':used/value.total_commit}

def evidence_file(entry):
 path=scoped(entry['path']);require(path.is_file() and sha(path)==entry['sha256'],'Measured evidence file changed: '+str(path));return path

def validate_cache_gpu(path,config):
 """Inspect causal A/B and actual production hook; do not infer equivalence."""
 manifest=read(scoped(path),{})
 require(manifest.get('schema')==1 and manifest.get('kind')=='cuda_cache_reclaim_launch_validation_v1',
   'Small linked causal/actual-hook validation manifest required')
 probe_path=evidence_file(manifest['probe']);probe=read(probe_path,{})
 probe_worker=evidence_file(manifest['probe_worker'])
 require(probe.get('status')=='complete' and probe.get('passed') is True and probe.get('solo_gpu') is True
   and probe.get('matched_sources_equal') is True and probe.get('shared_prefix_updates',0)>=2,
   'Bounded causal A/B TRAIN probe must complete with matched sources')
 require(probe.get('allocator_hard_cap_gb')==13 and probe.get('maximum_reserved_gb_guard')==12
   and probe.get('maximum_commit_fraction')==.9 and probe.get('probe_version')==3 and probe.get('cleanup_threshold_gb')==6,
   'Actual6GB causal probe requires the documented13GB cap and live safety guards')
 identity=probe['identity']
 require(sha(probe_worker)==identity['probe_sha256'],'Actual causal probe source differs from measured identity')
 require(identity['source_step']==907 and identity['source_run_id']==SOURCE_RUN
   and identity['helper_sha256']==FROZEN['cuda_cache_reclaim_v1.py']
   and identity['launcher_sha256']==FROZEN['launch_with_cache_reclaim_v1.py'],
   'Causal A/B checkpoint/helper/launcher identity differs')
 require(probe['probe_sha256_after']==identity['probe_sha256']
   and probe['checkpoint_sha256_after']==identity['checkpoint_sha256'] and probe.get('original_run_mutated') is False,
   'Causal probe source/checkpoint changed')
 require(sha(scoped(identity['checkpoint_path']))==identity['checkpoint_sha256']
   and identity['source_sha256']==config['code_sha256'],'Actual measured source907 checkpoint/training-code identity differs')
 count=probe['shared_prefix_updates'];control=probe['branches']['control'];reclaim=probe['branches']['reclaim']
 require(control['completed_updates']==len(control['updates'])==count and reclaim['completed_updates']==len(reclaim['updates'])>=count,
   'Measured A/B branch update coverage differs')
 require(probe.get('intervention_requested_updates_completed') is True and reclaim['completed_updates']>=100,
   'Reviewed6GB cleanup branch must complete at least100 diverse TRAIN updates without its memory safety stop')
 for before,after in zip(control['updates'],reclaim['updates']):
  require(before['source_ids']==after['source_ids'],'Causal A/B TRAIN sampling differs')
  require(set(before['component_losses'])==set(after['component_losses'])
    and all(math.isfinite(v) for v in [*before['component_losses'].values(),*after['component_losses'].values()]),
    'Causal A/B objective components must be finite with identical terms')
 for update in reclaim['updates']:
  require(update['system'].get('available') is True and 0<=update['system']['commit_fraction']<=.9
    and update['allocator']['reserved_bytes.all.current']<=12e9,
    'Every measured6GB intervention update must remain inside the actual memory safety bounds')
 require(set(probe['comparisons'])=={'1',str(count)},'Matched gradients/weights must cover first and last common A/B updates')
 for comparison in probe['comparisons'].values():
  require(comparison['sampler_rng_equal'] is True and comparison['python_cpu_cuda_rng_equal'] is True
    and comparison['quality_or_numerical_equivalence_waiver'] is False,'Causal A/B sampler/RNG changed or numerical limits waived')
  for category in ('gradients','weights'):
   require(all(math.isfinite(comparison[category][k]) for k in ('max_absolute_delta','relative_l2','cosine')),
     'Actual matched tensors must have finite descriptive numerical comparisons')
   require(comparison[category]['max_absolute_delta']>=0 and comparison[category]['relative_l2']>=0
     and -1<=comparison[category]['cosine']<=1.000000000001,'Actual tensor error metrics must have valid ranges')
 evidence_file({'path':probe['full_capture_path'],'sha256':probe['full_capture_sha256']})
 production=manifest['production_hook'];cache_path=evidence_file(production['cache_provenance'])
 memory_path=evidence_file(production['memory_provenance'])
 io_path=evidence_file(production['io_provenance']);history_path=evidence_file(production['training_history'])
 config_path=evidence_file(production['run_config']);actual_config=read(config_path,{})
 run_id=actual_config['run_id'];initial=production['initial_step'];completed=production['completed_steps']
 require(initial==907 and completed>=initial+2 and actual_config['config_identity']==config['config_identity'],
   'Actual production hook smoke must start from identical907 state/config')
 require(history_path.parent.name==run_id and config_path.parent==history_path.parent,'Actual production smoke config/history run differs')
 smoke_receipt_path=evidence_file(manifest['smoke_fork_receipt']);smoke_receipt=read(smoke_receipt_path,{})
 require(smoke_receipt.get('kind')=='cache_guard_frozen_worker_smoke_fork_v2'
   and smoke_receipt.get('source_run_id')==SOURCE_RUN and smoke_receipt.get('smoke_run_id')==run_id
   and manifest.get('smoke_fork_exact_CPU_checks',0)>=180000
   and manifest.get('independent_actual_tensor_and_snapshot_recomputation') is True,
   'Real hook must use independently verified full907/801 smoke forks and actual captured tensors/snapshots')
 require(manifest.get('builder_source_sha256')==sha(OUTPUT/'build_cache_launch_validation_v1.py'),
   'Actual independent GPU proof builder source changed')
 roles={item['role']:item for item in smoke_receipt['forks']}
 require(set(roles)=={'latest','best'},'Both exact smoke state forks required')
 for role,record in roles.items():
  require(record['source_step']==(907 if role=='latest' else 801) and record['only_changed_checkpoint_field']=='run_id'
    and sha(scoped(record['source_path']))==record['source_sha256']
    and sha(scoped(record['fork_path']))==record['fork_sha256'], 'Archived real-hook state fork source/bytes changed')
 warm=actual_config['warm_start'];latest=roles['latest']
 require(warm['mode']=='exact_resume' and warm['source_run_id']==run_id and warm['source_step']==907
   and warm['sha256']==latest['fork_sha256']
   and Path(warm['path']).resolve() in (Path(latest['runtime_path']).resolve(),Path(latest['fork_path']).resolve()),
   'Real frozen worker must actually load the exact907 archived smoke state')
 history=read(history_path,[]);tail=normalized_tail(history,initial)
 require(tail is not None and tail['completed_updates']==completed,'Actual hook smoke completed history/count mismatch')
 proof=cache_evidence(cache_path,io_path,run_id,initial,memory_path)
 require(proof['cache']['eligible_update_calls']==completed-initial,'Actual run:539 scope must cover every new smoke update')
 for actual,causal in zip(tail['new_training_records'],control['updates']):
  require(actual['sampled_source_ids']==[s for group in causal['source_ids'] for s in group],
    'Actual frozen worker sampling differs from matched causal907 prefix')
 numerical=manifest['strict_numerical_gate']
 limits={'gradient_relative_l2':.0005,'gradient_min_cosine':.9999995,
   'component_absolute_delta':.0001,'weights_relative_l2':.000001}
 require(numerical.get('waived') is False and isinstance(numerical.get('passed'),bool)
   and numerical.get('thresholds')==limits,'Cache-only numerical thresholds must equal the original strict gradient/loss limits; no waiver')
 loss_delta=max(abs(before['component_losses'][key]-after['component_losses'][key])
   for before,after in zip(control['updates'],reclaim['updates']) for key in before['component_losses'])
 require(loss_delta==probe['max_component_absolute_delta'],'Actual cache A/B maximum component delta differs')
 passed=loss_delta<limits['component_absolute_delta'] and all(
   v['gradients']['relative_l2']<limits['gradient_relative_l2']
   and v['gradients']['cosine']>limits['gradient_min_cosine']
   and v['weights']['relative_l2']<limits['weights_relative_l2'] for v in probe['comparisons'].values())
 require(numerical['passed']==passed,'Declared cache-only strict numerical gate differs from actual measured tensors/components')
 numerical={**numerical,'scope':'Cache A/B measured global gradients, trainable weights and objective components only',
   'recomputed_max_component_absolute_delta':loss_delta,'encoder_features_measured':False,
   'historical_full_BF16_original_vs_v2_equivalence_gate_passed':False}
 return {'validation_path':str(scoped(path)),'validation_sha256':sha(path),'causal_probe_path':str(probe_path),
   'causal_probe_sha256':sha(probe_path),'actual_hook_run_id':run_id,'actual_hook_initial_step':initial,
   'actual_hook_completed_steps':completed,'actual_hook_cache_provenance_path':str(cache_path),
   'actual_hook_cache_provenance_sha256':sha(cache_path),'actual_hook_memory_provenance_path':str(memory_path),
   'actual_hook_memory_provenance_sha256':sha(memory_path),'allocator_cap_bytes':13_000_000_000,
   'smoke_fork_receipt_path':str(smoke_receipt_path),'smoke_fork_receipt_sha256':sha(smoke_receipt_path),
   'independent_builder_sha256':manifest['builder_source_sha256'],
   'strict_numerical_gate':numerical,'training_math_source_unchanged':True,
   'rng_and_sampling_preserved':True,'real_frozen_optimizer_hook_executed':True}

def validate_entry(args):
 safe_id(args.loop_id);require(args.source_run_id==SOURCE_RUN,'This continuation forks the verified907 B8 source')
 require(0<args.window_hours<=3 and 0<args.final_reserve_seconds<=1800 and 30<=args.stop_grace_seconds<=300,'Bounded renewed window/reserve/grace required')
 require(all(math.isfinite(v) and v>0 for v in (args.continuation_seconds,args.challenger_seconds)), 'Stage quotas must be finite and positive')
 require(1<=args.poll_seconds<=30 and args.reserved_limit_gb==12 and args.reserved_observations==3
   and args.commit_limit_fraction==.9 and args.cache_threshold_gb==6,'Reviewed memory watchdog/cache thresholds required')
 require(0<args.regression_ratio<=.5 and args.rate_window_updates>=50 and args.regression_observations>=3,'Conservative throughput-regression guard required')
 require(args.publication_stall_seconds==600,'Reviewed all-phase publication-stall allowance is600seconds')
 prior_dir=OUTPUT/'runs'/PRIOR_LOOP
 audit_path=prior_dir/'adaptive_results_verification.json'
 receipt_path=scoped(args.prior_exit_receipt);receipt=read(receipt_path,{})
 require(sha(receipt_path)==EXIT_RECEIPT_SHA256 and receipt.get('schema')==1 and receipt.get('kind')=='root_owned_predecessor_exit_receipt_v1'
   and receipt.get('predecessor_loop_id')==PRIOR_LOOP and receipt.get('root_observed_owned_session_completed') is True
   and receipt.get('owned_exec_session_id')==13816 and receipt.get('root_observed_exit_code')==0
   and receipt.get('predecessor_status')=='complete' and receipt.get('predecessor_audit_passed') is True,
   'Parent-owned old session13816 complete exit receipt required; a PID guess is insufficient')
 require(receipt.get('predecessor_status_sha256')==sha(prior_dir/'loop_status.json')
   and receipt.get('predecessor_audit_sha256')==sha(audit_path)
   and Path(receipt['predecessor_status_path']).resolve()==(prior_dir/'loop_status.json').resolve()
   and Path(receipt['predecessor_audit_path']).resolve()==audit_path.resolve(),
   'Prior process-exit receipt must bind terminal controller and audited artifacts')
 require(stamp(receipt['observed_at_utc'])<=datetime.now(timezone.utc),'Process exit receipt timestamp is invalid')
 config=read(OUTPUT/'runs'/SOURCE_RUN/'run_config.json',{})
 identity=config.get('config_identity',{}).get('training_arguments',{})
 require(set(identity)==set(IDENTITY_KEYS) and identity['batch_size']==8 and identity['accumulation']==2
   and identity['epochs']==3 and identity['steps'] is None and identity['warmup_steps']==25
   and identity['tdt_implementation']=='v2' and identity['chunk_size']==16
   and identity['no_encoder_checkpoint'] is False,'Source exact configuration must be the reviewed B8/3epochs/9611updates')
 history=read(OUTPUT/'runs'/SOURCE_RUN/'training_history.json',[])
 selection=read(OUTPUT/'runs'/SOURCE_RUN/'training_selection.json',None) or read(OUTPUT/'runs'/SOURCE_RUN/'selection.json',{})
 require(len(history)==907 and history[-1]['step']==907 and selection.get('completed_steps')==907
   and selection.get('selected_step')==801,'Source exact continuation needs full907/latest and801/best')
 require(config['steps']==9611 and selection.get('accepted') is True,'Source best DEV-selected state must qualify')
 loop_dir=OUTPUT/'runs'/args.loop_id
 require(loop_dir.exists() and {p.name for p in loop_dir.iterdir()}=={'fork_snapshot','fork_snapshot_independent_verification.json','fork_initial_states_archive.json'},
   'Renewed loop may contain only its three owned immutable fork prerequisites before launch')
 fork_path=loop_dir/'fork_snapshot/fork_manifest.json';fork=read(fork_path,{})
 require(fork.get('passed') is True and fork.get('status')=='complete' and fork.get('kind')=='exact_training_state_run_id_fork_v1'
   and fork.get('source_run_id')==SOURCE_RUN and fork.get('target_run_id')==args.loop_id+'_resume_B8'
   and fork.get('source_completed_step')==907 and fork.get('inherited_selected_step')==801
   and fork.get('only_changed_path')=='run_id' and fork.get('inherited_config_identity')==config['config_identity'],
   'Pre-created exact907/801 fork manifest differs from source/target/config')
 require(fork.get('predecessor_exit_receipt_sha256')==EXIT_RECEIPT_SHA256
   and fork.get('source_verification_sha256')==sha(audit_path),'Fork predecessor receipt/audit linkage changed')
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
   require(sha(path)==value['sha256'] and value['step']==(907 if role=='latest' else 801),'Fork/source checkpoint bytes/step changed')
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
   'prior_exit_receipt_path':str(receipt_path),'prior_exit_receipt_sha256':sha(receipt_path),
   'cache_validation':validation,'fork_manifest_path':str(fork_path),'fork_manifest_sha256':sha(fork_path),
   'initial_fork_archive_path':str(archive_path),'initial_fork_archive_sha256':sha(archive_path),
   'old_deadline_at_utc':receipt['predecessor_deadline_unchanged'],'renewed_window_hours':args.window_hours,
   'source_config':config,'source_selection':selection,'challengers':list(CHALLENGERS),
   'first_stage_policy':'New run-ID fork only; all nested weights/Adam/sampler/RNG/history identical through907',
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
  self.args=args;self.start_step=start_step;self.reserved_streak=0;self.rate_streak=0;self.last_rate_step=None;self.rate_reference=None
 def observe(self,history,memory):
  observations={'system_memory':memory};reason=None
  if not memory.get('available'):return 'memory_telemetry_unavailable',observations
  if memory['commit_fraction']>self.args.commit_limit_fraction:reason='system_commit_above90percent'
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

def cache_evidence(path,io_path,run_id,start_step,memory_path=None):
 evidence=read(path,{});io=read(io_path,{})
 require(evidence.get('status')=='complete' and evidence.get('active_process_attachment') is False
   and evidence.get('frozen_worker_sha256')==FROZEN['train_lora_batched.py'],'Cache launcher must complete unchanged frozen worker')
 for key,name in [('launcher_sha256','launch_with_cache_reclaim_v1.py'),('cache_helper_sha256','cuda_cache_reclaim_v1.py'),
     ('frozen_io_launcher_sha256','launch_with_io_retry_v1.py'),('frozen_io_helper_sha256','atomic_io_retry_v1.py')]:
  require(evidence.get(key)==FROZEN[name] and evidence.get(key+'_after')==FROZEN[name],'Cache/IO source changed: '+name)
 cache=evidence['cache_reclamation'];require(cache['worker_sha256']==cache['worker_sha256_after']==FROZEN['train_lora_batched.py']
   and cache['caller_function']=='run' and cache['caller_line']==539 and cache['threshold_reserved_bytes']==6_000_000_000
   and cache['failed_reclamations']==0,'Scoped cache policy/evidence incorrect')
 require(io.get('status')=='complete' and io.get('training_semantics_changed') is False,'Nested IO adapter must complete without math changes')
 arguments=io.get('arguments',[]);require('--run-id' in arguments and arguments[arguments.index('--run-id')+1]==run_id,'Nested IO run identity mismatch')
 require(cache['eligible_update_calls']>=0 and cache['reclamations']>=0,
   'Optimizer update/reclamation counts must be nonnegative')
 compact={key:cache[key] for key in ('caller_function','caller_line','threshold_reserved_bytes','eligible_update_calls',
   'reclamations','failed_reclamations','worker_sha256','worker_sha256_after','cleanup_seconds_in_training_timing')}
 memory_ref={}
 if memory_path is not None:
  memory=read(memory_path,{})
  require(memory.get('status')=='complete' and memory.get('kind')=='memory_guard_training_launcher_v2'
    and memory.get('cleanup_threshold_bytes')==6_000_000_000
    and memory.get('cap_bytes')==13_000_000_000 and memory.get('allocator_cap_applied') is True
    and memory.get('training_math_changed') is False and memory.get('backend_flags_changed') is False
    and memory.get('RNG_sampler_optimizer_changed') is False and memory.get('global_driver_settings_changed') is False,
    'Production memory guard must apply the reviewed13GB per-process cap without changing training math')
  require(15e9<=memory['physical_cuda_device_bytes']<=18e9
    and memory['per_process_allocator_fraction']==13e9/memory['physical_cuda_device_bytes'],
    'Actual16GB-class device and exact13GB allocator fraction differ')
  for key,name in [('memory_launcher_sha256','launch_with_memory_guard_v2.py'),
    ('cache_launcher_sha256','launch_with_cache_reclaim_v1.py'),('cache_helper_sha256','cuda_cache_reclaim_v1.py')]:
   require(memory.get(key)==memory.get(key+'_after')==FROZEN[name],'Memory-cap launcher source changed: '+name)
  require(Path(memory['cache_provenance_path']).resolve()==Path(path).resolve()
    and Path(memory['io_provenance_path']).resolve()==Path(io_path).resolve(),'Memory/cache/IO evidence linkage differs')
  require(memory['arguments']==arguments or memory['arguments']==['--',*arguments],'Memory-cap worker argument identity differs')
  memory_ref={'memory_guard_provenance_path':str(memory_path),'memory_guard_provenance_sha256':sha(memory_path),
    'allocator_cap_bytes':memory['cap_bytes'],'per_process_allocator_fraction':memory['per_process_allocator_fraction']}
 return {'cache_provenance_path':str(path),'cache_provenance_sha256':sha(path),'io_provenance_path':str(io_path),
  'io_provenance_sha256':sha(io_path),'cache':compact,'inherited_completed_step':start_step,**memory_ref}

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
  target=scoped(args.preflight_output or OUTPUT/'continuation_v1_preflight.json')
  require(not target.exists(),'Preflight output already exists; preserve reviewed evidence')
  write(target,{k:v for k,v in preflight.items() if k not in ['source_config','source_selection']})
  print(json.dumps({'passed':True,'gpu_used':False,'preflight':str(target)}));return 0
 memory=system_memory();require(memory.get('available') and memory['commit_fraction']<=.9,'System commit pressure must be resolved before any new GPU child')
 loop_dir=OUTPUT/'runs'/args.loop_id
 start=time.monotonic();start_utc=datetime.now(timezone.utc);deadline=start+args.window_hours*3600
 training_deadline=deadline-args.final_reserve_seconds-args.stop_grace_seconds
 state={'schema':1,'kind':'renewed_finite_B8_continuation_v1','loop_id':args.loop_id,'status':'validating_preserved_fork',
  'started_at_utc':start_utc.isoformat(),'deadline_at_utc':(start_utc+timedelta(hours=args.window_hours)).isoformat(),
  'window_hours':args.window_hours,'renewal_authorization':'Explicit user request to continue adaptive experiments after prior export/verification',
  'prior_loop_id':PRIOR_LOOP,'prior_deadline_at_utc':preflight['old_deadline_at_utc'],'prior_exit_receipt':preflight['prior_exit_receipt_path'],
  'preflight':{k:v for k,v in preflight.items() if k not in ['source_config','source_selection']},
  'selection_split':'FLEURS validation only','heldout_policy':'No TEST parsing during trials or candidate selection; final frozen winner only. Existing fixed Whisper TEST reused read-only for final reporting.',
  'runtime':{'batch_size':8,'accumulation':2,'effective_batch':16,'threshold_reserved_gb':6,'allocator_cap_gb':13,
    'memory_guard_launcher_sha256':FROZEN['launch_with_memory_guard_v2.py'],'strict_BF16_numeric_equivalence_passed':False},
  'watchdog_policy':{'reserved_limit_gb':12,'reserved_observations':3,'commit_limit_fraction':.9,'rate_ratio':args.regression_ratio,
    'rate_window_updates':args.rate_window_updates,'rate_observations':args.regression_observations,
    'publication_stall_seconds':args.publication_stall_seconds,'resource_stop_blocks_all_further_GPU':True},
  'stage_plan':{'exact_continuation_cap_seconds':args.continuation_seconds,'challenger_cap_seconds':args.challenger_seconds,
    'challengers':list(CHALLENGERS),'winning_extension_remaining_training_budget':True,'minimum_fresh_TR_epochs_for_comparison':.25},
  'final_reserve_seconds':args.final_reserve_seconds,'stop_grace_seconds':args.stop_grace_seconds,'trials':[],
  'requires_resource_review':False,'warnings':[]}
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
    and fork.get('source_completed_step')==907 and fork.get('inherited_selected_step')==801,'Exact fork manifest incomplete/wrong-run')
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
   start_step=907 if exact else 0
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
   command=[sys.executable,'-X','utf8','-u',str(OUTPUT/'launch_with_memory_guard_v2.py'),
     '--provenance',str(io_path),'--cache-provenance',str(cache_path),'--memory-provenance',str(memory_path),
     '--cap-gb','13','--threshold-gb','6','--',*argv]
   entry={'name':name,'run_id':run_id,'status':'running','command':command,'started_at_utc':now(),'quota_seconds':quota,
     'initial_step':start_step,'inherited_completed_step':start_step,'start_weights':str(start_input),
     'start_weights_sha256':sha(start_input),'start_weights_provenance':input_identity,
     'cache_provenance_path':str(cache_path),'io_provenance_path':str(io_path),'memory_guard_provenance_path':str(memory_path),
     'optimizer_policy':'exact907optimizer/sampler/RNG continuation' if exact else 'freshAdam DEV-best warmstart',
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
      reason,observation=watch.observe(history,system_memory())
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
     adequate=bool(exact) or actual_epochs>=.25 or stop_requested in ('stage_wall_quota','system_commit_above90percent','reserved_above12GB_three_observations','sustained_training_rate_regression')
     entry.update(actual_turkish_epochs=actual_epochs,minimum_exposure_met=bool(exact) or actual_epochs>=.25,
        comparison_exception=None if adequate and (exact or actual_epochs>=.25) else stop_requested)
     require(adequate,'Fresh stage lacks quarter-epoch DEV comparison exposure and no deadline/resource exception')
     entry['selection_provenance']=provenance
     if selected['turkish_validation_wer']<best['wer']:
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
      'source_run_id':SOURCE_RUN,'inherited_completed_step':907,'inherited_selected_step':801}}
  freeze_path=loop_dir/'frozen_final_selection.json';write(freeze_path,frozen)
  state.update(status='resource_review_required' if state['requires_resource_review'] else 'development_selection_frozen',
    frozen_selection_sha256=sha(freeze_path),best_development_candidate=dict(best))
  write(status_path,state)
  if not state['requires_resource_review'] and not args.train_only:
   require(current_sources()==preflight['source_sha256'],'Runtime/controller sources changed during renewed training')
   if best['run_id']==SOURCE_RUN:
    final=read(OUTPUT/'runs'/SOURCE_RUN/'selection.json',{})
    require(final.get('merged') is True and final.get('heldout_tests_evaluated') is True,'Preserved fallback must reuse complete original export')
    state['reused_prior_finalization']=True
   else:
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
 p.add_argument('--loop-id',default='adaptive_20261010_continuation_01')
 p.add_argument('--source-run-id',default=SOURCE_RUN)
 p.add_argument('--prior-exit-receipt',type=Path,required=True)
 p.add_argument('--cache-validation',type=Path,required=True)
 p.add_argument('--window-hours',type=float,default=3.)
 p.add_argument('--continuation-seconds',type=float,default=4500.)
 p.add_argument('--challenger-seconds',type=float,default=1500.)
 p.add_argument('--final-reserve-seconds',type=float,default=900.)
 p.add_argument('--stop-grace-seconds',type=float,default=180.)
 p.add_argument('--poll-seconds',type=float,default=5.)
 p.add_argument('--cache-threshold-gb',type=float,default=6.)
 p.add_argument('--reserved-limit-gb',type=float,default=12.)
 p.add_argument('--reserved-observations',type=int,default=3)
 p.add_argument('--commit-limit-fraction',type=float,default=.9)
 p.add_argument('--regression-ratio',type=float,default=.5)
 p.add_argument('--rate-window-updates',type=int,default=100)
 p.add_argument('--regression-observations',type=int,default=3)
 p.add_argument('--publication-stall-seconds',type=float,default=600.)
 p.add_argument('--audit-worker',default='verify_continued_loop_results_v1.py')
 p.add_argument('--train-only',action='store_true')
 p.add_argument('--validate-only',action='store_true')
 p.add_argument('--preflight-output',type=Path)
 return p

if __name__=='__main__':raise SystemExit(main(parser().parse_args()))
