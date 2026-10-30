"""Finite expanded-data training window; fresh rank8 control, matched rank16 option.

Metadata-only Turkish inputs remove unnecessary frozen-encoder preparation.
The completed owned full PCM audit is reused only with fresh input fingerprints.
A stopped preparation worker did not perform any optimizer update.
Only development scores choose candidates. Short250-update screens are diagnostic
comparisons with explicit exposure uncertainty, not claims of competitive quality.
This worker never evaluates TEST or
publishes an export. An incomplete three-hour window is not objective completion.
New data uses new optimizer/sampler state, retaining the selected acoustic model.
GPU ownership belongs to the calling root; this controller owns one child at a time.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import re
from pathlib import Path
import subprocess
import sys
import time

from atomic_io_retry_v1 import read_json as read, write_json as write
import run_experiment_loop_recovery_v2 as prior
from resource_watchdog_v2 import ResourceGuard, sample_memory, worker_process_commit_reason

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parent.parent
SOURCE_RUN = 'adaptive_20261010_continuation_03_resume_B8'
RECEIPT_SHA = '0318e73a9ea151cfe5c8381a6ad3a0ebe90ea8ab82f03a54012710aee0b42275'
FROZEN = {
 'batched_lora_tr_metadata_v1.py':'b78236a851a9739c30423bb00863973d571cf9fee948be4473bcd1cb34f05075',
 'tr_frame_metadata_v1.py':'e74bc92b4c089ca92ca408397267eecfb665c51b43874909f6367def64d760af',
 'training_input_audit_reuse_v1.py':'f8cd68791b54e061e4012cfe2f715c19cb390c14229a1a76ee18fb5d2e60d07f',
 'train_lora_batched_data_v6.py':'2a2ef388faee61ad8b8cd1c28dc6827e0b61589c8709a42af19241bc7ded02ab',
 'public_turkish_data_policy_v3.py':'08eee3896bf8571c9e97dae1be688ecef7766816c019741dd7631d70f640ce41',
 'launch_with_cache_reclaim_v7.py':'c0003f97063a5fbe889bac764889f416729c8ca49fe549bbf40481aea4482b81',
 'resource_watchdog_v2.py':'64d1393e1e45cf65dac14f78ac955331c91b4a3f0eafbf7535196e2b61778641',
 'paired_windows_commit_telemetry_v1.py':'2ae9f2cdfc388a294914e000c60b62fe09cddac147700c5762851d626a95a683',
 'cuda_cache_reclaim_v1.py':'99ec5974fcd4714fa6fa341eb5cde70dc817085c997bd564c2b465aaaf98a543',
 'launch_with_io_retry_v4.py':'913343ac111a99c1207761c5d73543180e9627586d7e293cd2bfa400b27becf9',
 'atomic_io_retry_v1.py':'2d3fe379d51021fd15679e562c71286f7f9501180d28dde77893e8f72c80a1b0',
 'upgrade_lora_rank_v1.py':'360d446637152bf87ce32117aa5cb16d5ef80138c6951552129a4dfc6f69ef3b',
 'run_experiment_loop_recovery_v2.py':'c16d0c61e0c4aa48d4bb93ef5ef8b8192032581d264a3be8fee369f4077a1213',
}
WRAPPER = 'launch_with_memory_guard_v10.py'
WRAPPER_SHA = 'f105c3ad2e25543e145bb9b1b964f8f51cc36a943423e920b8fce6196335db8d'
IDENTITY_KEYS = ('head_lr','lora_lr','new_token_lr_multiplier','replay_kl','duration_kl',
 'feature_distillation','bucket_size','seed','max_tokens','chunk_size','no_encoder_checkpoint','phonetic_sh_donors')


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path): return prior.sha(path)
def require(value,message):
 if not value: raise RuntimeError(message)
def stamp(value):
 result=datetime.fromisoformat(value)
 require(result.tzinfo is not None,'Timezone-aware UTC timestamp required')
 return result
def scoped(value):
 value=Path(value).resolve(); require(value.is_relative_to(ROOT),'Workspace-only artifact required');return value
def safe_id(value):
 require(bool(re.fullmatch(r'[A-Za-z0-9_-]+',value)),'Safe run identifier required');return value
def record(path):return {'path':str(scoped(path)),'sha256':sha(path)}
def source_hashes():
 for name,digest in {**FROZEN,WRAPPER:WRAPPER_SHA}.items():
  require(sha(OUTPUT/name)==digest,'Reviewed runtime changed: '+name)
 return {**FROZEN,WRAPPER:WRAPPER_SHA,Path(__file__).name:sha(__file__)}
def revalidate(sources):
 for name,digest in sources.items():require(sha(OUTPUT/name)==digest,'Source changed during window: '+name)


def preflight(args):
 safe_id(args.loop_id)
 for value in (args.window_hours,args.poll_seconds,args.prepare_timeout_seconds,args.cooperative_grace_seconds):
  require(math.isfinite(value) and value>0,'Finite positive window/cadence required')
 require(args.window_hours<=3 and type(args.screen_updates) is int and 25<args.screen_updates<=500
   and args.poll_seconds>=5,'Reviewed window<=3h, short26..500-update screens, fresh>=5s polls')
 require(args.train_only is True,'This version is explicitly training-only; TEST/export needs a separate frozen winner audit')
 require(args.source_run_id==SOURCE_RUN,'This version starts from the reviewed1871 predecessor')
 require(not (OUTPUT/'runs'/args.loop_id).exists(),'Use a fresh loop namespace')
 failed_path=OUTPUT/'runs/adaptive_20261011_public_data_01/loop_status.json';failed=read(failed_path,{})
 require(failed.get('status')=='failed_review_required' and failed.get('active_run_id') is None
   and len(failed.get('trials',[]))==1 and failed['trials'][0].get('exit_code')==1
   and failed['trials'][0].get('published_steps')==0 and failed.get('awake_clear',{}).get('enabled') is False,
   'Preserved controller4 dispatch failure must have a closed child and no published optimizer update')
 require(sha(OUTPUT/'run_experiment_loop_continuation_v4.py')=='db0c467b060704826cadd3e9aca6f50a8d02f4ff6df06c36ea5fc81befc2db27',
   'Preserved failed controller4 source changed')
 receipt_path=scoped(args.prior_exit_receipt)
 require(sha(receipt_path)==RECEIPT_SHA,'Authoritative owned-exit receipt changed')
 receipt=read(receipt_path,{})
 require(receipt['kind']=='root_owned_resource_review_exit_receipt_v1'
   and receipt['root_observed_owned_session_completed'] is True
   and receipt['root_observed_exit_code']==1 and receipt['child_exit_code']==0
   and receipt['predecessor_run_id']==SOURCE_RUN
   and receipt['child_durable_step']==receipt['child_selected_step']==1871
   and receipt['predecessor_training_checkpoint_review_passed'] is True,'Closed reviewed1871 child required')
 for stem in ('predecessor_status','predecessor_controller','child_run_config','child_training_history','child_selection','source_best','source_latest'):
  require(sha(scoped(receipt[stem+'_path']))==receipt[stem+'_sha256'],'Predecessor artifact changed: '+stem)
 selected,provenance=prior.validate_selection(SOURCE_RUN)
 require(selected['selected_step']==1871 and selected['turkish_validation_wer']==receipt['latest_dev_wer'],'Reviewed source DEV selection differs')
 config=read(receipt['child_run_config_path'],{})
 require(config['rank']==8 and config['alpha']==16 and config['batch_size']==8 and config['accumulation']==2,'Reviewed source layout/runtime differs')
 proof_path=scoped(args.data_proof);manifest=scoped(args.tr_manifest);proof=read(proof_path,{})
 require(proof['status'].startswith('CPU_TDT_tokenizer_verified') and proof['tokenizer_decode_group_tokens'] is False
   and proof['vocabulary_changed'] is False and proof['all_protected_original_files_unchanged'] is True,'Corrected TDT ungrouped input proof required')
 require(manifest==scoped(proof['CPU_tokenizer_accepted_manifest_path'])
   and sha(manifest)==proof['CPU_tokenizer_accepted_manifest_sha256'],'Exact corrected TRAIN manifest required')
 require(proof['CPU_tokenizer_accepted_clips']==74883,'Full corrected public-v3 pool required')
 label_path=OUTPUT/'tr_public_v3_tokenizer_acceptance.json';label=read(label_path,{})
 require(label['passed'] is True and label['group_tokens'] is False
   and label['tokenizer_semantic_sha256']==config['config_identity']['extended_tokenizer_sha256'],'Corrected target proof/tokenizer identity differs')
 readiness=scoped(args.data_readiness);ready=read(readiness,{})
 require(ready.get('passed') is True and ready['accepted_manifest_sha256']==sha(manifest)
   and ready['accepted_clips']==proof['CPU_tokenizer_accepted_clips'],'Public-data runtime CPU readiness must pass and bind actual full input')
 for path,digest in ready['artifacts_sha256'].items():require(sha(scoped(path))==digest,'CPU readiness source/evidence changed: '+path)
 metadata=scoped(args.tr_frame_metadata);metadata_proof=scoped(args.tr_frame_metadata_proof)
 require(metadata==scoped(ready['TR_metadata_path']) and sha(metadata)==ready['TR_metadata_sha256'],'Exact no-feature TR metadata required')
 frame=read(metadata_proof,{})
 require(frame.get('passed') is True and frame['CPU_only'] is True and frame['clips']==74883
   and frame['metadata_sha256']==sha(metadata) and frame['manifest_sha256']==sha(manifest)
   and frame['cache_files_written']==0,'Full actual frame/target metadata CPU proof differs')
 reuse=scoped(args.reuse_input_audit);reuse_proof=read(reuse,{})
 require(reuse==scoped(ready['optional_input_audit_reuse_receipt_path'])
   and sha(reuse)==ready['optional_input_audit_reuse_receipt_sha256']
   and reuse_proof['passed'] is True and reuse_proof['observed_completed_full_audit'] is True
   and reuse_proof['actual_owned_worker_pid']==5212 and reuse_proof['source_loop_id']=='adaptive_20261011_public_data_02',
   'Actual completed owned full audit receipt required; fingerprints rechecked inside worker')
 gpu_path=scoped(args.metadata_gpu_proof);gpu=read(gpu_path,{})
 require(gpu['kind']=='TR_metadata_actual_GPU_masks_v1' and gpu['status']=='complete' and gpu['passed'] is True
   and gpu['TRAIN_only'] is True and gpu['physical_batch']==8 and gpu['training_updates']==0 and gpu['cache_files_written']==0
   and gpu['manifest_sha256']==sha(manifest) and gpu['metadata_sha256']==sha(metadata)
   and gpu['CPU_frame_proof_sha256']==sha(metadata_proof) and len(gpu['source_ids'])==16,
   'Actual diverse sixteen-TRAIN B8 frontend/encoder length smoke required')
 for name,digest in gpu['source_sha256'].items():require(sha(OUTPUT/name)==digest,'GPU length smoke source changed: '+name)
 dispatch_path=OUTPUT/'data_v6_actual_io_dispatch_cpu_verification.json';dispatch=read(dispatch_path,{})
 require(dispatch.get('passed') is True
   and dispatch['trainer_sha256']==FROZEN['train_lora_batched_data_v6.py']
   and dispatch['io_launcher_sha256']==FROZEN['launch_with_io_retry_v4.py']
   and dispatch['cache_launcher_sha256']==FROZEN['launch_with_cache_reclaim_v7.py']
   and dispatch['memory_launcher_sha256']==WRAPPER_SHA
   and 'actual_whole_nested_dispatch_reached_worker_run' in dispatch['checks']
   and 'exact_original_serializer_adapter_still_preserved' in dispatch['checks'],
   'Actual metadata nested I/O/cache/memory parser/run and original serializer CPU proof required')
 pause_path=scoped(args.preparation_exit_receipt);pause=read(pause_path,{})
 require(sha(pause_path)=='43352e1c276eaf91090b0526d752e356c678758ca8e84a3d4be3ad878ef97d15',
   'Authoritative stopped-before-optimizer receipt changed')
 require(pause['kind']=='root_owned_pre_optimizer_preparation_exit_receipt_v1'
   and pause['predecessor_loop_id']=='adaptive_20261011_public_data_02'
   and pause['root_observed_owned_session_completed'] is True and pause['root_observed_exit_code']==1
   and pause['owned_exec_session_id']==95658 and pause['actual_worker_pid']==5212
   and pause['worker_absent_confirmed'] is True and pause['published_optimizer_steps']==0,
   'Root-owned stopped zero-update predecessor evidence required; original nonterminal JSON remains preserved')
 absence=pause['worker_absence_observation']
 require(absence['pid']==5212 and absence['available'] is False and absence['error']==87 and absence['stage']=='OpenProcess',
   'Actual owned worker absence observation required')
 require(sha(scoped(pause['loop_status_path']))==pause['loop_status_sha256']
   and sha(scoped(pause['source_checkpoint_path']))==pause['source_checkpoint_sha256']
   and pause['source_checkpoint_sha256']==receipt['source_best_sha256'],'Paused predecessor/source artifacts changed')
 for item in pause['unpublished_training_artifacts']:
  require(item['exists'] is False and not scoped(item['path']).exists(),'Pre-optimizer predecessor artifact unexpectedly appeared')
 require(not (OUTPUT/'runs/adaptive_20261011_public_data_02_public_rank8_control/training_history.json').exists(),
   'Original source-model preparation must have zero published optimizer history')
 baseline,baseline_provenance=prior.validate_baseline()
 threshold=baseline['metrics']['wer']+.02
 runtime=source_hashes()
 for name,digest in config['code_sha256'].items():require(sha(OUTPUT/name)==digest,'Prior model/scoring implementation changed: '+name)
 inputs={str(path):sha(path) for path in (manifest,proof_path,label_path,scoped(args.replay_manifest),readiness,receipt_path,metadata,metadata_proof,reuse,gpu_path,dispatch_path,pause_path)}
 screen_steps=args.screen_updates
 return {'receipt':record(receipt_path),'failed_dispatch_predecessor':record(failed_path),
   'preparation_exit_receipt':record(pause_path),'metadata_GPU_smoke':record(gpu_path),'input_audit_reuse':record(reuse),
   'source_selection':selected,'source_provenance':provenance,
   'source_config':config,'source_checkpoint':record(selected['weights']),'runtime':runtime,'input_sha256':inputs,
   'public_data':{'clips':proof['CPU_tokenizer_accepted_clips'],'actual_PCM_hours':proof['CPU_tokenizer_accepted_hours'],
      'manifest':record(manifest),'proof':record(proof_path),'TDT_group_tokens':False,
      'TR_frozen_encoder_feature_cache_required':False,'TR_metadata_acceptance_pending':True,
      'genuine_replay_acoustic_cache_required':True,'TR_metadata':record(metadata),'TR_metadata_proof':record(metadata_proof)},
   'screen_updates':screen_steps,'strong_DEV_threshold':threshold,'baseline_provenance':baseline_provenance,
   'passed':True,'GPU_not_initialized':True}


def freeze_trainables(source,target,source_run,step):
 import torch
 require(not target.exists(),'Immutable warmstart must be new')
 before=sha(source);saved=torch.load(source,map_location='cpu',weights_only=True)
 require(saved['run_id']==source_run and saved['step']==step and len(saved['weights'])==207,'Warmstart identity207 tensor layout required')
 require(all(p.device.type=='cpu' and p.dtype==torch.float32 and torch.isfinite(p).all() for p in saved['weights'].values()),'Finite FP32 trainables required')
 value={'schema':2,'run_id':source_run,'step':step,'weights':saved['weights'],'lora_metadata':saved['lora_metadata'],
   'warmstart_only':True,'exact_optimizer_resume_supported':False,'parent_checkpoint_path':str(source),'parent_checkpoint_sha256':before}
 target.parent.mkdir(parents=True,exist_ok=True);torch.save(value,target)
 require(sha(source)==before,'Source checkpoint mutated while freezing')
 return {**record(target),'original_source_path':str(source),'original_source_sha256':before,'source_run_id':source_run,'source_step':step,
    'fresh_optimizer_and_sampler':True,'optimizer_copied':False}


def training_argv(args,config,run_id,start,rank,steps,wall,stop,*,replay_fraction=.5):
 values={key:config[key] for key in IDENTITY_KEYS}
 tr_per_update=16-round(16*replay_fraction)
 values.update(epochs=3.,steps=steps,batch_size=8,accumulation=2,replay_fraction=replay_fraction,
   rank=rank,alpha=2.*rank,tdt_implementation='v2',eval_per_epoch=4.,
   eval_epoch_fraction=min(.25,args.screen_updates*tr_per_update/74883),eval_every_updates=args.screen_updates,
   warmup_steps=25,schedule='constant',min_lr_ratio=.1,early_stop_patience=3,max_retention_failures=2,
   min_improvement=.002,target_turkish_wer=args.strong_threshold,log_every=10,run_id=run_id,
   tr_manifest=str(scoped(args.tr_manifest)),replay_manifest=str(scoped(args.replay_manifest)),
   data_v3_proof=str(scoped(args.data_proof)),tr_frame_metadata=str(scoped(args.tr_frame_metadata)),
   tr_frame_metadata_proof=str(scoped(args.tr_frame_metadata_proof)),reuse_input_audit=str(scoped(args.reuse_input_audit)),resume_weights=str(start),max_wall_seconds=wall,stop_file=str(stop))
 argv=[]
 for key,value in values.items():
  if value is False or value is None:continue
  argv.append('--'+key.replace('_','-'))
  if value is not True:argv.append(str(value))
 return argv


def argument_sha(argv):return hashlib.sha256(json.dumps(argv,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def validate_identity(value,run_id,argv,launcher_pid,started,sources):
 require(value['kind']=='owned_training_worker_identity_v1' and value['expected_run_id']==run_id
   and value['worker_arguments_sha256']==argument_sha(argv),'Owned worker run/argv identity differs')
 require(type(value['pid']) is int and value['pid']>0 and (value['pid']==launcher_pid or value['ppid']==launcher_pid),'Worker must be self or owned venv child')
 for key,name in [('memory_launcher_sha256',WRAPPER),('cache_launcher_sha256','launch_with_cache_reclaim_v7.py'),
   ('cache_helper_sha256','cuda_cache_reclaim_v1.py'),('resource_helper_sha256','resource_watchdog_v2.py'),
   ('resource_telemetry_sha256','paired_windows_commit_telemetry_v1.py'),('frozen_worker_sha256','train_lora_batched_data_v6.py'),
   ('data_policy_sha256','public_turkish_data_policy_v3.py')]:require(value[key]==sources[name],'Actual worker source changed: '+key)
 require(stamp(started)-timedelta(seconds=5)<=stamp(value['started_at_utc'])<=datetime.now(timezone.utc),'Worker identity timestamp differs')
 creation=value['process_creation_identity']
 require(creation['available'] is True and creation['pid']==value['pid']
   and stamp(started).timestamp()-5<=creation['creation_unix_seconds']<=stamp(value['started_at_utc']).timestamp()+1,'Owned process creation timestamp required')
 return value


def self_memory(live_path,identity,identity_path):
 if not live_path.exists():return None,{'available':False,'reason':'awaiting_actual_worker_sidecar'}
 value=read(live_path,{})
 require(value['kind']=='owned_worker_live_memory_v1' and value['pid']==identity['pid']
   and value['run_id']==identity['expected_run_id'] and value['worker_identity_sha256']==sha(identity_path)
   and value['process_creation_identity']==identity['process_creation_identity'],'Self sidecar identity differs')
 for key in ('memory_launcher_sha256','cache_launcher_sha256','cache_helper_sha256','resource_helper_sha256',
   'resource_telemetry_sha256','frozen_worker_sha256','data_policy_sha256'):
  require(value[key]==identity[key],'Self sidecar source identity differs: '+key)
 sample=value['sample'];age=(datetime.now(timezone.utc)-stamp(sample['observed_at_utc'])).total_seconds()
 require(age>=-1,'Future self telemetry invalid')
 if age>15:return None,{'available':False,'reason':'worker_self_sample_stale','age_seconds':age}
 observation=sample.get('calling_process_commit_memory',{})
 if not observation.get('available') or observation.get('pid')!=identity['pid']:
  return None,{'available':False,'reason':'actual_worker_process_available_commit_API_unavailable','sample':sample}
 reason=worker_process_commit_reason(observation,identity['pid'])
 return reason,{'available':True,'age_seconds':age,'sample':sample,'scope':'Actual owned worker self reading; separate from global GetPerformanceInfo'}


def log_phase(path,history):
 try:
  with path.open('rb') as stream:
   stream.seek(0,2);stream.seek(max(0,stream.tell()-65536));lines=stream.read().decode('utf-8',errors='replace').splitlines()
 except OSError:return {'phase':'worker_startup','inferred_from_bounded_log_tail':True}
 result={'phase':'worker_startup','inferred_from_bounded_log_tail':True}
 for line in reversed(lines):
  match=re.search(r'Reused full-audit fresh WAV/header verification (\d+)',line)
  if match:
   result.update(phase='fresh_verified_input_fingerprints_and_headers',inputs_processed=int(match[1]),inputs_total=77783,
     input_scope='All owned reuse-receipt TRAIN/replay/heldout WAV files; no predictions',optimizer_updates_started=False);break
  match=re.search(r'TR metadata (\d+)/(\d+)',line)
  if match:
   result.update(phase='Turkish_frame_target_metadata_preparation',metadata_processed=int(match[1]),metadata_total=int(match[2]),
     TR_frozen_encoder_feature_cache_required=False,optimizer_updates_started=False);break
  match=re.search(r'acoustic cache (\d+)/(\d+)',line)
  if match:
   result.update(phase='teacher_acoustic_cache_preparation',cache_processed=int(match[1]),cache_stream_total=int(match[2]),
     cache_count_is_stream_local=True,optimizer_updates_started=False);break
  match=re.search(r'public v3 PCM/input verification (\d+)/(\d+)',line)
  if match:
   result.update(phase='actual_training_input_verification',inputs_processed=int(match[1]),inputs_total=int(match[2]),optimizer_updates_started=False);break
  if 'validation ' in line or 'validation tr:' in line:result['phase']='development_evaluation';break
  if line.startswith('dev step'):result['phase']='development_complete_or_checkpoint';break
  if line.startswith('{"step"'):result['phase']='training_updates';break
  if line.startswith('Batch '):result['phase']='training_ready';break
  if line.startswith('{"accepted"'):result['phase']='final_checkpoint_or_exit';break
 result['latest_published_step']=history[-1]['step'] if history else 0
 return result


class ProgressGuard:
 def __init__(self,started):
  self.reserved=0;self.rate_streak=0;self.rate_reference=None;self.last_rate_step=0;self.latest_step=0;self.last_progress=started
 def observe(self,history,clock):
  if not history:return None,{}
  row=history[-1];step=row['step'];info={'published_step':step,'not_a_durable_checkpoint':True}
  require(step==len(history),'Fresh-stage history must be ordered and complete')
  if step>self.latest_step:self.latest_step=step;self.last_progress=clock
  if row.get('train_wall_seconds',0)>0 and 'turkish_audio_seen_seconds' in row:
   info.update(turkish_audio_minutes_per_train_wall_minute=row['turkish_audio_seen_seconds']/row['train_wall_seconds'],
     turkish_audio_seen_seconds=row['turkish_audio_seen_seconds'],turkish_clips_seen=row.get('turkish_clips_seen'),
     turkish_unique_clips_seen=row.get('turkish_unique_clips_seen'),turkish_epoch=row.get('turkish_epoch'))
  reserved=row['current_reserved_vram_gb'];require(math.isfinite(reserved),'Finite current reservation required')
  self.reserved=self.reserved+1 if reserved>12 else 0;info['reserved_observation_streak']=self.reserved
  if self.reserved>=3:return 'reserved_above12GB_three_observations',info
  if clock-self.last_progress>600:return 'training_publication_stall_600seconds',info
  if len(history)>50 and step!=self.last_rate_step:
   old=history[-51];audio=row['audio_seen_seconds']-old['audio_seen_seconds'];wall=row['train_wall_seconds']-old['train_wall_seconds']
   require(audio>0 and wall>0,'Positive update-only audio/time required');rate=audio/wall
   if self.rate_reference is None:self.rate_reference=rate
   self.rate_streak=self.rate_streak+1 if rate<.5*self.rate_reference else 0;self.last_rate_step=step
   info.update(audio_min_per_wall_min_last50=rate,rate_reference=self.rate_reference,rate_regression_streak=self.rate_streak)
   if 'turkish_audio_seen_seconds' in old and 'turkish_audio_seen_seconds' in row:
    info['turkish_audio_min_per_wall_min_last50']=(row['turkish_audio_seen_seconds']-old['turkish_audio_seen_seconds'])/wall
   if self.rate_streak>=3:return 'sustained_training_rate_regression',info
  return None,info


def check_runtime_evidence(trial,sources):
 revalidate(sources)
 memory=read(trial['memory_provenance_path'],{});cache=read(trial['cache_provenance_path'],{});io=read(trial['io_provenance_path'],{})
 require(memory['kind']=='memory_guard_training_launcher_v10' and memory['status']=='complete'
   and memory['cap_bytes']==9e9 and memory['cleanup_threshold_bytes']==4e9 and memory['allocator_cap_applied'] is True
   and memory['training_math_changed'] is False and memory['backend_flags_changed'] is False,'Actual new memory runtime policy differs')
 require(memory['all_phase_owned_memory_telemetry']['thread_join_completed'] is True,'Owned all-phase telemetry thread did not stop/join')
 require(memory['resource_telemetry_sha256']==memory['paired_windows_commit_telemetry_v1_sha256_after']==sources['paired_windows_commit_telemetry_v1.py']
   and memory['data_policy_sha256']==memory['public_turkish_data_policy_v3_sha256_after']==sources['public_turkish_data_policy_v3.py'],
   'Before/after system telemetry or data policy source differs')
 for key,name in [('memory_launcher_sha256',WRAPPER),('cache_launcher_sha256','launch_with_cache_reclaim_v7.py'),
   ('resource_helper_sha256','resource_watchdog_v2.py'),('frozen_worker_sha256','train_lora_batched_data_v6.py')]:
  require(memory[key]==memory[key+'_after']==sources[name],'Before/after runtime source differs: '+key)
 require(cache['kind']=='cache_reclamation_training_launcher_v7' and cache['status']=='complete'
   and cache['frozen_worker_sha256']==sources['train_lora_batched_data_v6.py'],'New cache runtime worker differs')
 evidence=cache['cache_reclamation'];require(evidence['eligible_update_calls']==trial['completed_steps'],'Cleanup hook did not observe each fresh-stage update')
 require(evidence['failed_reclamations']==0,'Cache release failed')
 require(io['status']=='complete' and io['worker_sha256']==io['worker_sha256_after']==sources['train_lora_batched_data_v6.py'],'Actual I/O worker source changed')
 return {'memory':record(trial['memory_provenance_path']),'cache':record(trial['cache_provenance_path']),
   'io':record(trial['io_provenance_path']),'eligible_updates':evidence['eligible_update_calls'],'mathematical_source_unchanged':True,
   'historical_cross_shape_BF16_equivalence_passed':False,'no_new_numerical_equivalence_claim':True}


def run_stage(args,state,stage,start,rank,steps,deadline,*,replay_fraction=.5):
 revalidate(state['code_sha256']);remaining=deadline-time.monotonic();require(remaining>120,'Insufficient time for safe stage start')
 loop_dir=Path(state['loop_dir']);run_id=safe_id(args.loop_id+'_'+stage);directory=OUTPUT/'runs'/run_id
 require(not directory.exists() and not (ROOT/'work'/(run_id+'_latest.pt')).exists(),'Stage namespace must be fresh')
 paths={key:loop_dir/(run_id+'_'+suffix) for key,suffix in [('io_provenance','io.json'),('cache_provenance','cache.json'),
   ('memory_provenance','memory.json'),('worker_identity','identity.json'),('stop','stop.txt'),('log','training.log')]}
 argv=training_argv(args,state['source_config'],run_id,start,rank,steps,max(1,remaining-120),paths['stop'],replay_fraction=replay_fraction)
 command=[str(scoped(args.python)),str(OUTPUT/WRAPPER),'--provenance',str(paths['io_provenance']),
   '--cache-provenance',str(paths['cache_provenance']),'--memory-provenance',str(paths['memory_provenance']),
   '--worker-identity',str(paths['worker_identity']),'--cap-gb','9','--threshold-gb','4','--',*argv]
 trial={'run_id':run_id,'stage':stage,'status':'starting','rank':rank,'alpha':rank*2,'requested_steps':steps,'initial_step':0,
   'replay_fraction':replay_fraction,'TR_clips_per_update':16-round(16*replay_fraction),
   'diagnostic_short_screen':stage!='public_winner_extension','exposure_uncertainty':'Short DEV diagnostic; limited source coverage, no competitive or heldout claim',
   'fresh_optimizer_and_sampler':True,'start_weights':record(start),'arguments':argv,'canonical_worker_arguments_sha256':argument_sha(argv),
   'started_at_utc':now(),'selection_split':'FLEURS validation only','heldout_tests_evaluated':False,
   **{key+'_path':str(path) for key,path in paths.items()}}
 state['trials'].append(trial);state['active_run_id']=run_id;write(loop_dir/'loop_status.json',state)
 child=None;resource=ResourceGuard();started=time.monotonic();progress=ProgressGuard(started);stop_reason=None;stop_at=None
 try:
  with paths['log'].open('w',encoding='utf-8') as log:
   child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);trial['owned_launcher_pid']=child.pid
   while child.poll() is None:
    history=read(directory/'training_history.json',[]);phase=log_phase(paths['log'],history)
    identity=None
    if paths['worker_identity'].exists():identity=validate_identity(read(paths['worker_identity'],{}),run_id,argv,child.pid,trial['started_at_utc'],state['code_sha256'])
    sample=sample_memory(identity['pid'] if identity else child.pid,phase['phase'])
    reason,observation=resource.observe(sample['system_memory'],sample['worker_private_memory'],observed_monotonic=sample['observed_monotonic'])
    self_reason,self_observation=self_memory(Path(str(paths['worker_identity'])+'.live_memory.json'),identity,paths['worker_identity']) if identity else (None,{'available':False})
    progress_reason,progress_info=progress.observe(history,time.monotonic())
    reason=reason or self_reason or progress_reason
    if not identity and time.monotonic()-started>120:reason=reason or 'owned_worker_identity_unavailable'
    if identity and not self_observation['available'] and time.monotonic()-started>120:reason=reason or 'actual_worker_self_memory_unavailable_or_stale'
    if not history and time.monotonic()-started>args.prepare_timeout_seconds:reason=reason or 'bounded_cache_preparation_timeout'
    if time.monotonic()>=deadline-120:reason=reason or 'window_budget_checkpoint_boundary'
    trial.update(status='running',phase=phase,published_steps=history[-1]['step'] if history else 0,
      latest_observation_at_utc=now(),system_and_owned_private_memory=observation,actual_worker_self_memory=self_observation,progress=progress_info)
    state['current_phase']=phase;state['updated_at_utc']=now();write(loop_dir/'loop_status.json',state)
    with (loop_dir/(run_id+'_memory_observations.jsonl')).open('a',encoding='utf-8') as out:
     out.write(json.dumps({'at_utc':now(),'phase':phase,'memory':observation,'worker_self':self_observation,'progress':progress_info},ensure_ascii=False)+'\n')
    if reason and stop_reason is None:
     stop_reason=reason;stop_at=time.monotonic();paths['stop'].write_text('Cooperative save/DEV requested: '+reason+'\n',encoding='utf-8')
     trial.update(cooperative_stop_reason=reason,cooperative_stop_requested_at_utc=now())
    if stop_at and time.monotonic()-stop_at>args.cooperative_grace_seconds:
     child.terminate();child.wait(timeout=30);raise RuntimeError('Owned child exceeded cooperative grace; no next GPU trial')
    time.sleep(args.poll_seconds)
   trial['exit_code']=child.returncode
  require(child.returncode==0,'GPU child failed; review before another trial')
  selection=read(directory/'selection.json',{});trial.update(completed_steps=selection['completed_steps'],finished_at_utc=now(),status='complete',stop_early_reason=selection['stop_early_reason'])
  trial['runtime_evidence']=check_runtime_evidence(trial,state['code_sha256'])
  config=read(directory/'run_config.json',{});actual=read(directory/'public_data_v3_acceptance.json',{})
  require(config['turkish_clips_used']==state['public_data']['clips'] and actual['actual_TR_metadata_accepted_clips']==state['public_data']['clips']
    and actual['TR_metadata_acceptance_pending'] is False and actual['TR_frozen_encoder_feature_cache_required'] is False
    and actual['acoustic_cache_acceptance_pending']=='not_applicable_to_TR_metadata','Full corrected pool actual metadata acceptance required')
  trial.update(actual_training_pool=record(directory/'training_pool.json'),actual_public_data_acceptance=record(directory/'public_data_v3_acceptance.json'),
    turkish_epochs_completed=selection['turkish_epochs_completed'],audio_seen_seconds=selection['audio_seen_seconds'],train_wall_seconds=selection['train_wall_seconds'],
    selected_step=selection['selected_step'],accepted=selection['accepted'],requested_screen_exposure_completed=selection['completed_steps']>=steps)
  history=read(directory/'training_history.json',[])
  if history:
   trial.update(turkish_audio_seen_seconds=history[-1]['turkish_audio_seen_seconds'],
     turkish_audio_minutes_per_train_wall_minute=history[-1]['turkish_audio_seen_seconds']/selection['train_wall_seconds'] if selection['train_wall_seconds'] else None,
     turkish_clips_seen=history[-1]['turkish_clips_seen'],turkish_unique_clips_seen=history[-1]['turkish_unique_clips_seen'])
  if selection['accepted']:
   valid,provenance=prior.validate_selection(run_id);candidate={'run_id':run_id,'selected_step':valid['selected_step'],
    'wer':valid['turkish_validation_wer'],'weights':valid['weights'],'rank':rank,'alpha':rank*2,'replay_fraction':replay_fraction,'provenance':provenance}
   trial['candidate']=candidate
   if state['best'] is None or candidate['wer']<state['best']['wer']:state['best']=candidate
  if stop_reason and stop_reason!='window_budget_checkpoint_boundary':
   state.update(status='resource_review_required',resource_review_required=True,resource_stop_reason=stop_reason)
  return trial
 finally:
  if child is not None and child.poll() is None:
   child.terminate()
   try:child.wait(timeout=30)
   except subprocess.TimeoutExpired:child.kill();child.wait(timeout=30)
  state['active_run_id']=None;state['updated_at_utc']=now();write(loop_dir/'loop_status.json',state)


def main(args):
 prepared=preflight(args)
 if args.validate_only:
  write(args.preflight_output,prepared);print(json.dumps({'passed':True,'screen_steps':prepared['screen_updates'],'CUDA_initialized':False}));return 0
 loop_dir=OUTPUT/'runs'/args.loop_id;loop_dir.mkdir();state={'schema':1,'kind':'expanded_public_TDT_training_window_v6',
  'loop_id':args.loop_id,'loop_dir':str(loop_dir),'status':'running','started_at_utc':now(),
  'deadline_at_utc':(datetime.now(timezone.utc)+timedelta(hours=args.window_hours)).isoformat(),
  'source_run_id':SOURCE_RUN,'source_step':1871,'source_config':prepared['source_config'],
  'predecessor_exit_receipt':prepared['receipt'],'source_provenance':prepared['source_provenance'],
  'failed_dispatch_predecessor':prepared['failed_dispatch_predecessor'],
  'preparation_exit_receipt':prepared['preparation_exit_receipt'],'metadata_GPU_smoke':prepared['metadata_GPU_smoke'],
  'input_audit_reuse':prepared['input_audit_reuse'],
  'code_sha256':prepared['runtime'],'input_sha256':prepared['input_sha256'],'public_data':prepared['public_data'],
  'screen_updates':prepared['screen_updates'],'strong_DEV_threshold':prepared['strong_DEV_threshold'],
  'baseline_provenance':prepared['baseline_provenance'],'trials':[],'best':None,'resource_review_required':False,
  'objective_complete':False,'objective_incomplete':True,'heldout_tests_evaluated':False,'export_published':False,
  'runtime_policy':{'physical_batch':8,'accumulation':2,'effective_batch':16,'allocator_cap_GB':9,'cleanup_threshold_GB':4,
     'system_commit_API':'GetPerformanceInfo','self_process_commit_API':'GlobalMemoryStatusEx from actual owned worker only',
     'historical_strict_cross_shape_BF16_equivalence_passed':False,'new_runtime_hardware_efficacy_pending':True},
  'next_action':'Continue/adapt DEV-only experiments until competitive; final export and fixed TEST are separate frozen-selection work'}
 args.strong_threshold=state['strong_DEV_threshold'];deadline=time.monotonic()+args.window_hours*3600
 try:
  state['awake_request']=prior.keep_awake(True);write(loop_dir/'loop_status.json',state)
  common=loop_dir/'trial_inputs/common_rank8_1871.pt'
  state['common_start']=freeze_trainables(Path(prepared['source_checkpoint']['path']),common,SOURCE_RUN,1871)
  control=run_stage(args,state,'public_rank8_control',common,8,prepared['screen_updates'],deadline)
  if state['resource_review_required']:return 1
  if args.rank16_validation and control['requested_screen_exposure_completed']:
   proof=read(scoped(args.rank16_validation),{})
   require(proof.get('passed') is True and proof.get('rank')==16 and proof.get('full_model_GPU_readiness_passed') is True,'Actual rank16 full-model readiness proof required')
   estimate=control['train_wall_seconds']+240
   if deadline-time.monotonic()>estimate+120:
    upgraded=loop_dir/'trial_inputs/common_rank16_1871.pt'
    import torch
    from upgrade_lora_rank_v1 import upgrade
    saved=torch.load(common,map_location='cpu',weights_only=True);result=upgrade(saved,seed=state['source_config']['seed'])
    result['rank_upgrade'].update(parent_checkpoint_sha256=sha(common),parent_checkpoint_path=str(common));torch.save(result,upgraded)
    state['rank16_readiness']=record(args.rank16_validation)
    arm=run_stage(args,state,'public_rank16_matched',upgraded,16,prepared['screen_updates'],deadline)
    state['matched_rank_comparison']={'control_run_id':control['run_id'],'rank16_run_id':arm['run_id'],
      'common_rank8_start_sha256':sha(common),'matched_update_count':prepared['screen_updates'],
      'complete_equal_exposure':arm['completed_steps']==control['completed_steps']==prepared['screen_updates'],
      'audio_and_source_sequence_equality_pending_independent_audit':True,'no_capacity_quality_promise':True}
   else:state['rank16_deferred_reason']='Remaining time insufficient for matched exposure; retain control and compare in next window'
  else:state['rank16_deferred_reason']='Optional actual GPU readiness not supplied or control exposure incomplete'
  if state['resource_review_required']:return 1
  if not args.no_replay_challenger and control['requested_screen_exposure_completed'] and deadline-time.monotonic()>control['train_wall_seconds']+240:
   challenger=run_stage(args,state,'public_rank8_replay25_screen',common,8,prepared['screen_updates'],deadline,replay_fraction=.25)
   state['replay_fraction_comparison']={'control_run_id':control['run_id'],'challenger_run_id':challenger['run_id'],
      'control_replay_fraction':.5,'challenger_replay_fraction':.25,'requested_updates_each':prepared['screen_updates'],
      'same_common_weights_seed_data':True,'TR_clip_exposure_changed_from8_to12_per_update':True,
      'TR_audio_and_wall_rates_measured_separately':True,'quality_and_speed_experiment_not_numerical_equivalence':True}
  if state['resource_review_required']:return 1
  if state['best'] and deadline-time.monotonic()>300 and control['requested_screen_exposure_completed']:
   best=state['best'];start=loop_dir/'trial_inputs/winner_extension_start.pt'
   freeze_trainables(Path(best['weights']),start,best['run_id'],best['selected_step'])
   tr_per_update=16-round(16*best['replay_fraction'])
   run_stage(args,state,'public_winner_extension',start,best['rank'],3*math.ceil(state['public_data']['clips']/tr_per_update),deadline,replay_fraction=best['replay_fraction'])
  if state['resource_review_required']:return 1
  state['status']='training_window_complete_objective_incomplete';return 0
 except BaseException as error:
  state.update(status='failed_review_required',resource_review_required=True,failure_type=type(error).__name__,failure_message=str(error));raise
 finally:
  state['awake_clear']=prior.keep_awake(False);state['finished_at_utc']=now();state['active_run_id']=None
  state['competitive_on_development']=bool(state['best'] and state['best']['wer']<=state['strong_DEV_threshold'])
  state['next_action']='Root reviews resource failure and resumes safely; no silent batch fallback' if state['resource_review_required'] else 'Root renews/adapts the next DEV-only window; freeze/export/audit before any TEST or objective completion'
  write(loop_dir/'loop_status.json',state)


def parser():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--loop-id',default='adaptive_20261011_public_data_03')
 p.add_argument('--source-run-id',default=SOURCE_RUN)
 p.add_argument('--prior-exit-receipt',type=Path,default=OUTPUT/'continuation_prerequisites_v4/predecessor_exit_receipt.json')
 p.add_argument('--tr-manifest',type=Path,default=ROOT/'work/data/tr_public_v3_train.jsonl')
 p.add_argument('--replay-manifest',type=Path,default=ROOT/'work/data/replay_train.jsonl')
 p.add_argument('--data-proof',type=Path,default=OUTPUT/'tr_public_v3_preparation.json')
 p.add_argument('--data-readiness',type=Path,default=OUTPUT/'TR_metadata_launch_readiness_v1.json')
 p.add_argument('--tr-frame-metadata',type=Path,default=ROOT/'work/data/tr_public_v3_frame_metadata_v1.jsonl')
 p.add_argument('--tr-frame-metadata-proof',type=Path,default=OUTPUT/'TR_frame_metadata_cpu_verification_v1.json')
 p.add_argument('--reuse-input-audit',type=Path,default=OUTPUT/'public_data_full_audit_reuse_v1/reuse_receipt.json')
 p.add_argument('--metadata-gpu-proof',type=Path,default=OUTPUT/'TR_metadata_actual_GPU_masks_v1.json')
 p.add_argument('--preparation-exit-receipt',type=Path,default=OUTPUT/'public_data_02_preparation_exit_receipt_v1.json')
 p.add_argument('--python',type=Path,default=ROOT/'work/training-env/Scripts/python.exe')
 p.add_argument('--window-hours',type=float,default=3.)
 p.add_argument('--screen-updates',type=int,default=250)
 p.add_argument('--no-replay-challenger',action='store_true')
 p.add_argument('--rank16-validation',type=Path)
 p.add_argument('--prepare-timeout-seconds',type=float,default=7200.)
 p.add_argument('--cooperative-grace-seconds',type=float,default=300.)
 p.add_argument('--poll-seconds',type=float,default=5.)
 p.add_argument('--train-only',action='store_true')
 p.add_argument('--validate-only',action='store_true')
 p.add_argument('--preflight-output',type=Path,default=OUTPUT/'continuation_v6_actual_preflight.json')
 return p


if __name__=='__main__':raise SystemExit(main(parser().parse_args()))
