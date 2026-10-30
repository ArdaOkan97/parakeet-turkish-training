"""New disabled metadata bindings for original-source9223 production assembly.

Only source/JSON/configuration files are read. No namespace, audio, model,
optimizer or training process is created. Historical authorities are retained;
new exact source/identity bindings require the final separate execution review.
"""
from collections import deque
from copy import deepcopy
import ast,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research'
sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from atomic_io_retry_v1 import read_json,write_json
E=Path(r'E:\gpu_training_scratchpad');A=E/'analysis/ASR18_memory_diagnosis_v1'
N=E/'analysis/BPE9223_training_launch_v1'
OUT=N/'full_configuration_inputs_01'
RUN=E/'runs/BPE9223_weak2_clean1_20261018_01'

REPLACEMENTS={
 'asr18_full_run_parent_controller_v11.py':'asr18_BPE9223_full_run_controller_v1.py',
 'asr18_full_run_parent_controller_v13.py':'asr18_BPE9223_full_run_controller_v1.py',
 'asr18_full_run_native_owner_v5.py':'asr18_BPE9223_full_run_native_v2.py',
 'asr18_full_run_owned_GPU_runtime_v6.py':'asr18_BPE9223_owned_GPU_runtime_v2.py',
 'run_ASR18_full_run_owner_v13.py':'run_ASR18_BPE9223_full_run_owner_v1.py',
 'run_ASR18_full_run_owner_v15.py':'run_ASR18_BPE9223_full_run_owner_v1.py',
 'asr18_full_run_persistence_prefix_v1.py':'asr18_BPE9223_persistence_prefix_v4.py',
 'asr18_full_run_metadata_resolver_v1.py':'asr18_BPE9223_metadata_resolver_v1.py',
 'asr18_full_run_source_context_v1.py':'asr18_BPE9223_source_context_v2.py',
 'bpe_full_run_training_state_v1.py':'asr18_BPE9223_training_state_v1.py',
 'asr18_full_run_current_recovery_v2.py':'asr18_BPE9223_current_recovery_v2.py',
 'bpe_full_run_lifecycle_v2.py':'asr18_BPE9223_lifecycle_v3.py',
 'bpe_full_run_durable_owner_adapter_v1.py':'asr18_BPE9223_durable_transitions_v3.py',
 'bpe_full_run_durable_owner_adapter_v2.py':'asr18_BPE9223_durable_owner_adapter_v3.py',
 'bpe_indexed_owned_checkpoint_storage_v3.py':'asr18_BPE9223_indexed_checkpoint_storage_v1.py',
 'bpe_owned_checkpoint_storage_v2.py':'asr18_BPE9223_owned_checkpoint_storage_v1.py',
 'asr18_full_run_serial_DEV_v8.py':'asr18_full_run_serial_DEV_9223_v2.py',
 'asr18_full_run_serial_DEV_v9.py':'asr18_full_run_serial_DEV_9223_v2.py',
 'asr18_full_run_DEV_runtime_v4.py':'asr18_full_run_DEV_runtime_9223_v2.py',
 'asr18_optimizer_free_DEV_structure_v2.py':'asr18_optimizer_free_DEV_structure_9223_v1.py',
 'asr18_DEV_weight_only_mapper_v1.py':'asr18_DEV_weight_only_mapper_9223_v1.py',
}

def source(name):
 for root in (S,R):
  path=root/name
  if path.is_file():return path
 raise ValueError('Missing exact owned source '+name)

def emit(name,value):
 path=OUT/name;require(not path.exists(),'New immutable input only');write_json(path,value);return record(path)

def main():
 require(not OUT.exists() and not RUN.exists(),'Fresh metadata and future production namespaces')
 require(read_json(E/'storage_config.json')['storage_root']==str(E),'Required E-only storage policy')
 old=read_json(A/'disabled_monitored_real_run_configuration_01/configuration_pending_execution_review.json')
 sustained=read_json(N/'sustained20_configuration_02/native_configuration_CPU_review_bound.json')
 data=read_json(N/'BPE9223_final_verified_TRAIN_population_evidence_v1.json')
 data['BPE9223_population']={k:data['BPE9223_population'][k] for k in ('path','bytes','sha256')}
 names=set(REPLACEMENTS.values())|{'prepare_ASR18_BPE9223_full_configuration_v2.py',Path(__file__).name,
  'asr18_BPE9223_CPU_window_configuration_v2.py','asr18_BPE9223_bootstrap_v2.py',
  'asr18_BPE9223_model_migration_v2.py','asr18_BPE9223_optimizer_v2.py'}
 queue=deque(source(name) for name in sorted(names));pins={}
 for item in read_json(sustained['source_manifest']['path'])['bindings']:pins[item['path']]=item
 while queue:
  path=queue.popleft().resolve()
  if str(path) in pins:continue
  raw=path.read_text(encoding='utf-8-sig');tree=ast.parse(raw);compile(tree,str(path),'exec');pins[str(path)]=record(path)
  for node in ast.walk(tree):
   imported=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
   for name in imported:
    if '.' not in name:
     for root in (S,R):
      candidate=root/(name+'.py')
      if candidate.is_file():queue.append(candidate);break
 for item in pins.values():require(record(item['path'])==item,'Prior frozen source changed')
 OUT.mkdir()
 manifest=emit('full_source_manifest.json',{'kind':'ASR18_BPE9223_full_original_source_manifest_v1',
  'bindings':list(pins.values()),'CPU_owner_bundle':record(N/'CPU_preparation_v1/source_handoff_v3.json'),
  'population_evidence':record(N/'BPE9223_final_verified_TRAIN_population_evidence_v1.json'),
  'row_migration_plan':sustained['row_migration_plan'],'production_enabled':False,
  'prototype_numerical_CPU_contract':sustained['CPU_contract'],
  'prototype_CPU_source_contract':sustained['recorded_room_source_contract']})
 oldrun=old['run_namespace'];oldidentity=digest(old['full_run_identity'])
 def remap(value):
  if isinstance(value,dict):
   if set(value) in ({'path','bytes','sha256'},{'path','bytes','sha256','mtime_ns'}) and Path(value['path']).name in REPLACEMENTS:
    return record(source(REPLACEMENTS[Path(value['path']).name]))
   return {REPLACEMENTS.get(k,k):remap(v) for k,v in value.items()}
  if isinstance(value,list):return [remap(v) for v in value]
  if isinstance(value,str):return value.replace(oldrun,str(RUN))
  return value
 template=remap(old)
 identity=template['full_run_identity'];identity['full_run_source_contract']=manifest
 identity.update(vocabulary=9223,blank_id=8192,joint_width=9228,
  BPE9223_population=data['BPE9223_population'],row_migration_plan=sustained['row_migration_plan'],
  permanent_namespace_scope='human_authorized_BPE9223_two_broad_one_cleaner')
 sha=digest(identity)
 template.update(TEMPLATE_NOT_LAUNCHABLE=True,source_manifest=manifest,vocabulary=9223,
  row_migration_plan=sustained['row_migration_plan'],run_namespace=str(RUN),
  checkpoint_storage_namespace=str(RUN/'checkpoints'),
  CPU_controller_source=record(R/'run_ASR18_BPE9223_CPU_window_v3.py'),
  augmentation_identity=sustained['augmentation_identity'],CUDA_capacity_policy={
   'allocator_cap_bytes':12000000000,'minimum_physical_bytes':17000000000,
   'driver_desktop_reserve_bytes':3000000000,'hard_free_physical_reserve_bytes':2000000000})
 template['planning_arguments']=deepcopy(sustained['planning_arguments'])
 template['planning_arguments']['DATA_evidence']=data
 template['forecast_calibration']['BPE9223_DATA_evidence']=data
 template['entire_worker_forecast'].update(native_private_forecast_bytes=7933712768,
  entire_worker_private_forecast_bytes=7997712768)
 for key in ('session','initial_sampler_sha256','prepared_metadata_plan','actual_metadata_bridge',
  'configuration_assembly_source','configuration_source_read_review','actual_ROOT_CPU_resource_review'):
  template.pop(key,None)
 policy_pins={}
 for key in ('ROOT_storage_review','durable_policy','ROOT_DEV_resource_review','ROOT_DEV_policy_review','ROOT_DEV_static_execution_review','actual_ROOT_CPU_resource_review'):
  value=remap(read_json(old[key]['path']))
  for stale in ('exact_current_metadata_inputs','exact_resume_input_rebinding','previous_trial_outcome','supersedes'):
   value.pop(stale,None)
  value.update(run_identity_sha256=sha,full_run_source_contract=manifest,
   production_approval_claimed=False,production_enabled=False,
   previous_source_scope=old[key],BPE9223_source_preparation_only=True)
  if key=='ROOT_storage_review':
   value['future_namespace']=str(RUN/'checkpoints')
   value['scope']='source_bound_BPE9223_future_storage_metadata_binding'
  elif key=='durable_policy':value['scope']='full_run_integration_validation'
  elif key=='ROOT_DEV_static_execution_review':
   value['scope']='static_original_full_run_pass_boundaries_only'
   value['resource_review']=policy_pins['ROOT_DEV_resource_review']
   value['source_bindings']={Path(item['path']).name:item for item in pins.values()}
  elif key=='ROOT_DEV_policy_review':
   sys.path.insert(0,str(S));import asr18_full_run_serial_DEV_9223_v2 as serial
   value['dependencies']=serial.sources()
  elif key=='actual_ROOT_CPU_resource_review':
   value.update(CPU_source_manifest=record(N/'CPU_preparation_v1/source_handoff_v3.json'),
    outer_parent_source=template['parent_driver_source'],scope='CPU_preparation_only')
  policy_pins[key]=emit(key+'.json',value);template[key]=policy_pins[key]
 # The static DEV map must be exactly the current implementation closure.
 template['DEV_source_bindings']=read_json(template['ROOT_DEV_static_execution_review']['path'])['source_bindings']
 template['full_run_identity']=identity
 tp=emit('disabled_full_configuration_template.json',template)
 print(json.dumps({'template':tp,'source_manifest':manifest,
  'CPU_resource_review':policy_pins['actual_ROOT_CPU_resource_review'],
  'run_identity_sha256':sha,'run_namespace_not_created':not RUN.exists(),
  'actual_ML_audio_GPU_jobs':0}),flush=True)

if __name__=='__main__':main()
