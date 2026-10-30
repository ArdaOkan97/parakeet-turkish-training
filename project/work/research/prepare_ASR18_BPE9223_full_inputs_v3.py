"""Inactive exact import-successor rebind from valid full inputs02.

Emits review candidates only. ROOT must independently accept the new authority
bindings before the disabled metadata assembler can execute. No ML or audio.
"""
import ast,json,sys
from collections import deque
from copy import deepcopy
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research'
sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1')
BASE=N/'full_configuration_inputs_02';OUT=N/'full_configuration_inputs_03'
REPLACEMENTS={
 'asr18_BPE9223_full_run_controller_v1.py':S/'asr18_BPE9223_full_run_controller_v2.py',
 'run_ASR18_BPE9223_full_run_owner_v1.py':R/'run_ASR18_BPE9223_full_run_owner_v2.py',
 'asr18_BPE9223_full_run_native_v2.py':S/'asr18_BPE9223_full_run_native_v3.py',
 'asr18_BPE9223_owned_GPU_runtime_v2.py':R/'asr18_BPE9223_owned_GPU_runtime_v3.py',
 'asr18_full_run_serial_DEV_9223_v2.py':S/'asr18_full_run_serial_DEV_9223_v3.py',
 'asr18_full_run_DEV_runtime_9223_v2.py':S/'asr18_full_run_DEV_runtime_9223_v3.py',
}
POLICIES=('ROOT_storage_review','durable_policy','ROOT_DEV_resource_review',
 'ROOT_DEV_policy_review','ROOT_DEV_static_execution_review','actual_ROOT_CPU_resource_review')

def emit(name,value):
 path=OUT/name;require(not path.exists(),'Immutable fresh metadata only');write_json(path,value);return record(path)

def main():
 require(not OUT.exists(),'Fresh successor inputs')
 old=read_json(BASE/'disabled_full_configuration_template.json')
 old_manifest=read_json(BASE/'full_source_manifest.json')
 require(not Path(old['run_namespace']).exists(),'Production namespace still uncreated')
 pins={p['path']:p for p in old_manifest['bindings']}
 queue=deque([*REPLACEMENTS.values(),R/'prepare_ASR18_BPE9223_full_configuration_v3.py',Path(__file__)])
 while queue:
  path=queue.popleft().resolve()
  if str(path) in pins:continue
  tree=ast.parse(path.read_text('utf-8-sig'));compile(tree,str(path),'exec');pins[str(path)]=record(path)
  for node in ast.walk(tree):
   modules=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
   for module in modules:
    if '.' not in module:
     for root in (S,R):
      candidate=root/(module+'.py')
      if candidate.is_file():queue.append(candidate);break
 for pin in pins.values():require(record(pin['path'])==pin,'Frozen source changed')
 OUT.mkdir()
 manifest=deepcopy(old_manifest);manifest['bindings']=list(pins.values())
 mp=emit('full_source_manifest.json',manifest)
 def remap(value):
  if isinstance(value,dict):
   if set(value) in ({'path','bytes','sha256'},{'path','bytes','mtime_ns','sha256'}):
    name=Path(value['path']).name
    if name in REPLACEMENTS:return record(REPLACEMENTS[name])
   return {(REPLACEMENTS[k].name if k in REPLACEMENTS else k):remap(v) for k,v in value.items()}
  if isinstance(value,list):return [remap(v) for v in value]
  return value
 template=remap(old);identity=template['full_run_identity'];identity['full_run_source_contract']=mp
 require(type(identity['owner_validation_only']) is bool and identity['owner_validation_only'] is False
  and identity['production_source_eligible'] is True,'Typed production qualification preserved')
 identity_sha=digest(identity);template['source_manifest']=mp
 bindings={Path(pin['path']).name:pin for pin in pins.values()}
 import asr18_full_run_serial_DEV_9223_v3 as serial
 candidate_pins={}
 for key in POLICIES:
  candidate=remap(read_json(BASE/(key+'.json')))
  candidate.update(run_identity_sha256=identity_sha,full_run_source_contract=mp,
   accepted=False,reviewer='/root/memory_diagnosis',ROOT_acceptance_required=True,
   prior_genuine_ROOT_authority=record(BASE/(key+'.json')))
  if key=='ROOT_DEV_policy_review':candidate['dependencies']=serial.sources()
  elif key=='ROOT_DEV_static_execution_review':
   candidate['source_bindings']=bindings
   candidate['resource_review']=candidate_pins['ROOT_DEV_resource_review']
   candidate['ROOT_policy_review']=candidate_pins['ROOT_DEV_policy_review']
  candidate_pins[key]=emit('candidate_'+key+'.json',candidate);template[key]=candidate_pins[key]
 template['DEV_source_bindings']=bindings
 tp=emit('candidate_disabled_full_configuration_template.json',template)
 handoff={'kind':'BPE9223_exact_full_import_successor_review_input_v1',
  'source':record(__file__),'assembler':record(R/'prepare_ASR18_BPE9223_full_configuration_v3.py'),
  'base_template':record(BASE/'disabled_full_configuration_template.json'),
  'base_manifest':record(BASE/'full_source_manifest.json'),'source_manifest':mp,'template_candidate':tp,
  'run_identity_sha256':identity_sha,'candidate_authorities':candidate_pins,
  'source_replacements':{k:record(v) for k,v in REPLACEMENTS.items()},
  'old_bindings_preserved':len(old_manifest['bindings']),'new_binding_count':len(pins)-len(old_manifest['bindings']),
  'production_enabled':False,'actual_audio_model_native_jobs':0,'ROOT_acceptance_required':True,
  'run_namespace_created':Path(template['run_namespace']).exists()}
 hp=emit('review_input_handoff.json',handoff);print(json.dumps({'handoff':hp,'manifest':mp,'identity':identity_sha}),flush=True)

if __name__=='__main__':main()
