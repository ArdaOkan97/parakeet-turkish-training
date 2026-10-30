import sys,json,copy,time,os
from pathlib import Path
P=Path(r'C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');sys.path[:0]=[str(P/'outputs/parakeet_turkish'),str(P/'work/research')]
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from run_ASR18_BPE9223_full_run_owner_v20 import storage,sample_memory
from asr18_BPE9223_full_run_controller_v25 import ACKed260_owner_state
N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1');F=Path(__file__).parent
storage();began=time.monotonic();peak=0;calls=0
def guard():
 global peak,calls
 v=sample_memory(os.getpid(),'fixed_control_derivation');calls+=1;peak=max(peak,v['worker_private_memory']['private_bytes'])
 require(time.monotonic()-began<60 and peak<128000000 and v['system_memory']['commit_fraction']<.95 and v['system_memory']['commit_available_bytes']>2000000000,'Same metadata guards')
def save(p,o):require(not p.exists(),'New only');write_json(p,o);return record(p)
old=record(P/'outputs/parakeet_turkish/asr18_BPE9223_full_run_controller_v24.py');new=record(P/'outputs/parakeet_turkish/asr18_BPE9223_full_run_controller_v25.py')
def remap(o,pairs):
 if isinstance(o,dict):
  for a,b in pairs:
   if o==a:return copy.deepcopy(b)
  return {k:remap(v,pairs)for k,v in o.items()}
 if isinstance(o,list):return [remap(v,pairs)for v in o]
 if isinstance(o,str):return o.replace('direct_controller24_with_parent20_singlepass_prefetch_services','direct_controller25_with_parent20_singlepass_prefetch_services')
 return o
basecomp=record(N/'pipeline_overlap_operational_bundle_v1/ROOT_compatibility_v1.json');comp=read_json(basecomp['path']);oldmanifest=comp['operational_source_manifest']
manifest=remap(read_json(oldmanifest['path']),[(old,new)])
mp=save(F/'active_source_manifest.json',manifest)
latest=record(N/'pipeline_overlap_RUN07_actual_01/actual_control_namespace_failure_TOOL1_v1.json')
candidate=remap(comp,[(old,new),(oldmanifest,mp)])
candidate.update(accepted=False,reviewer='/root/memory_diagnosis',candidate_only=True,prior_ACKed260_control_namespace_failure=latest,
 prior_operational_compatibility=basecomp)
cp=save(F/'proposed_compatibility.json',candidate)
base_failure=comp['prior_ACKed260_metadata_path_failure'];state=ACKed260_owner_state(base_failure,guard,control_failure=latest)
sp=save(F/'actual_latest_owner_state260.json',state)
devp=record(N/'pipeline_overlap_DEV_candidates_v1/ROOT_signed/handoff.json');dev=read_json(devp['path']);updates=copy.deepcopy(dev['full_configuration_updates'])
pairs=[(old,new),(oldmanifest,mp),(basecomp,cp)]
for k in ('ROOT_DEV_resource_review','ROOT_DEV_policy_review','ROOT_DEV_static_execution_review'):
 oldpin=updates[k];value=remap(read_json(oldpin['path']),pairs)
 value.update(accepted=False,reviewer='/root/memory_diagnosis',candidate_only=True)
 pin=save(F/(k+'_candidate.json'),value);pairs.append((oldpin,pin));updates[k]=pin
updates=remap(updates,pairs)
dp=save(F/'DEV_candidate_handoff.json',{'kind':'ASR18_proposed_singlepass_fixed_control_DEV_authorities_v1','accepted':False,'full_configuration_updates':updates})
report={'kind':'ASR18_singlepass_fixed_control_source_and_actual_snapshot_handoff_v1','controller':new,'previous_controller':old,
 'manifest':mp,'proposed_compatibility':cp,'DEV_candidates':dp,'latest_failure':latest,'actual_state':sp,
 'actual_state_builder_passed':True,'guard_calls':calls,'sampled_private_peak_bytes':peak,'elapsed_seconds':time.monotonic()-began,
 'original_parent_guard_implementation_changed':False,'current_forecast_or_science_changed':False,'source_count':len(manifest['bindings'])}
hp=save(F/'handoff.json',report);print(json.dumps(hp),flush=True)
