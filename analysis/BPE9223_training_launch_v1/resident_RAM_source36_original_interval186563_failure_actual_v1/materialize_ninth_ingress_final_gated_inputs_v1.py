"""Inactive metadata only: bind genuine ROOT gates, no candidate or plan reads."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import psutil
F=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_source36_original_interval186563_failure_actual_v1')
S=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20/outputs/parakeet_turkish');sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
def require(v,m):
    if not v:raise ValueError(m)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
    p=Path(p).resolve();before=p.stat();require(before.st_size<=8388608,'Compact SOURCE/metadata only')
    data=p.read_bytes();after=p.stat();require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Stable metadata/source')
    return {'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def encode(v):return (json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
def future(p,b):return {'path':str(p.resolve()),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def identity(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
require(os.environ['CUDA_VISIBLE_DEVICES']=='-1'and all(os.environ[k]=='1'for k in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
    and all(Path(os.environ[k]).resolve().is_relative_to(Path('E:/gpu_training_scratchpad').resolve())for k in('TEMP','TMP','HF_HOME','TORCH_EXTENSIONS_DIR')),'E process-local CPU metadata preparation only')
sample=sample_memory(me.pid,'inactive_ninth_final_ROOT_gate_input_materialization')
require(sample['paired_sampling_within_one_second']and sample['worker_private_memory']['private_bytes']<=536870912
    and sample['system_memory']['commit_available_bytes']>=5055306368
    and sample['system_memory']['physical_available_bytes']>=2305306368
    and shutil.disk_usage('E:/').free>=100000000000,'Original full metadata prospective/floors/FULLprefetch/E100GB')
draft_path=F/'ninth_ingress_frontend_DRAFT_inputs_PENDING_ROOT_gates_v1.json';draft=read(draft_path);draft_pin=pin(draft_path)
require(draft_pin['sha256']=='f3cce7b6d4afcbc223b4e4c817777c6426864e2151a45118caed020a07d2b918'
    and draft['overlay_source_review']is None and draft['CPU14_transport_compatibility_review']is None,'Unexecuted frozen draft with two explicit None gates')
overlay_gate=pin(F/'ROOT_ninth_overlay10_CPU14_inactive_source_constructor_review_v1.json')
transport_gate=pin(F/'ROOT_CPU14_one_overlay_import_only_transport_compatibility_successor_v1.json')
require(overlay_gate['bytes']==24672 and overlay_gate['sha256']=='64457284b30bd570875d5f5fa8db0fc995584a48f416959874e8f8924193d8be'
    and transport_gate['bytes']==4131 and transport_gate['sha256']=='b794ac22769d6eba2b78ddee87e40af8e0f89dbe23a32f96d8d580241aeabe41','Exact genuine new ROOT gate pins')
o=read(overlay_gate['path']);t=read(transport_gate['path'])
require(o['accepted']is True and o['reviewer']=='/root'and o['scope']=='ASR18_source_bound_ninth_original_full_clip_geometry_overlay_v1'
    and o['overlay_source']==draft['overlay_source']and o['certificate']==draft['certificate'],'Genuine exact ninth constructor source gate')
require(t['accepted']is True and t['reviewer']=='/root'and t['scope']=='ASR18_owned_two_logical_packet_one_receiver_transport_v1'
    and t['CPU_source']==draft['CPU14_source']and t['source_gate']==overlay_gate
    and t['maximum_logical_packets']==2 and t['maximum_live_receiver_packets']==1
    and t['one_import_only']is True and t['new_CPU14_transport_control_executed']is False and t['old_transport_control_rerun']is False,
    'Genuine derived CPU14 compatibility, no invented new transport test')
for value in(draft['overlay_source'],draft['CPU14_source'],draft['certificate']):require(pin(value['path'])==value,'Frozen candidate source/certificate unchanged, not imported')
old_handoff_path=F/'ninth_ingress_frontend_INACTIVE_fixture_handoff_v1.json';old_handoff=read(old_handoff_path)
for value in(old_handoff['fixture_source'],old_handoff['launcher_source'],old_handoff['full_RAW_inverse']):require(pin(value['path'])==value,'Frozen proof sources unchanged, no imports')
final=deepcopy(draft);final['overlay_source_review']=overlay_gate;final['CPU14_transport_compatibility_review']=transport_gate
require(all(v['path']not in{overlay_gate['path'],transport_gate['path']}for v in final['input_pins']),'Both gates are new exact final input pins')
final['input_pins'] +=[overlay_gate,transport_gate]
inverse=deepcopy(final);require(inverse['input_pins'][-2:]==[overlay_gate,transport_gate],'Exact ordered gate suffix')
inverse['input_pins']=inverse['input_pins'][:-2];inverse['overlay_source_review']=None;inverse['CPU14_transport_compatibility_review']=None
require(inverse==draft and encode(inverse)==draft_path.read_bytes(),'ENTIRE draft serialized RAW bytes reconstructed after only two gate fields plus two appended pins')
final_path=F/'ninth_ingress_frontend_frozen_inputs_v1.json';final_bytes=encode(final)
delta_path=F/'ninth_ingress_frontend_DRAFT_to_final_exact_metadata_delta_v1.json'
delta={'kind':'ASR18_INACTIVE_NINTH_DRAFT_to_final_exact_gate_metadata_delta_v1','old':draft_pin,'new':future(final_path,final_bytes),
    'JSON_operations':[{'path':'/overlay_source_review','old':None,'new':overlay_gate},
        {'path':'/CPU14_transport_compatibility_review','old':None,'new':transport_gate},
        {'path':'/input_pins','operation':'append_exact_ordered_suffix','old_count':len(draft['input_pins']),
            'new_count':len(final['input_pins']),'appended':[overlay_gate,transport_gate]}],
    'all_other_JSON_values_exact':True,'full_original_serialized_LF_RAW_reconstructed_after_only_declared_operations':True,
    'newline_normalization':'NONE; original JSON indentation/ensure_ascii=false/LF reproduced exactly',
    'historical_DRAFT_kind_description_retained_as_original_metadata_not_execution_gate':True,
    'no_source_or_config_or_plan_or_audio_or_actual_proof_operation':True}
delta_bytes=encode(delta)
handoff_path=F/'ninth_ingress_frontend_final_frozen_handoff_v1.json'
handoff={'kind':'ASR18_NINTH_original_ingress_frontend_final_frozen_source_handoff_v1','execution_admitted':False,
    'required_ROOT_CPU_execution_scope':'ASR18_ninth_exact_original_head_member_CPU_ingress_frontend_v1',
    'fixture_source':old_handoff['fixture_source'],'launcher_source':old_handoff['launcher_source'],
    'frozen_inputs':future(final_path,final_bytes),'input_pins':final['input_pins'],'input_count':len(final['input_pins']),
    'overlay_source':draft['overlay_source'],'CPU14_source':draft['CPU14_source'],'certificate':draft['certificate'],
    'overlay_source_review':overlay_gate,'CPU14_transport_compatibility_review':transport_gate,
    'exact_DRAFT_to_final_metadata_delta':future(delta_path,delta_bytes),'original_INACTIVE_fixture_handoff':pin(old_handoff_path),
    'full_RAW_fixture_external_inverse':old_handoff['full_RAW_inverse'],
    'full_RAW_overlay_CPU_inverse':pin(F/'ninth_overlay10_CPU14_full_RAW_inverse_and_aliases_v1.json'),
    'proper_source_base':draft['proper_source_base'],'original_prior8_policy_tree_remains_exact':True,
    'isolated_proof_configuration_created_only_by_future_admitted_fixture_in_fresh_output_after_exact3operational_fields':True,
    'no_production_driver_wrapper_config_materialized_or_imported':True,
    'original_plan_binding_declared_from_accepted_ROOT_UNREAD_during_preparation':draft['recorded_plan'],
    'historical_DRAFT_kind_in_final_manifest_is_preserved_description_ONLY_both_required_gates_now_genuine':True,
    'output_path':old_handoff['output_path'],'limits':old_handoff['limits'],
    'future_exact_external_arguments':old_handoff['future_exact_external_arguments'],
    'positive_and_negative_proof_scope':old_handoff['actual_positive'],'strict8_negatives':old_handoff['strict8_negatives'],
    'prior8_and_ordinary_delegation':old_handoff['prior8_and_ordinary_delegation'],
    'initial_numeric_full_resource_sample_BEFORE_predicate_required_in_future_execution_manual_preflight':True,
    'reader_unchanged_training5_stopped_no_recovery_or_speed_claim':True,
    'source_only_typed_transition_proposal':pin(F/'ninth_current_source36_typed_transition_SOURCE_ONLY_proposal_v1.json')}
prepared={final_path:final_bytes,delta_path:delta_bytes,handoff_path:encode(handoff)}
result_path=F/'ninth_ingress_frontend_final_input_materialization_result_v1.json'
require(all(not p.exists()for p in[*prepared,result_path]),'All immutable final metadata paths fresh BEFORE writes')
for p,data in prepared.items():p.write_bytes(data)
result={'kind':'ASR18_INACTIVE_ninth_final_input_materialization_result_v1','prepared':[pin(p)for p in prepared],
    'producer':identity(me),'launcher':identity(me.parent()),'materializer_source':pin(__file__),
    'initial_full_metadata_resource_sample':sample,'no_source_change_import_plan_read_proof_frontend_audio_or_production_action':True}
result_path.write_bytes(encode(result));print(json.dumps({'result':pin(result_path),**result}),flush=True)
