"""Inactive exact metadata materialization only; no source import or CPU/audio job."""
import ast
import copy
import hashlib
import json
from pathlib import Path

P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish'
N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1')
F=N/'resident_RAM_original_interval170469_failure_actual_v1'
def require(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def encoded(value):return (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')
def pin(path):
    p=Path(path).resolve();data=p.read_bytes();require(len(data)<=8*1024**2,'Compact/source only')
    return {'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def planned_pin(path,data):return {'path':str(Path(path).resolve()),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def write_exact(path,data,expected):
    with Path(path).open('xb')as file:file.write(data)
    require(pin(path)==expected,'Actual serialized LF bytes exactly match planned metadata pin')

fixture=F/'verify_eighth_exact_ingress_frontend_v2.py';external=F/'launch_eighth_exact_ingress_frontend_external_v2.py'
source_gate=pin(F/'ROOT_eighth_alias9_CPU13_inactive_source_review_v1.json')
credit_gate=pin(F/'ROOT_CPU13_overlay_import_only_transport_compatibility_successor_v1.json')
require(source_gate['sha256']=='fbd817459af476669632f93fd89a62cdfe785166c6b24b1f685ed6ffbdb79ae4'
    and credit_gate['sha256']=='7b1f87b098b0248f183faf348d3328312858b9bf77139243e95c6559371c2e1b','Exact two genuine new ROOT pins')
source=read(source_gate['path']);credit=read(credit_gate['path'])
overlay_pin=pin(S/'fastmss_original_clip_geometry_overlay_v9.py');cpu_pin=pin(S/'fastmss_resident_feature_stream_v13.py')
certificate_pin=pin(F/'eighth_original_full_clip_geometry_certificate_v1.json')
require(source['accepted']is True and source['reviewer']=='/root'
    and source['scope']=='ASR18_source_bound_eighth_original_full_clip_geometry_overlay_v1'
    and source['overlay_source']==overlay_pin and source['certificate']==certificate_pin
    and credit['accepted']is True and credit['reviewer']=='/root'
    and credit['scope']=='ASR18_owned_two_logical_packet_one_receiver_transport_v1'
    and credit['CPU_source']==cpu_pin and credit['new_CPU13_transport_control_executed']is False,
    'Genuine named source and derived transport gates, no new transport-test claim')
fixture_text=fixture.read_text(encoding='utf-8');external_text=external.read_text(encoding='utf-8')
ast.parse(fixture_text);ast.parse(external_text)
require('fastmss_original_clip_geometry_overlay_v8'not in fixture_text and 'fastmss_resident_feature_stream_v12'not in fixture_text
    and 'CPU12'not in fixture_text and 'eighth_ingress_frontend_actual_01'not in fixture_text
    and 'eighth_ingress_frontend_frozen_inputs_v1.json'not in fixture_text
    and 'eighth_ingress_frontend_actual_01'not in external_text
    and 'eighth_ingress_frontend_external_result_v1.json'not in external_text,
    'Actual02 executable routes contain no failed8/CPU12/old namespace; no source imports')
proper=N/'resident_RAM_recurrent_metadata_slowdown_fix_v1/production_configuration_source31_v1/configuration.json'
template=F/'configuration_source32_eighth_alias9_candidate_template_v3.json'
base=read(proper);candidate=read(template)
require(candidate['original_clip_geometry_overlay']['ROOT_review']is None
    and candidate['original_clip_geometry_overlay']['source']==overlay_pin
    and candidate['resident_RAM_resource_scheduling_policy']['CPU_protocol_source']==cpu_pin,
    'Immutable unarmed template targets corrected alias9/CPU13')
final=copy.deepcopy(candidate);final['original_clip_geometry_overlay']['ROOT_review']=source_gate
final['resident_RAM_packet_credit_policy']['actual_CPU_control_review']=credit_gate
delta=[key for key in final if final[key]!=base.get(key)]
require(set(delta)=={'original_clip_geometry_overlay','prior_resident_RAM_original_interval_failure',
    'resident_RAM_resource_scheduling_policy','resident_RAM_packet_credit_policy'},'Exactly four operational/provenance fields')
unchanged=copy.deepcopy(final)
for key in delta:unchanged[key]=copy.deepcopy(base[key])
resource=copy.deepcopy(final['resident_RAM_resource_scheduling_policy']);resource['CPU_protocol_source']=base['resident_RAM_resource_scheduling_policy']['CPU_protocol_source']
policy=copy.deepcopy(final['resident_RAM_packet_credit_policy']);policy['actual_CPU_control_review']=base['resident_RAM_packet_credit_policy']['actual_CPU_control_review']
require(unchanged==base and resource==base['resident_RAM_resource_scheduling_policy']
    and policy==base['resident_RAM_packet_credit_policy']
    and final['original_clip_geometry_overlay']['prior_overlay']==base['original_clip_geometry_overlay'],
    'All resource parameters/science/native scheduling/seven prior policies exact')
outputs=[F/'configuration_source32_eighth_alias9_scoped_source_candidate_v4.json',
    F/'eighth_alias9_scoped_candidate4_exact_configuration_delta_v1.json',
    F/'eighth_ingress_frontend_frozen_inputs_v2.json',F/'eighth_ingress_frontend_actual02_inactive_handoff_v2.json']
fresh=[*outputs,F/'eighth_ingress_frontend_actual_02',F/'eighth_ingress_frontend_external_result_v2.json',
    F/'eighth_ingress_frontend_stdout02.log',F/'eighth_ingress_frontend_stderr02.log']
require(all(not path.exists()for path in fresh),'All actual02 metadata/output destinations fresh before any write')
cfg_path,cfgdiff_path,inputs_path,handoff_path=outputs
cfg_data=encoded(final);cfg_pin=planned_pin(cfg_path,cfg_data)
cfgdiff={'proper_source_base':pin(proper),'immutable_template3':pin(template),'scoped_candidate4':cfg_pin,
    'four_operational_top_fields':delta,'template3_to_candidate4_only_gate_pin_fields':
    ['original_clip_geometry_overlay.ROOT_review','resident_RAM_packet_credit_policy.actual_CPU_control_review'],
    'all_original_numeric_resources_and_credit_helper_parameters_exact':True,'whole_prior7_tree_exact':True,
    'source_only_and_derived_compatibility_NOT_completed_ingress_or_production':True,
    'original_seven_runtime_binding_KINDS_remain_future_only_no_runtime_materialized':True}
cfgdiff_data=encoded(cfgdiff);cfgdiff_pin=planned_pin(cfgdiff_path,cfgdiff_data)
old_inputs=read(F/'eighth_ingress_frontend_frozen_inputs_v1.json')
extras=[source_gate,credit_gate,overlay_pin,cpu_pin,pin(template),pin(proper),
    pin(F/'eighth_alias9_CPU13_source_and_fresh02_inactive_handoff_v1.json'),
    pin(F/'eighth_helper_alias_exact_static_export_resolution_v1.json'),
    pin(F/'overlay8_failed_to9_alias_only_exact_RAW_inverse_v1.json'),
    pin(F/'CPU11_to13_one_overlay_import_exact_RAW_inverse_v1.json'),
    pin(F/'eighth_ingress_fixture1_to2_alias13_exact_RAW_inverse_v1.json'),
    pin(F/'eighth_ingress_external1_to2_fresh02_exact_RAW_inverse_v1.json'),
    pin(F/'eighth_ingress_frontend_actual01_failed_completion_v1.json'),
    pin(F/'eighth_ingress_frontend_actual01_failed_TOOL_v1.json')]
unique={}
for value in old_inputs['input_pins']+extras:
    actual=pin(value['path']);require(actual=={key:value[key]for key in('path','bytes','sha256')},'Actual unchanged source/metadata input pins')
    unique[actual['path']]=actual
input_pins=list(unique.values())+[cfg_pin,cfgdiff_pin]
frozen={'kind':'ASR18_INACTIVE_eighth_alias9_CPU13_real_ingress_actual02_frozen_inputs_v2',
    'proper_source_base':pin(proper),'candidate_configuration':cfg_pin,'overlay_source_review':source_gate,
    'CPU13_transport_compatibility_review':credit_gate,'overlay_source':overlay_pin,'CPU13_source':cpu_pin,
    'certificate':certificate_pin,'actual_failure_closure':old_inputs['actual_failure_closure'],
    'recorded_plan':old_inputs['recorded_plan'],'recorded_plan_canonical_sha256':old_inputs['recorded_plan_canonical_sha256'],
    'maximum_original_prospective_bytes':901350272,'input_pins':input_pins}
inputs_data=encoded(frozen);inputs_pin=planned_pin(inputs_path,inputs_data)
handoff={'kind':'ASR18_INACTIVE_eighth_alias9_CPU13_actual02_exact_CPU_source_handoff_v2','execution_admitted':False,
    'required_ROOT_CPU_scope':'ASR18_eighth_exact_original_head_member_CPU_ingress_frontend_v1',
    'fixture_source':pin(fixture),'launcher_source':pin(external),'frozen_inputs':inputs_pin,'input_pins':input_pins,
    'overlay_source_review':source_gate,'CPU13_derived_transport_compatibility_review':credit_gate,
    'candidate_configuration':cfg_pin,'configuration_delta':cfgdiff_pin,
    'source_alias9_and_CPU13_oneimport_no_failed8_CPU12_executable_route':True,
    'exact_scope_tag_is_same_semantic_CPU_interface_newactual_ROOT_scope_must_bind_newfixture_manifest_and_fresh02':True,
    'only_fixture1_to2_deltas':read(F/'eighth_ingress_fixture1_to2_alias13_exact_RAW_inverse_v1.json'),
    'only_external1_to2_deltas':read(F/'eighth_ingress_external1_to2_fresh02_exact_RAW_inverse_v1.json'),
    'static_alias_resolution':pin(F/'eighth_helper_alias_exact_static_export_resolution_v1.json'),
    'failed01_preserved':pin(F/'eighth_ingress_frontend_actual01_failed_completion_v1.json'),
    'actual_positive_scope':'ONE original CPU13_startup→alias9 eighth ingress for recorded plan154/index9/row6→originalCPU frontend/mask/finite/full99. No augmentation transform/native/model/GPU/sampler draw/credit transport constructor.',
    'expected_features_shape':[1,1711,128],'expected_valid_mask_frames':1710,'expected_T':214,'full_target_ids':99,
    'same8strict_negatives_and_METHOD_STUB_geometry_loader_prior7ordinary_delegation_restored':True,
    'wrongmember_negative_hashes_samecompressed_extent_before_PCM_no_seconddecode':True,
    'same_original_private8GB_CPU2point5prospective_raw901350272_FULLprefetch_floors_E100GB_180240guards':True,
    'limits':{'fixture_wall_seconds':180,'external_wall_seconds':240,'sampled_family_private_bytes':8000000000,
        'CPU_private_bytes':8000000000,'CPU_prospective_commit_reserve_bytes':2500000000,'FULLprefetch_bytes':268435456,
        'full_original_raw_media_MAXfrontend_transform_numerical_prospective_bytes':901350272,
        'external_admission_commit_bytes':3669785728,'external_admission_physical_bytes':2669785728,
        'runtime_commit_floor_plus_prefetch_bytes':2268435456,'runtime_physical_floor_plus_prefetch_bytes':1768435456,
        'E_free_bytes':100000000000,'CUDA_VISIBLE_DEVICES':'-1','four_environment_threadcaps':1,'BelowNormal':True},
    'output_path':str((F/'eighth_ingress_frontend_actual_02').resolve()),
    'future_exact_external_arguments':['E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe','-Xutf8','-B',str(external),'--ROOT-review','<NEW_EXACT_ROOT_CPU02_SCOPE_PATH>'],
    'actual02_not_executed_separate_ROOT_exactscope_required':True,'no_automaticretry_source32_or_production_launch':True,
    'actual_guard_error_None_even_when_negative_catches_ValueError_required':True,
    'actual_CPU_gate_schema_after_genuine_pass':{'accepted':True,'reviewer':'/root',
        'scope':'ASR18_source_bound_eighth_original_full_clip_geometry_overlay_v1','overlay_source':overlay_pin,
        'certificate':certificate_pin,'actual_CPU_control_report':'REAL immutable report pin','all_exact_metadata_identities_dead':True}}
handoff_data=encoded(handoff);handoff_pin=planned_pin(handoff_path,handoff_data)
# Every assertion above occurs before writing any of the four new artifacts.
for path,data,expected in[(cfg_path,cfg_data,cfg_pin),(cfgdiff_path,cfgdiff_data,cfgdiff_pin),
    (inputs_path,inputs_data,inputs_pin),(handoff_path,handoff_data,handoff_pin)]:write_exact(path,data,expected)
print(json.dumps({'candidate4':pin(cfg_path),'frozen_inputs2':pin(inputs_path),'handoff2':pin(handoff_path),
    'configuration_delta':pin(cfgdiff_path),'input_count':len(input_pins),'fixture2':pin(fixture),'external2':pin(external),
    'no_source_import_or_CPU_audio_proof_execution':True}))
