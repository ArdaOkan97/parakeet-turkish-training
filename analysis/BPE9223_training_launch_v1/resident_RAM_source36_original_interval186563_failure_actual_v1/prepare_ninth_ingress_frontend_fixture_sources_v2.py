"""INACTIVE fixture/launcher/draft-input preparation only, no candidate imports."""
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish'
N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1')
F=N/'resident_RAM_source36_original_interval186563_failure_actual_v1';OLD=N/'resident_RAM_original_interval170469_failure_actual_v1'
OLD_FIXTURE=OLD/'verify_eighth_exact_ingress_frontend_v2.py';OLD_EXTERNAL=OLD/'launch_eighth_exact_ingress_frontend_external_v2.py'
FIXTURE=F/'verify_ninth_exact_ingress_frontend_v1.py';EXTERNAL=F/'launch_ninth_exact_ingress_frontend_external_v1.py'
def require(v,m):
    if not v:raise ValueError(m)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
    p=Path(p).resolve();before=p.stat();require(before.st_size<=8388608,'Compact/source input only, never media/journal/payload')
    data=p.read_bytes();after=p.stat();require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Stable source/metadata pin')
    return {'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def encode(v):return (json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
def future(p,data):return {'path':str(p.resolve()),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def identity(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
def guard():
    v=sample_memory(me.pid,'inactive_ninth_ingress_frontend_fixture_SOURCE_preparation')
    require(v['paired_sampling_within_one_second']and v['worker_private_memory']['private_bytes']<=536870912
        and v['system_memory']['commit_available_bytes']>=5055306368 and v['system_memory']['physical_available_bytes']>=2305306368
        and shutil.disk_usage('E:/').free>=100000000000,'Original full metadata prospective/floors/FULLprefetch/E100GB')
    return v
initial=guard()
scope_path=F/'ROOT_recorded186564_row5_NINTH_certificate_overlay10_CPU14_INACTIVE_preparation_scope_v1.json';scope_pin=pin(scope_path)
require(scope_pin['sha256']=='fa409fdaeb7a91158377c73b712dc2306f6353040b85a951a13616e943e1db95','Exact preparation-only scope')
scope=read(scope_path);config_path=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01/invocations/owner_000025/configuration.json')
config_pin=pin(config_path);require(config_pin['sha256']=='f866afb3e9cd7e8db15c341e7df5ecbc8c6cc7d6714153783c389fafc54498a7','Current source36 actual runtime proper base, not stale33 base')
config=read(config_path);old_text=OLD_FIXTURE.read_bytes().decode('utf-8');text=old_text;changes=[]
external_text=OLD_EXTERNAL.read_bytes().decode('utf-8');external_changes=[]
def change(which,before,after,count=1):
    global text,external_text
    value=text if which=='fixture'else external_text
    require(value.count(before)==count,'Exact source replacement count: '+before[:90]);value=value.replace(before,after)
    item={'old_literal':before,'new_literal':after,'expected_count':count}
    if which=='fixture':text=value;changes.append(item)
    else:external_text=value;external_changes.append(item)
change('fixture','eighth','ninth',old_text.count('eighth'))
change('fixture','CPU13','CPU14',old_text.count('CPU13'))
for before,after in(('273624','210280'),('273632','210288'),('820872','630840'),('820896','630864'),('1711','1315'),('1710','1314'),('214','165'),('99','66')):
    change('fixture',before,after,text.count(before))
change('fixture','row6','row5',text.count('row6'))
change('fixture','Prior seven/ordinary','Prior eight/ordinary')
change('fixture',"OUTPUT=F/'ninth_ingress_frontend_actual_02'","OUTPUT=F/'ninth_ingress_frontend_actual_01'")
change('fixture',"INPUTS=F/'ninth_ingress_frontend_frozen_inputs_v2.json'","INPUTS=F/'ninth_ingress_frontend_frozen_inputs_v1.json'")
change('fixture','from fastmss_resident_feature_stream_v13 import','from fastmss_resident_feature_stream_v14 import')
change('fixture','fastmss_original_clip_geometry_overlay_v9','fastmss_original_clip_geometry_overlay_v10',2)
before=text[text.index("    base=read(fixed['proper_source_base']['path']);candidate=read(fixed['candidate_configuration']['path'])"):text.index('    OUTPUT.mkdir();configuration=')]
after="""    base=read(fixed['proper_source_base']['path'])
    # Isolated CPU-proof configuration only, created AFTER actual ROOT scope.
    # No production configuration or future driver/wrapper is materialized.
    runtime=copy.deepcopy(base)
    runtime['original_clip_geometry_overlay']={'prior_overlay':copy.deepcopy(base['original_clip_geometry_overlay']),
        'source':fixed['overlay_source'],'certificate':fixed['certificate'],'ROOT_review':fixed['overlay_source_review']}
    runtime['resident_RAM_resource_scheduling_policy']['CPU_protocol_source']=fixed['CPU14_source']
    runtime['resident_RAM_packet_credit_policy']['actual_CPU_control_review']=fixed['CPU14_transport_compatibility_review']
    inverse=copy.deepcopy(runtime)
    for key in ('original_clip_geometry_overlay','resident_RAM_resource_scheduling_policy','resident_RAM_packet_credit_policy'):
        inverse[key]=copy.deepcopy(base[key])
    require(inverse==base and runtime['original_clip_geometry_overlay']['prior_overlay']==base['original_clip_geometry_overlay'],
        'Exactly3 isolated CPU-proof operational fields; full original current-source36 values and prior8 preserved')
    require(checked_resource_profile(runtime)==runtime['resident_RAM_resource_scheduling_policy'],
        'Actual CPU14 resource binding, every original byte/floor/reserve unchanged')
    require(checked_packet_credit_policy(runtime)==runtime['resident_RAM_packet_credit_policy'],
        'Genuine derived CPU14 import-only compatibility accepts closed original transport proof; no transport constructor or rerun')
"""
change('fixture',before,after)
change('fixture',"'Actual fixed plan154 canonical identity'","'Actual fixed plan389 canonical identity; original pure metadata normalizer only'")
change('fixture',"        update=normalized[9];expected=update['rows'][5]\n        row=copy.deepcopy(update['coordinator_update']['rows'][5]);context=update['coordinator_update']['contexts'][5]",
    "        update=normalized[3];expected=update['rows'][4]\n        row=copy.deepcopy(update['coordinator_update']['rows'][4]);context=update['coordinator_update']['contexts'][4]")
change('fixture',"row['broad_original']['ordinal']==332560","row['broad_original']['ordinal']==156134")
change('fixture','alania:aedf14e56cf50416170171503e0dee9a_00000','alania:a280321c852958fbb53e9d7a2044d1a6_00000')
change('fixture',"'Exact real normalized ingress of recorded plan154/index9/row5/full66'",
    "'Exact real normalized ingress of recorded plan389/index3/row5/full66'")
change('fixture',"        write_new(OUTPUT/'before_real_ingress.json',{'recorded_plan':fixed['recorded_plan'],",
    "        require(row==fixed['row_identity']['runtime_call_row']and expected==fixed['row_identity']['expected_argument'],\n            'Entire exact current row/fulltargets/weakTR context agree with accepted ROOT identity before decode')\n        write_new(OUTPUT/'before_real_ingress.json',{'recorded_plan':fixed['recorded_plan'],")
change('fixture',"'local_update_index':9,'original_row_1based':6,'global_update':170470,",
    "'local_update_index':3,'original_row_1based':5,'global_update':186564,")
change('fixture',"changed['end']=17.101","changed['end']=13.142")
change('fixture','All seven original policy dispatch bindings','All eight original policy dispatch bindings')
change('fixture','original seven-certificate policy tree','original eight-certificate policy tree')
change('fixture','all_seven_prior_row_routes','all_eight_prior_row_routes')
change('fixture','prior7 arguments','prior8 arguments')
change('fixture','changed_prior7_certificate_pin','changed_prior8_certificate_pin')
change('fixture','prior7_policy_exact','prior8_policy_exact')
change('fixture',"require(len(prior_rows)==7,","require(len(prior_rows)==8,")
change('fixture',"        reader.close();require(reader.connections==[]and reader.locks is None,'Original scalar DB connections/locks actually closed')",
    "        require(guard_error is None,'Negative controls cannot promote an original resource guard failure')\n        reader.close();require(reader.connections==[]and reader.locks is None,'Original scalar DB connections/locks actually closed')")
change('external','eighth','ninth',external_text.count('eighth'))
change('external','row6','row5',external_text.count('row6'))
change('external',"FIXTURE=F/'verify_ninth_exact_ingress_frontend_v2.py'","FIXTURE=F/'verify_ninth_exact_ingress_frontend_v1.py'")
change('external',"RESULT=F/'ninth_ingress_frontend_external_result_v2.json'","RESULT=F/'ninth_ingress_frontend_external_result_v1.json'")
change('external',"OUTPUT=F/'ninth_ingress_frontend_actual_02'","OUTPUT=F/'ninth_ingress_frontend_actual_01'")
change('external','PROSPECTIVE=901350272','PROSPECTIVE=901306112')
change('external','ninth_ingress_frontend_frozen_inputs_v2.json','ninth_ingress_frontend_frozen_inputs_v1.json',2)
change('external',"stdout=F/'ninth_ingress_frontend_stdout02.log';stderr=F/'ninth_ingress_frontend_stderr02.log'",
    "stdout=F/'ninth_ingress_frontend_stdout01.log';stderr=F/'ninth_ingress_frontend_stderr01.log'")
change('external','(2456,38964,26892,41672,34044,40012,31044)','(26892,30128,21912,37108,29600,39856,35440)',2)
fixture_bytes=text.encode('utf-8');external_bytes=external_text.encode('utf-8');ast.parse(fixture_bytes);ast.parse(external_bytes)
for current,prior,edits in((text,OLD_FIXTURE.read_bytes(),changes),(external_text,OLD_EXTERNAL.read_bytes(),external_changes)):
    inverse=current
    for edit in reversed(edits):
        require(inverse.count(edit['new_literal'])==edit['expected_count'],'Exact inverse count: '+edit['new_literal'][:90])
        inverse=inverse.replace(edit['new_literal'],edit['old_literal'])
    require(inverse.encode('utf-8')==prior,'ENTIRE original source RAW reconstructed')
cert_path=F/'ninth_original_full_clip_geometry_certificate_v1.json'
source14=S/'fastmss_resident_feature_stream_v14.py';overlay=S/'fastmss_original_clip_geometry_overlay_v10.py'
old_overlay=S/'fastmss_original_clip_geometry_overlay_v9.py'
# Only explicitly allowed SOURCE / current config / ROOT receipt / immutable
# certificate metadata reads occur here; no original plan or raw report reads.
values=[scope_pin,scope['actual_physical_ROOT'],scope['existing_localization_ROOT'],config_pin,
    pin(cert_path),pin(overlay),pin(source14),pin(old_overlay),pin(S/'fastmss_resident_feature_stream_v13.py'),
    pin(OLD_FIXTURE),pin(OLD_EXTERNAL),pin(F/'ninth_overlay10_CPU14_full_RAW_inverse_and_aliases_v1.json'),
    pin('E:/gpu_training_scratchpad/storage_config.json'),pin('E:/gpu_training_scratchpad/use_storage.ps1'),
    pin('E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe')]
template=read(config['CPU_template_configuration']['path'])
values.extend((config['CPU_template_configuration'],config['session'],template['source_contract_template']['numerical_CPU_contract']))
policy=config['original_clip_geometry_overlay'];prior8=[]
while policy is not None:
    c=read(policy['certificate']['path']);prior8.append({'source':policy['source'],'certificate':policy['certificate'],'ROOT_review':policy['ROOT_review'],
        'ordinal':c['original_row']['ordinal'],'source_id':c['original_row']['unique_id']})
    values.extend((policy['source'],policy['certificate'],policy['ROOT_review']));policy=policy.get('prior_overlay')
require(len(prior8)==8,'All original eight policy bindings, no old raw proof read')
for name in('fastmss_BPE9223_cpu_common_v12.py','fastmss_BPE9223_plan_bridge_v12.py','fastmss_BPE9223_feature_engine_v12.py',
    'fastmss_singlepass_CPU_v1.py','fastmss_bounded_original_audio_v2.py','fastmss_raw_lifetime_v1.py','resource_watchdog_v2.py'):
    values.append(pin(S/name))
unique={}
for value in values:
    expected={k:value[k]for k in('path','bytes','sha256')};actual=pin(value['path']);require(actual==expected,'Current frozen SOURCE/ROOT/metadata input exact')
    unique[actual['path']]=actual
# The already accepted plan pin is declared, NOT opened/hashed during prep.
plan=deepcopy(scope['exact_original_row_identity']['plan']);unique[plan['path']]=plan
draft_inputs={'kind':'ASR18_INACTIVE_NINTH_original_ingress_frontend_input_DRAFT_PENDING_ROOT_gates_v1',
    'overlay_source_review':None,'CPU14_transport_compatibility_review':None,
    'overlay_source':pin(overlay),'CPU14_source':pin(source14),'certificate':pin(cert_path),'proper_source_base':config_pin,
    'row_identity':scope['exact_original_row_identity'],'recorded_plan':plan,
    'recorded_plan_canonical_sha256':scope['exact_original_row_identity']['canonical_plan_sha256'],
    'maximum_original_prospective_bytes':901306112,'all_original_prior8_policies':prior8,'input_pins':list(unique.values()),
    'source_only_plan_binding_from_accepted_ROOT_not_rehashed_or_read_during_preparation':True,
    'future_final_inputs_path':str((F/'ninth_ingress_frontend_frozen_inputs_v1.json').resolve()),
    'genuine_overlay_source_gate_AND_derived_transport_successor_REQUIRED_before_final_inputs_materialization':True,
    'not_production_configuration_and_not_execution_admission':True}
draft_path=F/'ninth_ingress_frontend_DRAFT_inputs_PENDING_ROOT_gates_v1.json';draft_bytes=encode(draft_inputs)
ledger_path=F/'ninth_ingress_frontend_fixture_external_full_RAW_inverses_v1.json'
ledger={'kind':'ASR18_INACTIVE_NINTH_ingress_frontend_full_RAW_inverse_v1','newline_normalization':'NONE; old exact LF bytes retained',
    'sources':[{'old':pin(OLD_FIXTURE),'new':future(FIXTURE,fixture_bytes),'replacements':changes,'entire_old_RAW_reconstructed':True},
        {'old':pin(OLD_EXTERNAL),'new':future(EXTERNAL,external_bytes),'replacements':external_changes,'entire_old_RAW_reconstructed':True}],
    'no_source_import_candidate_proof_or_numerical_execution':True}
ledger_bytes=encode(ledger)
handoff_path=F/'ninth_ingress_frontend_INACTIVE_fixture_handoff_v1.json'
handoff={'kind':'ASR18_INACTIVE_NINTH_one_original_ingress_frontend_fixture_handoff_v1','execution_admitted':False,
    'preparation_ROOT_scope':scope_pin,'fixture_source':future(FIXTURE,fixture_bytes),'launcher_source':future(EXTERNAL,external_bytes),
    'full_RAW_inverse':future(ledger_path,ledger_bytes),'DRAFT_inputs_PENDING_genuine_ROOT_gates':future(draft_path,draft_bytes),
    'required_future_ROOT_CPU_scope':'ASR18_ninth_exact_original_head_member_CPU_ingress_frontend_v1',
    'source_gate_REQUIRED':'ASR18_source_bound_ninth_original_full_clip_geometry_overlay_v1',
    'transport_gate_REQUIRED':'Genuine derived CPU14 compatibility successor, reusing closed original proof; no new transport test',
    'no_final_input_file_or_production_config_materialized_yet':True,
    'actual_CPU_proof_configuration':'Construct only inside fresh future admitted CPU proof output from exact currentf866 base after3operational fields: new overlay/certificate/genuineROOTsourcegate, CPU_protocol_source14, genuine transport provenance. All other bytes/values and prior8 tree exact.',
    'actual_positive':'One actual CPU14_startup original immutable DATA/hash/source guard + current root-bound plan389/index3/row5 normalize + full210280 clean ingress + unchanged original CPUfrontend; finite[1,1315,128]/1314valid/T165/full66; no weaktransform/model/GPU/transport constructor/authority draws',
    'actual_plan_metadata_read_must_be_explicitly_bound_in_future_ROOT_execution':'Only exact immutable plan389 pure normalize/pins; no journal/prefix/history/source-population selection',
    'no_extra_old_DATA_or_prior_proof_audit':'Existing original _startup/overlay prior8 source/immutable binding checks remain code-identical; no additional audit/old audio or transport reruns',
    'strict8_negatives':['annotation changed','historical declaration changed','full target changed','row ordinal changed',
        'source SHA before decode','member SHA with ONLY same compressed hash extent/no second PCM decode',
        'actual physical geometry via explicit METHOD_STUB metadata loader','prior8 certificate pin changed'],
    'prior8_and_ordinary_delegation':'Actual ninth-layer __call__ with labelled METHOD_STUB parent; full original arguments and8route keys, no old PCM; all stubs restored',
    'resource_guard_error_cannot_be_promoted_by_negative_ValueError_catch':True,
    'limits':{'fixture_wall_seconds':180,'external_wall_seconds':240,'sampled_whole_family_private_bytes':8000000000,
        'CPU_prospective_commit_reserve_bytes':2500000000,'CPU_actual_commit_headroom_bytes':1500000000,'FULL_prefetch_bytes':268435456,
        'full_proof_numerical_raw_media_frontend_transform_prospective_bytes':901306112,'external_admission_commit_bytes':3669741568,
        'external_admission_physical_bytes':2669741568,'runtime_commit_plus_prefetch_bytes':2268435456,
        'runtime_physical_plus_prefetch_bytes':1768435456,'E_free_bytes':100000000000,'all4threadcaps':1,'CUDA_VISIBLE_DEVICES':'-1',
        'priority':'BelowNormal','metadata_file_pin_cap_bytes':8388608,'sampled_not_hard_allocation_family_or_output_limit':True},
    'output_path':str((F/'ninth_ingress_frontend_actual_01').resolve()),
    'future_exact_external_arguments':['E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe','-Xutf8','-B',str(EXTERNAL),'--ROOT-review','<FUTURE_EXACT_ROOT_EXECUTION_SCOPE>'],
    'science_225_MAX2_LIVE1_allbyteguards_original_reservations_unchanged':True,
    'no_certificate_import_or_actual_proof_frontend_weaktransform_GC_state_or_production_action':True}
prepared={FIXTURE:fixture_bytes,EXTERNAL:external_bytes,draft_path:draft_bytes,ledger_path:ledger_bytes,handoff_path:encode(handoff)}
result_path=F/'ninth_ingress_frontend_fixture_preparation_result_v2.json'
require(all(not p.exists()for p in[*prepared,result_path]),'Fresh inactive source destinations BEFORE writes')
final=guard()
for path,data in prepared.items():path.write_bytes(data)
result={'kind':'ASR18_INACTIVE_ninth_ingress_frontend_fixture_preparation_result_v1','prepared':[pin(path)for path in prepared],
    'producer':identity(me),'launcher':identity(me.parent()),'preparation_source':pin(__file__),
    'initial_metadata_guard':initial,'final_metadata_guard':final,'declared_current_input_count_including_UNREAD_accepted_plan':len(unique),
    'no_candidate_import_original_plan_read_proof_audio_frontend_or_production_operation':True}
result_path.write_bytes(encode(result));print(json.dumps({'result':pin(result_path),**result}),flush=True)
