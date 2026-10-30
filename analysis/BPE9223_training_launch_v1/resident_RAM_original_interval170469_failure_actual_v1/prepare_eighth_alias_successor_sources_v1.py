"""INACTIVE alias-only successor and exact static export resolution; no imports/jobs."""
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
def pin(path):
    p=Path(path).resolve();data=p.read_bytes();require(len(data)<=8*1024**2,'Compact/source metadata only')
    return {'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def write(path,value):
    p=Path(path);require(not p.exists(),'Fresh immutable successor metadata')
    p.write_bytes((json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8'))
def source_successor(old,new,replacements):
    before=Path(old).read_bytes();text=before.decode('utf-8');ledger=[]
    for previous,next_value,count in replacements:
        actual=text.count(previous);require(actual>0 and (count is None or actual==count),'Exact source replacement: '+previous)
        text=text.replace(previous,next_value)
        ledger.append({'old_literal':previous,'new_literal':next_value,'expected_count':actual})
    ast.parse(text);require(not Path(new).exists(),'New immutable source path')
    Path(new).write_bytes(text.encode('utf-8'));inverse=text
    for item in reversed(ledger):
        require(inverse.count(item['new_literal'])==item['expected_count'],'Exact RAW reverse replacement')
        inverse=inverse.replace(item['new_literal'],item['old_literal'])
    require(inverse.encode('utf-8')==before,'Entire old RAW byte stream reconstructed')
    return {'old':pin(old),'new':pin(new),'newline_normalization':'NONE; original raw line endings retained',
        'replacements':ledger,'full_old_RAW_bytes_reconstructed':True}

overlay_old=S/'fastmss_original_clip_geometry_overlay_v8.py';overlay_new=S/'fastmss_original_clip_geometry_overlay_v9.py'
overlay_inverse=source_successor(overlay_old,overlay_new,[('decode_full_member=prior.prior.decode_full_member','decode_full_member=prior.decode_full_member',1)])
overlay_ledger=F/'overlay8_failed_to9_alias_only_exact_RAW_inverse_v1.json';write(overlay_ledger,overlay_inverse)
cpu_old=S/'fastmss_resident_feature_stream_v11.py';cpu_new=S/'fastmss_resident_feature_stream_v13.py'
cpu_inverse=source_successor(cpu_old,cpu_new,[('from fastmss_original_clip_geometry_overlay_v7 import OriginalRows',
    'from fastmss_original_clip_geometry_overlay_v9 import OriginalRows',1)])
cpu_ledger=F/'CPU11_to13_one_overlay_import_exact_RAW_inverse_v1.json';write(cpu_ledger,cpu_inverse)

# Static module bindings only. Do NOT import the failed or successor modules.
modules={f'fastmss_original_clip_geometry_overlay_v{i}':ast.parse((S/f'fastmss_original_clip_geometry_overlay_v{i}.py').read_bytes())for i in range(1,10)}
def resolve(module,name,trace,active=()):
    key=(module,name);require(key not in active,'No static export recursion')
    require(module in modules,'Only exact known pinned overlay modules')
    for node in modules[module].body:
        if isinstance(node,ast.FunctionDef)and node.name==name:
            trace.append({'module':module,'name':name,'binding':'original_function_definition','line':node.lineno})
            return {'kind':'function','module':module,'name':name,
                'AST_sha256':hashlib.sha256(ast.dump(node,include_attributes=False).encode('utf-8')).hexdigest()}
        if isinstance(node,ast.Import):
            for alias in node.names:
                if (alias.asname or alias.name)==name:
                    trace.append({'module':module,'name':name,'binding':'module_import','target_module':alias.name,'line':node.lineno})
                    return {'kind':'module','module':alias.name}
        if isinstance(node,ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name)==name:
                    trace.append({'module':module,'name':name,'binding':'named_import','target_module':node.module,'target_name':alias.name,'line':node.lineno})
                    return resolve(node.module,alias.name,trace,active+(key,))
        if isinstance(node,ast.Assign)and any(isinstance(target,ast.Name)and target.id==name for target in node.targets):
            trace.append({'module':module,'name':name,'binding':'assignment','expression':ast.unparse(node.value),'line':node.lineno})
            return expression(module,node.value,trace,active+(key,))
    raise ValueError('No statically exported binding: '+module+'.'+name)
def expression(module,node,trace,active):
    if isinstance(node,ast.Name):return resolve(module,node.id,trace,active)
    if isinstance(node,ast.Attribute):
        parent=expression(module,node.value,trace,active);require(parent['kind']=='module','Attribute target must resolve to original pinned module')
        return resolve(parent['module'],node.attr,trace,active)
    raise ValueError('Unsupported static export expression, no fallback')
trace_new=[];trace_old=[];trace_failed=[]
new_resolution=resolve('fastmss_original_clip_geometry_overlay_v9','decode_full_member',trace_new)
old_resolution=resolve('fastmss_original_clip_geometry_overlay_v7','decode_full_member',trace_old)
require(new_resolution==old_resolution and new_resolution['module']=='fastmss_original_clip_geometry_overlay_v1'
    and new_resolution['name']=='decode_full_member','New alias resolves to accepted original helper function, not merely equal text')
try:resolve('fastmss_original_clip_geometry_overlay_v8','decode_full_member',trace_failed)
except ValueError as error:
    failed_resolution=repr(error)
    require('fastmss_original_clip_geometry_overlay_v6.decode_full_member'in failed_resolution,'Static failed alias exactly matches actual import failure')
else:raise AssertionError('Frozen failed8 must retain missing export')
resolution=F/'eighth_helper_alias_exact_static_export_resolution_v1.json'
write(resolution,{'kind':'ASR18_static_owned_helper_export_resolution_no_runtime_import_v1',
    'module_source_pins':[pin(S/(name+'.py'))for name in modules],
    'accepted7_export_trace':trace_old,'corrected9_export_trace':trace_new,'same_original_helper_definition':new_resolution,
    'failed8_export_trace':trace_failed,'failed8_static_rejection':failed_resolution,
    'no_modules_imported_or_executed':True,'no_audio_or_numerical_or_state_operation':True})

proper=N/'resident_RAM_recurrent_metadata_slowdown_fix_v1/production_configuration_source31_v1/configuration.json'
base=read(proper);candidate=copy.deepcopy(base)
candidate['original_clip_geometry_overlay']={'prior_overlay':copy.deepcopy(base['original_clip_geometry_overlay']),
    'source':pin(overlay_new),'certificate':pin(F/'eighth_original_full_clip_geometry_certificate_v1.json'),'ROOT_review':None}
candidate['prior_resident_RAM_original_interval_failure']=pin(F/'actual_failure_closure.json')
candidate['resident_RAM_resource_scheduling_policy']['CPU_protocol_source']=pin(cpu_new)
delta=[key for key in candidate if candidate[key]!=base[key]]
require(set(delta)=={'original_clip_geometry_overlay','prior_resident_RAM_original_interval_failure','resident_RAM_resource_scheduling_policy'},
    'Only three unarmed template operational fields, original transport proof retained')
cfg=F/'configuration_source32_eighth_alias9_candidate_template_v3.json';write(cfg,candidate)

fixture_old=F/'verify_eighth_exact_ingress_frontend_v1.py';fixture_new=F/'verify_eighth_exact_ingress_frontend_v2.py'
fixture_inverse=source_successor(fixture_old,fixture_new,[
    ("OUTPUT=F/'eighth_ingress_frontend_actual_01'","OUTPUT=F/'eighth_ingress_frontend_actual_02'",1),
    ("INPUTS=F/'eighth_ingress_frontend_frozen_inputs_v1.json'","INPUTS=F/'eighth_ingress_frontend_frozen_inputs_v2.json'",1),
    ('from fastmss_resident_feature_stream_v12 import','from fastmss_resident_feature_stream_v13 import',1),
    ('fastmss_original_clip_geometry_overlay_v8','fastmss_original_clip_geometry_overlay_v9',2),
    ('CPU12','CPU13',None)])
fixture_ledger=F/'eighth_ingress_fixture1_to2_alias13_exact_RAW_inverse_v1.json';write(fixture_ledger,fixture_inverse)
external_old=F/'launch_eighth_exact_ingress_frontend_external_v1.py';external_new=F/'launch_eighth_exact_ingress_frontend_external_v2.py'
external_inverse=source_successor(external_old,external_new,[
    ("FIXTURE=F/'verify_eighth_exact_ingress_frontend_v1.py'","FIXTURE=F/'verify_eighth_exact_ingress_frontend_v2.py'",1),
    ("RESULT=F/'eighth_ingress_frontend_external_result_v1.json'","RESULT=F/'eighth_ingress_frontend_external_result_v2.json'",1),
    ("OUTPUT=F/'eighth_ingress_frontend_actual_01'","OUTPUT=F/'eighth_ingress_frontend_actual_02'",1),
    ('eighth_ingress_frontend_frozen_inputs_v1.json','eighth_ingress_frontend_frozen_inputs_v2.json',2),
    ("stdout=F/'eighth_ingress_frontend_stdout.log';stderr=F/'eighth_ingress_frontend_stderr.log'",
     "stdout=F/'eighth_ingress_frontend_stdout02.log';stderr=F/'eighth_ingress_frontend_stderr02.log'",1)])
external_ledger=F/'eighth_ingress_external1_to2_fresh02_exact_RAW_inverse_v1.json';write(external_ledger,external_inverse)
handoff=F/'eighth_alias9_CPU13_source_and_fresh02_inactive_handoff_v1.json'
write(handoff,{'kind':'ASR18_INACTIVE_eighth_alias_only_overlay9_CPU13_and_unexecuted_fresh02_source_handoff_v1',
    'execution_admitted':False,'overlay_source':pin(overlay_new),'certificate':pin(F/'eighth_original_full_clip_geometry_certificate_v1.json'),
    'CPU13_source':pin(cpu_new),'proper_source_base':pin(proper),'unarmed_candidate_template_v3':pin(cfg),
    'candidate_three_operational_source_fields':delta,'all_prior7_and_original_numeric_resource_credit_science_values_exact':True,
    'original_closed_transport_review_retained_pending_genuine_CPU13_compatibility_successor':True,
    'overlay_source_only_ROOT_gate_required':{'accepted':True,'reviewer':'/root',
        'scope':'ASR18_source_bound_eighth_original_full_clip_geometry_overlay_v1','overlay_source':pin(overlay_new),
        'certificate':pin(F/'eighth_original_full_clip_geometry_certificate_v1.json')},
    'CPU13_derived_transport_review_required':'Genuine ROOT source-compatible successor reusing closed original finite proof; no new transport test claim',
    'alias_only_entire_RAW_inverse':pin(overlay_ledger),'CPU11_oneimport_entire_RAW_inverse':pin(cpu_ledger),
    'static_alias_export_resolution':pin(resolution),'fixture2':pin(fixture_new),'external2':pin(external_new),
    'fixture_entire_RAW_inverse':pin(fixture_ledger),'external_entire_RAW_inverse':pin(external_ledger),
    'actual01_failed_closure':pin(F/'eighth_ingress_frontend_actual01_failed_completion_v1.json'),
    'all_failed01_sources_config_logs_and_completion_immutable_preserved':True,
    'no_actual02_frozen_inputs_materialized_until_two_genuine_ROOT_gates':True,
    'future_actual02_frozen_input_path':str((F/'eighth_ingress_frontend_frozen_inputs_v2.json').resolve()),
    'future_actual02_output':str((F/'eighth_ingress_frontend_actual_02').resolve()),
    'actual02_CPU_fixture_requires_separate_exact_ROOT_execution_scope_after_final_config_input_pins_frozen':True,
    'no_changes_to_actual_positive_frontend_or8negative_case_algorithm':True,
    'no_baseline_physical_old7audio_transport_rerun_or_GPU_native_source32_production_operation':True,
    'same_certificate_full99_T214_and_historical820896_273632_reservations':True,
    'same_all_original_private8GB_CPU2point5prospective_FULLprefetch_raw901350272_floors_and180240wallguards':True})
print(json.dumps({'overlay9':pin(overlay_new),'CPU13':pin(cpu_new),'candidate3':pin(cfg),'handoff':pin(handoff),
    'static_resolution':pin(resolution),'fixture2':pin(fixture_new),'external2':pin(external_new),
    'alias_RAW_inverse':pin(overlay_ledger),'CPU_RAW_inverse':pin(cpu_ledger),'no_source_import_or_job':True}))
