"""Inactive source/certificate/configuration preparation only; no numerical imports."""
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path

P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish'
N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1')
F=N/'resident_RAM_original_interval170469_failure_actual_v1'
BASE=N/'resident_RAM_recurrent_metadata_slowdown_fix_v1/production_configuration_source31_v1/configuration.json'
OLD=S/'fastmss_original_clip_geometry_overlay_v7.py';NEW=S/'fastmss_original_clip_geometry_overlay_v8.py'
CPU_OLD=S/'fastmss_resident_feature_stream_v11.py';CPU_NEW=S/'fastmss_resident_feature_stream_v12.py'
CERT=F/'eighth_original_full_clip_geometry_certificate_v1.json'
def require(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):
    p=Path(path).resolve();data=p.read_bytes();require(len(data)<=8*1024**2,'Compact/source only')
    return {'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def write(path,value):
    p=Path(path);require(not p.exists(),'Fresh immutable source metadata');p.write_bytes((json.dumps(value,indent=2,ensure_ascii=False)+'\n').encode('utf-8'))

physical=read(F/'full_head_member_physical_actual_01/report.json')
failure=read(F/'ordered14_decoder_actual_01/first_actual_exception.json')
item=failure['identity']['candidate']['full_original_plan_row'];row=physical['row'];decl=physical['original_declaration']
require(row==item['broad_original']and row['ordinal']==332560 and row['end']==17.102 and len(physical['full_target_ids'])==99,
    'Exact accepted physical/plan original row and all99 targets')
actual={k:physical['actual_interval_geometry'][k]for k in('native_sr','native_samples','clean16k_samples','mel_frames','encoder_frames','target_tokens')}
certificate={'kind':'ASR18_source_bound_one_original_full_clip_geometry_certificate_v1','original_row':row,
    'original_row_sha256':digest(row),'original_declaration':decl,'full_target_ids':physical['full_target_ids'],
    'full_target_ids_sha256':digest(physical['full_target_ids']),'raw_label_sha256':physical['raw_label_text_sha256'],
    'original_TAR':physical['original_TAR'],'recording_index_entry':physical['recording_index_entry'],
    'member_sha256':physical['member_sha256'],'actual_geometry':actual,'decoded_frames':physical['decoded_frames'],
    'max_abs_PTS_residual_samples':physical['max_abs_PTS_residual_samples'],
    'native_f32_sha256':physical['full_source_native_f32_sha256'],'clean_f32_sha256':physical['clean_f32_sha256'],
    'clean_i16_sha256':physical['clean_i16_sha256'],
    'independent_full_member_observation':pin(F/'full_head_member_physical_actual_01/report.json'),
    'actual_failure':pin(F/'actual_failure_closure.json'),
    'ordered_failure_localization':pin(F/'ordered14_decoder_actual_01/report.json'),
    'actual_causal_CPU_closure':pin(F/'full_head_member_physical_actual_completion_v1.json'),
    'original_annotations_indexes_and_plans_unchanged':True,'all_original_physical_samples_once':True,
    'full_labels_order_and_full_pass_denominator_unchanged':True,'no_generic_EOF_tolerance_or_clamp':True,
    'historical_native_reservation_retained':820896,'actual_physical_native_samples':820872,
    'historical_clean_reservation_retained':273632,'actual_full_clean_samples':273624,
    'unchanged_mel1710_features1711_T214_and_all99_targets':True,
    'actual_raw_codec_side_data':physical['packet_side_data'],
    'independent_physical_ROOT_review':pin(F/'ROOT_recorded170470_row6_full_head_member_physical_actual_CPU_review_v1.json')}
write(CERT,certificate)
old=OLD.read_bytes();text=old.decode('utf-8');edits=[]
def change(before,after,count):
    global text
    require(text.count(before)==count,'Exact overlay replacement count: '+before)
    text=text.replace(before,after);edits.append({'old_literal':before,'new_literal':after,'expected_count':count})
change('Seventh exact','Eighth exact',1)
change('mel1444 versus declared1444, with T181/U61.','mel1710 versus declared1710, with T214/U99.',1)
change('fastmss_original_clip_geometry_overlay_v6','fastmss_original_clip_geometry_overlay_v7',2)
change('alania:0ce0d55b73c946f80abea8bbe41f3b27_00000','alania:aedf14e56cf50416170171503e0dee9a_00000',1)
change('210880','332560',1)
change('ASR18_source_bound_seventh_original_full_clip_geometry_overlay_v1','ASR18_source_bound_eighth_original_full_clip_geometry_overlay_v1',1)
change('14.443','17.102',1)
change("len(certificate['full_target_ids']) == 61","len(certificate['full_target_ids']) == 99",1)
change('native_samples=693264, clean16k_samples=231088, mel_frames=1444,','native_samples=820896, clean16k_samples=273632, mel_frames=1710,',1)
change("encoder_frames=181, target_tokens=61, status='declared')","encoder_frames=214, target_tokens=99, status='declared')",1)
change('native_sr=48000, native_samples=693255,','native_sr=48000, native_samples=820872,',1)
change('clean16k_samples=231085, mel_frames=1444, encoder_frames=181, target_tokens=61)','clean16k_samples=273624, mel_frames=1710, encoder_frames=214, target_tokens=99)',1)
change("certificate['decoded_frames'] == 723","certificate['decoded_frames'] == 856",1)
change('seventh_overlay_policy','eighth_overlay_policy',4)
change('seventh_certificate','eighth_certificate',9)
change('its six prior certified rows','its seven prior certified rows',1)
ast.parse(text);require(not NEW.exists(),'Fresh immutable overlay8');NEW.write_bytes(text.encode('utf-8'))
inverse=text
for edit in reversed(edits):
    require(inverse.count(edit['new_literal'])==edit['expected_count'],'Reverse exact overlay literal')
    inverse=inverse.replace(edit['new_literal'],edit['old_literal'])
require(inverse.encode('utf-8')==old,'Entire overlay7 RAW bytes inverse')
ledger=F/'overlay7_to8_exact_raw_replacement_ledger_v1.json'
write(ledger,{'old':pin(OLD),'new':pin(NEW),'newline_normalization':'NONE; original CRLF bytes retained',
    'replacements':edits,'full_old_raw_reconstructed':True})
cpu_old=CPU_OLD.read_bytes();before=b'from fastmss_original_clip_geometry_overlay_v7 import OriginalRows';after=b'from fastmss_original_clip_geometry_overlay_v8 import OriginalRows'
require(cpu_old.count(before)==1 and not CPU_NEW.exists(),'Exactly one CPU11 overlay import replacement')
cpu_new=cpu_old.replace(before,after);ast.parse(cpu_new);CPU_NEW.write_bytes(cpu_new)
require(cpu_new.replace(after,before)==cpu_old,'Entire CPU11 RAW inverse')
cpu_ledger=F/'CPU11_to12_one_overlay_import_raw_inverse_v1.json'
write(cpu_ledger,{'old':pin(CPU_OLD),'new':pin(CPU_NEW),'newline_normalization':'NONE; original CRLF bytes retained',
    'replacements':[{'old_literal':before.decode(),'new_literal':after.decode(),'expected_count':1}],
    'full_old_raw_reconstructed':True,'all_transport_resource_credit_numerical_methods_byte_unchanged':True})
base=read(BASE);candidate=copy.deepcopy(base)
candidate['original_clip_geometry_overlay']={'prior_overlay':copy.deepcopy(base['original_clip_geometry_overlay']),
    'source':pin(NEW),'certificate':pin(CERT),'ROOT_review':None}
candidate['prior_resident_RAM_original_interval_failure']=pin(F/'actual_failure_closure.json')
candidate['resident_RAM_resource_scheduling_policy']['CPU_protocol_source']=pin(CPU_NEW)
delta=[k for k in candidate if candidate[k]!=base[k]]
require(set(delta)=={'original_clip_geometry_overlay','prior_resident_RAM_original_interval_failure','resident_RAM_resource_scheduling_policy'},
    'Only three source configuration operational fields')
inverse_cfg=copy.deepcopy(candidate)
for key in delta:inverse_cfg[key]=copy.deepcopy(base[key])
require(inverse_cfg==base,'Every original source configuration value exact')
cfg=F/'configuration_source32_overlay8_candidate_template_v1.json';write(cfg,candidate)
runtime=read('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01/invocations/owner_000023/configuration.json')
runtime_keys=[k for k in runtime if runtime.get(k)!=base.get(k)]
config_diff=F/'overlay8_candidate_source_base_and_runtime_difference_v1.json'
write(config_diff,{'proper_source_base':pin(BASE),'candidate':pin(cfg),'three_operational_source_fields':delta,
    'source_full_inverse_after_only_three_fields':True,'prior_seven_overlay_policy_exact':True,
    'all_resource_values_exact_except_CPU12_protocol_source':True,'runtime_source_not_confused_with_source_template':True,
    'historical_I23_effective_vs_proper_source_keys':runtime_keys,
    'actual_current_runtime_binding_values':{key:runtime.get(key)for key in runtime_keys},
    'future_seven_runtime_binding_KINDS_unchanged':'source_configuration,worker_driver_source,parent_driver_source,resident_RAM_stream_configuration,resident_branch_namespace,source_checkpoint,checkpoint_interval_updates',
    'future_saved170400_binding_only_after_final_review':True})
handoff=F/'eighth_overlay_certificate_CPU12_inactive_source_handoff_v1.json'
write(handoff,{'kind':'ASR18_INACTIVE_eighth_exact_original_head_member_geometry_source_handoff_v1',
    'overlay_source':pin(NEW),'certificate':pin(CERT),'CPU12_source':pin(CPU_NEW),'candidate_configuration':pin(cfg),
    'proper_source_base':pin(BASE),'configuration_difference_report':pin(config_diff),'overlay_entire_RAW_inverse':pin(ledger),
    'CPU_entire_RAW_inverse':pin(cpu_ledger),'source_only_scope_required':'ASR18_source_bound_eighth_original_full_clip_geometry_overlay_v1',
    'ROOT_review_gate_requires':{'accepted':True,'reviewer':'/root','scope':'ASR18_source_bound_eighth_original_full_clip_geometry_overlay_v1',
        'overlay_source':pin(NEW),'certificate':pin(CERT)},
    'source_gate_is_NOT_production_admission':True,
    'physical_ROOT_review':pin(F/'ROOT_recorded170470_row6_full_head_member_physical_actual_CPU_review_v1.json'),
    'real_original_ingress_frontend_mask_finite_full99_and_strict_negatives_still_required':True,
    'old_seven_correction_policy_trees_and_helpers_unchanged':True,
    'original_larger820896_native273632_clean_fullraw_media_workspace_numerical_reservations_retained':True,
    'no_genericEOF_skip_pad_crop_filter_label_population_or_science_change':True,
    'CPU12_transport_compatibility_review_needed_before_production':'CPU12 checked_packet_credit_policy still demands review CPU_source==pin(__file__). Final policy needs separate genuine ROOT successor bound to unchanged helper2/MAX2/humanapproval and closed323c transport proof via exact one-import CPU11 inverse; do not rerun old transport or use overlay gate as transport review.',
    'inactive_candidate_retains_old_transport_review_honestly_NOT_launchable':True,
    'no_control_audio_frontend_model_GPU_or_training_job_executed_by_this_preparation':True})
print(json.dumps({'overlay':pin(NEW),'certificate':pin(CERT),'CPU12':pin(CPU_NEW),'candidate':pin(cfg),
    'handoff':pin(handoff),'overlay_inverse':pin(ledger),'CPU_inverse':pin(cpu_ledger),'config_delta':delta}))
