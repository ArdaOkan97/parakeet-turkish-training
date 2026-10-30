"""Seal source/configuration metadata only; no numeric imports or native work."""
import ast,copy,hashlib,json,os
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research'
E=Path(r'E:\gpu_training_scratchpad');A=E/'analysis/ASR18_memory_diagnosis_v1';F=E/'analysis/FastMSS_integration_v1';B=E/'analysis/BPE_weak_then_clean_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'bytes':p.stat().st_size,'sha256':h}
def write(p,d):
 assert not p.exists(),str(p)
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert read(E/'storage_config.json')['allow_fallback_outside_root'] is False
assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'
for k in ('TEMP','TMP','HF_HOME','TORCH_EXTENSIONS_DIR'):assert Path(os.environ[k]).resolve().is_relative_to(E)
prior=A/'targeted_cleaner23_full_caller_source_review_manifest_v2.json';old=read(prior)
cpu_path=F/'FastMSS_source_typed_CPU8_source_review_manifest_v1.json';cpu=read(cpu_path)
parent=R/'run_ASR18_hard_effective16_v2.py';builder=R/'prepare_ASR18_hard_effective16_configuration_v2.py'
direct=[parent,builder,S/'asr18_hard_envelope_schedule_context_v1.py',S/'asr18_full_population_geometry_forecast_v1.py',S/'asr18_full_population_separated_forecast_v1.py',S/'asr18_hard_effective16_native_v1.py',S/'asr18_hard_effective16_recovery_v1.py',S/'asr18_hard_effective16_restore_runtime_v1.py']
for p in direct:ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
t=copy.deepcopy(read(S/'asr18_targeted_cleaner23.disabled.v2.json'))
for key in list(t):
 if 'cleaner' in key:t.pop(key)
t.update(kind='ASR18_hard_effective16_config_TEMPLATE_NOT_LAUNCHABLE_v1',parent_driver_source=pin(parent),worker_driver_source=str(parent.resolve()),
 CPU_controller_source=cpu['new_sources']['controller'],maximum_windows=1,maximum_total_native_updates=1,maximum_updates_per_window=1,execute_window_limit=1,
 corrected_CPU_contract_kind='FastMSS_source_typed_row_serial_CPU_preparation_contract_v1',
 hard_row_provenance={'actual_TOOL0':pin(A/'fullpopulation_firstwindow20_conv_actual_01/actual_owned_CLOSED0_v1.json'),
 'report':pin(A/'fullpopulation_firstwindow20_conv_actual_01/report.json'),'first_plan':pin(A/'fullpopulation_firstwindow20_conv_actual_01/coordinator_plan_0.json')},
 initial_sampler_sha256=None,provisional_plan=None,hard_envelope_diagnostic_context=None,actual_metadata_bridged_plan=None,
 launch_requires=['Exact hard effective16 source/resource/configuration ROOT reviews','Actual CPU8 metadata bridge before audio','Three fresh full family/system/storage/idle-lease admissions','Actual CPU0 then one full16 native update then native0 and separate exact checkpoint recovery0'])
t['forecast_calibration'].update(final_geometry=pin(B/'ASR18_final_declared_training_geometry_v1.json'),
 final_geometry_actual0=pin(B/'ASR18_final_declared_training_geometry_owned_tool_actual_CLOSED0_v1.json'),
 previous_full_hard_actual0=pin(A/'full_hard_2D_native_01/actual_owned_CLOSED0.json'),
 previous_full_hard_report=pin(A/'full_hard_2D_native_01/full_hard_native_validation.json'),
 conv_only_hard_summary=pin(A/'single_full_hard_TR207_actual_01/actual_full_hard_TR207_summary_v1.json'))
t['entire_worker_forecast'].update(native_private_forecast_bytes=6684646272,entire_worker_private_forecast_bytes=6748646272)
tp=S/'asr18_hard_effective16.disabled.v1.json';write(tp,t)
proposal={'kind':'ASR18_hard_effective16_source_resource_candidate_v1','production_enabled':False,'diagnostic_only':True,
 'scope':'ONE current CPU8 original raw/header/frontend full16 precomputed update, hard row lastTR; checkpoint and fresh recovery',
 'host_private_bytes':{'historical_full_hard_native_actual':5726646272,'unchanged_variation':750000000,'extra_replay':128000000,'extra_fixed_teacher':80000000,
 'native':6684646272,'separate_persistence':64000000,'whole_worker':6748646272,'separate_parent':128000000,'ALL_separate_GPU_shims':64000000,'whole_GPU_family':6940646272},
 'CPU_family_private_bytes':1934217728,'CPU_family_separate_parent_bytes':128000000,
 'CUDA_bytes':{'unchanged_allocator_and_actual_ceiling':8000000000,'endcache_reserved':3000000000,'complete_v7_joint_union':791167592,
 'prior_hard_graph_candidate':3980617640,'additional_teacher_and_replay_reserve':208000000,'graph_plus_reserve_candidate':4188617640},
 'source_geometry':{'full_samples':2547840,'physical_feature_frames':15925,'valid_feature_frames':15924,'T':1991,'U':574,'TR_rows':14,'replay_rows':2},
 'CNN_or_CUDA_savings_subtracted_from_host_forecast_bytes':0,'actual_new_effective16_fit_claimed':False,
 'global_system90_critical95_headroom2GB_E100GB_unchanged':True,'actual_current_and_OS_lifetime_worker_parent_caps_required':True,
 'metadata_assembly_only_forecast_bytes':512000000,'metadata_wall_seconds':180}
rp=A/'hard_effective16_resource_proposal_v1.json';write(rp,proposal)
bindings={item['path']:item for item in old['bindings']}
extra=[*direct,tp,rp,cpu_path,Path(cpu['template']['path']),Path(cpu['interface']['path']),Path(__file__)]
extra.extend(Path(v['path']) for v in cpu['new_sources'].values())
extra.extend(Path(v['path']) for v in t['forecast_calibration'].values() if type(v) is dict and set(('path','bytes','sha256'))<=set(v))
extra.extend(Path(v['path']) for v in t['hard_row_provenance'].values())
for p in extra:bindings[str(p.resolve())]=pin(p)
for value in bindings.values():assert pin(value['path'])==value,'Frozen dependency changed: '+value['path']
manifest={'kind':'ASR18_hard_effective16_CPU8_full_caller_source_review_manifest_v1','owner':'/root/memory_diagnosis',
 'requested_source_scope':'source_read_review_for_hard_effective16_current_CPU_native_diagnostic',
 'later_execution_review_scope_required':'finite_hard_effective16_CPU8_native_fresh_recovery_diagnostic_v1',
 'production_enabled':False,'direct_sources':[pin(p) for p in direct],'bindings':sorted(bindings.values(),key=lambda x:x['path']),
 'configuration_template':pin(tp),'CPU_owner_bundle':pin(cpu_path),'previous_source_manifest':pin(prior),
 'previous_source_read_review':pin(A/'ROOT_targeted_cleaner23_source_read_review_v2.json'),'resource_candidate':pin(rp),
 'source_core_native_update':pin(S/'asr18_precomputed_native_v2.py'),'source_generation_only_no_behavioral_execution':True,
 'actual_new_audio_frontend_or_GPU_jobs':0,'diagnostic_only':True,'qualified_checkpoint':False,
 'source_delta':'CPU8 explicit typed hard mode; full retained geometry forecast; original-source diagnostic14 cursor/full2replay; hard lastTR; exact parent128MB and all GPU shims64MB; one native update and separate fresh all-state recovery',
 'metadata_bridge_call_required_before_audio':True,'ROOT_exact_source_resource_config_review_still_required':True}
mp=A/'hard_effective16_full_caller_source_review_manifest_v1.json';write(mp,manifest)
print(json.dumps({'source_manifest':pin(mp),'template':pin(tp),'resource_proposal':pin(rp),'bindings':len(bindings),'parent':pin(parent),'builder':pin(builder)}))
