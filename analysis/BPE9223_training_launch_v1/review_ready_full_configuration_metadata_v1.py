"""Final immutable configuration read; no fixtures, model, audio or payload IO."""
import importlib,json,sys
from pathlib import Path
L=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1');sys.path.insert(0,str(L));m=importlib.import_module('audit_and_fit_append_only_v5')
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1');F=N/'full_configuration_01';O=N/'independent_ready_full_configuration_review_v1';O.mkdir(exist_ok=True);m.AREA=O
S=m.PROJECT/'outputs/parakeet_turkish'
def rec(path):return {k:v for k,v in m.record(path).items() if k in ('path','bytes','sha256')}
def r3(pin):return {k:pin[k] for k in ('path','bytes','sha256')}
def read(pin):assert rec(pin['path'])==r3(pin);return json.loads(Path(pin['path']).read_text(encoding='utf-8-sig'))
def main():
 g=m.Guard();g.phase='actual_final_configuration_three_field_delta_and_bindings_read'
 oldpath=F/'configuration_pending_execution_review.json';path=F/'configuration_ready_for_final_execution_review.json'
 assert m.sha(path)=='19f8abbab865753c37b31d09ed5c0f15f98ac4cd054191056254cd54dd84097c'
 old=json.loads(oldpath.read_text());c=json.loads(path.read_text())
 assert set(old)==set(c);delta=sorted(k for k in c if c[k]!=old[k]);assert delta==['CPU_review_join_required_before_any_execution','ROOT_CPU_execution_review','full_run_launch_enabled']
 assert old['ROOT_CPU_execution_review'] is None and old['CPU_review_join_required_before_any_execution'] is True and old['full_run_launch_enabled'] is False
 assert c['CPU_review_join_required_before_any_execution'] is False and c['full_run_launch_enabled'] is True and c['production_enabled'] is False
 cpu=read(c['ROOT_CPU_execution_review']);assert c['ROOT_CPU_execution_review']['sha256']=='dd819f4572680fb2f11f94e7ed734550ecd2f5984cb4044858e92407f2613e2f'
 assert cpu['accepted'] is True and cpu['reviewer']=='/root' and cpu['scope']=='BPE9223_recorded_noise_16room_bounded_CPU_window_v1' and cpu['configuration']==c['CPU_template_configuration']
 actual=read(cpu['metadata_actual0']);report=read(cpu['metadata_report'])
 assert actual['owner']=='/root' and actual['actual_native_exit_code']==actual['actual_exec_tool_exit_code']==0 and actual['actual_tool_chunk_id']=='b4cf93'
 assert actual['actual_audio_or_GPU_jobs']==actual['actual_authoritative_draws_consumed']==0 and actual['report']==cpu['metadata_report']
 assert report['passed'] is True and report['configuration']==rec(oldpath) and report['worker_source']==c['configuration_assembly_source']
 assert report['source_manifest']==c['source_manifest'] and report['CPU_resource_review']==c['actual_ROOT_CPU_resource_review'] and report['session']==c['session']
 assert report['plan']==c['prepared_metadata_plan'] and report['actual_metadata_bridge_call_passed'] is True and report['lightweight_exact_callback_constructor_preflight_passed'] is True
 preflight_path=F/'actual_CPU_template_derivation_preflight.json';preflight=json.loads(preflight_path.read_text())
 assert preflight['passed'] is True and preflight['template']==c['CPU_template_configuration'] and preflight['ROOT_CPU_review']==c['ROOT_CPU_execution_review'] and preflight['audio_GPU_model_executed'] is False and preflight['new_optimizer_or_sampler_draws']==0 and preflight['run_namespace_created'] is False
 template=read(c['CPU_template_configuration']);contract=read(cpu['numerical_CPU_contract']);session=read(c['session'])
 assert template['owner_validation_only'] is False and template['checkpoint_diagnostic_only'] is False and template['vocabulary']==9223
 assert session['owner_validation_only'] is False and session['checkpoint_diagnostic_only'] is False and session['diagnostic_only'] is False and session['production_source_eligible'] is True
 assert session['planned_native_updates']==401072 and session['initial_optimizer_updates']==0 and session['CPU_preparation_contract']==cpu['numerical_CPU_contract']
 assert contract['vocabulary']==9223 and contract['owner_validation_only'] is False and contract['checkpoint_diagnostic_only'] is False and contract['planned_native_updates']==401072
 identity=c['full_run_identity'];data=c['planning_arguments']['DATA_evidence'];inputs=json.loads((N/'full_configuration_inputs_02/disabled_full_configuration_template.json').read_text())
 assert identity==inputs['full_run_identity'] and c['planning_arguments']==inputs['planning_arguments']
 assert c['owner_validation_only'] is False and c['diagnostic_only'] is False and identity['production_source_eligible'] is True and identity['owner_validation_only'] is False
 assert c['planned_native_updates']==401072 and identity['required_broad_passes']==2 and identity['required_cleaner_passes']==1 and identity['source_optimizer_RNG_sampler_reset'] is False and identity['source_original_clocks']==[64250,2000,62250]
 assert c['vocabulary']==identity['vocabulary']==9223 and identity['blank_id']==8192 and identity['joint_width']==9228 and data['duration_rows_half_open']==[9223,9228]
 assert data['broad']['mean_U']==90746310/2631115 and data['cleaner']['mean_U']==10621773/352768
 assert c['row_migration_plan']==identity['row_migration_plan'] and c['augmentation_identity']==inputs['augmentation_identity']==cpu['augmentation_identity']
 plan=read(c['prepared_metadata_plan']);assert c['prepared_metadata_plan']['sha256']=='143e0807673daa69796409acc45d6cd03e78ddf73ecc3f672d13b50df4e90591'
 assert plan['initial_committed_updates']==0 and len(plan['updates'])==20 and all(len(u['TR_draws'])==14 and len(u['rows'])==16 and all(r['language']=='tr' for r in u['rows'][:14]) and u['rows'][-1]['language']=='pt' for u in plan['updates'])
 assert cpu['plan']==c['prepared_metadata_plan'] and cpu['bridge']==c['actual_metadata_bridge'] and cpu['outer_parent_source']==c['parent_driver_source'] and cpu['sources']==template['sources']
 assert Path(c['checkpoint_storage_namespace'])==Path(c['run_namespace'])/'checkpoints' and template['checkpoint_storage_namespace']==c['checkpoint_storage_namespace'] and Path(template['output_namespace'])==Path(c['checkpoint_storage_namespace'])/'transient'
 assert c['entire_worker_forecast']==inputs['entire_worker_forecast'] and c['final_recovery_worker_forecast']==inputs['final_recovery_worker_forecast'] and c['CUDA_capacity_policy']==inputs['CUDA_capacity_policy']
 forecast=c['entire_worker_forecast'];assert forecast['native_private_forecast_bytes']==7933712768 and forecast['checkpoint_write_increment_bytes']==64000000 and forecast['entire_worker_private_forecast_bytes']==7997712768
 assert forecast['post_native_private_drop_assumed'] is False and forecast['native_margin_or_high_water_reduced'] is False and forecast['checkpoint_restore_during_resident_training_supported'] is False
 assert c['CUDA_capacity_policy']=={'allocator_cap_bytes':12000000000,'minimum_physical_bytes':17000000000,'driver_desktop_reserve_bytes':3000000000,'hard_free_physical_reserve_bytes':2000000000}
 assert c['CPU_owned_allowance_bytes']==1934217728 and cpu['resource_policy']['complete_CPU_phase_bytes']==1998217728
 assert c['exact_ROOT_execution_configuration_review_still_required'] is True and not Path(c['run_namespace']).exists()
 reviewed=N/'independent_full_inputs2_review_v1/source_findings_v1.json';assert m.sha(reviewed)=='10126b0ff85ce3969ed46d785f5fff18b4d6c97050a0c0c4c2192cc31068ed90'
 grouped=(S/'asr18_BPE9223_grouped_native_v1.py').read_text();assert "1<=v['count']<=8" in grouped and 'Every original row exactly once in original order' in grouped
 g.check()
 findings={'kind':'ASR18_independent_actual_ready_full_configuration_metadata_read_v1','passed':True,'reviewer':'/root/label_bpe_audit','configuration':rec(path),'actual_assembler_output':rec(oldpath),'exact_top_level_changed_keys':delta,
  'genuine_ROOT_CPU_review':c['ROOT_CPU_execution_review'],'metadata_assembler_actual0':cpu['metadata_actual0'],'metadata_report':cpu['metadata_report'],'actual_static_CPU_derivative_preflight':rec(preflight_path),'prior_source_template_policy_review_reused':rec(reviewed),
  'all_source_science_policy_identity_maps_unchanged_from_reviewed_actual_assembler':True,'original_source0_no_diagnostic_seed_2broad_1clean_401072':True,'new_denominators_9223_blank8192_joint9228_duration_rows_verified':True,
  'effective_batch':{'TR':14,'protected_replay':2,'effective_rows':16,'maximum_rows_per_Transformer_group':8,'always_exactly_two_groups':False,'geometry_may_subdivide_including_hard_singleton':True},
  'resource_domains':{'native_entire_worker_private_forecast_bytes':7997712768,'parent_base_current_bytes':128000000,'ALL_GPU_shim_current_bytes':64000000,'base_phase_family_forecast_bytes':8189712768,'separate_parent_OS_lifetime_private_cap_bytes':384000000,'CPU_entire_family_bytes':1934217728,'CPU_with_parent_current64MB_bytes':1998217728,'CUDA_allocator_cap_bytes':12000000000,'physical_driver_desktop_reserve_bytes':3000000000,'physical_free_reserve_bytes':2000000000,'restore_only_worker_bytes':3350000000},
  'production_launch_boolean_tracks_human_authorization_but_is_not_final_ROOT_execution_receipt':True,'new_concrete_metadata_blockers':[],
  'remaining_execution_gates':['Existing exclusive GPU owner real fullhard current-lifecycle update/checkpoint/fresh recovery/cleanup0','Separate exact final ROOT source/config/resource/actual evidence execution review before launch','Finite350 functional completion is not retention qualification; failed retention does not promote a model or shorten declared passes'],
  'actual_audio_GPU_model_payload_jobs':0,'run_namespace_not_created':True,'production_admitted_by_this_review':False}
 (O/'source_findings_v1.json').write_text(json.dumps(findings,indent=2)+'\n');g.publish(completed=True,findings=rec(O/'source_findings_v1.json'));print(json.dumps({'passed':True,'findings':rec(O/'source_findings_v1.json')}))
if __name__=='__main__':main()
