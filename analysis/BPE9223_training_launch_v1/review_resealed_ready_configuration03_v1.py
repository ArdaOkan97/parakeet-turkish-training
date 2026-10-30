"""Current resealed ready03 metadata joins; no native/audio/payload or fixtures."""
import importlib,inspect,json,sys
from pathlib import Path
L=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1');sys.path.insert(0,str(L));m=importlib.import_module('audit_and_fit_append_only_v5')
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1');I=N/'full_configuration_inputs_03';F=N/'full_configuration_03';O=N/'independent_resealed_ready03_review_v1';O.mkdir(exist_ok=True);m.AREA=O
S=m.PROJECT/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
def rec(path):return {k:v for k,v in m.record(path).items() if k in ('path','bytes','sha256')}
def r3(pin):return {k:pin[k] for k in ('path','bytes','sha256')}
def read(pin):assert rec(pin['path'])==r3(pin);return json.loads(Path(pin['path']).read_text(encoding='utf-8-sig'))
def main():
 g=m.Guard();g.phase='exact_resealed_inputs03_ready_configuration_metadata_read'
 handoff=I/'ROOT_reviewed_inputs_handoff.json';assert m.sha(handoff)=='608be73de602a429f05622ca16ec3b8002921cffa9d4d65f716e914121d7d2ca'
 h=json.loads(handoff.read_text());assert h['accepted'] is True and h['reviewer']=='/root'
 t=read(h['template']);manifest=read(h['source_manifest']);assert h['template']['sha256']=='1391e6d4b1f31ca55a153a550ccf4e75ead6e69caf97cc670f8447cdc19bce61' and h['source_manifest']['sha256']=='463c220bb3279c3bffdfe7fdf6e019f0d5e6cbf973c2dfed6b8a99012cab6e84'
 oldmanifest=json.loads((N/'full_configuration_inputs_02/full_source_manifest.json').read_text());oldpins={x['path']:x for x in oldmanifest['bindings']};pins={x['path']:x for x in manifest['bindings']}
 assert len(oldpins)==424 and len(pins)==432 and all(pins[k]==v for k,v in oldpins.items())
 replacements=json.loads((I/'review_input_handoff.json').read_text())['source_replacements']
 assert len(replacements)==6
 for pin in replacements.values():assert rec(pin['path'])==pin and pins[pin['path']]==pin
 prior=N/'independent_DEV_lifecycle_successors_review_v1/source_findings_v1.json';assert m.sha(prior)=='fdbe80f9b916fe32b9fabe743ee45525e1861b6cae00f2963215020185b77690'
 ready=F/'configuration_ready_for_final_execution_review.json';pending=F/'configuration_pending_execution_review.json'
 assert m.sha(ready)=='a20e024606cc480d4d29188c34468620b7487741abcffc7c57dbb7f59c5256af' and m.sha(pending)=='067524cae699bd8338d43fe9233d8efa2c16cea025b76443097dea67f0d7e90f'
 c=json.loads(ready.read_text());p=json.loads(pending.read_text());assert set(c)==set(p)
 delta=sorted(k for k in c if c[k]!=p[k]);assert delta==['CPU_review_join_required_before_any_execution','ROOT_CPU_execution_review','full_run_launch_enabled']
 assert p['full_run_launch_enabled'] is False and c['full_run_launch_enabled'] is True and p['ROOT_CPU_execution_review'] is None and p['CPU_review_join_required_before_any_execution'] is True and c['CPU_review_join_required_before_any_execution'] is False
 cpu=read(c['ROOT_CPU_execution_review']);assert c['ROOT_CPU_execution_review']['sha256']=='74a703bbe459843b2ebc7983f96d7321cdf1c0b6f7385226659fc8a9f3650b0f'
 assert cpu['accepted'] is True and cpu['reviewer']=='/root' and cpu['configuration']==c['CPU_template_configuration']
 actual=read(cpu['metadata_actual0']);report=read(cpu['metadata_report']);assert cpu['metadata_actual0']['sha256']=='e4b28ad9f4017a309c9ea82aec13fd6545a41a9fc6215b4c1de7260a8910a20d'
 assert actual['actual_native_exit_code']==actual['actual_exec_tool_exit_code']==0 and actual['actual_tool_chunk_id']=='ea337f' and actual['actual_identity_dead'] is True
 assert actual['report']==cpu['metadata_report'] and report['configuration']==rec(pending) and report['worker_source']==c['configuration_assembly_source']
 assert report['exact_serial_DEV_policy_and_source_bound_callback_constructor_preflight_passed'] is True and report['DEV_callbacks_invoked_or_audio_model_loaded'] is False
 source_review=read(c['configuration_source_read_review']);assert c['configuration_source_read_review']['sha256']=='347d27222b0a5ac21abf1fb3bd92cd28aaa335ac8621914c246e341e9ba9cf4f'
 assert source_review['accepted'] is True and source_review['production_approval_claimed'] is False and source_review['source_manifest']==c['source_manifest']==h['source_manifest'] and source_review['configuration_template']==h['template']
 identity=c['full_run_identity'];from bpe_broad_production_state_v1 import digest
 assert digest(identity)==h['run_identity_sha256']=='786003dd1aa3921235551c144e46ceae93ef60ca5b7af788d9da76cea9214621'
 base=json.loads((N/'full_configuration_inputs_02/disabled_full_configuration_template.json').read_text())
 assert c['planning_arguments']==base['planning_arguments'] and c['entire_worker_forecast']==base['entire_worker_forecast'] and c['CUDA_capacity_policy']==base['CUDA_capacity_policy']
 assert identity['source_original_clocks']==[64250,2000,62250] and identity['source_optimizer_RNG_sampler_reset'] is False and identity['original_scientific_sources']==base['full_run_identity']['original_scientific_sources']
 assert c['planned_native_updates']==401072 and identity['required_broad_passes']==2 and identity['required_cleaner_passes']==1 and c['owner_validation_only'] is False and identity['production_source_eligible'] is True
 assert c['vocabulary']==identity['vocabulary']==9223 and identity['blank_id']==8192 and identity['joint_width']==9228
 data=c['planning_arguments']['DATA_evidence'];assert data['duration_rows_half_open']==[9223,9228] and data['broad']['mean_U']==90746310/2631115 and data['cleaner']['mean_U']==10621773/352768
 assert c['row_migration_plan']==base['row_migration_plan'] and c['augmentation_identity']==base['augmentation_identity']
 assert c['parent_driver_source']==replacements['run_ASR18_BPE9223_full_run_owner_v1.py'] and c['worker_driver_source']==replacements['asr18_BPE9223_owned_GPU_runtime_v2.py'] and c['full_owner_controller_source']==replacements['asr18_BPE9223_full_run_controller_v1.py']
 policies={k:read(c[k]) for k in h['authorities']}
 for key,value in policies.items():assert value['accepted'] is True and value['reviewer']=='/root' and value['run_identity_sha256']==digest(identity) and value['full_run_source_contract']==h['source_manifest']
 import asr18_full_run_serial_DEV_9223_v3 as serial
 import asr18_BPE9223_lifecycle_v3 as lifecycle
 from asr18_full_run_DEV_runtime_9223_v3 import resource_proposal
 from asr18_BPE9223_full_run_controller_v2 import FullOwner
 from asr18_BPE9223_lifecycle_v3 import CALLBACKS
 from asr18_BPE9223_persistence_prefix_v4 import OwnerPrefixVerifier
 functions={name:getattr(FullOwner,name) for name in CALLBACKS};callback_sources={name:rec(inspect.getsourcefile(fn)) for name,fn in functions.items()}
 methods={name:getattr(FullOwner,name) for name in ('verify_pending_window','verify_writer_authority','verify_durable_resume')};methods['verify_current_recovery_and_persistence_prefix']=OwnerPrefixVerifier.verify_current_recovery_and_persistence_prefix
 proof_sources={name:rec(inspect.getsourcefile(fn)) for name,fn in methods.items()}
 durable=policies['durable_policy'];assert durable['lifecycle_callback_sources']==callback_sources and durable['proof_callback_sources']==proof_sources
 policy=policies['ROOT_DEV_policy_review'];static=policies['ROOT_DEV_static_execution_review'];resource=policies['ROOT_DEV_resource_review']
 assert policy['implementation']==static['implementation']==rec(serial.__file__) and policy['dependencies']==serial.sources()
 assert policy['scope']=='exact_selective_optimizer_free_serial_DEV_runtime_sources_and_policy' and policy['execution_sources']==static['execution_sources']==resource['execution_sources']==c['DEV_execution_sources']
 assert static['ROOT_policy_review']==c['ROOT_DEV_policy_review'] and static['resource_review']==c['ROOT_DEV_resource_review'] and resource['resource_policy']==resource_proposal()
 assert static['source_bindings']==c['DEV_source_bindings'] and policy['native_boundary_sources']==c['DEV_native_boundary_sources'] and policy['durable_prefix_verifier_source']==identity['persistent_prefix_sources']['prefix_verifier']
 assert c['DEV_execution_sources']['worker_source']==replacements['asr18_full_run_DEV_runtime_9223_v2.py'] and c['DEV_execution_sources']['parent_source']==c['parent_driver_source']
 cpu_template=read(c['CPU_template_configuration']);numerical=read(report['CPU_contract']);session=read(c['session']);assert cpu['outer_parent_source']==cpu_template['outer_parent_source']==numerical['outer_parent_source']==c['parent_driver_source']
 assert cpu_template['session']==c['session'] and cpu_template['full_run_identity_sha256']==session['full_run_identity_sha256']==digest(identity) and session['production_source_eligible'] is True
 assert session['owner_validation_only'] is False and session['checkpoint_diagnostic_only'] is False and numerical['owner_validation_only'] is False and numerical['checkpoint_diagnostic_only'] is False
 assert cpu['sources']==cpu_template['sources'] and cpu['augmentation_identity']==c['augmentation_identity'] and c['CPU_owned_allowance_bytes']==1934217728
 assert Path(cpu_template['output_namespace'])==Path(c['checkpoint_storage_namespace'])/'transient' and Path(c['checkpoint_storage_namespace'])==Path(c['run_namespace'])/'checkpoints'
 plan=read(c['prepared_metadata_plan']);previous=json.loads((N/'full_configuration_01/prepared_original_first20_plan.json').read_text());assert plan==previous
 frontier=serial.progress(plan['final_private_planning_sampler']);assert serial.progress is lifecycle.progress and frontier['optimizer_updates']==20 and frontier['broad_draws']==280 and frontier['cleaner_draws']==0
 assert c['exact_ROOT_execution_configuration_review_still_required'] is True and c['production_enabled'] is False and not Path(c['run_namespace']).exists()
 assert not any(k in sys.modules for k in ('torch','numpy','transformers','nemo'));g.check()
 result={'kind':'ASR18_independent_resealed_inputs03_actual_ready_configuration_review_v1','passed':True,'reviewer':'/root/label_bpe_audit','configuration':rec(ready),'assembler_pending_config':rec(pending),'exact_ready_delta_keys':delta,
  'ROOT_accepted_inputs_handoff':rec(handoff),'six_source_delta_review_reused':rec(prior),'original424bindings_retained_plus8_bound_successors':True,
  'metadata_assembler_actual0':cpu['metadata_actual0'],'new_DEV3_policy_and_callback_constructor_actual_pass':True,'earlier_source_review_missing_field_failure_preserved':actual['failed_review_schema_outcome_preserved'],
  'source_review3_exact_schema_complete':True,'all_current_serialDEV3_runtime3_controller2_parent2_native3_GPUruntime3_policy_source_maps_exact':True,
  'lifecycle3_new_mean_plan_progress_actual':frontier,'new_plan_raw_JSON_order_changed_but_all_parsed_values_equal_prior143e':True,'new_plan':c['prepared_metadata_plan'],
  'source0_no_diagnostic_seed_full401072_2broad1clean_newmeans_labels_geometry_preserved':True,'CPU3_static_template_session_outer_parent_and_resource_joins_exact':True,
  'unchanged_private_forecast_and12GB_CUDA_distinct_domains':True,'effective16_maximum8_rows_per_geometry_bounded_group_preserved':True,
  'new_concrete_metadata_blockers':[],'ROOT_final_execution_review_and_existing_GPU_owner_launch_still_separate':True,'production_admitted_here':False,
  'actual_audio_model_GPU_or_payload_jobs':0,'run_namespace_not_created':True}
 (O/'source_findings_v1.json').write_text(json.dumps(result,indent=2)+'\n');g.publish(completed=True,findings=rec(O/'source_findings_v1.json'));print(json.dumps({'passed':True,'findings':rec(O/'source_findings_v1.json')}))
if __name__=='__main__':main()
