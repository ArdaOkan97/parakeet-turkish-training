"""Independent current disabled full-config metadata/source joins; no assembly."""
import ast,copy,importlib,inspect,json,sys
from pathlib import Path
L=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1');sys.path.insert(0,str(L));m=importlib.import_module('audit_and_fit_append_only_v5')
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1');I=N/'full_configuration_inputs_02';O=N/'independent_full_inputs2_review_v1';O.mkdir(exist_ok=True);m.AREA=O
S=m.PROJECT/'outputs/parakeet_turkish';R=m.PROJECT/'work/research';sys.path.insert(0,str(S))
def rec(path):return {k:v for k,v in m.record(path).items() if k in ('path','bytes','sha256')}
def r3(pin):return {k:pin[k] for k in ('path','bytes','sha256')}
def read(pin):assert rec(pin['path'])==r3(pin);return json.loads(Path(pin['path']).read_text(encoding='utf-8-sig'))
def main():
 g=m.Guard();g.phase='current_disabled_full_inputs2_exact_metadata_source_joins'
 a1=R/'prepare_ASR18_BPE9223_full_configuration_v1.py';a2=R/'prepare_ASR18_BPE9223_full_configuration_v2.py'
 assert m.sha(a2)=='17e1464e904b719b8435388064915ac2d2d687512bfae54059b61763f2e2ac40'
 assert a1.read_text().replace("        'diagnostic_only':False,'owner_validation_only':False,'checkpoint_diagnostic_only':False,","        'diagnostic_only':False,'owner_validation_only':False,'checkpoint_diagnostic_only':False,\n        'production_source_eligible':True,")==a2.read_text()
 inp=R/'prepare_ASR18_BPE9223_full_inputs_v2.py';assert m.sha(inp)=='a01b0940151d0cb1c5df1e5e69cd14fc1e69257ed06ac72fbf851bc13b7128dc'
 tp=I/'disabled_full_configuration_template.json';mp=I/'full_source_manifest.json'
 assert m.sha(tp)=='998c2eb9dcf4368cddcd961414c1a4f36ebeb21cabcf1b2ba35dd2599bdf58a1'
 assert m.sha(mp)=='c5a0d1b40d9f7e310692f20d7950ba50ee1f1e05757e6f1440f06d51c0828aac'
 t=json.loads(tp.read_text());manifest=json.loads(mp.read_text());identity=t['full_run_identity']
 from bpe_broad_production_state_v1 import digest
 assert t['TEMPLATE_NOT_LAUNCHABLE'] is True and t['full_run_launch_enabled'] is False and t['production_enabled'] is False
 assert t['owner_validation_only'] is False and t['diagnostic_only'] is False and identity['owner_validation_only'] is False and identity['production_source_eligible'] is True
 assert t['planned_native_updates']==401072 and identity['required_broad_passes']==2 and identity['required_cleaner_passes']==1 and identity['source_optimizer_RNG_sampler_reset'] is False
 assert identity['source_original_clocks']==[64250,2000,62250] and identity['vocabulary']==t['vocabulary']==9223 and identity['blank_id']==8192 and identity['joint_width']==9228
 assert identity['full_run_source_contract']==t['source_manifest']==rec(mp)
 assert t['normal_boundary_resume_enabled'] is True and not any(k in t for k in ('maximum_owner_validation_updates','maximum_validation_updates_per_window','owner_validation_pause_after_updates'))
 arguments=t['planning_arguments'];data=arguments['DATA_evidence'];runtime=read(data['BPE9223_population'])
 assert identity['BPE9223_population']==data['BPE9223_population'] and data['complete'] is True and data['vocabulary']==runtime['vocabulary']==9223
 assert data['blank_id']==runtime['blank_id']==8192 and data['duration_rows_half_open']==runtime['duration_rows_half_open']==[9223,9228]
 assert data['broad']['rows']==2631115 and data['broad']['tokens']==90746310 and data['broad']['mean_U']==90746310/2631115
 assert data['cleaner']['rows']==352768 and data['cleaner']['tokens']==10621773 and data['cleaner']['mean_U']==10621773/352768 and data['cleaner']['augmentation_allowed'] is False
 assert r3(runtime['tokenizer'])==r3(data['tokenizer']) and data['tokenizer']['sha256']=='f2d80ffee7ed1b006a83b52cc77bfb28bed4ba850352b5222bb3a49d6cfbac4a'
 assert arguments['expected_original_sources']==identity['original_scientific_sources']
 assert arguments['expected_original_sources']['bpe_parent']['sha256']=='71ca1897d95622c5556df6a3b568d26136004e25c4bcca515f7a242df9e9b63a'
 assert arguments['expected_original_sources']['parked_parent']['sha256']=='b0f5d5e207caeda3c030a5f11eda4ae9004e4826a7fd050a8ac8de78333e3e38'
 assert arguments['replay_pool_binding']['sha256']=='00b839c717762aba74326e9867969d85b7ffaaabd7ead7ddbe732fe6be8c6cf4'
 assert manifest['row_migration_plan']==t['row_migration_plan']==identity['row_migration_plan']
 donor=read(t['row_migration_plan']);assert donor['source_lexical_rows']==8711 and donor['target_lexical_rows']==9223 and donor['source_head_rows']==8716 and donor['target_head_rows']==9228 and donor['blank_id_unchanged']==8192
 assert donor['old_duration_to_new_rows']==[{'old':8711+i,'new':9223+i} for i in range(5)]
 assert len(donor['donors'])==512 and [d['HF_id'] for d in donor['donors']]==list(range(8711,9223)) and all(all(type(v) is int and 0<=v<8711 for v in d['original_leaf_donor_ids']) for d in donor['donors'])
 assert donor['recursive_averaging_or_repeated_bias_penalty'] is False and donor['training_token_ID_permutation'] is False and donor['protected1914_replay_and_teacher_targets_unchanged'] is True
 # Current explicit dependency pins only, no historical payload/database hashes.
 active=[a2,inp,S/'asr18_BPE9223_full_run_controller_v1.py',S/'asr18_BPE9223_lifecycle_v3.py',S/'asr18_BPE9223_durable_transitions_v3.py',S/'asr18_BPE9223_durable_owner_adapter_v3.py',S/'asr18_BPE9223_persistence_prefix_v4.py',S/'asr18_BPE9223_owned_checkpoint_storage_v1.py',S/'asr18_BPE9223_indexed_checkpoint_storage_v1.py',S/'asr18_BPE9223_source_context_v2.py',S/'asr18_BPE9223_bootstrap_v2.py',S/'asr18_BPE9223_model_migration_v2.py',S/'asr18_BPE9223_optimizer_v2.py',S/'asr18_BPE9223_CPU_window_configuration_v2.py',S/'fastmss_BPE9223_feature_protocol_v3.py',S/'fastmss_BPE9223_plan_bridge_v3.py',S/'fastmss_BPE9223_cpu_common_v3.py']
 pins={p['path']:p for p in manifest['bindings']}
 for path in active:assert pins[str(path)]==rec(path)
 for key in ('prototype_numerical_CPU_contract','prototype_CPU_source_contract'):read(manifest[key])
 policies={key:read(t[key]) for key in ('durable_policy','ROOT_storage_review','ROOT_DEV_policy_review','ROOT_DEV_static_execution_review','ROOT_DEV_resource_review','actual_ROOT_CPU_resource_review')}
 for p in policies.values():assert p['accepted'] is True and p['reviewer']=='/root' and p['run_identity_sha256']==digest(identity) and p['full_run_source_contract']==rec(mp)
 cpu=policies['actual_ROOT_CPU_resource_review'];assert cpu['scope']=='CPU_preparation_only' and cpu['CPU_source_manifest']==manifest['CPU_owner_bundle'] and cpu['outer_parent_source']==t['parent_driver_source']
 resource=cpu['resource_review'];assert resource['whole_owned_private_allowance_bytes']+resource['outer_controller_launcher_reserve_bytes']==1934217728 and resource['maximum_simultaneous_children']==1
 from asr18_BPE9223_full_run_controller_v1 import FullOwner
 from asr18_BPE9223_lifecycle_v3 import SourceBoundOwnerCallbacks,CALLBACKS
 from asr18_BPE9223_durable_transitions_v3 import OwnerProofCallbacks
 from asr18_BPE9223_persistence_prefix_v4 import OwnerPrefixVerifier
 functions={name:getattr(FullOwner,name) for name in CALLBACKS}
 callbacks=SourceBoundOwnerCallbacks(sources={name:rec(inspect.getsourcefile(fn)) for name,fn in functions.items()},callbacks=functions)
 methods={name:getattr(FullOwner,name) for name in ('verify_pending_window','verify_writer_authority','verify_durable_resume')};methods['verify_current_recovery_and_persistence_prefix']=OwnerPrefixVerifier.verify_current_recovery_and_persistence_prefix
 proofs=OwnerProofCallbacks(sources={name:rec(inspect.getsourcefile(fn)) for name,fn in methods.items()},functions=methods)
 durable=policies['durable_policy'];assert durable['proof_callback_sources']==proofs.sources and durable['lifecycle_callback_sources']==callbacks.sources and durable['source']==rec(S/'asr18_BPE9223_durable_owner_adapter_v3.py') and durable['transition_source']==rec(S/'asr18_BPE9223_durable_transitions_v3.py')
 assert Path(durable['namespace'])==Path(t['run_namespace'])/'durable'
 storage=policies['ROOT_storage_review'];assert storage['future_namespace']==t['checkpoint_storage_namespace'] and storage['policy']==t['checkpoint_storage_policy'] and storage['component_source']==rec(S/'asr18_BPE9223_indexed_checkpoint_storage_v1.py') and storage['frozen_operation_source']==rec(S/'asr18_BPE9223_owned_checkpoint_storage_v1.py') and storage['owner_verifier_source']==t['full_owner_controller_source']
 import asr18_full_run_serial_DEV_9223_v2 as serial
 from asr18_full_run_DEV_runtime_9223_v2 import resource_proposal
 policy=policies['ROOT_DEV_policy_review'];static=policies['ROOT_DEV_static_execution_review'];devresource=policies['ROOT_DEV_resource_review'];scorer=read(identity['scorer_policy'])
 assert policy['scope']=='exact_selective_optimizer_free_serial_DEV_runtime_sources_and_policy' and policy['implementation']==rec(serial.__file__) and policy['dependencies']==serial.sources()
 assert policy['execution_sources']==static['execution_sources']==devresource['execution_sources']==t['DEV_execution_sources']
 assert policy['native_boundary_sources']==t['DEV_native_boundary_sources'] and policy['durable_prefix_verifier_source']==identity['persistent_prefix_sources']['prefix_verifier']
 assert static['ROOT_policy_review']==t['ROOT_DEV_policy_review'] and static['resource_review']==t['ROOT_DEV_resource_review'] and static['implementation']==rec(serial.__file__)
 assert static['source_bindings']==t['DEV_source_bindings'] and static['base_assets']==t['DEV_base_assets'] and static['scorer_policy']==policy['scorer_policy']==identity['scorer_policy']
 assert devresource['resource_policy']==resource_proposal() and policy['canonical_DEV_manifests']==scorer['DEV_manifests'] and scorer['reference_words']==serial.REFERENCES and scorer['retention_limits']==serial.LIMITS
 assert Path(t['checkpoint_storage_namespace'])==Path(t['run_namespace'])/'checkpoints' and Path(t['run_namespace']).is_relative_to(m.ROOT/'runs') and not Path(t['run_namespace']).exists()
 # Evaluate only emitted session scalar/API expression, never the assembler.
 tree=ast.parse(a2.read_text());assemble=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='assemble');assignment=next(n for n in assemble.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='session' for x in n.targets))
 dummy={'path':str(O/'IN_MEMORY_ONLY.json'),'bytes':1,'sha256':'1'*64};scope={'record':lambda p:copy.deepcopy(dummy),'S':S,'digest':digest,'identity_sha':digest(identity),'expected_owner':dummy,'template':t,'plan':{'initial_sampler_sha256':'b'*64},'cp':O/'IN_MEMORY_ONLY.contract','own':rec(a2)}
 session=eval(compile(ast.Expression(body=copy.deepcopy(assignment.value)),str(a2),'eval'),scope)
 import fastmss_BPE9223_plan_bridge_v3 as bridge
 assert bridge.validate_session_scope(session)==bridge.FULL_MODE and session['production_source_eligible'] is True
 assert not any(k in sys.modules for k in ('torch','numpy','transformers','nemo'));g.check()
 report={'kind':'ASR18_independent_current_disabled_full_inputs2_source_schema_join_review_v1','reviewer':'/root/label_bpe_audit','passed':True,'accepted_for_disabled_metadata_assembly':True,'accepted_for_production_launch':False,
  'assembler2':rec(a2),'input_generator2':rec(inp),'template2':rec(tp),'manifest2':rec(mp),'verified_active_current_source_pins':len(active),'historical_payload_audio_or_full_database_hashes_read':False,
  'eligible_fix_is_only_assembler_delta':True,'actual_emitted_session_expression_passes_current_bridge3_API':True,
  'source0_2broad_1clean_401072_and_new_full_denominators_verified':True,'tokenizer9223_blank8192_joint9228_durations9223_to9228_verified':True,
  'same_original_sources_and1914replay_pins':True,'flattened_original8711_leaf_donors_old3Adam_newclock0_policy_preserved':True,
  'actual_SourceBoundOwnerCallbacks_and_OwnerProofCallbacks_constructors_passed':True,'actual_policy_callback_source_maps_exact':True,
  'storage_namespace_current_indexed_base_prefix_policy_source_maps_exact':True,'DEV_policy_static_resource_current_maps_and_scorer_caps_exact':True,
  'prior_static_DEV_old_policy_pin_and_scope_findings_fixed':True,'run_namespace_not_created':True,'launch_disabled_and_source_preparation_only':True,
  'remaining':['ROOT guarded actual metadata assembler/first20 plan and generic CPU derivative proof','Exact final execution/source/config/resource authority and source-bound actual hard/350/native checks remain ROOT and memory ownership'],
  'actual_audio_model_payload_GPU_jobs':0,'new_concrete_metadata_source_blockers':[]}
 (O/'source_findings_v1.json').write_text(json.dumps(report,indent=2)+'\n');g.publish(completed=True,findings=rec(O/'source_findings_v1.json'));print(json.dumps({'passed':True,'findings':rec(O/'source_findings_v1.json')}))
if __name__=='__main__':main()
