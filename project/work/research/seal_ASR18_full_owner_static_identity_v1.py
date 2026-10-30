"""Seal only source/configuration review inputs. No target imports or data scans."""
from pathlib import Path
import ast,hashlib,json
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research'
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1');F=A.parent/'FastMSS_integration_v1'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def write(path,value):
    assert not path.exists();path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    return pin(path)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
baseline=read(A/'hard_effective16_full_caller_source_review_manifest_v3.json')
hard=read(A/'hard_effective16_configuration_CPU_actual_02/configuration_pending_execution_review.json')
dev=read(A/'diagnostic_optimizer_free_serial350DEV_configuration_02/configuration_pending_execution_review.json')
bindings={item['path']:item for item in baseline['bindings']}
for item in dev['source_bindings'].values():bindings[item['path']]=item
names=[S/'asr18_full_run_parent_controller_v6.py',R/'run_ASR18_full_run_owner_v8.py',R/'run_ASR18_full_run_owner_v6.py',
 R/'asr18_full_run_owned_GPU_runtime_v5.py',S/'asr18_full_run_native_owner_v4.py',S/'asr18_full_run_source_context_v1.py',
 S/'bpe_full_run_training_state_v1.py',S/'asr18_full_run_current_recovery_v2.py',S/'asr18_full_run_metadata_resolver_v1.py',
 S/'asr18_full_run_persistence_prefix_v1.py',S/'bpe_full_run_lifecycle_v2.py',S/'bpe_full_run_durable_owner_adapter_v2.py',
 S/'bpe_full_run_durable_owner_adapter_v1.py',S/'bpe_indexed_owned_checkpoint_storage_v3.py',S/'bpe_owned_checkpoint_storage_v2.py',
 S/'bpe_bounded_production_journal_v4.py',S/'asr18_full_run_serial_DEV_v9.py',S/'asr18_full_run_DEV_runtime_v4.py',
 S/'asr18_full_run_serial_DEV_v8.py',R/'prepare_ASR18_full_owner_configuration_v1.py']
# Statically resolve project-local source imports without importing a target.
seen=set();pending=list(names)
while pending:
    path=pending.pop().resolve()
    if path in seen:continue
    seen.add(path);bindings[str(path)]=pin(path)
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    for node in ast.walk(tree):
        modules=([node.module] if isinstance(node,ast.ImportFrom) and node.module else
                 [item.name for item in node.names] if isinstance(node,ast.Import) else [])
        for module in modules:
            if '.' in module:continue
            for root in (S,R):
                candidate=root/(module+'.py')
                if candidate.is_file():pending.append(candidate);break
source={'kind':'ASR18_full_run_original207_two_broad_one_cleaner_source_contract_v1',
 'owner':'/root/memory_diagnosis','production_enabled':False,'execution_admitted':False,
 'CPU_owner_bundle':pin(F/'FastMSS_source_typed_CPU8_source_review_manifest_v1.json'),
 'bindings':sorted(bindings.values(),key=lambda item:item['path']),
 'direct_sources':{path.name:pin(path) for path in names},
 'preserved_numeric_source_scope':'original207/threeAdam/all4RNG/fullPCM/targets, canonicalFP32 joint, exact validated conv-only2',
 'normal_path':'pass-bounded original schedule; CPU8; resident complete-window persistence; indexed3 AUX; current-only fresh recovery; streamed durable prefix; real-pass publicDEV9; strict winner',
 'unsupported_partial_tail':'fail-stop and preserve unknown suffix; no automatic partial-Adam replay/adoption',
 'control_scope':'fixed1GiB +48 reserved64MiB GPU-worker directories; no unbounded restart claim',
 'finite_validation':'same body, original first8 in two complete windows4; no pass/DEV fabrication or production ancestry'}
source_pin=write(A/'full_owner_source_contract_v1.json',source)
scorer=pin(F/'Full_run_serial350DEV_scorer_policy_DRAFT_NOT_ACCEPTED_v1.json')
identity={'kind':'ASR18_reviewed_full_two_broad_one_cleaner_run_identity_v1','native_owner':'/root/memory_diagnosis',
 'required_broad_passes':2,'required_cleaner_passes':1,'source_original_clocks':[64250,2000,62250],
 'source_optimizer_RNG_sampler_reset':False,'full_run_source_contract':source_pin,'scorer_policy':scorer,
 'checkpoint_storage_policy':pin(A/'ROOT_indexed3_numeric_checkpoint_policy_v1.json'),
 'maximum_pass_window_publications':46985,'owner_validation_only':True,'production_source_eligible':False,
 'permanent_namespace_scope':'original_source_owner_validation_first8_two_windows_no_production_ancestry',
 'original_scientific_sources':hard['planning_arguments']['expected_original_sources'],
 'persistent_prefix_sources':{
  'prefix_verifier':pin(S/'asr18_full_run_persistence_prefix_v1.py'),
  'metadata_resolver':pin(S/'asr18_full_run_parent_controller_v6.py'),
  'native_worker':pin(R/'asr18_full_run_owned_GPU_runtime_v5.py'),'native_parent':pin(R/'run_ASR18_full_run_owner_v8.py'),
  'recovery_worker':pin(R/'asr18_full_run_owned_GPU_runtime_v5.py'),'recovery_parent':pin(R/'run_ASR18_full_run_owner_v8.py'),
  'persistence_checker':pin(S/'asr18_streamed_persistence_fingerprint_v1.py'),
  'process_validator':pin(S/'fastmss_owned_process_closure_v2.py'),
  'durable_adapter':pin(S/'bpe_full_run_durable_owner_adapter_v2.py'),
  'indexed_storage':pin(S/'bpe_indexed_owned_checkpoint_storage_v3.py'),
  'streamed_semantics':pin(S/'bpe_streaming_checkpoint_semantics_v2.py')}}
identity_pin=write(A/'full_owner_validation_identity_v1.json',identity)
controller=pin(S/'asr18_full_run_parent_controller_v6.py');parent=pin(R/'run_ASR18_full_run_owner_v8.py')
prefix=pin(S/'asr18_full_run_persistence_prefix_v1.py')
callbacks=('admit_operation','prepare_window','execute_window','verify_checkpoint','serial_DEV','verify_DEV',
 'preserve_winner','verify_winner','verify_resume','checkpoint_storage_boundary')
root=Path(r'E:\gpu_training_scratchpad\runs\ASR18_full_owner_validation_08_01')
qualified_summary=pin(P/'work/closed-sources/TSC_WorldSpeech_expanded_65948_closed_v1/best_DEV_summary.json')
summary=read(qualified_summary['path']);references={'tr':825,'en':1045,'es':1289,'it':1158,'fr':1072,'de':1004,'pt':1095}
winner={'strict_qualified':True,'word_errors':{l:summary['validation_'+l]['asr']['word_errors'] for l in references},
 'reference_words':references,'artifact':hard['planning_arguments']['expected_original_sources']['parked_parent'],
 'DEV':qualified_summary}
protection={'kind':'ASR18_future_owned_checkpoint_protection_plan_v1',
 'required_roles':['source','history','DEV','qualified_winner'],
 'source':[hard['planning_arguments']['snapshot_binding']],
 'history':[pin(A/'hard_effective16_native_actual_01/actual_hard_effective16_end_to_end_summary_v1.json')],
 'DEV':[pin(A/'ROOT_fixed_serial350DEV_manifest_membership_review_v1.json')],
 'qualified_winner':[qualified_summary],'adopt_external_sources_as_owned':False}
protected_pin=write(A/'full_owner_external_protection_plan_v1.json',protection)
execution={'worker_source':pin(S/'asr18_full_run_DEV_runtime_v4.py'),'parent_source':parent,
 'guard_source':parent,'bootstrap_source':pin(S/'asr18_optimizer_free_DEV_structure_v2.py')}
native_boundary={**{k:identity['persistent_prefix_sources'][k] for k in ('native_worker','native_parent','recovery_worker','recovery_parent')},
 'durable_prefix_verifier':prefix}
dev_sources=dev['source_bindings'].copy()
for path in names:
    if path.parent==S:dev_sources[path.name]=pin(path)
request={'kind':'ASR18_full_owner_exact_run_bound_ROOT_review_inputs_v1','production_enabled':False,
 'source_manifest':source_pin,'identity':identity,'identity_binding':identity_pin,'run_identity_sha256':digest(identity),
 'run_namespace':str(root),'storage_namespace':str(root/'checkpoint_storage'),'durable_namespace':str(root/'durable'),
 'actual_folder':str(A/'full_owner_eight_update_actual_01'),'configuration_output':str(A/'full_owner_eight_update_configuration_01'),
 'owner_verifier_source':controller,'protection_plan':protected_pin,'qualified_source_winner':winner,
 'qualified_source_DEV_summary':qualified_summary,'parent_source':parent,'worker_source':pin(R/'asr18_full_run_owned_GPU_runtime_v5.py'),
 'lifecycle_callback_sources':{name:controller for name in callbacks},
 'proof_callback_sources':{'verify_pending_window':controller,'verify_writer_authority':controller,'verify_durable_resume':controller,
   'verify_current_recovery_and_persistence_prefix':prefix},
 'DEV_execution_sources':execution,'DEV_native_boundary_sources':native_boundary,
 'DEV_base_assets':dev['base_assets'],'DEV_source_bindings':dev_sources,
 'CPU_owner_bundle':source['CPU_owner_bundle'],'CPU_numeric_prior':pin(A/'ROOT_reviewed_row_serial_CPU_resource_contract_v9.json'),
 'base_hard_configuration':pin(A/'hard_effective16_configuration_CPU_actual_02/configuration_pending_execution_review.json'),
 'metadata_builder_source':pin(R/'prepare_ASR18_full_owner_configuration_v1.py'),
 'finite_scope':'finite_full_run_owner_eight_update_two_window_pipeline_validation_v1',
 'maximum_owner_validation_updates':8,'maximum_validation_updates_per_window':4,
 'numeric_policy_only':identity['checkpoint_storage_policy'],
 'no_actual_future_checkpoint_PID_or_owner_creation_evidence_claimed':True}
request_pin=write(A/'full_owner_exact_ROOT_review_inputs_v1.json',request)
print(json.dumps({'source_contract':source_pin,'identity':identity_pin,'ROOT_inputs':request_pin,'bindings':len(bindings)},ensure_ascii=False))
