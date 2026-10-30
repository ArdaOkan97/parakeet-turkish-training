"""Stdlib-only disabled real-run assembler/source/identity successor; no jobs."""
from pathlib import Path
import ast,copy,hashlib,json
P=Path(__file__).resolve().parents[2];R=P/'work/research'
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def write(path,value):
    assert not path.exists();path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n');return pin(path)
old_builder=R/'prepare_ASR18_full_owner_configuration_v10.py'
builder=R/'prepare_ASR18_full_owner_configuration_v11.py'
text=old_builder.read_text(encoding='utf-8-sig')
replacements=[
 ('Metadata-only full-body/stop8/two-window configuration and real CPU8 bridge.',
  'Metadata-only DISABLED full-two-broad/one-cleaner configuration and original first20 CPU8 bridge.'),
 ("source_read_review_for_full_run_owner_eight_update_two_window_pipeline_validation",
  "source_read_review_for_disabled_full_run_owner_production_metadata_preparation"),
 ("and template['owner_validation_only'] is True and template['maximum_owner_validation_updates']==8\n        and template['maximum_validation_updates_per_window']==4",
  "and template['owner_validation_only'] is False and template['planned_native_updates']==401072\n        and template['full_run_identity']['owner_validation_only'] is False\n        and template['full_run_identity']['production_source_eligible'] is True"),
 ("and template['normal_boundary_resume_enabled'] is True and template['owner_validation_pause_after_updates']==4,\n        'No metadata action enables a long run; exact resume-enabled genesis and bounded split4/4')",
  "and template['normal_boundary_resume_enabled'] is True\n        and not any(key in template for key in ('maximum_owner_validation_updates',\n            'maximum_validation_updates_per_window','owner_validation_pause_after_updates')),\n        'Disabled real-run preparation only; no finite diagnostic fields or long-run launch authority')"),
 ("plan=context.validation_plan(schedule,completed_boundaries=0,maximum_updates=4)",
  "plan=context.plan(schedule,completed_boundaries=0)"),
 ("and len(plan['updates'])==4 and frontier['optimizer_updates']==4 and frontier['broad_draws']==56",
  "and len(plan['updates'])==20 and frontier['optimizer_updates']==20 and frontier['broad_draws']==280"),
 ("Actual original first4 private preview; no consumed draw or fabricated pass/DEV",
  "Actual original first20 full-mode private preview; no consumed draw or fabricated pass/DEV"),
 ("prepared_original_first4_plan.json","prepared_original_first20_plan.json"),
 ("len(bridge['updates'])==4","len(bridge['updates'])==20"),
 ("ASR18_actual_full_owner_eight_update_two_window_metadata_configuration_v1",
  "ASR18_actual_disabled_full_run_owner_production_metadata_configuration_v1"),
 ("'future_durable_owner_binding_is_NOT_actual_creation_evidence':True,'production_enabled':False}",
  "'future_durable_owner_binding_is_NOT_actual_creation_evidence':True,\n        'full_run_launch_enabled':False,'production_execution_admitted':False,\n        'legacy_fresh_invocation_token':'genesis_normal_pause4',\n        'legacy_token_does_not_enable_validation_pause_in_full_mode':True,'production_enabled':False}"),
]
for before,after in replacements:
    assert text.count(before)==1,(before,text.count(before));text=text.replace(before,after)
ast.parse(text);assert not builder.exists();builder.write_text(text,encoding='utf-8',newline='\n')
old=read(A/'full_owner_exact_ROOT_review_inputs_v12.json')
source=copy.deepcopy(read(A/'full_owner_source_contract_v11.json'))
bindings={item['path']:item for item in source['bindings']};bindings[str(builder)]=pin(builder)
source['bindings']=sorted(bindings.values(),key=lambda item:item['path'])
source['direct_sources'][builder.name]=pin(builder)
source.pop('finite_validation',None)
source.update(metadata_preparation_scope='Disabled full-mode original first20 preview/session/CPU8 bridge only; no audio/native/draw/namespace creation',
    actual_validation_evidence=pin(A/'ROOT_end_to_end_memory_pipeline_validation_review_v1.json'),
    production_configuration_is_not_execution_admission=True,
    normal_path_implementation_unchanged=True,
    legacy_fresh_invocation_token='genesis_normal_pause4',
    legacy_token_semantics='Frozen protocol enum; physical pause4 is validation-only and disabled when owner_validation_onlyFalse')
source_pin=write(A/'full_owner_source_contract_v12.json',source)
identity=copy.deepcopy(old['identity']);identity['full_run_source_contract']=source_pin
identity.update(owner_validation_only=False,production_source_eligible=True,
    permanent_namespace_scope='original_source_monitored_full_two_broad_one_cleaner_pending_human_launch',
    normal_boundary_resume_enabled=True,crash_tail_rollback_supported=False)
identity_pin=write(A/'full_owner_monitored_real_run_identity_v1.json',identity)
root=Path(r'E:\gpu_training_scratchpad\runs\ASR18_monitored_full_two_broad_one_cleaner_PROPOSAL_01')
assert not root.exists()
inputs=copy.deepcopy(old)
for key in ('finite_scope','maximum_owner_validation_updates','maximum_validation_updates_per_window',
            'owner_validation_pause_after_updates','uninterrupted_eight_update_oracle'):
    inputs.pop(key,None)
inputs.update(source_manifest=source_pin,identity=identity,identity_binding=identity_pin,run_identity_sha256=digest(identity),
    run_namespace=str(root),storage_namespace=str(root/'checkpoint_storage'),durable_namespace=str(root/'durable'),
    actual_folder=str(A/'monitored_real_run_NOT_EXECUTED'),
    configuration_output=str(A/'disabled_monitored_real_run_configuration_01'),metadata_builder_source=pin(builder),
    metadata_scope='disabled_full_mode_original_first20_metadata_preparation_only',
    owner_validation_only=False,full_run_launch_enabled=False,launch_disabled=True,
    proposed_maximum_owner_wall_seconds=86400,proposed_maximum_GPU_child_wall_seconds=86400,
    deployment_proposal=pin(A/'disabled_monitored_real_run_configuration_preparation_proposal_v1.json'),
    functional_validation_review=pin(A/'ROOT_end_to_end_memory_pipeline_validation_review_v1.json'),
    legacy_fresh_invocation_token='genesis_normal_pause4',
    legacy_token_semantics=source['legacy_token_semantics'],
    no_original_numerical_source_or_guard_changes=True,
    no_actual_future_checkpoint_PID_or_owner_creation_evidence_claimed=True)
inputs_pin=write(A/'full_owner_exact_ROOT_review_inputs_v13.json',inputs)
delta={'kind':'ASR18_disabled_full_mode_metadata_assembler_source_delta_v1',
    'old_assembler':pin(old_builder),'new_assembler':pin(builder),'source_contract':source_pin,
    'identity':identity_pin,'ROOT_review_inputs':inputs_pin,'bindings':len(bindings),
    'unchanged_core_sources':{key:old[key] for key in ('parent_source','worker_source','owner_verifier_source')},
    'changes':['disabled full-mode guards; original first20/TR280 preview; matching bridge length and report kind',
        'new original-source identity/namespace; no diagnostic ancestry adoption; full launch remains false'],
    'actual_jobs':0,'namespace_created':False,'production_enabled':False}
delta_pin=write(A/'disabled_full_mode_metadata_assembler_source_delta_v1.json',delta)
print(json.dumps({'builder':pin(builder),'source_contract':source_pin,'identity':identity_pin,
    'ROOT_inputs':inputs_pin,'source_delta':delta_pin,'run_identity_sha256':digest(identity),'bindings':len(bindings)}))
