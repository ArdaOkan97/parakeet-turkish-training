"""Actual closed metadata consumer coverage; no checkpoint byte read or ML."""
from pathlib import Path
import sys,ast,json,hashlib
from types import SimpleNamespace

root=Path(__file__).resolve().parent.parent;base=root/'outputs/parakeet_turkish'
sys.path.insert(0,str(base))
import tsc_uniform_pttoken_constraint_transition_v1 as boundary
import record_TSC_uniform_pttoken_constraint_source_v1 as creator
target='adaptive_20261012_TR_uniform_PT_token5x_01'
sources,receipt=creator.inspect_closed(SimpleNamespace(source_review=base/'TSC_PT_token5x_measured_retention_repair_review_v1.json',target_run_id=target))
review=boundary.read(receipt['ROOT_review']['path']);config=boundary.read(sources['run_config'])
checkpoints={key:item for key,item in review['artifacts'].items() if key.endswith('checkpoint')}
checkpoints['original_champion']=config['source_champion_parked']
records={key:(checkpoints[key] if key in checkpoints else boundary.record(path)) for key,path in sources.items()}
receipt.update(artifacts=records,original_artifact_records=records,original_artifact_paths={k:str(v) for k,v in sources.items()})
ephemeral=root/'work/PT_token_closed_metadata_verification_receipt_v1.json'
assert not ephemeral.exists()
ephemeral.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
context=boundary.source_context(str(ephemeral),target)
values={**context['config'],**context['source_recipe'],'run_id':target,
    'resume_majority':Path(context['source_checkpoint']['path']),'majority_source_receipt':ephemeral}
admitted=boundary.preflight(SimpleNamespace(**values))
assert admitted['source_step']==8750 and admitted['source_actual_failure_counter']==2
assert admitted['source_best_step']==8250 and admitted['prior_uniform_draws']==24774
assert admitted['remaining_updates_from_chosen_frontier']==16359 and admitted['configured_end_step']==25109
assert not admitted['source_retention_gates']['pt']['passed']
assert admitted['source_retention_gates']['pt']['word_errors']==70
coverage={}
for name in ('train_lora_batched_tsc_uniform_pttoken_v1.py','run_TSC_uniform_pttoken_v1.py'):
    tree=ast.parse((base/name).read_text(encoding='utf-8'))
    keys=sorted({node.slice.value for node in ast.walk(tree) if isinstance(node,ast.Subscript)
        and isinstance(node.value,ast.Name) and node.value.id=='context'
        and isinstance(node.slice,ast.Constant) and isinstance(node.slice.value,str)})
    assert not set(keys)-set(admitted),(name,set(keys)-set(admitted))
    coverage[name]=keys
assert boundary.token_constraint_declaration(context)['preservation_coefficients_by_language']['pt']==[200.,2.,20.]
for bad in (1.,3.):
    try:boundary.preflight(SimpleNamespace(**{**values,'portuguese_token_kl_multiplier':bad}))
    except ValueError:pass
    else:raise AssertionError('Undeclared token multiplier admitted')
result={'schema':1,'kind':'PT_token8750_actual_metadata_consumer_verification_v1','passed':True,
    'source_review':receipt['ROOT_review'],'boundary':boundary.record(base/'tsc_uniform_pttoken_constraint_transition_v1.py'),
    'creator':boundary.record(base/'record_TSC_uniform_pttoken_constraint_source_v1.py'),
    'actual_CLOSED_source_metadata_used':True,'ephemeral_metadata_only_receipt_not_adoption_receipt':str(ephemeral),
    'source_step':8750,'source_counter_preserved':2,'separate_BEST_step':8250,'PT_source_errors':70,'PT_source_limit':66,
    'uniform_unique_clips':24774,'configured_end_step':25109,'remaining_updates':16359,'coverage_rewind_clips':0,
    'consumer_context_key_coverage':coverage,'undeclared_multiplier_rejections':2,
    'checkpoint_bytes_read':False,'Torch_imported':False,'GPU_used':False,
    'runtime_state_restore_native_decode_and_positive_updates_verified':False}
output=base/'TSC_PT_token8750_actual_metadata_consumer_verification_v1.json'
assert not output.exists();output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'output':str(output),'boundary':result['boundary'],'creator':result['creator']},indent=2))
