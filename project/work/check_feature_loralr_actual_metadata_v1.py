"""Actual closed metadata and consumer-key admission, no tensors or GPU."""
from pathlib import Path
import sys,ast,json
from types import SimpleNamespace
root=Path(__file__).resolve().parent.parent;base=root/'outputs/parakeet_turkish';sys.path.insert(0,str(base))
import tsc_uniform_feature_loralr_transition_v1 as boundary
import record_TSC_uniform_feature_loralr_source_v1 as creator
target='adaptive_20261012_TR_uniform_feature_lora3e5_01'
sources,receipt=creator.inspect_closed(SimpleNamespace(source_review=base/'TSC_feature_loralr_measured_repair_review_v1.json',target_run_id=target))
review=boundary.read(receipt['ROOT_review']['path']);config=boundary.read(sources['run_config'])
checkpoints={key:item for key,item in review['artifacts'].items() if key.endswith('checkpoint')}
checkpoints['original_champion']=config['source_champion_parked']
records={key:(checkpoints[key] if key in checkpoints else boundary.record(path)) for key,path in sources.items()}
receipt.update(artifacts=records,original_artifact_records=records,original_artifact_paths={k:str(v) for k,v in sources.items()})
ephemeral=root/'work/feature_loralr_closed_metadata_verification_receipt_v1.json';assert not ephemeral.exists()
ephemeral.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
context=boundary.source_context(str(ephemeral),target)
values={**context['config'],**context['source_recipe'],'run_id':target,'resume_majority':Path(context['source_checkpoint']['path']),'majority_source_receipt':ephemeral}
admitted=boundary.preflight(SimpleNamespace(**values))
assert admitted['source_step']==8250 and admitted['source_actual_failure_counter']==0 and admitted['source_best_step']==8250
assert admitted['prior_uniform_draws']==17774 and admitted['remaining_updates_from_chosen_frontier']==16859
assert admitted['configured_end_step']==25109 and all(v['passed'] for v in admitted['source_retention_gates'].values())
assert admitted['parent_recipe']['lora_lr']==1e-4 and admitted['source_recipe']['lora_lr']==3e-5
assert admitted['source_recipe']['feature_distillation']==20 and admitted['source_recipe']['replay_kl']==40
coverage={}
for name in ('train_lora_batched_tsc_uniform_feature_loralr_v1.py','run_TSC_uniform_feature_loralr_v1.py'):
    tree=ast.parse((base/name).read_text(encoding='utf-8'))
    keys=sorted({node.slice.value for node in ast.walk(tree) if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name)
        and node.value.id=='context' and isinstance(node.slice,ast.Constant) and isinstance(node.slice.value,str)})
    assert not set(keys)-set(admitted),(name,set(keys)-set(admitted));coverage[name]=keys
for changes in ({'lora_lr':1e-4},{'feature_distillation':4.}):
    try:boundary.preflight(SimpleNamespace(**{**values,**changes}))
    except ValueError:pass
    else:raise AssertionError('Additional recipe change admitted')
result={'schema':1,'kind':'Feature_LoRA_LR8250_actual_metadata_consumer_verification_v1','passed':True,
    'source_review':receipt['ROOT_review'],'boundary':boundary.record(base/'tsc_uniform_feature_loralr_transition_v1.py'),
    'creator':boundary.record(base/'record_TSC_uniform_feature_loralr_source_v1.py'),
    'actual_CLOSED_source_metadata_used':True,'ephemeral_metadata_only_receipt_not_adoption_receipt':str(ephemeral),
    'source_step':8250,'source_counter_preserved':0,'coherent_source_role':'best','chosen_uniform_unique_clips':17774,
    'configured_end_step':25109,'remaining_updates':16859,'coverage_rewind_clips':10500,
    'source_LoRA_LR_to_audit':1e-4,'active_LoRA_LR':3e-5,'consumer_context_key_coverage':coverage,
    'additional_recipe_rejections':2,'checkpoint_bytes_read':False,'Torch_imported':False,'GPU_used':False,
    'runtime_state_restore_native_decode_and_positive_updates_verified':False}
output=base/'TSC_feature_LoRA_LR8250_actual_metadata_consumer_verification_v1.json';assert not output.exists()
output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'output':str(output),'boundary':result['boundary'],'creator':result['creator']},indent=2))
