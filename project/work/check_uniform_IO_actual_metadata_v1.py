"""Both technical modes on actual CLOSED metadata; no tensor bytes or ML."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,ast
root=Path(__file__).resolve().parent.parent;base=root/'outputs/parakeet_turkish';sys.path.insert(0,str(base))
import tsc_uniform_io_transition_v1 as boundary
import record_TSC_uniform_io_source_v1 as creator
results=[]
for mode in boundary.PERSISTENCE_MODES:
    target='adaptive_20261012_TR_uniform_IO_'+('double' if mode=='double_json' else 'single')+'_01'
    sources,receipt=creator.inspect_closed(SimpleNamespace(source_review=base/'TSC_uniform_IO25109_USER_control_review_v1.json',target_run_id=target,persistence_mode=mode))
    review=boundary.read(receipt['ROOT_review']['path']);config=boundary.read(sources['run_config'])
    checkpoints={k:v for k,v in review['artifacts'].items() if k.endswith('checkpoint')};checkpoints['original_champion']=config['source_champion_parked']
    records={k:(checkpoints[k] if k in checkpoints else boundary.record(p)) for k,p in sources.items()}
    receipt.update(artifacts=records,original_artifact_records=records,original_artifact_paths={k:str(p) for k,p in sources.items()})
    ephemeral=root/('work/IO_'+mode+'_metadata_verification_receipt_v1.json');assert not ephemeral.exists();ephemeral.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    context=boundary.source_context(str(ephemeral),target)
    values={**context['config'],**context['source_recipe'],'run_id':target,'resume_majority':Path(context['source_checkpoint']['path']),
        'majority_source_receipt':ephemeral,'persistence_benchmark_updates':100}
    context=boundary.preflight(SimpleNamespace(**values))
    assert context['source_step']==25109 and context['source_actual_failure_counter']==23 and context['source_best_step']==19750
    assert context['source_lineage_end_step']==25109 and context['technical_end_step']==25209 and context['technical_extension_updates']==100
    assert context['prior_uniform_draws']==253800 and not context['source_retention_gates']['es']['passed']
    assert context['parent_recipe']['steps']==25109 and context['source_recipe']['steps']==25209
    assert context['parent_recipe']['lora_lr']==context['source_recipe']['lora_lr']==1e-4
    assert context['parent_recipe']['portuguese_token_kl_multiplier']==context['source_recipe']['portuguese_token_kl_multiplier']==1
    assert boundary.io_declaration(context,mode)['technical_updates']==100
    for changes in ({'persistence_benchmark_updates':99},{'lora_lr':3e-5},{'persistence_mode':boundary.PERSISTENCE_MODES[1-boundary.PERSISTENCE_MODES.index(mode)]}):
        try:boundary.preflight(SimpleNamespace(**{**values,**changes}))
        except ValueError:pass
        else:raise AssertionError('Undeclared technical or science change admitted')
    coverage={}
    for name in ('train_lora_batched_tsc_uniform_io_v1.py','run_TSC_uniform_io_v1.py'):
        path=base/name
        if path.exists():
            tree=ast.parse(path.read_text(encoding='utf-8'));keys=sorted({n.slice.value for n in ast.walk(tree) if isinstance(n,ast.Subscript)
                and isinstance(n.value,ast.Name) and n.value.id=='context' and isinstance(n.slice,ast.Constant) and isinstance(n.slice.value,str)})
            assert not set(keys)-set(context),(name,set(keys)-set(context));coverage[name]=keys
    results.append({'persistence_mode':mode,'passed':True,'ephemeral_metadata_only_receipt_not_adoption_receipt':str(ephemeral),
        'rejected_undeclared_changes':3,'consumer_context_keys_at_check':coverage})
report={'schema':1,'kind':'IO25109_actual_closed_metadata_verification_v1','passed':True,'source_review':receipt['ROOT_review'],
    'boundary':boundary.record(base/'tsc_uniform_io_transition_v1.py'),'creator':boundary.record(base/'record_TSC_uniform_io_source_v1.py'),
    'source_step':25109,'source_counter_preserved':23,'selected_BEST_preserved':19750,'source_lineage_end_step':25109,
    'technical_end_step':25209,'additional_updates':100,'source_uniform_unique_clips':253798,'source_uniform_draws':253800,
    'results':results,'checkpoint_bytes_read':False,'Torch_imported':False,'GPU_used':False,
    'native_math_or_sampler_execution_measured':False}
output=base/'TSC_uniform_IO25109_actual_closed_metadata_verification_v1.json';assert not output.exists();output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'output':str(output),'boundary':report['boundary'],'creator':report['creator']},indent=2))
