"""Matched schedule modes on actual closed original25109 metadata only."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,ast
root=Path(__file__).resolve().parent.parent;base=root/'outputs/parakeet_turkish';sys.path.insert(0,str(base))
import tsc_uniform_encoder_schedule_transition_v1 as boundary
import record_TSC_uniform_encoder_schedule_source_v1 as creator
results=[]
for mode in boundary.ENCODER_SCHEDULE_MODES:
    target='adaptive_20261012_TR_uniform_encoder_'+mode+'_01'
    sources,receipt=creator.inspect_closed(SimpleNamespace(source_review=base/'TSC_uniform_encoder_schedule25109_USER_review_v1.json',target_run_id=target,encoder_lr_schedule=mode))
    review=boundary.read(receipt['ROOT_review']['path']);config=boundary.read(sources['run_config'])
    checkpoints={k:v for k,v in review['artifacts'].items() if k.endswith('checkpoint')};checkpoints['original_champion']=config['source_champion_parked']
    records={k:(checkpoints[k] if k in checkpoints else boundary.record(p)) for k,p in sources.items()}
    receipt.update(artifacts=records,original_artifact_records=records,original_artifact_paths={k:str(p) for k,p in sources.items()})
    ephemeral=root/('work/encoder_'+mode+'_metadata_verification_receipt_v1.json');assert not ephemeral.exists();ephemeral.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    context=boundary.source_context(str(ephemeral),target)
    values={**context['config'],**context['source_recipe'],'run_id':target,'resume_majority':Path(context['source_checkpoint']['path']),'majority_source_receipt':ephemeral}
    context=boundary.preflight(SimpleNamespace(**values))
    assert context['source_step']==25109 and context['source_actual_failure_counter']==23 and context['source_best_step']==19750
    assert context['source_lineage_end_step']==25109 and context['configured_end_step']==43238 and context['encoder_schedule_updates']==18129
    assert context['prior_uniform_draws']==253800 and not context['source_retention_gates']['es']['passed']
    assert context['parent_recipe']['steps']==25109 and context['source_recipe']['steps']==43238
    assert context['parent_recipe']['lora_lr']==context['source_recipe']['lora_lr']==1e-4 and context['source_recipe']['head_lr']==1e-5
    declared=boundary.encoder_schedule_declaration(context,mode)
    assert declared['last_applied_encoder_lr']==(1e-4 if mode=='constant' else 3e-5)
    for changes in ({'encoder_schedule_updates':100},{'head_lr':3e-5},{'persistence_mode':'single_serialize_two_commits'}):
        try:boundary.preflight(SimpleNamespace(**{**values,**changes}))
        except ValueError:pass
        else:raise AssertionError('Unapproved recipe or exposure admitted')
    coverage={}
    for name in ('train_lora_batched_tsc_uniform_encoder_schedule_v1.py','run_TSC_uniform_encoder_schedule_v1.py'):
        path=base/name
        if path.exists():
            tree=ast.parse(path.read_text(encoding='utf-8'));keys=sorted({n.slice.value for n in ast.walk(tree) if isinstance(n,ast.Subscript)
                and isinstance(n.value,ast.Name) and n.value.id=='context' and isinstance(n.slice,ast.Constant) and isinstance(n.slice.value,str)})
            assert not set(keys)-set(context),(name,set(keys)-set(context));coverage[name]=keys
    results.append({'mode':mode,'passed':True,'ephemeral_metadata_only_receipt_not_adoption':str(ephemeral),'rejected_changes':3,'consumer_context_keys':coverage})
report={'schema':1,'kind':'Encoder_schedule25109_actual_metadata_verification_v1','passed':True,'source_review':receipt['ROOT_review'],
    'boundary':boundary.record(base/'tsc_uniform_encoder_schedule_transition_v1.py'),'creator':boundary.record(base/'record_TSC_uniform_encoder_schedule_source_v1.py'),
    'source_step':25109,'source_counter':23,'separate_BEST_step':19750,'source_lineage_end_step':25109,'new_end_step':43238,
    'matched_updates':18129,'carried_cycle2_prefix':2,'remaining_cycle2_TR_visits':253796,'cycle3_tail_visits':10,'results':results,
    'checkpoint_bytes_read':False,'Torch_imported':False,'GPU_used':False,'actual_state_restore_math_quality_or_cycle_coverage_verified':False}
output=base/'TSC_uniform_encoder_schedule25109_actual_metadata_verification_v1.json';assert not output.exists();output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'output':str(output),'boundary':report['boundary'],'creator':report['creator']},indent=2))
