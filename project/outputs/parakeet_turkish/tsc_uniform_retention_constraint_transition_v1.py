"""Measured one-variable feature-KD fork from coherent qualified own BEST8250.

The failed8750 branch stays archived. The explicit7000-clip rewind is a new
scientific namespace, not exact latest continuation. No tensor/ML import.
"""
from __future__ import annotations
from functools import lru_cache
import math

import tsc_uniform_majority_ptfocus_integer_transition_v1 as parent
from tsc_uniform_majority_ptfocus_integer_transition_v1 import (
    HERE,ROOT,LANGUAGES,require,read,scoped,record,checked_record,canonical,_history_frontier,
    proof_path,retention_gates,compare_source_development,restore_parent_sampler)
from tsc_uniform_majority_transition_v1 import audit_source_optimizer

RECEIPT_KIND='ROOT_owned_closed_TSC_uniform_retention_constraint_v1'
REVIEW_KIND='ROOT_delegated_measured_TSC_retention_constraint_review_v1'
PARENT_WORKER='train_lora_batched_tsc_uniform_majority_ptfocus_v3.py'
PARENT_CONTROLLER='run_TSC_uniform_majority_ptfocus_v3.py'
PARENT_PINS={PARENT_WORKER:'cbe277fb8464008d755f7f88f0c26cc2662bcc871efd271b5759e99f7352ec5b',
    PARENT_CONTROLLER:'1de96cba7a708f461e8340ae39ea64a89b9653fecac6b76ca4ccde5330578e9f',
    'tsc_uniform_majority_ptfocus_integer_transition_v1.py':'8e9444372e9504298a5e9a38681ecede1279677c7b928f66f375cc529fc619b5'}


def constraint_declaration(context):
    return {'source_feature_distillation':4.,'active_feature_distillation':20.,'source_step':8250,
        'failed_completed_step':8750,'rewound_TR_clips':7000,
        'preservation_coefficients_by_language':{'pt':[40.,2.,20.],'old_five':[40*(5/3),2*(5/3),20*(5/3)]},
        'all207_Adam_sampler_RNG_history_preserved':True,'ROOT_review':context['receipt']['ROOT_review'],
        'actual_diagnostic':context['constraint_review']['actual_diagnostic'],
        'original_integer_gate_correction_preserved':True,'renewed_Portuguese_grace':False,
        'source_actual_retention_failures':0,'source_actual_selected_step':8250,'new_counter_migration':False,
        'no_hybrid_training_state':True,'diagnostic_is_conditional_not_universal_cause_proof':True}


def ptfocus_declaration(context):return context['config']['PT_replay_policy_transition']


def validate_receipt(value,run_id):
    require(value.get('schema')==1 and value.get('kind')==RECEIPT_KIND
        and value.get('status')=='actual_CLOSED_qualified_BEST_measured_feature_constraint'
        and value.get('target_run_id')==run_id and value.get('source_run_id')!=run_id
        and value.get('chosen_step')==8250 and value.get('completed_steps')==8750 and value.get('chosen_checkpoint_role')=='best'
        and value.get('explicit_coverage_rewind_clips')==7000,'Explicit coherent BEST8250 rewind/fork required')
    return value


@lru_cache(maxsize=2)
def source_context(receipt_path,run_id=None):
    path=scoped(receipt_path);receipt=read(path);validate_receipt(receipt,run_id or receipt['target_run_id'])
    review=read(checked_record(receipt['ROOT_review']));a=receipt['artifacts'];original=receipt['original_artifact_records']
    require(review.get('kind')==REVIEW_KIND and review.get('passed') is True and review.get('chosen_step')==8250
        and review.get('completed_steps')==8750 and review.get('chosen_checkpoint_role')=='best'
        and review.get('allow_declared_feature_distillation_constraint_fork') is True
        and review.get('source_counter_preserved')==0 and review.get('source_selected_BEST_preserved')==8250
        and review.get('source_unqualified_latest_used') is False and review.get('no_hybrid_training_state_used') is True
        and review.get('explicit_coverage_rewind_clips')==7000 and review.get('failed_branch_history_preserved') is True
        and review.get('renewed_Portuguese_grace') is False,'Actual delegated measured coherent-BEST constraint review required')
    change=review['declared_change'];require(change['parameter']=='feature_distillation' and change['source_value']==4
        and change['active_value']==20 and change['multiplier']==5 and change['token_KL_unchanged']==40
        and change['duration_KL_unchanged']==2 and change['head_LR_unchanged']==1e-5 and change['LoRA_LR_unchanged']==1e-4
        and change['TR_per_update']==14 and change['PT_replay_per_update']==change['old_five_replay_per_update']==1
        and change['old_five_language_weight_multiplier']==5/3 and change['fixed_integer_gate_limits_unchanged'] is True,
        'Only feature coefficient4 to20 may change')
    closed=read(checked_record(review['actual_closed_source_review']));diagnostic=read(checked_record(review['actual_diagnostic']))
    require(review['source_actual_owned_exit']['exit_code']==0 and review['actual_diagnostic_owned_exit']['exit_code']==0
        and closed['completed_steps']==8750 and closed['run_id']==receipt['source_run_id']
        and diagnostic.get('passed') is True,'Actual source/diagnostic CLOSED0 and measured result required')
    for key,item in a.items():checked_record(item,checkpoint=key in ('chosen_checkpoint','latest_checkpoint','original_champion'))
    for key,item in review['artifacts'].items():require(original[key]==item,'Actual reviewed source binding changed: '+key)
    for name,digest in PARENT_PINS.items():require(record(HERE/name)['sha256']==digest,'Frozen parent changed: '+name)
    config=read(a['run_config']['path']);selection=read(a['selection']['path']);controller=read(a['controller_status']['path']);identity=config['config_identity']
    require(config['run_id']==selection['run_id']==controller['run_id']==receipt['source_run_id']
        and controller['status']=='train_window_ended' and controller['child_exit_code']==0 and controller['resource_review_required'] is False
        and controller['completed_steps']==selection['completed_steps']==8750 and selection['accepted'] is True
        and selection['selected_step']==8250 and selection['heldout_tests_evaluated'] is False
        and scoped(original['chosen_checkpoint']['path'])==scoped(selection['weights'])
        and scoped(original['latest_checkpoint']['path'])==scoped(selection['latest_weights']),
        'Qualified BEST8250 must remain separate from failed latest8750')
    require(canonical(identity)==receipt['scientific_config_identity_sha256'],'Parent identity changed')
    for name,digest in identity['source_sha256'].items():require(record(HERE/name)['sha256']==digest,'Parent dependency changed: '+name)
    prefix,last=_history_frontier(a['training_history']['path'],8250,8750)
    require(prefix==receipt['source_history_prefix_sha256'],'Chosen BEST history prefix changed')
    origin_record=receipt['original_integer_source_receipt'];checked_record(origin_record)
    require(scoped(origin_record['path'])==scoped(config['majority_source_receipt']),'Original numeric source lineage changed')
    origin=parent.source_context(str(scoped(origin_record['path'])),receipt['source_run_id'])
    recipe=dict(identity['training_arguments']);require(recipe==origin['source_recipe'] and recipe['feature_distillation']==4,
        'Original source recipe must be feature4 and unchanged otherwise')
    summary=read(a['chosen_DEV_summary']['path']);baseline=read(a['Portuguese_baseline_proof']['path'])['baseline_overlay'];gates=retention_gates(summary,baseline)
    require(all(v['passed'] for v in gates.values()) and summary['validation_tr']['asr']['wer']==selection['turkish_validation_wer'],
        'Actual chosen8250 all-six-qualified source counts required')
    sampler_identity=identity['domain_sampling_identity'];quota=sampler_identity['majority_quota_transition'];focus=sampler_identity['PT_focus_transition']
    require(quota==origin['quota_transition'] and focus['source_step']==8000 and quota['source_step']==7621 and quota['configured_end_step']==25109,
        'Original same-pass and PT selector boundaries remain immutable')
    #8250 history is the original false Boolean; the selected whole state has
    #already applied the ROOT-reviewed integer accounting projection.
    require(origin['review']['kind']==parent.REVIEW_KIND and config['retention_numeric_correction']['ROOT_review']==origin['receipt']['ROOT_review'],
        'Existing ROOT integer correction must remain explicit')
    done=read(a['runtime_completion']['path']);owned=read(a['owned_worker_identity']['path'])
    require(done['status']=='complete' and done['sources_unchanged'] is True and done['pid']==owned['pid']
        and done['actual_dispatch_source_sha256']==PARENT_PINS[PARENT_CONTROLLER],'Actual original runtime closure required')
    for key in ('memory','cache','io'):
        value=read(a[key]['path']);require(value['status']=='complete' and value['sources_unchanged'] is True,'Actual runtime incomplete: '+key)
    context={**origin,'receipt':receipt,'receipt_record':record(path),'source_record':record(path),'constraint_review':review,
        'config':config,'selection':selection,'parent_identity':identity,'parent_recipe':recipe,'source_recipe':{**recipe,'feature_distillation':20.},
        'source_step':8250,'source_selected_step':8250,'source_checkpoint':a['chosen_checkpoint'],'source_summary':summary,
        'source_history_prefix_sha256':prefix,'source_frontier_row':last,'source_retention_gates':gates,
        'source_actual_selected_step':8250,'source_actual_failure_counter':0,'corrected_initial_retention_failures':0,
        'corrected_initial_best_score':summary['validation_tr']['asr']['wer'],'corrected_initial_selected_step':8250,
        'quota_transition':quota,'indexed_audit':read(a['indexed_TR_audit']['path']),'indexed_audit_record':a['indexed_TR_audit'],
        'source_best_checkpoint':a['chosen_checkpoint'],'source_best_step':8250,'original_source350':a['initial_source_equivalence'],
        'original_native_ready':a['original_native_ready'],'uniform_source_receipt':a['uniform_source_receipt'],
        'portuguese_preparation_proof':a['Portuguese_preparation_proof'],'portuguese_baseline_proof':a['Portuguese_baseline_proof'],
        'parked_original_champion':a['original_champion'],'combined_baseline':baseline,'failed_completed_step':8750}
    return context


def validate_portuguese_inputs(args,context):return parent.validate_portuguese_inputs(args,context)


def preflight(args):
    context=source_context(str(scoped(args.majority_source_receipt)),args.run_id).copy()
    require(scoped(args.resume_majority)==scoped(context['source_checkpoint']['path']),'Only coherent BEST8250 whole state may initialize')
    for key,value in context['source_recipe'].items():require(getattr(args,key)==value,'Unapproved additional change: '+key)
    for key in ('tsc_input_proof','tsc_training_approval','source_closure','tr_manifest','replay_manifest','data_v3_proof',
        'tr_frame_metadata','tr_frame_metadata_proof','reuse_input_audit','supplement','portuguese_preparation_proof','portuguese_baseline_proof',
        'portuguese_replay_manifest','portuguese_dev_manifest','indexed_tr_directory','resume_uniform_full','uniform_full_source_receipt'):
        before=context['config'].get(key);after=getattr(args,key,None)
        require((before is None and after is None) or (before is not None and after is not None and scoped(before)==scoped(after)),
            'Original input path changed: '+key)
    require(args.portuguese_seed==context['config']['portuguese_seed'],'Independent replay seed changed')
    validate_portuguese_inputs(args,context);return context


def audit_parent_state(model,state,context,*,torch_module):
    torch=torch_module;require(state.get('schema')==2 and state.get('run_id')==context['receipt']['source_run_id']
        and state.get('step')==8250 and canonical(state.get('config_identity'))==canonical(context['parent_identity'])
        and state['selected_step']==8250 and state['best_score']==context['corrected_initial_best_score'] and state['retention_failures']==0,
        'Actual coherent qualified BEST8250 counter0/selected8250 must remain intact')
    require(len(state['history'])==8250 and canonical(state['history'])==context['source_history_prefix_sha256']
        and state['sampler']['updates']==8250 and state['sampler']['identity']==context['parent_identity']['domain_sampling_identity'],
        'BEST8250 weights/moments/history/queues must share one frontier')
    for key in ('optimizer','lora_metadata','python_rng','torch_rng','cuda_rng','patience','next_eval_epoch','train_wall_seconds',
        'audio_seconds','language_audio_seconds','evaluation_wall_seconds','initial_gate','stop_reason'):
        require(key in state,'Incomplete coherent source runtime: '+key)
    expected={name:p for name,p in model.named_parameters() if p.requires_grad};require(len(expected)==207 and set(expected)==set(state['weights']),'All207 source weights required')
    for name,value in state['weights'].items():require(value.shape==expected[name].shape and value.dtype==torch.float32
        and bool(torch.isfinite(value).all()),'FP32 source shape/finiteness differs: '+name)
    from encoder_lora import encoder_lora_metadata
    require(encoder_lora_metadata(model)==state['lora_metadata'] and model.config.vocab_size==8199
        and model.config.blank_token_id==8192 and list(model.config.durations)==[0,1,2,3,4], 'Original207/8199 source geometry required')
    audit=audit_source_optimizer(state,{**context,'source_recipe':context['parent_recipe']},expected,torch_module=torch)
    require(record(context['source_checkpoint']['path'])==context['source_checkpoint'],'Chosen coherent8250 bytes changed')
    return {'passed':True,'source_Adam_audit':audit,'all207_Adam_sampler_RNG_history_preserved':True,
        'source_actual_selected_step':8250,'source_actual_failure_counter':0,'feature_coefficient_change_only':[4.,20.],
        'hybrid_or_latest8750_state_used':False}


def release_admission_caches():
    source_context.cache_clear();parent.release_admission_caches()
    return {'feature_constraint_and_original_source_caches_cleared':True,'disk_or_other_process_changed':False}
