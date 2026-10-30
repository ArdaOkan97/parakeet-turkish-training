"""Inactive PT-focused replay full-state fork from ROOT-reviewed CLOSED lowerLR state.

All207 parent Adam moments are audited at the actual unchanged1e-4 group LR.
Quota7621, uniformorigin6500, queues, grace and failure counters carry.
No checkpoint bytes or ML imports occur during metadata preflight.
"""
from __future__ import annotations
from copy import deepcopy
from functools import lru_cache
import math
from pathlib import Path

import tsc_uniform_majority_transition_v1 as base
import tsc_uniform_majority_loralr_transition_v1 as parent
from tsc_uniform_majority_transition_v1 import (
    HERE,ROOT,LANGUAGES,PHASE,require,read,scoped,record,checked_record,canonical,
    retention_gates,_history_frontier,proof_path,compare_source_development)

RECEIPT_KIND='ROOT_owned_closed_TSC_majority_for_PT_focused_replay_v1'
REVIEW_KIND='ROOT_owned_closed_TSC_majority_lowerLR_PT_replay_review_v1'
PARENT_WORKER='train_lora_batched_tsc_uniform_majority_loralr_v1.py'
PARENT_CONTROLLER='run_TSC_uniform_majority_loralr_v1.py'
PARENT_PINS={PARENT_WORKER:'1b2089821aac2923a0e516253610961b4dc5a25f35ced77a49947e7908151114',
    PARENT_CONTROLLER:'0684422301c926e0037bfa0c1423230ce39dbc95cac538521882db9a68f7746e',
    'tsc_uniform_majority_loralr_transition_v1.py':'e24eb0dda449fd42e7204c8e354020dd54caed208d4eb4c5784b44b3ef2dfea9'}
DECLARED_CHANGE={'head_lr':1e-5,'encoder_LoRA_lr':1e-4,'TR_per_update':14,'replay_per_update':2,
    'PT_replay_clips_per_update':1,'original_five_replay_clips_per_update':1,
    'original_five_replay_language_choice':'uniform among EN ES IT FR DE, preserve existing queues and RNG state with declared future choice change',
    'PT_preservation_coefficients':[40,2,4],
    'original_five_preservation_coefficients':[40*(5/3),2*(5/3),4*(5/3)],
    'normalization':'Oldfive replay probability perupdate falls from2/6 to1/5; multiply oldfive preservation terms5/3 to retain expected per-language density. PT draws rise2/6 to1 and PT preservation density triples. TR14 unchanged. Gradients/quality not guaranteed identical.'}


def ptfocus_declaration(context):
    return {'kind':'declared_PT_focused_replay_full_state_v1','source_step':context['source_step'],
        'quota_boundary_step':context['quota_boundary_step'],'uniform_origin_step':6500,
        'configured_end_step':context['configured_end_step'],'head_lr':1e-5,'lora_lr':1e-4,
        'declared_change':DECLARED_CHANGE,'all207_Adam_moments_and_global_clocks_preserved':True,
        'fresh_optimizer_or_sampler':False,'renewed_Portuguese_grace':False,
        'parent_failure_counter_preserved':True,'all_six_required_for_every_BEST':True,
        'future_replay_language_choices_change_explicitly':True}



def validate_receipt(value,run_id):
    require(value.get('schema')==1 and value.get('kind')==RECEIPT_KIND
        and value.get('status')=='actual_CLOSED_reviewed_majority_PT_fork'
        and value.get('target_run_id')==run_id and value.get('source_run_id')!=run_id
        and value.get('chosen_checkpoint_role')=='latest' and value.get('chosen_step')==value.get('completed_steps')
        and value.get('allow_declared_PT_focused_replay_continuation') is True
        and value.get('declared_change')==DECLARED_CHANGE,'Separate ROOT-declared latest PT-focused full-state fork required')
    return value


@lru_cache(maxsize=2)
def source_context(receipt_path,run_id=None):
    path=scoped(receipt_path);receipt=read(path);validate_receipt(receipt,run_id or receipt['target_run_id'])
    review=read(checked_record(receipt['ROOT_closed_retention_review']));a=receipt['artifacts'];original=receipt['original_artifact_records']
    for key,item in a.items():checked_record(item,checkpoint=key in ('chosen_checkpoint','source_best_checkpoint','original_champion'))
    for name,digest in PARENT_PINS.items():require(record(HERE/name)['sha256']==digest,'Frozen majority source changed: '+name)
    require(review.get('schema')==1 and review.get('kind')==REVIEW_KIND and review.get('passed') is True
        and review.get('source_run_id')==receipt['source_run_id'] and review.get('chosen_step')==receipt['chosen_step']
        and review.get('chosen_checkpoint_role')=='latest' and review.get('allow_declared_PT_focused_replay_continuation') is True
        and review.get('declared_change')==DECLARED_CHANGE and review.get('renewed_Portuguese_grace') is False
        and review.get('parent_failure_counter_preserved') is True
        and review.get('chosen_frontier_is_qualified_BEST') is False,'Actual ROOT retention/closure/PT-focused review required')
    for key,review_key in (('controller_status','controller_status'),('runtime_completion','native_completion'),
        ('selection','selection'),('chosen_DEV_summary','chosen_DEV_summary'),('chosen_checkpoint','chosen_checkpoint'),
        ('initial_source_equivalence','source350_equivalence')):
        require(original[key]==review[review_key],'Reviewed actual source artifact changed: '+key)
    controller=read(a['controller_status']['path']);selection=read(a['selection']['path']);config=read(a['run_config']['path'])
    identity=config['config_identity'];h=receipt['chosen_step'];recipe=dict(identity['training_arguments'])
    require(controller.get('status')=='train_window_ended' and controller.get('child_exit_code')==0
        and controller.get('resource_review_required') is False and controller.get('stop_category')=='retention'
        and controller.get('controller_sha256')==PARENT_PINS[PARENT_CONTROLLER]
        and review['owned_actual_exit']['exit_code']==0 and receipt['owned_actual_exit']==review['owned_actual_exit']
        and controller['completed_steps']==selection['completed_steps']==h and config['run_id']==controller['run_id']==selection['run_id']==receipt['source_run_id']
        and config['phase']=='HF_TSC_8199_uniform_Turkish_majority_lower_LoRA_v1' and config['trainable_parameter_tensors']==207
        and config['turkish_clips_used']==253798 and config['replay_clips_used']==1914
        and selection['heldout_tests_evaluated'] is False and selection['accepted'] is False and selection['selected_step'] is None,
        'Actually CLOSED latest unqualified majority whole-state boundary required')
    require(canonical(identity)==receipt['scientific_config_identity_sha256']
        and identity['extended_tokenizer_sha256']=='b91d500326aca290ed709d629dab2bee56f298ea7323b94855b9f5ef0caeac75'
        and recipe['head_lr']==1e-5 and recipe['lora_lr']==1e-4 and recipe['schedule']=='constant'
        and recipe['replay_fraction']==.125 and recipe['replay_kl']==40 and recipe['duration_kl']==2 and recipe['feature_distillation']==4
        and recipe['batch_size']==8 and recipe['accumulation']==2,'Exact source207/8199/14-2/40-2-4 science required')
    for name,digest in identity['source_sha256'].items():require(record(HERE/name)['sha256']==digest,'Actual source pin changed: '+name)
    require(scoped(original['chosen_checkpoint']['path'])==scoped(selection['latest_weights']), 'Chosen latest pointer changed')
    prefix,last=_history_frontier(a['training_history']['path'],h,h)
    require(prefix==receipt['source_history_prefix_sha256'] and last['selection_diagnostics']['consecutive_retention_failures']==review['parent_retention_failure_counter']
        and selection['consecutive_retention_failures']==review['parent_retention_failure_counter'],'Exact inherited history/failure counter required')
    old_record=receipt['original_LR_source_receipt'];checked_record(old_record)
    require(old_record==review['original_LR_source_receipt'],'Original quota source receipt changed')
    origin=parent.source_context(str(scoped(old_record['path'])),receipt['source_run_id'])
    quota=identity['domain_sampling_identity']['majority_quota_transition']
    require(quota==origin['quota_transition']
        and quota['configured_end_step']==recipe['steps']==selection['configured_steps']
        and config['uniform_origin_step']==6500,'Original immutable quota boundary/end/pass must not reset')
    coverage=last['Turkish_domain_coverage']['uniform_pass'];majority=last['Turkish_domain_coverage']['majority_transition']
    require(coverage['source_step']==6500 and coverage['draws']==coverage['unique_clips_seen']==review['actual_unique_uniform_clips']
        and coverage['repeat_visits']==0 and majority['local_updates']==h-quota['source_step']
        and h-origin['source_step']==review['actual_LR_fork_updates'],
        'Actual segmented coverage/local clock differs')
    summary=read(a['chosen_DEV_summary']['path']);baseline=read(a['Portuguese_baseline_proof']['path'])['baseline_overlay']
    gates=retention_gates(summary,baseline);require(gates==review['retention_gates']
        and all(gates[lang]['passed'] for lang in ('en','es','it','fr','de')) and gates['pt']['passed'] is False,
        'Actual old-five passed/Portuguese failed review facts required')
    require(summary['validation_tr']['asr']['wer']<origin['source_summary']['validation_tr']['asr']['wer'],
        'Reviewed latest must improve its source Turkish score')
    done=read(a['runtime_completion']['path']);owned=read(a['owned_worker_identity']['path']);ready=read(a['runtime_native_ready']['path'])
    require(done.get('status')=='complete' and done.get('sources_unchanged') is True and not done.get('error')
        and done['pid']==owned['pid']==ready['pid'] and done['run_id']==receipt['source_run_id']
        and done['actual_dispatch_source_sha256']==PARENT_PINS[PARENT_CONTROLLER]
        and done['backend_ready_sha256']==a['original_native_ready']['sha256']
        and done['fresh_runtime_sha256']==a['runtime_native_ready']['sha256'],'Actual native/current owned closure required')
    for key in ('memory','cache','io'):
        runtime=read(a[key]['path']);require(runtime.get('status')=='complete' and runtime.get('sources_unchanged') is True
            and not runtime.get('error'),'Actual runtime incomplete: '+key)
    memory=read(a['memory']['path']);cache=read(a['cache']['path'])
    require(memory['cap_bytes']==8_000_000_000 and memory['cleanup_threshold_bytes']==3_000_000_000
        and memory['allocator_cap_applied'] is True and memory['all_phase_owned_memory_telemetry']['thread_join_completed'] is True
        and cache['cache_reclamation']['eligible_update_calls']==h-controller['resume_step'],'Actual retained guard/hook frontier differs')
    audit=read(a['indexed_TR_audit']['path']);require(audit==config['public_turkish_data_audit'],'Original indexed audit changed')
    active=dict(recipe)
    context={'receipt':receipt,'receipt_record':record(path),'source_record':record(path),'config':config,'selection':selection,
        'parent_identity':identity,'parent_recipe':recipe,'source_recipe':active,'source_step':h,'source_selected_step':h,
        'uniform_origin_step':6500,'quota_boundary_step':quota['source_step'],'quota_transition':quota,'quota_origin_context':origin,
        'source_checkpoint':a['chosen_checkpoint'],'source_summary':summary,'source_frontier_row':last,
        'source_history_prefix_sha256':prefix,'source_retention_gates':gates,'prior_uniform_draws':coverage['draws'],
        'remaining_update_budget':quota['remaining_update_budget'],
        'remaining_updates_from_chosen_frontier':quota['configured_end_step']-h,'configured_end_step':quota['configured_end_step'],
        'indexed_audit':audit,'indexed_audit_record':a['indexed_TR_audit'],'original_source350':a['initial_source_equivalence'],
        'original_native_ready':a['original_native_ready'],'uniform_source_receipt':a['uniform_source_receipt'],
        'portuguese_preparation_proof':a['Portuguese_preparation_proof'],'portuguese_baseline_proof':a['Portuguese_baseline_proof'],
        'source_best_checkpoint':None,'source_best_step':None,'parked_original_champion':a['original_champion'],
        'original_LR_source_receipt':old_record,'combined_baseline':baseline}
    return context


def validate_portuguese_inputs(args,context):return base.validate_portuguese_inputs(args,context)


def preflight(args):
    context=source_context(str(scoped(args.majority_source_receipt)),args.run_id).copy()
    require(scoped(args.resume_majority)==scoped(context['source_checkpoint']['path']),'Chosen latest8000 whole-state input required')
    for key,value in context['source_recipe'].items():require(getattr(args,key)==value,'Unapproved science difference: '+key)
    for key in ('tsc_input_proof','tsc_training_approval','source_closure','tr_manifest','replay_manifest','data_v3_proof',
        'tr_frame_metadata','tr_frame_metadata_proof','reuse_input_audit','supplement','portuguese_preparation_proof','portuguese_baseline_proof',
        'portuguese_replay_manifest','portuguese_dev_manifest','indexed_tr_directory','resume_uniform_full','uniform_full_source_receipt'):
        before=context['config'].get(key);after=getattr(args,key,None)
        require((before is None and after is None) or (before is not None and after is not None and scoped(before)==scoped(after)),
            'Original input path changed: '+key)
    require(args.portuguese_seed==context['config']['portuguese_seed'],'Independent replay seed changed')
    validate_portuguese_inputs(args,context);return context


def audit_parent_state(model,state,context,*,torch_module):
    evidence=parent.audit_parent_state(model,state,context,torch_module=torch_module)
    evidence.pop('only_next_live_LoRA_group_lr_changed',None)
    evidence.update(source_LoRA_group_lr_audited=1e-4,active_LoRA_lr_unchanged=True,
        new_optimizer_or_moment_reset=False,declared_change=DECLARED_CHANGE)
    return evidence


def restore_parent_sampler(sampler,state,context):
    require(state['sampler']['updates']==context['source_step']
        and state['sampler']['identity']==context['parent_identity']['domain_sampling_identity'],
        'Same CLOSED8000 own sampler source required')
    evidence=sampler.migrate_closed_state(state['sampler'],context['parent_identity']['domain_sampling_identity'])
    require(sampler.updates==context['source_step'] and sampler.configured_end_step==context['configured_end_step']
        and sampler.transition_source_step==context['quota_boundary_step'],
        'PT selector transition must retain current frontier/original quota/horizon')
    return evidence



def release_admission_caches():
    source_context.cache_clear();parent.release_admission_caches()
    return {'PT_focus_and_original_source_caches_cleared':True,'disk_or_other_process_changed':False}
