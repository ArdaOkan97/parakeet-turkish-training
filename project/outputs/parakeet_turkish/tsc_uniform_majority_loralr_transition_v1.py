"""Inactive LR-only full-state fork from a ROOT-reviewed CLOSED majority state.

All207 parent Adam moments are audited at3e-4 before the live LoRA group changes
to1e-4. Quota7621, uniformorigin6500, queues, grace and failure counters carry.
No checkpoint bytes or ML imports occur during metadata preflight.
"""
from __future__ import annotations
from copy import deepcopy
from functools import lru_cache
import math
from pathlib import Path

import tsc_uniform_majority_transition_v1 as base
from tsc_uniform_majority_transition_v1 import (
    HERE,ROOT,LANGUAGES,PHASE,require,read,scoped,record,checked_record,canonical,
    retention_gates,_history_frontier,proof_path,compare_source_development)

RECEIPT_KIND='ROOT_owned_closed_TSC_majority_for_lower_LoRA_LR_v1'
REVIEW_KIND='ROOT_owned_closed_TSC_majority_retention_review_v1'
PARENT_WORKER='train_lora_batched_tsc_uniform_majority_v1.py'
PARENT_CONTROLLER='run_TSC_uniform_majority_v1.py'
PARENT_PINS={PARENT_WORKER:'b076b4a637f331736db0462da14fc5e5ddfc3059cb57b82dc95887f78e835680',
    PARENT_CONTROLLER:'6361896ba05893c07baf26644895ee68539a9471c539b46f1cdd74dd0db91105',
    'tsc_uniform_majority_transition_v1.py':'173664a651ae1529f2981b7150acae402410ee8708ad96c98735e0d30fb03738'}
DECLARED_CHANGE={'source_LoRA_Adam_group_lr':3e-4,'next_applied_LoRA_group_lr':1e-4,'head_lr':1e-5,
    'TR_per_update':14,'replay_per_update':2,'preservation_coefficients':[40,2,4],
    'sampler_weights_Adam_moments_RNG_history_coverage_preserved':True}


def loralr_declaration(context):
    return {'kind':'declared_lower_encoder_LR_full_state_v1','source_step':context['source_step'],
        'quota_boundary_step':context['quota_boundary_step'],'uniform_origin_step':6500,
        'configured_end_step':context['configured_end_step'],'source_LoRA_lr':3e-4,'active_LoRA_lr':1e-4,
        'head_lr':1e-5,'preservation_coefficients':[40,2,4],'TR_per_update':14,'replay_per_update':2,
        'all207_Adam_moments_and_global_clocks_preserved':True,'fresh_optimizer_or_sampler':False,
        'renewed_Portuguese_grace':False,'parent_failure_counter_preserved':True,
        'all_six_required_for_every_BEST':True,'only_new_scientific_variable':'encoder_LoRA_group_LR'}


def validate_receipt(value,run_id):
    require(value.get('schema')==1 and value.get('kind')==RECEIPT_KIND
        and value.get('status')=='actual_CLOSED_reviewed_majority_LR_fork'
        and value.get('target_run_id')==run_id and value.get('source_run_id')!=run_id
        and value.get('chosen_checkpoint_role')=='latest' and value.get('chosen_step')==value.get('completed_steps')
        and value.get('allow_declared_lower_encoder_LR_continuation') is True
        and value.get('declared_change')==DECLARED_CHANGE,'Separate ROOT-declared latest full-state lowerLR fork required')
    return value


@lru_cache(maxsize=2)
def source_context(receipt_path,run_id=None):
    path=scoped(receipt_path);receipt=read(path);validate_receipt(receipt,run_id or receipt['target_run_id'])
    review=read(checked_record(receipt['ROOT_closed_retention_review']));a=receipt['artifacts'];original=receipt['original_artifact_records']
    for key,item in a.items():checked_record(item,checkpoint=key in ('chosen_checkpoint','source_best_checkpoint','original_champion'))
    for name,digest in PARENT_PINS.items():require(record(HERE/name)['sha256']==digest,'Frozen majority source changed: '+name)
    require(review.get('schema')==1 and review.get('kind')==REVIEW_KIND and review.get('passed') is True
        and review.get('source_run_id')==receipt['source_run_id'] and review.get('chosen_step')==receipt['chosen_step']
        and review.get('chosen_checkpoint_role')=='latest' and review.get('allow_declared_lower_encoder_LR_continuation') is True
        and review.get('declared_change')==DECLARED_CHANGE and review.get('renewed_Portuguese_grace') is False
        and review.get('parent_failure_counter_preserved') is True
        and review.get('chosen_frontier_is_qualified_BEST') is False,'Actual ROOT retention/closure/LR review required')
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
        and config['phase']==PHASE and config['trainable_parameter_tensors']==207
        and config['turkish_clips_used']==253798 and config['replay_clips_used']==1914
        and selection['heldout_tests_evaluated'] is False and selection['accepted'] is False and selection['selected_step'] is None,
        'Actually CLOSED latest unqualified majority whole-state boundary required')
    require(canonical(identity)==receipt['scientific_config_identity_sha256']
        and identity['extended_tokenizer_sha256']=='b91d500326aca290ed709d629dab2bee56f298ea7323b94855b9f5ef0caeac75'
        and recipe['head_lr']==1e-5 and recipe['lora_lr']==3e-4 and recipe['schedule']=='constant'
        and recipe['replay_fraction']==.125 and recipe['replay_kl']==40 and recipe['duration_kl']==2 and recipe['feature_distillation']==4
        and recipe['batch_size']==8 and recipe['accumulation']==2,'Exact source207/8199/14-2/40-2-4 science required')
    for name,digest in identity['source_sha256'].items():require(record(HERE/name)['sha256']==digest,'Actual source pin changed: '+name)
    require(scoped(original['chosen_checkpoint']['path'])==scoped(selection['latest_weights']), 'Chosen latest pointer changed')
    prefix,last=_history_frontier(a['training_history']['path'],h,h)
    require(prefix==receipt['source_history_prefix_sha256'] and last['selection_diagnostics']['consecutive_retention_failures']==review['parent_retention_failure_counter']
        and selection['consecutive_retention_failures']==review['parent_retention_failure_counter'],'Exact inherited history/failure counter required')
    old_record=receipt['original_majority_source_receipt'];checked_record(old_record)
    require(old_record==review['original_majority_source_receipt'],'Original quota source receipt changed')
    origin=base.source_context(str(scoped(old_record['path'])),receipt['source_run_id'])
    quota=identity['domain_sampling_identity']['majority_quota_transition']
    require(quota['source_step']==origin['source_step'] and quota['prior_uniform_draws']==origin['prior_uniform_draws']
        and quota['configured_end_step']==recipe['steps']==selection['configured_steps']
        and config['uniform_origin_step']==6500,'Original immutable quota boundary/end/pass must not reset')
    coverage=last['Turkish_domain_coverage']['uniform_pass'];majority=last['Turkish_domain_coverage']['majority_transition']
    require(coverage['source_step']==6500 and coverage['draws']==coverage['unique_clips_seen']==review['actual_unique_uniform_clips']
        and coverage['repeat_visits']==0 and majority['local_updates']==h-quota['source_step']==review['actual_new_majority_updates'],
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
    active={**recipe,'lora_lr':1e-4}
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
        'original_majority_source_receipt':old_record,'combined_baseline':baseline}
    return context


def validate_portuguese_inputs(args,context):return base.validate_portuguese_inputs(args,context)


def preflight(args):
    context=source_context(str(scoped(args.majority_source_receipt)),args.run_id).copy()
    require(scoped(args.resume_majority)==scoped(context['source_checkpoint']['path']),'Chosen latest7750 whole-state input required')
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
    h=context['source_step'];require(state.get('schema')==2 and state.get('run_id')==context['receipt']['source_run_id']
        and state.get('step')==h and canonical(state.get('config_identity'))==canonical(context['parent_identity']),
        'Chosen own-majority7750 weights/Adam/runtime must be one frontier')
    require(len(state.get('history',[]))==h and canonical(state['history'])==context['source_history_prefix_sha256']
        and state['selected_step'] is None and state['best_score']==math.inf
        and state['retention_failures']==context['receipt']['parent_retention_failure_counter'], 'Inherited history/BEST/failure frontier differs')
    for key in ('weights','optimizer','sampler','lora_metadata','python_rng','torch_rng','cuda_rng','patience','next_eval_epoch',
        'train_wall_seconds','audio_seconds','language_audio_seconds','evaluation_wall_seconds','initial_gate','stop_reason'):
        require(key in state,'Complete source runtime required: '+key)
    sampler=state['sampler'];require(sampler.get('kind')=='TSC_uniform_majority_TR_six_replay_sampler_v1'
        and sampler['updates']==h and sampler['identity']==context['parent_identity']['domain_sampling_identity']
        and sampler['quota_transition']==context['quota_transition']
        and sampler['turkish_streams']['uniform']['draws']==context['prior_uniform_draws'], 'Own14/2 sampler frontier differs')
    expected={name:p for name,p in model.named_parameters() if p.requires_grad};torch=torch_module
    require(len(expected)==207 and set(expected)==set(state['weights']),'All207 named weights required')
    for name,value in state['weights'].items():require(value.shape==expected[name].shape and value.dtype==torch.float32
        and bool(torch.isfinite(value).all()),'Source FP32 shape/finiteness differs: '+name)
    from encoder_lora import encoder_lora_metadata
    require(encoder_lora_metadata(model)==state['lora_metadata'] and model.config.vocab_size==8199
        and model.config.blank_token_id==8192 and list(model.config.durations)==[0,1,2,3,4]
        and model.decoder.embedding.num_embeddings==8199 and model.joint.head.out_features==8204,'Original207/8199 geometry required')
    audit=base.audit_source_optimizer(state,{**context,'source_recipe':context['parent_recipe']},expected,torch_module=torch)
    require(record(context['source_checkpoint']['path'])==context['source_checkpoint'],'Chosen CLOSED7750 bytes changed')
    return {'passed':True,'source_step':h,'source_Adam_audit':audit,'all207_Adam_moments_and_global_clock_preserved':True,
        'only_next_live_LoRA_group_lr_changed':{'before':3e-4,'after':1e-4},'parent_failure_counter_preserved':state['retention_failures']}


def restore_parent_sampler(sampler,state,context):
    """Restore existing own-kind queues without loading another7621 checkpoint."""
    source=state['sampler'];identity=context['parent_identity']['domain_sampling_identity'];quota=context['quota_transition'];prefix=source['source_prefix']
    require(source['identity']==identity and source['quota_transition']==quota and source['updates']==context['source_step']
        and canonical(prefix)==quota['source_original_prefix_sha256'],'Same immutable quota origin/prefix required')
    expected=deepcopy(sampler.identity)
    expected.update(source_boundary_step=6500,source_domain_identity_sha256=canonical(prefix['source_identity']),
        source_sampler_sha256=prefix['source_sampler_sha256'],sampler_family=source['kind'],replay_per_update=2,
        majority_quota_transition=deepcopy(quota))
    require(expected==identity,'Original catalog/pool/seeds/grouping identity differs')
    sampler.source_prefix=deepcopy(prefix);sampler.quota_transition=deepcopy(quota);sampler.identity=deepcopy(identity)
    sampler.tr_per_update=14;sampler.replay_per_update=2;sampler.load_state_dict(source)
    require(sampler.updates==context['source_step'] and sampler.configured_end_step==context['configured_end_step'],
        'Loaded same-pass frontier/horizon changed')
    return {'kind':'actual_existing_majority_own_state_restore_v1','passed':True,'source_step':sampler.updates,
        'quota_boundary_step':quota['source_step'],'configured_end_step':sampler.configured_end_step,
        'queues_rng_seen_counters_and_quota_segments_preserved':True,'new_migration_or_shuffle':False}


def release_admission_caches():
    source_context.cache_clear();base.release_admission_caches()
    return {'lowerLR_and_original_source_caches_cleared':True,'disk_or_other_process_changed':False}
