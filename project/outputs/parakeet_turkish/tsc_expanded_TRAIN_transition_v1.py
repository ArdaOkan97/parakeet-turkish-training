"""Matched base/expanded TRAIN arms from the coherent qualified40750 full state.

This module reads metadata only. Actual207 weights/Adam/RNG/history admission and
source350 equality execute in the guarded native worker before its first update.
"""
from __future__ import annotations
from functools import lru_cache
import math

from tsc_uniform_retention_constraint_transition_v1 import (
    HERE,ROOT,LANGUAGES,require,read,scoped,record,checked_record,canonical,
    _history_frontier,proof_path,retention_gates,compare_source_development)
from tsc_uniform_majority_transition_v1 import audit_source_optimizer

RECEIPT_KIND='ROOT_owned_closed_TSC_expanded_TRAIN_v1'
PHASE='HF_TSC_8199_matched_expanded_TRAIN_v1'
SOURCE_STEP=40750
COMPLETED_SOURCE_STEP=43238
UPDATES=25198
END_STEP=65948
POPULATION_MEAN_U=26.642061797177284
DATA_ARMS=('base','expanded')
PRESERVATION_SHA='62af8a89873169313a7e9f4413195b4be489410ce51ad4cfc30f6b37514303fb'
POLICY_SHA='8173087bc993c692dd21d609d7626c13a2f376a651f95dd932c1eee408dac204'
PREPARATION_SHA='8aac435e3472d3e304a88c4ac952756f1a01682682ef1d8cc82beedbc694e9e7'
VERIFICATION_SHA='f4e1fe9b487d1a8fc8da25e9ed0faf9f8ff35cf293bcc87ace4b084fa0b55e3f'
SANITY_SHA='19b1469269dfca2c428c5da06e28f4c1fb7088f793050e9160982262ff7264b3'
INDEX_PROOF_SHA='2760e27504abad94bd64a8ac5a386bd311eb02fea0b43beee8f624fb3d9425b5'
EXPANDED_HELPER_SHA='aaf3b188669c05ab0ff67e2be004b41f2cc21a81c8be368f58e7d501800c2370'


def source_encoder_lr():
    local=SOURCE_STEP-25109
    fraction=(local-1)/(18129-1)
    return 3e-5+(1e-4-3e-5)*.5*(1+math.cos(math.pi*fraction))


def data_expansion_declaration(context,data_arm):
    require(data_arm in DATA_ARMS and data_arm==context['receipt']['data_arm'],'Hash-bound data arm required')
    pool=253798 if data_arm=='base' else 352768
    return {'kind':'matched_original8199_TRAIN_pool_expansion_v1','data_arm':data_arm,
        'source_step':SOURCE_STEP,'source_completed_step':COMPLETED_SOURCE_STEP,
        'source_checkpoint_role':'best','source_retention_failures':0,'source_selected_step':SOURCE_STEP,
        'additional_updates':UPDATES,'configured_end_step':END_STEP,'active_TR_pool_clips':pool,
        'base_TR_pool_clips':253798,'added_WorldSpeech_clips':98970 if data_arm=='expanded' else 0,
        'TR_per_update':14,'PT_per_update':1,'old_five_replay_per_update':1,'batch_size':8,'accumulation':2,
        'new_uniform_coverage_epoch_local_zero':True,'TR_rng_copied_before_new_queue':True,
        'prior_TR_queue_cursor_seen_parked_as_lineage':True,'six_replay_queues_cursors_RNG_preserved':True,
        'evaluation_epoch_is_inherited_source_epoch_plus_local_draws_over_active_pool':True,
        'source_next_eval_epoch_preserved':True,'fixed_DEV_every_updates':250,
        'plateau_target_and_retention_early_stop_disabled_for_entire_matched_budget':True,
        'source_patience_and_failure_counters_preserved_without_renewed_grace':True,
        'all207_weights_Adam_moments_global_clocks_history_RNG_preserved':True,
        'source_encoder_LR_audited_before_new_cycle_schedule':source_encoder_lr(),
        'encoder_lr_schedule':'cosine','encoder_peak_lr':1e-4,'encoder_min_lr':3e-5,
        'encoder_schedule_updates':UPDATES,'head_lr':1e-5,'warmup_or_Adam_clock_reset':False,
        'tr_tdt_weighting':'token_population','fixed_population_mean_U':POPULATION_MEAN_U,
        'denominator_reestimated_for_expansion':False,'normalization_population_is_frozen_original253798_not_active_arm':True,
        'expanded_population_mean_TR_weight_claimed_one':False,'replay_coefficients':{'pt':[40.,2.,20.],
            'old_five':[40*5/3,2*5/3,20*5/3]},'retention_stop_policy':'monitor_only',
        'strict_integer_all_six_champion_selection':True,'TEST_used':False,
        'matched_optimizer_replay_budget':True,'matched_audio_or_token_workload_claimed':False,
        'expanded_full_pass_tail_draws':4,'control_may_repeat_base_pool':data_arm=='base',
        'actual_unique_coverage_required':True,'failed_final43238_branch_preserved':True,
        'normalization_policy':context['receipt']['normalization_policy'],
        'candidate_approval':context['receipt']['candidate_approval']}


def validate_receipt(value,run_id):
    require(value.get('schema')==1 and value.get('kind')==RECEIPT_KIND
        and value.get('status')=='actual_CLOSED_qualified_BEST40750_matched_data_arms'
        and value.get('target_run_id')==run_id and value.get('source_run_id')!=run_id
        and value.get('chosen_step')==SOURCE_STEP and value.get('completed_steps')==COMPLETED_SOURCE_STEP
        and value.get('chosen_checkpoint_role')=='best' and value.get('data_arm') in DATA_ARMS
        and value.get('source_counter_preserved')==0 and value.get('additional_updates')==UPDATES
        and value.get('configured_end_step')==END_STEP,'Qualified coherent40750 and explicit matched data arm required')
    return value


def validate_closed_inputs(preservation_record,policy_record,index_proof_record):
    closed=read(checked_record(preservation_record));policy=read(checked_record(policy_record))
    require(preservation_record['sha256']==PRESERVATION_SHA and policy_record['sha256']==POLICY_SHA,
        'Exact actual completed normalization source and policy required')
    require(closed['passed'] is True and closed['completed_step']==COMPLETED_SOURCE_STEP
        and closed['selected_qualified_BEST_step']==SOURCE_STEP and closed['native_child_exit_code']==0
        and closed['native_sources_unchanged'] is True
        and closed['actual_owned_exit']=={'session_id':15227,'exit_code':0,'tool_chunk':'cb3509'},'Actual owned CLOSED0 normalization source required')
    require(policy['passed'] is True and policy['actual_curve_evidence_used'] is True
        and policy['both_matched_cycles_complete'] is True and policy['selected_TR_tdt_weighting']=='token_population'
        and policy['selected_frozen_population_mean_U']==POPULATION_MEAN_U and policy['selected_head_lr']==1e-5
        and policy['selected_encoder_policy']=='cosine'
        and policy['selected_coherent_checkpoint']==closed['artifacts']['best_checkpoint']['archived'],
        'Selected qualified normalization source/denominator changed')
    index=read(checked_record(index_proof_record))
    require(index_proof_record['sha256']==INDEX_PROOF_SHA and index['passed'] is True
        and index['rows']==352768 and index['source_group_counts']=={'cv':72846,'fleurs':2037,'tsc':178915,'worldspeech':98970}
        and index['source_data_eligible_for_controlled_trial'] is True and index['original_source_rows_unchanged'] is True
        and index['helper']['sha256']==EXPANDED_HELPER_SHA,'Completed exact expanded indexed carrier required')
    checked_record(index['index_provenance']);checked_record(index['helper'])
    return closed,policy,index


@lru_cache(maxsize=2)
def source_context(receipt_path,run_id=None):
    path=scoped(receipt_path);receipt=read(path);validate_receipt(receipt,run_id or receipt['target_run_id'])
    closed,policy,index=validate_closed_inputs(receipt['source_preservation'],receipt['normalization_policy'],receipt['expanded_index_proof'])
    a=receipt['artifacts'];original=receipt['original_artifact_records']
    for key,item in a.items():
        checked_record(item,checkpoint=key in ('chosen_checkpoint','latest_checkpoint','source_best_checkpoint','original_champion'))
    require(a['chosen_checkpoint']==a['source_best_checkpoint']==closed['artifacts']['best_checkpoint']['archived']
        and a['latest_checkpoint']==closed['artifacts']['latest_checkpoint']['archived'],'Coherent BEST archive differs; no final/BEST mixture')
    for key in ('run_config','training_history','selection','controller_status','runtime_completion','owned_worker_identity'):
        require(a[key]==closed['artifacts'][key]['archived'],'Closed source artifact differs: '+key)
    config=read(a['run_config']['path']);selection=read(a['selection']['path']);status=read(a['controller_status']['path'])
    identity=config['config_identity'];recipe=dict(identity['training_arguments'])
    require(config['run_id']==selection['run_id']==status['run_id']==receipt['source_run_id']==closed['run_id']
        and status['status']=='train_window_ended' and status['child_exit_code']==0
        and status['resource_review_required'] is False and status['completed_steps']==COMPLETED_SOURCE_STEP
        and selection['selected_step']==SOURCE_STEP and selection['completed_steps']==COMPLETED_SOURCE_STEP
        and selection['heldout_tests_evaluated'] is False and canonical(identity)==receipt['scientific_config_identity_sha256'],
        'Actual source runtime/config/selected checkpoint mismatch')
    require(scoped(selection['weights'])==scoped(closed['artifacts']['best_checkpoint']['original']['path'])
        and original['chosen_checkpoint']==closed['artifacts']['best_checkpoint']['original'], 'Chosen ownBEST pointer changed')
    require(config['tr_tdt_weighting']=='token_population' and config['tr_tdt_population_mean_U']==POPULATION_MEAN_U
        and config['TR_loss_normalization']['population_mean_U']==POPULATION_MEAN_U
        and recipe['head_lr']==1e-5 and recipe['lora_lr']==1e-4 and recipe['encoder_lr_schedule']=='cosine'
        and recipe['encoder_schedule_updates']==18129 and recipe['steps']==43238
        and recipe['batch_size']==8 and recipe['accumulation']==2 and recipe['replay_fraction']==.125
        and (recipe['replay_kl'],recipe['duration_kl'],recipe['feature_distillation'])==(40.,2.,20.)
        and recipe['portuguese_token_kl_multiplier']==1 and recipe['retention_stop_policy']=='monitor_only',
        'Original8199 selected recipe/LRs/normalization changed')
    for name,digest in identity['source_sha256'].items():require(record(HERE/name)['sha256']==digest,'Parent source changed: '+name)
    require(record(HERE/'expanded_training_rows_v1.py')['sha256']==EXPANDED_HELPER_SHA,'Expanded carrier source changed')
    prefix,last=_history_frontier(a['training_history']['path'],SOURCE_STEP,COMPLETED_SOURCE_STEP)
    require(prefix==receipt['source_history_prefix_sha256'],'Chosen40750 history prefix changed')
    require(last['head_lr']==1e-5 and last['lora_lr']==source_encoder_lr(),'Actual saved40750 optimizer rates differ from source cosine recipe')
    summary=read(a['chosen_DEV_summary']['path']);baseline=read(a['Portuguese_baseline_proof']['path'])['baseline_overlay']
    gates=retention_gates(summary,baseline)
    require(all(v['passed'] for v in gates.values()) and summary['validation_tr']['asr']['word_errors']==277
        and summary['validation_tr']['asr']['reference_words']==825
        and last['validation']['retention_gate_passed'] is True
        and last['validation']['turkish_wer']==summary['validation_tr']['asr']['wer']
        and last['selection_diagnostics']['consecutive_retention_failures']==0,'Source40750 must be the genuine all-six-qualified BEST')
    done=read(a['runtime_completion']['path']);owned=read(a['owned_worker_identity']['path'])
    require(done['status']=='complete' and done['sources_unchanged'] is True and done['pid']==owned['pid'], 'Actual source native closure required')
    approval=read(checked_record(receipt['candidate_approval']));verification=read(checked_record(receipt['candidate_verification']))
    preparation=read(checked_record(receipt['candidate_preparation']))
    require(receipt['candidate_approval']['sha256']==SANITY_SHA and receipt['candidate_verification']['sha256']==VERIFICATION_SHA
        and receipt['candidate_preparation']['sha256']==PREPARATION_SHA and preparation['status']=='complete'
        and verification['passed'] is True and verification['new_candidate_clips_after_exact_audio_exclusion']==98970
        and approval['approved_for_controlled_training_admission'] is True and approval['source_checkpoint']==a['chosen_checkpoint']
        and approval['optimizer_updates']==0 and approval['source_refs_unchanged'] is True,'Exact preparation/verification/actual TRAIN sanity required')
    active={**recipe,'steps':END_STEP,'encoder_schedule_updates':UPDATES,'tr_tdt_weighting':'token_population',
        'tr_tdt_population_mean_U':POPULATION_MEAN_U,'data_arm':receipt['data_arm']}
    parent_recipe={**recipe,'lora_lr':source_encoder_lr()}
    context={'receipt':receipt,'receipt_record':record(path),'source_record':record(path),'config':config,'selection':selection,
        'parent_identity':identity,'parent_recipe':parent_recipe,'source_recipe':active,'source_step':SOURCE_STEP,
        'source_selected_step':SOURCE_STEP,'completed_source_step':COMPLETED_SOURCE_STEP,'source_checkpoint':a['chosen_checkpoint'],
        'source_summary':summary,'source_best_summary':summary,'source_best_step':SOURCE_STEP,'source_best_checkpoint':a['chosen_checkpoint'],
        'source_history_prefix_sha256':prefix,'source_frontier_row':last,'source_retention_gates':gates,
        'source_actual_selected_step':SOURCE_STEP,'source_actual_failure_counter':0,'corrected_initial_retention_failures':0,
        'corrected_initial_selected_step':SOURCE_STEP,'corrected_initial_best_score':summary['validation_tr']['asr']['wer'],
        'source_encoder_lr':source_encoder_lr(),'parked_source_sampler_identity':identity['domain_sampling_identity'],
        'combined_baseline':baseline,'configured_end_step':END_STEP,'data_expansion_updates':UPDATES,'encoder_schedule_updates':UPDATES,
        'execution_extension_updates':UPDATES,'remaining_updates_from_chosen_frontier':UPDATES,'data_arm':receipt['data_arm'],
        'TR_pool_clips':253798 if receipt['data_arm']=='base' else 352768,'uniform_origin_step':6500,'quota_boundary_step':7621,
        'lora_boundary_step':7750,'PT_focus_boundary_step':8000,'focus_origin_step':8000,'source_lineage_end_step':43238,
        'quota_transition':identity['domain_sampling_identity']['majority_quota_transition'],
        'indexed_audit':read(a['indexed_TR_audit']['path']),'indexed_audit_record':a['indexed_TR_audit'],
        'expanded_index_proof':index,'expanded_index_proof_record':receipt['expanded_index_proof'],
        'expanded_index_directory':str(scoped(index['index_provenance']['path']).parent),
        'original_source350':a['initial_source_equivalence'],'original_native_ready':a['original_native_ready'],
        'original_index_directory':config['indexed_tr_directory'],
        'uniform_source_receipt':a['uniform_source_receipt'],'portuguese_preparation_proof':a['Portuguese_preparation_proof'],
        'portuguese_baseline_proof':a['Portuguese_baseline_proof'],'parked_original_champion':a['original_champion']}
    return context


def validate_portuguese_inputs(args,context):
    for key,record_key in (('portuguese_preparation_proof','Portuguese_preparation_proof'),('portuguese_baseline_proof','Portuguese_baseline_proof')):
        item=context['receipt']['artifacts'][record_key];checked_record(item)
        require(scoped(getattr(args,key))==scoped(context['config'][key]),'Original Portuguese input path changed: '+key)
    require(context['config']['portuguese_retention_baseline_proof']==context['portuguese_baseline_proof'], 'Original Portuguese baseline binding changed')
    return context


def preflight(args):
    context=source_context(str(scoped(args.data_expansion_source_receipt)),args.run_id).copy()
    require(scoped(args.resume_data_expansion)==scoped(context['source_checkpoint']['path']),'Only coherent chosen BEST40750 may initialize')
    for key,value in context['source_recipe'].items():require(getattr(args,key)==value,'Additional undeclared science change: '+key)
    for key in ('tsc_input_proof','tsc_training_approval','source_closure','tr_manifest','replay_manifest','data_v3_proof',
        'tr_frame_metadata','tr_frame_metadata_proof','resume_uniform_full','uniform_full_source_receipt',
        'resume_majority','majority_source_receipt','indexed_tr_directory',
        'portuguese_replay_manifest','portuguese_dev_manifest','portuguese_preparation_proof','portuguese_baseline_proof'):
        require(scoped(getattr(args,key))==scoped(context['config'][key]),'Original immutable input path changed: '+key)
    require(args.portuguese_seed==context['config']['portuguese_seed'] and not getattr(args,'resume_weights',None)
        and not getattr(args,'resume_optimization',None) and not getattr(args,'resume_runtime_migration',None),'No source/warmstart/RPseed substitution')
    require(args.native_max_joint_bytes==context['config']['native_max_joint_bytes']==1_700_000_000,
        'Keep actual original1.7GB native lattice bound')
    validate_portuguese_inputs(args,context);return context


def audit_parent_state(model,state,context,*,torch_module):
    torch=torch_module
    require(state.get('schema')==2 and state.get('run_id')==context['receipt']['source_run_id'] and state.get('step')==SOURCE_STEP
        and canonical(state.get('config_identity'))==canonical(context['parent_identity'])
        and state.get('selected_step')==SOURCE_STEP and state.get('best_score')==context['corrected_initial_best_score']
        and state.get('retention_failures')==0,'One coherent qualified40750 weights/moments/runtime required')
    require(len(state['history'])==SOURCE_STEP and canonical(state['history'])==context['source_history_prefix_sha256']
        and state['sampler']['updates']==SOURCE_STEP
        and canonical(state['sampler']['identity'])==canonical(context['parent_identity']['domain_sampling_identity']), 'Source history/sampler frontier mismatch')
    for key in ('optimizer','lora_metadata','python_rng','torch_rng','cuda_rng','patience','next_eval_epoch','train_wall_seconds',
        'audio_seconds','language_audio_seconds','evaluation_wall_seconds','initial_gate','stop_reason'):
        require(key in state,'Incomplete40750 runtime: '+key)
    require(state['patience']==context['source_frontier_row']['selection_diagnostics']['plateau_epoch_evaluations'], 'Source patience cannot reset')
    expected={name:p for name,p in model.named_parameters() if p.requires_grad}
    require(len(expected)==207 and set(expected)==set(state['weights']),'All207 source weights required')
    for name,value in state['weights'].items():require(value.shape==expected[name].shape and value.dtype==torch.float32
        and bool(torch.isfinite(value).all()),'Source FP32 shape/finiteness differs: '+name)
    from encoder_lora import encoder_lora_metadata
    require(encoder_lora_metadata(model)==state['lora_metadata'] and model.config.vocab_size==8199
        and model.config.blank_token_id==8192 and list(model.config.durations)==[0,1,2,3,4]
        and model.decoder.embedding.num_embeddings==8199 and model.joint.head.out_features==8204,'Original8199 geometry required')
    audit=audit_source_optimizer(state,{**context,'source_recipe':context['parent_recipe']},expected,torch_module=torch)
    require(record(context['source_checkpoint']['path'])==context['source_checkpoint'],'Chosen coherent40750 checkpoint bytes changed')
    return {'passed':True,'source_step':SOURCE_STEP,'source_Adam_audit':audit,'all207_weights_moments_clocks_retained':True,
        'RNG_history_and_six_replay_frontiers_retained':True,'source_encoder_lr_audited':source_encoder_lr(),
        'new_TR_coverage_epoch_is_separately_declared':True}


def release_admission_caches():
    source_context.cache_clear()
    return {'data_expansion_admission_contexts_released':True,'disk_or_other_process_changed':False}
