"""Matched100-update IO control from coherent CLOSED latest25109.

Counter23 and separate qualified BEST19750 are carried unchanged. No replay
selector migration, rewind, renewed grace, tensor import or optimizer reset.
"""
from __future__ import annotations
from copy import deepcopy
from functools import lru_cache
import math

import tsc_uniform_full_epoch_transition_v1 as parent
from tsc_uniform_retention_constraint_transition_v1 import (
    HERE,ROOT,LANGUAGES,require,read,scoped,record,checked_record,canonical,_history_frontier,
    proof_path,retention_gates,compare_source_development)
from tsc_uniform_majority_transition_v1 import audit_source_optimizer

RECEIPT_KIND='ROOT_owned_closed_TSC_uniform_IO_control_v1'
REVIEW_KIND='ROOT_USER_TSC_uniform_IO_control_review_v1'
PARENT_WORKER='train_lora_batched_tsc_uniform_full_epoch_v1.py'
PARENT_CONTROLLER='run_TSC_uniform_full_epoch_v1.py'
PARENT_PINS={PARENT_WORKER:'1c33cbde6933c55d0839d0ebf82253fb3b673dde7f5a105adc3905e14f7013c3',
    PARENT_CONTROLLER:'1943266496b654bc3439c0f667f8a197bb620a0c1d110d5bd9269cb26910d885',
    'tsc_uniform_full_epoch_transition_v1.py':'7cf6aaa78ef3372353cd3320aa606add08927dbecca17a3e91dfd482474b3ce3'}
PERSISTENCE_MODES=('double_json','single_serialize_two_commits')


def io_declaration(context,persistence_mode):
    require(persistence_mode in PERSISTENCE_MODES and persistence_mode==context['receipt']['persistence_mode'],
        'Exact reviewed persistence mode required')
    return {'persistence_mode':persistence_mode,'source_step':25109,'technical_updates':100,'technical_end_step':25209,
        'source_lineage_end_step':25109,'source_actual_retention_failures':23,'source_actual_selected_step':19750,
        'source_uniform_unique_clips':253798,'source_uniform_draws':253800,'source_repeat_visits':2,
        'all207_weights_Adam_sampler_RNG_history_preserved':True,'training_math_LRs_teacher_and_inputs_unchanged':True,
        'paired_publication_frequency_schema_and_commit_order_unchanged':True,'strict_all_six_champion_gates_unchanged':True,
        'resource_nonfinite_protections_unchanged':True,'retention_stop_policy':'monitor_only',
        'ROOT_review':context['receipt']['ROOT_review'],'coverage_rewind_clips':0,'sampler_queues_or_RNG_reset':False,
        'technical_test_is_quality_recipe_screen_or_new_full_pass':False}


def ptfocus_declaration(context):return context['config']['PT_replay_policy_transition']


def full_epoch_declaration(context):return context['config']['full_epoch_policy_transition']['declared']


def validate_receipt(value,run_id):
    require(value.get('schema')==1 and value.get('kind')==RECEIPT_KIND
        and value.get('status')=='actual_CLOSED_matched_IO_control'
        and value.get('target_run_id')==run_id and value.get('source_run_id')!=run_id
        and value.get('chosen_step')==value.get('completed_steps')==25109 and value.get('chosen_checkpoint_role')=='latest'
        and value.get('explicit_coverage_rewind_clips')==0 and value.get('source_counter_preserved')==23
        and value.get('persistence_mode') in PERSISTENCE_MODES,
        'Explicit coherent latest25109/counter23/no-rewind fork required')
    return value


@lru_cache(maxsize=2)
def source_context(receipt_path,run_id=None):
    path=scoped(receipt_path);receipt=read(path);validate_receipt(receipt,run_id or receipt['target_run_id'])
    review=read(checked_record(receipt['ROOT_review']));a=receipt['artifacts'];original=receipt['original_artifact_records']
    require(review.get('kind')==REVIEW_KIND and review.get('passed') is True
        and review.get('source_run_id')==receipt['source_run_id'] and review.get('chosen_step')==review.get('completed_steps')==25109
        and review.get('chosen_checkpoint_role')=='latest' and review.get('allow_matched_IO_control') is True
        and review.get('source_counter_preserved')==23 and review.get('source_selected_BEST_preserved')==19750
        and review.get('source_unqualified_latest_used') is True and review.get('no_hybrid_training_state_used') is True
        and review.get('explicit_coverage_rewind_clips')==0 and review.get('renewed_Portuguese_grace') is False
        and review['declared_change'].get('same_starting_all207_state_Adam_sampler_RNG_history') is True
        and review['declared_change'].get('same_future_Turkish_and_replay_examples') is True
        and review.get('technical_updates')==100 and review.get('technical_end_step')==25209,
        'Actual USER-directed latest25109 full-epoch review required')
    change=review['declared_change']
    require(change.get('parameter')=='paired_history_persistence' and change.get('choices')==list(PERSISTENCE_MODES)
        and change.get('additional_updates')==100 and change.get('training_math_and_scoring_unchanged') is True
        and change.get('same_format_frequency_two_targets_atomic_fsync') is True
        and change.get('technical_benchmark_not_a_quality_repair_screen') is True
        and change.get('strict_all_six_winner_gates_unchanged') is True and change.get('retention_stop_policy')=='monitor_only',
        'Only matched persistence implementation and explicit100-update execution extension may change')
    closed=read(checked_record(review['actual_closed_source_review']))
    require(review['source_actual_owned_exit']=={'session_id':67197,'exit_code':0,'tool_chunk':'90d47e'}
        and closed['completed_steps']==25109 and closed['source_run_id']==receipt['source_run_id']
        and closed.get('passed') is True and closed.get('full_pass_complete') is True, 'Actual source CLOSED0 required')
    checkpoint_keys=('chosen_checkpoint','latest_checkpoint','source_best_checkpoint','original_champion')
    for key,item in a.items():checked_record(item,checkpoint=key in checkpoint_keys)
    for key,item in review['artifacts'].items():require(original[key]==item,'Actual reviewed source binding changed: '+key)
    for name,digest in PARENT_PINS.items():require(record(HERE/name)['sha256']==digest,'Frozen parent changed: '+name)
    config=read(a['run_config']['path']);selection=read(a['selection']['path']);controller=read(a['controller_status']['path']);identity=config['config_identity']
    require(config['run_id']==selection['run_id']==controller['run_id']==receipt['source_run_id']
        and controller['status']=='train_window_ended' and controller['child_exit_code']==0 and controller['resource_review_required'] is False
        and controller['completed_steps']==selection['completed_steps']==25109 and selection['accepted'] is True
        and selection['selected_step']==19750 and selection['consecutive_retention_failures']==23 and selection['heldout_tests_evaluated'] is False
        and scoped(original['chosen_checkpoint']['path'])==scoped(selection['latest_weights'])
        and scoped(original['source_best_checkpoint']['path'])==scoped(selection['weights'])
        and original['chosen_checkpoint']==original['latest_checkpoint'],
        'Coherent unqualified latest25109 and separate qualified BEST19750 required')
    require(canonical(identity)==receipt['scientific_config_identity_sha256'],'Parent identity changed')
    for name,digest in identity['source_sha256'].items():require(record(HERE/name)['sha256']==digest,'Parent dependency changed: '+name)
    prefix,last=_history_frontier(a['training_history']['path'],25109,25109)
    require(prefix==receipt['source_history_prefix_sha256'],'Latest25109 history prefix changed')
    origin_record=receipt['original_full_epoch_source_receipt'];checked_record(origin_record)
    require(scoped(origin_record['path'])==scoped(config['majority_source_receipt']),'Original feature source lineage changed')
    origin=parent.source_context(str(scoped(origin_record['path'])),receipt['source_run_id'])
    recipe=dict(identity['training_arguments']);require(recipe==origin['source_recipe'] and recipe['feature_distillation']==20
        and recipe['replay_kl']==40 and recipe['duration_kl']==2 and recipe['lora_lr']==1e-4 and recipe['portuguese_token_kl_multiplier']==1 and recipe['retention_stop_policy']=='monitor_only' and recipe['steps']==25109,
        'Actual parent feature20 recipe and optimizer LR must remain unchanged')
    summary=read(a['chosen_DEV_summary']['path']);best_summary=read(a['source_best_DEV_summary']['path'])
    baseline=read(a['Portuguese_baseline_proof']['path'])['baseline_overlay'];gates=retention_gates(summary,baseline);best_gates=retention_gates(best_summary,baseline)
    require(not gates['es']['passed'] and gates['es']['word_errors']==53 and gates['es']['allowed_word_errors']==52
        and all(gates[lang]['passed'] for lang in ('en','it','fr','de','pt'))
        and all(v['passed'] for v in best_gates.values())
        and best_summary['validation_tr']['asr']['wer']==selection['turkish_validation_wer'],
        'Unqualified latest25109 and separate all-six BEST19750 must remain truthful')
    require(last['validation']['turkish_wer']==summary['validation_tr']['asr']['wer']
        and last['validation']['retention_gate_passed'] is False
        and last['validation']['retention_language_passed']=={lang:value['passed'] for lang,value in gates.items()}
        and last['selection_diagnostics']['consecutive_retention_failures']==23,'Latest history/DEV/counter frontier differs')
    sampler_identity=identity['domain_sampling_identity'];quota=sampler_identity['majority_quota_transition'];focus=sampler_identity['PT_focus_transition']
    require(quota==origin['quota_transition'] and focus['source_step']==8000 and quota['source_step']==7621
        and quota['configured_end_step']==25109 and selection['Turkish_domain_coverage']['uniform_pass']['unique_clips_seen']==253798
        and selection['Turkish_domain_coverage']['uniform_pass']['draws']==253800
        and selection['Turkish_domain_coverage']['uniform_pass']['repeat_visits']==2
        and selection['Turkish_domain_coverage']['uniform_pass']['full_pass_complete'] is True
        and identity['retention_numeric_correction']==origin['parent_identity']['retention_numeric_correction'],
        'Same pass/quota/PT/numeric lineage must remain immutable')
    done=read(a['runtime_completion']['path']);owned=read(a['owned_worker_identity']['path'])
    require(done['status']=='complete' and done['sources_unchanged'] is True and done['pid']==owned['pid']
        and done['actual_dispatch_source_sha256']==PARENT_PINS[PARENT_CONTROLLER],'Actual parent runtime closure required')
    for key in ('memory','cache','io'):
        value=read(a[key]['path']);require(value['status']=='complete' and value['sources_unchanged'] is True,'Actual runtime incomplete: '+key)
    context={**origin,'receipt':receipt,'receipt_record':record(path),'source_record':record(path),'io_review':review,
        'config':config,'selection':selection,'parent_identity':identity,'parent_recipe':recipe,
        'source_recipe':{**recipe,'steps':25209,'persistence_mode':receipt['persistence_mode']},
        'source_step':25109,'source_selected_step':25109,'source_checkpoint':a['chosen_checkpoint'],'source_summary':summary,
        'source_history_prefix_sha256':prefix,'source_frontier_row':last,'source_retention_gates':gates,
        'source_actual_selected_step':19750,'source_actual_failure_counter':23,'corrected_initial_retention_failures':23,
        'corrected_initial_best_score':best_summary['validation_tr']['asr']['wer'],'corrected_initial_selected_step':19750,
        'quota_transition':quota,'indexed_audit':read(a['indexed_TR_audit']['path']),'indexed_audit_record':a['indexed_TR_audit'],
        'source_best_checkpoint':a['source_best_checkpoint'],'source_best_step':19750,'source_best_summary':best_summary,
        'original_source350':a['initial_source_equivalence'],'original_native_ready':a['original_native_ready'],
        'uniform_source_receipt':a['uniform_source_receipt'],'portuguese_preparation_proof':a['Portuguese_preparation_proof'],
        'portuguese_baseline_proof':a['Portuguese_baseline_proof'],'parked_original_champion':a['original_champion'],
        'combined_baseline':baseline,'prior_uniform_draws':253800,'remaining_updates_from_chosen_frontier':100,
        'source_lineage_end_step':25109,'technical_end_step':25209,'technical_updates':100,'technical_extension_updates':100,
        'configured_end_step':25209,'uniform_origin_step':6500,'quota_boundary_step':7621,'lora_boundary_step':7750,
        'PT_focus_boundary_step':8000,'focus_origin_step':8000}
    return context


def validate_portuguese_inputs(args,context):return parent.validate_portuguese_inputs(args,context)


def preflight(args):
    context=source_context(str(scoped(args.majority_source_receipt)),args.run_id).copy()
    require(type(args.persistence_benchmark_updates) is int and args.persistence_benchmark_updates==100,
        'Only declared100-update technical control admitted')
    require(scoped(args.resume_majority)==scoped(context['source_checkpoint']['path']),'Only coherent latest25109 may initialize')
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
        and state.get('step')==25109 and canonical(state.get('config_identity'))==canonical(context['parent_identity'])
        and state['selected_step']==19750 and state['best_score']==context['corrected_initial_best_score'] and state['retention_failures']==23,
        'Actual coherent latest25109/counter23/BEST19750 must remain intact')
    require(len(state['history'])==25109 and canonical(state['history'])==context['source_history_prefix_sha256']
        and state['sampler']['updates']==25109 and state['sampler']['identity']==context['parent_identity']['domain_sampling_identity'],
        'Latest25109 weights/moments/history/queues must share one frontier')
    for key in ('optimizer','lora_metadata','python_rng','torch_rng','cuda_rng','patience','next_eval_epoch','train_wall_seconds',
        'audio_seconds','language_audio_seconds','evaluation_wall_seconds','initial_gate','stop_reason'):
        require(key in state,'Incomplete coherent source runtime: '+key)
    expected={name:p for name,p in model.named_parameters() if p.requires_grad};require(len(expected)==207 and set(expected)==set(state['weights']),'All207 source weights required')
    for name,value in state['weights'].items():require(value.shape==expected[name].shape and value.dtype==torch.float32
        and bool(torch.isfinite(value).all()),'FP32 source shape/finiteness differs: '+name)
    from encoder_lora import encoder_lora_metadata
    require(encoder_lora_metadata(model)==state['lora_metadata'] and model.config.vocab_size==8199
        and model.config.blank_token_id==8192 and list(model.config.durations)==[0,1,2,3,4], 'Original207/8199 geometry required')
    audit=audit_source_optimizer(state,{**context,'source_recipe':context['parent_recipe']},expected,torch_module=torch)
    require(record(context['source_checkpoint']['path'])==context['source_checkpoint'],'Chosen coherent latest25109 bytes changed')
    return {'passed':True,'source_Adam_audit':audit,'all207_Adam_sampler_RNG_history_preserved':True,
        'source_actual_selected_step':19750,'source_actual_failure_counter':23,'training_math_LRs_unchanged':True,
        'retention_stop_policy':'monitor_only',
        'coverage_rewind_clips':0,'hybrid_or_BEST_state_used_for_weights':False}


def restore_parent_sampler(sampler,state,context):
    saved=state['sampler'];identity=context['parent_identity']['domain_sampling_identity'];prefix=saved['source_prefix'];quota=saved['quota_transition'];focus=saved['PT_focus_transition']
    require(saved['identity']==identity and saved['updates']==context['source_step'] and quota==context['quota_transition']
        and focus==identity['PT_focus_transition'] and focus['source_step']==8000
        and canonical(prefix)==quota['source_original_prefix_sha256'],'Immutable selector/quota/source prefix required')
    parent_identity=deepcopy(identity);parent_identity.pop('PT_focus_transition');parent_identity['sampler_family']='TSC_uniform_majority_TR_six_replay_sampler_v1'
    expected=deepcopy(sampler.identity)
    expected.update(source_boundary_step=6500,source_domain_identity_sha256=canonical(prefix['source_identity']),
        source_sampler_sha256=prefix['source_sampler_sha256'],sampler_family=parent_identity['sampler_family'],replay_per_update=2,
        majority_quota_transition=deepcopy(quota))
    require(expected==parent_identity,'Original catalog/pool/seeds/layout identity differs')
    sampler.source_prefix=deepcopy(prefix);sampler.quota_transition=deepcopy(quota);sampler.parent_majority_identity=parent_identity
    sampler.PT_focus_transition=deepcopy(focus);sampler.identity=deepcopy(identity);sampler.tr_per_update=14;sampler.replay_per_update=2
    sampler.load_state_dict(saved)
    require(sampler.updates==context['source_step'] and sampler.configured_end_step==25109,'Same own25109 queues/clock/end required')
    return {'kind':'actual_PT_focus_own25109_sampler_restore_v1','passed':True,'source_step':context['source_step'],
        'focus_origin_step':8000,'quota_boundary_step':7621,'no_new_selector_migration_or_shuffle':True}


def release_admission_caches():
    source_context.cache_clear();parent.release_admission_caches()
    return {'IO_control_and_original_source_caches_cleared':True,'disk_or_other_process_changed':False}
