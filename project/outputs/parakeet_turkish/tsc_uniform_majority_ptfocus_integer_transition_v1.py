"""Count-equivalent gate correction over one reviewed CLOSED PT-focus state.

No threshold relaxation, recipe reset or historical gate rewrite. Runtime
audits the original failure5/NoneBEST state; only the declared new accounting
projection uses failure0/eligible8250 after actual identical350 evaluation.
"""
from __future__ import annotations
from copy import deepcopy
from functools import lru_cache
import math
from types import FunctionType

import tsc_uniform_majority_transition_v1 as comparison
import tsc_uniform_majority_ptfocus_transition_v1 as parent
from tsc_uniform_majority_transition_v1 import HERE,ROOT,LANGUAGES,require,read,scoped,record,checked_record,canonical,_history_frontier,proof_path

RECEIPT_KIND='ROOT_owned_closed_TSC_PT_focus_integer_gate_continuation_v1'
REVIEW_KIND='ROOT_PT_focus8250_integer_gate_migration_review_v1'
PARENT_WORKER='train_lora_batched_tsc_uniform_majority_ptfocus_v2.py'
PARENT_CONTROLLER='run_TSC_uniform_majority_ptfocus_v2.py'
PARENT_PINS={PARENT_WORKER:'7c55665c838909c7728af794fb448e49ecc10f2df1ef98e25b9da0c9e3faf40d',
    PARENT_CONTROLLER:'7f596b9188c8d7a23768bdbece3230c4c2fcf24bae787cf01a7b0b6a11097908',
    'tsc_uniform_majority_ptfocus_transition_v1.py':'e3b0063832b31b62648b08de9acb30fb9bfd67d49345360b65ab9bc312336658'}


def retention_gates(summary,baseline):
    result={}
    for language in LANGUAGES:
        candidate=summary['validation_'+language]['asr'];base=baseline['validation_'+language]['asr'];n=base['reference_words'];e=base['word_errors']
        require(type(n) is int and n>0 and type(e) is int and e>=0 and candidate['utterances']==50
            and candidate['reference_words']==n and type(candidate['word_errors']) is int and candidate['word_errors']>=0
            and candidate['wer']==candidate['word_errors']/n and base['wer']==e/n,'Exact equal reference counts/ASR ratios required: '+language)
        if language!='tr':
            allowed=e+max(n//200,e//10)
            result[language]={'word_errors':candidate['word_errors'],'reference_words':n,'allowed_word_errors':allowed,
                'passed':candidate['word_errors']<=allowed}
    return result


def ptfocus_declaration(context):
    return {**parent.ptfocus_declaration(context),'numeric_gate_migration':context['review']['declared_state_accounting_migration'],
        'gate_arithmetic':'base_errors + max(reference_words//200,base_errors//10)',
        'thresholds_relaxed':False,'historical_false_gate_records_rewritten':False,
        'PT_focus_origin_step':context['PT_focus_boundary_step']}


def validate_receipt(value,run_id):
    require(value.get('schema')==1 and value.get('kind')==RECEIPT_KIND and value.get('status')=='actual_CLOSED_count_equivalent_gate_migration'
        and value.get('target_run_id')==run_id and value.get('source_run_id')!=run_id
        and value.get('source_step')==8250 and value.get('chosen_checkpoint_role')=='latest'
        and value.get('thresholds_relaxed') is False and value.get('allow_count_equivalent_numerical_gate_correction') is True,
        'One new namespace for reviewed8250 numeric accounting correction required')
    return value


@lru_cache(maxsize=2)
def source_context(receipt_path,run_id=None):
    receipt_path=scoped(receipt_path);receipt=read(receipt_path);validate_receipt(receipt,run_id or receipt['target_run_id'])
    review=read(checked_record(receipt['ROOT_review']));a=receipt['artifacts'];original=receipt['original_artifact_records']
    require(review.get('schema')==1 and review.get('kind')==REVIEW_KIND and review.get('passed') is True
        and review.get('source_step')==8250 and review.get('source_run_id')==receipt['source_run_id']
        and review.get('allow_count_equivalent_numerical_gate_correction') is True and review.get('thresholds_relaxed') is False
        and review.get('word_counts_or_historical_gate_records_rewritten') is False and review.get('no_new_grace_or_optimizer_reset') is True,
        'Actual ROOT count-equivalent correction review required')
    closure=read(checked_record(review['actual_delegated_closure_review']))
    require(closure['status']=='actual_CLOSED0_retention_numeric_boundary_review_required'
        and closure['actual_owned_exit']==review['actual_owned_exit'] and closure['actual_owned_exit']['exit_code']==0
        and closure['native_child_exit_code']==0 and all(v['absent'] is True for v in closure['actual_owned_PID_absence']),
        'Actual delegated owned closure/native0 required')
    for key,review_key in (('controller_status','source_controller_status'),('selection','source_selection'),
        ('chosen_DEV_summary','source_summary'),('runtime_completion','source_native_completion')):
        require(original[key]==review[review_key],'Actual reviewed metadata changed: '+key)
    chosen=a['chosen_checkpoint'];unhashed=review['source_latest_checkpoint']
    require(scoped(original['chosen_checkpoint']['path'])==scoped(unhashed['path']) and chosen['bytes']==unhashed['bytes'],
        'Actual chosen latest path/size differs')
    for key,item in a.items():checked_record(item,checkpoint=key in ('chosen_checkpoint','original_champion'))
    for name,digest in PARENT_PINS.items():require(record(HERE/name)['sha256']==digest,'Actual parent source changed: '+name)
    config=read(a['run_config']['path']);selection=read(a['selection']['path']);controller=read(a['controller_status']['path']);identity=config['config_identity']
    require(config['run_id']==selection['run_id']==controller['run_id']==receipt['source_run_id']
        and config['phase']=='HF_TSC_8199_uniform_Turkish_majority_PT_focused_v1'
        and controller['status']=='train_window_ended' and controller['child_exit_code']==0 and controller['resource_review_required'] is False
        and controller['stop_category']=='retention' and controller['controller_sha256']==PARENT_PINS[PARENT_CONTROLLER]
        and controller['completed_steps']==selection['completed_steps']==8250 and selection['selected_step'] is None
        and selection['accepted'] is False and selection['consecutive_retention_failures']==5
        and selection['heldout_tests_evaluated'] is False,'Preserve actual false-gate source artifacts unchanged')
    require(canonical(identity)==receipt['scientific_config_identity_sha256'],'Whole source identity changed')
    for name,digest in identity['source_sha256'].items():require(record(HERE/name)['sha256']==digest,'Parent dependency changed: '+name)
    prefix,last=_history_frontier(a['training_history']['path'],8250,8250)
    require(prefix==receipt['source_history_prefix_sha256'] and last['selection_diagnostics']['consecutive_retention_failures']==5
        and last['validation']['retention_gate_passed'] is False,'Actual false history/step prefix must remain intact')
    origin_record=receipt['original_PT_source_receipt'];checked_record(origin_record)
    require(scoped(origin_record['path'])==scoped(config['majority_source_receipt']),'Original8000 selector source changed')
    origin=parent.source_context(str(scoped(origin_record['path'])),receipt['source_run_id'])
    sampler_identity=identity['domain_sampling_identity'];quota=sampler_identity['majority_quota_transition'];focus=sampler_identity['PT_focus_transition']
    require(quota==origin['quota_transition'] and focus['source_step']==8000 and quota['source_step']==7621
        and quota['configured_end_step']==25109 and sampler_identity['sampler_family']=='TSC_uniform_majority_PT_focus_six_replay_sampler_v1',
        'Original6500/7621/8000 same-pass and selector identity required')
    coverage=last['Turkish_domain_coverage']['uniform_pass']
    require(coverage['source_step']==6500 and coverage['draws']==coverage['unique_clips_seen']==review['uniform_unique_preserved']==17774
        and coverage['repeat_visits']==0,'Actual uniform progress must carry')
    summary=read(a['chosen_DEV_summary']['path']);baseline=read(a['Portuguese_baseline_proof']['path'])['baseline_overlay'];gates=retention_gates(summary,baseline)
    expected={lang:{'word_errors':v['candidate_errors'],'reference_words':v['reference_words'],'allowed_word_errors':v['allowed_errors'],'passed':v['passed']}
        for lang,v in review['strict_count_gates'].items()}
    migration=review['declared_state_accounting_migration']
    require(gates==expected and all(v['passed'] for v in gates.values()) and migration['previous_actual_failure_counter']==5
        and migration['new_counter_after_correctly_qualified_last_DEV']==0 and migration['correctly_eligible_source_step']==8250
        and migration['correctly_eligible_source_TR_errors']==summary['validation_tr']['asr']
        and migration['historical_false_gate_and_selection_are_preserved'] is True,'Exact original count gates/accounting correction required')
    recipe=dict(identity['training_arguments']);require(recipe==origin['source_recipe'] and recipe['lora_lr']==1e-4
        and recipe['head_lr']==1e-5 and recipe['steps']==25109,'No new optimizer/LR/loss/quota recipe allowed')
    done=read(a['runtime_completion']['path']);owned=read(a['owned_worker_identity']['path']);ready=read(a['runtime_native_ready']['path'])
    require(done['status']=='complete' and done['sources_unchanged'] is True and not done.get('error')
        and done['pid']==owned['pid']==ready['pid'] and done['actual_dispatch_source_sha256']==PARENT_PINS[PARENT_CONTROLLER],
        'Actual same-owned native runtime must complete')
    for key in ('memory','cache','io'):
        value=read(a[key]['path']);require(value.get('status')=='complete' and value.get('sources_unchanged') is True,'Actual runtime incomplete: '+key)
    context={**origin,'receipt':receipt,'receipt_record':record(receipt_path),'source_record':record(receipt_path),'review':review,
        'config':config,'selection':selection,'parent_identity':identity,'parent_recipe':recipe,'source_recipe':recipe,
        'source_step':8250,'source_selected_step':8250,'source_checkpoint':chosen,'source_summary':summary,'source_frontier_row':last,
        'source_history_prefix_sha256':prefix,'source_retention_gates':gates,'quota_transition':quota,'quota_boundary_step':7621,
        'focus_origin_step':8000,'PT_focus_boundary_step':8000,'lora_boundary_step':7750,'uniform_origin_step':6500,
        'remaining_updates_from_chosen_frontier':25109-8250,'prior_uniform_draws':17774,
        'original_PT_source_receipt':origin_record,'source_best_checkpoint':chosen,'source_best_step':8250,
        'source_actual_selected_step':None,'source_actual_failure_counter':5,'corrected_initial_retention_failures':0,
        'corrected_initial_best_score':summary['validation_tr']['asr']['wer'],'corrected_initial_selected_step':8250,
        'indexed_audit':read(a['indexed_TR_audit']['path']),'indexed_audit_record':a['indexed_TR_audit'],
        'original_source350':a['initial_source_equivalence'],'original_native_ready':a['original_native_ready'],
        'uniform_source_receipt':a['uniform_source_receipt'],'portuguese_preparation_proof':a['Portuguese_preparation_proof'],
        'portuguese_baseline_proof':a['Portuguese_baseline_proof'],'parked_original_champion':a['original_champion'],'combined_baseline':baseline}
    return context


def validate_portuguese_inputs(args,context):return parent.validate_portuguese_inputs(args,context)


def preflight(args):
    context=source_context(str(scoped(args.majority_source_receipt)),args.run_id).copy()
    require(scoped(args.resume_majority)==scoped(context['source_checkpoint']['path']),'Same actual8250 complete state required')
    for key,value in context['source_recipe'].items():require(getattr(args,key)==value,'Recipe change forbidden: '+key)
    for key in ('tsc_input_proof','tsc_training_approval','source_closure','tr_manifest','replay_manifest','data_v3_proof',
        'tr_frame_metadata','tr_frame_metadata_proof','reuse_input_audit','supplement','portuguese_preparation_proof','portuguese_baseline_proof',
        'portuguese_replay_manifest','portuguese_dev_manifest','indexed_tr_directory','resume_uniform_full','uniform_full_source_receipt'):
        before=context['config'].get(key);after=getattr(args,key,None)
        require((before is None and after is None) or (before is not None and after is not None and scoped(before)==scoped(after)),
            'Original admitted path changed: '+key)
    require(args.portuguese_seed==context['config']['portuguese_seed'],'Independent replay seed changed')
    validate_portuguese_inputs(args,context);return context


def audit_parent_state(model,state,context,*,torch_module):
    torch=torch_module;require(state.get('schema')==2 and state.get('run_id')==context['receipt']['source_run_id']
        and state.get('step')==8250 and canonical(state.get('config_identity'))==canonical(context['parent_identity'])
        and state['selected_step'] is None and state['best_score']==math.inf and state['retention_failures']==5,
        'Audit actual unmodified8250 failure5/NoneBEST source before accounting projection')
    require(len(state['history'])==8250 and canonical(state['history'])==context['source_history_prefix_sha256']
        and state['sampler']['updates']==8250 and state['sampler']['identity']==context['parent_identity']['domain_sampling_identity']
        and state['sampler']['kind']=='TSC_uniform_majority_PT_focus_six_replay_sampler_v1', 'Complete own PT runtime/frontier required')
    for key in ('optimizer','lora_metadata','python_rng','torch_rng','cuda_rng','patience','next_eval_epoch','train_wall_seconds',
        'audio_seconds','language_audio_seconds','evaluation_wall_seconds','initial_gate','stop_reason'):
        require(key in state,'Incomplete own8250 runtime: '+key)
    expected={name:p for name,p in model.named_parameters() if p.requires_grad};require(len(expected)==207 and set(expected)==set(state['weights']),'All207 weights required')
    for name,value in state['weights'].items():require(value.shape==expected[name].shape and value.dtype==torch.float32
        and bool(torch.isfinite(value).all()),'FP32 source shape/finiteness differs: '+name)
    from encoder_lora import encoder_lora_metadata
    require(encoder_lora_metadata(model)==state['lora_metadata'] and model.config.vocab_size==8199
        and model.config.blank_token_id==8192 and list(model.config.durations)==[0,1,2,3,4], 'Original8199 geometry required')
    audit=comparison.audit_source_optimizer(state,context,expected,torch_module=torch)
    require(record(context['source_checkpoint']['path'])==context['source_checkpoint'],'Actual closed8250 bytes changed')
    return {'passed':True,'source_Adam_audit':audit,'all207_weights_Adam_RNG_sampler_history_preserved':True,
        'source_actual_failure_counter':5,'declared_corrected_counter_after_actual350':0,'source_actual_selected_step':None}


def restore_parent_sampler(sampler,state,context):
    saved=state['sampler'];identity=context['parent_identity']['domain_sampling_identity'];prefix=saved['source_prefix'];quota=saved['quota_transition'];focus=saved['PT_focus_transition']
    require(saved['identity']==identity and saved['updates']==8250 and quota==context['quota_transition']
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
    require(sampler.updates==8250 and sampler.configured_end_step==25109,'Same own8250 queues/clock/end required')
    return {'kind':'actual_PT_focus_own8250_sampler_restore_v1','passed':True,'source_step':8250,
        'focus_origin_step':8000,'quota_boundary_step':7621,'no_new_selector_migration_or_shuffle':True}


def compare_source_development(*args,**kwargs):
    # Reuse the immutable exact350/RNG/path comparator with a PRIVATE globals
    # view changing only gate arithmetic. No frozen module globals are mutated.
    frozen=comparison.compare_source_development
    exact=FunctionType(frozen.__code__,{**frozen.__globals__,'retention_gates':retention_gates},
        frozen.__name__,frozen.__defaults__,frozen.__closure__)
    return exact(*args,**kwargs)


def release_admission_caches():
    source_context.cache_clear();parent.release_admission_caches()
    return {'integer_gate_and_source_admission_caches_cleared':True,'disk_or_other_process_changed':False}
