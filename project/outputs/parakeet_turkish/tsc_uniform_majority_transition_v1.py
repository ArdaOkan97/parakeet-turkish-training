"""Inactive CLOSED uniform full-state -> declared 14TR/2RP continuation.

Metadata admission never reads checkpoint bytes. Tensor checks are explicitly
caller-injected at future runtime. The original uniform pass, Portuguese grace,
all207 Adam clocks and inherited BEST remain separate from the quota boundary.
"""
from __future__ import annotations
from datetime import datetime
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import re

from tsc_uniform_full_transition_v1 import (
    HERE, require, scoped, read, record, checked_record, canonical, byte_record,
    source_context as original_source_context, validate_portuguese_inputs as original_PT_inputs,
    audit_source_optimizer, compare_development)

ROOT=HERE.parent.parent
RECEIPT_KIND='ROOT_owned_closed_TSC_uniform_for_Turkish_majority_v1'
PROOF_KIND='TSC_uniform_majority_before_update_source_equivalence_v1'
PROOF_NAME='majority_initial_source_equivalence.json'
PHASE='HF_TSC_8199_uniform_Turkish_majority_with_Portuguese_v1'
PARENT_PHASE='HF_TSC_8199_uniform_fullpass_with_Portuguese_v1'
LANGUAGES=('tr','en','es','it','fr','de','pt')
RETENTION_LANGUAGES=LANGUAGES[1:]
UNIFORM_ORIGIN=6500
TR_COUNT=253798
PARENT_CONTROLLER='run_TSC_uniform_full_v1.py'
PARENT_WORKER='train_lora_batched_tsc_uniform_full_v1.py'
SOURCE_PINS={PARENT_CONTROLLER:'dfbac1e473274728f287169a9b3db3edeb03f39046d694c615c026ae7e2ada4f',
    PARENT_WORKER:'b9e550ec145c56a7a75521a997a8a6a33bd98eee7d76c1cb0cabfb116a40631c',
    'tsc_uniform_full_transition_v1.py':'08ec46e079cadf2805f72b24078b92fc560e5701228a033228eb3519219079de',
    'tsc_uniform_full_sampler_v1.py':'8ebd56d20e47a568db4a7cfe50ed172ad86af253275b3b1172ad78a34b60439a',
    'tsc_uniform_majority_sampler_v1.py':'19adad00ed5d2c00ad3c07e9d31c221c8d171e1202b1ea83348176e276d61976'}
RESOURCE_REVIEW_KIND='ROOT_TSC_uniform_majority_resource_review_v1'
RETENTION_REVIEW_KIND='ROOT_TSC_uniform_majority_retention_review_v1'


def proof_path(args):return HERE/'runs'/args.run_id/PROOF_NAME


def majority_declaration(source_step,prior_uniform_draws):
    require(type(source_step) is int and source_step>UNIFORM_ORIGIN
        and type(prior_uniform_draws) is int and prior_uniform_draws==8*(source_step-UNIFORM_ORIGIN)
        and 0<prior_uniform_draws<TR_COUNT,'Actual unfinished 8/8 uniform frontier required')
    remaining=math.ceil((TR_COUNT-prior_uniform_draws)/14)
    return {'kind':'declared_existing_uniform_Turkish_majority_v1','source_step':source_step,
        'original_uniform_source_step':UNIFORM_ORIGIN,'prior_uniform_draws':prior_uniform_draws,
        'remaining_update_budget':remaining,'configured_end_step':source_step+remaining,
        'TR_per_update':14,'replay_per_update':2,'batch_size':8,'accumulation':2,
        'physical_microbatch_composition':[[8,0],[6,2]],'replay_fraction':.125,
        'preservation_coefficients':{'replay_kl':40.,'duration_kl':2.,'feature_distillation':4.},
        'source_preservation_coefficients':{'replay_kl':10.,'duration_kl':.5,'feature_distillation':1.},
        'coefficient_accounting':'RP fraction .5 to .125; mean16 expected preservation density retained',
        'equal_gradients_or_quality_claimed':False,'head_lr':1e-5,'lora_lr':3e-4,'schedule':'constant',
        'all207_weights_Adam_global_clocks_sampler_RNG_history_preserved':True,
        'new_queue_shuffle_or_optimizer_reset':False,'renewed_Portuguese_grace':False,
        'Portuguese_grace_updates_since_original6500':1000,'parent_retention_failure_counter_preserved':True,
        'all_six_required_for_every_BEST':True,'original_old_five_champion_parked':True,
        'actual_uniform_unique_clips_for_full_pass':TR_COUNT,'last_whole_update_next_cycle_repeats':remaining*14-(TR_COUNT-prior_uniform_draws),
        'plateau_patience_target_WER_disabled_before_full_unique_pass':True,'TEST_used':False}


def retention_gates(summary,baseline):
    result={}
    for lang in LANGUAGES:
        score=summary['validation_'+lang]['asr'];base=baseline['validation_'+lang]['asr']
        require(score.get('utterances')==50 and type(score.get('word_errors')) is int
            and type(score.get('reference_words')) is int and score['reference_words']>0
            and score['reference_words']==base['reference_words']
            and score.get('wer')==score['word_errors']/score['reference_words'],'Actual fixed DEV count/ratio differs: '+lang)
        if lang!='tr':
            cap=base['wer']+max(.005,.1*base['wer'])
            allowed=math.floor(cap*score['reference_words']+1e-9)
            result[lang]={'word_errors':score['word_errors'],'reference_words':score['reference_words'],
                'allowed_word_errors':allowed,'passed':score['wer']<=cap}
    return result


def validate_closure_header(controller):
    resource=controller.get('resource_review_required') is True
    require(controller.get('status')==('resource_review_required' if resource else 'train_window_ended')
        and controller.get('child_exit_code')==0 and controller.get('finished_at_utc')
        and controller.get('actual_runtime_completion') and controller.get('resource_review_required') is resource,
        'Actual terminal owned child0/runtime completion required before any checkpoint bytes')
    require(controller.get('controller_sha256')==SOURCE_PINS[PARENT_CONTROLLER]
        and controller.get('kind')=='guarded_TSC_8199_uniform_full_Portuguese_controller_v1'
        and ((resource and controller.get('stop_category')=='resource') or
             (not resource and controller.get('stop_category') in ('budget','manual','scientific','quality'))),
        'Pinned uniform executor and truthful terminal category required')
    return 1 if resource else 0


def validate_review(item,kind,receipt,controller,gates):
    value=read(checked_record(item));original=receipt['original_artifact_records']
    require(value.get('schema')==1 and value.get('kind')==kind and value.get('passed') is True
        and value.get('source_run_id')==receipt['source_run_id'] and value.get('chosen_step')==receipt['chosen_step']
        and value.get('chosen_checkpoint_role')==receipt['chosen_checkpoint_role']
        and value.get('source_controller_sha256')==SOURCE_PINS[PARENT_CONTROLLER]
        and value.get('reviewed_status_artifact')==original['controller_status']
        and value.get('chosen_DEV_summary')==original['chosen_DEV_summary']
        and value.get('owned_actual_exit')==receipt['owned_actual_exit']
        and value.get('allow_majority_continuation') is True and value.get('guards_retained') is True,
        'Exact ROOT review/status/DEV/owned exit binding required: '+kind)
    if kind==RESOURCE_REVIEW_KIND:
        require(value.get('stop_reason')==controller.get('cooperative_stop_reason')
            and controller.get('resource_review_required') is True
            and value.get('material_policy_change')=={'TR_per_update':14,'replay_per_update':2,'physical_microbatch_composition':[[8,0],[6,2]]}
            and value.get('actual_next_runtime_fit_not_claimed') is True,'Resource review must bind actual stop and changed layout; no fit claim')
    else:
        require(value.get('retention_gates')==gates and value.get('renewed_Portuguese_grace') is False
            and value.get('parent_failure_counter_preserved') is True,'Unqualified review must retain exact gate/failure/grace facts')
    return value


def validate_receipt(value,run_id):
    require(value.get('schema')==1 and value.get('kind')==RECEIPT_KIND and value.get('status')=='actual_CLOSED_reviewed_uniform_frontier'
        and value.get('target_run_id')==run_id and value.get('source_run_id')!=run_id
        and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{2,110}',run_id or '')
        and value.get('allow_Turkish_majority_continuation') is True and value.get('guards_retained') is True
        and value.get('chosen_checkpoint_role') in ('latest','best')
        and value.get('declared_majority')==majority_declaration(value['chosen_step'],value['prior_uniform_draws']),
        'Distinct ROOT-declared full-state majority scientific fork required')
    h=value['chosen_step'];c=value['completed_steps'];j=value.get('source_best_step')
    require(type(c) is int and h<=c and ((value['chosen_checkpoint_role']=='latest' and h==c)
        or (value['chosen_checkpoint_role']=='best' and h==j and value.get('explicit_earlier_BEST_full_state_fork') is True)),
        'One chosen latest or explicitly reviewed earlier BEST frontier; no mixed states')
    require((j is None and value.get('source_best_checkpoint') is None) or
        (type(j) is int and UNIFORM_ORIGIN<j<=h and type(value.get('source_best_checkpoint')) is dict),
        'Separate parent eligible BEST lineage required')
    exit_value=value['owned_actual_exit'];require(exit_value.get('role')=='trainer_controller'
        and type(exit_value.get('session_id')) is int and exit_value['session_id']>0
        and exit_value.get('exit_code') in (0,1) and re.fullmatch('[0-9a-f]+',exit_value.get('chunk_id','')),
        'ROOT actual owned controller exit record required')
    return value


def _history_frontier(path,step,completed):
    """Bounded JSON array decoder: retain only the chosen row, not full history."""
    decoder=json.JSONDecoder(parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite history')))
    buffer='';position=0;count=0;chosen=None;digest=hashlib.sha256(b'[')
    with Path(path).open(encoding='utf-8') as stream:
        def more():
            nonlocal buffer,position
            old=len(buffer)-position;buffer=buffer[position:]+stream.read(65536);position=0
            return len(buffer)>old
        def whitespace():
            nonlocal position
            while True:
                while position<len(buffer) and buffer[position].isspace():position+=1
                if position<len(buffer) or not more():return
        whitespace();require(buffer[position:position+1]=='[','History array required');position+=1
        while True:
            whitespace()
            if buffer[position:position+1]==']':position+=1;break
            while True:
                try:row,end=decoder.raw_decode(buffer,position);break
                except json.JSONDecodeError:require(more() and len(buffer)<=8_000_000,'Truncated/unbounded history row')
            position=end;count+=1;require(type(row) is dict and row.get('step')==count,'Contiguous full-state history required')
            if count<=step:
                if count>1:digest.update(b',')
                digest.update(json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())
            if count==step:chosen=row
            whitespace();separator=buffer[position:position+1];position+=1
            if separator==']':break
            require(separator==',','History separator changed')
        whitespace();require(position==len(buffer) and not stream.read(1),'Trailing history bytes')
    require(count==completed and UNIFORM_ORIGIN<step<=count and chosen is not None,'Chosen closed history frontier missing')
    digest.update(b']');return digest.hexdigest(),chosen


def validate_portuguese_inputs(args,context):
    # The original6500 proof remains its own lineage. Do not reinterpret H as
    # the original Portuguese baseline source or renew its recovery grace.
    original=read(context['uniform_source_receipt']['path']);origin=original['source_lineage_receipt']
    before=original_source_context(str(scoped(origin['path'])),origin['sha256']).copy()
    preparation=context['receipt']['original_artifact_records']['Portuguese_preparation_proof']
    checked_record(preparation)
    archived=context['portuguese_preparation_proof']
    require((preparation['sha256'],preparation['bytes'])==(archived['sha256'],archived['bytes']),
        'Original Portuguese preparation and archived alias bytes differ')
    before.update(portuguese_preparation_proof=preparation,
        portuguese_baseline_proof=context['portuguese_baseline_proof'])
    original_PT_inputs(args,before)
    for key in ('combined_baseline','portuguese_preparation','portuguese_baseline','portuguese_validation_directory',
        'portuguese_replay_manifest','portuguese_dev_manifest','Portuguese_source_result'):
        context[key]=before[key]
    return context


@lru_cache(maxsize=2)
def source_context(receipt_path,run_id=None):
    path=scoped(receipt_path);receipt=read(path);validate_receipt(receipt,run_id or receipt['target_run_id'])
    artifacts=receipt['artifacts']
    for name,digest in SOURCE_PINS.items():require(record(HERE/name)['sha256']==digest,'Pinned source changed: '+name)
    for name,item in artifacts.items():checked_record(item,checkpoint=name in ('chosen_checkpoint','source_best_checkpoint','original_champion'))
    controller=read(artifacts['controller_status']['path']);exit_code=validate_closure_header(controller)
    require(controller['run_id']==receipt['source_run_id'] and receipt['owned_actual_exit']['exit_code']==exit_code,
        'Actual resource controller exit1 must remain exit1')
    observed=datetime.fromisoformat(receipt['owned_actual_exit']['observed_at_utc'].replace('Z','+00:00'))
    finished=datetime.fromisoformat(controller['finished_at_utc'].replace('Z','+00:00'))
    require(observed.tzinfo is not None and observed>=finished,'ROOT owned exit observation must follow terminal publication')
    config=read(artifacts['run_config']['path']);selection=read(artifacts['selection']['path']);identity=config['config_identity']
    h=receipt['chosen_step'];c=receipt['completed_steps'];recipe=dict(identity['training_arguments'])
    require(config['run_id']==selection['run_id']==controller['run_id'] and config['phase']==PARENT_PHASE
        and config['trainable_parameter_tensors']==207 and config['turkish_clips_used']==TR_COUNT and config['replay_clips_used']==1914
        and selection['completed_steps']==controller['completed_steps']==c and selection['heldout_tests_evaluated'] is False
        and canonical(identity)==receipt['scientific_config_identity_sha256']
        and identity['extended_tokenizer_sha256']=='b91d500326aca290ed709d629dab2bee56f298ea7323b94855b9f5ef0caeac75'
        and recipe['head_lr']==1e-5 and recipe['lora_lr']==3e-4 and recipe['schedule']=='constant'
        and recipe['replay_fraction']==.5 and recipe['replay_kl']==10 and recipe['duration_kl']==.5 and recipe['feature_distillation']==1
        and recipe['batch_size']==8 and recipe['accumulation']==2 and recipe['sampling_mode']=='uniform'
        and config['uniform_full_pass']['source_step']==UNIFORM_ORIGIN,'Original8199/207 fulluniform science/pool required')
    for name,digest in identity['source_sha256'].items():require(record(HERE/name)['sha256']==digest,'Parent source changed: '+name)
    originals=receipt['original_artifact_records'];role=receipt['chosen_checkpoint_role']
    expected=selection['latest_weights'] if role=='latest' else selection['weights']
    require(scoped(originals['chosen_checkpoint']['path'])==scoped(expected),'Chosen role/checkpoint pointer differs')
    require(receipt['source_best_step']==selection['selected_step'] and (selection['accepted'] is (selection['selected_step'] is not None)),
        'Eligible parent BEST remains distinct from chosen latest')
    require(artifacts['original_champion']==config['source_champion_parked'], 'Original old-five champion must stay parked')
    if receipt['source_best_step'] is not None:
        require(receipt['source_best_checkpoint']==artifacts['source_best_checkpoint']
            and scoped(originals['source_best_checkpoint']['path'])==scoped(selection['weights']),
            'Separate inherited eligible BEST archive/pointer changed')
    prefix,last=_history_frontier(artifacts['training_history']['path'],h,c)
    require(prefix==receipt['source_history_prefix_sha256'] and last.get('uniform_full_updates')==h-UNIFORM_ORIGIN,
        'Chosen history prefix/counter changed')
    coverage=last['Turkish_domain_coverage']['uniform_pass']
    require(coverage['source_step']==UNIFORM_ORIGIN and coverage['draws']==receipt['prior_uniform_draws']
        and coverage['unique_clips_seen']==coverage['draws'] and coverage['repeat_visits']==0
        and coverage['pool_clips']==TR_COUNT,'Actual unfinished uniform queue/unique frontier required')
    summary=read(artifacts['chosen_DEV_summary']['path']);baseline=read(artifacts['Portuguese_baseline_proof']['path'])['baseline_overlay']
    gates=retention_gates(summary,baseline);require(gates==receipt['chosen_retention_gates'],'Saved chosen gate accounting differs')
    if receipt['source_best_step'] is not None:
        selected_summary=read(artifacts['source_best_DEV_summary']['path'])
        require(all(v['passed'] for v in retention_gates(selected_summary,baseline).values())
            and selected_summary['validation_tr']['asr']['wer']==selection['turkish_validation_wer'],
            'Inherited parent BEST must have actual all-six qualification and score')
    if exit_code==1:validate_review(receipt['resource_stop_review'],RESOURCE_REVIEW_KIND,receipt,controller,gates)
    else:require(receipt.get('resource_stop_review') is None,'Unneeded resource bypass prohibited')
    if not all(item['passed'] for item in gates.values()):
        require(role=='latest','Unqualified checkpoint cannot be described as selected BEST')
        validate_review(receipt['unqualified_source_review'],RETENTION_REVIEW_KIND,receipt,controller,gates)
    else:require(receipt.get('unqualified_source_review') is None,'Qualified source does not need an unqualified bypass')
    done=read(artifacts['runtime_completion']['path']);owned=read(artifacts['owned_worker_identity']['path']);runtime=read(artifacts['runtime_native_ready']['path'])
    require(done.get('kind')=='TSC_uniform_full_native_completion_v1' and done.get('status')=='complete'
        and done.get('sources_unchanged') is True and not done.get('error')
        and done['pid']==owned['pid']==runtime['pid'] and done['run_id']==controller['run_id']
        and done['actual_dispatch_source_sha256']==SOURCE_PINS[PARENT_CONTROLLER]
        and done['fresh_runtime_sha256']==artifacts['runtime_native_ready']['sha256']
        and done['backend_ready_sha256']==artifacts['original_native_ready']['sha256'],
        'Complete same-owned worker/native boundary required')
    for name,kind in (('memory','TSC_uniform_full_memory_runtime_v1'),('cache','TSC_uniform_full_cache_runtime_v1'),('io','TSC_uniform_full_atomic_IO_v1')):
        value=read(artifacts[name]['path']);require(value.get('kind')==kind and value.get('status')=='complete'
            and value.get('sources_unchanged') is True and not value.get('error'),'Runtime completion failed: '+name)
    memory=read(artifacts['memory']['path']);cache=read(artifacts['cache']['path'])
    require(memory['cap_bytes']==8_000_000_000 and memory['cleanup_threshold_bytes']==3_000_000_000
        and memory['allocator_cap_applied'] is True and memory['all_phase_owned_memory_telemetry']['thread_join_completed'] is True
        and cache['cache_reclamation']['eligible_update_calls']==c-controller['resume_step'],'Actual guard/cache suffix frontier differs')
    original_proof=read(artifacts['initial_source_equivalence']['path'])
    require(original_proof.get('passed') is True and original_proof.get('actual_GPU_decode') is True
        and original_proof.get('source_selected_step')==UNIFORM_ORIGIN
        and identity['input_sha256'].get(originals['initial_source_equivalence']['path'])==artifacts['initial_source_equivalence']['sha256'],
        'Original source350 token changed')
    audit=read(artifacts['indexed_TR_audit']['path'])
    require(audit==config['public_turkish_data_audit'],'Original admitted index audit changed')
    active={**recipe,'replay_fraction':.125,'replay_kl':40.,'duration_kl':2.,'feature_distillation':4.,
        'steps':receipt['declared_majority']['configured_end_step']}
    context={'receipt':receipt,'receipt_record':record(path),'source_record':record(path),'config':config,'selection':selection,
        'parent_identity':identity,'parent_recipe':recipe,'source_recipe':active,'source_step':h,'source_selected_step':h,
        'uniform_origin_step':UNIFORM_ORIGIN,'source_checkpoint':artifacts['chosen_checkpoint'],'source_summary':summary,
        'source_history_prefix_sha256':prefix,'source_frontier_row':last,'source_retention_gates':gates,
        'indexed_audit':audit,'indexed_audit_record':artifacts['indexed_TR_audit'],'original_source350':artifacts['initial_source_equivalence'],
        'original_native_ready':artifacts['original_native_ready'],'uniform_source_receipt':artifacts['uniform_source_receipt'],
        'portuguese_preparation_proof':artifacts['Portuguese_preparation_proof'],'portuguese_baseline_proof':artifacts['Portuguese_baseline_proof'],
        'source_best_checkpoint':receipt.get('source_best_checkpoint'),'source_best_step':receipt.get('source_best_step'),
        'parked_original_champion':artifacts['original_champion'],'prior_uniform_draws':receipt['prior_uniform_draws'],
        'remaining_update_budget':receipt['declared_majority']['remaining_update_budget'],
        'configured_end_step':active['steps']}
    return context


def preflight(args):
    require(getattr(args,'resume_majority',None) is not None and getattr(args,'majority_source_receipt',None) is not None,
        'Explicit majority checkpoint/ROOT receipt required')
    context=source_context(str(scoped(args.majority_source_receipt)),args.run_id).copy()
    require(scoped(args.resume_majority)==scoped(context['source_checkpoint']['path']),'Chosen CLOSED frontier changed')
    for key,value in context['source_recipe'].items():require(getattr(args,key)==value,'Unapproved scientific recipe change: '+key)
    for key in ('tsc_input_proof','tsc_training_approval','source_closure','tr_manifest','replay_manifest','data_v3_proof',
        'tr_frame_metadata','tr_frame_metadata_proof','reuse_input_audit','supplement','portuguese_preparation_proof','portuguese_baseline_proof',
        'portuguese_replay_manifest','portuguese_dev_manifest','indexed_tr_directory','resume_uniform_full','uniform_full_source_receipt'):
        before=context['config'].get(key);after=getattr(args,key,None)
        require((before is None and after is None) or (before is not None and after is not None and scoped(before)==scoped(after)),
            'Approved source/input path changed: '+key)
    require(args.portuguese_seed==context['config']['portuguese_seed'],'Portuguese independent queue seed changed')
    validate_portuguese_inputs(args,context)
    return context


def audit_parent_state(model,state,context,*,torch_module):
    h=context['source_step'];require(state.get('schema')==2 and state.get('run_id')==context['receipt']['source_run_id']
        and state.get('step')==h and canonical(state.get('config_identity'))==canonical(context['parent_identity']),
        'All weights/Adam/runtime must come from the single chosen parent frontier')
    required=('weights','lora_metadata','optimizer','sampler','python_rng','torch_rng','cuda_rng','history','best_score','selected_step',
        'patience','retention_failures','next_eval_epoch','train_wall_seconds','audio_seconds','language_audio_seconds',
        'evaluation_wall_seconds','initial_gate','stop_reason')
    require(all(key in state for key in required) and len(state['history'])==h
        and all(row.get('step')==index for index,row in enumerate(state['history'],1))
        and canonical(state['history'])==context['source_history_prefix_sha256'], 'Complete exact chosen history/RNG/runtime required')
    sampler=state['sampler'];require(sampler.get('kind')=='TSC_uniform_full_TR_six_replay_sampler_v1'
        and sampler.get('updates')==h and canonical(sampler.get('identity'))==canonical(context['parent_identity']['domain_sampling_identity'])
        and sampler['source_prefix']['updates']==UNIFORM_ORIGIN
        and sampler['turkish_streams']['uniform']['draws']==context['prior_uniform_draws'], 'Chosen sampler queue/clock/pool changed')
    require(state['selected_step']==context['source_best_step']
        and ((state['selected_step'] is None and math.isinf(state['best_score']) and state['best_score']>0)
             or (state['selected_step'] is not None and state['best_score']==context['selection']['turkish_validation_wer'])),
        'Actual selectedBEST lineage differs from chosen source runtime')
    diagnostics=context['source_frontier_row']['selection_diagnostics']
    require(state['retention_failures']==diagnostics['consecutive_retention_failures']
        and state['patience']==diagnostics['plateau_epoch_evaluations'],'Do not reset inherited retention/patience counters')
    require(all(type(state[key]) in (int,float) and math.isfinite(state[key]) and state[key]>=0
        for key in ('train_wall_seconds','audio_seconds','evaluation_wall_seconds')),'Finite inherited runtime counters required')
    expected={name:p for name,p in model.named_parameters() if p.requires_grad};torch=torch_module
    require(len(expected)==207 and set(expected)==set(state['weights']),'Exact all207 names required')
    for name,value in state['weights'].items():require(value.dtype==torch.float32 and value.shape==expected[name].shape
        and bool(torch.isfinite(value).all()),'FP32 source shape/finiteness differs: '+name)
    from encoder_lora import encoder_lora_metadata
    require(encoder_lora_metadata(model)==state['lora_metadata'] and model.config.vocab_size==8199
        and model.config.blank_token_id==8192 and list(model.config.durations)==[0,1,2,3,4]
        and model.decoder.embedding.num_embeddings==8199 and model.joint.head.out_features==8204,'Original8199 geometry/layout required')
    audit=audit_source_optimizer(state,{**context,'source_recipe':context['parent_recipe']},expected,torch_module=torch)
    require(record(context['source_checkpoint']['path'])==context['source_checkpoint'],'Chosen closed bytes changed')
    return {'passed':True,'source_step':h,'source_Adam_audit':audit,'all207_weights_and_Adam_restore_required':True,
        'history_RNG_sampler_and_failure_counters_preserved':True,'source_retention_gates':context['source_retention_gates']}


def compare_source_development(model,processor,args,context,state,evaluate_fn,torch_module):
    torch=torch_module;folder=HERE/'results'/f'{args.run_id}_step_{context["source_step"]}';target=proof_path(args)
    require(not target.exists(),'Immutable initial350 proof already exists')
    rng=(random.getstate(),torch.get_rng_state().clone(),[x.clone() for x in torch.cuda.get_rng_state_all()])
    modes=[(module,module.training) for module in model.modules()]
    try:candidate=evaluate_fn(model,processor,f'{args.run_id}_step_{context["source_step"]}',splits=('validation',))
    finally:
        random.setstate(rng[0]);torch.set_rng_state(rng[1]);torch.cuda.set_rng_state_all(rng[2])
        for module,mode in modes:module.train(mode)
    require(random.getstate()==rng[0] and torch.equal(torch.get_rng_state(),rng[1])
        and len(torch.cuda.get_rng_state_all())==len(rng[2])
        and all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),rng[2]))
        and all(module.training==mode for module,mode in modes),'Source350 RNG/module modes must be restored')
    artifacts=context['receipt']['artifacts'];comparisons={}
    for name in ('metadata.json','metadata_pt.json'):
        key='chosen_DEV_metadata_pt' if name=='metadata_pt.json' else 'chosen_DEV_metadata'
        before=read(artifacts[key]['path']);after=read(folder/name)
        require(before['model_fingerprint']==after['model_fingerprint'] and before['manifests']==after['manifests'],
            'Same source model/generation/manifest fingerprint required: '+name)
    for language in LANGUAGES:
        before=[json.loads(line) for line in Path(artifacts['chosen_DEV_'+language]['path']).read_text(encoding='utf-8').splitlines() if line.strip()]
        after=[json.loads(line) for line in (folder/f'validation_{language}.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        comparisons[language]=compare_development(before,after,context['source_summary']['validation_'+language]['asr'],
            candidate['validation_'+language]['asr'],language)
        del before,after
    gates=retention_gates(candidate,context['combined_baseline']);require(gates==context['source_retention_gates'],'Source actual gates changed')
    proof={'schema':1,'kind':PROOF_KIND,'passed':True,'run_id':args.run_id,'source_receipt':context['receipt_record'],
        'source_selected_step':context['source_step'],'uniform_origin_step':UNIFORM_ORIGIN,'actual_GPU_decode':True,
        'before_first_majority_update':True,'compared_fixed_DEV_utterances':350,'all_raw_transcripts_and_emitted_paths_exact':True,
        'all_seven_ASR_counts_exact':True,'all_six_retention_gates_passed':all(v['passed'] for v in gates.values()),
        'source_retention_gates':gates,'reviewed_unqualified_source_not_claimed_BEST':not all(v['passed'] for v in gates.values()),
        'RNG_restored_after_added_evaluation':True,'module_training_modes_restored_after_added_evaluation':True,
        'sampler_drawn_during_check':False,'comparisons':comparisons,'candidate_summary':record(folder/'summary.json'),
        'candidate_metadata':record(folder/'metadata.json'),'candidate_metadata_pt':record(folder/'metadata_pt.json')}
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='utf-8') as stream:json.dump(proof,stream,ensure_ascii=False,indent=2,allow_nan=False)
    return proof


def release_admission_caches():
    source_context.cache_clear();original_source_context.cache_clear()
    return {'majority_and_original_source_admission_caches_cleared':True,'disk_or_other_process_changed':False}
