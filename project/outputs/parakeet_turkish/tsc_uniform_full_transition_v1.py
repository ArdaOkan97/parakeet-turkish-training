"""INACTIVE original8199/207Adam BEST -> declared uniform full TRAIN fork.

The prior ROOT byte archive is lineage evidence, not authorization for this new
recipe. A new ROOT receipt explicitly approves uniform TR and added Portuguese.
No model/checkpoint/ML imports or actual source admission occur at import.
"""
from __future__ import annotations
from functools import lru_cache
import json
import math
from pathlib import Path
import random

from tsc_bpe_acquisition_transition_v1 import (
    HERE, LANGUAGES, require, sha, canonical, scoped, read, record, checked_record,
    validate_receipt as validate_lineage_receipt, validate_closed_runtime, validate_layout,
    validate_parent_state, audit_source_optimizer)
from tsc_optimization_transition_v1 import compare_development

RECEIPT_KIND='ROOT_owned_closed_TSC_8199_for_uniform_fullpass_v1'
PROOF_KIND='TSC_uniform_full_before_update_source_equivalence_v1'
PROOF_NAME='uniform_full_initial_source_equivalence.json'
PARENT_RUN='adaptive_20261011_hf_TSC_headlr02_01'
BEST_SHA='df299dd5e1918f37fc1fb83fe7b428b458daf10a759a021cb6fae07c4eb71a16'
SOURCE_STEP=6500
POOL_COUNTS={'fleurs':2037,'cv':72846,'tsc':178915}
TR_COUNT=253798
PASS_UPDATES=31725
END_STEP=38225
SOURCE_AUDIT_SHA='0c888b2bc1cdbde8fc7c8062a7c6bd6a404d6aa09b62050b93942ec16705d038'


def proof_path(args):
    return HERE/'runs'/args.run_id/PROOF_NAME


def byte_record(value):
    return {key:value[key] for key in ('path','sha256','bytes')}


def uniform_declaration(portuguese_seed):
    require(type(portuguese_seed) is int,'Explicit Portuguese shuffle seed required')
    return {'kind':'declared_TSC_8199_uniform_fullpass_with_Portuguese_v1',
        'source_selected_step':SOURCE_STEP,'original_lexical_size':8199,'blank_id':8192,'joint_width':8204,
        'full_TR_clips':TR_COUNT,'source_group_counts':POOL_COUNTS,'new_uniform_pass_updates':PASS_UPDATES,
        'new_uniform_pass_end_step':END_STEP,'new_uniform_TR_draws':253800,'last_two_draws_next_cycle_repeats':2,
        'historical_TR_domain_queues_not_relabelled_uniform':True,'historical_draws_seen_audio_preserved':True,
        'copied_source_TR_RNG_before_new_fullshuffle':True,'new_uniform_coverage_clock_starts_zero':True,
        'eval_every_updates':250,'final_actual_full_unique_coverage_required':True,
        'original_domain_horizon_lineage_only':44958,'head_lr':1e-5,'lora_lr':3e-4,'rank':8,'alpha':16,
        'batch_size':8,'accumulation':2,'replay_fraction':.5,'replay_kl':10.,'duration_kl':.5,'feature_distillation':1.,
        'new_six_glyph_multiplier':6.,'all207_weights_and_Adam_states_retained':True,'fresh_optimizer_or_warmup':False,
        'original_five_replay_clips':1595,'new_Portuguese_TRAIN_clips':319,'combined_replay_clips':1914,
        'replay_languages':['en','es','it','fr','de','pt'],'original_five_replay_boundary_state_preserved':True,
        'six_language_future_replay_selection_changes_explicitly':True,'old_five_future_replay_sequence_claimed':False,
        'new_Portuguese_shuffle_seed':portuguese_seed,'selection_fixed_DEV_only':True,'TEST_selection':False,
        'source_original_five_qualified_new_Portuguese_gate_failed':True,'initial_all_six_champion_claimed':False,
        'Portuguese_recovery_grace_new_updates':1000,'original_five_retention_always_guarded':True,
        'all_six_retention_required_for_every_BEST_selection':True,'after_grace_all_six_max_consecutive_failures':2,
        'full31725_pass_target_not_replaced_by_grace_screen_or_restart':True,
        'training_rows':'indexed_disk_metadata_with_bounded_cache',
        'TR_sampler_queues':'compact_indices_not_resident_rowdicts',
        'active_uniform_stop_policy':{'plateau_patience_target_WER_before_full_unique_pass':False,
            'actual_new_uniform_unique_clips_required':TR_COUNT,
            'early_stop_allowed':['resource','nonfinite','original_five_retention',
                'all_six_retention_after_Portuguese_1000_update_grace'],
            'legacy_plateau_patience_target_fields_preserved_as_lineage':True}}


def validate_receipt(value,run_id,portuguese_seed):
    require(value.get('schema')==1 and value.get('kind')==RECEIPT_KIND
        and value.get('status')=='actual_CLOSED_old_five_qualified_PT_recovery_source' and value.get('target_run_id')==run_id
        and value.get('source_run_id')==PARENT_RUN and value.get('selected_step')==SOURCE_STEP
        and value.get('allow_uniform_fullpass') is True and value.get('declared_uniform_fullpass')==uniform_declaration(portuguese_seed)
        and value.get('BEST_not_latest_state') is True and value.get('artifacts',{}).get('best_checkpoint',{}).get('sha256')==BEST_SHA,
        'New explicit ROOT uniform fullpass/PT recovery from old-five-qualified original8199 BEST6500 required')
    return value


def _history_prefix(path,step,completed):
    """Stream one history object at a time; no second6500-row decoded array."""
    def invalid_constant(value):raise ValueError('Nonfinite archived history constant: '+value)
    decoder=json.JSONDecoder(parse_constant=invalid_constant);buffer='';position=0;count=0;digest=__import__('hashlib').sha256();digest.update(b'[')
    with Path(path).open(encoding='utf-8') as stream:
        def more():
            nonlocal buffer,position
            buffer=buffer[position:]+stream.read(65536);position=0
        def whitespace():
            nonlocal position
            while True:
                while position<len(buffer) and buffer[position].isspace():position+=1
                if position<len(buffer):return
                old=len(buffer)-position;more()
                if len(buffer)==old:return
        whitespace();require(buffer[position:position+1]=='[','Canonical history array required');position+=1
        while True:
            whitespace()
            if buffer[position:position+1]==']':position+=1;break
            while True:
                try:value,end=decoder.raw_decode(buffer,position);break
                except json.JSONDecodeError:
                    prior=len(buffer)-position;more()
                    require(len(buffer)>prior and len(buffer)<=8_000_000,'Truncated/unbounded history row')
            position=end;count+=1;require(type(value) is dict and value.get('step')==count,'History order changed')
            if count<=step:
                if count>1:digest.update(b',')
                digest.update(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())
            del value
            whitespace();separator=buffer[position:position+1];position+=1
            if separator==']':break
            require(separator==',','Invalid history separator')
        whitespace();require(position==len(buffer) and not stream.read(1),'Trailing history content')
    require(count==completed and 0<step<=count,'Actual archived completed/history frontier differs')
    digest.update(b']');return digest.hexdigest()


@lru_cache(maxsize=2)
def source_context(lineage_path,lineage_sha):
    require(sha(lineage_path)==lineage_sha and sha(HERE/'tsc_bpe_acquisition_transition_v1.py')==SOURCE_AUDIT_SHA,
        'Original ROOT archive/source audit changed')
    lineage=read(lineage_path);validate_lineage_receipt(lineage,lineage['target_run_id']);artifacts=lineage['artifacts']
    require(lineage['selected_step']==SOURCE_STEP and artifacts['best_checkpoint']['sha256']==BEST_SHA,
        'Use actual immutable qualifying BEST6500; never latest6844 or BPE7000')
    for key in ('best_checkpoint','latest_checkpoint'):checked_record(artifacts[key],checkpoint=True)
    for key in ('run_config','training_history','selection','controller_status','runtime_completion','fresh_runtime',
                'DEV_summary','DEV_metadata',*('DEV_'+lang for lang in LANGUAGES)):checked_record(artifacts[key])
    config=read(artifacts['run_config']['path']);selection=read(artifacts['selection']['path']);identity=config['config_identity']
    require(config['run_id']==selection['run_id']==PARENT_RUN and canonical(identity)==lineage['scientific_config_identity_sha256']
        and identity['extended_tokenizer_sha256']=='b91d500326aca290ed709d629dab2bee56f298ea7323b94855b9f5ef0caeac75'
        and selection['selected_step']==SOURCE_STEP and selection['completed_steps']==lineage['completed_steps']
        and selection['accepted'] is True and selection['heldout_tests_evaluated'] is False,
        'Original8199 selected source config identity and actual qualified selection required')
    for name,digest in identity['source_sha256'].items():require(sha(HERE/name)==digest,'Parent source changed: '+name)
    validate_layout(identity['lora_layout']);recipe=dict(identity['training_arguments'])
    require(recipe['head_lr']==1e-5 and recipe['lora_lr']==3e-4 and recipe['new_token_lr_multiplier']==6
        and recipe['rank']==8 and recipe['alpha']==16 and recipe['batch_size']==8 and recipe['accumulation']==2
        and recipe['replay_kl']==10 and recipe['duration_kl']==.5 and recipe['feature_distillation']==1
        and recipe['sampling_mode']=='domain30' and recipe['schedule']=='constant'
        and recipe['steps']==44958 and recipe['eval_every_updates']==250
        and identity['domain_sampling_identity']['source_group_pool_counts']==POOL_COUNTS,
        'Preserved successful original207 recipe/full original target pool required')
    validate_closed_runtime(lineage,read(artifacts['controller_status']['path']),read(artifacts['runtime_completion']['path']),read(artifacts['fresh_runtime']['path']))
    prefix=_history_prefix(artifacts['training_history']['path'],SOURCE_STEP,lineage['completed_steps'])
    summary=read(artifacts['DEV_summary']['path']);base=read(HERE/'results/base_v3/summary.json')
    require(summary['validation_tr']['asr']['wer']==selection['turkish_validation_wer']
        and all(summary['validation_'+lang]['asr']['utterances']==50 for lang in LANGUAGES),'Actual selected source300 required')
    for lang in LANGUAGES[1:]:
        original=base['validation_'+lang]['asr']['wer']
        require(summary['validation_'+lang]['asr']['wer']<=original+max(.005,.1*original),'Selected old-language retention failed')
    return {'lineage_receipt':lineage,'lineage_receipt_record':record(lineage_path),'receipt':lineage,'config':config,
        'selection':selection,'parent_identity':identity,'source_recipe':recipe,'source_selected_step':SOURCE_STEP,'source_step':SOURCE_STEP,
        'source_checkpoint':artifacts['best_checkpoint'],'source_history_prefix_sha256':prefix,'source_summary':summary,'combined_baseline':base}


def preflight(args):
    require(getattr(args,'resume_uniform_full',None) is not None and getattr(args,'uniform_full_source_receipt',None) is not None,
        'Declared original207 uniform full-state fork required')
    path=scoped(args.uniform_full_source_receipt);receipt=validate_receipt(read(path),args.run_id,args.portuguese_seed)
    origin=receipt['source_lineage_receipt'];checked_record(origin)
    context=source_context(str(scoped(origin['path'])),origin['sha256'])
    require(receipt['artifacts']==context['lineage_receipt']['artifacts']
        and receipt['scientific_config_identity_sha256']==canonical(context['parent_identity'])
        and scoped(args.resume_uniform_full)==scoped(context['source_checkpoint']['path']), 'Same original byte archive/context required')
    for key,value in context['source_recipe'].items():
        expected='uniform' if key=='sampling_mode' else END_STEP if key=='steps' else value
        require(getattr(args,key)==expected,'Unapproved source recipe difference: '+key)
    for key in ('tsc_input_proof','tsc_training_approval','source_closure','tr_manifest','replay_manifest','data_v3_proof',
                'tr_frame_metadata','tr_frame_metadata_proof','reuse_input_audit','supplement'):
        before=context['config'].get(key);after=getattr(args,key,None)
        require((before is None and after is None) or (before is not None and after is not None and scoped(before)==scoped(after)),
            'Original data/source path changed: '+key)
    for key in ('portuguese_preparation_proof','portuguese_baseline_proof'):
        item=receipt[key];checked_record(item);require(scoped(getattr(args,key))==scoped(item['path']), 'Portuguese approved proof changed')
    result={**context,'receipt':receipt,'receipt_record':record(path),'source_record':record(path),
        'portuguese_preparation_proof':receipt['portuguese_preparation_proof'],'portuguese_baseline_proof':receipt['portuguese_baseline_proof']}
    validate_portuguese_inputs(args,result)
    return result


def validate_portuguese_inputs(args,context):
    """Bound319 TRAIN/50DEV proof plus separately measured original-base PT gate."""
    preparation=read(checked_record(context['portuguese_preparation_proof']))
    require(preparation.get('kind')=='Portuguese_FLEURS_retention_preparation_v1' and preparation.get('passed') is True
        and preparation.get('language')=='pt' and preparation.get('dataset_config')=='pt_br'
        and preparation.get('revision')=='70bb2e84b976b7e960aa89f1c648e09c59f894dd'
        and preparation.get('source_split_and_global_sentence_ID_isolation_passed') is True
        and preparation.get('new_PT_heldout_excluded_all_existing_FLEURS_TRAIN_ids_and_normalized_references') is True
        and preparation.get('new_PT_heldout_audio_checked_against_all_existing_TR_and_replay') is True
        and preparation.get('original_five_replay_rows_changed') is False
        and preparation.get('reserved_TEST_predictions_read_or_executed') is False
        and preparation.get('TRAIN_original_teacher',{}).get('vocab_size')==8193
        and preparation['TRAIN_original_teacher'].get('all_targets_below_original8193') is True,
        'Complete disjoint original-teacher Portuguese TRAIN/DEV preparation required')
    files=preparation['files']
    for key,count in (('replay_train_pt',319),('validation_pt',50),('test_pt',100)):
        require(files[key]['utterances']==count,'Full uncropped Portuguese manifests required')
    for flag,key in (('portuguese_replay_manifest','replay_train_pt'),('portuguese_dev_manifest','validation_pt')):
        require(scoped(getattr(args,flag))==scoped(files[key]['path']),'Portuguese manifest path changed: '+flag)
        checked_record(byte_record(files[key]))
    baseline=read(checked_record(context['portuguese_baseline_proof']))
    require(baseline.get('kind')=='Portuguese_retention_baselines_v1' and baseline.get('passed') is True
        and baseline.get('actual_GPU_decode') is True
        and baseline.get('base_revision')=='541d1f99c6b0c3cd0b11a95167540bb8edefd82b'
        and baseline.get('source_run_id')==PARENT_RUN and baseline.get('source_step')==SOURCE_STEP
        and baseline.get('source_checkpoint')==context['source_checkpoint']
        and baseline.get('source_receipt')==context['lineage_receipt_record']
        and baseline.get('pt_preparation')==context['portuguese_preparation_proof']
        and byte_record(baseline['fixed_PT_DEV_manifest'])==byte_record(files['validation_pt'])
        and baseline.get('fixed_PT_DEV_count')==50 and baseline.get('original_six_language_baseline_unchanged') is True
        and baseline.get('no_TEST_prediction_or_score_used') is True and baseline.get('no_optimizer_updates') is True,
        'Actual base/source6500 Portuguese proof required, never a CPU readiness receipt')
    checked_record(baseline['original_base_v3_summary'])
    for result in baseline['results'].values():
        for name in ('summary','metadata','predictions'):checked_record(result[name])
    score=baseline['baseline_PT']['validation_pt']['asr']
    require(score.get('utterances')==50 and type(score.get('word_errors')) is int
        and type(score.get('reference_words')) is int and score['reference_words']>0
        and type(score.get('wer')) in (int,float) and math.isfinite(score['wer'])
        and score['wer']==score['word_errors']/score['reference_words'],'Actual Portuguese baseline counts/ratio disagree')
    actual=baseline['champion6500_PT']['validation_pt']['asr'];gate=baseline['Portuguese_gate']
    limit=max(.005,.1*score['wer']);allowed=math.floor((score['wer']+limit)*score['reference_words']+1e-9)
    require(actual['utterances']==50 and actual['reference_words']==score['reference_words']
        and actual['wer']==actual['word_errors']/actual['reference_words']
        and gate['base_wer']==score['wer'] and gate['source_wer']==actual['wer']
        and gate['allowed_delta']==limit and gate['allowed_word_errors']==allowed
        and gate['actual_source_word_errors']==actual['word_errors'] and gate['passed'] is False
        and actual['word_errors']>allowed,'This declared recovery source must preserve its actual new PT gate failure')
    for lang in LANGUAGES:
        require(baseline['baseline_overlay']['validation_'+lang]==context['combined_baseline']['validation_'+lang],
            'Original six-language baseline overlay changed')
    require(baseline['baseline_overlay']['validation_pt']==baseline['baseline_PT']['validation_pt'],'Portuguese baseline overlay differs')
    context['combined_baseline']={**context['combined_baseline'],'validation_pt':baseline['baseline_PT']['validation_pt']}
    source_results=[result for result in baseline['results'].values()
        if read(result['summary']['path'])['validation_pt']['asr']==actual]
    require(len(source_results)==1,'Unique actual source6500 PT prediction/metadata records required')
    context.update(portuguese_replay_manifest=files['replay_train_pt']['path'],portuguese_dev_manifest=files['validation_pt']['path'],
        portuguese_validation_directory=str(scoped(files['validation_pt']['path']).parent),
        portuguese_preparation=preparation,portuguese_baseline=baseline,Portuguese_source_result=source_results[0],
        Portuguese_source_gate_passed=False,Portuguese_recovery_grace_new_updates=1000,initial_all_six_eligible_BEST=None)
    return context


def load_validate_source(model,state,context,*,torch_module):
    boundary=dict(validate_parent_state(state,context))
    # Reuse validation, not the acquisition-specific evidence labels.
    for key in ('first_acquisition_update','source_Adam_states_retained_dormant','new_active_Adam_tables'):boundary.pop(key,None)
    boundary.update(first_uniform_update=context['source_step']+1,active_Adam_states_to_restore=207,
        source_all207_Adam_moments_and_clocks_preserved=True,acquisition_or_new_rows_only_phase=False)
    expected={name:parameter for name,parameter in model.named_parameters() if parameter.requires_grad}
    require(len(expected)==207 and set(expected)==set(state['weights']),'Original head15/encoder192 parameter names required')
    for name,value in state['weights'].items():
        require(value.shape==expected[name].shape and value.dtype==torch_module.float32
            and bool(torch_module.isfinite(value).all()),'Source FP32/shape/finiteness changed: '+name)
    optimizer_audit=audit_source_optimizer(state,context,expected,torch_module=torch_module)
    from encoder_lora import encoder_lora_metadata
    require(encoder_lora_metadata(model)==state['lora_metadata'] and model.config.vocab_size==8199
        and model.config.blank_token_id==8192 and list(model.config.durations)==[0,1,2,3,4]
        and model.decoder.embedding.num_embeddings==8199 and model.joint.head.out_features==8204,
        'Original LoRA/tokenizer/blank/duration geometry required')
    require(sha(context['source_checkpoint']['path'])==context['source_checkpoint']['sha256'],'Immutable BEST6500 bytes changed')
    model.load_state_dict(state['weights'],strict=False)
    return {'state':state,'boundary':boundary,'optimizer_audit':optimizer_audit,'full207_optimizer_load_required':True,
        'three_table_acquisition_or_expansion_used':False}


def compare_source_development(model,processor,args,context,state,evaluate_fn,torch_module):
    torch=torch_module;validate_parent_state(state,context);folder=HERE/'results'/f'{args.run_id}_step_{SOURCE_STEP}'
    target=proof_path(args);require(not target.exists(),'Initial source proof must remain immutable')
    rng=(random.getstate(),torch.get_rng_state().clone(),[value.clone() for value in torch.cuda.get_rng_state_all()])
    modes=[(module,module.training) for module in model.modules()]
    try:candidate=evaluate_fn(model,processor,f'{args.run_id}_step_{SOURCE_STEP}',splits=('validation',))
    finally:
        random.setstate(rng[0]);torch.set_rng_state(rng[1]);torch.cuda.set_rng_state_all(rng[2])
        for module,training in modes:module.train(training)
    require(random.getstate()==rng[0] and torch.equal(torch.get_rng_state(),rng[1])
        and len(torch.cuda.get_rng_state_all())==len(rng[2])
        and all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),rng[2]))
        and all(module.training==training for module,training in modes),'Initial350 must restore all RNG/module modes')
    artifacts=context['receipt']['artifacts'];source_summary=context['source_summary']
    source_metadata=read(artifacts['DEV_metadata']['path']);candidate_metadata=read(folder/'metadata.json')
    require(source_metadata['model_fingerprint']==candidate_metadata['model_fingerprint']
        and source_metadata['manifests']==candidate_metadata['manifests'],'Original6 source model/generation/data fingerprint changed')
    comparisons={}
    for language in LANGUAGES:
        before=[json.loads(line) for line in Path(artifacts['DEV_'+language]['path']).read_text(encoding='utf-8').splitlines() if line.strip()]
        after=[json.loads(line) for line in (folder/f'validation_{language}.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        comparisons[language]=compare_development(before,after,source_summary['validation_'+language]['asr'],candidate['validation_'+language]['asr'],language)
        del before,after
    require('validation_pt' in candidate and candidate['validation_pt']['asr']['utterances']==50,
        'Fresh actual Portuguese source measurement required; no invented equality path')
    source_pt=context['Portuguese_source_result']
    before=[json.loads(line) for line in Path(source_pt['predictions']['path']).read_text(encoding='utf-8').splitlines() if line.strip()]
    after=[json.loads(line) for line in (folder/'validation_pt.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    comparisons['pt']=compare_development(before,after,context['portuguese_baseline']['champion6500_PT']['validation_pt']['asr'],
        candidate['validation_pt']['asr'],'pt')
    del before,after
    baseline=context['combined_baseline']
    require('validation_pt' in baseline,'Actual separate pinned Portuguese baseline required')
    for language in ('en','es','it','fr','de'):
        base=baseline['validation_'+language]['asr']['wer']
        require(candidate['validation_'+language]['asr']['wer']<=base+max(.005,.1*base),'Initial original-five retention gate failed: '+language)
    pt_score=candidate['validation_pt']['asr'];pt_allowed=context['portuguese_baseline']['Portuguese_gate']['allowed_word_errors']
    require(pt_score['word_errors']==context['portuguese_baseline']['champion6500_PT']['validation_pt']['asr']['word_errors']
        and pt_score['word_errors']>pt_allowed,'Initial PT recovery must reproduce the declared source failure, never claim all-six qualification')
    proof={'schema':1,'kind':PROOF_KIND,'passed':True,'run_id':args.run_id,'source_receipt':context['receipt_record'],
        'source_selected_step':SOURCE_STEP,'actual_GPU_decode':True,'before_first_uniform_update':True,
        'compared_fixed_DEV_utterances':300,'Portuguese_fixed_DEV_measured_utterances':50,
        'all_raw_transcripts_and_emitted_paths_exact':True,'all_six_ASR_counts_exact':True,'Portuguese50_raw_paths_and_ASR_exact':True,
        'all_five_original_retention_gates_passed':True,'all_six_retention_gates_passed':False,
        'Portuguese_source_word_errors':pt_score['word_errors'],'Portuguese_allowed_word_errors':pt_allowed,
        'Portuguese_recovery_grace_new_updates':1000,'initial_all_six_champion_claimed':False,
        'RNG_restored_after_added_evaluation':True,'module_training_modes_restored_after_added_evaluation':True,
        'old_source300_equal_new_Portuguese50_measurement_distinct':True,'sampler_drawn_during_check':False,
        'comparisons':comparisons,'candidate_summary':record(folder/'summary.json'),'candidate_metadata':record(folder/'metadata.json')}
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='utf-8') as stream:json.dump(proof,stream,ensure_ascii=False,indent=2,allow_nan=False)
    return proof


def release_admission_caches():
    before=source_context.cache_info().currsize;source_context.cache_clear()
    return {'uniform_source_admission_entries_before':before,'uniform_source_admission_entries_after':source_context.cache_info().currsize,
        'disk_artifacts_or_other_process_modified':False}
