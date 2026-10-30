"""CPU audit/report for one frozen DEV-selected adaptive Parakeet experiment.

No inference or GPU work. Original pilot reports and helpers are read-only.
Valid early stops and selected step zero are supported. Audit success describes
consistent evidence; competitive Turkish quality is a separate measured result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re

from confidence import confidence_report, fit_temperature
from metrics import normalize_text, paired_bootstrap_delta, score_transcriptions
from verify_final_results import (Checks, export_manifest, file_sha, json_load,
                                  json_sha, rows_load, verify_export_fingerprint,
                                  verify_split)

OUTPUT=Path(__file__).resolve().parent
ROOT=OUTPUT.parents[1]
DATA=ROOT/'work/data'
LANGUAGES=('en','es','it','fr','de','tr')
BASE_REVISION='541d1f99c6b0c3cd0b11a95167540bb8edefd82b'
STRONG_REVISION='06f233fe06e710322aca913c1bc4249a0d71fce1'


def local_path(value,check,label):
    path=Path(value).resolve()
    check.require(path.is_relative_to(ROOT.resolve()) and path.exists(),label+': existing workspace path')
    return path


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def near(check,actual,expected,label,rtol=1e-9,atol=1e-8):
    check.require(finite(actual) and finite(expected) and math.isclose(actual,expected,rel_tol=rtol,abs_tol=atol),label)


def safe_run(value,check,label):
    check.require(type(value) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}',value) is not None,label+': safe run ID')
    return value


def check_gate(gate,baseline,check,label):
    check.require(finite(gate['turkish_wer']) and gate['turkish_wer']>=0,label+': finite Turkish DEV WER')
    deltas=gate['retention_wer_deltas']
    check.equal(set(deltas),set(LANGUAGES)-{'tr'},label+': all five retention languages')
    limits={lang:max(.005,.1*baseline['validation_'+lang]['asr']['wer']) for lang in deltas}
    check.require(all(finite(value) for value in deltas.values()),label+': finite retention deltas')
    expected=all(baseline['validation_'+lang]['asr']['wer']+deltas[lang]
                 <=baseline['validation_'+lang]['asr']['wer']+limits[lang] for lang in deltas)
    check.equal(gate['retention_gate_passed'],expected,label+': retention gate arithmetic (direct WER comparison avoids subtraction-boundary rounding)')
    for key in ('retention_wer_limits','retention_wer_tolerances'):
        if key in gate:check.equal(gate[key],limits,label+': exact retention limits')


def selected_history(run_id,selection,config,history,baseline,check):
    """Strict DEV-only checkpoint selection, including the warm-start step zero."""
    check.require(type(history) is list and [row['step'] for row in history]==list(range(1,len(history)+1)),run_id+': contiguous completed optimizer steps')
    configured=config['steps']
    check.require(type(configured) is int and configured>0 and len(history)<=configured,run_id+': valid early-stop completion')
    for row in history:
        check.require(all(finite(row[key]) for key in ('tdt_loss','replay_kl','replay_duration_kl','replay_feature_distillation','gradient_norm')),run_id+f": finite losses/gradient step {row['step']}")
    step=selection['selected_step']
    check.require(type(step) is int and 0<=step<=len(history),run_id+': selected completed step including zero')
    check.require(selection['accepted'] is True and selection['selection_split']==config['selection_split']=='FLEURS validation only',run_id+': accepted DEV-only selection')
    check.equal(selection['heldout_tests_evaluated'],False,run_id+': training selection never evaluated TEST')
    if 'completed_steps' in selection:check.equal(selection['completed_steps'],len(history),run_id+': recorded completed steps')
    if 'configured_steps' in selection:check.equal(selection['configured_steps'],configured,run_id+': recorded configured steps')
    eligible=[]
    initial_path=OUTPUT/'runs'/run_id/'initial_development.json'
    if initial_path.exists():
        initial=json_load(initial_path);check_gate(initial,baseline,check,run_id+': initial DEV')
        if initial['retention_gate_passed'] and initial['turkish_wer']<baseline['validation_tr']['asr']['wer']:
            eligible.append((0,initial['turkish_wer']))
    for row in history:
        if 'validation' not in row:continue
        gate=row['validation'];check_gate(gate,baseline,check,run_id+f": DEV step {row['step']}")
        if 'development_selection_only' in row:check.equal(row['development_selection_only'],True,run_id+': explicit DEV-only history')
        if gate['retention_gate_passed'] and gate['turkish_wer']<baseline['validation_tr']['asr']['wer']:
            eligible.append((row['step'],gate['turkish_wer']))
    check.require(bool(eligible),run_id+': qualifying DEV checkpoint exists')
    expected=min(eligible,key=lambda pair:pair[1])
    check.equal((step,selection['turkish_validation_wer']),expected,run_id+': earliest best qualifying DEV step/score')
    return {'selected_step':step,'completed_steps':len(history),'configured_steps':configured,
            'selected_unmerged_dev_wer':expected[1],'valid_early_stop':len(history)<configured,
            'stop_reason':selection.get('stop_early_reason','configured_steps_complete')}


def selected_development_predictions(run_id,selection,baseline,check):
    """Ground the selected unmerged score in its actual saved DEV predictions."""
    folder=OUTPUT/'results'/f"{run_id}_step_{selection['selected_step']}"
    summary=json_load(folder/'summary.json')
    metrics={};hashes={}
    for lang in LANGUAGES:
        path=folder/f'validation_{lang}.jsonl';predictions=rows_load(path)
        manifest=rows_load(DATA/f'validation_{lang}.jsonl')
        check.equal(len(predictions),50,run_id+': selected DEV complete '+lang+' count')
        check.equal([r['source_id'] for r in predictions],[r['source_id'] for r in manifest],run_id+': selected DEV ordered '+lang+' sources')
        check.require(all(row['reference']==source['text'] and row['wav_sha256']==source['wav_sha256']
                          and row['sentence_id']==source['sentence_id'] and row['split']=='validation'
                          for row,source in zip(predictions,manifest)),run_id+': selected DEV exact '+lang+' references/audio/sentence/split')
        scored=score_transcriptions(predictions,language=lang);scored.pop('details')
        check.equal(scored,summary['validation_'+lang]['asr'],run_id+': selected DEV actual '+lang+' scoring')
        metrics[lang]=scored;hashes[path.name]=file_sha(path)
    check.equal(metrics['tr']['wer'],selection['turkish_validation_wer'],run_id+': selected score grounded in actual unmerged DEV predictions')
    check.require(all(metrics[lang]['wer']<=baseline['validation_'+lang]['asr']['wer']+max(.005,.1*baseline['validation_'+lang]['asr']['wer'])
                      for lang in LANGUAGES[:-1]),run_id+': selected actual DEV predictions pass all five direct-WER gates')
    return {'summary_path':str(folder/'summary.json'),'summary_sha256':file_sha(folder/'summary.json'),'prediction_files_sha256':hashes}


def check_frozen_selection(frozen_path,state,baseline,check):
    frozen=json_load(frozen_path)
    run_id=safe_run(frozen['run_id'],check,'frozen winner')
    run_dir=OUTPUT/'runs'/run_id
    config_path=run_dir/'run_config.json';history_path=run_dir/'training_history.json'
    training_path=run_dir/'training_selection.json'
    check.require(training_path.exists(),'winner preserved pre-finalization training selection')
    config=json_load(config_path);history=json_load(history_path);training=json_load(training_path)
    check.equal(frozen['selection_split'],'FLEURS validation only','frozen selection is DEV only')
    stamp=datetime.fromisoformat(frozen['frozen_at_utc'])
    check.require(stamp.tzinfo is not None and stamp<=datetime.now(timezone.utc),'valid timezone-aware selection freeze timestamp')
    check.equal(config['run_id'],run_id,'frozen/config winner run')
    check.equal(config['base_revision'],BASE_REVISION,'pinned original Parakeet base revision')
    history_summary=selected_history(run_id,training,config,history,baseline,check)
    history_summary['selected_unmerged_development_evidence']=selected_development_predictions(run_id,training,baseline,check)
    check.equal(frozen['selected_step'],training['selected_step'],'frozen/checkpoint selected step')
    check.equal(frozen['wer'],training['turkish_validation_wer'],'frozen exact unmerged DEV selection WER')
    provenance=frozen['provenance']
    for key,path in (('selection_sha256',training_path),('run_config_sha256',config_path),('history_sha256',history_path)):
        check.equal(file_sha(path),provenance[key],'frozen winner '+key)
    weights=local_path(frozen['weights'],check,'frozen selected trainables')
    check.equal(weights,Path(training['weights']).resolve(),'frozen/training weights path')
    check.equal(file_sha(weights),provenance['weights_sha256'],'frozen selected weights SHA256')
    check.equal(provenance['selected_step'],frozen['selected_step'],'frozen provenance selected step')
    check.equal(provenance['code_sha256'],config['code_sha256'],'frozen training source SHA dictionary')
    check.equal(provenance.get('input_sha256',{}),config.get('input_sha256',{}),'frozen training input SHA dictionary')
    for name,expected in config['code_sha256'].items():
        check.require(Path(name).name==name,'safe frozen source filename')
        check.equal(file_sha(OUTPUT/name),expected,'frozen source bytes: '+name)
    for value,expected in config.get('input_sha256',{}).items():
        path=Path(value);path=path if path.is_absolute() else ROOT/path
        check.equal(file_sha(local_path(path,check,'frozen input')),expected,'frozen input bytes: '+str(value))
    check.equal(state['selection_split'],'FLEURS validation only','controller DEV-only selection policy')
    check.equal(state['best_development_candidate']['run_id'],run_id,'controller winner matches frozen winner')
    check.equal(state['best_development_candidate']['selected_step'],frozen['selected_step'],'controller/frozen selected step')
    check.equal(state['best_development_candidate']['wer'],frozen['wer'],'controller/frozen DEV score')
    check.equal(state['best_development_candidate']['provenance'],provenance,'controller/frozen complete source provenance')
    check.equal(state['controller_sha256'],file_sha(OUTPUT/'run_experiment_loop.py'),'controller source bytes stable')
    # Screens may finish early or fail. Only successful, provenance-accepted DEV
    # selections enter the controller's winner pool; no TEST metric enters it.
    pilot_path=OUTPUT/'runs/lora_20261010_01/training_selection.json'
    pilot=json_load(pilot_path)
    candidates=[('lora_20261010_01',pilot['turkish_validation_wer'],pilot['selected_step'])]
    for trial in state['trials']:
        if trial.get('status')!='complete' or trial.get('exit_code')!=0 or 'selection_provenance' not in trial:continue
        trial_id=safe_run(trial['run_id'],check,'trial')
        directory=OUTPUT/'runs'/trial_id
        path=directory/'training_selection.json'
        if not path.exists():path=directory/'selection.json'
        selected=json_load(path);prov=trial['selection_provenance']
        check.require(selected['accepted'] is True and selected['heldout_tests_evaluated'] is False,'accepted trial is training-only selection')
        for key,target in (('selection_sha256',path),('run_config_sha256',directory/'run_config.json'),('history_sha256',directory/'training_history.json')):
            check.equal(file_sha(target),prov[key],trial_id+': completed trial '+key)
        trial_config=json_load(directory/'run_config.json')
        selected_history(trial_id,selected,trial_config,json_load(directory/'training_history.json'),baseline,check)
        selected_development_predictions(trial_id,selected,baseline,check)
        check.equal(file_sha(local_path(selected['weights'],check,trial_id+': best weights')),prov['weights_sha256'],trial_id+': completed trial weights SHA')
        check.equal(trial['selection']['selected_step'],selected['selected_step'],trial_id+': controller trial selected step')
        check.equal(trial['selection']['turkish_validation_wer'],selected['turkish_validation_wer'],trial_id+': controller trial DEV score')
        candidates.append((trial_id,selected['turkish_validation_wer'],selected['selected_step']))
    expected=min(candidates,key=lambda value:value[1])
    check.equal((run_id,frozen['wer'],frozen['selected_step']),expected,'one frozen overall winner is best accepted DEV trial; no TEST selection')
    return frozen,run_dir,config,training,history,history_summary


def audit_training_rate(run_id,check):
    directory=OUTPUT/'runs'/run_id
    config=json_load(directory/'run_config.json');history=json_load(directory/'training_history.json')
    if not history:return {'run_id':run_id,'completed_updates':0,'audio_seconds':0.,'training_wall_seconds':0.,'audio_min_per_wall_min':None}
    if 'sampled_source_ids' not in history[-1]:
        return {'run_id':run_id,'available':False,'reason':'Original pilot history lacks new per-update source/audio timers; historical reconstruction is separately labeled.'}
    pool=json_load(directory/'training_pool.json')
    check.equal(pool['input_sha256'],config['input_sha256'],run_id+': exact training pool input provenance')
    by_id={row['source_id']:row for row in pool['turkish']+pool['replay']}
    check.equal(len(by_id),len(pool['turkish'])+len(pool['replay']),run_id+': unique accepted training-pool IDs')
    prepared={}
    for value in config['input_sha256']:
        path=Path(value);path=path if path.is_absolute() else ROOT/path
        check.equal(file_sha(local_path(path,check,run_id+': input manifest')),config['input_sha256'][value],run_id+': input manifest stable')
        if path.suffix=='.jsonl':
            for row in rows_load(path):
                if row.get('dataset_split')=='train':prepared[row['source_id']]=row
    heldout=[row for lang in LANGUAGES for split in ('validation','test') for row in rows_load(DATA/f'{split}_{lang}.jsonl')]
    heldout_ids={row['source_id'] for row in heldout};heldout_sentences={row['sentence_id'] for row in heldout if row.get('sentence_id') is not None}
    heldout_audio={row[key] for row in heldout for key in ('audio_sha256','wav_sha256') if row.get(key)}
    heldout_text={(row['language'],normalize_text(row['text'],row['language'])) for row in heldout}
    for kind,key in (('turkish','tr_pool_sha256'),('replay','replay_pool_sha256')):
        rows=pool[kind]
        digest=hashlib.sha256('\n'.join(sorted(row['source_id'] for row in rows)).encode()).hexdigest()
        check.equal(digest,config['config_identity'][key],run_id+': accepted '+kind+' pool identity')
        for row in rows:
            source=prepared.get(row['source_id'])
            check.require(source is not None,run_id+': accepted TRAIN source belongs to frozen manifest')
            check.require(all(row[name]==source[name] for name in ('text','language','duration','audio_sha256','wav_sha256','dataset_split')),run_id+': accepted source reference/audio/duration identity')
            check.require(row['dataset_split']=='train' and row['source_id'] not in heldout_ids
                          and (row.get('sentence_id') is None or row['sentence_id'] not in heldout_sentences)
                          and not any(row.get(name) in heldout_audio for name in ('audio_sha256','wav_sha256'))
                          and (row['language'],normalize_text(row['text'],row['language'])) not in heldout_text,
                          run_id+': accepted TRAIN source isolated from every DEV/TEST source/reference')
    effective=config['effective_batch'];tr_per=config['tr_clips_per_update'];replay_per=config['replay_clips_per_update']
    check.equal(effective,config['batch_size']*config['accumulation'],run_id+': effective batch calculation')
    check.equal(effective,tr_per+replay_per,run_id+': TR/replay clip mix calculation')
    from train_lora_batched import BalancedBucketSampler
    sampler=BalancedBucketSampler(pool['turkish'],pool['replay'],seed=config['seed'],batch_size=config['batch_size'],
                                  accumulation=config['accumulation'],replay_fraction=config['replay_fraction'],bucket_size=config['bucket_size'])
    audio=0.;language_audio={lang:0. for lang in LANGUAGES};seen_tr=set();last_wall=0.;language_counts={lang:0 for lang in LANGUAGES if lang!='tr'}
    for index,row in enumerate(history,1):
        check.equal(row['step'],index,run_id+': source history contiguous')
        ids=row['sampled_source_ids'];check.equal(len(ids),effective,run_id+f': complete clip presentations step{index}')
        expected_ids=[item['source_id'] for batch in sampler.next_batches() for item in batch]
        check.equal(ids,expected_ids,run_id+f': exact seed/bucket/cycle/uniform-language sample stream step{index}')
        samples=[by_id[value] for value in ids]
        check.equal(sum(x['language']=='tr' for x in samples),tr_per,run_id+': exact TR presentations per update')
        step_audio=sum(float(x['duration']) for x in samples);audio+=step_audio
        for sample in samples:
            lang=sample['language'];language_audio[lang]+=float(sample['duration'])
            if lang=='tr':seen_tr.add(sample['source_id'])
            else:language_counts[lang]+=1
        near(check,row['step_audio_seen_seconds'],step_audio,run_id+': exact sampled step audio')
        near(check,row['audio_seen_seconds'],audio,run_id+': exact cumulative sampled audio')
        check.require(finite(row['train_wall_seconds']) and row['train_wall_seconds']>last_wall,run_id+': positive monotone measured training intervals')
        last_wall=row['train_wall_seconds']
        near(check,row['audio_minutes_per_wall_minute'],audio/last_wall,run_id+': actual audio/wall ratio')
        check.equal(row['turkish_clips_seen'],index*tr_per,run_id+': cumulative TR presentations')
        check.equal(row['replay_clips_seen'],index*replay_per,run_id+': cumulative replay presentations')
        check.equal(row['turkish_unique_clips_seen'],len(seen_tr),run_id+': unique sampled TR count')
        check.equal(row['replay_language_clip_counts'],language_counts,run_id+': sampled replay language counts')
        near(check,row['turkish_epoch'],index*tr_per/len(pool['turkish']),run_id+': actual Turkish epoch')
        for lang in LANGUAGES:near(check,row['language_audio_seen_seconds'][lang],language_audio[lang],run_id+': exact '+lang+' sampled audio')
        near(check,row['turkish_audio_seen_seconds'],language_audio['tr'],run_id+': cumulative TR audio')
        near(check,row['replay_audio_seen_seconds'],audio-language_audio['tr'],run_id+': cumulative replay audio')
    return {'run_id':run_id,'available':True,'completed_updates':len(history),'audio_seconds':audio,'training_wall_seconds':last_wall,
            'audio_min_per_wall_min':audio/last_wall,'language_audio_seconds':language_audio,'unique_tr_clips':len(seen_tr),
            'training_pool_sha256':file_sha(directory/'training_pool.json'),'history_sha256':file_sha(directory/'training_history.json'),
            'timing_scope':config['training_throughput_timing']}


def audit_export(run_dir,config,frozen,check):
    selection=json_load(run_dir/'selection.json');pointer=json_load(run_dir/'adapted_results_path.json')
    provenance=json_load(run_dir/'finalization_provenance.json')
    check.require(selection['accepted'] and selection['merged'] and selection['heldout_tests_evaluated'],'completed merged winner and heldout inference')
    for value in (selection,provenance):
        check.equal(value['run_id'],frozen['run_id'],'finalization frozen winner run')
        check.equal(value['selected_step'],frozen['selected_step'],'finalization frozen winner step')
    checkpoint=local_path(selection['checkpoint'],check,'published merged checkpoint')
    check.equal(checkpoint,Path(pointer['checkpoint']).resolve(),'published pointer/checkpoint identity')
    check.equal(selection['final_results_tag'],pointer['tag'],'complete final result tag')
    adapted=local_path(Path(pointer['summary']).parent,check,'adaptive results folder')
    check.equal(adapted.name,pointer['tag'],'adaptive result folder/tag')
    candidate=local_path(provenance['candidate'],check,'merged candidate')
    check.equal(json_load(checkpoint/'finalization_provenance.json'),provenance,'published checkpoint provenance equals winner run')
    check.equal(export_manifest(checkpoint),provenance['candidate_files_sha256'],'complete published HF file hashes')
    check.equal(export_manifest(candidate),provenance['candidate_files_sha256'],'complete candidate HF file hashes')
    base=local_path(provenance['base_path'],check,'pinned original base')
    check.equal(base.name,BASE_REVISION,'export reconstructed from pinned base revision')
    check.equal(export_manifest(base),provenance['base_files_sha256'],'complete pinned base file hashes')
    check.equal(file_sha(run_dir/'run_config.json'),provenance['run_config_sha256'],'merged run config SHA')
    check.equal(file_sha(run_dir/'vocabulary_audit.json'),provenance['training_vocabulary_audit_sha256'],'merged vocabulary audit SHA')
    check.equal(config['lora_layout'],provenance['adapter_layout'],'merged training LoRA layout')
    check.equal(json_sha(provenance['adapter_layout']),provenance['adapter_layout_sha256'],'merged LoRA layout SHA')
    check.equal(config['code_sha256'],provenance['training_code_sha256'],'merged training source dictionary')
    check.equal(file_sha(OUTPUT/'finalize_lora.py'),provenance['finalizer_sha256'],'stable finalizer source bytes')
    check.require(provenance['merged_fp32'] and provenance['reconstruction']['all_trainable_keys_loaded']
                  and provenance['reconstruction']['frozen_state_loaded_from_pinned_base'],'complete fresh-base FP32 merge recorded')
    archived=local_path(selection['selected_trainables'],check,'archived selected trainables')
    check.equal(file_sha(archived),frozen['provenance']['weights_sha256'],'archived checkpoint equals frozen winning trainables')
    check.equal(provenance['selected_state_sha256'],frozen['provenance']['weights_sha256'],'merged frozen state SHA')
    check.equal(file_sha(local_path(provenance['selected_state_path'],check,'merge source trainables')),provenance['selected_state_sha256'],'merge source state SHA')
    import torch
    torch.set_num_threads(1)
    state=torch.load(archived,map_location='cpu',weights_only=True)
    check.equal((state['run_id'],state['step']),(frozen['run_id'],frozen['selected_step']),'actual trainable checkpoint run/step')
    check.equal(state['lora_metadata'],config['lora_layout'],'actual saved checkpoint LoRA layout')
    check.equal(sum(value.numel() for value in state['weights'].values()),config['trainable_parameters'],'saved trainable parameter count')
    check.equal(len(state['weights']),provenance['reconstruction']['trainable_key_count'],'saved trainable tensor count')
    check.equal(json_sha(sorted(state['weights'])),provenance['reconstruction']['trainable_keys_sha256'],'saved complete trainable key SHA')
    check.require(all(value.dtype==torch.float32 and bool(torch.isfinite(value).all()) for value in state['weights'].values()),'finite FP32 selected trainables')
    if 'history' in state:
        check.equal(state['history'],json_load(run_dir/'training_history.json')[:frozen['selected_step']],'saved selected checkpoint exact completed history prefix')
    del state
    return selection,pointer,provenance,checkpoint,adapted


def english_confidence(before,after):
    observations=[];groups={}
    for old,new in zip(before,after):
        a={w['reference_word_index']:w for w in old['alignment']['words'] if w['correct']}
        b={w['reference_word_index']:w for w in new['alignment']['words'] if w['correct']}
        cluster=old.get('sentence_id')
        if cluster is None:cluster=old['source_id']
        for index in sorted(a.keys()&b.keys()):
            x,y=a[index],b[index]
            if x['word']!=y['word']:raise ValueError('Shared correct English reference words differ')
            item={'source_id':old['source_id'],'reference_word_index':index,'word':x['word'],
                  'baseline_confidence':x['confidence'],'candidate_confidence':y['confidence'],
                  'baseline_conditional':x['conditional_nonblank_confidence'],'candidate_conditional':y['conditional_nonblank_confidence']}
            observations.append(item);groups.setdefault(cluster,[]).append(item)
    result={'shared_correct_word_count':len(observations),'pairing':'source_id + zero-based reference_word_index; correct in both transcripts',
            'low_candidate_correct_word_examples':sorted(observations,key=lambda row:row['candidate_confidence'])[:10]}
    if not observations:return result
    for name in ('baseline','candidate'):
        result[name]={'mean_raw_confidence':sum(w[name+'_confidence'] for w in observations)/len(observations),
                      'mean_conditional_nonblank_confidence':sum(w[name+'_conditional'] for w in observations)/len(observations)}
    chunks=[(len(words),sum(w['candidate_confidence']-w['baseline_confidence'] for w in words),
             sum(w['candidate_conditional']-w['baseline_conditional'] for w in words)) for words in groups.values()]
    rng=random.Random(20261010);samples=[[],[]]
    for _ in range(2000):
        selected=[rng.choice(chunks) for _ in chunks];n=sum(value[0] for value in selected)
        for index in (0,1):samples[index].append(100*sum(value[index+1] for value in selected)/n)
    def quantile(values,p):
        values=sorted(values);q=p*(len(values)-1);i=math.floor(q);j=math.ceil(q)
        return values[i]+(values[j]-values[i])*(q-i)
    result['paired_sentence_bootstrap']={'resamples':2000,'seed':20261010,'units':len(chunks),'confidence':.95,
        'raw_confidence_delta_percentage_points':100*(result['candidate']['mean_raw_confidence']-result['baseline']['mean_raw_confidence']),
        'raw_confidence_delta_ci_percentage_points':[quantile(samples[0],.025),quantile(samples[0],.975)],
        'conditional_confidence_delta_ci_percentage_points':[quantile(samples[1],.025),quantile(samples[1],.975)]}
    result['limitation']='Matched-correct-word scores describe confidence only; omit words becoming incorrect. Full emitted-word ECE/Brier and WER remain the quality measures.'
    return result


def audit_predictions(run_dir,selection,provenance,checkpoint,adapted,baseline,check):
    summary=json_load(adapted/'summary.json');metadata=json_load(adapted/'metadata.json')
    base_summary=json_load(baseline/'summary.json');base_meta=json_load(baseline/'metadata.json')
    expected={split+'_'+lang for split in ('validation','test') for lang in LANGUAGES}
    check.equal(set(metadata['manifests']),expected,'complete six-language DEV+TEST metadata')
    check.equal(json_sha(metadata['model_fingerprint']['identity']),metadata['model_fingerprint']['sha256'],'adaptive model fingerprint digest')
    for name,digest in metadata['model_fingerprint']['identity']['evaluation_code_sha256'].items():
        check.equal(file_sha(OUTPUT/name),digest,'stable final evaluation bytes: '+name)
    for name in ('metrics.py','confidence.py','runtime.py'):
        check.equal(metadata['model_fingerprint']['identity']['evaluation_code_sha256'][name],base_meta['model_fingerprint']['identity']['evaluation_code_sha256'][name],'paired original/candidate helper: '+name)
    check.equal(metadata['model_fingerprint']['identity']['inference_dtype'],base_meta['model_fingerprint']['identity']['inference_dtype'],'paired inference compute dtype')
    verify_export_fingerprint(checkpoint,metadata,provenance,check)
    merged=OUTPUT/'results'/selection['merged_development_tag']
    check.equal(json_load(merged/'metadata.json')['model_fingerprint'],metadata['model_fingerprint'],'same merged DEV and final model state/config/tokenizer/generation')
    comparisons=json_load(run_dir/'benchmark_comparison.json');calibration=json_load(run_dir/'confidence_calibration.json')
    check.equal(set(comparisons),set(LANGUAGES),'all six heldout paired comparisons')
    check.equal(set(calibration),set(LANGUAGES),'all six DEV-fitted temperatures')
    results={};saved={};dev_sentences=set();test_sentences=set()
    for lang in LANGUAGES:
        splits={}
        for split in ('validation','test'):
            after,asr=verify_split(adapted,metadata,summary,lang,split,check)
            before,base_asr=verify_split(baseline,base_meta,base_summary,lang,split,check)
            check.equal(metadata['manifests'][split+'_'+lang],base_meta['manifests'][split+'_'+lang],lang+': exact paired manifest metadata')
            splits[split]=(before,after,base_asr,asr)
        dev=splits['validation'];test=splits['test']
        dev_sentences.update(row['sentence_id'] for row in dev[1]);test_sentences.update(row['sentence_id'] for row in test[1])
        check.equal(file_sha(adapted/f'validation_{lang}.jsonl'),file_sha(merged/f'validation_{lang}.jsonl'),lang+': final DEV predictions copied from reloaded model')
        fitted=fit_temperature([row['alignment'] for row in dev[1]])
        check.equal(fitted,calibration[lang],lang+': durable DEV-only temperature fit')
        check.equal(fitted,summary['calibration_'+lang],lang+': recorded DEV-only temperature fit')
        calibrated=confidence_report([row['alignment'] for row in test[1]],fitted['temperature'])
        check.equal(calibrated,summary['test_'+lang]['calibrated_confidence'],lang+': TEST calibration uses frozen DEV temperature')
        base_fitted=fit_temperature([row['alignment'] for row in dev[0]])
        check.equal(base_fitted,base_summary['calibration_'+lang],lang+': unchanged original DEV temperature')
        base_calibrated=confidence_report([row['alignment'] for row in test[0]],base_fitted['temperature'])
        check.equal(base_calibrated,base_summary['test_'+lang]['calibrated_confidence'],lang+': unchanged original heldout calibration')
        bootstrap=paired_bootstrap_delta([row['reference'] for row in test[0]],[row['hypothesis'] for row in test[0]],
                                         [row['hypothesis'] for row in test[1]],language=lang,groups=[row['sentence_id'] for row in test[0]])
        check.equal(bootstrap,comparisons[lang],lang+': independently reproduced sentence-cluster TEST bootstrap')
        retention=test[3]['wer']-test[2]['wer']<=max(.005,.1*test[2]['wer'])
        results[lang]={'baseline_development_asr':dev[2],'development_asr':dev[3],
                      'baseline_test_asr':test[2],'test_asr':test[3],'paired_test_comparison':bootstrap,
                      'heldout_quality_gate_passed':test[3]['wer']<test[2]['wer'] if lang=='tr' else retention,
                      'dev_fitted_temperature':fitted['temperature'],'raw_test_confidence':summary['test_'+lang]['raw_confidence'],
                      'calibrated_test_confidence':calibrated,'baseline_raw_test_confidence':base_summary['test_'+lang]['raw_confidence'],
                      'baseline_calibrated_test_confidence':base_calibrated}
        saved[lang]=splits
    check.require(not dev_sentences.intersection(test_sentences),'global DEV/TEST sentence clusters separate calibration and evaluation')
    gate={'passed':True,'retention_gate_passed':True,'turkish_improved':results['tr']['development_asr']['wer']<base_summary['validation_tr']['asr']['wer'],
          'turkish_wer':results['tr']['development_asr']['wer'],'baseline_turkish_wer':base_summary['validation_tr']['asr']['wer'],
          'retention_wer_deltas':{},'retention_wer_tolerances':{},
          'policy':'each dev WER <= base + max(0.5 percentage point,10% relative base WER); Turkish WER improves'}
    for lang in LANGUAGES[:-1]:
        original=results[lang]['baseline_development_asr']['wer']
        gate['retention_wer_deltas'][lang]=results[lang]['development_asr']['wer']-original
        gate['retention_wer_tolerances'][lang]=max(.005,.1*original)
    gate['retention_gate_passed']=all(gate['retention_wer_deltas'][lang]<=gate['retention_wer_tolerances'][lang] for lang in LANGUAGES[:-1])
    gate['passed']=gate['retention_gate_passed'] and gate['turkish_improved']
    check.require(gate['passed'],'reloaded merged winner passes every DEV gate')
    check.equal(gate,selection['merged_development_gate'],'exact final merged development gate')
    check.equal(gate,json_load(run_dir/'merged_development_gate.json'),'durable merged development gate')
    english=english_confidence(saved['en']['test'][0],saved['en']['test'][1])
    return results,gate,english,saved,metadata


def strong_development(frozen,check):
    provenance=frozen['baseline_provenance'];path=local_path(provenance['summary_path'],check,'strong DEV summary')
    summary=json_load(path)
    check.equal(file_sha(path),provenance['summary_sha256'],'frozen strong DEV summary SHA')
    predictions=rows_load(path.parent/'validation_tr.jsonl')
    check.equal(file_sha(path.parent/'validation_tr.jsonl'),provenance['predictions_sha256'],'frozen strong DEV predictions SHA')
    check.equal(summary['revision'],STRONG_REVISION,'strong DEV exact pinned model revision')
    check.equal(summary['repository'],'openai/whisper-large-v3','strong DEV exact model repository')
    check.equal(summary['protocol'],provenance['protocol'],'frozen strong DEV protocol')
    check.equal(summary['metrics_code_sha256'],file_sha(OUTPUT/'metrics.py'),'strong DEV same original scorer bytes')
    check.equal(summary['script_sha256'],file_sha(OUTPUT/'benchmark_strong_tr.py'),'original strong DEV worker bytes remain unchanged')
    check.equal(file_sha(OUTPUT/'strong_baseline_dev_verification.json'),provenance['verification_sha256'],'frozen strong DEV independent audit SHA')
    metrics=score_transcriptions(predictions,language='tr');metrics.pop('details')
    check.equal(metrics,summary['metrics'],'strong DEV exact scoring recomputation')
    check.equal(metrics,provenance['metrics'],'frozen strong DEV exact metrics')
    manifest=rows_load(DATA/'validation_tr.jsonl')
    check.equal([r['source_id'] for r in predictions],[r['source_id'] for r in manifest],'strong DEV exact fixed source order')
    check.require(all(row['reference']==source['text'] and row['wav_sha256']==source['wav_sha256'] for row,source in zip(predictions,manifest)),'strong DEV exact source references/WAV pairing')
    check.equal(frozen['competitive_threshold_wer'],metrics['wer']+.02,'provisional target is strong DEV WER plus2pp')
    return summary


def strong_test(path,frozen_path,frozen,saved,strong_dev,check):
    if path is None:return {'available':False,'reason':'No completed frozen-winner strong TEST comparison was supplied.'}
    comparison=json_load(local_path(path,check,'strong heldout comparison'))
    check.require(comparison['passed'] is True and comparison['status']=='complete','completed verified strong heldout comparison')
    check.equal(comparison['run_id'],frozen['run_id'],'strong TEST comparison frozen winner run')
    check.equal(comparison['selected_step'],frozen['selected_step'],'strong TEST comparison frozen winner step')
    check.equal(comparison['frozen_selection_sha256'],file_sha(frozen_path),'strong TEST comparison exact immutable frozen selection SHA')
    folder=OUTPUT/'results/strong_whisper_large_v3_greedy_test'
    summary=json_load(folder/'summary.json');predictions=rows_load(folder/'test_tr.jsonl')
    check.require(summary['status']=='complete' and summary['split']=='test' and summary['language']=='tr','complete strong Turkish TEST run')
    check.equal((summary['repository'],summary['revision']),('openai/whisper-large-v3',STRONG_REVISION),'strong TEST pinned official model')
    check.equal(summary['protocol'],strong_dev['protocol'],'same predeclared strong DEV/TEST decoding protocol')
    check.equal(summary['model_files_sha256'],strong_dev['model_files_sha256'],'strong DEV/TEST identical pinned model assets')
    check.equal((comparison['baseline_repository'],comparison['baseline_revision']),('openai/whisper-large-v3',STRONG_REVISION),'strong comparison model repository/revision')
    check.equal(comparison['baseline_protocol'],summary['protocol'],'strong comparison same fixed decoder protocol')
    check.equal((comparison['selection_split'],comparison['heldout_split'],comparison['language'],comparison['utterances']),
                ('FLEURS validation only','test','tr',100),'strong comparison DEV-only selection and fixed100TR TEST scope')
    check.equal(summary['metrics_code_sha256'],file_sha(OUTPUT/'metrics.py'),'strong TEST shared scorer bytes')
    check.equal(summary['predictions_sha256'],file_sha(folder/'test_tr.jsonl'),'strong TEST prediction bytes')
    check.equal(summary['script_sha256'],file_sha(OUTPUT/'benchmark_strong_tr_test.py'),'strong TEST worker source bytes')
    provenance=comparison['provenance']
    expected_paths={'strong_summary':folder/'summary.json','strong_predictions':folder/'test_tr.jsonl',
                    'selected_summary':Path(json_load(OUTPUT/'runs'/frozen['run_id']/'adapted_results_path.json')['summary']),
                    'selected_predictions':Path(json_load(OUTPUT/'runs'/frozen['run_id']/'adapted_results_path.json')['summary']).parent/'test_tr.jsonl'}
    for key,target in expected_paths.items():
        check.equal(Path(provenance[key+'_path']).resolve(),target.resolve(),'strong comparison exact '+key+' path')
        check.equal(provenance[key+'_sha256'],file_sha(target),'strong comparison exact '+key+' bytes')
    check.equal(provenance['test_script_sha256'],file_sha(OUTPUT/'benchmark_strong_tr_test.py'),'strong comparison worker SHA')
    check.equal(provenance['metrics_code_sha256'],file_sha(OUTPUT/'metrics.py'),'strong comparison scorer SHA')
    check.equal(provenance['verified_dev_summary_sha256'],frozen['baseline_provenance']['summary_sha256'],'strong TEST comparison same verified DEV baseline')
    check.equal(provenance['verified_dev_script_sha256'],strong_dev['script_sha256'],'strong comparison verified DEV source SHA')
    input_audit=summary['input_audit']
    check.equal(input_audit,provenance['input_audit'],'strong comparison exact input audit')
    for key,target in (('manifest',DATA/'test_tr.jsonl'),('paired_baseline_test',OUTPUT/'results/base_v3/test_tr.jsonl')):
        check.equal(Path(input_audit[key+'_path']).resolve(),target.resolve(),'strong TEST exact '+key+' path')
        check.equal(input_audit[key+'_sha256'],file_sha(target),'strong TEST exact '+key+' bytes')
    freeze=comparison['frozen_selection']
    check.equal(freeze['sha256'],file_sha(frozen_path),'strong comparison frozen selection digest')
    check.equal(Path(freeze['path']).resolve(),Path(frozen_path).resolve(),'strong comparison frozen selection path')
    check.equal((freeze['run_id'],freeze['selected_step']),(frozen['run_id'],frozen['selected_step']),'strong comparison frozen run/step')
    for key,target in (('selected_results_marker',OUTPUT/'runs'/frozen['run_id']/'adapted_results_path.json'),
                       ('selected_final_selection',OUTPUT/'runs'/frozen['run_id']/'selection.json'),
                       ('selected_finalization_provenance',OUTPUT/'runs'/frozen['run_id']/'finalization_provenance.json')):
        check.equal(freeze[key+'_sha256'],file_sha(target),'strong comparison exact '+key+' bytes')
    lock_path=ROOT/'work/strong_baseline_model.lock.json';lock=json_load(lock_path)
    check.equal((lock['repository'],lock['revision']),('openai/whisper-large-v3',STRONG_REVISION),'strong pinned model lock identity')
    check.equal(file_sha(lock_path),summary['model_lock_sha256'],'strong TEST actual model lock bytes')
    check.equal(summary['model_lock_sha256'],provenance['model_lock_sha256'],'strong comparison model lock SHA')
    check.equal(summary['model_files_sha256'],provenance['model_files_sha256'],'strong comparison complete pinned asset hashes')
    model_path=local_path(lock['local_path'],check,'strong pinned model folder')
    for name,expected in summary['model_files_sha256'].items():
        check.require(Path(name).name==name,'strong safe asset filename')
        check.equal(file_sha(model_path/name),expected,'strong actual pinned model asset: '+name)
    check.require(summary['all_full_prefixes_verified'] and summary['all_terminal_eos_verified'],'strong TEST full prefix/EOS summary checks')
    before=summary['frozen_selection_before_inference']
    check.equal(file_sha(local_path(before['path'],check,'strong baseline original selection freeze')),before['sha256'],'strong baseline original pre-inference freeze bytes')
    before_stamp=datetime.fromisoformat(before['frozen_at_utc']);created=datetime.fromisoformat(summary['created_at_utc'])
    check.require(before_stamp.tzinfo is not None and created.tzinfo is not None and before_stamp<=created,
                  'strong baseline inference follows its recorded timezone-aware selection freeze')
    adapted=saved['tr']['test'][1];manifest=rows_load(DATA/'test_tr.jsonl')
    check.equal(len(predictions),100,'strong TEST exactly100 fixed recordings')
    check.equal([r['source_id'] for r in predictions],[r['source_id'] for r in manifest],'strong TEST exact source IDs/order')
    for row,source in zip(predictions,manifest):
        check.require(all(row[key]==source['text' if key=='reference' else key] for key in
                          ('reference','sentence_id','dataset_revision','audio_sha256','wav_sha256','duration','language')),'strong TEST exact human reference/audio provenance')
        check.equal(row['split'],'test','strong TEST actual split')
        tokens=row['generated_token_ids']
        check.require(row['eos_emitted'] is True and type(tokens) is list and all(type(value) is int for value in tokens)
                      and tokens[:4]==[50258,50268,50360,50364] and tokens[-1:]==[50257] and tokens.count(50257)==1,
                      'strong TEST forced Turkish/transcribe/no-timestamp prefix and terminal EOS')
    metrics=score_transcriptions(predictions,language='tr');metrics.pop('details')
    candidate=score_transcriptions(adapted,language='tr');candidate.pop('details')
    check.equal(metrics,summary['metrics'],'strong TEST exact corpus WER/CER counters')
    check.equal(metrics,comparison['baseline_asr'],'comparison strong TEST exact metrics')
    check.equal(candidate,comparison['adapted_asr'],'comparison candidate TEST exact metrics')
    paired=paired_bootstrap_delta([row['reference'] for row in predictions],[row['hypothesis'] for row in predictions],
                                  [row['hypothesis'] for row in adapted],language='tr',groups=[row['sentence_id'] for row in predictions])
    check.equal(paired,comparison['paired_sentence_bootstrap'],'strong TEST exact paired sentence-cluster bootstrap')
    near(check,comparison['wer_delta_percentage_points'],paired['wer']['delta_percentage_points'],'strong TEST WER gap arithmetic')
    near(check,comparison['cer_delta_percentage_points'],paired['cer']['delta_percentage_points'],'strong TEST CER gap arithmetic')
    return {'available':True,'comparison_path':str(Path(path).resolve()),'comparison_sha256':file_sha(path),
            'baseline_asr':metrics,'candidate_asr':candidate,'paired_sentence_bootstrap':paired,
            'candidate_minus_strong_wer_gap_percentage_points':paired['wer']['delta_percentage_points'],
            'small_test_within_two_percentage_points':candidate['wer']<=metrics['wer']+.02,
            'scope':'100 fixed Turkish TEST recordings after one DEV winner was frozen; model/protocol sizes differ; no published-normalizer/SOTA/general-domain claim.'}


def audit(args,check):
    frozen_path=local_path(args.frozen_selection,check,'frozen selection')
    loop_dir=frozen_path.parent;state_path=loop_dir/'loop_status.json';state=json_load(state_path)
    loop_id=safe_run(state['loop_id'],check,'adaptive loop')
    check.equal(loop_dir.name,loop_id,'frozen selection is loop-specific')
    baseline=local_path(args.baseline_results,check,'original multilingual baseline')
    check.equal(baseline,(OUTPUT/'results/base_v3').resolve(),'same unchanged original-v3 benchmark')
    base_summary=json_load(baseline/'summary.json')
    frozen,run_dir,config,training,history,history_summary=check_frozen_selection(frozen_path,state,base_summary,check)
    strong_dev=strong_development(frozen,check)
    selection,pointer,provenance,checkpoint,adapted=audit_export(run_dir,config,frozen,check)
    languages,gate,english,saved,metadata=audit_predictions(run_dir,selection,provenance,checkpoint,adapted,baseline,check)
    check.equal(state['final_selection'],selection,'controller actual finalization selection matches durable winner')
    check.equal(state['merged_turkish_development_wer'],gate['turkish_wer'],'controller merged DEV score')
    competitive=gate['passed'] and gate['turkish_wer']<=frozen['competitive_threshold_wer']
    check.equal(state['competitive_on_development'],competitive,'controller competitive flag uses reloaded merged DEV gates')
    rates=[]
    for trial in state['trials']:
        path=OUTPUT/'runs'/trial['run_id']/'training_history.json'
        if path.exists():rates.append(audit_training_rate(trial['run_id'],check))
    measured=[r for r in rates if r.get('available',True) and r['training_wall_seconds']>0]
    total_audio=sum(r['audio_seconds'] for r in measured);total_wall=sum(r['training_wall_seconds'] for r in measured)
    actual={'trial_rates':rates,'audio_seconds':total_audio,'training_wall_seconds':total_wall,
            'audio_min_per_wall_min':total_audio/total_wall if total_wall else None,
            'scope':'Sum actual logged sampled-audio seconds / measured sampling/data/cache/STFT/H2D/forward/backward/Adam seconds across adaptive trials; excludes DEV/checkpoint/history serialization/startup.'}
    profile=json_load(OUTPUT/'batching_speed_comparison.json');profile_audit=json_load(OUTPUT/'batching_selection_independent_audit.json')
    check.require(profile_audit['passed'] is True,'independently audited selected quartet throughput')
    check.equal(profile_audit['selection_audit']['sha256'],file_sha(OUTPUT/'batching_speed_comparison.json'),'quartet throughput report remains independently verified')
    check.equal(state['profile_sha256'],file_sha(OUTPUT/'batching_speed_comparison.json'),'controller selected exact measured speed report')
    check.equal(state['selected_configuration'],profile['selected_configuration'],'controller uses verified B1 speed configuration')
    strong=strong_test(args.strong_test_comparison,frozen_path,frozen,saved,strong_dev,check)
    if args.strong_test_comparison is not None:
        check.equal(state['strong_test_comparison_sha256'],file_sha(args.strong_test_comparison),'controller strong TEST comparison bytes')
    return {'loop_id':loop_id,'run_id':frozen['run_id'],'selected_step':frozen['selected_step'],
            'frozen_selection_sha256':file_sha(frozen_path),'frozen_selection_path':str(frozen_path),
            'controller_state_sha256_at_audit':file_sha(state_path),'controller_state_snapshot_status':state['status'],
            'winner_history':history_summary,'checkpoint':str(checkpoint),'checkpoint_state_sha256':metadata['model_fingerprint']['identity']['model_state_sha256'],
            'candidate_export_files_sha256':provenance['candidate_files_sha256'],'adapted_results_path':str(adapted),
            'languages':languages,'merged_development_gate':gate,'competitive_on_development':competitive,
            'strong_development_wer':strong_dev['metrics']['wer'],'competitive_development_threshold_wer':frozen['competitive_threshold_wer'],
            'strong_test_comparison':strong,'english_confidence':english,'actual_adaptive_training_rate':actual,
            'matched_quartet_speed':{'before_audio_min_per_wall_min':profile['before_audio_min_per_wall_min'],
                                    'after_audio_min_per_wall_min':profile['after_audio_min_per_wall_min'],
                                    'throughput_ratio':profile['throughput_ratio'],'scope':profile['timing_scope'],
                                    'independent_audit_sha256':file_sha(OUTPUT/'batching_selection_independent_audit.json')},
            'heldout_old_language_gates_passed':all(languages[lang]['heldout_quality_gate_passed'] for lang in LANGUAGES[:-1]),
            'turkish_heldout_improved_over_original_v3':languages['tr']['heldout_quality_gate_passed'],
            'quality_scope':'Small fixed FLEURS pilot:50 DEV +100 TEST recordings per language. Multiple adaptive trials reuse DEV; TEST is evaluated only for one frozen winner. WER improvement alone does not establish useful Turkish ASR.'}


def text_report(result):
    if not result['passed']:
        return 'Adaptive final evidence audit failed.\n\n'+result['failed_check']+'\nNo verified final quality conclusion is issued.\n'
    history=result['winner_history'];tr=result['languages']['tr'];strong=result['strong_test_comparison']
    lines=[f"Adaptive Parakeet Turkish ASR: {result['run_id']} (selected step{result['selected_step']}, completed{history['completed_steps']}/{history['configured_steps']} updates).",
           'Saved-source, frozen DEV selection, merged-checkpoint, prediction, score and confidence evidence passed the independent CPU audit.',
           f"Turkish WER: DEV {100*tr['development_asr']['wer']:.2f}%; TEST {100*tr['test_asr']['wer']:.2f}% (original v3 {100*tr['baseline_test_asr']['wer']:.2f}%).",
           f"Strong pinned Whisper DEV WER {100*result['strong_development_wer']:.2f}%; provisional +2pp DEV target {100*result['competitive_development_threshold_wer']:.2f}%; target met: {result['competitive_on_development']}."]
    if strong['available']:
        paired=strong['paired_sentence_bootstrap']['wer'];ci=paired['ci_percentage_points']
        lines.append(f"Same100 Turkish TEST clips: Whisper WER {100*strong['baseline_asr']['wer']:.2f}%; Parakeet minus Whisper {paired['delta_percentage_points']:+.2f}pp (paired sentence-bootstrap95% CI {ci[0]:+.2f} to {ci[1]:+.2f}pp).")
        if strong['candidate_asr']['wer']>strong['baseline_asr']['wer']+.02:
            lines.append('The Turkish model remains behind the measured strong baseline; workflow completion and relative improvement do not make it competitive.')
        else:lines.append('The small Turkish TEST set is within2pp of this fixed greedy Whisper reference; broader useful-quality claims require more speech and domains.')
    else:lines.append('Strong Turkish heldout comparison was not available; competitive TEST quality is unverified.')
    if tr['test_asr']['wer']>=.5:lines.append('Turkish WER is still high: this is a research checkpoint, not useful Turkish transcription accuracy.')
    lines.extend(['','Existing-language TEST WER (original -> candidate; delta95% sentence-bootstrap CI):'])
    for lang in LANGUAGES[:-1]:
        item=result['languages'][lang];paired=item['paired_test_comparison']['wer'];ci=paired['ci_percentage_points']
        lines.append(f"{lang.upper()}: {100*item['baseline_test_asr']['wer']:.2f}% -> {100*item['test_asr']['wer']:.2f}%; {paired['delta_percentage_points']:+.2f}pp [{ci[0]:+.2f}, {ci[1]:+.2f}]; retention gate {item['heldout_quality_gate_passed']}.")
    en=result['languages']['en'];raw=en['raw_test_confidence'];cal=en['calibrated_test_confidence'];base=en['baseline_raw_test_confidence']
    lines.extend(['',f"English correct-word raw confidence: original {100*base['correct_word_mean_confidence']:.3f}% -> candidate {100*raw['correct_word_mean_confidence']:.3f}%; correctly decoded words below0.5: {100*raw['correct_words_below_0_5_fraction']:.3f}%.",
                  f"English emitted-word raw ECE {100*raw['ece']:.3f}pp, Brier {raw['brier']:.6f}; DEV-fitted temperature {en['dev_fitted_temperature']:g}, TEST ECE {100*cal['ece']:.3f}pp, Brier {cal['brier']:.6f}.",
                  'Word confidence is the geometric mean of actual emitted token probabilities including blank competition; duration probabilities are separate. Calibration uses DEV only and leaves transcripts/WER unchanged. Deletions have no emitted confidence.'])
    speed=result['matched_quartet_speed'];actual=result['actual_adaptive_training_rate']
    lines.extend(['',f"Measured fixed TRAIN quartet speed: {speed['before_audio_min_per_wall_min']:.3f} -> {speed['after_audio_min_per_wall_min']:.3f} audio min/wall min ({speed['throughput_ratio']:.3f}x). B1/accumulation4 uses the faster exact loss; unresolved B2 was rejected."])
    if actual['audio_min_per_wall_min'] is not None:
        lines.append(f"Actual adaptive trials: {actual['audio_seconds']/60:.2f} audio minutes / {actual['training_wall_seconds']/60:.2f} measured training minutes = {actual['audio_min_per_wall_min']:.3f} audio min/wall min.")
    else:lines.append('No completed adaptive training intervals were available for a new-run rate.')
    lines.extend(['Quartet and adaptive rates have different scopes. Actual intervals include input/cache/frontend/GPU/optimizer work and exclude DEV, startup and checkpoint/history serialization.',
                  '',result['quality_scope'],'Confidence changes on shared correct words are descriptive and omit words that become incorrect; ECE/Brier and WER determine reliability. No new GPU inference occurs in this audit.'])
    return '\n'.join(lines)+'\n'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(value,encoding='utf-8');temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-selection',type=Path,required=True)
    parser.add_argument('--strong-test-comparison',type=Path)
    parser.add_argument('--baseline-results',type=Path,default=OUTPUT/'results/base_v3')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();check=Checks()
    directory=(args.output_dir or args.frozen_selection.parent).resolve()
    if not directory.is_relative_to(OUTPUT.resolve()):parser.error('output-dir must stay within experiment outputs')
    result={'schema':1,'passed':False,'status':'failed','gpu_used':False,'verified_at_utc':datetime.now(timezone.utc).isoformat(),
            'auditor_sha256':file_sha(Path(__file__)),'protected_helper_sha256':file_sha(OUTPUT/'verify_final_results.py')}
    try:
        result.update(audit(args,check));result.update(passed=True,status='complete')
    except Exception as error:
        result.update(failed_check=str(error),error_type=type(error).__name__)
    result['checks']=check.items;result['passed_check_count']=len(check.items)
    report=text_report(result);result['human_report_sha256']=hashlib.sha256(report.encode('utf-8')).hexdigest()
    result['human_report_path']=str(directory/'adaptive_final_report.txt')
    for target in {directory,OUTPUT}:
        write(target/'adaptive_final_report.txt',report)
        write(target/'adaptive_results_verification.json',json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:result.get(key) for key in ('passed','status','run_id','loop_id','selected_step','frozen_selection_sha256','passed_check_count','failed_check')},indent=2))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
