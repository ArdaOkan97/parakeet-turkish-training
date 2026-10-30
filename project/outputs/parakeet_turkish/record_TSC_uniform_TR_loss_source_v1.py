"""Byte archive for inactive original8199 matched TR-only loss control.

Actual head-cycle completion and source CLOSED evidence precede .pt hashing.
Separate BEST19750/source25109 state is preserved; no actual admission now.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import shutil

from tsc_uniform_TR_loss_transition_v1 import (
    HERE,ROOT,RECEIPT_KIND,REVIEW_KIND,PARENT_PINS,PARENT_CONTROLLER,require,read,scoped,
    record,checked_record,canonical,_history_frontier,validate_receipt,source_context,validate_completed_head_choice,ENCODER_SCHEDULE_MODES,HEAD_LR_CHOICES,POLICY_CHOICE_KIND)
from record_TSC_uniform_majority_source_v1 import option
from TR_tdt_population_weighting_v1 import MODES, POPULATION_MEAN_U, POPULATION_REVIEW_SHA256


def inspect_closed(args):
    review_path=scoped(args.source_review);review=read(review_path)
    require(review.get('kind')==REVIEW_KIND and review.get('passed') is True
        and review.get('chosen_step')==25109 and review.get('completed_steps')==25109 and review.get('chosen_checkpoint_role')=='latest'
        and review.get('allow_matched_TR_loss_control') is True
        and review.get('source_counter_preserved')==23 and review.get('source_selected_BEST_preserved')==19750,
        'Actual delegated reviewed latest25109 inactive future matched head-rate review required')
    require(review.get('encoder_policy_selection_completed') is True
        and review.get('selected_encoder_policy')==args.encoder_lr_schedule,
        'Explicit actual encoder-policy selection required before any checkpoint bytes')
    policy=read(checked_record(review['encoder_policy_choice_review']))
    require(policy.get('kind')==POLICY_CHOICE_KIND and policy.get('passed') is True
        and policy.get('actual_curve_evidence_used') is True
        and policy.get('selected_encoder_lr_schedule')==args.encoder_lr_schedule,
        'Actual encoder-curve evidence must choose the future policy')

    require(review.get('head_policy_selection_completed') is True and review.get('selected_head_lr')==args.head_lr,
        'Actual head policy must be completed before checkpoint byte reads')
    head_policy=read(checked_record(review['head_policy_choice_review']))
    require(head_policy.get('kind')=='ROOT_completed_head_LR_policy_selection_v1' and head_policy.get('passed') is True
        and head_policy.get('actual_curve_evidence_used') is True and head_policy.get('full_matched_head_cycle_completed') is True
        and head_policy.get('selected_head_lr')==args.head_lr and head_policy.get('selected_encoder_lr_schedule')==args.encoder_lr_schedule,
        'Actual completed matched head policy required; inactive source cannot admit a choice')
    validate_completed_head_choice(head_policy)
    change=review['declared_change']
    require(change.get('parameter')=='TR_TDT_normalization' and change.get('active_mode')==args.tr_tdt_weighting
        and change.get('fixed_population_mean_U')==POPULATION_MEAN_U
        and change.get('active_head_lr')==args.head_lr and change.get('selected_encoder_policy')==args.encoder_lr_schedule
        and change.get('replay_TDT_and_preservation_unchanged') is True and change.get('natural_draws_unchanged') is True,
        'Only bound TR normalization relative to actual selected head/encoder policy required')
    population_record=record(HERE/'original8199_TR_loss_length_exposure_CPU_review_v1.json')
    require(population_record['sha256']==POPULATION_REVIEW_SHA256,'Pinned population evidence changed')
    closed=read(checked_record(review['actual_closed_source_review']))
    require(review['source_actual_owned_exit']['exit_code']==0 
        and closed['completed_steps']==25109 and closed['source_run_id']==review['source_run_id']
        and closed.get('passed') is True and closed.get('full_pass_complete') is True,'Actual source CLOSED0 required')
    a=review['artifacts'];status=read(checked_record(a['controller_status']));selection=read(checked_record(a['selection']))
    done=read(checked_record(a['runtime_completion']));config=read(checked_record(a['run_config']));run_id=review['source_run_id'];run=HERE/'runs'/run_id
    require(status['status']=='train_window_ended' and status['child_exit_code']==0 and status['resource_review_required'] is False
        and status['controller_sha256']==PARENT_PINS[PARENT_CONTROLLER] and status['completed_steps']==25109
        and done['status']=='complete' and done['sources_unchanged'] is True
        and selection['accepted'] is True and selection['selected_step']==19750 and selection['completed_steps']==25109
        and selection['consecutive_retention_failures']==23
        and scoped(selection['weights'])==scoped(a['source_best_checkpoint']['path'])
        and scoped(selection['latest_weights'])==scoped(a['chosen_checkpoint']['path'])
        and a['chosen_checkpoint']==a['latest_checkpoint'],
        'Actual coherent latest25109/counter23 and separate BEST19750 required before .pt bytes')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{2,110}',args.target_run_id) and args.target_run_id!=run_id,
        'New explicit inactive future matched head-rate scientific namespace required')
    prefix,_=_history_frontier(a['training_history']['path'],25109,25109)
    sources={key:scoped(item['path']) for key,item in a.items()}
    argv=status['worker_arguments']
    sources.update(original_native_ready=scoped(option(argv,'--native-backend-provenance')),
        indexed_TR_audit=run/'indexed_TR_audit.json',uniform_source_receipt=scoped(option(argv,'--uniform-full-source-receipt')),
        Portuguese_preparation_proof=scoped(option(argv,'--portuguese-preparation-proof')),
        Portuguese_baseline_proof=scoped(option(argv,'--portuguese-baseline-proof')),
        original_champion=scoped(config['source_champion_parked']['path']))
    receipt={'schema':1,'kind':RECEIPT_KIND,'status':'actual_CLOSED_selected_encoder_head_policy_TR_loss_control',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'source_run_id':run_id,'target_run_id':args.target_run_id,
        'chosen_checkpoint_role':'latest','chosen_step':25109,'completed_steps':25109,'explicit_coverage_rewind_clips':0,
        'ROOT_review':record(review_path),'original_full_epoch_source_receipt':record(config['majority_source_receipt']),
        'source_actual_owned_exit':review['source_actual_owned_exit'],'scientific_config_identity_sha256':canonical(config['config_identity']),
        'source_history_prefix_sha256':prefix,'source_chosen_is_qualified':False,'source_counter_preserved':23,'encoder_lr_schedule':args.encoder_lr_schedule,'head_lr':args.head_lr,
        'tr_tdt_weighting':args.tr_tdt_weighting,'tr_tdt_population_mean_U':POPULATION_MEAN_U,'population_review':population_record,
        'failed_branch_history_preserved':True,'retention_stop_policy':'monitor_only','creator':record(__file__),'GPU_launched':False,'tensor_deserialization_used':False}
    validate_receipt(receipt,args.target_run_id);return sources,receipt


def prepare(args):
    sources,receipt=inspect_closed(args)
    if args.validate_only:return {'passed':True,'source_step':25109,'separate_selected_BEST_step':19750,'checkpoint_bytes_read':False,
        'explicit_rewind_TR_clips':0,'encoder_lr_schedule':args.encoder_lr_schedule,'head_lr':args.head_lr,'encoder_schedule_end_step':43238,'ROOT_review':receipt['ROOT_review']}
    require(args.archive_dir is not None and args.receipt_output is not None,'Fresh archive/purpose receipt required')
    archive=scoped(args.archive_dir);output=scoped(args.receipt_output)
    require(archive.is_relative_to(ROOT/'work/closed-sources') and not archive.exists() and not output.exists(),'Fresh isolated archive/output required')
    original={key:record(path) for key,path in sources.items()};review=read(receipt['ROOT_review']['path'])
    for key,item in review['artifacts'].items():require(original[key]==item,'Actual reviewed closed bytes changed: '+key)
    archive.mkdir(parents=True);artifacts={};copies={};backups={};immutable=('original_native_ready','initial_source_equivalence','original_champion')
    for key,source in sources.items():
        if source not in copies:
            target=archive/(key+source.suffix);shutil.copy2(source,target);bound=record(target)
            require((bound['sha256'],bound['bytes'])==(original[key]['sha256'],original[key]['bytes']) and record(source)==original[key],
                'Source/archive changed: '+key)
            copies[source]=bound
        if key in immutable:artifacts[key]=original[key];backups[key]=copies[source]
        else:artifacts[key]=copies[source]
    require(record(sources['controller_status'])==original['controller_status'],'Closed status changed during archive')
    receipt.update(artifacts=artifacts,original_artifact_records=original,
        original_artifact_paths={key:str(path) for key,path in sources.items()},archived_immutable_scientific_tokens=backups,
        archive_copies_byte_identical=True,archive_directory=str(archive))
    provisional=archive/'receipt_validation.json'
    with provisional.open('x',encoding='utf-8') as stream:json.dump(receipt,stream,indent=2,ensure_ascii=False,allow_nan=False)
    source_context(str(provisional),args.target_run_id);source_context.cache_clear()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(receipt,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    return {'passed':True,'receipt':record(output),'source_step':25109,'GPU_launched':False,'tensor_deserialization_used':False}


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-review',type=Path,required=True)
    p.add_argument('--target-run-id',required=True);p.add_argument('--encoder-lr-schedule',choices=ENCODER_SCHEDULE_MODES,required=True);p.add_argument('--head-lr',type=float,choices=HEAD_LR_CHOICES,required=True);p.add_argument('--archive-dir',type=Path);p.add_argument('--receipt-output',type=Path)
    p.add_argument('--tr-tdt-weighting',choices=MODES,required=True)
    p.add_argument('--validate-only',action='store_true');return p


if __name__=='__main__':print(json.dumps(prepare(parser().parse_args()),indent=2))
