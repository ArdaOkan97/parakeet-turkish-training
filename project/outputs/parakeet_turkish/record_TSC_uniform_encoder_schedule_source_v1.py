"""Byte archive for matched encoder schedule continuation from coherent reviewed latest25109.

Actual source CLOSED evidence precedes .pt hashing. Separate qualified BEST19750
and original numeric/feature lineage remain archived without any rewind.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import shutil

from tsc_uniform_encoder_schedule_transition_v1 import (
    HERE,ROOT,RECEIPT_KIND,REVIEW_KIND,PARENT_PINS,PARENT_CONTROLLER,require,read,scoped,
    record,checked_record,canonical,_history_frontier,validate_receipt,source_context,ENCODER_SCHEDULE_MODES)
from record_TSC_uniform_majority_source_v1 import option


def inspect_closed(args):
    review_path=scoped(args.source_review);review=read(review_path)
    require(review.get('kind')==REVIEW_KIND and review.get('passed') is True
        and review.get('chosen_step')==25109 and review.get('completed_steps')==25109 and review.get('chosen_checkpoint_role')=='latest'
        and review.get('allow_matched_encoder_schedule_control') is True
        and review.get('source_counter_preserved')==23 and review.get('source_selected_BEST_preserved')==19750,
        'Actual delegated reviewed latest25109 matched encoder schedule review required')
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
        'New explicit matched encoder schedule scientific namespace required')
    prefix,_=_history_frontier(a['training_history']['path'],25109,25109)
    sources={key:scoped(item['path']) for key,item in a.items()}
    argv=status['worker_arguments']
    sources.update(original_native_ready=scoped(option(argv,'--native-backend-provenance')),
        indexed_TR_audit=run/'indexed_TR_audit.json',uniform_source_receipt=scoped(option(argv,'--uniform-full-source-receipt')),
        Portuguese_preparation_proof=scoped(option(argv,'--portuguese-preparation-proof')),
        Portuguese_baseline_proof=scoped(option(argv,'--portuguese-baseline-proof')),
        original_champion=scoped(config['source_champion_parked']['path']))
    receipt={'schema':1,'kind':RECEIPT_KIND,'status':'actual_CLOSED_matched_encoder_schedule',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'source_run_id':run_id,'target_run_id':args.target_run_id,
        'chosen_checkpoint_role':'latest','chosen_step':25109,'completed_steps':25109,'explicit_coverage_rewind_clips':0,
        'ROOT_review':record(review_path),'original_full_epoch_source_receipt':record(config['majority_source_receipt']),
        'source_actual_owned_exit':review['source_actual_owned_exit'],'scientific_config_identity_sha256':canonical(config['config_identity']),
        'source_history_prefix_sha256':prefix,'source_chosen_is_qualified':False,'source_counter_preserved':23,'encoder_lr_schedule':args.encoder_lr_schedule,
        'failed_branch_history_preserved':True,'retention_stop_policy':'monitor_only','creator':record(__file__),'GPU_launched':False,'tensor_deserialization_used':False}
    validate_receipt(receipt,args.target_run_id);return sources,receipt


def prepare(args):
    sources,receipt=inspect_closed(args)
    if args.validate_only:return {'passed':True,'source_step':25109,'separate_selected_BEST_step':19750,'checkpoint_bytes_read':False,
        'explicit_rewind_TR_clips':0,'encoder_lr_schedule':args.encoder_lr_schedule,'encoder_schedule_end_step':43238,'ROOT_review':receipt['ROOT_review']}
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
    p.add_argument('--target-run-id',required=True);p.add_argument('--encoder-lr-schedule',choices=ENCODER_SCHEDULE_MODES,required=True);p.add_argument('--archive-dir',type=Path);p.add_argument('--receipt-output',type=Path)
    p.add_argument('--validate-only',action='store_true');return p


if __name__=='__main__':print(json.dumps(prepare(parser().parse_args()),indent=2))
