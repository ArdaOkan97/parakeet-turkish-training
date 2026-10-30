"""Purpose receipt for matched data arms, reusing an already sealed BEST40750.

No model/checkpoint payload is read or duplicated here. The actual CLOSED0
preservation record already binds both byte archives; native startup must hash
and validate the one chosen full207 checkpoint before any optimizer update.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import re
from pathlib import Path

from tsc_expanded_TRAIN_transition_v1 import (
    HERE,ROOT,RECEIPT_KIND,DATA_ARMS,SOURCE_STEP,COMPLETED_SOURCE_STEP,UPDATES,END_STEP,
    require,read,scoped,record,checked_record,canonical,_history_frontier,
    validate_closed_inputs,validate_receipt,source_context)


def inspect_closed(args):
    preservation=record(args.source_preservation);policy=record(args.normalization_policy);index_record=record(args.expanded_index_proof)
    closed,selection_policy,index=validate_closed_inputs(preservation,policy,index_record)
    run_id=closed['run_id'];require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{2,110}',args.target_run_id)
        and args.target_run_id!=run_id and args.data_arm in DATA_ARMS,'New explicit matched data-arm namespace required')
    ca=closed['artifacts'];artifacts={key:item['archived'] for key,item in ca.items()}
    original={key:item['original'] for key,item in ca.items()}
    for key in ('run_config','selection','controller_status','runtime_completion','owned_worker_identity'):
        checked_record(artifacts[key])
    config=read(artifacts['run_config']['path']);selection=read(artifacts['selection']['path']);status=read(artifacts['controller_status']['path'])
    require(status['status']=='train_window_ended' and status['child_exit_code']==0 and status['resource_review_required'] is False
        and status['completed_steps']==selection['completed_steps']==COMPLETED_SOURCE_STEP
        and selection['selected_step']==SOURCE_STEP and selection['accepted'] is True
        and config['run_id']==selection['run_id']==run_id,'Actual closed normalization source required before receipt creation')
    artifacts.update(chosen_checkpoint=ca['best_checkpoint']['archived'],source_best_checkpoint=ca['best_checkpoint']['archived'],
        chosen_DEV_summary=ca['best_DEV_summary']['archived'],source_best_DEV_summary=ca['best_DEV_summary']['archived'])
    original.update(chosen_checkpoint=ca['best_checkpoint']['original'],source_best_checkpoint=ca['best_checkpoint']['original'])
    checked_record(artifacts['chosen_checkpoint'],checkpoint=True);checked_record(artifacts['latest_checkpoint'],checkpoint=True)
    folder=scoped(closed['best_summary']['path']).parent
    for key,name in (('chosen_DEV_metadata','metadata.json'),('chosen_DEV_metadata_pt','metadata_pt.json')):
        artifacts[key]=record(folder/name);original[key]=artifacts[key]
    for lang in ('tr','en','es','it','fr','de','pt'):
        key='chosen_DEV_'+lang;artifacts[key]=record(folder/f'validation_{lang}.jsonl');original[key]=artifacts[key]
    run=HERE/'runs'/run_id
    extra={'indexed_TR_audit':run/'indexed_TR_audit.json','uniform_source_receipt':config['uniform_full_source_receipt'],
        'Portuguese_preparation_proof':config['portuguese_preparation_proof'],
        'Portuguese_baseline_proof':config['portuguese_baseline_proof'],
        'original_champion':config['source_champion_parked']['path']}
    for key,p in extra.items():
        if key=='original_champion':
            bound=config['source_champion_parked'];checked_record(bound,checkpoint=True)
        else:bound=record(p)
        artifacts[key]=bound;original[key]=bound
    # These original tokens remain ancestry only; the new run gets fresh ready/source350.
    artifacts['original_native_ready']=ca['original_native_ready']['original']
    artifacts['initial_source_equivalence']=ca['initial_source_equivalence']['original']
    checked_record(artifacts['original_native_ready']);checked_record(artifacts['initial_source_equivalence'])
    prefix,last=_history_frontier(artifacts['training_history']['path'],SOURCE_STEP,COMPLETED_SOURCE_STEP)
    require(last['selection_diagnostics']['consecutive_retention_failures']==0
        and last['validation']['retention_gate_passed'] is True,'Chosen BEST40750 has genuine zero failure counter')
    receipt={'schema':1,'kind':RECEIPT_KIND,'status':'actual_CLOSED_qualified_BEST40750_matched_data_arms',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'source_run_id':run_id,'target_run_id':args.target_run_id,
        'data_arm':args.data_arm,'chosen_checkpoint_role':'best','chosen_step':SOURCE_STEP,'completed_steps':COMPLETED_SOURCE_STEP,
        'source_counter_preserved':0,'additional_updates':UPDATES,'configured_end_step':END_STEP,
        'source_preservation':preservation,'normalization_policy':policy,'expanded_index_proof':index_record,
        'candidate_preparation':record(HERE/'WorldSpeech_TR_TRAIN_audio_candidate_preparation_v1.json'),
        'candidate_verification':record(HERE/'WorldSpeech_complete_candidate_CPU_independent_verification_v1.json'),
        'candidate_approval':record(HERE/'WorldSpeech_TRAIN_sanity64_actual_review_v1.json'),
        'source_actual_owned_exit':closed['actual_owned_exit'],'artifacts':artifacts,'original_artifact_records':original,
        'scientific_config_identity_sha256':canonical(config['config_identity']),'source_history_prefix_sha256':prefix,
        'source_chosen_is_qualified':True,'chosen_source_not_final_latest':True,
        'explicit_completed_update_rewind':COMPLETED_SOURCE_STEP-SOURCE_STEP,
        'explicit_TR_draws_revisited_vs_final_branch':14*(COMPLETED_SOURCE_STEP-SOURCE_STEP),
        'failed_final_source_history_preserved':True,'new_TR_coverage_epoch_is_explicit':True,
        'source_checkpoint_archive_reused_without_tensor_or_byte_read':True,'creator':record(__file__),
        'GPU_launched':False,'tensor_deserialization_used':False,'TEST_used':False}
    validate_receipt(receipt,args.target_run_id)
    return receipt


def prepare(args):
    receipt=inspect_closed(args)
    if args.validate_only:return {'passed':True,'source_step':SOURCE_STEP,'source_role':'best','data_arm':args.data_arm,
        'configured_end_step':END_STEP,'additional_updates':UPDATES,'checkpoint_payload_read':False,'GPU_launched':False}
    require(args.archive_dir is not None and args.receipt_output is not None,'Fresh purpose archive and output required')
    archive=scoped(args.archive_dir);output=scoped(args.receipt_output)
    require(archive.is_relative_to(ROOT/'work/closed-sources') and not archive.exists() and not output.exists(),'Fresh isolated purpose namespace required')
    archive.mkdir(parents=True)
    receipt.update(purpose_archive_directory=str(archive),existing_sealed_source_archive_reused=True)
    provisional=archive/'purpose_receipt.json'
    with provisional.open('x',encoding='utf-8') as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    source_context(str(provisional),args.target_run_id);source_context.cache_clear()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    return {'passed':True,'receipt':record(output),'source_step':SOURCE_STEP,'data_arm':args.data_arm,
        'configured_end_step':END_STEP,'checkpoint_payload_read':False,'GPU_launched':False}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-preservation',type=Path,default=HERE/'TR_token_population_actual_CLOSED43238_preservation_v1.json')
    p.add_argument('--normalization-policy',type=Path,default=HERE/'completed_TR_TDT_normalization_policy_selection_v1.json')
    p.add_argument('--expanded-index-proof',type=Path,default=HERE/'expanded_TRAIN_index_actual_CPU_preparation_v1.json')
    p.add_argument('--target-run-id',required=True);p.add_argument('--data-arm',choices=DATA_ARMS,required=True)
    p.add_argument('--archive-dir',type=Path);p.add_argument('--receipt-output',type=Path)
    p.add_argument('--validate-only',action='store_true');return p


if __name__=='__main__':print(json.dumps(prepare(parser().parse_args()),indent=2))
