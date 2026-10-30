"""Inactive ROOT purpose+byte archive for reviewed majority latest PT-focused replay fork.

The existing actual ROOT closure/retention review is the authority; no second
review is fabricated. Whole207 latest and runtime archive together after
terminal/native checks. No tensor deserialization, launch or ML imports.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import shutil

from tsc_uniform_majority_ptfocus_transition_v1 import (
    HERE,ROOT,LANGUAGES,RECEIPT_KIND,REVIEW_KIND,DECLARED_CHANGE,PARENT_PINS,PARENT_CONTROLLER,
    require,read,scoped,record,canonical,checked_record,_history_frontier,source_context,validate_receipt)
from record_TSC_uniform_majority_source_v1 import option


def inspect_closed(args):
    review_path=scoped(args.source_review);review=read(review_path)
    require(review.get('schema')==1 and review.get('kind')==REVIEW_KIND and review.get('passed') is True
        and review.get('allow_declared_PT_focused_replay_continuation') is True
        and review.get('declared_change')==DECLARED_CHANGE,'Actual ROOT majority CLOSED/lowerLR review required')
    status=read(checked_record(review['controller_status']));selection=read(checked_record(review['selection']))
    done=read(checked_record(review['native_completion']));h=review['chosen_step'];run_id=review['source_run_id'];run=HERE/'runs'/run_id
    require(status.get('status')=='train_window_ended' and status.get('child_exit_code')==0
        and status.get('resource_review_required') is False and status.get('stop_category')=='retention'
        and status.get('controller_sha256')==PARENT_PINS[PARENT_CONTROLLER]
        and status.get('completed_steps')==selection.get('completed_steps')==h
        and review['owned_actual_exit']['exit_code']==0 and done.get('status')=='complete'
        and done.get('sources_unchanged') is True and not done.get('error'),'Real parent/native closure required before checkpoint bytes')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{2,110}',args.target_run_id)
        and args.target_run_id!=run_id,'Fresh lowerLR scientific namespace required')
    require(review['chosen_checkpoint_role']=='latest' and selection['accepted'] is False and selection['selected_step'] is None
        and scoped(review['chosen_checkpoint']['path'])==scoped(selection['latest_weights']),'One unqualified latest7750 frontier, no BEST mixture')
    observed=datetime.fromisoformat(review['owned_actual_exit']['observed_at_utc'].replace('Z','+00:00'))
    finished=datetime.fromisoformat(status['finished_at_utc'].replace('Z','+00:00'))
    require(observed.tzinfo is not None and observed>=finished,'Actual owned ROOT closure observation must follow terminal state')
    config=read(run/'run_config.json');prefix,row=_history_frontier(run/'training_history.json',h,h)
    require(row['selection_diagnostics']['consecutive_retention_failures']==review['parent_retention_failure_counter'],
        'Do not reset parent failure counter')
    paths=status['paths'];argv=status['worker_arguments'];folder=HERE/'results'/f'{run_id}_step_{h}'
    original_native=scoped(option(argv,'--native-backend-provenance'));runtime_native=scoped(paths.get('fresh_runtime',paths['native']))
    sources={'controller_status':scoped(review['controller_status']['path']),
        'run_config':run/'run_config.json','selection':scoped(review['selection']['path']),'training_history':run/'training_history.json',
        'chosen_checkpoint':scoped(review['chosen_checkpoint']['path']),'chosen_DEV_summary':scoped(review['chosen_DEV_summary']['path']),
        'chosen_DEV_metadata':folder/'metadata.json','chosen_DEV_metadata_pt':folder/'metadata_pt.json',
        'initial_source_equivalence':scoped(review['source350_equivalence']['path']),
        'original_native_ready':original_native,'runtime_native_ready':runtime_native,
        'runtime_completion':scoped(review['native_completion']['path']),'owned_worker_identity':scoped(paths['identity']),
        'indexed_TR_audit':run/'indexed_TR_audit.json','uniform_source_receipt':scoped(option(argv,'--uniform-full-source-receipt')),
        'Portuguese_preparation_proof':scoped(option(argv,'--portuguese-preparation-proof')),
        'Portuguese_baseline_proof':scoped(option(argv,'--portuguese-baseline-proof')),
        'original_champion':scoped(config['source_champion_parked']['path']),
        **{name:scoped(paths[name]) for name in ('memory','cache','io')},
        **{'chosen_DEV_'+lang:folder/f'validation_{lang}.jsonl' for lang in LANGUAGES}}
    for key in ('memory','cache','io'):
        value=read(sources[key]);require(value.get('status')=='complete' and value.get('sources_unchanged') is True,
            'Incomplete owned runtime: '+key)
    receipt={'schema':1,'kind':RECEIPT_KIND,'status':'actual_CLOSED_reviewed_majority_PT_fork',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'target_run_id':args.target_run_id,'source_run_id':run_id,
        'chosen_checkpoint_role':'latest','chosen_step':h,'completed_steps':h,'parent_retention_failure_counter':review['parent_retention_failure_counter'],
        'owned_actual_exit':review['owned_actual_exit'],'allow_declared_PT_focused_replay_continuation':True,
        'declared_change':DECLARED_CHANGE,'ROOT_closed_retention_review':record(review_path),
        'original_LR_source_receipt':review['original_LR_source_receipt'],
        'scientific_config_identity_sha256':canonical(config['config_identity']),'source_history_prefix_sha256':prefix,
        'renewed_Portuguese_grace':False,'parent_failure_counter_preserved':True,
        'creator':record(__file__),'checkpoint_tensors_audited':False,'actual_training_activation':False}
    validate_receipt(receipt,args.target_run_id)
    return sources,receipt


def prepare(args):
    sources,receipt=inspect_closed(args)
    if args.validate_only:return {'passed':True,'checkpoint_bytes_read':False,'actual_closed_review':receipt['ROOT_closed_retention_review'],
        'source_step':receipt['chosen_step'],'parent_failure_counter':receipt['parent_retention_failure_counter'],'new_replay_layout':'PT1_originalfive1'}
    require(args.archive_dir is not None and args.receipt_output is not None,'Fresh archive/purpose receipt required')
    archive=scoped(args.archive_dir);output=scoped(args.receipt_output)
    require(archive.is_relative_to(ROOT/'work/closed-sources') and not archive.exists() and not output.exists(),'Fresh isolated archive/output required')
    # Real terminal, native and ROOT review checks precede the first .pt byte.
    originals={name:record(path) for name,path in sources.items()}
    review=read(receipt['ROOT_closed_retention_review']['path']);require(originals['chosen_checkpoint']==review['chosen_checkpoint'],
        'Reviewed chosen latest bytes changed')
    archive.mkdir(parents=True);artifacts={};copies={};backups={}
    immutable=('original_native_ready','initial_source_equivalence','original_champion')
    for name,source in sources.items():
        if source not in copies:
            target=archive/(name+source.suffix);shutil.copy2(source,target);bound=record(target)
            require((bound['sha256'],bound['bytes'])==(originals[name]['sha256'],originals[name]['bytes'])
                and record(source)==originals[name],'Source/archive changed: '+name)
            copies[source]=bound
        if name in immutable:artifacts[name]=originals[name];backups[name]=copies[source]
        else:artifacts[name]=copies[source]
    require(record(sources['controller_status'])==originals['controller_status'],'Source terminal changed during archive')
    receipt.update(artifacts=artifacts,original_artifact_records=originals,
        original_artifact_paths={name:str(path) for name,path in sources.items()},archived_immutable_scientific_tokens=backups,
        archive_directory=str(archive),archive_copies_byte_identical=True)
    provisional=archive/'receipt_validation.json'
    with provisional.open('x',encoding='utf-8') as stream:json.dump(receipt,stream,indent=2,ensure_ascii=False,allow_nan=False)
    source_context(str(provisional),args.target_run_id);source_context.cache_clear()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(receipt,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    return {'passed':True,'receipt':record(output),'source_step':receipt['chosen_step'],'GPU_launched':False,'tensor_deserialization_used':False}


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-review',type=Path,required=True)
    p.add_argument('--target-run-id',required=True);p.add_argument('--archive-dir',type=Path);p.add_argument('--receipt-output',type=Path)
    p.add_argument('--validate-only',action='store_true');return p


if __name__=='__main__':print(json.dumps(prepare(parser().parse_args()),indent=2))
