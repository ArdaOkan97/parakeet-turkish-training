"""ROOT-reviewed8250 whole-state archive for a count-equivalent gate correction.

Metadata terminal/native/owned exit evidence precedes all checkpoint hashing.
Old failure5/NoneBEST artifacts stay immutable; new accounting is explicit.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import shutil

from tsc_uniform_majority_ptfocus_integer_transition_v1 import (
    HERE,ROOT,LANGUAGES,REVIEW_KIND,RECEIPT_KIND,PARENT_PINS,PARENT_CONTROLLER,
    require,read,scoped,record,canonical,checked_record,_history_frontier,source_context,validate_receipt)
from record_TSC_uniform_majority_source_v1 import option


def inspect_closed(args):
    review_path=scoped(args.source_review);review=read(review_path)
    require(review.get('kind')==REVIEW_KIND and review.get('passed') is True and review.get('source_step')==8250
        and review.get('allow_count_equivalent_numerical_gate_correction') is True and review.get('thresholds_relaxed') is False,
        'Actual ROOT integer gate migration review required')
    closure=read(checked_record(review['actual_delegated_closure_review']))
    status=read(checked_record(review['source_controller_status']));selection=read(checked_record(review['source_selection']))
    done=read(checked_record(review['source_native_completion']));run_id=review['source_run_id'];run=HERE/'runs'/run_id
    require(closure['actual_owned_exit']==review['actual_owned_exit'] and review['actual_owned_exit']['exit_code']==0
        and closure['native_child_exit_code']==0 and all(v['absent'] is True for v in closure['actual_owned_PID_absence'])
        and status['status']=='train_window_ended' and status['child_exit_code']==0 and status['resource_review_required'] is False
        and status['stop_category']=='retention' and status['controller_sha256']==PARENT_PINS[PARENT_CONTROLLER]
        and done['status']=='complete' and done['sources_unchanged'] is True and not done.get('error'),
        'Actual owned8250 CLOSED0/native0 required before checkpoint bytes')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{2,110}',args.target_run_id) and args.target_run_id!=run_id,
        'New declared numeric accounting namespace required')
    require(selection['completed_steps']==status['completed_steps']==8250 and selection['accepted'] is False
        and selection['selected_step'] is None and selection['consecutive_retention_failures']==5
        and scoped(selection['latest_weights'])==scoped(review['source_latest_checkpoint']['path']),
        'Actual false selection/failure5 latest must not be rewritten')
    config=read(run/'run_config.json');prefix,last=_history_frontier(run/'training_history.json',8250,8250)
    require(last['validation']['retention_gate_passed'] is False and last['selection_diagnostics']['consecutive_retention_failures']==5,
        'Source false-gate history remains canonical')
    paths=status['paths'];argv=status['worker_arguments'];folder=HERE/'results'/f'{run_id}_step_8250'
    original_ready=scoped(option(argv,'--native-backend-provenance'));runtime_ready=scoped(paths.get('fresh_runtime',paths['native']))
    sources={'controller_status':scoped(review['source_controller_status']['path']),'selection':scoped(review['source_selection']['path']),
        'run_config':run/'run_config.json','training_history':run/'training_history.json',
        'chosen_checkpoint':scoped(review['source_latest_checkpoint']['path']),'chosen_DEV_summary':scoped(review['source_summary']['path']),
        'chosen_DEV_metadata':folder/'metadata.json','chosen_DEV_metadata_pt':folder/'metadata_pt.json',
        'initial_source_equivalence':run/'majority_initial_source_equivalence.json',
        'original_native_ready':original_ready,'runtime_native_ready':runtime_ready,'runtime_completion':scoped(review['source_native_completion']['path']),
        'owned_worker_identity':scoped(paths['identity']),'indexed_TR_audit':run/'indexed_TR_audit.json',
        'uniform_source_receipt':scoped(option(argv,'--uniform-full-source-receipt')),
        'Portuguese_preparation_proof':scoped(option(argv,'--portuguese-preparation-proof')),
        'Portuguese_baseline_proof':scoped(option(argv,'--portuguese-baseline-proof')),
        'original_champion':scoped(config['source_champion_parked']['path']),
        **{key:scoped(paths[key]) for key in ('memory','cache','io')},
        **{'chosen_DEV_'+lang:folder/f'validation_{lang}.jsonl' for lang in LANGUAGES}}
    for key in ('memory','cache','io'):
        value=read(sources[key]);require(value['status']=='complete' and value['sources_unchanged'] is True,'Actual runtime incomplete: '+key)
    receipt={'schema':1,'kind':RECEIPT_KIND,'status':'actual_CLOSED_count_equivalent_gate_migration',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'source_run_id':run_id,'target_run_id':args.target_run_id,
        'source_step':8250,'chosen_checkpoint_role':'latest','ROOT_review':record(review_path),
        'actual_delegated_closure_review':review['actual_delegated_closure_review'],'actual_owned_exit':review['actual_owned_exit'],
        'allow_count_equivalent_numerical_gate_correction':True,'thresholds_relaxed':False,
        'declared_state_accounting_migration':review['declared_state_accounting_migration'],
        'original_PT_source_receipt':record(config['majority_source_receipt']),
        'scientific_config_identity_sha256':canonical(config['config_identity']),'source_history_prefix_sha256':prefix,
        'creator':record(__file__),'actual_training_activation':False,'checkpoint_tensors_audited':False}
    validate_receipt(receipt,args.target_run_id)
    return sources,receipt


def prepare(args):
    sources,receipt=inspect_closed(args)
    if args.validate_only:return {'passed':True,'source_step':8250,'checkpoint_bytes_read':False,
        'actual_source_counter':5,'approved_corrected_counter':0,'ROOT_review':receipt['ROOT_review']}
    require(args.archive_dir is not None and args.receipt_output is not None,'Fresh byte archive/purpose receipt required')
    archive=scoped(args.archive_dir);output=scoped(args.receipt_output)
    require(archive.is_relative_to(ROOT/'work/closed-sources') and not archive.exists() and not output.exists(),'Fresh isolated output paths required')
    originals={key:record(path) for key,path in sources.items()}
    archive.mkdir(parents=True);artifacts={};copies={};backups={};immutable=('original_native_ready','initial_source_equivalence','original_champion')
    for key,source in sources.items():
        if source not in copies:
            target=archive/(key+source.suffix);shutil.copy2(source,target);bound=record(target)
            require((bound['sha256'],bound['bytes'])==(originals[key]['sha256'],originals[key]['bytes'])
                and record(source)==originals[key],'Source/archive changed: '+key)
            copies[source]=bound
        if key in immutable:artifacts[key]=originals[key];backups[key]=copies[source]
        else:artifacts[key]=copies[source]
    require(record(sources['controller_status'])==originals['controller_status'],'Actual terminal record changed')
    receipt.update(artifacts=artifacts,original_artifact_records=originals,
        original_artifact_paths={key:str(path) for key,path in sources.items()},archived_immutable_scientific_tokens=backups,
        archive_copies_byte_identical=True,archive_directory=str(archive))
    provisional=archive/'receipt_validation.json'
    with provisional.open('x',encoding='utf-8') as stream:json.dump(receipt,stream,indent=2,ensure_ascii=False,allow_nan=False)
    source_context(str(provisional),args.target_run_id);source_context.cache_clear()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(receipt,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    return {'passed':True,'receipt':record(output),'source_step':8250,'tensor_deserialization_used':False,'GPU_launched':False}


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-review',type=Path,required=True)
    p.add_argument('--target-run-id',required=True);p.add_argument('--archive-dir',type=Path);p.add_argument('--receipt-output',type=Path)
    p.add_argument('--validate-only',action='store_true');return p


if __name__=='__main__':print(json.dumps(prepare(parser().parse_args()),indent=2))
