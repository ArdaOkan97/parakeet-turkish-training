"""ROOT-only closed uniform source archive for a NEW 14TR/2RP scientific fork.

Live/failed native boundaries are rejected before any checkpoint byte. A
resource controller exit1 is retained and needs its own exact ROOT review;
failed six-language retention needs a second review. No tensor or OS-closure
inference, process launch or actual source admission occurs on import.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import shutil

from tsc_uniform_majority_transition_v1 import (
    HERE,ROOT,LANGUAGES,UNIFORM_ORIGIN,SOURCE_PINS,PARENT_CONTROLLER,RECEIPT_KIND,
    RESOURCE_REVIEW_KIND,RETENTION_REVIEW_KIND,require,read,scoped,record,canonical,
    checked_record,majority_declaration,retention_gates,validate_closure_header,
    validate_receipt,validate_review,_history_frontier,source_context)


def option(arguments,flag):
    require(type(arguments) is list and arguments.count(flag)==1,'One actual worker option required: '+flag)
    index=arguments.index(flag);require(index+1<len(arguments),'Missing worker value: '+flag)
    return arguments[index+1]


def inspect_closed(args):
    loop=scoped(args.source_loop);status_path=loop/'loop_status.json';controller=read(status_path)
    exit_code=validate_closure_header(controller)
    require(args.owned_exit_code==exit_code,'ROOT must retain actual controller resource exit1')
    require(type(args.owned_session_id) is int and args.owned_session_id>0
        and re.fullmatch('[0-9a-f]+',args.owned_exit_chunk or ''),'Actual ROOT owned exit session/chunk required')
    observed=datetime.fromisoformat(args.owned_exit_observed_at.replace('Z','+00:00'))
    finished=datetime.fromisoformat(controller['finished_at_utc'].replace('Z','+00:00'))
    require(observed.tzinfo is not None and observed>=finished,'ROOT observed actual exit must follow controller terminal publication')
    run_id=controller['run_id'];run=HERE/'runs'/run_id;config=read(run/'run_config.json');selection=read(run/'selection.json')
    c=selection['completed_steps'];role=args.chosen_checkpoint_role;h=c if role=='latest' else selection['selected_step']
    require(type(h) is int and UNIFORM_ORIGIN<h<=c and controller.get('completed_steps')==c
        and (role=='latest' or (selection.get('accepted') is True and args.allow_earlier_best_full_state_fork)),
        'Actual chosen latest or explicitly declared earlier qualified BEST fork required')
    prefix,row=_history_frontier(run/'training_history.json',h,c)
    prior=row['Turkish_domain_coverage']['uniform_pass']['draws'];declaration=majority_declaration(h,prior)
    folder=HERE/'results'/f'{run_id}_step_{h}';summary=read(folder/'summary.json')
    baseline_path=scoped(config['portuguese_retention_baseline_proof']['path']);baseline=read(baseline_path)['baseline_overlay']
    gates=retention_gates(summary,baseline)
    require(args.target_run_id!=run_id and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{2,110}',args.target_run_id),
        'Distinct safe scientific target namespace required')
    paths=controller['paths'];arguments=controller['worker_arguments']
    original_ready=scoped(option(arguments,'--native-backend-provenance'))
    runtime_ready=scoped(paths.get('fresh_runtime',paths['native']))
    sources={'controller_status':status_path,'source_controller':HERE/PARENT_CONTROLLER,
        'run_config':run/'run_config.json','selection':run/'selection.json','training_history':run/'training_history.json',
        'chosen_checkpoint':scoped(selection['latest_weights'] if role=='latest' else selection['weights']),
        'chosen_DEV_summary':folder/'summary.json','chosen_DEV_metadata':folder/'metadata.json','chosen_DEV_metadata_pt':folder/'metadata_pt.json',
        'original_native_ready':original_ready,'runtime_native_ready':runtime_ready,
        'runtime_completion':runtime_ready.with_suffix('.completion.json'),'owned_worker_identity':scoped(paths['identity']),
        'initial_source_equivalence':run/'uniform_full_initial_source_equivalence.json',
        'indexed_TR_audit':run/'indexed_TR_audit.json','uniform_source_receipt':scoped(option(arguments,'--uniform-full-source-receipt')),
        'Portuguese_preparation_proof':scoped(option(arguments,'--portuguese-preparation-proof')),
        'Portuguese_baseline_proof':baseline_path,'original_champion':scoped(config['source_champion_parked']['path']),
        **{name:scoped(paths[name]) for name in ('memory','cache','io')},
        **{'chosen_DEV_'+lang:folder/f'validation_{lang}.jsonl' for lang in LANGUAGES}}
    j=selection['selected_step']
    if j is not None:
        sources['source_best_checkpoint']=scoped(selection['weights'])
        sources['source_best_DEV_summary']=HERE/'results'/f'{run_id}_step_{j}'/'summary.json'
        require(all(v['passed'] for v in retention_gates(read(sources['source_best_DEV_summary']),baseline).values()),
            'Separate inherited eligible BEST must pass every retention gate')
    # Native/runtime closure is checked before source/checkpoint hashing.
    done=read(sources['runtime_completion']);owned=read(sources['owned_worker_identity']);ready=read(runtime_ready)
    require(done.get('status')=='complete' and done.get('sources_unchanged') is True and not done.get('error')
        and done['run_id']==run_id and done['pid']==owned['pid']==ready['pid']
        and done['actual_dispatch_source_sha256']==SOURCE_PINS[PARENT_CONTROLLER],'Actual child native completion failed')
    for name in ('memory','cache','io'):
        value=read(sources[name]);require(value.get('status')=='complete' and value.get('sources_unchanged') is True
            and not value.get('error'),'Actual runtime boundary failed: '+name)
    original={name:record(path) for name,path in sources.items() if path.suffix!='.pt'}
    exit_record={'role':'trainer_controller','session_id':args.owned_session_id,'exit_code':exit_code,
        'chunk_id':args.owned_exit_chunk,'observed_at_utc':observed.isoformat()}
    receipt={'schema':1,'kind':RECEIPT_KIND,'status':'actual_CLOSED_reviewed_uniform_frontier',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'source_run_id':run_id,'target_run_id':args.target_run_id,
        'allow_Turkish_majority_continuation':True,'guards_retained':True,'chosen_checkpoint_role':role,
        'chosen_step':h,'completed_steps':c,'explicit_earlier_BEST_full_state_fork':role=='best',
        'source_best_step':j,'source_best_checkpoint':None if j is None else {'path':str(sources['source_best_checkpoint'])},
        'scientific_config_identity_sha256':canonical(config['config_identity']),'source_history_prefix_sha256':prefix,
        'prior_uniform_draws':prior,'declared_majority':declaration,'chosen_retention_gates':gates,
        'source_retention_failure_counter':row['selection_diagnostics']['consecutive_retention_failures'],
        'Portuguese_grace_elapsed_new_updates':h-UNIFORM_ORIGIN,'renewed_Portuguese_grace':False,
        'owned_actual_exit':exit_record,'source_stop_category':controller['stop_category'],
        'source_stop_reason':controller.get('cooperative_stop_reason'),'source_actual_exit_status_not_relabelled':True,
        'resource_stop_review':None if args.resource_stop_review is None else record(args.resource_stop_review),
        'unqualified_source_review':None if args.unqualified_source_review is None else record(args.unqualified_source_review),
        'original_artifact_records':original,'original_artifact_paths':{name:str(path) for name,path in sources.items()},
        'artifacts':original,'creator':record(__file__),'checkpoint_tensors_audited':False,'actual_training_activation':False}
    validate_receipt(receipt,args.target_run_id)
    if exit_code==1:
        require(receipt['resource_stop_review'] is not None,'Resource exit1 needs distinct exact ROOT review')
        validate_review(receipt['resource_stop_review'],RESOURCE_REVIEW_KIND,receipt,controller,gates)
    else:require(receipt['resource_stop_review'] is None,'No unnecessary resource bypass')
    if not all(v['passed'] for v in gates.values()):
        require(role=='latest' and receipt['unqualified_source_review'] is not None,'Failed gates need explicit latest recovery review')
        validate_review(receipt['unqualified_source_review'],RETENTION_REVIEW_KIND,receipt,controller,gates)
    else:require(receipt['unqualified_source_review'] is None,'No unnecessary unqualified bypass')
    return {'sources':sources,'receipt':receipt,'controller':controller}


def prepare(args):
    context=inspect_closed(args);receipt=context['receipt'];sources=context['sources']
    if args.validate_only:
        return {'passed':True,'actual_CLOSED_metadata_admitted':True,'checkpoint_bytes_read':False,
            'source_step':receipt['chosen_step'],'new_recipe':receipt['declared_majority'],
            'source_actual_exit_code':receipt['owned_actual_exit']['exit_code'],'archive_created':False}
    require(args.archive_dir is not None and args.receipt_output is not None,'Fresh archive/receipt paths required')
    archive=scoped(args.archive_dir);output=scoped(args.receipt_output)
    require(archive.is_relative_to(ROOT/'work/closed-sources') and not archive.exists() and not output.exists(),
        'Fresh isolated ROOT archive/receipt outputs required')
    # Every closure, native, review and qualification gate precedes .pt reads.
    originals={name:record(path) for name,path in sources.items()}
    archive.mkdir(parents=True);copies={};artifacts={}
    immutable=('original_native_ready','initial_source_equivalence','original_champion')
    backups={}
    for name,source in sources.items():
        if source not in copies:
            target=archive/(name+source.suffix);shutil.copy2(source,target);bound=record(target)
            require((bound['sha256'],bound['bytes'])==(originals[name]['sha256'],originals[name]['bytes'])
                and record(source)==originals[name],'Closed source/archive changed: '+name)
            copies[source]=bound
        if name in immutable:artifacts[name]=originals[name];backups[name]=copies[source]
        else:artifacts[name]=copies[source]
    require(record(sources['controller_status'])==originals['controller_status'],'Actual terminal source changed during archive')
    receipt.update(artifacts=artifacts,original_artifact_records=originals,
        source_best_checkpoint=artifacts.get('source_best_checkpoint'),archived_immutable_scientific_tokens=backups,
        archive_directory=str(archive),archive_copies_byte_identical=True)
    validate_receipt(receipt,args.target_run_id)
    output.parent.mkdir(parents=True,exist_ok=True)
    # Metadata preflight over archive precedes publication of the final receipt.
    provisional=archive/'receipt_validation.json'
    with provisional.open('x',encoding='utf-8') as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2,allow_nan=False)
    source_context(str(provisional),args.target_run_id);source_context.cache_clear()
    with output.open('x',encoding='utf-8') as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    return {'passed':True,'receipt':record(output),'source_step':receipt['chosen_step'],'source_actual_exit_code':receipt['owned_actual_exit']['exit_code'],
        'chosen_checkpoint':artifacts['chosen_checkpoint'],'checkpoint_tensor_audit_performed':False,'GPU_launched':False}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-loop',type=Path,required=True);p.add_argument('--target-run-id',required=True)
    p.add_argument('--chosen-checkpoint-role',choices=('latest','best'),default='latest')
    p.add_argument('--allow-earlier-best-full-state-fork',action='store_true')
    p.add_argument('--owned-session-id',type=int,required=True);p.add_argument('--owned-exit-code',type=int,choices=(0,1),required=True)
    p.add_argument('--owned-exit-chunk',required=True);p.add_argument('--owned-exit-observed-at',required=True)
    p.add_argument('--resource-stop-review',type=Path);p.add_argument('--unqualified-source-review',type=Path)
    p.add_argument('--archive-dir',type=Path);p.add_argument('--receipt-output',type=Path);p.add_argument('--validate-only',action='store_true')
    return p


if __name__=='__main__':print(json.dumps(prepare(parser().parse_args()),ensure_ascii=False,indent=2))
