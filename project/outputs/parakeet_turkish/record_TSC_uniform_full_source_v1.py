"""NEW ROOT-purpose receipt over the immutable actually CLOSED BEST6500 archive.

The archived exit1 and narrow ROOT resource review retain their actual status;
no new closure, tensor audit or training execution is fabricated. Creation
hashes the selected archive only after existing CLOSED lineage admission.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import re
from pathlib import Path
from types import SimpleNamespace

from tsc_uniform_full_transition_v1 import (HERE,PARENT_RUN,SOURCE_STEP,BEST_SHA,RECEIPT_KIND,require,
    scoped,read,record,checked_record,canonical,source_context,uniform_declaration,validate_receipt,validate_portuguese_inputs)


def prepare(args):
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{2,110}',args.target_run_id) and args.target_run_id!=PARENT_RUN,
        'Distinct safe full-uniform scientific run ID required')
    path=scoped(args.source_lineage_receipt);origin=record(path);context=source_context(str(path),origin['sha256'])
    preparation=record(args.portuguese_preparation_proof);baseline=record(args.portuguese_baseline_proof)
    receipt={'schema':1,'kind':RECEIPT_KIND,'status':'actual_CLOSED_old_five_qualified_PT_recovery_source','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_run_id':PARENT_RUN,'target_run_id':args.target_run_id,'selected_step':SOURCE_STEP,
        'completed_steps':context['lineage_receipt']['completed_steps'],'allow_uniform_fullpass':True,'BEST_not_latest_state':True,
        'declared_uniform_fullpass':uniform_declaration(args.portuguese_seed),'source_lineage_receipt':origin,
        'scientific_config_identity_sha256':canonical(context['parent_identity']),
        'source_history_prefix_sha256':context['source_history_prefix_sha256'],'artifacts':context['lineage_receipt']['artifacts'],
        'portuguese_preparation_proof':preparation,'portuguese_baseline_proof':baseline,
        'original_artifact_paths':context['lineage_receipt']['original_artifact_paths'],
        'source_actual_owned_exits_retained':context['lineage_receipt']['owned_actual_exits'],
        'source_actual_resource_status_not_relabelled_exit0':True,
        'source_resource_review':context['lineage_receipt'].get('resource_stop_review'),
        'new_initial_original300_equality_and_Portuguese50_actual_failure_reproduction_required':True,
        'initial_all_six_champion_claimed':False,'Portuguese_recovery_grace_new_updates':1000,
        'all207_optimizer_tensor_admission_pending_future_worker':True,'creator':record(__file__),
        'actual_GPU_or_training_activation':False,'actual_new_source_event_fabricated':False}
    pt=read(args.portuguese_preparation_proof)['files']
    pt_args=SimpleNamespace(portuguese_replay_manifest=Path(pt['replay_train_pt']['path']),portuguese_dev_manifest=Path(pt['validation_pt']['path']))
    validate_portuguese_inputs(pt_args,{**context,'portuguese_preparation_proof':preparation,'portuguese_baseline_proof':baseline})
    validate_receipt(receipt,args.target_run_id,args.portuguese_seed)
    if not args.validate_only:
        require(args.receipt_output is not None and not scoped(args.receipt_output).exists(),'Fresh ROOT-purpose receipt required')
        require(record(context['source_checkpoint']['path'])==context['source_checkpoint']
            and context['source_checkpoint']['sha256']==BEST_SHA,'Actually CLOSED selected archive byte hash changed')
        output=scoped(args.receipt_output);output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x',encoding='utf-8') as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
        return {'passed':True,'receipt':record(output),'source_checkpoint':context['source_checkpoint'],'GPU_launched':False,'source_tensor_audit_performed':False}
    return {'passed':True,'receipt_created':False,'source_checkpoint':context['source_checkpoint'],
        'selected_archive_checkpoint_hash_rechecked':False,'new_declaration':receipt['declared_uniform_fullpass'],'GPU_launched':False}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-lineage-receipt',type=Path,default=HERE/'TSC_headlr_BPE512_actual_CLOSED_reviewed_main_source_v2.json')
    p.add_argument('--target-run-id',required=True);p.add_argument('--portuguese-seed',type=int,required=True)
    p.add_argument('--portuguese-preparation-proof',type=Path,required=True);p.add_argument('--portuguese-baseline-proof',type=Path,required=True)
    p.add_argument('--receipt-output',type=Path);p.add_argument('--validate-only',action='store_true')
    return p


if __name__=='__main__':
    print(json.dumps(prepare(parser().parse_args()),ensure_ascii=False,indent=2))
