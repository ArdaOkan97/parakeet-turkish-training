"""CPU-only exact-state fork after a reviewed cooperative resource stop.

Every serialized leaf except the checkpoint's top-level run_id is preserved.
The predecessor has DEV/full-state evidence, not a new export or TEST result.
The manifest, proof and initial archives are immutable and committed last.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
sys.path.insert(0, str(OUTPUT))
import fork_training_state_v1 as base
from atomic_io_retry_v1 import commit_writer, read_bytes_shared, write_json
import torch

PRIOR_FORK_HELPER_SHA256 = 'd1195f520d6654bd787b960dc12f10320ff4d58fafacc4e166d2978735d6206b'


def raw_exact(left, right, check, path='state'):
    """Independent serialized-state comparison, including signed-zero bytes."""
    check.require(type(left) is type(right), path+' type')
    if isinstance(left, torch.Tensor):
        check.tensor_count += 1
        check.require(left.device.type == right.device.type == 'cpu', path+' CPU')
        check.require(left.dtype == right.dtype and left.shape == right.shape, path+' tensor shape/dtype')
        a=left.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        b=right.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        check.require(a == b, path+' raw tensor bytes')
    elif isinstance(left,dict):
        check.require(list(left)==list(right),path+' ordered dictionary keys')
        for key in left:raw_exact(left[key],right[key],check,path+'.'+str(key))
    elif isinstance(left,(list,tuple)):
        check.require(len(left)==len(right),path+' sequence length')
        for i,(a,b) in enumerate(zip(left,right)):raw_exact(a,b,check,path+'['+str(i)+']')
    else:
        check.require(left==right,path+' value')


def validate_resource_receipt(receipt, source_loop, source_run, check, args):
    check.require(receipt['kind']=='root_owned_resource_review_exit_receipt_v1','Reviewed resource-stop receipt required')
    check.require(receipt['root_observed_owned_session_completed'] is True
        and receipt['root_owned_exec_session']==args.expected_owned_session and receipt['root_observed_exit_code']==1,
        'Root must have observed the owned predecessor controller exit1')
    check.require(receipt['predecessor_loop_id']==source_loop.name
        and receipt['predecessor_status']=='resource_review_required'
        and receipt['predecessor_run_id']==source_run.name,'Receipt predecessor identity/status')
    check.require(receipt['child_exit_code']==0 and receipt['predecessor_training_checkpoint_review_passed'] is True,
        'Worker must have stopped cleanly and resource cause reviewed')
    check.require(receipt['child_durable_step']==args.expected_latest_step and receipt['child_selected_step']==args.expected_best_step,
        'Reviewed durable predecessor progress must match expected durable/selected steps')
    check.require(receipt['predecessor_status_sha256']==base.digest(source_loop/'loop_status.json'),
        'Receipt must bind immutable terminal predecessor status')
    state=base.read_json(source_loop/'loop_status.json')
    check.require(state['status']=='resource_review_required' and state['requires_resource_review'] is True,
        'Predecessor terminal resource stop differs')
    trial=next((x for x in state['trials'] if x['run_id']==source_run.name),None)
    check.require(trial is not None and trial['exit_code']==0 and trial['completed_steps']==args.expected_latest_step,
        'Owned predecessor trainer completion differs')
    check.require(state['deadline_at_utc']==receipt['predecessor_deadline_at_utc'],
        'Predecessor deadline unchanged')
    check.require(receipt['predecessor_audit_available'] is False,
        'No false completed final-model audit claim')
    for path_key,sha_key,expected in (
        ('child_run_config_path','child_run_config_sha256',source_run/'run_config.json'),
        ('child_training_history_path','child_training_history_sha256',source_run/'training_history.json'),
        ('child_selection_path','child_selection_sha256',source_run/'selection.json'),
        ('predecessor_controller_path','predecessor_controller_sha256',OUTPUT/'run_experiment_loop_continuation_v2.py')):
        check.require(Path(receipt[path_key]).resolve()==expected.resolve()
            and base.digest(expected)==receipt[sha_key],'Receipt exact file '+path_key)
    for path_key,sha_key in (('resource_review_path','resource_review_sha256'),
                            ('prior_audited_receipt_path','prior_audited_receipt_sha256')):
        path=Path(receipt[path_key]).resolve()
        check.require(base.inside(path,OUTPUT) and base.digest(path)==receipt[sha_key],'Bound resource/prior evidence')
    for record in receipt['runtime_provenance'].values():
        path=Path(record['path']).resolve()
        check.require(base.inside(path,source_loop) and base.digest(path)==record['sha256'],'Bound predecessor runtime evidence')
    return state


def run(args):
    torch.set_num_threads(2)
    check=base.Checks()
    check.require(base.digest(OUTPUT/'fork_training_state_v1.py')==PRIOR_FORK_HELPER_SHA256,
        'Original full-state comparison helper changed')
    source_id,target_id=args.source_run_id,args.target_run_id
    source_run=OUTPUT/'runs'/source_id;source_loop=OUTPUT/'runs'/args.source_loop_id
    target_run=OUTPUT/'runs'/target_id;snapshot=args.output_dir.resolve();loop=snapshot.parent
    check.require(source_id=='adaptive_20261010_continuation_02_resume_B8'
        and args.source_loop_id=='adaptive_20261010_continuation_02'
        and target_id=='adaptive_20261010_continuation_03_resume_B8', 'Reviewed fork source/target only')
    check.require(snapshot==OUTPUT/'runs/adaptive_20261010_continuation_03/fork_snapshot',
        'New owned snapshot directory only')
    check.require(not loop.exists() and not target_run.exists(),'Existing continuation evidence is immutable')
    check.require(isinstance(args.expected_latest_step,int) and isinstance(args.expected_best_step,int)
        and 0<args.expected_best_step<=args.expected_latest_step,'Reviewed latest/best valid ordered steps')
    targets={kind:ROOT/'work'/(target_id+'_'+kind+'.pt') for kind in ('latest','best')}
    for path in targets.values():check.require(not path.exists(),'Existing target checkpoint refused')
    receipt=base.read_json(args.predecessor_exit_receipt)
    check.require(base.digest(args.predecessor_exit_receipt)==
        args.predecessor_exit_receipt_sha256,'Root-owned reviewed receipt bytes')
    old=validate_resource_receipt(receipt,source_loop,source_run,check,args)
    receipt_sha=base.digest(args.predecessor_exit_receipt)
    config=base.read_json(source_run/'run_config.json')
    selection=base.read_json(source_run/'selection.json')
    history=base.read_json(source_run/'training_history.json')
    check.require(config['run_id']==selection['run_id']==source_id and selection['accepted'] is True,
        'Source DEV selection identity/acceptance')
    check.require(selection['completed_steps']==len(history)==args.expected_latest_step
        and selection['selected_step']==args.expected_best_step,
        'Source selection/history exact latest/best')
    check.require(history[args.expected_best_step-1]['validation']['retention_gate_passed'] is True,
        'Selected DEV state must pass all original DEV retention gates')
    check.require(selection['turkish_validation_wer']==history[args.expected_best_step-1]['validation']['turkish_wer'],
        'Selected state actual DEV WER')
    for name,digest in config['code_sha256'].items():
        check.require(base.digest(OUTPUT/name)==digest,'Frozen source '+name)
    for path,digest in config['input_sha256'].items():
        check.require(base.digest(Path(path))==digest,'Training input '+path)
    for name,digest in old['preflight']['source_sha256'].items():
        check.require(base.digest(OUTPUT/name)==digest,'Predecessor runtime source '+name)
    sources={};states={}
    for kind in ('latest','best'):
        path=ROOT/'work'/(source_id+'_'+kind+'.pt');data=read_bytes_shared(path)
        check.require(Path(receipt['source_'+kind+'_path']).resolve()==path.resolve()
            and hashlib.sha256(data).hexdigest()==receipt['source_'+kind+'_sha256'],'Receipt source '+kind+' checkpoint')
        state=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
        check.require(state['schema']==2 and state['run_id']==source_id
            and state['step']==(args.expected_latest_step if kind=='latest' else args.expected_best_step)
            and state['selected_step']==args.expected_best_step,'Full source '+kind+' identity')
        check.require(state['config_identity']==config['config_identity'],'Full source config identity')
        check.require(len(state['weights'])==len(state['optimizer']['state'])==207,'Full207 weights/Adam states')
        check.require(state['history']==history[:state['step']],'Full state checkpoint/durable history equality')
        for name,tensor in state['weights'].items():
            check.require(bool(torch.isfinite(tensor).all()),kind+' finite weight '+name)
        for key,value in state['optimizer']['state'].items():
            check.require(int(value['step'].item())==state['step'],kind+' Adam update count '+str(key))
            for field in ('exp_avg','exp_avg_sq'):
                check.require(bool(torch.isfinite(value[field]).all()),kind+' finite Adam '+str(key)+field)
        sources[kind]={'path':str(path.resolve()),'sha256':hashlib.sha256(data).hexdigest(),
            'size_bytes':len(data),'step':state['step']}
        states[kind]=state
        del data
    check.require(states['latest']['best_score']==states['best']['best_score']==selection['turkish_validation_wer'],
        'Full latest/best selected DEV score')
    latest_weights,best_weights=states['latest']['weights'],states['best']['weights']
    if args.expected_latest_step==args.expected_best_step:
        raw_exact(latest_weights,best_weights,check,'latest/best selected weights')
    for step in [r['step'] for r in history if 'validation' in r]:
        check.require(not (OUTPUT/'results'/(target_id+'_step_'+str(step))).exists(),'Existing inherited DEV copy')
    snapshot.mkdir(parents=True)
    records={}
    base.copy_tree(source_run,snapshot/'source_run',records,check)
    # Retain inherited immutable snapshot trees in place. Copy all direct loop
    # evidence and observation files, avoiding redundant old checkpoint trees.
    for path in sorted(source_loop.iterdir()):
        if path.is_file():base.copy_verified(path,snapshot/'source_loop'/path.name,records,check)
        elif path.name not in ('fork_snapshot',):
            base.copy_tree(path,snapshot/'source_loop'/path.name,records,check)
    base.copy_verified(args.predecessor_exit_receipt,snapshot/'predecessor_exit_receipt.json',records,check)
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file() and path.suffix in ('.json','.txt'):
            base.copy_verified(path,snapshot/'top_level'/path.name,records,check)
    for kind,entry in sources.items():
        base.copy_verified(Path(entry['path']),snapshot/'source_checkpoints'/Path(entry['path']).name,records,check)
    forks={};raw_check=base.Checks()
    for kind,path in targets.items():
        fork=dict(states[kind]);fork['run_id']=target_id
        base.validate_fork(states[kind],fork,source_id,target_id,check)
        commit_writer(path,lambda temporary,value=fork:torch.save(value,temporary))
        loaded=torch.load(io.BytesIO(read_bytes_shared(path)),map_location='cpu',weights_only=True)
        base.validate_fork(states[kind],loaded,source_id,target_id,check)
        for key in states[kind]:
            if key!='run_id':raw_exact(states[kind][key],loaded[key],raw_check,kind+'.'+key)
        forks[kind]={'path':str(path.resolve()),'sha256':base.digest(path),'size_bytes':path.stat().st_size,
            'step':sources[kind]['step'],'selected_step':args.expected_best_step,'source_sha256':sources[kind]['sha256']}
        del loaded
    dev={}
    for step in [r['step'] for r in history if 'validation' in r]:
        source=OUTPUT/'results'/(source_id+'_step_'+str(step));target=OUTPUT/'results'/(target_id+'_step_'+str(step))
        check.require({p.name for p in source.iterdir() if p.is_file()}==
            {'metadata.json','summary.json',*(f'validation_{lang}.jsonl' for lang in ('tr','en','es','it','fr','de'))},
            'Inherited DEV-only file set')
        copied={};base.copy_tree(source,target,copied,check)
        dev[str(step)]={'source_path':str(source.resolve()),'target_path':str(target.resolve()),'files':copied,
            'evaluation_reused_without_new_inference':True,'model_fingerprint':base.read_json(source/'metadata.json')['model_fingerprint'],
            'checkpoint_role':'selected_best' if step==args.expected_best_step else 'historical_development',
            'identity_basis':'Source and target checkpoint model tensors and inherited history are unchanged; predictions/metadata are reused byte-for-byte.'}
    # Ordinary export is the previously verified801 model, not a new checkpoint export.
    marker=base.read_json(OUTPUT/'checkpoint/finalization_provenance.json')
    published={p.name:base.digest(p) for p in (OUTPUT/'checkpoint').iterdir() if p.is_file()}
    for path in (OUTPUT/'checkpoint').iterdir():
        if path.is_file() and path.name!='model.safetensors':
            base.copy_verified(path,snapshot/'published_checkpoint_metadata'/path.name,records,check)
    old_candidate=Path(receipt['immutable_prior_candidate']).resolve()
    check.require(published==receipt['prior_hf_files_sha256'],'Prior ordinary published801 receipt equality')
    for name,digest in published.items():check.require(base.digest(old_candidate/name)==digest,'Published prior801 retained '+name)
    for kind,entry in sources.items():check.require(base.digest(Path(entry['path']))==entry['sha256'],'Original source checkpoint unchanged')
    check.require(base.digest(args.predecessor_exit_receipt)==receipt_sha,'Receipt unchanged during fork')
    check.require(not torch.cuda.is_initialized(),'No CUDA initialization')
    manifest={'schema':1,'kind':'exact_training_state_run_id_fork_v3','status':'complete','passed':True,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'source_loop_id':args.source_loop_id,
        'source_run_id':source_id,'target_run_id':target_id,'source_completed_step':args.expected_latest_step,'inherited_selected_step':args.expected_best_step,
        'source_review_kind':'resource_stop_full_state_DEV_review','fresh_TEST_evaluated':False,'fresh_export_published':False,
        'predecessor_exit_receipt_path':str(args.predecessor_exit_receipt.resolve()),'predecessor_exit_receipt_sha256':receipt_sha,
        'predecessor_status_sha256':base.digest(source_loop/'loop_status.json'),'source_checkpoints':sources,
        'fork_checkpoints':forks,'only_changed_path':'run_id','inherited_config_identity':config['config_identity'],
        'config_identity_sha256':base.json_digest(config['config_identity']),'inherited_development_reuse':dev,
        'snapshot_files':records,'snapshot_file_count':len(records),'snapshot_bytes':sum(x['size_bytes'] for x in records.values()),
        'retained_predecessor_fork_manifest_path':str(source_loop/'fork_snapshot/fork_manifest.json'),
        'retained_predecessor_fork_manifest_sha256':base.digest(source_loop/'fork_snapshot/fork_manifest.json'),
        'preserved_published_export':{'kind':'prior_verified801_export_not_new_checkpoint_export','run_id':marker['run_id'],
            'candidate_path':str(old_candidate.resolve()),'files_sha256':published},
        'fork_helper_path':str(Path(__file__).resolve()),'fork_helper_sha256':base.digest(Path(__file__)),
        'io_helper_sha256':base.digest(OUTPUT/'atomic_io_retry_v1.py'),'original_fork_helper_sha256':PRIOR_FORK_HELPER_SHA256,
        'checks':check.count,'raw_serialized_proof_checks':raw_check.count,'tensor_leaves_compared':raw_check.tensor_count,
        'cuda_initialized':False,'source_artifacts_mutated':False}
    manifest_path=snapshot/'fork_manifest.json';write_json(manifest_path,manifest)
    proof_path=loop/'fork_snapshot_independent_verification.json'
    proof={'schema':1,'kind':'raw_serialized_exact_fork_CPU_verification_v3','passed':True,
        'fork_manifest_sha256':base.digest(manifest_path),'source_run_id':source_id,'target_run_id':target_id,
        'only_changed_path':'run_id','checks':raw_check.count,'tensor_leaves_compared':raw_check.tensor_count,
        'signed_zero_byte_identity_checked':True,'CUDA_initialized':False,
        'scope':'Separate raw tensor-byte and container/value pass over both actual serialized checkpoints after reload.'}
    write_json(proof_path,proof)
    archive_dir=snapshot/'initial_fork_states';archive={}
    for kind,entry in forks.items():
        path=Path(entry['path']);destination=archive_dir/path.name
        copied={};base.copy_verified(path,destination,copied,check)
        archive[kind]={'mutable_work_path':str(path),'mutable_work_path_initial_sha256':entry['sha256'],
            'archive_path':str(destination.resolve()),'archive_sha256':entry['sha256'],'size_bytes':entry['size_bytes'],
            'step':entry['step'],'selected_step':args.expected_best_step,'parent_source_checkpoint_sha256':sources[kind]['sha256']}
    write_json(loop/'fork_initial_states_archive.json',{'schema':1,'kind':'immutable_initial_fork_states_archive_v3',
        'status':'complete','passed':True,'source_run_id':source_id,'target_run_id':target_id,
        'fork_manifest_sha256':base.digest(manifest_path),'independent_fork_verification_sha256':base.digest(proof_path),
        'states':archive,'CUDA_initialized':False})
    print(json.dumps({'passed':True,'manifest_path':str(manifest_path),'manifest_sha256':base.digest(manifest_path),
        'checks':check.count,'raw_proof_checks':raw_check.count,'CUDA_initialized':False}))
    return manifest


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    for key in ('source-loop-id','source-run-id','target-run-id'):
        result.add_argument('--'+key,type=base.checked_id,required=True)
    result.add_argument('--predecessor-exit-receipt',type=Path,required=True)
    result.add_argument('--output-dir',type=Path,required=True)
    result.add_argument('--expected-latest-step',type=int,default=1821)
    result.add_argument('--predecessor-exit-receipt-sha256',required=True)
    result.add_argument('--expected-owned-session',type=int,default=3634)
    result.add_argument('--expected-best-step',type=int,default=1821)
    return result


if __name__=='__main__':run(parser().parse_args())
