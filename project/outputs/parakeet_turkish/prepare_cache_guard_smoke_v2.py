"""CPU-only exact logical fork for a NEW frozen-worker cache/cap smoke.

Only checkpoint top-level run_id changes. Immutable fork snapshots and separate
runtime latest/best copies preserve originals. No trainer/GPU process launches.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import torch

from atomic_io_retry_v1 import read_bytes_shared, write_json
from cuda_cache_reclaim_v1 import file_sha
from runtime import ROOT, OUTPUT


def exact(left,right,path='state',counts=None):
    counts=counts if counts is not None else {'nodes':0,'tensors':0}
    counts['nodes']+=1
    if type(left) is not type(right):raise AssertionError('Changed type '+path)
    if torch.is_tensor(left):
        counts['tensors']+=1
        if left.dtype!=right.dtype or left.shape!=right.shape or not torch.equal(left,right):raise AssertionError('Changed tensor '+path)
    elif isinstance(left,dict):
        if left.keys()!=right.keys():raise AssertionError('Changed keys '+path)
        for key in left:exact(left[key],right[key],path+'.'+str(key),counts)
    elif isinstance(left,(tuple,list)):
        if len(left)!=len(right):raise AssertionError('Changed length '+path)
        for index,(a,b) in enumerate(zip(left,right)):exact(a,b,path+'.'+str(index),counts)
    elif left!=right:raise AssertionError('Changed scalar '+path)
    return counts


def main(args):
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    if not 120<=args.max_wall_seconds<=600:raise ValueError('Bounded smoke process-wall limit120..600sec required')
    if not args.smoke_id.replace('_','').isalnum() or args.smoke_id==args.source_run_id:raise ValueError('New simple smoke run ID required')
    directory=OUTPUT/'runs'/args.smoke_id
    source_dir=OUTPUT/'runs'/args.source_run_id
    source_config=source_dir/'run_config.json'
    config=json.loads(read_bytes_shared(source_config))
    snapshots=directory/'fork_snapshot'
    runtime_paths={role:ROOT/'work'/f'{args.smoke_id}_{role}.pt' for role in ('latest','best')}
    if directory.exists() or any(path.exists() for path in runtime_paths.values()):raise FileExistsError('Smoke fork already exists')
    snapshots.mkdir(parents=True)
    evidence=[]
    for role in ('latest','best'):
        source=ROOT/'work'/f'{args.source_run_id}_{role}.pt';source_sha=file_sha(source)
        state=torch.load(source,map_location='cpu',weights_only=True)
        if state['run_id']!=args.source_run_id or state['config_identity']!=config['config_identity']:
            raise ValueError('Full original checkpoint/config identity required')
        if len(state['weights'])!=207 or len(state['optimizer']['state'])!=207:raise ValueError('Complete207weights/Adamstates required')
        if role=='latest' and state['step']!=907:raise ValueError('Reviewed complete907-step source required')
        fork=dict(state);fork['run_id']=args.smoke_id
        immutable=snapshots/f'{role}.pt';torch.save(fork,immutable)
        restored=torch.load(immutable,map_location='cpu',weights_only=True)
        changed_run=restored['run_id'];restored['run_id']=args.source_run_id
        checks=exact(state,restored)
        restored['run_id']=changed_run
        shutil.copyfile(immutable,runtime_paths[role])
        if file_sha(immutable)!=file_sha(runtime_paths[role]) or file_sha(source)!=source_sha:
            raise RuntimeError('Fork copy/source byte identity changed')
        evidence.append({'role':role,'source_path':str(source),'source_sha256':source_sha,'source_run_id':state['run_id'],
            'source_step':state['step'],'selected_step':state['selected_step'],'fork_path':str(immutable),
            'fork_sha256':file_sha(immutable),'runtime_path':str(runtime_paths[role]),
            'runtime_sha256_at_creation':file_sha(runtime_paths[role]),'only_changed_checkpoint_field':'run_id',
            'recursive_exact_comparison':checks,'optimizer_states':207,'trainable_tensors':207})
        del state,fork,restored;gc.collect()
    if evidence[1]['source_step']!=evidence[0]['selected_step']:raise ValueError('Original best must match latest selection')
    io=directory/'smoke_io_retry_provenance.json';cache=directory/'smoke_cache_reclamation_provenance.json'
    memory=directory/'smoke_memory_guard_provenance.json';stop=ROOT/'work'/f'{args.smoke_id}.stop'
    if stop.exists():raise FileExistsError('Owned new stop path already exists')
    worker_args=['--run-id',args.smoke_id,'--tr-manifest',config['tr_manifest'],'--replay-manifest',config['replay_manifest'],
                 '--resume-exact',str(runtime_paths['latest']),'--max-wall-seconds',str(args.max_wall_seconds),'--stop-file',str(stop),'--log-every','1']
    for key,value in config['config_identity']['training_arguments'].items():
        if value is None or value is False:continue
        worker_args.append('--'+key.replace('_','-'))
        if value is not True:worker_args.append(str(value))
    command=[str(ROOT/'work/training-env/Scripts/python.exe'),'-X','utf8','-u',str(OUTPUT/'launch_with_memory_guard_v2.py'),
        '--provenance',str(io),'--cache-provenance',str(cache),'--memory-provenance',str(memory),'--threshold-gb','6','--cap-gb','13','--',*worker_args]
    receipt={'schema':1,'kind':'cache_guard_frozen_worker_smoke_fork_v2','status':'prepared_not_run',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'source_run_id':args.source_run_id,'smoke_run_id':args.smoke_id,
        'CPU_only':True,'CUDA_initialized':torch.cuda.is_initialized(),'forks':evidence,
        'original_config_sha256':file_sha(source_config),'original_config_path':str(source_config),
        'source_sha256':{name:file_sha(OUTPUT/name) for name in ('train_lora_batched.py','launch_with_io_retry_v1.py',
            'atomic_io_retry_v1.py','cuda_cache_reclaim_v1.py','launch_with_cache_reclaim_v1.py','launch_with_memory_guard_v2.py',
            'prepare_cache_guard_smoke_v2.py')},'command':command,'stop_path':str(stop),
        'all_provenance_paths_distinct_and_new':True,'allocator_cap_bytes':13_000_000_000,'cleanup_threshold_bytes':6_000_000_000,
        'original_model_optimizer_sampler_RNG_config_preserved':True,'batch_size':8,'accumulation':2,'effective_batch':16,
        'training_wall_argument_seconds':args.max_wall_seconds,'wall_semantics':'Frozen soft process-wall limit includes startup; final DEV/checkpoint work occurs after stopping.',
        'production_hook_assertion':'Pending actual GPU execution: require matching run:539 eligible calls>=2, cap applied, full finite updates and progress beyond907.',
        'quality_scope':'Functional memory/hook smoke only; no TEST evaluation or experiment selection; original full3epoch identity remains intact.'}
    write_json(directory/'fork_receipt.json',receipt)
    write_json(directory/'smoke_command.json',command)
    print(json.dumps({'prepared':True,'smoke_id':args.smoke_id,'receipt':str(directory/'fork_receipt.json'),
                      'recursive_checks':sum(row['recursive_exact_comparison']['nodes'] for row in evidence),'CUDA_initialized':torch.cuda.is_initialized()}))


def parser():
    value=argparse.ArgumentParser(description=__doc__)
    value.add_argument('--source-run-id',default='adaptive_20261010_recovery_b8_01_b8_quality')
    value.add_argument('--smoke-id',default='cache_guard_smoke_20261010_02')
    value.add_argument('--max-wall-seconds',type=int,default=360)
    return value


if __name__=='__main__':main(parser().parse_args())
