"""CPU-only immutable-input command preparation after an exact smoke fork.

Preserves the original fork receipt and emits new versioned command/evidence.
No checkpoint tensors are changed and no trainer or CUDA work is launched.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from atomic_io_retry_v1 import read_json, write_json
from cuda_cache_reclaim_v1 import file_sha
from runtime import ROOT, OUTPUT


def main(args):
    if not args.smoke_id.replace('_','').isalnum():raise ValueError('Simple owned smoke ID required')
    if not 360<=args.max_wall_seconds<=600:raise ValueError('Reviewed smoke wall limit360..600sec required')
    directory=OUTPUT/'runs'/args.smoke_id
    source=directory/'fork_receipt.json';receipt=read_json(source)
    if receipt['smoke_run_id']!=args.smoke_id or not receipt['CPU_only'] or receipt['CUDA_initialized']:
        raise ValueError('CPU-only exact fork receipt required')
    if sum(item['recursive_exact_comparison']['nodes'] for item in receipt['forks'])<180_000:
        raise ValueError('Complete full checkpoint recursive proof required')
    latest=next(item for item in receipt['forks'] if item['role']=='latest')
    best=next(item for item in receipt['forks'] if item['role']=='best')
    if latest['source_step']!=907 or best['source_step']!=801:raise ValueError('Reviewed907/801 fork required')
    for item in receipt['forks']:
        if item['only_changed_checkpoint_field']!='run_id':raise ValueError('Only run_id may change')
        for field,digest in [('source_path','source_sha256'),('fork_path','fork_sha256'),('runtime_path','runtime_sha256_at_creation')]:
            if file_sha(item[field])!=item[digest]:raise ValueError('Exact fork/source bytes changed: '+field)
    command=list(receipt['command'])
    if Path(command[4]).name!='launch_with_memory_guard_v2.py':raise ValueError('Exact6GB wrapperv2 required')
    command[command.index('--resume-exact')+1]=latest['fork_path']
    command[command.index('--max-wall-seconds')+1]=str(args.max_wall_seconds)
    for flag,basename in [('--provenance','smoke_io_retry_provenance_v2.json'),
                          ('--cache-provenance','smoke_cache_reclamation_provenance_v2.json'),
                          ('--memory-provenance','smoke_memory_guard_provenance_v2.json')]:
        path=directory/basename
        if path.exists():raise FileExistsError('New smoke evidence required: '+str(path))
        command[command.index(flag)+1]=str(path)
    if command[command.index('--threshold-gb')+1]!='6' or command[command.index('--cap-gb')+1]!='13':
        raise ValueError('Exact6GB cache/13GB cap required')
    result=dict(receipt)
    result.update(command=command,training_wall_argument_seconds=args.max_wall_seconds,
                  original_fork_receipt_path=str(source),original_fork_receipt_sha256=file_sha(source),
                  exact_resume_input_is_immutable_fork=True,exact_resume_input_path=latest['fork_path'])
    result['source_sha256']=dict(receipt['source_sha256'])
    result['source_sha256'][Path(__file__).name]=file_sha(__file__)
    for name,value in [('fork_receipt_v2.json',result),('smoke_command_v2.json',command)]:
        path=directory/name
        if path.exists():raise FileExistsError('Existing prepared evidence is immutable: '+str(path))
        write_json(path,value)
    print({'prepared':True,'command_path':str(directory/'smoke_command_v2.json'),
           'receipt_sha256':file_sha(directory/'fork_receipt_v2.json'),'wall_seconds':args.max_wall_seconds})


def parser():
    value=argparse.ArgumentParser(description=__doc__)
    value.add_argument('--smoke-id',default='cache_guard_smoke_20261010_02')
    value.add_argument('--max-wall-seconds',type=int,default=420)
    return value


if __name__=='__main__':main(parser().parse_args())
