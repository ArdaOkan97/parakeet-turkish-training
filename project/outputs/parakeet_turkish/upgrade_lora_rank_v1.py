"""CPU-only rank expansion for a NEW fresh-optimizer warm start.

Copy existing A rows/B columns, initialize additional A rows randomly, and zero
additional B columns. Preserve alpha/rank. This is not an exact optimizer resume.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import torch


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def upgrade(saved, *, rank=16, alpha=32., seed=20261010, output_run_id=None):
    metadata=copy.deepcopy(saved['lora_metadata'])
    targets=metadata['targets']
    if not targets or not isinstance(rank,int) or isinstance(rank,bool):
        raise ValueError('Nonempty known LoRA layout and integer rank required')
    generator=torch.Generator(device='cpu').manual_seed(seed)
    weights={}
    known_adapters=set()
    for target in targets:
        old_rank=target['rank'];scaling=target['alpha']/old_rank
        if old_rank!=8 or target['alpha']!=16. or rank!=16 or alpha!=32. or alpha/rank!=scaling:
            raise ValueError('This measured upgrade requires rank8/alpha16 to rank16/alpha32, scaling2')
        if target['dropout']!=0.:
            raise ValueError('Function-preserving proof requires the frozen zero-dropout adapter layout')
        a_key=target['module']+'.lora_A';b_key=target['module']+'.lora_B'
        a=saved['weights'][a_key];b=saved['weights'][b_key]
        if (a.shape!=(old_rank,target['in_features']) or b.shape!=(target['out_features'],old_rank)
            or a.dtype!=torch.float32 or b.dtype!=torch.float32 or a.device.type!='cpu' or b.device.type!='cpu'
            or not torch.isfinite(a).all() or not torch.isfinite(b).all()):
            raise ValueError('Original adapter tensor layout/FP32/finiteness differs')
        new_a=torch.empty((rank,target['in_features']),dtype=torch.float32)
        # Same distribution as the frozen nn.init.kaiming_uniform_(a=sqrt(5)).
        bound=1/math.sqrt(target['in_features'])
        new_a.uniform_(-bound,bound,generator=generator)
        new_b=torch.zeros((target['out_features'],rank),dtype=torch.float32)
        new_a[:old_rank].copy_(a);new_b[:,:old_rank].copy_(b)
        weights[a_key]=new_a;weights[b_key]=new_b;known_adapters.update((a_key,b_key))
        target.update(rank=rank,alpha=float(alpha),scaling=alpha/rank,
                      adapter_parameters=rank*(target['in_features']+target['out_features']))
    for key,value in saved['weights'].items():
        if key not in known_adapters:
            if key.endswith(('.lora_A','.lora_B')):raise ValueError('Untracked adapter tensor')
            weights[key]=value.detach().cpu().clone()
    metadata['adapter_parameter_count']=sum(x['adapter_parameters'] for x in targets)
    result={'schema':2,'run_id':output_run_id or saved['run_id']+'_rank16_warmstart',
            'step':saved['step'],'weights':weights,'lora_metadata':metadata,
            'warmstart_only':True,'exact_optimizer_resume_supported':False,
            'rank_upgrade':{'kind':'function_preserving_rank8_to16_v1','parent_run_id':saved['run_id'],
                            'parent_step':saved['step'],'rank':rank,'alpha':alpha,'scaling':alpha/rank,
                            'seed':seed,'old_A_rows_copied':True,'old_B_columns_copied':True,
                            'new_A_random':True,'new_B_zero':True,
                            'optimizer_policy':'NEW fresh optimizer; compare to fresh rank8 control',
                            'sampler_policy':'NEW matched seed/clip exposure; not inherited exact-resume cursor'}}
    return result


def main(args):
    source=args.checkpoint.resolve();target=args.output.resolve()
    root=Path(__file__).resolve().parents[2]
    if not source.is_relative_to(root) or not target.is_relative_to(root) or target.exists():
        raise ValueError('Workspace-only input/output and no overwrite required')
    before=sha(source)
    saved=torch.load(source,map_location='cpu',weights_only=True)
    result=upgrade(saved,rank=args.rank,alpha=args.alpha,seed=args.seed,output_run_id=args.output_run_id)
    result['rank_upgrade'].update(parent_checkpoint_path=str(source),parent_checkpoint_sha256=before,
                                 implementation_sha256=sha(__file__))
    target.parent.mkdir(parents=True,exist_ok=True)
    torch.save(result,target)
    if sha(source)!=before:raise RuntimeError('Immutable parent changed during upgrade')
    print(json.dumps({'output':str(target),'sha256':sha(target),'rank':args.rank,'gpu_initialized':torch.cuda.is_initialized()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--output-run-id')
    p.add_argument('--rank',type=int,default=16)
    p.add_argument('--alpha',type=float,default=32.)
    p.add_argument('--seed',type=int,default=20261010)
    raise SystemExit(main(p.parse_args()))
