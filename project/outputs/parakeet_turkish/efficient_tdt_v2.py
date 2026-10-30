"""Exact chunked TDT loss with host-planned, vectorized dynamic-program arcs.

The objective is the Apache-2.0 Transformers TDT loss from commit
536ecc007387a50e77603bb5d92100e9b07514cc. Full token normalization and the
checkpointed joint projection are reused from efficient_tdt.py. This variant
groups each anti-diagonal's incoming blank/label arcs into three index_select
operations and one logsumexp, instead of testing CUDA tensors in Python for
every duration. It uses no .any(), .item() or tensor-bool branch in the DP.

Pass logit_lengths/target_lengths as CPU integer tensors or integer sequences
when the data loader already knows them. CUDA length tensors remain supported,
with one combined small host copy for validation. Padded positions do not alter
the terminal probability. reduction='none' returns raw per-example NLL [B].
The pilot uses released durations [0,1,2,3,4]. Sparse duration sets without 1
inherit the reference implementation's unreachable-state NaN-gradient limit.
"""
from dataclasses import dataclass
from functools import lru_cache
import numbers

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from efficient_tdt import reduced_joint_probs

REDUCTIONS=('mean','mean_volume','mean_batch','sum','none')
INTEGER_DTYPES=(torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8)


@dataclass(frozen=True)
class ArcPlan:
    destination: torch.Tensor
    alpha_source: torch.Tensor
    token_source: torch.Tensor
    duration_source: torch.Tensor
    invalid: torch.Tensor
    terminal_duration_indices: torch.Tensor
    terminal_offsets: torch.Tensor
    diagonals: tuple
    original_tensor_truth_reads: int


def _durations(values):
    values=tuple(values)
    if not values or any(isinstance(x,bool) or not isinstance(x,numbers.Integral) or x<0 for x in values):
        raise ValueError('durations must be a nonempty sequence of nonnegative integers')
    if not any(x>0 for x in values):
        raise ValueError('TDT requires a positive duration for terminal blank arcs')
    return tuple(int(x) for x in values)


@lru_cache(maxsize=4)
def _arc_plan(max_t,max_u,durations,device_name):
    """Build index metadata on host, then transfer packed arrays once.

    Cache size is bounded because random-length clips can produce many shapes.
    No audio, labels, activations, or computation graphs are cached.
    """
    destinations=[];alpha_sources=[];token_sources=[];duration_sources=[];masks=[]
    diagonals=[];dst_offset=arc_offset=truth_reads=0
    num_durations=len(durations)
    token_width=2*max_u-1
    for n in range(1,max_t+max_u-1):
        us=np.arange(max(0,n-max_t+1),min(n+1,max_u),dtype=np.int64)
        ts=n-us
        diagonal_sources=[];diagonal_tokens=[];diagonal_durations=[];diagonal_masks=[]
        for index,duration in enumerate(durations):
            previous=ts-duration
            valid_t=previous>=0
            # These NumPy booleans are host-only, derived from static dimensions.
            truth_reads+=1
            if not bool(np.any(valid_t)):
                continue
            source_t=np.maximum(previous,0)
            if duration>0:
                source=source_t*max_u+us
                diagonal_sources.append(source)
                diagonal_tokens.append(source_t*token_width+us)
                diagonal_durations.append(source*num_durations+index)
                diagonal_masks.append(valid_t)
            valid_both=valid_t & (us>0)
            truth_reads+=1
            if bool(np.any(valid_both)):
                source_u=np.maximum(us-1,0)
                source=source_t*max_u+source_u
                diagonal_sources.append(source)
                diagonal_tokens.append(source_t*token_width+max_u+np.minimum(source_u,max_u-2))
                diagonal_durations.append(source*num_durations+index)
                diagonal_masks.append(valid_both)
        if not diagonal_sources:
            continue
        count=len(diagonal_sources);width=len(us);size=count*width
        destinations.append(ts*max_u+us)
        alpha_sources.append(np.stack(diagonal_sources).reshape(-1))
        token_sources.append(np.stack(diagonal_tokens).reshape(-1))
        duration_sources.append(np.stack(diagonal_durations).reshape(-1))
        # Cache the inverted mask, avoiding a GPU bitwise-not launch per diagonal.
        masks.append(~np.stack(diagonal_masks).reshape(-1))
        diagonals.append((dst_offset,dst_offset+width,arc_offset,arc_offset+size,count,width))
        dst_offset+=width;arc_offset+=size
    device=torch.device(device_name)
    def packed(arrays,dtype):
        array=np.concatenate(arrays) if arrays else np.empty(0,dtype=dtype)
        return torch.from_numpy(array).to(device=device)
    positive=[(index,duration) for index,duration in enumerate(durations) if 0<duration<=max_t]
    return ArcPlan(packed(destinations,np.int64),packed(alpha_sources,np.int64),
                   packed(token_sources,np.int64),packed(duration_sources,np.int64),
                   packed(masks,np.bool_),
                   torch.tensor([x[0] for x in positive],device=device,dtype=torch.long),
                   torch.tensor([x[1] for x in positive],device=device,dtype=torch.long),
                   tuple(diagonals),
                   truth_reads+sum(duration>0 for duration in durations))


def clear_dp_plan_cache():
    _arc_plan.cache_clear()


def dp_plan_statistics(max_t,max_u,durations,device='cpu'):
    """CPU-safe static plan counts; timings must be measured separately."""
    plan=_arc_plan(max_t,max_u,_durations(durations),str(torch.device(device)))
    tensors=(plan.destination,plan.alpha_source,plan.token_source,plan.duration_source,plan.invalid,
             plan.terminal_duration_indices,plan.terminal_offsets)
    return {'anti_diagonals':len(plan.diagonals),'padded_arc_candidates':plan.invalid.numel(),
            'index_metadata_bytes':sum(x.numel()*x.element_size() for x in tensors),
            'original_per_duration_tensor_truth_reads':plan.original_tensor_truth_reads,
            'optimized_per_duration_tensor_truth_reads':0,
            'metadata_cache_max_shapes':_arc_plan.cache_info().maxsize}


def _lengths(logit_lengths,target_lengths,batch,max_t,max_labels,device):
    supplied=[logit_lengths,target_lengths]
    defaults=[max_t,max_labels]
    values=[None,None]
    gpu_indices=[]
    for index,lengths in enumerate(supplied):
        if lengths is None:
            values[index]=[defaults[index]]*batch
        elif isinstance(lengths,torch.Tensor):
            if lengths.ndim!=1 or lengths.shape[0]!=batch or lengths.dtype not in INTEGER_DTYPES:
                raise ValueError('lengths must have shape (batch,) and integer dtype')
            if lengths.device.type=='cpu':
                values[index]=lengths.detach().tolist()
            else:
                gpu_indices.append(index)
        else:
            values[index]=list(lengths)
    if gpu_indices:
        tensors=[supplied[index].detach() for index in gpu_indices]
        if len(tensors)==2 and tensors[0].device==tensors[1].device:
            copied=torch.stack(tensors).cpu().tolist()
        else:
            copied=[tensor.cpu().tolist() for tensor in tensors]
        for index,copied_values in zip(gpu_indices,copied):
            values[index]=copied_values
    for lengths,maximum in zip(values,defaults):
        if len(lengths)!=batch or any(isinstance(x,bool) or not isinstance(x,numbers.Integral) or not 1<=x<=maximum for x in lengths):
            raise ValueError('TDT requires positive integer lengths within the padded tensor bounds')
    result=[]
    for lengths,host in zip(supplied,values):
        if isinstance(lengths,torch.Tensor):
            result.append(lengths.to(device=device,dtype=torch.long))
        else:
            result.append(torch.tensor(host,device=device,dtype=torch.long))
    return tuple(result)


def tdt_from_log_probs(blank_log_probs,label_log_probs,duration_log_probs,
                       logit_lengths,target_lengths,durations,reduction='mean'):
    if reduction not in REDUCTIONS:
        raise ValueError(f'Unsupported reduction {reduction}')
    if blank_log_probs.ndim!=3:
        raise ValueError('blank log probabilities must have shape (batch,T,U+1)')
    batch,max_t,max_u=blank_log_probs.shape
    durations=_durations(durations)
    if batch<1 or max_t<1 or max_u<2:
        raise ValueError('TDT requires a nonempty batch, positive frames and nonempty targets')
    if label_log_probs.shape!=(batch,max_t,max_u-1) or duration_log_probs.shape!=(batch,max_t,max_u,len(durations)):
        raise ValueError('Blank, label and duration log probability shapes must agree')
    device=blank_log_probs.device
    if label_log_probs.device!=device or duration_log_probs.device!=device:
        raise ValueError('All log probabilities must share a device')
    lengths,targets=_lengths(logit_lengths,target_lengths,batch,max_t,max_u-1,device)
    plan=_arc_plan(max_t,max_u,durations,str(device))
    alpha=torch.full((batch,max_t*max_u),float('-inf'),device=device,dtype=torch.float32)
    alpha[:,0]=0.
    token=torch.cat((blank_log_probs,label_log_probs),dim=-1).reshape(batch,-1)
    duration_flat=duration_log_probs.reshape(batch,-1)
    for dst_start,dst_end,arc_start,arc_end,count,width in plan.diagonals:
        arc_slice=slice(arc_start,arc_end)
        incoming=(alpha.index_select(1,plan.alpha_source[arc_slice])
                  +token.index_select(1,plan.token_source[arc_slice])
                  +duration_flat.index_select(1,plan.duration_source[arc_slice]))
        incoming=incoming.masked_fill(plan.invalid[arc_slice].unsqueeze(0),float('-inf'))
        alpha[:,plan.destination[dst_start:dst_end]]=torch.logsumexp(incoming.reshape(batch,count,width),dim=1)
    # Group all positive terminal durations. No per-duration CUDA valid.any().
    if plan.terminal_offsets.numel()==0:
        raise ValueError('No terminal duration fits the encoder frame count')
    final_t=lengths[:,None]-plan.terminal_offsets[None,:]
    valid=final_t>=0
    source=final_t.clamp(min=0)*max_u+targets[:,None]
    terminal=(alpha.gather(1,source)
              +blank_log_probs.reshape(batch,-1).gather(1,source)
              +duration_flat.gather(1,source*len(durations)+plan.terminal_duration_indices[None,:]))
    losses=-torch.logsumexp(terminal.masked_fill(~valid,float('-inf')),dim=1)
    denominator=targets.float()
    if reduction=='mean':return (losses/denominator).mean()
    if reduction=='mean_volume':return losses.sum()/denominator.sum()
    if reduction=='mean_batch':return losses.mean()
    if reduction=='sum':return losses.sum()
    return losses


def chunked_tdt_loss(joint,encoder_projected,decoder_projected,labels,
                     vocab_size,blank_id,durations,logit_lengths=None,
                     target_lengths=None,chunk_size=16,sigma=0.0,
                     reduction='mean',recompute=True):
    if type(chunk_size) is not int or chunk_size<1:
        raise ValueError('chunk_size must be a positive integer')
    if reduction not in REDUCTIONS:
        raise ValueError(f'Unsupported reduction {reduction}')
    if labels.ndim!=2 or labels.shape[1]<1:
        raise ValueError('TDT training requires a nonempty target sequence')
    if encoder_projected.ndim!=3 or decoder_projected.ndim!=3:
        raise ValueError('Projected tensors must have shape (batch,sequence,hidden)')
    if encoder_projected.shape[0]<1 or encoder_projected.shape[1]<1:
        raise ValueError('TDT requires a nonempty batch and positive frames')
    if decoder_projected.shape[1]!=labels.shape[1]+1:
        raise ValueError('Decoder sequence must include one initial blank before the targets')
    if not (labels.shape[0]==encoder_projected.shape[0]==decoder_projected.shape[0]):
        raise ValueError('Batch sizes must match')
    if not (labels.device==encoder_projected.device==decoder_projected.device):
        raise ValueError('Projected tensors and labels must share a device')
    if labels.dtype!=torch.long:labels=labels.long()
    parts=[]
    def project(ep,dp,target):
        return reduced_joint_probs(joint,ep,dp,target,vocab_size,blank_id,sigma)
    for start in range(0,encoder_projected.shape[1],chunk_size):
        ep=encoder_projected[:,start:start+chunk_size]
        if recompute and torch.is_grad_enabled():
            values=checkpoint(project,ep,decoder_projected,labels,use_reentrant=False)
        else:
            values=project(ep,decoder_projected,labels)
        parts.append(values)
    blank,label,duration=[torch.cat([part[index] for part in parts],dim=1) for index in range(3)]
    return tdt_from_log_probs(blank,label,duration,logit_lengths,target_lengths,durations,reduction)
