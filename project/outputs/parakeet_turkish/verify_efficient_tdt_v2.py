"""CPU original/new exact-loss and gradient audit; never uses the GPU.

Tests checkpointed full-vocabulary normalization plus the vectorized DP, with
padded B2/B4, BF16, all reductions, arbitrary upstream weights, sigma, reordered
duration IDs, input/head freezing, and explicit invalid-input behavior. CPU
timings concern synthetic DP alone and are not GPU training throughput claims.
"""
import copy
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn
from transformers.loss.loss_tdt import tdt_loss as full_tdt_loss

import efficient_tdt as original
import efficient_tdt_v2 as optimized

OUTPUT=Path(__file__).resolve().parent
torch.set_num_threads(2)


class Joint(nn.Module):
    def __init__(self,hidden,output):
        super().__init__();self.head=nn.Linear(hidden,output)
    def forward(self,ep,dp):
        return self.head(torch.relu(ep+dp))


def gradient_comparison(pairs):
    assert all(a is not None and b is not None for a,b in pairs)
    left=torch.cat([a.float().flatten() for a,b in pairs])
    right=torch.cat([b.float().flatten() for a,b in pairs])
    return {'all_gradients_finite':bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
            'maximum_absolute_gradient_difference':float((left-right).abs().max()),
            'relative_gradient_l2_difference':float(torch.linalg.vector_norm(left-right)/torch.linalg.vector_norm(left).clamp_min(1e-20)),
            'gradient_cosine_similarity':float(nn.functional.cosine_similarity(left,right,dim=0))}


def run_case(name,batch=2,frames=37,targets=16,vocab=41,chunk=7,
             sigma=0.,durations=(0,1,2,3,4),reduction='mean',bf16=False,
             recompute=True,full_reference=False,frozen_inputs=False,
             frozen_head=False,list_lengths=False,blank_id=None,seed=91023):
    torch.manual_seed(seed)
    models=[Joint(13,vocab+len(durations))]
    models.append(copy.deepcopy(models[0]))
    if full_reference:models.append(copy.deepcopy(models[0]))
    if frozen_head:
        for model in models:model.requires_grad_(False)
    ep0=torch.randn(batch,frames,13)
    dp0=torch.randn(batch,targets+1,13)
    eps=[ep0.clone().requires_grad_(not frozen_inputs) for _ in models]
    dps=[dp0.clone().requires_grad_(not frozen_inputs) for _ in models]
    lengths=torch.tensor([max(1,frames-index*max(1,frames//(batch+2))) for index in range(batch)])
    target_lengths=torch.tensor([max(1,targets-index*max(1,targets//(batch+2))) for index in range(batch)])
    blank_id=vocab-1 if blank_id is None else blank_id
    labels=torch.randint(0,vocab-1,(batch,targets))
    labels+=(labels>=blank_id).long()
    if blank_id<vocab-1:labels[:,0]=vocab-1
    for index in range(batch):labels[index,target_lengths[index]:]=blank_id
    upstream=torch.arange(1,batch+1,dtype=torch.float32)*.37 if reduction=='none' else torch.tensor(.83)
    losses=[];gradients=[];padding=[]
    for index,(model,ep,dp) in enumerate(zip(models,eps,dps)):
        with torch.autocast('cpu',dtype=torch.bfloat16,enabled=bf16):
            if index==2:
                logits=model(ep[:,:,None,:],dp[:,None,:,:])
                loss=full_tdt_loss(logits[...,:vocab],logits[...,vocab:],labels,lengths,target_lengths,
                                   blank_id,list(durations),sigma=sigma,reduction=reduction)
            else:
                function=original.chunked_tdt_loss if index==0 else optimized.chunked_tdt_loss
                loss=function(model,ep,dp,labels,vocab,blank_id,list(durations),
                              lengths.tolist() if index==1 and list_lengths else lengths,
                              target_lengths.tolist() if index==1 and list_lengths else target_lengths,
                              chunk_size=chunk,sigma=sigma,reduction=reduction,recompute=recompute)
        (loss*upstream).sum().backward()
        losses.append(loss.detach())
        parameters=[]
        if not frozen_inputs:parameters.extend([ep.grad.clone(),dp.grad.clone()])
        if not frozen_head:parameters.extend(p.grad.clone() for p in model.parameters())
        gradients.append(parameters)
        if not frozen_inputs:
            padded=max(max(float(ep.grad[b,lengths[b]:].abs().max()) if lengths[b]<frames else 0.,
                           float(dp.grad[b,target_lengths[b]+1:].abs().max()) if target_lengths[b]<targets else 0.)
                       for b in range(batch))
            padding.append(padded)
    result={'name':name,'batch':batch,'frames':frames,'targets':targets,'vocab':vocab,'chunk_size':chunk,
            'durations':list(durations),'sigma':sigma,'reduction':reduction,'blank_id':blank_id,
            'logit_lengths':lengths.tolist(),'target_lengths':target_lengths.tolist(),
            'dtype':'BF16 autocast with FP32 probability/DP' if bf16 else 'FP32',
            'recompute':recompute,'list_lengths':list_lengths,
            'loss_shape':list(losses[1].shape),
            'absolute_loss_difference':float((losses[0]-losses[1]).abs().max()),
            **gradient_comparison(list(zip(gradients[0],gradients[1]))),
            'maximum_gradient_in_padded_inputs':max(padding) if padding else None}
    scale=max(1.,float(losses[0].abs().max()))
    tolerance=.01 if bf16 else 2e-5
    passed=result['all_gradients_finite'] and result['absolute_loss_difference']<=scale*2e-6
    passed &= result['relative_gradient_l2_difference']<tolerance and result['gradient_cosine_similarity']>.9999
    passed &= not padding or max(padding)==0.
    if reduction=='none':passed &= result['loss_shape']==[batch]
    if blank_id<vocab-1 and not frozen_head:
        result['appended_token_head_gradient_nonzero']=bool(models[1].head.weight.grad[blank_id+1:vocab].abs().sum()>0)
        passed &= result['appended_token_head_gradient_nonzero']
    if full_reference:
        full={'absolute_loss_difference':float((losses[2]-losses[1]).abs().max()),
              **gradient_comparison(list(zip(gradients[2],gradients[1])))}
        full['passed']=full['all_gradients_finite'] and full['absolute_loss_difference']<=scale*2e-6 and full['relative_gradient_l2_difference']<tolerance
        result['pinned_full_reference']=full;passed &= full['passed']
    result['passed']=bool(passed)
    assert result['passed'],result
    print(name, 'passed',flush=True)
    return result


def replay_kl_case():
    """Combining the new DP with sparse replay KL preserves all gradients."""
    torch.manual_seed(64119)
    batch,frames,targets,old_vocab,vocab,hidden=2,31,12,31,37,11
    durations=[0,1,2,3,4]
    left=Joint(hidden,vocab+5);right=copy.deepcopy(left)
    teacher=Joint(hidden,old_vocab+5).eval().requires_grad_(False)
    ep=torch.randn(batch,frames,hidden,requires_grad=True);dp=torch.randn(batch,targets+1,hidden,requires_grad=True)
    ep2=ep.detach().clone().requires_grad_();dp2=dp.detach().clone().requires_grad_()
    labels=torch.randint(0,old_vocab-1,(batch,targets))
    lengths=torch.tensor([31,24]);target_lengths=torch.tensor([12,8])
    ts=torch.tensor([0,3,8,13,21]);us=torch.tensor([0,2,4,6,8])
    original_features=torch.randn(batch,frames,hidden)
    with torch.no_grad():distribution=teacher(original_features[:,ts,None],dp.detach()[:,None,us])[...,:old_vocab].softmax(-1)
    losses=[];kls=[]
    for function,model,enc,dec in ((original.chunked_tdt_loss,left,ep,dp),(optimized.chunked_tdt_loss,right,ep2,dp2)):
        student=model(enc[:,ts,None],dec[:,None,us])[...,:vocab].log_softmax(-1)
        kl=(distribution*(distribution.clamp_min(1e-12).log()-student[...,:old_vocab])).sum(-1).mean()
        loss=function(model,enc,dec,labels,vocab,old_vocab-1,durations,lengths,target_lengths,chunk_size=6)+5.*kl
        loss.backward();losses.append(loss.detach());kls.append(kl.detach())
    result={'name':'sparse_teacher_KL_with_new_DP','teacher_encoder_features_frozen_and_original':True,
            'coefficient':5.,'student_normalization_vocab':vocab,'teacher_vocab':old_vocab,
            'absolute_loss_difference':float((losses[0]-losses[1]).abs()),
            'absolute_replay_KL_difference':float((kls[0]-kls[1]).abs()),
            'new_row_gradient_nonzero':bool(right.head.weight.grad[old_vocab:vocab].abs().sum()>0),
            **gradient_comparison([(ep.grad,ep2.grad),(dp.grad,dp2.grad)]+[(p.grad,q.grad) for p,q in zip(left.parameters(),right.parameters())])}
    result['passed']=result['all_gradients_finite'] and result['absolute_loss_difference']<1e-5 and result['absolute_replay_KL_difference']==0. and result['relative_gradient_l2_difference']<2e-5 and result['new_row_gradient_nonzero']
    assert result['passed'],result
    print(result['name'],'passed',flush=True)
    return result


def direct_dp_case(name,batch,frames,targets,durations,reduction='none'):
    torch.manual_seed(781)
    blank=-torch.rand(batch,frames,targets+1)*5
    label=-torch.rand(batch,frames,targets)*5
    duration=torch.randn(batch,frames,targets+1,len(durations)).log_softmax(-1)
    leaves=[[value.clone().requires_grad_() for value in (blank,label,duration)] for _ in range(2)]
    lengths=torch.tensor([frames-3*b for b in range(batch)])
    target_lengths=torch.tensor([targets-2*b for b in range(batch)])
    weights=torch.linspace(.3,1.7,batch) if reduction=='none' else 1.
    losses=[]
    for function,inputs in zip((original.tdt_from_log_probs,optimized.tdt_from_log_probs),leaves):
        loss=function(*inputs,lengths,target_lengths,list(durations),reduction)
        (loss*weights).sum().backward();losses.append(loss.detach())
    result={'name':name,'batch':batch,'frames':frames,'targets':targets,'durations':list(durations),
            'absolute_loss_difference':float((losses[0]-losses[1]).abs().max()),
            **gradient_comparison([(a.grad,b.grad) for a,b in zip(*leaves)])}
    result['passed']=result['all_gradients_finite'] and result['absolute_loss_difference']<1e-4 and result['relative_gradient_l2_difference']<2e-5
    assert result['passed'],result
    print(name,'passed',flush=True)
    return result


def edge_cases():
    joint=Joint(7,16);ep=torch.randn(1,8,7);dp=torch.randn(1,3,7);labels=torch.tensor([[1,2]])
    base=(joint,ep,dp,labels,11,10,[0,1,2,3,4])
    checks=[]
    def rejects(name,args=base,**kwargs):
        try:optimized.chunked_tdt_loss(*args,**kwargs)
        except ValueError as error:checks.append({'name':name,'rejected':True,'error':str(error)})
        else:raise AssertionError(name)
    rejects('empty targets',args=(joint,ep,torch.randn(1,1,7),torch.empty(1,0,dtype=torch.long),11,10,[0,1,2,3,4]))
    rejects('zero effective targets',target_lengths=[0])
    rejects('zero encoder length',logit_lengths=[0])
    rejects('out of range encoder length',logit_lengths=[9])
    rejects('out of range target length',target_lengths=[3])
    rejects('fractional length',target_lengths=[1.5])
    rejects('wrong batch length count',target_lengths=[1,2])
    rejects('invalid reduction',reduction='unsupported')
    rejects('zero chunk',chunk_size=0)
    rejects('negative duration',args=base[:-1]+([0,1,-2,3,4],))
    rejects('no positive terminal duration',args=base[:-1]+([0],))
    # Non-unit duration sets inherit upstream unreachable-state NaN gradients.
    durations=[0,2,4];inputs=(torch.randn(1,9,3).log_softmax(-1),torch.randn(1,9,2).log_softmax(-1),torch.randn(1,9,3,3).log_softmax(-1))
    observations=[]
    for module in (original,optimized):
        leaves=[x.detach().clone().requires_grad_() for x in inputs]
        loss=module.tdt_from_log_probs(*leaves,torch.tensor([9]),torch.tensor([2]),durations)
        loss.backward()
        observations.append({'finite_loss':bool(torch.isfinite(loss)),
                             'all_gradients_finite':all(bool(torch.isfinite(x.grad).all()) for x in leaves)})
    assert observations[0]==observations[1],observations
    checks.append({'name':'duration set without one','duration_values':durations,'reference':observations[0],
                   'optimized':observations[1],'recommendation':'Use the released [0,1,2,3,4] durations; unreachable states inherit upstream nonfinite gradients.'})
    return checks


def dp_timing():
    measurements=[]
    for batch,frames,targets in ((1,105,67),(1,293,112),(4,70,30)):
        torch.manual_seed(731);args=(-torch.rand(batch,frames,targets+1),-torch.rand(batch,frames,targets),
            torch.randn(batch,frames,targets+1,5).log_softmax(-1),torch.full((batch,),frames,dtype=torch.long),
            torch.full((batch,),targets,dtype=torch.long),[0,1,2,3,4])
        optimized.clear_dp_plan_cache();start=time.perf_counter()
        plan=optimized.dp_plan_statistics(frames,targets+1,[0,1,2,3,4])
        row={'shape':[batch,frames,targets+1],'cpu_threads':torch.get_num_threads(),
             'cold_host_plan_construction_seconds':time.perf_counter()-start,'static_plan':plan}
        for name,function in (('original',original.tdt_from_log_probs),('optimized',optimized.tdt_from_log_probs)):
            samples=[]
            for repeat in range(4):
                leaves=[x.clone().requires_grad_() for x in args[:3]]
                start=time.perf_counter();loss=function(*leaves,*args[3:]);loss.backward();elapsed=time.perf_counter()-start
                if repeat:samples.append(elapsed)
            row[name+'_warm_forward_backward_seconds']=samples
        row['warm_median_cpu_speedup']=float(torch.tensor(row['original_warm_forward_backward_seconds']).median()/torch.tensor(row['optimized_warm_forward_backward_seconds']).median())
        measurements.append(row)
    return measurements


def main():
    cases=[]
    for batch in (2,4):
        for reduction in optimized.REDUCTIONS:
            cases.append(run_case(f'padded_B{batch}_{reduction}',batch=batch,reduction=reduction,sigma=.13,full_reference=reduction in ('mean','none')))
    for batch in (2,4):
        cases.append(run_case(f'bf16_padded_B{batch}',batch=batch,bf16=True,sigma=.17,reduction='none',full_reference=True))
    cases.append(run_case('bf16_no_recompute',batch=4,bf16=True,recompute=False,sigma=.17))
    cases.append(run_case('reordered_duration_indices',batch=4,durations=(4,0,2,1,3),sigma=.5,chunk=99,list_lengths=True,full_reference=True))
    cases.append(run_case('large_duration_and_sigma',batch=2,frames=57,targets=23,durations=(0,1,3,8,80),sigma=1.,chunk=11,full_reference=True))
    cases.append(run_case('small_frame_count',batch=2,frames=2,targets=5,chunk=1,full_reference=True))
    cases.append(run_case('frozen_projected_inputs',frozen_inputs=True))
    cases.append(run_case('frozen_joint_trainable_inputs',frozen_head=True))
    cases.append(run_case('single_example_no_padding',batch=1,frames=47,targets=10,reduction='none',list_lengths=True,full_reference=True))
    cases.append(run_case('interior_blank_appended_labels',batch=4,blank_id=30,vocab=37,reduction='none',full_reference=True))
    cases.append(run_case('bf16_interior_blank_appended_labels',batch=2,blank_id=30,vocab=37,reduction='none',full_reference=True,bf16=True))
    cases.append(replay_kl_case())
    cases.append(direct_dp_case('long_DP_B2',2,105,67,(0,1,2,3,4)))
    cases.append(direct_dp_case('long_DP_B4',4,80,39,(0,1,3,8)))
    edges=edge_cases();timings=dp_timing()
    report={'passed':True,'gpu_used':False,'source_sha256':hashlib.sha256((OUTPUT/'efficient_tdt_v2.py').read_bytes()).hexdigest(),
            'reference_source_sha256':hashlib.sha256((OUTPUT/'efficient_tdt.py').read_bytes()).hexdigest(),
            'reference':'Unchanged efficient_tdt.py and pinned Transformers full TDT loss 536ecc007387a50e77603bb5d92100e9b07514cc',
            'independent_source_review':{'reviewer':'retention_design worker','gpu_used':False,
                'findings':[
                    'Flattened blank/label token concatenation and incoming arc indices match the recurrence.',
                    'Per-item terminal T/U and monotone paths prevent padded states from affecting selected probabilities.',
                    'Reduction none returns raw NLL; caller must divide by target length to reproduce prior mean.',
                    'No source blocker found for released durations [0,1,2,3,4]; four-shape index cache is bounded.'
                ]},
            'cases':cases,'edge_cases':edges,'synthetic_cpu_dp_timings':timings,
            'scope':'CPU loss/all-gradient parity; full-vocabulary normalization, checkpoint recomputation and padded batches. No GPU measurements.',
            'limitations':[
                'FP32 terminal sum grouping changes rounding; mathematical objective is identical rather than bitwise equality.',
                'CPU BF16 parity does not replace GPU BF16/actual-clip checks.',
                'CPU synthetic DP timings do not establish end-to-end training speed or GPU memory.',
                'Duration sets missing one inherit upstream unreachable-state nonfinite-gradient behavior.',
                'Random shapes incur cold host-plan construction; the four-shape cache is bounded and stores indices only.',
                'CUDA length tensors need a combined host metadata copy; CPU lengths supplied by the data loader avoid that sync.'
            ]}
    path=OUTPUT/'efficient_tdt_v2_cpu_verification.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'passed':True,'cases':len(cases),'edge_cases':len(edges),
                      'max_loss_difference':max(case['absolute_loss_difference'] for case in cases),
                      'max_relative_gradient_l2_difference':max(case['relative_gradient_l2_difference'] for case in cases),
                      'output':str(path),'synthetic_cpu_timings':timings},indent=2))


if __name__=='__main__':main()
