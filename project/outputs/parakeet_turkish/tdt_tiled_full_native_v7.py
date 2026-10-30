"""NEW fullT/U nativeDP with canonicalT1/U128 fullvocabulary tiles; unadmitted."""
from __future__ import annotations
import random
import torch

def require(ok,message):
    if not ok:raise ValueError(message)

def bounded_tdt_loss(joint,encoder_projected,decoder_projected,labels,vocab_size,blank_id,durations,
                     logit_lengths=None,target_lengths=None,chunk_size=16,sigma=0.,reduction='mean',recompute=True,
                     *,omega=0.,max_joint_bytes=1_700_000_000,tile_joint_bytes=16_000_000,guard=lambda:None):
    from numba import cuda
    from nemo.collections.asr.parts.numba.rnnt_loss.utils import rnnt_helper,global_constants
    from nemo.collections.asr.parts.numba.rnnt_loss.utils.cuda_utils import gpu_rnnt,reduce
    from native_tdt_checkpointed_v4 import _native_loss
    from tdt_tile_native_gradient_v1 import tile_tdt_grad_kernel,GPU_RNNT_THREAD_SIZE
    require(GPU_RNNT_THREAD_SIZE==256,'Pinned official gradient launch width')
    # The constructor validates the pinned official source functions as before.
    _native_loss(8192,(0,1,2,3,4),0.)
    require(getattr(joint,'_ASR18_fp32_joint_policy',{}).get('kind')=='ASR18_joint_canonical_T1_U128_FP32_v2','Explicit FP32 joint policy required for FP32 tile accumulation')
    ep=encoder_projected;dp=decoder_projected
    require(ep.device.type=='cuda' and ep.shape[0]==dp.shape[0]==labels.shape[0]==1,'Serial one-clip CUDA scope')
    require(vocab_size==8711 and blank_id==8192 and tuple(durations)==(0,1,2,3,4),'Original full vocabulary/blank/duration identity')
    require(sigma==omega==0. and max_joint_bytes==1_700_000_000 and tile_joint_bytes<=16_000_000 and recompute,'Original scientific settings and bounds')
    t=int(logit_lengths[0]) if logit_lengths is not None else ep.shape[1];u=int(target_lengths[0]) if target_lengths is not None else labels.shape[1]
    require(1<=t<=ep.shape[1] and 1<=u<=labels.shape[1] and u+1<=dp.shape[1] and u+1<=1024,'Full native T/U block bounds')
    from tdt_full_native_allocation_budget_v1 import allocation_budget
    allocation_budget(t,u,projected_elements=ep.numel()+dp.numel(),head_elements=sum(p.numel() for p in joint.parameters()),
        hidden_width=ep.shape[-1],max_joint_bytes=max_joint_bytes,tile_joint_bytes=tile_joint_bytes)
    target=labels[:,:u].contiguous();require(bool(((target>=0)&(target<8711)&(target!=8192)).all()),'Full original target IDs')
    by_u=torch.cat((target[0],target.new_tensor([0])))
    uc=128;require(uc*8716*4<=tile_joint_bytes,'Actual fullvocabulary predictor tile exceeds16MB')
    tiles=[(a,a+1,b,min(b+uc,u+1)) for a in range(t) for b in range(0,u+1,uc)]
    autocast=torch.is_autocast_enabled('cuda');dtype=torch.get_autocast_dtype('cuda');parameters=tuple(p for p in joint.parameters() if p.requires_grad)
    require(len(parameters)==2,'Deterministic original linear joint head only')
    cuda.select_device(ep.device.index);stream=cuda.external_stream(torch.cuda.current_stream(ep.device).cuda_stream)

    def tile(e,d):
        packed=joint(e[:,:,None],d[:,None]);require(packed.shape[-1]==8716,'Full joint width')
        return packed[...,:8711].float().contiguous(),packed[...,8711:].float().contiguous()

    class FullNativeTiled(torch.autograd.Function):
        @staticmethod
        def forward(ctx,e,d,y,*head):
            cells=t*(u+1);acts=torch.empty((1,t,u+1,2),device=e.device,dtype=torch.float32)
            raw_duration=torch.empty((1,t,u+1,5),device=e.device,dtype=torch.float32);denom=torch.empty(cells,device=e.device,dtype=torch.float32)
            for a,z,b,c in tiles:
                guard()
                with torch.autocast('cuda',dtype=dtype,enabled=autocast):lex,raw=tile(e[:,a:z],d[:,b:c])
                local=(z-a)*(c-b);norm=torch.zeros(local,device=e.device,dtype=torch.float32)
                reduce.reduce_max(lex.reshape(-1),norm,rows=8711,cols=local,minus=False,stream=stream)
                reduce.reduce_exp(lex.reshape(-1),norm,rows=8711,cols=local,minus=True,stream=stream)
                denom.reshape(t,u+1)[a:z,b:c].copy_(norm.reshape(z-a,c-b))
                acts[:,a:z,b:c,0].copy_(lex.gather(-1,by_u[None,None,b:c,None].expand(1,z-a,c-b,1)).squeeze(-1))
                acts[:,a:z,b:c,1].copy_(lex[...,8192]);raw_duration[:,a:z,b:c].copy_(raw);del lex,raw,norm
            duration=torch.log_softmax(raw_duration,dim=-1).contiguous()
            size,status=rnnt_helper.get_workspace_size(t,u+1,1,gpu=True);require(status==global_constants.RNNTStatus.RNNT_STATUS_SUCCESS,'Official workspace calculation')
            workspace=torch.zeros(size,device=e.device,dtype=torch.float32);require(size>=3*cells+2,'Pinned workspace layout')
            duration_values=torch.tensor(durations,device=e.device,dtype=torch.long);costs=torch.zeros(1,device=e.device,dtype=torch.float32)
            # Only this wrapper instance supplies a precomputed denominator;
            # installed classes, kernels and global module attributes stay intact.
            from tdt_exact_denominator_instance_v1 import exact_denominator_type,bind,release
            wrapper=exact_denominator_type(gpu_rnnt.GPUTDT)(minibatch=1,maxT=t,maxU=u+1,alphabet_size=2,workspace=workspace,
                tdt_workspace=duration_values,num_durations=5,blank=1,fastemit_lambda=0.,clamp=0.,num_threads=1,stream=stream,sigma=0.,omega=0.)
            bind(wrapper,workspace,denom,cells)
            lengths_t=torch.tensor([t],device=e.device,dtype=torch.long);lengths_u=torch.tensor([u],device=e.device,dtype=torch.long)
            mapped=torch.zeros_like(y);cg=torch.zeros_like(acts);hg=torch.zeros_like(duration)
            before=random.getstate()
            try:
                status=wrapper.cost_and_grad(label_acts=acts.reshape(-1).data,duration_acts=duration.reshape(-1).data,
                    label_grads=cg.reshape(-1).data,duration_grads=hg.reshape(-1).data,costs=costs.data,
                    pad_labels=mapped.data,label_lengths=lengths_u.data,input_lengths=lengths_t.data)
            finally:
                random.setstate(before);release(wrapper);del wrapper
            require(status==global_constants.RNNTStatus.RNNT_STATUS_SUCCESS and bool(torch.isfinite(costs).all()),'Official full-DP cost/gradient call')
            ctx.save_for_backward(e,d,y,*head,workspace,duration_values,duration,raw_duration,hg,lengths_t,lengths_u)
            ctx.head_count=len(head);return costs

        @staticmethod
        def backward(ctx,scale):
            saved=ctx.saved_tensors;e,d,y=saved[:3];head=saved[3:3+ctx.head_count]
            workspace,duration_values,duration,raw_duration,hg,lengths_t,lengths_u=saved[3+ctx.head_count:]
            cells=t*(u+1);norm=workspace[:cells];alpha=workspace[cells:2*cells];beta=workspace[2*cells:3*cells];ll=workspace[3*cells:3*cells+1]
            # Match the original global FP32 duration log-softmax backward and
            # its scale-before-softmax ordering before entering any joint tile.
            with torch.enable_grad():
                raw=raw_duration.detach().requires_grad_();normalized=torch.log_softmax(raw,dim=-1)
                duration_logits_grad=torch.autograd.grad(normalized,raw,hg*scale.reshape(1,1,1,1))[0]
            del raw,normalized
            eg=torch.zeros(e.shape,device=e.device,dtype=torch.float32);dg=torch.zeros(d.shape,device=d.device,dtype=torch.float32);head_grad=[torch.zeros_like(p,dtype=torch.float32) for p in head]
            kernel_hg=torch.empty_like(duration)
            for a,z,b,c in tiles:
                guard()
                with torch.enable_grad():
                    et=e[:,a:z].detach().float().requires_grad_();dt=d[:,b:c].detach().float().requires_grad_()
                    with torch.autocast('cuda',dtype=dtype,enabled=autocast):lex,raw=tile(et,dt)
                    lexical_grad=torch.empty_like(lex)
                    tile_tdt_grad_kernel[(z-a)*(c-b),GPU_RNNT_THREAD_SIZE,stream,0](lexical_grad.reshape(-1),kernel_hg.reshape(-1),lex.detach().reshape(-1),duration.reshape(-1),
                        norm,0.,alpha,beta,ll,lengths_t,lengths_u,y,1,t,u+1,8711,a,b,c-b,8192,duration_values,5,0.,0.)
                    grads=torch.autograd.grad((lex,raw),(et,dt,*head),grad_outputs=(lexical_grad*scale.reshape(1,1,1,1),duration_logits_grad[:,a:z,b:c]))
                eg[:,a:z].add_(grads[0].float());dg[:,b:c].add_(grads[1].float())
                for total,part in zip(head_grad,grads[2:]):total.add_(part.float())
                del et,dt,lex,raw,lexical_grad,grads
            return (eg.to(e.dtype),dg.to(d.dtype),None,*[g.to(p.dtype) for g,p in zip(head_grad,head)])

    value=FullNativeTiled.apply(ep[:,:t],dp[:,:u+1],target,*parameters)
    if reduction=='none':return value
    require(reduction in ('mean','sum','mean_batch','mean_volume'),'Declared original reduction')
    return value.sum() if reduction in ('sum','mean_batch') else value.sum()/u
