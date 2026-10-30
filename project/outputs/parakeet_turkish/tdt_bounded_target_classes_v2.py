"""UNADMITTED bounded full-vocabulary TDT candidate for oversized clips.

Full T/U dynamic program is unchanged. Each cell retains target, sink and blank
logits, with sink=logsumexp(all other lexical logits). Full vocabulary joint
tiles are recomputed in backward; no input cropping/target partition/DP chunking.
Numerical equality, real-model gradients and long-shape resource proof REQUIRED.
"""
from __future__ import annotations
import random
import torch
from tdt_target_class_compression_v1 import compress_lexical

def require(ok,message):
    if not ok:raise ValueError(message)

def bounded_tdt_loss(joint,encoder_projected,decoder_projected,labels,vocab_size,blank_id,durations,
                     logit_lengths=None,target_lengths=None,chunk_size=16,sigma=0.,reduction='mean',recompute=True,
                     *,omega=0.,max_joint_bytes=1_700_000_000,tile_joint_bytes=16_000_000,guard=lambda:None):
    from native_tdt_checkpointed_v4 import _native_loss
    ep=encoder_projected;dp=decoder_projected
    require(ep.device.type=='cuda' and ep.shape[0]==dp.shape[0]==labels.shape[0]==1,'Finite serial one-clip CUDA scope only')
    require(vocab_size==8711 and blank_id==8192 and tuple(durations)==(0,1,2,3,4),'Exact admitted TDT vocabulary/blank/duration policy')
    require(sigma==omega==0. and max_joint_bytes==1_700_000_000 and tile_joint_bytes<=16_000_000,'Unchanged scientific settings and conservative tile bound')
    require(recompute and reduction in ('none','sum','mean','mean_batch','mean_volume'),'Recompute and declared reduction required')
    t=int(logit_lengths[0]) if logit_lengths is not None else ep.shape[1]
    u=int(target_lengths[0]) if target_lengths is not None else labels.shape[1]
    require(1<=t<=ep.shape[1] and 1<=u<=labels.shape[1] and u+1<=dp.shape[1],'Full positive original T/U lengths')
    require(u+1<=1024,'Official native full-U block exceeds1024; retained unresolved row, no consumed draw')
    require(t*(u+1)*8*4<=max_joint_bytes,'Actual compact full-DP operands exceed unchanged1.7GB bound')
    target=labels[:,:u].contiguous();require(bool(((target>=0)&(target<vocab_size)&(target!=blank_id)).all()),'Full unchanged lexical labels')
    by_u=torch.cat((target[0],target.new_tensor([0])))
    parameters=tuple(p for p in joint.parameters() if p.requires_grad)
    require(len(parameters)==2,'Only deterministic original Parakeet linear joint head admitted')
    require(not any(isinstance(m,torch.nn.Dropout) and m.training and m.p for m in joint.modules()),'Stochastic joint recomputation not admitted')
    autocast=torch.is_autocast_enabled('cuda');dtype=torch.get_autocast_dtype('cuda')
    cells=tile_joint_bytes//((vocab_size+5)*4);uc=min(u+1,32);tc=max(1,min(t,cells//uc))
    require(tc*uc*(vocab_size+5)*4<=tile_joint_bytes,'Full-vocabulary tile bound differs')
    tiles=[(a,min(t,a+tc),b,min(u+1,b+uc)) for a in range(0,t,tc) for b in range(0,u+1,uc)]
    backend=_native_loss(2,tuple(durations),0.)

    def joint_tile(e,d,target_ids):
        packed=joint(e[:,:,None],d[:,None])
        require(packed.shape[-1]==vocab_size+5,'Joint lexical/duration width changed')
        lexical=packed[...,:vocab_size].float()
        duration=torch.nn.functional.log_softmax(packed[...,vocab_size:].float(),dim=-1)
        compressed=compress_lexical(lexical,target_ids,blank_id,torch_module=torch)
        return compressed,duration

    class BoundedJoint(torch.autograd.Function):
        @staticmethod
        def forward(ctx,e,d,y,*head):
            compact=torch.empty((1,t,u+1,3),device=e.device,dtype=torch.float32)
            duration=torch.empty((1,t,u+1,5),device=e.device,dtype=torch.float32)
            for a,z,b,q in tiles:
                guard()
                with torch.autocast('cuda',dtype=dtype,enabled=autocast):c,h=joint_tile(e[:,a:z],d[:,b:q],by_u[b:q])
                compact[:,a:z,b:q].copy_(c);duration[:,a:z,b:q].copy_(h);del c,h
            # Execute unchanged official TDT DP on the sufficient statistic.
            # The unreachable sink remains in its denominator, not its target.
            with torch.enable_grad():
                compact.requires_grad_();duration.requires_grad_()
                mapped=torch.zeros_like(y);length_t=torch.tensor([t],device=e.device,dtype=torch.long);length_u=torch.tensor([u],device=e.device,dtype=torch.long)
                before=random.getstate()
                try:value=backend.loss(compact,duration,mapped,length_t,length_u,2,backend.durations,'none',0.,0.,0.,0.)
                finally:random.setstate(before)
                require(bool(torch.isfinite(value).all()),'Finite official compact-DP objective required')
                cg,hg=torch.autograd.grad(value.sum(),(compact,duration))
            ctx.save_for_backward(e,d,y,*head,cg,hg)
            ctx.head_count=len(head)
            return value.detach()

        @staticmethod
        def backward(ctx,scale):
            saved=ctx.saved_tensors;e,d,y=saved[:3];head=saved[3:3+ctx.head_count];cg,hg=saved[-2:]
            eg=torch.zeros(e.shape,device=e.device,dtype=torch.float32);dg=torch.zeros(d.shape,device=d.device,dtype=torch.float32)
            head_grad=[torch.zeros_like(p,dtype=torch.float32) for p in head]
            for a,z,b,q in tiles:
                guard()
                with torch.enable_grad():
                    et=e[:,a:z].detach().requires_grad_();dt=d[:,b:q].detach().requires_grad_()
                    with torch.autocast('cuda',dtype=dtype,enabled=autocast):
                        c,h=joint_tile(et,dt,by_u[b:q])
                    # Recompute under the original autocast context, then invoke
                    # autograd outside it, matching the dense native caller.
                    grads=torch.autograd.grad((c,h),(et,dt,*head),grad_outputs=(cg[:,a:z,b:q]*scale.reshape(1,1,1,1),hg[:,a:z,b:q]*scale.reshape(1,1,1,1)))
                eg[:,a:z].add_(grads[0].float());dg[:,b:q].add_(grads[1].float())
                for total,part in zip(head_grad,grads[2:]):total.add_(part.float())
                del et,dt,c,h,grads
            return (eg.to(e.dtype),dg.to(d.dtype),None,*[g.to(p.dtype) for g,p in zip(head_grad,head)])

    value=BoundedJoint.apply(ep[:,:t],dp[:,:u+1],target,*parameters)
    require(value.shape==(1,),'One full native DP result per uncut clip')
    return value if reduction=='none' else value.sum() if reduction in ('sum','mean_batch') else value.sum()/u
