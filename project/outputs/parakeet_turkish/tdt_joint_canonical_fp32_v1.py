"""NEW deterministic per-frame FP32 joint arithmetic for serial TDT training.

Both dense reference and bounded path use the same per-frame matrix shapes,
FP32 input-gradient reduction order, and one final cast to projected dtype.
This spends extra kernel launches to remove shape-dependent BF16 rounding.
"""
import threading
from types import MethodType
import torch
from torch.nn import functional as F

POLICY={'kind':'ASR18_joint_canonical_frame_FP32_v1','addition':'float32','activation':'float32_relu',
        'linear_and_backward':'float32_per_encoder_frame','projected_gradient_accumulation':'FP32_before_one_final_cast',
        'parameter_gradient_accumulation':'FP32_in_increasing_encoder_frame_order','allow_tf32':False,
        'full_vocabulary':8711,'durations':5,'encoder_decoder_autocast_unchanged':True,
        'original_BF16_trajectory_equivalence_claimed':False,'extra_kernel_launches_per_frame':True}

class Canonical(torch.autograd.Function):
    @staticmethod
    def forward(ctx,e,d,w,b):
        if not(e.ndim==d.ndim==4 and e.shape[0]==d.shape[0]==1 and e.shape[2]==d.shape[1]==1 and e.shape[3]==d.shape[3]):
            raise ValueError('Canonical native joint requires serial (1,T,1,H)+(1,1,U,H)')
        output=torch.empty((1,e.shape[1],d.shape[2],w.shape[0]),device=e.device,dtype=torch.float32)
        with torch.autocast('cuda',enabled=False):
            for i in range(e.shape[1]):output[:,i:i+1]=F.linear(F.relu(e[:,i:i+1].float()+d.float()),w,b)
        ctx.save_for_backward(e,d,w);return output

    @staticmethod
    def backward(ctx,gradient):
        e,d,w=ctx.saved_tensors
        eg=torch.empty_like(e,dtype=torch.float32);dg=torch.zeros_like(d,dtype=torch.float32)
        wg=torch.zeros_like(w);bg=torch.zeros(w.shape[0],device=w.device,dtype=torch.float32)
        with torch.autocast('cuda',enabled=False):
            for i in range(e.shape[1]):
                raw=e[:,i:i+1].float()+d.float();hidden=F.relu(raw);g=gradient[:,i:i+1].reshape(-1,w.shape[0]).float()
                gh=(g@w).reshape_as(raw);gh=gh.masked_fill(raw<=0,0.)
                eg[:,i:i+1]=gh.sum(dim=2,keepdim=True);dg.add_(gh)
                wg.add_(g.T@hidden.reshape(-1,w.shape[1]));bg.add_(g.sum(dim=0))
        return eg.to(e.dtype),dg.to(d.dtype),wg,bg

def forward(self,encoder_hidden_states,decoder_hidden_states):
    if not self._ASR18_joint_lock.acquire(blocking=False):raise ValueError('Concurrent joint forward is not admitted')
    try:
        e,d=encoder_hidden_states,decoder_hidden_states
        # Inference may supply lower-rank/broadcast inputs; it has no tiled
        # training backward and retains the same FP32 mathematical forward.
        if e.ndim==d.ndim==4 and e.shape[2]==d.shape[1]==1:
            return Canonical.apply(e,d,self.head.weight,self.head.bias)
        if d.ndim==e.ndim==4 and d.shape[2]==e.shape[1]==1:
            return Canonical.apply(d,e,self.head.weight,self.head.bias)
        with torch.autocast('cuda',enabled=False):return F.linear(F.relu(e.float()+d.float()),self.head.weight,self.head.bias)
    finally:self._ASR18_joint_lock.release()

def apply(joint):
    if hasattr(joint,'_ASR18_fp32_joint_policy'):raise ValueError('Joint policy already applied')
    if not isinstance(joint.head,torch.nn.Linear) or joint.head.out_features!=8716:raise ValueError('Pinned full8711+5 head required')
    if len(tuple(joint.parameters()))!=2 or any(p.dtype!=torch.float32 for p in joint.parameters()):raise ValueError('Existing two FP32 head parameters required')
    if torch.backends.cuda.matmul.allow_tf32:raise ValueError('Canonical FP32 requires TF32 disabled')
    joint._ASR18_joint_lock=threading.Lock();joint.forward=MethodType(forward,joint);joint._ASR18_fp32_joint_policy=dict(POLICY)
    return dict(POLICY)
