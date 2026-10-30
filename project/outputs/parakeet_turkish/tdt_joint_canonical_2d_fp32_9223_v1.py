"""NEW numerical recipe: canonical T1/U128 FP32 joint and reduction order.

The dense reference and bounded path use identical matrix shapes per block,
accumulate FP32 in increasing T then U-block order, and cast complete projected
gradients once. This is a new recipe, without v1/BF16 trajectory equivalence.
"""
import threading
from types import MethodType
import torch
from torch.nn import functional as F

U_BLOCK = 128
POLICY = {'kind': 'ASR18_joint_canonical_T1_U128_FP32_v2', 'addition': 'float32',
    'activation': 'float32_relu', 'linear_and_backward': 'float32_T1_U128_blocks',
    'projected_gradient_accumulation': 'FP32_full_gradient_before_one_final_cast',
    'parameter_gradient_accumulation': 'FP32_in_increasing_T_then_U128_block_order',
    'predictor_block_rows': U_BLOCK, 'encoder_block_frames': 1, 'allow_tf32': False,
    'full_vocabulary': 9223, 'durations': 5, 'encoder_decoder_autocast_unchanged': True,
    'v1_or_original_BF16_trajectory_equivalence_claimed': False,
    'maximum_packed_joint_block_bytes': U_BLOCK*9228*4,
    'extra_kernel_launches_per_T_U_block': True}

class Canonical2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, e, d, w, b):
        if not (e.ndim == d.ndim == 4 and e.shape[0] == d.shape[0] == 1
                and e.shape[2] == d.shape[1] == 1 and e.shape[3] == d.shape[3]):
            raise ValueError('Serial (1,T,1,H)+(1,1,U,H) joint scope required')
        result = torch.empty((1, e.shape[1], d.shape[2], w.shape[0]), device=e.device, dtype=torch.float32)
        with torch.autocast('cuda', enabled=False):
            for t in range(e.shape[1]):
                for u in range(0, d.shape[2], U_BLOCK):
                    end = min(u+U_BLOCK, d.shape[2])
                    hidden = F.relu(e[:,t:t+1].float()+d[:,:,u:end].float())
                    result[:,t:t+1,u:end] = F.linear(hidden, w, b)
        ctx.save_for_backward(e, d, w)
        return result

    @staticmethod
    def backward(ctx, gradient):
        e, d, w = ctx.saved_tensors
        eg = torch.zeros_like(e, dtype=torch.float32); dg = torch.zeros_like(d, dtype=torch.float32)
        wg = torch.zeros_like(w); bg = torch.zeros(w.shape[0], device=w.device, dtype=torch.float32)
        with torch.autocast('cuda', enabled=False):
            for t in range(e.shape[1]):
                for u in range(0, d.shape[2], U_BLOCK):
                    end = min(u+U_BLOCK, d.shape[2])
                    raw = e[:,t:t+1].float()+d[:,:,u:end].float(); hidden = F.relu(raw)
                    g = gradient[:,t:t+1,u:end].reshape(-1, w.shape[0]).float()
                    gh = (g@w).reshape_as(raw).masked_fill(raw <= 0, 0.)
                    eg[:,t:t+1].add_(gh.sum(dim=2, keepdim=True))
                    dg[:,:,u:end].add_(gh)
                    wg.add_(g.T@hidden.reshape(-1, w.shape[1])); bg.add_(g.sum(dim=0))
        return eg.to(e.dtype), dg.to(d.dtype), wg, bg

def forward(self, encoder_hidden_states, decoder_hidden_states):
    if not self._ASR18_joint_lock.acquire(blocking=False):
        raise ValueError('Concurrent joint forward is not admitted')
    try:
        e, d = encoder_hidden_states, decoder_hidden_states
        if e.ndim == d.ndim == 4 and e.shape[2] == d.shape[1] == 1:
            return Canonical2D.apply(e, d, self.head.weight, self.head.bias)
        if d.ndim == e.ndim == 4 and d.shape[2] == e.shape[1] == 1:
            return Canonical2D.apply(d, e, self.head.weight, self.head.bias)
        with torch.autocast('cuda', enabled=False):
            return F.linear(F.relu(e.float()+d.float()), self.head.weight, self.head.bias)
    finally:
        self._ASR18_joint_lock.release()

def apply(joint):
    if hasattr(joint, '_ASR18_fp32_joint_policy'):
        raise ValueError('Joint numerical policy already applied')
    if not isinstance(joint.head, torch.nn.Linear) or joint.head.out_features != 9228:
        raise ValueError('Pinned full9223+5 head required')
    if len(tuple(joint.parameters())) != 2 or any(p.dtype != torch.float32 for p in joint.parameters()):
        raise ValueError('Original two active FP32 head parameters required')
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError('Canonical FP32 requires TF32 disabled')
    joint._ASR18_joint_lock = threading.Lock()
    joint.forward = MethodType(forward, joint)
    joint._ASR18_fp32_joint_policy = dict(POLICY)
    return dict(POLICY)
