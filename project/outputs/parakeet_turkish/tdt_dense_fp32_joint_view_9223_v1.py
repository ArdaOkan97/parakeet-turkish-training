"""Explicit NEW full-clip FP32 GEMM policy; no canonical-T1 byte claim.

The view shares the exact original two active head Parameters. The student's
installed canonical joint and the teacher/preservation paths are unchanged.
"""
import torch
from torch.nn import functional as F

POLICY={'kind':'ASR18_full_clip_dense_FP32_joint_v1','addition':'float32','activation':'float32_relu',
    'linear_and_backward':'one_full_clip_T_times_U_FP32_GEMM_original_PyTorch_autograd',
    'full_vocabulary':9223,'durations':5,'allow_tf32':False,'canonical_T1_trajectory_equivalence_claimed':False,
    'BF16_original_joint_trajectory_equivalence_claimed':False,'head_Parameters_shared_not_copied':True}


class DenseFP32JointView:
    def __init__(self,original):
        from tdt_joint_canonical_2d_fp32_9223_v1 import POLICY as canonical_policy
        if getattr(original,'_ASR18_fp32_joint_policy',None)!=canonical_policy:
            raise ValueError('Exact unchanged canonical original joint remains installed')
        self.head=original.head;self.lock=original._ASR18_joint_lock
        parameters=tuple(self.head.parameters())
        if len(parameters)!=2 or any(p.dtype!=torch.float32 for p in parameters) or self.head.out_features!=9228:
            raise ValueError('Same two full9223+5 FP32 head Parameters')
        self._ASR18_fp32_joint_policy=dict(POLICY)

    def parameters(self):return self.head.parameters()

    def __call__(self,e,d):
        if not self.lock.acquire(blocking=False):raise ValueError('No overlapping original/dense joint forward')
        try:
            if not(e.ndim==d.ndim==4 and e.shape[0]==d.shape[0]==1 and e.shape[2]==d.shape[1]==1
                and e.shape[-1]==d.shape[-1]==self.head.in_features and not torch.backends.cuda.matmul.allow_tf32):
                raise ValueError('One complete unpadded clip/full predictor axis, no TF32')
            with torch.autocast('cuda',enabled=False):
                return F.linear(F.relu(e.float()+d.float()),self.head.weight,self.head.bias)
        finally:self.lock.release()
