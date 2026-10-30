"""NEW original-style BF16 ASR joint only; preservation remains canonicalFP32.

Original HF RNNT forward is activation(encoder+decoder), then SAME Linear;
TDT inherits it and only extends head width. Full-vocabulary normalization and
official DP remain FP32 in the separately source-bound loss adapter.
"""
import torch
from tdt_dense_fp32_joint_view_9223_v1 import DenseFP32JointView

POLICY={'kind': 'ASR18_full_U_larger_T_original_style_BF16_joint_v1', 'addition': 'bfloat16', 'activation': 'original_HF_ReLU_bfloat16', 'linear': 'original_FP32_head_Parameters_autocast_bfloat16', 'full_vocabulary': 9223, 'durations': 5, 'lexical_normalization_and_DP': 'float32', 'allow_tf32': False, 'head_Parameters_shared_FP32_not_copied': True, 'original_HF_source': {'path': 'C:\\Users\\ardao\\Documents\\Codex\\2026-10-10\\i-x20\\work\\training-env\\Lib\\site-packages\\transformers\\models\\parakeet\\modeling_parakeet.py', 'bytes': 48476, 'sha256': 'e1d75e98995f2d01fe6adcf51ac83b10f65366e1439c0859fc58475db62d6cdd'}, 'canonical_FP32_trajectory_equivalence_claimed': False, 'maximum_time_block': 32, 'maximum_complete_joint_union_bytes': 1700000000, 'conservative_FP32_packed_work_unit_bytes': 128000000, 'full_clip_BF16_GEMM_byte_equivalence_claimed': False}
ORIGINAL_HF_SOURCE=POLICY['original_HF_source']
REFERENCE_POLICY={key:value for key,value in POLICY.items() if key not in ('maximum_time_block',
    'maximum_complete_joint_union_bytes','conservative_FP32_packed_work_unit_bytes','full_clip_BF16_GEMM_byte_equivalence_claimed')}
REFERENCE_POLICY['kind']='ASR18_full_clip_original_style_BF16_joint_v1'


class DenseBF16JointView(DenseFP32JointView):
    def __init__(self,original):
        super().__init__(original)
        if not isinstance(getattr(original,'activation',None),torch.nn.ReLU) or original.activation.inplace:
            raise ValueError('Exact original HF config hidden_act=relu, no in-place activation')
        self.activation=original.activation
        del self._ASR18_fp32_joint_policy
        self._ASR18_loss_joint_policy=dict(REFERENCE_POLICY)

    def __call__(self,e,d):
        if not self.lock.acquire(blocking=False):raise ValueError('No overlapping original/BF16 loss joint')
        try:
            if not(e.ndim==d.ndim==4 and e.shape[0]==d.shape[0]==1 and e.shape[2]==d.shape[1]==1
                and e.shape[-1]==d.shape[-1]==self.head.in_features and not torch.backends.cuda.matmul.allow_tf32):
                raise ValueError('One exact unpadded full-U source work unit, no TF32')
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=True):
                # These casts retain the original BF16 projected operands when
                # the custom local adjoint presents detached FP32 leaves.
                return self.head(self.activation(e.to(torch.bfloat16)+d.to(torch.bfloat16)))
        finally:self.lock.release()


class BlockedBF16JointView(DenseBF16JointView):
    def __init__(self,original):
        super().__init__(original)
        self._ASR18_loss_joint_policy=dict(POLICY)

    def __call__(self,e,d):
        if not(1<=e.shape[1]<=32 and e.shape[1]*d.shape[2]*9228*4<=128_000_000):
            raise ValueError('Pre-admitted full-U/larger-T work unit, FULL FP32 allocation charge retained')
        return super().__call__(e,d)
