"""NEW T<=32/full-U FP32 GEMM recipe; no canonical/full-dense byte claim."""
from tdt_dense_fp32_joint_view_v1 import DenseFP32JointView

POLICY={'kind':'ASR18_full_U_larger_T_FP32_joint_v1','addition':'float32','activation':'float32_relu',
    'linear_and_backward':'source_budgeted_T_le32_full_U_FP32_GEMMs_original_PyTorch_local_autograd',
    'full_vocabulary':8711,'durations':5,'allow_tf32':False,'maximum_packed_work_unit_bytes':128_000_000,
    'maximum_complete_joint_union_bytes':1_700_000_000,'canonical_T1_trajectory_equivalence_claimed':False,
    'full_clip_dense_trajectory_equivalence_claimed':False,'BF16_original_joint_trajectory_equivalence_claimed':False,
    'head_Parameters_shared_not_copied':True}


class BlockedFP32JointView(DenseFP32JointView):
    def __init__(self,original):
        super().__init__(original)
        self._ASR18_fp32_joint_policy=dict(POLICY)

    def __call__(self,e,d):
        if not(1<=e.shape[1]<=32 and e.shape[1]*d.shape[2]*8716*4<=128_000_000):
            raise ValueError('Only exact pre-admitted larger-T/full-U work unit')
        return super().__call__(e,d)
