"""NEW full-U/T<=32 FP32 work units, including all earlier compact contexts.

The SAME 1.7GB complete union remains; only the packed work-unit limit rises
from16MB to128MB. No free GPU or host-memory credit is inferred.
"""

MAX_JOINT_BYTES=1_700_000_000
MAX_PACKED_BYTES=128_000_000


def allocation_budget(t,u,*,projected_elements,head_elements,hidden_width,prior_compact_cells=0,
                      max_joint_bytes=MAX_JOINT_BYTES,tile_joint_bytes=MAX_PACKED_BYTES):
    if not(type(t) is int and type(u) is int and t>0 and 0<u<1024 and projected_elements>0
        and head_elements>0 and hidden_width>0 and type(prior_compact_cells) is int and prior_compact_cells>=0
        and max_joint_bytes==MAX_JOINT_BYTES and tile_joint_bytes==MAX_PACKED_BYTES):
        raise ValueError('Exact full-T/U, earlier compact contexts and complete allocation policy')
    cells=t*(u+1)
    for block in range(min(32,t),0,-1):
        packed=block*(u+1)*8716*4
        parts={'all_full_DP_and_prior_compact_duration_tensor_allowance':256*(cells+prior_compact_cells)+8,
            'eight_complete_packed_FP32_work_unit_copies':8*packed,
            'work_unit_hidden_activation_and_adjoint':16*block*(u+1)*hidden_width,
            'complete_projected_gradient_and_cast_allowance':16*projected_elements,
            'head_gradient_and_reduction_copy_allowance':8*head_elements*4,
            'labels_lengths_duration_metadata_allowance':32*(u+1)+4096,
            'joint_runtime_workspace_reserve':256_000_000}
        total=sum(parts.values())
        if packed<=tile_joint_bytes and total<=max_joint_bytes:
            return {'kind':'ASR18_full_U_larger_T_FP32_joint_allocation_budget_v1','T':t,'U':u,'cells':cells,
                'prior_compact_cells':prior_compact_cells,'time_block':block,'predictor_block':u+1,
                'packed_FP32_work_unit_bytes':packed,'parts':parts,'total_budget_bytes':total,
                'maximum_joint_bytes':max_joint_bytes,'maximum_packed_work_unit_bytes':tile_joint_bytes,
                'CUDA_allocator_and_host_WDDM_commitment_admission_separate':True,'measured_fit_claimed':False}
    raise ValueError('No larger-T/full-U work unit fits complete union; use predeclared full-row canonical fallback before allocation')
