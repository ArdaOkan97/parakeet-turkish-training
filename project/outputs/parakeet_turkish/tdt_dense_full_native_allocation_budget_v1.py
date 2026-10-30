"""Full-clip FP32 allocation union, separate from allocator/host admission.

Eight COMPLETE packed matrices are reserved; the old128-row tile allowance is
never reused for a dense lattice. This is a logical bound, not measured fit.
"""


def allocation_budget(t,u,*,projected_elements,head_elements,hidden_width,
                      max_joint_bytes=8_000_000_000,tile_joint_bytes=1_000_000_000):
    if not(type(t) is int and type(u) is int and t>0 and 0<u<1024 and projected_elements>0
        and head_elements>0 and hidden_width>0 and max_joint_bytes==8_000_000_000
        and tile_joint_bytes==1_000_000_000):
        raise ValueError('Explicit positive full-clip geometry and new dense resource policy')
    cells=t*(u+1);packed=cells*8716*4
    parts={'all_full_DP_and_duration_tensor_allowance':256*cells+8,
        'eight_complete_packed_FP32_lattice_copies':8*packed,
        'full_lattice_hidden_activation_and_adjoint':16*cells*hidden_width,
        'complete_projected_gradient_and_cast_allowance':16*projected_elements,
        'head_gradient_and_reduction_copy_allowance':8*head_elements*4,
        'labels_lengths_duration_metadata_allowance':32*(u+1)+4096,
        'joint_runtime_workspace_reserve':256_000_000}
    total=sum(parts.values())
    if packed>tile_joint_bytes or total>max_joint_bytes:
        raise ValueError(f'Dense full-clip allocation not admitted: packed={packed},total={total}; use predeclared full-row canonical fallback, never crop/OOM retry')
    return {'kind':'ASR18_dense_full_clip_FP32_joint_allocation_budget_v1','T':t,'U':u,'cells':cells,
        'packed_FP32_lattice_bytes':packed,'parts':parts,'total_budget_bytes':total,
        'maximum_joint_bytes':max_joint_bytes,'maximum_packed_lattice_bytes':tile_joint_bytes,
        'CUDA_allocator_and_host_WDDM_commitment_admission_separate':True,'measured_fit_claimed':False}
