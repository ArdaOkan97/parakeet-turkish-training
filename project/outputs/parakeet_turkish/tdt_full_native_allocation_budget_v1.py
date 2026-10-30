"""Conservative explicit joint/DP tensor budget, separate from CUDA allocator cap.

The old48*cells check was not a total simultaneous allocation bound. This
budget sums a padded full-DP allowance, concurrent full-vocabulary tile copies,
projected gradients/casts, head-gradient copies and a fixed runtime allowance.
Encoder/decoder graphs remain covered separately by fresh stage/global/CUDA
admission and measurements. A logical estimate never proves corpus shape fit.
"""

def allocation_budget(t,u,*,projected_elements,head_elements,hidden_width,max_joint_bytes=1_700_000_000,tile_joint_bytes=16_000_000):
    if not (type(t) is int and type(u) is int and t>0 and 0<u<1024 and projected_elements>0 and head_elements>0 and hidden_width>0
            and max_joint_bytes==1_700_000_000 and tile_joint_bytes<=16_000_000):
        raise ValueError('Declared serial full-target geometry and unchanged bounds required')
    cells=t*(u+1);tile_rows=min(128,u+1);packed=tile_rows*8716*4
    # Explicit forward full-cell allocations total92B/cell (acts8, raw20,
    # denom4, normalized20, labelgrad8, durationgrad20, officialDP12).
    # Backward additionally has full duration logsoftmax intermediates,
    # scaled gradients and kernel duration output.256B/cell includes the
    # union of these allocations rather than assuming perfect early release.
    parts={'all_full_DP_and_duration_tensor_allowance':256*cells+8,
        'concurrent_packed_lexical_gradient_tile_allowance':8*packed,
        'tile_hidden_activation_and_gradient_allowance':16*tile_rows*hidden_width,
        'complete_projected_gradient_and_cast_allowance':16*projected_elements,
        'head_gradient_and_reduction_copy_allowance':8*head_elements*4,
        'labels_lengths_duration_metadata_allowance':32*(u+1)+4096,
        'joint_runtime_workspace_reserve':256_000_000}
    total=sum(parts.values())
    if packed>tile_joint_bytes or total>max_joint_bytes:
        raise ValueError(f'Unresolved full-DP/joint allocation budget: {total}>{max_joint_bytes}; no draw exclusion/consumption')
    return {'kind':'ASR18_conservative_full_native_joint_allocation_budget_v1','T':t,'U':u,'cells':cells,
        'explicit_forward_bytes_per_cell':92,'padded_full_cell_bytes':256,'parts':parts,'total_budget_bytes':total,
        'maximum_joint_bytes':max_joint_bytes,'maximum_packed_tile_bytes':packed,'CUDA_allocator_cap_is_separate':True,
        'encoder_decoder_graph_fit_or_final_corpus_geometry_claimed':False}
