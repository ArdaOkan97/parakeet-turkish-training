"""INACTIVE conv-only temporal tiling with one ORIGINAL full-shape Linear.

This does not alter encoder-block/LoRA/teacher/Adam math, targets, audio, draw
order, effective accumulation or global guards. Different convolution shapes
can select different CUDA algorithms: numerical equality is UNPROVED and no old
trajectory equivalence is claimed. It cannot be wired into a caller by default.
"""
import hashlib
import inspect
from pathlib import Path
import threading

from bpe_broad_production_state_v1 import require

PROJECT = Path(__file__).resolve().parents[2]
ORIGINAL_SOURCE = PROJECT/'work/training-env/Lib/site-packages/transformers/models/parakeet/modeling_parakeet.py'
ORIGINAL_SHA256 = 'e1d75e98995f2d01fe6adcf51ac83b10f65366e1439c0859fc58475db62d6cdd'
CORE_OUTPUT_FRAMES = 32
MAX_PHYSICAL_INPUT_FRAMES = 15925


def tile_plan(physical_frames, valid_mask_frames):
    require(type(physical_frames) is int and type(valid_mask_frames) is int
            and 1 <= valid_mask_frames <= physical_frames <= MAX_PHYSICAL_INPUT_FRAMES,
            'Full positive physical frames and exact valid-mask count, never crop/drop')
    physical_output = (physical_frames+7)//8
    tiles=[]
    for low in range(0,physical_output,CORE_OUTPUT_FRAMES):
        high=min(low+CORE_OUTPUT_FRAMES,physical_output)
        # Each k3/s2/p1 operation grows receptive width:3,7,15. Output j
        # depends only on input8*j+[-7,+7]. Crop origin is a MULTIPLE of8;
        # one whole output-cell halo on either side covers radius7 exactly.
        start=max(0,8*low-8);stop=min(physical_frames,8*high+8)
        origin=start//8
        require(start%8==0 and 0 <= low-origin < high-origin <= (stop-start+7)//8,
                'Exact output grid/core slice within original cropped forward')
        for index in (low,high-1):
            require(start <= max(0,8*index-7)
                    and stop >= min(physical_frames,8*index+8),
                    'Complete original receptive interval including both utterance edges')
        tiles.append({'output_start':low,'output_stop':high,'input_start':start,'input_stop':stop,
            'local_output_start':low-origin,'local_output_stop':high-origin,
            'local_valid_mask_frames':max(0,min(stop,valid_mask_frames)-start)})
    require(tiles[0]['output_start']==0 and tiles[-1]['output_stop']==physical_output
            and all(a['output_stop']==b['output_start'] for a,b in zip(tiles,tiles[1:])),
            'Every full physical output row exactly once, including masked suffix')
    return {'physical_input_frames':physical_frames,'valid_mask_frames':valid_mask_frames,
        'physical_output_frames':physical_output,'valid_encoder_frames':(valid_mask_frames+7)//8,
        'effective_stride':8,'receptive_field':15,'receptive_radius':7,'aligned_input_halo':8,
        'core_output_frames':CORE_OUTPUT_FRAMES,'tiles':tiles}


def logical_allocation(physical_frames, valid_mask_frames, *, activation_bytes=2):
    """Tensor-storage union bound only; NOT host/private/commit/CUDA workspace."""
    require(type(activation_bytes) is int and activation_bytes in (2,4), 'BF16 or conservative FP32 activation width')
    plan=tile_plan(physical_frames,valid_mask_frames)
    def extents(frames):
        t1=(frames+1)//2;t2=(t1+1)//2;t3=(t2+1)//2
        return [256*t1*64*activation_bytes,256*t2*32*activation_bytes,256*t3*16*activation_bytes]
    whole=extents(physical_frames)
    local=extents(max(tile['input_stop']-tile['input_start'] for tile in plan['tiles']))
    final_output=plan['physical_output_frames']*1024*activation_bytes
    largest_streamed_weight=1024*4096*2 # existing BF16 frozen final linear, not a new full-model copy
    input_whole=physical_frames*128*4 # original complete FP32 input remains resident
    # Sum more than the simultaneously live Python forward tensors: full input,
    # all local conv/pointwise/ReLU outputs/copies, flattened final conv, linear
    # output, whole final output and largest temporary streamed matrix. No full
    # first-conv activation is created or concatenated by the proposed path.
    full_flattened=plan['physical_output_frames']*4096*activation_bytes
    union=input_whole+sum(3*n for n in local)+local[-1]+CORE_OUTPUT_FRAMES*1024*activation_bytes+2*final_output+largest_streamed_weight+full_flattened
    return {'scope':'logical_CUDA_tensor_storage_only_no_private_or_workspace_attribution',
        'activation_bytes':activation_bytes,'physical_input_frames':physical_frames,
        'original_first_conv_output_bytes':whole[0],
        'original_first_ReLU_input_plus_output_bytes':2*whole[0],
        'proposed_full_union_conservative_bytes':union,'proposed_first_conv_output_upper_bytes':local[0],
        'whole_input_and_final_output_preserved':True,'maximum_tile_input_frames':max(t['input_stop']-t['input_start'] for t in plan['tiles']),
        'tile_forwards':len(plan['tiles']),'CUDA_workspace_bytes_included':False,
        'full_flattened_BF16_feature_bytes':full_flattened,'full_shape_linear_calls':1,
        'host_private_reduction_proved_bytes':None,'new_native_whole_worker_forecast_derived':False}


def apply(model, *, torch, guard):
    """Future reviewed instance hook only; requires an actual no-grad frozen CNN."""
    require(threading.current_thread() is threading.main_thread() and not hasattr(model,'_orig_mod')
            and not torch.distributed.is_initialized(), 'Serial uncompiled single-GPU owner only')
    module=model.encoder.subsampling
    require(type(module).__name__=='ParakeetEncoderSubsamplingConv2D'
            and not hasattr(module,'_ASR18_temporal_tile_policy')
            and Path(inspect.getsourcefile(type(module))).resolve()==ORIGINAL_SOURCE.resolve()
            and hashlib.sha256(ORIGINAL_SOURCE.read_bytes()).hexdigest()==ORIGINAL_SHA256,
            'Exact pinned original class/source and no nested application')
    require((module.kernel_size,module.stride,module.channels,module.padding,module.num_layers)==(3,2,256,1,3)
            and len(module.layers)==8 and module.linear.in_features==4096 and module.linear.out_features==1024,
            'Exact original three-stage CNN/frequency/hidden geometry')
    expected=[(0,1,256,3,2,1,1),(2,256,256,3,2,1,256),(3,256,256,1,1,0,1),
              (5,256,256,3,2,1,256),(6,256,256,1,1,0,1)]
    for index,cin,cout,kernel,stride,padding,groups in expected:
        layer=module.layers[index]
        require(isinstance(layer,torch.nn.Conv2d) and layer.in_channels==cin and layer.out_channels==cout
                and layer.kernel_size==(kernel,kernel) and layer.stride==(stride,stride)
                and layer.padding==(padding,padding) and layer.groups==groups
                and layer.dilation==(1,1) and layer.padding_mode=='zeros', 'Exact local convolution operation')
    require(all(isinstance(module.layers[index],torch.nn.ReLU) and not module.layers[index].inplace for index in (1,4,7)),
            'Only local original out-of-place ReLU; no normalization/stochastic branch')
    require(all(not parameter.requires_grad for parameter in module.parameters()),
            'No trainable CNN/linear parameter; input gradient path must remain absent')
    def forward(input_features,attention_mask=None):
        guard()
        require(not model.encoder.training and not input_features.requires_grad
                and input_features.device.type=='cuda' and input_features.ndim==3
                and input_features.shape[0]==1 and input_features.shape[2]==128
                and attention_mask is not None and attention_mask.dtype==torch.bool
                and attention_mask.shape==input_features.shape[:2]
                and all(not p.requires_grad for p in module.parameters()),
                'Full original physical1 input and frozen/no-gradient subsampling only')
        frames=input_features.shape[1];valid=int(attention_mask.sum().item())
        require(torch.equal(attention_mask,torch.arange(frames,device=attention_mask.device)[None,:]<valid),
                'Exact original right-prefix mask; no gap or approximate geometry')
        plan=tile_plan(frames,valid);features=None
        # Exact original Conv/ReLU/mask equations on aligned local inputs.
        # Stitch the final flattened representation, then retain the original
        # FULL Linear M/K/N shape and invoke its existing streamed hook ONCE.
        with torch.no_grad():
            for tile in plan['tiles']:
                guard();start,stop=tile['input_start'],tile['input_stop']
                hidden_states=input_features[:,start:stop].unsqueeze(1)
                current_lengths=attention_mask[:,start:stop].sum(-1)
                for layer in module.layers:
                    hidden_states=layer(hidden_states)
                    if isinstance(layer,torch.nn.Conv2d):
                        current_lengths=module._get_output_length(current_lengths,layer)
                        current_seq_length=hidden_states.shape[2]
                        channel_mask=(torch.arange(current_seq_length,device=attention_mask.device)<current_lengths[:,None])
                        hidden_states*=channel_mask[:,None,:,None]
                # RHS shape[2] is PRE-transpose TIME, exactly as original.
                part=hidden_states.transpose(1,2).reshape(hidden_states.shape[0],hidden_states.shape[2],-1)
                require(part.shape==(1,(stop-start+7)//8,4096) and not part.requires_grad,
                        'Original final Conv flatten geometry/no-grad')
                if features is None:
                    features=torch.empty((1,plan['physical_output_frames'],4096),device=part.device,dtype=part.dtype)
                require(features.dtype==part.dtype,'No Conv tile dtype change')
                features[:,tile['output_start']:tile['output_stop']].copy_(
                    part[:,tile['local_output_start']:tile['local_output_stop']])
                del part,hidden_states
            require(features is not None and features.is_contiguous() and features.shape==(1,plan['physical_output_frames'],4096),
                    'Complete original full-M/K contiguous Linear input')
            output=module.linear(features)
            del features
        require(output.shape==(1,plan['physical_output_frames'],1024) and output.is_contiguous() and not output.requires_grad,
                'Same contiguous whole physical output required')
        return output
    module.forward=forward
    policy={'kind':'ASR18_proposed_frozen_CNN_conv_only_temporal_tiled_lifetime_v2',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'original_source':str(ORIGINAL_SOURCE),'original_SHA256':ORIGINAL_SHA256,
        'core_output_frames':CORE_OUTPUT_FRAMES,'max_physical_input_frames':MAX_PHYSICAL_INPUT_FRAMES,
        'original_no_grad_CNN_operations_reused':True,'full_mask_and_physical_output_preserved':True,
        'full_original_Linear_shape_preserved':True,'full_shape_Linear_calls':1,
        'numerical_equality_proved':False,'old_training_trajectory_byte_equivalence_claimed':False,
        'host_private_memory_reduction_proved':False,'production_enabled':False}
    module._ASR18_temporal_tile_policy=policy
    return policy
