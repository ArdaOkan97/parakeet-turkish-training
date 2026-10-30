"""Diagnostic BF16 storage for frozen autocast Linear/Conv weights only.

All trainable207 weights/moments and normalization parameters remain FP32.
This policy requires separate numerical/DEV proof and is never implicit.
"""
import torch

def apply(model):
    converted=[];before=after=0
    for name,module in model.encoder.named_modules():
        if not isinstance(module,(torch.nn.Linear,torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.Conv3d)):continue
        parameter=module.weight
        if parameter.requires_grad:raise ValueError('Refuse precision change to active weight: '+name)
        if parameter.dtype!=torch.float32:raise ValueError('Frozen source must start FP32: '+name)
        before+=parameter.numel()*parameter.element_size()
        parameter.data=parameter.data.to(dtype=torch.bfloat16)
        after+=parameter.numel()*parameter.element_size()
        converted.append('encoder.'+name+'.weight')
    if not converted:raise ValueError('No frozen encoder matrix converted')
    if any(p.requires_grad and p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('Active207 must remain FP32')
    for module in model.encoder.modules():
        if isinstance(module,(torch.nn.LayerNorm,torch.nn.modules.batchnorm._BatchNorm)):
            if any(p.dtype!=torch.float32 for p in module.parameters(recurse=False)):raise ValueError('FP32 normalization required')
    torch.cuda.empty_cache()
    return {'kind':'frozen_encoder_Linear_Conv_BF16_storage_v1','converted_names':converted,
        'converted_weight_tensors':len(converted),'before_weight_storage_bytes':before,'after_weight_storage_bytes':after,
        'storage_bytes_saved':before-after,'active207_weights_and_Adam_moments_FP32':True,
        'normalization_weights_FP32':True,'biases_unchanged':True,'compute_policy':'existing_CUDA_BF16_autocast',
        'exact_native_update_equivalence_claimed':False,'numerical_and_full_DEV_equivalence_required':True}
