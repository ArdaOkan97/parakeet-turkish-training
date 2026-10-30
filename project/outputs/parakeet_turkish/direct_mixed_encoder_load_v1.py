"""Stream frozen autocast matrices directly as BF16, all other tensors FP32.

Uses the installed Transformers dtype-plan hook inspected in local source.
No complete CPU model and no FP32 GPU model followed by fragmented conversion.
The actual dtype of every loaded parameter/buffer is verified after loading.
"""
import gc
import re
import torch
from transformers import AutoConfig,AutoModelForTDT

def load(base):
    config=AutoConfig.from_pretrained(base,local_files_only=True)
    with torch.device('meta'):
        skeleton=AutoModelForTDT.from_config(config)
    if not all(p.device.type=='meta' for p in skeleton.parameters()):raise ValueError('Meta-only planning model required')
    targets={'encoder.'+name+'.weight' for name,module in skeleton.encoder.named_modules()
             if isinstance(module,(torch.nn.Linear,torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.Conv3d))}
    entries={**dict(skeleton.named_parameters()),**dict(skeleton.named_buffers())}
    plan={name:(torch.bfloat16 if name in targets else torch.float32) for name,tensor in entries.items() if tensor.is_floating_point()}
    saved_bytes=sum(entries[name].numel()*2 for name in targets)
    cls=type(skeleton);original=cls._get_dtype_plan
    def exact_dtype_plan(self,dtype):
        if dtype!=torch.bfloat16:raise ValueError('Explicit mixed BF16 loader dtype required')
        return {'^'+re.escape(name)+'$':value for name,value in plan.items()}
    del skeleton,entries;gc.collect()
    cls._get_dtype_plan=exact_dtype_plan
    try:
        model=cls.from_pretrained(base,dtype=torch.bfloat16,local_files_only=True,
                                  low_cpu_mem_usage=True,device_map={'':'cuda:0'})
    finally:cls._get_dtype_plan=original
    actual={**dict(model.named_parameters()),**dict(model.named_buffers())}
    for name,expected in plan.items():
        tensor=actual[name]
        if tensor.dtype!=expected or tensor.device.type!='cuda':raise ValueError('Direct mixed dtype/device mismatch: '+name)
    for name,p in model.named_parameters():
        if name in targets:p.requires_grad_(False)
    return model,{'kind':'direct_mixed_frozen_encoder_storage_v1','frozen_BF16_matrix_weights':sorted(targets),
        'frozen_matrix_count':len(targets),'frozen_matrix_storage_bytes_saved':saved_bytes,
        'all_other_floating_parameters_and_buffers_FP32':True,'active207_FP32_required_at_optimizer_bootstrap':True,
        'meta_only_planning_model':True,'complete_CPU_model_created':False,'complete_FP32_GPU_model_created':False,
        'existing_BF16_autocast_compute_policy':True,'source_weights_on_disk_modified':False}
