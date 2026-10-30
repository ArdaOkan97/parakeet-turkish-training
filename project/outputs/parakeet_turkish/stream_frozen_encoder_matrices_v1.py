"""Mapped frozen BF16 weights, bounded per-module GPU materialization.

Only frozen matrix storage moves. Active LoRA/head weights and all Adam state
stay FP32 on GPU. Existing nonreentrant encoder block checkpoints recompute
frozen matrices on demand; actual gradient/peak validation is mandatory.
"""
import gc
import json
import math
import mmap
from pathlib import Path
import struct
import zipfile
import torch
from transformers import AutoConfig,AutoModelForTDT,GenerationConfig
from asr18_diagnostic_bootstrap_v1 import AREA,record,require


def readonly_shard(path,shape):
    # torch.load(mmap=True) uses copy-on-write mappings, which reserve commit
    # on Windows. An explicitly read-only mapping avoids that reservation.
    with zipfile.ZipFile(path) as archive:
        entries=[info for info in archive.infolist() if info.filename.endswith('/data/0')]
        require(len(entries)==1,'Single contiguous BF16 storage shard required')
        entry=entries[0]
        require(entry.compress_type==zipfile.ZIP_STORED and entry.file_size==math.prod(shape)*2,'Raw BF16 shard length differs')
        byteorder=[name for name in archive.namelist() if name.endswith('/byteorder')]
        require(len(byteorder)==1 and archive.read(byteorder[0])==b'little','Pinned little-endian shard required')
    with path.open('rb') as stream:
        stream.seek(entry.header_offset);header=struct.unpack('<IHHHHHIIIHH',stream.read(30))
        require(header[0]==0x04034b50,'ZIP local storage header required')
        offset=entry.header_offset+30+header[-2]+header[-1]
        mapping=mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ)
    value=torch.frombuffer(mapping,dtype=torch.bfloat16,count=math.prod(shape),offset=offset).reshape(shape)
    return value,mapping


def load(base):
    bank_path=AREA/'frozen_weight_bank_v1/report.json'
    bank=json.loads(bank_path.read_text())
    require(bank['passed'] and not bank['GPU_used'] and bank['matrix_count']==294,'Complete guarded CPU weight bank required')
    require(record(Path(base)/'model.safetensors')==bank['source'],'Exact original base weight archive required')
    config=AutoConfig.from_pretrained(base,local_files_only=True)
    with torch.device('meta'):model=AutoModelForTDT.from_config(config,dtype=torch.bfloat16)
    model.generation_config=GenerationConfig.from_pretrained(base,local_files_only=True)
    meta=model.state_dict();state={};backing={}
    require(all(parameter.device.type=='meta' for parameter in model.parameters()),'No complete CPU/GPU planning model')
    mappings=[]
    with (Path(base)/'model.safetensors').open('rb') as archive:
        header_length=struct.unpack('<Q',archive.read(8))[0]
        header=json.loads(archive.read(header_length));data_offset=8+header_length
        require(set(header)-{'__metadata__'}==set(meta),'All original model state keys required')
        base_mapping=mmap.mmap(archive.fileno(),0,access=mmap.ACCESS_READ)
        for name,reference in meta.items():
            if name in bank['matrices']:
                entry=bank['matrices'][name];path=Path(entry['tensor']['path'])
                require(record(path)==entry['tensor'],'Pinned mapped weight changed: '+name)
                value,mapping=readonly_shard(path,entry['shape']);mappings.append(mapping)
                require(value.dtype==torch.bfloat16 and list(value.shape)==entry['shape'],'Mapped weight shape/dtype differs')
                backing[name]=value;state[name]=value
            else:
                info=header[name];require(info['dtype'] in ('F32','I64'),'Expected original F32/I64 storage')
                dtype=torch.float32 if info['dtype']=='F32' else torch.int64
                value=torch.frombuffer(base_mapping,dtype=dtype,count=math.prod(info['shape']),
                    offset=data_offset+info['data_offsets'][0]).reshape(info['shape'])
                state[name]=value.to(device='cuda',dtype=torch.float32 if value.is_floating_point() else value.dtype)
            del value
        base_mapping.close()
    model.load_state_dict(state,strict=True,assign=True)
    generated=[]
    for name,buffer in model.named_buffers():
        if buffer.device.type!='meta':continue
        parent,key=name.rsplit('.',1);module=model.get_submodule(parent)
        require(key=='inv_freq' and type(module).__name__=='ParakeetEncoderRelPositionalEncoding',
                'Unknown generated buffer requires explicit initializer: '+name)
        # Same public deterministic initializer used by Parakeet._init_weights
        # for its nonpersistent relative-position buffer; no parameters reset.
        with torch.device('cpu'):value=module.compute_default_relative_positional_parameters(module.config)
        require(value.shape==buffer.shape and value.dtype==torch.float32,'Relative position initializer differs')
        module._buffers[key]=value.to('cuda');generated.append(name)
    del state,meta;gc.collect();torch.cuda.empty_cache()
    require(not any(tensor.device.type=='meta' for tensor in (*model.parameters(),*model.buffers())),'Unmaterialized generated model buffer')
    modules=dict(model.named_modules());handles=[]
    metrics={'forward_materializations':0,'transferred_weight_bytes':0,'logical_current_matrix_bytes':0,'logical_peak_matrix_bytes':0}
    for name,cpu in backing.items():
        module_name,parameter_name=name.rsplit('.',1);module=modules[module_name]
        require(parameter_name=='weight' and isinstance(module,(torch.nn.Linear,torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.Conv3d)),
                'Only frozen Linear/Conv matrices may be streamed')
        require(module.weight.device.type=='cpu' and not cpu.requires_grad,'Mapped CPU storage required')
        module.weight.requires_grad_(False)
        cpu_parameter=module.weight
        size=cpu.numel()*cpu.element_size()
        def before(current,inputs,cpu=cpu,size=size,cpu_parameter=cpu_parameter):
            require(not current.weight.requires_grad and current.weight.device.type=='cpu','Frozen CPU matrix required at entry')
            # Preserve the GPU tensor object saved by checkpoint/autograd.
            # Mutating its .data back to CPU corrupts recompute metadata.
            current._parameters['weight']=torch.nn.Parameter(cpu.to(device='cuda',non_blocking=False),requires_grad=False)
            metrics['forward_materializations']+=1;metrics['transferred_weight_bytes']+=size
            metrics['logical_current_matrix_bytes']+=size
            metrics['logical_peak_matrix_bytes']=max(metrics['logical_peak_matrix_bytes'],metrics['logical_current_matrix_bytes'])
        def after(current,inputs,output,cpu_parameter=cpu_parameter,size=size):
            current._parameters['weight']=cpu_parameter
            metrics['logical_current_matrix_bytes']-=size
        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after,always_call=True))
    # Keep the mapped storages and hook handles alive for this model only.
    model._diagnostic_frozen_backing=backing
    model._diagnostic_frozen_mappings=mappings
    model._diagnostic_frozen_handles=handles
    model._diagnostic_frozen_metrics=metrics
    policy={'kind':'mapped_BF16_frozen_encoder_streaming_v1','bank':record(bank_path),'matrix_count':294,
        'mapped_weight_bytes':bank['mapped_BF16_weight_bytes'],'no_complete_GPU_base_copy':True,
        'explicit_readonly_mappings_avoid_Windows_copy_on_write_commit':True,
        'generated_relative_position_buffers':generated,
        'original_generation_config':record(Path(base)/'generation_config.json'),
        'active207_FP32_GPU_required':True,'backward_block_checkpoint_validation_required':True,
        'logical_hook_peak_is_not_total_CUDA_peak':True,'source_weights_modified':False}
    return model,policy
