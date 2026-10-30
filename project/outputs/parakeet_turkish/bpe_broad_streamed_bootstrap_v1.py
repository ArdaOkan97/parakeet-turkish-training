"""Opt-in production bootstrap: original BPE64250 and parked62250 only."""
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random

PROJECT=Path(__file__).resolve().parents[2]
STORAGE=Path(r'E:\gpu_training_scratchpad')
AREA=STORAGE/'analysis/ASR18_memory_diagnosis_v1'

def require(ok,message):
    if not ok: raise ValueError(message)

def record(path):
    path=Path(path)
    with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(path.resolve()),'sha256':digest,'bytes':path.stat().st_size}

def bootstrap(model,processor,*,torch,stage,admit):
    require(hasattr(model,'_diagnostic_frozen_backing'),'Explicit reviewed readonly streaming loader required')
    lease=json.loads((PROJECT/'work/gpu_training_owner.json').read_text())
    require(lease['owner']=='/root/memory_diagnosis' and lease['actual_GPU_running']
            and lease['actual_owned_worker_pid']==os.getpid(),'Actual delegated worker lease required')
    from encoder_lora import install_encoder_lora,encoder_lora_metadata
    from extend_vocab import extend_processor_and_model
    from train_lora import checkpoint_encoder_blocks
    from train_lora_batched_tsc_bpe_full_decoder_v1 import expand_candidate
    from bpe_broad_full207_optimizer_v1 import BPEBroadFull207Optimizer
    projection=json.loads((AREA/'source_projection_v4/report.json').read_text())
    preflight=json.loads((AREA/'optimizer_projected_v4/report.json').read_text())
    require(projection['passed'] and preflight['passed'],'Actual passed guarded CPU projection/optimizer preflight required')
    sources={key:value['parent'] for key,value in projection['projections'].items()}
    for key,item in projection['projections'].items():
        require(record(item['projected']['path'])==item['projected'],'Pinned compact source changed: '+key)
        require(sources[key]==preflight['pinned_sources'][key],'Original source provenance differs')
    extend_processor_and_model(processor,model)
    layout=install_encoder_lora(model,rank=8,alpha=16,last_n_layers=24)
    # A streamed frozen CPU base creates adapter tensors on CPU initially.
    # Materialize only active LoRA tensors on GPU before source restoration.
    if hasattr(model,'_diagnostic_frozen_backing'):
        for name,parameter in model.named_parameters():
            if name.endswith(('lora_A','lora_B')):parameter.data=parameter.data.to('cuda')
    model.config.encoder_lora_tracking={'layout':layout,'kind':'unmerged',
        'implementation_sha256':record(PROJECT/'outputs/parakeet_turkish/encoder_lora.py')['sha256']}
    admit('sequential_original_source_and_compaction',700_000_000)
    original=torch.load(projection['projections']['original']['projected']['path'],map_location='cpu',mmap=True,weights_only=True)
    require(original['step']==original['selected_step']==62250,'Original qualified source62250 required')
    model.load_state_dict(original['weights'],strict=False)
    expansion=expand_candidate(model,processor,original)
    # Constructor consumes only original LoRA weights/moments; copy those small
    # tensors so the entire original archive mapping/history can be released.
    names=tuple(original['weights']);lora={n for n in names if n.endswith(('lora_A','lora_B'))}
    compact={'step':62250,'selected_step':62250,
        'weights':{n:original['weights'][n].clone() if n in lora else None for n in names},
        'optimizer':{'param_groups':copy.deepcopy(original['optimizer']['param_groups']),
                     'state':{i:{k:v.clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v) for k,v in state.items()}
                              for i,state in original['optimizer']['state'].items() if i>=15}}}
    del original;gc.collect();stage('after_original_source_release_compact_parked192')
    admit('sequential_BPE_source_and_GPU_Adam_transfer',700_000_000)
    bpe=torch.load(projection['projections']['bpe']['projected']['path'],map_location='cpu',mmap=True,weights_only=True)
    result=model.load_state_dict(bpe['weights'],strict=False)
    require(not result.unexpected_keys and not any(n in bpe['weights'] for n in result.missing_keys),'All207 BPE source weights required')
    optimizer=BPEBroadFull207Optimizer(model,compact,bpe,source_records=sources,torch_module=torch)
    checkpoint_encoder_blocks(model)
    random.setstate(bpe['python_rng']);torch.set_rng_state(bpe['torch_rng']);torch.cuda.set_rng_state_all(bpe['cuda_rng'])
    provenance={'source_records':sources,'source_head_step':bpe['step'],'source_clocks':[64250,2000,62250],
                'source_RNG_restored':True,'source_optimizer_moments_reset':False,'source_history_retained_in_immutable_parents':True,
                'projection_report':record(AREA/'source_projection_v4/report.json'),
                'CPU_optimizer_preflight':record(AREA/'optimizer_projected_v4/report.json'),
                'optimizer_identity':optimizer.identity(),'LoRA':encoder_lora_metadata(model),'BPE_expansion':expansion}
    source_sampler=bpe['sampler']
    require(source_sampler['updates']==64250,'Original BPE sampler frontier64250 required')
    carried=source_sampler['continued_parent_sampler']
    replay_snapshot={'source_step':64250,'source_identity':sources['bpe'],
        **{key:copy.deepcopy(carried[key]) for key in ('replay','replay_language_counts','turkish_rng','replay_rng','Portuguese_rng')}}
    del compact,bpe,source_sampler,carried;gc.collect();torch.cuda.empty_cache()
    stage('after_sequential_sources_release_full207_training_ready')
    require(sum(p.requires_grad for p in model.parameters())==207,'All207 must be active')
    require(all(p.device.type=='cuda' and p.dtype==torch.float32 for p in model.parameters() if p.requires_grad),
            'All207 active tensors must remain FP32 GPU')
    model.encoder.eval()
    return optimizer,provenance,replay_snapshot
