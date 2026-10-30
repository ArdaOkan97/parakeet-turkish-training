"""Actual pretrained subsampling/first encoder blocks, CPU only, no GPU."""
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUT))
import torch
from torch import nn
from torch.nn.attention import sdpa_kernel, SDPBackend
from safetensors import safe_open
from transformers import AutoConfig,AutoProcessor
from transformers.models.parakeet.modeling_parakeet import (ParakeetEncoderSubsamplingConv2D,
    ParakeetEncoderRelPositionalEncoding,ParakeetEncoderBlock)
from encoder_lora import install_encoder_lora
from profile_batched_training import examples
from runtime import read_audio

torch.set_num_threads(4)
base=Path((ROOT/'work/base_model_path.txt').read_text().strip())
config=AutoConfig.from_pretrained(base,local_files_only=True).encoder_config
config._attn_implementation='sdpa'
sub=ParakeetEncoderSubsamplingConv2D(config).eval()
positions=ParakeetEncoderRelPositionalEncoding(config).eval()
blocks=nn.ModuleList([ParakeetEncoderBlock(config) for _ in range(3)]).eval()
loaded=[]
with safe_open(base/'model.safetensors',framework='pt',device='cpu') as f:
    for prefix,module in [('encoder.subsampling.',sub),*[(f'encoder.layers.{i}.',b) for i,b in enumerate(blocks)]]:
        weights={name[len(prefix):]:f.get_tensor(name).float() for name in f.keys() if name.startswith(prefix)}
        status=module.load_state_dict(weights,strict=False)
        if status.unexpected_keys or any(not k.endswith('num_batches_tracked') for k in status.missing_keys):
            raise ValueError(str(status))
        loaded.append({'prefix':prefix,'tensor_count':len(weights)})
wrapper=SimpleNamespace(encoder=nn.Module())
wrapper.encoder.layers=blocks
install_encoder_lora(wrapper,rank=8,alpha=16)
incoming=torch.load(ROOT/'work/lora_20261010_01_best.pt',map_location='cpu',weights_only=True)
for i,block in enumerate(blocks):
    prefix=f'encoder.layers.{i}.'
    adapters={name[len(prefix):]:tensor for name,tensor in incoming['weights'].items() if name.startswith(prefix)}
    block.load_state_dict(adapters,strict=False)
    block.eval()
processor=AutoProcessor.from_pretrained(base,local_files_only=True)
rows,items=examples(SimpleNamespace(quartet='median'))
waves=[read_audio(row) for row in rows]
single=[dict(processor(audio=[w],sampling_rate=16000,return_tensors='pt')) for w in waves]
def run(inputs):
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        h=sub(inputs['input_features'],inputs['attention_mask'])*math.sqrt(config.hidden_size)
        lengths=inputs['attention_mask'].sum(-1)
        for _ in range(int(math.log2(config.subsampling_factor))): lengths=(lengths+1)//2
        valid=torch.arange(h.shape[1])[None,:]<lengths[:,None]
        mask=(valid[:,None,:]&valid[:,:,None])[:,None]
        p=positions(h)
        snapshots=[h.clone()]
        for block in blocks:
            h=block(h,attention_mask=mask,position_embeddings=p)
            snapshots.append(h.clone())
    return snapshots,lengths,p,mask
before=torch.get_rng_state().clone()
reference=[run(x) for x in single]
results=[]
for backend in ('sdpa','eager'):
    for block in blocks: block.self_attn.config._attn_implementation=backend
    refs=reference if backend=='sdpa' else [run(x) for x in single]
    for layout,groups in [('B2_homogeneous',[[0,2],[1,3]]),('B4_mixed',[[0,1,2,3]])]:
        for group in groups:
            inputs=dict(processor(audio=[waves[i] for i in group],sampling_rate=16000,return_tensors='pt',padding='longest'))
            candidate,lengths,p,mask=run(inputs)
            for j,i in enumerate(group):
                n=int(lengths[j]);r,rl,rp,rm=refs[i]
                records=[]
                for layer,(a,b) in enumerate(zip(r,candidate)):
                    x=a[0,:n].float();y=b[j,:n].float();delta=y-x
                    records.append({'stage':'subsampling' if layer==0 else f'encoder_block_{layer-1}',
                                    'relative_l2_difference':float(delta.norm()/x.norm()),
                                    'max_abs_difference':float(delta.abs().max())})
                # Relative shift must align the same integer q/key offsets at
                # different padded lengths. Compare fixed shared-position values.
                short_len=r[0].shape[1];long_len=candidate[0].shape[1]
                overlap=min(short_len,long_len)-1
                pos_equal=torch.equal(rp[0,short_len-1-overlap:short_len+overlap],
                                      p[j,long_len-1-overlap:long_len+overlap])
                results.append({'backend':backend,'layout':layout,'source_id':rows[i]['source_id'],
                                'valid_frames':n,'same_valid_length':n==int(rl[0]),
                                'relative_position_shared_offsets_exact':pos_equal,'stages':records})
result={'schema':1,'gpu_used':False,'cpu_threads':4,'torch_version':torch.__version__,
        'sdpa_backend':'explicit torch.nn.attention.SDPBackend.MATH',
        'scope':'Actual1024-hidden pretrained subsampling and first3 encoder blocks with learned rank8 pilot adapters, CPU FP32; remaining21 layers/GPU not tested',
        'loaded_base_prefixes':loaded,'incoming_weights_sha256':hashlib.sha256((ROOT/'work/lora_20261010_01_best.pt').read_bytes()).hexdigest(),
        'rng_unchanged':bool(torch.equal(before,torch.get_rng_state())), 'results':results,
        'maximum_relative_l2_difference':max(s['relative_l2_difference'] for r in results for s in r['stages']),
        'maximum_absolute_difference':max(s['max_abs_difference'] for r in results for s in r['stages'])}
(OUT/'batched_actual_encoder_prefix_cpu_diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:result[k] for k in ['rng_unchanged','maximum_relative_l2_difference','maximum_absolute_difference']}))
