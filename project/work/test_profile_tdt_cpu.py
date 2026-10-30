"""CPU-only audit of profile input selection and 24-block checkpoint wrapper."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUTPUT))
import profile_tdt as profile
from encoder_lora import install_encoder_lora

torch.set_num_threads(2)

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ('q_proj','k_proj','v_proj','o_proj'):
            setattr(self,name,nn.Linear(16,16))
    def forward(self,value):
        return self.o_proj(torch.tanh(self.q_proj(value)+self.k_proj(value)+self.v_proj(value)))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn=Attention()
        self.norm=nn.BatchNorm1d(16)
        self.gradient_checkpointing=False
    def forward(self,value,attention_mask=None,position_embeddings=None):
        return value+0.05*self.norm(self.self_attn(value).transpose(1,2)).transpose(1,2)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.ModuleList([Block() for _ in range(24)])
    def forward(self,value):
        for layer in self.layers:
            value=layer(value,attention_mask=None,position_embeddings=None)
        return value

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=Encoder()

torch.manual_seed(809)
left=Model()
metadata=install_encoder_lora(left,rank=8,alpha=16.0)
left.encoder.eval()
right=copy.deepcopy(left)
profile.checkpoint_encoder_blocks(right)
value=torch.randn(1,11,16)  # no input gradient needed for first trainable LoRA block
buffers={name:tensor.clone() for name,tensor in right.named_buffers()}
with torch.autocast('cpu',dtype=torch.bfloat16):
    original=left.encoder(value).float().square().mean()
    changed=right.encoder(value).float().square().mean()
original.backward(); changed.backward()
deltas=[]
for (name,p),(other,q) in zip(left.named_parameters(),right.named_parameters()):
    assert name==other
    if p.requires_grad:
        assert p.grad is not None and q.grad is not None
        assert torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all()
        deltas.append(float((p.grad-q.grad).abs().max()))
    else:
        assert p.grad is None and q.grad is None
assert float((original.detach()-changed.detach()).abs())==0
assert max(deltas)==0
assert all(torch.equal(buffers[name],tensor) for name,tensor in right.named_buffers())
assert not right.encoder.training and all(not layer.training for layer in right.encoder.layers)
examples=profile.cpu_examples(OUTPUT/'memory_profile_examples.json')
assert set(examples)=={'tr','replay','short_parity'}
report={'status':'passed','gpu_used':False,
    'profile_source_sha256':hashlib.sha256((OUTPUT/'profile_tdt.py').read_bytes()).hexdigest(),
    'scope':'CPU BF16 toy 24-block encoder validates explicit wrapper/autograd/BN freezing; actual Parakeet GPU profile remains unrun.',
    'adapted_layers':len(metadata['adapted_layer_indices']),'adapted_projections':len(metadata['targets']),
    'maximum_adapter_gradient_difference':max(deltas),
    'same_encoder_loss':True,'original_encoder_parameters_have_no_gradients':True,
    'batchnorm_buffers_unchanged':True,'input_requires_grad':False,
    'actual_selected_cache_positive_T_U_and_WAV_hash_checks':True,
    'actual_selected_T_U':{name:[example['T'],example['U']] for name,example in examples.items()}}
profile.write_json(OUTPUT/'tdt_profile_cpu_verification.json',report)
print(json.dumps(report,indent=2))
