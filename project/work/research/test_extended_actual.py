"""One-step synthetic GPU integration check; not an ASR quality benchmark."""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'outputs' / 'parakeet_turkish'))
from runtime import load_model
from train import forward_cached, teacher_kl, train_mode
from extend_vocab import extend_processor_and_model

torch.manual_seed(123)
model, processor = load_model()
old_size = model.config.vocab_size
teacher = {name: copy.deepcopy(getattr(model, name)).eval()
           for name in ('encoder_projector', 'decoder', 'joint')}
for module in teacher.values():
    for param in module.parameters():
        param.requires_grad_(False)
audit = extend_processor_and_model(processor, model)
for param in model.encoder.parameters():
    param.requires_grad_(False)
params = [param for param in model.parameters() if param.requires_grad]
train_mode(model)

rng = np.random.default_rng(20261010)
audio = rng.normal(0, .03, 48000).astype(np.float32)
inputs = processor(audio=[audio], sampling_rate=16000, return_tensors='pt')
inputs = {key: value.cuda() for key, value in inputs.items()}
with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
    encoded = model.encoder(**inputs)
    length = int(encoded.attention_mask.sum())
    features = encoded.last_hidden_state[0, :length].cpu().clone()
# clone outside inference_mode produces ordinary tensors suitable for autograd.
features = features.clone()
new_params = (model.decoder.embedding.weight, model.joint.head.weight, model.joint.head.bias)
previous = [param[old_size:model.config.vocab_size].detach().clone() for param in new_params]
optimizer = torch.optim.AdamW(params, lr=1e-5, weight_decay=.01)
losses = []
kls = []
started = time.perf_counter()
for text, retention in [('ğĞıİşŞ', False), ('Hello world.', True)]:
    labels = processor.tokenizer(text, return_tensors='pt')['input_ids'][0]
    item = {'features': features, 'labels': labels}
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out, cuda_features, decoder_ids = forward_cached(model, item)
        kl = teacher_kl(model, teacher, cuda_features, decoder_ids, out.logits, old_size) if retention else torch.zeros((), device='cuda')
        total = (out.loss + .5 * kl) / 2
    assert torch.isfinite(total), (text, out.loss, kl)
    total.backward()
    losses.append(float(out.loss.detach()))
    kls.append(float(kl.detach()))
    del out, cuda_features, decoder_ids, total, kl
finite_grads = all(torch.isfinite(param.grad).all() for param in params if param.grad is not None)
assert finite_grads
new_gradient_norms = [float(param.grad[old_size:model.config.vocab_size].norm()) for param in new_params]
assert all(norm > 0 for norm in new_gradient_norms)
norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
optimizer.step()
with torch.no_grad():
    for param, old in zip(new_params, previous):
        rows = param[old_size:model.config.vocab_size]
        rows.copy_(old + 100 * (rows - old))
update_norms = [float((param[old_size:model.config.vocab_size] - old).norm())
                for param, old in zip(new_params, previous)]
assert all(norm > 0 for norm in update_norms)
assert all(torch.isfinite(param).all() for param in new_params)
assert not any(param.grad is not None for param in model.encoder.parameters())
report = {
    'test': 'Actual pretrained v3 extended tokenizer, synthetic 3-second audio, Turkish and English cached forward/backward, one optimizer step',
    'base_revision': '541d1f99c6b0c3cd0b11a95167540bb8edefd82b',
    'gpu': torch.cuda.get_device_name(0),
    'vocab_size': model.config.vocab_size,
    'blank_token_id': model.config.blank_token_id,
    'features_shape': list(features.shape),
    'tdt_losses_turkish_english': losses,
    'teacher_kl_turkish_english': kls,
    'finite_gradients': finite_grads,
    'new_embedding_head_weight_head_bias_gradient_norms': new_gradient_norms,
    'new_embedding_head_weight_head_bias_update_norms': update_norms,
    'gradient_norm_before_clip': float(norm),
    'encoder_gradients_absent': True,
    'effective_new_row_learning_rate': .001,
    'other_rows_learning_rate': 1e-5,
    'elapsed_training_seconds': time.perf_counter() - started,
    'peak_allocated_vram_gb': torch.cuda.max_memory_allocated() / 1e9,
    'scope': 'Integration smoke only; no real-speech quality claim and no model checkpoint saved',
}
path = ROOT / 'outputs' / 'parakeet_turkish' / 'actual_extension_gpu_smoke.json'
path.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2), flush=True)
