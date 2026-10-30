"""CPU-only LoRA integration tests and meta-device official model counts."""
import copy
import json
import sys
from pathlib import Path

import torch
from transformers import ParakeetEncoderConfig, ParakeetForTDT, ParakeetTDTConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'outputs' / 'parakeet_turkish'))
from encoder_lora import LoRALinear, adapter_state, encoder_lora_metadata, install_encoder_lora, merge_encoder_lora

torch.set_num_threads(4)
torch.manual_seed(20261010)
encoder_config = ParakeetEncoderConfig(
    hidden_size=16, intermediate_size=32, num_hidden_layers=3,
    num_attention_heads=2, num_key_value_heads=2, num_mel_bins=16,
    subsampling_conv_channels=4,
)
config = ParakeetTDTConfig(encoder_config=encoder_config, vocab_size=16,
                           blank_token_id=15, pad_token_id=2,
                           decoder_hidden_size=8, num_decoder_layers=1)
model = ParakeetForTDT(config).eval()
inputs = {'input_features': torch.randn(1, 40, 16),
          'attention_mask': torch.ones(1, 40, dtype=torch.long)}
with torch.no_grad():
    before = model.encoder(**inputs).last_hidden_state.clone()
decoder_flags = {name: param.requires_grad for name, param in model.named_parameters() if not name.startswith('encoder.')}
meta = install_encoder_lora(model, rank=4, alpha=8, last_n_layers=2)
with torch.no_grad():
    zero_initialized = model.encoder(**inputs).last_hidden_state
assert torch.equal(before, zero_initialized)
assert meta['adapted_layer_indices'] == [1, 2]
assert meta['adapter_parameter_count'] == 2 * 4 * 4 * (16 + 16)
assert meta['non_adapter_encoder_trainable_parameter_count'] == 0
assert decoder_flags == {name: param.requires_grad for name, param in model.named_parameters() if not name.startswith('encoder.')}
assert all(param.dtype == torch.float32 for name, param in model.encoder.named_parameters() if name.endswith(('lora_A', 'lora_B')))

parameters = [param for param in model.encoder.parameters() if param.requires_grad]
optimizer = torch.optim.SGD(parameters, lr=.1)
loss = model.encoder(**inputs).last_hidden_state[0, 0, 0]
loss.backward()
assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in parameters)
b_grad = sum(float(param.grad.norm()) for name, param in model.encoder.named_parameters() if name.endswith('lora_B'))
assert b_grad > 0
assert not any(param.grad is not None for name, param in model.encoder.named_parameters() if not name.endswith(('lora_A', 'lora_B')))
optimizer.step()
optimizer.zero_grad(set_to_none=True)
loss_second = model.encoder(**inputs).last_hidden_state[0, 0, 1]
loss_second.backward()
a_grad = sum(float(param.grad.norm()) for name, param in model.encoder.named_parameters() if name.endswith('lora_A'))
assert a_grad > 0
assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in parameters)
with torch.no_grad():
    unmerged = model.encoder(**inputs).last_hidden_state.clone()
state = adapter_state(model)
assert len(state['weights']) == 16
assert all(tensor.device.type == 'cpu' for tensor in state['weights'].values())
merged_meta = merge_encoder_lora(model)
with torch.no_grad():
    merged = model.encoder(**inputs).last_hidden_state
torch.testing.assert_close(unmerged, merged, rtol=3e-5, atol=2e-6)
assert not any(isinstance(module, LoRALinear) for module in model.modules())
assert encoder_lora_metadata(model)['adapter_parameter_count'] == 0
save_path = ROOT / 'work' / 'research' / 'lora_merged_tiny'
model.save_pretrained(save_path)
restored = ParakeetForTDT.from_pretrained(save_path, local_files_only=True).eval()
with torch.no_grad():
    reload_output = restored.encoder(**inputs).last_hidden_state
assert torch.equal(merged, reload_output)

# BF16 autocast on CPU exercises the same dtype contract without using CUDA.
linear = torch.nn.Linear(16, 16).eval()
wrapped = LoRALinear(copy.deepcopy(linear), rank=8, alpha=16)
linear_input = torch.randn(1, 3, 16)
with torch.autocast('cpu', dtype=torch.bfloat16):
    plain_bf16 = linear(linear_input)
    lora_bf16 = wrapped(linear_input)
assert torch.equal(plain_bf16, lora_bf16)
assert wrapped.lora_A.dtype == wrapped.lora_B.dtype == torch.float32
lora_bf16.float().square().mean().backward()
assert torch.isfinite(wrapped.lora_B.grad).all()

actual_config = ParakeetTDTConfig.from_dict(json.loads((ROOT / 'work/research/config.json').read_text(encoding='utf-8')))
actual_config.vocab_size = 8199
plans = []
for rank in (4, 8):
    for last_n_layers in (8, 12, None):
        with torch.device('meta'):
            actual = ParakeetForTDT(actual_config)
            base_total = sum(param.numel() for param in actual.parameters())
            base_encoder = sum(param.numel() for param in actual.encoder.parameters())
            installed = install_encoder_lora(actual, rank=rank, alpha=2 * rank, last_n_layers=last_n_layers)
        assert all(param.device.type == 'meta' for param in actual.parameters())
        plans.append({
            'rank': rank, 'alpha': 2 * rank,
            'last_n_layers': last_n_layers,
            'adapted_layer_indices': installed['adapted_layer_indices'],
            'adapter_parameter_count': installed['adapter_parameter_count'],
            'base_total_parameters_with_extended_vocab': base_total,
            'base_encoder_parameters': base_encoder,
            'full_decoder_joint_projection_parameters': base_total - base_encoder,
            'adapter_plus_full_heads_trainable_parameters': installed['adapter_parameter_count'] + base_total - base_encoder,
            'actual_weight_storage_allocated': False,
        })
        del actual

report = {
    'test_device': 'CPU; official architecture parameter counts use storage-free meta tensors',
    'zero_initialized_encoder_output_exact': True,
    'base_encoder_weights_and_biases_frozen': True,
    'decoder_joint_trainability_unchanged': True,
    'adapter_dtype_fp32': True,
    'all_adapter_gradients_finite': True,
    'initial_B_gradient_norm_sum': b_grad,
    'second_backward_A_gradient_norm_sum': a_grad,
    'merge_fp32_max_absolute_output_difference': float((unmerged - merged).abs().max()),
    'merged_hf_save_reload_output_exact': True,
    'cpu_bf16_autocast_zero_initialization_exact': True,
    'cpu_bf16_autocast_gradients_finite': True,
    'official_encoder_hidden_size': 1024,
    'official_encoder_layer_count': 24,
    'official_projection_sizes': 'q/k/v/o each1024x1024',
    'actual_config_plans': plans,
    'scope': 'Implementation smoke and exact architecture counts; no CUDA execution or real-speech ASR claim',
}
output = ROOT / 'outputs/parakeet_turkish/encoder_lora_verification.json'
output.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2), flush=True)
