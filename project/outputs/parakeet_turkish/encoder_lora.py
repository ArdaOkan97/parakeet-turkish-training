"""Small attention LoRA adapters for the native HF Parakeet encoder.

Install before constructing the optimizer. This helper freezes the encoder's
original parameters, leaves decoder/joint trainability unchanged, and does not
alter model configuration or tokenizer. Keep the encoder in eval mode while
training these adapters to freeze BatchNorm statistics and disable LayerDrop.
Eval mode does not disable gradients. Default adapter dropout is zero.

Use adapter_state for a separate adapter checkpoint. Merge into standard Linear
modules before exporting with HF save_pretrained; evaluate the merged export
because floating-point regrouping can slightly change outputs under autocast.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


TARGET_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoRALinear(nn.Module):
    """Frozen Linear plus ``(alpha / rank) * B(A(x))``.

    Adapter parameters remain FP32 and inherit the base layer's device. CUDA
    bf16 autocast can accelerate their matrix multiplies without lowering the
    optimizer-state precision. B starts at zero, making installation neutral.
    """

    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear requires a standard torch.nn.Linear.")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError("rank must be a positive integer.")
        if not math.isfinite(float(alpha)) or alpha <= 0:
            raise ValueError("alpha must be finite and positive.")
        if not math.isfinite(float(dropout)) or not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1).")
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / rank
        self.lora_dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(torch.empty(
            rank, self.in_features, device=base_layer.weight.device, dtype=torch.float32,
        ))
        self.lora_B = nn.Parameter(torch.zeros(
            self.out_features, rank, device=base_layer.weight.device, dtype=torch.float32,
        ))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)
        self.train(base_layer.training)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(hidden_states)
        adapter_input = self.lora_dropout(hidden_states).to(self.lora_A.dtype)
        update = F.linear(F.linear(adapter_input, self.lora_A), self.lora_B)
        return base_output + (update * self.scaling).to(base_output.dtype)

    def merged_linear(self) -> nn.Linear:
        """Return an ordinary Linear with the adapter added to its weights."""
        base = self.base_layer
        merged = nn.Linear(self.in_features, self.out_features, bias=base.bias is not None,
                           device=base.weight.device, dtype=base.weight.dtype)
        with torch.no_grad():
            delta = (self.lora_B @ self.lora_A) * self.scaling
            merged.weight.copy_(base.weight + delta.to(base.weight.dtype))
            if base.bias is not None:
                merged.bias.copy_(base.bias)
        merged.weight.requires_grad_(base.weight.requires_grad)
        if merged.bias is not None:
            merged.bias.requires_grad_(base.bias.requires_grad)
        merged.train(self.training)
        return merged


def _installed(model):
    for layer_index, layer in enumerate(model.encoder.layers):
        for projection in TARGET_PROJECTIONS:
            module = getattr(layer.self_attn, projection)
            if isinstance(module, LoRALinear):
                yield f"encoder.layers.{layer_index}.self_attn.{projection}", layer_index, projection, module


def encoder_lora_metadata(model) -> dict:
    """JSON-serializable adapter layout and exact trainable parameter counts."""
    targets = []
    parameter_count = 0
    for path, layer_index, projection, module in _installed(model):
        count = module.lora_A.numel() + module.lora_B.numel()
        parameter_count += count
        targets.append({
            "module": path, "layer_index": layer_index, "projection": projection,
            "rank": module.rank, "alpha": module.alpha, "scaling": module.scaling,
            "dropout": module.lora_dropout.p,
            "in_features": module.in_features, "out_features": module.out_features,
            "adapter_parameters": count, "adapter_dtype": str(module.lora_A.dtype),
        })
    non_adapter_trainable = sum(
        parameter.numel() for name, parameter in model.encoder.named_parameters()
        if parameter.requires_grad and not name.endswith(("lora_A", "lora_B"))
    )
    return {
        "format": "parakeet_encoder_attention_lora_v1",
        "num_encoder_layers": len(model.encoder.layers),
        "adapted_layer_indices": sorted({target["layer_index"] for target in targets}),
        "target_projection_names": list(TARGET_PROJECTIONS),
        "adapter_parameter_count": parameter_count,
        "non_adapter_encoder_trainable_parameter_count": non_adapter_trainable,
        "targets": targets,
    }


def install_encoder_lora(model, rank: int = 8, alpha: float = 16.0,
                         last_n_layers: int | None = None) -> dict:
    """Freeze encoder base weights and wrap q/k/v/o in all or upper blocks.

    ``last_n_layers=None`` selects all blocks; e.g. 8 selects blocks 16–23 of
    the official 24-block v3 encoder. Decoder and joint parameters are untouched.
    Returns layout metadata. Reinstallation raises to avoid nested adapters.
    """
    layer_count = len(model.encoder.layers)
    if last_n_layers is not None:
        if isinstance(last_n_layers, bool) or not isinstance(last_n_layers, int) or not 1 <= last_n_layers <= layer_count:
            raise ValueError("last_n_layers must be between 1 and the encoder layer count.")
    first_layer = 0 if last_n_layers is None else layer_count - last_n_layers
    if any(True for _ in _installed(model)):
        raise ValueError("Encoder LoRA is already installed; do not nest adapters.")
    replacements = []
    for index in range(first_layer, layer_count):
        attention = model.encoder.layers[index].self_attn
        for projection in TARGET_PROJECTIONS:
            original = getattr(attention, projection)
            if not isinstance(original, nn.Linear):
                raise TypeError(f"Unexpected projection module at encoder.layers.{index}.self_attn.{projection}.")
            replacements.append((attention, projection, LoRALinear(original, rank, alpha)))
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for attention, projection, replacement in replacements:
        setattr(attention, projection, replacement)
    metadata = encoder_lora_metadata(model)
    if metadata["non_adapter_encoder_trainable_parameter_count"]:
        raise AssertionError("An original encoder parameter remains trainable.")
    return metadata


def adapter_state(model) -> dict:
    """Return layout metadata and detached CPU clones of adapter tensors only."""
    weights = {}
    for path, _, _, module in _installed(model):
        weights[f"{path}.lora_A"] = module.lora_A.detach().cpu().clone()
        weights[f"{path}.lora_B"] = module.lora_B.detach().cpu().clone()
    return {"metadata": encoder_lora_metadata(model), "weights": weights}


def merge_encoder_lora(model) -> dict:
    """Replace adapters with standard Linear modules for ordinary HF export.

    Returns the pre-merge metadata. This mutates the model and invalidates any
    existing optimizer, so merge only for evaluation/export after training.
    """
    metadata = encoder_lora_metadata(model)
    for _, layer_index, projection, module in list(_installed(model)):
        setattr(model.encoder.layers[layer_index].self_attn, projection, module.merged_linear())
    if any(True for _ in _installed(model)):
        raise AssertionError("An adapter wrapper remained after merge.")
    return metadata
