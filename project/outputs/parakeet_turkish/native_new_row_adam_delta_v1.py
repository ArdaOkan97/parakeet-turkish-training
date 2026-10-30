"""Optional declared new-six-row AdamW delta scaling through official hooks.

This is a scientific recipe change, not a memory or gradient-equivalence fix.
The owner must validate an actual optimizer step before any training adoption.
Importing this module requires only the standard library.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def install_native_new_row_adam_delta(model, optimizer, multiplier=6.0, torch_module=None):
    if multiplier not in (1.0, 6.0):
        raise ValueError("Only declared control1 or experimental6 multiplier is supported")
    if torch_module is None:
        import torch as torch_module
    if getattr(optimizer, "_native_new_row_delta_hook_v1", None) is not None:
        raise ValueError("Native row-delta hooks are already installed")
    parameters = {
        "decoder.prediction.embed.weight": model.decoder.prediction["embed"].weight,
        "joint.joint_net.2.weight": model.joint.joint_net[2].weight,
        "joint.joint_net.2.bias": model.joint.joint_net[2].bias,
    }
    if model.decoder.blank_idx != 8198 or model.decoder.prediction["embed"].padding_idx != 8198:
        raise ValueError("Expected native lexical8198 with blank8198")
    if parameters["decoder.prediction.embed.weight"].shape[0] != 8199 or any(
            parameter.shape[0] != 8204 for name, parameter in parameters.items() if name.startswith("joint.")):
        raise ValueError("Unexpected native lexical/blank/duration row shapes")
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if any(not parameter.requires_grad or parameter.dtype != torch_module.float32 or id(parameter) not in optimizer_ids
           for parameter in parameters.values()):
        raise ValueError("Each row target must remain the actual FP32 trainable optimizer parameter")
    token_rows = slice(8192, 8198)
    snapshots = {}
    marker = object()
    statistics = {"pre_step_calls": 0, "post_step_calls": 0, "scaled_optimizer_updates": 0}

    def before_step(actual_optimizer, args, kwargs):
        if actual_optimizer is not optimizer or snapshots:
            raise ValueError("Native row delta hook ownership/nesting changed")
        for name, parameter in parameters.items():
            snapshots[name] = parameter[token_rows].detach().clone()
        statistics["pre_step_calls"] += 1
        return None

    def after_step(actual_optimizer, args, kwargs):
        if actual_optimizer is not optimizer or len(snapshots) != 3:
            raise ValueError("Native row delta hook has no matching pre-step snapshot")
        try:
            if multiplier != 1.0:
                with torch_module.no_grad():
                    for name, parameter in parameters.items():
                        previous = snapshots[name]
                        actual_delta = parameter[token_rows] - previous
                        parameter[token_rows].copy_(previous + multiplier * actual_delta)
                statistics["scaled_optimizer_updates"] += 1
            statistics["post_step_calls"] += 1
        finally:
            snapshots.clear()
        return None

    pre_handle = optimizer.register_step_pre_hook(before_step)
    try:
        post_handle = optimizer.register_step_post_hook(after_step)
    except BaseException:
        pre_handle.remove()
        raise
    optimizer._native_new_row_delta_hook_v1 = marker

    def remove():
        if getattr(optimizer, "_native_new_row_delta_hook_v1", None) is not marker:
            raise ValueError("Native row-delta hook ownership changed")
        pre_handle.remove()
        post_handle.remove()
        snapshots.clear()
        del optimizer._native_new_row_delta_hook_v1

    return {"remove": remove, "identity": {
        "kind": "native_new_six_rows_actual_AdamW_delta_scaling_v1",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "multiplier": multiplier, "parameter_names": list(parameters), "new_token_rows": list(range(8192, 8198)),
        "unchanged_native_blank_row": 8198, "unchanged_duration_rows": list(range(8199, 8204)),
        "implementation": "official optimizer.register_step_pre_hook/post_hook; snapshot then scale actual parameter delta",
        "optimizer_state_or_moments_mutated_by_hook": False, "gradient_scale_or_global_head_LR_changed": False,
        "RNG_or_backend_policy_changed": False, "scientific_recipe_change": multiplier != 1.0,
        "actual_GPU_step_validation_pending": True, "statistics": statistics}}
