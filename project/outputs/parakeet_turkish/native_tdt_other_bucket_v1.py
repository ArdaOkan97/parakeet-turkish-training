"""INACTIVE exact-mass OTHER-bucket TDT projection prototype.

No Torch/NeMo import, model load or kernel launch occurs at import. A future
caller injects Torch, its nonreentrant checkpoint function and the independently
verified official native callback. This does not replace a trainer/loss adapter.
Mathematical first-order equivalence is conditional on sigma/omega/FastEmit0 and
effective native clamp0. Floating projection/reduction order is different and
requires actual full-logit/native/model-gradient and memory comparisons.
"""
from __future__ import annotations

import math
import random

SOURCE_FILES = {
    "collections/asr/parts/numba/rnnt_loss/rnnt_pytorch.py": "7b5ff7fe789c23a36068d20e2e3fce3d2931be6e55bcebff7efc4d86c399918e",
    "collections/asr/parts/numba/rnnt_loss/utils/cuda_utils/gpu_rnnt.py": "df40a679bbabaeb9bc05f92bb3fbc436b80ef2f4c311564a00391934087c985c",
    "collections/asr/parts/numba/rnnt_loss/utils/cuda_utils/gpu_rnnt_kernel.py": "385de70753b6c1b2a2e5b402a689d816ece82faa846368b3d52cca7e47bb9045",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def retained_plan(target_ids, vocab_size, blank_id, durations=(0, 1, 2, 3, 4), vocab_chunk_size=1024):
    """Metadata only. Remap ONLY native DP labels, never decoder/sampler IDs."""
    target_ids = tuple(target_ids)
    durations = tuple(durations)
    require(type(vocab_size) is int and vocab_size > 1 and type(blank_id) is int
        and 0 <= blank_id < vocab_size, "Positive full vocabulary and valid explicit blank required")
    require(1 <= len(target_ids) <= 180 and all(type(value) is int and 0 <= value < vocab_size
        and value != blank_id for value in target_ids), "Whole positive accepted target, max180 and no blank/truncation")
    require(type(vocab_chunk_size) is int and 1 <= vocab_chunk_size <= vocab_size,
            "Positive bounded vocabulary chunk required")
    require(len(durations) >= 2 and durations == tuple(range(len(durations))), "Duration0/1 and contiguous duration order required")
    retained = tuple(sorted(set(target_ids) | {blank_id}))
    mapping = {value: index for index, value in enumerate(retained)}
    chunks = []
    for start in range(0, vocab_size, vocab_chunk_size):
        stop = min(vocab_size, start + vocab_chunk_size)
        local = tuple(value - start for value in retained if start <= value < stop)
        chunks.append({"start": start, "stop": stop, "retained_local_indices": local,
                       "other_count": stop - start - len(local)})
    other_count = vocab_size - len(retained)
    return {"vocab_size": vocab_size, "source_blank_id": blank_id, "target_ids": target_ids,
        "target_length": len(target_ids), "retained_ids": retained,
        "remapped_target_ids": tuple(mapping[value] for value in target_ids),
        "native_blank_id": mapping[blank_id], "other_index": len(retained) if other_count else None,
        "other_class_count": other_count, "native_vocab_size": len(retained) + bool(other_count),
        "durations": durations, "vocabulary_chunks": tuple(chunks),
        "all_non_target_nonblank_mass_included": True, "decoder_ids_or_target_length_changed": False}


def scalar_logsumexp(values):
    values = tuple(values)
    require(values and all(math.isfinite(value) for value in values), "Nonempty finite scalar logit sequence required")
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def scalar_coalesce(logits, plan):
    """Float64 stdlib sanity primitive, not an actual Torch/native execution."""
    require(len(logits) == plan["vocab_size"] and all(math.isfinite(value) for value in logits),
            "Complete finite full-vocabulary logits required")
    kept = [logits[index] for index in plan["retained_ids"]]
    if plan["other_class_count"]:
        retained = set(plan["retained_ids"])
        kept.append(scalar_logsumexp(value for index, value in enumerate(logits) if index not in retained))
    return kept


def projection_bounds(plan, frames, hidden_size, time_chunk_size=16):
    """Arithmetic allocations only; not a VRAM/private-commit safety proof."""
    require(type(frames) is int and frames > 0 and type(hidden_size) is int and hidden_size > 0
        and type(time_chunk_size) is int and 1 <= time_chunk_size <= frames,
        "Positive whole frames/hidden and bounded time chunk required")
    cells = frames * (plan["target_length"] + 1)
    block_cells = time_chunk_size * (plan["target_length"] + 1)
    widest = max(chunk["stop"] - chunk["start"] for chunk in plan["vocabulary_chunks"])
    return {"full_reference_FP32_joint_bytes": cells * (plan["vocab_size"] + len(plan["durations"])) * 4,
        "compressed_native_FP32_token_bytes": cells * plan["native_vocab_size"] * 4,
        "compressed_native_FP32_token_gradient_bytes": cells * plan["native_vocab_size"] * 4,
        "FP32_duration_tensor_bytes": cells * len(plan["durations"]) * 4,
        "largest_FP32_projection_chunk_bytes": block_cells * widest * 4,
        "FP32_hidden_time_block_bytes": block_cells * hidden_size * 4,
        "vocabulary_chunk_count": len(plan["vocabulary_chunks"]), "total_memory_or_GPU_fit_proven": False,
        "full_vocabulary_projection_FLOPs_removed": False}


def _validate_native_policy(policy):
    require(policy.get("sigma") == policy.get("omega") == policy.get("fastemit_lambda") == 0.
        and policy.get("effective_native_clamp") == 0. and policy.get("reduction") == "none"
        and policy.get("source_files_sha256") == SOURCE_FILES,
        "Pinned official raw-NLL callback with sigma/omega/FastEmit0/effectiveclamp0 required")


def compressed_clip_loss(encoder_projected, decoder_projected, head_weight, head_bias, target_ids,
                         *, vocab_size, blank_id, activation, torch_module, checkpoint_fn,
                         native_callback, native_policy, durations=(0, 1, 2, 3, 4),
                         time_chunk_size=16, vocab_chunk_size=1024, recompute=True,
                         max_compressed_token_bytes=128_000_000, max_projection_FP32_bytes=64_000_000):
    """Future one-clip prototype, returns RAW native NLL[1] plus metadata.

    Callback receives FP32 token logits, FP32 duration LOG probabilities, remapped
    DP labels, actual length tensors, and explicit blank/duration/policy kwargs.
    It must call the verified low-level backend.loss; high-level RNNTLoss's
    blank-last conventions are not silently assumed. No model config/head,
    tokenizer, decoder input or optimizer is mutated. Existing full-denominator
    teacher KL remains outside this TDT-only prototype.
    """
    torch = torch_module
    _validate_native_policy(native_policy)
    require(callable(activation) and callable(native_callback) and callable(checkpoint_fn),
            "Explicit deterministic joint activation/native/checkpoint callbacks required")
    require(encoder_projected.ndim == decoder_projected.ndim == 3
        and encoder_projected.shape[0] == decoder_projected.shape[0] == 1, "One whole unpadded clip required")
    frames, hidden = encoder_projected.shape[1:]
    plan = retained_plan(target_ids, vocab_size, blank_id, durations, min(vocab_chunk_size, vocab_size))
    require(frames > 0 and hidden > 0 and decoder_projected.shape[1:] == (plan["target_length"] + 1, hidden)
        and head_weight.shape == (vocab_size + len(plan["durations"]), hidden)
        and head_bias.shape == (vocab_size + len(plan["durations"]),), "Full source T/U+1/head/duration geometry required")
    require(encoder_projected.device == decoder_projected.device == head_weight.device == head_bias.device
        and head_weight.dtype == head_bias.dtype == torch.float32,
        "Same-device projected tensors and source FP32 trainable head required")
    require(type(time_chunk_size) is int and time_chunk_size > 0
        and type(vocab_chunk_size) is int and vocab_chunk_size > 0
        and type(max_compressed_token_bytes) is int and max_compressed_token_bytes > 0
        and type(max_projection_FP32_bytes) is int and max_projection_FP32_bytes > 0,
        "Explicit positive prototype chunk/byte limits required")
    bounds = projection_bounds(plan, frames, hidden, min(time_chunk_size, frames))
    require(bounds["compressed_native_FP32_token_bytes"] <= max_compressed_token_bytes,
            "Compressed lattice exceeds prototype byte limit; no clip truncation/cropping")
    require(bounds["largest_FP32_projection_chunk_bytes"] <= max_projection_FP32_bytes
        and bounds["FP32_hidden_time_block_bytes"] <= max_projection_FP32_bytes,
        "Prototype projection/hidden block exceeds explicit arithmetic byte bound")

    def project_time(e, d, weight, bias):
        h = activation(e[:, :, None, :] + d[:, None, :, :])
        kept_parts, other_parts = [], []
        for chunk in plan["vocabulary_chunks"]:
            start, stop = chunk["start"], chunk["stop"]
            # Bind loop metadata now: backward recomputation must not observe a
            # later vocabulary chunk's labels or excluded-class mask.
            def project_vocab(x, w, b, retained=chunk["retained_local_indices"], other_count=chunk["other_count"]):
                raw = torch.nn.functional.linear(x, w, b).float()
                ids = torch.tensor(retained, device=x.device, dtype=torch.long)
                kept = raw.index_select(-1, ids)
                if other_count:
                    excluded = torch.zeros(raw.shape[-1], device=x.device, dtype=torch.bool)
                    if retained:
                        excluded[ids] = True
                    other = torch.logsumexp(raw.masked_fill(excluded, float("-inf")), dim=-1)
                else:
                    # Never logsumexp an empty/all-masked chunk: that would give
                    # undefined backward even if a later chunk has OTHER mass.
                    other = None
                return kept, other
            inputs = (h, weight[start:stop], bias[start:stop])
            kept, other = (checkpoint_fn(project_vocab, *inputs, use_reentrant=False, preserve_rng_state=True)
                           if recompute and torch.is_grad_enabled() else project_vocab(*inputs))
            if chunk["retained_local_indices"]:
                kept_parts.append(kept)
            if other is not None:
                other_parts.append(other)
        label_acts = torch.cat(kept_parts, dim=-1)
        if other_parts:
            other = other_parts[0]
            for part in other_parts[1:]:
                other = torch.logaddexp(other, part)
            label_acts = torch.cat((label_acts, other.unsqueeze(-1)), dim=-1)
        duration_logits = torch.nn.functional.linear(h, weight[vocab_size:], bias[vocab_size:]).float()
        duration_log_probs = torch.nn.functional.log_softmax(duration_logits, dim=-1)
        return label_acts, duration_log_probs

    parts = []
    for start in range(0, frames, time_chunk_size):
        inputs = (encoder_projected[:, start:start + time_chunk_size], decoder_projected, head_weight, head_bias)
        # Only projections are checkpointed. The official native callback keeps
        # compact gradients and is not recomputed by a whole-loss checkpoint.
        part = (checkpoint_fn(project_time, *inputs, use_reentrant=False, preserve_rng_state=True)
                if recompute and torch.is_grad_enabled() else project_time(*inputs))
        parts.append(part)
    label_acts = torch.cat([part[0] for part in parts], dim=1).contiguous()
    duration_acts = torch.cat([part[1] for part in parts], dim=1).contiguous()
    require(label_acts.shape == (1, frames, plan["target_length"] + 1, plan["native_vocab_size"])
        and duration_acts.shape == (1, frames, plan["target_length"] + 1, len(plan["durations"]))
        and label_acts.dtype == duration_acts.dtype == torch.float32, "Exact compressed/native duration operands required")
    device = encoder_projected.device
    labels = torch.tensor([plan["remapped_target_ids"]], device=device, dtype=torch.long)
    frame_lengths = torch.tensor([frames], device=device, dtype=torch.long)
    target_lengths = torch.tensor([plan["target_length"]], device=device, dtype=torch.long)
    state = random.getstate()
    try:
        value = native_callback(label_acts, duration_acts, labels, frame_lengths, target_lengths,
            blank_id=plan["native_blank_id"], durations=plan["durations"], reduction="none",
            sigma=0., omega=0., fastemit_lambda=0., clamp=0.)
    finally:
        random.setstate(state)
    require(value.shape == (1,), "Official callback must return one RAW clip NLL; caller's original U reduction unchanged")
    return value, {"plan": plan, "allocation_arithmetic": bounds,
        "all_full_vocabulary_mass_included": True, "native_DP_recomputed_by_checkpoint": False,
        "head_projection_recomputed_by_checkpoint": bool(recompute), "Python_RNG_preserved": True,
        "native_effective_clamp": 0., "raw_NLL_and_first_order_gradient_equivalence_mathematical_only": True,
        "floating_point_bitwise_equivalence_claimed": False, "actual_GPU_or_ASR_readiness_proven": False,
        "model_decoder_tokenizer_optimizer_or_sampler_modified": False}
