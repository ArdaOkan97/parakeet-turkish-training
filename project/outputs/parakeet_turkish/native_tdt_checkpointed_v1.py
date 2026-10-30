"""Full-vocabulary official NeMo TDT kernel, checkpointed one clip at a time.

The physical encoder batch is retained. Each loss invocation uses the exact
un-padded T and U, FP32 joint logits, sigma=0 by default, omega=0, no FastEmit or
elementwise gradient clamp. reduction='none' returns raw NLL[B]; callers retain
their existing per-target-length averaging and preservation losses.

The kernel needs one complete joint lattice per clip; chunk_size is accepted
for the existing dependency-injection API and does not split the native DP.
This is a candidate for a measured training benchmark, not an adoption claim.
"""
from functools import lru_cache
import hashlib
import math
import numbers
from pathlib import Path
import random

import torch
from torch.utils.checkpoint import checkpoint

REDUCTIONS = frozenset(('none', 'sum', 'mean', 'mean_batch', 'mean_volume'))
DEFAULT_MAX_JOINT_BYTES = 1_600_000_000
SOURCE_FILES = {
 'collections/asr/parts/numba/rnnt_loss/rnnt_pytorch.py': '7b5ff7fe789c23a36068d20e2e3fce3d2931be6e55bcebff7efc4d86c399918e',
 'collections/asr/parts/numba/rnnt_loss/utils/cuda_utils/gpu_rnnt.py': 'df40a679bbabaeb9bc05f92bb3fbc436b80ef2f4c311564a00391934087c985c',
 'collections/asr/parts/numba/rnnt_loss/utils/cuda_utils/gpu_rnnt_kernel.py': '385de70753b6c1b2a2e5b402a689d816ece82faa846368b3d52cca7e47bb9045',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def configuration_identity():
    """Export immutable scientific settings and pinned official kernel files."""
    return {'kind': 'native_nemo_checkpointed_per_clip_v1',
            'token_logit_dtype': 'float32', 'full_vocabulary_normalization': True,
            'native_loss_reduction': 'none', 'default_sigma': 0.0, 'omega': 0.0,
            'fastemit_lambda': 0.0, 'native_gradient_clamp': -1.0,
            'recompute': 'torch.checkpoint(use_reentrant=False) per clip',
            'native_time_chunking': False, 'max_joint_bytes_default': DEFAULT_MAX_JOINT_BYTES,
            'python_rng_preserved': True, 'source_files_sha256': dict(SOURCE_FILES),
            'scope': 'Official loss kernel with existing HF projections; no full NeMo trainer/model equivalence'}


@lru_cache(maxsize=8)
def _native_loss(blank_id, durations, sigma):
    import nemo
    from nemo.collections.asr.parts.numba.rnnt_loss import TDTLossNumba
    root = Path(nemo.__file__).resolve().parent
    for relative, digest in SOURCE_FILES.items():
        _require(sha(root / relative) == digest, 'Official native source changed: ' + relative)
    return TDTLossNumba(blank=blank_id, durations=list(durations), reduction='none',
                        fastemit_lambda=0.0, clamp=-1.0, sigma=sigma, omega=0.0)


def _host_lengths(value, batch, maximum, name):
    if value is None:
        values = [maximum] * batch
    elif isinstance(value, torch.Tensor):
        _require(value.ndim == 1 and value.shape[0] == batch and value.dtype in
                 (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8),
                 name + ' must be an integer vector with batch elements')
        values = value.detach().cpu().tolist()
    else:
        values = list(value)
    _require(len(values) == batch and all(isinstance(x, numbers.Integral)
             and not isinstance(x, bool) and 1 <= x <= maximum for x in values),
             name + ' must contain positive integers within padded bounds')
    return tuple(int(x) for x in values)


def _validated_plan(ep, dp, labels, vocab_size, blank_id, durations,
                    logit_lengths, target_lengths, max_joint_bytes):
    _require(ep.ndim == 3 and dp.ndim == 3 and labels.ndim == 2,
             'Projected tensors must be (B,sequence,H), labels (B,U)')
    batch, max_t, hidden = ep.shape
    _require(batch > 0 and max_t > 0 and labels.shape[1] > 0 and
             labels.shape[0] == dp.shape[0] == batch and
             dp.shape[1] == labels.shape[1] + 1 and dp.shape[2] == hidden,
             'Positive matching batches, hidden size, frames and U+1 decoder states required')
    _require(ep.device == dp.device == labels.device and labels.dtype == torch.long,
             'All inputs must share a device and labels must be int64')
    _require(type(vocab_size) is int and vocab_size > 1 and
             type(blank_id) is int and 0 <= blank_id < vocab_size, 'Invalid vocabulary/blank')
    durations = tuple(durations)
    _require(durations == tuple(range(len(durations))) and len(durations) >= 2,
             'Native adapter requires contiguous durations beginning with 0 and 1')
    _require(type(max_joint_bytes) is int and max_joint_bytes > 0, 'Positive joint byte bound required')
    frames = _host_lengths(logit_lengths, batch, max_t, 'logit_lengths')
    targets = _host_lengths(target_lengths, batch, labels.shape[1], 'target_lengths')
    host_labels = labels.detach().cpu()
    for index, (t, u) in enumerate(zip(frames, targets)):
        valid = host_labels[index, :u]
        _require(bool(((valid >= 0) & (valid < vocab_size) & (valid != blank_id)).all()),
                 'Valid target tokens must be lexical vocabulary IDs, excluding blank')
        _require(t * (u + 1) * (vocab_size + len(durations)) * 4 <= max_joint_bytes,
                 'Exact full native joint exceeds the configured per-clip byte bound; no cropping allowed')
    return frames, targets, durations


def _clip_loss(joint, ep, dp, target, frame_length, target_length, backend,
               vocab_size, duration_count):
    acts = joint(ep[:, :, None], dp[:, None]).float().contiguous()
    _require(acts.shape == (1, ep.shape[1], dp.shape[1], vocab_size + duration_count),
             'Joint output vocabulary/duration width changed')
    # Official NeMo calls random.uniform even with omega=0. The branch is
    # deterministic here; preserve global Python RNG so the sampler is unaffected
    # and recomputation does not add random draws to the original training stream.
    state = random.getstate()
    try:
        value = backend(acts, target, frame_length, target_length)
    finally:
        random.setstate(state)
    _require(value.shape == (1,), "Native reduction='none' must return one raw clip NLL")
    return value


def _loss_with_backend(joint, ep, dp, labels, vocab_size, blank_id, durations,
                       logit_lengths, target_lengths, reduction, recompute,
                       max_joint_bytes, backend):
    """Core scheduling, independently CPU-testable with an explicit loss oracle."""
    frames, targets, durations = _validated_plan(ep, dp, labels, vocab_size, blank_id,
                    durations, logit_lengths, target_lengths, max_joint_bytes)
    losses = []
    for index, (t, u) in enumerate(zip(frames, targets)):
        clip_ep = ep[index:index + 1, :t]
        clip_dp = dp[index:index + 1, :u + 1]
        target = labels[index:index + 1, :u].contiguous()
        frame_length = torch.tensor([t], device=ep.device, dtype=torch.long)
        target_length = torch.tensor([u], device=ep.device, dtype=torch.long)

        # Bind clip-specific lengths/labels as defaults. Backward recomputation
        # must not observe the next loop iteration's closure values.
        def evaluate(e, d, y, tl=frame_length, ul=target_length):
            return _clip_loss(joint, e, d, y, tl, ul, backend, vocab_size, len(durations))

        value = (checkpoint(evaluate, clip_ep, clip_dp, target, use_reentrant=False)
                 if recompute and torch.is_grad_enabled() else evaluate(clip_ep, clip_dp, target))
        losses.append(value)
    raw = torch.cat(losses)
    denominator = raw.new_tensor(targets)
    if reduction == 'none':
        return raw
    if reduction == 'sum':
        return raw.sum()
    if reduction == 'mean_batch':
        return raw.mean()
    if reduction == 'mean_volume':
        return raw.sum() / denominator.sum()
    return (raw / denominator).mean()


def chunked_tdt_loss(joint, encoder_projected, decoder_projected, labels,
                     vocab_size, blank_id, durations, logit_lengths=None,
                     target_lengths=None, chunk_size=16, sigma=0.0,
                     reduction='mean', recompute=True, *, omega=0.0,
                     max_joint_bytes=DEFAULT_MAX_JOINT_BYTES):
    """Compatible current-control TDT loss; full native DP cannot time-chunk."""
    _require(encoder_projected.device.type == 'cuda', 'Official TDT kernel requires CUDA')
    _require(type(chunk_size) is int and chunk_size > 0, 'Compatibility chunk_size must be positive')
    _require(reduction in REDUCTIONS, 'Unsupported loss reduction')
    _require(isinstance(sigma, numbers.Real) and math.isfinite(float(sigma)) and sigma >= 0,
             'Sigma must be finite and nonnegative')
    _require(omega == 0.0, 'Matched checkpointed adapter requires omega=0; no random branch resampling')
    backend = _native_loss(blank_id, tuple(durations), float(sigma))
    return _loss_with_backend(joint, encoder_projected, decoder_projected, labels,
                vocab_size, blank_id, durations, logit_lengths, target_lengths,
                reduction, recompute, max_joint_bytes, backend)
