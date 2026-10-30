"""Checkpointed, chunked joint projection for exact full-vocabulary TDT loss.

The dynamic program is adapted from the Apache-2.0 Transformers TDT loss at
commit 536ecc007387a50e77603bb5d92100e9b07514cc. Only blank, target-label and
duration log probabilities are retained. Full token normalization is computed
inside recomputed time chunks; this does not prune tokens or change the loss.
"""
import torch
from torch.utils.checkpoint import checkpoint


def tdt_from_log_probs(blank_log_probs, label_log_probs, duration_log_probs,
                       logit_lengths, target_lengths, durations, reduction='mean'):
    batch_size, max_t, max_u = blank_log_probs.shape
    device = blank_log_probs.device
    log_alpha = torch.full((batch_size, max_t, max_u), float('-inf'), device=device)
    log_alpha[:, 0, 0] = 0.0
    neg_inf = torch.tensor(float('-inf'), device=device)
    for n in range(1, max_t + max_u - 1):
        us = torch.arange(max(0, n - max_t + 1), min(n + 1, max_u), device=device)
        ts = n - us
        candidates = []
        for index, duration in enumerate(durations):
            previous = ts - duration
            valid_t = previous >= 0
            if not valid_t.any():
                continue
            source_t = previous.clamp(min=0)
            if duration > 0:
                contribution = (log_alpha[:, source_t, us] + blank_log_probs[:, source_t, us]
                                + duration_log_probs[:, source_t, us, index])
                candidates.append(torch.where(valid_t.unsqueeze(0), contribution, neg_inf))
            valid_both = valid_t & (us > 0)
            if valid_both.any():
                source_u = (us - 1).clamp(min=0)
                source_label = source_u.clamp(max=max_u - 2) if max_u > 1 else source_u
                contribution = (log_alpha[:, source_t, source_u] + label_log_probs[:, source_t, source_label]
                                + duration_log_probs[:, source_t, source_u, index])
                candidates.append(torch.where(valid_both.unsqueeze(0), contribution, neg_inf))
        if candidates:
            log_alpha[:, ts, us] = torch.logsumexp(torch.stack(candidates), dim=0)
    batch_indices = torch.arange(batch_size, device=device)
    log_probs = torch.full((batch_size,), float('-inf'), device=device)
    for index, duration in enumerate(durations):
        if duration == 0:
            continue
        final_t = logit_lengths - duration
        valid = final_t >= 0
        if not valid.any():
            continue
        final_t = final_t.clamp(min=0)
        terminal = (log_alpha[batch_indices, final_t, target_lengths]
                    + blank_log_probs[batch_indices, final_t, target_lengths]
                    + duration_log_probs[batch_indices, final_t, target_lengths, index])
        combined = torch.logsumexp(torch.stack([log_probs, terminal]), dim=0)
        log_probs = torch.where(valid, combined, log_probs)
    losses = -log_probs
    target_lengths = target_lengths.float()
    if reduction == 'mean':
        return (losses / target_lengths).mean()
    if reduction == 'mean_volume':
        return losses.sum() / target_lengths.sum()
    if reduction == 'mean_batch':
        return losses.mean()
    if reduction == 'sum':
        return losses.sum()
    if reduction == 'none':
        return losses
    raise ValueError(f'Unsupported reduction {reduction}')


def reduced_joint_probs(joint, encoder_projected, decoder_projected, labels,
                        vocab_size, blank_id, sigma=0.0):
    logits = joint(encoder_projected[:, :, None, :], decoder_projected[:, None, :, :])
    tokens = logits[..., :vocab_size].float()
    normalizer = torch.logsumexp(tokens, dim=-1)
    blank = tokens[..., blank_id] - normalizer - sigma
    indices = labels[:, None, :, None].expand(-1, tokens.shape[1], -1, -1)
    label = torch.gather(tokens[:, :, :-1, :], -1, indices).squeeze(-1)
    label = label - normalizer[:, :, :-1] - sigma
    duration = logits[..., vocab_size:].float().log_softmax(-1)
    return blank, label, duration


def chunked_tdt_loss(joint, encoder_projected, decoder_projected, labels,
                     vocab_size, blank_id, durations, logit_lengths=None,
                     target_lengths=None, chunk_size=16, sigma=0.0,
                     reduction='mean', recompute=True):
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    if labels.ndim != 2 or labels.shape[-1] == 0:
        raise ValueError('TDT training requires a nonempty target sequence')
    if encoder_projected.ndim != 3 or decoder_projected.ndim != 3:
        raise ValueError('Projected encoder and decoder tensors must have shape (batch, sequence, hidden)')
    if decoder_projected.shape[1] != labels.shape[1] + 1:
        raise ValueError('The decoder sequence must include one initial blank before the targets')
    if not (labels.shape[0] == encoder_projected.shape[0] == decoder_projected.shape[0]):
        raise ValueError('Batch sizes must match')
    if target_lengths is not None and bool((target_lengths <= 0).any()):
        raise ValueError('TDT training requires positive target lengths')
    if labels.dtype != torch.long:
        labels = labels.long()
    parts = []
    def project(ep, dp, target):
        return reduced_joint_probs(joint, ep, dp, target, vocab_size, blank_id, sigma)
    for start in range(0, encoder_projected.shape[1], chunk_size):
        ep = encoder_projected[:, start:start + chunk_size]
        if recompute and torch.is_grad_enabled():
            values = checkpoint(project, ep, decoder_projected, labels, use_reentrant=False)
        else:
            values = project(ep, decoder_projected, labels)
        parts.append(values)
    blank, label, duration = [torch.cat([part[index] for part in parts], dim=1) for index in range(3)]
    batch = encoder_projected.shape[0]
    if logit_lengths is None:
        logit_lengths = torch.full((batch,), encoder_projected.shape[1], dtype=torch.long, device=labels.device)
    if target_lengths is None:
        target_lengths = torch.full((batch,), labels.shape[1], dtype=torch.long, device=labels.device)
    return tdt_from_log_probs(blank, label, duration, logit_lengths, target_lengths, durations, reduction)


def cpu_parity_check():
    """Compare actual losses and every gradient against the pinned full loss."""
    import copy
    import json
    from pathlib import Path
    from torch import nn
    from transformers.loss.loss_tdt import tdt_loss
    class Joint(nn.Module):
        def __init__(self, hidden, output):
            super().__init__()
            self.head = nn.Linear(hidden, output)
        def forward(self, ep, dp):
            return self.head(torch.relu(ep + dp))
    torch.set_num_threads(2)
    cases = []
    for seed, batch, frames, targets, vocab, chunk in ((7, 1, 9, 4, 19, 2), (8, 2, 13, 7, 31, 5), (9, 1, 3, 1, 7, 1)):
        torch.manual_seed(seed)
        durations = [0, 1, 2, 3, 4]
        left = Joint(11, vocab + len(durations))
        right = copy.deepcopy(left)
        enc = torch.randn(batch, frames, 11, requires_grad=True)
        dec = torch.randn(batch, targets + 1, 11, requires_grad=True)
        enc2 = enc.detach().clone().requires_grad_()
        dec2 = dec.detach().clone().requires_grad_()
        labels = torch.randint(0, vocab - 1, (batch, targets))
        lengths = torch.full((batch,), frames, dtype=torch.long)
        target_lengths = torch.full((batch,), targets, dtype=torch.long)
        if batch > 1:
            lengths[-1] -= 2
            target_lengths[-1] -= 1
        logits = left(enc[:, :, None], dec[:, None])
        original = tdt_loss(logits[..., :vocab], logits[..., vocab:], labels, lengths, target_lengths,
                            vocab - 1, durations, sigma=0.1)
        optimized = chunked_tdt_loss(right, enc2, dec2, labels, vocab, vocab - 1, durations,
                                     lengths, target_lengths, chunk_size=chunk, sigma=0.1)
        original.backward()
        optimized.backward()
        pairs = [(enc.grad, enc2.grad), (dec.grad, dec2.grad)]
        pairs += [(p.grad, q.grad) for p, q in zip(left.parameters(), right.parameters())]
        finite = all(torch.isfinite(p).all() and torch.isfinite(q).all() for p, q in pairs)
        loss_delta = abs(float(original.detach()) - float(optimized.detach()))
        grad_delta = max(float((p - q).abs().max()) for p, q in pairs)
        assert finite and loss_delta < 3e-6 and grad_delta < 3e-6, (finite, loss_delta, grad_delta)
        cases.append({'seed': seed, 'shape': [batch, frames, targets + 1, vocab],
                      'chunk_size': chunk, 'original_loss': float(original.detach()),
                      'optimized_loss': float(optimized.detach()), 'absolute_loss_difference': loss_delta,
                      'maximum_absolute_gradient_difference': grad_delta, 'all_gradients_finite': finite})
    report = {'passed': True, 'reference': 'Transformers tdt_loss at 536ecc007387a50e77603bb5d92100e9b07514cc',
              'scope': 'CPU FP32 forward/backward parity, padded and unpadded cases; GPU memory not yet measured',
              'cases': cases}
    Path(__file__).with_name('efficient_tdt_cpu_parity.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    cpu_parity_check()
