"""Independent CPU parity audit; never uses CUDA or edits training helpers."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
from transformers.loss.loss_tdt import tdt_loss

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/parakeet_turkish'
sys.path.insert(0, str(OUTPUT))
from efficient_tdt import chunked_tdt_loss

torch.set_num_threads(2)


class Joint(nn.Module):
    def __init__(self, hidden, output):
        super().__init__()
        self.head = nn.Linear(hidden, output)

    def forward(self, ep, dp):
        return self.head(torch.relu(ep + dp))


def differences(pairs):
    flat_left = torch.cat([left.float().flatten() for left, _ in pairs])
    flat_right = torch.cat([right.float().flatten() for _, right in pairs])
    return {'all_gradients_finite': bool(torch.isfinite(flat_left).all() and torch.isfinite(flat_right).all()),
        'maximum_absolute_gradient_difference': float((flat_left-flat_right).abs().max()),
        'relative_gradient_l2_difference': float(torch.linalg.vector_norm(flat_left-flat_right) /
            torch.linalg.vector_norm(flat_left).clamp_min(1e-20)),
        'gradient_cosine_similarity': float(torch.nn.functional.cosine_similarity(flat_left, flat_right, dim=0))}


def run_case(name, batch=2, frames=47, targets=19, vocab=53, chunk=7, sigma=0.0,
             durations=(0,1,2,3,4), reduction='mean', recompute=True, bf16=False,
             input_grads=True, frozen_joint=False):
    torch.manual_seed(21417)
    left = Joint(17, vocab + len(durations))
    if frozen_joint:
        left.requires_grad_(False)
    right = copy.deepcopy(left)
    enc = torch.randn(batch, frames, 17, requires_grad=input_grads)
    dec = torch.randn(batch, targets + 1, 17, requires_grad=input_grads)
    enc2 = enc.detach().clone().requires_grad_(input_grads)
    dec2 = dec.detach().clone().requires_grad_(input_grads)
    labels = torch.randint(0, vocab-1, (batch, targets))
    lengths = torch.full((batch,), frames, dtype=torch.long)
    target_lengths = torch.full((batch,), targets, dtype=torch.long)
    if batch > 1:
        lengths[-1] = max(1, frames-11)
        target_lengths[-1] = max(1, targets-8)
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
        logits = left(enc[:,:,None], dec[:,None])
        original = tdt_loss(logits[...,:vocab], logits[...,vocab:], labels, lengths,
            target_lengths, vocab-1, list(durations), sigma=sigma, reduction=reduction)
        optimized = chunked_tdt_loss(right, enc2, dec2, labels, vocab, vocab-1, list(durations),
            lengths, target_lengths, chunk_size=chunk, sigma=sigma, reduction=reduction, recompute=recompute)
    # None reduction must support arbitrary upstream per-example weights.
    weights = torch.arange(1, batch+1) if reduction == 'none' else 1
    (original * weights).sum().backward()
    (optimized * weights).sum().backward()
    pairs = []
    if input_grads:
        pairs.extend([(enc.grad, enc2.grad), (dec.grad, dec2.grad)])
    if not frozen_joint:
        pairs.extend((p.grad, q.grad) for p,q in zip(left.parameters(), right.parameters()))
    result = dict(name=name, batch=batch, frames=frames, targets=targets, vocab=vocab,
        chunk=chunk, sigma=sigma, durations=list(durations), reduction=reduction,
        recompute=recompute, dtype='CPU BF16 autocast/FP32 probabilities' if bf16 else 'FP32',
        absolute_loss_difference=float((original.detach()-optimized.detach()).abs().max()), **differences(pairs))
    tolerance = 2e-4 if not bf16 else 0.01
    result['passed'] = result['all_gradients_finite'] and result['absolute_loss_difference'] < 3e-4 and result['relative_gradient_l2_difference'] < tolerance
    assert result['passed'], result
    return result


def teacher_kl_case():
    """Sparse student projection preserves KL and gradients, including new rows."""
    torch.manual_seed(661)
    vocab, old_vocab, frames, targets, hidden = 37, 31, 41, 14, 13
    durations = [0,1,2,3,4]
    left = Joint(hidden, vocab+len(durations))
    right = copy.deepcopy(left)
    teacher = Joint(hidden, old_vocab+len(durations)).eval().requires_grad_(False)
    enc = torch.randn(1,frames,hidden,requires_grad=True)
    dec = torch.randn(1,targets+1,hidden,requires_grad=True)
    enc2 = enc.detach().clone().requires_grad_()
    dec2 = dec.detach().clone().requires_grad_()
    labels = torch.randint(0,old_vocab-1,(1,targets))
    lengths=torch.tensor([frames]); target_lengths=torch.tensor([targets])
    ts=torch.linspace(0,frames-1,8).long(); us=torch.linspace(0,targets,8).long()
    with torch.no_grad():
        target=teacher(enc.detach()[:,ts,None], dec.detach()[:,None,us])[...,:old_vocab].softmax(-1)
    full=left(enc[:,:,None],dec[:,None])
    student=full[:,ts][:,:,us,:vocab].log_softmax(-1)
    kl=(target*(target.clamp_min(1e-12).log()-student[...,:old_vocab])).sum(-1).mean()
    coefficient=5.0
    original=tdt_loss(full[...,:vocab], full[...,vocab:], labels, lengths, target_lengths, vocab-1,durations) + coefficient*kl
    sparse=right(enc2[:,ts,None],dec2[:,None,us])[...,:vocab].log_softmax(-1)
    sparse_kl=(target*(target.clamp_min(1e-12).log()-sparse[...,:old_vocab])).sum(-1).mean()
    optimized=chunked_tdt_loss(right,enc2,dec2,labels,vocab,vocab-1,durations,lengths,target_lengths,chunk_size=6)+coefficient*sparse_kl
    original.backward(); optimized.backward()
    pairs=[(enc.grad,enc2.grad),(dec.grad,dec2.grad)] + [(p.grad,q.grad) for p,q in zip(left.parameters(),right.parameters())]
    report=dict(name='sparse_student_replay_KL_plus_checkpointed_TDT', replay_kl_coefficient=coefficient,
        absolute_loss_difference=abs(float(original.detach()-optimized.detach())),
        kl_absolute_difference=abs(float(kl.detach()-sparse_kl.detach())),
        new_token_head_gradient_nonzero=bool(right.head.weight.grad[old_vocab:vocab].abs().sum()>0), **differences(pairs))
    assert report['all_gradients_finite'] and report['maximum_absolute_gradient_difference'] < 1e-5
    assert report['new_token_head_gradient_nonzero'] and report['kl_absolute_difference'] < 1e-6
    report['passed']=True
    return report


def edge_cases():
    """Expose invalid inputs rather than silently claiming support."""
    torch.manual_seed(11)
    joint=Joint(7,16)
    enc=torch.randn(1,8,7,requires_grad=True)
    dec=torch.randn(1,1,7,requires_grad=True)
    try:
        chunked_tdt_loss(joint,enc,dec,torch.empty(1,0,dtype=torch.long),11,10,[0,1,2,3,4])
        raise AssertionError('Empty targets must be rejected before projection')
    except ValueError as exc:
        report=[{'name':'empty_labels_with_mean_reduction', 'behavior':str(exc),
            'expected_supported':False, 'rejected':True}]
    try:
        chunked_tdt_loss(joint,enc,torch.randn(1,3,7),torch.tensor([[1,2]]),11,10,[0,1,2,3,4],target_lengths=torch.tensor([0]))
        raise AssertionError('Zero effective target length must be rejected')
    except ValueError as exc:
        report.append({'name':'zero_target_lengths', 'behavior':str(exc), 'expected_supported':False, 'rejected':True})
    try:
        chunked_tdt_loss(joint,enc,dec,torch.empty(1,0,dtype=torch.long),11,10,[0,1,2,3,4],chunk_size=0)
        raise AssertionError('Invalid chunk accepted')
    except ValueError as exc:
        report.append({'name':'zero_chunk', 'behavior':str(exc), 'expected_supported':False, 'rejected':True})
    # The pinned upstream DP also has undefined gradients for unreachable states
    # when duration 1 is absent; this release's actual duration set includes 1.
    durations=[0,2,4]; model=Joint(7,14)
    labels=torch.tensor([[1,3]])
    ep=torch.randn(1,9,7,requires_grad=True); dp=torch.randn(1,3,7,requires_grad=True)
    logits=model(ep[:,:,None],dp[:,None])
    loss=tdt_loss(logits[...,:11],logits[...,11:],labels,torch.tensor([9]),torch.tensor([2]),10,durations)
    loss.backward()
    reference_finite=bool(torch.isfinite(ep.grad).all())
    model.zero_grad(set_to_none=True); ep=ep.detach().clone().requires_grad_(); dp=dp.detach().clone().requires_grad_()
    optimized=chunked_tdt_loss(model,ep,dp,labels,11,10,durations,chunk_size=3)
    optimized.backward()
    optimized_finite=bool(torch.isfinite(ep.grad).all())
    report.append({'name':'duration_set_without_one', 'durations':durations,
        'reference_finite_loss':bool(torch.isfinite(loss)), 'optimized_finite_loss':bool(torch.isfinite(optimized)),
        'reference_gradients_finite':reference_finite, 'optimized_gradients_finite':optimized_finite,
        'recommendation':'Restrict this pilot to the released durations [0,1,2,3,4]; unreachable DP states inherit upstream nonfinite-gradient behavior.'})
    return report


cases=[]
for reduction in ('mean','mean_volume','mean_batch','sum','none'):
    cases.append(run_case('long_padded_'+reduction, reduction=reduction, sigma=0.13))
cases.append(run_case('nondividing_chunks_large_duration', durations=(0,1,3,8), sigma=1.0, frames=57,targets=23,chunk=11))
cases.append(run_case('reordered_duration_ids', durations=(4,0,2,1,3), sigma=0.0, chunk=80))
cases.append(run_case('without_recomputation', recompute=False, sigma=0.25))
cases.append(run_case('checkpoint_trainable_head_frozen_inputs', input_grads=False))
cases.append(run_case('checkpoint_frozen_head_trainable_inputs', frozen_joint=True))
cases.append(run_case('bf16_checkpoint', bf16=True, sigma=0.1))
cases.append(run_case('bf16_no_recompute', bf16=True, recompute=False, sigma=0.1))
cases.append(teacher_kl_case())
report={'status':'passed_supported_cases', 'gpu_used':False,
    'source_sha256':hashlib.sha256((OUTPUT/'efficient_tdt.py').read_bytes()).hexdigest(),
    'reference':'Installed Transformers tdt_loss, pinned commit 536ecc007387a50e77603bb5d92100e9b07514cc',
    'scope':'CPU FP32 and CPU BF16 forward/backward, complete full-vocabulary normalization; actual GPU parity and peak VRAM remain unmeasured.',
    'cases':cases, 'unsupported_edge_cases':edge_cases(),
    'review_findings':[
        'Token normalization includes the complete student vocabulary, including appended rows and blank; reducing stored states does not prune the objective.',
        'Blank and selected-target token log probabilities receive sigma undernormalization; duration log probabilities use their separate normalized distribution.',
        'The antidiagonal recurrence and terminal blank arcs match the pinned upstream implementation. All five target-length reduction rules match.',
        'Nonreentrant checkpoints propagate gradients even when checkpoint inputs are frozen but joint parameters are trainable, and when joint parameters are frozen but encoder inputs are trainable.',
        'BF16 chunked accumulation gives small rounding differences from full GEMM; forward loss is not bitwise identical and GPU precision must be checked separately.',
        'Replay KL must project sampled student states directly rather than materialize the original full vocabulary lattice; student normalization must still include new rows.',
        'For encoder LoRA, freeze the original teacher features/cache. Feeding teacher heads adapted student encoder features would change the preservation target.',
        'Do not update parameters or change module state between checkpointed forward and backward. Keep encoder eval mode to freeze BatchNorm statistics and disable dropout/LayerDrop.',
        'The dynamic-program autograd graph is unchanged and still costs memory/launches; actual GPU peak allocated/reserved memory and timing cannot be inferred from this CPU parity audit.'
    ]}
(OUTPUT/'efficient_tdt_independent_cpu_review.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(report,indent=2))
