"""CPU scheduling/checkpoint/gradient tests, not a native CUDA execution claim."""
import copy
import hashlib
import json
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/parakeet_turkish'
sys.path.insert(0, str(OUT))
import torch
from torch import nn
import native_tdt_checkpointed_v1 as candidate
import verify_nemo_native_loss_v3 as oracle

torch.set_num_threads(2)
torch.manual_seed(2211)
Reference, edits = oracle.reference_class('TDTLossPytorch')
checks, cases = 0, []


def check(value, message):
    global checks
    checks += 1
    if not value:
        raise AssertionError(message)


class Joint(nn.Module):
    def __init__(self, hidden, width):
        super().__init__()
        self.linear = nn.Linear(hidden, width)

    def forward(self, ep, dp):
        return self.linear(torch.tanh(ep + dp))


for batch, t, u in ((1, 5, 3), (2, 6, 4), (4, 7, 5)):
    frames = [t - (i % 3) for i in range(batch)]
    targets = [u - (i % 3) for i in range(batch)]
    vocab, blank, durations = 11, 3, (0, 1, 2, 3, 4)
    labels = torch.randint(0, vocab - 1, (batch, u))
    labels[labels >= blank] += 1
    for i, size in enumerate(targets):
        labels[i, size:] = blank
    ep, dp = torch.randn(batch, t, 5), torch.randn(batch, u + 1, 5)
    for reduction in ('none', 'mean', 'mean_volume'):
        ref_joint, new_joint = Joint(5, vocab + len(durations)), Joint(5, vocab + len(durations))
        new_joint.load_state_dict(ref_joint.state_dict())
        a, b = ep.clone().requires_grad_(), dp.clone().requires_grad_()
        x, y = ep.clone().requires_grad_(), dp.clone().requires_grad_()
        reference = Reference(blank=blank, durations=list(durations), reduction=reduction, sigma=0.0)
        expected = reference(ref_joint(a[:, :, None], b[:, None]), labels,
                             torch.tensor(frames), torch.tensor(targets))
        per_clip = Reference(blank=blank, durations=list(durations), reduction='none', sigma=0.0)

        def backend(acts, target, tl, ul):
            random.uniform(0, 1)  # mirror official omega=0's otherwise-unused draw
            return per_clip(acts, target, tl, ul)

        state = random.getstate()
        actual = candidate._loss_with_backend(new_joint, x, y, labels, vocab, blank, durations,
                       frames, targets, reduction, True, 10_000_000, backend)
        check(random.getstate() == state, 'Python RNG changed in native scheduling forward')
        check(actual.shape == expected.shape, 'Raw/reduced shape mismatch')
        check(torch.allclose(actual, expected, atol=8e-6, rtol=2e-6), 'Per-clip vs padded official raw loss differs')
        expected.sum().backward()
        actual.sum().backward()
        check(random.getstate() == state, 'Python RNG changed during recomputation')
        maximum = 0.0
        for left, right in [(a.grad, x.grad), (b.grad, y.grad)] + [
                (p.grad, q.grad) for p, q in zip(ref_joint.parameters(), new_joint.parameters())]:
            maximum = max(maximum, float((left - right).abs().max()))
            check(torch.isfinite(right).all(), 'Nonfinite checkpoint gradient')
            check(torch.allclose(left, right, atol=7e-6, rtol=2e-5), 'Per-clip checkpoint gradient differs')
        for i, (ft, ut) in enumerate(zip(frames, targets)):
            check(torch.count_nonzero(x.grad[i, ft:]) == 0 and
                  torch.count_nonzero(y.grad[i, ut + 1:]) == 0, 'Padded state received gradient')
        cases.append({'B': batch, 'T': frames, 'U': targets, 'reduction': reduction,
                      'raw_loss_max_absolute_delta': float((actual - expected).abs().max()),
                      'maximum_gradient_absolute_delta': maximum})

rejections = []
base = dict(ep=ep, dp=dp, labels=labels, vocab_size=vocab, blank_id=blank,
            durations=durations, logit_lengths=frames, target_lengths=targets,
            max_joint_bytes=10_000_000)
for name, changed in (('empty_targets', {'target_lengths': [0] * batch}),
                      ('joint_byte_bound', {'max_joint_bytes': 1}),
                      ('unsupported_duration', {'durations': (0, 2)}),
                      ('invalid_lexical_blank', {'labels': torch.full_like(labels, blank)})):
    try:
        candidate._validated_plan(**dict(base, **changed))
    except ValueError:
        checks += 1
        rejections.append(name)
    else:
        raise AssertionError('Rejected guard not enforced: ' + name)
check(not torch.cuda.is_initialized(), 'CPU guards initialized CUDA')
report = {'schema': 1, 'passed': True, 'checks': checks, 'cases': cases, 'rejections': rejections,
          'adapter_sha256': candidate.sha(candidate.__file__), 'script_sha256': candidate.sha(__file__),
          'official_reference_path': str(oracle.SITE / oracle.PYTORCH),
          'official_reference_sha256': oracle.sha(oracle.SITE / oracle.PYTORCH),
          'reference_allocation_edits': edits, 'CUDA_executed': False,
          'scope': 'Exact per-clip/unpadding/reduction/closure/recompute/full joint gradients vs official CPU reference; actual native checkpoint GPU parity/throughput pending'}
path = OUT / 'native_tdt_checkpointed_cpu_verification_v1.json'
check(not path.exists(), 'Preserve CPU verification report')
path.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({'passed': True, 'checks': checks, 'cases': len(cases), 'CUDA_executed': False}))
