"""CPU-only comparison against a captured NVIDIA NeMo PyTorch TDT recurrence.

This is not a NeMo installation or native Numba-kernel test. The official class
is extracted from a SHA-bound source snapshot and exactly two CUDA allocation
calls are changed to the input device. NeMo metadata/base classes are replaced
with a torch Module; no recurrence, normalization or reduction is replaced.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
from transformers.loss.loss_tdt import tdt_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
import efficient_tdt_v2

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'work/research/nemo_reference/rnnt_pytorch_web_main.py'
URL = 'https://raw.githubusercontent.com/NVIDIA-NeMo/Speech/main/nemo/collections/asr/losses/rnnt_pytorch.py'


def reference_class():
    source = SOURCE.read_text(encoding='utf-8')
    tree = ast.parse(source)
    selected = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'TDTLossPytorch')
    # Allocation-only portability patch. Never replace a computational node.
    edits = []
    class DeviceAllocation(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'cuda':
                if node.args or node.keywords:
                    raise ValueError('Unexpected CUDA allocation signature')
                edits.append({'line': node.lineno, 'before': ast.unparse(node),
                              'after': ast.unparse(node.func.value) + '.to(acts.device)'})
                node.func.attr = 'to'
                node.args = [ast.Attribute(ast.Name(id='acts', ctx=ast.Load()), attr='device', ctx=ast.Load())]
            return node
    selected = DeviceAllocation().visit(selected)
    if len(edits) != 2:
        raise ValueError(f'Official source changed: expected two allocation edits, found {edits}')
    module = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
    scope = {'torch': torch, 'Loss': nn.Module, 'List': list}
    exec(compile(module, str(SOURCE), 'exec'), scope)
    return scope['TDTLossPytorch'], edits, hashlib.sha256(source.encode()).hexdigest()


class Joint(nn.Module):
    def __init__(self, vocab, duration_count):
        super().__init__()
        self.head = nn.Linear(11, vocab + duration_count)
    def forward(self, ep, dp):
        return self.head(torch.relu(ep + dp))


def compare_gradients(reference, candidate):
    left = torch.cat([p.flatten().double() for p in reference])
    right = torch.cat([p.flatten().double() for p in candidate])
    return {'finite': bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
            'maximum_absolute_difference': float((left - right).abs().max()),
            'relative_l2_difference': float((left - right).norm() / left.norm().clamp_min(1e-30)),
            'cosine': float(nn.functional.cosine_similarity(left, right, dim=0))}


def case(reference_type, name, batch, frames, targets, vocab, blank, sigma, reduction, durations):
    torch.manual_seed(961131 + batch)
    original = Joint(vocab, len(durations))
    ep_base = torch.randn(batch, frames, 11)
    dp_base = torch.randn(batch, targets + 1, 11)
    ts = torch.tensor([max(1, frames - i) for i in range(batch)])
    us = torch.tensor([max(1, targets - i) for i in range(batch)])
    labels = torch.randint(vocab - 1, (batch, targets))
    labels += (labels >= blank).long()
    if blank != vocab - 1:
        labels[:, 0] = vocab - 1
    for i in range(batch):
        labels[i, us[i]:] = blank
    raw_losses, losses, grads, logit_grads = [], [], [], []
    upstream = torch.arange(1, batch + 1).float() / 3 if reduction == 'none' else torch.tensor(0.73)
    for kind in ('nemo_cpu_device_adapted', 'transformers', 'custom_chunked_v2'):
        joint = copy.deepcopy(original)
        ep, dp = ep_base.clone().requires_grad_(), dp_base.clone().requires_grad_()
        logits = joint(ep[:, :, None], dp[:, None]) if kind != 'custom_chunked_v2' else None
        if logits is not None:
            logits.retain_grad()
        if kind == 'nemo_cpu_device_adapted':
            raw = reference_type(blank=blank, durations=list(durations), reduction=None, sigma=sigma)(
                logits, labels, ts, us)
            loss = reference_type(blank=blank, durations=list(durations), reduction=reduction, sigma=sigma)(
                logits, labels, ts, us)
        elif kind == 'transformers':
            raw = tdt_loss(logits[..., :vocab], logits[..., vocab:], labels, ts, us,
                           blank, list(durations), sigma=sigma, reduction='none')
            loss = tdt_loss(logits[..., :vocab], logits[..., vocab:], labels, ts, us,
                            blank, list(durations), sigma=sigma, reduction=reduction)
        else:
            raw = efficient_tdt_v2.chunked_tdt_loss(joint, ep, dp, labels, vocab, blank, list(durations),
                logit_lengths=ts, target_lengths=us, sigma=sigma, reduction='none', chunk_size=3)
            loss = efficient_tdt_v2.chunked_tdt_loss(joint, ep, dp, labels, vocab, blank, list(durations),
                logit_lengths=ts, target_lengths=us, sigma=sigma, reduction=reduction, chunk_size=3)
        # NeMo treats arbitrary non-reducer values (e.g. none) as unreduced.
        (loss * upstream).sum().backward()
        raw_losses.append(raw.detach())
        losses.append(loss.detach())
        grads.append([ep.grad, dp.grad, *[p.grad for p in joint.parameters()]])
        logit_grads.append(logits.grad if logits is not None else None)
    comparisons = []
    for i, kind in enumerate(('transformers', 'custom_chunked_v2'), 1):
        grad = compare_gradients(grads[0], grads[i])
        raw_delta = float((raw_losses[0] - raw_losses[i]).abs().max())
        delta = float((losses[0] - losses[i]).abs().max())
        scale = max(1.0, float(raw_losses[0].abs().max()))
        passed = grad['finite'] and raw_delta <= 3e-6 * scale and delta <= 3e-6 * scale
        passed &= grad['relative_l2_difference'] < 1e-4 and grad['cosine'] > 0.999999
        item = {'candidate': kind, 'raw_nll_max_absolute_difference': raw_delta,
                'reduced_loss_max_absolute_difference': delta, 'gradient': grad, 'passed': passed}
        if logit_grads[i] is not None:
            item['joint_logit_gradient'] = compare_gradients([logit_grads[0]], [logit_grads[i]])
        comparisons.append(item)
    return {'name': name, 'batch': batch, 'frames': frames, 'target_capacity': targets,
            'frame_lengths': ts.tolist(), 'target_lengths': us.tolist(),
            'token_vocabulary_including_blank': vocab, 'blank_id': blank,
            'durations': list(durations), 'sigma': sigma, 'reduction': reduction,
            'raw_nll': raw_losses[0].tolist(), 'comparison': comparisons,
            'passed': all(item['passed'] for item in comparisons)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('nemo_tdt_reference_cpu_verification_v1.json'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU-only audit must not initialize CUDA')
    cls, edits, source_sha = reference_class()
    report = {'schema_version': 1, 'scope': 'Captured official NeMo PyTorch recurrence with two device-allocation edits; synthetic CPU FP32, not native NeMo/Numba or end-to-end model parity',
              'official_source_url': URL, 'official_source_snapshot_path': str(SOURCE),
              'official_source_snapshot_sha256': source_sha, 'allocation_only_edits': edits,
              'unsupported_reference_features': ['Numba CUDA backend', 'omega RNNT/TDT stochastic mixing', 'FastEmit', 'native NeMo metadata/typecheck integration'],
              'cases': []}
    specifications = [
        ('B1_base_none', 1, 8, 4, 41, 40, 0., 'none', (0,1,2,3,4)),
        ('B2_padded_none', 2, 9, 4, 41, 40, 0., 'none', (0,1,2,3,4)),
        ('B4_padded_mean', 4, 10, 5, 41, 40, 0., 'mean', (0,1,2,3,4)),
        ('B2_sigma002_mean_volume', 2, 7, 3, 41, 40, .02, 'mean_volume', (0,1,2,3,4)),
        ('B2_sigma01_sum', 2, 8, 3, 41, 40, .1, 'sum', (0,1,2,3,4)),
        ('B2_sigma002_mean_batch', 2, 7, 3, 41, 40, .02, 'mean_batch', (0,1,2,3,4)),
        ('B2_nonlast_blank_appended_TR', 2, 6, 3, 8199, 8192, 0., 'none', (0,1,2,3,4)),
        ('B2_reordered_durations', 2, 7, 3, 41, 40, .02, 'mean', (4,0,2,1,3)),
        ('B1_large_duration', 1, 7, 3, 41, 40, 0., 'none', (0,1,3,8)),
    ]
    for values in specifications:
        report['cases'].append(case(cls, *values))
    report['passed'] = all(item['passed'] for item in report['cases'])
    report['cuda_initialized'] = torch.cuda.is_initialized()
    report['code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['protected_source_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('efficient_tdt.py', 'efficient_tdt_v2.py', 'runtime.py', 'batched_lora.py')}
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': report['passed'], 'cases': len(report['cases']), 'cuda_initialized': report['cuda_initialized']}))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
