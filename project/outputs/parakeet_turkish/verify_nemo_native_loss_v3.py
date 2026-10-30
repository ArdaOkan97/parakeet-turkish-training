"""Bounded native NeMo loss oracle, intended for a parent-owned solo GPU window.

--cpu-source-only audits installed reference source without GPU initialization.
The GPU path consumes saved ACTUAL joint logits, never loads/trains an encoder.
It checks omega0 TDT versus two FP32 CPU references and omega1 RNNT versus the
native PyTorch RNNT reference. omega.1 is sampled, not an arithmetic mixture.

CPU reference gradients contract all unused token classes into one logsumexp
OTHER logit. This preserves every blank/target probability mathematically and
backpropagates to the original full logits. The native CUDA input is unchanged.
Actual full-width official reference losses are independently checked without
autograd, avoiding full-width SelectBackward tensors at each recurrence arc.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SITE = ROOT/'work/nemo-native-env-v1/Lib/site-packages'
PIN = ROOT/'work/research/nemo_reference/pinned_e4a34fd25f89'
PYTORCH = Path('nemo/collections/asr/losses/rnnt_pytorch.py')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024), b''):value.update(block)
    return value.hexdigest()


def node(path, name):
    return next(x for x in ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(x, ast.ClassDef) and x.name == name)


def source_report():
    paths = {'installed_pytorch': SITE/PYTORCH, 'pinned_pytorch': PIN/PYTORCH,
             'installed_resolver': SITE/'nemo/collections/asr/losses/rnnt.py',
             'installed_numba_wrapper': SITE/'nemo/collections/asr/parts/numba/rnnt_loss/rnnt_pytorch.py',
             'installed_numba_gpu': SITE/'nemo/collections/asr/parts/numba/rnnt_loss/utils/cuda_utils/gpu_rnnt.py',
             'installed_numba_gpu_kernels': SITE/'nemo/collections/asr/parts/numba/rnnt_loss/utils/cuda_utils/gpu_rnnt_kernel.py'}
    ast_equal = {}
    for name in ('TDTLossPytorch', 'RNNTLossPytorch'):
        ast_equal[name] = ast.dump(node(paths['installed_pytorch'], name), include_attributes=False) == ast.dump(node(paths['pinned_pytorch'], name), include_attributes=False)
    return {'installed_versions': {name:importlib.metadata.version(name) for name in
                                  ('nemo-toolkit', 'torch', 'pytorch-lightning', 'lightning')},
            'sources': {k:{'path':str(v.resolve()), 'sha256':sha(v)} for k,v in paths.items()},
            'reference_class_AST_equal_installed_vs_pin': ast_equal,
            'omega_semantics': 'Python random.uniform once per native TDT loss invocation; RNNT branch probability omega; RNNT branch does not use sigma and yields zero duration gradient',
            'whole_model_scope': 'No encoder/tokenizer/decoder/optimizer equivalence claim'}


def reference_class(name):
    selected = node(SITE/PYTORCH, name)
    edits = []
    class DeviceAllocation(ast.NodeTransformer):
        def visit_Call(self, value):
            self.generic_visit(value)
            if isinstance(value.func, ast.Attribute) and value.func.attr == 'cuda':
                require(not value.args and not value.keywords, 'Unexpected CUDA allocation')
                edits.append({'line':value.lineno, 'before':ast.unparse(value)})
                value.func.attr = 'to'
                value.args = [ast.Attribute(ast.Name(id='acts', ctx=ast.Load()), attr='device', ctx=ast.Load())]
            return value
    selected = DeviceAllocation().visit(selected)
    require(len(edits) == (2 if name == 'TDTLossPytorch' else 0), 'Reference allocation edits differ')
    scope = {'torch':torch, 'Loss':nn.Module, 'List':list}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[])), str(SITE/PYTORCH), 'exec'), scope)
    return scope[name], edits


def grad_stats(left, right):
    # Stream float64 reductions rather than allocate two full float64 gradients.
    left, right = left.flatten(), right.flatten()
    require(left.shape == right.shape, 'Gradient shapes differ')
    finite = True
    norm_left = norm_right = delta2 = dot = maximum = 0.
    for start in range(0,left.numel(),1_000_000):
        a,b=left[start:start+1_000_000].double(),right[start:start+1_000_000].double()
        finite &= bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        difference=a-b
        norm_left+=float(a.square().sum());norm_right+=float(b.square().sum())
        delta2+=float(difference.square().sum());dot+=float((a*b).sum())
        maximum=max(maximum,float(difference.abs().max()))
    return {'finite':finite,'maximum_absolute_delta':maximum,
            'relative_l2':math.sqrt(delta2/max(norm_left,1e-60)),
            'cosine':dot/max(math.sqrt(norm_left*norm_right),1e-60),
            'reduction':'float64 streamed1M values; no full float64 gradient copies'}


def validate_input(data):
    require(data.get('schema') == 1 and data.get('kind') == 'actual_fleurs_training_joint_logits', 'Actual fixed TRAIN-logit input required; no synthetic claim')
    acts, labels = data['joint_logits'], data['labels']
    lengths, targets = data['frame_lengths'], data['target_lengths']
    durations, blank = data['durations'], data['blank_id']
    require(acts.device.type == 'cpu' and acts.dtype == torch.float32 and acts.ndim == 4 and acts.is_contiguous(), 'Saved joint logits must be contiguous CPU FP32')
    require(acts.numel() <= 80_000_000 and bool(torch.isfinite(acts).all()), 'Bounded finite logits <=320MB required')
    batch, frames, states, width = acts.shape
    require(1 <= batch <= 4 and 1 <= frames <= 128 and 2 <= states <= 81 and durations == [0,1,2,3,4], 'Bounded actual T/U/B and released durations required')
    vocab = width-len(durations)
    require(vocab == 8199 and blank == 8192, 'Expected exact extended current vocabulary and retained blank')
    require(labels.shape == (batch,states-1) and labels.dtype == torch.long and labels.device.type == 'cpu', 'Labels shape/type differ')
    require(lengths.shape == targets.shape == (batch,) and lengths.dtype == targets.dtype == torch.long, 'Lengths shape/type differ')
    require(int(lengths.max()) == frames and int(targets.max())+1 == states and int(lengths.min()) >= 1 and int(targets.min()) >= 1, 'Native inputs require exact maximum T/U')
    for i in range(batch):
        valid = labels[i,:int(targets[i])]
        require(bool(((0 <= valid)&(valid < vocab)&(valid != blank)).all()), 'Invalid positive labels')
    provenance = data['provenance']
    require(provenance.get('split') == 'train' and len(provenance['source_ids']) == batch, 'No DEV/TEST or fabricated logit evidence')
    require(len(set(provenance['source_ids'])) == batch, 'Unique fixed TRAIN recordings required')
    require(provenance.get('model_training') is False and provenance.get('dropout_disabled') is True, 'Deterministic eval-captured logits required')
    for key in ('checkpoint_sha256','tokenizer_sha256'):
        require(isinstance(provenance.get(key),str) and len(provenance[key]) == 64, 'Source checkpoint/tokenizer hash missing')
    return acts, labels, lengths, targets, durations, blank, vocab


def custom_tdt(acts, labels, lengths, targets, durations, blank, vocab, sigma):
    import efficient_tdt_v2
    token = nn.functional.log_softmax(acts[...,:vocab], dim=-1)-sigma
    duration = nn.functional.log_softmax(acts[...,vocab:], dim=-1)
    selected = token[...,:-1,:].gather(-1, labels[:,None,:,None].expand(-1,acts.shape[1],-1,1)).squeeze(-1)
    return efficient_tdt_v2.tdt_from_log_probs(token[...,blank], selected, duration, lengths, targets, durations, reduction='none')


def compress_reference_logits(acts, labels, blank, vocab):
    """Exact categorical contraction for reference math, never native targets.

    The official recurrence only visits the blank and the supplied target
    classes. Keeping those logits and replacing all other classes by their
    logsumexp preserves the partition function and the entire loss derivative.
    Indexing is differentiable, including OTHER's full-class logsumexp gradient.
    All padded labels are retained too; no source target is cropped or changed.
    """
    require(acts.ndim == 4 and 0 <= blank < vocab <= acts.shape[-1], 'Reference compression dimensions')
    kept = sorted(set(labels.reshape(-1).tolist()) | {blank})
    require(all(isinstance(value,int) and 0 <= value < vocab for value in kept), 'Reference class index invalid')
    remaining = sorted(set(range(vocab))-set(kept))
    keep_index = torch.tensor(kept, device=acts.device, dtype=torch.long)
    token = acts[...,:vocab]
    selected = token.index_select(-1, keep_index)
    if remaining:
        other_index = torch.tensor(remaining, device=acts.device, dtype=torch.long)
        other = token.index_select(-1, other_index).logsumexp(-1, keepdim=True)
        selected = torch.cat((selected,other), dim=-1)
    reduced = torch.cat((selected,acts[...,vocab:]), dim=-1)
    mapping = torch.full((vocab,), -1, device=labels.device, dtype=torch.long)
    mapping[keep_index.to(labels.device)] = torch.arange(len(kept), device=labels.device)
    remapped = mapping[labels]
    require(bool((remapped >= 0).all()), 'A reference label was omitted')
    compressed_blank = kept.index(blank)
    metadata = {'kept_original_token_ids':kept, 'other_original_token_class_count':len(remaining),
        'original_token_classes':vocab, 'compressed_token_classes':selected.shape[-1],
        'compressed_blank_id':compressed_blank,
        'scope':'Reference-only exact categorical contraction; actual native logits and targets remain original'}
    return reduced, remapped, compressed_blank, selected.shape[-1], metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-source-only', action='store_true')
    parser.add_argument('--gpu-window-authorized', action='store_true')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--input-sha256')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and args.output.resolve().is_relative_to(ROOT), 'New workspace output required')
    torch.set_num_threads(2)
    report = {'schema':1, 'kind':'native_nemo_loss_oracle', 'source':source_report(), 'code_sha256':sha(__file__)}
    require(importlib.metadata.version('nemo-toolkit') == '3.0.0', 'Expected isolated native NeMo3.0')
    if args.cpu_source_only:
        require(not torch.cuda.is_initialized(), 'CPU-only audit initialized CUDA')
        report.update(passed_source_checks=True, cuda_initialized=False, native_kernel_executed=False,
                      scope='Read-only installed-vs-pinned source; no kernel/full model proof')
    else:
        require(args.gpu_window_authorized and args.input and args.input_sha256, 'Parent-owned solo GPU window and hash-bound input required')
        require(sha(args.input) == args.input_sha256 and args.input.resolve().is_relative_to(ROOT), 'Input evidence hash/path differs')
        data = torch.load(args.input, map_location='cpu', weights_only=True)
        acts, labels, lengths, targets, durations, blank, vocab = validate_input(data)
        report.update(input_path=str(args.input.resolve()), input_sha256=args.input_sha256,
                      input_provenance=data['provenance'], input_shape=list(acts.shape), cases=[])
        from nemo.collections.asr.parts.numba.rnnt_loss import TDTLossNumba
        device = torch.device('cuda:0')
        require(torch.cuda.get_device_properties(device).total_memory >= 8_000_000_000, 'Expected local GPU')
        torch.cuda.reset_peak_memory_stats(device)
        tdt_ref, edits = reference_class('TDTLossPytorch')
        rnnt_ref, _ = reference_class('RNNTLossPytorch')
        from resource_watchdog_v2 import sample_memory
        report['memory_events']=[]
        def memory_event(phase):
            event=sample_memory(os.getpid(),phase)
            report['memory_events'].append(event)
            print(json.dumps({'memory_phase':phase,'system_memory':event['system_memory'],'own_private_memory':event['worker_private_memory']}),flush=True)
        memory_event('before_reference_and_native_loss')
        upstream = torch.arange(1, len(lengths)+1).float()/3
        for sigma, omega in ((0.,0.),(.02,0.),(.02,1.)):
            raw_started = time.perf_counter()
            with torch.no_grad():
                if omega == 1:
                    full_reference_loss = rnnt_ref(blank=blank,reduction=None)(acts[...,:vocab],labels,lengths,targets)
                else:
                    full_reference_loss = tdt_ref(blank=blank,durations=durations,reduction=None,sigma=sigma)(acts,labels,lengths,targets)
            raw_seconds = time.perf_counter()-raw_started
            reference_started = time.perf_counter()
            expected_acts = acts.clone().requires_grad_()
            contracted, reference_labels, reference_blank, reference_vocab, compression = compress_reference_logits(expected_acts,labels,blank,vocab)
            if omega == 1:
                expected = rnnt_ref(blank=reference_blank,reduction=None)(contracted[...,:reference_vocab],reference_labels,lengths,targets)
            else:
                expected = tdt_ref(blank=reference_blank,durations=durations,reduction=None,sigma=sigma)(contracted,reference_labels,lengths,targets)
            (expected*upstream).sum().backward()
            expected_loss=expected.detach()
            expected_grad=expected_acts.grad
            reference_loss_delta=float((full_reference_loss-expected_loss).abs().max())
            reference_seconds=time.perf_counter()-reference_started
            del expected,expected_acts,contracted,reference_labels
            memory_event(f'after_CPU_reference_sigma{sigma}_omega{omega}')
            candidate_acts = acts.to(device).detach().requires_grad_()
            random.seed(941177)
            started = time.perf_counter()
            native = TDTLossNumba(blank=blank,durations=durations,reduction='none',sigma=sigma,omega=omega,fastemit_lambda=0,clamp=-1)
            actual = native(candidate_acts,labels.to(device),lengths.to(device),targets.to(device))
            (actual*upstream.to(device)).sum().backward()
            torch.cuda.synchronize(device)
            seconds = time.perf_counter()-started
            actual_cpu, native_grad = actual.detach().cpu(), candidate_acts.grad.cpu()
            delta = float((full_reference_loss-actual_cpu).abs().max())
            stats = grad_stats(expected_grad,native_grad)
            memory_event(f'after_native_gradient_sigma{sigma}_omega{omega}')
            duration_gradient_max_abs=float(native_grad[...,vocab:].abs().max())
            item = {'sigma':sigma,'omega':omega,'expected_loss':expected_loss.tolist(),
                    'full_width_official_reference_loss':full_reference_loss.tolist(),
                    'compressed_reference_raw_loss_delta':reference_loss_delta,
                    'full_width_reference_forward_seconds':raw_seconds,
                    'compressed_reference_forward_backward_seconds':reference_seconds,
                    'reference_compression':compression,
                    'native_loss':actual_cpu.tolist(),'maximum_raw_loss_delta':delta,'gradient':stats,
                    'native_seconds_including_first_JIT':seconds,'device_allocation_reference_edits':edits if omega == 0 else [],
                    'duration_gradient_max_abs':duration_gradient_max_abs}
            del actual,candidate_acts,native_grad,actual_cpu,native
            if omega == 0:
                custom_acts = acts.clone().requires_grad_()
                custom = custom_tdt(custom_acts,labels,lengths,targets,durations,blank,vocab,sigma)
                (custom*upstream).sum().backward()
                item['custom_v2_raw_loss_delta'] = float((expected_loss-custom.detach()).abs().max())
                item['custom_v2_gradient'] = grad_stats(expected_grad, custom_acts.grad)
                del custom,custom_acts
                memory_event(f'after_custom_gradient_sigma{sigma}_omega{omega}')
            item['thresholds'] = {'raw_loss_absolute':1e-4, 'gradient_relative_l2':2e-4,'gradient_min_cosine':.999999}
            item['passed'] = (delta < 1e-4 and reference_loss_delta < 1e-4 and stats['finite'] and stats['relative_l2'] < 2e-4 and stats['cosine'] > .999999)
            if omega == 1:
                item['passed'] &= item['duration_gradient_max_abs'] == 0
            else:
                g=item['custom_v2_gradient']
                item['passed'] &= item['custom_v2_raw_loss_delta'] < 1e-4 and g['finite'] and g['relative_l2'] < 2e-4 and g['cosine'] > .999999
            report['cases'].append(item)
            del expected_grad,expected_loss,full_reference_loss
        report.update(passed=all(x['passed'] for x in report['cases']), native_kernel_executed=True,
                      gpu_max_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      scope='Native NeMo CUDA Numba loss on identical actual saved logits; no native full encoder/tokenizer/decoder or omega.1 quality validation')
    for record in report['source']['sources'].values():
        require(sha(record['path']) == record['sha256'], 'Native/pinned source changed during audit')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(args.output),'passed':report.get('passed',report.get('passed_source_checks')),
                      'native_kernel_executed':report['native_kernel_executed']}))
    if report.get('passed') is False:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
