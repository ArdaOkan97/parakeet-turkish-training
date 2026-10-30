"""Root-only comparison of the old/new wrappers on saved actual TRAIN logits."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUT_SHA = '8c3481e742f7df332102f718bc632314cd8869896ef292b28b2e0654bbfe1259'
OLD_SHA = '24e6f01a63c482d8451d72af9e21994332c0d0be53cbefbdf4c295f659090fd5'
NEW_SHA = '839194dddf83c3b08bf608d01cb540b2f160feee426a91373d14cfc426ab7494'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def main(args):
    require(args.gpu_window_authorized and not args.output.exists(), 'New root-owned proof required')
    require(sha(args.input) == INPUT_SHA, 'Actual TRAIN input changed')
    require(sha(HERE / 'native_tdt_checkpointed_v2.py') == OLD_SHA, 'Old adapter changed')
    require(sha(HERE / 'native_tdt_checkpointed_v3.py') == NEW_SHA, 'New adapter changed')
    from preload_nemo_linker_v1 import preload
    preloaded = preload()
    import torch
    import native_tdt_checkpointed_v2 as old
    import native_tdt_checkpointed_v3 as new
    from resource_watchdog_v2 import ResourceGuard, sample_memory, worker_process_commit_reason
    from verify_nemo_native_loss_v3 import grad_stats, validate_input
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(6e9 / torch.cuda.get_device_properties(0).total_memory)
    data = torch.load(args.input, map_location='cpu', weights_only=True)
    _, _, _, _, durations, blank, vocab = validate_input(data)
    backend = old._native_loss(blank, tuple(durations), 0.)
    guard = ResourceGuard()
    report = {'kind': 'native_split_actual_TRAIN_logits_GPU_comparison_v1', 'passed': False,
              'input_sha256': INPUT_SHA, 'old_adapter_sha256': OLD_SHA,
              'new_adapter_sha256': NEW_SHA, 'preload': preloaded,
              'native_full_model_or_optimizer_equivalence_claimed': False,
              'scope': 'Identical real TRAIN joint logits; official kernel and full logit gradient',
              'cases': [], 'memory': [], 'pid': os.getpid()}
    began = time.monotonic()

    def observe(phase):
        value = sample_memory(os.getpid(), phase)
        reason, _ = guard.observe(value['system_memory'], value['worker_private_memory'],
                                  observed_monotonic=value['observed_monotonic'])
        reason = reason or worker_process_commit_reason(value['calling_process_commit_memory'], os.getpid())
        report['memory'].append(value)
        require(reason is None and time.monotonic() - began < 150, 'Resource/time guard: ' + str(reason))

    class SavedJoint(torch.nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.logits = logits

        def forward(self, encoder, decoder):
            return self.logits

    try:
        observe('before_native_comparison')
        for index in range(data['joint_logits'].shape[0]):
            t, u = int(data['frame_lengths'][index]), int(data['target_lengths'][index])
            labels = data['labels'][index:index+1, :u].to('cuda').contiguous()
            frames = torch.tensor([t], dtype=torch.long, device='cuda')
            targets = torch.tensor([u], dtype=torch.long, device='cuda')
            ep = torch.empty((1, t, 1), device='cuda')
            dp = torch.empty((1, u+1, 1), device='cuda')
            for dtype in (torch.float32, torch.bfloat16):
                results = []
                for name, module in (('old', old), ('split', new)):
                    logits = data['joint_logits'][index:index+1, :t, :u+1].to(
                        device='cuda', dtype=dtype).contiguous().detach().requires_grad_(True)
                    rng = random.getstate()
                    loss = module._clip_loss(SavedJoint(logits), ep, dp, labels, frames,
                                             targets, backend, vocab, len(durations))
                    loss.sum().backward()
                    torch.cuda.synchronize()
                    require(random.getstate() == rng, 'Wrapper changed Python RNG')
                    results.append((loss.detach().cpu(), logits.grad.detach().float().cpu()))
                    del loss, logits
                    torch.cuda.empty_cache()
                    observe(name + '_after_full_logit_gradient')
                a, b = results
                stats = grad_stats(a[1], b[1])
                delta = float((a[0] - b[0]).abs().max())
                passed = delta <= 1e-5 and stats['finite'] and stats['relative_l2'] <= 1e-6
                record = {'clip_index': index, 'source_id': data['provenance']['source_ids'][index],
                          'T': t, 'U': u, 'dtype': str(dtype), 'old_loss': a[0].tolist(),
                          'new_loss': b[0].tolist(), 'loss_max_delta': delta,
                          'gradient': stats, 'loss_bitexact': torch.equal(a[0], b[0]),
                          'gradient_bitexact': torch.equal(a[1], b[1]), 'passed': passed}
                report['cases'].append(record)
                print(json.dumps(record), flush=True)
                require(passed, 'Actual native gradient comparison failed')
                del results, a, b
        report['passed'] = True
    finally:
        report.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      actual_native_kernel_and_backward_executed=bool(report['cases']))
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': report['passed'], 'cases': len(report['cases']),
                      'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'work/nemo_actual_train_logits_v1.pt')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu-window-authorized', action='store_true')
    main(parser.parse_args())
