"""Finite CPU-only byte/state and bounded-slice checkpoint fixture."""
import argparse, gc, hashlib, json, os, random, sys, threading, time
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path(r'E:\gpu_training_scratchpad')
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json
from bpe_streaming_checkpoint_v1 import MappedCheckpoint, write_checkpoint, record, require, CHUNK

def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); a = p.parse_args()
    folder = Path(a.output).resolve(); require(folder.is_relative_to(ROOT.resolve()), 'E fixture storage required')
    folder.mkdir(exist_ok=False)
    import torch
    require(torch.__version__.endswith('+cpu') and not torch.cuda.is_available(), 'Isolated actual CPU-only Torch required')
    # Reuse only the existing compatible NumPy module files; no environment edit
    # or alternate Torch distribution is loaded into this CPU fixture.
    numpy_path = PROJECT / 'work/nemo-native-env-v2/Lib/site-packages'
    sys.path.insert(0, str(numpy_path))
    import numpy as np
    sys.path.remove(str(numpy_path))
    torch.set_num_threads(1)
    from resource_watchdog_v2 import sample_memory
    counts = 0; peak = 0; last = None; error = None; done = threading.Event()
    def watch():
        nonlocal counts, peak, last, error
        try:
            while not done.is_set():
                value = sample_memory(os.getpid(), 'streaming_checkpoint_write')['worker_private_memory']
                require(value['available'], 'Available checkpoint phase sample required')
                counts += 1; peak = max(peak, value['private_bytes']); last = time.monotonic(); done.wait(.01)
        except BaseException as exc:
            error = repr(exc)
    checks = 0
    def check(value, message):
        nonlocal checks
        require(value, message); checks += 1
    generator = torch.Generator().manual_seed(991)
    weights = {'large_head': torch.randn((8193, 1024), generator=generator)}
    weights.update({'adapter_'+str(i): torch.randn((16, 64), generator=generator) for i in range(206)})
    moments = {i: {'step': torch.tensor(float((64250, 2000, 62250)[i % 3])),
                   'exp_avg': tensor * .1, 'exp_avg_sq': tensor.square() * .01}
               for i, tensor in enumerate(weights.values())}
    value = {'kind': 'fixture_not_training', 'diagnostic_only': True, 'weights': weights,
             'optimizers': [{'state': moments, 'param_groups': [{'params': list(moments), 'lr': 1e-5}]}],
             'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state(),
             'types': [None, False, True, -5, 1.25, 'Türkçe', ('tuple', [1, 2])],
             'other_tensors': [torch.arange(17, dtype=torch.int64), torch.arange(17, dtype=torch.uint8),
                               torch.arange(17, dtype=torch.float32).to(torch.bfloat16), torch.empty(0)]}
    gc.collect()
    baseline = sample_memory(os.getpid(), 'streaming_checkpoint_before')['worker_private_memory']['private_bytes']
    thread = threading.Thread(target=watch, daemon=True); thread.start(); started = time.monotonic()
    try:
        saved = write_checkpoint(folder/'state.json', value, torch=torch, guard=lambda: None)
    finally:
        done.set(); thread.join(timeout=2)
    check(not thread.is_alive() and error is None and counts >= 3 and last is not None, 'Clean available phase monitor required')
    check(peak-baseline < 64_000_000, 'Observed streaming CPU increment exceeded declared64MB diagnostic forecast')
    elapsed = time.monotonic()-started
    restored = MappedCheckpoint(saved, torch=torch, guard=lambda: None)
    def equal(actual, expected):
        if isinstance(expected, torch.Tensor):
            check(actual.dtype == expected.dtype and actual.shape == expected.shape and torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), 'Tensor byte identity')
        elif type(expected) is dict:
            check(type(actual) is dict and list(actual) == list(expected), 'Exact ordered key/type identity')
            for key in expected: equal(actual[key], expected[key])
        elif type(expected) in (list, tuple):
            check(type(actual) is type(expected) and len(actual) == len(expected), 'Exact container identity')
            for av, ev in zip(actual, expected): equal(av, ev)
        else:
            check(type(actual) is type(expected) and actual == expected, 'Exact scalar identity')
    equal(restored.value, value)
    # Active CPU Adam clocks must be cloned before in-place use. Read-only
    # checkpoint-backed tensor objects are never writable optimizer state.
    step = restored.value['optimizers'][0]['state'][0]['step'].clone()
    step.add_(1)
    check(step.item() == 64251 and restored.value['optimizers'][0]['state'][0]['step'].item() == 64250,
          'Independent mutable Adam clock copy preserves read-only backing')
    del step
    restored.close()
    check(restored.closed and restored.value is None, 'Mapped checkpoint closed after all borrowed views released')
    tiny = write_checkpoint(folder/'tamper.json', {'weight': torch.ones(3)}, torch=torch, guard=lambda: None)
    document = json.loads(Path(tiny['path']).read_text())
    with Path(document['payload']['path']).open('r+b') as stream:
        old = stream.read(1); stream.seek(0); stream.write(bytes([old[0] ^ 1])); stream.flush(); os.fsync(stream.fileno())
    try:
        MappedCheckpoint(tiny, torch=torch, guard=lambda: None)
    except ValueError:
        check(True, 'Changed immutable payload rejected')
    else:
        raise ValueError('Changed payload unexpectedly accepted')
    try:
        write_checkpoint(folder/'state.json', value, torch=torch, guard=lambda: None)
    except ValueError:
        check(True, 'Existing checkpoint cannot be overwritten')
    else:
        raise ValueError('Existing checkpoint overwritten')
    write_json(folder/'report.json', {'passed': True, 'CPU_only': True, 'checks': checks, 'PID': os.getpid(),
        'source': record(__file__), 'checkpoint_source': record(PROJECT/'outputs/parakeet_turkish/bpe_streaming_checkpoint_v1.py'),
        'checkpoint': saved, 'named_weights': 207, 'moment_partitions_fixture_only': True,
        'maximum_CPU_tensor_slice_bytes': CHUNK, 'private_baseline_bytes': baseline,
        'sampled_write_private_peak_bytes': peak, 'sampled_write_increment_bytes': peak-baseline,
        'write_available_samples': counts, 'monitor_error': error, 'write_seconds': elapsed,
        'all_nested_types_and_tensor_bytes_exact': True, 'mutable_CPU_Adam_step_clone_required': True,
        'GPU_native_memory_or_production_approval_claimed': False})
    print(json.dumps({'passed': True, 'checks': checks, 'increment': peak-baseline, 'seconds': elapsed}))

if __name__ == '__main__':
    main()
