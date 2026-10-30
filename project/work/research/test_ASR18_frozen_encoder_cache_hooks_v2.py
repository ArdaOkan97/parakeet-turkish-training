"""Inactive focused CPU2.7.1 protocol fixture for real cache hook bodies.

Only tiny BF16 CPU Linear tensors are used. The entry called gpu_parameter is
explicitly a CPU stand-in; full294 installation, active207, native CUDA, teacher,
forecast fit and performance are NOT tested. No source/environment installation.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import threading
import time
import warnings

PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path(r'E:\gpu_training_scratchpad')
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
import asr18_frozen_encoder_cuda_cache_v1 as old_cache
import asr18_frozen_encoder_cuda_cache_v2 as cache
from atomic_io_retry_v1 import write_json
from fastmss_owned_process_closure_v2 import identity
from resource_watchdog_v2 import ResourceGuard, sample_memory

CPU_SITE = ROOT / 'scratch/BPE_broad_CPU_preflight_env_v2/Lib/site-packages'
NUMPY_SITE = ROOT / 'scratch/CPU_validation_v2_env/Lib/site-packages'
EXPECTED_OLD = '8b11c838875028ee334d08f713e6e0e7eed12d98ab17bedf13b707859086ef23'
EXPECTED_NEW = 'e861fb57fad35d20517961f0e44d3fa3f766dbb188abc1d070593f6a1f967b2b'
watch = ResourceGuard(); peak = 0; checks = []
torch = None; np = None


def require(value, message):
    if not value:
        raise ValueError(message)


def pin(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT) or path.is_relative_to(PROJECT), 'Existing project/E only')
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def guard():
    global peak
    sample = sample_memory(os.getpid(), 'CPU_cache_hook_protocol')
    require(sample['paired_sampling_within_one_second'] and sample['worker_private_memory']['available']
            and sample['system_memory']['available'], 'Fresh paired telemetry')
    stop, _ = watch.observe(sample['system_memory'], sample['worker_private_memory'])
    require(stop is None, 'Unchanged system/worker guard: ' + str(stop))
    peak = max(peak, sample['worker_private_memory']['private_bytes'])
    require(peak <= 512_000_000 and shutil.disk_usage(ROOT).free > 100_000_000_000,
            'Proposed512MB private/E100GB, unchanged2GB outer hard limit')
    return sample


def check(name, value, **detail):
    require(value, name); checks.append({'name': name, 'passed': True, **detail}); guard()


def rng():
    state = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': [state[0], state[1].tobytes().hex(), *state[2:]],
            'torch': bytes(torch.get_rng_state().tolist()).hex(),
            'CUDA': [bytes(x.tolist()).hex() for x in torch.cuda.get_rng_state_all()]}


def toy(candidate):
    model = torch.nn.Module(); model.encoder = torch.nn.Sequential(
        torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16))
    model.encoder.eval(); linear = model.encoder[0]
    linear.weight.requires_grad_(False)
    with torch.no_grad():
        linear.weight.copy_((torch.arange(64, dtype=torch.float32).reshape(8, 8) / 128 - .25).to(torch.bfloat16))
    parameter = linear.weight
    model._diagnostic_frozen_metrics = {'forward_materializations': 0,
        'transferred_weight_bytes': 0, 'logical_current_matrix_bytes': 0,
        'logical_peak_matrix_bytes': 0}
    # This intentionally does NOT call install(): it is a CPU stand-in test of
    # exact hooks/exception lifetime, not a pretend294 CUDA bank/model fixture.
    entry = {'module': linear, 'cpu': parameter.detach(), 'cpu_parameter': parameter,
        'parameter_version': parameter._version, 'cpu_version': parameter._version,
        'gpu_parameter': torch.nn.Parameter(parameter.detach().clone(), requires_grad=False),
        'size': parameter.numel() * parameter.element_size(), 'busy': False}
    entry['gpu_version'] = entry['gpu_parameter']._version
    state = candidate._InstanceCache(model, [entry])
    before, after = state.hooks(entry, torch)
    return model, linear, entry, state, before, after


def arm(candidate=None):
    model, linear, entry, state, before, after = toy(cache if candidate is None else candidate)
    events = []
    if candidate is not None:
        linear.register_forward_pre_hook(before)
        linear.register_forward_hook(after, always_call=True)
        def observer(module, inputs, output):
            events.append({'output_is_None': output is None,
                'exception': type(sys.exc_info()[1]).__name__,
                'main_thread': threading.current_thread() is threading.main_thread(),
                'CPU_parameter_restored': module.weight is entry['cpu_parameter']})
        linear.register_forward_hook(observer, always_call=True)
    random.seed(812); np.random.seed(813); torch.manual_seed(814)
    initial = rng()
    x = torch.arange(16, dtype=torch.float32).reshape(2, 8).to(torch.bfloat16).requires_grad_()
    pointer, version = x.data_ptr(), x._version
    from torch.utils.checkpoint import checkpoint
    y = checkpoint(linear, x, use_reentrant=False); y.sum().backward()
    return {'output': y.detach().clone(), 'gradient': x.grad.detach().clone(),
            'RNG_unchanged': initial == rng(), 'events': events,
            'poisoned': state.poisoned, 'CPU_restored': linear.weight is entry['cpu_parameter'],
            'busy': entry['busy'], 'inflight': state.metrics['inflight_matrices'],
            'logical_current': model._diagnostic_frozen_metrics['logical_current_matrix_bytes'],
            'input_unchanged': x.data_ptr() == pointer and x._version == version,
            'materializations': model._diagnostic_frozen_metrics['forward_materializations']}


def failure_case(mode):
    model, linear, entry, state, before, after = toy(cache)
    linear.register_forward_pre_hook(before)
    linear.register_forward_hook(after, always_call=True)
    original = linear.forward
    if mode == 'forward':
        def fail(*args): raise RuntimeError('PRIVATE true module failure')
        linear.forward = fail
    elif mode == 'prehook':
        def fail(*args): raise RuntimeError('PRIVATE later prehook failure')
        linear.register_forward_pre_hook(fail)
    elif mode == 'version':
        def mutate(x):
            with torch.no_grad(): entry['gpu_parameter'].add_(1)
            return original(x)
        linear.forward = mutate
    elif mode == 'replace':
        def replace(x):
            linear._parameters['weight'] = torch.nn.Parameter(entry['gpu_parameter'].detach().clone(), requires_grad=False)
            return original(x)
        linear.forward = replace
    elif mode == 'duplicate':
        linear.register_forward_pre_hook(before)
    elif mode == 'busy':
        entry['busy'] = True; state.metrics['inflight_matrices'] = 1
        model._diagnostic_frozen_metrics['logical_current_matrix_bytes'] = entry['size']
    else:
        raise ValueError('Unknown fixture failure mode')
    initial = rng(); raised = False
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        try:
            linear(torch.ones(2, 8, dtype=torch.bfloat16, requires_grad=True))
        except (ValueError, RuntimeError) as error:
            raised = True; exception = type(error).__name__
    check(mode + ': actual failure rejects/poisons and restores original CPU parameter/counts',
          raised and state.poisoned and linear.weight is entry['cpu_parameter'] and not entry['busy']
          and state.metrics['inflight_matrices'] == 0
          and model._diagnostic_frozen_metrics['logical_current_matrix_bytes'] == 0,
          actual_exception=exception if raised else None, suppressed_hook_warning_count=len(recorded))
    check(mode + ': Python/NumPy/Torch/empty-CUDA RNG unchanged', initial == rng())


def main(output):
    global torch, np
    began = time.monotonic(); output = Path(output).resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), 'Fresh E fixture output')
    config = json.loads((ROOT / 'storage_config.json').read_text())
    require(config['storage_root'] == str(ROOT) and config['allow_fallback_outside_root'] is False,
            'E storage only/no fallback')
    require(os.environ['CUDA_VISIBLE_DEVICES'] == '-1', 'CUDA masked')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        require(os.environ[name] == '1', 'Four thread caps1')
    admissions = []
    for _ in range(3):
        sample = guard(); system = sample['system_memory']; budget = 576_000_000
        require(system['commit_used_bytes'] + budget < .90 * system['commit_limit_bytes']
                and system['commit_available_bytes'] - budget > 2_000_000_000, 'Fresh whole576MB admission')
        admissions.append(sample); time.sleep(.1)
    require(pin(old_cache.__file__)['sha256'] == EXPECTED_OLD and pin(cache.__file__)['sha256'] == EXPECTED_NEW,
            'Exact frozen candidate sources')
    # Both existing E environments are Python3.11.13; this child alone gains
    # the explicitly pinned NumPy site. No installed file/environment changes.
    sys.path.insert(0, str(NUMPY_SITE))
    import numpy as imported_numpy
    np = imported_numpy
    import torch as imported_torch
    torch = imported_torch
    require(Path(np.__file__).resolve() == (NUMPY_SITE / 'numpy/__init__.py').resolve()
            and Path(torch.__file__).resolve() == (CPU_SITE / 'torch/__init__.py').resolve()
            and torch.__version__ == '2.7.1+cpu' and not torch.cuda.is_available()
            and not torch.cuda.is_initialized(), 'Known exact CPU2.7.1/NumPy E roots, no GPU runtime')
    torch.set_num_threads(1); torch.set_num_interop_threads(1); output.mkdir()
    baseline = arm(); previous = arm(old_cache); current = arm(cache)
    check('Real nonreentrant checkpoint baseline runs without RNG/input mutation', baseline['RNG_unchanged'] and baseline['input_unchanged'])
    check('Legacy hook poison reproduced on normal checkpoint early-stop', previous['poisoned'] and
          any(x['output_is_None'] and x['exception'] == '_StopRecomputationError' for x in previous['events']))
    check('v2 accepts only actual normal checkpoint early-stop', not current['poisoned'] and
          any(x['output_is_None'] and x['exception'] == '_StopRecomputationError' for x in current['events']))
    check('CPU BF16 full output and input gradient match baseline byte-for-byte',
          torch.equal(current['output'].view(torch.uint8), baseline['output'].view(torch.uint8))
          and torch.equal(current['gradient'].view(torch.uint8), baseline['gradient'].view(torch.uint8)))
    check('Successful recompute restores identical CPU Parameter/no busy or logical bytes',
          current['CPU_restored'] and not current['busy'] and current['inflight'] == current['logical_current'] == 0
          and current['materializations'] == 2)
    check('v2 all available RNG and input pointers/versions unchanged', current['RNG_unchanged'] and current['input_unchanged'])
    check('Strict main-thread callback rule holds in this CPU recomputation only', all(x['main_thread'] for x in current['events']))
    for mode in ('forward', 'prehook', 'version', 'replace', 'duplicate', 'busy'):
        failure_case(mode)
    model, linear, entry, state, before, after = toy(cache)
    errors = []
    def foreign_thread():
        try: before(linear, (torch.ones(2, 8, dtype=torch.bfloat16),))
        except ValueError as error: errors.append(str(error))
    thread = threading.Thread(target=foreign_thread); thread.start(); thread.join(timeout=2)
    check('Foreign-thread prehook remains forbidden and poisons candidate', not thread.is_alive() and errors
          and state.poisoned and linear.weight is entry['cpu_parameter'] and not entry['busy'])
    check('No CUDA state/context was initialized by CPU fixture', rng()['CUDA'] == [] and not torch.cuda.is_initialized())
    check('Frozen v1/v2 sources unchanged', pin(old_cache.__file__)['sha256'] == EXPECTED_OLD
          and pin(cache.__file__)['sha256'] == EXPECTED_NEW)
    report = {'kind': 'ASR18_frozen_cache2_CPU_checkpoint_hook_protocol_controls_v1', 'passed': True,
        'producer': identity(), 'worker_source': pin(__file__), 'parent_source': pin(__file__),
        'candidate': pin(cache.__file__), 'legacy_candidate': pin(old_cache.__file__),
        'checks': checks, 'check_count': len(checks), 'CPU_recompute_events': current['events'],
        'CPU_stand_in_for_CUDA_weight_storage': True, 'full294_install_exercised': False,
        'all207_or_native_GPU_equivalence_claimed': False, 'CUDA_RNG_scope': 'CPU-only empty state, not native CUDA generator validation',
        'production_enabled': False, 'performance_claimed': False, 'private_peak_bytes': peak,
        'whole_forecast_bytes': 576_000_000, 'wall_seconds': time.monotonic() - began,
        'fresh_three_admissions': admissions, 'torch_version': torch.__version__, 'numpy_version': np.__version__,
        'runtime_sources': [pin(CPU_SITE / 'torch/utils/checkpoint.py'), pin(CPU_SITE / 'torch/nn/modules/module.py'),
            pin(CPU_SITE / 'torch/version.py'), pin(NUMPY_SITE / 'numpy/__init__.py'),
            pin(CPU_SITE.parent.parent / 'pyvenv.cfg'), pin(NUMPY_SITE.parent.parent / 'pyvenv.cfg')]}
    write_json(output / 'report.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
