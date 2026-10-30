"""NEW-process 9GB allocator cap, 4GB cache cleanup and owned-worker telemetry.

The frozen trainer, cache and I/O helpers remain unchanged. The optional read-only
update callback samples host/process memory only after the exact frozen update
synchronize call; its cost is inside the original measured training interval.
CPU validation never initializes CUDA. Hardware efficacy requires an actual run.
"""
from __future__ import annotations
import argparse
from collections import deque
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from atomic_io_retry_v1 import write_json
from cuda_cache_reclaim_v1 import file_sha, FROZEN_TRAINER_SHA256
from resource_watchdog_v1 import sample_memory

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
CACHE_LAUNCHER_SHA256 = 'db1ad326f26a89c5163a34a839cc781e457793b83d625a7c99b8366db9301459'
CACHE_HELPER_SHA256 = '99ec5974fcd4714fa6fa341eb5cde70dc817085c997bd564c2b465aaaf98a543'
RESOURCE_HELPER_SHA256 = 'cc4ca1f11fdef19ac38d2aa878334451a100fd79f9a2e25a5afb4ceef89f0d7c'


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def safe_sample(pid, phase):
    try:
        return sample_memory(pid, phase)
    except Exception as error:
        # Observational API failures never replace a worker's original error.
        return {'phase': phase, 'observed_at_utc': timestamp(),
                'system_memory': {'available': False, 'failure_type': type(error).__name__},
                'worker_private_memory': {'available': False, 'pid': pid},
                'telemetry_failed': True}


def argument_identity(arguments):
    values = list(arguments)
    if values[:1] == ['--']:
        values = values[1:]
    if values.count('--run-id') != 1:
        raise ValueError('Exactly one explicit frozen-worker run ID is required')
    index = values.index('--run-id')
    if index + 1 >= len(values):
        raise ValueError('Missing frozen-worker run ID')
    run_id = values[index + 1]
    if not run_id or not run_id.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Safe explicit run ID required')
    return run_id, hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def creation_identity(pid):
    base = {'pid': pid, 'API': 'OpenProcess/GetProcessTimes; owned self PID only'}
    if os.name != 'nt':
        return {**base, 'available': False, 'reason': 'Windows native API required'}
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]; kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION only
    if not handle:
        return {**base, 'available': False, 'error': ctypes.get_last_error()}
    try:
        values = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in values)):
            return {**base, 'available': False, 'error': ctypes.get_last_error()}
        ticks = (int(values[0].dwHighDateTime) << 32) | int(values[0].dwLowDateTime)
        return {**base, 'available': True, 'creation_filetime_100ns': ticks,
                'creation_unix_seconds': ticks / 10_000_000 - 11_644_473_600}
    finally:
        kernel.CloseHandle(handle)


def validate_capacity(total_bytes, cap_gb=9.):
    if not 15e9 <= total_bytes <= 18e9:
        raise ValueError('Reviewed 16GB-class CUDA device required')
    if cap_gb != 9.:
        raise ValueError('This reviewed version requires exactly 9GB allocator cap')
    fraction = 9e9 / total_bytes
    return {'physical_cuda_device_bytes': int(total_bytes), 'cap_bytes': 9_000_000_000,
            'per_process_allocator_fraction': fraction, 'cuda_device_index': 0}


class UpdateTelemetry:
    """No Torch operations; bounded same-time host/self-process observations."""
    def __init__(self, pid, *, sample=sample_memory, clock=time.monotonic):
        self.pid = pid; self.sample = sample; self.clock = clock
        self.last_sample = None; self.eligible_calls = 0; self.sample_count = 0
        self.failures = 0; self.samples = deque(maxlen=128)

    def observe_update(self):
        self.eligible_calls += 1
        now = self.clock()
        if self.last_sample is not None and now - self.last_sample < 5.:
            return
        self.last_sample = now
        try:
            value = self.sample(self.pid, 'frozen_optimizer_end_before_cache_reclamation')
            self.sample_count += 1
            self.samples.append({'eligible_update_call': self.eligible_calls, **value})
        except Exception as error:
            self.failures += 1
            self.samples.append({'eligible_update_call': self.eligible_calls, 'status': 'telemetry_failed',
                                 'failure_type': type(error).__name__, 'observed_at_utc': timestamp()})

    def evidence(self):
        return {'callback_scope': 'Read-only _eligible callback after exact frozen run:539 synchronize, before cache reclamation',
                'eligible_update_calls': self.eligible_calls, 'successful_samples': self.sample_count,
                'failed_samples': self.failures, 'minimum_sampling_interval_seconds': 5.,
                'sample_capacity': 128, 'samples_omitted': max(0, self.sample_count + self.failures - len(self.samples)),
                'samples': list(self.samples), 'inside_original_training_timing': True}


def telemetry_reclaimer_class(base, telemetry):
    class ObservedReclaimer(base):
        def _eligible(self, frame):
            eligible = super()._eligible(frame)
            if eligible:
                telemetry.observe_update()
            return eligible
    return ObservedReclaimer


def main(args):
    pins = [('launch_with_cache_reclaim_v1.py', CACHE_LAUNCHER_SHA256),
            ('cuda_cache_reclaim_v1.py', CACHE_HELPER_SHA256),
            ('resource_watchdog_v1.py', RESOURCE_HELPER_SHA256),
            ('train_lora_batched.py', FROZEN_TRAINER_SHA256)]
    for name, expected in pins:
        if file_sha(OUTPUT / name) != expected:
            raise RuntimeError('Reviewed source differs: ' + name)
    paths = [args.provenance, args.cache_provenance, args.memory_provenance, args.worker_identity]
    if len({path.resolve() for path in paths}) != 4:
        raise ValueError('Four distinct NEW provenance/identity paths required')
    for path in paths:
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('Every evidence path must be in this workspace')
        if path.exists():
            raise FileExistsError('Existing evidence is immutable: ' + str(path))
    if args.cap_gb != 9. or args.threshold_gb != 4.:
        raise ValueError('Reviewed policy is exactly cap9GB/cache4GB')
    run_id, argument_sha = argument_identity(args.arguments)
    sources = {'memory_launcher_sha256': file_sha(__file__), 'cache_launcher_sha256': CACHE_LAUNCHER_SHA256,
               'cache_helper_sha256': CACHE_HELPER_SHA256, 'resource_helper_sha256': RESOURCE_HELPER_SHA256,
               'frozen_worker_sha256': FROZEN_TRAINER_SHA256}
    if args.validate_only:
        print({'passed': True, 'CUDA_initialized': False, 'Torch_not_imported': True,
               'device_validation_pending_GPU_launch': True, 'cap_gb': 9., 'cleanup_threshold_gb': 4.,
               'run_id': run_id, 'worker_arguments_sha256': argument_sha, **sources})
        return
    pid = os.getpid()
    identity = {'schema': 1, 'kind': 'owned_training_worker_identity_v1', 'pid': pid, 'ppid': os.getppid(),
                'expected_run_id': run_id, 'started_at_utc': timestamp(), **sources,
                'worker_arguments_sha256': argument_sha, 'process_creation_identity': creation_identity(pid),
                'telemetry_before_torch_import': safe_sample(pid, 'before_torch_import'),
                'scope': 'Actual interpreter running nested frozen worker; no command-line or other-process inspection'}
    write_json(args.worker_identity, identity)
    import torch
    properties = torch.cuda.get_device_properties(0)
    cap = validate_capacity(properties.total_memory, args.cap_gb)
    telemetry = UpdateTelemetry(pid)
    evidence = {'schema': 1, 'kind': 'memory_guard_training_launcher_v3', **sources, **cap,
                'cuda_device_name': properties.name, 'cap_scope': 'This process only; PyTorch CUDA allocator only',
                'actual_API': 'torch.cuda.set_per_process_memory_fraction(fraction, device=0)',
                'allocator_cap_may_trigger_native_reclaim_or_OOM': True, 'cleanup_threshold_bytes': 4_000_000_000,
                'global_driver_settings_changed': False, 'training_math_changed': False,
                'backend_flags_changed': False, 'RNG_sampler_optimizer_changed': False,
                'worker_pid': pid, 'worker_ppid': identity['ppid'], 'worker_identity_path': str(args.worker_identity.resolve()),
                'worker_identity_sha256': file_sha(args.worker_identity), 'worker_arguments_sha256': argument_sha,
                'run_id': run_id, 'cache_provenance_path': str(args.cache_provenance.resolve()),
                'io_provenance_path': str(args.provenance.resolve()), 'started_at_utc': timestamp(), 'status': 'running',
                'telemetry_after_torch_import': safe_sample(pid, 'before_frozen_worker_startup')}
    write_json(args.memory_provenance, evidence)
    original_class = cache_launcher = None
    try:
        torch.cuda.set_per_process_memory_fraction(cap['per_process_allocator_fraction'], device=0)
        evidence['allocator_cap_applied'] = True
        write_json(args.memory_provenance, evidence)
        import launch_with_cache_reclaim_v1 as cache_launcher
        original_class = cache_launcher.CudaCacheReclaimer
        cache_launcher.CudaCacheReclaimer = telemetry_reclaimer_class(original_class, telemetry)
        cache_launcher.main(SimpleNamespace(provenance=args.provenance, cache_provenance=args.cache_provenance,
            threshold_gb=4., max_events=args.max_events, retry_seconds=args.retry_seconds,
            validate_only=False, arguments=args.arguments))
        evidence['status'] = 'complete'
    except BaseException as error:
        evidence.update(status='failed', failure_type=type(error).__name__, failure_message=str(error))
        raise
    finally:
        if original_class is not None:
            cache_launcher.CudaCacheReclaimer = original_class
        evidence.update(finished_at_utc=timestamp(), owned_update_memory_telemetry=telemetry.evidence(),
                        telemetry_after_worker=safe_sample(pid, 'after_frozen_worker_exit'),
                        memory_launcher_sha256_after=file_sha(__file__),
                        cache_launcher_sha256_after=file_sha(OUTPUT/'launch_with_cache_reclaim_v1.py'),
                        cache_helper_sha256_after=file_sha(OUTPUT/'cuda_cache_reclaim_v1.py'),
                        resource_helper_sha256_after=file_sha(OUTPUT/'resource_watchdog_v1.py'),
                        frozen_worker_sha256_after=file_sha(OUTPUT/'train_lora_batched.py'))
        for name, expected in pins:
            evidence[name.removesuffix('.py') + '_sha256_after'] = file_sha(OUTPUT / name)
        write_json(args.memory_provenance, evidence)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('provenance', 'cache-provenance', 'memory-provenance', 'worker-identity'):
        result.add_argument('--' + name, type=Path, required=True)
    result.add_argument('--cap-gb', type=float, default=9.)
    result.add_argument('--threshold-gb', type=float, default=4.)
    result.add_argument('--max-events', type=int, default=128)
    result.add_argument('--retry-seconds', type=float, default=30.)
    result.add_argument('--validate-only', action='store_true')
    result.add_argument('arguments', nargs=argparse.REMAINDER)
    return result


if __name__ == '__main__':
    main(parser().parse_args())
