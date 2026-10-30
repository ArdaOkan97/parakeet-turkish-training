"""SAME-run TSC exact resume with unchanged8GB cap/4GB cache/owned telemetry.

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
import threading
import sys
from types import SimpleNamespace

from atomic_io_retry_v1 import write_json
from cuda_cache_reclaim_v2 import file_sha
FROZEN_TRAINER_SHA256 = '1bf121e4d25b39112c8a05412b856ac4d7c8315c92819f001d7183146d2a0f8b'
NATIVE_ADAPTER_SHA256 = '839194dddf83c3b08bf608d01cb540b2f160feee426a91373d14cfc426ab7494'
DATA_POLICY_SHA256 = '08eee3896bf8571c9e97dae1be688ecef7766816c019741dd7631d70f640ce41'
from resource_watchdog_v2 import sample_memory

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
CACHE_LAUNCHER_SHA256 = '0516a1ed15b33b34e93eb964bd11a2f7701330bd93c3d73b015b2fa73b70eca6'
CACHE_HELPER_SHA256 = '869b9ffaac70c9b7b6e7590ba1d330363100d6b2ddb9f8f7af9b2b9865c1c7f9'
RESOURCE_HELPER_SHA256 = '64d1393e1e45cf65dac14f78ac955331c91b4a3f0eafbf7535196e2b61778641'
RESOURCE_TELEMETRY_SHA256 = '2ae9f2cdfc388a294914e000c60b62fe09cddac147700c5762851d626a95a683'


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


def tsc_argument_records(arguments):
    """Require explicit isolated paths; do not inherit the old worker's defaults."""
    values = list(arguments)
    if values[:1] == ['--']:
        values = values[1:]
    records = {}
    for flag in ('--tsc-input-proof', '--tsc-training-approval', '--source-closure',
                 '--tr-manifest', '--replay-manifest', '--tr-frame-metadata',
                 '--tr-frame-metadata-proof', '--data-v3-proof', '--resume-exact'):
        if values.count(flag) != 1 or values.index(flag)+1 >= len(values):
            raise ValueError('Exactly one explicit isolated TSC argument required: '+flag)
        path = Path(values[values.index(flag)+1]).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError('Approved existing workspace TSC asset required: '+flag)
        records[flag.removeprefix('--').replace('-', '_')] = {'path': str(path), 'sha256': file_sha(path)}
    if '--resume-weights' in values or '--resume-runtime-migration' in values:
        raise ValueError('This SAME-run TSC route admits only exact full-state resume, never warm weights or migration')
    return records


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


def validate_capacity(total_bytes, cap_gb=8.):
    if not 15e9 <= total_bytes <= 18e9:
        raise ValueError('Reviewed 16GB-class CUDA device required')
    if cap_gb != 8.:
        raise ValueError('This version permits explicit8GB caps; root must select a hardware-validated cap')
    fraction = cap_gb * 1e9 / total_bytes
    return {'physical_cuda_device_bytes': int(total_bytes), 'cap_bytes': int(cap_gb * 1e9),
            'per_process_allocator_fraction': fraction, 'cuda_device_index': 0}


class LiveTelemetrySampler:
    """Read-only owned-self Windows telemetry; no Torch, CUDA or RNG operations.

    A latest atomic record stays fresh through cache preparation and development.
    The thread is stopped and joined before final evidence is serialized. Every
    deque is bounded; observational failure never replaces worker exceptions.
    """
    def __init__(self, pid, sink, *, sample=sample_memory):
        self.pid = pid; self.sink = sink; self.sample = sample
        self.stop_event = threading.Event(); self.thread = None
        self.samples = deque(maxlen=128); self.successes = 0; self.failures = 0
        self.publish_failures = 0; self.join_completed = True

    def sample_once(self):
        try:
            value = self.sample(self.pid, 'owned_worker_all_phases_observational_sampler')
            self.successes += 1; self.samples.append(value)
            try:
                self.sink(value)
            except Exception as error:
                self.publish_failures += 1
                self.samples[-1]['live_publish_failure_type'] = type(error).__name__
        except Exception as error:
            self.failures += 1
            self.samples.append({'status': 'telemetry_failed', 'failure_type': type(error).__name__,
                                 'observed_at_utc': timestamp()})

    def _run(self):
        while not self.stop_event.is_set():
            self.sample_once()
            if self.stop_event.wait(5.):
                return

    def start(self):
        if self.thread is not None:
            raise RuntimeError('Sampler may start only once')
        self.thread = threading.Thread(target=self._run, name='owned-self-memory-telemetry', daemon=True)
        self.thread.start()

    def stop_and_join(self):
        self.stop_event.set()
        if self.thread is not None and self.thread.ident is not None:
            self.thread.join(timeout=2.)
            self.join_completed = not self.thread.is_alive()

    def evidence(self):
        return {'scope': 'Actual owned-self process; every5sec while nestedworkeractive includingcache/DEV; stdlibWindowsAPI only',
                'successful_samples': self.successes, 'failed_samples': self.failures,
                'live_publish_failures': self.publish_failures, 'sample_capacity': 128,
                'samples_omitted': max(0, self.successes+self.failures-len(self.samples)),
                'minimum_sampling_interval_seconds': 5., 'stop_requested': self.stop_event.is_set(),
                'thread_join_completed': self.join_completed, 'samples': list(self.samples),
                'CUDA_model_RNG_operations': False}


class UpdateTelemetry:
    """No Torch operations; bounded same-time host/self-process observations."""
    def __init__(self, pid, *, sample=sample_memory, clock=time.monotonic, sink=None):
        self.pid = pid; self.sample = sample; self.clock = clock
        self.last_sample = None; self.eligible_calls = 0; self.sample_count = 0
        self.failures = 0; self.samples = deque(maxlen=128)
        self.sink = sink
        self.publish_failures = 0

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
            if self.sink is not None:
                try:
                    self.sink(value)
                except Exception as error:
                    self.publish_failures += 1
                    self.samples[-1]['live_publish_failure_type'] = type(error).__name__
        except Exception as error:
            self.failures += 1
            self.samples.append({'eligible_update_call': self.eligible_calls, 'status': 'telemetry_failed',
                                 'failure_type': type(error).__name__, 'observed_at_utc': timestamp()})

    def evidence(self):
        return {'callback_scope': 'Read-only _eligible callback after exact frozen run:545 synchronize, before cache reclamation',
                'eligible_update_calls': self.eligible_calls, 'successful_samples': self.sample_count,
                'failed_samples': self.failures, 'minimum_sampling_interval_seconds': 5.,
                'live_publish_failures': self.publish_failures,
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
    pins = [('launch_with_cache_reclaim_tsc_v1.py', CACHE_LAUNCHER_SHA256),
            ('cuda_cache_reclaim_v2.py', CACHE_HELPER_SHA256),
            ('resource_watchdog_v2.py', RESOURCE_HELPER_SHA256),
            ('paired_windows_commit_telemetry_v1.py', RESOURCE_TELEMETRY_SHA256),
            ('train_lora_batched_tsc_v1.py', FROZEN_TRAINER_SHA256),
            ('public_turkish_data_policy_v3.py', DATA_POLICY_SHA256),
            ('native_tdt_checkpointed_v3.py', NATIVE_ADAPTER_SHA256),
            ('cuda_cache_caller_tsc_v1.py', '6cf4c021eb5c3683f09119995b1b3c9cd4bce15491818e355272ea25b53c77ec'),
            ('tsc_training_inputs_v1.py', '7123bd6978294af4fe15c6c475699db825af98d245f1940d827ee68f961f3f7c'),
            ('tsc_supplemental_data_policy_v1.py', 'a10b1cbaa94b4e6810f6a5fd749de4b7743bbaf68504c63bbcd5070e8cdd7cdb'),
            ('prepare_TSC_TRAIN_v1.py', '08785c2446229ae6716feee5e9103d77c468faf0eaf68a602e0477bc2d36063e'),
            ('train_lora_batched_data_v11.py', '4b4c9e3008205c61bb5a881972187b5b132881dc7a1d9249583d1adcd52a7d20')]
    for name, expected in pins:
        if file_sha(OUTPUT / name) != expected:
            raise RuntimeError('Reviewed source differs: ' + name)
    live_memory_path = Path(str(args.worker_identity) + '.live_memory.json')
    paths = [args.provenance, args.cache_provenance, args.memory_provenance, args.worker_identity, live_memory_path]
    if len({path.resolve() for path in paths}) != 5:
        raise ValueError('Four CLI evidence paths and their derived NEW live-memory path must be distinct')
    for path in paths:
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('Every evidence path must be in this workspace')
        if path.exists():
            raise FileExistsError('Existing evidence is immutable: ' + str(path))
    if args.cap_gb != 8. or args.threshold_gb != 4.:
        raise ValueError('Explicit8GB cap required; optimizer-end cache4GB unchanged')
    run_id, argument_sha = argument_identity(args.arguments)
    tsc_assets = tsc_argument_records(args.arguments)
    sources = {'memory_launcher_sha256': file_sha(__file__), 'cache_launcher_sha256': CACHE_LAUNCHER_SHA256,
               'cache_helper_sha256': CACHE_HELPER_SHA256, 'resource_helper_sha256': RESOURCE_HELPER_SHA256,
               'resource_telemetry_sha256': RESOURCE_TELEMETRY_SHA256,
               'frozen_worker_sha256': FROZEN_TRAINER_SHA256, 'data_policy_sha256': DATA_POLICY_SHA256,
               'native_cleanup_adapter_sha256': NATIVE_ADAPTER_SHA256,
               'TSC_inputs_helper_sha256': '7123bd6978294af4fe15c6c475699db825af98d245f1940d827ee68f961f3f7c',
               'TSC_supplemental_policy_sha256': 'a10b1cbaa94b4e6810f6a5fd749de4b7743bbaf68504c63bbcd5070e8cdd7cdb',
               'TSC_importer_sha256': '08785c2446229ae6716feee5e9103d77c468faf0eaf68a602e0477bc2d36063e',
               'TSC_explicit_asset_records': tsc_assets,
               'TSC_resume_mode': 'SAME-run exact full-state continuation; original immutable native-ready/data/closure identities retained; fresh process evidence is separate',
               'memory_policy_change': 'Declared allocator cap8GB; original endcleanup4GB/criticalWindowsguards unchanged; nativeadapter3 splits operands before FP32 and owns perclip inactivecache cleanup'}
    if args.validate_only:
        print({'passed': True, 'CUDA_initialized': False, 'Torch_not_imported': True,
               'device_validation_pending_GPU_launch': True, 'cap_gb': args.cap_gb, 'cleanup_threshold_gb': 4.,
               'run_id': run_id, 'worker_arguments_sha256': argument_sha, **sources})
        return
    pid = os.getpid()
    identity = {'schema': 1, 'kind': 'owned_training_worker_identity_v1', 'pid': pid, 'ppid': os.getppid(),
                'expected_run_id': run_id, 'started_at_utc': timestamp(), **sources,
                'worker_arguments_sha256': argument_sha, 'process_creation_identity': creation_identity(pid),
                'telemetry_before_torch_import': safe_sample(pid, 'before_torch_import'),
                'live_memory_path': str(live_memory_path.resolve()),
                'scope': 'Actual interpreter running nested frozen worker; no command-line or other-process inspection'}
    write_json(args.worker_identity, identity)
    identity_sha = file_sha(args.worker_identity)
    publish_lock = threading.Lock(); latest_published = [float('-inf')]
    def publish_memory(value):
        observed = value.get('observed_monotonic', time.monotonic())
        with publish_lock:
            if observed < latest_published[0]:
                return  # Concurrent sampling cannot replace a newer observation.
            write_json(live_memory_path, {'schema': 1, 'kind': 'owned_worker_live_memory_v1',
                'pid': pid, 'ppid': identity['ppid'], 'run_id': run_id, 'worker_identity_sha256': identity_sha,
                'process_creation_identity': identity['process_creation_identity'], **sources,
                'sample': value, 'scope': 'Self-observed telemetry; latest bounded atomic record; no cross-process memory access'},
                retry_seconds=1.)
            latest_published[0] = observed
    publish_memory(identity['telemetry_before_torch_import'])
    import torch
    properties = torch.cuda.get_device_properties(0)
    cap = validate_capacity(properties.total_memory, args.cap_gb)
    telemetry = UpdateTelemetry(pid, sink=publish_memory)
    live_sampler = LiveTelemetrySampler(pid, publish_memory)
    evidence = {'schema': 1, 'kind': 'memory_guard_training_launcher_TSC_resume_v1', **sources, **cap,
                'cuda_device_name': properties.name, 'cap_scope': 'This process only; PyTorch CUDA allocator only',
                'actual_API': 'torch.cuda.set_per_process_memory_fraction(fraction, device=0)',
                'allocator_cap_may_trigger_native_reclaim_or_OOM': True, 'cleanup_threshold_bytes': 4_000_000_000,
                'global_driver_settings_changed': False, 'training_math_changed': False,
                'backend_flags_changed': False, 'RNG_sampler_optimizer_changed': False,
                'worker_pid': pid, 'worker_ppid': identity['ppid'], 'worker_identity_path': str(args.worker_identity.resolve()),
                'worker_identity_sha256': file_sha(args.worker_identity), 'worker_arguments_sha256': argument_sha,
                'live_memory_path': str(live_memory_path.resolve()),
                'live_memory_update_policy': 'Before/after Torch startup; boundedstdlibdaemon every5sec duringcache/DEV/training plus exact optimizer-end samples;1sec atomic-write retry; stop/join finally; observational failures recorded',
                'run_id': run_id, 'cache_provenance_path': str(args.cache_provenance.resolve()),
                'io_provenance_path': str(args.provenance.resolve()), 'started_at_utc': timestamp(), 'status': 'running',
                'telemetry_after_torch_import': safe_sample(pid, 'before_frozen_worker_startup')}
    write_json(args.memory_provenance, evidence)
    publish_memory(evidence['telemetry_after_torch_import'])
    original_class = cache_launcher = None
    try:
        torch.cuda.set_per_process_memory_fraction(cap['per_process_allocator_fraction'], device=0)
        evidence['allocator_cap_applied'] = True
        write_json(args.memory_provenance, evidence)
        import launch_with_cache_reclaim_tsc_v1 as cache_launcher
        original_class = cache_launcher.CudaCacheReclaimer
        cache_launcher.CudaCacheReclaimer = telemetry_reclaimer_class(original_class, telemetry)
        live_sampler.start()
        cache_launcher.main(SimpleNamespace(provenance=args.provenance, cache_provenance=args.cache_provenance,
            threshold_gb=4., max_events=args.max_events, retry_seconds=args.retry_seconds,
            validate_only=False, arguments=args.arguments))
        evidence['status'] = 'complete'
    except BaseException as error:
        evidence.update(status='failed', failure_type=type(error).__name__, failure_message=str(error))
        raise
    finally:
        live_sampler.stop_and_join()
        if original_class is not None:
            cache_launcher.CudaCacheReclaimer = original_class
        evidence.update(finished_at_utc=timestamp(), owned_update_memory_telemetry=telemetry.evidence(),
                        all_phase_owned_memory_telemetry=live_sampler.evidence(),
                        telemetry_after_worker=safe_sample(pid, 'after_frozen_worker_exit'),
                        memory_launcher_sha256_after=file_sha(__file__),
                        cache_launcher_sha256_after=file_sha(OUTPUT/'launch_with_cache_reclaim_tsc_v1.py'),
                        cache_helper_sha256_after=file_sha(OUTPUT/'cuda_cache_reclaim_v2.py'),
                        resource_helper_sha256_after=file_sha(OUTPUT/'resource_watchdog_v2.py'),
                        frozen_worker_sha256_after=file_sha(OUTPUT/'train_lora_batched_tsc_v1.py'))
        try:
            publish_memory(evidence['telemetry_after_worker'])
            evidence['live_memory_sha256_after'] = file_sha(live_memory_path)
        except Exception as error:
            evidence['live_memory_final_publish_failure_type'] = type(error).__name__
        try:
            native_adapter = sys.modules.get('native_tdt_checkpointed_v3')
            evidence['native_cleanup_evidence'] = (native_adapter.cache_cleanup_evidence()
                if native_adapter is not None else {'available': False, 'reason': 'Native adapter not imported before worker exit'})
        except Exception as error:
            evidence['native_cleanup_evidence'] = {'available': False, 'failure_type': type(error).__name__,
                'scope': 'Read-only evidence callback; original worker result preserved'}
        for name, expected in pins:
            evidence[name.removesuffix('.py') + '_sha256_after'] = file_sha(OUTPUT / name)
        write_json(args.memory_provenance, evidence)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('provenance', 'cache-provenance', 'memory-provenance', 'worker-identity'):
        result.add_argument('--' + name, type=Path, required=True)
    result.add_argument('--cap-gb', type=float, default=8.)
    result.add_argument('--threshold-gb', type=float, default=4.)
    result.add_argument('--max-events', type=int, default=128)
    result.add_argument('--retry-seconds', type=float, default=30.)
    result.add_argument('--validate-only', action='store_true')
    result.add_argument('arguments', nargs=argparse.REMAINDER)
    return result


if __name__ == '__main__':
    main(parser().parse_args())
