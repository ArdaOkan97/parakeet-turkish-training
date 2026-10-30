"""Scoped inactive CUDA-cache reclamation for the frozen trainer's update end.

This helper is for a NEW worker process only. It changes no model, objective,
optimizer, RNG, backend flags or batching. Cache cleanup runs inside the frozen
trainer's measured update interval. CUDA calls from every other caller pass
through untouched. No CUDA operation runs during helper import or validation.
"""
from __future__ import annotations

import ast
from collections import deque
import functools
import hashlib
import sys
import time
from pathlib import Path

FROZEN_TRAINER_SHA256 = 'b32aa1dc09d5ff7dd85be9c309e8d69a095b8cf859b5498136a27d5d5af7cd9d'


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validated_update_sync_line(worker_path, expected_sha=FROZEN_TRAINER_SHA256):
    """Bind the only synchronize call in the original trainer's run function."""
    worker_path = Path(worker_path).resolve()
    if file_sha(worker_path) != expected_sha:
        raise RuntimeError('Frozen trainer source differs from cache-hook expectation')
    tree = ast.parse(worker_path.read_text(encoding='utf-8'))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run']
    if len(functions) != 1:
        raise RuntimeError('Expected exactly one frozen training run function')
    calls = []
    for node in ast.walk(functions[0]):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'synchronize' and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == 'cuda' and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == 'torch'):
            calls.append(node)
    if len(calls) != 1 or calls[0].args or calls[0].keywords:
        raise RuntimeError('Expected one argument-free optimizer-end CUDA synchronize call')
    # Independent parent/source review binds this exact optimizer-end location.
    if calls[0].lineno != 539:
        raise RuntimeError('Frozen optimizer-end call line differs')
    return calls[0].lineno


class CudaCacheReclaimer:
    def __init__(self, cuda, worker_path, *, threshold_gb=8., max_events=128,
                 expected_sha=FROZEN_TRAINER_SHA256):
        if not 0 < threshold_gb <= 12 or not 1 <= max_events <= 1024:
            raise ValueError('Cache threshold/events must be conservatively bounded')
        self.cuda = cuda
        self.worker_path = Path(worker_path).resolve()
        self.expected_sha = expected_sha
        self.line = validated_update_sync_line(self.worker_path, expected_sha)
        self.threshold_bytes = int(threshold_gb * 1e9)
        self.max_events = max_events
        self.events = deque(maxlen=max_events)
        self.eligible_calls = self.ignored_calls = self.reclamations = 0
        self.failed_reclamations = 0
        self.cleanup_seconds = 0.
        self.original = self.wrapped = None

    @staticmethod
    def _caller():
        # _caller -> wrapped synchronize -> original worker call site.
        return sys._getframe(2)

    def _eligible(self, frame):
        return (Path(frame.f_code.co_filename).resolve() == self.worker_path
                and frame.f_code.co_name == 'run' and frame.f_lineno == self.line)

    def _stats(self):
        return {'allocated_bytes': int(self.cuda.memory_allocated()),
                'reserved_bytes': int(self.cuda.memory_reserved()),
                'peak_allocated_bytes': int(self.cuda.max_memory_allocated())}

    def install(self):
        if self.original is not None:
            raise RuntimeError('Cache hook is already installed')
        original = self.cuda.synchronize
        self.original = original

        @functools.wraps(original)
        def synchronize(*arguments, **keywords):
            # Original return and original exception behavior are preserved.
            result = original(*arguments, **keywords)
            if not self._eligible(self._caller()):
                self.ignored_calls += 1
                return result
            self.eligible_calls += 1
            before = self._stats()
            if before['reserved_bytes'] <= self.threshold_bytes:
                return result
            began = time.perf_counter()
            event = {'eligible_update_call': self.eligible_calls, 'before': before,
                     'worker_sha256': self.expected_sha, 'caller_function': 'run',
                     'caller_line': self.line}
            try:
                self.cuda.empty_cache()
                event.update(status='complete', after=self._stats())
                self.reclamations += 1
            except BaseException as exception:
                self.failed_reclamations += 1
                event.update(status='failed', failure_type=type(exception).__name__,
                             failure_message=str(exception))
                raise
            finally:
                duration = time.perf_counter()-began
                event['cleanup_seconds'] = duration
                self.cleanup_seconds += duration
                self.events.append(event)
            return result

        self.wrapped = synchronize
        self.cuda.synchronize = synchronize
        return self

    def restore(self):
        if self.original is None:
            return
        if self.cuda.synchronize is not self.wrapped:
            raise RuntimeError('Scoped CUDA synchronize hook was unexpectedly replaced')
        self.cuda.synchronize = self.original
        self.original = self.wrapped = None

    def evidence(self):
        return {'kind': 'cuda_cache_reclaim_v1', 'worker_path': str(self.worker_path),
                'worker_sha256': self.expected_sha, 'worker_sha256_after': file_sha(self.worker_path),
                'caller_function': 'run', 'caller_line': self.line,
                'threshold_reserved_bytes': self.threshold_bytes,
                'eligible_update_calls': self.eligible_calls, 'ignored_other_calls': self.ignored_calls,
                'reclamations': self.reclamations, 'failed_reclamations': self.failed_reclamations,
                'cleanup_seconds_in_training_timing': self.cleanup_seconds,
                'event_capacity': self.max_events, 'events_omitted': max(0, self.reclamations+self.failed_reclamations-len(self.events)),
                'events': list(self.events), 'changes_training_math': False,
                'changes_backend_flags': False, 'changes_rng_or_sampler': False,
                'changes_optimizer_state': False, 'changes_cuda_allocator_policy': False,
                'memory_management_changed': True,
                'scope': 'Only unused CUDA allocator cache at the exact frozen optimizer-end synchronize call; measured update timing includes cleanup.'}
