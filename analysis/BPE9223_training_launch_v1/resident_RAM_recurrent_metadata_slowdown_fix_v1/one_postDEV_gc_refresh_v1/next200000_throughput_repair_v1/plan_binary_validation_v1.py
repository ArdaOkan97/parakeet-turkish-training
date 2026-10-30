"""INACTIVE bounded complete-plan witness; binary bytes are never unpickled.

Every original boundary still inspects the ENTIRE current plan. A fresh strict C
Pickler is only an equality witness for the original canonical SHA. Uncertain,
changed or unsupported input takes the original canonical path, not a cached
mutable-object hash. The cache retains no scientific object or Pickler/memo.
"""
import _pickle
import hashlib
import math
import sys
import threading
import time

EXTRA_RESERVATION_BYTES = 64 * 1024**2
MAX_WIRE_BYTES = 2 * 1024**2
MAX_VISITS = 262144
MAX_CONTAINER_ITEMS = 65536
MAX_STRING_CODEPOINTS = 131072
MAX_INTEGER_BITS = 4096
MAX_ORIGINAL_RECURSION_LIMIT = 1000
PROTOCOL = 4


class UnsupportedBinaryDomain(ValueError):
    """Expected unsupported domain, before any user reduction is invoked."""


class BinaryCapacity(ValueError):
    """Expected local eligibility/capacity limit; not a resource guard denial."""


class _BoundedWriter:
    __slots__ = ('_parts', 'size', 'closed')

    def __init__(self):
        self._parts = []
        self.size = 0
        self.closed = False

    def write(self, raw):
        if self.closed or type(raw) is not bytes:
            raise BinaryCapacity('Only live bounded C-Pickler byte writes')
        size = self.size + len(raw)
        if size > MAX_WIRE_BYTES:
            raise BinaryCapacity('Complete binary wire exceeds local eligibility bound')
        self._parts.append(raw)
        self.size = size
        return len(raw)

    def finish(self):
        return b''.join(self._parts)

    def close(self):
        self._parts.clear()
        self.closed = True


class _StrictPickler(_pickle.Pickler):
    """persistent_id runs BEFORE builtin fast paths; always returns None.

    This is an admission hook, never a persisted scientific persistent ID.
    reducer_override is a second refusal barrier, never custom reduction.
    C recursion uses the unchanged original interpreter bound, not a new limit.
    """
    def __init__(self, writer):
        self.visits = 0
        super().__init__(writer, protocol=PROTOCOL, fix_imports=False)

    def persistent_id(self, value):
        self.visits += 1
        if self.visits > MAX_VISITS:
            raise BinaryCapacity('Full typed visit bound')
        kind = type(value)
        if kind is dict or kind is list or kind is tuple:
            if len(value) > MAX_CONTAINER_ITEMS:
                raise BinaryCapacity('Exact builtin container-size bound')
        elif kind is str:
            if len(value) > MAX_STRING_CODEPOINTS:
                raise BinaryCapacity('Exact Unicode scalar-size bound before C encoding')
        elif kind is int:
            if int.bit_length(value) > MAX_INTEGER_BITS:
                raise BinaryCapacity('Exact integer-size bound before C encoding')
        elif kind is float:
            if not math.isfinite(value):
                raise UnsupportedBinaryDomain('Nonfinite float uses original canonical semantics')
        elif value is None or kind is bool:
            pass
        else:
            # Do not format value/type names: even a custom metaclass is untrusted.
            raise UnsupportedBinaryDomain('Only exact JSON builtins and original tuple representation')
        return None

    def reducer_override(self, value):
        raise UnsupportedBinaryDomain('Custom reduction is never invoked')


def complete_binary(value):
    """Fresh writer/Pickler/memo per call, all references dropped on ALL paths.

    The source protocol requires CPython3.11's original recursion limit <=1000.
    MAX_VISITS bounds memo insertions as well as visits; <=262144 insertions,
    conservative <=16MiB C table allowance under pinned CPython growth policy.
    Wire/parts/join/baseline <=8MiB; Unicode transient/cache <=4MiB; recursion,
    counters, object headers and headroom are covered by the separate64MiB.
    This bound must be proved by the frozen CPU fixture before integration.
    """
    if sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 11):
        raise UnsupportedBinaryDomain('Unproved interpreter uses original canonical path')
    if sys.getrecursionlimit() > MAX_ORIGINAL_RECURSION_LIMIT:
        raise UnsupportedBinaryDomain('Unproved original recursion limit; never change it')
    writer = _BoundedWriter()
    pickler = None
    try:
        pickler = _StrictPickler(writer)
        pickler.dump(value)
        raw = writer.finish()
        return raw, pickler.visits
    finally:
        if pickler is not None:
            # Explicitly release scientific objects even if dump raised. No GC call.
            pickler.memo.clear()
        pickler = None
        writer.close()


def _empty_cost():
    return {'count': 0, 'wall_seconds': 0.0, 'wall_max_seconds': 0.0,
            'thread_CPU_seconds': 0.0, 'thread_CPU_max_seconds': 0.0,
            'wall_observations': 0, 'thread_CPU_observations': 0,
            'raised': 0, 'clock_errors': 0, 'bytes': 0, 'visits': 0}


class PlanBinaryWitness:
    """ONE plan lifetime, MainThread only, no baseline refresh on mismatch."""
    __slots__ = ('_canonical', '_baseline', '_sha', '_started', '_owner',
                 '_wall', '_cpu', '_costs', '_counts', '_closed')

    def __init__(self, canonical, *, admitted_extra_reservation_bytes,
                 wall_clock=time.perf_counter, thread_clock=time.thread_time):
        if admitted_extra_reservation_bytes != EXTRA_RESERVATION_BYTES:
            raise ValueError('Distinct full64MiB reservation required before witness allocations')
        if not callable(canonical):
            raise ValueError('Original source-bound canonical callable')
        self._canonical = canonical
        self._baseline = None
        self._sha = None
        self._started = False
        self._owner = threading.get_ident()
        self._wall, self._cpu = wall_clock, thread_clock
        self._costs = {label: _empty_cost() for label in ('original_baseline', 'binary_validation', 'original_fallback')}
        self._counts = {label: 0 for label in ('baseline_eligible', 'baseline_unsupported', 'baseline_capacity',
            'baseline_recursion', 'binary_equal', 'binary_mismatch', 'binary_unsupported',
            'binary_capacity', 'binary_recursion', 'uncached_original')}
        self._closed = False

    def _start(self, label):
        cost = self._costs[label]
        cost['count'] += 1
        values = []
        for clock in (self._wall, self._cpu):
            try:
                value = clock()
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError('Invalid clock scalar')
                values.append(value)
            except Exception:
                cost['clock_errors'] += 1
                values.append(None)
        return tuple(values)

    def _finish(self, label, start, raised, byte_count=0, visits=0):
        cost = self._costs[label]
        cost['raised'] += bool(raised)
        cost['bytes'] += byte_count
        cost['visits'] += visits
        for index, (clock, total, maximum, observed) in enumerate((
                (self._wall, 'wall_seconds', 'wall_max_seconds', 'wall_observations'),
                (self._cpu, 'thread_CPU_seconds', 'thread_CPU_max_seconds', 'thread_CPU_observations'))):
            try:
                end = clock()
                if start[index] is None or type(end) not in (int, float) or not math.isfinite(end) or end < start[index]:
                    raise ValueError('Invalid clock interval')
                elapsed = end - start[index]
                cost[total] += elapsed
                cost[maximum] = max(cost[maximum], elapsed)
                cost[observed] += 1
            except Exception:
                cost['clock_errors'] += 1

    def _original(self, value, label):
        stamp = self._start(label)
        raised, count = True, 0
        try:
            raw = self._canonical(value)
            count = len(raw)
            result = hashlib.sha256(raw).hexdigest()
            raised = False
            return result
        finally:
            # No exception/frame/value retained in the statistics.
            self._finish(label, stamp, raised, count)

    def _binary(self, value):
        stamp = self._start('binary_validation')
        raised, count, visits = True, 0, 0
        try:
            raw, visits = complete_binary(value)
            count = len(raw)
            raised = False
            return raw
        finally:
            self._finish('binary_validation', stamp, raised, count, visits)

    def digest(self, value):
        if self._closed or threading.get_ident() != self._owner or threading.current_thread() is not threading.main_thread():
            raise ValueError('Live MainThread-only per-plan witness')
        if not self._started:
            # Original canonical runs FIRST, exactly once. Original exceptions
            # propagate unchanged outside any binary eligibility catch.
            result = self._original(value, 'original_baseline')
            self._started = True
            try:
                baseline = self._binary(value)
            except UnsupportedBinaryDomain:
                self._counts['baseline_unsupported'] += 1
            except BinaryCapacity:
                self._counts['baseline_capacity'] += 1
            except RecursionError:
                self._counts['baseline_recursion'] += 1
            else:
                self._baseline, self._sha = baseline, result
                self._counts['baseline_eligible'] += 1
            return result
        if self._baseline is None:
            self._counts['uncached_original'] += 1
            return self._original(value, 'original_fallback')
        try:
            current = self._binary(value)
        except UnsupportedBinaryDomain:
            self._counts['binary_unsupported'] += 1
        except BinaryCapacity:
            self._counts['binary_capacity'] += 1
        except RecursionError:
            self._counts['binary_recursion'] += 1
        else:
            if current == self._baseline:
                self._counts['binary_equal'] += 1
                return self._sha
            self._counts['binary_mismatch'] += 1
        # Original canonical failure identity is never caught/replaced.
        return self._original(value, 'original_fallback')

    def snapshot_and_reset(self):
        if threading.get_ident() != self._owner:
            raise ValueError('MainThread-only scalar observation')
        result = {'costs': self._costs, 'counts': self._counts,
                  'baseline_binary_bytes_retained': 0 if self._baseline is None else len(self._baseline),
                  'full_extra_reservation_bytes': EXTRA_RESERVATION_BYTES,
                  'wall_and_threadCPU_are_distinct_not_exclusive_decomposition': True}
        self._costs = {label: _empty_cost() for label in self._costs}
        self._counts = {label: 0 for label in self._counts}
        return result

    def close(self):
        if threading.get_ident() != self._owner:
            raise ValueError('Original witness owner closes it')
        self._baseline = None
        self._sha = None
        self._closed = True

