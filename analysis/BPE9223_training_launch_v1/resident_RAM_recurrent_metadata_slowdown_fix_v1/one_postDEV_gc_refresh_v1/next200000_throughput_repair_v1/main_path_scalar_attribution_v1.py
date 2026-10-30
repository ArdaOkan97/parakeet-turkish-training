"""INACTIVE scalar clocks for unchanged inline original main-path operations.
No operation callable, payload, traceback, exception or frame is retained.
"""
import math
import threading
import time

LABELS = ('host_gate', 'digest_readiness', 'readiness_wait', 'digest_take',
          'packet_take', 'digest_binding', 'authoritative_propose',
          'metadata_binding', 'native_and_commit', 'archive_drop_release', 'progress_emit')


def _blank():
    return {'count': 0, 'wall_seconds': 0.0, 'wall_max_seconds': 0.0,
            'thread_CPU_seconds': 0.0, 'thread_CPU_max_seconds': 0.0,
            'wall_observations': 0, 'thread_CPU_observations': 0,
            'raised': 0, 'clock_errors': 0, 'nested_calls': 0}


class MainPathScalarAttribution:
    __slots__ = ('_owner', '_wall', '_cpu', '_costs', '_active', '_sequence', '_preparation')

    def __init__(self, *, wall_clock=time.perf_counter, thread_clock=time.thread_time):
        self._owner = threading.get_ident()
        self._wall, self._cpu = wall_clock, thread_clock
        self._costs = {label: _blank() for label in LABELS}
        self._active = []  # <=11 live integer tokens, never a history or payload.
        self._sequence = 0
        self._preparation = {'count': 0, 'seconds': 0.0, 'max_seconds': 0.0}

    def start(self, label):
        if threading.get_ident() != self._owner or threading.current_thread() is not threading.main_thread():
            raise ValueError('Only original MainThread phases')
        if label not in self._costs or len(self._active) >= len(LABELS):
            raise ValueError('Fixed coarse labels and finite active depth')
        cost = self._costs[label]
        cost['count'] += 1
        cost['nested_calls'] += bool(self._active)
        values = []
        for clock in (self._wall, self._cpu):
            try:
                value = clock()
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError('Invalid clock')
                values.append(value)
            except Exception:
                cost['clock_errors'] += 1
                values.append(None)
        self._sequence += 1
        self._active.append(self._sequence)
        return label, self._sequence, values[0], values[1]

    def finish(self, stamp, *, raised):
        label, sequence, wall, cpu = stamp
        cost = self._costs[label]
        cost['raised'] += bool(raised)
        if threading.get_ident() != self._owner or not self._active or self._active[-1] != sequence:
            cost['clock_errors'] += 1
            return  # Observation cannot mask the original operation's exception.
        self._active.pop()
        for clock, start, total, maximum, observed in (
                (self._wall, wall, 'wall_seconds', 'wall_max_seconds', 'wall_observations'),
                (self._cpu, cpu, 'thread_CPU_seconds', 'thread_CPU_max_seconds', 'thread_CPU_observations')):
            try:
                end = clock()
                if start is None or type(end) not in (int, float) or not math.isfinite(end) or end < start:
                    raise ValueError('Invalid clock interval')
                elapsed = end - start
                cost[total] += elapsed
                cost[maximum] = max(cost[maximum], elapsed)
                cost[observed] += 1
            except Exception:
                cost['clock_errors'] += 1

    def copy_preparation_scalar(self, seconds):
        if threading.get_ident() != self._owner or threading.current_thread() is not threading.main_thread():
            raise ValueError('Only original MainThread copies existing packet scalars')
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Existing finite nonnegative preparation_seconds scalar only')
        self._preparation['count'] += 1
        self._preparation['seconds'] += seconds
        self._preparation['max_seconds'] = max(self._preparation['max_seconds'], seconds)

    def snapshot_and_reset(self):
        if threading.get_ident() != self._owner or self._active:
            raise ValueError('Only completed MainThread window scalar summary')
        result = {'phases': self._costs, 'existing_packet_preparation_seconds': self._preparation,
                  'producer_preparation_excludes_credit_acquire_and_send': True,
                  'nested_wall_not_addable_or_exclusive_CPU_by_function': True}
        self._costs = {label: _blank() for label in LABELS}
        self._preparation = {'count': 0, 'seconds': 0.0, 'max_seconds': 0.0}
        return result

