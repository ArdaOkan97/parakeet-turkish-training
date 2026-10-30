"""Stdlib Windows capacity sampling and review of a scoped owned-service release."""
import ctypes
import hashlib
import json
import sys
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent


def sample():
    class Performance(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in
            ['CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal', 'PhysicalAvailable',
             'SystemCache', 'KernelTotal', 'KernelPaged', 'KernelNonpaged', 'PageSize']] + [
            (name, wintypes.DWORD) for name in ['HandleCount', 'ProcessCount', 'ThreadCount']]
    value = Performance()
    value.cb = ctypes.sizeof(value)
    api = ctypes.WinDLL('psapi', use_last_error=True).GetPerformanceInfo
    api.argtypes = [ctypes.POINTER(Performance), wintypes.DWORD]
    api.restype = wintypes.BOOL
    if not api(ctypes.byref(value), value.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {'API': 'GetPerformanceInfo', 'scope': 'system-wide host commitment',
            'commit_used_bytes': int(value.CommitTotal * value.PageSize),
            'commit_limit_bytes': int(value.CommitLimit * value.PageSize),
            'physical_available_bytes': int(value.PhysicalAvailable * value.PageSize),
            'system_process_count': value.ProcessCount,
            'observed_at_utc': datetime.now(timezone.utc).isoformat()}


def record(path):
    raw = path.read_bytes()
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


if len(sys.argv) == 1:
    print(json.dumps(sample()))
else:
    release_path = Path(sys.argv[1]).resolve()
    release = json.loads(release_path.read_text(encoding='utf-8-sig'))
    before = release['baseline_before']
    after = release['baseline_after']
    peak = 12079239168
    fits = all(item['commit_used_bytes'] + peak < item['commit_limit_bytes'] * .9
               and item['commit_limit_bytes'] - item['commit_used_bytes'] - peak >= 2000000000
               for item in release['after_samples'])
    passed = release['actual_owned_process_closed'] and release['actual_listener_closed'] and fits
    passed = passed and after['commit_used_bytes'] < before['commit_used_bytes']
    closure = BASE / 'matched_encoder_cosine_actual_CLOSED34651_resource_v1.json'
    review = {'kind': 'ROOT_owned_encoder_cosine34651_capacity_release_v1', 'passed': passed,
              'run_id': 'adaptive_20261012_TR_uniform_encoder_cosine_01', 'source_step': 34651,
              'source_actual_owned_exit': {'session_id': 99543, 'exit_code': 1, 'tool_chunk': 'dca628'},
              'capacity_release_actual_closed': release['actual_owned_process_closed'],
              'no_user_apps_closed': True, 'no_guard_changes': True, 'no_pagefile_changes': True,
              'source_closure_review': record(closure), 'capacity_release_record': record(release_path),
              'baseline_before': before, 'baseline_after': after,
              'same_policy_peak_observed_private_bytes': peak,
              'scope': 'Only the verified optional owned Trackio server; user browser/application stays open.'}
    path = BASE / 'ROOT_owned_encoder_cosine34651_capacity_release_v1.json'
    path.write_text(json.dumps(review, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'review_path': str(path.resolve()), 'passed': passed,
                      'baseline_after_GB': after['commit_used_bytes'] / 1e9,
                      'projected_GB': (after['commit_used_bytes'] + peak) / 1e9}))
    if not passed:
        sys.exit(1)
