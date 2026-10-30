"""Fresh resource-only forecast immediately before one finalization command.

This is not a reusable permit. The caller must run it anew immediately before
each new job. Native jobs retain their own unchanged hard/private/system guards.
"""
import argparse
import os
from pathlib import Path
import sys
import time

from ASR18_continuous_data_guard_v5 import DataIOGuard, AREA
from cpu_preparation_guard_v2 import memory, now
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import read_json, write_json

EXPECTED_PEAK = {'catalog_finalize': 1_000_000_000, 'labels_finalize': 128_000_000,
    'source_inventory': 64_000_000, 'carrier_coverage': 750_000_000,
    'declared_geometry': 128_000_000, 'two_pass_metadata': 64_000_000}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=tuple(EXPECTED_PEAK), required=True)
    args = parser.parse_args()
    assert all(os.environ.get(key) == '1' for key in (
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
    guard = DataIOGuard('fresh_finalization_forecast_' + args.role, heavy=False, maximum_bytes=16_000_000)
    plan = read_json(AREA / 'preparation_plan_v1.json')
    missing = sum(job['size'] for job in plan['download_jobs'] if not Path(job['destination']).exists())
    expected = EXPECTED_PEAK[args.role]
    reserved = expected + 250_000_000
    started, fresh = time.monotonic(), 0
    while time.monotonic() - started < 3600:
        guard.wait(reserve=missing + 2_000_000_000)
        state = memory()
        limit = state['commit_headroom_bytes'] / (1 - state['commit_ratio'])
        projected = state['commit_ratio'] + reserved / limit
        admitted = projected < .90 and state['commit_headroom_bytes'] - reserved >= 2_000_000_000
        fresh = fresh + 1 if admitted else 0
        value = dict(kind='ASR18_immediate_resource_only_finalization_forecast_v1', owner='/root/cpu_data_preparation',
            at_utc=now(), PID=guard.process.pid, role=args.role, current_memory=state,
            expected_incremental_job_peak_private_bytes=expected, additional_margin_bytes=250_000_000,
            reserved_incremental_bytes=reserved, projected_commit_ratio=projected,
            projected_headroom_bytes=state['commit_headroom_bytes']-reserved,
            reserved_missing_download_bytes=missing, reserved_additional_temporary_disk_bytes=2_000_000_000,
            consecutive_fresh_samples=fresh, admitted=admitted and fresh >= 3,
            reusable_permit=False, GPU_idle_permission_required=False,
            original_job_CPU_private_cap_bytes=2_000_000_000, TRAIN_admitted=False)
        write_json(AREA / ('immediate_' + args.role + '_resource_forecast_v1.json'), value)
        if fresh >= 3:
            print(value, flush=True)
            return
        time.sleep(.1 if admitted else .25)
    raise RuntimeError('Finite resource admission window elapsed; no new finalization job started')


if __name__ == '__main__':
    main()
