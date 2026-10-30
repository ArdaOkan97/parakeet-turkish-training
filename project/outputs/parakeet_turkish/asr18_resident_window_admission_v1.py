"""Prospective resident-between-windows policy, separate from native fit.

During resident CPU preparation the GPU process must be synchronized, coherent
and idle. Current system commit ALREADY includes its measured private memory;
only the reviewed CPU child allowance is added. A failed combined admission
selects an actual exit/restore fallback; it never changes DATA activity/guards.
"""
import time
from datetime import datetime, timezone
from bpe_broad_production_state_v1 import require


def fresh_pair(sample, *, expected_pid, maximum_age_seconds=2.):
    require(type(expected_pid) is int and expected_pid > 0 and type(sample) is dict
            and sample['paired_sampling_within_one_second'] is True
            and sample['system_memory']['available'] is True
            and sample['worker_private_memory']['available'] is True
            and sample['worker_private_memory']['pid'] == expected_pid,
            'Available fresh paired system/actual-owned-private telemetry required')
    age = (datetime.now(timezone.utc)-datetime.fromisoformat(sample['observed_at_utc'])).total_seconds()
    require(-.1 <= age <= maximum_age_seconds and 0 <= time.monotonic()-sample['observed_monotonic'] <= maximum_age_seconds,
            'Resident admission cannot reuse stale or future telemetry')
    system = sample['system_memory']; private = sample['worker_private_memory']['private_bytes']
    require(type(private) is int and private > 0
            and all(type(system[key]) is int and system[key] >= 0
                    for key in ('commit_used_bytes', 'commit_limit_bytes', 'commit_available_bytes'))
            and abs(system['commit_used_bytes']+system['commit_available_bytes']-system['commit_limit_bytes']) <= 4096,
            'Exact available commit/private byte domains required')
    return system, private


def resident_preparation_admission(sample, *, native_pid, entire_GPU_worker_forecast_bytes,
                                   CPU_private_allowance_bytes, CPU_contract_binding, coherent_idle_verified):
    require(coherent_idle_verified is True and type(entire_GPU_worker_forecast_bytes) is int
            and entire_GPU_worker_forecast_bytes > 0 and type(CPU_private_allowance_bytes) is int
            and 0 < CPU_private_allowance_bytes <= 2_000_000_000
            and type(CPU_contract_binding) is dict
            and set(CPU_contract_binding) == {'path', 'bytes', 'sha256'},
            'Reviewed bounded CPU contract and independently verified coherent idle boundary required')
    system, private = fresh_pair(sample, expected_pid=native_pid)
    remaining = system['commit_available_bytes']-CPU_private_allowance_bytes
    projected = system['commit_used_bytes']+CPU_private_allowance_bytes
    reasons = []
    if private > entire_GPU_worker_forecast_bytes: reasons.append('actual_GPU_private_above_reviewed_whole_worker_forecast')
    if projected >= .90*system['commit_limit_bytes']: reasons.append('combined_CPU_preparation_projection_not_below90percent')
    if remaining <= 2_000_000_000: reasons.append('combined_CPU_preparation_projection_not_above2GB_headroom')
    return {'kind': 'ASR18_resident_GPU_plus_serial_CPU_preparation_admission_v1',
            'resident_preparation_admitted': not reasons,
            'selection': 'keep_coherent_GPU_resident' if not reasons else 'actual_GPU_exit_then_separately_admit_CPU_and_restore',
            'reasons': reasons, 'native_PID': native_pid, 'actual_native_private_bytes': private,
            'reviewed_GPU_entire_worker_forecast_bytes': entire_GPU_worker_forecast_bytes,
            'CPU_private_allowance_bytes': CPU_private_allowance_bytes, 'CPU_contract': CPU_contract_binding,
            'projected_system_commit_bytes': projected, 'projected_system_headroom_bytes': remaining,
            'combined_owned_private_upper_bound_bytes': private+CPU_private_allowance_bytes,
            'reviewed_combined_owned_private_upper_bound_bytes': entire_GPU_worker_forecast_bytes+CPU_private_allowance_bytes,
            'fresh_sample': sample, 'GPU_process_exit_claimed': False,
            'all_guards_unchanged': True, 'DATA_pause_requested': False, 'production_enabled': False}
