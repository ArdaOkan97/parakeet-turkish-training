"""Entire GPU-worker envelope plus conditionally admitted idle CPU preparation.

Native/write/restore mathematics is unchanged from v2. A resident idle phase
adds a separately pinned CPU-child allowance to fresh SYSTEM commit; it does
not infer a GPU-private drop, consume a draw, or allow CPU work during native.
"""
from asr18_precomputed_worker_forecast_v2 import whole_worker_forecast as previous_whole
from asr18_precomputed_worker_forecast_v2 import phase_projection as previous_phase
from bpe_broad_production_state_v1 import require


def whole_worker_forecast(native):
    result = previous_whole(native)
    result.update(kind='ASR18_precomputed_entire_worker_phase_forecast_v3',
        CPU_feature_preparation_during_GPU_residency_supported='only_freshly_admitted_source_bound_coherent_idle',
        resident_preference=True, concurrent_CPU_preparation_requires_separate_combined_admission=True,
        fallback_requires_actual_GPU_exit_and_fresh_full_forecast_before_restore=True)
    return result


def phase_projection(admitted, *, phase, current_private_bytes, increment_bytes):
    require(admitted['kind'] == 'ASR18_precomputed_entire_worker_phase_forecast_v3', 'Exact whole-worker policy')
    previous = {**admitted, 'kind': 'ASR18_precomputed_entire_worker_phase_forecast_v2'}
    return previous_phase(previous, phase=phase, current_private_bytes=current_private_bytes, increment_bytes=increment_bytes)
