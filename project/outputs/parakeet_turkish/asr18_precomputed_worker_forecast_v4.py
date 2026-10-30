"""Whole-worker/resident policy binding the explicit finite40 native successor."""
from asr18_precomputed_worker_forecast_v3 import whole_worker_forecast as previous_whole
from asr18_precomputed_worker_forecast_v3 import phase_projection as previous_phase
from bpe_broad_production_state_v1 import require


def whole_worker_forecast(native):
    require(native['kind'] == 'ASR18_finite_measured_geometry_native_forecast_v6'
            and native['maximum_native_updates_in_process'] == 40, 'Exact finite40 prospective native policy required')
    old = {**native, 'kind': 'ASR18_finite_measured_geometry_native_forecast_v5'}
    result = previous_whole(old)
    result.update(kind='ASR18_precomputed_entire_worker_phase_forecast_v4',
                  finite_native_policy='ASR18_finite_measured_geometry_native_forecast_v6')
    return result


def phase_projection(admitted, *, phase, current_private_bytes, increment_bytes):
    require(admitted['kind'] == 'ASR18_precomputed_entire_worker_phase_forecast_v4', 'Exact whole-worker finite40 policy')
    return previous_phase({**admitted, 'kind': 'ASR18_precomputed_entire_worker_phase_forecast_v3'},
        phase=phase, current_private_bytes=current_private_bytes, increment_bytes=increment_bytes)
