"""Unadmitted workload separation; existing native/write/restore margins intact.

Resident training performs native + write/read-only persistence phases only.
All exact checkpoint restores happen in a distinct final restore-only process,
or as a startup restore after measured resident admission forces a real exit.
This policy cannot make an old worker with in-residency restores fit.
"""
from bpe_broad_production_state_v1 import require

NATIVE_KIND = 'ASR18_finite_measured_geometry_native_forecast_v6'
WRITE = 64_000_000
RESTORE = 350_000_000
STARTUP = 3_000_000_000


def resident_worker_forecast(native):
    require(native['kind'] == NATIVE_KIND and native['maximum_native_updates_in_process'] == 40
            and native['prospective_native_success_claimed'] is False
            and native['production_forecast_approved'] is False,
            'Exact unchanged finite40 native policy; no reduced native margin')
    ceiling = native['absolute_private_forecast_bytes']
    require(type(ceiling) is int and ceiling > 0, 'Explicit original native ceiling')
    return {'kind':'ASR18_resident_native_write_only_entire_worker_forecast_v1',
        'native_private_forecast_bytes':ceiling,'checkpoint_write_increment_bytes':WRITE,
        'startup_restore_increment_bytes':RESTORE,'startup_private_upper_bound_bytes':STARTUP,
        'entire_worker_private_forecast_bytes':max(ceiling+WRITE,STARTUP+RESTORE),
        'checkpoint_restore_during_resident_training_supported':False,
        'post_native_private_drop_assumed':False,'native_margin_or_high_water_reduced':False,
        'final_actual_closed_native_then_separate_exact_restore_required':True,
        'CPU_preparation_resident_first_combined_admission_unchanged':True,
        'production_forecast_approved':False}


def restore_only_worker_forecast():
    return {'kind':'ASR18_separate_final_restore_only_worker_forecast_v1',
        'entire_worker_private_forecast_bytes':STARTUP+RESTORE,
        'startup_private_upper_bound_bytes':STARTUP,'checkpoint_restore_increment_bytes':RESTORE,
        'native_training_updates_allowed':0,'frontend_or_augmentation_allowed':False,
        'previous_native_process_actual_exit_required':True,
        'production_forecast_approved':False}


def phase_projection(admitted, *, phase, current_private_bytes, increment_bytes):
    require(type(current_private_bytes) is int and current_private_bytes > 0
            and type(increment_bytes) is int and increment_bytes > 0,
            'Fresh private and explicit unchanged phase increment required')
    kind = admitted['kind']
    if kind == 'ASR18_resident_native_write_only_entire_worker_forecast_v1':
        require(phase in ('native','checkpoint_write','startup_restore'),
                'Resident exact restore is unsupported; it must run in a separate final worker')
        if phase == 'native':
            required = current_private_bytes+increment_bytes+WRITE
        elif phase == 'checkpoint_write':
            require(increment_bytes == WRITE, 'Keep complete64MB writer/persistence increment')
            required = current_private_bytes+WRITE
        else:
            require(current_private_bytes <= STARTUP and increment_bytes == RESTORE,
                    'A measured fallback restore may only occur before native residency')
            required = current_private_bytes+RESTORE
    else:
        require(kind == 'ASR18_separate_final_restore_only_worker_forecast_v1'
                and phase == 'checkpoint_restore' and current_private_bytes <= STARTUP
                and increment_bytes == RESTORE, 'Separate restore-only exact350MB stage')
        required = current_private_bytes+RESTORE
    require(required <= admitted['entire_worker_private_forecast_bytes'],
            'Unchanged phase allowance exceeds supported whole-worker workload bound')
    return {'phase':phase,'projected_phase_private_bytes':current_private_bytes+increment_bytes,
        'required_entire_private_bytes':required,'whole_worker_private_forecast_bytes':admitted['entire_worker_private_forecast_bytes'],
        'fresh_system_and_actual_phase_guard_still_required':True,
        'no_native_write_or_restore_increment_reduced':True}
