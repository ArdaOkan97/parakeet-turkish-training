"""Prospective whole-worker admission across native, write and restore phases.

Feature preparation is separately admitted before GPU residency. This finite
proposal retains the existing64MB write and350MB restore reserves and makes
no assumption that post-native private memory has fallen before either phase.
"""
from bpe_broad_production_state_v1 import require

WRITE=64_000_000
RESTORE=350_000_000
STARTUP=3_000_000_000

def whole_worker_forecast(native):
    require(native['kind']=='ASR18_finite_measured_geometry_native_forecast_v5'
        and native['prospective_native_success_claimed'] is False and native['production_forecast_approved'] is False,
        'Exact separately proposed native policy required; this cannot grant approval')
    ceiling=native['absolute_private_forecast_bytes']
    require(type(ceiling) is int and ceiling>0,'Positive explicit native admission ceiling')
    whole=max(STARTUP,ceiling+WRITE,ceiling+RESTORE)
    return {'kind':'ASR18_precomputed_entire_worker_phase_forecast_v2','native_private_forecast_bytes':ceiling,
        'startup_private_upper_bound_bytes':STARTUP,'checkpoint_write_increment_bytes':WRITE,'checkpoint_restore_increment_bytes':RESTORE,
        'entire_worker_private_forecast_bytes':whole,'native_then_checkpoint_write_upper_bound_bytes':ceiling+WRITE,
        'native_then_checkpoint_restore_upper_bound_bytes':ceiling+RESTORE,
        'post_native_residency_drop_assumed':False,'CPU_feature_preparation_during_GPU_residency_supported':False,
        'CUDA_allocator_8GB_is_separate_from_worker_private_forecast':True,
        'fresh_system90_three_observations_95critical_2GBheadroom_still_required':True,
        'prospective_native_or_entire_worker_success_claimed':False,'production_forecast_approved':False}

def phase_projection(admitted,*,phase,current_private_bytes,increment_bytes):
    require(admitted['kind']=='ASR18_precomputed_entire_worker_phase_forecast_v2'
        and type(current_private_bytes) is int and current_private_bytes>0 and type(increment_bytes) is int and increment_bytes>0,
        'Available actual private observation and explicit phase allowance required')
    require(phase in ('native','checkpoint_write','checkpoint_restore'),'No guessed unsupported worker phase')
    projected=current_private_bytes+increment_bytes
    required=projected+RESTORE if phase=='native' else projected
    require((phase!='checkpoint_write' or increment_bytes==WRITE)
        and (phase!='checkpoint_restore' or increment_bytes==RESTORE)
        and required<=admitted['entire_worker_private_forecast_bytes'],
        'Phase exceeds full prospective admission including later restore; stop before work, no margin shrink')
    return {'phase':phase,'projected_phase_private_bytes':projected,'required_entire_private_bytes':required,
        'fresh_system_admission_still_required':True,'forecast_is_not_CPU_private_resource_cap':True}
