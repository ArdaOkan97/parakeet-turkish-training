"""Whole candidate for native/persistence, with fresh recovery in another worker."""
from bpe_broad_production_state_v1 import require
from asr18_full_population_geometry_forecast_v1 import MAX_UPDATES
from asr18_separated_worker_forecast_v1 import phase_projection,restore_only_worker_forecast

def resident_worker_forecast(native):
    require(native['kind']=='ASR18_full_retained_geometry_native_forecast_candidate_v1'
            and native['maximum_native_updates_in_process']==MAX_UPDATES
            and native['prospective_native_success_claimed'] is False
            and native['production_forecast_approved'] is False,'Exact new all-retained candidate, no smaller margin')
    ceiling=native['absolute_private_forecast_bytes']
    require(type(ceiling) is int and ceiling>=6684646272 and ceiling+64000000<8000000000,
            'Historical5.726646272GB +750MBvariation +128MBreplay +80MBteacher, writer separate; unchanged8GB guard')
    return {'kind':'ASR18_resident_native_write_only_entire_worker_forecast_v1',
        'native_private_forecast_bytes':ceiling,'checkpoint_write_increment_bytes':64000000,
        'startup_restore_increment_bytes':350000000,'startup_private_upper_bound_bytes':3000000000,
        'entire_worker_private_forecast_bytes':max(ceiling+64000000,3350000000),
        'checkpoint_restore_during_resident_training_supported':False,'post_native_private_drop_assumed':False,
        'native_margin_or_high_water_reduced':False,'final_actual_closed_native_then_separate_exact_restore_required':True,
        'CPU_preparation_resident_first_combined_admission_unchanged':True,'production_forecast_approved':False}
