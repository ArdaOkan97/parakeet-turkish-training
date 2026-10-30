"""Preserved historical full-hard envelope plus entire frozen bank/new-row charge.

New9223 label geometry is independently bound; historical larger U574 is only
the conservative source resource basis. Actual9223 sustained/hard fit is a new
execution requirement, never inferred from that old numerical trajectory.
"""
from asr18_full_population_geometry_forecast_v1 import CalibratedGeometryForecast as Previous,MAX_UPDATES
from asr18_BPE9223_source_context_v2 import validate_CPU_population
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest

BANK=1_217_066_496
NEW_ROWS=32_000_000


class CalibratedGeometryForecast(Previous):
    def __init__(self,*,BPE9223_DATA_evidence,guard=lambda:None,**calibration):
        super().__init__(guard=guard,**calibration)
        v=validate_CPU_population(BPE9223_DATA_evidence,guard)
        self.calibrated_peak+=BANK+NEW_ROWS
        self.high_water=self.calibrated_peak
        self.envelopes['student_tr']['target_tokens']=517
        self.full_scope.update(BPE9223_population=v['new_population'],
            new_student_target_ceiling=517,old_resource_reference_target_ceiling=574,
            full_frozen_CUDA_bank_and_WDDM_private_charge_bytes=BANK,
            appended512_parameters_moments_gradients_and_bounded_metadata_pool_bytes=NEW_ROWS,
            new_checkpoint_tensor_entries=854,host_savings_credited_bytes=0)
    def budget(self,*,rows,current_private_bytes,update_in_process):
        require(type(update_in_process) is int and 1<=update_in_process<=MAX_UPDATES
            and type(current_private_bytes) is int and current_private_bytes>0 and 3<=len(rows)<=16,
            'Exact current full/tail update and fresh private observation')
        for row in rows:
            require(row['role'] in self.envelopes,'Only student or original protected replay')
            env=self.envelopes[row['role']]
            actual={'encoder_frames':row['encoder_frames'],'target_tokens':len(row['full_target_ids']),
                'feature_frames':row['input_features']['shape'][1],'samples':row['samples']}
            require(all(type(x) is int and 0<x<=env[k] for k,x in actual.items()),
                'Full new retained geometry before draw; no crop/drop/fallback after failure')
        if update_in_process==1:
            require(current_private_bytes<=3_000_000_000+BANK+NEW_ROWS,'Entire bank/newrow-inclusive first-native ready bound')
        target=max(self.calibrated_peak+750_000_000+self.replay_teacher_feature_extension_reserve_bytes,
            self.high_water+250_000_000+self.replay_teacher_feature_extension_reserve_bytes)
        increment=max(2_500_000_000 if update_in_process==1 else 250_000_000,target-current_private_bytes)
        return {'kind':'ASR18_BPE9223_full_geometry_native_forecast_v1','source':record(__file__),
            'current_private_bytes':current_private_bytes,'incremental_bytes':increment,
            'absolute_private_forecast_bytes':max(target,current_private_bytes+increment),
            'calibrated_native_peak_bytes':self.calibrated_peak,'running_native_high_water_bytes':self.high_water,
            'warmup_variation_allowance_bytes':750_000_000,'high_water_margin_bytes':250_000_000,
            'maximum_native_updates_in_process':MAX_UPDATES,'complete_geometry_scope':self.full_scope,
            'all_shape_or_long_loop_fit_claimed':False,'prospective_native_success_claimed':False,
            'production_forecast_approved':False}
    def state_dict(self):
        return {'kind':'ASR18_BPE9223_full_geometry_monotonic_forecast_v1','source':record(__file__),
            'complete_geometry_scope':self.full_scope,'geometry_envelopes':self.envelopes,
            'running_native_high_water_bytes':self.high_water,'production_forecast_approved':False}
    def load_state_dict(self,value):
        expected=self.state_dict()
        require(set(value)==set(expected) and digest({k:v for k,v in value.items() if k!='running_native_high_water_bytes'})
            ==digest({k:v for k,v in expected.items() if k!='running_native_high_water_bytes'})
            and type(value['running_native_high_water_bytes']) is int and value['running_native_high_water_bytes']>=self.calibrated_peak,
            'Same source/geometry/resource identity and monotonic current high water')
        self.high_water=max(self.high_water,value['running_native_high_water_bytes'])


def resident_worker_forecast(native):
    require(native['kind']=='ASR18_BPE9223_full_geometry_native_forecast_v1'
        and native['maximum_native_updates_in_process']==MAX_UPDATES,'Exact9223 complete native proposal')
    ceiling=native['absolute_private_forecast_bytes']
    require(type(ceiling) is int and ceiling>=7_933_712_768 and ceiling+64_000_000<8_000_000_000,
        'Entire historical+750M+replay+teacher+bank+newrows, same hard8GB private cap')
    return {'kind':'ASR18_resident_native_write_only_entire_worker_forecast_v1',
        'native_private_forecast_bytes':ceiling,'checkpoint_write_increment_bytes':64_000_000,
        'startup_restore_increment_bytes':350_000_000,'startup_private_upper_bound_bytes':3_000_000_000,
        'entire_worker_private_forecast_bytes':max(ceiling+64_000_000,3_350_000_000),
        'checkpoint_restore_during_resident_training_supported':False,'post_native_private_drop_assumed':False,
        'native_margin_or_high_water_reduced':False,'final_actual_closed_native_then_separate_exact_restore_required':True,
        'CPU_preparation_resident_first_combined_admission_unchanged':True,'production_forecast_approved':False}
