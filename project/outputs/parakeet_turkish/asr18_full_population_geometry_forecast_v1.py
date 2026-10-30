"""Conservative source-bound all-retained geometry admission candidate.

The older full hard native peak is retained without CNN credit. This policy
does not prove all-shape/long-loop fit; fresh admission and coherent stops are
required before each real update. No crop/drop/shortened targets are allowed.
"""
import json
from pathlib import Path
from asr18_calibrated_geometry_forecast_v6 import CalibratedGeometryForecast as Previous
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from fastmss_phase_aware_plan_bridge_v6 import actual0,read_bound
from tdt_full_native_allocation_budget_v1 import allocation_budget

MAX_UPDATES=2*((2631115+13)//14)+(352768+13)//14
HISTORICAL_HARD_NATIVE=5_726_646_272
TEACHER_FIXED_ALLOWANCE=80_000_000
REPLAY_EXTENSION=128_000_000

class CalibratedGeometryForecast(Previous):
    def __init__(self,*,final_geometry,final_geometry_actual0,previous_full_hard_actual0,
                 previous_full_hard_report,conv_only_hard_summary,guard=lambda:None,**calibration):
        super().__init__(guard=guard,**calibration)
        geometry=read_bound(final_geometry,guard)
        actual0(final_geometry_actual0,final_geometry,guard,owner='/root/cpu_data_preparation')
        p=geometry['population']
        require(geometry['complete'] is True and geometry['TRAIN_admitted'] is False
                and p['retained_rows']==2631115 and p['total_target_tokens']==101470757
                and p['MAX_T']==1991 and p['MAX_U']==574 and p['MAX_T_times_U_plus_1']==1144825
                and p['MAX_clean16k_samples']==2547840
                and all(type(value) is int and value==0 for value in geometry['unknown_invalid_counts'].values()),
                'Genuine final unfiltered scalar envelope, never a native-fit certificate')
        old=read_bound(previous_full_hard_actual0,guard);native=read_bound(previous_full_hard_report,guard)
        require(old['kind']=='ASR18_owned_actual_tool_closure_v1' and old['owner']=='/root/memory_diagnosis'
                and type(old['actual_exit_code']) is int and old['actual_exit_code']==0
                and old['actual_tool_chunk_id']=='4790ca' and old['native_and_state_report']==previous_full_hard_report
                and native['history'][0]['full_T']==1991 and native['history'][0]['full_U']==574
                and native['history'][0]['native_memory']['private_peak_bytes']==HISTORICAL_HARD_NATIVE,
                'Keep actual historical full hard native peak; no restore or component subtraction')
        hard=read_bound(conv_only_hard_summary,guard)
        require(hard['kind']=='ASR18_single_full_hard_TR207_actual_CLOSED0_summary_v1'
                and type(hard['actual_tool_exit_code']) is int and hard['actual_tool_exit_code']==0
                and hard['actual_tool_chunk_id']=='719ae2' and hard['actual_updates']==1
                and hard['full_encoder_frames']==1991 and hard['full_target_tokens']==574
                and hard['whole_audio_samples']==2547840
                and hard['all207_threeAdam_all4RNG_and_source_snapshot_exact'] is True,
                'Prior complete conv-only hard proof remains its narrower source/lifecycle scope')
        for key in ('overall','positive_frontier','checkpoint','PRE_save','fresh_recovery'):
            require(record(hard[key]['path'])==hard[key],'Immutable prior hard proof changed')
        self.calibrated_peak=HISTORICAL_HARD_NATIVE;self.high_water=max(self.high_water,self.calibrated_peak)
        self.envelopes['student_tr']={'encoder_frames':1991,'target_tokens':574,'feature_frames':15925,'samples':2547840}
        # Include actual full cleaner maxima independently of broad declarations.
        require(362<1991 and 164<574,'Original cleaner scalar envelope remains inside the admitted student bounds')
        self.replay_teacher_feature_extension_reserve_bytes=REPLAY_EXTENSION+TEACHER_FIXED_ALLOWANCE
        full_joint=allocation_budget(1991,574,projected_elements=(1991+574+1)*640,head_elements=8716*641,hidden_width=640)
        require(full_joint['total_budget_bytes']==791167592 and full_joint['total_budget_bytes']<1700000000,
                'Correct complete v7 DP/joint union, separate from encoder/private fit')
        self.full_scope={'final_geometry':final_geometry,'final_geometry_actual0':final_geometry_actual0,
            'historical_full_hard_actual0':previous_full_hard_actual0,'historical_full_hard_report':previous_full_hard_report,
            'prior_conv_only_hard_summary':conv_only_hard_summary,'full_v7_joint_bytes':791167592,
            'teacher_fixed_additional_allowance_bytes':TEACHER_FIXED_ALLOWANCE,'replay_extension_allowance_bytes':REPLAY_EXTENSION,
            'CNN_private_credit_subtracted_bytes':0,'new_effective16_current_CPU_and_native_hard_proof_still_required':True}
    def budget(self,*,rows,current_private_bytes,update_in_process):
        require(type(update_in_process) is int and 1<=update_in_process<=MAX_UPDATES,
                'Finite declared scientific schedule; no hidden extra updates')
        value=super().budget(rows=rows,current_private_bytes=current_private_bytes,
                             update_in_process=1 if update_in_process==1 else 2)
        value.update(kind='ASR18_full_retained_geometry_native_forecast_candidate_v1',source=record(__file__),
            actual_update_index_in_this_process=update_in_process,maximum_native_updates_in_process=MAX_UPDATES,
            complete_geometry_scope=self.full_scope,all_shape_or_long_loop_fit_claimed=False,
            production_forecast_approved=False)
        return value
    def state_dict(self):
        value=super().state_dict()
        value.update(kind='ASR18_full_retained_geometry_high_water_candidate_v1',source=record(__file__),
                     complete_geometry_scope=self.full_scope)
        return value
    def load_state_dict(self,state):
        expected=self.state_dict()
        require(type(state) is dict and set(state)==set(expected)
                and digest({k:v for k,v in state.items() if k!='running_native_high_water_bytes'})
                    ==digest({k:v for k,v in expected.items() if k!='running_native_high_water_bytes'})
                and type(state['running_native_high_water_bytes']) is int
                and state['running_native_high_water_bytes']>=self.calibrated_peak,
                'Source/geometry/calibration identity and monotonic actual high water')
        self.high_water=max(self.high_water,state['running_native_high_water_bytes'])
