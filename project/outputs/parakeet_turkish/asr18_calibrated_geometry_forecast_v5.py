"""NEW prospective bounded-prefix policy with all original replay target scope.

Student TR geometry remains the measured20 envelope. Original replay T/U is
bounded by the whole1914-row metadata audit, with explicit extra padding.
This is a proposed empirical admission policy, not prospective native proof.
"""
import json
from pathlib import Path
from asr18_calibrated_geometry_forecast_v2 import CalibratedGeometryForecast as Previous
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from fastmss_phase_aware_plan_bridge_v3 import actual0
from tdt_full_native_allocation_budget_v1 import allocation_budget

class CalibratedGeometryForecast(Previous):
    def __init__(self,*,whole_replay_metadata,whole_replay_actual0,guard=lambda:None,**previous_bindings):
        super().__init__(**previous_bindings)
        guard();require(record(whole_replay_metadata['path'])==whole_replay_metadata,'Whole original replay metadata binding changed')
        actual0(whole_replay_actual0,whole_replay_metadata,guard,owner='/root/memory_diagnosis')
        report=json.loads(Path(whole_replay_metadata['path']).read_text(encoding='utf-8'));group=report['groups']['all']
        require(report['kind']=='ASR18_whole_original_replay_declared_metadata_geometry_v1' and report['passed'] is True
            and report['all_original1914_rows_retained'] is True and group['rows']==1914
            and group['MAX_declared_encoder_frames']==326 and group['MAX_declared_target_tokens']==126
            and group['count_outside_measured_replay_U75']==116 and group['count_outside_shared_student_T1115']==0,
            'Exact all1914 original replay metadata scope required; no filtering')
        old=self.original_role_envelopes['replay'];t=326;u=126
        require(t<=self.envelopes['student_tr']['encoder_frames'],'Replay student encoder exceeds measured TR scope')
        def joint(target):return allocation_budget(t,target,projected_elements=(t+target+1)*640,head_elements=8716*641,hidden_width=640)['total_budget_bytes']
        logical_extra=max(0,joint(u)-joint(old['target_tokens']))
        teacher_features=32*max(0,t-old['encoder_frames'])*(1024+640)
        require(logical_extra<=64_000_000 and teacher_features<=64_000_000,'New replay logical extension exceeds predeclared separate allowances')
        self.envelopes['replay']={'encoder_frames':t,'target_tokens':u,'feature_frames':8*t+1,'samples':8*t*160+159}
        # Preserve prior64MB and add64MB. These conservative host-policy pads
        # are not an attribution or equality between CUDA and CPU domains.
        self.replay_teacher_feature_extension_reserve_bytes=128_000_000
        self.new_replay_scope={'whole_metadata':whole_replay_metadata,'whole_metadata_actual0':whole_replay_actual0,
            'logical_joint_extension_bytes':logical_extra,'logical_teacher_feature_extension_bytes':teacher_features,
            'additional_private_policy_padding_bytes':64_000_000,'all_original_replay_rows_retained':1914,
            'metadata_requires_actual_full_audio_cache_feature_target_match_before_draw':True}

    def budget(self,*,rows,current_private_bytes,update_in_process):
        require(update_in_process!=1 or current_private_bytes<=3_000_000_000,'First native private observation exceeds reviewed3GB startup envelope; stop before native work')
        value=super().budget(rows=rows,current_private_bytes=current_private_bytes,update_in_process=update_in_process)
        value['calibrated_target_private_bytes']=value['absolute_private_forecast_bytes']
        value['absolute_private_forecast_bytes']=max(value['absolute_private_forecast_bytes'],current_private_bytes+value['incremental_bytes'])
        value['first_native_startup_private_upper_bound_bytes']=3_000_000_000
        value.update(kind='ASR18_finite_measured_geometry_native_forecast_v5',replay_teacher_target_U_ceiling_unchanged=False,
            replay_scope=self.new_replay_scope,student_TR_geometry_scope_unchanged=True,
            source=record(__file__),prospective_native_success_claimed=False,production_forecast_approved=False)
        return value

    def state_dict(self):
        return {'kind':'ASR18_finite_native_forecast_high_water_v5','source':record(__file__),'calibration':self.bindings,
            'replay_scope':self.new_replay_scope,'geometry_envelopes':self.envelopes,
            'running_native_high_water_bytes':self.high_water,'production_forecast_approved':False}

    def load_state_dict(self,state):
        require(type(state) is dict and set(state)==set(self.state_dict())
            and digest({k:v for k,v in state.items() if k!='running_native_high_water_bytes'})==digest({k:v for k,v in self.state_dict().items() if k!='running_native_high_water_bytes'})
            and type(state['running_native_high_water_bytes']) is int and state['running_native_high_water_bytes']>=self.calibrated_peak,
            'Exact policy/source/calibration geometry and valid actual high-water state required')
        self.high_water=max(self.high_water,state['running_native_high_water_bytes'])
