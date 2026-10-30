"""Prospective finite-geometry native forecast from actual closed20 evidence.

This empirical envelope is admission evidence for bounded controls, not a proof
that every smaller shape shares an allocator peak. Independent fresh global,
CUDA, full-DP and actual native monitors remain required. No broad final fit.
"""
import json
from pathlib import Path
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import canonical,digest

class CalibratedGeometryForecast:
    def __init__(self,*,summary_binding,archive_binding,retirement_binding,native_source_binding):
        for b in (summary_binding,archive_binding,retirement_binding,native_source_binding):require(record(b['path'])==b,'Immutable prospective calibration binding changed')
        summary=json.loads(Path(summary_binding['path']).read_text());archive=json.loads(Path(archive_binding['path']).read_text())
        retirement=json.loads(Path(retirement_binding['path']).read_text())
        require(summary['passed'] and summary['actual_native_updates']==20 and summary['same_PID']==23568 and
            summary['all207_weights_three_Adam_Python_NumPy_Torch_CUDA_RNG_both_samplers_and_prefix_exact'], 'Actual sameprocess20/state prerequisite')
        actual=summary['actual_owned_exit'];require(record(actual['path'])==actual,'Actual owned20 exit binding')
        closed=json.loads(Path(actual['path']).read_text());require(closed['actual_native_exit_code']==closed['actual_tool_exit_code']==0
            and closed['actual_tool_chunk_id']=='b5e417' and closed['native_updates']==20,'Actual native20 closure required')
        require(retirement['passed'] and retirement['preserved_metadata']==archive_binding and archive['kind']=='FastMSS_feature_window_durable_metadata_v1'
            and retirement['manifest']==archive['manifest_binding'],'Owned archived metadata must preserve exact retired feature geometry')
        manifest=archive['metadata']['manifest.json'];require(digest(manifest)==archive['manifest_binding']['sha256'] and
            manifest['owner']==archive['owner_binding'],'Exact original canonical manifest geometry binding')
        self.envelopes={};count=0
        for part in manifest['updates']:
            name=Path(part['update']['path']).name;update=archive['metadata'][name]
            require(digest(update)==part['update']['sha256'],'Exact canonical update metadata identity')
            for row in update['rows']:
                group='student_tr' if row['role']=='student_tr' else 'replay'
                require(row['role'] in ('student_tr','replay') and len(row['input_features']['shape'])==3 and row['input_features']['shape'][2]==128,'Actual original frontend scope')
                e=self.envelopes.setdefault(group,{'encoder_frames':0,'target_tokens':0,'feature_frames':0,'samples':0})
                for key,value in (('encoder_frames',row['encoder_frames']),('target_tokens',len(row['full_target_ids'])),
                    ('feature_frames',row['input_features']['shape'][1]),('samples',row['samples'])):e[key]=max(e[key],value)
                count+=1
        require(count==320 and set(self.envelopes)=={'student_tr','replay'},'All320 actual calibrated rows required')
        self.original_role_envelopes=json.loads(json.dumps(self.envelopes))
        # Every role runs the same student encoder. Replay retains its measured
        # target-U ceiling; its teacher head uses at most8x8 sampled cells.
        # Only the full cached teacher features/projector/cosine arrays extend
        # with T.32 bytes times both1024/640 feature widths pads their copies.
        # This is an empirical prospective control policy, not all-shape proof.
        shared={key:max(env[key] for env in self.envelopes.values()) for key in ('encoder_frames','feature_frames','samples')}
        delta=shared['encoder_frames']-self.envelopes['replay']['encoder_frames']
        self.replay_teacher_feature_extension_reserve_bytes=64_000_000
        require(32*delta*(1024+640)<=self.replay_teacher_feature_extension_reserve_bytes,'New teacher-feature extension needs a new explicit budget')
        self.envelopes['replay'].update(shared)
        self.calibrated_peak=summary['native_phase_private_peak_50ms_bytes'];self.high_water=self.calibrated_peak
        self.bindings={'summary':summary_binding,'archive':archive_binding,'retirement':retirement_binding,'native_source':native_source_binding}

    def budget(self,*,rows,current_private_bytes,update_in_process):
        require(type(rows) is list and 3<=len(rows)<=16 and type(current_private_bytes) is int and current_private_bytes>0
            and type(update_in_process) is int and 1<=update_in_process<=20,'Bounded physical1 native input/current observation required')
        for row in rows:
            role=row['role'];require(role in self.envelopes,'Protected teacher/eval cannot enter training admission')
            env=self.envelopes[role]
            observed={'encoder_frames':row['encoder_frames'],'target_tokens':len(row['full_target_ids']),
                'feature_frames':row['input_features']['shape'][1],'samples':row['samples']}
            require(all(type(v) is int and 0<v<=env[key] for key,v in observed.items()),
                    'New geometry lies outside measured envelope; stop before consumed draw, no exclusion')
        target=max(self.calibrated_peak+750_000_000+self.replay_teacher_feature_extension_reserve_bytes,self.high_water+250_000_000+self.replay_teacher_feature_extension_reserve_bytes)
        return {'kind':'ASR18_finite_measured_geometry_native_forecast_v2','current_private_bytes':current_private_bytes,
            'incremental_bytes':max(2_500_000_000 if update_in_process==1 else 250_000_000,target-current_private_bytes),
            'absolute_private_forecast_bytes':target,'calibrated_native_peak_bytes':self.calibrated_peak,'running_native_high_water_bytes':self.high_water,
            'warmup_variation_allowance_bytes':750_000_000,'high_water_margin_bytes':250_000_000,'geometry_envelopes':self.envelopes,'original_role_envelopes':self.original_role_envelopes,
            'replay_teacher_feature_extension_reserve_bytes':self.replay_teacher_feature_extension_reserve_bytes,
            'replay_teacher_target_U_ceiling_unchanged':True,'teacher_joint_sample_grid_maximum_cells':64,
            'bindings':self.bindings,'geometry_is_empirical_admission_basis_not_every_shape_fit_proof':True,'production_forecast_approved':False}

    def observe(self,proof):
        require(proof['error'] is None and proof['joined'] is True and proof['available_sample_count']>=3
            and type(proof['private_peak_bytes']) is int and proof['private_peak_bytes']>0,'Clean available completed native-phase samples required')
        self.high_water=max(self.high_water,proof['private_peak_bytes'])
