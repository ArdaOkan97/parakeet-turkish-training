"""Preserve role-specific preflight rejection; pin a larger prospective budget."""
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish'
p=S/'asr18_calibrated_geometry_forecast_v1.py';q=p.with_name('asr18_calibrated_geometry_forecast_v2.py');t=p.read_text()
needle="        self.calibrated_peak=summary['native_phase_private_peak_50ms_bytes'];self.high_water=self.calibrated_peak"
replacement="""        self.original_role_envelopes=json.loads(json.dumps(self.envelopes))
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
        self.calibrated_peak=summary['native_phase_private_peak_50ms_bytes'];self.high_water=self.calibrated_peak"""
assert needle in t;t=t.replace(needle,replacement)
t=t.replace("self.calibrated_peak+750_000_000,self.high_water+250_000_000", "self.calibrated_peak+750_000_000+self.replay_teacher_feature_extension_reserve_bytes,self.high_water+250_000_000+self.replay_teacher_feature_extension_reserve_bytes")
t=t.replace("'geometry_envelopes':self.envelopes,", "'geometry_envelopes':self.envelopes,'original_role_envelopes':self.original_role_envelopes,\n            'replay_teacher_feature_extension_reserve_bytes':self.replay_teacher_feature_extension_reserve_bytes,\n            'replay_teacher_target_U_ceiling_unchanged':True,'teacher_joint_sample_grid_maximum_cells':64,")
t=t.replace('ASR18_finite_measured_geometry_native_forecast_v1','ASR18_finite_measured_geometry_native_forecast_v2')
assert not q.exists();q.write_text(t,encoding='utf-8',newline='\n')
p=Path(__file__).with_name('validate_ASR18_geometry_forecast_v1.py');q=p.with_name('validate_ASR18_geometry_forecast_v2.py')
t=p.read_text().replace('asr18_calibrated_geometry_forecast_v1','asr18_calibrated_geometry_forecast_v2').replace('5432813440','5496813440')
assert not q.exists();q.write_text(t,encoding='utf-8',newline='\n')
