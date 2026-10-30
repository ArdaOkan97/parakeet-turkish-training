"""Candidate forecast for this exact closed feature window, never broad geometry.

First activation and varying replay rows exceeded the earlier2GB/last-peak
estimates. A future caller must bind actual control evidence, retain a running
high-water mark and reserve explicit warmup variation. This module does not
change any running source or the independent global/CUDA resource guards.
"""
import json
from pathlib import Path
from bpe_broad_audio_v1 import record,require

class FixedWindowForecast:
    def __init__(self,*,feature_window,control_summary,summary):
        require(record(control_summary['path'])==control_summary and summary['passed'] is True and
            summary['sustained20_proven'] is False,'Exact actual ONE-control evidence required')
        require(summary['full_state_checkpoint_restore']['all207_weights_all3_Adam_and_Python_NumPy_Torch_CUDA_RNG_exact'] is True,
            'Actual complete state restoration prerequisite')
        actual=summary['actual_closed'];require(record(actual['path'])==actual,'Actual control closure binding changed')
        closed=json.loads(Path(actual['path']).read_text(encoding='utf-8'))
        require(closed['kind']=='ASR18_owned_actual_closed_GPU_child_v1' and closed['owner']=='/root/memory_diagnosis' and
            closed['actual_tool_exit_code']==closed['actual_native_exit_code']==0 and closed['actual_tool_chunk_id'],
            'Actual owned control0 prerequisite')
        report=closed['report'];require(record(report['path'])==report,'Actual control report changed')
        complete=json.loads(Path(report['path']).read_text(encoding='utf-8'))
        require(complete['passed'] is True and complete['completion']['native_updates']==1 and
            complete['completion']['feature_window']==feature_window and record(feature_window['path'])==feature_window,
            'Calibration cannot silently generalize to different feature geometry')
        self.window=feature_window;self.control=control_summary
        self.calibrated_peak=summary['native_50ms_sampled_private_peak_bytes']
        require(type(self.calibrated_peak) is int and self.calibrated_peak>0,'Available native calibration required')
        self.high_water=self.calibrated_peak

    def observe(self,proof):
        require(proof['error'] is None and proof['joined'] is True and proof['available_sample_count']>=3,
            'Clean measured native phase required; no guessed fallback')
        self.high_water=max(self.high_water,proof['private_peak_bytes'])

    def budget(self,*,current_private_bytes,feature_window,update):
        require(feature_window==self.window and type(update) is int and 1<=update<=20 and
            type(current_private_bytes) is int and current_private_bytes>0,'This exact fixed20 input window only')
        target=max(self.calibrated_peak+750_000_000,self.high_water+250_000_000)
        return {'incremental_bytes':max(2_500_000_000 if update==1 else 250_000_000,target-current_private_bytes),
            'current_private_bytes':current_private_bytes,'absolute_private_forecast_bytes':target,
            'calibrated_native_peak_bytes':self.calibrated_peak,'running_native_high_water_bytes':self.high_water,
            'warmup_variation_allowance_bytes':750_000_000,'minimum_high_water_margin_bytes':250_000_000,
            'actual_control':self.control,'feature_window':self.window,'production_geometry_or_admission_claimed':False}
