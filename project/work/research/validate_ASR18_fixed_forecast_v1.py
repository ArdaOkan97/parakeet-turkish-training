"""Metadata replay of already-closed phases; no GPU/global-admission claim."""
import argparse,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import write_json
from asr18_fixed_window_forecast_v1 import FixedWindowForecast
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();folder=Path(a.folder)
    require(folder.is_relative_to(A),'E-only candidate forecast fixture');folder.mkdir(exist_ok=False)
    sp=A/'feature_window_native_control_01/actual_one_update_control_summary_v2.json';summary=json.loads(sp.read_text())
    result=json.loads((A/'feature_window_native_control_01/report.json').read_text());window=result['completion']['feature_window']
    policy=FixedWindowForecast(feature_window=window,control_summary=record(sp),summary=summary)
    observations=[];f=A/'feature_window_native20_01'
    paths=sorted(f.glob('native_phase_*.json'),key=lambda value:int(value.stem.rsplit('_',1)[1]))
    for path in paths:
        n=int(path.stem.rsplit('_',1)[1]);proof=json.loads(path.read_text());old=json.loads((f/f'update_{n}_budget.json').read_text())
        proposed=policy.budget(current_private_bytes=old['current_private_bytes'],feature_window=window,update=n)
        growth=proof['private_peak_bytes']-old['current_private_bytes']
        require(growth<=proposed['incremental_bytes'],'Candidate does not cover an observed actual phase; preserve failure')
        observations.append({'update':n,'old_budget':old['incremental_bytes'],'candidate_budget':proposed['incremental_bytes'],
            'actual_increase':growth,'old_margin':old['incremental_bytes']-growth,'candidate_margin':proposed['incremental_bytes']-growth,
            'phase_record':record(path),'original_admission_baseline':record(f/f'update_{n}_budget.json')})
        policy.observe(proof)
    require(len(observations)>=3,'Actual closed multistep calibration required')
    def reject(call):
        try:call()
        except ValueError:return True
        raise ValueError('Expected forecast fail-closed rejection')
    checks={'mismatched_window_rejected':reject(lambda:policy.budget(current_private_bytes=1,feature_window={**window,'sha256':'0'*64},update=1)),
        'dead_or_error_monitor_rejected':reject(lambda:policy.observe({'error':'intentional','joined':True,'available_sample_count':3,'private_peak_bytes':1})),
        'unavailable_monitor_rejected':reject(lambda:policy.observe({'error':None,'joined':True,'available_sample_count':0,'private_peak_bytes':1}))}
    high=policy.high_water;policy.observe({'error':None,'joined':True,'available_sample_count':3,'private_peak_bytes':1})
    checks['running_high_water_never_decreases']=policy.high_water==high
    write_json(folder/'report.json',{'passed':True,'kind':'ASR18_candidate_forecast_actual_phase_metadata_replay_v1','source':record(__file__),
        'candidate_policy':record(P/'outputs/parakeet_turkish/asr18_fixed_window_forecast_v1.py'),'actual_control':record(sp),
        'observations':observations,'checks':checks,'closed_native_phases_replayed':len(observations),'all20_native_closed_claimed':False,
        'actual_GPU_launched':False,'actual_global_admission_or_future_shape_fit_claimed':False,'current_admitted_sources_changed':False})
    print(json.dumps({'passed':True,'closed_native_phases':len(observations),'checks':checks}))

if __name__=='__main__':main()
