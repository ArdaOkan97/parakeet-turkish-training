"""Metadata-only proposed replay-scope/high-water controls, no native proof."""
import argparse,copy,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import write_json
from asr18_calibrated_geometry_forecast_v4 import CalibratedGeometryForecast
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1');F=A.parent/'FastMSS_integration_v1'
def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();f=Path(a.folder);require(f.is_relative_to(A),'E-only policy controls');f.mkdir(exist_ok=False)
    args={'summary_binding':record(A/'feature_window_native20_01/actual_sameprocess20_summary_v1.json'),
        'archive_binding':record(F/'weak_fixed14_feature_window20_durable_metadata_after_native20_v1.json'),
        'retirement_binding':record(F/'weak_fixed14_feature_window20_actual_retirement_v1.json'),
        'native_source_binding':record(S/'asr18_precomputed_native_v2.py'),
        'whole_replay_metadata':record(A/'whole_replay_geometry_CPU_actual_01/report.json'),
        'whole_replay_actual0':record(A/'whole_replay_geometry_CPU_actual_01/actual_owned_CLOSED0.json')}
    policy=CalibratedGeometryForecast(**args);checks=[]
    def reject(name,call):
        try:call()
        except ValueError:checks.append(name);return
        raise ValueError('Expected rejection: '+name)
    archive=json.loads(Path(args['archive_binding']['path']).read_text(encoding='utf-8'));rows=archive['metadata']['update_00.json']['rows']
    proposed=policy.budget(rows=rows,current_private_bytes=2_047_344_640,update_in_process=1)
    require(proposed['absolute_private_forecast_bytes']==5_560_813_440 and proposed['incremental_bytes']>=2_500_000_000,'New explicit reserve preserves old warmup and adds64MB')
    checks.append('Native private target is5,560,813,440B with unchanged warmup/high-water margins plus explicit new64MB')
    scope=copy.deepcopy(rows)
    for row in scope:
        if row['role']=='replay':
            row.update(encoder_frames=326,full_target_ids=list(range(126)),samples=417439)
            row['input_features']['shape']=[1,2608,128]
    policy.budget(rows=scope,current_private_bytes=1,update_in_process=1)
    checks.append('Proposed full original replay T326/U126 declared scope passes policy; no actual native success claim')
    bad=copy.deepcopy(scope);next(r for r in bad if r['role']=='replay')['full_target_ids']=list(range(127))
    reject('Outside original replay U126 stops without source exclusion',lambda:policy.budget(rows=bad,current_private_bytes=1,update_in_process=1))
    bad=copy.deepcopy(scope);next(r for r in bad if r['role']=='student_tr')['encoder_frames']=1116
    reject('Unmeasured TR geometry remains blocked',lambda:policy.budget(rows=bad,current_private_bytes=1,update_in_process=1))
    saved=policy.state_dict();policy.observe({'error':None,'joined':True,'available_sample_count':3,'private_peak_bytes':5_100_000_000})
    high=policy.high_water;policy.load_state_dict(saved);require(policy.high_water==high,'High-water cannot be reduced by older state')
    checks.append('Carried multiwindow high-water never decreases')
    saved=policy.state_dict();other=CalibratedGeometryForecast(**args);other.load_state_dict(saved);require(other.high_water==high,'Exact high-water survives new policy instance')
    checks.append('Exact source-bound high-water state carries to next process/window')
    bad=copy.deepcopy(saved);bad['running_native_high_water_bytes']=True
    reject('Boolean high-water rejected',lambda:policy.load_state_dict(bad))
    bad=copy.deepcopy(saved);bad['geometry_envelopes']['replay']['target_tokens']=127
    reject('Changed role geometry on restore rejected',lambda:policy.load_state_dict(bad))
    reject('First native observation outside3GB startup envelope stops',lambda:policy.budget(rows=scope,current_private_bytes=3_000_000_001,update_in_process=1))
    elevated=policy.budget(rows=scope,current_private_bytes=6_000_000_000,update_in_process=2)
    require(elevated['absolute_private_forecast_bytes']>=6_250_000_000,'Reported absolute ceiling cannot understate current plus required increment')
    checks.append('Later elevated residency reports full actual incremental admission ceiling')
    require(not any(n in sys.modules for n in ('torch','numpy','scipy','transformers','pyarrow')),'No numeric/model/native imports');checks.append('No new native/CPU frontend/ASR work')
    write_json(f/'report.json',{'passed':True,'checks':checks,'check_count':len(checks),'source':record(__file__),
        'forecast_source':record(S/'asr18_calibrated_geometry_forecast_v4.py'),'bindings':args,'proposed_initial_budget':proposed,
        'whole_original_replay_metadata_scope_only':True,'prospective_native_forecast_accepted_or_measured':False,
        'actual_optimizer_updates':0,'production_forecast_approved':False})
    print(json.dumps({'passed':True,'checks':len(checks),'proposed_private_target_bytes':proposed['absolute_private_forecast_bytes'],'actual_native_updates':0}))
if __name__=='__main__':main()
