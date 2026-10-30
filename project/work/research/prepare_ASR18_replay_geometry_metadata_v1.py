"""Whole immutable original replay metadata envelope; not decoded validation."""
import argparse,collections,json,math,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import write_json
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def group():return {'rows':0,'total_target_tokens':0,'MAX_declared_encoder_frames':0,'MAX_declared_target_tokens':0,'MAX_declared_duration_seconds':0.0,'count_outside_measured_replay_U75':0,'count_outside_shared_student_T1115':0}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--folder',required=True);a=parser.parse_args();f=Path(a.folder);require(f.is_relative_to(A),'E-only metadata output');f.mkdir(exist_ok=False)
    plan=json.loads((A/'cleaner_after20_plan_01/plan.json').read_text(encoding='utf-8'));binding=plan['original_replay_pool'];require(record(binding['path'])==binding,'Exact original immutable replay pool required')
    value=json.loads(Path(binding['path']).read_text(encoding='utf-8'));rows=value['replay'];require(type(rows) is list and len(rows)==1914,'Entire original1914 replay population required')
    groups={k:group() for k in ('all','en','es','it','fr','de','pt')};seen=set();largest=[]
    for row in rows:
        lang=row['language'];identity=(lang,row['source_id']);t=row['encoder_frames'];u=row['target_tokens'];duration=row['duration']
        require(lang in groups and lang!='all' and identity not in seen and type(t) is int and t>0 and type(u) is int and u>0
            and type(duration) in (float,int) and math.isfinite(duration) and duration>0,'Unique original replay identities and declared scalar geometry required')
        seen.add(identity)
        for key in ('all',lang):
            g=groups[key];g['rows']+=1;g['total_target_tokens']+=u;g['MAX_declared_encoder_frames']=max(g['MAX_declared_encoder_frames'],t)
            g['MAX_declared_target_tokens']=max(g['MAX_declared_target_tokens'],u);g['MAX_declared_duration_seconds']=max(g['MAX_declared_duration_seconds'],duration)
            g['count_outside_measured_replay_U75']+=int(u>75);g['count_outside_shared_student_T1115']+=int(t>1115)
        largest.append({k:row[k] for k in ('source_id','language','encoder_frames','target_tokens','duration','audio_filepath','cache_path')})
    topU=sorted(largest,key=lambda r:r['target_tokens'],reverse=True)[:8];topT=sorted(largest,key=lambda r:r['encoder_frames'],reverse=True)[:8]
    require({k:v['rows'] for k,v in groups.items() if k!='all'}=={'en':312,'es':316,'it':328,'fr':317,'de':322,'pt':319},'Every original replay row retained without selection')
    require(not any(v in sys.modules for v in ('torch','numpy','scipy','transformers','pyarrow','soundfile')),'Metadata only; no decoder/cache tensor/runtime import')
    result={'kind':'ASR18_whole_original_replay_declared_metadata_geometry_v1','passed':True,'original_pool':binding,'groups':groups,'top_U_rows':topU,'top_T_rows':topT,
        'source':record(__file__),'all_original1914_rows_retained':True,'new_audio_or_cache_tensor_reads':0,'encoder_frames_and_U_are_original_metadata_not_new_independent_cache_validation':True,
        'future_before_draw_full_audio_feature_cache_and_target_identity_check_required':True,'sampler_or_replay_RNG_advanced':False,
        'actual_native_updates':0,'full_native_envelope_or_forecast_or_production_approval_claimed':False}
    write_json(f/'report.json',result);print(json.dumps({'passed':True,'whole_original_replay_rows':len(rows),'all':groups['all'],'native_updates':0}))
if __name__=='__main__':main()
