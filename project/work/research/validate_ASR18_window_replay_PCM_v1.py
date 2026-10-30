"""Independent clean replay waveform/full cached-target references, no PCM pool."""
import argparse,gc,hashlib,json,struct,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';T=P/'work/training-env/Lib/site-packages'
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1');E=Path(r'E:\gpu_training_scratchpad')
sys.path.insert(0,str(T))
import numpy as np
sys.path.remove(str(T))
import torch
assert torch.__version__=='2.7.1+cpu' and torch.version.cuda is None
sys.path.extend([str(T),str(S)])
import soundfile as sf
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();folder=Path(a.folder)
    require(folder.is_relative_to(A),'E-only metadata');folder.mkdir(exist_ok=False);torch.set_num_threads(1)
    plan_path=A/'feature_window20_plan_01/plan.json';plan=json.loads(plan_path.read_text())
    require(plan['kind']=='ASR18_provisional_fixed14_feature_window_plan_v1' and plan['actual_optimizer_updates']==0,'Actual pinned provisional-only inputs')
    rows={}
    for update in plan['updates']:
        for row in update['replay_rows']:
            key=row['source_id'];require(key not in rows or rows[key]==row,'Conflicting planned replay metadata');rows[key]=row
    references=[]
    for key,row in sorted(rows.items()):
        audio_path=Path(row['audio_filepath']).resolve();cache_path=Path(row['cache_path']).resolve()
        require(all(p.is_relative_to(P) or p.is_relative_to(E) for p in (audio_path,cache_path)),'Existing project/replay or E sources only')
        before=[(p.stat().st_size,p.stat().st_mtime_ns) for p in (audio_path,cache_path)]
        audio_binding=record(audio_path);cache_binding=record(cache_path)
        require(cache_binding['bytes']<=128*1024**2,'Bounded one original genuine feature/target cache')
        audio,sr=sf.read(audio_path,dtype='float32')
        require(sr==16000 and audio.dtype==np.float32 and audio.ndim==1 and audio.flags.c_contiguous and bool(np.isfinite(audio).all()),'Full original mono16k float32 replay audio')
        require((len(audio)//160+7)//8==row['encoder_frames'],'Exact original replay encoder frames')
        expected={'sample_rate':16000,'samples':len(audio),'waveform_sha256':hashlib.sha256(audio.tobytes()).hexdigest()}
        cached=torch.load(cache_path,map_location='cpu',weights_only=True)
        require(cached['source_id']==key and 'kind' not in cached and cached['features'].ndim==2 and cached['features'].shape[0]==row['encoder_frames'],'Genuine original replay teacher cache/source')
        ids=cached['labels'].detach().cpu().long().tolist()
        require(len(ids)==row['target_tokens'] and all(type(v) is int and 0<=v<8193 and v!=8192 for v in ids),'Every unchanged old-vocabulary replay target')
        require(before==[(p.stat().st_size,p.stat().st_mtime_ns) for p in (audio_path,cache_path)],'Original audio/cache mutated during independent reference read')
        references.append({'source_id':key,'language':row['language'],'source_row':row,'audio_file':{**audio_binding,'mtime_ns':before[0][1]},
            'genuine_teacher_feature_cache':{**cache_binding,'mtime_ns':before[1][1]},'expected_audio':expected,
            'full_target_ids':ids,'target_ids_u32_sha256':hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest(),
            'frozen_encoder_feature_shape':list(cached['features'].shape),'full_clean_original_audio':True,'augmentation':False})
        del audio,cached,ids;gc.collect();write_json(folder/'progress.json',{'completed_unique_replay_rows':len(references)})
    report={'kind':'ASR18_independent_planned_replay_PCM_targets_v1','passed':True,'plan':record(plan_path),'source':record(__file__),
        'original_clean_reader_source':record(S/'runtime.py'),'Torch':torch.__version__,'references':references,
        'unique_replay_sources':len(references),'planned_replay_draws':40,'new_PCM_or_teacher_feature_cache_created':False,
        'actual_optimizer_updates':0,'actual_draw_frontiers_advanced':False,'full_all_six_language_coverage_in_plan':sorted({r['language'] for r in references})}
    write_json(folder/'report.json',report);print(json.dumps(record(folder/'report.json')))

if __name__=='__main__':main()
