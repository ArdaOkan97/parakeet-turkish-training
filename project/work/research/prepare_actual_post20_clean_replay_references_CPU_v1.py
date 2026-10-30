"""Exact next original carried FR/PT replay PCM and genuine cached full targets."""
import argparse,gc,hashlib,json,os,shutil,struct,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';T=P/'work/training-env/Lib/site-packages'
R=Path(r'E:\gpu_training_scratchpad');CPU=R/'scratch/BPE_broad_CPU_preflight_env_v2'
sys.path.insert(0,str(S))
from fastmss_feature_window_v1 import scoped,read_bound,record,write_new,require
from resource_watchdog_v2 import sample_memory
def guard():
 v=sample_memory(os.getpid(),'actual_original_carried_replay_reference');w,s=v['worker_private_memory'],v['system_memory']
 require(w['available'] and s['available'] and v['paired_sampling_within_one_second'] and w['private_bytes']<2_000_000_000,'Fresh private2GB')
 require(s['commit_used_bytes']/s['commit_limit_bytes']<.90 and s['commit_available_bytes']>=2_000_000_000 and shutil.disk_usage(R).free>=100_000_000_000,'System90/2GB/E100GB')
 return v
def main(output):
 require(os.environ['CUDA_VISIBLE_DEVICES']=='-1','CPU only')
 for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):require(os.environ[k]=='1','Fixed threadcaps')
 out=scoped(output);out.mkdir(exist_ok=False);initial=guard();system=initial['system_memory']
 require(system['commit_used_bytes']+700_000_000<.90*system['commit_limit_bytes'] and system['commit_available_bytes']-700_000_000>=2_000_000_000,'Fresh700MB reference admission')
 meta=R/'analysis/ASR18_memory_diagnosis_v1/post20_replay_metadata_actual_03/report.json'
 binding=record(meta,guard);require(binding['sha256']=='df9ac85057d448555fb6bb8c52a5810555c1eef0e5727488fdd43739c54b6d38','Exact actual post20 carried replay metadata')
 parent=json.loads(meta.read_bytes());rows=parent['next_carried_replay_rows']
 require(len(rows)==2 and rows[0]['language']=='fr' and rows[1]['language']=='pt','Exact next original carried FR/PT pair')
 sys.path.insert(0,str(T))
 import numpy as np
 sys.path.remove(str(T))
 import torch
 require(torch.__version__=='2.7.1+cpu' and torch.version.cuda is None and Path(torch.__file__).resolve().is_relative_to(CPU),'Pinned E CPU-only Torch')
 sys.path.append(str(T));torch.set_num_threads(1)
 import soundfile as sf
 references=[]
 for row in rows:
  guard();audio=scoped(row['audio_filepath'],existing_input=True);cache=scoped(row['cache_path'],existing_input=True)
  before=[(p.stat().st_size,p.stat().st_mtime_ns) for p in (audio,cache)];ab=record(audio,guard);cb=record(cache,guard)
  require(cb['bytes']<=128*1024**2,'One original replay cache128MiB bound')
  wave,sr=sf.read(audio,dtype='float32',always_2d=False)
  require(sr==16000 and wave.ndim==1 and wave.dtype==np.float32 and wave.flags.c_contiguous and np.isfinite(wave).all() and
   (len(wave)//160+7)//8==row['encoder_frames'],'Whole original replay16kPCM/full frame identity')
  actual=torch.load(cache,map_location='cpu',weights_only=True)
  require(actual['source_id']==row['source_id'] and 'kind' not in actual and actual['features'].ndim==2 and
   actual['features'].shape[0]==row['encoder_frames'] and torch.isfinite(actual['features']).all(),'Genuine original teacher/source cache')
  ids=actual['labels'].detach().cpu().long().tolist()
  require(len(ids)==row['target_tokens'] and all(type(v) is int and 0<=v<8193 and v!=8192 for v in ids),'Full unchanged old replay labels')
  require(before==[(p.stat().st_size,p.stat().st_mtime_ns) for p in (audio,cache)],'Original replay files changed during reference')
  references.append({'source_id':row['source_id'],'language':row['language'],'source_row':row,
   'source_audio_file':{**ab,'mtime_ns':before[0][1]},'audio_file':{**ab,'mtime_ns':before[0][1]},
   'genuine_teacher_feature_cache':{**cb,'mtime_ns':before[1][1]},'full_target_ids':ids,
   'target_ids_u32_sha256':hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest(),
   'expected_audio':{'sample_rate':16000,'samples':len(wave),'waveform_sha256':hashlib.sha256(wave.tobytes()).hexdigest()},
   'frozen_encoder_feature_shape':list(actual['features'].shape),'full_clean_original_audio':True,'augmentation':False})
  del wave,actual;gc.collect()
 write_new(out/'report.json',{'kind':'FastMSS_actual_post20_clean_replay_reference_v1','passed':True,
  'post20_carried_replay_metadata':binding,'references':references,'actual_optimizer_updates':0,'actual_draw_frontiers_advanced':False,
  'new_PCM_or_teacher_feature_cache_created':False,'Torch':torch.__version__,'source':record(__file__,guard),
  'fresh700MB_admission':{'passed':True,'forecast_bytes':700_000_000,'sample':initial}})
 print(json.dumps({'passed':True,'references':2,'report':record(out/'report.json',guard)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
