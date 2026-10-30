"""Two complete original cleaner TRAIN rows/PCM identities, never a draw commit."""
from pathlib import Path
from dataclasses import asdict
import argparse,hashlib,json,os,shutil,sys
PROJECT=Path(__file__).resolve().parents[2];ROOT=Path(r'E:\gpu_training_scratchpad')
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from bpe_cleaner_data_v1 import CleanerRows
from fastmss_clean_feature_contract_v1 import target_hash
from fastmss_feature_window_v1 import record,write_new,scoped
from resource_watchdog_v2 import sample_memory

def guard():
 v=sample_memory(os.getpid(),'actual_cleaner_original_PCM_reference');w,s=v['worker_private_memory'],v['system_memory']
 assert w['available'] and s['available'] and v['paired_sampling_within_one_second'] and w['private_bytes']<2_000_000_000
 assert s['commit_available_bytes']>=2_000_000_000 and s['commit_used_bytes']/s['commit_limit_bytes']<.90
 assert shutil.disk_usage(ROOT).free>=100_000_000_000
 return v
def main(output):
 assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'
 for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):assert os.environ[k]=='1'
 out=scoped(output);out.mkdir(exist_ok=False);admission=guard();s=admission['system_memory']
 assert s['commit_used_bytes']+700_000_000<.90*s['commit_limit_bytes'] and s['commit_available_bytes']-700_000_000>=2_000_000_000
 write_new(out/'fresh700MB_admission.json',{'passed':True,'budget_bytes':700_000_000,'sample':admission,'original_source_audio_only':True})
 import numpy as np
 import soundfile as sf
 rows=CleanerRows();entries=[]
 try:
  for ordinal in (1,352768):
   guard();row=rows.row(ordinal);p=scoped(row['audio_filepath'],existing_input=True)
   wave,sr=sf.read(p,dtype='float32',always_2d=False)
   assert sr==16000 and wave.dtype==np.float32 and wave.ndim==1 and wave.flags.c_contiguous and np.isfinite(wave).all()
   assert int(((len(wave)//160)+7)//8)==row['encoder_frames'] and len(row['tr_metadata_labels'])==row['target_tokens']
   ids=list(row['tr_metadata_labels']);stat=p.stat();source={**record(p,guard),'mtime_ns':stat.st_mtime_ns}
   entries.append({'source_id':row['source_id'],'language':'tr','cleaner_original_ordinal':ordinal,'source_row':row,
    'source_audio_file':source,'full_target_ids':ids,'target_ids_u32_sha256':target_hash(ids,'tr'),
    'expected_audio':{'sample_rate':16000,'samples':len(wave),'waveform_sha256':hashlib.sha256(wave.tobytes()).hexdigest()},
    'full_clean_original_audio':True,'augmentation':False,'genuine_teacher_feature_cache':None})
   del wave
  report={'kind':'FastMSS_actual_original_cleaner_TR_reference_v1','passed':True,'cleaner_catalog_identity':rows.identity,
   'phase':'cleaner_data_fine_tuning','selected_exact_ordinals':[1,352768],'references':entries,
   'source':record(__file__,guard),'actual_optimizer_updates':0,'actual_cleaner_draws_consumed':0,
   'sampler_journal_frontier_advanced':False,'original_audio_or_labels_modified':False,'PCM_copies_created':False,
   'replay_references_pending_native_owner_plan':True,'full_cleaner_pass_or_native_compatibility_claimed':False}
  assert not any(v in sys.modules for v in ('torch','transformers','scipy','pyarrow'))
  write_new(out/'report.json',report)
  print(json.dumps({'passed':True,'references':len(entries),'report':record(out/'report.json',guard)}))
 finally:rows.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
