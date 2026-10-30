"""Fresh full-WAV identity at exact planned clean contexts, no frontend or commit."""
import argparse,hashlib,json,os,shutil,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];R=Path(r'E:\gpu_training_scratchpad');sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from fastmss_feature_window_v1 import scoped,record,read_bound,write_new,require
from fastmss_clean_feature_contract_v1 import PHASE,PROOF_KIND,digest,target_hash
from resource_watchdog_v2 import sample_memory
def guard():
 v=sample_memory(os.getpid(),'bound_original_clean_WAV_reference');w,s=v['worker_private_memory'],v['system_memory']
 require(w['available'] and s['available'] and v['paired_sampling_within_one_second'] and w['private_bytes']<2_000_000_000,'Fresh2GB private')
 require(s['commit_used_bytes']/s['commit_limit_bytes']<.90 and s['commit_available_bytes']>=2_000_000_000 and shutil.disk_usage(R).free>=100_000_000_000,'System90/2GB/E100GB')
def main(request_path,output):
 import numpy as np
 import soundfile as sf
 require(os.environ['CUDA_VISIBLE_DEVICES']=='-1','CPU-only reference')
 request=json.loads(scoped(request_path).read_bytes());require(request['phase']==PHASE and request['actual_optimizer_updates']==0,'Provisional cleaner input')
 references=[]
 for row,context,source in zip(request['rows'],request['contexts'],request['reference_sources']):
  guard();require(all(row[k]==v for k,v in source['source_row'].items()),'Full original source-row fields must remain unchanged')
  require(row['source_id']==source['source_id']==context['source_id'] and row['language']==source['language']==context['language'],'Exact source order')
  path=scoped(row['audio_filepath'],existing_input=True);before=(path.stat().st_size,path.stat().st_mtime_ns)
  original=source.get('source_audio_file',source.get('audio_file'))
  require(record(path,guard)=={k:original[k] for k in ('path','bytes','sha256')} and before[1]==original['mtime_ns'],'Independent original WAV binding')
  wave,sr=sf.read(path,dtype='float32',always_2d=False);pcm=source['expected_audio']
  require(sr==16000 and wave.dtype==np.float32 and wave.ndim==1 and wave.flags.c_contiguous and np.isfinite(wave).all() and
   len(wave)==pcm['samples'] and hashlib.sha256(wave.tobytes()).hexdigest()==pcm['waveform_sha256'] and
   (len(wave)//160+7)//8==row['encoder_frames'],'Fresh whole original PCM/hash/full frame identity')
  ids=source['full_target_ids'];require(row['target_tokens']==len(ids) and target_hash(ids,row['language'])==source['target_ids_u32_sha256'],'Full independent original targets')
  if row['language']=='tr':require(ids==row['tr_metadata_labels'],'All original cleaner BPE IDs')
  else:
   require(ids==row['clean_target_ids'],'All genuine replay IDs');cache=source['genuine_teacher_feature_cache'];cp=scoped(cache['path'],existing_input=True)
   require(record(cp,guard)=={k:cache[k] for k in ('path','bytes','sha256')} and cp.stat().st_mtime_ns==cache['mtime_ns'],'Genuine original replay cache unchanged')
  require(before==(path.stat().st_size,path.stat().st_mtime_ns),'Original WAV changed during reference read')
  references.append({'source_id':row['source_id'],'language':row['language'],'source_row_sha256':digest(row),'context':context,
   'source_audio_file':original,'expected_audio':pcm,'full_target_ids':ids,'target_ids_u32_sha256':source['target_ids_u32_sha256'],
   'cleaner_original_ordinal':row.get('cleaner_original_ordinal'),'genuine_teacher_feature_cache':source.get('genuine_teacher_feature_cache'),
   'full_clean_original_audio':True,'augmentation':False})
  del wave
 require(len(references)==len(request['rows'])==len(request['contexts'])==len(request['reference_sources']),'Exact full reference cardinality')
 require(not any(v in sys.modules for v in ('torch','transformers','scipy','pyarrow')),'No ML/frontend imports')
 write_new(scoped(output),{'kind':PROOF_KIND,'passed':True,'phase':PHASE,'cleaner_catalog_identity':request['cleaner_catalog_identity'],
  'references':references,'source':record(__file__,guard),'input_request':record(request_path,guard),
  'actual_optimizer_updates':0,'sampler_journal_frontier_advanced':False,'original_sources_labels_caches_unchanged':True,
  'source_reference_proofs':request['source_reference_proofs'],'whole_actual_original_WAV_reverified_at_bound_contexts':True})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.request,a.output)
