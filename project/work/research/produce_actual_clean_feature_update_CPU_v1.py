"""Original CPU Parakeet frontend on explicit clean TRAIN; no augmentation."""
from __future__ import annotations
import argparse,gc,inspect,json,os,shutil,sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2];SOURCE=PROJECT/'outputs/parakeet_turkish'
TRAINING_SITE=PROJECT/'work/training-env/Lib/site-packages'
CPU_ENV=Path(r'E:\gpu_training_scratchpad\scratch\BPE_broad_CPU_preflight_env_v2')
sys.path.insert(0,str(SOURCE))
from fastmss_clean_feature_window_v1 import (ROOT,CACHE,CAPACITY,FRONTEND_ARGS,begin_update,finish_update,
 capacity_lock,inventory,read_bound,record,save_array,scoped,write_new,require)
from fastmss_clean_feature_contract_v1 import CleanFeatureUpdateProducer,PHASE

def main(request_path,output):
 request=json.loads(scoped(request_path).read_bytes())
 require(request['kind']=='FastMSS_actual_CPU_clean_feature_request_v1' and os.environ['CUDA_VISIBLE_DEVICES']=='-1','Explicit original clean CPU feature request')
 for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):require(os.environ[k]=='1','Four CPU threadcaps')
 from resource_watchdog_v2 import sample_memory
 def guard():
  v=sample_memory(os.getpid(),'original_clean_CPU_frontend');w,s=v['worker_private_memory'],v['system_memory']
  require(w['available'] and s['available'] and v['paired_sampling_within_one_second'] and w['private_bytes']<2_000_000_000,'Fresh paired private2GB')
  require(s['commit_used_bytes']/s['commit_limit_bytes']<.90 and s['commit_available_bytes']>=2_000_000_000 and shutil.disk_usage(ROOT).free>=100_000_000_000,'Global90/2GB/E100GB guards')
  return v
 initial=guard();system=initial['system_memory']
 require(system['commit_used_bytes']+700_000_000<.90*system['commit_limit_bytes'] and system['commit_available_bytes']-700_000_000>=2_000_000_000,'Fresh original CPU frontend700MB admission')
 owner,folder,planned,active=begin_update(request['owner'],request['update_index'],request['prior_closures'],guard)
 clean_request=read_bound(request['clean_contract_request'],guard)
 require(clean_request['feature_cache_owner']==request['owner'] and clean_request['update_index']==request['update_index'] and
  clean_request['phase']==PHASE and clean_request['batch_id']==planned['batch_id'] and clean_request['contexts']==planned['contexts'] and
  clean_request['native_transaction_sha256']==planned['expected_native_transaction_sha256'] and
  clean_request['before_sampler_sha256']==planned['expected_before_sampler_sha256'] and
  clean_request['after_sampler_sha256']==planned['expected_after_sampler_sha256'] and clean_request['frontend']==owner['frontend'] and
  clean_request['augmentation_identity']==owner['augmentation_identity'],'Exact independently pinned clean source/draw/frontend request')
 sys.path.insert(0,str(TRAINING_SITE))
 import numpy as np
 sys.path.remove(str(TRAINING_SITE))
 import torch
 require(torch.__version__=='2.7.1+cpu' and torch.version.cuda is None and Path(torch.__file__).resolve().is_relative_to(CPU_ENV),'Pinned E CPU-only Torch')
 sys.path.append(str(TRAINING_SITE));torch.set_num_threads(4)
 from transformers import AutoFeatureExtractor
 import soundfile as sf
 declared=owner['frontend'];extractor=AutoFeatureExtractor.from_pretrained(declared['base_path'],local_files_only=True)
 require(type(extractor).__name__=='ParakeetFeatureExtractor' and record(inspect.getfile(type(extractor)),guard)==declared['source'] and
  record(declared['processor_config']['path'],guard)==declared['processor_config'] and declared['arguments']==FRONTEND_ARGS,'Exact original frontend source/config/defaults')
 def clean_reader(row):
  wave,sr=sf.read(scoped(row['audio_filepath'],existing_input=True),dtype='float32',always_2d=False)
  require(sr==16000 and wave.ndim==1,'Whole original mono16k clean WAV only; no resampling/trimming')
  return wave
 def frontend(wave):
  guard();inputs=dict(extractor([wave],**FRONTEND_ARGS));require(set(inputs)=={'input_features','attention_mask'},'Original complete frontend keys')
  mask=inputs['attention_mask'].bool();features=inputs['input_features'].masked_fill(~mask.unsqueeze(-1),0).detach().cpu().contiguous().numpy()
  return {'input_features':features,'attention_mask':mask.detach().cpu().contiguous().numpy()}
 def sink(index,key,array):
  suffix='features' if key=='input_features' else 'mask'
  return save_array(owner,folder,f"u{request['update_index']:02}_r{index:02}_{suffix}.npy",array,guard)
 def capacity(additional):
  guard()
  with capacity_lock():
   require(list(CACHE.iterdir())==[folder],'One shared512MiB feature window; no parallel cache')
   return inventory(owner,folder)+additional<=CAPACITY
 producer=CleanFeatureUpdateProducer(clean_request,clean_reader=clean_reader,frontend=frontend,save_array=sink,capacity_guard=capacity,guard=guard)
 for index in range(len(clean_request['rows'])):producer.prepare_next(index);gc.collect()
 prepared=producer.prepared_update()
 actual_frontend={**declared,'Torch_file':str(Path(torch.__file__).resolve()),'NumPy':np.__version__,
  'NumPy_file':str(Path(np.__file__).resolve()),'Torch_CUDA_runtime':None,'actual_torch_threads':torch.get_num_threads(),
  'ASR_model_loaded':False,'cleaner_phase':PHASE,'fresh700MB_stage_admission':{'sample':initial,'forecast_bytes':700_000_000,'passed':True}}
 result=finish_update(request['owner'],request['update_index'],active,prepared['rows'],prepared['prepared_payload'],
  input_evidence={**prepared['input_evidence'],'clean_contract_request':request['clean_contract_request'],
   'whole_original_clean_WAV_no_transport':True,'actual_CPU_child_exit_observed_in_child':False},frontend=actual_frontend,guard=guard)
 require(not any(v.startswith('nemo.collections.asr.models') for v in sys.modules),'No ASR model')
 write_new(scoped(output),{'passed':True,'update_record':result,'actual_producer_PID':os.getpid(),'phase':PHASE,
  'original_frontend_executed':True,'augmentation_or_weak_transport_invoked':False,'no_optimizer_or_sampler_updates':True})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.request,a.output)
