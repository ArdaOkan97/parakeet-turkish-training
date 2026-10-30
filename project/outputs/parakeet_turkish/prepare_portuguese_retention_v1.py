"""Separate pinned FLEURS pt_br319 TRAIN/50DEV/100 reserved TEST preparation.

--prepare-only is stdlib/offline. Actual public downloads require the explicit
network flag. No Torch/CUDA/model is imported, and old manifests stay unchanged.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,sys,time,wave
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent;DATA=ROOT/'work/data'
REVISION='70bb2e84b976b7e960aa89f1c648e09c59f894dd';REPO='google/fleurs';CONFIG='pt_br'
FILES=[f'parquet-data/{CONFIG}/{split}-00000-of-00001.parquet' for split in ('train','validation','test')]

def require(ok,message):
 if not ok:raise ValueError(message)
def record(path):
 path=Path(path).resolve();before=path.stat();h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 require((before.st_size,before.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns),'File changed while hashing')
 return {'path':str(path),'sha256':h.hexdigest(),'bytes':before.st_size}
def write(path,value):
 path=Path(path);require(not path.exists(),'New immutable output required: '+str(path));path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def below_normal():
 if os.name!='nt':return {'available':False,'reason':'notWindows'}
 import ctypes
 a=ctypes.WinDLL('kernel32',use_last_error=True);a.GetCurrentProcess.restype=ctypes.c_void_p
 a.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_uint];a.SetPriorityClass.restype=ctypes.c_int
 ok=bool(a.SetPriorityClass(a.GetCurrentProcess(),0x4000));return {'applied':ok,'winerror':0 if ok else ctypes.get_last_error()}
def raw_rows(path):
 with Path(path).open(encoding='utf-8') as f:
  for line in f:
   if line.strip():yield json.loads(line)
def planned(args):
 lock=json.loads((DATA/'sources.lock.json').read_text(encoding='utf-8'));source=lock[REPO]
 require(source['revision']==REVISION and all(x in source['files'] for x in FILES),'Exact locked public pt_br shards required')
 base=Path((ROOT/'work/base_model_path.txt').read_text().strip()).resolve()
 return {'schema':1,'kind':'Portuguese_FLEURS_retention_preparation_plan_v1','mode':'offline_source_only',
  'repository':REPO,'revision':REVISION,'configuration':CONFIG,'language':'pt','files':FILES,
  'cache_dir':str(DATA/'hf_cache'),'cached':{x:(DATA/'hf_cache'/f'datasets--google--fleurs/snapshots/{REVISION}'/x).exists() for x in FILES},
  'output_dir':str(args.output_dir.resolve()),'proof_path':str(args.proof.resolve()),'teacher_tokenizer':record(base/'tokenizer.json'),
  'original_TR_manifest':str(args.original_tr.resolve()),'original_replay_manifest':str(args.original_replay.resolve()),
  'requested':{'TRAIN_replay':319,'fixed_DEV':50,'reserved_TEST':100},'network_called':False,'Torch_imported':False,'TEST_predictions_requested':False}
def main(args):
 plan=planned(args)
 if args.prepare_only:print(json.dumps(plan,ensure_ascii=True,indent=2));return 0
 require(not args.output_dir.exists() and not args.proof.exists(),'Fresh PT directory/proof required')
 os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_DISABLE_XET']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
 for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
 os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='60';priority=below_normal();start=time.monotonic()
 from huggingface_hub import hf_hub_download
 local=[]
 for filename in FILES:
  print('Pinned public PT asset '+filename,flush=True)
  path=hf_hub_download(repo_id=REPO,repo_type='dataset',revision=REVISION,filename=filename,
   cache_dir=str(DATA/'hf_cache'),local_files_only=not args.network_download_authorized,token=False)
  local.append(record(path))
 require('torch' not in sys.modules,'PT preparation must not import Torch')
 import prepare_data as original
 from tokenizers import Tokenizer
 import numpy as np
 require('torch' not in sys.modules,'Data/audio/tokenizer imports unexpectedly imported Torch')
 lock=json.loads((DATA/'sources.lock.json').read_text(encoding='utf-8'));source=lock[REPO]
 protected=[args.original_tr,args.original_replay,DATA/'sources.lock.json']+[DATA/f'{split}_{lang}.jsonl' for split in ('validation','test') for lang in ('en','es','it','fr','de','tr')]
 protected_records={str(p.resolve()):record(p) for p in protected}
 train_sentences=set();train_refs=set();heldout={'validation':set(),'test':set()};old_heldout_audio=set()
 for path in (args.original_tr,args.original_replay):
  for row in raw_rows(path):
   if row.get('dataset')==REPO:train_sentences.add(row['sentence_id'])
   train_refs.add(hashlib.sha256(original.normalize_text(row['text'],'pt').encode()).digest())
 for split in ('validation','test'):
  for lang in ('en','es','it','fr','de','tr'):
   for row in raw_rows(DATA/f'{split}_{lang}.jsonl'):
    if row.get('dataset')==REPO:heldout[split].add(row['sentence_id'])
    old_heldout_audio.update(row[k] for k in ('audio_sha256','wav_sha256','canonical_pcm_sha256') if row.get(k))
 metadata={};parquets=[]
 for split in ('test','validation','train'):
  metadata[split],prov=original.collect_metadata(REPO,CONFIG,split,'pt',source,DATA,True);parquets+=prov
 def new_heldout_allowed(row,banned):
  return row['sentence_id'] not in train_sentences|banned and hashlib.sha256(original.normalize_text(row['text'],'pt').encode()).digest() not in train_refs
 test=original.stable_select([r for r in metadata['test'] if new_heldout_allowed(r,heldout['validation'])],100,args.seed,'fleurs_pt_test')
 test_ids=original.sentence_ids(test)
 dev=original.stable_select([r for r in metadata['validation'] if new_heldout_allowed(r,heldout['test']|test_ids)],50,args.seed,'fleurs_pt_validation')
 heldout_ids=heldout['validation']|heldout['test']|test_ids|original.sentence_ids(dev)
 tokenizer=Tokenizer.from_file(plan['teacher_tokenizer']['path'])
 require(tokenizer.get_vocab_size()==8193 and tokenizer.token_to_id('<blank>')==8192,'Original8193 tokenizer/blank8192 required')
 exclusions={};targets={};eligible=[]
 for row in metadata['train']:
  reason=None;ids=tokenizer.encode(row['text'],add_special_tokens=False).ids
  if row['sentence_id'] in heldout_ids:reason='protected_global_heldout_sentence'
  elif row['source_duration'] is not None and not .5<=row['source_duration']<=30.:reason='TRAIN_duration_bounds'
  elif not ids or len(ids)>180 or 8192 in ids or min(ids)<0 or max(ids)>=8193:reason='original_teacher_target_bounds'
  elif original.normalize_text(tokenizer.decode(ids,skip_special_tokens=True),'pt')!=original.normalize_text(row['text'],'pt'):reason='ungrouped_TDT_normalized_roundtrip'
  if reason:exclusions[reason]=exclusions.get(reason,0)+1;continue
  eligible.append(row);targets[row['source_id']]=ids
 train=original.stable_select(eligible,319,args.seed,'fleurs_pt_replay')
 original.assert_split_isolation(train,dev,test)
 args.output_dir.mkdir(parents=True,exist_ok=False)
 converted=original.export_audio(train+dev+test,args.output_dir,.5,30.)
 by_id={r['source_id']:r for r in converted};train=[by_id[r['source_id']] for r in train];dev=[by_id[r['source_id']] for r in dev];test=[by_id[r['source_id']] for r in test]
 pcm_evidence=[]
 for row in train+dev+test:
  with wave.open(row['audio_filepath'],'rb') as audio:
   require(audio.getframerate()==16000 and audio.getnchannels()==1 and audio.getsampwidth()==2 and audio.getcomptype()=='NONE','Canonical16k mono PCM16 required')
   raw=audio.readframes(audio.getnframes());frames=audio.getnframes()
  require(len(raw)==frames*2 and frames>0,'Complete finite integer PCM required');values=np.frombuffer(raw,dtype='<i2')
  require(np.any(values!=0),'Allzero audio rejected')
  row['canonical_pcm_sha256']=hashlib.sha256(raw).hexdigest();row['training_role']='train' if row in train else 'heldout'
  pcm_evidence.append({'source_id':row['source_id'],'frames':frames,'canonical_pcm_sha256':row['canonical_pcm_sha256'],'zero_fraction':float(np.count_nonzero(values==0)/frames),'clipping_fraction':float(np.count_nonzero((values==-32768)|(values==32767))/frames)})
  if row in train:
   require(.5<=row['duration']<=30.,'Actual PT TRAIN duration differs from eligible metadata')
   ids=targets[row['source_id']];row.update(teacher_vocab_size=8193,target_tokens=len(ids),teacher_target_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest())
 original.assert_split_isolation(train,dev,test)
 train_audio={row[k] for row in train for k in ('audio_sha256','wav_sha256','canonical_pcm_sha256')};new_eval_audio={row[k] for row in dev+test for k in ('audio_sha256','wav_sha256','canonical_pcm_sha256')}
 require(not train_audio&new_eval_audio and not train_audio&old_heldout_audio,'PT TRAIN/heldout audio leakage')
 require(not {r[k] for r in dev for k in ('audio_sha256','wav_sha256','canonical_pcm_sha256')}&{r[k] for r in test for k in ('audio_sha256','wav_sha256','canonical_pcm_sha256')},'PT DEV/TEST audio leakage')
 for path in (args.original_tr,args.original_replay):
  for row in raw_rows(path):require(not any(row.get(k) in new_eval_audio for k in ('audio_sha256','wav_sha256','canonical_pcm_sha256') if row.get(k)),'New PT heldout audio already in protected TRAIN')
 files={}
 for name,rows in (('replay_train_pt',train),('validation_pt',dev),('test_pt',test)):
  path=args.output_dir/(name+'.jsonl');original.write_jsonl(path,rows)
  files[name]={**record(path),'utterances':len(rows),'hours':math.fsum(r['duration'] for r in rows)/3600,'source_ids_sha256':hashlib.sha256('\n'.join(r['source_id'] for r in rows).encode()).hexdigest()}
 for path,value in protected_records.items():require(record(path)==value,'Protected original source changed')
 require('torch' not in sys.modules,'PT actual CPU prep imported Torch')
 proof={'schema':1,'kind':'Portuguese_FLEURS_retention_preparation_v1','passed':True,'language':'pt','dataset_config':CONFIG,'repository':REPO,'revision':REVISION,'seed':args.seed,'files':files,'manifests':files,'source_parquet_records':local,'source_split_metadata_rows':{k:len(v) for k,v in metadata.items()},'TRAIN_eligibility_exclusions':exclusions,'TRAIN_original_teacher':{'vocab_size':8193,'blank_id':8192,'max_tokens':180,'tokenizer':plan['teacher_tokenizer'],'normalized_ungrouped_roundtrip_passed':True,'all_targets_below_original8193':True},'audio_PCM_evidence':pcm_evidence,'protected_original_files':protected_records,'source_split_and_global_sentence_ID_isolation_passed':True,'new_PT_heldout_excluded_all_existing_FLEURS_TRAIN_ids_and_normalized_references':True,'new_PT_heldout_audio_checked_against_all_existing_TR_and_replay':True,'original_five_replay_rows_changed':False,'reserved_TEST_predictions_read_or_executed':False,'Torch_imported':False,'GPU_or_model_inference':False,'public_download_authorized':args.network_download_authorized,'below_normal_priority':priority,'elapsed_seconds':time.monotonic()-start,'created_at_utc':datetime.now(timezone.utc).isoformat(),'source_script':record(__file__)}
 write(args.proof,proof);print(json.dumps({'passed':True,'proof':record(args.proof),'files':files,'elapsed_seconds':proof['elapsed_seconds']},ensure_ascii=True,indent=2));return 0

def parser():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-only',action='store_true');p.add_argument('--network-download-authorized',action='store_true');p.add_argument('--seed',type=int,default=20261010)
 p.add_argument('--output-dir',type=Path,default=DATA/'portuguese_retention_v1');p.add_argument('--proof',type=Path,default=HERE/'Portuguese_FLEURS_retention_preparation_v1.json')
 p.add_argument('--original-tr',type=Path,default=DATA/'TSC_quality_inputs_v2/combined_TR_train.jsonl');p.add_argument('--original-replay',type=Path,default=DATA/'TSC_quality_inputs_v2/original_replay_train.jsonl');return p
if __name__=='__main__':sys.exit(main(parser().parse_args()))
