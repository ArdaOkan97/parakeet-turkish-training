"""Fork native step63 and qualifying best0 without changing learned state.

CPU-only. Only the top-level run_id changes. Runtime source/config migration is
declared separately; this file never edits the inherited config identity.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import pickle
import re
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
RECEIPT_SHA='a597c2e72d0818370e3fa1a536e1cd8dff3cc526eeead3618beed1b9d8b63449'
SOURCE='adaptive_20261011_native_data_01_public_rank8_control'
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def scoped(path):
 p=Path(path).resolve();require(p.is_relative_to(ROOT),'Workspace-scoped state required');return p
def record(path):return {'path':str(path),'sha256':sha(path),'size_bytes':path.stat().st_size}
def identity(value,torch):
 digest=hashlib.sha256();leaves=0
 def visit(x):
  nonlocal leaves
  if isinstance(x,torch.Tensor):
   require(x.device.type=='cpu','CPU state required')
   leaves+=1;digest.update(b'T'+str(x.dtype).encode()+str(tuple(x.shape)).encode())
   digest.update(x.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
  elif isinstance(x,dict):
   digest.update(b'D')
   for key in sorted(x,key=lambda k:repr(k)):visit(key);visit(x[key])
  elif isinstance(x,(tuple,list)):
   digest.update(b't' if isinstance(x,tuple) else b'l')
   for item in x:visit(item)
  else:digest.update(pickle.dumps(x,protocol=5))
 visit({k:v for k,v in value.items() if k!='run_id'})
 return {'sha256':digest.hexdigest(),'tensor_leaves':leaves}
def main(args):
 import torch
 require(not torch.cuda.is_initialized(),'Fork never initializes CUDA')
 require(re.fullmatch(r'[A-Za-z0-9_-]+',args.target_run_id),'Safe target run ID required')
 receipt_path=scoped(args.predecessor_exit_receipt);require(sha(receipt_path)==RECEIPT_SHA,'Closed root-owned receipt changed')
 receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
 require(receipt['root_observed_owned_session_completed'] is True and receipt['root_observed_exit_code']==1
  and receipt['actual_child_exit_code']==0 and receipt['worker_absent_confirmed'] is True
  and receipt['predecessor_run_id']==SOURCE and receipt['completed_optimizer_updates']==63
  and receipt['selected_step']==0 and receipt['test_evaluated'] is False,'Actual closed native63/best0 receipt required')
 for key in ('loop_status','controller','worker_identity','selection','run_config','history','selected_checkpoint','durable_latest_checkpoint','original_source_checkpoint','diagnosis'):
  require(sha(scoped(receipt[key]['path']))==receipt[key]['sha256'],'Closed predecessor changed: '+key)
 folder=scoped(args.output_dir);require(not folder.exists(),'New immutable snapshot namespace required')
 require(not (HERE/'runs'/args.target_run_id).exists(),'Target run config namespace must be absent')
 for role in ('best','latest'):require(not (ROOT/'work'/(args.target_run_id+'_'+role+'.pt')).exists(),'Target work state must be absent')
 folder.mkdir(parents=True)
 manifest={'schema':1,'kind':'native_exact_state_run_id_fork_v1','status':'CPU_prepared','passed':True,
  'source_run_id':SOURCE,'target_run_id':args.target_run_id,'source_step':63,'inherited_selected_step':0,
  'root_owned_exit_receipt':record(receipt_path),'only_changed_state_path':'run_id',
  'created_at_utc':datetime.now(timezone.utc).isoformat(),'source_run_config':receipt['run_config'],
  'source_checkpoint':{},'fork_checkpoint':{},'development_reuse':{},
  'runtime_identity_migration_not_performed_by_fork':True}
 for role,key,step in (('latest','durable_latest_checkpoint',63),('best','selected_checkpoint',0)):
  source=scoped(receipt[key]['path']);saved=torch.load(source,map_location='cpu',weights_only=True)
  require(saved['run_id']==SOURCE and saved['step']==step and saved['selected_step']==0
   and len(saved['weights'])==207 and len(saved['history'])==step,'Actual learned state/selection shape differs')
  old_identity=identity(saved,torch)
  source_config_identity=saved['config_identity']
  saved['run_id']=args.target_run_id
  archive=folder/(args.target_run_id+'_'+role+'.pt');torch.save(saved,archive)
  actual=torch.load(archive,map_location='cpu',weights_only=True)
  require(actual['run_id']==args.target_run_id and identity(actual,torch)==old_identity,'Learned state bytes changed in fork')
  target=ROOT/'work'/(args.target_run_id+'_'+role+'.pt');shutil.copyfile(archive,target)
  require(sha(target)==sha(archive) and sha(source)==receipt[key]['sha256'],'Fork copy/source changed')
  manifest['source_checkpoint'][role]=receipt[key]
  manifest['fork_checkpoint'][role]={**record(archive),'mutable_work_path':str(target),'step':step,'identity_without_top_run_id':old_identity}
  if role=='latest':
   manifest['source_config_identity']=source_config_identity
   manifest['source_config_identity_sha256']=hashlib.sha256(json.dumps(source_config_identity,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
  del saved,actual
 for step in (0,63):
  source=HERE/'results'/(SOURCE+'_step_'+str(step));target=HERE/'results'/(args.target_run_id+'_step_'+str(step))
  require(source.is_dir() and not target.exists(),'Actual DEV source and fresh target required')
  shutil.copytree(source,target)
  files={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}
  require(all(sha(target/name)==digest for name,digest in files.items()),'Inherited DEV bytes changed')
  manifest['development_reuse'][str(step)]={'source_path':str(source),'target_path':str(target),'files_sha256':files,
   'evaluation_reused_without_new_inference':True,'heldout_test_copied':False}
 require(not torch.cuda.is_initialized(),'Unexpected CUDA initialization')
 manifest['CUDA_initialized']=False
 path=folder/'fork_manifest.json'
 with path.open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2,ensure_ascii=False)
 print(json.dumps({'passed':True,'manifest':str(path),'manifest_sha256':sha(path),'target_run_id':args.target_run_id}))
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--target-run-id',required=True)
 parser.add_argument('--predecessor-exit-receipt',type=Path,default=HERE/'native_data_01_resource_exit_receipt_v1.json')
 parser.add_argument('--output-dir',type=Path,required=True)
 main(parser.parse_args())
