"""Preload verified native CUDA12.9 linkage before the NEW memory/trainer chain.

The ready backend provenance is immutable. A separate completion file records
the outer result; the inner memory/cache/I/O wrappers retain their own evidence.
--validate-only checks source/argument routing without Torch or CUDA imports.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
PRELOAD_SHA='f8787b0f3346c9f526903ff0c06de4c266f940ca3b0f41c55907cc795e7cc2ba'
ADAPTER_SHA='839194dddf83c3b08bf608d01cb540b2f160feee426a91373d14cfc426ab7494'


def sha(path):
 value=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):value.update(b)
 return value.hexdigest()
def require(ok,message):
 if not ok:raise ValueError(message)
def argument_sha(argv):return hashlib.sha256(json.dumps(argv,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def scoped(path):
 p=Path(path).resolve();require(p.is_relative_to(ROOT),'Workspace-only native evidence/source path required');return p
def option(argv,name):
 require(argv.count(name)==1 and argv.index(name)+1<len(argv),'Exactly one '+name+' required')
 return argv[argv.index(name)+1]
def write_new(path,value):
 path=scoped(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',encoding='utf-8') as f:f.write(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def validate(args):
 wrapper=scoped(args.memory_launcher);trainer=HERE/'train_lora_batched_bpe512_v1.py'
 require(wrapper==HERE/'launch_with_memory_guard_bpe512_v1.py','Only reviewed isolated BPE memory dispatch allowed')
 require(sha(wrapper)=='ce67bf8580c28060a852cf4f979657b960a3d346823f8411e63c92f45fdef7df' and sha(trainer)=='9dca06b398a028d1f8fdc84c984f297f3874cb58917150a5199d6f6dfeb52f11','Pinned isolated BPE source changed')
 require(sha(wrapper)==args.memory_launcher_sha256 and sha(trainer)==args.trainer_sha256,'Native candidate source changed')
 require(sha(HERE/'preload_nemo_linker_v1.py')==PRELOAD_SHA and sha(HERE/'native_tdt_checkpointed_v3.py')==ADAPTER_SHA,
   'Frozen successful native adapter/preload changed')
 argv=list(args.memory_arguments)
 if argv and argv[0]=='--':argv=argv[1:]
 require(argv.count('--')==1,'Exactly one memory-wrapper/trainer separator required')
 worker=argv[argv.index('--')+1:]
 require(option(worker,'--tdt-implementation')=='nemo','This version dispatches official NeMo loss only')
 require(scoped(option(worker,'--native-backend-provenance'))==scoped(args.native_provenance),
   'Native backend provenance must already be included in canonical worker arguments')
 require(option(argv[:argv.index('--')],'--cap-gb') == '8' and option(argv[:argv.index('--')],'--threshold-gb')=='4',
   'Explicit root-selected8GB cap and unchanged4GB endcleanup required; efficacy needs the chosen hardware proof')
 require('--resume-exact' not in worker and '--resume-runtime-migration' not in worker,'BPE arm must use fresh model-only warmstart')
 for flag in ('--resume-weights','--bpe-source-checkpoint','--bpe-source-selection','--bpe-source-closure','--bpe-conversion-report'):
  scoped(option(worker,flag))
 run_id=option(worker,'--run-id')
 require(run_id and all(c.isalnum() or c in '_-' for c in run_id),'Safe owned native run ID required')
 ready=scoped(args.native_provenance);completion=ready.with_suffix('.completion.json')
 require(not ready.exists() and not completion.exists(),'Native backend evidence must be new; no overwrite')
 return wrapper,trainer,argv,worker,run_id,ready,completion


def main(args):
 wrapper,trainer,argv,worker,run_id,ready,completion=validate(args)
 code={'launcher':sha(__file__),'trainer':sha(trainer),'memory_launcher':sha(wrapper),
       'adapter':ADAPTER_SHA,'preload':PRELOAD_SHA}
 if args.validate_only:
  print(json.dumps({'passed':True,'run_id':run_id,'worker_arguments_sha256':argument_sha(worker),
    'source_sha256':code,'Torch_imported':'torch' in sys.modules,'CUDA_initialized':False,'runtime_preload_pending':True}))
  return
 require('torch' not in sys.modules,'Actual preload must precede Torch')
 from preload_nemo_linker_v1 import preload
 result=preload()  # No Torch/Numba/memory wrapper import may precede this call.
 require(result.get('kind')=='coherent_native_cuda12_9_preload_v1' and result.get('actual_binding_linker',{}).get('version')==[12,9] and result.get('cuda_initialized') is False,'Actual coherent pre-Torch linkage required')
 value={'schema':1,'kind':'native_tdt_training_backend_v3','status':'ready',
   'run_id':run_id,'tdt_implementation':'nemo','preload':result,'pid':os.getpid(),'ppid':os.getppid(),
   'ready_at_utc':datetime.now(timezone.utc).isoformat(),
   'trainer_path':str(trainer),'trainer_sha256':code['trainer'],
   'adapter_path':str(HERE/'native_tdt_checkpointed_v3.py'),'adapter_sha256':ADAPTER_SHA,
   'launcher_path':str(Path(__file__).resolve()),'launcher_sha256':code['launcher'],
   'memory_launcher_path':str(wrapper),'memory_launcher_sha256':code['memory_launcher'],
   'worker_arguments_sha256':argument_sha(worker),
   'declared_allocator_policy':{'cap_gb':int(option(argv[:argv.index('--')],'--cap-gb')),'end_cache_gb':4,'native_inactive_threshold_bytes':750000000},
   'native_full_model_or_Lightning_trainer_equivalence_claimed':False,
   'loss_scope':'Official NeMo loss kernel, sigma0/omega0, raw per-clip NLL; existing HF LoRA/retention training'}
 write_new(ready,value)
 old=sys.argv;outcome='complete';error=None
 try:
  sys.argv=[str(wrapper),*argv]
  runpy.run_path(str(wrapper),run_name='__main__')
 except BaseException as e:
  if not isinstance(e,SystemExit) or e.code not in (None,0):outcome='failed';error=repr(e)
  raise
 finally:
  sys.argv=old
  unchanged=(sha(trainer)==code['trainer'] and sha(wrapper)==code['memory_launcher'] and sha(__file__)==code['launcher']
    and sha(HERE/'native_tdt_checkpointed_v3.py')==ADAPTER_SHA and sha(HERE/'preload_nemo_linker_v1.py')==PRELOAD_SHA)
  write_new(completion,{'kind':'native_tdt_training_backend_completion_v3','status':outcome,
    'run_id':run_id,'backend_ready_path':str(ready),'backend_ready_sha256':sha(ready),
    'sources_unchanged':unchanged,'error':error,'finished_at_utc':datetime.now(timezone.utc).isoformat()})
  require(unchanged,'Native launcher/trainer/memory source changed during the candidate')


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--native-provenance',required=True,type=Path)
 p.add_argument('--memory-launcher',required=True,type=Path)
 p.add_argument('--memory-launcher-sha256',required=True)
 p.add_argument('--trainer-sha256',required=True)
 p.add_argument('--validate-only',action='store_true')
 p.add_argument('memory_arguments',nargs=argparse.REMAINDER)
 main(p.parse_args())
