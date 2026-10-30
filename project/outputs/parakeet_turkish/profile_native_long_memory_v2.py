"""Root-owned resource-only native v2 full B8x2 TRAIN stress update.

Uses sixteen accepted TRAIN rows, including all four largest Turkish joint
lattices. No data or model is modified, no heldout prediction is generated.
"""
import argparse
from datetime import datetime,timezone
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
ADAPTER='24e6f01a63c482d8451d72af9e21994332c0d0be53cbefbdf4c295f659090fd5'
PLAN_SHA='3ad2f08d7c6d3bcc8b4f39882cea9953054c5b461cfd10ab2d235dd1af9eee95'
WEIGHT_SHA='56e51fae1fde387034942707895ce6ff030c8eb5a52ef2bf49bcc066e25e9b81'
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)
def main(args):
 require(args.gpu_window_authorized,'Only root launches a sole-GPU authorized window')
 require(not args.output.exists(),'New report path required')
 require(sha(args.selection)==PLAN_SHA and sha(args.weights)==WEIGHT_SHA,'Accepted TRAIN selection/native63 state changed')
 require(sha(HERE/'native_tdt_checkpointed_v2.py')==ADAPTER,'Frozen cleanup-only adapter2 changed')
 plan=json.loads(args.selection.read_text(encoding='utf-8'))
 require(all(len(group)==8 for group in plan['batches']) and len(plan['batches'])==2,'Actual B8x2 required')
 for row in sum(plan['batches'],[]):require(sha(Path(row['audio_filepath']))==row['wav_sha256'],'Selected WAV changed')
 from preload_nemo_linker_v1 import preload
 preloaded=preload()  # Every Torch/model/native import must follow preload.
 import torch
 from profile_batched_training import initialize
 from batched_lora_tr_metadata_v1 import collate_training_batch,batch_objective
 import native_tdt_checkpointed_v2 as native
 from resource_watchdog_v2 import ResourceGuard,sample_memory,worker_process_commit_reason
 torch.set_num_threads(4)
 total=torch.cuda.get_device_properties(0).total_memory
 torch.cuda.set_per_process_memory_fraction(args.cap_gb*1e9/total)
 guard=ResourceGuard();samples=[];start=time.monotonic()
 report={'kind':'native_v2_long_TRAIN_memory_resource_pilot','status':'running','passed':False,
  'cap_gb':args.cap_gb,'end_cache_threshold_gb':4,'native_inactive_threshold_bytes':750000000,
  'max_joint_bytes':1700000000,'source_checkpoint':{'path':str(args.weights),'sha256':WEIGHT_SHA,'step':63},
  'selection':{'path':str(args.selection),'sha256':PLAN_SHA},'source_ids':plan['source_ids'],
  'source_sha256':{str(HERE/n):sha(HERE/n) for n in ('profile_native_long_memory_v2.py','profile_batched_training.py','native_tdt_checkpointed_v2.py','batched_lora_tr_metadata_v1.py','preload_nemo_linker_v1.py')},
  'preload':preloaded,'pid':os.getpid(),'updates':[],'requested_updates':args.updates,
  'fresh_Adam_resource_only':True,'no_exact_optimizer_resume_claim':True,'heldout_evaluated':False,
  'throughput_representative':False,'cache_cleanup_inside_update_time':True}
 def observe(phase):
  sample=sample_memory(os.getpid(),phase)
  reason,_=guard.observe(sample['system_memory'],sample['worker_private_memory'],observed_monotonic=sample['observed_monotonic'])
  reason=reason or worker_process_commit_reason(sample['calling_process_commit_memory'],os.getpid())
  samples.append(sample)
  require(reason is None,'Resource guard: '+str(reason))
  require(time.monotonic()-start<args.max_wall_seconds,'Bounded resource pilot deadline')
  return sample
 original=native.reclaim_inactive_cuda_cache
 def monitored(device,phase):
  observe('native_'+phase+'_before_cleanup')
  original(device,phase)
 native.reclaim_inactive_cuda_cache=monitored
 try:
  observe('pre_model')
  model,processor,teacher,old_size,state=initialize(SimpleNamespace(weights=args.weights,no_encoder_checkpoint=False))
  require(state['step']==63 and len(state['weights'])==207,'Actual native63 layout required')
  del state
  trainables=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
  optimizer=torch.optim.AdamW([{'params':[p for n,p in trainables if not n.endswith(('lora_A','lora_B'))],'lr':5e-5,'weight_decay':.01},
   {'params':[p for n,p in trainables if n.endswith(('lora_A','lora_B'))],'lr':3e-4,'weight_decay':0.}])
  rowparams=[model.decoder.embedding.weight,model.joint.head.weight,model.joint.head.bias]
  fn=partial(native.chunked_tdt_loss,max_joint_bytes=1700000000)
  for step in range(1,args.updates+1):
   observe('before_whole_B8x2_update');torch.cuda.synchronize();began=time.perf_counter();optimizer.zero_grad(set_to_none=True)
   values=[]
   for group in plan['batches']:
    batch=collate_training_batch(processor,group,blank_id=model.config.blank_token_id,old_size=old_size,device='cuda')
    result=batch_objective(model,teacher,batch,old_size=old_size,replay_kl=5.,duration_kl=.5,feature_distillation=1.,
     chunk_size=16,tdt_loss_fn=fn,tdt_host_lengths=True,use_autocast=True)
    require(bool(torch.isfinite(result.loss)),'Nonfinite whole objective');(result.loss/2).backward()
    values.append(result.detached_metrics());del result,batch
   norm=torch.nn.utils.clip_grad_norm_([p for _,p in trainables],1.,error_if_nonfinite=True)
   before=[p[old_size:model.config.vocab_size].detach().clone() for p in rowparams];optimizer.step()
   with torch.no_grad():
    for p,old in zip(rowparams,before):
     new=p[old_size:model.config.vocab_size];new.copy_(old+6*(new-old))
   torch.cuda.synchronize()
   if torch.cuda.memory_reserved()>4e9:torch.cuda.empty_cache()
   wall=time.perf_counter()-began
   sample=observe('after_whole_B8x2_Adam')
   report['updates'].append({'update':step,'wall_seconds_including_first_JIT_if_any':wall,
    'components':values,'gradient_norm':float(norm),'allocated_bytes':torch.cuda.memory_allocated(),
    'reserved_bytes':torch.cuda.memory_reserved(),'system_commit_fraction':sample['system_memory']['commit_fraction']})
   write(args.output,report);print(json.dumps(report['updates'][-1]),flush=True)
  report.update(status='complete',passed=True)
 except Exception as e:
  report.update(status='resource_or_OOM_failure',error=str(e),traceback=traceback.format_exc(limit=8))
 finally:
  native.reclaim_inactive_cuda_cache=original
  report.update(native_cleanup_evidence=native.cache_cleanup_evidence(),
   peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
   memory_observations=samples[-128:],max_system_commit_fraction=max((r['system_memory']['commit_fraction'] for r in samples if r['system_memory'].get('available')),default=None),
   minimum_owned_self_headroom_bytes=min((r['calling_process_commit_memory']['ullAvailPageFile_bytes'] for r in samples if r['calling_process_commit_memory'].get('available')),default=None),
   finished_at_utc=datetime.now(timezone.utc).isoformat())
  write(args.output,report)
 print(json.dumps({'passed':report['passed'],'status':report['status'],'output':str(args.output)}),flush=True)
 return 0 if report['passed'] else 1
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--selection',type=Path,default=HERE/'native_long_resource_pilot_selection_v1.json')
 p.add_argument('--weights',type=Path,default=ROOT/'work/adaptive_20261011_native_data_01_public_rank8_control_latest.pt')
 p.add_argument('--cap-gb',type=int,choices=(6,7),default=6)
 p.add_argument('--updates',type=int,default=3)
 p.add_argument('--max-wall-seconds',type=float,default=300)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--gpu-window-authorized',action='store_true')
 raise SystemExit(main(p.parse_args()))
