"""Independent real-stack CPU guard review; imports no torch/CUDA runtime."""
import ast
import hashlib
import importlib.util
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/parakeet_turkish'
worker=OUT/'train_lora_batched.py'
helper=OUT/'cuda_cache_reclaim_v1.py'
launcher=OUT/'launch_with_cache_reclaim_v1.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('review_cache_helper',helper)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
checks=[]
def check(label,result):
 if not result:raise AssertionError(label)
 checks.append(label)
class MockCuda:
 def __init__(self):
  self.allocated=3_000_000_000;self.reserved=9_000_000_000;self.peak=4_000_000_000
  self.syncs=0;self.clears=0;self.fail_sync=False;self.fail_clear=False
  self.sentinel=object();self.rng={'seed':31,'position':19};self.backends={'tf32':True};self.live={'weights':17,'optimizer':39,'gradients':43}
 def synchronize(self,*args,**kwargs):
  self.syncs+=1
  if self.fail_sync:raise RuntimeError('original sync failure')
  return self.sentinel
 def memory_allocated(self):return self.allocated
 def memory_reserved(self):return self.reserved
 def max_memory_allocated(self):return self.peak
 def empty_cache(self):
  self.clears+=1
  if self.fail_clear:raise RuntimeError('original clear failure')
  self.reserved=self.allocated
def callsite(path=worker,line=539,name='run'):
 # Real CPython frames execute synthetic source in memory only. No worker
 # source file is edited/imported; exact scope and stack depth are exercised.
 namespace={'torch':SimpleNamespace(cuda=cuda)}
 source='\n'*(line-2)+f'def {name}():\n    return torch.cuda.synchronize()\n'
 exec(compile(source,str(path),'exec'),namespace)
 return namespace[name]

source=worker.read_text(encoding='utf-8');tree=ast.parse(source)
run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
check('frozen worker byte identity',sha(worker)==module.FROZEN_TRAINER_SHA256)
check('validated unique synchronize line539',module.validated_update_sync_line(worker)==539)
check('Adam before hook',source.index('optimizer.step()') < source.index('torch.cuda.synchronize()',source.index('optimizer.step()')))
check('new token row scaling before hook',source.index('updated.copy_(old + args.new_token_lr_multiplier*(updated-old))') < source.index('torch.cuda.synchronize()',source.index('optimizer.step()')))
check('cleanup in measured training interval',source.index('step_start = time.perf_counter()') < source.index('optimizer.step()') < source.index('torch.cuda.synchronize()',source.index('optimizer.step()')) < source.index('train_wall += time.perf_counter()-step_start'))
helper_text=helper.read_text(encoding='utf-8')
check('no RNG/backend/optimizer mutation APIs',not any(x in helper_text for x in ['manual_seed(','set_rng_state(','allow_tf32 =','optimizer.step(','backward(','register_forward_hook(']))
launchtext=launcher.read_text(encoding='utf-8')
check('launcher cannot attach active process',not any(x in launchtext for x in ['OpenProcess(','CreateRemoteThread','WriteProcessMemory','settrace(','setprofile(']))
check('cache evidence separate from IO evidence',"args.provenance.resolve() == args.cache_provenance.resolve()" in launchtext and "'active_process_attachment': False" in launchtext)
check('existing evidence is rejected before worker import',launchtext.index('if path.exists():') < launchtext.index('import torch') and "raise FileExistsError('Use a new distinct provenance path" in launchtext)
check('launcher binds frozen IO before and after hashes','frozen_io_launcher_sha256_after=file_sha' in launchtext and 'frozen_io_helper_sha256_after=file_sha' in launchtext)

cuda=MockCuda();hook=module.CudaCacheReclaimer(cuda,worker,threshold_gb=8,max_events=2)
original=cuda.synchronize
hook.install();rng=random.getstate();snapshot=(dict(cuda.rng),dict(cuda.backends),dict(cuda.live))
check('correct real frame invokes original return',callsite()() is cuda.sentinel)
check('one post-sync reclamation after threshold exceeded',cuda.syncs==1 and cuda.clears==1 and hook.eligible_calls==1 and hook.reclamations==1 and cuda.reserved==cuda.allocated)
check('mock live state remains unchanged',snapshot==(cuda.rng,cuda.backends,cuda.live))
check('Python RNG state unchanged',random.getstate()==rng)
cuda.reserved=8_000_000_000
callsite()()
check('exact threshold does not reclaim',cuda.clears==1 and hook.eligible_calls==2)
cuda.reserved=7_000_000_000
callsite()()
check('below threshold does not reclaim',cuda.clears==1 and hook.eligible_calls==3)
cuda.reserved=9_000_000_000
for fn in [callsite(line=540),callsite(name='evaluate'),callsite(path=ROOT/'work/nonworker_cpu_fixture.py')]:fn()
check('wrong line/function/path calls ignored',cuda.clears==1 and hook.eligible_calls==3 and hook.ignored_calls==3)
cuda.fail_sync=True
try:callsite()();raise AssertionError('missing sync exception')
except RuntimeError as e:check('original synchronize exception propagated',str(e)=='original sync failure')
check('sync failure no stats/cleanup eligibility increment',hook.eligible_calls==3 and cuda.clears==1)
cuda.fail_sync=False
for _ in range(3):cuda.reserved=9_000_000_000;callsite()()
e=hook.evidence()
check('bounded event queue and omission arithmetic',len(e['events'])==2 and e['reclamations']==4 and e['events_omitted']==2)
check('positive cleanup accounted in measured timer',e['cleanup_seconds_in_training_timing']>=sum(x['cleanup_seconds'] for x in e['events'])>=0)
cuda.reserved=9_000_000_000;cuda.fail_clear=True
try:callsite()();raise AssertionError('missing cleanup exception')
except RuntimeError as e:check('cleanup failure propagated',str(e)=='original clear failure')
e=hook.evidence()
check('failure event preserved/counts correct',e['failed_reclamations']==1 and e['events'][-1]['status']=='failed' and e['events_omitted']==3)
hook.restore()
check('original synchronize restored after failure',cuda.synchronize==original)
hook.restore();check('second restore idempotent',cuda.synchronize==original)
for value in [0.,-1.,12.01,float('nan')]:
 try:module.CudaCacheReclaimer(cuda,worker,threshold_gb=value);raise AssertionError('bad threshold accepted')
 except ValueError:check('invalid threshold rejected '+str(value),True)
for value in [0,1025]:
 try:module.CudaCacheReclaimer(cuda,worker,max_events=value);raise AssertionError('bad event cap accepted')
 except ValueError:check('invalid event bound rejected '+str(value),True)
check('worker bytes remain frozen after all real-stack guards',sha(worker)==module.FROZEN_TRAINER_SHA256)
report={'schema':1,'passed':True,'observed_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Independent CPU source/real-stack mocked allocator guard review; no torch import, GPU/model/process mutation or original source edits','checks_passed':len(checks),'checks':checks,'source_sha256':{p.name:sha(p) for p in [worker,helper,launcher]},'scope_binding':{'worker_function':'run','worker_line':539,'optimizer_end':True,'cleanup_in_train_wall_seconds':True,'microbatch_and_evaluation_callers_ignored':True,'existing_provenance_overwrite_rejected':True},'limits':['Mocked CUDA release cannot establish actual GPU numerical parity, memory recovery, driver shared-memory behavior or cleanup performance.','State preservation is direct source/API inspection plus mocked live/RNG/backend invariants, not an executed full-model CPU/GPU optimizer test.','Successor orchestration must supply NEW distinct cache/I/O provenance paths; launcher explicitly rejects existing paths.','An8GB total-reserved trigger may still fire when live allocations exceed8GB; empty_cache only reclaims inactive cache and cannot force live tensor memory below threshold.']}
destination=OUT/'cuda_cache_reclaim_independent_cpu_review.json'
destination.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'checks':len(checks),'helper_sha256':report['source_sha256'][helper.name],'output':str(destination)}))
