"""INACTIVE finite CPU-only method-stub controls for the proposed pressure gate.
No actual native/model/audio/packet/optimizer/sampler/journal replay or speed claim.
Execution requires a separate exact ROOT scope; this source does not authorize itself.
"""
import argparse,copy,hashlib,importlib.util,json,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).parent
EXPECTED_GATE={'path':str(F/'asr18_native_update_commit_pressure_gate_v1.py'),'bytes':2957,
 'sha256':'6e1473a81ff18478a6cd303fa67e1dd76f617f6a80a197696e4c383c72486047'}
def require(v,m):
 if not v:raise ValueError(m)
def pin(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def main(a):
 began=time.monotonic();psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'CPU-only original thread caps')
 def guard():
  s=sample_memory(os.getpid(),'inactive_native_pressure_gate_method_stub_CPU_control');m=s['system_memory']
  require(s['paired_sampling_within_one_second']and s['worker_private_memory']['private_bytes']<512*1024**2 and m['commit_available_bytes']>=2000000000+268435456 and m['physical_available_bytes']>=1500000000+268435456 and shutil.disk_usage('E:/').free>=100000000000 and time.monotonic()-began<180,'Original finite metadata limits and full reservations')
  return s
 initial=guard();require(initial['system_memory']['commit_available_bytes']>=2000000000+268435456+512*1024**2 and initial['system_memory']['physical_available_bytes']>=1500000000+268435456+512*1024**2,'Full additional512MiB control admission')
 require(pin(EXPECTED_GATE['path'])==EXPECTED_GATE,'Frozen gate source')
 review=json.loads(Path(a.ROOT_review).read_text(encoding='utf-8-sig'))
 require(review['accepted']is True and review['reviewer']=='/root' and review['scope']=='ASR18_native_preupdate_commit_pressure_scheduling_CPU_control_v1' and review['gate_source']==EXPECTED_GATE,'Separate exact ROOT finite-control authorization')
 out=Path(a.output).resolve();require(out.is_relative_to(Path(r'E:\gpu_training_scratchpad'))and not out.exists(),'Fresh E output only');out.mkdir()
 me=psutil.Process();pa=me.parent();identities={'producer':{'pid':me.pid,'create_time':me.create_time(),'ppid':me.ppid()},'launcher':None if pa is None else {'pid':pa.pid,'create_time':pa.create_time(),'ppid':pa.ppid()}}
 (out/'actual_process_entry.json').write_text(json.dumps(identities,indent=2)+'\n',encoding='utf-8')
 spec=importlib.util.spec_from_file_location('inactive_pressure_gate_exact',EXPECTED_GATE['path']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 policy=copy.deepcopy(module.POLICY);gate=module.wait_for_native_update_commit_pressure
 require(policy['native_actual_commit_headroom_bytes']==2000000000 and policy['native_actual_physical_headroom_bytes']==1500000000 and policy['full_additional_prefetch_host_reservation_bytes']==268435456 and policy['additional_native_commit_pressure_buffer_bytes']==2500000000 and not policy['freed_memory_credit_claimed']and not policy['workload_forecast_claimed'],'Explicit stricter scheduling, no changed original guards or credit')
 commit=4768435456;physical=1768435456;checks={}
 def case(samples,*,observe_error=None,check_error=None,changed_policy=None):
  now=[0.0];calls=[];queue=list(samples);state={'sampler':17,'draws':0,'native':0,'journal':0}
  def check():
   calls.append('check')
   if check_error is not None:raise check_error
  def observe():
   calls.append('observe')
   if observe_error is not None:raise observe_error
   c,p=queue.pop(0)if len(queue)>1 else queue[0]
   if c<2000000000 or p<1500000000:raise ValueError('Original actual floor retained')
   return {'system_memory':{'commit_available_bytes':c,'physical_available_bytes':p}}
  def sleep(seconds):
   calls.append(('sleep',seconds));now[0]+=seconds
  try:result=gate(observe=observe,check=check,policy=policy if changed_policy is None else changed_policy,clock=lambda:now[0],sleep=sleep);error=None
  except BaseException as e:result=None;error=e
  require(state=={'sampler':17,'draws':0,'native':0,'journal':0},'Stub has no authoritative callbacks or operation')
  return result,error,calls,now[0]
 r,e,c,w=case([(commit,physical)]);require(e is None and r['observations']==1 and w==0 and c==['check','observe','check'],'Exact threshold/order immediate admission');checks['exact_threshold_and_order']=True
 r,e,c,w=case([(commit-1,physical),(commit,physical)]);require(e is None and r['observations']==2 and w==.5 and c==['check','observe',('sleep',.5),'check','observe','check'],'One-byte commit deficit defers');checks['one_byte_commit_deficit']=True
 r,e,c,w=case([(commit,physical-1),(commit,physical)]);require(e is None and r['observations']==2 and w==.5,'One-byte physical deficit defers');checks['one_byte_physical_deficit']=True
 r,e,c,w=case([(commit-1,physical)]);require(isinstance(e,TimeoutError)and r is None and w==300 and c.count('observe')==601 and all(x[1]<=.5 for x in c if isinstance(x,tuple)),'Finite exact300s fake-clock timeout, no retry');checks['bounded_timeout_no_retry']=True
 marker=RuntimeError('exact original monitor check failure');r,e,c,w=case([(commit,physical)],check_error=marker);require(e is marker and c==['check'],'Same check exception object');checks['check_exception_identity']=True
 marker=ValueError('exact original observer failure');r,e,c,w=case([(commit,physical)],observe_error=marker);require(e is marker and c==['check','observe'],'Same observe exception object');checks['observe_exception_identity']=True
 changed=copy.deepcopy(policy);changed['additional_native_commit_pressure_buffer_bytes']-=1
 r,e,c,w=case([(commit,physical)],changed_policy=changed);require(isinstance(e,ValueError)and not c,'Changed policy rejected before callbacks');checks['changed_policy_rejected']=True
 changed=copy.deepcopy(policy);changed['freed_memory_credit_claimed']=True
 r,e,c,w=case([(commit,physical)],changed_policy=changed);require(isinstance(e,ValueError)and not c,'Freed credit rejected');checks['freed_credit_rejected']=True
 r,e,c,w=case([(1943629824,10373808128)]);require(isinstance(e,ValueError)and str(e)=='Original actual floor retained'and r is None,'Historical actual crossing still fails; no softened floor');checks['actual_failure_still_rejected']=True
 require(pin(EXPECTED_GATE['path'])==EXPECTED_GATE,'Gate unchanged after control');guard()
 result={'kind':'ASR18_actual_finite_native_pressure_gate_METHOD_STUB_CPU_control_v1','passed':True,'proof_scope':'method-stub arithmetic/order/exception/no-authoritative-operation only; not real native allocation or demonstrated failure prevention','gate_source':EXPECTED_GATE,'policy':policy,'checks':checks,'actual_process_identities':identities,'additional_NATIVE_pressure_buffer_is_unmeasured_conservative_policy':True,'external_midupdate_commit_change_still_can_fail_original_guard':True,'actual_native_optimizer_updates':0,'actual_sampler_draws':0,'actual_audio_decode':0,'actual_GPU_activity':False,'guard_initial':initial,'guard_final':guard()}
 path=out/'report.json';path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({'report':pin(path),'identities':identities}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ROOT-review',required=True);p.add_argument('--output',required=True);main(p.parse_args())
