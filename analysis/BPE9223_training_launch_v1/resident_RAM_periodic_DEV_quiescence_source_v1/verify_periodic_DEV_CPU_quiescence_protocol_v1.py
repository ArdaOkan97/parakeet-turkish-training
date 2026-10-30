"""Finite CPU-only helper order/exception and typed-predecessor metadata controls."""
import ast,copy,hashlib,json,os,shutil,sys,threading,time
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research';sys.path[:0]=[str(S),str(R)]
from atomic_io_retry_v1 import read_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from resource_watchdog_v2 import sample_memory
from asr18_resident_periodic_CPU_quiescence_v1 import close_private_preparation_for_DEV
D=Path(__file__).parent; OUT=D/'actual_CPU_01';require(not OUT.exists(),'One actual control only');OUT.mkdir()
process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);began=time.monotonic();stop=threading.Event();errors=[];samples=[]
require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('','-1') and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'CPU-only four thread caps')
def guard():
 s=sample_memory(os.getpid(),'periodic_DEV_CPU_quiescence_metadata_control');m=s['system_memory'];require(s['paired_sampling_within_one_second'],'Fresh paired sample')
 require(m['commit_available_bytes']>=2000000000 and m['physical_available_bytes']>=1500000000,'Unchanged system and physical headroom')
 require(s['worker_private_memory']['private_bytes']<512*1024**2 and time.monotonic()-began<180,'Finite512MiB/180s metadata control')
 require(shutil.disk_usage('E:/gpu_training_scratchpad').free>=100000000000,'E100GB reserve');samples.append(s)
 return s
initial=guard();require(initial['system_memory']['commit_available_bytes']-512*1024**2-268435456>=2000000000,'Full prospective512MiB fixture plus extra256MiB and unchanged2GB reserve')
def watch():
 try:
  while not stop.wait(.1):guard()
 except BaseException as error:errors.append(repr(error))
thread=threading.Thread(target=watch);thread.start()
plan_path=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01\invocations\owner_000012\plans\plan_000081.json');plan=read_json(plan_path);plan_pin=record(plan_path)
require(plan['initial_committed_updates']==136000 and len(plan['updates'])==20,'Exact real unconsumed ordered plan')
class Injected(Exception):pass
class FakeStream:
 def __init__(self,mode='good'):
  self.mode=mode;self.calls=[];self.live=None
 def next_update(self,sha,index):
  self.calls.append(['next_update',sha,index]);require(self.live is None,'Only one fixture live packet')
  if self.mode=='next_error':raise Injected('next error')
  entry=plan['updates'][index];metadata={k:copy.deepcopy(entry[k])for k in ('native_transaction_sha256','before_sampler_sha256','after_sampler_sha256','batch_id')};metadata['update_index']=index
  packet={'plan_sha256':sha,'metadata':metadata,'update':metadata,'token':str(index),'arrays':['fixture-only no tensor/audio arrays']}
  if index==0:
   if self.mode=='wrong_plan':packet['plan_sha256']='0'*64
   if self.mode=='wrong_index':metadata['update_index']=1
   if self.mode=='wrong_transaction':metadata['native_transaction_sha256']='0'*64
   if self.mode=='wrong_sampler':metadata['after_sampler_sha256']='0'*64
  self.live=packet['token'];return packet
 def release(self,token):
  self.calls.append(['release',token]);require(token==self.live,'Exact live fixture token')
  if self.mode=='release_error':raise Injected('release error')
  self.live=None
 def close(self):
  self.calls.append(['close']);require(self.live is None,'No unreleased fixture packet')
  if self.mode=='close_error':raise Injected('close error')
  return {'kind':'CPU_METHOD_STUB_NOT_ACTUAL_PROCESS0','actual_Popen_wait_code':1 if self.mode=='nonzero_close' else 0,'transport_threads_joined':True,'pipe_handles_closed':True}
controls=[]
try:
 before_global_state={'sampler':'unchanged','journal':'unchanged','native_updates':136000};frozen=copy.deepcopy(before_global_state)
 stream=FakeStream();result=close_private_preparation_for_DEV(stream,(plan,plan_pin),committed_updates=136000,guard=guard)
 expected=[]
 for index in range(20):expected.extend([['next_update',digest(plan),index],['release',str(index)]])
 expected.append(['close']);require(stream.calls==expected and result['private_packets_discarded']==20 and result['unconsumed_private_plan']==plan_pin and before_global_state==frozen,'All20 exact ordered release/close calls and no authoritative operation')
 controls.append({'name':'real20update_plan_order_release_close','passed':True,'CPU_transport_is_explicit_method_stub':True,'native_or_sampler_operations':0})
 empty=FakeStream();empty_result=close_private_preparation_for_DEV(empty,None,committed_updates=136000,guard=guard);require(empty.calls==[['close']] and empty_result['private_packets_discarded']==0,'Empty private queue uses genuine-close API only');controls.append({'name':'no_private_plan_close_only','passed':True})
 for mode in ('wrong_plan','wrong_index','wrong_transaction','wrong_sampler','next_error','release_error','close_error','nonzero_close'):
  stream=FakeStream(mode)
  try:close_private_preparation_for_DEV(stream,(plan,plan_pin),committed_updates=136000,guard=guard)
  except (ValueError,Injected) as error:
   require(before_global_state==frozen,'No authoritative state mutation on exception');controls.append({'name':mode,'passed':True,'actual_exception_type':type(error).__name__,'close_not_falsely_claimed':True})
  else:raise ValueError('Expected rejection '+mode)
 bad_pin=dict(plan_pin,sha256='0'*64)
 try:close_private_preparation_for_DEV(FakeStream(),(plan,bad_pin),committed_updates=136000,guard=guard)
 except ValueError:controls.append({'name':'wrong_plan_pin','passed':True})
 else:raise ValueError('Wrong immutable pin accepted')
 stream=FakeStream()
 def reject_guard():raise Injected('guard error')
 try:close_private_preparation_for_DEV(stream,(plan,plan_pin),committed_updates=136000,guard=reject_guard)
 except Injected:require(stream.calls==[],'Guard exception before packet actions');controls.append({'name':'guard_exception_before_transport','passed':True})
 else:raise ValueError('Guard exception swallowed')
 from run_ASR18_BPE9223_resident_RAM_training_v17 import checked_DEV_admission_resume
 config=read_json(D/'configuration_candidate.json');pointer_pin=record(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01\checkpoints\update_000136000\pointer.json')
 predecessor=checked_DEV_admission_resume(config['prior_resident_RAM_DEV_admission_failure'],base=config,folder=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01'),pointer_pin=pointer_pin)
 require(predecessor['pending_periodic_DEV_updates']==136000 and predecessor['unsaved_tail_bytes']==0,'Real typed saved136000 metadata gate')
 controls.append({'name':'actual_typed136000_predecessor_metadata','passed':True})
 altered=dict(pointer_pin,sha256='0'*64)
 try:checked_DEV_admission_resume(config['prior_resident_RAM_DEV_admission_failure'],base=config,folder=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01'),pointer_pin=altered)
 except ValueError:controls.append({'name':'typed_pointer_pin_mismatch','passed':True})
 else:raise ValueError('Wrong typed pointer accepted')
 old_tree=ast.parse((R/'run_ASR18_BPE9223_resident_RAM_training_v16.py').read_text(encoding='utf-8-sig'));new_tree=ast.parse((R/'run_ASR18_BPE9223_resident_RAM_training_v17.py').read_text(encoding='utf-8-sig'))
 old_nodes={n.name:ast.dump(n,include_attributes=False)for n in old_tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name!='main'};new_nodes={n.name:ast.dump(n,include_attributes=False)for n in new_tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name!='main'}
 require(all(new_nodes[k]==v for k,v in old_nodes.items()),'All existing non-entry ASTs unchanged');controls.append({'name':'all_existing_nonentry_ASTs_unchanged','passed':True,'names':list(old_nodes)})
 guard();require(not errors,'Actual monitor has no error')
finally:
 stop.set();thread.join(3);require(not thread.is_alive(),'Actual observer monitor joined')
report={'kind':'ASR18_actual_CPU_periodic_DEV_private_quiescence_order_exception_metadata_control_v1','passed':True,'producer':{'pid':os.getpid(),'create_time':process.create_time(),'ppid':os.getppid()},'controls':controls,'real_frozen_plan':plan_pin,'helper':record(S/'asr18_resident_periodic_CPU_quiescence_v1.py'),'driver':record(R/'run_ASR18_BPE9223_resident_RAM_training_v17.py'),'configuration':record(D/'configuration_candidate.json'),'fixture':record(__file__),'CPU_transport_method_stubs_not_real_CPU_producer_run':True,'no_model_audio_GPU_native_or_sampler_draw':True,'existing_CPU7_transport_source_unchanged':record(S/'fastmss_resident_feature_stream_v7.py'),'monitor_joined':True,'monitor_errors':errors,'sample_count':len(samples),'peak_private_bytes':max(s['worker_private_memory']['private_bytes']for s in samples),'initial_prospective_sample':initial,'elapsed_seconds':time.monotonic()-began,'no_production_admission_claimed':True}
out=OUT/'report.json';out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(record(out)),flush=True)
