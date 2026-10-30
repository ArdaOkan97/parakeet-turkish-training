"""One guarded genuine-plan plus real CPU7 transport closure control; no GPU/native."""
from copy import deepcopy
import json,os,random,sys,threading,time
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research';sys.path[:0]=[str(S),str(R)]
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import canonical,digest
from bpe_final_DATA_schedule_context_v2 import normalize_schedule_state
from asr18_BPE9223_source_context_v2 import FullRunSourceContext9223
from asr18_resident_periodic_CPU_quiescence_v1 import close_private_preparation_for_DEV as old_close
from asr18_resident_periodic_CPU_quiescence_v2 import close_private_preparation_for_DEV
from fastmss_resident_feature_stream_v7 import ResidentFeatureProducer
from resource_watchdog_v2 import sample_memory
D=Path(__file__).parent;B=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000013';OUT=D/'genuine_context_real_CPU_actual_02'
require(not OUT.exists(),'One fresh E control');OUT.mkdir();process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
require(sys.flags.utf8_mode==1 and os.environ.get('PYTHONUTF8')=='1' and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'CPU mask, UTF8, all four thread caps1 before source entry')
began=time.monotonic();stop=threading.Event();monitor_errors=[];peak_parent=0;peak_family=0;sample_count=0

def guard():
 global peak_parent,peak_family,sample_count
 s=sample_memory(os.getpid(),'periodic_quiescence_genuine_context_real_CPU_control');m=s['system_memory'];sample_count+=1
 require(s['paired_sampling_within_one_second'] and m['commit_available_bytes']>=2000000000 and m['physical_available_bytes']>=1500000000,'Original real system/physical headroom')
 parent=s['worker_private_memory']['private_bytes'];family=parent
 for child in process.children(recursive=True):
  try:family+=child.memory_info().private
  except psutil.NoSuchProcess:pass
 peak_parent=max(peak_parent,parent);peak_family=max(peak_family,family)
 require(parent<=512*1024**2 and family<=2*1024**3 and time.monotonic()-began<180,'Finite512MiB parent/2GiB owned family/180s')
 require(__import__('shutil').disk_usage('E:/gpu_training_scratchpad').free>=100000000000,'Original E reserve')
 return s

def watch():
 try:
  while not stop.wait(.1):guard()
 except BaseException as error:monitor_errors.append(repr(error))
initial=guard();require(initial['system_memory']['commit_available_bytes']-2*1024**3-268435456>=2000000000 and initial['system_memory']['physical_available_bytes']-2*1024**3-268435456>=1500000000,'Full prospective family plus full extra256MiB, no freed-memory credit')
monitor=threading.Thread(target=watch);monitor.start();context=None;stream=None;global_rng=random.getstate();results=[]
try:
 config_pin=record(D/'configuration_candidate.json');config=read_json(config_pin['path'])
 previous_pin=record(I/'plans/plan_000200.json');expected_pin=record(I/'plans/plan_000201.json');previous=read_json(previous_pin['path']);expected=read_json(expected_pin['path'])
 state=normalize_schedule_state(previous['final_private_planning_sampler']);pointer=read_json(B/'checkpoints/update_000140000/pointer.json')
 require(digest(state)==expected['initial_sampler_sha256']==pointer['sampler_state_sha256'] and expected['initial_committed_updates']==140000 and len(expected['updates'])==20,'Real saved140k frontier and exact next private20')
 context=FullRunSourceContext9223(config['planning_arguments'],guard=guard);schedule=context.new_schedule();schedule.load_state_dict(deepcopy(state));before=deepcopy(schedule.state_dict())
 lifecycle_path=B/'lifecycle.json';lifecycle_existed=lifecycle_path.exists();lifecycle_pin=record(lifecycle_path)if lifecycle_existed else None
 lifecycle=read_json(lifecycle_path)if lifecycle_existed else {'completed_boundaries':[],'winner':config['qualified_source_winner']}
 require(isinstance(lifecycle,dict) and isinstance(lifecycle.get('completed_boundaries'),list) and isinstance(lifecycle.get('winner'),dict),'Exact production lifecycle exists/read/default branch and non-None schema before genuine planning')
 require(not lifecycle_existed or record(lifecycle_path)==lifecycle_pin,'Existing lifecycle immutable pin before genuine planning')
 lifecycle_evidence={'path':str(lifecycle_path),'actual_file_existed':lifecycle_existed,'existing_pin':lifecycle_pin,'production_default_used':not lifecycle_existed,'production_default_exact':lifecycle_existed or lifecycle=={'completed_boundaries':[],'winner':config['qualified_source_winner']},'source18':record(R/'run_ASR18_BPE9223_resident_RAM_training_v18.py'),'value':lifecycle}
 completed=len(lifecycle['completed_boundaries']);live=context.plan(schedule,completed_boundaries=completed)
 require(canonical(live)==canonical(expected) and digest(live)==digest(expected) and schedule.state_dict()==before,'Genuine original full coordinator plan equals frozen201 with no authoritative advance')
 require(live!=expected,'Reproduce original Python representation inequality')
 paths=[]
 for key in ('replay_rng','Portuguese_rng'):
  a=live['final_private_planning_sampler']['carried_replay'][key];b=expected['final_private_planning_sampler']['carried_replay'][key]
  require(type(a) is tuple and type(b) is list and canonical(a)==canonical(b),'Original carried Random.getstate tuple versus JSON list')
  paths.append('final_private_planning_sampler.carried_replay.'+key)
 normalized=deepcopy(expected);normalized['final_private_planning_sampler']=normalize_schedule_state(normalized['final_private_planning_sampler']);require(normalized==live,'ONLY original two RNG representation paths restore exact full equality')
 class NoTransport:
  def __getattr__(self,name):raise AssertionError('No transport side effect expected '+name)
 try:old_close(NoTransport(),(live,expected_pin),committed_updates=140000,guard=guard)
 except ValueError as error:require(str(error)=='Keep the exact previously submitted private plan','Exact real-plan original failure reproduced');results.append({'name':'original_helper1_genuine_plan_representation_failure','passed':True})
 else:raise AssertionError('Original helper1 unexpectedly accepted raw tuple/list comparison')
 for name in ('wrong_pin','target_change','transaction_change','sampler_RNG_change'):
  altered=deepcopy(live);bound=dict(expected_pin)
  if name=='wrong_pin':bound['sha256']='0'*64
  elif name=='target_change':altered['updates'][0]['rows'][0]['target_tokens']+=1
  elif name=='transaction_change':altered['updates'][0]['native_transaction_sha256']='0'*64
  else:altered['final_private_planning_sampler']['carried_replay']['replay_rng']=(0,(),None)
  try:close_private_preparation_for_DEV(NoTransport(),(altered,bound),committed_updates=140000,guard=guard)
  except ValueError:results.append({'name':name,'passed':True,'before_transport':True})
  else:raise AssertionError('Changed actual plan accepted '+name)
 journal=Path(pointer['journal_prefix']['path']);journal_stat=(journal.stat().st_size,journal.stat().st_mtime_ns)
 stream=ResidentFeatureProducer(config_pin,OUT/'CPU',queue_bytes=268435456,guard=guard)
 launcher={'pid':stream.process.pid,'create_time':psutil.Process(stream.process.pid).create_time(),'ppid':os.getpid()}
 stream.submit(expected_pin)
 quiescence=close_private_preparation_for_DEV(stream,(live,expected_pin),committed_updates=140000,guard=guard);stream=None
 closed=quiescence['CPU_actual0'];require(closed['actual_Popen_wait_code']==0 and closed['completed_updates']==20 and closed['transport_threads_joined'] and closed['pipe_handles_closed'] and quiescence['private_packets_discarded']==20,'ACTUAL CPU7 full20 drain, release, wait0, joined handles')
 require(closed['source']['protocol']==record(S/'fastmss_resident_feature_stream_v7.py'),'Actual unchanged CPU7 source, no fixture producer/source alias')
 for who in (launcher,closed['producer_identity']):
  try:alive=abs(psutil.Process(who['pid']).create_time()-who['create_time'])<.001
  except psutil.NoSuchProcess:alive=False
  require(not alive,'Actual CPU launcher/producer exact identities dead')
 require(schedule.state_dict()==before and canonical(live)==canonical(expected) and random.getstate()==global_rng and journal_stat==(journal.stat().st_size,journal.stat().st_mtime_ns),'Original sampler/RNG/plan and untouched whole training journal')
 from run_ASR18_BPE9223_resident_RAM_training_v18 import checked_periodic_quiescence_resume
 typed=checked_periodic_quiescence_resume(config['prior_resident_RAM_periodic_quiescence_failure'],base=config,folder=B,pointer_pin=record(B/'checkpoints/update_000140000/pointer.json'))
 require(typed['pending_periodic_DEV_updates']==140000 and typed['unsaved_tail_bytes']==0,'Actual final typed saved140k failure gate')
 context.close();require(context.closed and not context.resolver.metadata_locks.handles,'Actual readonly source context and deny-write handles closed');context=None
 require(not monitor_errors and 'torch'not in sys.modules,'Actual no parent model/GPU work and clean guard');guard()
 report={'kind':'ASR18_actual_genuine_plan140000_and_real_CPU7_quiescence_control_v1','passed':True,'producer':{'pid':os.getpid(),'create_time':process.create_time(),'ppid':os.getppid()},'source':record(__file__),'configuration':config_pin,'previous_plan':previous_pin,'frozen_next_plan':expected_pin,'actual_production_lifecycle_reader':lifecycle_evidence,'genuine_live_plan_canonical_sha256':digest(live),'full_canonical_equality':True,'raw_Python_equality_reproduced_false':True,'ONLY_two_original_RNG_representation_paths':paths,'all_other_python_plan_values_exact':True,'negative_controls':results,'actual_real_CPU_quiescence':quiescence,'actual_CPU_launcher':launcher,'actual_CPU_producer_and_launcher_observed_dead':True,'CPU_transport_is_REAL_unchanged_CPU7_not_method_stub':True,'original_context_and_lock_handles_closed':True,'actual_authoritative_sampler_native_or_journal_advance':False,'no_model_or_GPU_work':True,'sample_count':sample_count,'parent_private_peak_bytes':peak_parent,'owned_family_private_peak_bytes':peak_family,'initial_prospective_sample':initial,'elapsed_seconds':time.monotonic()-began,'no_pipeline_speed_claim':True}
finally:
 try:
  if stream is not None:stream.abort()
 finally:
  if context is not None:context.close()
  stop.set();monitor.join(3);require(not monitor.is_alive(),'Actual control monitor joined')
require(not monitor_errors,'No deferred monitor error');report['monitor_joined']=True;report['monitor_errors']=monitor_errors;write_json(OUT/'report.json',report);print(json.dumps(record(OUT/'report.json')),flush=True)
