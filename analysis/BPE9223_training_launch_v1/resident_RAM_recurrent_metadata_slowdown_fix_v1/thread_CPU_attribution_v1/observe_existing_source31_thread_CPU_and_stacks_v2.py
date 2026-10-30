"""INACTIVE one60s exact existing-process read-only attribution. No GPU or state call."""
import argparse,hashlib,json,math,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import psutil
from owned_thread_times_v2 import ThreadQuery,ThreadSampler,profile_role_mapping
from finite_thread_counter_mapping_fixture_v2 import profile_profiles
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).parent;EXE=F.parent/'py_spy_0_4_2_win_amd64_v1/py-spy.exe'
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000023'
OUT=F/'existing_source31_thread_CPU_actual_01'
TARGET={'pid':2456,'create_time':1792752131.0863705,'ppid':41672}
DRIVER_SHA='4d73b710332552cbbad10246f1fe07a627fa0690b9657f4c61a6cfac20160bc1'
RUNTIME_SHA='d62979c101b8cc4e5c95d4930c6f2a27d5333b893363db8b191505148f3cb7ed'
PROTECTED={2456,38964,26892,41672,34044}
LIMIT=536870912;MAX_PROFILE=67108864;MAX_COUNTER_BYTES=33554432
def require(v,m):
 if not v:raise ValueError(m)
def read(p,cap=8*1024**2):
 p=Path(p);require(p.stat().st_size<=cap,'Bounded compact/profile JSON');return json.loads(p.read_text(encoding='utf-8-sig'))
def pin(p,cap=8*1024**2):
 p=Path(p).resolve();require(p.stat().st_size<=cap,'Bounded pin');b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,indent=2);f.write('\n')
def identity(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def exact(v):
 try:return abs(psutil.Process(v['pid']).create_time()-v['create_time'])<.001
 except psutil.NoSuchProcess:return False
def health():
 require(exact(TARGET),'Same exact native2456 must remain alive')
 boot=read(I/'boot.json');progress=read(B/'progress/latest.json')
 require(boot['producer']==TARGET and boot['source']['sha256']==DRIVER_SHA and boot['configuration']['sha256']==RUNTIME_SHA,'Exact original source31/I23/native/runtime')
 require(pin(boot['source']['path'])==boot['source']and pin(I/'configuration.json')==boot['configuration'],'Same immutable admitted source/config')
 require(progress['producer']==TARGET and progress['source']==boot['source']and progress['planned_native_updates']==401072 and progress['administrative_update_stop']is None and not progress['optimizer_poisoned']and not progress['schedule_poisoned']and not(I/'actual_failure.json').exists(),'Healthy same full401072 frontier; no state changes')
 return {'progress':progress,'boot':pin(I/'boot.json'),'runtime':pin(I/'configuration.json')}
def io_sample():return dict(psutil.Process(TARGET['pid']).io_counters()._asdict())
def main(a):
 me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);began=time.monotonic();exact_external_interpreter_launcher=identity(me.parent())
 require(os.environ['CUDA_VISIBLE_DEVICES']=='-1'and os.environ['PYTHONUTF8']=='1'and all(os.environ[k]=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),'Separate E CPU-only observer process environment')
 review=read(a.ROOT_review)
 require(review['accepted']is True and review['reviewer']=='/root'and review['scope']=='ASR18_existing_source31_GetThreadTimes_nonblocking60s_attribution_v1','Separate exact LIVE execution scope required')
 require(review['live_observer_source']==pin(__file__)and review['target_identity']==TARGET and review['output_path']==str(OUT.resolve()),'Exact source/target/output scope')
 proof=read(review['completed_new_counter_CPU_review']['path']);require(pin(review['completed_new_counter_CPU_review']['path'])==review['completed_new_counter_CPU_review']and proof['accepted']is True and proof['reviewer']=='/root'and proof['scope']=='ASR18_GetThreadTimes_new_toy_nativeTID_mapping_actual_CPU_v1'and proof['thread_counter_helper_source']==pin(F/'owned_thread_times_v2.py')and proof['all_exact_metadata_identities_dead']is True,'Genuine NEW counter/role/handle CPU proof, not old py-spy toy substitution')
 inputs=review['input_pins'];require(all(pin(v['path'])==v for v in inputs),'All exact inputs before LIVE observation')
 require(not OUT.exists(),'One fresh bounded source31 observation; no automatic rerun')
 child=None;child_id=None;sampler=None;peak=0;last=None;observed={};cleanup=[];error=None;value={};opened_before=ThreadQuery.opened-ThreadQuery.closed_count
 def remember(p):
  v=identity(p);observed[(v['pid'],v['create_time'])]=v;return v
 remember(me)
 def guard(admission=False):
  nonlocal peak,last
  require(time.monotonic()-began<120 and exact(TARGET),'Bounded observer; actual original native must stay live')
  require(exact(exact_external_interpreter_launcher),'Exact observer interpreter-launcher parent sampled in family')
  family=[me,psutil.Process(exact_external_interpreter_launcher['pid'])]
  if child_id is not None and exact(child_id):
   proc=psutil.Process(child_id['pid']);family.extend([proc,*proc.children(recursive=True)])
  total=0
  for proc in family:
   try:
    who=identity(proc)
    if proc.pid!=exact_external_interpreter_launcher['pid']:remember(proc)
    require(who['pid']not in PROTECTED,'Owned observer/profiler family only')
    s=sample_memory(proc.pid,'source31_thread_CPU_readonly_attribution');require(s['paired_sampling_within_one_second'],'Original paired memory sample');total+=s['worker_private_memory']['private_bytes'];last=s
   except psutil.NoSuchProcess:continue
  peak=max(peak,total);require(total<=LIMIT,'Total owned observer/profiler family512MiB')
  m=last['system_memory'];extra=LIMIT if admission else 0
  require(m['commit_available_bytes']>=(4518435456+extra if admission else 2268435456)and m['physical_available_bytes']>=1768435456+extra and shutil.disk_usage('E:/').free>=100000000000,'FULL225 cushion+FULLprefetch+prospective512 admission; original runtimefloors+FULLprefetch; E100GB')
  return s
 initial=guard(True);before_health=health();OUT.mkdir();before_io=io_sample();alignment={};profile=OUT/'profile.speedscope.json'
 command=[str(EXE),'record','--pid',str(TARGET['pid']),'--duration','60','--rate','100','--nonblocking','--idle','--threads','--full-filenames','--format','speedscope','--output',str(profile)]
 try:
  alignment['before_profiler_Popen_monotonic_ns']=time.monotonic_ns()
  with(OUT/'profiler.stdout.log').open('xb')as o,(OUT/'profiler.stderr.log').open('xb')as e:child=subprocess.Popen(command,cwd=F,stdout=o,stderr=e,env=dict(os.environ),creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
  child_id=remember(psutil.Process(child.pid));alignment['after_profiler_Popen_monotonic_ns']=time.monotonic_ns()
  sampler=ThreadSampler(TARGET['pid'],max_threads=128,max_identities=256,max_polls=700,max_records=80000)
  next_poll=time.monotonic()
  while child.poll()is None:
   guard();sampler.poll([t.id for t in psutil.Process(TARGET['pid']).threads()]);next_poll+=.1
   time.sleep(max(0,min(.1,next_poll-time.monotonic())))
  code=child.wait(timeout=10);alignment['profiler_exit_observed_monotonic_ns']=time.monotonic_ns();require(code==0,'Actual owned profiler exit0; preserve failures')
  guard();sampler.poll([t.id for t in psutil.Process(TARGET['pid']).threads()]);alignment['final_counter_poll_monotonic_ns']=time.monotonic_ns()
  after_io=io_sample();after_health=health();require(profile.stat().st_size<=MAX_PROFILE,'Bounded64MiB original profile')
  profiles=profile_profiles(read(profile,MAX_PROFILE));summary=sampler.summary()
  mapping=profile_role_mapping(profiles,summary,role_specs={'main':{'profile_name':'MainThread','file':str((P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v31.py').resolve()),'functions':['main']},'scalar_worker':{'profile_name':'private-broad-row-prefetch','file':str((S/'asr18_BPE9223_private_scalar_prefetch_v1.py').resolve()),'functions':['_run']}})
  # Role mapping uses both the actual profile name and the exact source frame when available.
  for v in mapping['mapped']:
   pp=next(p for p in profiles if p['name']==v['profile_name'])
   v['exact_source_role_frame_present']=any(str(fr.get('file','')).replace('\\','/').lower()==str((S/'asr18_BPE9223_private_scalar_prefetch_v1.py').resolve()).replace('\\','/').lower()and fr.get('name')=='_run'for fr in pp['sampled_frames'])if v['role']=='scalar_worker'else any(str(fr.get('file','')).replace('\\','/').lower()==str((P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v31.py').resolve()).replace('\\','/').lower()and fr.get('name')=='main'for fr in pp['sampled_frames'])
   v['role_label_and_source_frame_agree']=v['exact_source_role_frame_present']and(('MainThread'if v['role']=='main'else'private-broad-row-prefetch')in v['profile_name'])
  sampler.close();counter=OUT/'thread_CPU_counters.jsonl'
  with counter.open('xb')as f:
   total=0
   for row in sampler.records:
    b=(json.dumps(row,separators=(',',':'))+'\n').encode('utf-8');total+=len(b);require(total<=MAX_COUNTER_BYTES,'Counter output32MiB cap');f.write(b)
  stderr=(OUT/'profiler.stderr.log').read_text(encoding='utf-8',errors='replace');require(len(stderr.encode('utf-8'))<=4194304,'Bounded original error log')
  delta={k:after_io[k]-before_io[k]for k in before_io};require(all(v>=0 for v in delta.values()),'Same exact target process I/O counters monotonic')
  value={'measurement_completed':True,'actual_profiler_Popen_exit':code,'exact_command':command,'profile':pin(profile,MAX_PROFILE),'counter_snapshot':pin(counter,MAX_COUNTER_BYTES),'profile_summary':profiles,'thread_CPU_summary':summary,'role_mapping':mapping,'all_requested_roles_name_and_source_agree':all(sum(v['role']==role and v['role_label_and_source_frame_agree']for v in mapping['mapped'])==1 for role in ('main','scalar_worker')),'sampler_open_or_exit_races':sampler.failures,'counter_poll_brackets':sampler.poll_brackets,'counter_sample_stamp_is_poll_start_not_simultaneous_thread_query':True,'alignment':alignment,'target_owned_process_IO_before':before_io,'target_owned_process_IO_after':after_io,'target_owned_process_IO_delta':delta,'health_before':before_health,'health_after':after_health,'profiler_stderr':pin(OUT/'profiler.stderr.log'),'profiler_stdout':pin(OUT/'profiler.stdout.log'),'profiler_stderr_nonempty_preserved':bool(stderr),'inference_limits':['GetThreadTimes user+kernel is actual thread CPU, inclusive of any finalizers/C work executed there','CPU interval first/last sample is distinct from sampled stack wall duration; startup/end alignment uncertainty explicit','Profile sampled wall and thread CPU cannot be added or called exact exclusive CPU-by-stack/phase','Nonblocking stack reads can be nonatomic or fail; stderr and unknown/reused/unmapped nativeTIDs retained','Process I/O counters include all target threads/cache and do not establish physical disk or per-thread/phase I/O','No before/after speed success or GC-policy recommendation follows automatically']}
 except BaseException as e:error={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
 finally:
  if sampler is not None and not sampler.closed:
   try:sampler.close()
   except BaseException as e:cleanup.append({'kind':'query_handle_close_error','type':type(e).__name__,'message':str(e)})
  if child is not None and child.poll()is None:
   try:
    require(child_id is not None and child_id['pid']not in PROTECTED and exact(child_id),'Cleanup exact owned profiler only')
    root=psutil.Process(child_id['pid']);owned=[identity(p)for p in root.children(recursive=True)]
    for v in reversed(owned):
     if not exact(v):continue
     require(v['pid']not in PROTECTED and exact(child_id),'Never target production family')
     proc=psutil.Process(v['pid']);require(any(p.pid==child_id['pid']and abs(p.create_time()-child_id['create_time'])<.001 for p in proc.parents()),'Fresh exact profiler descendant ancestry')
     if exact(v):proc.kill();cleanup.append({'kind':'owned_profiler_descendant_killed','identity':v})
    if exact(child_id):psutil.Process(child_id['pid']).kill();cleanup.append({'kind':'owned_profiler_killed','identity':child_id})
    child.wait(timeout=10)
   except BaseException as e:cleanup.append({'kind':'owned_profiler_cleanup_error','type':type(e).__name__,'message':str(e)})
  value.update({'kind':'ASR18_existing_source31_thread_CPU_and_nonblocking_stacks_actual_v1','passed':value.get('measurement_completed')is True and error is None and not cleanup,'error':error,'required_cleanup':cleanup,'observer_identity':identity(me),'observer_launcher_identity':identity(me.parent()),'owned_profiler_identity':child_id,'recorded_owned_identities':list(observed.values()),'all_owned_query_handles_closed':ThreadQuery.opened-ThreadQuery.closed_count==opened_before,'input_pins_before':inputs,'input_pins_after':[pin(v['path'])for v in inputs],'inputs_before_after_exact':all(pin(v['path'])==v for v in inputs),'ROOT_execution_scope':pin(a.ROOT_review),'completed_new_counter_CPU_review':review['completed_new_counter_CPU_review'],'initial_resource_sample':initial,'last_resource_sample':last,'owned_family_private_peak_bytes':peak,'actual_wall_seconds':time.monotonic()-began,'no_target_signal_suspend_injection_debugger_extra_GC_GPU_model_audio_state_restart':True})
  value['passed']=value['passed']and value['all_owned_query_handles_closed']and value['inputs_before_after_exact']
  write(OUT/'report.json',value)
 print(json.dumps({'report':pin(OUT/'report.json'),'passed':value['passed'],'error':error,'cleanup':cleanup}),flush=True)
 return 0 if value['passed']else 1
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ROOT-review',required=True);sys.exit(main(p.parse_args()))
