"""INACTIVE fixed60s CPU and nominal20s20Hz role profile. No GPU or state call."""
import argparse,hashlib,json,math,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import psutil
from owned_thread_times_v2 import ThreadQuery,ThreadSampler,profile_role_mapping
from finite_thread_counter_mapping_fixture_v2 import profile_profiles
from fixed_counter_window_and_scalar_files_v1 import FixedCounterWindow,BoundedScalarFiles
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).parent;EXE=F.parent/'py_spy_0_4_2_win_amd64_v1/py-spy.exe'
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000025'
OUT=F/'existing_source36_thread_CPU_actual_02'
TARGET={'pid':21912,'create_time':1792780981.1092248,'ppid':37108}
DRIVER_SHA='d8b9797195bf5dd1981e3a4043f86fba369fd4e7706d9535038e7f7b788e1d5a'
RUNTIME_SHA='f866afb3e9cd7e8db15c341e7df5ecbc8c6cc7d6714153783c389fafc54498a7'
PROTECTED={21912,37108,29600,26892}
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
def health(review):
 require(exact(TARGET),'Same exact native21912 must remain alive')
 boot=read(I/'boot.json');progress=read(B/'progress/latest.json')
 require(boot['producer']==TARGET and boot['source']['sha256']==DRIVER_SHA and boot['configuration']['sha256']==RUNTIME_SHA,'Exact current source36/I25/native/runtime')
 require(pin(boot['source']['path'])==boot['source']and pin(I/'configuration.json')==boot['configuration'],'Same immutable admitted source/config')
 require(progress['producer']==TARGET and progress['source']==boot['source']and progress['planned_native_updates']==401072 and progress['administrative_update_stop']is None and not progress['optimizer_poisoned']and not progress['schedule_poisoned']and not(I/'actual_failure.json').exists(),'Healthy same full401072 frontier; no state changes')
 cpu_pin=review['current_CPU_ready'];require(pin(cpu_pin['path'])==cpu_pin,'Exact current phase CPU-ready metadata binding')
 cpu_ready=read(cpu_pin['path']);cpu=cpu_ready['producer_identity'];cpu_launcher=identity(psutil.Process(cpu['ppid']))
 require(cpu_ready['source']['protocol']==review['CPU_source']and exact(cpu)
  and cpu['pid']in PROTECTED and cpu_launcher['pid']in PROTECTED
  and any(w['pid']==cpu['pid']and abs(w['create_time']-cpu['create_time'])<.001 for w in review['protected_production_identities'])
  and any(w['pid']==cpu_launcher['pid']and abs(w['create_time']-cpu_launcher['create_time'])<.001 for w in review['protected_production_identities']),
  'Actual current CPU13 cohort and launcher, exact ROOT-bound post184k identities')
 reader=review['reader_identity'];require(reader['pid']==26892 and abs(reader['create_time']-1792652904.3359847)<.001 and exact(reader),'Same exact reader identity; raw compact records captured without guessed schema')
 reader_paths=review['reader_compact_paths'];reader_root=Path('E:/gpu_training_scratchpad/analysis/BPE9223_Trackio_live_v1')
 require(reader_paths=={name:str((reader_root/file).resolve())for name,file in [('current_status','current_status.json'),('mirror_state','mirror_state.json'),('owner','owner.json')]},
  'Only three exact authoritative existing reader compact paths; no restart or mutation')
 reader_raw={name:read(path,1048576)for name,path in reader_paths.items()}
 return {'progress':progress,'boot':pin(I/'boot.json'),'runtime':pin(I/'configuration.json'),
  'current_CPU_ready':cpu_pin,'current_CPU_identity':cpu,'current_CPU_launcher_identity':cpu_launcher,
  'reader_identity':reader,'reader_compact_paths':reader_paths,'reader_compact_raw_without_schema_inference':reader_raw}
def io_sample():return dict(psutil.Process(TARGET['pid']).io_counters()._asdict())

def capture_current_timing_tail(output):
 path=I/'private_metadata_window_timings.jsonl';size=path.stat().st_size;start=max(0,size-65536)
 with path.open('rb')as f:f.seek(start);raw=f.read(size-start)
 if start:
  first=raw.find(b'\n');require(first>=0,'Bounded complete timing-tail row');start+=first+1;raw=raw[first+1:]
 if not raw.endswith(b'\n'):
  last=raw.rfind(b'\n');require(last>=0,'Preserve only complete existing timing rows');raw=raw[:last+1]
 lines=raw.splitlines(keepends=True);require(len(lines)>=5,'Five complete late existing timing rows')
 selected=b''.join(lines[-5:]);selected_start=start+len(raw)-len(selected)
 rows=[json.loads(line)for line in lines[-5:]]
 snapshot=output/'trigger_existing_last5_timing_rows.jsonl'
 with snapshot.open('xb')as f:f.write(selected)
 return {'source_path':str(path),'source_size_observed':size,'start_offset':selected_start,
  'end_offset':selected_start+len(selected),'bytes':len(selected),'sha256':hashlib.sha256(selected).hexdigest(),
  'snapshot':pin(snapshot),'rows':rows,'selection':'last5 complete existing rows from <=64KiB after actual185020',
  'no_new_timing_span_benchmark_or_cause_or_speed_claim':True}

def main(a):
 me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);began=time.monotonic();exact_external_interpreter_launcher=identity(me.parent())
 require(os.environ['CUDA_VISIBLE_DEVICES']=='-1'and os.environ['PYTHONUTF8']=='1'and all(os.environ[k]=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),'Separate E CPU-only observer process environment')
 review=read(a.ROOT_review)
 require(review['accepted']is True and review['reviewer']=='/root'and review['scope']=='ASR18_existing_source36_GetThreadTimes_short20Hz20s_fixed60_attribution_v1','Separate exact LIVE execution scope required')
 require(review['live_observer_source']==pin(__file__)and review['target_identity']==TARGET and review['output_path']==str(OUT.resolve()),'Exact source/target/output scope')
 proof=read(review['completed_new_counter_CPU_review']['path']);require(pin(review['completed_new_counter_CPU_review']['path'])==review['completed_new_counter_CPU_review']and proof['accepted']is True and proof['reviewer']=='/root'and proof['scope']=='ASR18_GetThreadTimes_new_toy_nativeTID_mapping_actual_CPU_v1'and proof['thread_counter_helper_source']==pin(F/'owned_thread_times_v2.py')and proof['all_exact_metadata_identities_dead']is True,'Genuine NEW counter/role/handle CPU proof, not old py-spy toy substitution')
 controller_pin=review['completed_fixed_counter_controller_CPU_review'];controller_proof=read(controller_pin['path'])
 require(pin(controller_pin['path'])==controller_pin and controller_proof['accepted']is True
  and controller_proof['reviewer']=='/root'
  and controller_proof['scope']=='ASR18_fixed60_counter_window_persistence_METHOD_STUB_actual_CPU_v1'
  and controller_proof['controller_source']==pin(F/'fixed_counter_window_and_scalar_files_v1.py')
  and pin(controller_proof['actual_CPU_report']['path'])==controller_proof['actual_CPU_report']
  and controller_proof['all_exact_metadata_identities_dead']is True,
  'Genuine completed fixed60/persistence METHOD_STUB ROOT gate; no live or timing success inferred')
 require(review['CPU_source']==pin(S/'fastmss_resident_feature_stream_v13.py')
  and review['runtime_configuration']==pin(I/'configuration.json'), 'Exact current CPU13 and immutable source36 runtime')
 require(all(type(w['pid'])is int for w in review['protected_production_identities'])
  and {TARGET['pid'],TARGET['ppid'],29600,26892}.issubset({w['pid']for w in review['protected_production_identities']}),
  'ROOT-bound current production family and reader are never owned cleanup targets')
 PROTECTED.update(w['pid']for w in review['protected_production_identities'])
 inputs=review['input_pins'];require(all(pin(v['path'])==v for v in inputs),'All exact inputs before LIVE observation')
 require(not OUT.exists(),'One fresh bounded source36 observation; no automatic rerun')
 child=None;child_id=None;sampler=None;peak=0;last=None;observed={};cleanup=[];error=None;value={};opened_before=ThreadQuery.opened-ThreadQuery.closed_count;scalar_files=None;controller=None;lifetime_queries={};lifetime_acquire_errors=[];lifetime_at_profile_end=[];lifetime_captured=False;peak_query_handles=0
 def remember(p):
  v=identity(p);observed[(v['pid'],v['create_time'])]=v;return v
 remember(me)
 def guard(admission=False):
  nonlocal peak,last,peak_query_handles
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
    s=sample_memory(proc.pid,'source36_thread_CPU_readonly_attribution');require(s['paired_sampling_within_one_second'],'Original paired memory sample');total+=s['worker_private_memory']['private_bytes'];last=s
   except psutil.NoSuchProcess:continue
  peak=max(peak,total);require(total<=LIMIT,'Total owned observer/profiler family512MiB')
  peak_query_handles=max(peak_query_handles,ThreadQuery.opened-ThreadQuery.closed_count)
  require(0<=ThreadQuery.opened-ThreadQuery.closed_count<=256,'Original total256 owned query-handle bound')
  m=last['system_memory'];extra=LIMIT if admission else 0
  require(m['commit_available_bytes']>=(4518435456+extra if admission else 2268435456)and m['physical_available_bytes']>=1768435456+extra and shutil.disk_usage('E:/').free>=100000000000,'FULL225 cushion+FULLprefetch+prospective512 admission; original runtimefloors+FULLprefetch; E100GB')
  return s
 initial=guard(True);before_health=health(review)
 require(before_health['progress']['actual_optimizer_updates']>=185020 and before_health['progress']['last_saved_updates']>=185000,
  'Sample only after fixed actual185020 and natural185000 on source36 post184k late phase; no startup/postDEV-warm attribution')
 OUT.mkdir();trigger_tail=capture_current_timing_tail(OUT);before_io=io_sample();alignment={};profile=OUT/'profile.speedscope.json'
 command=[str(EXE),'record','--pid',str(TARGET['pid']),'--duration','20','--rate','20','--nonblocking','--idle','--threads','--full-filenames','--format','speedscope','--output',str(profile)]
 try:
  alignment['before_profiler_Popen_monotonic_ns']=time.monotonic_ns()
  with(OUT/'profiler.stdout.log').open('xb')as o,(OUT/'profiler.stderr.log').open('xb')as e:child=subprocess.Popen(command,cwd=F,stdout=o,stderr=e,env=dict(os.environ),creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
  child_id=remember(psutil.Process(child.pid));alignment['after_profiler_Popen_monotonic_ns']=time.monotonic_ns()
  sampler=ThreadSampler(TARGET['pid'],max_threads=128,max_identities=256,max_polls=700,max_records=80000)
  scalar_files=BoundedScalarFiles(OUT);controller=FixedCounterWindow(time.monotonic())
  alignment['counter_explicit_start_monotonic_ns']=time.monotonic_ns();alignment['counter_intended_deadline_seconds']=controller.deadline
  while True:
   guard();now=time.monotonic();phase=controller.phase(now,child.poll())
   if phase!='counter_window'and not lifetime_captured:
    lifetime_queries={(tid,q.creation):q for tid,q in sampler.active.items()}
    lifetime_captured=True
    alignment['existing_sampler_handles_retained_at_counter_end_monotonic_ns']=time.monotonic_ns()
    require(len(lifetime_queries)<=256 and ThreadQuery.opened-ThreadQuery.closed_count<=256,
     'Only original sampler handles retained through actual profiler end; no duplicate OpenThread')
   if phase=='done':break
   if phase=='counter_window'and controller.due(now):
    tids=[t.id for t in psutil.Process(TARGET['pid']).threads()]
    vector={'monotonic_ns':time.monotonic_ns(),'attempted_poll':sampler.polls+1,'poll_limit':700,'current_TID_count':len(tids),'unique_TID_count':len(set(tids)),'max_current_threads':128,'total_identity_count':len(sampler.identities),'max_identities':256,'prior_counter_records':len(sampler.records),'counter_record_limit':80000,'live_query_handles':ThreadQuery.opened-ThreadQuery.closed_count,'max_live_query_handles':256,'conservative_new_TIDs':len(set(tids)-set(sampler.active))}
    scalar_files.append('preconditions',[vector]) # Flush BEFORE every original helper assertion.
    require(ThreadQuery.opened-ThreadQuery.closed_count+len(set(tids)-set(sampler.active))<=256,
     'Conservative before-poll original256 total handle budget, vector already durable')
    before_count=len(sampler.records);query_start=time.monotonic()
    if query_start<controller.deadline:
     sampler.poll(tids);controller.recorded(query_start)
     scalar_files.append('counters',sampler.records[before_count:]);scalar_files.append('brackets',[sampler.poll_brackets[-1]])
     require(ThreadQuery.opened-ThreadQuery.closed_count<=256,'Original total handle bound after original sampler poll')
   if phase=='counter_window':
    time.sleep(controller.sleep_seconds(time.monotonic()))
   else:time.sleep(.05) # Only guarded profiler completion wait; NO new counter acquisition.
  code=child.wait(timeout=10);alignment['profiler_exit_observed_monotonic_ns']=time.monotonic_ns();require(code==0,'Actual owned profiler exit0; preserve failures')
  for key,q in lifetime_queries.items():lifetime_at_profile_end.append({'native_tid':key[0],'creation_filetime_100ns':key[1],'actual_same_retained_handle_at_profile_end':q.sample()})
  alignment['counter_queries_stop_at_fixed_window_not_profiler_exit']=True
  after_io=io_sample();after_health=health(review);require(profile.stat().st_size<=MAX_PROFILE,'Bounded64MiB original profile')
  profiles=profile_profiles(read(profile,MAX_PROFILE));summary=sampler.summary()
  mapping=profile_role_mapping(profiles,summary,role_specs={'main':{'profile_name':'MainThread','file':str((P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v36.py').resolve()),'functions':['main']},'scalar_worker':{'profile_name':'private-broad-row-prefetch','file':str((S/'asr18_BPE9223_private_scalar_prefetch_v1.py').resolve()),'functions':['_run']}})
  invalid_lifetimes={(v['native_tid'],v['creation_filetime_100ns'])for v in lifetime_acquire_errors}
  invalid_lifetimes.update((v['native_tid'],v['creation_filetime_100ns'])for v in lifetime_at_profile_end if v['actual_same_retained_handle_at_profile_end']['exit_filetime_100ns'])
  invalid_lifetimes.update((v['native_tid'],v['creation_filetime_100ns'])for v in summary if v['thread_exited_observed'])
  kept=[]
  for v in mapping['mapped']:
   if(v['native_tid'],v['creation_filetime_100ns'])in invalid_lifetimes:mapping['unknown'].append({'name':v['profile_name'],'reason':'Retained lifetime exited/unproven during expanded profiler interval; no role/CPU attribution'})
   else:kept.append(v)
  mapping['mapped']=kept
  # Role mapping uses both the actual profile name and the exact source frame when available.
  for v in mapping['mapped']:
   pp=next(p for p in profiles if p['name']==v['profile_name'])
   v['exact_source_role_frame_present']=any(str(fr.get('file','')).replace('\\','/').lower()==str((S/'asr18_BPE9223_private_scalar_prefetch_v1.py').resolve()).replace('\\','/').lower()and fr.get('name')=='_run'for fr in pp['sampled_frames'])if v['role']=='scalar_worker'else any(str(fr.get('file','')).replace('\\','/').lower()==str((P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v36.py').resolve()).replace('\\','/').lower()and fr.get('name')=='main'for fr in pp['sampled_frames'])
   v['role_label_and_source_frame_agree']=v['exact_source_role_frame_present']and(('MainThread'if v['role']=='main'else'private-broad-row-prefetch')in v['profile_name'])
  sampler.close()
  scalar_files.close();counter=OUT/'thread_CPU_counters.jsonl'
  stderr=(OUT/'profiler.stderr.log').read_text(encoding='utf-8',errors='replace');require(len(stderr.encode('utf-8'))<=4194304,'Bounded original error log')
  delta={k:after_io[k]-before_io[k]for k in before_io};require(all(v>=0 for v in delta.values()),'Same exact target process I/O counters monotonic')
  value={'measurement_completed':True,'actual_profiler_Popen_exit':code,'exact_command':command,'profile':pin(profile,MAX_PROFILE),'counter_snapshot':pin(counter,MAX_COUNTER_BYTES),'profile_summary':profiles,'thread_CPU_summary':summary,'role_mapping':mapping,'all_requested_roles_name_and_source_agree':all(sum(v['role']==role and v['role_label_and_source_frame_agree']for v in mapping['mapped'])==1 for role in ('main','scalar_worker')),'sampler_open_or_exit_races':sampler.failures,'counter_poll_brackets':sampler.poll_brackets,'counter_sample_stamp_is_poll_start_not_simultaneous_thread_query':True,'alignment':alignment,'target_owned_process_IO_before':before_io,'target_owned_process_IO_after':after_io,'target_owned_process_IO_delta':delta,'health_before':before_health,'health_after':after_health,'profiler_stderr':pin(OUT/'profiler.stderr.log'),'profiler_stdout':pin(OUT/'profiler.stdout.log'),'profiler_stderr_nonempty_preserved':bool(stderr),'inference_limits':['GetThreadTimes user+kernel is actual thread CPU, inclusive of any finalizers/C work executed there','CPU interval first/last sample is distinct from sampled stack wall duration; startup/end alignment uncertainty explicit','Profile sampled wall and thread CPU cannot be added or called exact exclusive CPU-by-stack/phase','Nonblocking stack reads can be nonatomic or fail; stderr and unknown/reused/unmapped nativeTIDs retained','Process I/O counters include all target threads/cache and do not establish physical disk or per-thread/phase I/O','No before/after speed success or GC-policy recommendation follows automatically']}
 except BaseException as e:error={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
 finally:
  if scalar_files is not None and not scalar_files.closed:
   try:scalar_files.close()
   except BaseException as e:cleanup.append({'kind':'scalar_evidence_file_close_error','type':type(e).__name__,'message':str(e)})
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
  partial_artifacts={}
  for key,(name,cap)in BoundedScalarFiles.NAMES.items():
   p=OUT/name;partial_artifacts[key]=pin(p,cap)if p.exists()else{'absent':True,'path':str(p.resolve())}
  value.update({'partial_scalar_artifacts':partial_artifacts,'counter_partial_summary':sampler.summary()if sampler is not None else[],'counter_partial_poll_brackets':sampler.poll_brackets if sampler is not None else[],'counter_controller_state':None if controller is None else{'explicit_start':controller.start,'intended_deadline':controller.deadline,'recorded_polls':controller.polls,'last_clock_observation':controller.last},'retained_lifetime_acquire_errors':lifetime_acquire_errors,'retained_lifetime_checks_at_profiler_end':lifetime_at_profile_end,'trigger_existing_timing_tail':trigger_tail,'retained_lifetime_handle_origin':'EXISTING_sampler.active_references_NO_duplicate_OpenThread','peak_total_owned_query_handles':peak_query_handles,'completed_fixed_counter_controller_CPU_review':controller_pin,'CPU_counter_and_profiler_windows_are_different':True,'no_exclusive_CPU_by_profile_stack_claim':True,'kind':'ASR18_existing_source36_thread_CPU_and_short20Hz20s_fixed60_stacks_actual_v2','passed':value.get('measurement_completed')is True and error is None and not cleanup,'error':error,'required_cleanup':cleanup,'observer_identity':identity(me),'observer_launcher_identity':identity(me.parent()),'owned_profiler_identity':child_id,'recorded_owned_identities':list(observed.values()),'all_owned_query_handles_closed':ThreadQuery.opened-ThreadQuery.closed_count==opened_before,'input_pins_before':inputs,'input_pins_after':[pin(v['path'])for v in inputs],'inputs_before_after_exact':all(pin(v['path'])==v for v in inputs),'ROOT_execution_scope':pin(a.ROOT_review),'completed_new_counter_CPU_review':review['completed_new_counter_CPU_review'],'initial_resource_sample':initial,'last_resource_sample':last,'owned_family_private_peak_bytes':peak,'actual_wall_seconds':time.monotonic()-began,'no_target_signal_suspend_injection_debugger_extra_GC_GPU_model_audio_state_restart':True})
  value['passed']=value['passed']and value['all_owned_query_handles_closed']and value['inputs_before_after_exact']
  write(OUT/'report.json',value)
 print(json.dumps({'report':pin(OUT/'report.json'),'passed':value['passed'],'error':error,'cleanup':cleanup}),flush=True)
 return 0 if value['passed']else 1
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ROOT-review',required=True);sys.exit(main(p.parse_args()))
