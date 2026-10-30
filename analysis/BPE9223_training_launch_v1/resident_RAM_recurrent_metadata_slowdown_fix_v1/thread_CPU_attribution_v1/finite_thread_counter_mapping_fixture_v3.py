"""NEW finite GetThreadTimes/nativeTID proof; synthetic CPU threads, no training attach."""
import argparse,ctypes,hashlib,json,math,os,shutil,subprocess,sys,threading,time,traceback
from pathlib import Path
import psutil
from owned_thread_times_v2 import ThreadQuery,ThreadSampler,raw_times,profile_role_mapping
F=Path(__file__).parent;P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
PYTHON=P/'work/nemo-native-env-v2/Scripts/python.exe'
EXE=F.parent/'py_spy_0_4_2_win_amd64_v1/py-spy.exe';TOY=F/'toy_native_threads_v2.py'
OUT=F/'finite_thread_counter_mapping_actual_02'
PROTECTED={2456,38964,26892,41672,34044}
def require(v,m):
 if not v:raise ValueError(m)
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,indent=2);f.write('\n')
def identity(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def exact(v):
 try:return abs(psutil.Process(v['pid']).create_time()-v['create_time'])<.001
 except psutil.NoSuchProcess:return False
def negative(cases,name,fn,kind=ValueError):
 try:fn()
 except kind as e:cases.append({'case':name,'passed':True,'actual_exception':type(e).__name__,'message':str(e)});return
 raise AssertionError('Required negative not rejected: '+name)
def owned_stop(popen,root,records):
 if popen is None or popen.poll()is not None:return
 require(root['pid']not in PROTECTED and exact(root),'Cleanup only exact newly owned Popen root')
 descendants=[]
 for p in psutil.Process(root['pid']).children(recursive=True):
  v=identity(p);require(v['pid']not in PROTECTED,'Never protected production identity');descendants.append(v)
 for v in reversed(descendants):
  if not exact(v):continue
  p=psutil.Process(v['pid']);ancestry=[identity(x)for x in p.parents()]
  require(any(x['pid']==root['pid']and abs(x['create_time']-root['create_time'])<.001 for x in ancestry),'Exact owned root ancestry immediately before kill')
  if exact(v):p.kill();records.append(v)
 if exact(root):psutil.Process(root['pid']).kill();records.append(root)
 popen.wait(timeout=10)
def profile_profiles(value):
 require(value.get('profiles')and type(value.get('shared',{}).get('frames'))is list,'Actual speedscope sampled profile')
 frames=value['shared']['frames'];out=[]
 for p in value['profiles']:
  require(p['type']=='sampled'and p['unit']=='seconds'and len(p['samples'])==len(p['weights']),'Actual sampled seconds schema')
  indices=set()
  for stack,w in zip(p['samples'],p['weights']):
   require(type(w)in(int,float)and math.isfinite(w)and w>=0,'Finite actual sample weight')
   require(type(stack)is list and all(type(i)is int and 0<=i<len(frames)for i in stack),'Bounded stack indices');indices.update(stack)
  out.append({'name':p['name'],'samples':len(p['samples']),'wall_sample_weight_seconds':math.fsum(p['weights']),'sampled_frames':[frames[i]for i in sorted(indices)]})
 return out
def main(a):
 require(os.environ['CUDA_VISIBLE_DEVICES']=='-1'and os.environ['PYTHONUTF8']=='1'and all(os.environ[k]=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'Isolated CPU-only proof environment')
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 review=read(a.ROOT_review)
 require(review['accepted']is True and review['reviewer']=='/root'and review['scope']=='ASR18_GetThreadTimes_new_toy_nativeTID_mapping_finite_CPU_v1','Exact separate finite proof scope')
 inputs=review['input_pins'];require(all(pin(v['path'])==v for v in inputs),'All exact frozen input pins before invocation')
 require(review['fixture_source']==pin(__file__)and review['output_path']==str(OUT.resolve()),'Exact fixture/fresh output binding')
 require(not OUT.exists(),'ONE fresh actual finite proof');OUT.mkdir()
 cases=[];cleanup=[];toy=spy=None;toy_id=spy_id=None;queries=[];began=time.monotonic();sampler=None;error=None;report={}
 start_handles=ThreadQuery.opened-ThreadQuery.closed_count;exit_observations=[]
 try:
  own=ThreadQuery(os.getpid(),threading.get_native_id());queries.append(own);before=own.sample();x=.1
  for i in range(100000):x=math.sin(x)+.1
  after=own.sample();require(after['user_100ns']+after['kernel_100ns']>=before['user_100ns']+before['kernel_100ns'],'Actual current-thread monotonic CPU')
  cases.append({'case':'real_current_nativeTID_creation_and_monotonic_counters','passed':True,'before':before,'after':after,'threading_ident':threading.get_ident()})
  negative(cases,'wrong_creation_rejected',lambda:ThreadQuery(os.getpid(),threading.get_native_id(),own.creation+1))
  negative(cases,'wrong_process_rejected',lambda:ThreadQuery(os.getpid()+1,threading.get_native_id()))
  negative(cases,'invalid_zero_handle_actual_Win32_error',lambda:raw_times(None),OSError)
  own.close();negative(cases,'closed_handle_rejected_before_query',own.sample);negative(cases,'double_close_rejected',own.close)
  baseline=ThreadQuery.opened-ThreadQuery.closed_count;constructor_owned=[];constructor_failed=False
  try:
   for expected in (own.creation,own.creation+1):
    q=ThreadQuery(os.getpid(),threading.get_native_id(),expected);constructor_owned.append(q)
  except ValueError:constructor_failed=True
  finally:
   for q in constructor_owned:
    if not q.closed:q.close()
  require(constructor_failed and ThreadQuery.opened-ThreadQuery.closed_count==baseline,'Real second-constructor failure leaves no first-query or failed-constructor handle')
  cases.append({'case':'real_second_constructor_failure_incremental_ownership_cleanup','passed':True})
  ready=threading.Event();release=threading.Event();thread_data={}
  def finite_owned_thread():
   thread_data['tid']=threading.get_native_id();ready.set();release.wait(5)
  t=threading.Thread(target=finite_owned_thread,name='CounterExitFixture');t.start();require(ready.wait(5),'Real owned thread ready')
  q=ThreadQuery(os.getpid(),thread_data['tid']);queries.append(q)
  exit_path=OUT/'retained_thread_exit_observations.jsonl'
  require(not exit_path.exists(),'Fresh bounded exit observation metadata')
  def capture_exit(label):
   observed=q.sample();record={'label':label,'monotonic_ns':time.monotonic_ns(),'actual_same_handle_counter_values':observed}
   with exit_path.open('ab')as f:f.write((json.dumps(record,separators=(',',':'))+'\n').encode('utf-8'));f.flush()
   exit_observations.append(record);return observed
  alive=capture_exit('before_thread_release');release.set();t.join(5);require(not t.is_alive(),'Real Python thread joined without signal')
  post_join_began=time.monotonic();post_join_polls=0
  while True:
   require(post_join_polls<202,'Bounded same-handle kernel-exit observations')
   exited=capture_exit('post_Python_join_actual_kernel_exit_observation');post_join_polls+=1
   if exited['exit_filetime_100ns']>0:break
   require(time.monotonic()-post_join_began<2,'Actual kernel thread exit must be observed within2s; no inferred exit')
   time.sleep(.01)
  again=capture_exit('stable_requery_after_actual_nonzero_exit_FILETIME')
  require(exited['exit_filetime_100ns']>0 and again==exited and exited['creation_filetime_100ns']==alive['creation_filetime_100ns'],'Same retained handle actual exit identity and stable raw counters');q.close()
  cases.append({'case':'real_retained_handle_after_thread_exit_and_join','passed':True,'before':alive,'after':exited})
  toy_out=OUT/'toy';toy_out.mkdir()
  with(OUT/'toy.stdout.log').open('xb')as o,(OUT/'toy.stderr.log').open('xb')as e:
   toy=subprocess.Popen([str(PYTHON),'-Xutf8','-B',str(TOY),'--output',str(toy_out)],cwd=F,stdout=o,stderr=e,env=dict(os.environ),creationflags=subprocess.CREATE_NO_WINDOW)
  toy_id=identity(psutil.Process(toy.pid));require(toy_id['pid']not in PROTECTED,'Owned new toy launcher, never production')
  while not(toy_out/'toy_ready.json').exists():
   require(toy.poll()is None and time.monotonic()-began<30,'Bounded new toy readiness');time.sleep(.05)
  actual=read(toy_out/'toy_ready.json');native=actual['producer'];require(native['pid']not in PROTECTED and exact(native)and exact(toy_id),'Exact toy actual and Popen launcher alive')
  require(native['pid']==toy.pid or any(p.pid==toy.pid and abs(p.create_time()-toy_id['create_time'])<.001 for p in psutil.Process(native['pid']).parents()),'Actual ready native is exact launcher or trusted descendant')
  require(actual['source']['sha256']==pin(TOY)['sha256'],'Same frozen toy source')
  retained=[]
  for k in ('main','scalar_worker'):
   q=ThreadQuery(native['pid'],actual[k]['native_tid'],actual[k]['creation_filetime_100ns']);queries.append(q);retained.append(q)
  profile=OUT/'new_toy_profile.speedscope.json';command=[str(EXE),'record','--pid',str(native['pid']),'--duration','8','--rate','100','--nonblocking','--idle','--threads','--full-filenames','--format','speedscope','--output',str(profile)]
  alignment={'before_profiler_Popen_monotonic_ns':time.monotonic_ns()}
  with(OUT/'profiler.stdout.log').open('xb')as o,(OUT/'profiler.stderr.log').open('xb')as e:spy=subprocess.Popen(command,cwd=F,stdout=o,stderr=e,env=dict(os.environ),creationflags=subprocess.CREATE_NO_WINDOW)
  spy_id=identity(psutil.Process(spy.pid));alignment['after_profiler_Popen_monotonic_ns']=time.monotonic_ns()
  sampler=ThreadSampler(native['pid'],max_polls=200,max_records=20000)
  while spy.poll()is None:
   require(exact(native)and exact(toy_id)and time.monotonic()-began<60,'Exact NEW toy alive/bounded proof')
   sampler.poll([v.id for v in psutil.Process(native['pid']).threads()]);time.sleep(.1)
  spy_code=spy.wait(timeout=5);alignment['observed_profiler_exit_monotonic_ns']=time.monotonic_ns();require(spy_code==0,'Actual new short paired profiler exit0')
  require(profile.stat().st_size<=67108864,'Bounded profile bytes');summary=sampler.summary();profiles=profile_profiles(read(profile))
  roles={'main':{'profile_name':'MainThread','file':str(TOY.resolve()),'functions':['main']},'scalar_worker':{'profile_name':'ToyScalarWorker','file':str(TOY.resolve()),'functions':['toy_scalar_worker']}}
  mapping=profile_role_mapping(profiles,summary,role_specs=roles)
  for role,key in [('main','main'),('scalar_worker','scalar_worker')]:
   found=[v for v in mapping['mapped']if v['role']==role];require(len(found)==1 and found[0]['native_tid']==actual[key]['native_tid']and found[0]['creation_filetime_100ns']==actual[key]['creation_filetime_100ns'],'Actual profile nativeTID+creation maps exact ready role, never PID assumption')
   require(found[0]['user_seconds_delta']+found[0]['kernel_seconds_delta']>0,'Actual role has measured positive thread CPU delta')
  cases.append({'case':'NEW_actual_short_profile_role_mapping_with_exact_nativeTID_creation','passed':True,'mapping':mapping,'ready':actual,'alignment':alignment})
  negative(cases,'duplicate_current_TID_list',lambda:sampler.poll([actual['main']['native_tid']]*2))
  negative(cases,'bounded_poll_count',lambda:ThreadSampler(native['pid'],max_polls=701))
  stale=[dict(summary[0]),dict(summary[0],creation_filetime_100ns=summary[0]['creation_filetime_100ns']+1)]
  unknown=profile_role_mapping([{'name':'Thread '+str(summary[0]['native_tid'])+' \"MainThread\"','sampled_frames':[]}],stale,role_specs={'main':{'profile_name':'MainThread'}})
  require(not unknown['mapped']and unknown['unknown'],'METHOD_STUB reused/ambiguous creation stays unknown');cases.append({'case':'METHOD_STUB_TID_reuse_ambiguity_not_fabricated','passed':True,'result':unknown})
  unknown=profile_role_mapping([{'name':'unsupported-label','sampled_frames':[]}],summary,role_specs=roles);require(unknown['unknown']and not unknown['mapped'],'Unknown profile label retained');cases.append({'case':'unknown_label_explicit','passed':True})
  for label in ('Thread 0x1234 \"MainThread\"','Thread 1234 (MainThread)'):
   unknown=profile_role_mapping([{'name':label,'sampled_frames':[]}],summary,role_specs=roles)
   require(unknown['unknown']and not unknown['mapped'],'Hex PythonID or old parentheses not normalized to nativeTID')
  cases.append({'case':'hex_PythonID_and_parenthesized_labels_unknown','passed':True})
  main_profile_name=next(v['profile_name']for v in mapping['mapped']if v['role']=='main')
  main_profile=next(v for v in profiles if v['name']==main_profile_name)
  negative(cases,'ambiguous_role_rejected',lambda:profile_role_mapping([main_profile,main_profile],summary,role_specs=roles))
  toy_code=toy.wait(timeout=15);require(toy_code==0 and read(toy_out/'toy_completion.json')['passed']is True,'Actual new toy Popen0/completion/join')
  after_exit=[q.sample()for q in retained];require(all(v['exit_filetime_100ns']>0 for v in after_exit),'Both retained toy thread handles after process exit')
  cases.append({'case':'actual_toy_process_exit_retained_threads_and_join','passed':True,'threads':after_exit,'Popen_exit':toy_code})
  sampler.close()
  for q in queries:
   if not q.closed:q.close()
  require(ThreadQuery.opened-ThreadQuery.closed_count==start_handles,'All owned query handles closed')
  report={'passed':True,'cases':cases,'counter_samples':sampler.records,'thread_summary':summary,'sampler_open_races':sampler.failures,'counter_poll_brackets':sampler.poll_brackets,'counter_sample_stamp_is_poll_start_not_simultaneous_thread_query':True,'profile':pin(profile),'profile_summary':profiles,'role_mapping':mapping,'alignment':alignment,'actual_profiler_Popen_exit':spy_code,'actual_toy_Popen_exit':toy_code,'toy_native_identity':native,'toy_launcher_identity':toy_id,'profiler_identity':spy_id,'owned_thread_handles_opened':ThreadQuery.opened,'owned_thread_handles_closed':ThreadQuery.closed_count,'all_owned_thread_query_handles_closed':True,'unreachable_handle_reuse_test_is_METHOD_STUB_not_actual_OS_reuse':True,'CPU_and_profile_weights_not_exact_exclusive_phase_attribution':True,'new_toy_only_no_training_attach':True}
 except BaseException as e:error={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
 finally:
  for q in queries:
   if not q.closed:
    try:q.close()
    except BaseException as e:cleanup.append({'type':type(e).__name__,'message':str(e),'kind':'query_close_error'})
  if sampler is not None and not sampler.closed:
   try:sampler.close()
   except BaseException as e:cleanup.append({'type':type(e).__name__,'message':str(e),'kind':'sampler_close_error'})
  for pop,v in [(spy,spy_id),(toy,toy_id)]:
   if pop is not None and pop.poll()is None:
    try:owned_stop(pop,v,cleanup)
    except BaseException as e:cleanup.append({'type':type(e).__name__,'message':str(e),'kind':'owned_cleanup_error'})
  report.update({'cases':cases,'retained_thread_exit_observations':exit_observations,'exit_observation_snapshot':pin(OUT/'retained_thread_exit_observations.jsonl')if(OUT/'retained_thread_exit_observations.jsonl').exists()else None,'kind':'ASR18_GetThreadTimes_new_toy_finite_CPU_actual_v1','passed':bool(report.get('passed'))and error is None and not cleanup,'error':error,'required_cleanup':cleanup,'fixture_identity':identity(psutil.Process()),'fixture_launcher_identity':identity(psutil.Process().parent()),'input_pins':inputs,'input_pins_before_after_exact':all(pin(v['path'])==v for v in inputs),'ROOT_review':pin(a.ROOT_review),'elapsed_seconds':time.monotonic()-began,'all_owned_thread_query_handles_closed':ThreadQuery.opened==ThreadQuery.closed_count,'no_extra_collect_model_audio_GPU_or_training_operation':True})
  write(OUT/'report.json',report)
 print(json.dumps({'report':pin(OUT/'report.json'),'passed':report['passed'],'error':error,'required_cleanup':cleanup}),flush=True)
 return 0 if report['passed']else 1
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ROOT-review',required=True);sys.exit(main(p.parse_args()))
