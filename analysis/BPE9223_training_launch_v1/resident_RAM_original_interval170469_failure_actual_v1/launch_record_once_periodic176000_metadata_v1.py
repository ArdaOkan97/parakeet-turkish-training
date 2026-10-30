"""One admitted routine metadata command plus actual prelaunch/Popen/identity records.
No production session consumption, signal, evaluator, journal or tensor read.
"""
import hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).parent;N=F.parent;A=N/'resident_RAM_training_actual_25'
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000024'
SCOPE=N/'ROOT_periodic176000_source33_CPU13_metadata_only_execution_scope_v1.json'
def read(p):return json.loads(Path(p).read_bytes())
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();assert len(b)<=2*1024**2
 return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def identity(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n')as out:json.dump(v,out,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
assert pin(SCOPE)['sha256']=='462ae7e6b96f695d559e75342136f7d76d911c259eed49ee8cec7077255afd84'
scope=read(SCOPE);assert scope['accepted']is True and scope['reviewer']=='/root'
command=scope['actual_future_arguments'];assert pin(command[-1])==scope['source']
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and os.environ.get('PYTHONUTF8')=='1'
assert all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
assert Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
progress=read(B/'progress/latest.json');native=progress['producer']
assert native['pid']==3652 and abs(native['create_time']-1792764340.2396317)<.001
assert abs(psutil.Process(native['pid']).create_time()-native['create_time'])<.001
assert progress['source']==pin(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v33.py')
assert progress['actual_optimizer_updates']>176200 and progress['last_saved_updates']>=176200
assert not progress['optimizer_poisoned']and not progress['schedule_poisoned']and not(I/'actual_failure.json').exists()
assert progress['planned_native_updates']==401072 and progress['administrative_update_stop']is None
persistence=read(B/'checkpoints/update_000176200/persistence_verified.json')
assert persistence['passed']is True and persistence['optimizer_updates']==176200
assert (persistence['named_tensors_verified'],persistence['weights_verified'],persistence['Adam_states_verified'])==(854,207,215)
state=read(B/'periodic_DEV.json');entry=[v for v in state['completed_evaluations']if v['optimizer_updates']==176000]
assert len(entry)==1 and Path(entry[0]['DEV']['path']).is_file()and Path(entry[0]['frontier_preservation']['path']).is_file()
q=read(I/'CPU_periodic_000176000_quiescence.json');assert q['CPU_actual0']['completed_updates']==4020
assert q['CPU_actual0']['producer_identity']['pid']==31568
assert all(not Path(p).exists()for p in scope['outputs_once'])
paths={k:A/('actual_periodic176000_'+v)for k,v in {
 'prelaunch':'metadata_prelaunch_v1.json','entry':'metadata_Popen_entry_v1.json',
 'result':'metadata_Popen_result_v1.json','stdout':'metadata_stdout_v1.log','stderr':'metadata_stderr_v1.log'}.items()}
assert all(not p.exists()for p in paths.values())
sample=sample_memory(me.pid,'periodic176000_full225_plus_full_prospective_metadata_admission')
assert sample['paired_sampling_within_one_second']
assert sample['system_memory']['commit_available_bytes']>=5055306368
assert sample['system_memory']['physical_available_bytes']>=2305306368
assert shutil.disk_usage('E:/').free>=100000000000
write(paths['prelaunch'],{'ROOT_scope':pin(SCOPE),'source':pin(command[-1]),'exact_arguments':command,
 'actual_sample':sample,'full_admission_commit_bytes':5055306368,'full_admission_physical_bytes':2305306368,
 'executor_identity':identity(me),'executor_interpreter_launcher':identity(me.parent()),
 'actual_production_progress_at_admission':progress,'production_environment_or_state_changed':False})
observed={};popen=None;code=None;error=None;began=time.monotonic()
def remember(proc):
 who=identity(proc);observed[(who['pid'],who['create_time'])]=who;return who
try:
 with paths['stdout'].open('xb')as out,paths['stderr'].open('xb')as err:
  popen=subprocess.Popen(command,cwd=P,stdout=out,stderr=err,creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
  launcher=remember(psutil.Process(popen.pid))
  write(paths['entry'],{'exact_Popen_launcher_identity':launcher,'actual_arguments':command,
   'executor_identity':identity(me),'executor_launcher_identity':identity(me.parent()),'ROOT_scope':pin(SCOPE)})
  while popen.poll()is None:
   assert time.monotonic()-began<200,'Original sealer internal180s; observation wait exceeded200s, no automatic kill/retry'
   try:
    root=psutil.Process(launcher['pid'])
    if abs(root.create_time()-launcher['create_time'])<.001:
     for child in root.children(recursive=True):remember(child)
   except psutil.NoSuchProcess:pass
   time.sleep(.025)
  code=popen.wait(timeout=10)
except BaseException as exc:error={'error':repr(exc),'traceback':traceback.format_exc()}
finally:
 status=Path(scope['outputs_once'][0]);record=None
 if status.exists():
  record=read(status)
  for who in (record['sealer_identity'],record['sealer_launcher']):observed[(who['pid'],who['create_time'])]=who
 result={'kind':'ASR18_actual_periodic176000_metadata_Popen_result_v1','Popen_wait_code':code,
  'actual_exception':error,'observed_exact_job_identities':list(observed.values()),
  'executor_identity':identity(me),'executor_launcher_identity':identity(me.parent()),
  'ROOT_scope':pin(SCOPE),'prelaunch':pin(paths['prelaunch']),'source':pin(command[-1]),'actual_arguments':command,
  'status':pin(status)if status.exists()else None,'status_passed':None if record is None else record['passed'],
  'stdout':pin(paths['stdout'])if paths['stdout'].exists()else None,'stderr':pin(paths['stderr'])if paths['stderr'].exists()else None,
  'owned_production_session65067_consumed':False,'production_signals_or_changes':False,
  'no_new_fullfamily_CPU_control_proof_claim':True,'wall_seconds':time.monotonic()-began}
 write(paths['result'],result)
print(json.dumps({'Popen_result':pin(paths['result']),'Popen_wait_code':code,'status':result['status'],'error':error}),flush=True)
assert code==0 and error is None and result['status_passed']is True,'Preserve actual metadata failure, no automatic rerun'
