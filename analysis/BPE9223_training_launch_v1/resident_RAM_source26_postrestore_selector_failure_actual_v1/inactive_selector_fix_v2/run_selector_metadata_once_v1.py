"""ONE scoped CPU Popen invocation with external original resource guards."""
import hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).parent;N=F.parents[1];SCOPE=N/'ROOT_saved156600_source27_exact_postrestore_selector_metadata_CPU_scope_v1.json'
def pin(p):
 p=Path(p);raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 with Path(p).open('x',encoding='utf-8')as f:json.dump(v,f,indent=2);f.write('\n')
def identity(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
scope=read(SCOPE);assert pin(SCOPE)['sha256']=='e5b5229e4dd90370e2a0fa961e93b68acf91f15c2ff00d971eb4510243f79f3f'
assert scope['accepted']is True and scope['reviewer']=='/root'
for x in scope['input_pins']:assert pin(x['path'])==x
assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'and all(os.environ[k]=='1'for k in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
assert Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
assert not Path(scope['output']).exists()
me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);parent=me.parent();began=time.monotonic();peak=0;observed={};process=None;error=None;code=None
def guard(initial=False):
 global peak
 sample=sample_memory(me.pid,'postrestore_selector_external_guard');m=sample['system_memory']
 family=[me]
 if process is not None:
  try:
   child=psutil.Process(process.pid);family.extend([child]+child.children(recursive=True))
  except psutil.NoSuchProcess:pass
 total=0
 for item in family:
  try:
   who=identity(item);observed[(who['pid'],who['create_time'])]=who;total+=item.memory_info().private
  except psutil.NoSuchProcess:pass
 peak=max(peak,total)
 assert peak<536870912 and m['commit_available_bytes']>=2000000000+268435456+(536870912 if initial else 0)
 assert m['physical_available_bytes']>=1500000000+268435456+(536870912 if initial else 0)
 assert shutil.disk_usage('E:/').free>=100000000000 and time.monotonic()-began<240
 return sample
initial=guard(True)
external_entry={'controller':identity(me),'controller_interpreter_launcher':identity(parent),'scope':pin(SCOPE),'guard_initial':initial}
write(F/'typed_selector_actual_01_external_entry.json',external_entry)
command=[scope['runtime'],'-X','utf8','-B',scope['typed_fixture_source']['path'],*scope['exact_fixture_arguments']]
stdout=F/'typed_selector_actual_01.stdout.log';stderr=F/'typed_selector_actual_01.stderr.log'
try:
 with stdout.open('xb')as out,stderr.open('xb')as err:
  process=subprocess.Popen(command,cwd=P,env=os.environ.copy(),stdout=out,stderr=err,creationflags=0x08000000|0x00004000)
  popen_identity=identity(psutil.Process(process.pid))
  while process.poll()is None:guard();time.sleep(.025)
  code=process.wait()
except BaseException as exc:
 error={'error':repr(exc),'traceback':traceback.format_exc()}
 if process is not None and process.poll()is None:
  try:
   owned=psutil.Process(process.pid)
   for child in reversed(owned.children(recursive=True)):child.terminate()
   owned.terminate();code=process.wait(timeout=30)
  except BaseException as cleanup:error['cleanup_error']=repr(cleanup)
entry_path=Path(scope['output'])/'actual_process_entry.json';report_path=Path(scope['output'])/'report.json'
entry=read(entry_path)if entry_path.exists()else None
for x in scope['input_pins']:assert pin(x['path'])==x
value={'kind':'ASR18_exact_postrestore_selector_actual_external_Popen_result_v1','passed':code==0 and error is None and report_path.exists(),
 'actual_Popen_wait_code':code,'error':error,'command':command,'scope':pin(SCOPE),'controller':external_entry['controller'],'controller_interpreter_launcher':external_entry['controller_interpreter_launcher'],
 'Popen_identity':locals().get('popen_identity'),'actual_fixture_identities':entry,'observed_owned_family_identities':list(observed.values()),
 'elapsed_seconds':time.monotonic()-began,'maximum_sampled_family_private_bytes':peak,'stdout':pin(stdout),'stderr':pin(stderr),
 'report':pin(report_path)if report_path.exists()else None,'input_pins_before_after_exact':scope['input_pins'],'guard_initial':initial,'guard_final':guard()}
result=F/'typed_selector_actual_01_external_result.json';write(result,value)
print(json.dumps({'external_result':pin(result),'actual_Popen_wait_code':code,'passed':value['passed'],'report':value['report'],'identities':entry,'controller':external_entry['controller'],'controller_launcher':external_entry['controller_interpreter_launcher']}),flush=True)
sys.exit(0 if value['passed']else 1)

