"""External metadata/finite guard for ONE separately ROOT-admitted CPU fixture."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).parent;began=time.monotonic();known={};peak=0
def require(v,m):
 if not v:raise ValueError(m)
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with Path(p).open('x',encoding='utf-8')as f:json.dump(v,f,indent=2);f.write('\n')
def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def same(old):
 try:return abs(psutil.Process(old['pid']).create_time()-old['create_time'])<.001
 except psutil.NoSuchProcess:return False
def observe():
 global peak
 value=sample_memory(os.getpid(),'external_finite_packet_credit_CPU_control');m=value['system_memory']
 family=value['worker_private_memory']['private_bytes']
 for old in list(known.values()):
  if same(old):
   info=psutil.Process(old['pid']).memory_info();family+=getattr(info,'private',info.rss)
 peak=max(peak,family)
 require(value['paired_sampling_within_one_second'] and family<512*1024**2
  and m['commit_available_bytes']>=2000000000+268435456 and m['physical_available_bytes']>=1500000000+268435456
  and shutil.disk_usage('E:/').free>=100000000000 and time.monotonic()-began<240,
  'Original external240s/512MiB-family/full256MiB/commit+physical/E100GB control guards')
 return value
def main(a):
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 review=json.loads(Path(a.ROOT_review).read_text());inputs=review['input_pins']
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and os.environ.get('PYTHONUTF8')=='1' and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'Original CPU-only process-local masks')
 require(Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),'Applied E runtime including extensions')
 require(review['fixture_source']==pin(F/'verify_source30_actual_checkpoint_archive_entire_selectors_v1.py') and review['external_launcher_source']==pin(__file__),'Exact frozen fixture/external guard scope')
 require(review['accepted'] is True and review['reviewer']=='/root','Exact finite ROOT scope')
 for item in inputs:require(pin(item['path'])==item,'Exact canonical input pin '+item['path'])
 output=F/'typed_selector_actual_01';require(not output.exists()and not(F/'typed_selector_actual_01_external_result.json').exists(),'Exactly one fresh actual control')
 initial=observe();m=initial['system_memory']
 require(m['commit_available_bytes']>=2000000000+268435456+512*1024**2 and m['physical_available_bytes']>=1500000000+268435456+512*1024**2,'Full prospective512MiB allowance')
 entry={'external':who(psutil.Process()),'external_launcher':who(psutil.Process().parent()),'ROOT_scope':pin(a.ROOT_review),'inputs_before':inputs,'initial_sample':initial}
 write(F/'typed_selector_actual_01_external_entry.json',entry)
 command=[sys.executable,'-Xutf8','-B',review['fixture_source']['path'],'--ROOT-review',str(Path(a.ROOT_review).resolve())]
 error=None;cleanup=[]
 with (F/'typed_selector_actual_01_stdout.log').open('xb')as log:
  child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy(),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
  launcher=who(psutil.Process(child.pid));known[(launcher['pid'],launcher['create_time'])]=launcher
  write(F/'typed_selector_actual_01_external_launch.json',{'command':command,'Popen_launcher':launcher})
  try:
   while child.poll()is None:
    try:
     for p in psutil.Process(child.pid).children(recursive=True):
      try:observed=who(p);known[(observed['pid'],observed['create_time'])]=observed
      except psutil.NoSuchProcess:pass
    except psutil.NoSuchProcess:pass
    observe();time.sleep(.05)
   code=child.wait(timeout=1)
  except BaseException as exc:error=repr(exc);code=None
  finally:
   # Only exact descendants already recorded under this owned Popen may be stopped.
   for old in reversed(list(known.values())):
    if same(old):
     p=psutil.Process(old['pid']);p.terminate();observed_code=p.wait(timeout=30)
     cleanup.append({'identity':old,'actual_targeted_wait_code':observed_code})
   if child.poll()is None:child.terminate()
   final_code=child.wait(timeout=30)
   if code is None:code=final_code
 for item in inputs:require(pin(item['path'])==item,'Every frozen input unchanged '+item['path'])
 if cleanup and error is None:error='Owned family required cleanup after fixture result; not a clean finite pass'
 deaths=[{'identity':old,'exact_identity_dead':not same(old)}for old in known.values()]
 require(all(x['exact_identity_dead']for x in deaths),'All externally observed owned identities dead')
 result={'kind':'ASR18_actual_one_owned_packet_credit_CPU_external_wait_v1','actual_Popen_wait_code':code,
  'external_guard_error':error,'targeted_cleanup':cleanup,'actual_owned_identities':deaths,'inputs_before_and_after_exact':inputs,
  'ROOT_scope':pin(a.ROOT_review),'entry':pin(F/'typed_selector_actual_01_external_entry.json'),
  'launch':pin(F/'typed_selector_actual_01_external_launch.json'),'stdout':pin(F/'typed_selector_actual_01_stdout.log'),
  'sampled_external_plus_owned_family_peak_bytes':peak,'elapsed_seconds':time.monotonic()-began,
  'report':pin(output/'report.json')if(output/'report.json').exists()else None,
  'failure':pin(output/'actual_failure.json')if(output/'actual_failure.json').exists()else None}
 write(F/'typed_selector_actual_01_external_result.json',result)
 print(json.dumps({'result':pin(F/'typed_selector_actual_01_external_result.json'),'Popen_code':code,'report':result['report'],'failure':result['failure']}),flush=True)
 if code!=0 or error is not None:raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ROOT-review',required=True);main(p.parse_args())
