"""INACTIVE external guard for one separately ROOT-scoped metadata-only proof.
Full sampled family includes exact external interpreter launcher parent; that
parent is never a cleanup target or waited on by its live child.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).resolve().parent;BEGIN=time.monotonic();known={};PEAK=0;PARENT=None
FORBIDDEN={2456,41672,34044,38964,24620,26892}
def require(v,m):
 if not v:raise ValueError(m)
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,indent=2);f.write('\n')
def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def same(old):
 try:return abs(psutil.Process(old['pid']).create_time()-old['create_time'])<.001
 except psutil.NoSuchProcess:return False
def sampled_private(old):
 if not same(old):return 0
 info=psutil.Process(old['pid']).memory_info();return getattr(info,'private',info.rss)
def owned_under(old,root):
 if old['pid']in FORBIDDEN or not same(old)or not same(root):return False
 if old['pid']==root['pid']:return old==root
 try:
  return any(p.pid==root['pid']and abs(p.create_time()-root['create_time'])<.001 for p in psutil.Process(old['pid']).parents())
 except psutil.NoSuchProcess:return False
def discover(root):
 if not same(root):return
 for p in psutil.Process(root['pid']).children(recursive=True):
  try:
   old=who(p)
   if owned_under(old,root):known[(old['pid'],old['create_time'])]=old
  except psutil.NoSuchProcess:pass
def observe():
 global PEAK
 v=sample_memory(os.getpid(),'source33_eighth_typed_metadata_external');m=v['system_memory']
 identities={k:x for k,x in known.items()}
 if PARENT is not None:identities[(PARENT['pid'],PARENT['create_time'])]=PARENT
 total=v['worker_private_memory']['private_bytes']+sum(sampled_private(x)for x in identities.values())
 PEAK=max(PEAK,total)
 require(v['paired_sampling_within_one_second']and total<536870912
  and m['commit_available_bytes']>=2268435456 and m['physical_available_bytes']>=1768435456
  and shutil.disk_usage('E:/').free>=100000000000 and time.monotonic()-BEGIN<240,
  'External240s/512MiB fullfamily including exact parent/FULLprefetch/originalfloors/E100GB')
 return v
def main(a):
 global PARENT
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and os.environ.get('PYTHONUTF8')=='1'
  and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
  and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),'Exact E CPU-only metadata runtime')
 review=read(a.ROOT_review);inputs=review['input_pins'];fixture=F/'verify_source33_eighth_actual_failure_entire_selector_metadata_v1.py'
 require(review['accepted']is True and review['reviewer']=='/root'
  and review['scope']=='ASR18_source33_eighth_original_failure_entire_selector_metadata_CPU_v1'
  and review['fixture_source']==pin(fixture)and review['external_launcher_source']==pin(__file__),'Exact separate ROOT metadata scope')
 for item in inputs:require(pin(item['path'])==item,'Exact canonical input before launch')
 output=F/'source33_typed_eighth_selector_actual_01';stem='source33_typed_eighth_selector_actual_01'
 require(not output.exists()and not(F/(stem+'_external_result.json')).exists()
  and not(F/(stem+'_external_entry.json')).exists(),'Fresh one-invocation namespace')
 PARENT=who(psutil.Process().parent());selfid=who(psutil.Process());initial=observe();m=initial['system_memory']
 require(m['commit_available_bytes']>=2805306368 and m['physical_available_bytes']>=2305306368,
  'Full prospective512MiB ON TOP of FULLprefetch/originalfloors; no measured/freed credit')
 write(F/(stem+'_external_entry.json'),{'external':selfid,'external_launcher':PARENT,'parent_live_at_write':True,
  'ROOT_scope':pin(a.ROOT_review),'initial_guard':initial,'input_pins':inputs})
 command=[sys.executable,'-Xutf8','-B',str(fixture),'--ROOT-review',str(Path(a.ROOT_review).resolve())]
 error=None;cleanup=[];code=None;root=None;final=None
 with (F/(stem+'_stdout.log')).open('xb')as log:
  child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy(),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
  root=who(psutil.Process(child.pid));require(root['pid']not in FORBIDDEN,'New owned metadata launcher only')
  known[(root['pid'],root['create_time'])]=root
  write(F/(stem+'_external_launch.json'),{'command':command,'Popen_launcher':root})
  try:
   while child.poll()is None:
    discover(root);observe();time.sleep(.05)
   code=child.wait(timeout=1);final=observe()
  except BaseException as exc:error=repr(exc)
  finally:
   # Capture/recheck exact identity plus current ancestry. Never act on parent,
   # production/reader IDs or a root PID that has been reused since Popen.
   discover(root)
   for old in reversed(list(known.values())):
    if owned_under(old,root):
     p=psutil.Process(old['pid'])
     if same(old)and owned_under(old,root):
      p.terminate();wait=p.wait(timeout=30);cleanup.append({'identity':old,'actual_targeted_wait_code':wait})
   if child.poll()is None:
    require(owned_under(root,root),'Only exact owned Popen root may be cleaned')
    child.terminate()
   actual=child.wait(timeout=30)
   if code is None:code=actual
 if cleanup and error is None:error='Owned metadata process required cleanup; not a clean proof'
 unchanged=[]
 for item in inputs:
  require(pin(item['path'])==item,'Every frozen input exact after metadata proof');unchanged.append(item)
 deaths=[{'identity':old,'exact_identity_dead':not same(old)}for old in known.values()]
 require(all(x['exact_identity_dead']for x in deaths),'All observed owned metadata descendants dead')
 result={'kind':'ASR18_source33_eighth_typed_metadata_actual_external_result_v1','actual_Popen_wait_code':code,
  'external_guard_error':error,'targeted_cleanup':cleanup,'actual_owned_identities':deaths,
  'external_identity':selfid,'external_launcher_identity':PARENT,'external_parent_live_at_write':same(PARENT),
  'external_parent_in_sampled_family_not_cleanup_targets':True,'sampled_full_family_peak_bytes':PEAK,
  'initial_guard':initial,'final_guard':final,'elapsed_seconds':time.monotonic()-BEGIN,
  'ROOT_scope':pin(a.ROOT_review),'inputs_before_after_exact':unchanged,
  'entry':pin(F/(stem+'_external_entry.json')),'launch':pin(F/(stem+'_external_launch.json')),
  'stdout':pin(F/(stem+'_stdout.log')),'report':pin(output/'report.json')if(output/'report.json').exists()else None,
  'failure':pin(output/'actual_failure.json')if(output/'actual_failure.json').exists()else None}
 write(F/(stem+'_external_result.json'),result)
 print(json.dumps({'external_result':pin(F/(stem+'_external_result.json')),'actual_Popen_wait_code':code,
  'external_guard_error':error,'report':result['report'],'failure':result['failure']}),flush=True)
 if code!=0 or error is not None:raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ROOT-review',required=True);main(p.parse_args())
