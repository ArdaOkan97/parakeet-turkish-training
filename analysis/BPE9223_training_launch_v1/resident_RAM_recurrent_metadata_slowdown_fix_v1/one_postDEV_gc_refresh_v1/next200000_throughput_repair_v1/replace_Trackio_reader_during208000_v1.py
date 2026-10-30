"""One exact reader-only throughput repair at the genuine scheduled208000 pause."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import psutil

P=Path("C:/Users/ardao/Documents/Codex/2026-10-10/i-x20")
ROOT=Path("E:/gpu_training_scratchpad")
W=ROOT/"analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/next200000_throughput_repair_v1"
B=ROOT/"runs/BPE9223_resident_RAM_20261018_01"
I=B/"invocations/owner_000027"
T=ROOT/"analysis/BPE9223_Trackio_live_v1"

def read(path):return json.loads(Path(path).read_bytes())
def require(value,message):
 if not value:raise ValueError(message)
def pin(path):
 path=Path(path);raw=path.read_bytes()
 return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def write(path,value):
 raw=json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()
 with path.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def exact(who):
 try:
  p=psutil.Process(who["pid"])
  return abs(p.create_time()-who["create_time"])<.001 and p.ppid()==who["ppid"]
 except psutil.NoSuchProcess:return False

def main():
 me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 scope_path=Path(sys.argv[1]);scope=read(scope_path)
 require(scope["kind"]=="ASR18_exact_ONE_Trackio6_reader_repair_at208000_scope_v1","scopekind")
 require(pin(Path(__file__))==scope["source"],"sourcepin")
 require(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"and sys.flags.utf8_mode==1,"CPUhidden UTF8")
 require(all(os.environ.get(k)=="1"for k in("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")),"caps1")
 out=W/"actual_Trackio6_reader_repair_at208000_v1";out.mkdir(exist_ok=False)
 sys.path.insert(0,str(P/"outputs/parakeet_turkish"))
 from resource_watchdog_v2 import sample_memory
 started=time.monotonic();initial=sample_memory(os.getpid(),"reader_repair208_initial")
 initial_pin=write(out/"initial_numeric_BEFORE_predicate.json",initial)
 result=dict(kind=scope["kind"],scope=pin(scope_path),entry=dict(pid=me.pid,create_time=me.create_time(),ppid=me.ppid()),
             initial=initial_pin,source_copied=False,old_stop_actions=[],new_launch_count=0)
 logs=[]
 child=None
 try:
  sm=initial["system_memory"]
  require(initial["paired_sampling_within_one_second"],"paired")
  require(sm["commit_available_bytes"]>=5399239296 and sm["physical_available_bytes"]>=2649239296,
          "full native+72+metadata512+newcache256 prospective")
  def guard():
   m=sample_memory(os.getpid(),"reader_repair208_runtime")
   require(time.monotonic()-started<180 and m["worker_private_memory"]["private_bytes"]<536870912,"self512/wall180")
   require(m["system_memory"]["commit_available_bytes"]>=2880803840 and m["system_memory"]["physical_available_bytes"]>=2380803840,"runtimefloors+72+512+cache256")
   require(shutil.disk_usage(ROOT).free>=100001048576,"E100GB")
  guard()
  for field in("reader_source","helper","source_review","CPU_proof_ROOT"):
   require(pin(scope[field]["path"])==scope[field],field+"pin")
  require(read(scope["CPU_proof_ROOT"]["path"])["accepted"]is True,"ROOTacceptedactualCPUproof")
  target=P/"work/research/serve_BPE9223_Trackio_live_v6.py"
  require(not target.exists(),"fresh finalsource")
  db=T/"database/parakeet-turkish.db";require(db.exists(),"existingDB required, nevercopy/reseed")
  oldbody,oldlauncher=scope["old_reader"],scope["old_reader_launcher"]
  require(exact(oldbody)and exact(oldlauncher),"exact old reader pair")
  p=read(B/"progress/latest.json")
  require(207980<=p["actual_optimizer_updates"]<=208000,"only imminent natural208pause")
  require(p["producer"]==scope["native"]and p["planned_native_updates"]==401072 and p["administrative_update_stop"]is None,"native/fulltarget")
  while True:
   guard()
   p=read(B/"progress/latest.json")
   require(p["producer"]==scope["native"]and not p["optimizer_poisoned"]and not p["schedule_poisoned"]and not(I/"actual_failure.json").exists(),"healthy native")
   require(p["actual_optimizer_updates"]<=208000,"natural pause window passed; no action")
   qpath=I/"CPU_periodic_000208000_quiescence.json"
   after=I/"CPU_after_periodic_DEV_000208000/actual_CPU_ready.json"
   if p["actual_optimizer_updates"]==208000 and qpath.exists()and not after.exists():break
   time.sleep(.5)
  q=read(qpath)
  require(q["optimizer_updates"]==208000 and q["CPU_actual0"]["actual_Popen_wait_code"]==0,"actual normal208CPU closure")
  require(not exact(scope["old_CPU"])and not exact(scope["old_CPU_launcher"]),"oldCPU exactdead in naturalDEV")
  require(not after.exists(),"postDEV CPU not yet started")
  completed={r["optimizer_updates"]for r in read(B/"periodic_DEV.json")["completed_evaluations"]}
  require(208000 not in completed,"natural208DEV still in progress")
  result["natural_pause"]={"progress":p,"quiescence":q,"quiescence_pin":pin(qpath),"post_CPU_ready_absent":True,"DEV208_not_complete":True}
  raw=Path(scope["reader_source"]["path"]).read_bytes()
  with target.open("xb")as f:f.write(raw);f.flush();os.fsync(f.fileno())
  require(pin(target)["sha256"]==scope["reader_source"]["sha256"],"final exactsource bytes")
  result["source_copied"]=True;result["final_reader_source"]=pin(target)
  # Recheck the natural pause immediately before touching the dashboard reader.
  require(read(B/"progress/latest.json")["actual_optimizer_updates"]==208000 and not after.exists(),"still native-idle before readerrepair")
  require(exact(oldbody)and exact(oldlauncher),"same targeted reader pair before stop")
  for who in(oldbody,oldlauncher):
   if exact(who):
    proc=psutil.Process(who["pid"]);proc.terminate()
    try:exitcode=proc.wait(timeout=15)
    except psutil.TimeoutExpired:raise RuntimeError("Targeted oldreader did not exit within15seconds")
    result["old_stop_actions"].append(dict(identity=who,terminate_called=True,wait_return=exitcode))
   else:result["old_stop_actions"].append(dict(identity=who,terminate_called=False,numeric_OS_exit="UNRECORDED_UNKNOWN",exact_identity_absent=True))
  for who in(oldbody,oldlauncher):require(not exact(who),"oldpair exactdead")
  with socket.socket()as probe:require(probe.connect_ex(("127.0.0.1",7860))!=0,"oldportreleased")
  guard()
  require(read(B/"progress/latest.json")["actual_optimizer_updates"]==208000 and not after.exists(),"still native-idle before newreader launch")
  stdout=(out/"reader_stdout.log").open("xb");stderr=(out/"reader_stderr.log").open("xb");logs=[stdout,stderr]
  argv=[str(P/"work/monitor-env/Scripts/python.exe"),"-X","utf8","-B",str(target)]
  child=subprocess.Popen(argv,cwd=str(P),env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                         creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
  result["new_launch_count"]=1
  result["new_launcher"]={"pid":child.pid,"create_time":psutil.Process(child.pid).create_time(),"ppid":os.getpid()}
  result["argv"]=argv
  observed=None;deadline=time.monotonic()+25
  while time.monotonic()<deadline:
   guard()
   require(child.poll()is None,"newreader launcher ended")
   try:
    owner=read(T/"owner.json")
    if owner["source_sha256"]==scope["reader_source"]["sha256"]and owner["pid"]!=oldbody["pid"]:
     observed=owner;break
   except(OSError,ValueError,KeyError):pass
   time.sleep(.2)
  require(observed is not None,"newreader owner not observed; no secondlaunch")
  np=psutil.Process(observed["pid"])
  require(abs(np.create_time()-observed["create_time"])<.001 and np.ppid()==child.pid,"new body/launcher exactancestry")
  result["new_reader"]={"pid":np.pid,"create_time":np.create_time(),"ppid":np.ppid()}
  result["new_owner"]=observed
  result["passed"]=True;result["error"]=None
 except BaseException as exc:
  result["passed"]=False;result["error"]={"type":type(exc).__name__,"message":str(exc)}
 finally:
  for f in logs:f.close()
 result["wall_seconds"]=time.monotonic()-started
 result["training_or_DB_state_changed"]=False
 result["no_extra_DEV_or_GC_or_training_stop"]=True
 rp=write(out/"result.json",result)
 print(json.dumps(dict(pin=rp,passed=result["passed"],error=result.get("error"),new_launch_count=result["new_launch_count"],new_reader=result.get("new_reader"))),flush=True)
 return 0 if result["passed"] else 1

if __name__=="__main__":raise SystemExit(main())
