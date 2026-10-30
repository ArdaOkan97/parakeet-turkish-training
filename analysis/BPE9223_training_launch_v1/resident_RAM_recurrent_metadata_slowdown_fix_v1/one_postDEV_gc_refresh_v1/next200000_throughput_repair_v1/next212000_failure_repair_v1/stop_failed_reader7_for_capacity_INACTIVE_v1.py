"""INACTIVE stop-only proposal for the exact already-failed Reader7 pair.
No replacement, training control, data edits, DB access or automatic proof.
Requires a separately approved ROOT execution scope with an explicit human
exception to the standing proof-before-reader-control ordering.
"""
import argparse, ctypes, hashlib, json, os, shutil, time
from pathlib import Path
import psutil
ROOT=Path("E:/gpu_training_scratchpad")
X=ROOT/"analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/next200000_throughput_repair_v1/next212000_failure_repair_v1"
AREA=ROOT/"analysis/BPE9223_Trackio_live_v1"
BRANCH=ROOT/"runs/BPE9223_resident_RAM_20261018_01"
TARGETS=[{"pid":25012,"create_time":1792872869.5210469,"ppid":34368},
         {"pid":34368,"create_time":1792872869.5039415,"ppid":21280}]
NATIVE={"pid":27664,"create_time":1792877280.3418417,"ppid":23820}
OUT=X/"reader7_failed_stop_only_actual_01"
READER_SHA="d7bb6494d197cfe4b4f7bc8c5af1f4b720be199e2fb8d3077cb8af45e450da22"
DRIVER_SHA="bd1017bb044c21961a850861f7dc362c5fbcd83463c523428cd1ef0a185f4774"
class MEMORY(ctypes.Structure):
    _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(n,ctypes.c_ulonglong)for n in ("totalphysical","physical","commitlimit","commit","totalvirtual","virtual","extended")]
class FILETIME(ctypes.Structure):
    _fields_=[("low",ctypes.c_ulong),("high",ctypes.c_ulong)]
def require(value,message):
    if not value:raise RuntimeError(message)
def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode()
def write_new(path,value):
    raw=canonical(value)
    with path.open("xb")as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return {"path":str(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
def identity(expected):
    process=psutil.Process(expected["pid"])
    value={"pid":process.pid,"create_time":process.create_time(),"ppid":process.ppid()}
    require(value["ppid"]==expected["ppid"]and abs(value["create_time"]-expected["create_time"])<.001,"Exact original process identity required")
    return value
def main(review_path):
    policy=json.loads((ROOT/"storage_config.json").read_text(encoding="utf-8-sig"))
    require(type(policy)is dict and ROOT.is_dir(),"Designated E storage required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"and all(os.environ.get(n)=="1"for n in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")),"CPU-only original metadata environment")
    for name in ("TEMP","TMP","TORCH_EXTENSIONS_DIR"):
        require(os.environ.get(name)and Path(os.environ[name]).resolve().is_relative_to(ROOT.resolve()),"Required process-local E storage environment")
    process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);began=time.monotonic()
    require(not OUT.exists(),"Fresh exact stop-only output required");OUT.mkdir()
    def sample():
        m=MEMORY();m.length=ctypes.sizeof(m)
        require(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)),"Available memory sample")
        return {"time":time.time(),"pid":process.pid,"create_time":process.create_time(),"ppid":process.ppid(),"commit":m.commit,"physical":m.physical,"private":process.memory_info().private,"E_free":shutil.disk_usage(ROOT).free,"elapsed":time.monotonic()-began}
    initial_sample=sample()
    initial=write_new(OUT/"initial_BEFORE_predicate.json",{"sample":initial_sample,"prospective":[6028384896,3278384896],"runtime":[3509949440,3009949440],"self_private":536870912,"wall":180,"E_free":100000000000})
    def limits(value,commit,physical):
        require(value["commit"]>=commit and value["physical"]>=physical and value["private"]<=536870912 and value["E_free"]>=100000000000 and value["elapsed"]<180,"Unchanged complete metadata resource predicate")
    fresh=sample();gate=write_new(OUT/"fresh_gate_numeric_BEFORE_predicate.json",{"sample":fresh,"initial":initial})
    for value in (initial_sample,fresh):limits(value,6028384896,3278384896)
    def guard():
        value=sample();limits(value,3509949440,3009949440);return value
    def read_metadata(path,maximum,label):
        guard();before=path.stat();require(0<before.st_size<=maximum,"Bounded exact metadata")
        raw=path.read_bytes()
        witness=write_new(OUT/(label+"_FIRST_bytes.json"),{"path":str(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"UTF8_text":raw.decode("utf-8-sig")})
        after=path.stat();require(before.st_size==len(raw)==after.st_size and before.st_mtime_ns==after.st_mtime_ns,"Stable exact metadata during capture")
        return json.loads(raw),witness
    scope,scope_capture=read_metadata(Path(review_path),65536,"ROOT_execution_scope")
    require(scope.get("kind")=="ROOT_exact_ONE_failed_reader7_stop_only_for_capacity_v1"and scope.get("accepted")is True and scope.get("reviewer")=="/root","Separate exact approved ROOT execution scope")
    require(scope.get("reader_control_order_exception")is True and type(scope.get("explicit_human_authorization"))is str and bool(scope["explicit_human_authorization"].strip()),"Explicit human exception to proof-before-reader-control ordering")
    require(scope["targets"]==TARGETS and scope["native"]==NATIVE and scope["output"]==str(OUT)and scope["action"]=="stop_failed_reader7_only_no_replacement","Exact unchanged stop-only targets/action")
    guard();own=Path(__file__).read_bytes();require(scope["actor_source"]=={"path":str(Path(__file__).resolve()),"bytes":len(own),"sha256":hashlib.sha256(own).hexdigest()},"Exact independently reviewed Source")
    status,status_capture=read_metadata(AREA/"current_status.json",131072,"current_failure_status")
    progress,progress_capture=read_metadata(BRANCH/"progress/latest.json",131072,"current_training_progress")
    require(status["pid"]==25012 and abs(status["create_time"]-TARGETS[0]["create_time"])<.001 and status["source_sha256"]==READER_SHA,"Same failed installed reader")
    require(status["status"]=="mirror_thread_failed"and status["mirror_thread_alive"]is False and status["reader_ready"]is False,"Never stop a newly healthy reader")
    require(status["mirrored_update"]==212859 and status["journal_offset"]==427713850 and "WinError 5"in str(status.get("error")),"Exact known failed publication/frontier fence")
    installed=Path(status["source"]).resolve()
    require(installed==Path(scope["installed_source"]).resolve(),"Exact freshly bound installed reader path")
    require(installed.stat().st_size==26204 and hashlib.sha256(installed.read_bytes()).hexdigest()==READER_SHA,"Unchanged installed reader7 bytes")
    require(progress["source"]["sha256"]==DRIVER_SHA and progress["optimizer_poisoned"]is False and progress["schedule_poisoned"]is False and progress["actual_optimizer_updates"]>=226620 and progress["planned_native_updates"]==401072 and progress["administrative_update_stop"]is None,"Genuine unchanged healthy Source41 training")
    require(progress["producer"]["pid"]==27664 and Path(progress["journal_prefix"]["path"]).resolve()==(BRANCH/"journals/commits_000029.jsonl").resolve(),"Source41 native/J29 binding")
    identity(NATIVE)
    for target in TARGETS:identity(target)
    require({p.pid for p in psutil.Process(34368).children(recursive=True)}=={25012}and not psutil.Process(25012).children(recursive=True),"Exact pair only; no unbound descendants")
    def listeners():return [{"pid":c.pid,"address":c.laddr.ip,"port":c.laddr.port}for c in psutil.net_connections(kind="tcp")if c.status==psutil.CONN_LISTEN and c.laddr.port==7860]
    require(listeners()==[{"pid":25012,"address":"127.0.0.1","port":7860}],"Exact failed reader's listening port")
    db=(AREA/"database/parakeet-turkish.db").resolve()
    require(Path(status["database"]).resolve()==db,"Same actual reader database path")
    def stat_storage():
        return [{"path":str(q),"exists":q.exists(),"bytes":q.stat().st_size if q.exists()else None,"mtime_ns":q.stat().st_mtime_ns if q.exists()else None}for q in (AREA/"mirror_state.json",db,Path(str(db)+"-wal"),Path(str(db)+"-shm"))]
    before=stat_storage();require(before[0]["exists"]and before[1]["exists"],"Preserve original STATE and database")
    scope_admission=write_new(OUT/"admission_BEFORE_ANY_control.json",{"scope":scope_capture,"initial":initial,"gate":gate,"status":status_capture,"progress":progress_capture,"targets":[identity(t)for t in TARGETS],"native":identity(NATIVE),"storage_STAT_only":before,"no_DB_STATE_journal_or_training_writes":True,"fresh":guard()})
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.GetProcessTimes.argtypes=[ctypes.c_void_p,ctypes.POINTER(FILETIME),ctypes.POINTER(FILETIME),ctypes.POINTER(FILETIME),ctypes.POINTER(FILETIME)]
    kernel.GetProcessId.argtypes=[ctypes.c_void_p];kernel.GetProcessId.restype=ctypes.c_ulong
    kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
    kernel.TerminateProcess.argtypes=[ctypes.c_void_p,ctypes.c_uint]
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong];kernel.WaitForSingleObject.restype=ctypes.c_ulong
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handles=[];outcomes=[];error=None;primary=None
    try:
        for target in TARGETS:
            handle=kernel.OpenProcess(0x00100000|0x1000|0x0001,False,target["pid"])
            require(bool(handle),"Open exact owned process handle")
            handles.append(handle)
            created,exited,kt,ut=FILETIME(),FILETIME(),FILETIME(),FILETIME()
            require(kernel.GetProcessTimes(handle,ctypes.byref(created),ctypes.byref(exited),ctypes.byref(kt),ctypes.byref(ut)),"Owned handle creation time")
            ctime=((created.high<<32)|created.low)/10000000-11644473600
            require(kernel.GetProcessId(handle)==target["pid"]and abs(ctime-target["create_time"])<.001,"Handle binds exact original identity")
            identity(target)
        for index,(target,handle)in enumerate(zip(TARGETS,handles)):
            guard();identity(NATIVE);code=ctypes.c_ulong()
            require(kernel.GetExitCodeProcess(handle,ctypes.byref(code)),"Actual owned exit code before action")
            terminated=False
            if code.value==259:
                identity(target);require(kernel.TerminateProcess(handle,15),"Exact failed-reader termination");terminated=True
            deadline=time.monotonic()+15
            while True:
                guard();remaining=deadline-time.monotonic();require(remaining>0,"Exact15s owned wait")
                wait=kernel.WaitForSingleObject(handle,max(1,min(250,int(remaining*1000))))
                if wait==0:break
                require(wait==258 and time.monotonic()<deadline,"Exact15s owned wait; no extended retry")
            require(kernel.GetExitCodeProcess(handle,ctypes.byref(code)),"Actual owned process exit code")
            outcomes.append({"identity":target,"termination_called":terminated,"actual_GetExitCodeProcess":code.value,"actual_handle_wait_signaled":True,"normal_reader_exit_claimed":False})
        deadline=time.monotonic()+5
        while listeners():
            guard();require(time.monotonic()<deadline,"Owned port release within5s");time.sleep(.25)
        for target in TARGETS:
            try:require(abs(psutil.Process(target["pid"]).create_time()-target["create_time"])>=.001,"Exact original reader identity dead")
            except psutil.NoSuchProcess:pass
        require(stat_storage()[0]==before[0],"Original STATE STAT unchanged by stop-only action")
        identity(NATIVE)
    except BaseException as exc:
        primary=exc;error=type(exc).__name__+": "+str(exc)
    finally:
        handle_results=[]
        for handle in reversed(handles):
            closed=bool(kernel.CloseHandle(handle));handle_results.append(closed)
            if not closed and primary is None:primary=RuntimeError("Owned process handle closure failed");error=str(primary)
        try:native_after=identity(NATIVE)
        except BaseException as observation_error:native_after={"unavailable":type(observation_error).__name__}
        try:after_storage=stat_storage()
        except BaseException as observation_error:after_storage={"unavailable":type(observation_error).__name__}
        result={"kind":"ROOT_actual_failed_reader7_stop_only_result_v1","completed":primary is None,"admission":scope_admission,"outcomes":outcomes,"owned_handle_CloseHandle_returns":handle_results,"storage_STAT_before":before,"storage_STAT_after":after_storage,"native_after":native_after,"reader_restart_or_replacement":0,"production_training_or_DB_STATE_journal_writes":0,"proof_admission":0,"no_freed_capacity_credit_or_cause_claim":True,"error":error,"final_observation":sample()}
        write_new(OUT/"result.json",result)
    if primary is not None:raise primary
    print(json.dumps(result,separators=(",",":")))
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--ROOT-review",required=True)
    args=parser.parse_args()
    try:main(args.ROOT_review)
    except BaseException as error:
        if OUT.is_dir():
            try:write_new(OUT/"failure_seal.json",{"kind":"ROOT_failed_reader7_stop_only_failure_seal_v1","error":type(error).__name__+": "+str(error),"result_exists":(OUT/"result.json").is_file(),"actual_control_count":"SEE_RECORDED_RESULT_OR_UNKNOWN","automatic_retry_authorized":False,"time":time.time()})
            except BaseException as seal_error:error.add_note("Failure seal: "+type(seal_error).__name__+": "+str(seal_error))
        raise
