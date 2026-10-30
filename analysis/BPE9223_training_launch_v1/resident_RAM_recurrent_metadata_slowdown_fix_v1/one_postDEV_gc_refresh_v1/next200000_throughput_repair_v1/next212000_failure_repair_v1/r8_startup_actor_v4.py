"""INACTIVE SOURCE DRAFT: exact failed Reader7 pair -> one Reader8 launch.
No execution scope is present. The draft manifest is accepted=False/public_entry=None.
A future ROOT scope must bind every pending input, exact source pin and entry.
No production restart, old-reader fallback, DB/STATE edit or native control.
"""
import argparse,ctypes,hashlib,json,os,shutil,subprocess,sys,time,urllib.request
from pathlib import Path
import psutil

P=Path("C:/Users/ardao/Documents/Codex/2026-10-10/i-x20")
E=Path("E:/gpu_training_scratchpad")
X=E/"analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/next200000_throughput_repair_v1/next212000_failure_repair_v1"
T=E/"analysis/BPE9223_Trackio_live_v1"
B=E/"runs/BPE9223_resident_RAM_20261018_01"
PYTHON=P/"work/monitor-env/Scripts/python.exe"
TARGET=P/"outputs/parakeet_turkish/serve_BPE9223_Trackio_live_v8_repaired_v1.py"
CANDIDATE=X/"serve_BPE9223_Trackio_live_v8_INACTIVE_v1.py"
OLD=[dict(pid=25012,ctime=1792872869.5210469,ppid=34368),
     dict(pid=34368,ctime=1792872869.5039415,ppid=21280)]
OLD_EXTRA=[dict(pid=18592,ctime=1792872869.5103593,ppid=34368)]
NATIVE=dict(pid=27664,ctime=1792877280.3418417,ppid=23820)
CLOSED_LEASE=dict(path=str((P/"work/gpu_training_owner.json").resolve()),bytes=22011,sha256="57743f5c1950eacc1718ac53446eaffcf0eeb98e8889db51f1294fb420314c06")
SHA_READER7="d7bb6494d197cfe4b4f7bc8c5af1f4b720be199e2fb8d3077cb8af45e450da22"
SHA_READER8="efd774dc960f54191260e2f883bdb95ec970985af12935d850fc9ced5b30d724"
SHA_DRIVER="bd1017bb044c21961a850861f7dc362c5fbcd83463c523428cd1ef0a185f4774"
COMBINED=dict(path=str(X/"ROOT_reader8_both500k_actual_acceptance_0110_v1/acceptance.json"),bytes=4228,sha256="c2482a80cadd1fc2ab4755621c19dc1860815bcf73f38ffe85c0a854a31adfbd")
LIMITS=dict(prospective_commit=8066304384,prospective_physical=5316304384,
 runtime_commit=5547868928,runtime_physical=5047868928,self_private=536870912,
 reader_private=1500000000,family_private=2573741824,body_seconds=160,
 external_seconds=180,HTTP_seconds=20,readiness_seconds=60,interval=.1,
 E_free=100000000000,log_bytes=16777216,descendants=16,known_identities=32)
BODY_OUT=X/"reader8_startup_actual04"
EXT_OUT=X/"reader8_startup_external_actual04"

class Memory(ctypes.Structure):
    _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(n,ctypes.c_ulonglong)for n in ("totalphysical","physical","commitlimit","commit","TotalVirtual","AvailableVirtual","extended")]
class FileTime(ctypes.Structure):
    _fields_=[("low",ctypes.c_ulong),("high",ctypes.c_ulong)]
def need(value,message):
    if not value:raise RuntimeError(message)
def raw_json(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False,separators=(",",":")).encode()
def pin(path,raw):
    return dict(path=str(Path(path).resolve()),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def write_raw(path,raw):
    with Path(path).open("xb")as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return pin(path,raw)
def write(path,value):return write_raw(path,raw_json(value))
def publish_once(path,value):
    path=Path(path);need(not path.exists(),"Exclusive one-time actor receipt")
    temporary=path.with_name(path.name+".tmp")
    write_raw(temporary,raw_json(value))
    os.replace(temporary,path)
    return pin(path,raw_json(value))
def observation(process):
    return dict(pid=process.pid,ctime=process.create_time(),ppid=process.ppid())
def exact_process(expected,missing=False):
    try:
        held=psutil.Process(expected["pid"])
        got=observation(held)
    except psutil.NoSuchProcess:
        need(not psutil.pid_exists(expected["pid"]),"Live/reused PID after an identity sampling race")
        if missing:return None
        raise
    need(got=={k:expected[k]for k in ("pid","ctime","ppid")},"Exact PID/creation-time/parent identity required")
    return held
def exact_dead(expected):
    try:q=psutil.Process(expected["pid"]);ctime=q.create_time()
    except psutil.NoSuchProcess:
        need(not psutil.pid_exists(expected["pid"]),"Live/reused PID during exact-death observation");return True
    need(ctime!=expected["ctime"],"Exact prior identity remains alive")
    return True
def stat_only(path):
    q=Path(path)
    try:s=q.stat();return dict(path=str(q.resolve()),exists=True,bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    except FileNotFoundError:return dict(path=str(q.resolve()),exists=False,bytes=None,mtime_ns=None)
def listeners():
    return [dict(pid=c.pid,address=c.laddr.ip,port=c.laddr.port)for c in psutil.net_connections(kind="tcp")if c.status==psutil.CONN_LISTEN and c.laddr.port==7860]
def listener_key(item):return(item["pid"],item["address"],item["port"])
def listener_fence(value,reader_pid,allow_reader_absent=False):
    actual=listeners()
    reader=dict(pid=reader_pid,address="127.0.0.1",port=7860)
    approved=value["approved_non_reader_7860_listeners"]
    need(type(approved)is list and len(approved)<=8 and all(type(x.get("pid"))is int and x["pid"]not in [t["pid"]for t in OLD]and x["port"]==7860 for x in approved),"Separately ROOT-reviewed non-reader listener roles")
    count=actual.count(reader)
    need(count==1 or(allow_reader_absent and count==0),"Exact loopback reader listener role")
    remaining=[x for x in actual if x!=reader]
    need(sorted(remaining,key=listener_key)==sorted(approved,key=listener_key),"Preserve approved LAN proxy listeners exactly; no proxy control")
    return dict(actual=actual,reader_present=count==1,approved_non_reader=approved)

class Context:
    def __init__(self,out,seconds,label):
        self.process=psutil.Process();self.process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        self.out=out;self.seconds=seconds;self.began=time.monotonic()
        need(not out.exists(),"Fresh exclusive output directory required");out.mkdir()
        first=self.sample()
        self.initial=write(out/"initial_BEFORE_predicate.json",dict(sample=first,limits=LIMITS,native_full_reserve_remains_charged=True))
        fresh=self.sample()
        self.fresh=write(out/"fresh_numeric_BEFORE_predicate.json",dict(sample=fresh,initial=self.initial))
        print(json.dumps(dict(record=label+"_numeric_BEFORE_predicate",first=first,fresh=fresh,initial=self.initial,gate=self.fresh),separators=(",",":")),flush=True)
        self.check(first,True);self.check(fresh,True)
        self.access_scope=write(out/"scope_BEFORE_any_input_or_target_stat.json",dict(kind="ROOT_READER8_replacement_actor_access_scope_v1",initial=self.initial,fresh=self.fresh,actual_source_is_DRAFT_until_ROOT_execution_scope_verified=True,no_DB_STATE_history_or_journal_content_reads=True))
        self.raw_inputs=[];self.logs=[]
    def sample(self):
        m=Memory();m.length=ctypes.sizeof(m);need(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)),"GlobalMemoryStatusEx")
        return dict(time=time.time(),identity=observation(self.process),commit=m.commit,physical=m.physical,private=self.process.memory_info().private,E_free=shutil.disk_usage(E).free,elapsed=time.monotonic()-self.began)
    def check(self,s,initial=False):
        c=LIMITS["prospective_commit"]if initial else LIMITS["runtime_commit"]
        p=LIMITS["prospective_physical"]if initial else LIMITS["runtime_physical"]
        need(s["commit"]>=c and s["physical"]>=p and s["private"]<=LIMITS["self_private"]and s["E_free"]>=LIMITS["E_free"]and s["elapsed"]<self.seconds,"Unchanged complete actor resource predicate")
    def guard(self):
        s=self.sample();self.check(s)
        for q in self.logs:
            if q.exists():need(q.stat().st_size<=LIMITS["log_bytes"],"Owned log cap")
        return s
    def read(self,path,maximum,label):
        self.guard();q=Path(path);before=q.stat();need(0<before.st_size<=maximum,"Bounded declared input")
        raw=q.read_bytes()
        capture=write_raw(self.out/(label+"_FIRST_RAW.bin"),raw)
        index=write(self.out/(label+"_FIRST_RAW_before_decode_index.json"),dict(original=pin(q,raw),RAW=capture,fsynced_before_decode_pin_semantics=True))
        after=q.stat();need(before.st_size==len(raw)==after.st_size and before.st_mtime_ns==after.st_mtime_ns,"Stable input capture")
        self.raw_inputs.append(index)
        return raw
    def json(self,path,maximum,label):
        raw=self.read(path,maximum,label);return json.loads(raw),pin(path,raw)
    def env(self):
        need(os.name=="nt"and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"and sys.flags.utf8_mode==1 and sys.dont_write_bytecode,"Windows CPU-only UTF8-B actor")
        need(all(os.environ.get(k)=="1"for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")),"Original thread caps")
        for k in ("TEMP","TMP","TORCH_EXTENSIONS_DIR"):
            need(os.environ.get(k)and Path(os.environ[k]).resolve().is_relative_to(E.resolve()),"Process-local E storage")
        need(os.environ.get("TORCH_EXTENSIONS_DIR")==r"E:\gpu_training_scratchpad\cache\torch_extensions","Exact candidate-compatible backslash TORCH_EXTENSIONS_DIR before any old control")
        policy,_=self.json(E/"storage_config.json",65536,"storage_policy")
        need(type(policy)is dict,"E storage policy object")

HANDLE_EVENTS=[]
class OwnedHandle:
    def __init__(self,expected):
        self.expected={k:expected[k]for k in ("pid","ctime","ppid")}
        self.k=ctypes.WinDLL("kernel32",use_last_error=True)
        self.k.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];self.k.OpenProcess.restype=ctypes.c_void_p
        self.k.GetProcessId.argtypes=[ctypes.c_void_p];self.k.GetProcessId.restype=ctypes.c_ulong
        self.k.GetProcessTimes.argtypes=[ctypes.c_void_p,ctypes.POINTER(FileTime),ctypes.POINTER(FileTime),ctypes.POINTER(FileTime),ctypes.POINTER(FileTime)]
        self.k.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
        self.k.TerminateProcess.argtypes=[ctypes.c_void_p,ctypes.c_uint]
        self.k.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong];self.k.WaitForSingleObject.restype=ctypes.c_ulong
        self.k.CloseHandle.argtypes=[ctypes.c_void_p]
        exact_process(self.expected)
        self.handle=self.k.OpenProcess(0x00100000|0x1000|0x0001,False,self.expected["pid"])
        need(bool(self.handle),"Owned exact Windows process handle")
        a,b,c,d=FileTime(),FileTime(),FileTime(),FileTime()
        if not self.k.GetProcessTimes(self.handle,ctypes.byref(a),ctypes.byref(b),ctypes.byref(c),ctypes.byref(d)):
            closed=bool(self.k.CloseHandle(self.handle))
            HANDLE_EVENTS.append(dict(identity=self.expected,phase="GetProcessTimes_failure",CloseHandle=closed))
            raise RuntimeError("GetProcessTimes")
        ticks=(a.high<<32)|a.low
        ctime=float(ticks-116444736000000000)/10000000 # Same explicit C-double conversion as psutil; strict equality retained.
        self.closed=False
        self.binding=dict(identity=self.expected,GetProcessId=self.k.GetProcessId(self.handle),creation_FILETIME=ticks,creation_time=ctime)
        try:
            need(self.binding["GetProcessId"]==self.expected["pid"]and ctime==self.expected["ctime"],"Strict creation-time-bound owned handle")
            exact_process(self.expected)
        except BaseException:
            HANDLE_EVENTS.append(dict(identity=self.expected,phase="handle_constructor_failure",binding=self.binding,CloseHandle=self.close()))
            raise
    def code(self):
        v=ctypes.c_ulong();need(self.k.GetExitCodeProcess(self.handle,ctypes.byref(v)),"Actual GetExitCodeProcess");return v.value
    def stop(self,ctx,seconds=15):
        ctx.guard();before=self.code();called=False
        if before==259:
            exact_process(self.expected);need(self.k.TerminateProcess(self.handle,15),"Owned exact-pair termination");called=True
        deadline=time.monotonic()+seconds;wait=None
        while True:
            ctx.guard();remaining=deadline-time.monotonic();need(remaining>0,"Unextended owned process wait")
            wait=self.k.WaitForSingleObject(self.handle,max(1,min(100,int(remaining*1000))))
            if wait==0:break
            need(wait==258,"Owned WaitForSingleObject failed")
        return dict(binding=self.binding,before_GetExitCodeProcess=before,TerminateProcess_called=called,WaitForSingleObject=wait,GetExitCodeProcess=self.code(),normal_exit_claimed=False)
    def close(self):
        if self.closed:return None
        self.closed=True;return bool(self.k.CloseHandle(self.handle))

def scope_review(ctx,path):
    value,got=ctx.json(path,65536,"ROOT_execution_scope")
    need(value.get("kind")=="ROOT_exact_ONE_installed_Reader8_startup_pending_catchup_v1"and value.get("accepted")is True and value.get("reviewer")=="/root"and value.get("owner")=="/root/memory_diagnosis","Separately approved exact ROOT execution scope")
    need(value["resources"]==LIMITS and value["old_targets"]==OLD and value["native_full_reserve"]==NATIVE,"All original reserves and exact old targets")
    need(value["allowed_old_extra_descendants"]==OLD_EXTRA and value["closed_lease"]==CLOSED_LEASE,"ROOT current extra old identity and exact closed-lease pin")
    need(value["target_source"]==str(TARGET.resolve())and value["candidate"]==dict(path=str(CANDIDATE.resolve()),bytes=28141,sha256=SHA_READER8),"Exact reviewed candidate and NEW S placement")
    old_source=Path(value["old_source"]["path"]).resolve()
    need(old_source.is_relative_to(P.resolve())and value["old_source"]==dict(path=str(old_source),bytes=26204,sha256=SHA_READER7),"Old Source path comes ONLY from separately reviewed ROOT current binding")
    need(value["body_output"]==str(BODY_OUT.resolve())and value["external_output"]==str(EXT_OUT.resolve()),"Fresh exact output bindings")
    need(value["reader_argv"]==[str(PYTHON.resolve()),"-X","utf8","-B",str(TARGET.resolve())],"Exact production CLI; no fixture bindings")
    need(value["health_URL"]=="http://127.0.0.1:7860/version"and value["all_isolated_proofs_ROOT"]==COMBINED,"Reviewed health route and combined ROOT proof authority")
    own=ctx.read(Path(__file__),131072,"actor_Source")
    need(pin(__file__,own)==value["actor_source"],"Complete canonical actor Source pin")
    receipt=ctx.read(COMBINED["path"],COMBINED["bytes"],"both500k_ROOT")
    need(pin(COMBINED["path"],receipt)==COMBINED,"Actual combined ROOT acceptance pin")
    need(value["latest_update"]==259800 and value["journal_path"]==str((B/"journals/commits_000029.jsonl").resolve())and value["journal_bytes"]==519927641,"Exact final J29 frontier")
    need(value["lease_closed"]is True and value["production_stopped"]is True and value["unknown_I30_descendants_outside_cleanup"]is True,"No production or unknown-I30 cleanup authority")
    for item in value["required_exact_dead"]:exact_dead(item)
    need(any({k:item[k]for k in ("pid","ctime","ppid")}==NATIVE for item in value["required_exact_dead"]),"Native exact identity must be independently dead, full reserve charged")
    if ctx.out==EXT_OUT:
        need(value["public_entry"]==[str(PYTHON.resolve()),"-X","utf8","-B",str(Path(__file__).resolve()),"--ROOT-review",str(Path(path).resolve())],"Exact approved PUBLIC actor entry")
    need(value["startup_only"]is True and value["full_catchup_Root_and_training_pending"]is True,"Separate startup readiness from full-journal production gate; no timeout increase")
    return value,got

def storage_stats():
    db=T/"database/parakeet-turkish.db"
    return [stat_only(p)for p in (T/"mirror_state.json",db,Path(str(db)+"-wal"),Path(str(db)+"-shm"))]

def body(private_path):
    ctx=Context(BODY_OUT,LIMITS["body_seconds"],"body");ctx.env()
    private,_=ctx.json(private_path,65536,"external_body_binding")
    external=private["external_identity"]
    exact_process(external)
    parents=ctx.process.parents()
    need(any(observation(q)==external for q in parents),"Body must descend from its exact external controller")
    value,scope_pin=scope_review(ctx,private["scope_RAW_copy"])
    need(scope_pin==private["scope_copy_pin"],"Captured ROOT scope handed off unchanged")
    need(TARGET.is_file(),"Preserve the already installed verified Reader8 Source")
    # Changed post-stop lane: both old exact identities have independently exited.
    for old in OLD:exact_dead(old)
    pre_control_listeners=listener_fence(value,25012,True)
    need(pre_control_listeners["reader_present"]is False,"Old loopback reader absent after recorded partial stop")
    prior=value["prior_partial_stop_ROOT"]
    prior_raw=ctx.read(prior["path"],65536,"prior_partial_stop_ROOT")
    need(pin(prior["path"],prior_raw)==prior,"Exact independent ROOT post-stop acceptance")
    prior_value=json.loads(prior_raw)
    need(prior_value["kind"]=="ROOT_Reader7_partial_stop_actual_acceptance_v1"and prior_value["accepted"]is True and prior_value["reviewer"]=="/root","Genuine changed post-stop lane")
    need(prior_value["old_pair_exact_dead"]==OLD and prior_value["new_reader_was_not_launched"]is True,"Recorded partial transition, no unchanged actor retry")
    progress,progress_pin=ctx.json(B/"progress/latest.json",131072,"stopped_Source41_progress")
    need(progress["actual_optimizer_updates"]==259800 and progress["source"]["sha256"]==SHA_DRIVER and progress["optimizer_poisoned"]is False and progress["schedule_poisoned"]is False,"Exact stopped final259800 science identity")
    need(progress["producer"]["pid"]==NATIVE["pid"]and progress["administrative_update_stop"]is None,"Unchanged original native binding")
    lease,lease_pin=ctx.json(CLOSED_LEASE["path"],CLOSED_LEASE["bytes"],"closed_lease")
    need(lease_pin==CLOSED_LEASE,"Exact ROOT current closed-lease bytes")
    lease_selected=dict(status="production_resident_RAM_process_closed",running=False,actual_GPU_running=False,actual_training_optimizer_running=False,actual_owned_worker_pid=None,committed_updates=259800,current_run_folder=str(B.resolve()),run_namespace=str(B.resolve()))
    need(all(lease.get(k)==v for k,v in lease_selected.items()),"Selected current closed flags and committed259800; other lease fields STALE/unused")
    journal_stat=stat_only(value["journal_path"])
    need(journal_stat==value["journal_STAT"]and journal_stat["bytes"]==519927641,"Unchanged final journal STAT only; no content read")
    # Recheck the recorded old pair without opening or controlling old handles.
    for old in OLD:exact_dead(old)
    before=storage_stats();need(before[0]["exists"]and before[1]["exists"],"Original STATE/DB must exist")
    candidate=ctx.read(CANDIDATE,28141,"candidate_Source")
    need(pin(CANDIDATE,candidate)==value["candidate"],"Candidate verified BEFORE any old-pair termination")
    old_handles=[];old_results=[];new_handles=[];new_ids=[];new_cleanup=[];new_descendants=[];new_descendant_races=[];launch=None;error=None;passed=False;close_results=[]
    try:
        # No old-process control remains in this genuinely new post-stop phase.
        for old in OLD:exact_dead(old)
        need(not listener_fence(value,25012,True)["reader_present"],"Port remains free; approved LAN proxy preserved")
        write(BODY_OUT/"admission_BEFORE_ANY_control.json",dict(scope=scope_pin,prior_partial_stop_ROOT=prior,progress=progress_pin,journal_STAT=journal_stat,storage_STAT=before,listener_roles=pre_control_listeners,closed_lease=lease_pin,selected_lease_fields=lease_selected,old_pair_exact_dead=OLD,no_old_process_control=True,fresh=ctx.guard(),native_full_reserve_remains_charged=True))
        installed=ctx.read(TARGET,28141,"installed_Reader8_Source")
        need(installed==candidate,"Exact installed candidate; no Source overwrite")
        target_pin=pin(TARGET,installed)
        stdout=BODY_OUT/"new_reader_stdout.log";stderr=BODY_OUT/"new_reader_stderr.log";ctx.logs.extend([stdout,stderr])
        launched=time.time()
        with stdout.open("xb")as out,stderr.open("xb")as err:
            launch=subprocess.Popen(value["reader_argv"],cwd=str(P),env=os.environ.copy(),stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        launcher=observation(psutil.Process(launch.pid));new_ids.append(launcher);new_handles.append(OwnedHandle(launcher))
        publish_once(BODY_OUT/"new_reader_actual_Popen_identity.json",dict(actual_Popen_pid=launch.pid,launcher=launcher,argv=value["reader_argv"],source=target_pin,launch_time=launched))
        deadline=time.monotonic()+LIMITS["readiness_seconds"];open_errors=[];probes=0;ready=None
        while time.monotonic()<deadline:
            ctx.guard();need(launch.poll()is None,"Owned reader launcher must remain alive")
            probes+=1;need(probes<=1024,"Bounded readiness observations")
            stream=None
            try:stream=(T/"current_status.json").open("rb")
            except PermissionError as exc:
                winerror=getattr(exc,"winerror",None)
                if not(os.name=="nt"and(winerror in(5,32,33)or(winerror is None and exc.errno==13))):raise
                need(len(open_errors)<256,"Bounded Windows OPEN-only observation")
                open_errors.append(dict(time=time.time(),errno=exc.errno,winerror=winerror,filename=exc.filename))
            if stream is not None:
                with stream:raw=stream.read(131073)
                need(len(raw)<=131072,"Bounded mutable STATUS observation")
                observed=json.loads(raw)
                if observed.get("updated_unix",0)>=launched and observed.get("create_time",0)>=launched and observed.get("source_sha256")==SHA_READER8 and observed.get("reader_ready")is True and observed.get("mirror_thread_alive")is True and observed.get("HTTP_ready")is True and observed.get("error")is None and observed.get("journal_path")==str((B/"journals/commits_000029.jsonl").resolve()) and 214971<=observed.get("mirrored_update",-1)<=259800 and 431862444<=observed.get("journal_offset",-1)<=519927641:
                    ready=observed;break
            time.sleep(.1)
        need(ready is not None,"Original60s readiness/catchup deadline")
        reader=observation(psutil.Process(ready["pid"]))
        need(reader["ctime"]==ready["create_time"]and(reader==launcher or reader["ppid"]==launcher["pid"]),"Reader identity from STATUS, then exact same launcher OR direct owned child")
        need(ready["updated_unix"]>=launched and Path(ready["source"]).resolve()==TARGET.resolve()and Path(ready["database"]).resolve()==(T/"database/parakeet-turkish.db").resolve(),"Fresh exact Reader8 Source/database")
        need(Path(ready["journal_path"]).resolve()==(B/"journals/commits_000029.jsonl").resolve(),"Actual current journal startup; full catchup remains a separate pending gate")
        if reader!=launcher:
            need(reader["pid"]!=launcher["pid"],"No conflicting same-PID identity")
            new_ids.append(reader);new_handles.append(OwnedHandle(reader))
        descendants=exact_process(launcher).children(recursive=True)
        need(len(descendants)<=LIMITS["descendants"],"Bounded actual new owned descendants; helpers allowed")
        for member in descendants:
            partial=dict(pid=member.pid)
            try:partial["ctime"]=member.create_time();partial["ppid"]=member.ppid()
            except psutil.NoSuchProcess:
                need(not psutil.pid_exists(member.pid),"Live/reused new descendant PID")
                need(len(new_descendant_races)<32,"Bounded actual new descendant races")
                new_descendant_races.append(partial);continue
            new_descendants.append(partial)
        need(reader==launcher or reader in new_descendants,"STATUS reader role is exact launcher or observed direct owned child; no order inference")
        healthy_listener_roles=listener_fence(value,reader["pid"])
        ctx.guard()
        with urllib.request.urlopen(value["health_URL"],timeout=LIMITS["HTTP_seconds"])as response:
            need(response.status==200,"Actual /version HTTP response")
            payload=response.read(65537);need(len(payload)<=65536,"Bounded /version response")
        version=json.loads(payload);need(isinstance(version,dict)and "error"not in version,"Actual /version error-field fence")
        terminal,terminal_pin=ctx.json(T/"current_status.json",131072,"healthy_Reader8_terminal_status")
        need(terminal["pid"]==reader["pid"]and terminal["create_time"]==reader["ctime"]and terminal["source_sha256"]==SHA_READER8 and terminal["mirror_thread_alive"]is True and terminal["HTTP_ready"]is True and terminal["reader_ready"]is True and terminal["error"]is None,"Unsuppressed fresh terminal health")
        need(214971<=terminal["mirrored_update"]<=259800 and 431862444<=terminal["journal_offset"]<=519927641,"Unsuppressed safe committed catchup frontier")
        need(Path(terminal["journal_path"]).resolve()==(B/"journals/commits_000029.jsonl").resolve(),"Same actual J29 during catchup")
        full_catchup=(terminal["mirrored_update"]==259800 and terminal["journal_offset"]==519927641 and terminal.get("last_verified_checkpoint_update")==259800)
        for item in new_ids:exact_process(item)
        need(stat_only(value["journal_path"])==journal_stat,"Actor must not change journal")
        write(BODY_OUT/"new_reader_startup_and_pending_catchup.json",dict(terminal_status=terminal_pin,version_response=version,reader=reader,launcher=launcher,listener_roles=healthy_listener_roles,new_owned_descendants=new_descendants,new_descendant_sample_races=new_descendant_races,actual_Popen_poll=launch.poll(),readiness_probes=probes,OPEN_errors=open_errors,source=target_pin,all_original_resource_reserves_unchanged=True,healthy_reader_retained_for_independent_ROOT_acceptance=True,not_production_restart=True,startup_only=True,full_current_journal_catchup_observed=full_catchup,full_catchup_Root_acceptance_PENDING=True,production_and_CPU_grants=False))
        passed=True
    except BaseException as exc:error=type(exc).__name__+": "+str(exc)
    finally:
        if not passed:
            for handle in reversed(new_handles):
                try:new_cleanup.append(handle.stop(ctx,15))
                except BaseException as exc:new_cleanup.append(dict(identity=handle.expected,error=type(exc).__name__+": "+str(exc),OS_or_method_closure="UNKNOWN"))
        for label,handles in (("new",new_handles),("old",old_handles)):
            for h in reversed(handles):
                try:close_results.append(dict(group=label,identity=h.expected,GetExitCodeProcess_before_CloseHandle=h.code(),CloseHandle=h.close()))
                except BaseException as exc:close_results.append(dict(group=label,identity=h.expected,error=repr(exc)))
        result=dict(kind="Reader8_replacement_body_result_v1",passed=passed,error=error,old_outcomes=old_results,new_recorded_identities=new_ids,new_owned_descendant_identities=new_descendants,new_descendant_sample_races=new_descendant_races,old_extra18592_not_controlled=True,whole_old_family_closed_NOT_claimed=True,new_cleanup=new_cleanup,owned_handle_closures=close_results,handle_constructor_events=HANDLE_EVENTS,actual_reader_launcher_Popen_poll=launch.poll()if launch else None,reader_retained=passed,startup_only=True,full_catchup_Root_acceptance_PENDING=True,production_restart=0,old_reader_fallback=0,manual_DB_STATE_history_or_journal_writes=0,final=ctx.sample())
        write(BODY_OUT/"result.json",result)
    need(passed,error or"Reader8 replacement failed")
    return result

def external(review_path):
    ctx=Context(EXT_OUT,LIMITS["external_seconds"],"external");ctx.env()
    value,scope_pin=scope_review(ctx,review_path)
    need(not BODY_OUT.exists()and TARGET.is_file(),"Fresh body; preserve exact installed Source from failed catchup attempt")
    raw_scope=Path(EXT_OUT/"ROOT_execution_scope_FIRST_RAW.bin")
    copy_pin=dict(path=str(raw_scope.resolve()),bytes=scope_pin["bytes"],sha256=scope_pin["sha256"])
    private=write(EXT_OUT/"private_body_binding.json",dict(external_identity=observation(ctx.process),scope_RAW_copy=str(raw_scope),scope_copy_pin=copy_pin))
    stdout=EXT_OUT/"body_stdout.log";stderr=EXT_OUT/"body_stderr.log";ctx.logs.extend([stdout,stderr,BODY_OUT/"new_reader_stdout.log",BODY_OUT/"new_reader_stderr.log"])
    process=None;known={};handles={};races=[];cleanup=[];primary=None;body_code=None;success=False;peak=0
    def race(pid,phase,partial=None):
        need(not psutil.pid_exists(pid),"Live/reused PID during owned process observation")
        need(len(races)<32,"Bounded actual process sampling races")
        races.append(dict(pid=pid,phase=phase,partial=partial,identity_complete=False if partial is None else all(k in partial for k in ("pid","ctime","ppid"))))
    def private_bytes(held,item,phase):
        try:return held.memory_info().private
        except psutil.NoSuchProcess:race(held.pid,phase,item);return 0
    def descendants(held,phase):
        try:return held.children(recursive=True)
        except psutil.NoSuchProcess:race(held.pid,phase);return []
    def capture():
        nonlocal peak
        candidates=descendants(ctx.process,"external_descendant_enumeration")
        need(len(candidates)<=LIMITS["descendants"],"Original descendant bound")
        for held in candidates:
            partial=dict(pid=held.pid)
            try:
                partial["ctime"]=held.create_time();partial["ppid"]=held.ppid()
            except psutil.NoSuchProcess:race(held.pid,"owned_member_identity",partial);continue
            key=(partial["pid"],partial["ctime"])
            need(key not in known or known[key]==partial,"Consistent exact owned-family identity")
            if key not in known:
                need(len(known)<LIMITS["known_identities"],"Original known-identity bound")
                try:handle=OwnedHandle(partial)
                except psutil.NoSuchProcess:race(held.pid,"owned_handle_capture",partial);continue
                known[key]=partial;handles[key]=handle
        total=private_bytes(ctx.process,observation(ctx.process),"external_private")
        for item in known.values():
            held=exact_process(item,missing=True)
            if held is not None:total+=private_bytes(held,item,"known_member_private")
        need(total<=LIMITS["family_private"],"Unchanged complete owned-family private cap")
        peak=max(peak,total)
        role_path=BODY_OUT/"new_reader_actual_Popen_identity.json"
        if role_path.exists():
            need(role_path.stat().st_size<=65536,"Bounded owned actor identity receipt")
            # This file has one exclusive fsync+replace publication, never a partial JSON.
            with role_path.open("rb")as f:role=json.loads(f.read(65537))
            launcher=role["launcher"];held=exact_process(launcher,missing=True)
            if held is not None:
                family=[held]+descendants(held,"new_reader_family_enumeration")
                private=0
                for member in family:private+=private_bytes(member,dict(pid=member.pid),"new_reader_family_private")
                need(private<=LIMITS["reader_private"],"Unchanged new reader1.5GB private cap")
        return total
    try:
        argv=[str(PYTHON.resolve()),"-X","utf8","-B",str(Path(__file__).resolve()),"--owned-body",private["path"]]
        with stdout.open("xb")as out,stderr.open("xb")as err:
            process=subprocess.Popen(argv,cwd=str(P),env=os.environ.copy(),stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        write(EXT_OUT/"actual_body_Popen.json",dict(actual_Popen_pid=process.pid,argv=argv,scope=scope_pin))
        while process.poll()is None:ctx.guard();capture();time.sleep(LIMITS["interval"])
        body_code=process.wait(timeout=1);ctx.guard();capture()
        need(body_code==0,"Actual body Popen exit must be0")
        result,_=ctx.json(BODY_OUT/"result.json",1048576,"body_result")
        need(result["passed"]is True and result["reader_retained"]is True and not result["new_cleanup"],"Actual healthy-reader handoff; no cleanup-as-pass")
        new_ids=result["new_recorded_identities"]
        need(1<=len(new_ids)<=2 and len({(item["pid"],item["ctime"])for item in new_ids})==len(new_ids),"Exactly the deduplicated actual new launcher/reader identities")
        for item in new_ids:exact_process(item)
        need(all(row.get("CloseHandle")is True for row in result["owned_handle_closures"]),"Every body-owned handle closed")
        success=True
    except BaseException as exc:primary=type(exc).__name__+": "+str(exc)
    finally:
        if not success:
            for key,handle in reversed(list(handles.items())):
                try:
                    held=exact_process(known[key],missing=True)
                    if held is not None:
                        write(EXT_OUT/("forced_cleanup_"+str(len(cleanup))+".json"),dict(identity=known[key],reason=primary))
                        cleanup.append(handle.stop(ctx,15))
                except BaseException as exc:cleanup.append(dict(identity=known[key],error=repr(exc),closure="UNKNOWN"))
        closed=[]
        for key,h in reversed(list(handles.items())):
            try:closed.append(dict(identity=known[key],CloseHandle=h.close()))
            except BaseException as exc:closed.append(dict(identity=known[key],error=repr(exc)))
        result=dict(kind="Reader8_replacement_external_result_v1",passed=success,error=primary,actual_body_Popen_exit=body_code,known_identities=list(known.values()),owned_family_sample_races=races,cleanup=cleanup,owned_handle_closures=closed,handle_constructor_events=HANDLE_EVENTS,healthy_new_reader_intentionally_retained=success,startup_only=True,full_catchup_Root_acceptance_PENDING=True,normal_reader_exit_NOT_claimed=True,native_and_old_unowned_processes_outside_cleanup=True,native_full_reserve_remains_charged=True,peak_sampled_family_private=peak,final=ctx.sample())
        write(EXT_OUT/"external_result.json",result)
    need(success,primary or"Replacement actor failed")
    need(all(x.get("CloseHandle")is True for x in closed),"External owned-handle closure")
    print(json.dumps(result,separators=(",",":")),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ROOT-review");group.add_argument("--owned-body")
    args=parser.parse_args()
    try:
        if args.owned_body:body(args.owned_body)
        else:external(args.ROOT_review)
    except BaseException as error:
        output=BODY_OUT if args.owned_body else EXT_OUT
        if output.is_dir():
            try:write(output/"failure_seal.json",dict(kind="Reader8_replacement_actor_failure_seal_v1",error=type(error).__name__+": "+str(error),control_or_cleanup_facts="SEE_RECORDED_RESULTS_OR_UNKNOWN",automatic_retry=False,time=time.time()))
            except BaseException as seal_error:error.add_note("Failure seal: "+repr(seal_error))
        raise
