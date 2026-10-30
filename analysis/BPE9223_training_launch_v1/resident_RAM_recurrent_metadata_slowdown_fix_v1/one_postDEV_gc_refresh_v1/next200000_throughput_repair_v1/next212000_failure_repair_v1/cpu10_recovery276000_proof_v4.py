"""INACTIVE new CPU10 final259800 continuation-path full20x16 validation.
Original CPU14/adapter/runner transport and all scientific rows are unchanged.
Separate ROOT accepted scope is required; no GPU/native/production control.
"""
import argparse,ast,copy,ctypes,hashlib,json,os,shutil,subprocess,sys,time,weakref
from pathlib import Path
import psutil
P=Path("C:/Users/ardao/Documents/Codex/2026-10-10/i-x20")
E=Path("E:/gpu_training_scratchpad")
X=Path(__file__).resolve().parent
PYTHON=E/"scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe"
BODY_OUT=X/"cpu10_recovery276000_actual01"
EXT_OUT=X/"cpu10_recovery276000_external_actual01"
LIMITS=dict(prospective_commit=8066304384,prospective_physical=5316304384,runtime_commit=5547868928,runtime_physical=5047868928,self_private=536870912,reader_private=1500000000,family_private=2573741824,body_seconds=160,external_seconds=180,HTTP_seconds=20,readiness_seconds=60,interval=.1,E_free=100000000000,log_bytes=16777216,descendants=16,known_identities=32)
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
        self.access_scope=write(out/"scope_BEFORE_any_input_or_target_stat.json",dict(kind="ROOT_CPU10_full20x16_access_scope_v1",initial=self.initial,fresh=self.fresh,actual_source_is_DRAFT_until_ROOT_execution_scope_verified=True,no_DB_STATE_history_or_journal_content_reads=True))
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
        ctime=float(ticks-116444736000000000)/10000000
        self.closed=False
        self.binding=dict(identity=self.expected,GetProcessId=self.k.GetProcessId(self.handle),creation_FILETIME=ticks,creation_time=ctime)
        try:
            need(self.binding["GetProcessId"]==self.expected["pid"]and ctime==self.expected["ctime"],"Strict creation-time-bound owned handle")
            exact_process(self.expected)
        except BaseException:
            HANDLE_EVENTS.append(dict(identity=self.expected,phase="handle_constructor_failure",CloseHandle=self.close()))
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


def checked_scope(ctx,path):
    scope,scope_pin=ctx.json(path,131072,"CPU10_ROOT_scope")
    need(scope["accepted"]is True and scope["reviewer"]=="/root"and scope["owner"]=="/root","Root CPU10 scope")
    need(scope["kind"]=="ROOT_CPU14_startup_adapter_firstpacket_exact_ONE_CPU_scope_v1"and scope["execution_phase"]=="NEW276000_recovery43_full20x16_transport","Exact new CPU10 phase")
    need(scope["resources"]==LIMITS and scope["body_output"]==str(BODY_OUT.resolve())and scope["external_output"]==str(EXT_OUT.resolve()),"Unchanged complete reservations and fresh outputs")
    source=ctx.read(__file__,65536,"CPU10_proof_Source")
    need(pin(__file__,source)==scope["proof_source"],"Whole new proof Source")
    for old in scope["required_exact_dead"]:exact_dead(old)
    need(scope["native_full_reserve_remains_charged"]is True and scope["no_production_control"]is True,"Native absent reserve remains charged")
    if ctx.out==EXT_OUT:
        need(scope["public_entry"]==[str(PYTHON.resolve()),"-X","utf8","-B",str(Path(__file__).resolve()),"--ROOT-review",str(Path(path).resolve())],"Exact single CPU10 public entry")
    return scope,scope_pin

def bound_json(ctx,item,label,maximum=8388608):
    raw=ctx.read(item["path"],maximum,label)
    need(pin(item["path"],raw)==item,"Pinned new input")
    return json.loads(raw),raw

def cpu_body(private_path):
    ctx=Context(BODY_OUT,LIMITS["body_seconds"],"CPU10_body");ctx.env()
    private,_=ctx.json(private_path,65536,"external_binding")
    need(any(observation(q)==private["external_identity"]for q in ctx.process.parents()),"Exact external-owned body")
    scope,scope_pin=checked_scope(ctx,private["scope_copy"])
    need(scope_pin==private["scope_copy_pin"],"Exact handed-off Root scope")
    cfg,cfg_raw=bound_json(ctx,scope["configuration_Root_copy"],"incoming_science_Root_copy")
    plan,plan_raw=bound_json(ctx,scope["plan_Root_copy"],"new_final20_plan")
    reference,_=bound_json(ctx,scope["CPU20_expected_Root_copy"],"new_final20_original_reference")
    need(len(plan["updates"])==len(reference)==20 and [r["metadata"]["global_update"]for r in reference]==list(range(276181,276201)),"New genuine last20 full16 reference")
    canonical=lambda value:json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()
    plan_sha=hashlib.sha256(canonical(plan)).hexdigest()
    need(plan_sha==scope["canonical_plan_sha256"],"Original canonical plan")
    need(all(len(e["rows"])==len(e["contexts"])==len(e["draws"])==16 and len(e["TR_draws"])==14 and len(e["replay_rows"])==2 for e in plan["updates"]),"Every full14TR2replay packet")
    incoming=write_raw(BODY_OUT/"incoming_stream_configuration.json",cfg_raw)
    plan_pin=write_raw(BODY_OUT/"new_original_last20_plan.json",plan_raw)
    driver_raw=ctx.read(scope["driver_source"]["path"],262144,"new_driver_Source")
    need(pin(scope["driver_source"]["path"],driver_raw)==scope["driver_source"],"Actual new driver Source pin")
    tree=ast.parse(driver_raw.decode())
    configure=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=="configure_startup_receipts"]
    need(len(configure)==1 and len(configure[0].args)==2 and not configure[0].keywords and ast.unparse(configure[0].args[1])=="record(a.stream_configuration)","Actual driver configures INCOMING wire pin")
    constructors=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=="ResidentFeatureProducer"]
    need(len(constructors)==3 and all(ast.unparse(n.args[0])=="record(a.stream_configuration)"for n in constructors),"All actual driver producer routes use INCOMING wire config")
    runtime=copy.deepcopy(cfg)
    runtime.update(worker_driver_source=scope["driver_source"],parent_driver_source=scope["driver_source"],resident_RAM_stream_configuration=incoming,resident_branch_namespace=str((E/"runs/BPE9223_resident_RAM_20261018_01").resolve()),checkpoint_interval_updates=200,source_checkpoint=scope["final_pointer"],source_configuration=incoming)
    runtime_pin=write(BODY_OUT/"distinct_parent_runtime_configuration.json",runtime)
    need(runtime_pin["path"]!=incoming["path"]and runtime_pin["sha256"]!=incoming["sha256"],"Genuine distinct parent runtime/worker configuration roles")
    policy=copy.deepcopy(cfg["original_CPU14_startup_reader7_recovery_policy"]["CPU_startup_policy"])
    need(policy["adapter_source"]==scope["CPU_startup_adapter_source"]and policy["runner_source"]==scope["CPU_startup_runner_source"]and policy["original_CPU14_source"]==scope["CPU14_source"],"Unchanged actual adapter/runner/CPU14")
    sys.path[:0]=[str(X),str(P/"outputs/parakeet_turkish"),str(P/"work/research")]
    # Original CPU worker uses this exact CPU-only NumPy runtime initializer.
    common_raw=ctx.read(scope["parent_numpy_runtime_source"]["path"],65536,"parent_original_NumPy_runtime_Source")
    need(pin(scope["parent_numpy_runtime_source"]["path"],common_raw)==scope["parent_numpy_runtime_source"],"Original NumPy runtime Source pin")
    import fastmss_BPE9223_cpu_common_v12 as parent_core
    parent_core.environment()
    parent_numpy,parent_torch=parent_core.numpy_runtime(frontend=False)
    need(parent_torch is None and "torch"not in sys.modules,"Parent transport NumPy only; no Torch/GPU import")
    need(Path(parent_numpy.__file__).resolve().is_relative_to(parent_core.TRAINING_SITE.resolve()),"Original numerical NumPy Source placement")
    parent_numpy_witness=dict(numpy_file=parent_numpy.__file__,numpy_version=parent_numpy.__version__,frontend=False,Torch_not_imported=True,original_runtime_source=scope["parent_numpy_runtime_source"])
    import CPU14_startup_launch_adapter_v15 as adapter
    adapter.configure_startup_receipts(policy,incoming,startup_scope_path=private["scope_copy"])
    stream=None;normal=None;rows_checked=0;packet_records=[];error=None;aborted=False;original_transport_error=None
    started=time.monotonic();ctx.logs.append(BODY_OUT/"CPU"/"CPU_stderr.log")
    try:
        stream=adapter.ResidentFeatureProducer(incoming,BODY_OUT/"CPU",queue_bytes=268435456,private_limit_bytes=8000000000,guard=ctx.guard)
        submitted=stream.submit(plan_pin);need(submitted==plan_sha,"Exact original plan actual submission")
        for i,expected in enumerate(reference):
            ctx.guard();packet=stream.next_update(submitted,i,timeout=300)
            metadata=packet["metadata"]
            need(packet["plan_sha256"]==plan_sha and packet["update_index"]==i and metadata["global_update"]==276181+i,"Original ordered fullpacket")
            actual_stable={k:v for k,v in metadata.items()if k not in("owner","producer")}
            expected_stable={k:v for k,v in expected["metadata"].items()if k not in("owner","producer")}
            need(metadata["global_update"]==plan["updates"][i]["global_update"]and metadata["phase"]==plan["updates"][i]["phase"],"Exact new plan update/phase; full array transport integrity follows")
            need(metadata["owner"]["plan_sha256"]==plan_sha and metadata["owner"]["source"]==scope["CPU14_source"],"Actual producer owner pin")
            need(len(packet["arrays"])==16 and len(metadata["rows"])==16,"Complete full16 arrays")
            refs=[];checks=[]
            for row_index,pair in enumerate(packet["arrays"]):
                for name in("input_features","attention_mask"):
                    array=pair[name];refs.append(weakref.ref(array))
                    expected_array=metadata["rows"][row_index][name]
                    need(str(array.dtype)==expected_array["dtype"]and list(array.shape)==expected_array["shape"]and hashlib.sha256(memoryview(array).cast("B")).hexdigest()==expected_array["array_sha256"],"Byteexact original numerical features/masks")
                    checks.append(dict(row=row_index,name=name,bytes=int(array.nbytes),sha256=expected_array["array_sha256"]))
                    del array
            rows_checked+=16
            token=packet["token"];packet_records.append(dict(update=metadata["global_update"],index=i,metadata_stable_sha256=hashlib.sha256(canonical(actual_stable)).hexdigest(),array_checks=checks))
            del packet,pair,metadata,actual_stable,expected_stable,expected_array,checks
            need(all(r()is None for r in refs),"Every payload reference retired BEFORE original release-credit")
            stream.release(token)
            del token,refs
        normal=stream.close(timeout=90)
        need(normal["completed_updates"]==20 and normal["actual_Popen_wait_code"]==0 and normal["transport_threads_joined"]is True and normal["pipe_handles_closed"]is True and normal["queue_capacity_bytes"]==268435456 and normal["maximum_live_receiver_packets"]==1 and normal["peak_live_receiver_packets"]<=1,"Actual complete CPU20 normal method closure")
        need(normal["packet_credit"]["all_owned_kernel_handles_closed"]is True and normal["packet_credit_owner"]["all_owned_kernel_handles_closed"]is True,"Both original owned credit handle sets closed")
        need(all(not x.is_alive()for x in(stream._sender,stream._receiver)),"Original transport thread methods joined")
    except BaseException as exc:
        error=type(exc).__name__+": "+str(exc)
        if stream is not None and stream.error is not None:
            original_transport_error=dict(type=type(stream.error).__name__,message=str(stream.error),exception_identity=id(stream.error))
        raise
    finally:
        if stream is not None and normal is None:
            stream.abort();aborted=True
        write(BODY_OUT/"CPU10_body_result.json",dict(kind="CPU10_final259800_adapter_full20x16_result_v1",passed=normal is not None and error is None,error=error,original_transport_error=original_transport_error,parent_original_NumPy_runtime=parent_numpy_witness,normal_report=normal,abnormal_abort_used=aborted,normal0_never_inferred_from_abort=True,rows_checked=rows_checked,updates_checked=len(packet_records),all_packet_checks=packet_records,incoming_stream_configuration=incoming,distinct_parent_runtime_configuration=runtime_pin,actual_driver_AST_routes_passed=True,actual_worker_wire_expected_incoming_configuration=incoming,adapter_source=scope["CPU_startup_adapter_source"],runner_source=scope["CPU_startup_runner_source"],CPU14_source=scope["CPU14_source"],driver_source=scope["driver_source"],full_native_update_or_training_control=0,no_GPU_model_or_authoritative_cursor_mutation=True,wall_seconds=time.monotonic()-started,final=ctx.sample()))

def external(review_path):
    ctx=Context(EXT_OUT,LIMITS["external_seconds"],"external");ctx.env()
    value,scope_pin=checked_scope(ctx,review_path)
    need(not BODY_OUT.exists(),"Fresh exact CPU10 body output required")
    raw_scope=Path(EXT_OUT/"CPU10_ROOT_scope_FIRST_RAW.bin")
    copy_pin=dict(path=str(raw_scope.resolve()),bytes=scope_pin["bytes"],sha256=scope_pin["sha256"])
    private=write(EXT_OUT/"private_body_binding.json",dict(external_identity=observation(ctx.process),scope_copy=str(raw_scope),scope_copy_pin=copy_pin))
    stdout=EXT_OUT/"body_stdout.log";stderr=EXT_OUT/"body_stderr.log";ctx.logs.extend([stdout,stderr,BODY_OUT/"CPU"/"CPU_stderr.log"])
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
        return total
    try:
        argv=[str(PYTHON.resolve()),"-X","utf8","-B",str(Path(__file__).resolve()),"--owned-body",private["path"]]
        with stdout.open("xb")as out,stderr.open("xb")as err:
            process=subprocess.Popen(argv,cwd=str(P),env=os.environ.copy(),stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        write(EXT_OUT/"actual_body_Popen.json",dict(actual_Popen_pid=process.pid,argv=argv,scope=scope_pin))
        while process.poll()is None:ctx.guard();capture();time.sleep(LIMITS["interval"])
        body_code=process.wait(timeout=1);ctx.guard();capture()
        need(body_code==0,"Actual body Popen exit must be0")
        result,_=ctx.json(BODY_OUT/"CPU10_body_result.json",1048576,"CPU10_body_result")
        need(result["passed"]is True and result["abnormal_abort_used"]is False and result["updates_checked"]==20 and result["rows_checked"]==320,"Actual full20x16 and normal closure")
        for item in known.values():exact_dead(item)
        need(result["normal_report"]["actual_Popen_wait_code"]==0 and result["normal_report"]["transport_threads_joined"]is True and result["normal_report"]["pipe_handles_closed"]is True,"Actual CPU method closure plus wait0")
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
        result=dict(kind="CPU10_final259800_external_result_v1",passed=success,error=primary,actual_body_Popen_exit=body_code,known_identities=list(known.values()),owned_family_sample_races=races,cleanup=cleanup,owned_handle_closures=closed,handle_constructor_events=HANDLE_EVENTS,all_recorded_owned_identities_observed_dead=success,normal0_never_inferred_from_cleanup=True,native_and_live_reader_outside_cleanup=True,native_full_reserve_remains_charged=True,peak_sampled_family_private=peak,final=ctx.sample())
        write(EXT_OUT/"external_result.json",result)
    need(success,primary or"CPU10 full20x16 validation failed")
    need(all(x.get("CloseHandle")is True for x in closed),"External owned-handle closure")
    print(json.dumps(result,separators=(",",":")),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ROOT-review");group.add_argument("--owned-body")
    args=parser.parse_args()
    try:
        if args.owned_body:cpu_body(args.owned_body)
        else:external(args.ROOT_review)
    except BaseException as error:
        output=BODY_OUT if args.owned_body else EXT_OUT
        if output.is_dir():
            try:write(output/"failure_seal.json",dict(kind="CPU10_final259800_failure_seal_v1",error=type(error).__name__+": "+str(error),normal0_NOT_inferred_from_abort_or_death=True,automatic_retry=False,time=time.time()))
            except BaseException as seal_error:error.add_note("Failure seal: "+repr(seal_error))
        raise
