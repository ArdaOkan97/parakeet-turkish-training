"""INACTIVE exact failed-reader6 -> tested reader7 replacement.
SOURCE only until a distinct exact ROOT scope; no training/session control.
"""
import argparse,ctypes,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
import psutil
P=Path(r"C:/Users/ardao/Documents/Codex/2026-10-10/i-x20")
HERE=Path(__file__).resolve().parent
ROOT=Path(r"E:/gpu_training_scratchpad")
FAILED=[dict(pid=33388,ctime=1792855546.8493547,ppid=38980),dict(pid=38980,ctime=1792855546.824607,ppid=26648)]
KNOWN_STOPPED_NATIVE=[{"pid":34788,"ctime":1792842518.865024,"ppid":32332},{"pid":32332,"ctime":1792842518.85846,"ppid":4964},{"pid":4964,"ctime":1792842518.480302,"ppid":6664},{"pid":6664,"ctime":1792842518.4723651,"ppid":37480},{"pid":23496,"ctime":1792848390.67016,"ppid":39100},{"pid":39100,"ctime":1792848390.6293392,"ppid":34788},{"pid":33668,"ctime":1792855655.0849347,"ppid":34788}]
SOURCE=HERE/"serve_BPE9223_Trackio_live_v7_INACTIVE_v5.py"
DESTINATION=P/"work/research/serve_BPE9223_Trackio_live_v7.py"
PYTHON=P/"work/monitor-env/Scripts/python.exe"
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def pin(p):
    p=Path(p);b=p.read_bytes()
    return dict(path=str(p.resolve()),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def save(p,obj):
    raw=(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2)+"\n").encode()
    with Path(p).open("xb")as f:f.write(raw);f.flush();os.fsync(f.fileno())
def who(p):return dict(pid=p.pid,ctime=p.create_time(),ppid=p.ppid())
def exact(w):
    p=psutil.Process(w["pid"])
    assert abs(p.create_time()-w["ctime"])<.001 and p.ppid()==w["ppid"],"Exact identity/ancestry; never retarget"
    return p
def dead(w):
    try:return abs(psutil.Process(w["pid"]).create_time()-w["ctime"])>=.001
    except psutil.NoSuchProcess:return True
def sample():
    class M(ctypes.Structure):
        _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(n,ctypes.c_ulonglong)for n in ("totalphysical","physical","commitlimit","commit","totalvirtual","virtual","extended")]
    m=M();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError()
    return dict(commit=m.commit,physical=m.physical,private=psutil.Process().memory_info().private,E_free=shutil.disk_usage(ROOT).free)
def listening():
    return [dict(pid=c.pid,address=list(c.laddr))for c in psutil.net_connections(kind="tcp")if c.status==psutil.CONN_LISTEN and c.laddr.port==7860]
def run(scope_path):
    s=read(scope_path)
    assert s["accepted"]is True and s["reviewer"]=="/root"and s["kind"]=="ROOT_reader7_replace_failed_reader6_exact_ONE_scope_v1"
    assert s["failed_reader_pair"]==FAILED and pin(__file__)==s["actor_source"]
    assert Path(os.environ["GPU_TRAINING_STORAGE_ROOT"]).resolve()==ROOT.resolve()
    assert Path(os.environ["TORCH_EXTENSIONS_DIR"]).resolve()==(ROOT/"cache/torch_extensions").resolve()
    assert os.environ["CUDA_VISIBLE_DEVICES"]=="-1"
    assert all(os.environ[k]=="1"for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"))
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    out=Path(s["output_directory"]).resolve()
    assert out.is_relative_to(HERE)and out!=HERE and not out.exists();out.mkdir()
    began=time.monotonic();limits=s["resources"]
    # Proposal remains ABOVE original source39/native72/metadata512+reader256.
    assert limits["prospective_commit"]>=5491513984 and limits["prospective_physical"]>=2741513984
    assert limits["runtime_commit"]>=2973078528 and limits["runtime_physical"]>=2473078528
    assert limits["self_private_bytes"]==536870912 and limits["reader_private_bytes"]==1500000000
    assert limits["wall_seconds"]==180 and limits["reader_stop_wait_seconds"]==15
    first=sample();save(out/"initial_BEFORE_predicate.json",dict(sample=first,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),resources=limits))
    assert first["commit"]>=limits["prospective_commit"]and first["physical"]>=limits["prospective_physical"]
    result=dict(old_targets=[],source_copied=False,new_launch_attempted=False,new_launch_identity=None,primary=None,training_control=False,inner_old_OS_codes="UNKNOWN unless separately observed")
    handles=[]
    def guard():
        m=sample()
        assert m["commit"]>=limits["runtime_commit"]and m["physical"]>=limits["runtime_physical"]
        assert m["private"]<=limits["self_private_bytes"]and m["E_free"]>=100000000000
        assert time.monotonic()-began<180
        for q in (result.get("new_launch_identity"),result.get("new_reader_identity")):
            if q is not None:
                try:p=exact(q)
                except psutil.NoSuchProcess:continue
                assert p.memory_info().private<=1500000000
        return m
    def wait_until(test,maximum):
        until=min(began+180,time.monotonic()+maximum)
        while time.monotonic()<until:
            guard()
            value=test()
            if value:return value
            time.sleep(.25)
        raise TimeoutError("Bounded exact reader replacement gate")
    try:
        guard()
        for p in s["input_pins"]:assert pin(p["path"])==p
        assert pin(SOURCE)==s["reader_template"]
        assert s["reader_template"]["sha256"]=="d7bb6494d197cfe4b4f7bc8c5af1f4b720be199e2fb8d3077cb8af45e450da22"
        assert Path(s["destination"]).resolve()==DESTINATION.resolve()and not DESTINATION.exists()
        assert Path(s["reader_python"]["path"]).resolve()==PYTHON.resolve()and pin(PYTHON)==s["reader_python"]
        db=Path(s["database_STAT"]["path"]);state=Path(s["state_STAT"]["path"])
        for path,expected in ((db,s["database_STAT"]),(state,s["state_STAT"])):
            stat=path.stat();assert path.is_file()and stat.st_size==expected["bytes"]and stat.st_mtime_ns==expected["mtime_ns"]
        owner=read(s["current_owner_path"])
        assert owner["pid"]==FAILED[0]["pid"]and abs(owner["create_time"]-FAILED[0]["ctime"])<.001
        assert owner["source_sha256"]==s["old_reader_source"]["sha256"]
        for q in FAILED:exact(q)
        assert all(x["pid"]in {q["pid"]for q in FAILED}for x in listening())
        assert all(q in s["accepted_stopped_native_identities"]for q in KNOWN_STOPPED_NATIVE)
        assert all(dead(q)for q in s["accepted_stopped_native_identities"])
        authority=s["API_both_shapes_ROOT"]
        assert authority["sha256"]=="970f937ef2904938980742b8a132e67de758bbd7ed927f4dd5997ad29d0aaa2f"
        assert pin(authority["path"])==authority
        API_acceptance=read(authority["path"])
        assert API_acceptance["kind"]=="ROOT_reader5_helper4_projector2_both_original500000_real_API_shapes_acceptance_v1"
        assert API_acceptance["reviewer"]=="/root"and API_acceptance["accepted_both_shapes"]is True and API_acceptance["accepted_all_original_lifetime"]is True
        assert API_acceptance["candidate_reader"]==s["reader_template"]
        assert API_acceptance["projection_source"]==pin(HERE/"incremental_scalar_dashboard_projection_v2.py")
        # One exact body stop, then its OWN exact launcher, each observed separately.
        for q in FAILED:
            if dead(q):
                result["old_targets"].append(dict(identity=q,exact_dead_before_turn=True,observed_wait_code="UNKNOWN"));continue
            p=exact(q);p.terminate();code=p.wait(timeout=15)
            assert dead(q);result["old_targets"].append(dict(identity=q,exact_dead=True,observed_psutil_wait_return=code,independent_original_OS_exit="UNKNOWN"))
        wait_until(lambda:not listening(),15)
        guard();assert all(dead(q)for q in FAILED)and not DESTINATION.exists()
        # Later exact ROOT scope alone authorizes this fresh P SOURCE copy.
        raw=SOURCE.read_bytes()
        with DESTINATION.open("xb")as f:f.write(raw);f.flush();os.fsync(f.fileno())
        result["source_copied"]=True;assert pin(DESTINATION)["sha256"]==s["reader_template"]["sha256"]
        assert len(raw)==s["reader_template"]["bytes"]
        stdout=(out/"reader_stdout.log").open("xb");stderr=(out/"reader_stderr.log").open("xb");handles.extend((stdout,stderr))
        env=os.environ.copy()
        command=[str(PYTHON),"-X","utf8","-B",str(DESTINATION)]
        assert command==s["exact_reader_command"]
        guard();assert all(dead(q)for q in FAILED)and not listening()
        result["new_launch_attempted"]=True
        child=subprocess.Popen(command,cwd=P,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        result["new_launch_identity"]=who(psutil.Process(child.pid))
        result["exact_command"]=command;result["observed_child_Popen_poll"]=child.poll()
        def accepted_current():
            if child.poll()is not None:raise RuntimeError("New reader launcher exited before genuine ready")
            try:o=read(s["current_owner_path"]);status=read(s["current_status_path"])
            except (FileNotFoundError,json.JSONDecodeError,PermissionError):return None
            if o.get("source_sha256")!=s["reader_template"]["sha256"]or o.get("pid")==FAILED[0]["pid"]:return None
            q=who(psutil.Process(o["pid"]));assert q["ppid"]==child.pid and abs(q["ctime"]-o["create_time"])<.001
            result["new_reader_identity"]=q
            assert o["database"]==str(db)and Path(o["source"]).resolve()==DESTINATION.resolve()
            if status.get("pid")!=q["pid"]or status.get("create_time")!=q["ctime"]:return None
            if status.get("error")is not None:raise RuntimeError("New local reader error: "+str(status["error"])[:500])
            if not(status.get("HTTP_ready")is True and status.get("mirror_thread_alive")is True and status.get("reader_ready")is True):return None
            if status.get("pending_batch")is not None:return None
            if status.get("journal_path")!=s["expected_saved_journal"]["path"]or status.get("journal_offset")!=426026870:return None
            if status.get("last_verified_checkpoint_update",0)<212000:return None
            assert all(x["pid"]==q["pid"]for x in listening())and listening()
            return dict(owner=o,status=status)
        ready=wait_until(accepted_current,90)
        # The ORIGINAL reader's own started-gate + real API/committed mirror health
        # is authoritative, not a GET/version against an unrelated existing port.
        result["actual_current_acceptance"]=ready
        result["HTTP_ready"]=result["reader_ready"]=result["mirror_thread_alive"]=True
        result["old_exact_pair_dead"]=all(dead(q)for q in FAILED)
        result["observed_child_Popen_poll"]=child.poll()
        result["passed"]=True
    except BaseException as error:
        result["primary"]=dict(type=type(error).__name__,message=str(error)[:2000]);result["passed"]=False
        raise
    finally:
        for h in reversed(handles):h.close()
        result["wall"]=time.monotonic()-began
        try:result["final_sample"]=sample()
        except BaseException as final_error:result["final_sample"]=dict(UNKNOWN=type(final_error).__name__)
        result["no_retry_or_retarget"]=True
        result["new_reader_lifetime"]="If launched, remains owned normal reader; do not kill/relaunch after controller failure"
        save(out/"result.json",result)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--ROOT-review",required=True)
    run(Path(p.parse_args().ROOT_review).resolve())
