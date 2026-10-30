"""Owned proof supervisor only. No DB/HTTP/model imports. Execution needs ROOT scope."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import psutil
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def save(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+".owned_tmp")
    with tmp.open("w",encoding="utf-8")as f:json.dump(x,f,allow_nan=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def who(p):return dict(pid=p.pid,ctime=p.create_time(),ppid=p.ppid())
def same(w):
    try:
        p=psutil.Process(w["pid"])
        return p.is_running()and p.create_time()==w["ctime"]and p.ppid()==w["ppid"]
    except psutil.NoSuchProcess:return False
def sample():
    class M(ctypes.Structure):
        _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(n,ctypes.c_ulonglong)for n in ("totalphysical","physical","commitlimit","commit","totalvirtual","virtual","extended")]
    m=M();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError()
    return dict(commit=m.commit,physical=m.physical,private=psutil.Process().memory_info().private,E_free=shutil.disk_usage("E:/gpu_training_scratchpad").free)
def execute(scope_path,fixture_path,mode):
    scope=read(scope_path)
    kind={"reader":"ROOT_Trackio7_real_API_mirror_integration_exact_ONE_CPU_scope_v1","CPU":"ROOT_CPU14_original_startup_firstpacket_exact_ONE_CPU_scope_v1"}[mode]
    assert scope["accepted"]is True and scope["reviewer"]=="/root"and scope["kind"]==kind
    root=Path(scope["external_output_directory"]).resolve()
    assert root.is_relative_to(HERE)and root!=HERE and not root.exists();root.mkdir()
    limits=scope["resources"]
    body_limit=limits["proof_body_wall_seconds"];total_limit=limits["external_wall_seconds"]
    assert 0<body_limit<=total_limit-20
    assert total_limit<=(180 if mode=="reader"else 420)
    required={"reader":{"prospective_commit":7452887424,"prospective_physical":4702887424,"runtime_commit":4934451968,"runtime_physical":4434451968,"family_private":2573741824},"CPU":{"prospective_commit":8437158784,"prospective_physical":5687158784,"runtime_commit":5918723328,"runtime_physical":5418723328,"family_private":8000000000}}[mode]
    assert limits["prospective_commit"]>=required["prospective_commit"]and limits["prospective_physical"]>=required["prospective_physical"]
    assert limits["runtime_commit"]>=required["runtime_commit"]and limits["runtime_physical"]>=required["runtime_physical"]
    assert limits["family_private_bytes"]<=required["family_private"]
    self_limit=limits["watchdog_private_bytes"];assert self_limit<=536870912
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    initial=sample();entry=who(psutil.Process());launcher=who(psutil.Process().parent())
    save(root/"initial_BEFORE_full_predicate.json",dict(sample=initial,entry=entry,launcher=launcher,limits=limits))
    assert initial["commit"]>=limits["prospective_commit"]and initial["physical"]>=limits["prospective_physical"]
    assert initial["private"]<=self_limit and initial["E_free"]>=100000000000
    assert os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
    assert all(os.environ.get(k)=="1"for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"))
    assert Path(os.environ["TORCH_EXTENSIONS_DIR"]).resolve()==Path(r"E:/gpu_training_scratchpad/cache/torch_extensions").resolve()
    for item in scope["input_pins"]:
        raw=Path(item["path"]).read_bytes()
        assert len(raw)==item["bytes"]and hashlib.sha256(raw).hexdigest()==item["sha256"]
    assert Path(scope["fixture_source"]["path"]).resolve()==Path(fixture_path).resolve()
    start=time.monotonic();known={};races=[];cleanup=[];guard_error=None;peak=0;last=initial;code=None
    stdout=(root/"fixture_stdout.log").open("xb");stderr=(root/"fixture_stderr.log").open("xb")
    argv=[sys.executable,"-X","utf8","-B",str(fixture_path),"--ROOT-review",str(scope_path),"--proof-body"]
    child=None
    def capture():
        nonlocal peak,last
        last=sample();family=last["private"]
        members_by_pid={}
        seeds=[]
        if child is not None:
            try:
                p=psutil.Process(child.pid)
                assert p.create_time()==known[child.pid]["ctime"]
                seeds.append(p)
            except psutil.NoSuchProcess:pass
        for identity in list(known.values()):
            if same(identity):seeds.append(psutil.Process(identity["pid"]))
        for seed in seeds:
            try:
                for p in [seed]+seed.children(recursive=True):members_by_pid[p.pid]=p
            except psutil.NoSuchProcess:
                if len(races)<64:races.append(dict(identity=known.get(seed.pid,"UNRECORDED"),phase="owned_seed_enumeration"))
        members=list(members_by_pid.values())
        if len(members)>16:raise RuntimeError("Owned proof descendant bound16")
        for p in members:
            w=known.get(p.pid)
            try:
                actual=who(p)
                if w is not None and w!=actual:raise RuntimeError("Owned proof identity changed")
                known[p.pid]=actual
                observed_private=p.memory_info().private
                arguments=p.cmdline()
                if "--server-child"in arguments:
                    if observed_private>1500000000:raise RuntimeError("Original fixture-reader private1.5GB cap")
                elif "--worker"in arguments:
                    if observed_private>8000000000:raise RuntimeError("Original CPU private8GB cap")
                elif observed_private>536870912:
                    raise RuntimeError("Original controller/unclassified private512MiB cap")
                family+=observed_private
            except psutil.NoSuchProcess:
                # Only a child enumerated under the exact OWNED proof root.
                if len(races)<64:races.append(dict(pid=p.pid,identity=w if w is not None else "UNRECORDED",phase="owned_family_sample"))
        if len(known)>32:raise RuntimeError("Owned lifetime identity bound32")
        peak=max(peak,family)
        if last["commit"]<limits["runtime_commit"]or last["physical"]<limits["runtime_physical"]:raise RuntimeError("Original runtime headroom guard")
        if last["private"]>self_limit or family>limits["family_private_bytes"]:raise RuntimeError("Original full family/private cap")
        if last["E_free"]<100000000000:raise RuntimeError("Original E100GB guard")
        if any(p.stat().st_size>16777216 for p in (root/"fixture_stdout.log",root/"fixture_stderr.log")):raise RuntimeError("Owned log16MiB bound")
        if time.monotonic()-start>=body_limit:raise TimeoutError("Independent proof wall bound")
    try:
        child=subprocess.Popen(argv,stdout=stdout,stderr=stderr,cwd=HERE,creationflags=subprocess.CREATE_NO_WINDOW)
        identity=who(psutil.Process(child.pid));known[child.pid]=identity
        save(root/"actual_Popen_identity.json",dict(identity=identity,argv=argv))
        while child.poll()is None:
            capture();time.sleep(.1)
        code=child.wait()
        # A failing fixture is preserved/cleaned immediately, not recast as a timeout.
        if code==0:
            while any(same(w)for w in known.values()):
                capture();time.sleep(.1)
    except BaseException as error:
        guard_error=dict(type=type(error).__name__,message=str(error)[:1000])
        raise
    finally:
        # Parent stopped first; only frozen exact owned identities may be terminated.
        targets=sorted(known.values(),key=lambda w:w["ctime"])
        for w in targets:
            if same(w):
                try:
                    psutil.Process(w["pid"]).terminate()
                    cleanup.append(dict(identity=w,action="terminate_exact_owned",method_close="UNKNOWN"))
                except psutil.NoSuchProcess:pass
                except BaseException as error:cleanup.append(dict(identity=w,error=type(error).__name__,exact_death="UNKNOWN"))
        deadline=start+total_limit
        while any(same(w)for w in known.values())and time.monotonic()<deadline:time.sleep(.05)
        if child is not None and child.poll()is not None:code=child.wait()
        deaths=[dict(identity=w,exact_dead=not same(w))for w in known.values()]
        stdout.close();stderr.close()
        save(root/"external_result.json",dict(entry=entry,launcher=launcher,initial=initial,last_sample=last,actual_Popen_code=code,guard_error=guard_error,cleanup=cleanup,known_deaths=deaths,races=races,peak_sampled_family_private=peak,wall_seconds=time.monotonic()-start,full_family_guard_sampled_every_seconds=.1,hard_OS_allocation_limit_claim=False,all_exact_known_dead=all(d["exact_dead"]for d in deaths),production_touched=False))

    if code!=0:raise RuntimeError("Owned fixture exited nonzero: "+str(code))
    return code
