"""INACTIVE independent bounds for typed SOURCE40 metadata fixture only."""
import argparse,ctypes,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
import psutil
HERE=Path(__file__).resolve().parent
E_PYTHON=Path(r"E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe").resolve()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def pin(p):
    p=Path(p);b=p.read_bytes()
    return dict(path=str(p.resolve()),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def save(p,v):
    with Path(p).open("x",encoding="utf-8")as f:json.dump(v,f,allow_nan=False,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
def who(p):return dict(pid=p.pid,ctime=p.create_time(),ppid=p.ppid())
def sample():
    class M(ctypes.Structure):
        _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(n,ctypes.c_ulonglong)for n in ("totalphysical","physical","commitlimit","commit","totalvirtual","virtual","extended")]
    m=M();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError()
    return dict(commit=m.commit,physical=m.physical,private=psutil.Process().memory_info().private,E_free=shutil.disk_usage("E:/gpu_training_scratchpad").free)
def main(scope_path):
    s=read(scope_path)
    assert s["accepted"]is True and s["reviewer"]=="/root"and s["kind"]=="ROOT_source40_saved212000_entire_typed_metadata_exact_ONE_CPU_scope_v1"
    assert pin(__file__)==s["external_source"]and Path(s["fixture_python"]["path"]).resolve()==E_PYTHON
    assert pin(E_PYTHON)==s["fixture_python"]and os.environ["CUDA_VISIBLE_DEVICES"]=="-1"
    assert all(os.environ[k]=="1"for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"))
    root=Path(s["external_output_directory"]).resolve();assert root.is_relative_to(HERE)and root!=HERE and not root.exists();root.mkdir()
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    limits=s["resources"]
    assert limits["prospective_commit"]>=6028384896 and limits["prospective_physical"]>=3278384896
    assert limits["runtime_commit"]>=3509949440 and limits["runtime_physical"]>=3009949440
    assert limits["fixture_wall_seconds"]==100 and limits["external_wall_seconds"]==120
    assert limits["self_private_bytes"]==limits["family_private_bytes"]==536870912
    started=time.monotonic();first=sample()
    save(root/"initial_BEFORE_full_predicate.json",dict(sample=first,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),resources=limits))
    assert first["commit"]>=limits["prospective_commit"]and first["physical"]>=limits["prospective_physical"]
    assert first["private"]<=536870912 and first["E_free"]>=100000000000
    for q in s["input_pins"]:assert pin(q["path"])==q
    fixture=Path(s["fixture_source"]["path"]).resolve();assert fixture.is_relative_to(HERE)and pin(fixture)==s["fixture_source"]
    known={};cleanup=[];error=None;child=None;peak=0;observations=0
    command=[str(E_PYTHON),"-X","utf8","-B",str(fixture),"--ROOT-review",str(scope_path)]
    try:
        with (root/"stdout.log").open("xb")as stdout,(root/"stderr.log").open("xb")as stderr:
            child=subprocess.Popen(command,cwd=HERE,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            while True:
                m=sample();family=[psutil.Process()]+psutil.Process().children(recursive=True);assert len(family)<=8
                private=0
                for p in family:
                    q=who(p)
                    if p.pid in known:assert known[p.pid]==q
                    known[p.pid]=q
                    try:private+=p.memory_info().private
                    except psutil.NoSuchProcess:
                        if p.pid==os.getpid():raise
                        cleanup.append(dict(identity=q,phase="exited_after_identity_before_private",private="UNKNOWN"))
                peak=max(peak,private);observations+=1
                assert private<=536870912 and m["commit"]>=limits["runtime_commit"]and m["physical"]>=limits["runtime_physical"]
                assert m["E_free"]>=100000000000 and time.monotonic()-started<=120
                code=child.poll()
                if code is not None:break
                time.sleep(.1)
            assert child.wait()==0
    except BaseException as e:
        error=dict(type=type(e).__name__,message=str(e)[:2000]);raise
    finally:
        if child is not None and child.poll()is None:
            for q in reversed(list(known.values())):
                if q["pid"]==os.getpid():continue
                try:
                    p=psutil.Process(q["pid"]);assert p.create_time()==q["ctime"]and p.ppid()==q["ppid"]
                    p.terminate();code=p.wait(timeout=15);cleanup.append(dict(identity=q,observed_cleanup_wait=code))
                except psutil.NoSuchProcess:pass
            if child.poll()is None:child.wait(timeout=15)
        save(root/"result.json",dict(passed=error is None and child is not None and child.returncode==0,Popen_returncode=None if child is None else child.returncode,
            guard_error=error,cleanup=cleanup,known_identities=list(known.values()),peak_family_private=peak,observations=observations,
            wall=time.monotonic()-started,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),command=command))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--ROOT-review",required=True);main(Path(p.parse_args().ROOT_review).resolve())
