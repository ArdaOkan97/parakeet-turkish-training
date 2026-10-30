"""INACTIVE original-CPU14 startup-only receipt entry; separately ROOT-admitted use only.
Binary stdout and the original worker remain unchanged. No per-update observer.
"""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time

ROOT=Path(r"E:/gpu_training_scratchpad").resolve()
ORIGINAL_SHA="e3d01517ac3af9e481a27b0127474c7a8fbc2d9351f7fcc6aefcf36691675d0e"
MAXIMUM_RECEIPT_BYTES=1048576
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def pin(p):
    p=Path(p);b=p.read_bytes()
    return dict(path=str(p.resolve()),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def run(args):
    import psutil
    scope=read(args.ROOT_startup_scope)
    assert scope["accepted"]is True and scope["reviewer"]=="/root"
    assert scope["kind"]in (
        "ROOT_CPU14_startup_adapter_firstpacket_exact_ONE_CPU_scope_v1",
        "ROOT_ASR18_source40_saved212000_training_exact_ONE_scope_v1")
    source=Path(args.original_source).resolve()
    assert pin(source)==scope["CPU14_source"]and pin(source)["sha256"]==ORIGINAL_SHA
    assert pin(__file__)==scope["CPU_startup_runner_source"]
    assert pin(args.configuration)==json.loads(args.expected_configuration_pin)
    root=Path(args.receipt_directory).resolve()
    allowed=Path(scope["CPU_receipt_root"]).resolve()
    assert root.is_relative_to(allowed)and root!=allowed and root.is_relative_to(ROOT)
    root.mkdir(exist_ok=False)
    me=psutil.Process();lock=threading.Lock()
    counter=[0];first_error=[False];write_error=[False];written_bytes=[0]
    observed=set();wrappers={};namespace=None
    def atomic(name,value):
        raw=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
        # Include a full maximum temporary record ABOVE all already saved bytes.
        if len(raw)>65536 or written_bytes[0]+len(raw)+65536>MAXIMUM_RECEIPT_BYTES:
            raise ValueError("Startup receipts plus owned temporary exceed1MiB")
        temporary=root/(name+".owned_tmp")
        with temporary.open("xb")as f:f.write(raw);f.flush();os.fsync(f.fileno())
        os.replace(temporary,root/name);written_bytes[0]+=len(raw)
    atomic("earliest_owned_worker_identity.json",dict(pid=me.pid,ctime=me.create_time(),ppid=me.ppid(),
        argv=list(sys.argv),original_source=pin(source),runner_source=pin(__file__),
        CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
        TORCH_EXTENSIONS_DIR=os.environ.get("TORCH_EXTENSIONS_DIR")))
    started=time.monotonic()
    def record(phase,**scalars):
        try:
            with lock:
                if counter[0]>=128:return
                counter[0]+=1
                atomic(str(counter[0]).zfill(3)+".json",dict(
                    index=counter[0],phase=phase,elapsed=time.monotonic()-started,**scalars))
        except BaseException:
            write_error[0]=True
            try:sys.stderr.write("OWNED_CPU_STARTUP_RECEIPT_WRITE_FAILED\n");sys.stderr.flush()
            except BaseException:pass
    def describe(error):
        try:message=str(error)[:2048]
        except BaseException:message="UNPRINTABLE_ORIGINAL_EXCEPTION"
        return dict(exception_type=type(error).__name__,message=message)
    def first_exception(error,phase):
        if first_error[0]:return
        first_error[0]=True;record("first_original_exception",at=phase,**describe(error))
    def owned_timeout_stack():
        frames=None;frame=None
        try:
            frames=sys._current_frames()
            if len(frames)>128:
                record("startup_timeout_stack_unavailable",reason="Own thread-count bound");return
            frame=frames.get(threading.main_thread().ident);rows=[]
            while frame is not None and len(rows)<64:
                rows.append(dict(file=frame.f_code.co_filename[:512],function=frame.f_code.co_name[:128],line=frame.f_lineno))
                frame=frame.f_back
            record("own_startup_main_thread_timeout_stack",frames=rows)
        except BaseException as error:record("startup_timeout_stack_failed",**describe(error))
        finally:del frame,frames
    timer=threading.Timer(295,owned_timeout_stack)
    timer.name="owned-CPU14-startup-timeout-receipt";timer.daemon=True;timer.start()
    try:
        record("before_original_module_definitions")
        exported=runpy.run_path(str(source),run_name="CPU14_owned_startup_definitions")
        namespace=exported["worker_main"].__globals__
        record("after_original_module_definitions")
        for name in ("checked_resource_profile","_startup","_header","_send"):
            original=namespace[name]
            def make_wrapper(original,name):
                @functools.wraps(original)
                def observed_once(*positional,**keywords):
                    # Detach BEFORE the first original call; future calls have
                    # zero wrapper/clock/receipt overhead, even on failure.
                    if namespace.get(name)is wrappers.get(name):namespace[name]=original
                    wrappers.pop(name,None)
                    record("before_first_"+name)
                    try:result=original(*positional,**keywords)
                    except BaseException as error:
                        first_exception(error,name);raise
                    finally:
                        observed.add(name)
                        if len(observed)==4:timer.cancel()
                    record("after_first_"+name)
                    return result
                return observed_once
            wrappers[name]=make_wrapper(original,name);namespace[name]=wrappers[name]
        record("before_original_worker_main")
        code=exported["worker_main"](args.configuration,args.private_limit)
        record("after_original_worker_main",returned_code=code,receipt_write_failed=write_error[0])
        return code
    except BaseException as error:
        first_exception(error,"module_or_worker_main");raise
    finally:
        timer.cancel();timer.join(timeout=2)
        if timer.is_alive():
            try:sys.stderr.write("OWNED_STARTUP_TIMEOUT_THREAD_JOIN_UNKNOWN\n")
            except BaseException:pass
        if namespace is not None:
            for name,wrapper in list(wrappers.items()):
                if namespace.get(name)is wrapper:namespace[name]=wrapper.__wrapped__
            wrappers.clear()
        record("runner_finally",receipt_write_failed=write_error[0],
            startup_hooks_observed_once=sorted(observed),startup_wrappers_remaining=len(wrappers))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--ROOT-startup-scope",required=True)
    p.add_argument("--original-source",required=True)
    p.add_argument("--receipt-directory",required=True)
    p.add_argument("--expected-configuration-pin",required=True)
    p.add_argument("--worker",action="store_true",required=True)
    p.add_argument("--configuration",required=True)
    p.add_argument("--private-limit",type=int,default=8000000000)
    raise SystemExit(run(p.parse_args()))
