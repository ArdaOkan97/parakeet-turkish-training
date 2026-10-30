"""INACTIVE proof-owned CPU14 entry wrapper, not a production CPU source.
No model/NumPy/Torch import at module import. Original protocol stays binary stdout.
Execution requires a separately ROOT-bound real first-packet proof scope.
"""
import argparse
import ctypes
import functools
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
ORIGINAL_SHA="e3d01517ac3af9e481a27b0127474c7a8fbc2d9351f7fcc6aefcf36691675d0e"
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def pin(p):
    p=Path(p);b=p.read_bytes()
    return dict(path=str(p.resolve()),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def run(args):
    import psutil
    scope=read(args.ROOT_proof_scope)
    assert scope["accepted"]is True and scope["reviewer"]=="/root"
    assert scope["kind"]=="ROOT_CPU14_original_startup_firstpacket_exact_ONE_CPU_scope_v1"
    source=Path(args.original_source).resolve();assert pin(source)==scope["CPU14_source"]
    assert pin(source)["sha256"]==ORIGINAL_SHA
    root=Path(args.receipt_directory).resolve()
    assert root.is_relative_to(Path(scope["output_directory"]).resolve())and root.is_relative_to(HERE)
    root.mkdir(exist_ok=False)
    me=psutil.Process()
    lock=threading.Lock()
    counter=[0];first_error=[False];write_error=[False]
    def atomic(name,value):
        temporary=root/(name+".owned_tmp")
        with temporary.open("x",encoding="utf-8")as f:
            json.dump(value,f,ensure_ascii=False,allow_nan=False);f.flush();os.fsync(f.fileno())
        os.replace(temporary,root/name)
    identity=dict(pid=me.pid,ctime=me.create_time(),ppid=me.ppid(),argv=list(sys.argv),original_source=pin(source),CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),TORCH_EXTENSIONS_DIR=os.environ.get("TORCH_EXTENSIONS_DIR"))
    # First receipt precedes original source import/profile/frontend/bootstrap.
    atomic("earliest_owned_worker_identity.json",identity)
    started=time.monotonic()
    def record(phase,**scalars):
        try:
            with lock:
                if counter[0]>=128:return
                counter[0]+=1
                value=dict(index=counter[0],phase=phase,elapsed=time.monotonic()-started,**scalars)
                raw=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
                if len(raw)>65536:raise ValueError("Bounded receipt record")
                atomic(str(counter[0]).zfill(3)+".json",value)
        except BaseException:
            write_error[0]=True
            # Receipt failure never masks/repeats the original operation.
            try:sys.stderr.write("OWNED_CPU_RECEIPT_WRITE_FAILED\n");sys.stderr.flush()
            except BaseException:pass
    def describe(error):
        try:
            message=str(error)[:2048]
        except BaseException:message="UNPRINTABLE_ORIGINAL_EXCEPTION"
        return dict(exception_type=type(error).__name__,message=message)
    def first_exception(error,phase):
        if first_error[0]:return
        first_error[0]=True
        record("first_original_exception",at=phase,**describe(error))
    def owned_timeout_stack():
        # Own main thread only. Copy code filename/name/line scalars, no f_locals.
        frames=None;frame=None
        try:
            frames=sys._current_frames()
            if len(frames)>128:
                record("timeout_stack_unavailable",reason="Own process thread-count bound");return
            frame=frames.get(threading.main_thread().ident)
            rows=[]
            while frame is not None and len(rows)<64:
                rows.append(dict(file=frame.f_code.co_filename[:512],function=frame.f_code.co_name[:128],line=frame.f_lineno))
                frame=frame.f_back
            record("own_main_thread_timeout_stack",frames=rows)
        except BaseException as error:record("timeout_stack_failed",**describe(error))
        finally:
            del frame,frames
    timer=threading.Timer(295,owned_timeout_stack)
    timer.name="owned-CPU14-timeout-receipt";timer.daemon=True
    timer.start()
    try:
        record("before_original_module_definitions")
        exported=runpy.run_path(str(source),run_name="CPU14_owned_proof_definitions")
        namespace=exported["worker_main"].__globals__
        record("after_original_module_definitions")
        for name in ("checked_resource_profile","_startup","_header","_send"):
            original=namespace[name]
            def make_wrapper(original,name):
                @functools.wraps(original)
                def observed(*positional,**keywords):
                    # No argument/packet/array values copied into retained receipts.
                    record("before_"+name)
                    try:
                        result=original(*positional,**keywords)
                    except BaseException as error:
                        first_exception(error,name)
                        raise
                    record("after_"+name)
                    return result
                return observed
            namespace[name]=make_wrapper(original,name)
        record("before_original_worker_main")
        code=exported["worker_main"](args.configuration,args.private_limit)
        record("after_original_worker_main",returned_code=code,receipt_write_failed=write_error[0])
        return code
    except BaseException as error:
        first_exception(error,"module_or_worker_main")
        raise
    finally:
        timer.cancel();timer.join(timeout=2)
        if timer.is_alive():
            try:sys.stderr.write("OWNED_TIMEOUT_RECEIPT_THREAD_JOIN_UNKNOWN\n")
            except BaseException:pass
        record("runner_finally",receipt_write_failed=write_error[0])
        # No historical worker-body death or normal0 is inferred by this file.

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--ROOT-proof-scope",required=True)
    p.add_argument("--original-source",required=True)
    p.add_argument("--receipt-directory",required=True)
    p.add_argument("--worker",action="store_true",required=True)
    p.add_argument("--configuration",required=True)
    p.add_argument("--private-limit",type=int,default=8000000000)
    args=p.parse_args()
    raise SystemExit(run(args))
