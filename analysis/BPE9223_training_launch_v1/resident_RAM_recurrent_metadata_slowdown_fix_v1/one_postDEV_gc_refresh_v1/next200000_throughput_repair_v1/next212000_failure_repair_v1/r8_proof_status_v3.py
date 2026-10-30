"""INACTIVE SOURCE ONLY. Actual execution needs an exact ROOT scope.
Fresh diagnostic E DB/journal only. Never opens live reader/trainer paths.
"""
import argparse
import concurrent.futures
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE=Path(__file__).resolve().parent
E=Path(r"E:/gpu_training_scratchpad").resolve()
CANDIDATE=HERE/"serve_BPE9223_Trackio_live_v8_INACTIVE_v1.py"
PROJECTION=HERE/"incremental_scalar_dashboard_projection_v2.py"
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def save(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+".owned_tmp")
    with tmp.open("x",encoding="utf-8")as f:
        json.dump(x,f,ensure_ascii=False,allow_nan=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def pin(p):
    p=Path(p);b=p.read_bytes()
    return dict(path=str(p.resolve()),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def load(p,name):
    spec=importlib.util.spec_from_file_location(name,p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def who(p):return dict(pid=p.pid,ctime=p.create_time(),ppid=p.ppid())
def under(p,root):
    p=Path(p).resolve()
    if not p.is_relative_to(root)or p==root:raise ValueError("Owned fixture path required")
    return p
def resources():
    import psutil
    class M(ctypes.Structure):
        _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(n,ctypes.c_ulonglong)for n in ("totalphysical","physical","commitlimit","commit","totalvirtual","virtual","extended")]
    m=M();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError()
    return dict(commit=m.commit,physical=m.physical,private=psutil.Process().memory_info().private,E_free=shutil.disk_usage(E).free)
def request(base,name,kwargs=None,token=None):
    if name=="/version":req=urllib.request.Request(base+name)
    else:
        req=urllib.request.Request(base+"/api/"+name,data=json.dumps({"kwargs":kwargs}).encode(),headers={"Content-Type":"application/json",**({"X-Trackio-Write-Token":token}if token else{})})
    t=time.monotonic()
    with urllib.request.urlopen(req,timeout=20)as response:answer=json.load(response)
    elapsed=time.monotonic()-t
    if elapsed>=20:raise AssertionError("Original20s API bound")
    if "error"in answer:raise RuntimeError(answer["error"])
    return answer.get("data",answer),elapsed

def server_child(bindings):
    """Owned real-server child; only fixture stop/closure control is doubled."""
    import psutil,uvicorn
    root=under(read(bindings)["fixture_root"],HERE)
    entry=who(psutil.Process());save(root/"child_entry.json",entry)
    instance=[];stop=threading.Event()
    original_init=uvicorn.Server.__init__
    def capture(self,*args,**kwargs):
        original_init(self,*args,**kwargs);instance.append(self)
    uvicorn.Server.__init__=capture
    def fixture_stop():
        while not stop.wait(.1):
            if (root/"fixture_stop.json").exists()and instance:
                instance[0].should_exit=True;return
    thread=threading.Thread(target=fixture_stop,name="owned-fixture-stop",daemon=False)
    thread.start()
    argv=sys.argv
    original_urlopen=None
    try:
        reader=load(CANDIDATE,"owned_reader7_candidate")
        original_write=reader.write
        original_urlopen=reader.urllib.request.urlopen
        fixture_area=under(read(bindings)["area"],root)
        def negative_STATE_publication_fence():
            # METHOD_STUB fixture observation only. The actual failing STATE or
            # response is untouched; prevent a later retry/progress write from
            # racing the parent's exact negative-state assertion.
            for observed,released in (
                ("STATE_publication_failure_observed.json","STATE_failure_fence_release.json"),
                ("response_loss_observed.json","response_loss_fence_release.json"),
                ("Windows_STATE_failure_observed.json","Windows_STATE_failure_fence_release.json")):
                if (root/observed).exists()and not(root/released).exists():
                    deadline=time.monotonic()+15
                    while not(root/released).exists():
                        if time.monotonic()>=deadline:
                            raise TimeoutError("METHOD_STUB negative STATE observation fence15s")
                        time.sleep(.01)
        def publication_failure(path,value):
            if Path(path).name=="mirror_state.json":
                negative_STATE_publication_fence()
            marker=root/"fail_next_STATE_publication.json"
            if Path(path).name=="mirror_state.json"and marker.exists():
                fault=read(marker)
                if value.get("offset")==fault["when_offset"]:
                    # This snapshot is inside the exact intercepted write,
                    # immediately before rejecting it, not a polling sample.
                    pre_state=read(path)
                    marker.unlink() # Owned diagnostic fault token only.
                    save(root/"STATE_publication_failure_observed.json",dict(
                        pre_STATE=pre_state,attempted_STATE=value,attempted_offset=value["offset"],
                        original_STATE_write_called=False,
                        observation_barrier="METHOD_STUB later STATE publications await explicit release; bounded15s",
                        fault_owner="METHOD_STUB one local write failure"))
                    raise OSError("fixture local STATE publication failed")
            windows_marker=root/"Windows_STATE_publication_gate.json"
            if Path(path).name=="mirror_state.json"and windows_marker.exists():
                fault=read(windows_marker)
                if value.get("offset")==fault["when_offset"]:
                    pre_state=read(path);windows_marker.unlink()
                    save(root/"Windows_STATE_gate_ready.json",dict(pre_STATE=pre_state,attempted_STATE=value))
                    deadline=time.monotonic()+10
                    while not(root/"Windows_STATE_gate_release.json").exists():
                        if time.monotonic()>=deadline:raise TimeoutError("Owned Windows STATE gate10s")
                        time.sleep(.01)
                    original_reader_os=reader.os
                    replace_error_identities=[]
                    def observed_replace(source,destination):
                        try:return original_reader_os.replace(source,destination)
                        except OSError as error:
                            assert len(replace_error_identities)<64
                            replace_error_identities.append(id(error))
                            raise
                    class WindowsOSProxy:
                        replace=staticmethod(observed_replace)
                        def __getattr__(self,key):return getattr(original_reader_os,key)
                    reader.os=WindowsOSProxy()
                    try:
                        try:return original_write(path,value)
                        except OSError as error:
                            save(root/"Windows_STATE_failure_observed.json",dict(pre_STATE=pre_state,
                                attempted_STATE=value,error_type=type(error).__name__,winerror=getattr(error,"winerror",None),
                                original_candidate_write_called=True,first_exception_identity=id(error),
                                first_actual_replace_error_identity=replace_error_identities[0]if replace_error_identities else None,
                                actual_replace_error_attempts=len(replace_error_identities),
                                fault_owner="REAL Windows deny-FILE_SHARE_DELETE handle; METHOD_STUB bounded observation fence only"))
                            raise
                    finally:
                        reader.os=original_reader_os
                        replace_error_identities.clear()
            return original_write(path,value)
        def response_lost(request,*args,**kwargs):
            response=original_urlopen(request,*args,**kwargs)
            marker=root/"lose_next_mirror_response.json"
            if marker.exists()and getattr(request,"data",None):
                fault=read(marker);body=json.loads(request.data)
                ids=[row.get("log_id")for row in body.get("kwargs",{}).get("logs",[])]
                if fault["when_log_id"]in ids and response.status==200:
                    response.read(1048576);response.close();marker.unlink()
                    pre_state=read(fixture_area/"mirror_state.json")
                    save(root/"response_loss_observed.json",dict(
                        pre_STATE=pre_state,original_HTTP200_consumed=True,
                        intercepted_log_id=fault["when_log_id"],
                        SQLite_durable_commit_from_HTTP200_alone="NOT_CLAIMED",
                        observation_barrier="METHOD_STUB later STATE publications await explicit release; bounded15s",
                        fault_owner="METHOD_STUB transport response loss after real API return"))
                    raise TimeoutError("fixture response lost after original HTTP200")
            return response
        reader.write=publication_failure
        reader.urllib.request.urlopen=response_lost
        sys.argv=[str(CANDIDATE),"--fixture-bindings",str(bindings)]
        reader.main()
        # main returned only after actual mirror.join and API.close completed.
        names=[t.name for t in threading.enumerate()if t.name.startswith("Trackio-owned-")or t.name=="read-only-training-mirror"]
        if names:raise AssertionError("Owned reader method closure incomplete")
        save(root/"child_method_closure.json",dict(entry=entry,reader_main_returned=True,owned_threads_remaining=names,real_API_and_mirror_close_returned=True))
    finally:
        if original_urlopen is not None:urllib.request.urlopen=original_urlopen
        sys.argv=argv;uvicorn.Server.__init__=original_init
        stop.set();thread.join(2)
        if thread.is_alive():raise RuntimeError("Fixture stop thread did not join")



def deny_delete(path):
    """REAL Windows handle: allow readers/writers, deny FILE_SHARE_DELETE."""
    from ctypes import wintypes
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    kernel.CreateFileW.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle.restype=wintypes.BOOL
    handle=kernel.CreateFileW(str(Path(path).resolve()),0x80000000,0x1|0x2,None,3,0x80,None)
    if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    return kernel,handle

def close_deny_delete(pair):
    kernel,handle=pair
    if not kernel.CloseHandle(handle):raise ctypes.WinError(ctypes.get_last_error())

def publication_negatives(root,guard):
    """Owned JSON only; no API/DB/history in focused publication mode."""
    import types
    case=root/"Windows_publication";case.mkdir()
    reader=load(CANDIDATE,"owned_reader8_publication")
    original_os=reader.os
    target=case/"current_status.json"
    original_value={"offset":0,"pending":None,"rows":list(range(64))}
    next_value={"offset":64,"pending":None,"rows":list(range(64))}
    reader.write(target,original_value);old_bytes=target.read_bytes()
    foreign=target.with_suffix(target.suffix+".tmp");foreign.write_bytes(b"foreign-original-tmp")
    foreign_owned=case/(target.name+".owned.foreign.tmp");foreign_owned.write_bytes(b"foreign-owned-looking")
    attempts=[];attempt_lock=threading.Lock();first_errors=[];observed=threading.Event()
    original_replace=original_os.replace
    def actual_replace(source,destination):
        try:return original_replace(source,destination)
        except OSError as error:
            with attempt_lock:
                assert len(attempts)<64
                attempts.append(dict(source=str(source),destination=str(destination),
                    error_type=type(error).__name__,winerror=getattr(error,"winerror",None),error_identity=id(error)))
                if not first_errors:first_errors.append(error)
            observed.set()
            raise
    class OSProxy:
        replace=staticmethod(actual_replace)
        def __getattr__(self,key):return getattr(original_os,key)
    reader.os=OSProxy()
    records=[];thread=None;held=None;all_threads=[]
    try:
        guard()
        held=deny_delete(target);done=threading.Event();outcome={}
        def publish():
            try:reader.write(target,next_value);outcome["returned"]=True
            except BaseException as error:outcome["error"]=error
            finally:done.set()
        thread=threading.Thread(target=publish,name="owned-Windows-publication",daemon=False);all_threads.append(thread);thread.start()
        assert observed.wait(.5),"Real Windows sharing error was not observed"
        assert target.read_bytes()==old_bytes
        close_deny_delete(held);held=None
        assert done.wait(2);thread.join(.1);assert not thread.is_alive()
        save(case/"released_contention_witness.json",dict(attempts=attempts,writer_returned=outcome.get("returned",False),
            writer_error=None if outcome.get("error")is None else type(outcome["error"]).__name__,
            final_bytes=target.stat().st_size,actual_real_deny_delete=True))
        assert outcome.get("error")is None and read(target)==next_value
        assert foreign.read_bytes()==b"foreign-original-tmp"and foreign_owned.read_bytes()==b"foreign-owned-looking"
        records.append(dict(case="real_transient_deny_delete_released_after_actual_error",passed=True))
        guard()
        stable=target.read_bytes();attempts.clear();first_errors.clear();observed.clear()
        held=deny_delete(target);began=time.monotonic();caught=None
        try:reader.write(target,{"offset":128,"rows":list(range(64))})
        except OSError as error:caught=error
        elapsed=time.monotonic()-began
        save(case/"persistent_contention_witness.json",dict(elapsed_seconds=elapsed,attempts=attempts,
            error_type=None if caught is None else type(caught).__name__,error_identity=None if caught is None else id(caught),
            first_error_identity=None if not first_errors else id(first_errors[0]),target_unchanged=target.read_bytes()==stable))
        assert caught is first_errors[0]and getattr(caught,"winerror",None)in(5,32,33)
        assert elapsed<=1.5 and target.read_bytes()==stable
        close_deny_delete(held);held=None
        assert all(not Path(item["source"]).exists()for item in attempts)
        reader.write(target,{"offset":128,"rows":list(range(64))})
        assert read(target)["offset"]==128
        caught.__traceback__=None;caught=None
        for error in first_errors:error.__traceback__=None
        first_errors.clear()
        records.append(dict(case="persistent_lock_preserves_first_exception_and_old_file_then_replay",passed=True))
        guard()
        sentinel=ctypes.WinError(87);calls=[]
        class NonTransientProxy:
            def replace(self,*args):calls.append(1);raise sentinel
            def __getattr__(self,key):return getattr(original_os,key)
        reader.os=NonTransientProxy();snapshot=target.read_bytes();caught=None
        try:reader.write(target,{"offset":999})
        except OSError as error:caught=error
        save(case/"nontransient_error_witness.json",dict(calls=len(calls),same_exception=caught is sentinel,target_unchanged=target.read_bytes()==snapshot))
        assert caught is sentinel and calls==[1]and target.read_bytes()==snapshot
        sentinel.__traceback__=None;caught=None;reader.os=original_os
        before={p.name for p in case.iterdir()};caught=None
        try:reader.write(target,{"unsupported":object()})
        except TypeError as error:caught=error
        assert caught is not None and target.read_bytes()==snapshot
        assert {p.name for p in case.iterdir()}==before
        caught.__traceback__=None;caught=None
        records.append(dict(case="nontransient_and_serialization_errors_fail_closed_owned_stage_only",passed=True))
        guard()
        barrier=threading.Barrier(3);sources=[];entered_sources=set();values=[{"writer":i,"rows":list(range(64))}for i in range(2)];errors=[]
        class ConcurrentProxy:
            def replace(self,source,destination):
                with attempt_lock:
                    first=str(source)not in entered_sources
                    if first:entered_sources.add(str(source));sources.append(str(source))
                if first:barrier.wait(timeout=2)
                return original_replace(source,destination)
            def __getattr__(self,key):return getattr(original_os,key)
        reader.os=ConcurrentProxy()
        def concurrent_write(value):
            try:reader.write(target,value)
            except BaseException as error:errors.append(error)
        threads=[threading.Thread(target=concurrent_write,args=(value,),daemon=False,name="owned-publication-pair")for value in values]
        all_threads.extend(threads)
        for item in threads:item.start()
        barrier.wait(timeout=2)
        observed_values=[];observer_open_errors=[];observer_first_error=None
        for i in range(64):
            if i%16==0:guard()
            try:observer_stream=target.open("r",encoding="utf-8-sig")
            except PermissionError as error:
                # Record an open failure during observation, without a locker cause.
                # Successful observations still must contain complete JSON only.
                # Only the open is sampled again; JSON/decode/read errors propagate.
                winerror=getattr(error,"winerror",None)
                if os.name!="nt"or not(winerror in(5,32,33)or(winerror is None and error.errno==13)):raise
                if observer_first_error is None:observer_first_error=error
                observer_open_errors.append(dict(error_type=type(error).__name__,errno=error.errno,
                    winerror=winerror,error_identity=id(error),sample_index=i))
            else:
                with observer_stream:observed_values.append(json.load(observer_stream))
            if all(not item.is_alive()for item in threads):break
        for item in threads:item.join(2)
        assert not errors and all(not item.is_alive()for item in threads)
        # No exception is suppressed after both writers have joined. A persistent
        # permission/open error or malformed JSON still fails this proof.
        final_observed_value=read(target)
        observed_values.append(final_observed_value)
        save(case/"concurrent_stage_ownership_witness.json",dict(source_paths=sources,
            original_or_complete_new_only=all(value=={"offset":128,"rows":list(range(64))}or value in values for value in observed_values),
            errors=[type(error).__name__ for error in errors],all_threads_joined=all(not item.is_alive()for item in threads),
            observer_open_errors=observer_open_errors,observer_successful_reads=len(observed_values),
            observed_complete_values=observed_values,expected_writer_values=values,
            old_complete_value={"offset":128,"rows":list(range(64))},terminal_observed_value=final_observed_value,
            observer_first_error_identity=None if observer_first_error is None else id(observer_first_error),
            observer_open_errors_bounded_64=len(observer_open_errors)<=64,
            terminal_read_after_all_writer_joins=True,terminal_complete_new_value=final_observed_value in values,
            no_added_wait_or_timeout=True,JSON_or_read_errors_never_suppressed=True))
        assert len(set(sources))==2 and all(not Path(path).exists()for path in sources)
        assert final_observed_value in values
        assert all(value=={"offset":128,"rows":list(range(64))}or value in values for value in observed_values)
        assert foreign.read_bytes()==b"foreign-original-tmp"and foreign_owned.read_bytes()==b"foreign-owned-looking"
        records.append(dict(case="unique_exclusive_stages_and_atomic_old_or_complete_new",passed=True))
        return dict(publication_cases=records,rows_in_owned_JSON=64,real_Windows_handles_closed=True,
            no_HTTP_DB_or_production=True,full_new_candidate500000_shapes_and_API_lifetime="PENDING_SEPARATE_ROOT")
    finally:
        primary=sys.exc_info()[1]
        reader.os=original_os
        if held is not None:close_deny_delete(held)
        for item in all_threads:
            if item.ident is not None:item.join(2)
            if item.is_alive():
                error=RuntimeError("Owned publication thread closure UNKNOWN")
                if primary is None:raise error
                primary.add_note(str(error))
        for error in first_errors:error.__traceback__=None
        first_errors.clear()

def source_bound_lifetime_negatives(binding,scope):
    """Actual helper methods/reader main; scientific numerical owners absent.
    Pure headroom branch uses explicitly labelled metadata owners.
    """
    import ast,asyncio,http.server,types,weakref
    import trackio.asgi_app as asgi,trackio.server as official
    helper=load(HERE/"Trackio_owned_bounded_sync_API_dispatch_v4.py","fixture_API_helper")
    sources={m.__name__:pin(m.__file__)for m in (asgi,official)}
    original_invoke=asgi._invoke_handler
    original_start=helper._OwnedSlot.start
    events=[];sentinel=RuntimeError("fixture second worker start")
    def fail_second(slot):
        events.append(slot.thread.name)
        if len(events)==2:raise sentinel
        return original_start(slot)
    helper._OwnedSlot.start=fail_second
    try:
        try:helper.OwnedDispatch(asgi,official,sources,lambda:None)
        except RuntimeError as error:assert error is sentinel
        else:raise AssertionError("Second-start negative did not reject")
        assert asgi._invoke_handler is original_invoke
        assert not [t for t in threading.enumerate()if t.name.startswith("Trackio-owned-")]
    finally:helper._OwnedSlot.start=original_start;sentinel.__traceback__=None

    # Fault exactly after both workers start, during hook installation.
    class FailingModule:
        def __init__(self):
            object.__setattr__(self,"original",asgi._invoke_handler)
            object.__setattr__(self,"value",asgi._invoke_handler)
            object.__setattr__(self,"error",RuntimeError("fixture post-start hook install"))
        def __getattr__(self,key):
            if key=="_invoke_handler":return self.value
            return getattr(asgi,key)
        def __setattr__(self,key,value):
            if key=="_invoke_handler"and value is not self.original:
                assert len([t for t in threading.enumerate()if t.name.startswith("Trackio-owned-")])==2
                object.__setattr__(self,"value",value) # Exercise restore-after-partial-install too.
                raise self.error
            if key=="_invoke_handler":object.__setattr__(self,"value",value)
            else:object.__setattr__(self,key,value)
    failing=FailingModule()
    try:
        try:helper.OwnedDispatch(failing,official,sources,lambda:None)
        except RuntimeError as caught:assert caught is failing.error
        else:raise AssertionError("Post-start install failure not preserved")
        assert failing._invoke_handler is failing.original
        assert not [t for t in threading.enumerate()if t.name.startswith("Trackio-owned-")]
    finally:failing.error.__traceback__=None

    async def cancellation_and_exception():
        enter=threading.Event();release=threading.Event();calls=[]
        old=official.get_logs_batch
        marker=ValueError("fixture original exception identity")
        class RequestOwner:pass
        def owned_function(request,which=0):
            calls.append(which)
            if which==0:enter.set();assert release.wait(5)
            if which==2:raise marker
            return which
        official.get_logs_batch=owned_function
        dispatch=helper.OwnedDispatch(asgi,official,sources,lambda:None)
        reference=None;task_flags={};exception_identity_seen=False;escaped=None
        def lifecycle_witness(stage):
            import psutil
            slots={lane:dict(lifecycle=dict(slot.lifecycle),queue_size=slot.queue.qsize(),
                queue_unfinished=slot.queue.unfinished_tasks,closed=slot.closed,
                thread_alive=slot.thread.is_alive())for lane,slot in dispatch.slots.items()}
            save(Path(binding).with_name("request_lifecycle_"+stage+".json"),dict(
                stage=stage,entry=who(psutil.Process()),request_weakref_alive=None if reference is None else reference()is not None,
                original_call_order=list(calls),task_flags=dict(task_flags),owned_slots=slots,
                original_exception_identity=id(marker),same_original_exception_handled=exception_identity_seen,
                escaped_exception=escaped,
                witness="Scalar weakref/task/owned-slot flags only; no frames/locals/request contents/traceback dump"))
        try:
            request_owner=RequestOwner();reference=weakref.ref(request_owner)
            first=asyncio.create_task(asgi._invoke_handler(owned_function,request_owner,kwargs={"which":0}))
            while not enter.is_set():await asyncio.sleep(.001)
            first.cancel()
            try:await first
            except asyncio.CancelledError:pass
            task_flags["first"]=dict(done=first.done(),cancelled=first.cancelled())
            second=asyncio.create_task(asgi._invoke_handler(owned_function,request_owner,kwargs={"which":1}))
            await asyncio.sleep(.02);assert calls==[0]
            release.set();assert await second==1
            task_flags["second"]=dict(done=second.done(),cancelled=second.cancelled())
            try:await asgi._invoke_handler(owned_function,request_owner,kwargs={"which":2})
            except ValueError as error:
                assert error is marker
                exception_identity_seen=True
            else:raise AssertionError("Original exception masked")
            # Caller has handled the exact ORIGINAL exception; only now release
            # its diagnostic traceback as in the original lifetime fixture.
            marker.__traceback__=None
            del first,second,request_owner
            await asyncio.sleep(0);await asyncio.sleep(0)
            lifecycle_witness("before_weakref_assert")
            assert reference()is None
        except BaseException as error:
            escaped=dict(type=type(error).__name__,message=str(error))
            raise
        finally:
            primary=sys.exc_info()[1]
            try:lifecycle_witness("finally")
            except BaseException as publication_error:
                if primary is None:raise
                primary.add_note("Lifecycle scalar witness publication failed: "+type(publication_error).__name__+": "+str(publication_error)[:500])
            finally:
                release.set();dispatch.close();official.get_logs_batch=old
                marker.__traceback__=None
    asyncio.run(cancellation_and_exception())

    # Real reader main partial-startup failures, before any server.run.
    reader=load(CANDIDATE,"fixture_reader_partial_startup")
    oldargv=sys.argv;sys.argv=[str(CANDIDATE),"--fixture-bindings",str(binding)]
    original_write=reader.write
    error=RuntimeError("fixture owner publication failed")
    def write_fail(path,value):
        if Path(path).name=="owner.json":raise error
        return original_write(path,value)
    reader.write=write_fail
    try:
        try:reader.main()
        except RuntimeError as caught:assert caught is error
        else:raise AssertionError("Owner-write negative did not fail")
        assert asgi._invoke_handler is original_invoke
        assert not [t for t in threading.enumerate()if t.name.startswith("Trackio-owned-")]
    finally:reader.write=original_write;sys.argv=oldargv;error.__traceback__=None
    original_thread_start=threading.Thread.start
    error=RuntimeError("fixture mirror thread start failed")
    def mirror_start_fail(thread):
        if thread.name=="read-only-training-mirror":raise error
        return original_thread_start(thread)
    threading.Thread.start=mirror_start_fail;sys.argv=[str(CANDIDATE),"--fixture-bindings",str(binding)]
    try:
        try:reader.main()
        except RuntimeError as caught:assert caught is error
        else:raise AssertionError("Mirror-start negative did not fail")
        assert asgi._invoke_handler is original_invoke
        assert not [t for t in threading.enumerate()if t.name.startswith("Trackio-owned-")]
    finally:threading.Thread.start=original_thread_start;sys.argv=oldargv;error.__traceback__=None

    b=read(binding);port_bad=Path(binding).with_name("invalid_port_binding.json")
    save(port_bad,dict(b,port=7860))
    try:reader.fixture_bindings(port_bad)
    except AssertionError:pass
    else:raise AssertionError("Production port accepted by fixture")
    # Own an actual foreign listener; fake only startup clock/owned-started flag.
    requests=[]
    class Foreign(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path);body=b'{"version":"wrong_fixture_instance"}'
            self.send_response(200);self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    foreign=http.server.HTTPServer(("127.0.0.1",0),Foreign)
    foreign_thread=threading.Thread(target=foreign.serve_forever,daemon=False);foreign_thread.start()
    actual_time=helper.time;clock=[0.]
    helper.time=types.SimpleNamespace(monotonic=lambda:clock[0],sleep=lambda amount:clock.__setitem__(0,clock[0]+amount))
    try:
        try:helper.wait_http_ready("http://127.0.0.1:"+str(foreign.server_port),lambda:None,lambda:False,lambda:False,total_seconds=.5)
        except TimeoutError:pass
        else:raise AssertionError("Wrong service promoted before own bind")
        assert requests==[]
        # Remaining deadline is passed to actual request interface (metadata spy).
        timeouts=[];oldurl=helper.urllib.request.urlopen
        def refused(req,timeout):
            timeouts.append(timeout);clock[0]+=.09;raise OSError("fixture refused")
        helper.urllib.request.urlopen=refused;clock[0]=0.
        try:
            try:helper.wait_http_ready("http://127.0.0.1:1",lambda:None,lambda:False,lambda:True,total_seconds=.1)
            except (TimeoutError,OSError):pass
            assert timeouts and all(0<t<=.1 for t in timeouts)
        finally:helper.urllib.request.urlopen=oldurl
    finally:
        helper.time=actual_time;foreign.shutdown();foreign.server_close();foreign_thread.join(2)
    # Execute both official API and affected official Gradio delegation on a real app.
    from starlette.requests import Request
    # Use the exact original public constructor's Spaces-only route/state
    # branch. Only its selector is METHOD_STUBbed; no state field is fabricated.
    normal_app,_=official.build_starlette_app_only(mcp_server=False)
    utils_pin=pin(asgi.utils.__file__)
    assert utils_pin==scope["official_utils_source"]
    assert utils_pin in scope["input_pins"]
    original_spaces_selector=asgi.utils.on_spaces
    assert original_spaces_selector()is False
    def owned_constructor_spaces_selector():return True
    oauth_paths={official.OAUTH_START_PATH,official.OAUTH_CALLBACK_PATH,"/oauth/logout"}
    oauth_routes=[route for route in normal_app.routes if getattr(route,"path",None)in oauth_paths]
    assert len(oauth_routes)==3
    asgi.utils.on_spaces=owned_constructor_spaces_selector
    try:
        gradio_app=asgi.create_trackio_starlette_app(
            oauth_routes,normal_app.state.api_registry,
            write_apis=normal_app.state.write_apis,
            allowed_file_roots=list(normal_app.state.allowed_file_roots),
            upload_authorizer=normal_app.state.upload_authorizer)
    finally:
        # Restore only our owned selector; never retarget a foreign replacement.
        if asgi.utils.on_spaces is owned_constructor_spaces_selector:
            asgi.utils.on_spaces=original_spaces_selector
    assert asgi.utils.on_spaces is original_spaces_selector
    assert original_spaces_selector()is False
    assert any(getattr(route,"endpoint",None)is asgi.gradio_call_post_handler for route in gradio_app.routes)
    assert not any(getattr(route,"endpoint",None)is asgi.gradio_call_post_handler for route in normal_app.routes)
    # No handler, storage, backend or auth call ran while selector was doubled.
    d=helper.OwnedDispatch(asgi,official,sources,lambda:None)
    async def affected_Gradio():
        binding_value=read(binding)
        payload=dict(kwargs=dict(project=binding_value["project"],runs=[dict(run=binding_value["run_name"],run_id=binding_value["run_name"])],max_points=3000,scalar_only=True))
        def req(api_name,event_id=None,app_owner=normal_app):
            data=json.dumps(payload).encode();sent=[False]
            async def receive():
                if sent[0]:return {"type":"http.disconnect"}
                sent[0]=True;return {"type":"http.request","body":data,"more_body":False}
            path=dict(api_name=api_name)
            if event_id is not None:path["event_id"]=event_id
            return Request(dict(type="http",method="POST",path="/fixture",headers=[(b"content-type",b"application/json")],query_string=b"",path_params=path,app=app_owner),receive)
        normal=await asyncio.wait_for(asgi.run_api_request(req("get_logs_batch"),"get_logs_batch"),20)
        assert normal.status_code==200
        expected=json.loads(normal.body)["data"]
        post=await asyncio.wait_for(asgi.gradio_call_post_handler(req("get_logs_batch",app_owner=gradio_app)),20)
        assert post.status_code==200
        event=json.loads(post.body)["event_id"]
        poll=await asgi.gradio_call_poll_handler(req("get_logs_batch",event,app_owner=gradio_app))
        body=b""
        async for chunk in poll.body_iterator:body+=chunk.encode()if isinstance(chunk,str)else chunk
        data_line=next(line[6:]for line in body.decode().splitlines()if line.startswith("data: "))
        assert json.loads(data_line)==[expected]
    try:asyncio.run(affected_Gradio())
    finally:d.close()

    # Actual original reader first-headroom branch with complete fixture globals.
    # The ORIGINAL readiness helper must call the supplied guard before HTTP.
    tree=ast.parse(CANDIDATE.read_text(encoding="utf-8"))
    functions={n.name:n for n in ast.walk(tree)if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    selected=[functions[n]for n in ("DashboardMemoryLimit","local_status","mirror_body","mirror")]
    binding_value=read(binding)
    namespace=dict(threading=threading,time=time,Path=Path,read=read,
        identity={"fixture":True},BASE="http://127.0.0.1:"+str(binding_value["port"]),
        RUN=Path(binding_value["run"]).resolve(),
        mirror_activity={"HTTP_ready":False,"committed_HTTP_batch":False},
        pending_holder={"value":None},terminal_holder={"error":None},
        server=types.SimpleNamespace(should_exit=False,started=False),
        STATE=Path(binding).with_name("never_exists_state.json"),
        STATUS=Path(binding).with_name("captured_status.json"),memory=lambda:{},
        api_dispatch=types.SimpleNamespace(stats={}),dispatch_module=helper)
    assert not namespace["STATE"].exists()
    statuses=[];namespace["write"]=lambda path,value:statuses.append(value)
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(CANDIDATE),"exec"),namespace)
    headroom=namespace["DashboardMemoryLimit"]("fixture original headroom cause")
    guard_calls=[]
    def fail_headroom():
        guard_calls.append(dict(call=len(guard_calls)+1,supplied_exception_identity=id(headroom)))
        raise headroom
    namespace["headroom"]=fail_headroom
    import psutil
    def headroom_witness(stage,escaped=None):
        terminal=namespace["terminal_holder"]["error"]
        save(Path(binding).with_name("headroom_first_guard_"+stage+".json"),dict(
            stage=stage,entry=who(psutil.Process()),statuses=statuses,guard_calls=guard_calls,
            terminal_type=None if terminal is None else type(terminal).__name__,
            terminal_message=None if terminal is None else str(terminal),
            terminal_identity=None if terminal is None else id(terminal),
            supplied_exception_identity=id(headroom),terminal_is_supplied_exception=terminal is headroom,
            escaped_exception=escaped,original_wait_http_ready=True,
            namespace_BASE=namespace["BASE"],server_started=namespace["server"].started,
            HTTP_ready=namespace["mirror_activity"]["HTTP_ready"],
            state_existed_before_call=False,
            original_first_guard_SOURCE_precedes_owned_started_HTTP_and_producer_read=True,
            owner="METHOD_STUB guard/status sink; ORIGINAL helper and reader branch AST"))
    escaped=None
    try:
        namespace["mirror"]()
        headroom_witness("before_assert")
        assert guard_calls==[dict(call=1,supplied_exception_identity=id(headroom))]
        assert namespace["terminal_holder"]["error"]is headroom
        assert statuses[-1]["status"]=="stopped_headroom"and "fixture original headroom cause"in statuses[-1]["error"]
        assert statuses[-1]["mirror_thread_alive"]is False
        assert namespace["mirror_activity"]["HTTP_ready"]is False
    except BaseException as error:
        escaped=dict(type=type(error).__name__,message=str(error))
        raise
    finally:
        primary=sys.exc_info()[1]
        try:headroom_witness("finally",escaped)
        except BaseException as publication_error:
            if primary is None:raise
            primary.add_note("Headroom scalar witness publication failed: "+type(publication_error).__name__+": "+str(publication_error)[:500])
        finally:
            terminal=namespace["terminal_holder"]["error"]
            if terminal is not None:terminal.__traceback__=None
            headroom.__traceback__=None
            namespace["terminal_holder"]["error"]=None
            namespace["headroom"]=None
            statuses.clear();guard_calls.clear()
    return dict(post_both_workers_hook_install_failure_unwind=True,second_start_unwind=True,owner_write_unwind=True,mirror_start_unwind=True,cancellation_keeps_slot_until_actual_completion=True,original_exception_identity=True,request_weakref_released=True,fixture_port7860_rejected=True,foreign_listener_received_no_request_before_own_bind=True,remaining_deadline_bound=True,actual_official_Gradio_post_poll_and_API_same_result=True,official_Spaces_constructor_branch_METHOD_STUB=True,selector_restored_BEFORE_handlers=True,deployed_local_Gradio_route_claimed=False,headroom_first_cause_preserved=True,headroom_owner="METHOD_STUB metadata and HTTP; actual reader branch AST")


def prepare_focused_real_server_STATE_fixture(root,scope,guard):
    """Fresh <=256 REAL official-schema rows; no accepted500k workload rerun."""
    assert scope["focused_real_server_initial_rows"]==64
    guard()
    case=root/"focused_real_server_STATE";case.mkdir()
    area=case/"area";area.mkdir()
    run=case/"run";run.mkdir();(run/"progress").mkdir()
    dbdir=area/"database";dbdir.mkdir()
    project="fixture_remaining_lifetime";name="fixture_training"
    os.environ["TRACKIO_DIR"]=str(dbdir)
    for key in ("TRACKIO_SPACE_ID","TRACKIO_SERVER_URL","TRACKIO_DATASET_ID","TRACKIO_BUCKET_ID","TRACKIO_WEBHOOK_URL"):
        os.environ.pop(key,None)
    import orjson,sqlite3,psutil
    from trackio.sqlite_storage import SQLiteStorage
    from trackio.utils import on_spaces
    assert not on_spaces(),"Only owned local fixture storage is admitted"
    db=SQLiteStorage.init_db(project)
    expected=(dbdir/(project+".db")).resolve()
    save(case/"actual_fixture_DB_path_binding.json",dict(shape="focused_real_server_STATE",
        actual_returned_DB_path=str(db.resolve()),expected_DB_path=str(expected),
        TRACKIO_DIR=os.environ.get("TRACKIO_DIR"),diagnostic_rows=64))
    assert db.resolve()==expected
    rows=[(name,"2026-01-01T00:00:%02d.000000"%(i%60),name,i,
        orjson.dumps({"train/objective":1.+i%7,"progress/update":i}).decode())for i in range(64)]
    conn=sqlite3.connect(db)
    try:
        conn.executemany("INSERT INTO metrics(run_id,timestamp,run_name,step,metrics)VALUES(?,?,?,?,?)",rows)
        conn.commit()
        assert conn.execute("SELECT COUNT(*)FROM metrics").fetchone()[0]==64
    finally:conn.close()
    journal=run/"fixture_journal.jsonl";journal.write_bytes(b"")
    save(run/"progress/latest.json",dict(actual_optimizer_updates=1,journal_prefix=dict(path=str(journal)),
        producer=dict(pid=os.getpid(),create_time=psutil.Process().create_time()),
        last_saved_updates=0,native_updates_this_process=1,native_seconds=1.,
        active_pipeline_seconds=1.,source_audio_seconds=1.,audio_minutes_per_wall_minute=1.,
        optimizer_poisoned=False,schedule_poisoned=False))
    save(run/"latest_checkpoint.json",dict(committed_updates=0))
    with socket.socket()as sock:sock.bind(("127.0.0.1",0));port=sock.getsockname()[1]
    assert port!=7860
    binding=case/"bindings.json"
    save(binding,dict(kind="Trackio7_owned_E_fixture_bindings_v1",fixture_root=str(case),area=str(area),
        run=str(run),source_db=str(case/"unused_source.db"),project=project,run_name=name,port=port))
    save(case/"focused_real_server_fixture_receipt.json",dict(real_original_schema=True,actual_rows=64,
        completed_metadata500000_main_rerun=False,fixture_only_no_training_authority=True))
    guard()
    return binding


def focused_real_server_STATE(root,scope,guard,wait,config_progress,children,identities):
    """Fresh64 original history rows plus64 real mirror rows, never a500k substitute."""
    import psutil,sqlite3
    binding=prepare_focused_real_server_STATE_fixture(root,scope,guard)
    value=read(binding)
    case=Path(value["fixture_root"]);area=Path(value["area"]);run=Path(value["run"])
    project=value["project"];name=value["run_name"];port=value["port"]
    assert port!=7860 and scope["focused_real_server_initial_rows"]==64
    journal=run/"fixture_journal.jsonl";db=area/"database"/(project+".db")
    out=(case/"stdout.log").open("xb");err=(case/"stderr.log").open("xb")
    try:
        argv=[sys.executable,"-X","utf8","-B",str(Path(__file__).resolve()),"--server-child",str(binding)]
        child=subprocess.Popen(argv,stdout=out,stderr=err,cwd=HERE,creationflags=subprocess.CREATE_NO_WINDOW)
        children.append(child)
        identities.append(who(psutil.Process(child.pid)))
    finally:
        out.close();err.close()
    wait(lambda:(case/"child_entry.json").exists(),10)
    wait(lambda:(area/"current_status.json").exists()and read(area/"current_status.json").get("reader_ready")is True,60)
    wait(lambda:(area/"mirror_state.json").exists(),15)
    token=(area/"local_write_token.txt").read_text().strip();base="http://127.0.0.1:"+str(port)
    payload=dict(project=project,runs=[dict(run=name,run_id=name)],max_points=3000,scalar_only=True)
    version,version_elapsed=request(base,"/version")
    logs,query_elapsed=request(base,"get_logs_batch",payload,token)
    assert isinstance(logs,list)and len(logs)==1
    pre=read(area/"mirror_state.json")
    assert pre.get("offset",0)==0 and pre.get("last_update",0)==0
    raw=b"".join((json.dumps(dict(kind="ASR18_committed_training_update_v1",optimizer_updates=i,
        native=dict(mean_objective=1.,global_grad_norm=.25),broad_draws=14*i,replay_updates=i,
        cleaner_draws=0))+"\n").encode()for i in range(1,65))
    target_offset=len(raw)
    save(case/"Windows_STATE_publication_gate.json",dict(when_offset=target_offset))
    with journal.open("ab")as handle:handle.write(raw);handle.flush();os.fsync(handle.fileno())
    config_progress(run,journal,65)
    held=None
    try:
        wait(lambda:(case/"Windows_STATE_gate_ready.json").exists(),15)
        held=deny_delete(area/"mirror_state.json")
        save(case/"Windows_STATE_gate_release.json",dict(release_to_actual_candidate_write=True))
        wait(lambda:(case/"Windows_STATE_failure_observed.json").exists(),5)
        wait(lambda:read(area/"current_status.json").get("error")is not None,5)
        fault=read(case/"Windows_STATE_failure_observed.json")
        actual=read(area/"mirror_state.json");status=read(area/"current_status.json")
        save(case/"focused_STATE_before_assert.json",dict(fault=fault,STATE=actual,status=status,
            HTTP200_proves_SQLite_commit=False))
        assert fault["original_candidate_write_called"]is True and fault["winerror"]in(5,32,33)
        assert fault["actual_replace_error_attempts"]>=1
        assert fault["first_exception_identity"]==fault["first_actual_replace_error_identity"]
        assert actual==fault["pre_STATE"]
        assert actual.get("offset",0)==0 and actual.get("last_update",0)==0
        assert status["mirror_thread_alive"]is True and status["pending_batch"]==dict(
            journal_path=str(journal),start_offset=0,end_offset=target_offset,last_update=64,count=64)
    finally:
        if held is not None:close_deny_delete(held)
        save(case/"Windows_STATE_failure_fence_release.json",dict(release_owned_observation_fence=True))
        if not(case/"Windows_STATE_gate_release.json").exists():
            save(case/"Windows_STATE_gate_release.json",dict(release_owned_gate_on_all_paths=True))
    wait(lambda:read(area/"mirror_state.json").get("offset")==target_offset,25)
    ids=[hashlib.sha256((name+":journal:"+hashlib.sha256(line).hexdigest()).encode()).hexdigest()
        for line in raw.splitlines(keepends=True)]
    assert len(ids)==len(set(ids))==64
    with sqlite3.connect(db)as check:
        count=check.execute("SELECT COUNT(*)FROM metrics WHERE log_id IN("+",".join("?"for _ in ids)+")",ids).fetchone()[0]
        total_rows=check.execute("SELECT COUNT(*)FROM metrics").fetchone()[0]
    assert count==64 and 128<=total_rows<=256
    wait(lambda:read(area/"current_status.json").get("status")=="live"and read(area/"current_status.json").get("error")is None,15)
    final=read(area/"mirror_state.json")
    assert final["offset"]==target_offset and final["last_update"]==64
    save(case/"focused_STATE_replay_before_close.json",dict(STATE=final,unique_journal_ids=64,
        SQL_count_for_exact_journal_ids=count,total_owned_original_schema_rows=total_rows,
        all_rows_at_most256=True,actual_real_bulk_log_and_original_mirror=True,
        HTTP200_SQLite_durability_NOT_inferred=True))
    save(case/"fixture_stop.json",{"stop_owned_fixture_only":True})
    wait(lambda:child.poll()is not None,40);assert child.wait()==0
    closure=read(case/"child_method_closure.json")
    assert closure["real_API_and_mirror_close_returned"]is True
    with socket.socket()as released:released.bind(("127.0.0.1",port))
    return dict(label="focused_real_server_STATE",initial_original_schema_rows=64,
        actual_journal_rows=64,total_owned_original_schema_rows=total_rows,
        actual_version_elapsed=version_elapsed,actual_scalar_elapsed=query_elapsed,
        first_actual_error_identity_preserved=True,pending_offset_preserved=True,
        same64_log_ids_replayed_exactly_once=True,SQLite_exact_id_count=count,
        real_Windows_deny_FILE_SHARE_DELETE=True,actual_child_identity=read(case/"child_entry.json"),
        launcher_identity=identities[-1],Popen_wait=child.returncode,closure=closure,port_released=True,
        full500k_cold_concurrency_lifetime_activation_NOT_claimed=True)


def fixture_body(scopepath):
    import psutil
    scope=read(scopepath)
    assert scope["accepted"]is True and scope["reviewer"]=="/root"
    assert scope["kind"]=="ROOT_Trackio7_real_API_mirror_integration_exact_ONE_CPU_scope_v1"
    root=under(scope["output_directory"],HERE);assert not root.exists();root.mkdir()
    limits=scope["resources"]
    # All original reservations plus new cache/executor/fixture family must be explicit.
    for k in ("prospective_commit","prospective_physical","runtime_commit","runtime_physical","family_private","self_private","wall_seconds"):
        assert type(limits[k])is int and limits[k]>0
    assert limits["prospective_commit"]>=8066304384
    assert limits["prospective_physical"]>=5316304384
    assert limits["runtime_commit"]>=5547868928 and limits["runtime_physical"]>=5047868928
    assert limits["family_private"]==limits["family_private_bytes"]==2573741824
    assert limits["self_private"]<=536870912 and limits["wall_seconds"]==limits["proof_body_wall_seconds"]<=160
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    first=resources();save(root/"initial_BEFORE_predicate.json",dict(sample=first,entry=who(psutil.Process()),launcher=who(psutil.Process().parent())))
    assert first["commit"]>=limits["prospective_commit"]and first["physical"]>=limits["prospective_physical"]
    assert first["private"]<=limits["self_private"]and first["E_free"]>=100000000000
    for item in scope["input_pins"]:assert pin(item["path"])==item
    assert scope["candidate"]==pin(CANDIDATE)
    execution_phase=scope["execution_phase"]
    assert execution_phase in ("publication_only","focused_real_server_STATE","original_metadata_shape","original_adverse_shape")
    projection_pin=pin(PROJECTION)
    assert scope["projection_source"]==projection_pin and projection_pin in scope["input_pins"]
    required_shape=scope["required_original_shape"]
    main_shape_pin=scope["accepted_metadata_main_ROOT"]
    prior_matching_pin=scope["prior_matching_candidate_first_shape_ROOT"]
    if execution_phase in ("publication_only","focused_real_server_STATE"):
        assert required_shape is None and main_shape_pin is None and prior_matching_pin is None
        assert scope["focused_publication_fixture_rows"]==64
        if execution_phase=="focused_real_server_STATE":assert scope["focused_real_server_initial_rows"]==64
    else:
        focused_pin=scope["accepted_focused_publication_ROOT"]
        assert focused_pin in scope["input_pins"]and pin(focused_pin["path"])==focused_pin
        focused=read(focused_pin["path"])
        assert focused["kind"]=="ROOT_reader8_Windows_publication_focused_actual_acceptance_v1"
        assert focused["accepted"]is True and focused["reviewer"]=="/root"
        assert focused["candidate_reader"]==scope["candidate"]
        assert focused["completed_focused_real_server_STATE_contention"]is True
        assert focused["completed_direct_publication_negatives"]is True
        if execution_phase=="original_metadata_shape":
            assert required_shape=="metadata_live_order"and main_shape_pin is None and prior_matching_pin is None
        else:
            assert required_shape=="adverse_reverse_interleaved"and main_shape_pin==prior_matching_pin
            assert prior_matching_pin in scope["input_pins"]and pin(prior_matching_pin["path"])==prior_matching_pin
            matching=read(prior_matching_pin["path"])
            assert matching["kind"]=="ROOT_reader8_first_original500000_shape_and_lifetime_actual_acceptance_v1"
            assert matching["accepted_first_shape"]is True and matching["reviewer"]=="/root"
            assert matching["candidate_reader"]==scope["candidate"]and matching["projection_source"]==projection_pin
            assert matching["completed_first_shape"]["shape"]=="metadata_live_order"
            assert matching["completed_first_shape"]["original_history_rows"]==500000
    start=time.monotonic();children=[];identities=[];known_family={};records=[]
    def guard():
        s=resources()
        assert s["commit"]>=limits["runtime_commit"]and s["physical"]>=limits["runtime_physical"]
        assert s["private"]<=limits["self_private"]and s["E_free"]>=100000000000
        assert time.monotonic()-start<limits["wall_seconds"]
        family=s["private"]
        for index,child in enumerate(children):
            try:
                parent=psutil.Process(child.pid)
                assert parent.create_time()==identities[index]["ctime"]
                members=[parent]+parent.children(recursive=True)
                assert len(members)<=8
                for member in members:
                    identity=who(member)
                    prior=known_family.get(member.pid)
                    if prior is not None:assert prior==identity
                    known_family[member.pid]=identity
                    family+=member.memory_info().private
            except psutil.NoSuchProcess:
                if child.poll()is None:raise
        assert family<=limits["family_private_bytes"]
    def wait(predicate,seconds):
        end=time.monotonic()+seconds
        while not predicate():
            guard()
            if time.monotonic()>=end:raise TimeoutError("Fixture condition deadline")
            time.sleep(.1)
    def config_progress(folder,journal,step):
        save(folder/"progress/latest.json",dict(actual_optimizer_updates=step,journal_prefix=dict(path=str(journal)),producer=dict(pid=os.getpid(),create_time=psutil.Process().create_time()),last_saved_updates=0,native_updates_this_process=step,native_seconds=float(step),active_pipeline_seconds=float(step),source_audio_seconds=float(step),audio_minutes_per_wall_minute=1.,optimizer_poisoned=False,schedule_poisoned=False))
    error=None
    try:
        if execution_phase in ("publication_only","focused_real_server_STATE"):
            records.append(publication_negatives(root,guard))
        if execution_phase=="focused_real_server_STATE":
            records.append(focused_real_server_STATE(root,scope,guard,wait,config_progress,children,identities))
        for shape in (() if execution_phase in ("publication_only","focused_real_server_STATE") else (required_shape,)):
            guard()
            case=root/shape;case.mkdir();area=case/"area";run=case/"run";area.mkdir();run.mkdir();(run/"progress").mkdir()
            dbdir=area/"database";dbdir.mkdir()
            project="fixture_"+shape;name="fixture_training"
            os.environ["TRACKIO_DIR"]=str(dbdir)
            for key in ("TRACKIO_SPACE_ID","TRACKIO_SERVER_URL","TRACKIO_DATASET_ID","TRACKIO_BUCKET_ID","TRACKIO_WEBHOOK_URL"):os.environ.pop(key,None)
            import orjson,sqlite3
            from trackio.sqlite_storage import SQLiteStorage
            from trackio.utils import on_spaces
            assert not on_spaces(),"Exact local SQLite fixture required; no remote backend"
            db=SQLiteStorage.init_db(project)
            save(case/"actual_fixture_DB_path_binding.json",dict(
                shape=shape,actual_returned_DB_path=str(db.resolve()),
                expected_DB_path=str((dbdir/(project+".db")).resolve()),
                TRACKIO_DIR=os.environ.get("TRACKIO_DIR"),strict_isolation_check_unchanged=True))
            assert db.resolve()==(dbdir/(project+".db")).resolve()
            # Real official schema; owned diagnostic history only.
            conn=sqlite3.connect(db)
            try:
                for lo in range(0,500000,1000):
                    guard();rows=[]
                    for i in range(lo,lo+1000):
                        order=i if shape=="metadata_live_order"else(500000-i if i%2 else i//2)
                        metrics={"train/objective":1.+i%17,"train/grad":.25,"progress/update":i}
                        if i%997==0:metrics.update({"dev/wer":25.,"bool":True,"hist":{"_type":"histogram","values":[1,2]}})
                        stamp=f"2026-01-{1+order//86400:02d}T{(order//3600)%24:02d}:{(order//60)%60:02d}:{order%60:02d}.000000"
                        rows.append((name,stamp,name,i,orjson.dumps(metrics).decode()))
                    conn.executemany("INSERT INTO metrics(run_id,timestamp,run_name,step,metrics)VALUES(?,?,?,?,?)",rows)
                conn.commit()
            finally:conn.close()
            journal=run/"fixture_journal.jsonl";journal.write_bytes(b"")
            config_progress(run,journal,1);save(run/"latest_checkpoint.json",{"committed_updates":0})
            with socket.socket()as sock:sock.bind(("127.0.0.1",0));port=sock.getsockname()[1]
            binding=case/"bindings.json"
            save(binding,dict(kind="Trackio7_owned_E_fixture_bindings_v1",fixture_root=str(case),area=str(area),run=str(run),source_db=str(case/"unused_source.db"),project=project,run_name=name,port=port))
            out=(case/"stdout.log").open("xb");err=(case/"stderr.log").open("xb")
            argv=[sys.executable,"-X","utf8","-B",str(Path(__file__).resolve()),"--server-child",str(binding)]
            child=subprocess.Popen(argv,stdout=out,stderr=err,cwd=HERE,creationflags=subprocess.CREATE_NO_WINDOW)
            out.close();err.close();children.append(child);identities.append(who(psutil.Process(child.pid)))
            wait(lambda:(case/"child_entry.json").exists(),10)
            status_poll_errors=[];status_poll_matches=[];status_poll_attempts=0
            def wait_status(label,predicate,seconds,exists_first=False):
                # A single open per EXISTING wait tick; no inner retry or extra sleep.
                matched=[]
                def probe():
                    nonlocal status_poll_attempts
                    status_poll_attempts+=1
                    if status_poll_attempts>2048:raise AssertionError("Bounded status polling probes")
                    target=area/"current_status.json"
                    if exists_first and not target.exists():return False
                    try:
                        stream=target.open("r",encoding="utf-8-sig")
                    except PermissionError as caught:
                        winerror=getattr(caught,"winerror",None)
                        if os.name!="nt"or not(winerror in(5,32,33)or(winerror is None and caught.errno==13)):raise
                        if len(status_poll_errors)>=256:raise AssertionError("Bounded status-open errors")
                        status_poll_errors.append(dict(label=label,attempt=status_poll_attempts,exception_identity=id(caught),type=type(caught).__name__,errno=caught.errno,winerror=winerror,path=str(target),time=time.monotonic()))
                        return False
                    # Stream read/decode/JSON errors remain fatal, outside the open-only catch.
                    with stream:value=json.load(stream)
                    if not isinstance(value,dict):raise AssertionError("Complete status object required")
                    if not predicate(value):return False
                    matched.append(value)
                    return True
                wait(probe,seconds) # Original outer deadline, .1s cadence, and resource guard.
                assert len(matched)==1
                status_poll_matches.append(dict(label=label,status=matched[0]))
                return matched[0]
            ready_status=wait_status("reader_ready",lambda status:status.get("reader_ready")is True,60,exists_first=True)
            token=(area/"local_write_token.txt").read_text().strip();base="http://127.0.0.1:"+str(port)
            payload=dict(project=project,runs=[dict(run=name,run_id=name)],max_points=3000,scalar_only=True)
            def reference(runs):
                # Original complete fetch/samplers, without a stale parent-process read cache.
                with sqlite3.connect(db)as reference_conn:
                    reference_conn.row_factory=sqlite3.Row
                    result=[]
                    for r in runs:
                        identity=SQLiteStorage._resolve_run_identity(reference_conn,run_name=r.get("run"),run_id=r.get("run_id"))
                        logs=[]if identity is None else SQLiteStorage._fetch_metric_logs_with_cursor(reference_conn.cursor(),identity,3000,scalar_only=True)
                        result.append(dict(run=r.get("run"),run_id=r.get("run_id"),logs=logs))
                    return result
            # First actual get_logs_batch is COLD and released in the same burst
            # as eight scalar clients, append64, and the existing GET/version.
            cold_status=ready_status # Same complete pre-request readiness snapshot; no query ran between.
            cold_projection=cold_status["scalar_query_projection"]
            assert cold_projection["calls"]==0 and cold_projection["source_rows_decoded"]==0
            extra=[dict(project=project,run=name,run_id=name,step=500001+i,metrics={"append/value":i},log_id="api_append_"+str(i))for i in range(64)]
            burst_gate=threading.Barrier(11) # Ten real requests plus controller.
            burst_events=[]
            burst_events_lock=threading.Lock()
            burst_origin=time.monotonic()
            save(case/"cold_burst_requests.json",dict(expected_requests=10,events=[]))
            def burst_request(label,api_name,kwargs=None):
                burst_gate.wait(timeout=10)
                began=time.monotonic()
                failure=None
                try:
                    answer,actual_elapsed=request(base,api_name,kwargs,token)
                    # Do not retain concurrent response graphs; independent original
                    # equivalence is checked after ALL requests have settled.
                    del answer
                    return dict(label=label,elapsed_seconds=actual_elapsed)
                except BaseException as caught:
                    failure=dict(type=type(caught).__name__,message=str(caught)[:500])
                    raise
                finally:
                    ended=time.monotonic()
                    event=dict(label=label,API=api_name,start_seconds=began-burst_origin,
                        end_seconds=ended-burst_origin,elapsed_seconds=ended-began,error=failure)
                    with burst_events_lock:
                        assert len(burst_events)<10
                        burst_events.append(event)
                        save(case/"cold_burst_requests.json",dict(expected_requests=10,events=list(burst_events)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=10)as pool:
                scalar_futures=[pool.submit(burst_request,"scalar_"+str(i),"get_logs_batch",payload)for i in range(8)]
                write_future=pool.submit(burst_request,"append64","bulk_log",dict(logs=extra,hf_token=None))
                version_future=pool.submit(burst_request,"version","/version")
                burst_gate.wait(timeout=10)
                concurrent_results=[future.result(timeout=20)for future in scalar_futures]
                write_result=write_future.result(timeout=20)
                version_result=version_future.result(timeout=20)
            assert len(burst_events)==10 and all(e["error"]is None and e["elapsed_seconds"]<20 for e in burst_events)
            scalar_events=[e for e in burst_events if e["API"]=="get_logs_batch"]
            write_event=next(e for e in burst_events if e["label"]=="append64")
            version_event=next(e for e in burst_events if e["label"]=="version")
            assert len(scalar_events)==8
            # Observed CLIENT intervals only, no claim about exclusive backend CPU.
            assert any(e["start_seconds"]<=write_event["end_seconds"]and e["end_seconds"]>=write_event["start_seconds"]for e in scalar_events)
            assert any(e["start_seconds"]<=version_event["end_seconds"]and e["end_seconds"]>=version_event["start_seconds"]for e in scalar_events)
            status_before=wait_status("cold_history_decoded",lambda status:status.get("scalar_query_projection",{}).get("source_rows_decoded",0)>=500000,15)
            decoded_before=status_before["scalar_query_projection"]["source_rows_decoded"]
            # No original reference query or isolated dashboard query warmed the
            # first burst. Compare only at the settled post-burst fence.
            after_expected=reference([dict(run=name,run_id=name)])
            after_actual,after_elapsed=request(base,"get_logs_batch",payload,token);assert after_actual==after_expected
            # Name-only identity and explicit fallback use official reference, too.
            for runs in ([dict(run=name,run_id=None)],[dict(run="fixture_absent",run_id="fixture_absent")]):
                body=dict(payload,runs=runs)
                ref=reference(runs)
                got,_=request(base,"get_logs_batch",body,token);assert got==ref
            try:request(base,"bulk_log",dict(logs=extra,hf_token=None),token=None)
            except urllib.error.HTTPError as auth_error:assert auth_error.code==400 # Original TrackioAPIError mapping
            else:raise AssertionError("Official write auth unexpectedly bypassed")
            try:request(base,"get_logs_batch",{},token)
            except urllib.error.HTTPError as argument_error:assert argument_error.code==400
            else:raise AssertionError("Official missing argument error lost")
            # A real DB writer lock makes the real mirror HTTP fail. No mocked API.
            blocker=sqlite3.connect(db);blocker.execute("BEGIN EXCLUSIVE")
            old_state=read(area/"mirror_state.json")
            raw=b"".join((json.dumps(dict(kind="ASR18_committed_training_update_v1",optimizer_updates=i,native=dict(mean_objective=1.,global_grad_norm=.25),broad_draws=14*i,replay_updates=i,cleaner_draws=0))+"\n").encode()for i in range(1,65))
            with journal.open("ab")as f:f.write(raw);f.flush();os.fsync(f.fileno())
            config_progress(run,journal,65)
            try:
                failed=wait_status("real_DB_lock_error",lambda status:status.get("error")is not None,35)
                assert failed["mirror_thread_alive"]is True and failed["pending_batch"]["count"]==64
                assert read(area/"mirror_state.json")==old_state
            finally:blocker.rollback();blocker.close()
            wait(lambda:read(area/"mirror_state.json").get("offset")==len(raw),35)
            # Stable per-line IDs and exactly64 persisted journal metrics despite timeout replay.
            ids=[hashlib.sha256((name+":journal:"+hashlib.sha256(line).hexdigest()).encode()).hexdigest()for line in raw.splitlines(keepends=True)]
            with sqlite3.connect(db)as check:
                count=check.execute("SELECT COUNT(*)FROM metrics WHERE log_id IN("+",".join("?"for _ in ids)+")",ids).fetchone()[0]
            assert count==64
            # Inject local STATE publication failure AFTER a real HTTP acknowledgement.
            checkpoint_state=read(area/"mirror_state.json")
            raw2=b"".join((json.dumps(dict(kind="ASR18_committed_training_update_v1",optimizer_updates=i,native=dict(mean_objective=1.,global_grad_norm=.25),broad_draws=14*i,replay_updates=i,cleaner_draws=0))+"\n").encode()for i in range(65,129))
            save(case/"fail_next_STATE_publication.json",dict(when_offset=len(raw)+len(raw2)))
            with journal.open("ab")as f:f.write(raw2);f.flush();os.fsync(f.fileno())
            config_progress(run,journal,129)
            try:
                wait(lambda:(case/"STATE_publication_failure_observed.json").exists(),20)
                failed_status=wait_status("stub_STATE_error",lambda status:"local STATE publication failed"in str(status.get("error")),10)
                fault=read(case/"STATE_publication_failure_observed.json")
                at_fence=read(area/"mirror_state.json")
                assert fault["original_STATE_write_called"]is False
                assert fault["attempted_offset"]==len(raw)+len(raw2)
                assert at_fence==fault["pre_STATE"]
                # Journal frontiers are independent of unrelated progress_signature.
                assert {k:fault["pre_STATE"].get(k)for k in ("journal_path","offset","last_update")}=={k:checkpoint_state.get(k)for k in ("journal_path","offset","last_update")}
                assert fault["pre_STATE"]["journal_path"]==str(journal)and fault["pre_STATE"]["offset"]==len(raw)and fault["pre_STATE"]["last_update"]==64
                pending=failed_status["pending_batch"]
                assert pending==dict(journal_path=str(journal),start_offset=len(raw),end_offset=len(raw)+len(raw2),last_update=128,count=64)
                assert failed_status["mirror_thread_alive"]is True
                save(case/"STATE_failure_fence_assertion.json",dict(
                    exact_pre_write_STATE=fault["pre_STATE"],actual_STATE_at_error_fence=at_fence,
                    attempted_STATE=fault["attempted_STATE"],pending_batch=pending,
                    journal_frontier_unchanged=True,original_STATE_write_called=False,
                    scientific_owner="METHOD_STUB observation fence only"))
            finally:
                save(case/"STATE_failure_fence_release.json",dict(release_owned_METHOD_STUB_fence=True))
            wait(lambda:read(area/"mirror_state.json").get("offset")==len(raw)+len(raw2),25)
            # Explicit response-loss negative; real official HTTP call runs first.
            raw3=b"".join((json.dumps(dict(kind="ASR18_committed_training_update_v1",optimizer_updates=i,native=dict(mean_objective=1.,global_grad_norm=.25),broad_draws=14*i,replay_updates=i,cleaner_draws=0))+"\n").encode()for i in range(129,193))
            first_id=hashlib.sha256((name+":journal:"+hashlib.sha256(raw3.splitlines(keepends=True)[0]).hexdigest()).encode()).hexdigest()
            checkpoint_state=read(area/"mirror_state.json")
            save(case/"lose_next_mirror_response.json",dict(when_log_id=first_id))
            with journal.open("ab")as f:f.write(raw3);f.flush();os.fsync(f.fileno())
            config_progress(run,journal,193)
            try:
                wait(lambda:(case/"response_loss_observed.json").exists(),20)
                failed_status=wait_status("lost_HTTP_response_error",lambda status:"response lost"in str(status.get("error")),10)
                loss=read(case/"response_loss_observed.json")
                at_fence=read(area/"mirror_state.json")
                assert loss["original_HTTP200_consumed"]is True and loss["intercepted_log_id"]==first_id
                assert at_fence==loss["pre_STATE"]
                assert {k:loss["pre_STATE"].get(k)for k in ("journal_path","offset","last_update")}=={k:checkpoint_state.get(k)for k in ("journal_path","offset","last_update")}
                assert loss["pre_STATE"]["journal_path"]==str(journal)and loss["pre_STATE"]["offset"]==len(raw)+len(raw2)and loss["pre_STATE"]["last_update"]==128
                pending=failed_status["pending_batch"]
                assert pending==dict(journal_path=str(journal),start_offset=len(raw)+len(raw2),end_offset=len(raw)+len(raw2)+len(raw3),last_update=192,count=64)
                assert failed_status["mirror_thread_alive"]is True
                save(case/"response_loss_fence_assertion.json",dict(
                    exact_pre_lost_response_STATE=loss["pre_STATE"],actual_STATE_at_error_fence=at_fence,
                    pending_batch=pending,journal_frontier_unchanged=True,
                    original_HTTP200_consumed=True,SQLite_durable_commit_from_HTTP200_alone="NOT_CLAIMED",
                    scientific_owner="METHOD_STUB observation fence only"))
            finally:
                save(case/"response_loss_fence_release.json",dict(release_owned_METHOD_STUB_fence=True))
            wait(lambda:read(area/"mirror_state.json").get("offset")==len(raw)+len(raw2)+len(raw3),25)
            all_ids=[hashlib.sha256((name+":journal:"+hashlib.sha256(line).hexdigest()).encode()).hexdigest()for line in (raw+raw2+raw3).splitlines(keepends=True)]
            with sqlite3.connect(db)as check:
                all_count=check.execute("SELECT COUNT(*)FROM metrics WHERE log_id IN("+",".join("?"for _ in all_ids)+")",all_ids).fetchone()[0]
            assert all_count==192
            # REAL cross-process Windows STATE contention after the original192 tests.
            raw4=b"".join((json.dumps(dict(kind="ASR18_committed_training_update_v1",optimizer_updates=i,native=dict(mean_objective=1.,global_grad_norm=.25),broad_draws=14*i,replay_updates=i,cleaner_draws=0))+"\n").encode()for i in range(193,257))
            target_offset=len(raw)+len(raw2)+len(raw3)+len(raw4)
            save(case/"Windows_STATE_publication_gate.json",dict(when_offset=target_offset))
            with journal.open("ab")as f:f.write(raw4);f.flush();os.fsync(f.fileno())
            config_progress(run,journal,257)
            held=None
            try:
                wait(lambda:(case/"Windows_STATE_gate_ready.json").exists(),15)
                held=deny_delete(area/"mirror_state.json")
                save(case/"Windows_STATE_gate_release.json",dict(release_to_actual_candidate_write=True))
                wait(lambda:(case/"Windows_STATE_failure_observed.json").exists(),5)
                failed_status=wait_status("real_Windows_STATE_error",lambda status:status.get("error")is not None,5)
                actual_fault=read(case/"Windows_STATE_failure_observed.json")
                actual_state=read(area/"mirror_state.json")
                save(case/"Windows_STATE_before_assert.json",dict(fault=actual_fault,STATE=actual_state,status=failed_status))
                assert actual_fault["original_candidate_write_called"]is True and actual_fault["winerror"]in(5,32,33)
                assert actual_state==actual_fault["pre_STATE"]
                assert actual_state["offset"]==len(raw)+len(raw2)+len(raw3)and actual_state["last_update"]==192
                assert failed_status["mirror_thread_alive"]is True and failed_status["pending_batch"]==dict(
                    journal_path=str(journal),start_offset=actual_state["offset"],end_offset=target_offset,last_update=256,count=64)
            finally:
                if held is not None:close_deny_delete(held)
                save(case/"Windows_STATE_failure_fence_release.json",dict(release_owned_observation_fence=True))
                if not(case/"Windows_STATE_gate_release.json").exists():
                    save(case/"Windows_STATE_gate_release.json",dict(release_owned_gate_on_all_paths=True))
            wait(lambda:read(area/"mirror_state.json").get("offset")==target_offset,25)
            every_id=[hashlib.sha256((name+":journal:"+hashlib.sha256(line).hexdigest()).encode()).hexdigest()for line in (raw+raw2+raw3+raw4).splitlines(keepends=True)]
            with sqlite3.connect(db)as check:
                windows_count=check.execute("SELECT COUNT(*)FROM metrics WHERE log_id IN("+",".join("?"for _ in every_id)+")",every_id).fetchone()[0]
            assert windows_count==256
            request(base,"get_logs_batch",payload,token)
            live_status=wait_status("final_live_after_replay",lambda status:status.get("status")=="live"and status.get("error")is None,15)
            decoded_after=live_status["scalar_query_projection"]["source_rows_decoded"]
            # Cold history decoded once; later source growth is bounded to appended new rows.
            assert decoded_after-decoded_before<=512
            save(case/"fixture_stop.json",{"stop_owned_fixture_only":True})
            wait(lambda:child.poll()is not None,40);assert child.returncode==0
            closure=read(case/"child_method_closure.json");assert closure["real_API_and_mirror_close_returned"]is True
            with socket.socket()as released:released.bind(("127.0.0.1",port))
            terminal_status=read(area/"current_status.json") # Unsuppressed, after normal child wait0/method close.
            assert isinstance(terminal_status,dict)
            save(case/"status_polling_open_witness.json",dict(attempts=status_poll_attempts,open_errors=status_poll_errors,matched_complete_statuses=status_poll_matches,terminal_complete_status=terminal_status,original_wait_deadlines_and_cadence_unchanged=True,open_only_Windows_PermissionError_classification=True,stream_decode_JSON_errors_FATAL=True,normal_child_wait0=True,no_inner_retry_or_added_wait=True))
            records.append(dict(shape=shape,original_history_rows=500000,queries=8,first_actual_get_logs_batch_in_cold_concurrent_burst=True,cold_burst_every_request_elapsed=list(burst_events),cold_burst_query_write_and_version_client_overlap=True,all_normal_requests_below20=True,quiescent_original_equivalence=True,DB_lock_timeout_then_potential_late_original_commit_and_idempotent_replay=True,local_STATE_publication_failure_preserves_committed_STATE=True,explicit_lost_response_exact_batch_replayed=True,all192_journal_rows_unique=all_count,real_Windows_STATE_failure_preserves_pending_and_offset=True,all256_journal_ids_unique_after_real_Windows_replay=windows_count,idempotent64_rows=count,warm_added_source_rows=decoded_after-decoded_before,actual_child_identity=read(case/"child_entry.json"),launcher_identity=identities[-1],Popen_wait=child.wait(),closure=closure,port_released=True))
        if execution_phase in ("original_metadata_shape","original_adverse_shape"):records.append(source_bound_lifetime_negatives(binding,scope))
        if execution_phase in ("original_metadata_shape","original_adverse_shape"):
            # Source-bind exact official caller/Gradio delegation, not a guessed route.
            import ast,trackio.asgi_app as asgi
            tree=ast.parse(Path(asgi.__file__).read_text(encoding="utf-8"))
            calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=="_invoke_handler"]
            assert len(calls)==1
            records.append(dict(label="official_Gradio_delegates_run_api_request_and_single_awaiting_dispatch_SOURCE_ASSERTION",scientific_owner="SOURCE_ONLY; live local Gradio route not fabricated"))
    except BaseException as caught:
        error=dict(type=type(caught).__name__,message=str(caught)[:2000])
        raise
    finally:
        cleanup=[]
        for i,child in enumerate(children):
            if child.poll()is None:
                # Only known owned fixture; never target readers/trainers.
                p=psutil.Process(child.pid)
                if p.create_time()!=identities[i]["ctime"]:raise RuntimeError("Owned child identity changed")
                save(root/("forced_cleanup_"+str(i)+".json"),dict(known_identity=identities[i],method_close="UNKNOWN",reason="fixture failed before normal close"))
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:cleanup.append(dict(identity=identities[i],exact_death="UNKNOWN"))
        deaths=[]
        for identity in known_family.values():
            try:
                current=psutil.Process(identity["pid"])
                alive=current.is_running()and current.create_time()==identity["ctime"]
            except psutil.NoSuchProcess:alive=False
            deaths.append(dict(identity=identity,exact_dead=not alive))
        save(root/"result.json",dict(passed=error is None,error=error,cases=records,children=identities,known_family=deaths,cleanup=cleanup,wall=time.monotonic()-start,production_touched=False,no_durable_live_cause_or_speed_claim=True,prior_matching_candidate_first_shape_ROOT=prior_matching_pin,required_original_shape=required_shape,candidate_reader=scope["candidate"],projection_source=projection_pin,this_run_one_explicit_original_shape=execution_phase in ("original_metadata_shape","original_adverse_shape"),execution_phase=execution_phase,accepted_metadata_main_ROOT=main_shape_pin,completed_metadata500000_main_rerun=False,historical_reader3_first_shape_NOT_activation_authority=True,historical_reader4_MAIN_NOT_newcandidate_authority=True,historical_reader7_API08_API09_NOT_reader8_activation_authority=True,focused_publication_only_no_full_API_claim=execution_phase=="publication_only",focused_real_server_no_full500k_activation_claim=execution_phase=="focused_real_server_STATE"))


def controller(scopepath):
    supervisor_path=HERE/"owned_source_CPU_proof_watchdog_v1.py"
    scope=read(scopepath)
    assert scope["watchdog_source"]==pin(supervisor_path)
    supervisor=load(supervisor_path,"owned_reader_proof_watchdog")
    return supervisor.execute(scopepath,Path(__file__).resolve(),"reader")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--ROOT-review");parser.add_argument("--server-child");parser.add_argument("--proof-body",action="store_true")
    args=parser.parse_args()
    if args.server_child:
        server_child(args.server_child)
    elif args.proof_body:
        assert args.ROOT_review
        fixture_body(args.ROOT_review)
    else:
        assert args.ROOT_review
        controller(args.ROOT_review)
