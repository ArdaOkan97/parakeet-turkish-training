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
CANDIDATE=HERE/"serve_BPE9223_Trackio_live_v7_INACTIVE_v3.py"
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
    try:
        reader=load(CANDIDATE,"owned_reader7_candidate")
        sys.argv=[str(CANDIDATE),"--fixture-bindings",str(bindings)]
        reader.main()
        # main returned only after actual mirror.join and API.close completed.
        names=[t.name for t in threading.enumerate()if t.name.startswith("Trackio-owned-")or t.name=="read-only-training-mirror"]
        if names:raise AssertionError("Owned reader method closure incomplete")
        save(root/"child_method_closure.json",dict(entry=entry,reader_main_returned=True,owned_threads_remaining=names,real_API_and_mirror_close_returned=True))
    finally:
        sys.argv=argv;uvicorn.Server.__init__=original_init
        stop.set();thread.join(2)
        if thread.is_alive():raise RuntimeError("Fixture stop thread did not join")


def source_bound_lifetime_negatives(binding,scope):
    """Actual helper methods/reader main; scientific numerical owners absent.
    Pure headroom branch uses explicitly labelled metadata owners.
    """
    import ast,asyncio,http.server,types,weakref
    import trackio.asgi_app as asgi,trackio.server as official
    helper=load(HERE/"Trackio_owned_bounded_sync_API_dispatch_v3.py","fixture_API_helper")
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
        try:
            request_owner=RequestOwner();reference=weakref.ref(request_owner)
            first=asyncio.create_task(asgi._invoke_handler(owned_function,request_owner,kwargs={"which":0}))
            while not enter.is_set():await asyncio.sleep(.001)
            first.cancel()
            try:await first
            except asyncio.CancelledError:pass
            second=asyncio.create_task(asgi._invoke_handler(owned_function,request_owner,kwargs={"which":1}))
            await asyncio.sleep(.02);assert calls==[0]
            release.set();assert await second==1
            try:await asgi._invoke_handler(owned_function,request_owner,kwargs={"which":2})
            except ValueError as error:assert error is marker
            else:raise AssertionError("Original exception masked")
            marker.__traceback__=None
            del first,second,request_owner
            await asyncio.sleep(0);await asyncio.sleep(0)
            assert reference()is None
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
    app,_=official.build_starlette_app_only(mcp_server=False)
    d=helper.OwnedDispatch(asgi,official,sources,lambda:None)
    async def affected_Gradio():
        binding_value=read(binding)
        payload=dict(kwargs=dict(project=binding_value["project"],runs=[dict(run=binding_value["run_name"],run_id=binding_value["run_name"])],max_points=3000,scalar_only=True))
        def req(api_name,event_id=None):
            data=json.dumps(payload).encode();sent=[False]
            async def receive():
                if sent[0]:return {"type":"http.disconnect"}
                sent[0]=True;return {"type":"http.request","body":data,"more_body":False}
            path=dict(api_name=api_name)
            if event_id is not None:path["event_id"]=event_id
            return Request(dict(type="http",method="POST",path="/fixture",headers=[(b"content-type",b"application/json")],query_string=b"",path_params=path,app=app),receive)
        normal=await asyncio.wait_for(asgi.run_api_request(req("get_logs_batch"),"get_logs_batch"),20)
        assert normal.status_code==200
        expected=json.loads(normal.body)["data"]
        post=await asyncio.wait_for(asgi.gradio_call_post_handler(req("get_logs_batch")),20)
        assert post.status_code==200
        event=json.loads(post.body)["event_id"]
        poll=await asgi.gradio_call_poll_handler(req("get_logs_batch",event))
        body=b""
        async for chunk in poll.body_iterator:body+=chunk.encode()if isinstance(chunk,str)else chunk
        data_line=next(line[6:]for line in body.decode().splitlines()if line.startswith("data: "))
        assert json.loads(data_line)==[expected]
    try:asyncio.run(affected_Gradio())
    finally:d.close()

    # Actual reader2 first-headroom terminal branch, metadata/HTTP owners doubled.
    tree=ast.parse(CANDIDATE.read_text(encoding="utf-8"))
    functions={n.name:n for n in ast.walk(tree)if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    selected=[functions[n]for n in ("DashboardMemoryLimit","local_status","mirror_body","mirror")]
    namespace=dict(threading=threading,time=time,identity={"fixture":True},mirror_activity={"HTTP_ready":False,"committed_HTTP_batch":False},pending_holder={"value":None},terminal_holder={"error":None},server=types.SimpleNamespace(should_exit=False),STATE=Path(binding).with_name("never_exists_state.json"),STATUS=Path(binding).with_name("captured_status.json"),memory=lambda:{},api_dispatch=types.SimpleNamespace(stats={}),dispatch_module=types.SimpleNamespace(wait_http_ready=lambda *a,**k:None))
    statuses=[];namespace["write"]=lambda path,value:statuses.append(value)
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(CANDIDATE),"exec"),namespace)
    headroom=namespace["DashboardMemoryLimit"]("fixture original headroom cause")
    def fail_headroom():raise headroom
    namespace["headroom"]=fail_headroom
    namespace["mirror"]()
    assert statuses[-1]["status"]=="stopped_headroom"and "fixture original headroom cause"in statuses[-1]["error"]
    assert statuses[-1]["mirror_thread_alive"]is False
    return dict(post_both_workers_hook_install_failure_unwind=True,second_start_unwind=True,owner_write_unwind=True,mirror_start_unwind=True,cancellation_keeps_slot_until_actual_completion=True,original_exception_identity=True,request_weakref_released=True,fixture_port7860_rejected=True,foreign_listener_received_no_request_before_own_bind=True,remaining_deadline_bound=True,actual_official_Gradio_post_poll_and_API_same_result=True,headroom_first_cause_preserved=True,headroom_owner="METHOD_STUB metadata and HTTP; actual reader branch AST")


def controller(scopepath):
    import psutil
    scope=read(scopepath)
    assert scope["accepted"]is True and scope["reviewer"]=="/root"
    assert scope["kind"]=="ROOT_Trackio7_real_API_mirror_integration_exact_ONE_CPU_scope_v1"
    root=under(scope["output_directory"],HERE);assert not root.exists();root.mkdir()
    limits=scope["resources"]
    # All original reservations plus new cache/executor/fixture family must be explicit.
    for k in ("prospective_commit","prospective_physical","runtime_commit","runtime_physical","family_private","self_private","wall_seconds"):
        assert type(limits[k])is int and limits[k]>0
    assert limits["prospective_commit"]>=5399239296+16777216
    assert limits["prospective_physical"]>=2649239296+16777216
    assert limits["self_private"]<=536870912 and limits["wall_seconds"]<=180
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    first=resources();save(root/"initial_BEFORE_predicate.json",dict(sample=first,entry=who(psutil.Process()),launcher=who(psutil.Process().parent())))
    assert first["commit"]>=limits["prospective_commit"]and first["physical"]>=limits["prospective_physical"]
    assert first["private"]<=limits["self_private"]and first["E_free"]>=100000000000
    for item in scope["input_pins"]:assert pin(item["path"])==item
    assert scope["candidate"]==pin(CANDIDATE)
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
        assert family<=limits["family_private"]
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
        for shape in ("metadata_live_order","adverse_reverse_interleaved"):
            guard()
            case=root/shape;case.mkdir();area=case/"area";run=case/"run";area.mkdir();run.mkdir();(run/"progress").mkdir()
            dbdir=area/"database";dbdir.mkdir()
            project="fixture_"+shape;name="fixture_training"
            os.environ["TRACKIO_DIR"]=str(dbdir)
            for key in ("TRACKIO_SPACE_ID","TRACKIO_SERVER_URL","TRACKIO_DATASET_ID","TRACKIO_BUCKET_ID","TRACKIO_WEBHOOK_URL"):os.environ.pop(key,None)
            import orjson,sqlite3
            from trackio.sqlite_storage import SQLiteStorage
            db=SQLiteStorage.init_db(project)
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
            child=subprocess.Popen(argv,stdout=out,stderr=err,cwd=case,creationflags=subprocess.CREATE_NO_WINDOW)
            out.close();err.close();children.append(child);identities.append(who(psutil.Process(child.pid)))
            wait(lambda:(case/"child_entry.json").exists(),10)
            wait(lambda:(area/"current_status.json").exists()and read(area/"current_status.json").get("reader_ready")is True,60)
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
            expected=reference([dict(run=name,run_id=name)])
            actual,elapsed=request(base,"get_logs_batch",payload,token)
            assert actual==expected
            wait(lambda:read(area/"current_status.json").get("scalar_query_projection",{}).get("source_rows_decoded",0)>=500000,15)
            status_before=read(area/"current_status.json")
            decoded_before=status_before["scalar_query_projection"]["source_rows_decoded"]
            with concurrent.futures.ThreadPoolExecutor(max_workers=9)as pool:
                futures=[pool.submit(request,base,"get_logs_batch",payload,token)for _ in range(8)]
                extra=[dict(project=project,run=name,run_id=name,step=500001+i,metrics={"append/value":i},log_id="api_append_"+str(i))for i in range(64)]
                write_future=pool.submit(request,base,"bulk_log",dict(logs=extra,hf_token=None),token)
                concurrent_results=[f.result(timeout=20)for f in futures]
                write_future.result(timeout=20)
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
                wait(lambda:read(area/"current_status.json").get("error")is not None,35)
                failed=read(area/"current_status.json")
                assert failed["mirror_thread_alive"]is True and failed["pending_batch"]["count"]==64
                assert read(area/"mirror_state.json")==old_state
            finally:blocker.rollback();blocker.close()
            wait(lambda:read(area/"mirror_state.json").get("offset")==len(raw),35)
            # Stable per-line IDs and exactly64 persisted journal metrics despite timeout replay.
            ids=[hashlib.sha256((name+":journal:"+hashlib.sha256(line).hexdigest()).encode()).hexdigest()for line in raw.splitlines(keepends=True)]
            with sqlite3.connect(db)as check:
                count=check.execute("SELECT COUNT(*)FROM metrics WHERE log_id IN("+",".join("?"for _ in ids)+")",ids).fetchone()[0]
            assert count==64
            request(base,"get_logs_batch",payload,token)
            wait(lambda:read(area/"current_status.json").get("status")=="live"and read(area/"current_status.json").get("error")is None,15)
            decoded_after=read(area/"current_status.json")["scalar_query_projection"]["source_rows_decoded"]
            # Cold history decoded once; later source growth is bounded to appended new rows.
            assert decoded_after-decoded_before<=256
            save(case/"fixture_stop.json",{"stop_owned_fixture_only":True})
            wait(lambda:child.poll()is not None,40);assert child.returncode==0
            closure=read(case/"child_method_closure.json");assert closure["real_API_and_mirror_close_returned"]is True
            with socket.socket()as released:released.bind(("127.0.0.1",port))
            records.append(dict(shape=shape,original_history_rows=500000,queries=8,all_normal_requests_below20=True,quiescent_original_equivalence=True,timeout_pending_and_exact_STATE_preserved=True,idempotent64_rows=count,warm_added_source_rows=decoded_after-decoded_before,actual_child_identity=read(case/"child_entry.json"),launcher_identity=identities[-1],Popen_wait=child.wait(),closure=closure,port_released=True))
        records.append(source_bound_lifetime_negatives(binding,scope))
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
        save(root/"result.json",dict(passed=error is None,error=error,cases=records,children=identities,known_family=deaths,cleanup=cleanup,wall=time.monotonic()-start,production_touched=False,no_durable_live_cause_or_speed_claim=True))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--ROOT-review");parser.add_argument("--server-child")
    args=parser.parse_args()
    if args.server_child:
        server_child(args.server_child)
    else:
        assert args.ROOT_review
        controller(args.ROOT_review)
