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
CANDIDATE=HERE/"serve_BPE9223_Trackio_live_v7_INACTIVE_v4.py"
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
                ("response_loss_observed.json","response_loss_fence_release.json")):
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
    return dict(post_both_workers_hook_install_failure_unwind=True,second_start_unwind=True,owner_write_unwind=True,mirror_start_unwind=True,cancellation_keeps_slot_until_actual_completion=True,original_exception_identity=True,request_weakref_released=True,fixture_port7860_rejected=True,foreign_listener_received_no_request_before_own_bind=True,remaining_deadline_bound=True,actual_official_Gradio_post_poll_and_API_same_result=True,official_Spaces_constructor_branch_METHOD_STUB=True,selector_restored_BEFORE_handlers=True,deployed_local_Gradio_route_claimed=False,headroom_first_cause_preserved=True,headroom_owner="METHOD_STUB metadata and HTTP; actual reader branch AST")


def prepare_remaining_lifetime_fixture(root,scope,guard):
    """Fresh <=256 REAL official-schema rows; no accepted500k workload rerun."""
    assert scope["remaining_lifetime_fixture_rows"]==64
    guard()
    case=root/"remaining_lifetime_only";case.mkdir()
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
    save(case/"actual_fixture_DB_path_binding.json",dict(shape="remaining_lifetime_only",
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
    save(case/"remaining_lifetime_fixture_receipt.json",dict(real_original_schema=True,actual_rows=64,
        completed_metadata500000_main_rerun=False,fixture_only_no_training_authority=True))
    guard()
    return binding


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
    assert limits["prospective_commit"]>=8056867200
    assert limits["prospective_physical"]>=5306867200
    assert limits["runtime_commit"]>=5538431744 and limits["runtime_physical"]>=5038431744
    assert limits["family_private"]==limits["family_private_bytes"]==2573741824
    assert limits["self_private"]<=536870912 and limits["wall_seconds"]==limits["proof_body_wall_seconds"]<=160
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    first=resources();save(root/"initial_BEFORE_predicate.json",dict(sample=first,entry=who(psutil.Process()),launcher=who(psutil.Process().parent())))
    assert first["commit"]>=limits["prospective_commit"]and first["physical"]>=limits["prospective_physical"]
    assert first["private"]<=limits["self_private"]and first["E_free"]>=100000000000
    for item in scope["input_pins"]:assert pin(item["path"])==item
    assert scope["candidate"]==pin(CANDIDATE)
    execution_phase=scope["execution_phase"]
    assert execution_phase in ("remaining_lifetime_only","original_adverse_shape")
    projection_pin=pin(PROJECTION)
    assert scope["projection_source"]==projection_pin
    assert projection_pin in scope["input_pins"]
    main_shape_pin=scope["accepted_metadata_main_ROOT"]
    assert main_shape_pin=={'path': 'E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_recurrent_metadata_slowdown_fix_v1\\one_postDEV_gc_refresh_v1\\next200000_throughput_repair_v1\\next212000_failure_repair_v1\\ROOT_projection2_reader4_API05_mainshape_partial_acceptance_v1\\acceptance.json', 'bytes': 8177, 'sha256': '55fd4f475438e52d239ba7b23cf0e5b083f590011aef2e0f733933b594d5891b'}
    assert main_shape_pin in scope["input_pins"]and pin(main_shape_pin["path"])==main_shape_pin
    main_shape=read(main_shape_pin["path"])
    assert main_shape["kind"]=="ROOT_Trackio7_projection2_reader4_first_original_MAIN_shape_partial_acceptance_v1"
    assert main_shape["accepted_main_shape"]is True and main_shape["reviewer"]=="/root"
    assert main_shape["candidate_reader"]==scope["candidate"]and main_shape["projection_source"]==projection_pin
    assert main_shape["completed_first_shape"]["shape"]=="metadata_live_order"
    assert main_shape["completed_first_shape"]["original_history_rows"]==500000
    required_shape=scope["required_original_shape"]
    prior_matching_pin=scope["prior_matching_candidate_first_shape_ROOT"]
    if execution_phase=="remaining_lifetime_only":
        assert required_shape is None and prior_matching_pin is None
        assert scope["remaining_lifetime_fixture_rows"]==64
    else:
        # Only the still-required adverse500k may run; accepted metadata MAIN
        # cannot be rerun, and its PARTIAL receipt cannot authorize adverse.
        assert required_shape=="adverse_reverse_interleaved"
        assert type(prior_matching_pin)is dict and prior_matching_pin in scope["input_pins"]
        assert pin(prior_matching_pin["path"])==prior_matching_pin
        matching=read(prior_matching_pin["path"])
        assert matching["kind"]=="ROOT_Trackio7_projection2_reader4_first_original_shape_actual_acceptance_v1"
        assert matching["accepted_first_shape"]is True and matching["reviewer"]=="/root"
        assert matching["candidate_reader"]==scope["candidate"]
        assert matching["projection_source"]==projection_pin
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
        if execution_phase=="remaining_lifetime_only":
            binding=prepare_remaining_lifetime_fixture(root,scope,guard)
        for shape in (() if execution_phase=="remaining_lifetime_only" else (required_shape,)):
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
            # First actual get_logs_batch is COLD and released in the same burst
            # as eight scalar clients, append64, and the existing GET/version.
            cold_status=read(area/"current_status.json")
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
            wait(lambda:read(area/"current_status.json").get("scalar_query_projection",{}).get("source_rows_decoded",0)>=500000,15)
            status_before=read(area/"current_status.json")
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
            # Inject local STATE publication failure AFTER a real HTTP acknowledgement.
            checkpoint_state=read(area/"mirror_state.json")
            raw2=b"".join((json.dumps(dict(kind="ASR18_committed_training_update_v1",optimizer_updates=i,native=dict(mean_objective=1.,global_grad_norm=.25),broad_draws=14*i,replay_updates=i,cleaner_draws=0))+"\n").encode()for i in range(65,129))
            save(case/"fail_next_STATE_publication.json",dict(when_offset=len(raw)+len(raw2)))
            with journal.open("ab")as f:f.write(raw2);f.flush();os.fsync(f.fileno())
            config_progress(run,journal,129)
            try:
                wait(lambda:(case/"STATE_publication_failure_observed.json").exists(),20)
                wait(lambda:"local STATE publication failed"in str(read(area/"current_status.json").get("error")),10)
                fault=read(case/"STATE_publication_failure_observed.json")
                at_fence=read(area/"mirror_state.json")
                failed_status=read(area/"current_status.json")
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
                wait(lambda:"response lost"in str(read(area/"current_status.json").get("error")),10)
                loss=read(case/"response_loss_observed.json")
                at_fence=read(area/"mirror_state.json")
                failed_status=read(area/"current_status.json")
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
            request(base,"get_logs_batch",payload,token)
            wait(lambda:read(area/"current_status.json").get("status")=="live"and read(area/"current_status.json").get("error")is None,15)
            decoded_after=read(area/"current_status.json")["scalar_query_projection"]["source_rows_decoded"]
            # Cold history decoded once; later source growth is bounded to appended new rows.
            assert decoded_after-decoded_before<=512
            save(case/"fixture_stop.json",{"stop_owned_fixture_only":True})
            wait(lambda:child.poll()is not None,40);assert child.returncode==0
            closure=read(case/"child_method_closure.json");assert closure["real_API_and_mirror_close_returned"]is True
            with socket.socket()as released:released.bind(("127.0.0.1",port))
            records.append(dict(shape=shape,original_history_rows=500000,queries=8,first_actual_get_logs_batch_in_cold_concurrent_burst=True,cold_burst_every_request_elapsed=list(burst_events),cold_burst_query_write_and_version_client_overlap=True,all_normal_requests_below20=True,quiescent_original_equivalence=True,DB_lock_timeout_then_potential_late_original_commit_and_idempotent_replay=True,local_STATE_publication_failure_preserves_committed_STATE=True,explicit_lost_response_exact_batch_replayed=True,all192_journal_rows_unique=all_count,idempotent64_rows=count,warm_added_source_rows=decoded_after-decoded_before,actual_child_identity=read(case/"child_entry.json"),launcher_identity=identities[-1],Popen_wait=child.wait(),closure=closure,port_released=True))
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
        save(root/"result.json",dict(passed=error is None,error=error,cases=records,children=identities,known_family=deaths,cleanup=cleanup,wall=time.monotonic()-start,production_touched=False,no_durable_live_cause_or_speed_claim=True,prior_matching_candidate_first_shape_ROOT=prior_matching_pin,required_original_shape=required_shape,candidate_reader=scope["candidate"],projection_source=projection_pin,this_run_one_explicit_original_shape=execution_phase=="original_adverse_shape",execution_phase=execution_phase,accepted_metadata_main_ROOT=main_shape_pin,completed_metadata500000_main_rerun=False,historical_reader3_first_shape_NOT_activation_authority=True))


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
