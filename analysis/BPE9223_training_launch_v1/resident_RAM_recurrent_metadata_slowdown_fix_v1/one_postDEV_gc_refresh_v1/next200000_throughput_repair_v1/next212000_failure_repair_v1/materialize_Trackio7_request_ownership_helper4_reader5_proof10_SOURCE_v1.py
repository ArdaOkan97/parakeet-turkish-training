import ast,ctypes,hashlib,json,os,pathlib,shutil,time
import psutil
P=pathlib.Path(r'C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
D=pathlib.Path(r'E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/next200000_throughput_repair_v1/next212000_failure_repair_v1')
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
class MS(ctypes.Structure):
    _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(n,ctypes.c_ulonglong)for n in ("totalphysical","physical","commitlimit","commit","totalvirtual","virtual","extended")]
def sample():
    m=MS();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError()
    return dict(commit=m.commit,physical=m.physical,private=psutil.Process().memory_info().private,E_free=shutil.disk_usage('E:/gpu_training_scratchpad').free)
def who(p):return dict(pid=p.pid,ctime=p.create_time(),ppid=p.ppid())
def sha(b):return hashlib.sha256(b).hexdigest()
def pin_bytes(p,b):return dict(path=str(p),bytes=len(b),sha256=sha(b))
def pin(p):return pin_bytes(p,p.read_bytes())
def astsha(b):return sha(ast.dump(ast.parse(b.decode('utf-8-sig')),include_attributes=False).encode())
def write_source(p,b):
    ast.parse(b.decode('utf-8-sig'))
    with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
    return pin_bytes(p,b)
def write_json(p,obj):
    with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    return pin(p)
t0=time.monotonic();s=sample()
initial=write_json(D/'Trackio7_request_ownership_proof10_SOURCE_initial_v1.json',dict(kind='SOURCE_preparation_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=s,required=dict(commit=5399239296,physical=2649239296,private_max=536870912,E_free_min=100000000000),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),candidate_import_execution=False))
assert s['commit']>=5399239296 and s['physical']>=2649239296 and s['private']<=536870912 and s['E_free']>=100000000000

import difflib
def raw_diff(old,new,source_path,target_path,labels):
    work=old;records=[];shift=0
    for tag,a,b,c,d in difflib.SequenceMatcher(None,old,new,autojunk=False).get_opcodes():
        if tag=='equal':continue
        at=a+shift;before=work
        assert work[at:at+(b-a)]==old[a:b]
        work=work[:at]+new[c:d]+work[at+(b-a):]
        records.append(dict(offset_before=at,old_utf8=old[a:b].decode(),new_utf8=new[c:d].decode(),before_sha256=sha(before),after_sha256=sha(work)))
        shift+=(d-c)-(b-a)
    assert work==new
    for e in reversed(records):
        at=e['offset_before'];x=e['new_utf8'].encode();y=e['old_utf8'].encode()
        assert work[at:at+len(x)]==x
        work=work[:at]+y+work[at+len(x):]
    assert work==old
    return dict(base=pin_bytes(source_path,old),candidate=pin_bytes(target_path,new),ordered_edits=records,whole_byte_inverse_exact=True,semantic_changes=labels,base_AST_sha256=astsha(old),candidate_AST_sha256=astsha(new))

new_work="    def _work(self):\n        while True:\n            job=self.queue.get()\n            if job is None:\n                self.queue.task_done();del job\n                return\n            future,call=job\n            result=None;failure=None\n            self.lifecycle[\"stage\"]=\"running\"\n            try:\n                try:result=call()\n                except BaseException as error:failure=error\n                finally:\n                    # Drop owned request-bearing closures BEFORE callbacks can\n                    # resume an awaiting/cancelled request on the event loop.\n                    del call,job\n                    self.queue.task_done()\n                    self.lifecycle[\"owned_job_drops\"]+=1\n                    self.lifecycle[\"stage\"]=\"owned_refs_dropped\"\n                if failure is None:future.set_result(result)\n                else:future.set_exception(failure)\n                self.lifecycle[\"publication_returns\"]+=1\n            finally:\n                # Keep the original exception object/traceback for its caller;\n                # release only this worker's local references, on every path.\n                del future,result,failure\n                self.lifecycle[\"stage\"]=\"idle\"\n"
new_cancel="    async def cancellation_and_exception():\n        enter=threading.Event();release=threading.Event();calls=[]\n        old=official.get_logs_batch\n        marker=ValueError(\"fixture original exception identity\")\n        class RequestOwner:pass\n        def owned_function(request,which=0):\n            calls.append(which)\n            if which==0:enter.set();assert release.wait(5)\n            if which==2:raise marker\n            return which\n        official.get_logs_batch=owned_function\n        dispatch=helper.OwnedDispatch(asgi,official,sources,lambda:None)\n        reference=None;task_flags={};exception_identity_seen=False;escaped=None\n        def lifecycle_witness(stage):\n            import psutil\n            slots={lane:dict(lifecycle=dict(slot.lifecycle),queue_size=slot.queue.qsize(),\n                queue_unfinished=slot.queue.unfinished_tasks,closed=slot.closed,\n                thread_alive=slot.thread.is_alive())for lane,slot in dispatch.slots.items()}\n            save(Path(binding).with_name(\"request_lifecycle_\"+stage+\".json\"),dict(\n                stage=stage,entry=who(psutil.Process()),request_weakref_alive=None if reference is None else reference()is not None,\n                original_call_order=list(calls),task_flags=dict(task_flags),owned_slots=slots,\n                original_exception_identity=id(marker),same_original_exception_handled=exception_identity_seen,\n                escaped_exception=escaped,\n                witness=\"Scalar weakref/task/owned-slot flags only; no frames/locals/request contents/traceback dump\"))\n        try:\n            request_owner=RequestOwner();reference=weakref.ref(request_owner)\n            first=asyncio.create_task(asgi._invoke_handler(owned_function,request_owner,kwargs={\"which\":0}))\n            while not enter.is_set():await asyncio.sleep(.001)\n            first.cancel()\n            try:await first\n            except asyncio.CancelledError:pass\n            task_flags[\"first\"]=dict(done=first.done(),cancelled=first.cancelled())\n            second=asyncio.create_task(asgi._invoke_handler(owned_function,request_owner,kwargs={\"which\":1}))\n            await asyncio.sleep(.02);assert calls==[0]\n            release.set();assert await second==1\n            task_flags[\"second\"]=dict(done=second.done(),cancelled=second.cancelled())\n            try:await asgi._invoke_handler(owned_function,request_owner,kwargs={\"which\":2})\n            except ValueError as error:\n                assert error is marker\n                exception_identity_seen=True\n            else:raise AssertionError(\"Original exception masked\")\n            # Caller has handled the exact ORIGINAL exception; only now release\n            # its diagnostic traceback as in the original lifetime fixture.\n            marker.__traceback__=None\n            del first,second,request_owner\n            await asyncio.sleep(0);await asyncio.sleep(0)\n            lifecycle_witness(\"before_weakref_assert\")\n            assert reference()is None\n        except BaseException as error:\n            escaped=dict(type=type(error).__name__,message=str(error))\n            raise\n        finally:\n            primary=sys.exc_info()[1]\n            try:lifecycle_witness(\"finally\")\n            except BaseException as publication_error:\n                if primary is None:raise\n                primary.add_note(\"Lifecycle scalar witness publication failed: \"+type(publication_error).__name__+\": \"+str(publication_error)[:500])\n            finally:\n                release.set();dispatch.close();official.get_logs_batch=old\n                marker.__traceback__=None\n"
new_gate="    execution_phase=scope[\"execution_phase\"]\n    assert execution_phase in (\"original_metadata_shape\",\"remaining_lifetime_only\",\"original_adverse_shape\")\n    projection_pin=pin(PROJECTION)\n    assert scope[\"projection_source\"]==projection_pin\n    assert projection_pin in scope[\"input_pins\"]\n    required_shape=scope[\"required_original_shape\"]\n    main_shape_pin=scope[\"accepted_metadata_main_ROOT\"]\n    prior_matching_pin=scope[\"prior_matching_candidate_first_shape_ROOT\"]\n    if execution_phase==\"original_metadata_shape\":\n        # NEW reader5 must earn its own original500k MAIN and full lifetime.\n        # The old reader4 MAIN receipt is historical, never activation authority.\n        assert required_shape==\"metadata_live_order\"\n        assert main_shape_pin is None and prior_matching_pin is None\n    else:\n        assert type(main_shape_pin)is dict and main_shape_pin in scope[\"input_pins\"]\n        assert pin(main_shape_pin[\"path\"])==main_shape_pin\n        main_shape=read(main_shape_pin[\"path\"])\n        assert main_shape[\"kind\"]==\"ROOT_Trackio7_projection2_reader5_first_original_MAIN_shape_partial_acceptance_v1\"\n        assert main_shape[\"accepted_main_shape\"]is True and main_shape[\"reviewer\"]==\"/root\"\n        assert main_shape[\"candidate_reader\"]==scope[\"candidate\"]and main_shape[\"projection_source\"]==projection_pin\n        assert main_shape[\"completed_first_shape\"][\"shape\"]==\"metadata_live_order\"\n        assert main_shape[\"completed_first_shape\"][\"original_history_rows\"]==500000\n        if execution_phase==\"remaining_lifetime_only\":\n            assert required_shape is None and prior_matching_pin is None\n            assert scope[\"remaining_lifetime_fixture_rows\"]==64\n        else:\n            assert required_shape==\"adverse_reverse_interleaved\"\n            assert type(prior_matching_pin)is dict and prior_matching_pin in scope[\"input_pins\"]\n            assert pin(prior_matching_pin[\"path\"])==prior_matching_pin\n            matching=read(prior_matching_pin[\"path\"])\n            assert matching[\"kind\"]==\"ROOT_Trackio7_projection2_reader5_first_original_shape_actual_acceptance_v1\"\n            assert matching[\"accepted_first_shape\"]is True and matching[\"reviewer\"]==\"/root\"\n            assert matching[\"candidate_reader\"]==scope[\"candidate\"]\n            assert matching[\"projection_source\"]==projection_pin\n            assert matching[\"completed_first_shape\"][\"shape\"]==\"metadata_live_order\"\n            assert matching[\"completed_first_shape\"][\"original_history_rows\"]==500000\n"
scopepath=D/'ROOT_API_request_reference_ownership_INACTIVE_SOURCE_scope_v1.json'
scope=json.loads(scopepath.read_text(encoding='utf-8-sig'))
assert pin(scopepath)['sha256']=='e334847939b9c59e05b8ba78a85646865ba381eb19a75c8d98547b7e0f9dc48b'
assert scope['accepted']is True and scope['actual_execution_admitted']is False
helperoldpath=pathlib.Path(scope['helper3']['path']);helperold=helperoldpath.read_bytes()
readeroldpath=pathlib.Path(scope['reader4']['path']);readerold=readeroldpath.read_bytes()
proofoldpath=pathlib.Path(scope['proof9']['path']);proofold=proofoldpath.read_bytes()
for p,b,k in ((helperoldpath,helperold,'helper3'),(readeroldpath,readerold,'reader4'),(proofoldpath,proofold,'proof9')):
    assert pin_bytes(p,b)==scope[k]
helperpath=D/'Trackio_owned_bounded_sync_API_dispatch_v4.py'
readerpath=D/'serve_BPE9223_Trackio_live_v7_INACTIVE_v5.py'
proofpath=D/'Trackio7_owned_completion_lifecycle_and_original_shapes_CPU_proof_v10.py'
rawpath=D/'Trackio7_request_ownership_proof10_full_ordered_RAW_AST_v1.json'
analysispath=D/'Trackio7_request_reference_SOURCE_ownership_analysis_v1.json'
resourcepath=D/'Trackio7_request_ownership_proof10_resource_calculation_v1.json'
handoffpath=D/'Trackio7_request_ownership_proof10_SOURCE_handoff_v1.json'
resultpath=D/'Trackio7_request_ownership_proof10_SOURCE_result_v1.json'
for p in (helperpath,readerpath,proofpath,rawpath,analysispath,resourcepath,handoffpath,resultpath):assert not p.exists(),str(p)
def guard():
    q=sample()
    assert q['commit']>=2612368384 and q['physical']>=2112368384 and q['private']<=536870912 and q['E_free']>=100000000000
    assert time.monotonic()-t0<180
    return q
def tops(b):
    return {n.name:sha(ast.dump(n,include_attributes=False).encode())for n in ast.parse(b.decode('utf-8-sig')).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
def nested(b):
    return {n.name:sha(ast.dump(n,include_attributes=False).encode())for n in ast.walk(ast.parse(b.decode('utf-8-sig')))if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
def replace_one(text,a,b):
    assert text.count(a)==1,(a[:100],text.count(a))
    return text.replace(a,b)
# Worker owns request-bearing call/job only until before completion publication.
h=helperold.decode()
a=h.index('    def _work(self):');b=h.index('    async def invoke(self,call):',a)
h=h[:a]+new_work+h[b:]
h=replace_one(h,'        self.closed=False\n    def start(self):',
'        self.closed=False\n        self.lifecycle={"stage":"idle","owned_job_drops":0,"publication_returns":0}\n    def start(self):')
helpernew=h.encode()
hn0=nested(helperold);hn1=nested(helpernew)
hchanged=[k for k in hn0 if hn0[k]!=hn1[k]]
assert hchanged==['_work'],hchanged
# Name collision in nested() means __init__ is last OwnedDispatch.__init__;
# explicitly verify whole OwnedDispatch and all non-work slot methods exact.
hc0=ast.parse(helperold.decode()).body;hc1=ast.parse(h)
c0={n.name:n for n in hc0 if isinstance(n,ast.ClassDef)};c1={n.name:n for n in hc1 if isinstance(n,ast.ClassDef)}
assert ast.dump(c0['OwnedDispatch'],include_attributes=False)==ast.dump(c1['OwnedDispatch'],include_attributes=False)
sm0={n.name:ast.dump(n,include_attributes=False)for n in c0['_OwnedSlot'].body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
sm1={n.name:ast.dump(n,include_attributes=False)for n in c1['_OwnedSlot'].body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
assert [k for k in sm0 if sm0[k]!=sm1[k]]==['__init__','_work']
guard();helperpin=write_source(helperpath,helpernew)
# Reader successor changes ONLY helper dependency path/byte-count/SHA.
r=readerold
for a,b in ((b'Trackio_owned_bounded_sync_API_dispatch_v3.py',helperpath.name.encode()),
(b'len(dispatch_raw)==6038',('len(dispatch_raw)=='+str(helperpin['bytes'])).encode()),
(b'9f478b8a732d63d710d8163eb9ae179e3fc57a1d8ade40ef3af130c7a0311198',helperpin['sha256'].encode())):
    assert r.count(a)==1;r=r.replace(a,b)
readerpin=write_source(readerpath,r)
# New candidate needs its own500k shape obligations, never old reader4 evidence.
p=proofold.decode()
p=replace_one(p,'serve_BPE9223_Trackio_live_v7_INACTIVE_v4.py',readerpath.name)
p=replace_one(p,'Trackio_owned_bounded_sync_API_dispatch_v3.py',helperpath.name)
a=p.index('    async def cancellation_and_exception():');b=p.index('    asyncio.run(cancellation_and_exception())',a)
p=p[:a]+new_cancel+p[b:]
a=p.index('    execution_phase=scope["execution_phase"]');b=p.index('    start=time.monotonic();children=[];',a)
p=p[:a]+new_gate+p[b:]
p=replace_one(p,'this_run_one_explicit_original_shape=execution_phase=="original_adverse_shape"',
'this_run_one_explicit_original_shape=execution_phase!="remaining_lifetime_only"')
p=replace_one(p,'historical_reader3_first_shape_NOT_activation_authority=True',
'historical_reader3_first_shape_NOT_activation_authority=True,historical_reader4_MAIN_NOT_newcandidate_authority=True')
proofnew=p.encode()
pt0=tops(proofold);pt1=tops(proofnew)
assert list(pt0)==list(pt1)
pchanged=[k for k in pt0 if pt0[k]!=pt1[k]]
assert pchanged==['source_bound_lifetime_negatives','fixture_body'],pchanged
# Header candidate constant is a separate literal delta, no other function changes.
proofpin=write_source(proofpath,proofnew)
raws=[
raw_diff(helperold,helpernew,helperoldpath,helperpath,[
'Add fixed scalar per-slot lifecycle stage/drop/publication counters only, initialized before any start.',
'Drop call/job closures and task_done BEFORE Future publication, preserving original result/exception object and traceback until caller handles it.',
'Worker future/result/failure locals removed in bounded finally after publication, including error/cancellation paths. Original invoke/gate/hook/constructor unwind/close/readiness AST exact.']),
raw_diff(readerold,r,readeroldpath,readerpath,['ONLY helper dependency basename+bytes+SHA; all runtime/science/source/resource/route/reader code otherwise byte inverse exact.']),
raw_diff(proofold,proofnew,proofoldpath,proofpath,[
'Bind new reader5/helper4; original projector2 unchanged.',
'Add scalar task/weakref/slot-stage/drop/publication witness before original weakref assertion andfinally, without new sleeps/GC/referrer/frame/request inspection.',
'Preserve exact original call order, same exception object caught before original marker traceback release, cancellation owns slot until actual work ends.',
'Fresh typed original_metadata_shape500k MAIN required for NEWreader5; oldreader4MAIN55fd historical only. Remaining64 phase needs genuine matching newMAIN, adverse500k needs composed matching newfirst ROOT. All workload/timeouts/guards unchanged.',
'Original headroom SOURCE9/Spaces/official API results/fences/server/64fixture remain exact outside narrow ownership and typed-gate edits.'])]
rawobj=dict(kind='Trackio7_helper4_reader5_proof10_full_ordered_RAW_AST_v1',SOURCE_ONLY=True,scope=pin(scopepath),
sources=raws,helper_slot_changed_methods=['__init__','_work'],helper_all_other_methods_exact=True,
reader_only_three_literal_dependency_edits=True,proof_changed_functions=pchanged,
base_proof_top_AST=pt0,candidate_proof_top_AST=pt1,
all_forward_inverse_verified=True,all_whole_byte_inverse_exact=True)
rawpin=write_json(rawpath,rawobj)
analysis=dict(kind='Trackio7_request_reference_SOURCE_ownership_analysis_v1',source_only=True,
failure_ROOT=scope['failure'],API07_actual_owner_and_permanence='UNKNOWN; no original actual/capture read',
definite_SOURCE_gap='Helper3 Future.set_result/set_exception publishes before finally del call/future and outer finally task_done/del job. Event-loop callback can resume while worker still owns request-bearing call/job.',
success_owners=['OwnedDispatch awaited coroutine captures fn/request/args/kwargs until completion','slot.invoke call argument and queued tuple contain request closure until original worker executes it','original result can legitimately retain values; unchanged result delivered'],
cancel_owners=['Cancelled shielded caller does not cancel concurrent original work','wrapped completion release callback retains slot/gate but no direct request; queued/running call remains owned until actual completion','gate release remains after original completion; no new cancellation or earlycapacity'],
exception_owners=['Original exception and traceback legitimately retain original invoke/lambda/request frames until caller handles it','Helper4 preserves SAME exception object and traceback, never clears it','Fixture handles SAME marker, then performs its ORIGINAL marker traceback release; no GC/new sleep/exception erasure added'],
remedy='Drop OWNED call/job refs before Future publication; queue.task_done once, then publish same result/exception. Finally drop worker future/result/failure locals. Fixed scalar lifecycle flags only; not proof of API07 actual cause.',
unresolved=['asyncio callback/task/Future scheduling may expose other ownership; actual weakref witness required','No retained actual strong-owner path or leak permanence claim','Newcandidate performance/full500k obligations remain unproved until future exact runs'],
changed_candidate_obligations='Reader5/helper4 must earn BOTH original500k shapes plus full lifetime under separately reviewed exact scopes. Reader4MAIN55fd cannot activate newcandidate.',
no_extra_runtime_SOURCE_or_frame_inspection=True)
analysispin=write_json(analysispath,analysis)
resources=dict(kind='Trackio7_reader5_unchanged_resource_calculation_v1',prospective_commit=8056867200,prospective_physical=5306867200,
runtime_commit=5538431744,runtime_physical=5038431744,family_private=2573741824,reader_private=1500000000,
controller_private=536870912,watchdog_private=536870912,self_private=536870912,HTTP_seconds=20,body_seconds=160,external_seconds=180,E_free_min=100000000000,
original_cache=268435456,API_reserve=16777216,projection_workspace=67108864,source39_reserve=75497472,
fixed_new_lifecycle_metadata='Two slot dictionaries, each3fixedkeys/string stage+2integer counters; no history/request/packet refs. Conservatively<=4096bytes accommodated WITHIN unchangedAPI16MiB, no reserve reduction or implicit capraise.',
guards_and_sampling='All original native225/FULL numerical/floors +72/cache256/API16/projection64/controller512/watchdog512/ROOT512 and1.5GB reader private retained. No freedcredit or HTTP/time/private increase.')
resourcepin=write_json(resourcepath,resources)
handoff=dict(kind='Trackio7_helper4_reader5_proof10_INACTIVE_SOURCE_handoff_v1',owner='/root/memory_diagnosis',
actual_execution_admitted=False,scope=pin(scopepath),failure_ROOT=scope['failure'],initial=initial,
helper=helperpin,reader=readerpin,proof=proofpin,projection_unchanged=scope['projection2'],
watchdog_unchanged=pin(D/'owned_source_CPU_proof_watchdog_v1.py'),RAW=rawpin,ownership_analysis=analysispin,resources=resourcepin,
semantics='Original invoke/OwnedDispatch/official auth-arg-return-error/guard/cancellation-gate/constructor-unwind/joins unchanged. Worker refdrop beforepublish only; same original exception object/traceback retained for caller; fixed scalar metadata.',
witness='request_lifecycle_before_weakref_assert.json + request_lifecycle_finally.json. Scalar weakref boolean, original callorder, task done/cancel flags, queue bounded metadata, worker stage/drop/publication counts, original exception identity and handledflag. No f_locals/frame/request dumps/referrer enumeration/GC/new sleep.',
first_newcandidate_mode=dict(execution_phase='original_metadata_shape',required_original_shape='metadata_live_order',accepted_metadata_main_ROOT=None,prior_matching_candidate_first_shape_ROOT=None,
output_directory=str(D/'a7_f08'),external_output_directory=str(D/'a7_e08'),
gate_status='NONACTIONABLE until separate ROOTwhole review and exactONE scope; allnewcandidate500k cold8+bulk64+version/reference/SQLite/state-loss192/warm/closure+full lifetime required.'),
remaining_mode=dict(execution_phase='remaining_lifetime_only',remaining_lifetime_fixture_rows=64,required_original_shape=None,accepted_metadata_main_ROOT=None,prior_matching_candidate_first_shape_ROOT=None,
gate_status='UNKNOWN until newreader5 MAIN ROOT exact matching candidate/projection, accepted_main_shape:true and required newreader5 kind; old55fd not allowed.'),
adverse_mode=dict(execution_phase='original_adverse_shape',required_original_shape='adverse_reverse_interleaved',accepted_metadata_main_ROOT=None,prior_matching_candidate_first_shape_ROOT=None,
gate_status='UNKNOWN until newreader5 MAIN+full lifetime composed matching first ROOT; original fulladverse500k and all tests retained.'),
proposed_prefix=[str(P/'work/monitor-env/Scripts/python.exe'),'-X','utf8','-B',str(proofpath),'--ROOT-review','FUTURE_ROOT_EXACT_SCOPE_UNKNOWN'],
new_candidate_ROOT_main_kind='ROOT_Trackio7_projection2_reader5_first_original_MAIN_shape_partial_acceptance_v1',
new_candidate_ROOT_first_kind='ROOT_Trackio7_projection2_reader5_first_original_shape_actual_acceptance_v1',
historical_reader4_MAIN_NOT_newcandidate_authority=True,all_actual_API07_artifacts_CLOSED_no_reread=True,
new_preparer=dict(entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),owned_children=[]),
no_candidate_import_execution=True,final_sample=guard(),elapsed=time.monotonic()-t0)
handoffpin=write_json(handoffpath,handoff)
result=dict(kind='Trackio7_request_ownership_proof10_SOURCE_result_v1',prepared=True,actual_execution=False,
helper=helperpin,reader=readerpin,proof=proofpin,RAW=rawpin,analysis=analysispin,resources=resourcepin,
handoff=handoffpin,initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),
ordered_RAW_counts=[len(x['ordered_edits'])for x in raws],whole_byte_inverse_exact=True,final_sample=guard(),elapsed=time.monotonic()-t0)
resultpin=write_json(resultpath,result)
print(json.dumps(dict(helper=helperpin,reader=readerpin,proof=proofpin,RAW=rawpin,analysis=analysispin,resources=resourcepin,
handoff=handoffpin,result=resultpin,initial=initial,entry=result['entry'],launcher=result['launcher'],ordered_RAW_counts=result['ordered_RAW_counts']),ensure_ascii=False))

