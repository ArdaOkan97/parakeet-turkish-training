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
initial=write_json(D/'Trackio7_complete_headroom_proof9_SOURCE_initial_v1.json',dict(kind='SOURCE_preparation_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=s,required=dict(commit=5399239296,physical=2649239296,private_max=536870912,E_free_min=100000000000),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),candidate_import_execution=False))
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

replacement="    # Actual original reader first-headroom branch with complete fixture globals.\n    # The ORIGINAL readiness helper must call the supplied guard before HTTP.\n    tree=ast.parse(CANDIDATE.read_text(encoding=\"utf-8\"))\n    functions={n.name:n for n in ast.walk(tree)if isinstance(n,(ast.FunctionDef,ast.ClassDef))}\n    selected=[functions[n]for n in (\"DashboardMemoryLimit\",\"local_status\",\"mirror_body\",\"mirror\")]\n    binding_value=read(binding)\n    namespace=dict(threading=threading,time=time,Path=Path,read=read,\n        identity={\"fixture\":True},BASE=\"http://127.0.0.1:\"+str(binding_value[\"port\"]),\n        RUN=Path(binding_value[\"run\"]).resolve(),\n        mirror_activity={\"HTTP_ready\":False,\"committed_HTTP_batch\":False},\n        pending_holder={\"value\":None},terminal_holder={\"error\":None},\n        server=types.SimpleNamespace(should_exit=False,started=False),\n        STATE=Path(binding).with_name(\"never_exists_state.json\"),\n        STATUS=Path(binding).with_name(\"captured_status.json\"),memory=lambda:{},\n        api_dispatch=types.SimpleNamespace(stats={}),dispatch_module=helper)\n    assert not namespace[\"STATE\"].exists()\n    statuses=[];namespace[\"write\"]=lambda path,value:statuses.append(value)\n    exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(CANDIDATE),\"exec\"),namespace)\n    headroom=namespace[\"DashboardMemoryLimit\"](\"fixture original headroom cause\")\n    guard_calls=[]\n    def fail_headroom():\n        guard_calls.append(dict(call=len(guard_calls)+1,supplied_exception_identity=id(headroom)))\n        raise headroom\n    namespace[\"headroom\"]=fail_headroom\n    import psutil\n    def headroom_witness(stage,escaped=None):\n        terminal=namespace[\"terminal_holder\"][\"error\"]\n        save(Path(binding).with_name(\"headroom_first_guard_\"+stage+\".json\"),dict(\n            stage=stage,entry=who(psutil.Process()),statuses=statuses,guard_calls=guard_calls,\n            terminal_type=None if terminal is None else type(terminal).__name__,\n            terminal_message=None if terminal is None else str(terminal),\n            terminal_identity=None if terminal is None else id(terminal),\n            supplied_exception_identity=id(headroom),terminal_is_supplied_exception=terminal is headroom,\n            escaped_exception=escaped,original_wait_http_ready=True,\n            namespace_BASE=namespace[\"BASE\"],server_started=namespace[\"server\"].started,\n            HTTP_ready=namespace[\"mirror_activity\"][\"HTTP_ready\"],\n            state_existed_before_call=False,\n            original_first_guard_SOURCE_precedes_owned_started_HTTP_and_producer_read=True,\n            owner=\"METHOD_STUB guard/status sink; ORIGINAL helper and reader branch AST\"))\n    escaped=None\n    try:\n        namespace[\"mirror\"]()\n        headroom_witness(\"before_assert\")\n        assert guard_calls==[dict(call=1,supplied_exception_identity=id(headroom))]\n        assert namespace[\"terminal_holder\"][\"error\"]is headroom\n        assert statuses[-1][\"status\"]==\"stopped_headroom\"and \"fixture original headroom cause\"in statuses[-1][\"error\"]\n        assert statuses[-1][\"mirror_thread_alive\"]is False\n        assert namespace[\"mirror_activity\"][\"HTTP_ready\"]is False\n    except BaseException as error:\n        escaped=dict(type=type(error).__name__,message=str(error))\n        raise\n    finally:\n        primary=sys.exc_info()[1]\n        try:headroom_witness(\"finally\",escaped)\n        except BaseException as publication_error:\n            if primary is None:raise\n            primary.add_note(\"Headroom scalar witness publication failed: \"+type(publication_error).__name__+\": \"+str(publication_error)[:500])\n        finally:\n            terminal=namespace[\"terminal_holder\"][\"error\"]\n            if terminal is not None:terminal.__traceback__=None\n            headroom.__traceback__=None\n            namespace[\"terminal_holder\"][\"error\"]=None\n            namespace[\"headroom\"]=None\n            statuses.clear();guard_calls.clear()\n"
scopepath=D/'ROOT_proof9_complete_headroom_namespace_INACTIVE_SOURCE_scope_v1.json'
scope=json.loads(scopepath.read_text(encoding='utf-8-sig'))
assert pin(scopepath)['sha256']=='1e17cf32ada5b2d92f1630850765ff8ec7010192af88065f0ae6c51322e8cb5d'
assert scope['accepted']is True and scope['actual_execution_admitted']is False
base=pathlib.Path(scope['base_proof']['path']);old=base.read_bytes()
assert pin_bytes(base,old)==scope['base_proof']
target=D/'Trackio7_complete_headroom_lifetime_and_adverse_CPU_proof_v9.py'
rawpath=D/'Trackio7_complete_headroom_proof9_full_ordered_RAW_AST_v1.json'
handoffpath=D/'Trackio7_complete_headroom_proof9_SOURCE_handoff_v1.json'
resultpath=D/'Trackio7_complete_headroom_proof9_SOURCE_result_v1.json'
for p in (target,rawpath,handoffpath,resultpath):assert not p.exists()
text=old.decode('utf-8')
a=text.index('    # Actual reader2 first-headroom terminal branch, metadata/HTTP owners doubled.')
b=text.index('    return dict(post_both_workers_hook_install_failure_unwind=',a)
text=text[:a]+replacement+text[b:]
new=text.encode('utf-8')
def tops(raw):
    return {n.name:sha(ast.dump(n,include_attributes=False).encode())for n in ast.parse(raw.decode()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
ot=tops(old);nt=tops(new)
assert list(ot)==list(nt)
changed=[k for k in ot if ot[k]!=nt[k]]
assert changed==['source_bound_lifetime_negatives'],changed
# Check every statement outside the replaced headroom sub-block, including all
# original Spaces/constructor/API/cancellation/lifetime code, remains byte exact.
assert new[:a]==old[:a]
assert text[text.index('    return dict(post_both_workers_hook_install_failure_unwind=',a):]==old.decode()[b:]
def guard():
    q=sample()
    assert q['commit']>=2612368384 and q['physical']>=2112368384 and q['private']<=536870912 and q['E_free']>=100000000000
    assert time.monotonic()-t0<180
    return q
guard()
sourcepin=write_source(target,new)
raw=raw_diff(old,new,base,target,[
'Complete original headroom AST namespace with fixture BASE/port, RUN, Path/read and server.started=False; no production binding or live read.',
'Use ORIGINAL helper.wait_http_ready instead of lambda bypass; original first supplied guard raises exact DashboardMemoryLimit before HTTP/producer read.',
'Persist scalar statuses/terminal type-text-identity/guard-call witness BEFORE strict assertions and separately in finally; no traceback/frame/packet retention.',
'Keep exact stopped_headroom/original cause/threadfalse assertions, plus guard-once/terminal identity/HTTPnotready checks; all other proof8 code is byte exact.'
])
raw.update(kind='Trackio7_proof9_complete_headroom_ordered_offset_RAW_AST_v1',
SOURCE_ONLY=True,scope=pin(scopepath),base_AST_top=ot,candidate_AST_top=nt,changed_functions=changed,
all_other_functions_AST_exact=True,all_outside_replaced_subblock_byte_exact=True)
rawpin=write_json(rawpath,raw)
helperpin=pin(D/'Trackio_owned_bounded_sync_API_dispatch_v3.py')
watchdogpin=pin(D/'owned_source_CPU_proof_watchdog_v1.py')
for item in (scope['reader_unchanged'],scope['projection_unchanged'],scope['ROOT_API06_failure_seal']):
    assert pin(pathlib.Path(item['path']))==item
helperraw=pathlib.Path(helperpin['path']).read_text(encoding='utf-8')
htree=ast.parse(helperraw)
hnode=next(n for n in htree.body if isinstance(n,ast.FunctionDef)and n.name=='wait_http_ready')
helpercite=dict(source=helperpin,start_line=hnode.lineno,end_line=hnode.end_lineno,
complete_original_SOURCE='\n'.join(helperraw.splitlines()[hnode.lineno-1:hnode.end_lineno]))
resources=dict(prospective_commit=8056867200,prospective_physical=5306867200,runtime_commit=5538431744,runtime_physical=5038431744,
family_private=2573741824,self_private=536870912,reader_private=1500000000,controller_private=536870912,watchdog_private=536870912,
request_seconds=20,proof_body_seconds=160,external_seconds=180,E_free_min=100000000000)
handoff=dict(kind='Trackio7_proof9_complete_headroom_INACTIVE_SOURCE_handoff_v1',
owner='/root/memory_diagnosis',actual_execution_admitted=False,scope=pin(scopepath),
base=scope['base_proof'],proof=sourcepin,RAW=rawpin,initial=initial,
root_failure_seal=scope['ROOT_API06_failure_seal'],failure_originals_and_captures_CLOSED=True,
actual_API06_terminal_status_cause='UNKNOWN; source predicts BASE NameError but no actual statuses were persisted. No original actual artifact reread.',
candidate_reader_unchanged=scope['reader_unchanged'],projection_unchanged=scope['projection_unchanged'],API_helper_unchanged=helperpin,watchdog_unchanged=watchdogpin,
headroom_change=dict(namespace='Fixture port->BASE, original Path/read, fixture RUN, server.started=False, original helper module.',
guard='Exact supplied DashboardMemoryLimit raised once by first original readiness helper guard before network/producer read; metadata status sink only.',
witness='headroom_first_guard_before_assert.json and headroom_first_guard_finally.json; full statuses scalar list, terminal type/text/id/supplied identity, entry PIDctime, guard count, HTTP_ready and server.started. Finally removes exception tracebacks/namespace exception refs; no frames/locals retained.',
assertions='Exact old stopped_headroom/original cause/threadfalse unchanged, with guardonce/terminal-is-original/HTTPnotready checks added.',
helper_source_citation=helpercite),
unchanged_proof8='All other lifetime/Spaces constructor selector finally/API/auth/result/cancellation/ownership blocks byte exact; remaining64 function and fixture_body full AST exact. No accepted500k MAIN rerun; future full adverse gate unchanged.',
future_mode=dict(execution_phase='remaining_lifetime_only',remaining_lifetime_fixture_rows=64,required_original_shape=None,prior_matching_candidate_first_shape_ROOT=None,
output_directory=str(D/'a7_l07'),external_output_directory=str(D/'a7_le07'),exact_argv_proposal=[str(P/'work/monitor-env/Scripts/python.exe'),'-X','utf8','-B',str(target),'--ROOT-review','FUTURE_ROOT_EXACT_API07_SCOPE_UNKNOWN']),
future_adverse='Still UNKNOWN/nonactionable until matching composed MAIN55fd+actual remaining lifetime ROOT; separate scope required. Same adverse500k original workload and all guards; no automatic execution.',
resources=resources,source_prep_limits=dict(commit=5399239296,physical=2649239296,runtime_commit=2612368384,runtime_physical=2112368384,self_private=536870912,wall_seconds=180,E_free_min=100000000000),
input_pins=[sourcepin,scope['reader_unchanged'],scope['projection_unchanged'],helperpin,watchdogpin,pin(scopepath),scope['ROOT_API06_failure_seal']],
known_preparer=dict(entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),owned_children=[]),
final_sample=guard(),elapsed=time.monotonic()-t0,no_candidate_import_or_execution=True)
handoffpin=write_json(handoffpath,handoff)
result=dict(kind='Trackio7_proof9_complete_headroom_SOURCE_result_v1',prepared=True,actual_execution=False,
proof=sourcepin,RAW=rawpin,handoff=handoffpin,initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),
changed_functions=changed,whole_byte_inverse_exact=True,all_other_functions_AST_exact=True,
final_sample=guard(),elapsed=time.monotonic()-t0)
resultpin=write_json(resultpath,result)
print(json.dumps(dict(proof=sourcepin,RAW=rawpin,handoff=handoffpin,result=resultpin,initial=initial,
entry=result['entry'],launcher=result['launcher'],ordered_RAW_edits=len(raw['ordered_edits']),
final_sample=result['final_sample']),ensure_ascii=False))

