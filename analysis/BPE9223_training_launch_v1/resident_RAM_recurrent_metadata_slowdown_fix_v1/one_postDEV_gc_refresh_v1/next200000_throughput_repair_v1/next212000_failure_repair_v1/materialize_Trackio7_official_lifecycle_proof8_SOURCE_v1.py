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
initial=write_json(D/'Trackio7_official_lifecycle_proof8_SOURCE_initial_v1.json',dict(kind='SOURCE_preparation_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=s,required=dict(commit=5399239296,physical=2649239296,private_max=536870912,E_free_min=100000000000),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),candidate_import_execution=False))
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

constructor_block="    # Use the exact original public constructor's Spaces-only route/state\n    # branch. Only its selector is METHOD_STUBbed; no state field is fabricated.\n    normal_app,_=official.build_starlette_app_only(mcp_server=False)\n    utils_pin=pin(asgi.utils.__file__)\n    assert utils_pin==scope[\"official_utils_source\"]\n    assert utils_pin in scope[\"input_pins\"]\n    original_spaces_selector=asgi.utils.on_spaces\n    assert original_spaces_selector()is False\n    def owned_constructor_spaces_selector():return True\n    oauth_paths={official.OAUTH_START_PATH,official.OAUTH_CALLBACK_PATH,\"/oauth/logout\"}\n    oauth_routes=[route for route in normal_app.routes if getattr(route,\"path\",None)in oauth_paths]\n    assert len(oauth_routes)==3\n    asgi.utils.on_spaces=owned_constructor_spaces_selector\n    try:\n        gradio_app=asgi.create_trackio_starlette_app(\n            oauth_routes,normal_app.state.api_registry,\n            write_apis=normal_app.state.write_apis,\n            allowed_file_roots=list(normal_app.state.allowed_file_roots),\n            upload_authorizer=normal_app.state.upload_authorizer)\n    finally:\n        # Restore only our owned selector; never retarget a foreign replacement.\n        if asgi.utils.on_spaces is owned_constructor_spaces_selector:\n            asgi.utils.on_spaces=original_spaces_selector\n    assert asgi.utils.on_spaces is original_spaces_selector\n    assert original_spaces_selector()is False\n    assert any(getattr(route,\"endpoint\",None)is asgi.gradio_call_post_handler for route in gradio_app.routes)\n    assert not any(getattr(route,\"endpoint\",None)is asgi.gradio_call_post_handler for route in normal_app.routes)\n    # No handler, storage, backend or auth call ran while selector was doubled.\n"
remaining_block="\ndef prepare_remaining_lifetime_fixture(root,scope,guard):\n    \"\"\"Fresh <=256 REAL official-schema rows; no accepted500k workload rerun.\"\"\"\n    assert scope[\"remaining_lifetime_fixture_rows\"]==64\n    guard()\n    case=root/\"remaining_lifetime_only\";case.mkdir()\n    area=case/\"area\";area.mkdir()\n    run=case/\"run\";run.mkdir();(run/\"progress\").mkdir()\n    dbdir=area/\"database\";dbdir.mkdir()\n    project=\"fixture_remaining_lifetime\";name=\"fixture_training\"\n    os.environ[\"TRACKIO_DIR\"]=str(dbdir)\n    for key in (\"TRACKIO_SPACE_ID\",\"TRACKIO_SERVER_URL\",\"TRACKIO_DATASET_ID\",\"TRACKIO_BUCKET_ID\",\"TRACKIO_WEBHOOK_URL\"):\n        os.environ.pop(key,None)\n    import orjson,sqlite3,psutil\n    from trackio.sqlite_storage import SQLiteStorage\n    from trackio.utils import on_spaces\n    assert not on_spaces(),\"Only owned local fixture storage is admitted\"\n    db=SQLiteStorage.init_db(project)\n    expected=(dbdir/(project+\".db\")).resolve()\n    save(case/\"actual_fixture_DB_path_binding.json\",dict(shape=\"remaining_lifetime_only\",\n        actual_returned_DB_path=str(db.resolve()),expected_DB_path=str(expected),\n        TRACKIO_DIR=os.environ.get(\"TRACKIO_DIR\"),diagnostic_rows=64))\n    assert db.resolve()==expected\n    rows=[(name,\"2026-01-01T00:00:%02d.000000\"%(i%60),name,i,\n        orjson.dumps({\"train/objective\":1.+i%7,\"progress/update\":i}).decode())for i in range(64)]\n    conn=sqlite3.connect(db)\n    try:\n        conn.executemany(\"INSERT INTO metrics(run_id,timestamp,run_name,step,metrics)VALUES(?,?,?,?,?)\",rows)\n        conn.commit()\n        assert conn.execute(\"SELECT COUNT(*)FROM metrics\").fetchone()[0]==64\n    finally:conn.close()\n    journal=run/\"fixture_journal.jsonl\";journal.write_bytes(b\"\")\n    save(run/\"progress/latest.json\",dict(actual_optimizer_updates=1,journal_prefix=dict(path=str(journal)),\n        producer=dict(pid=os.getpid(),create_time=psutil.Process().create_time()),\n        last_saved_updates=0,native_updates_this_process=1,native_seconds=1.,\n        active_pipeline_seconds=1.,source_audio_seconds=1.,audio_minutes_per_wall_minute=1.,\n        optimizer_poisoned=False,schedule_poisoned=False))\n    save(run/\"latest_checkpoint.json\",dict(committed_updates=0))\n    with socket.socket()as sock:sock.bind((\"127.0.0.1\",0));port=sock.getsockname()[1]\n    assert port!=7860\n    binding=case/\"bindings.json\"\n    save(binding,dict(kind=\"Trackio7_owned_E_fixture_bindings_v1\",fixture_root=str(case),area=str(area),\n        run=str(run),source_db=str(case/\"unused_source.db\"),project=project,run_name=name,port=port))\n    save(case/\"remaining_lifetime_fixture_receipt.json\",dict(real_original_schema=True,actual_rows=64,\n        completed_metadata500000_main_rerun=False,fixture_only_no_training_authority=True))\n    guard()\n    return binding\n\n"
mode_gate="    execution_phase=scope[\"execution_phase\"]\n    assert execution_phase in (\"remaining_lifetime_only\",\"original_adverse_shape\")\n    projection_pin=pin(PROJECTION)\n    assert scope[\"projection_source\"]==projection_pin\n    assert projection_pin in scope[\"input_pins\"]\n    main_shape_pin=scope[\"accepted_metadata_main_ROOT\"]\n    assert main_shape_pin==MAIN_PIN_LITERAL\n    assert main_shape_pin in scope[\"input_pins\"]and pin(main_shape_pin[\"path\"])==main_shape_pin\n    main_shape=read(main_shape_pin[\"path\"])\n    assert main_shape[\"kind\"]==\"ROOT_Trackio7_projection2_reader4_first_original_MAIN_shape_partial_acceptance_v1\"\n    assert main_shape[\"accepted_main_shape\"]is True and main_shape[\"reviewer\"]==\"/root\"\n    assert main_shape[\"candidate_reader\"]==scope[\"candidate\"]and main_shape[\"projection_source\"]==projection_pin\n    assert main_shape[\"completed_first_shape\"][\"shape\"]==\"metadata_live_order\"\n    assert main_shape[\"completed_first_shape\"][\"original_history_rows\"]==500000\n    required_shape=scope[\"required_original_shape\"]\n    prior_matching_pin=scope[\"prior_matching_candidate_first_shape_ROOT\"]\n    if execution_phase==\"remaining_lifetime_only\":\n        assert required_shape is None and prior_matching_pin is None\n        assert scope[\"remaining_lifetime_fixture_rows\"]==64\n    else:\n        # Only the still-required adverse500k may run; accepted metadata MAIN\n        # cannot be rerun, and its PARTIAL receipt cannot authorize adverse.\n        assert required_shape==\"adverse_reverse_interleaved\"\n        assert type(prior_matching_pin)is dict and prior_matching_pin in scope[\"input_pins\"]\n        assert pin(prior_matching_pin[\"path\"])==prior_matching_pin\n        matching=read(prior_matching_pin[\"path\"])\n        assert matching[\"kind\"]==\"ROOT_Trackio7_projection2_reader4_first_original_shape_actual_acceptance_v1\"\n        assert matching[\"accepted_first_shape\"]is True and matching[\"reviewer\"]==\"/root\"\n        assert matching[\"candidate_reader\"]==scope[\"candidate\"]\n        assert matching[\"projection_source\"]==projection_pin\n        assert matching[\"completed_first_shape\"][\"shape\"]==\"metadata_live_order\"\n        assert matching[\"completed_first_shape\"][\"original_history_rows\"]==500000\n"
scopepath=D/'ROOT_official_Gradio_lifecycle_remaining_proof8_INACTIVE_SOURCE_scope_v1.json'
amendpath=D/'ROOT_official_Gradio_utils_context_SOURCE_amendment_v1.json'
scope=json.loads(scopepath.read_text(encoding='utf-8-sig'))
amend=json.loads(amendpath.read_text(encoding='utf-8-sig'))
assert pin(scopepath)['sha256']=='dc64ef20afc8382e569bbac6169e07f0b270bd7e0afd23d35fe128348e7ab245'
assert pin(amendpath)['sha256']=='fd0164354a93e6fe8cd469fe56ed834a715206be1fec6233bd9aa6c60e0c0461'
assert scope['actual_execution_admitted'] is False and amend['actual_execution_admitted'] is False
base=pathlib.Path(scope['base_proof7']['path'])
old=base.read_bytes();assert pin_bytes(base,old)==scope['base_proof7']
target=D/'Trackio7_official_lifecycle_remaining_and_adverse_CPU_proof_v8.py'
rawpath=D/'Trackio7_official_lifecycle_proof8_full_ordered_RAW_AST_v1.json'
handoffpath=D/'Trackio7_official_lifecycle_proof8_SOURCE_handoff_v1.json'
resultpath=D/'Trackio7_official_lifecycle_proof8_SOURCE_result_v1.json'
for p in (target,rawpath,handoffpath,resultpath):assert not p.exists(),str(p)
text=old.decode('utf-8')
def exact(a,b):
    global text
    assert text.count(a)==1,(a[:120],text.count(a))
    text=text.replace(a,b)
exact('    app,_=official.build_starlette_app_only(mcp_server=False)\n',constructor_block)
exact('        def req(api_name,event_id=None):','        def req(api_name,event_id=None,app_owner=normal_app):')
exact('path_params=path,app=app),receive)','path_params=path,app=app_owner),receive)')
exact('asgi.gradio_call_post_handler(req("get_logs_batch"))','asgi.gradio_call_post_handler(req("get_logs_batch",app_owner=gradio_app))')
exact('asgi.gradio_call_poll_handler(req("get_logs_batch",event))','asgi.gradio_call_poll_handler(req("get_logs_batch",event,app_owner=gradio_app))')
exact('actual_official_Gradio_post_poll_and_API_same_result=True,headroom_first_cause_preserved=True',
'actual_official_Gradio_post_poll_and_API_same_result=True,official_Spaces_constructor_branch_METHOD_STUB=True,selector_restored_BEFORE_handlers=True,deployed_local_Gradio_route_claimed=False,headroom_first_cause_preserved=True')
exact('\ndef fixture_body(scopepath):',remaining_block+'\ndef fixture_body(scopepath):')
a=text.index('    # BOTH original shapes must pass for THIS exact new candidate.')
b=text.index('    start=time.monotonic();children=[];',a)
text=text[:a]+mode_gate.replace('MAIN_PIN_LITERAL',repr(scope['completed_new_metadata_MAIN_ROOT']))+text[b:]
exact('    try:\n        for shape in (required_shape,):',
'''    try:
        if execution_phase=="remaining_lifetime_only":
            binding=prepare_remaining_lifetime_fixture(root,scope,guard)
        for shape in (() if execution_phase=="remaining_lifetime_only" else (required_shape,)):''')
exact('this_run_one_explicit_original_shape=True,historical_reader3_first_shape_NOT_activation_authority=True',
'this_run_one_explicit_original_shape=execution_phase=="original_adverse_shape",execution_phase=execution_phase,accepted_metadata_main_ROOT=main_shape_pin,completed_metadata500000_main_rerun=False,historical_reader3_first_shape_NOT_activation_authority=True')
new=text.encode('utf-8')
oldtree=ast.parse(old.decode('utf-8'));newtree=ast.parse(text)
def top(tree):
    return {n.name:sha(ast.dump(n,include_attributes=False).encode())for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
ot=top(oldtree);nt=top(newtree)
added=sorted(set(nt)-set(ot));removed=sorted(set(ot)-set(nt))
changed=[k for k in ot if k in nt and ot[k]!=nt[k]]
assert added==['prepare_remaining_lifetime_fixture'] and not removed
assert changed==['source_bound_lifetime_negatives','fixture_body'],changed
for k in ot:
    if k not in changed:assert ot[k]==nt[k]
assert text.count('asgi.utils.on_spaces=owned_constructor_spaces_selector')==1
assert text.count('asgi.utils.on_spaces=original_spaces_selector')==1
assert 'gradio_app.state.'not in text and 'normal_app.state.gradio_call_events'not in text
def guard():
    s2=sample()
    assert s2['commit']>=2612368384 and s2['physical']>=2112368384 and s2['private']<=536870912 and s2['E_free']>=100000000000
    assert time.monotonic()-t0<180
    return s2
guard()
candidatepin=write_source(target,new)
rawledger=raw_diff(old,new,base,target,[
'Original installed public ASGI constructor called in explicitly labelled fixture-only Spaces branch via owned selector METHOD_STUB, finally restored before dispatch/handlers; no fabricated app.state/routes/auth/results.',
'Original local normal app and original constructor-built Gradio app own their respective requests; all official handler logic and original full lifetime negatives remain.',
'Add typed remaining_lifetime_only path with exactly64 fresh official-schema diagnostic rows and original lifetime/source-caller assertions; accepted metadata500000 MAIN is not rerun.',
'Original adverse500000 path remains strict and requires future matching composed MAIN+remaining-lifetime ROOT; no automatic continuation or historical-reader authority.',
'Require pinned original utils SOURCE; preserve original resource/private/time/HTTP/independent-watchdog guards and unchanged server child/failure fences.'
])
rawledger.update(kind='Trackio7_proof8_full_ordered_offset_RAW_forward_inverse_AST_v1',
SOURCE_ONLY=True,base_AST_top=ot,candidate_AST_top=nt,changed_functions=changed,
added_functions=added,removed_functions=removed,unchanged_top_functions=[k for k in ot if k not in changed],
scope=pin(scopepath),amendment=pin(amendpath))
rawpin=write_json(rawpath,rawledger)
site=P/'work/monitor-env/Lib/site-packages/trackio'
officialpaths=dict(asgi=site/'asgi_app.py',server=site/'server.py',utils=site/'utils.py')
officialraw={k:p.read_bytes()for k,p in officialpaths.items()}
officialpins={k:pin_bytes(officialpaths[k],b)for k,b in officialraw.items()}
assert officialpins['asgi']['sha256']=='708a5a8426df2b4b993bcd327f8418166ced73cc077a878529fd2cd4e4860e6b'
assert officialpins['server']['sha256']=='7d846d65fd7ba012e53ff211548dbddb6a626f1376f2b0841a8d1ff188c94e52'
assert officialpins['utils']==amend['additional_exact_SOURCE'] or (officialpins['utils']['bytes']==amend['additional_exact_SOURCE']['bytes']and officialpins['utils']['sha256']==amend['additional_exact_SOURCE']['sha256']and pathlib.Path(officialpins['utils']['path']).resolve()==pathlib.Path(amend['additional_exact_SOURCE']['path']).resolve())
def citation(label,function):
    src=officialraw[label].decode('utf-8-sig');tree=ast.parse(src)
    node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))and n.name==function)
    return dict(source=officialpins[label],function=function,start_line=node.lineno,end_line=node.end_lineno,complete_original_SOURCE='\n'.join(src.splitlines()[node.lineno-1:node.end_lineno]))
citations=[citation('utils','on_spaces'),citation('asgi','_normalize_allowed_file_roots'),citation('asgi','create_trackio_starlette_app'),citation('server','_api_registry'),citation('server','build_starlette_app_only')]
sources=[scope['candidate_reader4'],scope['projection2'],
pin(D/'Trackio_owned_bounded_sync_API_dispatch_v3.py'),pin(D/'owned_source_CPU_proof_watchdog_v1.py')]
for p in sources:assert pin(pathlib.Path(p['path']))==p
mainpin=scope['completed_new_metadata_MAIN_ROOT'];assert pin(pathlib.Path(mainpin['path']))==mainpin
resources=dict(prospective_commit=8056867200,prospective_physical=5306867200,runtime_commit=5538431744,runtime_physical=5038431744,family_private=2573741824,family_private_bytes=2573741824,self_private=536870912,wall_seconds=160,proof_body_wall_seconds=160,external_wall_seconds=180,reader_private=1610612736,controller_private=536870912,watchdog_private=536870912,request_seconds=20,E_free_min=100000000000,projection_pointer_workspace_extra=67108864,ROOT_independent_metadata_reserve=536870912)
handoff=dict(kind='Trackio7_official_lifecycle_proof8_INACTIVE_SOURCE_handoff_v1',
owner='/root/memory_diagnosis',actual_execution_admitted=False,scope=pin(scopepath),amendment=pin(amendpath),
base=scope['base_proof7'],proof=candidatepin,RAW=rawpin,source_initial=initial,
unchanged_candidate_reader=scope['candidate_reader4'],unchanged_projection=scope['projection2'],
unchanged_API_helper=sources[2],unchanged_external_watchdog=sources[3],
completed_new_metadata_MAIN_ROOT=mainpin,CPU03_ROOT=scope['CPU03_ROOT'],
official_SOURCE_pins=officialpins,official_context_citations=citations,
context_findings=dict(selector='Original utils.on_spaces is only os.environ SYSTEM==spaces; no env mutation. Original public constructor branch creates local Route objects, Starlette and original locks/maps only. Normal original builder called with mcp_server=False outside stub. Its OAuth routes/registry/write APIs/roots/authorizer are passed unchanged into original public constructor.',
effect_scope='Fixture-only owned selector true solely for original public create_trackio_starlette_app call; finally restore before dispatch/normal API/Gradio/auth/storage/backend calls. No fabricated state/route/auth/handler/result. This covers original Spaces-only constructor branch, not deployed-local Gradio route.',
caller='All original normal run_api_request/Gradio post+poll methods remain original; helper3 hook lifetime and original single-awaiting caller SOURCE assertion unchanged.'),
remaining_mode=dict(execution_phase='remaining_lifetime_only',remaining_lifetime_fixture_rows=64,required_original_shape=None,prior_matching_candidate_first_shape_ROOT=None,accepted_metadata_main_ROOT=mainpin,official_utils_source=officialpins['utils'],output_directory=str(D/'a7_l06'),external_output_directory=str(D/'a7_le06'),proposed_exact_argv=[str(P/'work/monitor-env/Scripts/python.exe'),'-X','utf8','-B',str(target),'--ROOT-review','FUTURE_ROOT_EXACT_REMAINING_LIFETIME_SCOPE_UNKNOWN'],tests='Fresh64 REAL SQLite schema diagnostic rows; ORIGINAL entire source_bound_lifetime_negatives + official caller AST check only; completed metadata500k MAIN not rerun, no proxy claim.'),
future_adverse_mode=dict(execution_phase='original_adverse_shape',required_original_shape='adverse_reverse_interleaved',accepted_metadata_main_ROOT=mainpin,prior_matching_candidate_first_shape_ROOT=None,gate_status='UNKNOWN_NONACTIONABLE until genuine matching composed ROOT main+life acceptance',required_composed_ROOT_schema=dict(kind='ROOT_Trackio7_projection2_reader4_first_original_shape_actual_acceptance_v1',accepted_first_shape=True,reviewer='/root',candidate_reader=scope['candidate_reader4'],projection_source=scope['projection2'],completed_first_shape=dict(shape='metadata_live_order',original_history_rows=500000)),output_directory=str(D/'a7_f06'),external_output_directory=str(D/'a7_e06'),tests='All ORIGINAL adverse500k cold8+bulk64+version/strict20/reference/SQLite timeout latecommit-idempotence192/STATElossfences/warm/reader0closure/port+full lifetime preserved; separate exact execution required.'),
resources=resources,source_prep_limits=dict(initial_commit=5399239296,initial_physical=2649239296,runtime_commit=2612368384,runtime_physical=2112368384,self_private=536870912,wall_seconds=180,E_free_min=100000000000),
resource_arithmetic='Unchanged API05: native full numerical225/FULL+source39 extra72+cache256+API16+projection pointerworkspace64+owned controller512+watchdog512+fixture reader private1.5GB+independent ROOT512; no freed credit/cap or timeout increase. Smaller64 lifetime fixture retains exact full prospective/runtime/private guards.',
required_future_input_pins=[candidatepin,*sources,*officialpins.values(),mainpin,pin(scopepath),pin(amendpath)],
source_only_exclusions=['No candidate or official module import','No py_compile/proof execution/server/HTTP/DB','No live reader/production/control','No closed actual report/journal/audio/checkpoint read','No Backlog/refresh'],
known_SOURCE_process=dict(entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),numeric_OS_exit='Not yet observed inside running source generator',owned_child_processes=[]),
final_sample=guard(),elapsed_seconds=time.monotonic()-t0)
handoffpin=write_json(handoffpath,handoff)
result=dict(kind='Trackio7_official_lifecycle_proof8_SOURCE_result_v1',prepared=True,actual_execution=False,
candidate=candidatepin,RAW=rawpin,handoff=handoffpin,initial=initial,
official_SOURCE_pins=officialpins,changed_functions=changed,added_functions=added,whole_byte_inverse_exact=True,
entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),owned_child_processes=[],
final_sample=guard(),elapsed_seconds=time.monotonic()-t0)
resultpin=write_json(resultpath,result)
print(json.dumps(dict(candidate=candidatepin,RAW=rawpin,handoff=handoffpin,result=resultpin,initial=initial,
entry=result['entry'],launcher=result['launcher'],ordered_RAW_edits=len(rawledger['ordered_edits']),
changed_functions=changed,added_functions=added,final_sample=result['final_sample']),ensure_ascii=False))

