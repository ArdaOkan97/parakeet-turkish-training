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
initial=write_json(D/'Trackio7_cold_first_burst_proof_SOURCE_v3_initial_v1.json',dict(kind='SOURCE_preparation_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=s,required=dict(commit=5399239296,physical=2649239296,private_max=536870912,E_free_min=100000000000),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),candidate_import_execution=False))
assert s['commit']>=5399239296 and s['physical']>=2649239296 and s['private']<=536870912 and s['E_free']>=100000000000

burst_code="            # First actual get_logs_batch is COLD and released in the same burst\n            # as eight scalar clients, append64, and the existing GET/version.\n            cold_status=read(area/\"current_status.json\")\n            cold_projection=cold_status[\"scalar_query_projection\"]\n            assert cold_projection[\"calls\"]==0 and cold_projection[\"source_rows_decoded\"]==0\n            extra=[dict(project=project,run=name,run_id=name,step=500001+i,metrics={\"append/value\":i},log_id=\"api_append_\"+str(i))for i in range(64)]\n            burst_gate=threading.Barrier(11) # Ten real requests plus controller.\n            burst_events=[]\n            burst_events_lock=threading.Lock()\n            burst_origin=time.monotonic()\n            save(case/\"cold_burst_requests.json\",dict(expected_requests=10,events=[]))\n            def burst_request(label,api_name,kwargs=None):\n                burst_gate.wait(timeout=10)\n                began=time.monotonic()\n                failure=None\n                try:\n                    answer,actual_elapsed=request(base,api_name,kwargs,token)\n                    # Do not retain concurrent response graphs; independent original\n                    # equivalence is checked after ALL requests have settled.\n                    del answer\n                    return dict(label=label,elapsed_seconds=actual_elapsed)\n                except BaseException as caught:\n                    failure=dict(type=type(caught).__name__,message=str(caught)[:500])\n                    raise\n                finally:\n                    ended=time.monotonic()\n                    event=dict(label=label,API=api_name,start_seconds=began-burst_origin,\n                        end_seconds=ended-burst_origin,elapsed_seconds=ended-began,error=failure)\n                    with burst_events_lock:\n                        assert len(burst_events)<10\n                        burst_events.append(event)\n                        save(case/\"cold_burst_requests.json\",dict(expected_requests=10,events=list(burst_events)))\n            with concurrent.futures.ThreadPoolExecutor(max_workers=10)as pool:\n                scalar_futures=[pool.submit(burst_request,\"scalar_\"+str(i),\"get_logs_batch\",payload)for i in range(8)]\n                write_future=pool.submit(burst_request,\"append64\",\"bulk_log\",dict(logs=extra,hf_token=None))\n                version_future=pool.submit(burst_request,\"version\",\"/version\")\n                burst_gate.wait(timeout=10)\n                concurrent_results=[future.result(timeout=20)for future in scalar_futures]\n                write_result=write_future.result(timeout=20)\n                version_result=version_future.result(timeout=20)\n            assert len(burst_events)==10 and all(e[\"error\"]is None and e[\"elapsed_seconds\"]<20 for e in burst_events)\n            scalar_events=[e for e in burst_events if e[\"API\"]==\"get_logs_batch\"]\n            write_event=next(e for e in burst_events if e[\"label\"]==\"append64\")\n            version_event=next(e for e in burst_events if e[\"label\"]==\"version\")\n            assert len(scalar_events)==8\n            # Observed CLIENT intervals only, no claim about exclusive backend CPU.\n            assert any(e[\"start_seconds\"]<=write_event[\"end_seconds\"]and e[\"end_seconds\"]>=write_event[\"start_seconds\"]for e in scalar_events)\n            assert any(e[\"start_seconds\"]<=version_event[\"end_seconds\"]and e[\"end_seconds\"]>=version_event[\"start_seconds\"]for e in scalar_events)\n            wait(lambda:read(area/\"current_status.json\").get(\"scalar_query_projection\",{}).get(\"source_rows_decoded\",0)>=500000,15)\n            status_before=read(area/\"current_status.json\")\n            decoded_before=status_before[\"scalar_query_projection\"][\"source_rows_decoded\"]\n            # No original reference query or isolated dashboard query warmed the\n            # first burst. Compare only at the settled post-burst fence.\n            after_expected=reference([dict(run=name,run_id=name)])\n            after_actual,after_elapsed=request(base,\"get_logs_batch\",payload,token);assert after_actual==after_expected\n"

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

base_path=D/'Trackio7_real_API_mirror_scalar_CPU_integration_proof_v2.py'
target_path=D/'Trackio7_real_API_mirror_scalar_CPU_integration_proof_v3.py'
base=base_path.read_bytes()
assert len(base)==35138 and sha(base)=='255091d5cb9bf33f1abb0cad3b571c7c9c51d87cbb1898e9586b2d32da49df39'
text=base.decode('utf-8')
start=text.index('            expected=reference([dict(run=name,run_id=name)])\n')
end=text.index('            # Name-only identity and explicit fallback',start)
old_block=text[start:end]
assert 'max_workers=9'in old_block and 'after_actual==after_expected'in old_block
text=text[:start]+burst_code+text[end:]
old_record='original_history_rows=500000,queries=8,all_normal_requests_below20=True,'
new_record='original_history_rows=500000,queries=8,first_actual_get_logs_batch_in_cold_concurrent_burst=True,cold_burst_every_request_elapsed=list(burst_events),cold_burst_query_write_and_version_client_overlap=True,all_normal_requests_below20=True,'
assert text.count(old_record)==1
text=text.replace(old_record,new_record)
candidate=text.encode('utf-8')
assert not target_path.exists()
ast.parse(text)
base_tree=ast.parse(base.decode('utf-8'));candidate_tree=ast.parse(text)
base_functions={n.name:ast.dump(n,include_attributes=False)for n in base_tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
candidate_functions={n.name:ast.dump(n,include_attributes=False)for n in candidate_tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
assert base_functions.keys()==candidate_functions.keys()
unchanged=[name for name in base_functions if base_functions[name]==candidate_functions[name]]
changed=[name for name in base_functions if base_functions[name]!=candidate_functions[name]]
assert changed==['fixture_body']
assert 'for shape in ("metadata_live_order","adverse_reverse_interleaved")'in text
assert 'range(0,500000,1000)'in text and 'decoded_after-decoded_before<=512'in text
assert 'limits["wall_seconds"]==limits["proof_body_wall_seconds"]<=160'in text
for label in ('DB_lock_timeout_then_potential_late_original_commit_and_idempotent_replay','local_STATE_publication_failure_preserves_committed_STATE','explicit_lost_response_exact_batch_replayed','source_bound_lifetime_negatives'):
    assert label in text
pin_candidate=write_source(target_path,candidate)
rawrecord=raw_diff(base,candidate,base_path,target_path,[
'FIRST actual get_logs_batch moved into10request barrier burst:8scalar+originalbulk_log64+existingGET/version',
'No isolated query or original reference scan before cold burst; all original-equivalence comparisons at settled post-burst fence',
'Every10cold-burst request records start/end/elapsed/error as it finishes, in bounded E-only diagnostic receipt',
'Require observed CLIENT interval overlap with write and version, without claiming internal exclusive backend timing',
'Both500000row shapes, original20s request bound,180external/160body cap, warm-new-decode bound512 and all failure/lifetime/cleanup tests unchanged'])
raw=write_json(D/'Trackio7_cold_first_burst_proof_SOURCE_v3_ordered_RAW_v1.json',dict(record=rawrecord,full_AST_base_sha256=astsha(base),full_AST_candidate_sha256=astsha(candidate),top_functions_unchanged=unchanged,top_functions_changed=changed,entire_base_inverse_exact=True,candidate_import_execution=False))
immutable_paths=[D/'owned_source_CPU_proof_watchdog_v1.py',D/'serve_BPE9223_Trackio_live_v7_INACTIVE_v3.py',D/'Trackio_owned_bounded_sync_API_dispatch_v3.py',D/'CPU14_original_startup_firstpacket_CPU_proof_v2.py']
immutable_pins=[pin(q)for q in immutable_paths]
prior=dict(pid=41420,ctime=1792860699.3796904,ppid=35456)
prior_launcher=dict(pid=35456,ctime=1792860699.368893,ppid=35844)
prior_deaths=[]
for w in (prior,prior_launcher):
    try:p=psutil.Process(w['pid']);alive=p.is_running()and p.create_time()==w['ctime']
    except psutil.NoSuchProcess:alive=False
    prior_deaths.append(dict(identity=w,exact_dead=not alive,OS_exit='UNKNOWN unless already directly recorded'))
assert time.monotonic()-t0<180
last=sample();assert last['commit']>=2612368384 and last['physical']>=2112368384 and last['private']<=536870912 and last['E_free']>=100000000000
handoff=write_json(D/'Trackio7_cold_first_burst_proof_SOURCE_v3_handoff_v1.json',dict(kind='INACTIVE_Trackio7_cold_first_concurrent_burst_SOURCE_v3_handoff_v1',proof=pin_candidate,RAW=raw,initial=initial,initial_sample=s,final_sample=last,immutable_source_inputs=immutable_pins,original_resource_PROPOSAL_unchanged=pin(D/'NEW212000_future_proof_resource_calculator_PROPOSAL_v1.json'),ROOT_mechanical_SOURCE_review_named='ROOT_final_proofs_whole_RAW_AST_resource_review_v1/review.json',ROOT_review_bytes=51242,ROOT_review_sha256='fd766323d889c376d1e91639d8478d3696eb1ee3d33e81eeb00f5f9849da034a',proposed_argv=[str(P/'work/monitor-env/Scripts/python.exe'),'-X','utf8','-B',str(target_path),'--ROOT-review','UNKNOWN_FUTURE_SEPARATE_ROOT_SCOPE'],future_body_and_external_output_names='UNKNOWN_FRESH_E_ONLY; no implicit reuse',actual_proof_execution_admitted=False,prior_known_metadata_deaths=prior_deaths,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),wall_seconds=time.monotonic()-t0))
result=write_json(D/'Trackio7_cold_first_burst_proof_SOURCE_v3_result_v1.json',dict(passed=True,stage='SOURCE materialization and AST/whole-byte RAW only',proof=pin_candidate,RAW=raw,handoff=handoff,initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),candidate_import_execution=False,actual_proof_runs=0,production_or_reader_control=False,all_other_candidate_sources_unchanged=True))
print(json.dumps(dict(proof=pin_candidate,RAW=raw,handoff=handoff,result=result,initial=initial,initial_sample=s,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),prior_metadata=prior_deaths)))

