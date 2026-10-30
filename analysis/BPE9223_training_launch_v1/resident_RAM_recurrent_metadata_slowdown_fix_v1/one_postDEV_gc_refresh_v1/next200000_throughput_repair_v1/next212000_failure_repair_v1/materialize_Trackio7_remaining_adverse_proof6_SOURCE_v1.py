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
initial=write_json(D/'Trackio7_remaining_adverse_proof6_SOURCE_initial_v1.json',dict(kind='SOURCE_preparation_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=s,required=dict(commit=5399239296,physical=2649239296,private_max=536870912,E_free_min=100000000000),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),candidate_import_execution=False))
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

scopepath=D/'ROOT_API_remaining_original_adverse_shape_INACTIVE_SOURCE_scope_v1.json'
rawscope=scopepath.read_bytes();assert len(rawscope)==4797 and sha(rawscope)=='259d040b9d7f01d80ca4c9f0cc632b59ce5c84ef51ba1c8f4f7fcdb527933e50'
scope=json.loads(rawscope)
base=D/'Trackio7_real_API_mirror_scalar_CPU_integration_proof_v5.py'
old=base.read_bytes();assert len(old)==42945 and sha(old)=='c5f53f21d948e395b2bae2d76e7bacd1423166b0a2be0b43c7367d38c12d6372'
new=old.decode()
def rep(a,b):
    global new
    assert new.count(a)==1,(a[:80],new.count(a))
    new=new.replace(a,b)
binding=scope['completed_original_first_shape_ROOT']
binding_literal='dict(path='+repr(binding['path'])+',bytes=7286,sha256="ad1079d8e9247f824f9f27c9b51a0f5afce826fb1e013cfdff335252ac788368")'
gate='''    # Compose the CLOSED accepted original first shape with this still-unrun
    # original adverse shape. The receipt is authority only, no data re-audit.
    accepted_first_shape_pin=scope["accepted_first_shape_ROOT"]
    assert accepted_first_shape_pin==BINDING_LITERAL
    assert accepted_first_shape_pin in scope["input_pins"]
    assert pin(accepted_first_shape_pin["path"])==accepted_first_shape_pin
    accepted_first_shape=read(accepted_first_shape_pin["path"])
    assert accepted_first_shape["accepted_first_shape"]is True
    assert accepted_first_shape["reviewer"]=="/root"
    assert accepted_first_shape["completed_first_shape"]["shape"]=="metadata_live_order"
    assert accepted_first_shape["completed_first_shape"]["original_history_rows"]==500000
'''.replace('BINDING_LITERAL',binding_literal)
rep('    start=time.monotonic();children=[];identities=[];known_family={};records=[]\n',gate+'    start=time.monotonic();children=[];identities=[];known_family={};records=[]\n')
rep('        for shape in ("metadata_live_order","adverse_reverse_interleaved"):','        for shape in ("adverse_reverse_interleaved",):')
rep('            assert db.resolve()==(dbdir/(project+".db")).resolve()\n','''            save(case/"actual_fixture_DB_path_binding.json",dict(
                shape=shape,actual_returned_DB_path=str(db.resolve()),
                expected_DB_path=str((dbdir/(project+".db")).resolve()),
                TRACKIO_DIR=os.environ.get("TRACKIO_DIR"),strict_isolation_check_unchanged=True))
            assert db.resolve()==(dbdir/(project+".db")).resolve()
''')
rep('production_touched=False,no_durable_live_cause_or_speed_claim=True))','production_touched=False,no_durable_live_cause_or_speed_claim=True,accepted_original_first_shape_ROOT=accepted_first_shape_pin,this_run_only_remaining_original_adverse_shape=True,completed_first_shape_rerun=False))')
path=D/'Trackio7_real_API_original_remaining_adverse_CPU_proof_v6.py'
candidate=write_source(path,new.encode())
def tops(raw):
    return {n.name:ast.dump(n,include_attributes=False)for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
a=tops(old.decode());b=tops(new);assert set(a)==set(b)
assert [k for k in a if a[k]!=b[k]]==['fixture_body']
# Existing detailed lifetime and exact negative owners are unchanged.
for name in ('server_child','source_bound_lifetime_negatives','request','controller'):
    assert a[name]==b[name]
labels=['Run ONLY still-unexecuted original adverse_reverse_interleaved500000 in fresh process; completed original metadata_live_order shape never rerun',
'Exact immutable accepted_first_shape_ROOT byte/path pin required in future scope/input_pins, accepted_first_shape:true and completed metadata_live_order500000 checked only as ROOT authority, not actual data audit',
'Record actual returned DB path/expected DB path/shape/TRACKIO_DIR immediately before unchanged strict isolation assert',
'Final result explicitly binds composed prior first-shape ROOT evidence and states this run only remaining adverse shape',
'Every original adverse generator/cold8+bulk64+version/20s/equivalence/SQLite lock+latecommit+idempotence192/STATE+loss fences/warmdecode/serverclosure/portrelease and all lifetime negatives/officialcaller/Gradio assertion unchanged']
ledger=raw_diff(old,new.encode(),base,path,labels)
ledger.update(top_function_delta=['fixture_body'],all_other_top_functions_AST_equal=True,server_child_negative_fences_AST_equal=True,lifetime_negatives_AST_equal=True,ROOT_source_scope=pin_bytes(scopepath,rawscope),accepted_original_first_shape_ROOT=binding)
rp=write_json(D/'Trackio7_remaining_adverse_proof6_ordered_RAW_AST_v1.json',ledger)
prior=[]
for identity in [dict(pid=29924,ctime=1792863718.8840935,ppid=10164),dict(pid=10164,ctime=1792863718.87449,ppid=29168)]:
    try:q=psutil.Process(identity['pid']);same=q.create_time()==identity['ctime']and q.ppid()==identity['ppid']
    except psutil.NoSuchProcess:same=False
    prior.append(dict(identity=identity,exact_dead=not same))
assert all(x['exact_dead']for x in prior)
hp=write_json(D/'Trackio7_remaining_adverse_proof6_SOURCE_handoff_v1.json',dict(kind='Trackio7_remaining_original_adverse_proof6_INACTIVE_SOURCE_handoff_v1',
candidate=candidate,base=pin_bytes(base,old),RAW_AST=rp,source_scope=pin_bytes(scopepath,rawscope),
accepted_first_shape_ROOT=binding,prior_CPU03_ROOT=scope['prior_CPU03_ROOT'],this_run_only_shape='adverse_reverse_interleaved',original_history_rows=500000,
proposed_command=[str(P/'work/monitor-env/Scripts/python.exe'),'-X','utf8','-B',str(path),'--ROOT-review','UNKNOWN'],
future_fresh_outputs=dict(body=str(D/'a7_f04'),external=str(D/'a7_e04')),
future_exact_ROOT_execution_scope='UNKNOWN / REQUIRED',candidate_import_execution=False,
watchdog1_unchanged=dict(path=str(D/'owned_source_CPU_proof_watchdog_v1.py'),bytes=8988,sha256='04d188dd774c4a9bed0ce896505a1b8563adbc1b5135c207da26d6266daad16a'),
reader3_helper3_projector1_unchanged=True,no_additional_controller=True,
resources_unchanged=dict(prospective_commit=7989758336,prospective_physical=5239758336,runtime_commit=5471322880,runtime_physical=4971322880,family_private=2573741824,reader_private=1500000000,controller_private=536870912,watchdog_private=536870912,body_wall=160,external_wall=180,HTTP=20),
retained_tests=labels[4],initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),prior_API03_completion_exact_deaths=prior,
limitations=['SOURCE preparation only, no import or execution','Completed first shape not rerun; future composed acceptance requires this remaining shape plus all lifetime negatives pass','No original production212 cause or permanent speed claim']))
s=sample()
assert s['commit']>=2612368384 and s['physical']>=2112368384 and s['private']<=536870912 and s['E_free']>=100000000000 and time.monotonic()-t0<=180
result=write_json(D/'Trackio7_remaining_adverse_proof6_SOURCE_result_v1.json',dict(candidate=candidate,RAW_AST=rp,handoff=hp,initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),prior_API03_completion_exact_deaths=prior,final=s,candidate_import_execution=False,wall=time.monotonic()-t0))
print(json.dumps(dict(candidate=candidate,RAW_AST=rp,handoff=hp,result=result,initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()))))

