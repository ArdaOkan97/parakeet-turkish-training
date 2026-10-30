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
initial=write_json(D/'Trackio7_APIproof4_CWD_SOURCE_initial_v1.json',dict(kind='SOURCE_preparation_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=s,required=dict(commit=5936110208,physical=3186110208,private_max=536870912,E_free_min=100000000000),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),candidate_import_execution=False))
assert s['commit']>=5936110208 and s['physical']>=3186110208 and s['private']<=536870912 and s['E_free']>=100000000000

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

scopepath=D/'ROOT_Trackio7_APIproof4_invalid_fixture_CWD_INACTIVE_SOURCE_correction_scope_v1.json'
scope_raw=scopepath.read_bytes()
assert len(scope_raw)==3952 and sha(scope_raw)=='1e87e9723d8b233d999889f816cc8343993f75ceb3cc1e0729dcfadc5c3899fe'
scope=json.loads(scope_raw)
oldpath=D/'Trackio7_real_API_mirror_scalar_CPU_integration_proof_v3.py'
old=oldpath.read_bytes()
assert len(old)==38186 and sha(old)=='0620a206d634ee0a8e68f8d706eac0da4462b5cabb85eb22267d7143fcebfb59'
a=b'child=subprocess.Popen(argv,stdout=out,stderr=err,cwd=case,creationflags=subprocess.CREATE_NO_WINDOW)'
b=b'child=subprocess.Popen(argv,stdout=out,stderr=err,cwd=HERE,creationflags=subprocess.CREATE_NO_WINDOW)'
assert old.count(a)==1
new=old.replace(a,b)
path=D/'Trackio7_real_API_mirror_scalar_CPU_integration_proof_v4.py'
candidate=write_source(path,new)
ot=ast.parse(old.decode());nt=ast.parse(new.decode())
old_funcs={n.name:ast.dump(n,include_attributes=False)for n in ot.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
new_funcs={n.name:ast.dump(n,include_attributes=False)for n in nt.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
assert set(old_funcs)==set(new_funcs)
assert [k for k in old_funcs if old_funcs[k]!=new_funcs[k]]==['fixture_body']
changed=[]
for n in ast.walk(nt):
    if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='subprocess'and n.func.attr=='Popen':
        for kw in n.keywords:
            if kw.arg=='cwd'and isinstance(kw.value,ast.Name)and kw.value.id=='HERE':
                changed.append(kw);kw.value.id='case'
assert len(changed)==1 and ast.dump(nt,include_attributes=False)==ast.dump(ot,include_attributes=False)
labels=['One exact original fixture server-child Popen cwd=case -> cwd=HERE; source E root is shorter and all typed fixture paths remain absolute','Every remaining source byte and AST node unchanged, cold-first8query+bulk_log64+version/two500000 shapes/20s and originallimits unchanged','Future fresh short namespaces a7_f02/a7_e02 only proposals; no actual execution admitted']
ledger=raw_diff(old,new,oldpath,path,labels)
ledger.update(one_exact_literal_edit=True,AST_inverse_exact=True,all_other_top_functions_AST_equal=True,source_scope=pin_bytes(scopepath,scope_raw))
rp=write_json(D/'Trackio7_APIproof4_CWD_ordered_RAW_AST_v1.json',ledger)
prior=[]
for ident in [dict(pid=20892,ctime=1792861933.7874331,ppid=33728),dict(pid=33728,ctime=1792861933.7771382,ppid=41996),dict(pid=33076,ctime=1792862083.007372,ppid=32976),dict(pid=32976,ctime=1792862082.9965632,ppid=12444)]:
    try:q=psutil.Process(ident['pid']);same=q.create_time()==ident['ctime']and q.ppid()==ident['ppid']
    except psutil.NoSuchProcess:same=False
    prior.append(dict(identity=ident,exact_dead=not same))
assert all(x['exact_dead']for x in prior)
hp=write_json(D/'Trackio7_APIproof4_CWD_SOURCE_handoff_v1.json',dict(kind='Trackio7_APIproof4_CWD_INACTIVE_SOURCE_handoff_v1',candidate=candidate,base=pin_bytes(oldpath,old),RAW_AST=rp,source_scope=pin_bytes(scopepath,scope_raw),source_only=True,candidate_import_execution=False,
closed_actual01_failure=scope['actual01_failure_binding'],future_outputs=scope['future_exact_output_path_proposal'],future_execution_scope='UNKNOWN / REQUIRED',
proposed_command_prefix=[str(P/'work/monitor-env/Scripts/python.exe'),'-X','utf8','-B',str(path),'--ROOT-review','UNKNOWN'],
watchdog_unchanged=dict(path=str(D/'owned_source_CPU_proof_watchdog_v1.py'),bytes=8988,sha256='04d188dd774c4a9bed0ce896505a1b8563adbc1b5135c207da26d6266daad16a'),
reader_candidate_unchanged=dict(path=str(D/'serve_BPE9223_Trackio_live_v7_INACTIVE_v3.py'),bytes=25922,sha256='2bf6dbf3518667cf7f4f943c87c6f966101c6f55cab1d5679ced25c3f401db9c'),
resources_future_unchanged=dict(prospective_commit=7989758336,prospective_physical=5239758336,runtime_commit=5471322880,runtime_physical=4971322880,family_private_bytes=2573741824,reader_private_bytes=1500000000,controller_private_bytes=536870912,watchdog_private_bytes=536870912,body_wall_seconds=160,external_wall_seconds=180,HTTP_seconds=20),
initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),prior_metadata_exact_deaths=prior))
s=sample()
assert s['commit']>=3149239296 and s['physical']>=2649239296 and s['private']<=536870912 and s['E_free']>=100000000000 and time.monotonic()-t0<=180
result=write_json(D/'Trackio7_APIproof4_CWD_SOURCE_result_v1.json',dict(candidate=candidate,RAW_AST=rp,handoff=hp,initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),prior_metadata_exact_deaths=prior,final=s,candidate_import_execution=False))
print(json.dumps(dict(candidate=candidate,RAW_AST=rp,handoff=hp,result=result,initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()))))

