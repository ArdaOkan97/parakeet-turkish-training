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
initial=write_json(D/'source40_typed02_actual_completion_initial_BEFORE_predicate_v1.json',dict(kind='SOURCE_preparation_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=s,required=dict(commit=6565255808,physical=3815255808,private_max=536870912,E_free_min=100000000000),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),candidate_import_execution=False))
assert s['commit']>=6565255808 and s['physical']>=3815255808 and s['private']<=536870912 and s['E_free']>=100000000000

actual_tool="{\"chunk_id\":\"4d279b\",\"wall_time_seconds\":1.2491273999999999,\"exit_code\":0,\"original_token_count\":0,\"output\":\"\"}"
actual_tool=json.loads(actual_tool)

tool_pin=write_json(D/'source40_typed02_actual_owned_TOOL_result_v1.json',actual_tool)
roots=[D/'source40_saved212000_typed_metadata_actual02_v1',D/'source40_saved212000_typed_metadata_external_actual02_v1']
captures=[];values={};identities={};references={}
def walk(v,ref):
    if isinstance(v,dict):
        ct=v.get('ctime',v.get('create_time'))
        if type(v.get('pid'))is int and type(ct)in(int,float):
            q=dict(pid=v['pid'],ctime=ct)
            if type(v.get('ppid'))is int:q['ppid']=v['ppid']
            key=(q['pid'],q['ctime']);identities.setdefault(key,q);references.setdefault(key,[]).append(ref)
        for k,x in v.items():walk(x,ref+'.'+str(k))
    elif isinstance(v,list):
        for i,x in enumerate(v):walk(x,ref+'['+str(i)+']')
for root in roots:
    files=sorted(root.rglob('*'));assert len(files)<=64
    for p in files:
        if not p.is_file():continue
        assert p.suffix in('.json','.log')
        b=p.read_bytes();assert len(b)<=262144
        rel=str(p.relative_to(D));q=dict(pin=pin_bytes(p,b),relative=rel)
        if p.suffix=='.json':
            q['value']=json.loads(b);values[rel]=q['value'];walk(q['value'],rel)
        else:q['text_utf8']=b.decode('utf-8',errors='replace')
        captures.append(q)
numericpath=D/'source40_saved212000_typed02_direct_numeric_exit_v1.json';nb=numericpath.read_bytes();numeric=dict(pin=pin_bytes(numericpath,nb),value=json.loads(nb))
deaths=[]
for key,w in identities.items():
    try:
        p=psutil.Process(w['pid']);same=abs(p.create_time()-w['ctime'])<.001
        q=dict(identity=w,exact_dead=not same,current=who(p)if same else None)
    except psutil.NoSuchProcess:q=dict(identity=w,exact_dead=True)
    except BaseException as e:q=dict(identity=w,exact_dead='UNKNOWN',error=type(e).__name__+':'+str(e)[:500])
    q['artifact_references']=references[key];deaths.append(q)
body=values[str(roots[0].relative_to(D)/'result.json')];external=values[str(roots[1].relative_to(D)/'result.json')]
summary=dict(actual_TOOL_exit=actual_tool['exit_code'],direct_Python_numeric_OS_exit=numeric['value']['direct_Python_numeric_OS_exit'],
    external_Popen_returncode=external['Popen_returncode'],external_passed=external['passed'],guard_error=external['guard_error'],cleanup=external['cleanup'],
    body_passed=body['passed'],actual_count=body['count'],positive=sum(r['outcome']=='positive'for r in body['results']),
    strict_rejects=sum(r['outcome']=='reject'for r in body['results']),method_scope=body['scope'],peak_family_private=external['peak_family_private'],
    known_job_identity_count=len(deaths),all_exact_known_job_identities_dead=all(r['exact_dead']is True for r in deaths),
    no_scientific_or_control_execution=body['all_original_science_restore_audio_GPU_draw_journal_control']is False)
completion=write_json(D/'source40_typed02_actual_completion_v1.json',dict(kind='owned_source40_typed02_actual_completion_v1',actual_TOOL=tool_pin,direct_numeric=numeric,
    summary=summary,NEW_scalar_artifacts=captures,new_output_pins=[q['pin']for q in captures],all_known_original_metadata_identity_deaths=deaths,
    completion_entry=who(psutil.Process()),completion_launcher=who(psutil.Process().parent()),initial=initial,
    all_external_ROOT_reader_STAT_psutil_authorities_explicit_METHOD_STUB=True,
    no_real_pointer_journal_checkpoint_plan_audio_GPU_native_reader_training_P_S_GC_or_Backlog_action=True,
    no_retry_rebind_source_or_guard_change=True,CPU04_and_typed01_closed_artifacts_not_read=True,final_sample=sample(),elapsed=time.monotonic()-t0))
print(json.dumps(dict(completion=completion,summary=summary,body_result=next(q['pin']for q in captures if q['relative']==str(roots[0].relative_to(D)/'result.json')),
    external_result=next(q['pin']for q in captures if q['relative']==str(roots[1].relative_to(D)/'result.json')),
    original_job_deaths=deaths,completion_entry=who(psutil.Process()),completion_launcher=who(psutil.Process().parent()),initial=initial),indent=2))

