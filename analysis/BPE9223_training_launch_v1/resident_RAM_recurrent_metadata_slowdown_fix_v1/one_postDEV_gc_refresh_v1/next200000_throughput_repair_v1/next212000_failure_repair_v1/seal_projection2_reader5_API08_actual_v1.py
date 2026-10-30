import ctypes,hashlib,json,os,pathlib,shutil,time
import psutil
D=pathlib.Path(r'E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/next200000_throughput_repair_v1/next212000_failure_repair_v1')
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
t0=time.monotonic()
def who(p):return dict(pid=p.pid,ctime=p.create_time(),ppid=p.ppid())
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):
    with p.open('x',encoding='utf-8',newline='\n')as f:
        json.dump(x,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def sha(b):return hashlib.sha256(b).hexdigest()
def pin(p,b=None):
    if b is None:b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=sha(b))
class M(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong)for n in ('totalphysical','physical','commitlimit','commit','totalvirtual','virtual','extended')]
def sample():
    m=M();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError()
    return dict(commit=m.commit,physical=m.physical,private=psutil.Process().memory_info().private,E_free=shutil.disk_usage(D).free)
scopepath=D/'ROOT_helper4_reader5_original_metadata_exact_ONE_API08_scope_v1.json'
scope=read(scopepath);limits=scope['resources']
initialpath=D/'projection2_reader5_API08_completion_initial_v1.json'
initial=dict(kind='API08_scalar_completion_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=sample(),required=limits,TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'))
save(initialpath,initial)
assert initial['sample']['commit']>=limits['prospective_commit']and initial['sample']['physical']>=limits['prospective_physical']
assert initial['sample']['private']<=limits['self_private']and initial['sample']['E_free']>=100000000000
def guard():
    s=sample()
    assert s['commit']>=limits['runtime_commit']and s['physical']>=limits['runtime_physical']
    assert s['private']<=limits['self_private']and s['E_free']>=100000000000
    assert time.monotonic()-t0<180
    return s
roots=[D/'a7_f08',D/'a7_e08']
scalars=[];stat_only=[];known={};total=0
def identities(value,origin):
    if isinstance(value,dict):
        c=value.get('ctime',value.get('create_time'))
        if type(value.get('pid'))is int and type(c)in(int,float):
            key=(value['pid'],float(c));entry=dict(pid=value['pid'],ctime=float(c))
            if type(value.get('ppid'))is int:entry['ppid']=value['ppid']
            if key not in known:known[key]=dict(identity=entry,origins=[])
            known[key]['origins'].append(origin)
        for v in value.values():identities(v,origin)
    elif isinstance(value,list):
        for v in value:identities(v,origin)
for root in roots:
    if not root.exists():
        stat_only.append(dict(path=str(root),exists=False));continue
    for p in sorted(root.rglob('*')):
        guard()
        if not p.is_file():continue
        st=p.stat()
        if p.suffix.lower()not in('.json','.log')or p.name.endswith('.db')or '-wal'in p.name or '-shm'in p.name:
            stat_only.append(dict(path=str(p),bytes=st.st_size,mtime=st.st_mtime,content_read=False));continue
        if st.st_size>262144:
            with p.open('rb')as f:f.seek(max(0,st.st_size-65536));b=f.read(65536)
            scalars.append(dict(path=str(p),file_bytes=st.st_size,captured_tail_bytes=len(b),captured_tail_sha256=sha(b),whole_file_read=False,text=b.decode('utf-8','replace')))
        else:
            b=p.read_bytes();total+=len(b);assert total<=4194304
            item=dict(pin=pin(p,b),whole_file_read=True)
            try:
                value=json.loads(b.decode('utf-8-sig'));item['value']=value;identities(value,str(p))
            except (UnicodeError,json.JSONDecodeError):item['text']=b.decode('utf-8','replace')
            scalars.append(item)
directpath=D/'projection2_reader5_API08_direct_OS_exit_v1.json'
direct=read(directpath)
assert direct['numeric_OS_exit']==0
scalars.append(dict(pin=pin(directpath),value=direct,whole_file_read=True))
input_pins=[]
for item in scope['input_pins']:
    p=pathlib.Path(item['path']);now=pin(p)
    input_pins.append(dict(expected=item,current=now,exact=now==item));assert now==item
deaths=[]
for v in known.values():
    item=v.copy();ident=v['identity']
    try:
        p=psutil.Process(ident['pid'])
        current=who(p)
        exact=current['ctime']==ident['ctime']
        item.update(exact_dead=not exact,current=current if exact else dict(pid_reused=current))
    except psutil.NoSuchProcess:item.update(exact_dead=True,current_absent=True)
    deaths.append(item)
completion=dict(kind='API08_original_metadata_shape_actual_scalar_completion_v1',owner='/root/memory_diagnosis',
actual_TOOL=dict(exit_code=0,chunk_id='0766fb',terminal=True),direct_Python_OS_exit=direct,
scope=pin(scopepath),initial=pin(initialpath),sealer_source=pin(pathlib.Path(__file__)),
entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),new_scalar_artifacts=scalars,
new_artifact_STAT_ONLY=stat_only,recorded_actual_identities=deaths,input_pins_current=input_pins,
no_retry_or_control=True,no_old_reader4_MAIN_authority_or_completed_shape_rerun=True,no_adverse_execution=True,
actual_method_closure='Only as explicitly recorded in captured receipts; no method or OS exit inferred from process death',
final_sample=guard(),elapsed_seconds=time.monotonic()-t0)
out=D/'projection2_reader5_API08_actual_completion_v1.json';save(out,completion)
result=[x.get('value')for x in scalars if x.get('pin',{}).get('path','').endswith('result.json')]
stderr=[x for x in scalars if 'stderr'in x.get('pin',{}).get('path','')]
print(json.dumps(dict(completion=pin(out),initial=pin(initialpath),sealer=completion['entry'],launcher=completion['launcher'],
results=result,stderr=stderr,recorded_actual_identities=deaths,STAT_ONLY=stat_only),ensure_ascii=False))

