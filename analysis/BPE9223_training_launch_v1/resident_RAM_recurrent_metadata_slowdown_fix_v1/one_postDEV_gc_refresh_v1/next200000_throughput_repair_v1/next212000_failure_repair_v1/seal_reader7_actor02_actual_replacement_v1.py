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
initial=write_json(D/'reader7_actor02_completion_initial_BEFORE_predicate_v1.json',dict(kind='SOURCE_preparation_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=s,required=dict(commit=6028384896,physical=3278384896,private_max=536870912,E_free_min=100000000000),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),candidate_import_execution=False))
assert s['commit']>=6028384896 and s['physical']>=3278384896 and s['private']<=536870912 and s['E_free']>=100000000000

actual_tool="{\"chunk_id\":\"2347e4\",\"wall_time_seconds\":1.8184054,\"exit_code\":0,\"original_token_count\":0,\"output\":\"\"}"
actual_tool=json.loads(actual_tool)

tool=write_json(D/'reader7_actor02_actual_owned_TOOL_result_v1.json',actual_tool)
root=D/'reader7_failed6_replacement_actual02_v1';artifacts=[];values={}
for p in sorted(root.iterdir()):
    if not p.is_file():continue
    assert p.suffix in('.json','.log')
    b=p.read_bytes();assert len(b)<=65536
    q=dict(pin=pin_bytes(p,b),live_reader_log=p.suffix=='.log')
    if p.suffix=='.json':q['value']=json.loads(b);values[p.name]=q['value']
    else:q['text_utf8']=b.decode('utf-8',errors='replace');q['qualification']='Snapshot of open/live reader log; no future immutability claim'
    artifacts.append(q)
result=values['result.json'];first=values['initial_BEFORE_predicate.json']
numericpath=D/'reader7_failed6_replacement_actual02_direct_numeric_exit_v1.json';b=numericpath.read_bytes();numeric=dict(pin=pin_bytes(numericpath,b),value=json.loads(b))
deaths=[]
for w in (first['entry'],first['launcher']):
    try:
        p=psutil.Process(w['pid']);same=abs(p.create_time()-w['ctime'])<.001
        q=dict(identity=w,exact_dead=not same,current=who(p)if same else None)
    except psutil.NoSuchProcess:q=dict(identity=w,exact_dead=True)
    except BaseException as e:q=dict(identity=w,exact_dead='UNKNOWN',error=type(e).__name__+':'+str(e)[:500])
    deaths.append(q)
old=[]
for w in (dict(pid=33388,ctime=1792855546.8493547,ppid=38980),dict(pid=38980,ctime=1792855546.824607,ppid=26648)):
    try:p=psutil.Process(w['pid']);same=abs(p.create_time()-w['ctime'])<.001;old.append(dict(identity=w,exact_dead=not same,current=who(p)if same else None))
    except psutil.NoSuchProcess:old.append(dict(identity=w,exact_dead=True))
area=pathlib.Path('E:/gpu_training_scratchpad/analysis/BPE9223_Trackio_live_v1')
compacts=[]
for name in ('owner.json','current_status.json','mirror_state.json'):
    p=area/name
    try:
        b=p.read_bytes();assert len(b)<=65536;st=p.stat()
        q=dict(pin=pin_bytes(p,b),mtime_ns=st.st_mtime_ns,value=json.loads(b),mutable_current_snapshot=True)
    except (OSError,json.JSONDecodeError)as e:q=dict(path=str(p),current_capture='UNKNOWN',error=type(e).__name__+':'+str(e)[:500])
    compacts.append(q)
pair=[]
for label in ('new_launch_identity','new_reader_identity'):
    w=result.get(label)
    if w is None:pair.append(dict(role=label,actual='UNRECORDED'));continue
    try:
        p=psutil.Process(w['pid']);observed=who(p)
        pair.append(dict(role=label,recorded=w,observed=observed,exact_alive=abs(observed['ctime']-w['ctime'])<.001 and observed['ppid']==w['ppid'],
            cmdline=p.cmdline(),private=p.memory_info().private))
    except psutil.NoSuchProcess:pair.append(dict(role=label,recorded=w,exact_alive=False))
try:ports=[dict(pid=c.pid,address=list(c.laddr))for c in psutil.net_connections(kind='tcp')if c.status==psutil.CONN_LISTEN and c.laddr.port==7860]
except BaseException as e:ports=dict(UNKNOWN=type(e).__name__+':'+str(e)[:500])
destination=P/'work/research/serve_BPE9223_Trackio_live_v7.py';source_pin=pin(destination)if destination.exists()else None
currentstatus=next((q.get('value')for q in compacts if q.get('pin',{}).get('path','').endswith('current_status.json')),None)
summary=dict(actor_TOOL_exit=actual_tool['exit_code'],direct_Python_numeric_OS_exit=numeric['value']['direct_Python_numeric_OS_exit'],
    actor_passed=result['passed'],primary=result['primary'],old_actions=result['old_targets'],source_copied=result['source_copied'],
    new_launch_attempted=result['new_launch_attempted'],actual_new_pair=pair,
    actual_new_reader_current_snapshot=None if currentstatus is None else {k:currentstatus.get(k)for k in ('pid','create_time','source','source_sha256','mirror_thread_alive','HTTP_ready','reader_ready','pending_batch','error','journal_path','journal_offset','last_verified_checkpoint_update','training_worker_alive','updated_unix','API_dispatch','scalar_query_projection')},
    old_pair_exactdead=all(x['exact_dead']is True for x in old),current_port=ports,new_source_pin=source_pin,
    training_or_database_manual_control=False,new_reader_Popen_exit='RUNNING pollNone if recorded; no wait0 or OSexit inferred')
cp=write_json(D/'reader7_actor02_actual_replacement_completion_v1.json',dict(kind='owned_reader7_actor02_actual_replacement_completion_v1',summary=summary,
    actual_TOOL=tool,direct_numeric=numeric,NEW_actor_scalar_artifacts=artifacts,new_actor_output_pins=[q['pin']for q in artifacts],
    actual_old_reader_exact_deaths=old,actual_original_actor_metadata_deaths=deaths,actual_new_reader_pair=pair,
    scoped_current_compact_snapshots=compacts,current_port=ports,actual_copied_source=source_pin,
    health_limits='Actual ready snapshot only; ROOT independent current acceptance still required. Processalive alone is insufficient; stopped native workerfalse is expected.',
    no_second_launch_retry_retarget_newreader_kill_or_training_action=True,
    no_DBbody_journal_checkpoint_plan_audio_or_closed_original_artifacts_reread=True,
    completion_entry=who(psutil.Process()),completion_launcher=who(psutil.Process().parent()),initial=initial,final_sample=sample(),elapsed=time.monotonic()-t0))
print(json.dumps(dict(completion=cp,summary=summary,original_actor_metadata_deaths=deaths,
    completion_entry=who(psutil.Process()),completion_launcher=who(psutil.Process().parent()),initial=initial),indent=2))

