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
initial=write_json(D/'reader7_actor01_completion_initial_BEFORE_predicate_v1.json',dict(kind='SOURCE_preparation_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=s,required=dict(commit=6028384896,physical=3278384896,private_max=536870912,E_free_min=100000000000),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),candidate_import_execution=False))
assert s['commit']>=6028384896 and s['physical']>=3278384896 and s['private']<=536870912 and s['E_free']>=100000000000

actual_tool="{\"chunk_id\":\"c9d4a6\",\"wall_time_seconds\":0.3928979,\"exit_code\":1,\"original_token_count\":215,\"output\":\"Traceback (most recent call last):\\r\\n  File \\\"E:\\\\gpu_training_scratchpad\\\\analysis\\\\BPE9223_training_launch_v1\\\\resident_RAM_recurrent_metadata_slowdown_fix_v1\\\\one_postDEV_gc_refresh_v1\\\\next200000_throughput_repair_v1\\\\next212000_failure_repair_v1\\\\replace_failed_reader6_with_tested_reader7_SOURCE_v1.py\\\", line 158, in <module>\\r\\n    run(Path(p.parse_args().ROOT_review).resolve())\\r\\n  File \\\"E:\\\\gpu_training_scratchpad\\\\analysis\\\\BPE9223_training_launch_v1\\\\resident_RAM_recurrent_metadata_slowdown_fix_v1\\\\one_postDEV_gc_refresh_v1\\\\next200000_throughput_repair_v1\\\\next212000_failure_repair_v1\\\\replace_failed_reader6_with_tested_reader7_SOURCE_v1.py\\\", line 97, in run\\r\\n    assert pin(authority[\\\"path\\\"])==authority and read(authority[\\\"path\\\"])[\\\"accepted\\\"]is True\\r\\n                                                 ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^\\r\\nKeyError: 'accepted'\\r\\n\"}"
actual_tool=json.loads(actual_tool)

tool=write_json(D/'reader7_actor01_actual_owned_TOOL_result_v1.json',actual_tool)
root=D/'reader7_failed6_replacement_actual01_v1'
artifacts=[];values={}
for p in sorted(root.iterdir()):
    if not p.is_file():continue
    assert p.suffix in('.json','.log')
    b=p.read_bytes();assert len(b)<=65536
    q=dict(pin=pin_bytes(p,b))
    if p.suffix=='.json':q['value']=json.loads(b);values[p.name]=q['value']
    else:q['text_utf8']=b.decode('utf-8',errors='replace')
    artifacts.append(q)
result=values['result.json'];first=values['initial_BEFORE_predicate.json']
numericpath=D/'reader7_failed6_replacement_actual01_direct_numeric_exit_v1.json';b=numericpath.read_bytes();numeric=dict(pin=pin_bytes(numericpath,b),value=json.loads(b))
job_deaths=[]
for w in (first['entry'],first['launcher']):
    try:
        p=psutil.Process(w['pid']);same=abs(p.create_time()-w['ctime'])<.001
        q=dict(identity=w,exact_dead=not same,current=who(p)if same else None)
    except psutil.NoSuchProcess:q=dict(identity=w,exact_dead=True)
    except BaseException as e:q=dict(identity=w,exact_dead='UNKNOWN',error=type(e).__name__+':'+str(e)[:500])
    job_deaths.append(q)
current_compacts=[]
area=pathlib.Path('E:/gpu_training_scratchpad/analysis/BPE9223_Trackio_live_v1')
for name in ('owner.json','current_status.json','mirror_state.json'):
    p=area/name
    b=p.read_bytes();assert len(b)<=65536
    st=p.stat();current_compacts.append(dict(pin=pin_bytes(p,b),mtime_ns=st.st_mtime_ns,value=json.loads(b)))
expected=[dict(pid=33388,ctime=1792855546.8493547,ppid=38980),dict(pid=38980,ctime=1792855546.824607,ppid=26648)]
pair=[]
for w in expected:
    try:
        p=psutil.Process(w['pid']);observed=who(p);same=abs(observed['ctime']-w['ctime'])<.001 and observed['ppid']==w['ppid']
        pair.append(dict(expected=w,observed=observed,exact_alive=same,cmdline=p.cmdline()if same else 'NOT_READ_FOR_MISMATCHED_PID'))
    except psutil.NoSuchProcess:pair.append(dict(expected=w,exact_alive=False))
    except BaseException as e:pair.append(dict(expected=w,exact_alive='UNKNOWN',error=type(e).__name__+':'+str(e)[:500]))
try:
    ports=[dict(pid=c.pid,address=list(c.laddr))for c in psutil.net_connections(kind='tcp')if c.status==psutil.CONN_LISTEN and c.laddr.port==7860]
except BaseException as e:ports=dict(UNKNOWN=type(e).__name__+':'+str(e)[:500])
db=area/'database/parakeet-turkish.db';ds=db.stat()
destination=P/'work/research/serve_BPE9223_Trackio_live_v7.py'
summary=dict(actor_direct_numeric_OS_exit=numeric['value']['direct_Python_numeric_OS_exit'],actor_TOOL_exit=actual_tool['exit_code'],
    passed=result['passed'],primary=result['primary'],old_target_actions=result['old_targets'],source_copied=result['source_copied'],
    new_launch_attempted=result['new_launch_attempted'],new_launch_identity=result['new_launch_identity'],
    new_destination_exists_STAT=destination.exists(),old_pair_exactalive=all(q['exact_alive']is True for q in pair),
    control_before_failure='Targeted terminate/sourcecopy/Popen NOT_REACHED; actual failure at pinned ROOT API09 missing accepted accessor',
    API09_top_level_keys='UNRETAINED; no closed receipt reread',no_retry_retarget_or_control=True)
cp=write_json(D/'reader7_actor01_authority_schema_failure_actual_completion_v1.json',
    dict(kind='owned_reader7_actor01_authority_schema_failure_completion_v1',summary=summary,actual_TOOL=tool,direct_numeric=numeric,
    NEW_scalar_artifacts=artifacts,new_output_pins=[q['pin']for q in artifacts],exact_original_actor_metadata_deaths=job_deaths,
    scoped_current_compacts=current_compacts,current_exact_failed_reader_pair=pair,current_7860_listeners=ports,
    current_database_STAT=dict(path=str(db),bytes=ds.st_size,mtime_ns=ds.st_mtime_ns,body_read=False),
    no_API_HTTP_DB_body_journal_plan_checkpoint_or_old_closed_artifact_rereads=True,
    health_qualification='Failed reader6 remains in place. Process/port/errorNone alone is not healthy mirror evidence; prior mirror failure is unresolved.',
    completion_entry=who(psutil.Process()),completion_launcher=who(psutil.Process().parent()),initial=initial,final_sample=sample(),elapsed=time.monotonic()-t0))
print(json.dumps(dict(completion=cp,summary=summary,initial=initial,original_actor_metadata_deaths=job_deaths,
    current_exact_failed_reader_pair=pair,current_7860_listeners=ports,completion_entry=who(psutil.Process()),completion_launcher=who(psutil.Process().parent())),indent=2))

