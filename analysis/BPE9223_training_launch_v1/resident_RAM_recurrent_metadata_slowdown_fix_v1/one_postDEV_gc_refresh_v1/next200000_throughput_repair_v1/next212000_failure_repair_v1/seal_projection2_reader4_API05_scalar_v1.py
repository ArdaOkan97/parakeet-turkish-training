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
initial=write_json(D/'projection2_reader4_API05_completion_initial_v1.json',dict(kind='SOURCE_preparation_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=s,required=dict(commit=8056867200,physical=5306867200,private_max=536870912,E_free_min=100000000000),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),candidate_import_execution=False))
assert s['commit']>=8056867200 and s['physical']>=5306867200 and s['private']<=536870912 and s['E_free']>=100000000000

ids=[
dict(pid=18164,ctime=1792865383.2202396,ppid=38404),
dict(pid=38404,ctime=1792865383.211644,ppid=24740),
dict(pid=22284,ctime=1792865383.401624,ppid=18164),
dict(pid=21300,ctime=1792865383.406959,ppid=22284),
dict(pid=37564,ctime=1792865383.4171593,ppid=22284),
dict(pid=21944,ctime=1792865392.4603357,ppid=37564),
dict(pid=35028,ctime=1792865392.4664023,ppid=21944),
dict(pid=16476,ctime=1792865392.4773896,ppid=21944)]
deaths=[]
for identity in ids:
    try:q=psutil.Process(identity['pid']);same=q.create_time()==identity['ctime']and q.ppid()==identity['ppid']
    except psutil.NoSuchProcess:same=False
    deaths.append(dict(identity=identity,exact_dead=not same))
assert all(x['exact_dead']for x in deaths)
outputs=[]
for root in [D/'a7_e05',D/'a7_f05']:
    count=0
    for parent,dirs,files in os.walk(root):
        assert len(dirs)<=32
        for name in files:
            p=pathlib.Path(parent)/name;count+=1;assert count<=160
            st=p.stat()
            if(p.suffix=='.json'and st.st_size<=1048576)or(p.suffix=='.log'and st.st_size<=16777216):outputs.append(pin(p))
            else:outputs.append(dict(path=str(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns,STAT_ONLY=True))
outputs.append(pin(D/'projection2_reader4_API05_direct_OS_result_v1.json'))
result=dict(kind='projection2_reader4_API05_partial_failure_scalar_completion_v1',
initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),
actual_TOOL=dict(exit_code=1,session_id=65473,chunk='49ce96'),numeric_direct_Python_OS_exit=1,actual_fixture_Popen_code=1,guard_error=None,cleanup=[],
actual_external_initial=dict(commit=12793372672,physical=16374943744,private=18956288,E_free=657334992896),
external_wall_seconds=88.09299999999348,peak_sampled_family_private=446042112,
new_candidate_metadata_shape_main_integration=dict(shape='metadata_live_order',history_rows=500000,cold10_all_success_below20=True,scalar_elapsed_range_seconds=[1.0940000000409782,1.437999999965541],append64_elapsed_seconds=.01600000006146729,version_elapsed_seconds=0.,quiescent_original_equivalence=True,actual_SQLite_timeout_late_commit_idempotence=True,exact_STATE_and_response_loss_fences=True,all192_unique=True,warm_source_rows_decoded=0,reader_Popen_code=0,reader_main_mirror_API_methods_closed=True,owned_threads_remaining=[],port_released=True),
full_requested_proof_passed=False,all_lifetime_Gradio_checks_passed=False,matching_first_shape_ROOT_NOT_yet_authorized=True,adverse_shape_NOT_executed=True,
primary=dict(type='AttributeError',message="'State' object has no attribute 'gradio_call_events_lock'",proof_source_line=318,proof_call_line=310,official_function='_store_gradio_call_result',official_line=280),
first_unstored_lifetime_result='Function failed before returning its aggregate result; no fabricated fullsuite outcome',
known_job_exact_deaths=deaths,all_recorded_job_identities_exact_dead=True,outer_shell=dict(pid=24740,ctime='UNRECORDED',exact_death='UNKNOWN'),
new_output_pins=outputs,fixture_DB_STAT_only=True,
production_or_live_reader_control=False,retry=False,no_timeout_or_resource_change=True,production212_cause='UNKNOWN',durable_speed_fix_claim=False,final=sample(),wall=time.monotonic()-t0)
rp=write_json(D/'projection2_reader4_API05_partial_completion_v1.json',result)
print(json.dumps(dict(result=rp,initial=initial,entry=result['entry'],launcher=result['launcher'],known_exact_dead=len(deaths),output_items=len(outputs))))

