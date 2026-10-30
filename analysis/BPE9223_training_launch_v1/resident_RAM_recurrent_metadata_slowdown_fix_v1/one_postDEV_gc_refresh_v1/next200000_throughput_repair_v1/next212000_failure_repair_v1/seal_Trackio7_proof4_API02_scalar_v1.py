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
initial=write_json(D/'Trackio7_proof4_API02_completion_initial_v1.json',dict(kind='SOURCE_preparation_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=s,required=dict(commit=7989758336,physical=5239758336,private_max=536870912,E_free_min=100000000000),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),candidate_import_execution=False))
assert s['commit']>=7989758336 and s['physical']>=5239758336 and s['private']<=536870912 and s['E_free']>=100000000000

identities=[
dict(pid=29024,ctime=1792862640.6495523,ppid=21608),
dict(pid=21608,ctime=1792862640.641772,ppid=40520),
dict(pid=39532,ctime=1792862640.84224,ppid=29024),
dict(pid=34508,ctime=1792862640.8484411,ppid=39532),
dict(pid=23960,ctime=1792862640.8626585,ppid=39532),
dict(pid=37964,ctime=1792862650.058597,ppid=23960),
dict(pid=25048,ctime=1792862650.064348,ppid=37964),
dict(pid=12232,ctime=1792862650.0762973,ppid=37964)]
deaths=[]
for identity in identities:
    try:q=psutil.Process(identity['pid']);same=q.create_time()==identity['ctime']and q.ppid()==identity['ppid']
    except psutil.NoSuchProcess:same=False
    deaths.append(dict(identity=identity,exact_dead=not same))
assert all(x['exact_dead']for x in deaths)
outputs=[]
for root in [D/'a7_e02',D/'a7_f02']:
    count=0
    for parent,dirs,files in os.walk(root):
        assert len(dirs)<=32
        for name in files:
            p=pathlib.Path(parent)/name;count+=1;assert count<=128
            st=p.stat()
            if (p.suffix=='.json'and st.st_size<=1048576)or(p.suffix=='.log'and st.st_size<=16777216):
                outputs.append(pin(p))
            else:outputs.append(dict(path=str(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns,STAT_ONLY=True))
outputs.append(pin(D/'Trackio7_proof4_API02_direct_OS_result_v1.json'))
prior=[]
for identity in [dict(pid=33772,ctime=1792862613.3265252,ppid=30504),dict(pid=30504,ctime=1792862613.31643,ppid=22680)]:
    try:q=psutil.Process(identity['pid']);same=q.create_time()==identity['ctime']and q.ppid()==identity['ppid']
    except psutil.NoSuchProcess:same=False
    prior.append(dict(identity=identity,exact_dead=not same))
assert all(x['exact_dead']for x in prior)
result=dict(kind='Trackio7_proof4_API02_failure_scalar_completion_v1',
initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),
actual_TOOL=dict(exit_code=1,session_id=7888,chunk='fc7ab3'),numeric_direct_Python_OS_exit=1,
actual_fixture_Popen_code=1,guard_error=None,cleanup=[],
external_wall_seconds=56.375,peak_sampled_family_private=442105856,
actual_external_initial=dict(commit=13011030016,physical=16689188864,private=18182144,E_free=657961517056),
primary=dict(type='AssertionError',message='',source_line=500,expression='read(area/"mirror_state.json")==checkpoint_state'),
cause_of_changed_STATE='UNKNOWN; no source or fixture correction selected',
fixture_shape_cases_completed=[],fixture_reader_child_launch_observed=True,
body_earlier_liveness_snapshot_preserved_without_rewriting=True,
later_known_job_exact_deaths=deaths,all_recorded_job_identities_exact_dead=True,
CPU02_completion_pair_exact_deaths=prior,outer_shell=dict(pid=40520,ctime='UNRECORDED',exact_death='UNKNOWN'),
new_output_pins=outputs,new_fixture_DB_files='STAT ONLY / no SQLite contents read during completion',
retry=False,production_or_live_reader_control=False,no_durable_speed_or_production212_cause_claim=True,
final=sample(),wall=time.monotonic()-t0)
rp=write_json(D/'Trackio7_proof4_API02_failure_completion_v1.json',result)
print(json.dumps(dict(result=rp,initial=initial,entry=result['entry'],launcher=result['launcher'],known_exact_dead=len(deaths),new_output_items=len(outputs))))

