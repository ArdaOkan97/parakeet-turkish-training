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
initial=write_json(D/'API6_API04_completion_initial_v1.json',dict(kind='SOURCE_preparation_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=s,required=dict(commit=7989758336,physical=5239758336,private_max=536870912,E_free_min=100000000000),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),candidate_import_execution=False))
assert s['commit']>=7989758336 and s['physical']>=5239758336 and s['private']<=536870912 and s['E_free']>=100000000000

ids=[
dict(pid=37136,ctime=1792864162.04898,ppid=24056),
dict(pid=24056,ctime=1792864162.0412192,ppid=26248),
dict(pid=8304,ctime=1792864162.2839947,ppid=37136),
dict(pid=10048,ctime=1792864162.2893023,ppid=8304),
dict(pid=38760,ctime=1792864162.3001237,ppid=8304),
dict(pid=20800,ctime=1792864173.102553,ppid=38760),
dict(pid=15016,ctime=1792864173.2023928,ppid=20800),
dict(pid=39776,ctime=1792864173.3054583,ppid=20800)]
deaths=[]
for identity in ids:
    try:q=psutil.Process(identity['pid']);same=q.create_time()==identity['ctime']and q.ppid()==identity['ppid']
    except psutil.NoSuchProcess:same=False
    deaths.append(dict(identity=identity,exact_dead=not same))
assert all(x['exact_dead']for x in deaths)
outputs=[]
for root in [D/'a7_e04',D/'a7_f04']:
    count=0
    for parent,dirs,files in os.walk(root):
        assert len(dirs)<=32
        for name in files:
            p=pathlib.Path(parent)/name;count+=1;assert count<=160
            st=p.stat()
            if(p.suffix=='.json'and st.st_size<=1048576)or(p.suffix=='.log'and st.st_size<=16777216):outputs.append(pin(p))
            else:outputs.append(dict(path=str(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns,STAT_ONLY=True))
outputs.append(pin(D/'API6_API04_direct_OS_result_v1.json'))
result=dict(kind='API6_API04_remaining_adverse_failure_scalar_completion_v1',
initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),
actual_TOOL=dict(exit_code=1,session_id=60255,chunk='0b5d0f'),numeric_direct_Python_OS_exit=1,
actual_fixture_Popen_code=1,guard_error=None,cleanup=[],
actual_external_initial=dict(commit=13048324096,physical=16667635712,private=19148800,E_free=657551523840),
external_wall_seconds=32.71799999999348,peak_sampled_family_private=225439744,
only_shape_attempted='adverse_reverse_interleaved',original_history_rows=500000,strict_DB_isolation_passed=True,actual_expected_DB_path_receipt=str(D/'a7_f04/adverse_reverse_interleaved/actual_fixture_DB_path_binding.json'),
primary=dict(type='TimeoutError',message='',source_line=489,expression='concurrent_results=[future.result(timeout=20)for future in scalar_futures]'),
request_failure_details='Preserved in fresh cold_burst_requests.json; no cause/individual HTTP outcome inferred solely from Future timeout',
completed_shape_cases=[],all_lifetime_negatives_reached=False,full_composed_integration_pass=False,completed_original_first_shape_rerun=False,
reader_child_launch_observed=True,reader_method_close='UNKNOWN on failed cold burst; no inference from death',
body_earlier_liveness_snapshot_preserved=True,
known_job_exact_deaths=deaths,all_recorded_job_identities_exact_dead=True,
sampling_races='All persisted race identities are known exact identities; no new classification inferred',
outer_shell=dict(pid=26248,ctime='UNRECORDED',exact_death='UNKNOWN'),
new_output_pins=outputs,fixture_DB_STAT_only=True,
production_or_live_reader_control=False,retry=False,no_timeout_or_resource_change=True,production212_cause='UNKNOWN',durable_speed_fix_claim=False,final=sample(),wall=time.monotonic()-t0)
rp=write_json(D/'API6_API04_failure_completion_v1.json',result)
print(json.dumps(dict(result=rp,initial=initial,entry=result['entry'],launcher=result['launcher'],known_exact_dead=len(deaths),output_items=len(outputs))))

