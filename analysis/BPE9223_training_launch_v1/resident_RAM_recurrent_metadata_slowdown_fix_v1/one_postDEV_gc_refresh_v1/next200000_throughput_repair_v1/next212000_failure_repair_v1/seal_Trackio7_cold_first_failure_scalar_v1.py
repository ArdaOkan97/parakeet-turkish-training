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
initial=write_json(D/'actual_Trackio7_cold_first_completion_initial_v1.json',dict(kind='SOURCE_preparation_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=s,required=dict(commit=5399239296,physical=2649239296,private_max=536870912,E_free_min=100000000000),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),candidate_import_execution=False))
assert s['commit']>=5399239296 and s['physical']>=2649239296 and s['private']<=536870912 and s['E_free']>=100000000000

external=D/'actual_Trackio7_real_API_cold_first_external_01'
fixture=D/'actual_Trackio7_real_API_cold_first_fixture_01'
e=json.loads((external/'external_result.json').read_text(encoding='utf-8-sig'))
identities=[e['entry'],e['launcher']]+[x['identity']for x in e['known_deaths']]
deaths=[]
for identity in identities:
    try:
        q=psutil.Process(identity['pid'])
        same=q.create_time()==identity['ctime']and q.ppid()==identity['ppid']
    except psutil.NoSuchProcess:same=False
    deaths.append(dict(identity=identity,exact_dead=not same))
assert all(x['exact_dead']for x in deaths)
new_output_paths=[external/'external_result.json',external/'initial_BEFORE_predicate.json',external/'fixture_stderr.log',external/'fixture_stdout.log',fixture/'result.json',fixture/'initial_BEFORE_predicate.json',D/'actual_Trackio7_cold_first_TOOL_direct_result_v1.json']
pins=[]
for p in new_output_paths:
    pins.append(pin(p)if p.is_file()else dict(path=str(p),exists=False))
result=dict(kind='Trackio7_cold_first_actual01_failure_scalar_completion_v1',accepted_as_success=False,
entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),initial=initial,
actual_TOOL=dict(exit_code=1,session_id=74049,chunk='7adfda'),
numeric_direct_OS_exit=1,actual_fixture_Popen_code=e['actual_Popen_code'],guard_error=e['guard_error'],cleanup=e['cleanup'],
initial_external_sample=e['initial'],external_wall_seconds=e['wall_seconds'],peak_sampled_family_private=e['peak_sampled_family_private'],
primary=dict(type='NotADirectoryError',message='[WinError 267] The directory name is invalid',source_line=392,operation='fixture child subprocess.Popen with cwd=case'),
completed_shape_cases=[],cold_burst_executed=False,new_reader_child_launch_completed=False,
known_job_exact_deaths=deaths,all_recorded_job_identities_exact_dead=all(x['exact_dead']for x in deaths),
historical_outer_shell=dict(pid=21720,ctime='UNRECORDED',exact_death='UNKNOWN'),
new_output_pins=pins,production_or_reader_control=False,retry=False,
source_amendment_authority='NONE for API source correction; awaiting ROOT review',
final_sample=sample(),wall=time.monotonic()-t0)
p=D/'actual_Trackio7_cold_first_failure_completion_v1.json'
out=write_json(p,result)
print(json.dumps(dict(result=out,initial=initial,entry=result['entry'],launcher=result['launcher'],deaths=deaths)))

