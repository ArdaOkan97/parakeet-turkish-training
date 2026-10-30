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
initial=write_json(D/'CPU14_proof3_CPU02_completion_initial_v1.json',dict(kind='SOURCE_preparation_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=s,required=dict(commit=8974029696,physical=6224029696,private_max=536870912,E_free_min=100000000000),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),candidate_import_execution=False))
assert s['commit']>=8974029696 and s['physical']>=6224029696 and s['private']<=536870912 and s['E_free']>=100000000000

identities=[
dict(pid=8568,ctime=1792862556.444024,ppid=34120),
dict(pid=34120,ctime=1792862556.4346976,ppid=12340),
dict(pid=37048,ctime=1792862556.5942411,ppid=8568),
dict(pid=34636,ctime=1792862556.6018074,ppid=37048),
dict(pid=2336,ctime=1792862556.612824,ppid=37048),
dict(pid=33024,ctime=1792862556.8938951,ppid=2336),
dict(pid=2912,ctime=1792862556.90084,ppid=33024)]
deaths=[]
for identity in identities:
    try:q=psutil.Process(identity['pid']);same=q.create_time()==identity['ctime']and q.ppid()==identity['ppid']
    except psutil.NoSuchProcess:same=False
    deaths.append(dict(identity=identity,exact_dead=not same))
assert all(x['exact_dead']for x in deaths)
paths=[D/'c_e02/external_result.json',D/'c_e02/initial_BEFORE_full_predicate.json',D/'c_e02/fixture_stderr.log',D/'c_e02/fixture_stdout.log',D/'c_f02/result.json',D/'c_f02/initial_BEFORE_predicate.json',D/'c_f02/parent_first_fail_scalars.json',D/'c_f02/actual_parent_Popen_identity.json',D/'c_f02/original_CPU14/CPU_stderr.log',D/'CPU14_proof3_CPU02_direct_OS_result_v1.json']
pins=[pin(p)if p.is_file()else dict(path=str(p),exists=False)for p in paths]
receipts=D/'c_f02/worker_receipts'
worker_receipts=[pin(p)for p in receipts.iterdir()if p.is_file()]if receipts.is_dir()else []
assert len(worker_receipts)<=128
result=dict(kind='CPU14_proof3_CPU02_failure_scalar_completion_v1',initial=initial,
entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),
actual_TOOL=dict(exit_code=1,chunk='f0f6d8',session_id=None),numeric_direct_Python_OS_exit=1,
fixture_Popen_code=1,guard_error=None,cleanup=[],
external_wall_seconds=.7179999999934807,peak_sampled_family_private=39100416,
actual_external_initial=dict(commit=13027762176,physical=16695549952,private=13422592,E_free=657970954240),
first_original_parent_failure=dict(type='ModuleNotFoundError',message="No module named 'numpy'",source_function='_receive',source_line=463,deliberate=False,original_fail_returned=True),
propagated_submit_error=dict(type='ValueError',message='Distinct ordered plan on healthy producer'),
deliberate_original_abort_record_separate=True,parent_receipt_errors=0,
no_ready_or_first_packet_observed=True,worker_receipts=worker_receipts,
known_job_exact_deaths=deaths,all_recorded_job_identities_exact_dead=True,
outer_shell=dict(pid=12340,ctime='UNRECORDED',exact_death='UNKNOWN'),
unrecorded_early_native_body_identity='UNKNOWN; no earliest-worker receipt and no extra death inference',
new_output_pins=pins,production212_cause='UNKNOWN',production_or_reader_action=False,retry=False,final=sample(),wall=time.monotonic()-t0)
p=D/'CPU14_proof3_CPU02_failure_completion_v1.json'
rp=write_json(p,result)
print(json.dumps(dict(result=rp,initial=initial,entry=result['entry'],launcher=result['launcher'],known_exact_dead=len(deaths))))

