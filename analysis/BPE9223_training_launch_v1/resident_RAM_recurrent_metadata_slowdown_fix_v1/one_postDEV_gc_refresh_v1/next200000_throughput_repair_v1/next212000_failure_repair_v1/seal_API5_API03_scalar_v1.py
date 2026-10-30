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
initial=write_json(D/'API5_API03_completion_initial_v1.json',dict(kind='SOURCE_preparation_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=s,required=dict(commit=7989758336,physical=5239758336,private_max=536870912,E_free_min=100000000000),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),candidate_import_execution=False))
assert s['commit']>=7989758336 and s['physical']>=5239758336 and s['private']<=536870912 and s['E_free']>=100000000000

ids=[
dict(pid=37080,ctime=1792863544.554876,ppid=28328),
dict(pid=28328,ctime=1792863544.5475805,ppid=31460),
dict(pid=26448,ctime=1792863544.730811,ppid=37080),
dict(pid=31232,ctime=1792863544.7360077,ppid=26448),
dict(pid=21924,ctime=1792863544.7465792,ppid=26448),
dict(pid=37036,ctime=1792863554.0284345,ppid=21924),
dict(pid=42804,ctime=1792863554.0344887,ppid=37036),
dict(pid=33740,ctime=1792863554.0457768,ppid=37036)]
deaths=[]
for identity in ids:
    try:q=psutil.Process(identity['pid']);same=q.create_time()==identity['ctime']and q.ppid()==identity['ppid']
    except psutil.NoSuchProcess:same=False
    deaths.append(dict(identity=identity,exact_dead=not same))
assert all(x['exact_dead']for x in deaths)
outputs=[]
for root in [D/'a7_e03',D/'a7_f03']:
    count=0
    for parent,dirs,files in os.walk(root):
        assert len(dirs)<=32
        for name in files:
            p=pathlib.Path(parent)/name;count+=1;assert count<=160
            st=p.stat()
            if(p.suffix=='.json'and st.st_size<=1048576)or(p.suffix=='.log'and st.st_size<=16777216):outputs.append(pin(p))
            else:outputs.append(dict(path=str(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns,STAT_ONLY=True))
outputs.append(pin(D/'API5_API03_direct_OS_result_v1.json'))
result=dict(kind='API5_API03_partial_failure_scalar_completion_v1',
initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),
actual_TOOL=dict(exit_code=1,session_id=24888,chunk='404b2b'),numeric_direct_Python_OS_exit=1,
fixture_Popen_code=1,guard_error=None,cleanup=[],
actual_external_initial=dict(commit=13074141184,physical=16722292736,private=18329600,E_free=657747607552),
wall_seconds=86.90599999995902,peak_sampled_family_private=444772352,
passed_full_two_shape_proof=False,
first_shape=dict(name='metadata_live_order',original_fixture_history_rows=500000,cold10_requests_all_success_below20=True,scalar_request_elapsed_seconds_range=[1.077999999979511,1.327999999979511],append64_seconds=.015999999945051968,version_seconds=0.,quiescent_original_equivalence=True,real_SQLite_timeout_late_commit_idempotence=True,exact_STATE_and_response_loss_negative_fences=True,all192_unique=True,warm_new_rows_decoded=1,reader_main_and_API_and_mirror_methods_closed=True,owned_threads_remaining=[],reader_Popen_code=0,port_released=True),
second_shape=dict(name='adverse_reverse_interleaved',primary_type='AssertionError',source_line=399,failed_expression='db.resolve()==(dbdir/(project+".db")).resolve()',returned_DB_path='UNRECORDED',cause='UNKNOWN',second_history_or_reader_work_not_reached=True),
known_job_exact_deaths=deaths,all_recorded_job_identities_exact_dead=True,
unrecorded_sample_race=dict(pid=36716,ctime='UNRECORDED',phase='owned_family_sample',role='UNKNOWN',exact_death='UNKNOWN'),
outer_shell=dict(pid=31460,ctime='UNRECORDED',exact_death='UNKNOWN'),
new_output_pins=outputs,fixture_DB_files_STAT_only=True,no_originaldata_audio_tensor_or_closedstudy_read=True,
retry=False,production_or_live_reader_control=False,production212_cause='UNKNOWN',durable_speed_fix_claim=False,final=sample(),wall=time.monotonic()-t0)
rp=write_json(D/'API5_API03_partial_completion_v1.json',result)
print(json.dumps(dict(result=rp,initial=initial,entry=result['entry'],launcher=result['launcher'],known_exact_dead=len(deaths),output_items=len(outputs))))

