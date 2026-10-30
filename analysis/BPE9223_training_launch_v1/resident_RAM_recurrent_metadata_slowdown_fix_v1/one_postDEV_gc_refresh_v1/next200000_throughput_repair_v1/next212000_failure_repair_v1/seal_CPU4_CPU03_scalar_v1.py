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
initial=write_json(D/'CPU4_CPU03_completion_initial_v1.json',dict(kind='SOURCE_preparation_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=s,required=dict(commit=8974029696,physical=6224029696,private_max=536870912,E_free_min=100000000000),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),candidate_import_execution=False))
assert s['commit']>=8974029696 and s['physical']>=6224029696 and s['private']<=536870912 and s['E_free']>=100000000000

ids=[
dict(pid=41460,ctime=1792863413.849085,ppid=19548),
dict(pid=19548,ctime=1792863413.8391404,ppid=5656),
dict(pid=33840,ctime=1792863414.009086,ppid=41460),
dict(pid=36500,ctime=1792863414.0147195,ppid=33840),
dict(pid=29176,ctime=1792863414.025438,ppid=33840),
dict(pid=42320,ctime=1792863414.3427377,ppid=29176),
dict(pid=40464,ctime=1792863414.3500943,ppid=42320),
dict(pid=10860,ctime=1792863414.3609676,ppid=42320)]
deaths=[]
for identity in ids:
    try:q=psutil.Process(identity['pid']);same=q.create_time()==identity['ctime']and q.ppid()==identity['ppid']
    except psutil.NoSuchProcess:same=False
    deaths.append(dict(identity=identity,exact_dead=not same))
assert all(x['exact_dead']for x in deaths)
paths=[D/'c_e03/external_result.json',D/'c_e03/initial_BEFORE_full_predicate.json',D/'c_e03/actual_typed_parent_worker_interpreters.json',D/'c_e03/actual_Popen_identity.json',D/'c_e03/fixture_stdout.log',D/'c_e03/fixture_stderr.log',D/'c_f03/result.json',D/'c_f03/first_packet_result.json',D/'c_f03/parent_first_fail_scalars.json',D/'c_f03/actual_parent_worker_interpreter_bindings.json',D/'c_f03/actual_parent_Popen_identity.json',D/'c_f03/initial_BEFORE_predicate.json',D/'CPU4_CPU03_direct_OS_result_v1.json']
outputs=[pin(p)if p.is_file()else dict(path=str(p),exists=False)for p in paths]
for folder in [D/'c_f03/worker_receipts',D/'c_f03/original_CPU14']:
    if folder.is_dir():
        entries=list(folder.iterdir());assert len(entries)<=128
        for p in entries:
            if not p.is_file():continue
            st=p.stat()
            if(p.suffix=='.json'and st.st_size<=1048576)or(p.suffix=='.log'and st.st_size<=16777216):outputs.append(pin(p))
            else:outputs.append(dict(path=str(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns,STAT_ONLY=True))
result=dict(kind='CPU4_CPU03_firstpacket_scalar_completion_v1',initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),
actual_TOOL=dict(exit_code=0,session_id=32005,chunk='06ad77'),numeric_direct_Python_OS_exit=0,fixture_Popen_code=0,guard_error=None,cleanup=[],
initial_external=dict(commit=12989243392,physical=16681664512,private=13635584,E_free=657766481920),
wall_seconds=40.95400000002701,peak_sampled_family_private=536129536,
first_packet=dict(delivered_rows=16,TR_rows=14,replay_rows=2,all_features_masks_finite=True,TR_full_ID_equality=True,replay_original_cache_checks=True,all_payload_weakrefs_dead_before_release=True,original_release_count=1),
parent_methods=dict(original_abort_returned=True,sender_joined=True,receiver_joined=True,pipes_closed=True,log_closed=True,all_owned_kernel_handles_closed=True,remaining_owned_handles=0),
child_Popen_code=1,child_credit_method_close='UNKNOWN under deliberate finite abort',normal0_claimed=False,
parent_failure_receipt_interpretation='Deliberate abort followed by receiver EOF during teardown; no pre-firstpacket error in observed sequence, no production212 cause inference',
known_job_exact_deaths=deaths,all_recorded_job_identities_exact_dead=True,outer_shell=dict(pid=5656,ctime='UNRECORDED',exact_death='UNKNOWN'),
new_output_pins=outputs,original_plan_or_audio_or_tensor_reread_in_completion=False,
production_or_reader_action=False,production212_cause='UNKNOWN',durable_speed_fix_claim=False,final=sample(),wall=time.monotonic()-t0)
rp=write_json(D/'CPU4_CPU03_completion_v1.json',result)
print(json.dumps(dict(result=rp,initial=initial,entry=result['entry'],launcher=result['launcher'],known_exact_dead=len(deaths),output_items=len(outputs))))

