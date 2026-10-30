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
initial=write_json(D/'source40_typed01_actual_completion_initial_BEFORE_predicate_v1.json',dict(kind='SOURCE_preparation_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=s,required=dict(commit=6565255808,physical=3815255808,private_max=536870912,E_free_min=100000000000),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),candidate_import_execution=False))
assert s['commit']>=6565255808 and s['physical']>=3815255808 and s['private']<=536870912 and s['E_free']>=100000000000

actual_tool="{\"chunk_id\":\"ae67a8\",\"wall_time_seconds\":0.4244989,\"exit_code\":1,\"original_token_count\":248,\"output\":\"Traceback (most recent call last):\\r\\n  File \\\"E:\\\\gpu_training_scratchpad\\\\analysis\\\\BPE9223_training_launch_v1\\\\resident_RAM_recurrent_metadata_slowdown_fix_v1\\\\one_postDEV_gc_refresh_v1\\\\next200000_throughput_repair_v1\\\\next212000_failure_repair_v1\\\\run_source40_saved212000_typed_metadata_CPU_external_v1.py\\\", line 78, in <module>\\r\\n    p=argparse.ArgumentParser();p.add_argument(\\\"--ROOT-review\\\",required=True);main(Path(p.parse_args().ROOT_review).resolve())\\r\\n                                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\r\\n  File \\\"E:\\\\gpu_training_scratchpad\\\\analysis\\\\BPE9223_training_launch_v1\\\\resident_RAM_recurrent_metadata_slowdown_fix_v1\\\\one_postDEV_gc_refresh_v1\\\\next200000_throughput_repair_v1\\\\next212000_failure_repair_v1\\\\run_source40_saved212000_typed_metadata_CPU_external_v1.py\\\", line 37, in main\\r\\n    for q in s[\\\"input_pins\\\"]:assert pin(q[\\\"path\\\"])==q\\r\\n                                    ^^^^^^^^^^^^^^^^^\\r\\nAssertionError\\r\\n\"}"
actual_tool=json.loads(actual_tool)

tool_pin=write_json(D/'source40_typed01_actual_owned_TOOL_result_v1.json',actual_tool)
root=D/'source40_saved212000_typed_metadata_external_actual01_v1'
body=D/'source40_saved212000_typed_metadata_actual01_v1'
assert root.is_dir()
captures=[];initial_value=None
for p in sorted(root.iterdir()):
    assert p.is_file()and p.suffix in ('.json','.log')
    b=p.read_bytes();assert len(b)<=65536
    q=dict(pin=pin_bytes(p,b))
    if p.suffix=='.json':
        q['value']=json.loads(b)
        if p.name=='initial_BEFORE_full_predicate.json':initial_value=q['value']
    else:q['text_utf8']=b.decode('utf-8',errors='replace')
    captures.append(q)
assert initial_value is not None
numeric_path=D/'source40_saved212000_typed01_direct_numeric_exit_v1.json';b=numeric_path.read_bytes();numeric=dict(pin=pin_bytes(numeric_path,b),value=json.loads(b))
deaths=[]
for w in (initial_value['entry'],initial_value['launcher']):
    try:
        p=psutil.Process(w['pid']);same=abs(p.create_time()-w['ctime'])<.001
        value=dict(identity=w,exact_dead=not same,current=who(p)if same else None)
    except psutil.NoSuchProcess:value=dict(identity=w,exact_dead=True)
    except BaseException as e:value=dict(identity=w,exact_dead='UNKNOWN',error=type(e).__name__+':'+str(e)[:500])
    deaths.append(value)
cp=write_json(D/'source40_typed01_pre_fixture_pin_rejection_actual_completion_v1.json',
    dict(kind='owned_source40_typed01_exact_pin_failure_completion_v1',actual_TOOL=tool_pin,direct_numeric=numeric,NEW_scalar_artifacts=captures,new_output_pins=[a['pin']for a in captures],
         source_bound_failure=dict(type='AssertionError',source='run_source40_saved212000_typed_metadata_CPU_external_v1.py',line=37,expression='for q in s["input_pins"]:assert pin(q["path"])==q',
             failing_input_index='UNRECORDED_UNKNOWN',particular_false_operand='UNKNOWN pending independent SOURCE/pin review',
             fixture_Popen='NOT_REACHED based actual trace/source order',candidate_fixture_import_execution='NOT_REACHED',typed_case_outcomes='NOT_REACHED'),
         actual_absence_STAT=dict(body_output_directory_exists=body.exists(),external_result_exists=(root/'result.json').exists(),stdout_exists=(root/'stdout.log').exists(),stderr_exists=(root/'stderr.log').exists()),
         original_guard_outcomes=dict(actual_initial_BEFORE_predicate=initial_value,resource_assertions_passed_before_pin_loop_SOURCE_bound=True,
             runtime_family_sampling='NOT_REACHED',guard_error_field='UNRECORDED: no external result created'),
         exact_known_original_metadata_identity_deaths=deaths,completion_entry=who(psutil.Process()),completion_launcher=who(psutil.Process().parent()),
         initial=initial,no_retry_rebind_or_source_edit=True,no_real_pointer_checkpoint_plan_journal_audio_reader_training_or_GPU_action=True,
         final_sample=sample(),elapsed=time.monotonic()-t0))
print(json.dumps(dict(completion=cp,original_initial=next(a['pin']for a in captures if a['pin']['path'].endswith('initial_BEFORE_full_predicate.json')),
    initial_value=initial_value,original_known_deaths=deaths,tool=tool_pin,numeric=numeric,
    completion_entry=who(psutil.Process()),completion_launcher=who(psutil.Process().parent()),initial=initial),indent=2))

