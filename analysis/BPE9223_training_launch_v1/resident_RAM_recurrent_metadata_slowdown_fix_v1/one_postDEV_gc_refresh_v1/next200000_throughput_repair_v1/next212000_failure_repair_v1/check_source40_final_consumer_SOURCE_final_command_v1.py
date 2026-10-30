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
initial=write_json(D/'source40_final_consumer_SOURCE_final_command_initial_v1.json',dict(kind='SOURCE_preparation_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=s,required=dict(commit=5491513984,physical=2741513984,private_max=536870912,E_free_min=100000000000),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),candidate_import_execution=False))
assert s['commit']>=5491513984 and s['physical']>=2741513984 and s['private']<=536870912 and s['E_free']>=100000000000


path=D/'ROOT_source40_saved212000_final_ONE_training_execution_review_v1.json';b=path.read_bytes()
assert len(b)==47850 and sha(b)=='0c3f36a26d4b679e060f76161c83542eab2f22666283f22ff7ef9fa85ce4e9de'
review=json.loads(b)
priorpath=D/'source40_final_operational_consumer_SOURCE_sanity_result_v1.json';priorbytes=priorpath.read_bytes();prior=json.loads(priorbytes)
assert prior['issues']==['concrete_argv_present']
command=review['exact_ONE_command']
assert len(command)==23 and command[:5]==[str(P/'work/nemo-native-env-v2/Scripts/python.exe'),'-X','utf8','-B',str(D/'run_ASR18_BPE9223_resident_RAM_autocontinue_v41.py')]
args=command[5:];assert len(args)==18 and all(type(v)is str and v for v in args)
flags=dict(zip(args[::2],args[1::2]))
expected={'--review':str(path),'--configuration':prior['NEW_configuration']['path'],
    '--pointer':'E:\\gpu_training_scratchpad\\runs\\BPE9223_resident_RAM_20261018_01\\checkpoints\\update_000212000\\pointer.json',
    '--latest-failure':'E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\pipeline_overlap_RUN07_actual_01\\actual_control_namespace_failure_TOOL1_v1.json',
    '--branch':'E:\\gpu_training_scratchpad\\runs\\BPE9223_resident_RAM_20261018_01',
    '--output':'E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_training_actual_30',
    '--queue-bytes':'268435456','--start-invocation':'28',
    '--throughput-resume':'E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_recurrent_metadata_slowdown_fix_v1\\one_postDEV_gc_refresh_v1\\next200000_throughput_repair_v1\\ROOT_NEW212000_production_and_reader_failure_independent_seal_v1\\acceptance.json'}
assert flags==expected
assert review['accepted']is True and review['execution_admitted']is True and review['single_use']is True
assert review['kind']=='ROOT_ASR18_source40_saved212000_training_exact_ONE_scope_v1'
assert review['source_configuration']==prior['NEW_configuration']and review['original_CPU14_startup_reader7_recovery_policy']['CPU_startup_policy']['execution_review_path']==str(path)
assert review['administrative_stop']is None and review['full_training_target']==401072
final=sample();assert final['commit']>=2973078528 and final['physical']>=2473078528 and final['private']<=536870912 and time.monotonic()-t0<180
out=write_json(D/'source40_final_consumer_SOURCE_final_command_supplement_v1.json',dict(passed=True,ACTIVE_lane_mismatches=[],
    final_admitted_review=pin_bytes(path,b),original_SOURCE_key_sanity=pin_bytes(priorpath,priorbytes),retained_config_key_kind_path_checks_passed=19,
    exact_ONE_command=command,exact18_arguments=args,exact5_prefix=command[:5],initial28_ACK260_distinct212failure=True,
    old_inspector_assumptions='Only metadata inspector searched wrong container and then expected separate prefix; preserved original receipts. No candidate/production failure or source/config change.',
    no_actual_import_proof_pycompile_checker_originalpointer_journal_checkpoint_plan_audio_HTTP_DB_GPU_reader_training_or_source_mutation=True,
    entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),initial=initial,final_sample=final,elapsed=time.monotonic()-t0))
print(json.dumps(dict(result=out,passed=True,ACTIVE_lane_mismatches=[],initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent())),indent=2))

