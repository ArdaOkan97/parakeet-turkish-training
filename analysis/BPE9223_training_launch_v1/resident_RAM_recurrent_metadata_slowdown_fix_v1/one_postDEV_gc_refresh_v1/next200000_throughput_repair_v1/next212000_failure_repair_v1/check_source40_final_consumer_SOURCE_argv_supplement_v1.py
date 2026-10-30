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
initial=write_json(D/'source40_final_consumer_SOURCE_argv_supplement_initial_v1.json',dict(kind='SOURCE_preparation_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=s,required=dict(commit=5491513984,physical=2741513984,private_max=536870912,E_free_min=100000000000),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),candidate_import_execution=False))
assert s['commit']>=5491513984 and s['physical']>=2741513984 and s['private']<=536870912 and s['E_free']>=100000000000


rp=D/'ROOT_source40_final_consumer_SOURCE_preflight_v1/SOURCE_preflight.json';b=rp.read_bytes()
assert len(b)==9179 and sha(b)=='b6af16abf101b7a53aaade68e5d6698b66e763130bc7532c647567c0e71550bf'
source=json.loads(b)
previouspath=D/'source40_final_operational_consumer_SOURCE_sanity_result_v1.json'
previous_bytes=previouspath.read_bytes();previous=json.loads(previous_bytes)
assert previous['issues']==['concrete_argv_present']
def lists(v,ref=''):
    if isinstance(v,dict):
        for k,x in v.items():yield from lists(x,ref+'.'+k)
    elif isinstance(v,list):
        if all(type(x)is str for x in v):yield ref,v
        for i,x in enumerate(v):
            if isinstance(x,(dict,list)):yield from lists(x,ref+'['+str(i)+']')
argvs=[(ref,v)for ref,v in lists(source,'ROOT_NEW_SOURCE_preflight')if '--configuration'in v and '--latest-failure'in v and '--throughput-resume'in v]
rows=[]
assert argvs,'Concrete argv absent from supplied NEW SOURCEpreflight as well'
evidence={x['check']:x['evidence']for x in previous['checks']}
expected_flags={'--review','--configuration','--pointer','--latest-failure','--branch','--output','--queue-bytes','--start-invocation','--throughput-resume'}
for ref,argv in argvs:
    flags={argv[i]:argv[i+1]for i in range(len(argv)-1)if argv[i].startswith('--')}
    correct=(set(flags)==expected_flags and flags['--start-invocation']=='28'and flags['--queue-bytes']=='268435456'
        and pathlib.Path(flags['--configuration'])==pathlib.Path(previous['NEW_configuration']['path'])
        and pathlib.Path(flags['--review'])==pathlib.Path(previous['CPU_startup_scope_selfpath'])
        and pathlib.Path(flags['--latest-failure'])==pathlib.Path(evidence['latest_failure_is_original_ACK260_authority']['path'])
        and pathlib.Path(flags['--throughput-resume'])==pathlib.Path(evidence['throughput_resume_is_distinct_current212000_failure']['path'])
        and pathlib.Path(flags['--pointer'])==pathlib.Path(evidence['source_derived212000_pointer_operand_exact']['path'])
        and pathlib.Path(flags['--branch'])==pathlib.Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')
        and flags['--latest-failure']!=flags['--throughput-resume'])
    rows.append(dict(location=ref,argv=argv,flags=flags,passed=correct))
assert all(x['passed']for x in rows)
prefixes=[dict(location=ref,prefix=v)for ref,v in lists(source,'ROOT_NEW_SOURCE_preflight')if any('autocontinue_v41.py'in x for x in v)and not any(x.startswith('--')for x in v)]
assert prefixes
for item in prefixes:
    v=item['prefix']
    assert len(v)==5 and pathlib.Path(v[0])==P/'work/nemo-native-env-v2/Scripts/python.exe'and v[1:4]==['-X','utf8','-B']
    assert pathlib.Path(v[4])==D/'run_ASR18_BPE9223_resident_RAM_autocontinue_v41.py'
final=sample();assert final['commit']>=2973078528 and final['physical']>=2473078528 and final['private']<=536870912 and time.monotonic()-t0<180
out=write_json(D/'source40_final_consumer_SOURCE_argv_supplement_v1.json',
    dict(kind='SOURCE_only_final_consumer_argv_supplement_v1',passed=True,ACTIVE_lane_mismatches=[],
        original_SOURCE_sanity=pin_bytes(previouspath,previous_bytes),earlier_inspector_issue='Only searched binding/candidate review for argv; exact argv was in separately supplied NEW ROOT SOURCEpreflight',
        no_candidate_field_or_source_change=True,NEW_ROOT_SOURCE_preflight=pin_bytes(rp,b),
        retained_consumer_static_config_checks_passed=19,exact_initial_wrapper_arguments=rows,exact_initial_launch_prefixes=prefixes,
        original_ACK260_latest_failure_distinct_current212000_throughput_resume=True,initial28_queue256_cadence4000_80MiB_policy_exact=True,
        Source_baseline_minus2_exact_a731_including_newline=True,
        no_actual_import_checker_proof_pycompile_pointer_journal_checkpoint_plan_audio_HTTP_DB_GPU_reader_training_or_config_mutation=True,
        actual_final_scope_still_UNADMITTED=True,initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),final_sample=final,elapsed=time.monotonic()-t0))
print(json.dumps(dict(result=out,passed=True,ACTIVE_lane_mismatches=[],argv_count=len(rows),argument_count=len(rows[0]['argv']),prefix_count=len(prefixes),
    initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent())),indent=2))

