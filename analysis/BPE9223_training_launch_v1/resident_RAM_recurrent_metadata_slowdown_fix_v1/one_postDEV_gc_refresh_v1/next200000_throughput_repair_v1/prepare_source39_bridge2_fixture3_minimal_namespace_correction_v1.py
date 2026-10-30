import ast, copy, hashlib, json, os, pathlib, shutil, sys, time, traceback
import psutil
W=pathlib.Path(__file__).resolve().parent
S=pathlib.Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20/outputs/parakeet_turkish')
sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
SCOPE=W/'ROOT_source39_CPU_actual01_failure_minimal_fixture_correction_SOURCE_ONLY_scope_v1.json'
INITIAL=W/'source39_bridge2_fixture3_source_preparation_initial_v1.json'
RESULT=W/'source39_bridge2_fixture3_source_preparation_result_v1.json'
def require(value,message):
    if not value:raise ValueError(message)
def identity(p):return dict(pid=p.pid,create_time=p.create_time(),ppid=p.ppid())
def pin(path):
    p=pathlib.Path(path);raw=p.read_bytes()
    return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    return pin(path)
def alive(who):
    try:return psutil.Process(who['pid']).create_time()==who['create_time']
    except psutil.NoSuchProcess:return False
me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
entry=dict(identity=identity(me),launcher=identity(me.parent()),recorded_before_predicate=True)
began=time.monotonic()
scope=json.loads(SCOPE.read_text(encoding='utf-8'))
require(pin(SCOPE)['sha256']=='335f20dde95f95d834574ecdc9b898552b777548ec8ec8ed549c7d5a05bebc00','Exact source-only ROOT scope')
sample=sample_memory(me.pid,'source39_minimal_namespace_source_preparation')
initial=dict(entry=entry,sample=sample,E_free=shutil.disk_usage('E:/gpu_training_scratchpad').free,
    private_bytes=me.memory_info().private,launcher_private_bytes=me.parent().memory_info().private,
    TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),ROOT_review=pin(SCOPE))
write(INITIAL,initial)
result=dict(kind='ASR18_source39_bridge2_fixture3_SOURCE_ONLY_preparation_v1',entry=entry,initial=pin(INITIAL),
    actual_CPU_execution=False,candidate_import=False,scientific_or_production_change=False,outputs={},error=None)
try:
    memory=sample['system_memory']
    require(memory['commit_available_bytes']>=5130803840 and memory['physical_available_bytes']>=2380803840,'Unchanged full prospective guard')
    require(initial['private_bytes']+initial['launcher_private_bytes']<=536870912 and initial['E_free']>=100000000000,'Original private/E guard')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'Exact isolated CPU environment')
    require(os.environ.get('TORCH_EXTENSIONS_DIR')=='E:\\gpu_training_scratchpad\\cache\\torch_extensions','Exact dot-sourced Torch extension path')
    require(all(not alive(who) for who in scope['job_identities_to_independently_check']),'Closed four actual01 exact identities')
    old_handoff=W/'source39_actual204800_entire_transition_CPU_INACTIVE_handoff_v3.json'
    old_handoff_pin=pin(old_handoff)
    require(old_handoff_pin['sha256']=='30a9bf35eb1dd3d22323fffbf6af4c5b380334a612a681d21d1a881dbc2882b4','Exact prior frozen handoff')
    handoff=json.loads(old_handoff.read_text(encoding='utf-8'))
    for who in handoff['protected_runtime_identities']:
        require(alive(who)==who['expected_alive'],'Original training dead/readers alive')
    for key in ('input_bridge','input_fixture'):
        require(pin(scope[key]['path'])==scope[key],'Exact source input')
    for key,value in scope['original_unchanged'].items():require(pin(value['path'])==value,'All original unchanged source/configuration pins')
    output_paths={key:pathlib.Path(value)for key,value in scope['fresh_outputs'].items()}
    require(all(not path.exists() for path in output_paths.values()),'All successor files fresh')
    require(not (W/'source39_entire_transition_integration_CPU_actual_02').exists()and not (W/'source39_entire_transition_integration_CPU_external_actual_02').exists(),'Future actual02 namespaces absent')
    raw_checks=[]
    for target,input_key,out_key in (('bridge_v2','input_bridge','bridge'),('fixture_v3','input_fixture','fixture')):
        before=pathlib.Path(scope[input_key]['path']).read_bytes()
        steps=[edit for edit in scope['required_edits']if edit['target']==target]
        require(len(steps)==1,'Only one declared source edit each')
        after=before
        edits=[]
        for step in steps:
            old=step['old'].encode('utf-8');new=step['new'].encode('utf-8')
            require(after.count(old)==step['count']==1,'Exact one literal occurrence')
            after=after.replace(old,new,1);edits.append(dict(old=step['old'],new=step['new'],count=1))
        inverse=after
        for edit in reversed(edits):
            old=edit['old'].encode('utf-8');new=edit['new'].encode('utf-8')
            require(inverse.count(new)==1,'Exact inverse literal occurrence');inverse=inverse.replace(new,old,1)
        require(inverse==before,'Full original whole-byte inverse')
        old_ast=ast.dump(ast.parse(before.decode('utf-8')),include_attributes=False)
        new_ast=ast.dump(ast.parse(after.decode('utf-8')),include_attributes=False)
        with output_paths[out_key].open('xb')as f:f.write(after);f.flush();os.fsync(f.fileno())
        result['outputs'][out_key]=pin(output_paths[out_key])
        raw_checks.append(dict(base=scope[input_key],successor=pin(output_paths[out_key]),ordered_edits=edits,
            forward_verified=True,whole_original_byte_inverse=True,reconstructed_original_bytes=len(inverse),
            reconstructed_original_sha256=hashlib.sha256(inverse).hexdigest(),
            base_AST_sha256=hashlib.sha256(old_ast.encode()).hexdigest(),successor_AST_sha256=hashlib.sha256(new_ast.encode()).hexdigest()))
    raw=dict(kind='ASR18_source39_bridge1_to2_fixture2_to3_exact_two_literal_RAW_AST_v1',ROOT_review=pin(SCOPE),
        all_sources_parsed_without_import=True,checks=raw_checks,unchanged=scope['original_unchanged'])
    result['outputs']['RAW']=write(output_paths['RAW'],raw)
    original_handoff=copy.deepcopy(handoff)
    replacements={handoff['bridge_source']['path']:result['outputs']['bridge'],handoff['fixture_source']['path']:result['outputs']['fixture'],handoff['full_RAW']['path']:result['outputs']['RAW']}
    handoff['bridge_source']=result['outputs']['bridge'];handoff['fixture_source']=result['outputs']['fixture'];handoff['full_RAW']=result['outputs']['RAW']
    handoff['input_pins']=[copy.deepcopy(replacements.get(item['path'],item))for item in handoff['input_pins']]+[pin(SCOPE)]
    handoff['kind']='ASR18_source39_actual204800_entire_transition_CPU_INACTIVE_handoff_v4'
    handoff['fixture_output']=str(W/'source39_entire_transition_integration_CPU_actual_02')
    handoff['external_output']=str(W/'source39_entire_transition_integration_CPU_external_actual_02')
    handoff['proposed_arguments']=[handoff['external_source']['path'],'--ROOT-review',str(W/'ROOT_source39_actual204800_entire_transition_main_path_CPU_exact_ONCE_scope_v2.json')]
    handoff['initial']=pin(INITIAL);handoff['preparer']=entry['identity'];handoff['launcher']=entry['launcher']
    handoff['metadata_only_finalization']=dict(ROOT_SOURCE_ONLY_scope=pin(SCOPE),prior_handoff=old_handoff_pin,
        declared_only_source_edits=scope['required_edits'],actual01_failure_preserved=scope['actual_failure'],
        actual01_known4_exactdead_reused_ROOT=True,
        failed_sealer=dict(source=pin(W/'seal_source39_CPU_actual01_failed_metadata_v1.py'),actual_TOOL_exit=1,chunk_id='018248',
            error='ModuleNotFoundError: paired_windows_commit_telemetry_v1',PID_creation_time_and_launcher='UNKNOWN_UNRECORDED',
            no_retry=True),actual_CPU_execution=False,separate_ROOT_exact_CPU_v2_required=True)
    result['outputs']['handoff']=write(output_paths['handoff'],handoff)
    result['handoff_changed_top_fields']=[key for key in handoff if handoff.get(key)!=original_handoff.get(key)]
    final=sample_memory(me.pid,'source39_minimal_namespace_source_preparation_final')
    require(final['system_memory']['commit_available_bytes']>=2343932928 and final['system_memory']['physical_available_bytes']>=1843932928,'Unchanged runtime floors')
    require(me.memory_info().private+me.parent().memory_info().private<=536870912 and time.monotonic()-began<180,'Original private/wall bound')
    result['final_sample']=final;result['passed']=True
except BaseException as exc:
    result['passed']=False;result['error']=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
finally:
    result['wall_seconds']=time.monotonic()-began
    result_pin=write(RESULT,result)
    print(json.dumps(dict(result=result_pin,passed=result.get('passed'),outputs=result['outputs'],entry=entry),sort_keys=True))
require(result.get('passed'),'Preserve source preparation failure; no automatic retry')
