import copy, hashlib, json, os, pathlib, shutil, sys, time, traceback
import psutil
W=pathlib.Path(__file__).resolve().parent
S=pathlib.Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20/outputs/parakeet_turkish')
sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
SCOPE=W/'ROOT_source39_saved204800_final_actual_configuration_INACTIVE_preparation_scope_v1.json'
INITIAL=W/'source39_actual204800_configuration_preparation_initial_v1.json'
def require(value,message):
    if not value:raise ValueError(message)
def identity(p):return dict(pid=p.pid,create_time=p.create_time(),ppid=p.ppid())
def alive(who):
    try:return psutil.Process(who['pid']).create_time()==who['create_time']
    except psutil.NoSuchProcess:return False
def pin(path):
    p=pathlib.Path(path);raw=p.read_bytes()
    return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n')as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    return pin(path)
me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
entry=dict(identity=identity(me),launcher=identity(me.parent()),recorded_before_predicate=True)
began=time.monotonic();scope=json.loads(SCOPE.read_text(encoding='utf-8'))
require(pin(SCOPE)['sha256']=='db5a6b677b77bcc97ef6a5891cfb0362eabd5bc1f3f074af8afee4db0dd74e51','Exact configuration-only ROOT scope')
sample=sample_memory(me.pid,'source39_actual_final_configuration_preparation')
initial=dict(entry=entry,sample=sample,private_bytes=me.memory_info().private,
    E_free=shutil.disk_usage('E:/gpu_training_scratchpad').free,TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),ROOT_review=pin(SCOPE))
write(INITIAL,initial)
outputs={key:pathlib.Path(value)for key,value in scope['fresh_outputs'].items()}
require(not outputs['result'].exists(),'Fresh result namespace')
result=dict(kind='ASR18_source39_actual204800_final_configuration_SOURCE_ONLY_preparation_v1',entry=entry,
    initial=pin(INITIAL),outputs={},error=None,actual_candidate_execution=False,production_activation=False)
try:
    memory=sample['system_memory']
    require(memory['commit_available_bytes']>=5130803840 and memory['physical_available_bytes']>=2380803840,'Original full prospective guard')
    require(initial['private_bytes']<=536870912 and initial['E_free']>=100000000000,'Original self512MiB/E100GB guard')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'Original hiddenCPU/caps')
    require(initial['TORCH_EXTENSIONS_DIR']=='E:\\gpu_training_scratchpad\\cache\\torch_extensions','Exact dot-sourced path')
    require(all(not path.exists()for path in outputs.values()),'All final outputs fresh')
    for key in ('base_candidate','proper_source37_configuration','driver_source','wrapper_source','transition_helper','actual_pause','CPU_actual02_acceptance','initial_checkpoint'):
        require(pin(scope[key]['path'])==scope[key],'Exact frozen allowed source/config/receipt/pointer pin')
    handoff_prior=json.loads((W/'source39_actual204800_entire_transition_CPU_INACTIVE_handoff_v4.json').read_text(encoding='utf-8'))
    protected=handoff_prior['protected_runtime_identities']
    require(all(alive(who)==who['expected_alive']for who in protected),'All9 production exactdead/readers2 exactalive')
    B=pathlib.Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')
    future_namespace=dict(invocation=str(B/'invocations'/'owner_000027'),journal=str(B/'journals'/'commits_000027.jsonl'),output=scope['proposed_future_GPU_argv'][-5])
    # The exact reviewed argv is authoritative; take the output using its flag.
    argv=list(scope['proposed_future_GPU_argv']);future_namespace['output']=argv[argv.index('--output')+1]
    require(all(not pathlib.Path(path).exists()for path in future_namespace.values()),'Future I27/J27/A28 absent')
    raw=pathlib.Path(scope['base_candidate']['path']).read_bytes();candidate=json.loads(raw)
    field=scope['allowed_exact_delta']['field'];require(candidate[field]is None,'Only pending transition selector')
    old_text='"'+field+'": null'
    pin_text=json.dumps(scope['actual_pause'],ensure_ascii=False,sort_keys=True,allow_nan=False,indent=2)
    new_text='"'+field+'": '+pin_text.replace('\n','\n  ')
    require(raw.count(old_text.encode())==1,'Exactly one final selector literal')
    new_raw=raw.replace(old_text.encode(),new_text.encode(),1)
    actual=json.loads(new_raw)
    expected=copy.deepcopy(candidate);expected[field]=scope['actual_pause']
    require(actual==expected,'Exact one dictionary field delta')
    undo=copy.deepcopy(actual);undo[field]=None
    require(undo==candidate,'All other candidate fields byte-value equivalent')
    stripped=copy.deepcopy(actual)
    for key in ('authoritative_current_plan_identity_and_main_path_attribution_policy',field):del stripped[key]
    proper=json.loads(pathlib.Path(scope['proper_source37_configuration']['path']).read_text(encoding='utf-8'))
    require(stripped==proper,'Proper original37 complete dictionary after ONLY2 operational fields')
    require(new_raw.replace(new_text.encode(),old_text.encode(),1)==raw,'Whole-byte inverse exact')
    with outputs['configuration'].open('xb')as f:f.write(new_raw);f.flush();os.fsync(f.fileno())
    result['outputs']['configuration']=pin(outputs['configuration'])
    delta=dict(kind='ASR18_source39_actual204800_configuration_exact_one_field_delta_v1',ROOT_review=pin(SCOPE),
        base_candidate=scope['base_candidate'],successor=result['outputs']['configuration'],proper_source37=scope['proper_source37_configuration'],
        field=field,old=None,new=scope['actual_pause'],all_other_candidate_dictionary_fields_equal=True,
        proper37_exact_after_strip2=True,ordered_RAW_edits=[dict(old=old_text,new=new_text,count=1)],
        whole_original_byte_inverse=True,reconstructed_original_bytes=len(raw),reconstructed_original_sha256=hashlib.sha256(raw).hexdigest(),
        source_science_resource_LR_GC_changes=False)
    result['outputs']['delta']=write(outputs['delta'],delta)
    future_runtime7=dict(parent_driver_source=scope['driver_source'],worker_driver_source=scope['driver_source'],
        resident_RAM_stream_configuration=result['outputs']['configuration'],source_configuration=result['outputs']['configuration'],
        source_checkpoint=scope['initial_checkpoint'],checkpoint_interval_updates=200,resident_branch_namespace=str(B))
    source_accept=pin(W/'ROOT_actual204800_source39_wrapper40_whole_SOURCE_RAW_AST_INACTIVE_acceptance_v1.json')
    stop_accept=pin(W/'ROOT_actual_source37_saved204800_pause_closed_independent_acceptance_v1.json')
    saved_accept=pin(W/'ROOT_saved204800_full854_PRE_SAVED_historical87_tail_independent_acceptance_v1.json')
    inputs=[pin(SCOPE),scope['base_candidate'],scope['proper_source37_configuration'],scope['driver_source'],scope['wrapper_source'],
        scope['transition_helper'],scope['actual_pause'],scope['CPU_actual02_acceptance'],scope['initial_checkpoint'],
        source_accept,stop_accept,saved_accept,result['outputs']['configuration'],result['outputs']['delta']]
    handoff=dict(kind='ASR18_source39_saved204800_final_actual_configuration_INACTIVE_handoff_v1',ROOT_preparation_scope=pin(SCOPE),
        source_configuration=result['outputs']['configuration'],configuration_delta=result['outputs']['delta'],
        source39=scope['driver_source'],wrapper40=scope['wrapper_source'],transition_helper4=scope['transition_helper'],
        CPU_actual02_review=scope['CPU_actual02_acceptance'],actual_pause=scope['actual_pause'],actual_pause_ROOT_review=stop_accept,
        saved_full854_and_historical87_ROOT_review=saved_accept,source_review=source_accept,initial_checkpoint=scope['initial_checkpoint'],
        future_runtime7=future_runtime7,future_namespace=future_namespace,
        proposed_exact_command=argv,proposed_prefix=argv[:4],proposed_wrapper_arguments=argv[4:],
        proposed_ROOT_final_review=argv[argv.index('--review')+1],input_pins=inputs,
        protected_runtime_identities=protected,preparer=entry['identity'],launcher=entry['launcher'],initial=pin(INITIAL),
        no_actual_GPU_execution=True,original_I26_marker_preserved=True,
        recovery_requirements=dict(full_saved204800_all_state_restore_before_only_savedprefix_copy=True,
            original_J26='STAT ONLY here; no closed tail/prefix bytes read during preparation',historical87_genuinely_redone=True,
            new88_expected_once_range=[204801,204888],natural_standard=205000,post_positive_min=205020,
            inherited180k_consumed_no_second_capture_GC_or_sync=True,reader_healthy_advancing_catchup_no_restart=True),
        durable_validation=dict(start=204820,end=209020,original20rows_all_saves_pauses_inclusive=True,
            next_real_DEV=208000,post_DEV_updates=1020,fixed_first_final400=True,two_postDEV400=True,
            primary_minimum_sustained_inclusive_rate=70,primary_maximum_inclusive_first_final_decline_percent=5,
            optional_save_exclusions_separately_exact_aligned=True,no_warm_only_or_pure_cause_claim=True),
        full401072_adminNone=True,LR_unchanged=[1e-5,1e-4,3e-5],guards_and_separate72MiB_preserved=True,
        pending='Independent ROOT final config/currentpin/ONE production execution binding REQUIRED')
    result['outputs']['handoff']=write(outputs['handoff'],handoff)
    final=sample_memory(me.pid,'source39_final_configuration_preparation_final')
    require(final['system_memory']['commit_available_bytes']>=2343932928 and final['system_memory']['physical_available_bytes']>=1843932928,'Unchanged runtime floors')
    require(me.memory_info().private<=536870912 and time.monotonic()-began<180,'Original self/wall guard')
    result['final_sample']=final;result['passed']=True
except BaseException as exc:
    result['passed']=False;result['error']=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
finally:
    result['wall_seconds']=time.monotonic()-began
    result_pin=write(outputs['result'],result)
    print(json.dumps(dict(result=result_pin,outputs=result['outputs'],entry=entry,passed=result.get('passed')),sort_keys=True))
require(result.get('passed'),'Preserve actual preparation failure; no automatic retry')
