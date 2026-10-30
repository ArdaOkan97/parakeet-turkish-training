"""Derive one inactive exact dispatcher using the frozen existing guard APIs."""
import ast
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fn(text,name):
 n=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
 return ''.join(text.splitlines(keepends=True)[n.lineno-1:n.end_lineno])
def setfn(text,name,value):return text.replace(fn(text,name),value.rstrip(),1)
base=(HERE/'run_TSC_bpe_shared_head_memory_v2.py').read_text(encoding='utf-8')
text=base.replace(base[:base.index('from __future__')],'''"""INACTIVE guarded SAME-ID own-shared-state exact executor.

Original config1000/source300/native-ready stay immutable. A ROOT actual CLOSED
qualified improving latest supplies full207/two-Adam/sampler/RNG/history; an
approved stopping-bound extension is separate. One owned child only, no automatic
retry, rewind, unfreeze or successor. Full coverage requires actual unique IDs.
"""
''',1)
worker='train_lora_batched_tsc_bpe_shared_head_exact_v1.py'
sync=[n.lineno for n in ast.walk(next(n for n in ast.parse((HERE/worker).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.cuda.synchronize'][0]
text=text.replace("WORKER = 'train_lora_batched_tsc_bpe_shared_head_memory_v2.py'",f"WORKER = '{worker}'",1)
text=text.replace("TRANSITION = 'tsc_bpe_shared_head_memory_v2.py'","TRANSITION = 'tsc_bpe_shared_head_exact_transition_v1.py'",1)
text=text.replace("WORKER_SHA = 'ea74795363e0c09114e0a496ccbea55dab34172202d40c6a9aefb5f7cdfa8290'",f"WORKER_SHA = '{sha(HERE/worker)}'",1)
text=text.replace("TRANSITION_SHA = 'a0e89e83f2d8253a10434a56af3603c88be13ea306b74338cae381dca5794051'",f"TRANSITION_SHA = '{sha(HERE/'tsc_bpe_shared_head_exact_transition_v1.py')}'",1)
text=text.replace('EXPECTED_SYNC_LINE = 532',f'EXPECTED_SYNC_LINE = {sync}',1)
text=text.replace("CONTROLLER_KIND = 'guarded_TSC_BPE512_new_rows_plus_shared_heads_controller_v1'","CONTROLLER_KIND = 'guarded_TSC_BPE_shared_heads_exact_controller_v1'",1)
text=setfn(text,'check_phase_arguments', '''def check_phase_arguments(argv):
    for flag in ('--resume-exact','--resume-weights','--resume-runtime-migration','--resume-optimization',
        '--optimization-source-receipt','--resume-acquisition','--acquisition-source-receipt','--unfreeze','--smoke-updates'):
        require(flag not in argv,'Unsupported own exact state mode:'+flag)
    require(option(argv,'--batch-size')=='8' and option(argv,'--accumulation')=='2'
        and option(argv,'--sampling-mode')=='domain30' and float(option(argv,'--fleurs-fraction'))==.3
        and float(option(argv,'--row-lr')) in (1e-3,3e-4) and float(option(argv,'--shared-lr')) in (1e-5,3e-6)
        and int(option(argv,'--native-max-joint-bytes'))==NATIVE_MAX_BYTES
        and int(option(argv,'--adaptation-max-updates'))==1000,
        'Immutable initial1000/B8x2/domain30/row+shared recipe required')
    for flag in ('--resume-shared-head-exact','--shared-head-exact-source-receipt','--shared-head-exact-runtime-proof'):
        scoped(option(argv,flag))
    return 1000,0''')
text=setfn(text,'boundary_context', '''def boundary_context(argv):
    # Pure boundary only; no worker/Torch import before coherent preload.
    transition=importlib.import_module(Path(TRANSITION).stem)
    receipt=read_small(option(argv,'--shared-head-exact-source-receipt'))
    config=read_small(receipt['artifacts']['run_config']['path'])
    values=dict(config);values.update(config['config_identity']['training_arguments'])
    for flag in ('--resume-shared-head-exact','--shared-head-exact-source-receipt','--shared-head-exact-runtime-proof',
        '--native-backend-provenance','--stop-file'):
        values[flag[2:].replace('-','_')]=scoped(option(argv,flag))
    values.update(run_id=option(argv,'--run-id'),max_wall_seconds=float(option(argv,'--max-wall-seconds')),
        resume_exact=None,resume_weights=None,resume_runtime_migration=None,resume_optimization=None,
        optimization_source_receipt=None,resume_acquisition=None,resume_adaptation=None,smoke_updates=None)
    # Ensure the actual command, including all immutable original arguments,
    # matches its source. Only these five exact/runtime paths may differ.
    source_args=read_small(receipt['artifacts']['controller_status']['path'])['worker_arguments']
    ignored=('--resume-shared-head-exact','--shared-head-exact-source-receipt','--shared-head-exact-runtime-proof',
        '--stop-file','--max-wall-seconds')
    def stripped(items):
        for flag in ignored:items=remove_option(items,flag)
        return items
    require(stripped(list(argv))==stripped(list(source_args)), 'No scientific CLI or original native-ready change')
    return transition.preflight(SimpleNamespace(**values))''')
text=setfn(text,'validate_plan', '''def validate_plan(plan):
    finalized_core();argv=plan['worker_arguments'];check_phase_arguments(argv)
    context=boundary_context(argv);pins=source_pins(context);receipt=context['exact_receipt']
    require(plan['kind']==CONTROLLER_KIND and plan['controller_sha256']==sha(__file__)
        and plan['source_sha256']==pins and option(argv,'--run-id')==plan['run_id']
        and domain.guards.argument_sha(argv)==plan['worker_arguments_sha256'], 'Frozen exact plan/source/arguments changed')
    require(plan['source_closure']==context['exact_receipt_record']
        and plan['source_checkpoint']==receipt['artifacts']['latest_checkpoint']
        and plan['preserved_BEST']==receipt['artifacts']['best_checkpoint']
        and plan['source_selected_step']==context['source_phase1_step']
        and plan['resume_step']==context['resume_step']
        and plan['shared_updates_at_window_start']==context['shared_updates_at_resume']
        and plan['phase_update_limit']==context['approved_total_shared_head_updates']
        and plan['original_source_selected_step']==receipt['original_step']
        and plan['selected_acquisition_updates']==receipt['source_acquisition_updates']
        and plan['original_config_identity_sha256']==canonical(context['parent_identity'])
        and plan['original_native_ready']==receipt['artifacts']['original_native_ready']
        and plan['original_source300']==receipt['artifacts']['initial_source_equivalence'], 'One exact own/latest/BEST/S-K-N/science frontier required')
    require(.1<=plan['window_hours']<=3. and math.isfinite(plan['window_hours'])
        and float(option(argv,'--max-wall-seconds'))==plan['window_hours']*3600-300,
        'Bounded3h whole-update window/grace required')
    paths={key:scoped(value) for key,value in plan['paths'].items()}
    require(len(set(paths.values()))==len(paths)
        and option(argv,'--stop-file')==str(paths['stop'])
        and option(argv,'--shared-head-exact-runtime-proof')==str(paths['native'])
        and scoped(option(argv,'--native-backend-provenance'))==scoped(plan['original_native_ready']['path'])
        and paths['native']!=scoped(plan['original_native_ready']['path']),
        'Fresh PID proof must remain separate from original scientific token')
    caller_line(HERE/WORKER);return paths''')
text=setfn(text,'prepare', '''def prepare(args):
    finalized_core()
    require(math.isfinite(args.window_hours) and .1<=args.window_hours<=3. and 5<=args.poll_seconds<=15,
        'Bounded3h window/fresh5-15s polls required')
    output=scoped(args.output_dir);source_loop=scoped(args.source_loop)
    require(output.parent==HERE/'runs' and not output.exists(), 'Fresh controller window directory required')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{2,110}',args.run_id) is not None
        and (HERE/'runs'/args.run_id).is_dir() and output!=HERE/'runs'/args.run_id,
        'SAME existing worker run namespace, separate fresh window required')
    receipt_path=scoped(args.source_receipt);receipt=read_small(receipt_path)
    original_path=receipt.get('original_artifact_paths',{}).get('controller_status',receipt['artifacts']['controller_status']['path'])
    require(scoped(source_loop/'loop_status.json')==scoped(original_path),'ROOT archived source loop differs')
    source=read_small(receipt['artifacts']['controller_status']['path']);argv=list(source['worker_arguments'])
    require(args.approved_total_shared_head_updates==receipt['approved_total_shared_head_updates'],
        'Explicit CLI total must match ROOT separate extension')
    paths={key:output/f'{args.run_id}_{name}' for key,name in (('native','fresh_native_backend.json'),
        ('identity','identity.json'),('memory','memory.json'),('cache','cache.json'),('io','io.json'),('stop','stop.txt'),('log','worker.log'))}
    for flag,value in (('--resume-shared-head-exact',scoped(receipt['artifacts']['latest_checkpoint']['path'])),
        ('--shared-head-exact-source-receipt',receipt_path),('--shared-head-exact-runtime-proof',paths['native']),
        ('--stop-file',paths['stop']),('--max-wall-seconds',args.window_hours*3600-300)):
        argv=set_option(argv,flag,value)
    check_phase_arguments(argv);context=boundary_context(argv);pins=source_pins(context)
    python=scoped(source['command'][0]);require(python==ROOT/'work/nemo-native-env-v2/Scripts/python.exe' and python.is_file(),
        'Keep actual native-env-v2 interpreter')
    original=context['original_context']
    plan={'schema':1,'kind':CONTROLLER_KIND,'phase':PHASE,'run_id':args.run_id,'loop_id':output.name,
        'controller_sha256':sha(__file__),'source_loop':str(source_loop),'source_closure':context['exact_receipt_record'],
        'source_checkpoint':receipt['artifacts']['latest_checkpoint'],'preserved_BEST':receipt['artifacts']['best_checkpoint'],
        'source_selected_step':context['source_phase1_step'],'source_phase1_step':context['source_phase1_step'],
        'original_source_selected_step':receipt['original_step'],'selected_acquisition_updates':receipt['source_acquisition_updates'],
        'resume_step':context['resume_step'],'shared_updates_at_window_start':context['shared_updates_at_resume'],
        'phase_update_limit':context['approved_total_shared_head_updates'],'approved_total_shared_head_updates':context['approved_total_shared_head_updates'],
        'source_history_prefix_sha256':canonical(context['archived_history']),
        'original_config_identity_sha256':canonical(context['parent_identity']),
        'original_native_ready':receipt['artifacts']['original_native_ready'],
        'original_source300':receipt['artifacts']['initial_source_equivalence'],
        'original207_checkpoint':original['original_context']['receipt']['artifacts']['best_checkpoint'],
        'phase1_checkpoint':original['phase1_record'],'source_recipe':original['source_recipe'],
        'source_sha256':pins,'worker_arguments':argv,'worker_arguments_sha256':domain.guards.argument_sha(argv),
        'command':[str(python),str(Path(__file__).resolve()),'--worker-plan',str(output/'plan.json'),'--gpu-window-authorized'],
        'paths':{key:str(value) for key,value in paths.items()},'window_hours':args.window_hours,
        'row_lr':float(option(argv,'--row-lr')),'shared_lr':float(option(argv,'--shared-lr')),
        'adaptation_max_updates':1000,'smoke_updates':0,'inherited_scientific_coverage_horizon':44958,
        'same_phase_exact_continuation_supported':True,'original_config_or_BEST_reinitialized':False,
        'source300_repeated':False,'unfreeze_supported':False,'objective_complete':False,'heldout_TEST_evaluated':False}
    validate_plan(plan);return plan''')
run=fn(text,'run_worker')
start=run.index('    flags = ');end=run.index('    sources = ',start)
run=run[:start]+'''    transition=importlib.import_module(Path(TRANSITION).stem)
    expected=plan['source_checkpoint']
    for value in (expected,plan['preserved_BEST'],plan['original207_checkpoint'],plan['phase1_checkpoint']):
        transition.checked_record(value,checkpoint=True)
        require(sha(value['path'])==value['sha256'],'Immutable own/original/phase1 bytes changed before runtime')
    ready={'schema':1,'kind':'TSC_BPE_shared_heads_exact_fresh_preload_v1','status':'ready',
        'phase':PHASE,'geometry':GEOMETRY,'run_id':plan['run_id'],'pid':os.getpid(),'ppid':os.getppid(),
        'ready_at_utc':now(),'worker_executor':transition.record(HERE/WORKER),
        'controller_executor':transition.record(__file__),'exact_helper_sha256':TRANSITION_SHA,
        'original_native_ready':plan['original_native_ready'],'source_receipt':plan['source_closure'],
        'preload':preloaded,'allocator_policy':{'cap_bytes':8000000000,'cleanup_threshold_bytes':3000000000,'allocator_cap_applied':False},
        'compactor_sha256':UTILITY_PINS['compact_training_row_objects_v1.py'],
        'native4_adapter_sha256':SUPPORT_PINS['native_tdt_checkpointed_v4.py'],
        'immutable_original_config_identity_sha256':plan['original_config_identity_sha256'],
        'row_lr':plan['row_lr'],'shared_lr':plan['shared_lr'],
        'shared_updates_at_window_start':plan['shared_updates_at_window_start'],
        'approved_total_shared_head_updates':plan['phase_update_limit']}
    write(paths['native'],ready)
''' +run[end:]
run=run.replace("memory['allocator_cap_applied'] = True; write(paths['memory'], memory)",
    "memory['allocator_cap_applied'] = True; write(paths['memory'], memory)\n        ready['allocator_policy']['allocator_cap_applied']=True;write(paths['native'],ready)",1)
text=setfn(text,'run_worker',run)
text=setfn(text,'adaptation_suffix', '''def adaptation_suffix(history,resume_step):
    require(all(row.get('step')==i for i,row in enumerate(history,1)) and len(history)>=resume_step,
        'Exact cumulative history prefix lost')
    suffix=history[resume_step:]
    for fresh,row in enumerate(suffix,1):
        require(row.get('adaptation_updates_completed')==row['step']-row['source_phase1_step'],
            'Exact actual localN differs')
    # Guard observation is fresh1..windowN; saved rows remain cumulative.
    return [{**row,'step':fresh} for fresh,row in enumerate(suffix,1)]''')
text=setfn(text,'verify_completion', '''def verify_completion(plan,selection):
    paths={key:Path(value) for key,value in plan['paths'].items()};run=HERE/'runs'/plan['run_id']
    n=selection.get('adaptation_updates_completed');initial=plan['shared_updates_at_window_start']
    require(type(n) is int and initial<=n<=plan['phase_update_limit']
        and selection['run_id']==plan['run_id'] and selection['phase']==PHASE
        and selection['completed_steps']==plan['source_selected_step']+n
        and selection['source_phase1_step']==plan['source_selected_step']
        and selection['original_source_selected_step']==plan['original_source_selected_step']
        and selection['selected_acquisition_updates']==plan['selected_acquisition_updates']
        and selection['row_Adam_clock']==plan['selected_acquisition_updates']+n
        and selection['shared_Adam_clock']==plan['original_source_selected_step']+n
        and selection['external_exact_resume_supported'] is True and selection['further_unfreeze_allowed'] is False
        and selection['heldout_tests_evaluated'] is False,'Exact S-K-N/latest/BEST frontier required')
    identity=read(paths['identity']);completion=read(paths['native'].with_suffix('.completion.json'))
    require(completion.get('kind')=='TSC_BPE_adaptation_native_completion_v1' and completion.get('status')=='complete'
        and completion['sources_unchanged'] is True and completion['pid']==identity['pid']
        and completion['run_id']==plan['run_id'] and completion['backend_ready_sha256']==sha(paths['native'])
        and completion['actual_dispatch_source_sha256']==sha(__file__),'Actual owned exact completion required')
    for key,kind in (('io','TSC_BPE_adaptation_atomic_IO_v1'),('cache','TSC_BPE_adaptation_cache_runtime_v1'),('memory','TSC_BPE_adaptation_memory_runtime_v1')):
        value=read(paths[key]);require(value['kind']==kind and value['status']=='complete' and value['sources_unchanged'] is True,
            'Actual exact runtime missing:'+key)
    memory=read(paths['memory']);cache=read(paths['cache'])
    require(memory['cap_bytes']==8000000000 and memory['cleanup_threshold_bytes']==3000000000
        and memory['allocator_cap_applied'] is True and memory['all_phase_owned_memory_telemetry']['thread_join_completed'] is True
        and cache['cache_reclamation']['eligible_update_calls']==n-initial,'Cap8/end3/join/current-window suffix hook required')
    config=read(run/'run_config.json');history=read(run/'training_history.json')
    require(canonical(config['config_identity'])==plan['original_config_identity_sha256']
        and len(history)==selection['completed_steps']
        and canonical(history[:plan['resume_step']])==plan['source_history_prefix_sha256'],
        'Original config/history prefix cannot be reset or relabelled')
    require(record(plan['original_native_ready']['path'])=={key:plan['original_native_ready'][key] for key in ('path','sha256')}
        and record(run/PROOF_NAME)=={key:plan['original_source300'][key] for key in ('path','sha256')},
        'Original immutable native/source300 tokens changed')
    events=list(run.glob('shared_head_exact_execution_from_'+str(plan['resume_step'])+'_'+plan['source_closure']['sha256'][:16]+'.json'))
    require(len(events)==1,'One exact execution event required');execution=read(events[0])
    require(selection['exact_execution']==execution and execution['original_config_or_BEST_reinitialized'] is False
        and execution['source300_repeated'] is False and execution['approved_total_shared_head_updates']==plan['phase_update_limit'],
        'Separate stopping extension/executor evidence changed')
    startup=read(events[0].with_name(events[0].stem+'_startup.json'));compact=startup['training_row_compaction']
    require(startup['kind']=='shared_heads_exact_startup_compaction_v1' and startup['passed'] is True
        and compact['passed'] is True and compact['rows_verified']==253798
        and compact['all_rows_canonical_equal'] is True and compact['all_row_key_orders_equal'] is True
        and compact['same_row_list_and_row_dict_objects'] is True and compact['same_label_list_objects'] is True,
        'Actual exact cold metadata compaction required')
    source_pins(boundary_context(plan['worker_arguments']))
    return {'new_adaptation_updates':n-initial,'shared_updates_at_window_start':initial,
        'completed_shared_updates':n,'native_completion':record(paths['native'].with_suffix('.completion.json')),
        'exact_execution':record(events[0]),'original_config_unchanged':True,
        'full_unique_TRAIN_coverage':selection['full_unique_TRAIN_coverage'],'goal_complete':False}''')
window=fn(text,'run_window').replace("inherited = state['source_selected_step']; last_dev = inherited", "inherited = state['resume_step']; last_dev = inherited")
window=window.replace("adaptation_updates_published=len(suffix)","new_window_updates_published=len(suffix), adaptation_updates_published=state['shared_updates_at_window_start']+len(suffix)")
window=window.replace("next_action='ROOT reviews row diagnostics, protected bytes, GPU fit and DEV; no automatic exact continuation or unfreeze'",
    "next_action='ROOT reviews actual qualifying latest/full coverage before a separately approved CLOSED exact successor; no automatic reset/unfreeze'")
text=setfn(text,'run_window',window)
text=setfn(text,'describe_contract', '''def describe_contract():
    return {'kind':CONTROLLER_KIND,'status':'sources_pinned_only','phase':PHASE,
        'SAME_ID_qualified_improving_latest_only':True,'earlier_BEST_fork_supported':False,
        'full207_twoAdam_current_sampler_RNG_history_preserved':True,'original_config1000_native_source300_immutable':True,
        'approved_totalN_extension_separate':True,'maximum_global_horizon':44958,'budget_alone_proves_fullcoverage':False,
        'cap_bytes':8000000000,'end_cache_bytes':3000000000,'native_max_joint_bytes':NATIVE_MAX_BYTES,
        'worker_sha256':WORKER_SHA,'boundary_sha256':TRANSITION_SHA,'sync_line':EXPECTED_SYNC_LINE,
        'GPU_launched':False,'actual_state_or_fit_claimed':False,'unfreeze_supported':False}''')
main=fn(text,'main').replace("    require(not args.resume_exact and not args.resume_closed_loop, 'Exact adaptation continuation is deliberately unsupported')\n",'')
text=setfn(text,'main',main)
text=setfn(text,'parser', '''def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--describe-contract',action='store_true');result.add_argument('--worker-plan',type=Path)
    result.add_argument('--source-loop',type=Path);result.add_argument('--source-receipt',type=Path)
    result.add_argument('--run-id');result.add_argument('--output-dir',type=Path)
    result.add_argument('--approved-total-shared-head-updates',type=int)
    result.add_argument('--window-hours',type=float,default=3.);result.add_argument('--poll-seconds',type=float,default=5.)
    result.add_argument('--validate-only',action='store_true');result.add_argument('--preflight-output',type=Path)
    result.add_argument('--gpu-window-authorized',action='store_true');return result''')
ast.parse(text)
target=HERE/'run_TSC_bpe_shared_head_exact_v1.py'
with target.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
print({'controller_sha256':sha(target),'worker_sha256':sha(HERE/worker),'sync_line':sync})
