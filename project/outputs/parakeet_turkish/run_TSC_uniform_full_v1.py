"""INACTIVE original8199 full253798 uniform TRAIN with Portuguese retention.

One actual owned child per bounded window. Only observed exit0/complete budget
closures continue SAME full207/Adam/sampler/RNG/history automatically. Resource,
manual, retention, plateau or unknown exits stop for review; no fresh reset.
ROOT exclusive GPU authorization and current memory admission are required.
"""
from __future__ import annotations
import argparse, ast, functools, hashlib, importlib, json, math, os, re, shutil, subprocess, sys, threading, time
from collections import deque
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from atomic_io_retry_v1 import read_json as read,write_json as write,commit_writer
from cuda_cache_reclaim_v2 import CudaCacheReclaimer as CacheOperations
import launch_with_memory_guard_TSC_resume_v3 as shared_memory
import run_hf_TSC_arm_v3 as domain
from resource_watchdog_v2 import ResourceGuard,sample_memory
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
WORKER='train_lora_batched_tsc_uniform_full_v1.py'
TRANSITION='tsc_uniform_full_transition_v1.py'
CONTROLLER_KIND='guarded_TSC_8199_uniform_full_Portuguese_controller_v1'
WORKER_SHA='b9e550ec145c56a7a75521a997a8a6a33bd98eee7d76c1cb0cabfb116a40631c'
TRANSITION_SHA='08ec46e079cadf2805f72b24078b92fc560e5701228a033228eb3519219079de'
SAMPLER_SHA='8ebd56d20e47a568db4a7cfe50ed172ad86af253275b3b1172ad78a34b60439a'
LAZY_ROWS_SHA='faf2b549d2e5e3d2f2fd78d282b612c6943e813cbedbc3f6b16caedab8249fd2'
EXPECTED_SYNC_LINE=667
SHARED_MEMORY_SHA='61fec6d3ef911665bdbe876dc333f3c6a640badc012596fe92ded1b80a10c7fa'
PRELOAD_SHA='f8787b0f3346c9f526903ff0c06de4c266f940ca3b0f41c55907cc795e7cc2ba'
ADAPTER_SHA='24b9cf5b8f3e11d5b043c22d0940da3735a46f8f70b6f5810b97f17d545cd90e'
COMPACTOR_SHA='1b3f9f0e3bdd19c285f2825a53685d7cf9fd964040c9087560f610a5ed1ec96d'
def require(ok,message):
    if not ok: raise ValueError(message)


def now(): return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def record(path): return {'path':str(Path(path).resolve()),'sha256':sha(path)}


def canonical(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def scoped(path):
    p=Path(path).resolve();require(p.is_relative_to(ROOT),'Workspace-only path required');return p


def option(argv,flag):
    require(argv.count(flag)==1 and argv.index(flag)+1<len(argv),'Exactly one '+flag+' required')
    return argv[argv.index(flag)+1]


def set_option(argv,flag,value):
    result=list(argv)
    if flag in result:
        require(result.count(flag)==1,'Duplicate '+flag);i=result.index(flag);result[i+1]=str(value)
    else: result.extend((flag,str(value)))
    return result


def remove_option(argv,flag):
    result=list(argv)
    if flag in result:
        require(result.count(flag)==1,'Duplicate '+flag);i=result.index(flag);del result[i:i+2]
    return result


def boundary_context(argv):
    from tsc_uniform_full_transition_v1 import preflight,source_context
    receipt=read(option(argv,'--uniform-full-source-receipt'));origin=receipt['source_lineage_receipt']
    source=source_context(str(scoped(origin['path'])),origin['sha256'])
    values=dict(source['config']);values.update(source['source_recipe'])
    values.update(sampling_mode='uniform',steps=38225,run_id=option(argv,'--run-id'),resume_optimization=None,
        optimization_source_receipt=None,resume_weights=None,resume_runtime_migration=None,
        resume_exact=scoped(option(argv,'--resume-exact')) if '--resume-exact' in argv else None,
        portuguese_seed=int(option(argv,'--portuguese-seed')))
    for flag in ('--resume-uniform-full','--uniform-full-source-receipt','--uniform-full-runtime-proof',
        '--native-backend-provenance','--portuguese-preparation-proof','--portuguese-baseline-proof',
        '--portuguese-replay-manifest','--portuguese-dev-manifest','--indexed-tr-directory','--stop-file'):
        values[flag[2:].replace('-','_')]=scoped(option(argv,flag))
    values['max_wall_seconds']=float(option(argv,'--max-wall-seconds'))
    return preflight(SimpleNamespace(**values))


def source_pins(context=None):
    require('PENDING' not in (WORKER_SHA,TRANSITION_SHA,SAMPLER_SHA) and EXPECTED_SYNC_LINE>0,'Full worker/helper/sampler pins not frozen')
    pins=dict(domain.chain_identity()['source_sha256'])
    if context:
        for name,value in context['parent_identity']['source_sha256'].items():
            require(name not in pins or pins[name]==value,'Original support science conflict:'+name);pins[name]=value
    pins.update({WORKER:WORKER_SHA,TRANSITION:TRANSITION_SHA,'tsc_uniform_full_sampler_v1.py':SAMPLER_SHA,
        'lazy_training_rows_v1.py':LAZY_ROWS_SHA,
        'native_tdt_checkpointed_v4.py':ADAPTER_SHA,'compact_training_row_objects_v1.py':COMPACTOR_SHA,
        'launch_with_memory_guard_TSC_resume_v3.py':SHARED_MEMORY_SHA,Path(__file__).name:sha(__file__),
        'run_hf_TSC_arm_v3.py':sha(domain.__file__)})
    for name,value in pins.items():require(sha(HERE/name)==value,'Frozen full-run dependency differs:'+name)
    require(pins['preload_nemo_linker_v1.py']==PRELOAD_SHA,'Coherent preloader changed');return pins


def caller_line(worker_path):
    require(sha(worker_path)==WORKER_SHA,'Reviewed fork worker changed')
    tree=ast.parse(Path(worker_path).read_text(encoding='utf-8'))
    functions=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='run']
    calls=[x for f in functions for x in ast.walk(f) if isinstance(x,ast.Call)
        and isinstance(x.func,ast.Attribute) and x.func.attr=='synchronize'
        and ast.unparse(x.func.value)=='torch.cuda']
    require(len(functions)==1 and len(calls)==1 and not calls[0].args and not calls[0].keywords
        and calls[0].lineno==EXPECTED_SYNC_LINE,'Single reviewed optimizer-end call required')
    return calls[0].lineno


class ForkCaller(CacheOperations):
    def __init__(self,cuda,worker_path):
        self.cuda=cuda;self.worker_path=Path(worker_path).resolve();self.expected_sha=WORKER_SHA
        self.line=caller_line(self.worker_path);self.threshold_bytes=3000000000;self.max_events=128
        self.events=deque(maxlen=128);self.eligible_calls=self.ignored_calls=self.reclamations=self.failed_reclamations=0
        self.cleanup_seconds=0.;self.original=self.wrapped=None


def validate_plan(plan):
    argv=plan['worker_arguments'];context=boundary_context(argv);pins=source_pins(context)
    require(plan['kind']==CONTROLLER_KIND and plan['source_sha256']==pins and plan['controller_sha256']==sha(__file__)
        and option(argv,'--run-id')==plan['run_id'] and domain.guards.argument_sha(argv)==plan['worker_arguments_sha256'],
        'Actual full-run plan/source/argument identity required')
    require(option(argv,'--sampling-mode')=='uniform' and option(argv,'--head-lr')=='1e-05'
        and float(option(argv,'--lora-lr'))==.0003 and option(argv,'--batch-size')=='8' and option(argv,'--accumulation')=='2'
        and option(argv,'--steps')=='38225' and option(argv,'--eval-every-updates')=='250'
        and '--resume-weights' not in argv and '--resume-optimization' not in argv and '--resume-runtime-migration' not in argv,
        'Full207 originalrecipe, physicalB8x2 and complete newpass required')
    exact=plan.get('continuation_mode')=='same_new_run_exact_budget_continuation'
    require(('--resume-exact' in argv)==exact and plan['source_closure']==context['receipt_record']
        and plan['initial_source_checkpoint']==context['source_checkpoint'],'Declared source6500 or SAME-own exact only')
    state_path=scoped(option(argv,'--resume-exact' if exact else '--resume-uniform-full'))
    require(record(state_path)==plan['source_checkpoint'],'Actual complete207/Adam frontier changed')
    paths={key:scoped(value) for key,value in plan['paths'].items()}
    runtime=paths['fresh_runtime'] if exact else paths['native']
    require(option(argv,'--uniform-full-runtime-proof')==str(runtime)
        and option(argv,'--native-backend-provenance')==str(paths['native'])
        and option(argv,'--stop-file')==str(paths['stop']) and runtime!=paths['identity'],
        'Fresh process evidence separate from immutable original ready')
    if exact:require(record(paths['native'])==plan['original_ready'],'Original first native token changed')
    caller_line(HERE/WORKER);return paths


def run_worker(plan):
    """New actual child: preload -> cap -> unchanged cache/IO APIs -> worker.run."""
    paths=validate_plan(plan);argv=plan['worker_arguments'];pins=plan['source_sha256']
    exact=plan.get('continuation_mode')=='same_new_run_exact_budget_continuation'
    require(scoped(option(argv,'--resume-exact' if exact else '--resume-uniform-full')).is_file(),
        'Actual dispatch requires the verified complete state artifact')
    for k in (('fresh_runtime','identity','memory','cache','io') if exact else ('native','identity','memory','cache','io')):
        require(not paths[k].exists(),'Existing runtime evidence is immutable: '+k)
    runtime_path=paths['fresh_runtime'] if exact else paths['native']
    live=Path(str(paths['identity'])+'.live_memory.json');completion=runtime_path.with_suffix('.completion.json')
    require(not live.exists() and not completion.exists(),'Fresh self/native evidence required')
    require('torch' not in sys.modules,'Actual preload must precede Torch import')
    before_preload=shared_memory.safe_sample(os.getpid(),'before_native_preload_and_Torch_import')
    from preload_nemo_linker_v1 import preload
    preloaded=preload()
    require(preloaded.get('kind')=='coherent_native_cuda12_9_preload_v1'
        and preloaded.get('actual_binding_linker',{}).get('version')==[12,9]
        and preloaded.get('cuda_initialized') is False,'Coherent pre-Torch native preload required')
    assets={flag[2:].replace('-','_'):record(option(argv,flag)) for flag in (
        '--tsc-input-proof','--tsc-training-approval','--source-closure','--tr-manifest','--replay-manifest',
        '--tr-frame-metadata','--tr-frame-metadata-proof','--data-v3-proof',
        '--resume-exact' if exact else '--resume-uniform-full','--uniform-full-source-receipt',
        '--portuguese-preparation-proof','--portuguese-baseline-proof','--portuguese-replay-manifest','--portuguese-dev-manifest')}
    ready={'schema':1,'kind':'native_tdt_training_backend_v3','status':'ready','run_id':plan['run_id'],
        'tdt_implementation':'nemo','preload':preloaded,'pid':os.getpid(),'ppid':os.getppid(),
        'ready_at_utc':now(),'trainer_path':str(HERE/WORKER),'trainer_sha256':WORKER_SHA,
        'adapter_path':str(HERE/'native_tdt_checkpointed_v4.py'),'adapter_sha256':ADAPTER_SHA,
        'launcher_path':str(Path(__file__).resolve()),'launcher_sha256':sha(__file__),
        'memory_launcher_path':str(Path(__file__).resolve()),'memory_launcher_sha256':sha(__file__),
        'worker_arguments_sha256':plan['worker_arguments_sha256'],'TSC_explicit_asset_records':assets,
        'declared_allocator_policy':{'cap_gb':8,'end_cache_gb':3,'native_inactive_threshold_bytes':750000000},
        'allocation_policy':{'TR_storage':'immutable indexed JSONL/mmap/SQLite','decoded_TR_cache_cap_rows':256,
            'resident_full_TR_rowdict_pool':False,'training_pool_artifact':'compact index provenance plus1914 replay headers',
            'lazy_helper_sha256':LAZY_ROWS_SHA,'source_sha256':pins,'measured_host_relief_claimed':False},
        'declared_optimization_change':'Original207 BEST6500 retained; fulluniformTR+Portuguese replay/retention explicitly added; native4 memory-only cleanup',
        'native_full_model_or_Lightning_trainer_equivalence_claimed':False,'allocator_cap_applied':False}
    if exact:
        ready.update(kind='TSC_uniform_full_fresh_preload_v1',original_first_start_ready=plan['original_ready'],
            fresh_PID_separate_from_original_ready=True,scientific_config_identity_changed=False)
    write(runtime_path,ready)
    sources={'memory_launcher_sha256':sha(__file__),'cache_launcher_sha256':sha(__file__),
        'cache_helper_sha256':pins['cuda_cache_reclaim_v2.py'],'resource_helper_sha256':pins['resource_watchdog_v2.py'],
        'resource_telemetry_sha256':pins['paired_windows_commit_telemetry_v1.py'],
        'frozen_worker_sha256':WORKER_SHA,'data_policy_sha256':pins['public_turkish_data_policy_v3.py']}
    identity={'schema':1,'kind':'owned_training_worker_identity_v1','pid':os.getpid(),'ppid':os.getppid(),
        'expected_run_id':plan['run_id'],'started_at_utc':now(),'worker_arguments_sha256':plan['worker_arguments_sha256'],
        'process_creation_identity':shared_memory.creation_identity(os.getpid()),'live_memory_path':str(live),
        'actual_combined_dispatch_source':record(__file__),'borrowed_implementation_sources':pins,
        'telemetry_before_torch_import':shared_memory.safe_sample(os.getpid(),'before_torch_import'),**sources}
    write(paths['identity'],identity);identity_sha=sha(paths['identity']);lock=threading.Lock();latest=[float('-inf')]
    with (runtime_path.parent/'preTorch_stage_memory.jsonl').open('x',encoding='utf-8') as trace:
        trace.write(json.dumps({'stage':'before_native_preload_and_Torch_import','pid':os.getpid(),'at_utc':identity['started_at_utc'],
            'actual_paired_memory':before_preload,'CUDA_initialized':False})+'\n')
        trace.write(json.dumps({'stage':'after_native_preload_before_Torch_import','pid':os.getpid(),'at_utc':identity['started_at_utc'],
            'actual_paired_memory':identity['telemetry_before_torch_import'],'CUDA_initialized':False})+'\n')
    def publish(value):
        observed=value.get('observed_monotonic',time.monotonic())
        with lock:
            if observed<latest[0]:return
            write(live,{'schema':1,'kind':'owned_worker_live_memory_v1','pid':identity['pid'],'ppid':identity['ppid'],
                'run_id':plan['run_id'],'worker_identity_sha256':identity_sha,
                'process_creation_identity':identity['process_creation_identity'],**sources,'sample':value},retry_seconds=1.)
            latest[0]=observed
    publish(identity['telemetry_before_torch_import'])
    import torch
    cap=shared_memory.validate_capacity(torch.cuda.get_device_properties(0).total_memory,8.)
    telemetry=shared_memory.UpdateTelemetry(identity['pid'],sink=publish)
    sampler=shared_memory.LiveTelemetrySampler(identity['pid'],publish)
    caller=shared_memory.telemetry_reclaimer_class(ForkCaller,telemetry)(torch.cuda,HERE/WORKER)
    memory={'kind':'TSC_uniform_full_memory_runtime_v1','status':'running',**sources,**cap,
        'cleanup_threshold_bytes':3_000_000_000,'declared_head_lr':1e-5,'run_id':plan['run_id'],
        'backend_flags_changed':False,'allocator_scope':'Owned process PyTorch CUDA allocator only',
        'borrowed_memory_implementation':record(shared_memory.__file__),'started_at_utc':now()}
    cache={'kind':'TSC_uniform_full_cache_runtime_v1','status':'running','worker_sha256':WORKER_SHA,
        'inherited_cache_implementation':record(HERE/'cuda_cache_reclaim_v2.py'),'started_at_utc':now()}
    io={'kind':'TSC_uniform_full_atomic_IO_v1','status':'running','worker_sha256':WORKER_SHA,
        'original_serializer_preserved':True,'helper':record(HERE/'atomic_io_retry_v1.py'),'started_at_utc':now()}
    retry_events=deque(maxlen=128);worker=None;original_save=original_write=benchmark_write=None;outcome='failed';error=None
    def event(value):retry_events.append(value)
    atomic_write=functools.partial(write,retry_seconds=30.,event=event)
    for p,e in ((paths['memory'],memory),(paths['cache'],cache),(paths['io'],io)):write(p,e)
    try:
        torch.cuda.set_per_process_memory_fraction(cap['per_process_allocator_fraction'],device=0)
        memory['allocator_cap_applied']=True;write(paths['memory'],memory)
        ready['allocator_cap_applied']=True;write(runtime_path,ready)
        sampler.start();caller.install()
        import benchmark
        benchmark_write=benchmark.write_json;benchmark.write_json=atomic_write
        worker=importlib.import_module(Path(WORKER).stem)
        original_write=worker.write_json;worker.write_json=atomic_write;original_save=worker.save_training_state
        @functools.wraps(original_save)
        def save(path,*values,**keywords):
            return commit_writer(path,lambda target:original_save(target,*values,**keywords),retry_seconds=30.,event=event)
        worker.save_training_state=save
        parsed=worker.validate_arguments(worker.parser().parse_args(argv));worker.run(parsed)
        outcome='complete'
    except BaseException as exc:
        error=repr(exc);raise
    finally:
        sampler.stop_and_join();caller.restore()
        if worker is not None:
            if original_save is not None:worker.save_training_state=original_save
            if original_write is not None:worker.write_json=original_write
        if benchmark_write is not None:benchmark.write_json=benchmark_write
        unchanged=all(sha(HERE/name)==digest for name,digest in pins.items())
        io.update(status=outcome,error=error,finished_at_utc=now(),retry_events=list(retry_events),sources_unchanged=unchanged)
        cache.update(status=outcome,error=error,finished_at_utc=now(),cache_reclamation=caller.evidence(),sources_unchanged=unchanged)
        native=sys.modules.get('native_tdt_checkpointed_v4')
        memory.update(status=outcome,error=error,finished_at_utc=now(),sources_unchanged=unchanged,
            owned_update_memory_telemetry=telemetry.evidence(),all_phase_owned_memory_telemetry=sampler.evidence(),
            native_cleanup_evidence=native.cache_cleanup_evidence() if native else {'available':False})
        memory['owned_update_memory_telemetry']['callback_scope']=f'Exact reviewed worker.run optimizer-end synchronize line{EXPECTED_SYNC_LINE}; observation only'
        for p,e in ((paths['memory'],memory),(paths['cache'],cache),(paths['io'],io)):write(p,e)
        write(completion,{'kind':'TSC_uniform_full_native_completion_v1','status':outcome,'run_id':plan['run_id'],
            'pid':os.getpid(),'sources_unchanged':unchanged,'backend_ready_sha256':sha(paths['native']),
            'fresh_runtime_sha256':sha(runtime_path),'original_ready_reused_as_current_PID_evidence':False,
            'actual_dispatch_source_sha256':sha(__file__),'finished_at_utc':now(),'error':error})
        require(unchanged,'Source changed during the optimization fork')


def validate_identity(value,plan,child_pid,started):
    require(value.get('kind')=='owned_training_worker_identity_v1' and value.get('expected_run_id')==plan['run_id']
        and value.get('worker_arguments_sha256')==plan['worker_arguments_sha256'],'Actual owned argument identity differs')
    require(type(value.get('pid')) is int and value['pid']>0
        and (value['pid']==child_pid or value.get('ppid')==child_pid),'Actual worker is not owned child/self')
    names={'memory_launcher_sha256':Path(__file__).name,'cache_launcher_sha256':Path(__file__).name,
        'cache_helper_sha256':'cuda_cache_reclaim_v2.py','resource_helper_sha256':'resource_watchdog_v2.py',
        'resource_telemetry_sha256':'paired_windows_commit_telemetry_v1.py','frozen_worker_sha256':WORKER,
        'data_policy_sha256':'public_turkish_data_policy_v3.py'}
    for key,name in names.items():require(value.get(key)==plan['source_sha256'][name],'Actual source differs: '+key)
    creation=value.get('process_creation_identity',{})
    require(creation.get('available') is True and creation.get('pid')==value['pid']
        and domain.guards.stamp(started).timestamp()-5<=creation['creation_unix_seconds']
        <=domain.guards.stamp(value['started_at_utc']).timestamp()+1,'Owned worker creation identity differs')
    require(value.get('actual_combined_dispatch_source')==record(__file__),'Actual one-file dispatcher differs')
    return value


def prepare(args):
    from tsc_uniform_full_transition_v1 import source_context
    require(math.isfinite(args.window_hours) and .1<=args.window_hours<=3. and 5<=args.poll_seconds<=15,'Bounded windows/freshpolls required')
    output=scoped(args.output_dir);receipt_path=scoped(args.source_receipt);receipt=read(receipt_path)
    require(output.parent==HERE/'runs' and not output.exists(),'Fresh first controller directory required')
    origin=receipt['source_lineage_receipt'];context=source_context(str(scoped(origin['path'])),origin['sha256'])
    source_status=read(context['lineage_receipt']['artifacts']['controller_status']['path'])
    argv=list(source_status['worker_arguments'])
    for flag in ('--resume-optimization','--optimization-source-receipt','--resume-exact','--resume-weights','--resume-runtime-migration'):
        argv=remove_option(argv,flag)
    paths={key:output/(args.run_id+'_'+name) for key,name in (('native','native_backend.json'),('identity','identity.json'),
        ('memory','memory.json'),('cache','cache.json'),('io','io.json'),('stop','stop.txt'),('log','worker.log'))}
    from tsc_uniform_full_transition_v1 import SOURCE_STEP
    for flag,value in (('--run-id',args.run_id),('--resume-uniform-full',context['source_checkpoint']['path']),
        ('--uniform-full-source-receipt',receipt_path),('--uniform-full-runtime-proof',paths['native']),
        ('--indexed-tr-directory',args.indexed_tr_directory or ROOT/'work/indexed-TRAIN'/args.run_id),
        ('--portuguese-preparation-proof',receipt['portuguese_preparation_proof']['path']),
        ('--portuguese-baseline-proof',receipt['portuguese_baseline_proof']['path']),
        ('--portuguese-replay-manifest',args.portuguese_replay_manifest),('--portuguese-dev-manifest',args.portuguese_dev_manifest),
        ('--portuguese-seed',args.portuguese_seed),('--sampling-mode','uniform'),('--steps',38225),
        ('--native-backend-provenance',paths['native']),('--stop-file',paths['stop']),('--max-wall-seconds',args.window_hours*3600-300)):
        argv=set_option(argv,flag,value)
    context=boundary_context(argv);pins=source_pins(context)
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{2,110}',args.run_id) and not (HERE/'runs'/args.run_id).exists(),
        'Fresh fullrun scientific namespace required')
    for suffix in ('best.pt','latest.pt'):require(not (ROOT/'work'/f'{args.run_id}_{suffix}').exists(),'Fullrun checkpoint namespace collision')
    python=ROOT/'work/nemo-native-env-v2/Scripts/python.exe'
    plan={'schema':1,'kind':CONTROLLER_KIND,'run_id':args.run_id,'controller_sha256':sha(__file__),
        'source_sha256':pins,'source_closure':context['receipt_record'],'source_checkpoint':record(context['source_checkpoint']['path']),
        'initial_source_checkpoint':context['source_checkpoint'],'source_selected_step':SOURCE_STEP,'resume_step':SOURCE_STEP,
        'source_history_prefix_sha256':context['source_history_prefix_sha256'],'window_hours':args.window_hours,
        'worker_arguments':argv,'worker_arguments_sha256':domain.guards.argument_sha(argv),
        'command':[str(python),str(Path(__file__).resolve()),'--worker-plan',str(output/'plan.json'),'--gpu-window-authorized'],
        'paths':{key:str(value) for key,value in paths.items()},'configured_steps':38225,'full_new_uniform_updates':31725,
        'Portuguese_replay_and_retention_added':True,'competitive_champion_parked':context['source_checkpoint'],
        'objective_complete':False}
    validate_plan(plan);return plan


def verify_completion(plan,selection):
    paths={key:Path(value) for key,value in plan['paths'].items()};run=HERE/'runs'/plan['run_id']
    exact=plan.get('continuation_mode')=='same_new_run_exact_budget_continuation';runtime=paths['fresh_runtime'] if exact else paths['native']
    identity=read(paths['identity']);done=read(runtime.with_suffix('.completion.json'))
    require(done['kind']=='TSC_uniform_full_native_completion_v1' and done['status']=='complete'
        and done['sources_unchanged'] is True and done['pid']==identity['pid'] and done['run_id']==plan['run_id']
        and done['fresh_runtime_sha256']==sha(runtime) and done['actual_dispatch_source_sha256']==sha(__file__),
        'Actual owned complete uniform child/runtime required')
    for key,kind in (('memory','TSC_uniform_full_memory_runtime_v1'),('cache','TSC_uniform_full_cache_runtime_v1'),('io','TSC_uniform_full_atomic_IO_v1')):
        value=read(paths[key]);require(value['kind']==kind and value['status']=='complete' and value['sources_unchanged'] is True,'Runtimefailed:'+key)
    memory,cache=read(paths['memory']),read(paths['cache'])
    require(memory['cap_bytes']==8000000000 and memory['cleanup_threshold_bytes']==3000000000
        and memory['allocator_cap_applied'] is True and memory['all_phase_owned_memory_telemetry']['thread_join_completed'] is True
        and cache['cache_reclamation']['eligible_update_calls']==selection['completed_steps']-plan['resume_step'],
        'Retain cap8/end3/join/currentwindow hook scope')
    history=read(run/'training_history.json');config=read(run/'run_config.json')
    require(selection['run_id']==plan['run_id'] and selection['configured_steps']==38225
        and len(history)==selection['completed_steps'] and 6500<=len(history)<=38225
        and selection['heldout_tests_evaluated'] is False and config['trainable_parameter_tensors']==207
        and config['turkish_clips_used']==253798 and config['replay_clips_used']==1914,'Full207/253798/1914 frontier required')
    proof=read(run/'uniform_full_initial_source_equivalence.json')
    require(proof['passed'] is True and proof['actual_GPU_decode'] is True
        and proof['compared_fixed_DEV_utterances']==300 and proof['Portuguese_fixed_DEV_measured_utterances']==50
        and proof['all_five_original_retention_gates_passed'] is True
        and proof['Portuguese50_raw_paths_and_ASR_exact'] is True
        and proof['all_six_retention_gates_passed'] is False,
        'Actual original300+Portuguese96-error recovery source measurement required')
    return {'full207_frontier':selection['completed_steps'],'actual_initial_source350':record(run/'uniform_full_initial_source_equivalence.json'),
        'runtime_completion':record(runtime.with_suffix('.completion.json')),'full_unique_uniform_pass_complete':selection['full_new_uniform_pass_complete']}


def request_stop(path,reason):
    if not path.exists():
        with path.open('x',encoding='utf-8') as stream:stream.write('Cooperative whole-update save/DEV: '+reason+'\n')


def publish_queue(state):
    write(HERE/'experiment_queue_state.json',{'kind':CONTROLLER_KIND,'queue_id':state['run_id']+'_full_pass',
        'pid':os.getpid(),'ppid':os.getppid(),'status':state['status'],'updated_at_utc':now(),
        'current_run_id':state['run_id'],'current_controller_loop':str(Path(state['paths']['log']).parent),
        'current_child_pid':state.get('owned_launcher_pid'),'current_stage':state.get('phase',{'phase':'starting'}),
        'published_step':state.get('published_step',state['resume_step']),
        'actual_new_uniform_updates':max(0,state.get('published_step',state['resume_step'])-6500),
        'configured_steps':38225,'new_uniform_pass_updates':31725,'objective_complete':False,
        'heartbeat_is_not_gradient_progress':True,'stop_category':state.get('stop_category'),
        'actual_runtime_completion':state.get('actual_runtime_completion')})


def run_window(state,deadline,args):
    paths={key:Path(value) for key,value in state['paths'].items()};loop=scoped(args.output_dir)
    child=None;stop_at=None;stop_reason=None;started=time.monotonic();started_utc=now();last_phase=None;last_dev=state['resume_step']
    resource=ResourceGuard();progress=domain.guards.ProgressGuard(started)
    try:
        with paths['log'].open('x',encoding='utf-8') as log:
            child=subprocess.Popen(state['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            state.update(status='running',owned_launcher_pid=child.pid,started_at_utc=started_utc)
            print(f"ACTUAL owned full-pass {state['run_id']} PID{child.pid}; original6500 full207, uniform31725 updates",flush=True)
            while child.poll() is None:
                history=read(HERE/'runs'/state['run_id']/'training_history.json',[])
                identity=validate_identity(read(paths['identity']),state,child.pid,started_utc) if paths['identity'].exists() else None
                suffix=[{**row,'step':i} for i,row in enumerate(history[state['resume_step']:],1)]
                phase=domain.guards.log_phase(paths['log'],suffix)
                sample=sample_memory(identity['pid'] if identity else child.pid,phase['phase'])
                reason,observation=resource.observe(sample['system_memory'],sample['worker_private_memory'],observed_monotonic=sample['observed_monotonic'])
                self_reason,self_observation=domain.guards.self_memory(Path(str(paths['identity'])+'.live_memory.json'),identity,paths['identity']) if identity else (None,{'available':False})
                progress_reason,progress_info=progress.observe(suffix,time.monotonic());reason=reason or self_reason or progress_reason
                if not identity and time.monotonic()-started>120:reason=reason or 'owned_worker_identity_unavailable'
                if identity and not self_observation['available'] and time.monotonic()-started>120:reason=reason or 'actual_worker_self_memory_unavailable_or_stale'
                if not suffix and time.monotonic()-started>1800:reason=reason or 'bounded_preparation_timeout'
                if paths['stop'].exists() and stop_at is None:reason=reason or 'ROOT_manual_cooperative_stop'
                if time.monotonic()>=deadline-300:reason=reason or 'window_budget_checkpoint_boundary'
                state.update(phase=phase,published_step=history[-1]['step'] if history else state['resume_step'],
                    system_and_owned_private_memory=observation,actual_worker_self_memory=self_observation,
                    progress=progress_info,updated_at_utc=now())
                private=observation.get('worker_private_memory') or {}
                if private.get('available') is True and type(private.get('private_bytes')) is int:
                    state['peak_observed_owned_private_bytes']=max(state.get('peak_observed_owned_private_bytes',0),private['private_bytes'])
                if phase['phase']!=last_phase:print(state['run_id']+' phase: '+phase['phase'],flush=True);last_phase=phase['phase']
                for row in history:
                    if row['step']>last_dev and 'validation' in row:
                        gate=row['validation'];print(f"{state['run_id']} DEV{row['step']}: TR{100*gate['turkish_wer']:.4f}%, all6={gate['retention_gate_passed']}",flush=True);last_dev=row['step']
                with (loop/'memory_observations.jsonl').open('a',encoding='utf-8') as out:
                    out.write(json.dumps({'at_utc':now(),'phase':phase,'memory':observation,'worker_self':self_observation,'progress':progress_info})+'\n')
                if reason and stop_at is None:
                    stop_at=time.monotonic();stop_reason=reason
                    request_stop(paths['stop'],reason)
                    state.update(cooperative_stop_reason=reason,cooperative_stop_requested_at_utc=now())
                write(loop/'loop_status.json',state)
                publish_queue(state)
                if stop_at and time.monotonic()-stop_at>300:raise RuntimeError('Owned child exceeded cooperative grace; preserve and review')
                time.sleep(args.poll_seconds)
        state['child_exit_code']=child.wait()
        require(child.returncode==0,'Actual child failed; no automatic GPU retry')
        selection_path=HERE/'runs'/state['run_id']/'selection.json';selection=read(selection_path,{})
        state['actual_runtime_completion']=verify_completion(state,selection)
        resource_stop=bool(stop_reason and stop_reason not in ('window_budget_checkpoint_boundary','ROOT_manual_cooperative_stop'))
        state.update(status='resource_review_required' if resource_stop else 'train_window_ended',
            selected_step=selection['selected_step'],completed_steps=selection['completed_steps'],
            selection=record(selection_path),stop_early_reason=selection['stop_early_reason'],
            resource_review_required=resource_stop,finished_at_utc=now(),objective_complete=False,
            stop_category=('resource' if resource_stop else 'manual' if stop_reason=='ROOT_manual_cooperative_stop' else 'retention' if selection['stop_early_reason']=='consecutive_retention_gate_failures'
                else 'budget' if stop_reason=='window_budget_checkpoint_boundary' or selection['stop_early_reason']=='soft_max_wall_seconds'
                else selection['stop_early_reason']),
            next_action='Only actual budget CLOSED0 may continue the same fullstate; resource/manual/scientific stops require review')
        return 1 if resource_stop else 0
    finally:
        if child is not None and child.poll() is None:
            if not paths['stop'].exists():write_stop=paths['stop'].open('x',encoding='utf-8');write_stop.write('Controller exit: cooperative save/DEV\n');write_stop.close()
            try:child.wait(timeout=300)
            except subprocess.TimeoutExpired:
                child.terminate();child.wait(timeout=30)


def current_memory_admission(*,first_measurement=False,previous=None):
    sample=sample_memory(os.getpid(),'full_uniform_before_GPU_window');memory=sample['system_memory']
    historical_peak=12176424960
    if first_measurement:
        require(previous is None and memory['available'] is True and memory['commit_fraction']<.9
            and memory['commit_available_bytes']>=8000000000 and memory['physical_available_bytes']>=8000000000,
            'First changed-allocation measurement requires8GB actual commit and physical reserve below90%; not a fit claim')
        return {'sample':memory,'kind':'explicit_first_material_lazy_allocation_measurement_v1',
            'actual_minimum_commit_and_physical_reserve_bytes':8000000000,
            'predecessor_peak_private_bytes':historical_peak,'predecessor_peak_used_as_new_policy_fit_predictor':False,
            'lazy_helper_sha256':LAZY_ROWS_SHA,'worker_sha256':WORKER_SHA,
            'all_live90x3_95percent_2GB_cap8_guards_unchanged':True,'first_update_fit_measured_not_proven':True}
    peak=previous.get('peak_observed_owned_private_bytes') if previous else historical_peak
    require(type(peak) is int and peak>0 and memory['available'] is True
        and memory['commit_used_bytes']+peak<.9*memory['commit_limit_bytes']
        and memory['commit_limit_bytes']-memory['commit_used_bytes']-peak>2000000000,
        'Fresh baseline plus observed same-policy owned peak fails admission; no automatic resource retry')
    return {'sample':memory,'same_policy_peak_observed_private_bytes':peak,
        'same_policy_previous_owned_complete_window':previous.get('finished_at_utc') if previous else None,
        'projection_is_not_actual_fit_proof':True}


def exact_successor(previous,output,args):
    require(previous['status']=='train_window_ended' and previous['child_exit_code']==0
        and previous['resource_review_required'] is False and previous['stop_category']=='budget',
        'Only actual owned budget closure may auto-continue')
    run=HERE/'runs'/args.run_id;selection=read(run/'selection.json');history=read(run/'training_history.json')
    require(6500<selection['completed_steps']<38225 and 'validation' in history[-1]
        and selection['consecutive_retention_failures']<2,
        'Latest complete fixed350 DEV and carried failure counter below the approved two-stop threshold required')
    latest=scoped(selection['latest_weights']);saved=output/(args.run_id+'_closed_latest.pt')
    require(not output.exists(),'Fresh successor namespace');output.mkdir(parents=True)
    bound=record(latest);shutil.copy2(latest,saved);require(sha(saved)==bound['sha256'] and record(latest)==bound,'Owned closed latest archive changed')
    plan=dict(previous);argv=list(previous['worker_arguments'])
    paths={key:output/(args.run_id+'_'+name) for key,name in (('native','unused_original.json'),('fresh_runtime','fresh_native.json'),
        ('identity','identity.json'),('memory','memory.json'),('cache','cache.json'),('io','io.json'),('stop','stop.txt'),('log','worker.log'))}
    paths['native']=Path(previous['paths']['native'])
    for flag,value in (('--resume-exact',saved),('--uniform-full-runtime-proof',paths['fresh_runtime']),
        ('--stop-file',paths['stop']),('--max-wall-seconds',args.window_hours*3600-300)):
        argv=set_option(argv,flag,value)
    plan.update(kind=CONTROLLER_KIND,continuation_mode='same_new_run_exact_budget_continuation',
        original_ready=record(paths['native']),source_checkpoint=record(saved),resume_step=selection['completed_steps'],
        worker_arguments=argv,worker_arguments_sha256=domain.guards.argument_sha(argv),paths={key:str(value) for key,value in paths.items()},
        command=[str(ROOT/'work/nemo-native-env-v2/Scripts/python.exe'),str(Path(__file__).resolve()),'--worker-plan',str(output/'plan.json'),'--gpu-window-authorized'])
    for key in ('child_exit_code','resource_review_required','stop_category','cooperative_stop_reason','actual_runtime_completion','finished_at_utc'):
        plan.pop(key,None)
    validate_plan(plan);return plan


def main(args):
    if args.worker_plan:
        require(args.gpu_window_authorized,'ROOT exclusive GPU authorization required');run_worker(read(scoped(args.worker_plan)));return 0
    require(all((args.source_receipt,args.output_dir,args.run_id)),'ROOT source/newrun/output required')
    require(math.isfinite(args.max_hours) and args.window_hours<=args.max_hours<=96,'Finite overall fullrun time bound required')
    plan=prepare(args)
    if args.validate_only:
        if args.preflight_output:
            require(not scoped(args.preflight_output).exists() and not scoped(args.preflight_output).is_relative_to(scoped(args.output_dir)),'Fresh outside preflight evidence required')
            write(args.preflight_output,{**plan,'passed':True,'GPU_launched':False,'Torch_imported':'torch' in sys.modules})
        print(json.dumps({'passed':True,'GPU_launched':False,'configured_steps':38225,'new_uniform_updates':31725}));return 0
    require(args.gpu_window_authorized,'ROOT sole GPU authorized only after actualsource/PT baseline and resource repair')
    first=scoped(args.output_dir)
    initial_admission=current_memory_admission(first_measurement=args.measure_new_allocation_policy)
    first.mkdir(parents=True);overall=time.monotonic()+args.max_hours*3600;window=1;previous=None
    while True:
        admission=initial_admission if window==1 else current_memory_admission(previous=previous)
        loop=Path(plan['paths']['log']).parent
        state={**plan,'status':'starting','fresh_memory_admission':admission,'updated_at_utc':now()}
        deadline=min(overall,time.monotonic()+args.window_hours*3600)
        state['awake_request']=domain.guards.prior.keep_awake(True)
        try:
            write(loop/'plan.json',plan);write(loop/'loop_status.json',state)
            publish_queue(state)
            args.output_dir=loop;code=run_window(state,deadline,args)
        except BaseException as error:
            state.update(status='failed',error=repr(error),finished_at_utc=now());raise
        finally:
            state['awake_clear']=domain.guards.prior.keep_awake(False);write(loop/'loop_status.json',state)
            publish_queue(state)
        write(first/'full_run_status.json',{'status':state['status'],'current_closed_window':str(loop),
            'completed_steps':state.get('completed_steps'),'objective_complete':False,'fullcoverage_by_budget_alone':False})
        selection=read(HERE/'runs'/args.run_id/'selection.json')
        if code or state['stop_category']!='budget' or selection['completed_steps']>=38225:return code
        if overall-time.monotonic()<600:return 0
        window+=1;next_dir=first.with_name(first.name+'_window_'+str(window))
        previous=state
        plan=exact_successor(state,next_dir,args)


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker-plan',type=Path)
    p.add_argument('--source-receipt',type=Path);p.add_argument('--run-id');p.add_argument('--output-dir',type=Path)
    p.add_argument('--indexed-tr-directory',type=Path)
    p.add_argument('--portuguese-replay-manifest',type=Path,default=ROOT/'work/data/portuguese_retention_v1/replay_train_pt.jsonl')
    p.add_argument('--portuguese-dev-manifest',type=Path,default=ROOT/'work/data/portuguese_retention_v1/validation_pt.jsonl')
    p.add_argument('--portuguese-seed',type=int,default=20261012)
    p.add_argument('--window-hours',type=float,default=3.);p.add_argument('--max-hours',type=float,default=48.)
    p.add_argument('--poll-seconds',type=float,default=5.);p.add_argument('--validate-only',action='store_true')
    p.add_argument('--preflight-output',type=Path);p.add_argument('--gpu-window-authorized',action='store_true')
    p.add_argument('--measure-new-allocation-policy',action='store_true',help='ROOT-reviewed FIRST lazy-runtime measurement only: reserve8GB commit+physical; retain every live guard')
    return p


if __name__=='__main__':sys.exit(main(parser().parse_args()))
