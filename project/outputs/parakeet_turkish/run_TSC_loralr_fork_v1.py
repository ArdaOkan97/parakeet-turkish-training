"""Declared TSC LoRA-LR fork, with one owned guarded child and shared runtime APIs.

INACTIVE draft: closure/worker/helper pins remain PENDING; no launch admission.
The CLOSED parent BEST full state is restored directly. Only LoRA LR changes
from3e-4 to1e-4 at the next optimizer update; head LR remains1e-5.
No Torch/model/CUDA import occurs in controller/preflight mode. ROOT alone may
authorize the exclusive GPU child. Existing runtime sources are never edited.
"""
from __future__ import annotations
import argparse
import ast
from collections import deque
from datetime import datetime, timezone
import functools
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

from atomic_io_retry_v1 import read_json as read, write_json as write, commit_writer
from cuda_cache_reclaim_v2 import CudaCacheReclaimer as CacheOperations
import launch_with_memory_guard_TSC_resume_v3 as shared_memory
import run_hf_TSC_arm_v3 as domain
from resource_watchdog_v2 import ResourceGuard, sample_memory

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
WORKER='train_lora_batched_tsc_loralr_v1.py'
TRANSITION='tsc_loralr_transition_v1.py'
# These two pins are finalized only after the worker's focused source checks.
WORKER_SHA='PENDING'
TRANSITION_SHA='PENDING'
ROOT_RECEIPT_SHA='PENDING'
EXPECTED_SYNC_LINE=590
SHARED_MEMORY_SHA='61fec6d3ef911665bdbe876dc333f3c6a640badc012596fe92ded1b80a10c7fa'
PRELOAD_SHA='f8787b0f3346c9f526903ff0c06de4c266f940ca3b0f41c55907cc795e7cc2ba'
ADAPTER_SHA='839194dddf83c3b08bf608d01cb540b2f160feee426a91373d14cfc426ab7494'


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


def source_pins():
    # Includes unchanged data/teacher/guard sources plus the original caller APIs.
    chain=domain.chain_identity();pins=dict(chain['source_sha256'])
    receipt_path=HERE/'TSC_headlr02_01_actual_CLOSED_branch_receipt_v1.json'
    require(ROOT_RECEIPT_SHA!='PENDING' and sha(receipt_path)==ROOT_RECEIPT_SHA,
        'Actual ROOT-owned source closure receipt changed')
    config_record=read(receipt_path)['artifacts']['run_config']
    require(record(config_record['path'])=={k:config_record[k] for k in ('path','sha256')},'Closed parent config changed')
    for name,digest in read(config_record['path'])['config_identity']['source_sha256'].items():
        require(name not in pins or pins[name]==digest,'Parent versus shared implementation source differs')
        pins[name]=digest
    require(WORKER_SHA!='PENDING' and TRANSITION_SHA!='PENDING','New sources not frozen yet')
    pins.update({WORKER:WORKER_SHA,TRANSITION:TRANSITION_SHA,
        'run_hf_TSC_arm_v3.py':sha(domain.__file__),
        'launch_with_memory_guard_TSC_resume_v3.py':SHARED_MEMORY_SHA,
        Path(__file__).name:sha(__file__)})
    for name,digest in pins.items():require(sha(HERE/name)==digest,'Reviewed dependency changed: '+name)
    require(pins['preload_nemo_linker_v1.py']==PRELOAD_SHA
        and pins['native_tdt_checkpointed_v3.py']==ADAPTER_SHA,'Native dependency differs')
    return pins


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
    """Only caller admission changes; all cache operations are inherited unchanged."""
    def __init__(self,cuda,worker_path):
        self.cuda=cuda;self.worker_path=Path(worker_path).resolve();self.expected_sha=WORKER_SHA
        self.line=caller_line(self.worker_path);self.threshold_bytes=4_000_000_000;self.max_events=128
        self.events=deque(maxlen=128)
        self.eligible_calls=self.ignored_calls=self.reclamations=self.failed_reclamations=0
        self.cleanup_seconds=0.;self.original=self.wrapped=None


def validate_plan(plan):
    pins=source_pins();require(plan['source_sha256']==pins,'Frozen plan/runtime source map differs')
    argv=plan['worker_arguments'];rid=plan['run_id']
    require(option(argv,'--run-id')==rid and option(argv,'--head-lr')=='1e-05' and option(argv,'--lora-lr')=='0.0001'
        and option(argv,'--batch-size')=='8' and option(argv,'--accumulation')=='2',
        'Declared LoRA1e-4/head1e-5/B8x2 boundary required')
    exact=plan.get('continuation_mode')=='same_new_run_exact_budget_continuation'
    require(('--resume-exact' in argv)==exact and ('--resume-optimization' in argv)!=exact
        and '--resume-weights' not in argv and '--resume-runtime-migration' not in argv,
        'Choose declared initial fork or SAME new-run exact budget continuation only')
    require(record(option(argv,'--optimization-source-receipt'))==plan['source_closure'],
        'ROOT closed branch receipt changed')
    state_path=scoped(option(argv,'--resume-exact' if exact else '--resume-optimization'))
    if state_path.exists():require(record(state_path)==plan['source_checkpoint'],'Closed full state changed')
    else:require(exact and record(plan['closed_latest_source']['path'])==plan['closed_latest_source'],
        'Only a verified closed latest may await its new immutable archive')
    if exact:require(record(option(argv,'--native-backend-provenance'))==plan['original_ready'],
        'Original immutable ready must remain in scientific identity')
    require(domain.guards.argument_sha(argv)==plan['worker_arguments_sha256'],'Canonical worker argv changed')
    paths={k:scoped(v) for k,v in plan['paths'].items()}
    require(option(argv,'--native-backend-provenance')==str(paths['native']),
        'Native ready must precede canonical argv identity')
    require(len({p for k,p in paths.items() if k not in ('log','stop')})==len(paths)-2,
        'Distinct evidence paths required')
    caller_line(HERE/WORKER)
    return paths


def run_worker(plan):
    """New actual child: preload -> cap -> unchanged cache/IO APIs -> worker.run."""
    paths=validate_plan(plan);argv=plan['worker_arguments'];pins=plan['source_sha256']
    exact=plan.get('continuation_mode')=='same_new_run_exact_budget_continuation'
    require(scoped(option(argv,'--resume-exact' if exact else '--resume-optimization')).is_file(),
        'Actual dispatch requires the verified complete state artifact')
    for k in (('fresh_runtime','identity','memory','cache','io') if exact else ('native','identity','memory','cache','io')):
        require(not paths[k].exists(),'Existing runtime evidence is immutable: '+k)
    runtime_path=paths['fresh_runtime'] if exact else paths['native']
    live=Path(str(paths['identity'])+'.live_memory.json');completion=runtime_path.with_suffix('.completion.json')
    require(not live.exists() and not completion.exists(),'Fresh self/native evidence required')
    require('torch' not in sys.modules,'Actual preload must precede Torch import')
    from preload_nemo_linker_v1 import preload
    preloaded=preload()
    require(preloaded.get('kind')=='coherent_native_cuda12_9_preload_v1'
        and preloaded.get('actual_binding_linker',{}).get('version')==[12,9]
        and preloaded.get('cuda_initialized') is False,'Coherent pre-Torch native preload required')
    assets={flag[2:].replace('-','_'):record(option(argv,flag)) for flag in (
        '--tsc-input-proof','--tsc-training-approval','--source-closure','--tr-manifest','--replay-manifest',
        '--tr-frame-metadata','--tr-frame-metadata-proof','--data-v3-proof',
        '--resume-exact' if exact else '--resume-optimization','--optimization-source-receipt')}
    ready={'schema':1,'kind':'native_tdt_training_backend_v3','status':'ready','run_id':plan['run_id'],
        'tdt_implementation':'nemo','preload':preloaded,'pid':os.getpid(),'ppid':os.getppid(),
        'ready_at_utc':now(),'trainer_path':str(HERE/WORKER),'trainer_sha256':WORKER_SHA,
        'adapter_path':str(HERE/'native_tdt_checkpointed_v3.py'),'adapter_sha256':ADAPTER_SHA,
        'launcher_path':str(Path(__file__).resolve()),'launcher_sha256':sha(__file__),
        'memory_launcher_path':str(Path(__file__).resolve()),'memory_launcher_sha256':sha(__file__),
        'worker_arguments_sha256':plan['worker_arguments_sha256'],'TSC_explicit_asset_records':assets,
        'declared_allocator_policy':{'cap_gb':8,'end_cache_gb':4,'native_inactive_threshold_bytes':750000000},
        'declared_optimization_change':'LoRA LR3e-4 ->1e-4 at next update; head1e-5 and state/frontier preserved',
        'native_full_model_or_Lightning_trainer_equivalence_claimed':False}
    if exact:
        ready.update(kind='TSC_loralr_same_new_run_fresh_preload_v1',original_first_start_ready=plan['original_ready'],
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
    memory={'kind':'TSC_loralr_fork_memory_runtime_v1','status':'running',**sources,**cap,
        'cleanup_threshold_bytes':4_000_000_000,'declared_head_lr':1e-5,'declared_lora_lr':1e-4,'run_id':plan['run_id'],
        'backend_flags_changed':False,'allocator_scope':'Owned process PyTorch CUDA allocator only',
        'borrowed_memory_implementation':record(shared_memory.__file__),'started_at_utc':now()}
    cache={'kind':'TSC_loralr_fork_cache_runtime_v1','status':'running','worker_sha256':WORKER_SHA,
        'inherited_cache_implementation':record(HERE/'cuda_cache_reclaim_v2.py'),'started_at_utc':now()}
    io={'kind':'TSC_loralr_fork_atomic_IO_v1','status':'running','worker_sha256':WORKER_SHA,
        'original_serializer_preserved':True,'helper':record(HERE/'atomic_io_retry_v1.py'),'started_at_utc':now()}
    retry_events=deque(maxlen=128);worker=None;original_save=original_write=benchmark_write=None;outcome='failed';error=None
    def event(value):retry_events.append(value)
    atomic_write=functools.partial(write,retry_seconds=30.,event=event)
    for p,e in ((paths['memory'],memory),(paths['cache'],cache),(paths['io'],io)):write(p,e)
    try:
        torch.cuda.set_per_process_memory_fraction(cap['per_process_allocator_fraction'],device=0)
        memory['allocator_cap_applied']=True;write(paths['memory'],memory)
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
        native=sys.modules.get('native_tdt_checkpointed_v3')
        memory.update(status=outcome,error=error,finished_at_utc=now(),sources_unchanged=unchanged,
            owned_update_memory_telemetry=telemetry.evidence(),all_phase_owned_memory_telemetry=sampler.evidence(),
            native_cleanup_evidence=native.cache_cleanup_evidence() if native else {'available':False})
        memory['owned_update_memory_telemetry']['callback_scope']=f'Exact reviewed worker.run optimizer-end synchronize line{EXPECTED_SYNC_LINE}; observation only'
        for p,e in ((paths['memory'],memory),(paths['cache'],cache),(paths['io'],io)):write(p,e)
        write(completion,{'kind':'TSC_loralr_fork_native_completion_v1','status':outcome,'run_id':plan['run_id'],
            'pid':os.getpid(),'sources_unchanged':unchanged,'backend_ready_sha256':sha(paths['native']),
            'fresh_runtime_sha256':sha(runtime_path),'original_ready_reused_as_current_PID_evidence':False,
            'actual_dispatch_source_sha256':sha(__file__),'finished_at_utc':now(),'error':error})
        require(unchanged,'Source changed during the optimization fork')


def prepare(args):
    from types import SimpleNamespace
    if args.resume_closed_loop:return prepare_exact(args)
    pins=source_pins()
    require(math.isfinite(args.window_hours) and 0.1<=args.window_hours<=3.,'Bounded window <=3 hours required')
    require(5.<=args.poll_seconds<=15.,'Fresh guard polls must be5 to15 seconds apart')
    output=scoped(args.output_dir);source_loop=scoped(args.source_loop)
    require(output.parent==HERE/'runs' and not output.exists(),'A fresh controller namespace under runs is required')
    require(re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id) is not None,'Safe new run ID required')
    require(not (HERE/'runs'/args.run_id).exists(),'Worker namespace must be new')
    for suffix in ('best.pt','latest.pt'):
        require(not (ROOT/'work'/f'{args.run_id}_{suffix}').exists(),'New worker checkpoint path already exists')
    receipt_path=scoped(args.source_receipt);receipt=read(receipt_path,{})
    transition=importlib.import_module(Path(TRANSITION).stem)
    transition.validate_receipt_payload(receipt,args.run_id)
    require(sha(receipt_path)==transition.RECEIPT_SHA,'ROOT closure receipt changed')
    source=read(source_loop/'loop_status.json',{})
    require(record(source_loop/'loop_status.json')=={k:receipt['artifacts']['controller_status'][k] for k in ('path','sha256')}
        and source.get('status')=='train_window_ended' and source.get('child_exit_code')==0
        and source.get('resource_review_required') is False and source.get('stop_category')=='retention'
        and source.get('completed_steps')==receipt['completed_steps'] and source.get('selected_step')==receipt['selected_step'],
        'Pinned genuinely closed parent required; never reuse live/resource-failed state')
    boundary=receipt['selected_step']
    config=read(transition.checked_record(receipt['artifacts']['run_config']))
    argv=list(source['worker_arguments']);original_argv=list(argv)
    require(option(argv,'--run-id')==receipt['source_run_id'],'Closed source argv run differs')
    recipe=dict(config['config_identity']['training_arguments']);recipe['lora_lr']=1e-4
    require(recipe['steps']==44958 and recipe['eval_every_updates']==250,'Preserve original horizon and cadence')
    paths={key:output/f'{args.run_id}_{name}' for key,name in (
        ('native','native_backend.json'),('identity','identity.json'),('memory','memory.json'),
        ('cache','cache.json'),('io','io.json'),('stop','stop.txt'),('log','worker.log'))}
    for flag in ('--resume-exact','--resume-weights','--resume-runtime-migration'):argv=remove_option(argv,flag)
    best=scoped(receipt['artifacts']['best_checkpoint']['path'])
    for flag,value in (('--run-id',args.run_id),('--lora-lr',1e-4),('--resume-optimization',best),
        ('--optimization-source-receipt',receipt_path),('--native-backend-provenance',paths['native']),
        ('--stop-file',paths['stop']),('--max-wall-seconds',args.window_hours*3600-300)):
        argv=set_option(argv,flag,value)
    # Pure admission receives the complete original parser namespace, without importing the ML worker.
    namespace=dict(config)
    namespace.update(run_id=args.run_id,lora_lr=1e-4,resume_optimization=best,
        optimization_source_receipt=receipt_path,resume_exact=None,resume_weights=None,
        resume_runtime_migration=None,native_backend_provenance=paths['native'],
        stop_file=paths['stop'],max_wall_seconds=args.window_hours*3600-300)
    ctx=transition.preflight(SimpleNamespace(**namespace))
    python=scoped(source['command'][0]);require(python.is_file(),'Pinned native Python unavailable')
    command=[str(python),str(Path(__file__).resolve()),'--worker-plan',str(output/'plan.json'),'--gpu-window-authorized']
    inputproof=scoped(option(argv,'--tsc-input-proof'));approval=scoped(option(argv,'--tsc-training-approval'))
    plan={'schema':1,'kind':'declared_TSC_loralr_full_state_fork_controller_v1','run_id':args.run_id,
        'loop_id':output.name,'source_loop':str(source_loop),'controller_sha256':sha(__file__),
        'source_closure':record(receipt_path),'source_checkpoint':record(best),
        'source_run_config':record(receipt['artifacts']['run_config']['path']),
        'source_sha256':pins,'worker_arguments':argv,'worker_arguments_sha256':domain.guards.argument_sha(argv),
        'command':command,'paths':{key:str(value) for key,value in paths.items()},'window_hours':args.window_hours,
        'source_original_worker_arguments':original_argv,'actual_stage_recipe':recipe,
        'source_config_identity':ctx['parent_identity'],'inherited_step':boundary,'optimization_boundary':boundary,'first_changed_lr_step':boundary+1,
        'combined_TR_clips':source['combined_TR_clips'],'domain_pool_sizes':source['domain_pool_sizes'],
        'configured_steps':source['configured_steps'],'TSC_input_proof':record(inputproof),
        'TSC_training_approval':record(approval),'TSC_quarantine_proof':source['TSC_quarantine_proof'],
        'declared_scientific_change':{'head_lr_unchanged':1e-5,'lora_lr_from':3e-4,'lora_lr_to':1e-4,'boundary_step':boundary+1},
        'optimizer_sampler_RNG_history_reset':False,'BEST_latest_state_mixed':False,
        'parent_future_exposure_revisited_from_BEST':{'first_step':boundary+1,'last_step':receipt['completed_steps']},
        'initial_actual300_DEV_source_equality_before_updates_required':True,
        'source_checkpoint_tensor_state_independently_read_by_preflight':False,
        'heldout_TEST_evaluated':False,'objective_complete':False,'objective_status':'incomplete'}
    validate_plan(plan)
    return plan


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


def prepare_exact(args):
    """Admit only a closed scheduled window of this NEW scientific family."""
    require(math.isfinite(args.window_hours) and 0.1<=args.window_hours<=3.,'Bounded window <=3 hours required')
    require(5.<=args.poll_seconds<=15.,'Guard poll interval must remain5 to15 seconds')
    output=scoped(args.output_dir);source_loop=scoped(args.resume_closed_loop)
    require(output.parent==HERE/'runs' and not output.exists(),'Fresh exact controller namespace required')
    prior=read(source_loop/'loop_status.json',{});pins=source_pins()
    require(prior.get('kind')=='declared_TSC_loralr_full_state_fork_controller_v1'
        and prior.get('controller_sha256')==sha(__file__) and prior.get('source_sha256')==pins
        and prior.get('status')=='train_window_ended' and prior.get('child_exit_code')==0
        and prior.get('resource_review_required') is False and prior.get('stop_category')=='budget'
        and prior.get('run_id')==args.run_id,'Only actually closed same-family scheduled budget windows may continue')
    run=HERE/'runs'/args.run_id;config=read(run/'run_config.json');selection=read(run/'selection.json')
    require(selection.get('accepted') is True and selection.get('stop_early_reason') in ('soft_max_wall_seconds','controller_stop_file')
        and selection['completed_steps']==prior['completed_steps']<prior['configured_steps']
        and selection['selected_step']==prior['selected_step'] and record(run/'selection.json')==prior['selection'],
        'Never bypass retention/plateau/resource/horizon closure or lose qualifying BEST')
    validate_identity(read(prior['paths']['identity']),prior,prior['owned_launcher_pid'],prior['started_at_utc'])
    verify_completion(prior,selection)
    latest=scoped(selection['latest_weights']);best=scoped(selection['weights'])
    require(latest==ROOT/'work'/f'{args.run_id}_latest.pt' and best==ROOT/'work'/f'{args.run_id}_best.pt',
        'Keep existing SAME-run full latest and qualifying BEST')
    require(best.is_file(),'Existing full BEST required for frozen worker exact restore')
    plan={key:value for key,value in prior.items() if key in read(source_loop/'plan.json',{})}
    # Preserve original native-ready/proof and every scientific CLI field. Runtime evidence is fresh.
    paths={key:output/f'{args.run_id}_{name}' for key,name in (
        ('fresh_runtime','fresh_runtime.json'),('identity','identity.json'),('memory','memory.json'),
        ('cache','cache.json'),('io','io.json'),('stop','stop.txt'),('log','worker.log'))}
    original_ready=scoped(config['native_backend_provenance']);paths['native']=original_ready
    source_latest=record(latest);archive=output/'immutable_closed_latest.pt'
    argv=remove_option(prior['worker_arguments'],'--resume-optimization')
    for flag,value in (('--resume-exact',archive),('--stop-file',paths['stop']),('--max-wall-seconds',args.window_hours*3600-300)):
        argv=set_option(argv,flag,value)
    require(option(argv,'--native-backend-provenance')==str(original_ready), 'Original ready CLI must remain unchanged')
    history=read(run/'training_history.json')
    plan.update(continuation_mode='same_new_run_exact_budget_continuation',loop_id=output.name,
        source_loop=str(source_loop),source_sha256=pins,controller_sha256=sha(__file__),worker_arguments=argv,
        worker_arguments_sha256=domain.guards.argument_sha(argv),paths={k:str(v) for k,v in paths.items()},
        source_checkpoint={'path':str(archive),'sha256':source_latest['sha256']},closed_latest_source=source_latest,
        original_ready=record(original_ready),inherited_step=selection['completed_steps'],
        inherited_history_sha256=canonical(history),prior_closed_controller=record(source_loop/'loop_status.json'),
        preserved_scientific_config_identity_sha256=canonical(config['config_identity']),window_hours=args.window_hours,
        command=[prior['command'][0],str(Path(__file__).resolve()),'--worker-plan',str(output/'plan.json'),'--gpu-window-authorized'])
    validate_plan(plan)
    return plan


def verify_completion(plan,selection):
    paths={key:Path(value) for key,value in plan['paths'].items()};run=HERE/'runs'/plan['run_id']
    require(selection.get('run_id')==plan['run_id'] and selection.get('heldout_tests_evaluated') is False,
        'Durable DEV-only new-namespace selection required')
    completed=selection['completed_steps'];require(completed>=plan['optimization_boundary'],'Fork cannot reset cumulative updates')
    exact=plan.get('continuation_mode')=='same_new_run_exact_budget_continuation'
    runtime_path=paths['fresh_runtime'] if exact else paths['native']
    identity=read(paths['identity'],{});completion=read(runtime_path.with_suffix('.completion.json'),{})
    require(completion.get('kind')=='TSC_loralr_fork_native_completion_v1' and completion.get('status')=='complete'
        and completion.get('pid')==identity['pid'] and completion.get('run_id')==plan['run_id']
        and completion.get('sources_unchanged') is True and completion.get('backend_ready_sha256')==sha(paths['native'])
        and completion.get('fresh_runtime_sha256')==sha(runtime_path)
        and completion.get('actual_dispatch_source_sha256')==sha(__file__),'Actual fresh native completion required')
    for key,kind in (('io','TSC_loralr_fork_atomic_IO_v1'),('cache','TSC_loralr_fork_cache_runtime_v1'),
        ('memory','TSC_loralr_fork_memory_runtime_v1')):
        value=read(paths[key],{});require(value.get('kind')==kind and value.get('status')=='complete'
            and value.get('sources_unchanged') is True,'Actual new runtime completion missing: '+key)
    memory=read(paths['memory']);cache=read(paths['cache'])
    require(memory.get('cap_bytes')==8_000_000_000 and memory.get('cleanup_threshold_bytes')==4_000_000_000
        and memory.get('allocator_cap_applied') is True
        and memory.get('all_phase_owned_memory_telemetry',{}).get('thread_join_completed') is True,
        'Actual cap8/end4/all-phase telemetry completion required')
    require(cache['cache_reclamation']['eligible_update_calls']==completed-plan['inherited_step'],
        'Actual optimizer hook must cover only newly trained suffix, not inherited history')
    config=read(run/'run_config.json');history=read(run/'training_history.json')
    require(config['config_identity']['training_arguments']==plan['actual_stage_recipe'],'Fork recipe differs')
    transition=importlib.import_module(Path(TRANSITION).stem);proof=read(run/transition.PROOF_NAME,{})
    require(proof.get('kind')=='TSC_loralr_before_update_source_equivalence_v1' and proof.get('passed') is True
        and proof.get('run_id')==plan['run_id'] and proof.get('compared_fixed_DEV_utterances')==300
        and proof.get('RNG_restored_after_added_evaluation') is True
        and {k:proof['source_receipt'][k] for k in ('path','sha256')}==plan['source_closure'],
        'Actual unsuppressed300 source equivalence before updates required')
    require(canonical(history[:plan['optimization_boundary']])==proof['state_boundary']['history_prefix_sha256'],
        'Inherited selected BEST history prefix changed or later failed rows mixed')
    if exact:
        require(canonical(history[:plan['inherited_step']])==plan['inherited_history_sha256']
            and canonical(config['config_identity'])==plan['preserved_scientific_config_identity_sha256'],
            'Exact new-family history/scientific identity changed')
        fresh=read(runtime_path,{})
        require(fresh.get('kind')=='TSC_loralr_same_new_run_fresh_preload_v1' and fresh.get('status')=='ready'
            and fresh.get('pid')==identity['pid'] and fresh.get('original_first_start_ready')==plan['original_ready']
            and fresh.get('worker_arguments_sha256')==plan['worker_arguments_sha256']
            and fresh.get('scientific_config_identity_changed') is False,'Actual fresh exact preload/PID proof differs')
    change=config['config_identity']['optimization_transition']
    require(change.get('kind')=='declared_TSC_LoRA_LR_full_state_fork_v1'
        and change.get('first_changed_lr_step')==plan['optimization_boundary']+1
        and change.get('lora_lr_to')==1e-4 and change.get('head_lr_unchanged')==1e-5
        and change.get('parent_optimization_transition')==plan['source_config_identity']['optimization_transition']
        and change.get('fresh_optimizer_sampler_or_warmup') is False,'Explicit full-state LR transition identity missing')
    coverage=domain.verify_pool_frontier(config,selection,history,plan)
    require(record(config['tsc_input_proof'])==plan['TSC_input_proof']
        and record(config['tsc_training_approval'])==plan['TSC_training_approval'], 'Original data/approval changed')
    source_pins()
    return {'actual_pool_frontier':coverage,'actual300_DEV_source_equality':record(run/transition.PROOF_NAME),
        'newly_trained_updates':completed-plan['inherited_step'],'inherited_steps':plan['inherited_step'],
        'run_config':record(run/'run_config.json'),'native_completion':record(runtime_path.with_suffix('.completion.json'))}


def run_window(state,deadline,args):
    paths={key:Path(value) for key,value in state['paths'].items()};loop=scoped(args.output_dir)
    child=None;stop_at=None;stop_reason=None;started=time.monotonic();started_utc=now();last_phase=None;last_dev=state['inherited_step']
    resource=ResourceGuard();progress=domain.guards.ProgressGuard(started)
    try:
        with paths['log'].open('x',encoding='utf-8') as log:
            child=subprocess.Popen(state['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            state.update(status='running',owned_launcher_pid=child.pid,started_at_utc=started_utc)
            print(f"ACTUAL owned launch {state['run_id']} PID{child.pid}; BEST{state['optimization_boundary']} full state, declared LoRALR1e-4/head1e-5",flush=True)
            while child.poll() is None:
                history=read(HERE/'runs'/state['run_id']/'training_history.json',[])
                identity=validate_identity(read(paths['identity']),state,child.pid,started_utc) if paths['identity'].exists() else None
                phase=domain.guards.log_phase(paths['log'],history)
                sample=sample_memory(identity['pid'] if identity else child.pid,phase['phase'])
                reason,observation=resource.observe(sample['system_memory'],sample['worker_private_memory'],observed_monotonic=sample['observed_monotonic'])
                self_reason,self_observation=domain.guards.self_memory(Path(str(paths['identity'])+'.live_memory.json'),identity,paths['identity']) if identity else (None,{'available':False})
                progress_reason,progress_info=progress.observe(history,time.monotonic());reason=reason or self_reason or progress_reason
                if not identity and time.monotonic()-started>120:reason=reason or 'owned_worker_identity_unavailable'
                if identity and not self_observation['available'] and time.monotonic()-started>120:reason=reason or 'actual_worker_self_memory_unavailable_or_stale'
                if not history and time.monotonic()-started>1800:reason=reason or 'bounded_preparation_timeout'
                if time.monotonic()>=deadline-300:reason=reason or 'window_budget_checkpoint_boundary'
                state.update(phase=phase,published_step=history[-1]['step'] if history else state['inherited_step'],
                    system_and_owned_private_memory=observation,actual_worker_self_memory=self_observation,
                    progress=progress_info,updated_at_utc=now())
                if phase['phase']!=last_phase:print(state['run_id']+' phase: '+phase['phase'],flush=True);last_phase=phase['phase']
                for row in history:
                    if row['step']>last_dev and 'validation' in row:
                        gate=row['validation'];print(f"{state['run_id']} DEV{row['step']}: TR{100*gate['turkish_wer']:.4f}%, all5={gate['retention_gate_passed']}",flush=True);last_dev=row['step']
                with (loop/'memory_observations.jsonl').open('a',encoding='utf-8') as out:
                    out.write(json.dumps({'at_utc':now(),'phase':phase,'memory':observation,'worker_self':self_observation,'progress':progress_info})+'\n')
                if reason and stop_at is None:
                    stop_at=time.monotonic();stop_reason=reason
                    with paths['stop'].open('x',encoding='utf-8') as out:out.write('Cooperative save/DEV requested: '+reason+'\n')
                    state.update(cooperative_stop_reason=reason,cooperative_stop_requested_at_utc=now())
                write(loop/'loop_status.json',state)
                if stop_at and time.monotonic()-stop_at>300:raise RuntimeError('Owned child exceeded cooperative grace; preserve and review')
                time.sleep(args.poll_seconds)
        state['child_exit_code']=child.returncode
        require(child.returncode==0,'Actual child failed; no automatic GPU retry')
        selection_path=HERE/'runs'/state['run_id']/'selection.json';selection=read(selection_path,{})
        state['actual_runtime_completion']=verify_completion(state,selection)
        resource_stop=bool(stop_reason and stop_reason!='window_budget_checkpoint_boundary')
        state.update(status='resource_review_required' if resource_stop else 'train_window_ended',
            selected_step=selection['selected_step'],completed_steps=selection['completed_steps'],
            selection=record(selection_path),stop_early_reason=selection['stop_early_reason'],
            resource_review_required=resource_stop,finished_at_utc=now(),objective_complete=False,
            stop_category=('resource' if resource_stop else 'retention' if selection['stop_early_reason']=='consecutive_retention_gate_failures'
                else 'budget' if stop_reason=='window_budget_checkpoint_boundary' or selection['stop_early_reason']=='soft_max_wall_seconds'
                else selection['stop_early_reason']),
            next_action='ROOT reviews actual DEV and closed full state; no automatic fresh optimizer or next GPU job')
        return 1 if resource_stop else 0
    finally:
        if child is not None and child.poll() is None:
            if not paths['stop'].exists():write_stop=paths['stop'].open('x',encoding='utf-8');write_stop.write('Controller exit: cooperative save/DEV\n');write_stop.close()
            try:child.wait(timeout=300)
            except subprocess.TimeoutExpired:
                child.terminate();child.wait(timeout=30)


def main(args):
    if args.worker_plan:
        require(args.gpu_window_authorized,'ROOT exclusive GPU authorization required for child dispatch')
        run_worker(read(scoped(args.worker_plan)));return 0
    require(all((args.output_dir,args.run_id)) and (args.resume_closed_loop or all((args.source_loop,args.source_receipt))),
        'Controller source/run/output arguments required')
    plan=prepare(args)
    if args.validate_only:
        if args.preflight_output:
            output=scoped(args.preflight_output);require(not output.exists(),'Fresh preflight output required')
            require(not output.is_relative_to(scoped(args.output_dir)),'Keep future controller namespace absent')
            write(output,{**plan,'passed':True,'Torch_imported':'torch' in sys.modules,'GPU_launched':False})
        print(json.dumps({'passed':True,'run_id':plan['run_id'],'GPU_launched':False,'Torch_imported':'torch' in sys.modules}));return 0
    require(args.gpu_window_authorized,'ROOT must confirm exclusive GPU ownership; no unrelated process claim')
    loop=scoped(args.output_dir);loop.mkdir(parents=True,exist_ok=False)
    state={**plan,'status':'starting','absolute_deadline_utc':datetime.fromtimestamp(time.time()+args.window_hours*3600,timezone.utc).isoformat()}
    deadline=time.monotonic()+args.window_hours*3600;state['awake_request']=domain.guards.prior.keep_awake(True)
    try:
        if plan.get('closed_latest_source'):
            require(record(plan['closed_latest_source']['path'])==plan['closed_latest_source'],'Closed latest changed before archiving')
            target=Path(plan['source_checkpoint']['path']);require(not target.exists(),'Never overwrite immutable exact input')
            shutil.copyfile(plan['closed_latest_source']['path'],target)
            require(record(target)==plan['source_checkpoint'] and record(plan['closed_latest_source']['path'])==plan['closed_latest_source'],
                'Exact archive must remain byte-identical to closed latest')
        write(loop/'plan.json',plan);write(loop/'loop_status.json',state)
        return run_window(state,deadline,args)
    except BaseException as error:
        state.update(status='failed',error=repr(error),finished_at_utc=now());raise
    finally:
        state['awake_clear']=domain.guards.prior.keep_awake(False);write(loop/'loop_status.json',state)


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--worker-plan',type=Path)
    result.add_argument('--resume-closed-loop',type=Path)
    result.add_argument('--source-loop',type=Path);result.add_argument('--source-receipt',type=Path)
    result.add_argument('--run-id');result.add_argument('--output-dir',type=Path)
    result.add_argument('--window-hours',type=float,default=3.);result.add_argument('--poll-seconds',type=float,default=5.)
    result.add_argument('--validate-only',action='store_true');result.add_argument('--preflight-output',type=Path)
    result.add_argument('--gpu-window-authorized',action='store_true')
    return result


if __name__=='__main__':sys.exit(main(parser().parse_args()))

