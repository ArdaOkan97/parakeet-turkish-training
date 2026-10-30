"""ROOT-authorized, bounded budget-only exact continuation of an approved TSC combined-pool arm.

The initial FIRST or EXACT cross-job controller is bound through its live owned-worker metadata
and pinned child-close/completion/publication path, without OS handle attachment.
Every successor is this supervisor's Popen child and must actually exit0 before
another may start. Original native-ready, source closure, science and full state
remain unchanged. Resource/retention/plateau/manual stops require ROOT review.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from atomic_io_retry_v1 import read_json as read,write_json as write
import run_hf_TSC_arm_v3 as domain
import resume_TSC_same_science_v3 as exact
import launch_TSC_same_science_resume_v3 as preload

HERE,ROOT=domain.HERE,domain.ROOT
FIRST=HERE/'run_hf_TSC_arm_v3.py'
FIRST_SHA = '7eccccd31c294e04bc962eed7cc3d0e385768386607b491064cbe21955cfc24b'
HELPER=HERE/'resume_TSC_same_science_v3.py'
HELPER_SHA = 'd9794733dc943e62c043572ee817f9feaee1c98877a1881b4aa5850a08d19581'
PRELOADER=HERE/'launch_TSC_same_science_resume_v3.py'
PRELOADER_SHA = 'c011e3f412f016419005f2513d35f7d4b100f0e461063914cfdc6088d0d5d522'
FIRST_KIND='isolated_HF_TSC_guarded_quality_arm_v3'
EXACT_KIND='TSC_same_run_exact_science_continuation_v3'
# Dynamic horizon is pinned by actual approved combined-pool config, never copied from another arm.


def require(value,message):
    if not value:raise ValueError(message)
def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def scoped(path):return domain.scoped(path)
def record(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def canonical_sha(value):return exact.canonical_sha(value)


def frozen_sources():
    require(sha(FIRST)==FIRST_SHA and sha(HELPER)==HELPER_SHA and sha(PRELOADER)==PRELOADER_SHA,
        'Frozen first-TSC controller/exact helper/fresh preloader changed')
    chain=domain.chain_identity()
    require(preload.PINS==chain['source_sha256'],'Exact runtime must use the unchanged original v3 domain-preserving TSC science chain')
    require(sha(HERE/preload.RUNTIME_MEMORY_NAME)==preload.RUNTIME_MEMORY_SHA,'Exact-only memory admission source changed')
    return chain['source_sha256']


def controller_order_proof():
    """Source-bound ownership/order evidence, not an independent OS exit query."""
    frozen_sources();proof={}
    for role,path,function in (('first',FIRST,'run_arm'),('exact',HELPER,'main')):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        run=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==function)
        calls=[node for node in ast.walk(run) if isinstance(node,ast.Call)]
        spawns=[node for node in calls if ast.unparse(node.func)=='subprocess.Popen']
        require(len(spawns)==1,'Pinned controller must own only one GPU child: '+role)
        loop=next(node for node in ast.walk(run) if isinstance(node,ast.While)
            and ast.unparse(node.test)=='child.poll() is None')
        close=next(node for node in calls if ast.unparse(node.func)=='require' and node.args
            and ast.unparse(node.args[0])=='child.returncode == 0')
        completion=next(node for node in calls if ast.unparse(node.func)=='verify_completion')
        terminal=next(node for node in calls if ast.unparse(node.func)=='state.update'
            and any(key.arg=='status' for key in node.keywords)
            and any(key.arg=='selected_step' for key in node.keywords))
        require(spawns[0].lineno<loop.lineno<close.lineno<completion.lineno<terminal.lineno,
            'Owned child close/completion/terminal assignment ordering changed: '+role)
        require(not any(node.lineno>terminal.lineno for node in spawns),'No GPU spawn after terminal state')
        main=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
        final_publish=[node for outer in ast.walk(main) if isinstance(outer,ast.Try)
            for final in outer.finalbody for node in ast.walk(final)
            if isinstance(node,ast.Call) and ast.unparse(node.func)=='write']
        require(final_publish and all(node.lineno>terminal.lineno for node in final_publish),
            'Atomic terminal publication after run close is missing: '+role)
        if role=='first':
            require(any(isinstance(node,ast.Return) and isinstance(node.value,ast.Call)
                and ast.unparse(node.value.func)=='run_arm' for node in ast.walk(main)),
                'First main must wait for its owned run_arm before final publication')
        proof[role]={'controller':record(path),'owned_child_spawn':spawns[0].lineno,
            'owned_child_poll':loop.lineno,'actual_returncode_zero':close.lineno,
            'verified_runtime_completion':completion.lineno,'terminal_assignment':terminal.lineno,
            'atomic_final_publish':[node.lineno for node in final_publish]}
    return {'kind':'pinned_TSC_controller_child_close_publication_source_proof_v3',
        'controllers':proof,'initial_independent_OS_exit_observation':False,
        'future_GPU_spawn_after_terminal':False}


def source_state(folder,run_id):
    pins=frozen_sources();state=read(folder/'loop_status.json',{})
    require(state.get('kind') in (FIRST_KIND,EXACT_KIND) and state.get('run_id')==run_id,
        'Only the original TSC arm or its SAME-run exact windows may be supervised')
    expected=FIRST_SHA if state['kind']==FIRST_KIND else HELPER_SHA
    require(state.get('controller_sha256')==expected and state.get('source_sha256')==pins,
        'Pinned actual controller/runtime identity differs')
    if state['kind']==EXACT_KIND:
        require(state.get('new_preloader',{}).get('sha256')==PRELOADER_SHA
            and state.get('scientific_arguments_unchanged') is True
            and state.get('optimizer_sampler_RNG_counters_reset') is False,
            'Exact full-state science preservation differs')
    config=read(HERE/'runs'/run_id/'run_config.json',{})
    identity=exact.validate_config(config,state['worker_arguments'])
    require(scoped(config['native_backend_provenance'])==scoped(preload.option(state['worker_arguments'],'--native-backend-provenance')),
        'Original native-ready config/argument path differs')
    counts=domain.domain_counts(config)
    require(config.get('steps')==domain.domain_horizon(counts['fleurs'],counts['cv_tsc'])
        and identity['training_arguments']['epochs']==1.,
        'Preserve actual one minimum-domain exposure horizon and fullstate frontier')
    if state['kind']==EXACT_KIND:
        require(identity==state['source_config_identity'],'Previous exact window changed scientific identity')
    return state,identity


def worker_proof(state,run_id,expected_launcher=None,expected_worker=None):
    identity_path=scoped(state['paths']['identity']);identity=read(identity_path,{})
    launcher=state.get('owned_launcher_pid')
    require(type(launcher) is int and launcher>0,'Owned launcher PID missing')
    domain.validate_identity(identity,state,launcher,state['started_at_utc'])
    pid=identity['pid']
    require(expected_launcher is None or launcher==expected_launcher,'ROOT declared launcher differs')
    require(expected_worker is None or pid==expected_worker,'ROOT declared worker differs')
    creation=identity.get('process_creation_identity',{})
    require(creation.get('creation_filetime_100ns',0)>0,'Owned self-reported process creation identity missing')
    argument_sha=domain.guards.argument_sha(state['worker_arguments'])
    if state['kind']==FIRST_KIND:
        ready_path=scoped(state['paths']['native']);ready=read(ready_path,{})
        require(scoped(preload.option(state['worker_arguments'],'--native-backend-provenance'))==ready_path
            and ready.get('kind')=='native_tdt_training_backend_v3' and ready.get('status')=='ready'
            and ready.get('run_id')==run_id and ready.get('pid')==pid and ready.get('ppid')==identity['ppid']
            and ready.get('worker_arguments_sha256')==argument_sha
            and ready.get('trainer_sha256')==preload.PINS['train_lora_batched_tsc_v3.py']
            and ready.get('adapter_sha256')==preload.PINS['native_tdt_checkpointed_v3.py']
            and ready.get('launcher_sha256')==preload.PINS['launch_native_training_tsc_v3.py']
            and ready.get('memory_launcher_sha256')==preload.PINS['launch_with_memory_guard_tsc_v3.py']
            and ready.get('declared_allocator_policy')=={'cap_gb':8,'end_cache_gb':4,'native_inactive_threshold_bytes':750000000},
            'Actual first-TSC native-ready/PID/argv/source policy differs')
        first_preload=ready.get('preload',{})
        require(first_preload.get('kind')=='coherent_native_cuda12_9_preload_v1'
            and first_preload.get('actual_binding_linker',{}).get('version')==[12,9]
            and first_preload.get('cuda_initialized') is False
            and first_preload.get('module_sha256')==preload.PINS['preload_nemo_linker_v1.py'],
            'First-TSC actual coherent pre-Torch native preload differs')
        original_ready=record(ready_path)
    else:
        ready_path=scoped(state['paths']['fresh_runtime']);ready=read(ready_path,{})
        exact.validate_fresh_runtime(ready,state,identity)
        require(ready.get('ppid')==identity['ppid'],'Fresh exact ready/self parent identity differs')
        original_ready=state['original_ready']
        require(record(scoped(original_ready['path']))==original_ready,'Original immutable TSC-ready bytes changed')
    closure_path=scoped(preload.option(state['worker_arguments'],'--source-closure'))
    return {'worker_identity':record(identity_path),'current_runtime_proof':record(ready_path),
        'worker_pid':pid,'launcher_pid':launcher,'worker_ppid':identity['ppid'],
        'original_TSC_ready':original_ready,'original_source_closure':record(closure_path),
        'worker_creation_identity_scope':'Worker self-report validated by pinned parent; no supervisor-independent process query'}


def bootstrap(folder,run_id,launcher_pid,worker_pid):
    state,science=source_state(folder,run_id)
    require(state.get('status')=='running','Arm supervisor only while the pinned FIRST/EXACT TSC controller is live')
    require(state['kind'] in (FIRST_KIND,EXACT_KIND),'Initial window must be a pinned FIRST or SAME-run EXACT TSC controller')
    timestamp=domain.guards.stamp(state['updated_at_utc'])
    age=(datetime.now(timezone.utc)-timestamp).total_seconds()
    require(-5<=age<=30,'Initial actual controller progress observation is stale')
    worker=worker_proof(state,run_id,launcher_pid,worker_pid)
    return {'kind':'pinned_live_TSC_controller_metadata_bootstrap_v4','run_id':run_id,'source_loop':str(folder),
        'source_state':record(folder/'loop_status.json'),'controller_updated_at_utc':state['updated_at_utc'],
        'controller_observation_age_seconds':age,'initial_controller_kind':state['kind'],
        'initial_controller_sha256':state['controller_sha256'],**worker,'scientific_config_identity_sha256':canonical_sha(science),'configured_steps':state['configured_steps'],
        'controller_close_publish_order':controller_order_proof(),'initial_independent_OS_exit_observation':False,
        'no_process_enumeration_or_permission_bypass':True,'observed_at_utc':now()}


def automatic_budget_handoff(selection,state):
    reason=selection.get('stop_early_reason');helper_reason=state.get('cooperative_stop_reason')
    return (state.get('resource_review_required') is False
        and state.get('stop_category')=='budget'
        and helper_reason in (None,'window_budget_checkpoint_boundary')
        and (reason=='soft_max_wall_seconds'
            or (reason=='controller_stop_file' and helper_reason=='window_budget_checkpoint_boundary')))


def closure(folder,run_id,initial_bootstrap,owned_child=None):
    owned_exit=None
    if owned_child is not None:
        require(owned_child.poll() is not None,'Supervisor-owned controller remains live')
        owned_exit=owned_child.wait(timeout=30)
    state,science=source_state(folder,run_id)
    expected_status='quality_window_ended' if state['kind']==FIRST_KIND else 'train_window_ended'
    require(state.get('status') in (expected_status,'resource_review_required')
        and state.get('child_exit_code')==0,'Pinned controller must have closed/saved its worker successfully')
    require(canonical_sha(science)==initial_bootstrap['scientific_config_identity_sha256'],
        'Original TSC science/seeds/pool/native-ready/source-closure identity changed')
    worker=worker_proof(state,run_id)
    require(worker['original_TSC_ready']==initial_bootstrap['original_TSC_ready']
        and worker['original_source_closure']==initial_bootstrap['original_source_closure'],
        'Immutable original TSC-ready/source-closure identity changed')
    if owned_child is None:
        require(state['kind']==initial_bootstrap['initial_controller_kind']
            and state['controller_sha256']==initial_bootstrap['initial_controller_sha256']
            and worker['worker_identity']==initial_bootstrap['worker_identity']
            and worker['current_runtime_proof']==initial_bootstrap['current_runtime_proof'],
            'Initial cross-job metadata ownership changed before closure')
    else:
        require(state['kind']==EXACT_KIND and owned_exit in (0,1),
            'Supervisor-owned exact controller failed instead of publishing a verified closed window')
        require(owned_exit==(1 if state.get('resource_review_required') else 0),
            'Actual owned controller exit differs from saved resource status')
    selection_path=HERE/'runs'/run_id/'selection.json';selection=read(selection_path,{})
    require(state.get('selection')==record(selection_path) and selection.get('run_id')==run_id
        and selection.get('completed_steps')==state.get('completed_steps')
        and selection.get('selected_step')==state.get('selected_step') and selection.get('accepted') is True
        and selection.get('heldout_tests_evaluated') is False,
        'Actual saved full-state/qualifying DEV frontier differs')
    if state['kind']==FIRST_KIND:runtime=domain.verify_completion(state,selection)
    else:runtime=exact.verify_completion(state,selection,read(scoped(state['paths']['identity']),{}))
    checkpoints={role:record(scoped(selection[key])) for role,key in (('full_latest','latest_weights'),('qualified_best','weights'))}
    budget=automatic_budget_handoff(selection,state)
    require(not budget or owned_exit in (None,0),'No successor after an actual owned controller failure')
    return {'kind':'pinned_TSC_controller_evidence_verified_window_closure_v4','run_id':run_id,'source_loop':str(folder),
        'closure_observed_from_pinned_controller_evidence':True,'supervisor_independent_initial_OS_exit_observation':False,
        'actual_supervisor_owned_controller_exit_code':owned_exit,'supervisor_owned_controller_pid':owned_child.pid if owned_child else None,
        'helper_final_state':record(folder/'loop_status.json'),'selection':record(selection_path),
        'saved_checkpoint_bytes':checkpoints,'verified_runtime_completion':runtime,**worker,
        'scientific_config_identity_sha256':canonical_sha(science),'configured_steps':state['configured_steps'],'completed_steps':selection['completed_steps'],
        'selected_step':selection['selected_step'],'minimum_domain_exposure_epochs_completed':selection['turkish_epochs_completed'],
        'selection_stop_reason':selection['stop_early_reason'],'helper_stop_reason':state.get('cooperative_stop_reason'),
        'resource_review_required':state.get('resource_review_required') is True,'stop_category':state.get('stop_category'),
        'automatic_budget_handoff_allowed':budget,'objective_complete':False,'verified_at_utc':now()}


def successor_command(folder,target,run_id,hours,poll_seconds):
    frozen_sources()
    require(math.isfinite(hours) and .25<=hours<=3 and not target.exists(),'New bounded exact-window namespace required')
    return [str(ROOT/'work/nemo-native-env-v2/Scripts/python.exe'),str(HELPER),'--source-loop',str(folder),
        '--run-id',run_id,'--output-dir',str(target),'--window-hours',str(hours),'--poll-seconds',str(poll_seconds),
        '--source-closed-confirmed','--gpu-window-authorized']


def main(args):
    folder,evidence=scoped(args.source_loop),scoped(args.supervisor_dir)
    require(evidence!=folder and not evidence.exists(),'NEW supervisor namespace required')
    require(math.isfinite(args.max_hours) and .25<=args.max_hours<=12
        and math.isfinite(args.window_hours) and .25<=args.window_hours<=3
        and math.isfinite(args.poll_seconds) and 5<=args.poll_seconds<=15,'Bounded12h supervisor/3h window/fresh5-15s polls required')
    require(re.fullmatch(r'[A-Za-z0-9_-]+',args.successor_prefix),'Safe successor evidence prefix required')
    require(type(args.launcher_pid) is int and type(args.worker_pid) is int and min(args.launcher_pid,args.worker_pid)>0,
        'ROOT must supply actual owned launcher/worker PIDs')
    initial=bootstrap(folder,args.run_id,args.launcher_pid,args.worker_pid)
    evidence.mkdir(parents=True,exist_ok=False);write(evidence/'initial_metadata_bootstrap.json',initial)
    state={'kind':'automatic_TSC_same_science_window_supervisor_v4','status':'metadata_bootstrap_verified',
        'run_id':args.run_id,'supervisor_pid':os.getpid(),'supervisor_source_sha256':sha(__file__),
        'first_controller_sha256':FIRST_SHA,'helper_sha256':HELPER_SHA,'preloader_sha256':PRELOADER_SHA,
        'initial_closure_scope':'Pinned initially observed FIRST/EXACT controller owned-child close/completion publication; no independent cross-job OS exit query',
        'initial_controller_kind':initial['initial_controller_kind'],'initial_controller_sha256':initial['initial_controller_sha256'],
        'configured_optimizer_horizon_unchanged':initial['configured_steps'],'minimum_domain_exposure_epoch_horizon':1,
        'scientific_config_identity_sha256':initial['scientific_config_identity_sha256'],
        'window_hours':args.window_hours,'max_hours':args.max_hours,'source_loop':str(folder),'windows':[],
        'objective_complete':False,'objective_status':'incomplete','started_at_utc':now(),
        'next_action':'Keep exact TSC state through scheduled windows; ROOT reviews quality/internal/resource/horizon stops'}
    write(evidence/'supervisor_status.json',state)
    if args.bootstrap_only:
        print(json.dumps({'passed':True,'metadata_bootstrap_verified':True,'GPU_launch':False,
            'independent_OS_attachment':False,'receipt':str(evidence/'initial_metadata_bootstrap.json')}));return 0
    require(args.gpu_window_authorized,'ROOT must authorize exclusive automatic budget-only GPU successors')
    deadline=time.monotonic()+args.max_hours*3600;child=None;index=2
    try:
        while True:
            supervisor_budget_stop=False;budget_request_at=None
            while True:
                pending=read(folder/'loop_status.json',{})
                if child is not None and child.poll() is not None:break
                if child is None and pending.get('status')!='running':break
                state.update(status='waiting_pinned_owned_controller_closure',current_source_loop=str(folder),updated_at_utc=now())
                write(evidence/'supervisor_status.json',state)
                if time.monotonic()>=deadline:
                    if child is None:
                        state.update(status='bounded_supervisor_horizon_current_initial_window',next_experiment_required=True,
                            objective_complete=False,next_action='ROOT reviews current independently finite initial controller; no successor launched',finished_at_utc=now())
                        write(evidence/'supervisor_status.json',state);return 0
                    if budget_request_at is None:
                        # This affects only an owned child's existing cooperative stop file.
                        stop=scoped(pending['paths']['stop'])
                        if not stop.exists():
                            with stop.open('x',encoding='utf-8') as stream:stream.write('Supervisor bounded wall horizon: cooperative full checkpoint/DEV requested\n')
                        budget_request_at=time.monotonic();supervisor_budget_stop=True
                        state.update(supervisor_budget_stop_requested_at_utc=now(),owned_controller_pending_exit=child.pid)
                    require(time.monotonic()-budget_request_at<=600,'Owned controller did not close within bounded supervisor grace; ROOT review, no restart')
                if pending.get('status')=='running':
                    stamp=domain.guards.stamp(pending['updated_at_utc'])
                    require((datetime.now(timezone.utc)-stamp).total_seconds()<=120,
                        'Initial/owned controller publication stale; no inferred process exit or retry')
                time.sleep(args.poll_seconds)
            closed=closure(folder,args.run_id,initial,owned_child=child)
            receipt=evidence/('window_'+str(index-1).zfill(2)+'_controller_closure.json')
            write(receipt,closed);state['windows'].append({'receipt':record(receipt),
                'completed_steps':closed['completed_steps'],'selected_step':closed['selected_step'],
                'automatic_budget_handoff_allowed':closed['automatic_budget_handoff_allowed']})
            remaining=deadline-time.monotonic()
            if closed['completed_steps']>=initial['configured_steps'] or not closed['automatic_budget_handoff_allowed'] or remaining<960 or supervisor_budget_stop:
                state.update(status='next_experiment_required',stop_reason='configured_TSC_horizon' if closed['completed_steps']>=initial['configured_steps']
                    else 'bounded_supervisor_wall_horizon' if remaining<960 or supervisor_budget_stop else closed['selection_stop_reason'],
                    resource_review_required=closed['resource_review_required'],objective_complete=False,
                    next_action='ROOT reviews actual DEV/retention and the active quality goal; full state and BEST are preserved',finished_at_utc=now())
                write(evidence/'supervisor_status.json',state);return 1 if closed['resource_review_required'] else 0
            target=folder.parent/(args.successor_prefix+'_'+str(index).zfill(2))
            command=successor_command(folder,target,args.run_id,min(args.window_hours,(remaining-60)/3600),args.poll_seconds)
            with (evidence/('successor_'+str(index).zfill(2)+'.log')).open('x',encoding='utf-8') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            state.update(status='successor_started',latest_launch={'source_verified_closure':record(receipt),'target':str(target),
                'command':command,'owned_controller_Popen_pid':child.pid,'actual_launch_completed':True,
                'actual_owned_controller_exit_code':None,'started_at_utc':now()})
            write(evidence/'supervisor_status.json',state);folder=target;index+=1
    except BaseException as error:
        state.update(status='failed_no_automatic_retry',error=repr(error),objective_complete=False,
            owned_controller_pending_exit=child.pid if child is not None and child.poll() is None else None,
            next_action='ROOT reviews failure; no automatic GPU restart; existing owned helper retains its finite guards',finished_at_utc=now())
        write(evidence/'supervisor_status.json',state);raise


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    for name in ('source-loop','supervisor-dir'):result.add_argument('--'+name,type=Path,required=True)
    result.add_argument('--run-id',required=True);result.add_argument('--successor-prefix',required=True)
    result.add_argument('--launcher-pid',type=int,required=True);result.add_argument('--worker-pid',type=int,required=True)
    result.add_argument('--window-hours',type=float,default=3.);result.add_argument('--max-hours',type=float,default=12.)
    result.add_argument('--poll-seconds',type=float,default=5.);result.add_argument('--bootstrap-only',action='store_true')
    result.add_argument('--gpu-window-authorized',action='store_true');return result


if __name__=='__main__':sys.exit(main(parser().parse_args()))
