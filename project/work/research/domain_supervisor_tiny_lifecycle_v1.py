"""One synthetic FIRST→owned exact→retention stop fixture; stdlib, no checkpoints/GPU."""
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2];OUTPUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUTPUT))
import supervise_domain30_same_science_v1 as controller

checks=[]
def check(value,name):
    if not value:raise AssertionError(name)
    checks.append(name)
def reject(call,name):
    try:call()
    except ValueError:checks.append(name)
    else:raise AssertionError('Expected rejection: '+name)
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')

actual_order=controller.controller_order_proof()
check(set(actual_order['controllers'])=={'first','exact'},'Actual pinned controllers close→verify→publish ordering checked')
check(not any(name in sys.modules for name in ('torch','nemo','numpy','soundfile')),'Actual supervisor and guards import no ML/audio')
original={key:getattr(controller,key) for key in ('HERE','source_state','bootstrap')}
original_first=controller.domain.verify_completion;original_exact=controller.exact.verify_completion
original_popen=controller.subprocess.Popen
launched=[];waited=[]
try:
    with tempfile.TemporaryDirectory(dir=ROOT/'work/research',prefix='domain_supervisor_fixture_') as temp:
        fixture=Path(temp);controller.HERE=fixture/'outputs';run_id='synthetic_domain30';parent=fixture/'first';evidence=fixture/'supervisor'
        ready=parent/'native_ready.json';closure=fixture/'uniform_source_closure.json';save(closure,{'fixture_only':True})
        started=datetime.now(timezone.utc).isoformat()
        initial_argv=['--run-id',run_id,'--native-backend-provenance',str(ready),'--source-closure',str(closure)]
        pins=controller.preload.PINS
        fields={'memory_launcher_sha256':'launch_with_memory_guard_domain30_v1.py',
            'cache_launcher_sha256':'launch_with_cache_reclaim_domain30_v1.py','cache_helper_sha256':'cuda_cache_reclaim_v2.py',
            'resource_helper_sha256':'resource_watchdog_v2.py','resource_telemetry_sha256':'paired_windows_commit_telemetry_v1.py',
            'frozen_worker_sha256':'train_lora_batched_domain30_v1.py','data_policy_sha256':'public_turkish_data_policy_v3.py'}
        def identity(pid,ppid,argv):
            return {'kind':'owned_training_worker_identity_v1','expected_run_id':run_id,'pid':pid,'ppid':ppid,
                'worker_arguments_sha256':controller.domain.guards.argument_sha(argv),'started_at_utc':started,
                'process_creation_identity':{'available':True,'pid':pid,'creation_filetime_100ns':123,
                    'creation_unix_seconds':controller.domain.guards.stamp(started).timestamp()},
                **{key:pins[name] for key,name in fields.items()}}
        save(parent/'identity.json',identity(101,100,initial_argv))
        preload={'kind':'coherent_native_cuda12_9_preload_v1','module_sha256':pins['preload_nemo_linker_v1.py'],
            'actual_binding_linker':{'version':[12,9]},'cuda_initialized':False}
        save(ready,{'kind':'native_tdt_training_backend_v3','status':'ready','run_id':run_id,'pid':101,'ppid':100,
            'worker_arguments_sha256':controller.domain.guards.argument_sha(initial_argv),
            'trainer_sha256':pins['train_lora_batched_domain30_v1.py'],'adapter_sha256':pins['native_tdt_checkpointed_v3.py'],
            'launcher_sha256':pins['launch_native_training_domain30_v1.py'],
            'memory_launcher_sha256':pins['launch_with_memory_guard_domain30_v1.py'],'preload':preload,
            'declared_allocator_policy':{'cap_gb':8,'end_cache_gb':4,'native_inactive_threshold_bytes':750000000}})
        first={'kind':controller.FIRST_KIND,'run_id':run_id,'status':'running','owned_launcher_pid':100,
            'started_at_utc':started,'updated_at_utc':started,'source_sha256':pins,'worker_arguments':initial_argv,
            'paths':{'native':str(ready),'identity':str(parent/'identity.json')}}
        save(parent/'loop_status.json',first)
        # Source/science validation was exercised by the exact-helper fixture;
        # this fixture focuses on source-bound metadata and lifecycle routing.
        science={'fixture_only':True,'training_arguments':{'sampling_mode':'domain30','replay_seed':20261012,'steps':13009,'epochs':1.}}
        states={str(parent):(first,science)}
        controller.source_state=lambda folder,run:states[str(folder)]
        initial=controller.bootstrap(parent,run_id,100,101)
        check(not (controller.HERE/'runs'/run_id/'selection.json').exists(),
            'Actual bootstrap does not require unpublished final selection during first live window')
        check(initial['initial_independent_OS_exit_observation'] is False,
            'Initial live ownership explicitly relies on pinned metadata, not OS attachment')
        check(initial['original_domain_ready']==controller.record(ready),'Original first ready bound, not rewritten')
        reject(lambda:controller.bootstrap(parent,run_id,999,101),'Foreign declared first launcher rejected')
        # Completion functions have their own real-function18-check fixture.
        controller.domain.verify_completion=lambda plan,selected:{'fixture_only':True,'calls':'first_actual_completion_contract_stub'}
        controller.exact.verify_completion=lambda plan,selected,owned:{'fixture_only':True,'calls':'exact_actual_completion_contract_stub'}
        run=controller.HERE/'runs'/run_id;latest=fixture/'latest.pt';best=fixture/'best.pt'
        latest.write_bytes(b'Opaque synthetic latest; not a checkpoint');best.write_bytes(b'Opaque synthetic BEST')
        def selected(step,reason):
            return {'run_id':run_id,'accepted':True,'heldout_tests_evaluated':False,'completed_steps':step,
                'selected_step':0,'latest_weights':str(latest),'weights':str(best),'stop_early_reason':reason,
                'turkish_epochs_completed':step*8*.7/72846}
        chosen=selected(100,'soft_max_wall_seconds');save(run/'selection.json',chosen)
        first.update(status='quality_window_ended',child_exit_code=0,resource_review_required=False,
            stop_category='budget',completed_steps=100,selected_step=0,selection=controller.record(run/'selection.json'))
        save(parent/'loop_status.json',first)
        closed=controller.closure(parent,run_id,initial)
        check(closed['automatic_budget_handoff_allowed'] and closed['actual_supervisor_owned_controller_exit_code'] is None,
            'Verified first budget closure authorizes successor without false independent OS exit claim')
        for reason,category,helper in [('controller_stop_file','manual',None),('consecutive_retention_gate_failures','retention',None),
            ('development_plateau','plateau',None),('controller_stop_file','resource','actual_worker_process_commit_headroom_below2GB')]:
            check(not controller.automatic_budget_handoff({'stop_early_reason':reason},
                {'stop_category':category,'cooperative_stop_reason':helper,'resource_review_required':category=='resource'}),
                'No automatic repeat after '+category)
        check(controller.automatic_budget_handoff({'stop_early_reason':'controller_stop_file'},
            {'stop_category':'budget','cooperative_stop_reason':'window_budget_checkpoint_boundary','resource_review_required':False}),
            'Only scheduled controller stop counts as a budget handoff')
        controller.bootstrap=lambda *values:copy.deepcopy(initial)
        class ClosedChild:
            pid=1000
            def poll(self):return 0
            def wait(self,timeout):waited.append((self.pid,timeout));return 0
        def popen(command,**kwargs):
            launched.append(command)
            check(command[1]==str(controller.HELPER) and '--source-closed-confirmed' in command
                and '--gpu-window-authorized' in command and '--resource-review-cleared' not in command,
                'Actual generated command invokes only frozen same-run exact helper, never clears resource review')
            target=Path(command[command.index('--output-dir')+1]);new_argv=initial_argv+['--resume-exact',str(fixture/'immutable_full_state.pt')]
            paths={'identity':str(target/'identity.json'),'fresh_runtime':str(target/'fresh_runtime.json')}
            save(Path(paths['identity']),identity(201,200,new_argv))
            save(Path(paths['fresh_runtime']),{'kind':'domain_same_science_native_preload_resume_v1','status':'ready',
                'run_id':run_id,'pid':201,'ppid':200,'worker_arguments_sha256':controller.domain.guards.argument_sha(new_argv),
                'launcher_sha256':controller.PRELOADER_SHA,'source_sha256':pins,'preload':preload,
                'original_first_start_ready':controller.record(ready),'scientific_config_identity_changed':False,
                'fresh_runtime_proof_separate_from_original_ready':True})
            terminal={'kind':controller.EXACT_KIND,'run_id':run_id,'status':'train_window_ended','owned_launcher_pid':200,
                'started_at_utc':started,'updated_at_utc':started,'source_sha256':pins,'worker_arguments':new_argv,
                'paths':paths,'original_ready':controller.record(ready),'new_preloader':{'sha256':controller.PRELOADER_SHA},
                'child_exit_code':0,'resource_review_required':False,'stop_category':'retention','completed_steps':200,'selected_step':0}
            chosen=selected(200,'consecutive_retention_gate_failures');save(run/'selection.json',chosen)
            terminal['selection']=controller.record(run/'selection.json');save(target/'loop_status.json',terminal)
            states[str(target)]=(terminal,science)
            return ClosedChild()
        controller.subprocess.Popen=popen
        args=controller.parser().parse_args(['--source-loop',str(parent),'--supervisor-dir',str(evidence),
            '--run-id',run_id,'--successor-prefix','synthetic_successor','--launcher-pid','100','--worker-pid','101',
            '--window-hours','3','--max-hours','12','--gpu-window-authorized'])
        check(controller.main(args)==0 and len(launched)==1 and waited==[(1000,30)],
            'Actual supervisor owns and waits exactly one successor then stops on retention, no duplicate GPU launch')
        final=controller.read(evidence/'supervisor_status.json',{})
        check(final['status']=='next_experiment_required' and final['stop_reason']=='consecutive_retention_gate_failures'
            and final['objective_complete'] is False and len(final['windows'])==2,
            'Internal stop surfaces review with preserved unfinished goal and both closure receipts')
finally:
    for name,value in original.items():setattr(controller,name,value)
    controller.domain.verify_completion=original_first;controller.exact.verify_completion=original_exact
    controller.subprocess.Popen=original_popen

check(not any(name in sys.modules for name in ('torch','nemo','numpy','soundfile')),'Fixture imported no ML/audio, native/GPU or actual checkpoint state')
result={'kind':'domain_same_science_supervisor_tiny_lifecycle_verification_v1','passed':True,'checks':checks,'check_count':len(checks),
    'actual_source_order_proof':actual_order,'synthetic_metadata_and_checkpoint_bytes_only':True,
    'source_state_and_runtime_completion_contracts_stubbed':True,'actual_owned_Popen_mocked':True,
    'complementary_actual_exact_helper_fixture':controller.record(OUTPUT/'domain_same_science_exact_resume_tiny_verification_v1.json'),
    'supervisor':controller.record(Path(controller.__file__)),'fixture':controller.record(Path(__file__)),
    'actual_GPU_or_model_or_checkpoint_read':False,'created_at_utc':datetime.now(timezone.utc).isoformat()}
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
args=parser.parse_args();target=(args.output or OUTPUT/('domain_supervisor_tiny_'+str(time.time_ns())+'.json')).resolve()
if not target.is_relative_to(ROOT):raise ValueError('Workspace-only report')
with target.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
print(json.dumps({'passed':True,'checks':len(checks),'report':str(target)}))
