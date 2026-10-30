"""One narrow stdlib routing/lifecycle fixture. No actual source or GPU use."""
from __future__ import annotations
import argparse
import ast
import copy
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/parakeet_turkish"
sys.path.insert(0, str(OUTPUT))
import run_hf_domain30_arm_v1 as controller
import launch_native_training_domain30_v1 as outer

checks = []


def check(value, name):
    if not value:raise AssertionError(name)
    checks.append(name)


def reject(call, name):
    try:call()
    except ValueError:checks.append(name)
    else:raise AssertionError("Expected rejection: " + name)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


chain = controller.chain_identity()
check(len(chain['source_sha256']) == 21, "Actual full21-source routing identity and dispatch proof verified")
check('torch' not in sys.modules and 'nemo' not in sys.modules, "Actual controller and outer import stdlib without ML")
recipe = {
    'epochs': 3., 'steps': 28082, 'batch_size': 8, 'accumulation': 2, 'replay_fraction': .5,
    'bucket_size': 128, 'seed': 20261010, 'max_tokens': 180, 'head_lr': 5e-5, 'lora_lr': 3e-4,
    'new_token_lr_multiplier': 6., 'replay_kl': 10., 'duration_kl': .5, 'feature_distillation': 1.,
    'rank': 8, 'alpha': 16., 'last_n_layers': None, 'chunk_size': 16, 'tdt_implementation': 'nemo',
    'warmup_steps': 25, 'schedule': 'constant', 'min_lr_ratio': .1, 'eval_per_epoch': 1.,
    'eval_epoch_fraction': .25, 'eval_every_updates': 250, 'min_improvement': .002,
    'no_encoder_checkpoint': False, 'phonetic_sh_donors': False, 'train_limit': None,
    'early_stop_patience': 3, 'max_retention_failures': 2, 'target_turkish_wer': None}
original_closed = controller.closed_source_identity
try:
    with tempfile.TemporaryDirectory(dir=ROOT / 'work/research', prefix='domain_controller_fixture_') as directory:
        fixture = Path(directory)
        source_config, source_selection, closure_path = [fixture / name for name in ('config.json','selection.json','closure.json')]
        source_weights = fixture / 'source_weights.pt'
        source_weights.write_bytes(b'Synthetic fixture only, not a real checkpoint')
        config = {'run_id':'closed_source_fixture','base_revision':'541d1f99c6b0c3cd0b11a95167540bb8edefd82b',
            'config_identity': {'extended_tokenizer_sha256':controller.TOKENIZER_SHA,
                'native_loss':{'sigma':0.,'omega':0.,'caller_reduction':'raw per-clip NLL / each target length, then physical-batch mean'},
                'training_arguments':recipe},
            'code_sha256':{'train_lora_batched_data_v11.py':chain['source_sha256']['train_lora_batched_data_v11.py']},
            'input_sha256':{}, 'native_max_joint_bytes':1700000000}
        save(source_config,config);save(source_selection,{'fixture_only':True})
        save(closure_path,{'source_run_config':controller.record(source_config),'source_selection':controller.record(source_selection)})
        identity={'source_run_id':'closed_source_fixture','turkish_wer':.54,'selected_step':6705,
            'closure_sha256':controller.sha(closure_path),'fixture_only':True}
        controller.closed_source_identity=lambda context:copy.deepcopy(identity)
        loop_id='domain_mock_'+Path(directory).name.replace('-','_')
        args=controller.parser().parse_args(['--loop-id',loop_id,'--source-checkpoint',str(source_weights),
            '--source-selection',str(source_selection),'--source-run-config',str(source_config),
            '--source-closure',str(closure_path),'--replay-seed','20261012','--validate-only'])
        plan=controller.prepare(args)
        check(plan['actual_stage_recipe']=={**recipe,'epochs':1.,'steps':13009},
            "Actual prepare copies every recipe field except explicit truthful domain horizon")
        check(plan['copied_source_recipe']['steps']==28082 and plan['horizon']['legacy_uniform_cap_copied'] is False,
            "Legacy28082 cap retained only as historical source metadata")
        check(plan['objective_complete'] is False and plan['exact_continuation_supported_by_this_outer'] is False,
            "First-launch window cannot falsely complete goal or claim exact continuation")
        command=plan['command'];separator=command.index('--');memory=command[separator+1:]
        outer_args=argparse.Namespace(native_provenance=Path(plan['paths']['native']),
            memory_launcher=OUTPUT/chain['memory_launcher'],
            memory_launcher_sha256=chain['source_sha256'][chain['memory_launcher']],
            trainer_sha256=chain['worker_sha256'],memory_arguments=memory,validate_only=True)
        _,trainer,_,worker_argv,run_id,_,_=outer.validate(outer_args)
        check(run_id==plan['run_id'] and worker_argv==plan['worker_arguments']
              and trainer.name=='train_lora_batched_domain30_v1.py',
              "Actual pinned native outer validates generated nested worker command without preload or ML")
        for flag,value in (('--epochs','1.0'),('--steps','13009'),('--replay-kl','10.0'),
                           ('--sampling-mode','domain30'),('--replay-seed','20261012')):
            check(worker_argv[worker_argv.index(flag)+1]==value,"Actual worker argument retained: "+flag)
        reject(lambda:controller.prepare(argparse.Namespace(**{**vars(args),'window_hours':float('nan')})),
               "NaN budget rejected")
        check(controller.main(args)==0,'Actual validate-only main launches no subprocess or loop state')
        check(not (OUTPUT/'runs'/plan['loop_id']).exists(),'Preflight creates no live controller namespace')
        reject(lambda:controller.main(argparse.Namespace(**{**vars(args),
            'preflight_output':OUTPUT/'runs'/plan['loop_id']/'preflight.json'})),
            'Preflight cannot accidentally create the fresh loop namespace before launch')

        started=datetime.now(timezone.utc).isoformat()
        owned={'kind':'owned_training_worker_identity_v1','expected_run_id':plan['run_id'],
            'worker_arguments_sha256':controller.guards.argument_sha(plan['worker_arguments']),
            'pid':101,'ppid':100,'started_at_utc':started,
            'process_creation_identity':{'available':True,'pid':101,'creation_unix_seconds':controller.guards.stamp(started).timestamp()}}
        for key,name in {'memory_launcher_sha256':'launch_with_memory_guard_domain30_v1.py',
            'cache_launcher_sha256':'launch_with_cache_reclaim_domain30_v1.py','cache_helper_sha256':'cuda_cache_reclaim_v2.py',
            'resource_helper_sha256':'resource_watchdog_v2.py','resource_telemetry_sha256':'paired_windows_commit_telemetry_v1.py',
            'frozen_worker_sha256':'train_lora_batched_domain30_v1.py','data_policy_sha256':'public_turkish_data_policy_v3.py'}.items():
            owned[key]=plan['source_sha256'][name]
        check(controller.validate_identity(owned,plan,100,started)==owned,
            "Actual owned PID/PPID/creation/argument/source validator accepts correct fields")
        bad=copy.deepcopy(owned);bad['ppid']=999
        reject(lambda:controller.validate_identity(bad,plan,100,started),'Foreign worker PID attribution rejected')

        # Completion schema fixture uses temporary evidence and real validators;
        # actual GPU/memory/optimizer values are deliberately not claimed.
        original_here=controller.HERE;original_chain=controller.chain_identity
        controller.HERE=fixture/'outputs/parakeet_turkish'
        controller.chain_identity=lambda:chain
        evidence_paths={key:str(fixture/key/(plan['run_id']+('.json' if key!='log' else '.log'))) for key in plan['paths']}
        completion_plan={**plan,'paths':evidence_paths}
        for key in ('io','cache','memory'):save(Path(evidence_paths[key]),{'status':'complete'})
        save(Path(evidence_paths['native']),{'fixture_only':True})
        save(Path(evidence_paths['native']).with_suffix('.completion.json'),{'status':'complete','sources_unchanged':True,
            'backend_ready_sha256':controller.sha(evidence_paths['native'])})
        save(Path(evidence_paths['memory']),{'status':'complete','kind':'memory_guard_training_launcher_domain30_v1',
            'cap_bytes':8000000000,'cleanup_threshold_bytes':4000000000,'allocator_cap_applied':True,
            'all_phase_owned_memory_telemetry':{'thread_join_completed':True}})
        save(Path(evidence_paths['cache']),{'status':'complete','cache_reclamation':{'eligible_update_calls':100}})
        run=controller.HERE/'runs'/plan['run_id']
        save(run/'run_config.json',{'epochs':1.,'steps':13009,'sampling_mode':'domain30','replay_seed':args.replay_seed,
            'warm_start':{'mode':'new_optimizer_warm_start'}})
        save(run/'initial_development.json',{'fixture_only':True})
        coverage={'turkish_draws':800,'domain':{'fleurs':{'draws':240},'cv':{'draws':560}},
            'all_Turkish_pool_sources_seen':False}
        selection={'run_id':plan['run_id'],'heldout_tests_evaluated':False,'completed_steps':100,
            'Turkish_domain_coverage':coverage,'turkish_epoch_semantics':'minimum_domain_exposure',
            'turkish_epochs_completed':min(240/2037,560/72846),'accepted':True,'selected_step':0,
            'stop_early_reason':'soft_max_wall_seconds','turkish_validation_wer':.54}
        save(run/'selection.json',selection)
        check('memory_provenance' in controller.verify_completion(completion_plan,selection),
            "Actual completion field names bind cap/thread join/hook/update/quota/min-domain clock")
        bad=copy.deepcopy(selection);bad['turkish_epochs_completed']=800/74883
        reject(lambda:controller.verify_completion(completion_plan,bad),
               "Global draw-equivalent epoch cannot masquerade as minimum-domain exposure")

        class DoneChild:
            pid=100;returncode=0
            def poll(self):return 0
        original_popen=controller.subprocess.Popen;original_note=controller.publish_note
        invocations=[]
        controller.subprocess.Popen=lambda command,**kwargs:(invocations.append(command) or DoneChild())
        controller.publish_note=lambda state,text:None
        Path(evidence_paths['log']).parent.mkdir(parents=True,exist_ok=True)
        state=copy.deepcopy(completion_plan)
        try:
            check(controller.run_arm(state,time.monotonic()+10800,args)==0,
                "Mock owned child completion exercises real lifecycle/completion branch")
            check(invocations==[plan['command']] and state['completed_steps']==100
                and state['stop_category']=='budget' and state['objective_complete'] is False
                and state['configured_domain_epoch_completed'] is False,
                "Budget-partial candidate remains checkpointed incomplete objective, not a full domain pass")
        finally:
            controller.subprocess.Popen=original_popen;controller.publish_note=original_note
            controller.HERE=original_here;controller.chain_identity=original_chain
finally:
    controller.closed_source_identity=original_closed

check('torch' not in sys.modules and 'nemo' not in sys.modules,'No ML imports throughout narrow fixture')
report={'kind':'domain30_controller_tiny_routing_lifecycle_verification_v1','passed':True,'checks':len(checks),
    'check_names':checks,'controller_sha256':controller.sha(OUTPUT/'run_hf_domain30_arm_v1.py'),
    'actual21_source_chain_validated':True,'actual_outer_command_validator_executed':True,
    'closed_source_and_process_GPU_runtime_evidence_synthetic_only':True,
    'actual_GPU_or_source_checkpoint_read':False,'Torch_or_NeMo_imported':False,
    'scope':'One tiny command/PID/completion/mock-child fixture; existing guard semantics reused unchanged'}
fixture_cli=argparse.ArgumentParser(description=__doc__)
fixture_cli.add_argument('--output',type=Path)
fixture_options=fixture_cli.parse_args()
destination=(fixture_options.output or OUTPUT/('domain30_controller_tiny_routing_'+str(time.time_ns())+'.json')).resolve()
assert destination.is_relative_to(ROOT) and not destination.exists()
with destination.open('x',encoding='utf-8') as stream:
    stream.write(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':True,'checks':len(checks),'controller_sha256':report['controller_sha256'],'output':str(destination)}))
