"""Focused stdlib synthetic closure/identity/command test; no real checkpoints or GPU."""
import argparse
import copy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2];OUTPUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUTPUT))
import resume_domain30_same_science_v1 as controller
import launch_domain30_same_science_resume_v1 as outer
import run_hf_domain30_arm_v1 as domain

checks=[]
def check(value,name):
    if not value:raise AssertionError(name)
    checks.append(name)
def reject(call,name):
    try:call()
    except (ValueError,KeyError):checks.append(name)
    else:raise AssertionError('Expected rejection: '+name)
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')

chain=domain.chain_identity()
check(outer.PINS==chain['source_sha256'] and len(outer.PINS)==21,'Exact original21 source pins; no duplicate key override')
check(not any(name in sys.modules for name in ('torch','nemo','numpy','soundfile')),'Real helper imports require no ML/audio modules')
recipe={'epochs':1.,'steps':13009,'batch_size':8,'accumulation':2,'replay_fraction':.5,'bucket_size':128,
    'seed':20261010,'max_tokens':180,'head_lr':5e-5,'lora_lr':3e-4,'new_token_lr_multiplier':6.,'replay_kl':10.,
    'duration_kl':.5,'feature_distillation':1.,'rank':8,'alpha':16.,'last_n_layers':None,'chunk_size':16,
    'tdt_implementation':'nemo','warmup_steps':25,'schedule':'constant','min_lr_ratio':.1,'eval_per_epoch':1.,
    'eval_epoch_fraction':.25,'eval_every_updates':250,'min_improvement':.002,'no_encoder_checkpoint':False,
    'phonetic_sh_donors':False,'train_limit':None,'early_stop_patience':3,'max_retention_failures':2,'target_turkish_wer':None}
originals={name:getattr(controller,name) for name in ('HERE','ROOT','sha')}
domain_originals={name:getattr(domain,name) for name in ('HERE','sha','chain_identity')}
original_preload=sys.modules.get('preload_nemo_linker_v1');original_run_path=outer.runpy.run_path
original_executable=sys.executable
try:
    with tempfile.TemporaryDirectory(dir=ROOT/'work/research',prefix='domain_exact_fixture_') as temp:
        fixture=Path(temp);controller.ROOT=fixture;controller.HERE=fixture/'outputs/parakeet_turkish';domain.HERE=controller.HERE
        real_sha=originals['sha']
        def fixture_sha(path):
            path=Path(path)
            return real_sha(OUTPUT/path.name if path.suffix=='.py' and not path.exists() else path)
        controller.sha=domain.sha=fixture_sha;domain.chain_identity=lambda:chain
        run_id='synthetic_domain30';run=controller.HERE/'runs'/run_id;parent=fixture/'parent';next_window=fixture/'next'
        closure=fixture/'original_uniform_closure.json';save(closure,{'fixture_only':True})
        ready=parent/'original_native_ready.json'
        native={'schema':1,'kind':'native_tdt_training_backend_v3','status':'ready','run_id':run_id,
            'tdt_implementation':'nemo','pid':999,'trainer_sha256':outer.PINS['train_lora_batched_domain30_v1.py'],
            'adapter_sha256':outer.PINS['native_tdt_checkpointed_v3.py'],
            'launcher_sha256':outer.PINS['launch_native_training_domain30_v1.py'],
            'memory_launcher_sha256':outer.PINS['launch_with_memory_guard_domain30_v1.py'],
            'declared_allocator_policy':{'cap_gb':8,'end_cache_gb':4,'native_inactive_threshold_bytes':750000000}}
        save(ready,native);ready_sha=controller.sha(ready)
        argv=domain.recipe_arguments(recipe)+['--run-id',run_id,'--resume-weights',str(fixture/'old_uniform_best.pt'),
            '--source-closure',str(closure),'--sampling-mode','domain30','--fleurs-fraction','.3','--replay-seed','20261012',
            '--native-backend-provenance',str(ready),'--native-max-joint-bytes','1700000000',
            '--stop-file',str(parent/'old_stop.txt'),'--max-wall-seconds','10500']
        sampler={'sampling_mode':'domain30','replay_seed':20261012,'turkish_seed':20261010,'fixture_only':True}
        config={**recipe,'run_id':run_id,'sampling_mode':'domain30','replay_seed':20261012,
            'native_backend_provenance':str(ready),'warm_start':{'mode':'new_optimizer_warm_start'},'code_sha256':outer.PINS,
            'config_identity':{'training_arguments':{**recipe,'sampling_mode':'domain30','fleurs_fraction':.3,'replay_seed':20261012},
                'extended_tokenizer_sha256':domain.TOKENIZER_SHA,'input_sha256':{str(ready):ready_sha,str(closure):controller.sha(closure)},
                'domain_sampling_identity':sampler,'closed_source':{'fixture_only':True}}}
        save(run/'run_config.json',config);save(run/'initial_development.json',{'fixture_only':True})
        history=[{'step':1},{'step':2}];save(run/'training_history.json',history)
        latest=fixture/'work'/(run_id+'_latest.pt');best=fixture/'work'/(run_id+'_best.pt')
        latest.parent.mkdir();latest.write_bytes(b'Opaque SYNTHETIC full-state bytes, not a model');best.write_bytes(b'Opaque SYNTHETIC BEST bytes')
        def selection(step):
            f=step*8*3//10;c=step*8-f
            return {'run_id':run_id,'accepted':True,'completed_steps':step,'selected_step':0,'weights':str(best),
                'latest_weights':str(latest),'heldout_tests_evaluated':False,'stop_early_reason':'soft_max_wall_seconds',
                'Turkish_domain_coverage':{'turkish_draws':step*8,'domain':{'fleurs':{'draws':f},'cv':{'draws':c}},
                    'all_Turkish_pool_sources_seen':False},'sampling_identity':sampler,
                'turkish_epoch_semantics':'minimum_domain_exposure','turkish_epochs_completed':min(f/2037,c/72846)}
        selected=selection(2);save(run/'selection.json',selected)
        paths={'native':str(ready),**{k:str(parent/(k+'.json')) for k in ('io','cache','memory')}}
        save(Path(paths['io']),{'status':'complete'})
        save(Path(paths['cache']),{'status':'complete','cache_reclamation':{'eligible_update_calls':2}})
        save(Path(paths['memory']),{'status':'complete','kind':'memory_guard_training_launcher_domain30_v1',
            'cap_bytes':8000000000,'cleanup_threshold_bytes':4000000000,'allocator_cap_applied':True,
            'all_phase_owned_memory_telemetry':{'thread_join_completed':True}})
        save(ready.with_suffix('.completion.json'),{'status':'complete','sources_unchanged':True,'backend_ready_sha256':ready_sha})
        state={'kind':'isolated_HF_domain30_guarded_quality_arm_v1','status':'quality_window_ended','run_id':run_id,
            'child_exit_code':0,'completed_steps':2,'selected_step':0,'selection':controller.record(run/'selection.json'),
            'worker_arguments':argv,'paths':paths,'replay_seed':20261012,'source_sha256':outer.PINS,
            'controller_sha256':controller.DOMAIN_CONTROLLER_SHA}
        save(parent/'loop_status.json',state)
        args=SimpleNamespace(source_loop=parent,run_id=run_id,output_dir=next_window,source_closed_confirmed=True,
            resource_review_cleared=False,window_hours=3.,poll_seconds=5.)
        plan=controller.source_plan(args,True)
        check(plan['initial_step']==2 and controller.sha(plan['archive_path'])==controller.sha(latest),
            'Actual closed-state preflight archives identical opaque full state without deserialization')
        check(plan['source_config_identity']==config['config_identity'] and plan['optimizer_sampler_RNG_counters_reset'] is False,
            'Exact original sampler/science identity and no state reset declared')
        for flag,path in (('--native-backend-provenance',ready),('--source-closure',closure)):
            check(outer.option(plan['worker_arguments'],flag)==str(path),'Original immutable argument preserved: '+flag)
        check('--resume-weights' not in plan['worker_arguments'] and '--resume-runtime-migration' not in plan['worker_arguments']
            and outer.option(plan['worker_arguments'],'--resume-exact')==plan['archive_path'],
            'Only full exact-state path; no warmweights/migration fallback')
        reject(lambda:controller.source_plan(SimpleNamespace(**{**vars(args),'window_hours':float('nan')}),False),'NaN window rejected')
        bad=copy.deepcopy(config);bad['config_identity']['training_arguments']['replay_seed']+=1
        reject(lambda:controller.validate_config(bad,argv),'Different replay seed rejected')
        bad=copy.deepcopy(config);bad['config_identity']['training_arguments']['head_lr']=1e-4
        reject(lambda:controller.validate_config(bad,argv),'Scientific learning rate change rejected')
        events=[]
        def preload():
            check('torch' not in sys.modules,'Fresh preload runs before Torch import')
            events.append('preload')
            return {'kind':'coherent_native_cuda12_9_preload_v1','module_sha256':outer.PINS['preload_nemo_linker_v1.py'],
                'actual_binding_linker':{'version':[12,9]},'cuda_initialized':False}
        def dispatch(path,run_name):
            events.append('dispatch')
            check(Path(path).name=='launch_with_memory_guard_domain30_v1.py' and events==['preload','dispatch'],
                'Actual new preloader routes unchanged domain wrapper after fresh preload')
        sys.modules['preload_nemo_linker_v1']=SimpleNamespace(preload=preload)
        outer.runpy.run_path=dispatch;sys.executable=str(ROOT/'work/nemo-native-env-v2/Scripts/python.exe')
        memory=plan['command'][plan['command'].index('--')+1:]
        outer_args=SimpleNamespace(original_native_provenance=ready,original_native_provenance_sha256=ready_sha,
            fresh_runtime_provenance=Path(plan['paths']['fresh_runtime']),memory_arguments=memory,validate_only=False)
        outer.main(outer_args)
        check(controller.sha(ready)==ready_sha,'Actual fresh preloader leaves original ready bytes unchanged')
        owned={'pid':os.getpid()}
        check(controller.validate_fresh_runtime(controller.read(plan['paths']['fresh_runtime'],{}),plan,owned)['pid']==os.getpid(),
            'Separate fresh PID/source/argv/preload proof matches current synthetic dispatched process')
        exact_config=copy.deepcopy(config);exact_config['warm_start']={'mode':'exact_resume','path':plan['archive_path'],
            'sha256':plan['snapshots']['latest']['snapshot']['sha256'],'source_run_id':run_id,'source_step':2}
        save(run/'run_config.json',exact_config);save(run/'training_history.json',history+[{'step':3}]);selected=selection(3)
        save(run/'selection.json',selected)
        new_paths={k:Path(v) for k,v in plan['paths'].items()}
        runtime={'status':'complete','kind':'memory_guard_training_launcher_domain30_v1','cap_bytes':8000000000,
            'cleanup_threshold_bytes':4000000000,'allocator_cap_applied':True,'training_math_changed':False,
            'backend_flags_changed':False,'all_phase_owned_memory_telemetry':{'thread_join_completed':True}}
        mapping={'memory_launcher_sha256':'launch_with_memory_guard_domain30_v1.py',
            'cache_launcher_sha256':'launch_with_cache_reclaim_domain30_v1.py','cache_helper_sha256':'cuda_cache_reclaim_v2.py',
            'resource_helper_sha256':'resource_watchdog_v2.py','frozen_worker_sha256':'train_lora_batched_domain30_v1.py'}
        for key,name in mapping.items():runtime[key]=runtime[key+'_after']=outer.PINS[name]
        for key,name in {'resource_telemetry_sha256':'paired_windows_commit_telemetry_v1.py',
            'data_policy_sha256':'public_turkish_data_policy_v3.py'}.items():
            runtime[key]=runtime[name.removesuffix('.py')+'_sha256_after']=outer.PINS[name]
        save(new_paths['memory'],runtime)
        cache={'kind':'cache_reclamation_training_launcher_domain30_v1','status':'complete',
            'frozen_worker_sha256':outer.PINS['train_lora_batched_domain30_v1.py'],
            'cache_reclamation':{'eligible_update_calls':1,'failed_reclamations':0}}
        save(new_paths['cache'],cache)
        save(new_paths['io'],{'kind':'atomic_io_retry_domain30_v1','status':'complete',
            'worker_sha256':outer.PINS['train_lora_batched_domain30_v1.py'],'worker_sha256_after':outer.PINS['train_lora_batched_domain30_v1.py']})
        completed=controller.verify_completion(plan,selected,owned)
        check(completed['new_optimizer_end_calls']==1 and completed['cumulative_completed_steps']==3,
            'Actual completion binds one new hook call vs three cumulative updates')
        cache['cache_reclamation']['eligible_update_calls']=3;save(new_paths['cache'],cache)
        reject(lambda:controller.verify_completion(plan,selected,owned),'Inherited full count cannot replace suffix hook count')
        cache['cache_reclamation']['eligible_update_calls']=1;save(new_paths['cache'],cache)
        save(run/'training_history.json',[{'step':1,'tampered':True},{'step':2},{'step':3}])
        reject(lambda:controller.verify_completion(plan,selected,owned),'Inherited history mutation rejected')
finally:
    for name,value in originals.items():setattr(controller,name,value)
    for name,value in domain_originals.items():setattr(domain,name,value)
    outer.runpy.run_path=original_run_path;sys.executable=original_executable
    if original_preload is None:sys.modules.pop('preload_nemo_linker_v1',None)
    else:sys.modules['preload_nemo_linker_v1']=original_preload

check(not any(name in sys.modules for name in ('torch','nemo','numpy','soundfile')),'Fixture imported no ML/audio and launched no child')
result={'schema':1,'kind':'domain_same_science_exact_resume_tiny_verification_v1','passed':True,
    'checks':checks,'check_count':len(checks),'synthetic_only':True,'actual_checkpoint_read_or_deserialization':False,
    'actual_process_GPU_or_native_preload_execution':False,'runtime_preload_and_dispatch_mocked':True,
    'source_sha256':{p.name:controller.sha(p) for p in (Path(controller.__file__),Path(outer.__file__))},
    'fixture_path':str(Path(__file__).resolve()),'fixture_sha256':controller.sha(__file__),
    'frozen_first_domain_controller_sha256':controller.sha(OUTPUT/'run_hf_domain30_arm_v1.py'),
    'original_domain_chain_sha256':controller.sha(domain.CHAIN_PATH),'created_at_utc':datetime.now(timezone.utc).isoformat()}
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
args=parser.parse_args();output=(args.output or OUTPUT/('domain_same_science_exact_resume_tiny_verification_'+str(time.time_ns())+'.json')).resolve()
if not output.is_relative_to(ROOT):raise ValueError('Workspace-only NEW report')
with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
print(json.dumps({'passed':True,'checks':len(checks),'output':str(output)}))
