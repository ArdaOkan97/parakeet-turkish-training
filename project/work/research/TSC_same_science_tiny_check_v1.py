"""Small synthetic TSC exact-state routing check; stdlib only, no GPU/data/model reads."""
import argparse,ast,copy,json,os,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(HERE))
import run_hf_TSC_arm_v1 as first
import resume_TSC_same_science_v1 as exact
import launch_TSC_same_science_resume_v1 as outer
import supervise_TSC_same_science_v1 as supervisor
checks=[]
def check(value,name):
    if not value:raise AssertionError(name)
    checks.append(name)
def reject(call,name):
    try:call()
    except (ValueError,KeyError):checks.append(name)
    else:raise AssertionError('Bad evidence accepted: '+name)
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')
def recipe_fixture():
    tree=ast.parse((ROOT/'work/research/domain_same_science_tiny_check_v1.py').read_text(encoding='utf-8'))
    recipe=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='recipe' for t in n.targets)))
    recipe['steps']=9363;recipe['eval_epoch_fraction']=.026708331664062605
    return recipe
def run(output):
    output=first.scoped(output);check(not output.exists(),'New report only')
    chain=supervisor.frozen_sources();check(chain==outer.PINS and len(chain)==21,'Pinned original 21 science sources plus separate exact memory sibling')
    proof=supervisor.controller_order_proof();check(set(proof['controllers'])=={'first','exact'},'Actual first/exact source-owned close-completion-terminal-publish order')
    old={(module,name):getattr(module,name) for module,names in ((exact,('HERE','ROOT','sha')),(first,('HERE','sha','chain_identity'))) for name in names}
    old_preload=sys.modules.get('preload_nemo_linker_v1');old_dispatch=outer.runpy.run_path;old_executable=sys.executable
    try:
        with tempfile.TemporaryDirectory(dir=ROOT/'work/research',prefix='TSC_exact_fixture_') as temp:
            d=Path(temp);exact.ROOT=d;exact.HERE=d/'outputs/parakeet_turkish';first.HERE=exact.HERE
            real_sha=old[exact,'sha']
            def fixture_sha(path):
                p=Path(path);return real_sha(HERE/p.name if p.suffix=='.py' and not p.exists() else p)
            exact.sha=first.sha=fixture_sha;first.chain_identity=lambda:{'source_sha256':chain}
            n=74899;recipe=recipe_fixture();runid='synthetic_TSC';run=exact.HERE/'runs'/runid
            parent=d/'first_window';successor=d/'second_window';source=d/'closed_original_config.json'
            save(source,{'config_identity':{'training_arguments':recipe}})
            assets={key:d/(key+'.json') for key in ('tsc_input_proof','tsc_training_approval','source_closure','tr_manifest','replay_manifest','tr_frame_metadata','tr_frame_metadata_proof','data_v3_proof')}
            for key,path in assets.items():save(path,{'source_run_config':exact.record(source)} if key=='source_closure' else {'synthetic_only':True})
            ready=parent/'original_ready.json'
            native={'kind':'native_tdt_training_backend_v3','status':'ready','run_id':runid,'tdt_implementation':'nemo','pid':999,
                'trainer_sha256':chain['train_lora_batched_tsc_v1.py'],'adapter_sha256':chain['native_tdt_checkpointed_v3.py'],
                'launcher_sha256':chain['launch_native_training_tsc_v1.py'],'memory_launcher_sha256':chain['launch_with_memory_guard_tsc_v1.py'],
                'declared_allocator_policy':{'cap_gb':8,'end_cache_gb':4,'native_inactive_threshold_bytes':750000000},
                'TSC_explicit_asset_records':{k:exact.record(p) for k,p in assets.items()}}
            save(ready,native);ready_sha=exact.sha(ready)
            argv=first.recipe_arguments(recipe)+['--run-id',runid,'--resume-weights',str(d/'original_best.pt'),
                '--native-backend-provenance',str(ready),'--native-max-joint-bytes','1700000000','--stop-file',str(parent/'stop.txt'),'--max-wall-seconds','10500']
            for key,path in assets.items():argv+=['--'+key.replace('_','-'),str(path)]
            config={**recipe,'run_id':runid,'turkish_clips_used':n,'replay_clips_used':1595,'tr_clips_per_update':8,'replay_clips_per_update':8,
                'native_backend_provenance':str(ready),'tsc_input_proof':str(assets['tsc_input_proof']),'tsc_training_approval':str(assets['tsc_training_approval']),
                'warm_start':{'mode':'new_optimizer_warm_start'},'code_sha256':chain,'config_identity':{'training_arguments':recipe,
                    'public_turkish_data_policy':{'accepted_clips':n},'extended_tokenizer_sha256':first.TOKENIZER_SHA,
                    'input_sha256':{str(p):exact.sha(p) for p in [ready,*assets.values()]}}}
            save(run/'run_config.json',config);save(run/'initial_development.json',{'synthetic_only':True})
            latest=d/'work'/(runid+'_latest.pt');best=d/'work'/(runid+'_best.pt');latest.parent.mkdir()
            latest.write_bytes(b'Opaque synthetic full Adam sampler RNG checkpoint, no tensor load');best.write_bytes(b'Opaque synthetic best')
            def selection(step):return {'run_id':runid,'accepted':True,'completed_steps':step,'selected_step':0,'weights':str(best),'latest_weights':str(latest),
                'heldout_tests_evaluated':False,'stop_early_reason':'soft_max_wall_seconds','turkish_epochs_completed':step*8/n}
            def history(step):return [{'step':i,'turkish_clips_seen':i*8,'turkish_unique_clips_seen':min(i*8,n)} for i in range(1,step+1)]
            save(run/'training_history.json',history(2));save(run/'selection.json',selection(2))
            paths={'native':str(ready),**{key:str(parent/(key+'.json')) for key in ('io','cache','memory')}}
            save(Path(paths['io']),{'status':'complete'});save(Path(paths['cache']),{'status':'complete','cache_reclamation':{'eligible_update_calls':2}})
            save(Path(paths['memory']),{'status':'complete','kind':'memory_guard_training_launcher_tsc_v1','cap_bytes':8000000000,'cleanup_threshold_bytes':4000000000,
                'allocator_cap_applied':True,'all_phase_owned_memory_telemetry':{'thread_join_completed':True}})
            save(ready.with_suffix('.completion.json'),{'kind':'native_tdt_training_backend_completion_v3','status':'complete','sources_unchanged':True,'backend_ready_sha256':ready_sha})
            state={'kind':'isolated_HF_TSC_guarded_quality_arm_v1','status':'quality_window_ended','run_id':runid,'worker_arguments':argv,'paths':paths,
                'source_sha256':chain,'controller_sha256':exact.TSC_CONTROLLER_SHA,'child_exit_code':0,'completed_steps':2,'selected_step':0,
                'selection':exact.record(run/'selection.json'),'combined_TR_clips':n,'configured_steps':recipe['steps'],'actual_stage_recipe':recipe,
                'TSC_input_proof':exact.record(assets['tsc_input_proof']),'TSC_training_approval':exact.record(assets['tsc_training_approval'])}
            save(parent/'loop_status.json',state)
            args=SimpleNamespace(source_loop=parent,run_id=runid,output_dir=successor,source_closed_confirmed=True,resource_review_cleared=False,window_hours=3.,poll_seconds=5.)
            plan=exact.source_plan(args,True)
            # Code remains at its genuine frozen path; only synthetic run artifacts move.
            plan['runtime_memory_launcher']=exact.record(HERE/outer.RUNTIME_MEMORY_NAME)
            check(plan['initial_step']==2 and exact.sha(plan['archive_path'])==exact.sha(latest),'Actual closed preflight archives opaque full-state bytes unchanged')
            check(plan['configured_steps']==9363 and plan['source_config_identity']==config['config_identity'],'Dynamic non-domain horizon and source science preserved')
            check(outer.option(plan['worker_arguments'],'--native-backend-provenance')==str(ready) and '--resume-weights' not in plan['worker_arguments'],
                'Original ready preserved and full exact restore replaces warmweights')
            bad=copy.deepcopy(config);bad['config_identity']['training_arguments']['head_lr']=1e-4
            reject(lambda:exact.validate_config(bad,argv),'Scientific LR mutation rejected')
            bad=copy.deepcopy(config);bad['config_identity']['training_arguments']['eval_epoch_fraction']=.25
            changed=list(argv);exact.replace_option(changed,'--eval-epoch-fraction',.25)
            reject(lambda:exact.validate_config(bad,changed),'Quarter-fraction mutation rejected against original source')
            events=[]
            def mock_preload():
                events.append('preload');check('torch' not in sys.modules,'Fresh preload before Torch')
                return {'kind':'coherent_native_cuda12_9_preload_v1','module_sha256':chain['preload_nemo_linker_v1.py'],'actual_binding_linker':{'version':[12,9]},'cuda_initialized':False}
            def mock_dispatch(path,run_name):
                events.append('dispatch');check(Path(path).name==outer.RUNTIME_MEMORY_NAME and events==['preload','dispatch'],'Fresh preloader routes exact-only memory after preload')
            sys.modules['preload_nemo_linker_v1']=SimpleNamespace(preload=mock_preload);outer.runpy.run_path=mock_dispatch
            sys.executable=str(ROOT/'work/nemo-native-env-v2/Scripts/python.exe')
            outer.main(SimpleNamespace(original_native_provenance=ready,original_native_provenance_sha256=ready_sha,
                fresh_runtime_provenance=Path(plan['paths']['fresh_runtime']),memory_arguments=plan['command'][plan['command'].index('--')+1:],validate_only=False))
            check(exact.sha(ready)==ready_sha,'Original native-ready bytes remain unchanged')
            owned={'pid':os.getpid()};exact.validate_fresh_runtime(exact.read(plan['paths']['fresh_runtime'],{}),plan,owned)
            check(True,'Fresh PID/preload proof separate from old scientific ready')
            resumed=copy.deepcopy(config);resumed['warm_start']={'mode':'exact_resume','path':plan['archive_path'],
                'sha256':plan['snapshots']['latest']['snapshot']['sha256'],'source_run_id':runid,'source_step':2}
            save(run/'run_config.json',resumed);save(run/'training_history.json',history(3));save(run/'selection.json',selection(3))
            new={k:Path(p) for k,p in plan['paths'].items()}
            memory={'kind':'memory_guard_training_launcher_TSC_resume_v1','status':'complete','cap_bytes':8000000000,'cleanup_threshold_bytes':4000000000,
                'allocator_cap_applied':True,'training_math_changed':False,'backend_flags_changed':False,'all_phase_owned_memory_telemetry':{'thread_join_completed':True}}
            mapping={'memory_launcher_sha256':outer.RUNTIME_MEMORY_NAME,'cache_launcher_sha256':'launch_with_cache_reclaim_tsc_v1.py','cache_helper_sha256':'cuda_cache_reclaim_v2.py',
                'resource_helper_sha256':'resource_watchdog_v2.py','frozen_worker_sha256':'train_lora_batched_tsc_v1.py'}
            for key,name in mapping.items():memory[key]=memory[key+'_after']=outer.RUNTIME_MEMORY_SHA if name==outer.RUNTIME_MEMORY_NAME else chain[name]
            for key,name in {'resource_telemetry_sha256':'paired_windows_commit_telemetry_v1.py','data_policy_sha256':'public_turkish_data_policy_v3.py'}.items():
                memory[key]=memory[name.removesuffix('.py')+'_sha256_after']=chain[name]
            save(new['memory'],memory);cache={'kind':'cache_reclamation_training_launcher_tsc_v1','status':'complete','frozen_worker_sha256':chain['train_lora_batched_tsc_v1.py'],
                'cache_reclamation':{'eligible_update_calls':1,'failed_reclamations':0}}
            save(new['cache'],cache);save(new['io'],{'kind':'atomic_io_retry_tsc_v1','status':'complete','worker_sha256':chain['train_lora_batched_tsc_v1.py'],'worker_sha256_after':chain['train_lora_batched_tsc_v1.py']})
            done=exact.verify_completion(plan,selection(3),owned)
            check(done['new_optimizer_end_calls']==1 and done['cumulative_completed_steps']==3,'Suffix-only hook count and full inherited frontier')
            cache['cache_reclamation']['eligible_update_calls']=3;save(new['cache'],cache)
            reject(lambda:exact.verify_completion(plan,selection(3),owned),'Cumulative hook count cannot replace process suffix count')
            cache['cache_reclamation']['eligible_update_calls']=1;save(new['cache'],cache)
            corrupt=history(3);corrupt[0]['tampered']=True;save(run/'training_history.json',corrupt)
            reject(lambda:exact.verify_completion(plan,selection(3),owned),'Inherited history mutation rejected')
    finally:
        for (module,name),value in old.items():setattr(module,name,value)
        outer.runpy.run_path=old_dispatch;sys.executable=old_executable
        if old_preload is None:sys.modules.pop('preload_nemo_linker_v1',None)
        else:sys.modules['preload_nemo_linker_v1']=old_preload
    check(supervisor.automatic_budget_handoff({'stop_early_reason':'soft_max_wall_seconds'},{'resource_review_required':False,'stop_category':'budget'}),'Scheduled soft budget permits exact handoff')
    for category in ('retention','plateau','resource','manual'):
        check(not supervisor.automatic_budget_handoff({'stop_early_reason':'controller_stop_file'},{'resource_review_required':category=='resource','stop_category':category}),category+' stop cannot auto-resume')
    check(not any(n in sys.modules for n in ('torch','nemo','numpy','soundfile','transformers')),'No ML/audio imports, checkpoint parsing or GPU')
    report={'kind':'TSC_same_science_exact_resume_tiny_verification_v1','passed':True,'check_count':len(checks),'checks':checks,
        'synthetic_only':True,'actual_checkpoint_read_or_deserialization':False,'actual_process_GPU_or_native_preload_execution':False,
        'runtime_preload_and_dispatch_mocked':True,'source_order_proof':proof,'source_sha256':{Path(m.__file__).name:exact.sha(m.__file__) for m in (first,exact,outer,supervisor)},
        'created_at_utc':datetime.now(timezone.utc).isoformat()}
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({'passed':True,'checks':len(checks),'output':str(output)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
