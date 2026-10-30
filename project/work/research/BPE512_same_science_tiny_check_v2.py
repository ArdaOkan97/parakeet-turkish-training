"""Synthetic exact-window routing, stdlib only; no model or real checkpoint reads."""
import argparse,ast,copy,json,os,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(HERE))
import run_hf_bpe512_arm_v2 as first
import resume_BPE512_same_science_v2 as exact
import launch_BPE512_same_science_resume_v2 as outer
import supervise_BPE512_same_science_v2 as supervisor
import hf_bpe512_training_v2 as training
checks=[]
def check(value,name):
    if not value:raise AssertionError(name)
    checks.append(name)
def reject(call,name):
    try:call()
    except (ValueError,KeyError):checks.append(name)
    else:raise AssertionError('Bad evidence accepted: '+name)
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')
def recipe_fixture():
    tree=ast.parse((ROOT/'work/research/domain_same_science_tiny_check_v1.py').read_text(encoding='utf-8'))
    recipe=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='recipe' for t in n.targets)))
    recipe.update(steps=13009,eval_epoch_fraction=.026708331664062605,sampling_mode='domain30',fleurs_fraction=.3,replay_seed=20261012)
    return recipe
def run(output):
    output=first.scoped(output);check(not output.exists(),'New report only')
    chain=supervisor.frozen_sources();check(chain==outer.PINS and len(chain)==21,'All21 original source pins unchanged')
    order=supervisor.controller_order_proof();check(set(order['controllers'])=={'first','exact'},'Actual source child-close-completion-terminal publication order')
    changed={(module,name):getattr(module,name) for module,names in ((exact,('HERE','ROOT','sha')),(first,('HERE','sha','chain_identity')),(training,('sha_file',))) for name in names}
    old_preload=sys.modules.get('preload_nemo_linker_v1');old_dispatch=outer.runpy.run_path;old_executable=sys.executable
    try:
        with tempfile.TemporaryDirectory(dir=ROOT/'work/research',prefix='BPE_exact_fixture_') as temp:
            d=Path(temp);exact.ROOT=d;exact.HERE=d/'outputs/parakeet_turkish';first.HERE=exact.HERE
            real_sha=changed[exact,'sha']
            def fixture_sha(path):
                p=Path(path);return real_sha(HERE/p.name if p.suffix=='.py' and not p.exists() else p)
            exact.sha=first.sha=training.sha_file=fixture_sha;first.chain_identity=lambda:{'source_sha256':chain}
            recipe=recipe_fixture();runid='synthetic_BPE';run=exact.HERE/'runs'/runid
            parent=d/'first_window';successor=d/'second_window';source=d/'original_closed_config.json'
            sampling={'sampling_mode':'domain30','replay_seed':20261012,'turkish_seed':recipe['seed'],
                'replay_rng_independent_of_turkish':True,'fleurs_fraction_numerator':3,'fleurs_fraction_denominator':10,
                'batch_size':8,'accumulation':2,'replay_per_update':8}
            save(source,{'config_identity':{'training_arguments':recipe,'domain_sampling_identity':sampling},'code_sha256':training.DOMAIN_SOURCES})
            assets={key:d/(key+'.json') for key in ('bpe_source_checkpoint','bpe_source_selection','bpe_source_closure','bpe_conversion_report')}
            for key,path in assets.items():save(path,{'source_run_config':exact.record(source),
                'allow_BPE512_vocabulary_stage':True,'source_sampling_metadata':sampling} if key=='bpe_source_closure' else {'synthetic_only':True})
            candidate=ROOT/'work/hf_turkish_bpe_candidate_512_v1'
            # Only the four small immutable candidate assets are read; no TRAIN/DEV manifests.
            first.candidate_assets(candidate)
            ready=parent/'original_ready.json'
            native={'kind':'native_tdt_training_backend_v3','status':'ready','run_id':runid,'tdt_implementation':'nemo','pid':999,
                'trainer_sha256':chain['train_lora_batched_bpe512_v2.py'],'adapter_sha256':chain['native_tdt_checkpointed_v3.py'],
                'launcher_sha256':chain['launch_native_training_bpe512_v2.py'],'memory_launcher_sha256':chain['launch_with_memory_guard_bpe512_v2.py'],
                'declared_allocator_policy':{'cap_gb':8,'end_cache_gb':4,'native_inactive_threshold_bytes':750000000}}
            save(ready,native);ready_sha=exact.sha(ready)
            argv=first.recipe_arguments(recipe)+['--run-id',runid,'--resume-weights',str(d/'original_converted.pt'),
                '--native-backend-provenance',str(ready),'--native-max-joint-bytes','1800000000','--bpe-candidate-dir',str(candidate),
                '--stop-file',str(parent/'stop.txt'),'--max-wall-seconds','10500']
            for key,path in assets.items():argv+=['--'+key.replace('_','-'),str(path)]
            source_identity={'source_scientific_arguments':recipe,'converted_weights_sha256':'synthetic_converted_digest'}
            proof={'kind':'HF_BPE512_actual_unsuppressed_GPU_source_equivalence_v1','passed':True,
                'actual_GPU_decode':True,'before_optimizer_creation':True,'decoded_clips':300,'run_id':runid,
                'source_identity':source_identity,'student_geometry':training.GEOMETRY,'converted_weights_sha256':source_identity['converted_weights_sha256'],
                'trainer_sha256':chain['train_lora_batched_bpe512_v2.py'],'candidate_runtime_sha256':chain['hf_bpe512_training_v2.py'],
                'candidate_tokenizer_sha256':training.TOKENIZER_SEMANTIC_SHA,'raw_transcript_changes':[],
                'token_duration_emission_changes':[],'source_gate':{'retention_gate_passed':True},'candidate_gate':{'retention_gate_passed':True}}
            proof_path=run/'BPE512_actual_GPU_source_equivalence.json';save(proof_path,proof)
            config={**recipe,'run_id':runid,'turkish_clips_used':74883,'replay_clips_used':1595,'tr_clips_per_update':8,'replay_clips_per_update':8,
                'native_backend_provenance':str(ready),'sampling':{'initial_pool':{'domain':{'fleurs':{'pool_clips':2037},'cv':{'pool_clips':72846}}}},
                'initial_domain_to_BPE_model_only_admission':source_identity,'warm_start':{'mode':'new_optimizer_warm_start'},'code_sha256':chain,
                'config_identity':{'training_arguments':{**recipe,'BPE_delta_multiplier':1.},'extended_tokenizer_sha256':training.TOKENIZER_SEMANTIC_SHA,
                    'domain_sampling_identity':sampling,'BPE512':{**training.GEOMETRY,'source_equivalence_sha256':exact.sha(proof_path)},
                    'input_sha256':{str(p):exact.sha(p) for p in [ready,*assets.values(),*[candidate/name for name in
                        ('tokenizer.json','added_pieces.json','row_migration_plan.json','candidate_frozen_before_measurement.json')]]}}}
            save(run/'run_config.json',config)
            latest=d/'work'/(runid+'_latest.pt');best=d/'work'/(runid+'_best.pt');latest.parent.mkdir()
            latest.write_bytes(b'Opaque synthetic full Adam sampler RNG checkpoint');best.write_bytes(b'Opaque synthetic best')
            def coverage(step):
                draws=step*8;nf=draws*3//10;counts={'fleurs':2037,'cv':72846};nd={'fleurs':nf,'cv':draws-nf}
                return {'epoch_clock':'minimum_domain_exposure','domain':{k:{'pool_clips':v,'draws':nd[k],
                    'unique_clips_seen':min(nd[k],v),'repeat_visits':nd[k]-min(nd[k],v)} for k,v in counts.items()},
                    'turkish_draws':draws,'turkish_unique_clips_seen':sum(min(nd[k],v) for k,v in counts.items()),
                    'minimum_domain_exposure_epochs':min(nd[k]/v for k,v in counts.items()),'all_Turkish_pool_sources_seen':False}
            def selection(step):return {'run_id':runid,'accepted':True,'completed_steps':step,'selected_step':0,'weights':str(best),'latest_weights':str(latest),
                'heldout_tests_evaluated':False,'stop_early_reason':'soft_max_wall_seconds','sampling_identity':sampling,
                'turkish_epochs_completed':coverage(step)['minimum_domain_exposure_epochs'],
                'turkish_epoch_semantics':'minimum_domain_exposure','Turkish_domain_coverage':coverage(step)}
            def history(step):return [{'step':i,'Turkish_domain_coverage':coverage(i)} for i in range(1,step+1)]
            save(run/'training_history.json',history(2));save(run/'selection.json',selection(2))
            paths={'native':str(ready),**{key:str(parent/(key+'.json')) for key in ('io','cache','memory')}}
            save(Path(paths['io']),{'status':'complete'});save(Path(paths['cache']),{'status':'complete','cache_reclamation':{'eligible_update_calls':2}})
            save(Path(paths['memory']),{'status':'complete'});save(ready.with_suffix('.completion.json'),{'status':'complete','sources_unchanged':True})
            state={'kind':'isolated_HF_BPE512_guarded_quality_arm_v2','status':'quality_window_ended','run_id':runid,'worker_arguments':argv,'paths':paths,
                'source_sha256':chain,'controller_sha256':exact.BPE_CONTROLLER_SHA,'child_exit_code':0,'completed_steps':2,'selected_step':0,
                'selection':exact.record(run/'selection.json'),'resource_review_required':False}
            save(parent/'loop_status.json',state)
            args=SimpleNamespace(source_loop=parent,run_id=runid,output_dir=successor,source_closed_confirmed=True,resource_review_cleared=False,window_hours=3.,poll_seconds=5.)
            plan=exact.source_plan(args,True)
            check(plan['runtime_memory_launcher']==exact.record(HERE/outer.RUNTIME_MEMORY_NAME),'Exact admission runtime record is part of actual plan')
            check(plan['initial_step']==2 and exact.sha(plan['archive_path'])==exact.sha(latest),'Closed preflight archives opaque full-state bytes unchanged')
            check(plan['configured_steps']==13009 and plan['source_config_identity']==config['config_identity'],'Original min-domain horizon and36-field science retained')
            check(plan['source_equivalence']==exact.record(proof_path),'Initial300proof preserved without new decode')
            check(outer.option(plan['worker_arguments'],'--native-backend-provenance')==str(ready) and '--resume-weights' not in plan['worker_arguments'],
                'Original ready preserved and full exact restore replaces warmweights')
            bad=copy.deepcopy(config);bad['config_identity']['training_arguments']['head_lr']=1e-4
            reject(lambda:exact.validate_config(bad,argv),'LR mutation rejected')
            bad=copy.deepcopy(config);bad['config_identity']['training_arguments']['BPE_delta_multiplier']=6.
            reject(lambda:exact.validate_config(bad,argv),'BPE-row multiplier mutation rejected')
            events=[]
            def mock_preload():
                events.append('preload');check('torch' not in sys.modules,'Preload precedesTorch')
                return {'kind':'coherent_native_cuda12_9_preload_v1','module_sha256':chain['preload_nemo_linker_v1.py'],'actual_binding_linker':{'version':[12,9]},'cuda_initialized':False}
            def mock_dispatch(path,run_name):
                events.append('dispatch');check(Path(path).name==outer.RUNTIME_MEMORY_NAME and events==['preload','dispatch'],'Fresh preloader dispatches exact-only memory after preload')
            sys.modules['preload_nemo_linker_v1']=SimpleNamespace(preload=mock_preload);outer.runpy.run_path=mock_dispatch
            sys.executable=str(ROOT/'work/nemo-native-env-v2/Scripts/python.exe')
            outer.main(SimpleNamespace(original_native_provenance=ready,original_native_provenance_sha256=ready_sha,
                fresh_runtime_provenance=Path(plan['paths']['fresh_runtime']),memory_arguments=plan['command'][plan['command'].index('--')+1:],validate_only=False))
            check(exact.sha(ready)==ready_sha,'Original ready bytes unchanged')
            started=datetime.now(timezone.utc).isoformat()
            owned={'kind':'owned_training_worker_identity_v1','expected_run_id':runid,'pid':os.getpid(),'ppid':42,
                'worker_arguments_sha256':first.guards.argument_sha(plan['worker_arguments']),'started_at_utc':started,
                'process_creation_identity':{'available':True,'pid':os.getpid(),
                    'creation_unix_seconds':datetime.now(timezone.utc).timestamp()},'BPE_explicit_asset_records':{}}
            names={'memory_launcher_sha256':outer.RUNTIME_MEMORY_NAME,'cache_launcher_sha256':'launch_with_cache_reclaim_bpe512_v2.py',
                'cache_helper_sha256':'cuda_cache_reclaim_v2.py','resource_helper_sha256':'resource_watchdog_v2.py',
                'resource_telemetry_sha256':'paired_windows_commit_telemetry_v1.py','frozen_worker_sha256':'train_lora_batched_bpe512_v2.py',
                'data_policy_sha256':'public_turkish_data_policy_v3.py'}
            for key,name in names.items():owned[key]=chain[name]
            for flag in ('--bpe-source-checkpoint','--bpe-source-selection','--bpe-source-closure','--bpe-conversion-report',
                         '--native-backend-provenance','--resume-exact'):
                p=Path(outer.option(plan['worker_arguments'],flag));owned['BPE_explicit_asset_records'][flag.removeprefix('--').replace('-','_')]=exact.record(p)
            exact.validate_identity(owned,plan,os.getpid(),started)
            check(True,'Actual exact-plan runtime source and explicit asset identity route')
            bad_owned=copy.deepcopy(owned);bad_owned['memory_launcher_sha256']=chain['launch_with_memory_guard_bpe512_v2.py']
            reject(lambda:exact.validate_identity(bad_owned,plan,os.getpid(),started),'Initial fresh-memory source cannot substitute exact admission')
            exact.validate_fresh_runtime(exact.read(plan['paths']['fresh_runtime'],{}),plan,owned)
            check(True,'Fresh PID evidence separate from immutable first-start ready')
            resumed=copy.deepcopy(config);resumed['warm_start']={'mode':'exact_resume','path':plan['archive_path'],
                'sha256':plan['snapshots']['latest']['snapshot']['sha256'],'source_run_id':runid,'source_step':2}
            save(run/'run_config.json',resumed);save(run/'training_history.json',history(3));save(run/'selection.json',selection(3))
            new={k:Path(p) for k,p in plan['paths'].items()}
            memory={'kind':'memory_guard_training_launcher_bpe512_v1','status':'complete','cap_bytes':8000000000,'cleanup_threshold_bytes':4000000000,
                'allocator_cap_applied':True,'training_math_changed':False,'backend_flags_changed':False,'all_phase_owned_memory_telemetry':{'thread_join_completed':True}}
            mapping={'memory_launcher_sha256':outer.RUNTIME_MEMORY_NAME,'cache_launcher_sha256':'launch_with_cache_reclaim_bpe512_v2.py','cache_helper_sha256':'cuda_cache_reclaim_v2.py',
                'resource_helper_sha256':'resource_watchdog_v2.py','frozen_worker_sha256':'train_lora_batched_bpe512_v2.py'}
            for key,name in mapping.items():memory[key]=memory[key+'_after']=chain[name]
            for key,name in {'resource_telemetry_sha256':'paired_windows_commit_telemetry_v1.py','data_policy_sha256':'public_turkish_data_policy_v3.py'}.items():
                memory[key]=memory[name.removesuffix('.py')+'_sha256_after']=chain[name]
            save(new['memory'],memory);cache={'kind':'cache_reclamation_training_launcher_bpe512_v1','status':'complete','frozen_worker_sha256':chain['train_lora_batched_bpe512_v2.py'],
                'cache_reclamation':{'eligible_update_calls':1,'failed_reclamations':0}}
            save(new['cache'],cache);save(new['io'],{'kind':'atomic_io_retry_bpe512_v1','status':'complete','worker_sha256':chain['train_lora_batched_bpe512_v2.py'],'worker_sha256_after':chain['train_lora_batched_bpe512_v2.py']})
            done=exact.verify_completion(plan,selection(3),owned)
            check(done['new_optimizer_end_calls']==1 and done['cumulative_completed_steps']==3 and done['source_equivalence_rerun'] is False,'Suffix-only hook count and inherited frontier/proof')
            cache['cache_reclamation']['eligible_update_calls']=3;save(new['cache'],cache)
            reject(lambda:exact.verify_completion(plan,selection(3),owned),'Cumulative hook count rejected')
            cache['cache_reclamation']['eligible_update_calls']=1;save(new['cache'],cache)
            corrupt=history(3);corrupt[0]['tampered']=True;save(run/'training_history.json',corrupt)
            reject(lambda:exact.verify_completion(plan,selection(3),owned),'Inherited history mutation rejected')
    finally:
        for (module,name),value in changed.items():setattr(module,name,value)
        outer.runpy.run_path=old_dispatch;sys.executable=old_executable
        if old_preload is None:sys.modules.pop('preload_nemo_linker_v1',None)
        else:sys.modules['preload_nemo_linker_v1']=old_preload
    check(supervisor.automatic_budget_handoff({'stop_early_reason':'soft_max_wall_seconds'},{'kind':supervisor.FIRST_KIND,'resource_review_required':False}),
        'First producer without category supports only actual soft budget')
    for category in ('retention','plateau','resource','manual'):
        check(not supervisor.automatic_budget_handoff({'stop_early_reason':'controller_stop_file'},{'resource_review_required':category=='resource','stop_category':category}),category+' cannot auto-resume')
    check(not any(n in sys.modules for n in ('torch','nemo','numpy','soundfile','transformers')),'No ML/audio imports or actualcheckpoint parsing/GPU')
    report={'kind':'BPE512_same_science_exact_resume_tiny_verification_v2','passed':True,'check_count':len(checks),'checks':checks,
        'synthetic_only':True,'actual_checkpoint_read_or_deserialization':False,'runtime_preload_and_dispatch_mocked':True,
        'actual_GPU_decode_or_training':False,'source_order_proof':order,
        'source_sha256':{Path(m.__file__).name:exact.sha(m.__file__) for m in (first,exact,outer,supervisor)},
        'created_at_utc':datetime.now(timezone.utc).isoformat()}
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({'passed':True,'checks':len(checks),'output':str(output)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
