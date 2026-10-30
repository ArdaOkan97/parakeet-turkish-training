"""Isolated exact-branch fixture. Opaque state, CUDA, ML run and native/Win32 calls are mocked."""
from pathlib import Path
import argparse,ast,builtins,json,math,os,shutil,sys,tempfile,time,types
REAL_ROOT=Path(__file__).resolve().parent.parent
REAL_HERE=REAL_ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(REAL_HERE))
import run_TSC_headlr_fork_v1 as c
import tsc_optimization_transition_v1 as transition


def coverage(step):
    draws=step*8;nf=draws*3//10;no=draws-nf
    def group(n,p):return {'draws':n,'pool_clips':p,'unique_clips_seen':min(n,p),'repeat_visits':n-min(n,p)}
    epoch=min(nf/2037,no/251761)
    return {'epoch_clock':'minimum_domain_exposure','domain':{'fleurs':group(nf,2037),'cv_tsc':group(no,251761)},
        'source_groups':{'fleurs':group(nf,2037),'cv':group(no,72846),'tsc':group(0,178915)},
        'turkish_draws':draws,'turkish_unique_clips_seen':min(nf,2037)+no,
        'minimum_domain_exposure_epochs':epoch,'all_Turkish_pool_sources_seen':False}


def main(output):
    checks=[]
    def check(ok,label):
        if not ok:raise AssertionError(label)
        checks.append(label)
    def reject(fn,label):
        try:fn()
        except (ValueError,KeyError):checks.append(label);return
        raise AssertionError('Accepted '+label)
    # Read sources only. No actual runs, manifests or checkpoint paths are opened.
    real_controller=c.record(c.__file__);real_worker=c.record(REAL_HERE/c.WORKER)
    names=(c.WORKER,c.TRANSITION,Path(c.__file__).name,'cuda_cache_reclaim_v2.py','resource_watchdog_v2.py',
        'paired_windows_commit_telemetry_v1.py','public_turkish_data_policy_v3.py','preload_nemo_linker_v1.py',
        'launch_with_memory_guard_TSC_resume_v3.py','atomic_io_retry_v1.py')
    pins={name:c.sha(REAL_HERE/name) for name in names}
    originals={'ROOT':c.ROOT,'HERE':c.HERE,'source_pins':c.source_pins,
        'creation':c.shared_memory.creation_identity,'safe_sample':c.shared_memory.safe_sample,
        'sampler':c.shared_memory.LiveTelemetrySampler,'telemetry':c.shared_memory.UpdateTelemetry}
    original_import=builtins.__import__
    module_originals={}
    try:
        with tempfile.TemporaryDirectory(prefix='TSC_headlr_exact_fixture_',dir=REAL_ROOT/'work') as temporary:
            root=Path(temporary);here=root/'outputs/parakeet_turkish';here.mkdir(parents=True);(root/'work').mkdir()
            for name in names:shutil.copyfile(REAL_HERE/name,here/name)
            c.ROOT=root;c.HERE=here;c.source_pins=lambda:dict(pins)
            rid='fixture_TSC_headlr_same';run=here/'runs'/rid;run.mkdir(parents=True)
            first=here/'runs'/'fixture_closed_first';first.mkdir();next_loop=here/'runs'/'fixture_exact_next'
            asset=root/'work'/'opaque_asset.json';c.write(asset,{'fixture':True})
            ready=first/'original_native.json';c.write(ready,{'kind':'native_tdt_training_backend_v3','pid':111,'immutable_fixture':True})
            ready_before=c.record(ready);receipt=c.record(asset)
            recipe={'head_lr':1e-5,'steps':44958,'batch_size':8,'accumulation':2,'eval_every_updates':250}
            identity={'training_arguments':recipe,'optimization_transition':{
                'kind':'declared_TSC_head_LR_full_state_fork_v1','first_changed_lr_step':1501,'head_lr_to':1e-5,
                'fresh_optimizer_sampler_or_warmup':False}}
            config={'run_id':rid,'native_backend_provenance':str(ready),'config_identity':identity,
                'epochs':1.,'steps':44958,'turkish_clips_used':253798,'replay_clips_used':1595,
                'tr_clips_per_update':8,'replay_clips_per_update':8,'tsc_input_proof':str(asset),'tsc_training_approval':str(asset),
                'sampling':{'initial_pool':{'domain':{'fleurs':{'pool_clips':2037},'cv_tsc':{'pool_clips':251761}}}}}
            history=[{'step':step} for step in range(1,2001)]
            for step in (1500,2000):
                cov=coverage(step);history[step-1].update(turkish_clips_seen=cov['turkish_draws'],
                    turkish_unique_clips_seen=cov['turkish_unique_clips_seen'],Turkish_domain_coverage=cov)
            proof={'kind':'TSC_headlr_before_update_source_equivalence_v1','passed':True,'run_id':rid,
                'compared_fixed_DEV_utterances':300,'RNG_restored_after_added_evaluation':True,'source_receipt':receipt,
                'state_boundary':{'history_prefix_sha256':c.canonical(history[:1500])}}
            c.write(run/transition.PROOF_NAME,proof);proof_before=c.record(run/transition.PROOF_NAME)
            c.write(run/'run_config.json',config);c.write(run/'training_history.json',history)
            latest=root/'work'/f'{rid}_latest.pt';best=root/'work'/f'{rid}_best.pt'
            latest.write_bytes(b'opaque-complete-own-latest-2000-Adam-sampler-RNG-counters');best.write_bytes(b'opaque-own-BEST-1500')
            cov=coverage(2000)
            selection={'run_id':rid,'accepted':True,'heldout_tests_evaluated':False,'completed_steps':2000,'selected_step':1500,
                'stop_early_reason':'controller_stop_file','latest_weights':str(latest),'weights':str(best),
                'Turkish_domain_coverage':cov,'turkish_epochs_completed':cov['minimum_domain_exposure_epochs'],
                'turkish_epoch_semantics':'minimum_domain_exposure'}
            c.write(run/'selection.json',selection)
            paths={key:str(first/(key+'.json')) for key in ('identity','memory','cache','io','stop','log')};paths['native']=str(ready)
            argv=['--run-id',rid,'--head-lr','1e-05','--batch-size','8','--accumulation','2','--steps','44958',
                '--eval-every-updates','250','--replay-seed','20261012','--resume-optimization',str(asset),
                '--optimization-source-receipt',str(asset),'--native-backend-provenance',str(ready),
                '--stop-file',paths['stop'],'--max-wall-seconds','10500']
            for flag in ('--tsc-input-proof','--tsc-training-approval','--source-closure','--tr-manifest',
                '--replay-manifest','--tr-frame-metadata','--tr-frame-metadata-proof','--data-v3-proof'):
                argv.extend((flag,str(asset)))
            plan={'kind':'declared_TSC_headlr_full_state_fork_controller_v1','run_id':rid,'loop_id':first.name,
                'controller_sha256':c.sha(c.__file__),'source_sha256':pins,'source_closure':receipt,
                'source_checkpoint':c.record(asset),'paths':paths,'worker_arguments':argv,
                'worker_arguments_sha256':c.domain.guards.argument_sha(argv),'actual_stage_recipe':recipe,
                'command':['mock-python',str(c.__file__)],'inherited_step':1500,
                'configured_steps':44958,'combined_TR_clips':253798,'domain_pool_sizes':{'fleurs':2037,'cv_tsc':251761},
                'TSC_input_proof':receipt,'TSC_training_approval':receipt,'TSC_quarantine_proof':receipt}
            # Completed first-window producer metadata is synthetic, checked by actual consumer functions.
            started=c.now();pid=12345;args_sha=plan['worker_arguments_sha256']
            worker_id={'kind':'owned_training_worker_identity_v1','expected_run_id':rid,'pid':pid,'ppid':23456,
                'started_at_utc':started,'worker_arguments_sha256':args_sha,'actual_combined_dispatch_source':c.record(c.__file__),
                'process_creation_identity':{'available':True,'pid':pid,'creation_unix_seconds':c.domain.guards.stamp(started).timestamp()}}
            for key,name in {'memory_launcher_sha256':Path(c.__file__).name,'cache_launcher_sha256':Path(c.__file__).name,
                'cache_helper_sha256':'cuda_cache_reclaim_v2.py','resource_helper_sha256':'resource_watchdog_v2.py',
                'resource_telemetry_sha256':'paired_windows_commit_telemetry_v1.py','frozen_worker_sha256':c.WORKER,
                'data_policy_sha256':'public_turkish_data_policy_v3.py'}.items():worker_id[key]=pins[name]
            c.write(paths['identity'],worker_id)
            for key,kind in (('io','TSC_headlr_fork_atomic_IO_v1'),('cache','TSC_headlr_fork_cache_runtime_v1'),('memory','TSC_headlr_fork_memory_runtime_v1')):
                value={'kind':kind,'status':'complete','sources_unchanged':True}
                if key=='cache':value['cache_reclamation']={'eligible_update_calls':500}
                if key=='memory':value.update(cap_bytes=8_000_000_000,cleanup_threshold_bytes=4_000_000_000,
                    allocator_cap_applied=True,all_phase_owned_memory_telemetry={'thread_join_completed':True})
                c.write(paths[key],value)
            c.write(ready.with_suffix('.completion.json'),{'kind':'TSC_headlr_fork_native_completion_v1','status':'complete',
                'pid':pid,'run_id':rid,'sources_unchanged':True,'backend_ready_sha256':c.sha(ready),
                'fresh_runtime_sha256':c.sha(ready),'actual_dispatch_source_sha256':c.sha(c.__file__)})
            state={**plan,'status':'train_window_ended','child_exit_code':0,'resource_review_required':False,
                'stop_category':'budget','completed_steps':2000,'selected_step':1500,'selection':c.record(run/'selection.json'),
                'owned_launcher_pid':23456,'started_at_utc':started}
            c.write(first/'plan.json',plan);c.write(first/'loop_status.json',state)
            args=types.SimpleNamespace(window_hours=3.,poll_seconds=5.,output_dir=next_loop,resume_closed_loop=first,run_id=rid)
            actual_plan=c.prepare_exact(args)
            check(actual_plan['inherited_step']==2000 and actual_plan['original_ready']==ready_before,'Actual prepare_exact binds closed frontier and original ready')
            check(c.option(actual_plan['worker_arguments'],'--native-backend-provenance')==str(ready)
                and '--resume-optimization' not in actual_plan['worker_arguments'],'Exact argv retains original ready, removes initial fork mode')
            check(actual_plan['preserved_scientific_config_identity_sha256']==c.canonical(identity),'Exact scientific identity bound without rewriting')
            for label,changes in (('retention',{'stop_category':'retention'}),('resource',{'resource_review_required':True,'status':'resource_review_required'}),
                ('horizon',{'stop_category':'configured_steps_complete','completed_steps':44958})):
                c.write(first/'loop_status.json',{**state,**changes});reject(lambda:c.prepare_exact(args),'Actual prepare_exact rejects '+label+' closure')
            c.write(first/'loop_status.json',state)
            # Exercise existing main's closed byte-copy branch and real verify_completion; owned child is mocked.
            events=[];cuda=types.SimpleNamespace();reserved=[5_000_000_000]
            cuda.synchronize=lambda:events.append('synchronize') or 42
            cuda.memory_allocated=lambda:2_800_000_000;cuda.memory_reserved=lambda:reserved[0];cuda.max_memory_allocated=lambda:6_000_000_000
            def clear():events.append('empty_cache');reserved[0]=3_000_000_000
            cuda.empty_cache=clear;cuda.get_device_properties=lambda index:types.SimpleNamespace(total_memory=16_000_000_000)
            cuda.set_per_process_memory_fraction=lambda fraction,device:events.append(('cap',fraction,device))
            fake_torch=types.SimpleNamespace(cuda=cuda)
            tree=ast.parse((here/c.WORKER).read_text());functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('parser','validate_arguments')]
            ns={'argparse':argparse,'Path':Path,'DATA':root/'work','OUTPUT':here,'math':math,
                'native_loss_identity':lambda a:None,'closed_source_identity':lambda a:None}
            exec(compile(ast.Module(body=functions,type_ignores=[]),str(here/c.WORKER),'exec'),ns)
            worker=types.ModuleType(Path(c.WORKER).stem);worker.parser=ns['parser'];worker.validate_arguments=ns['validate_arguments']
            worker.write_json=lambda path,value:Path(path).write_text(json.dumps(value))
            worker.save_training_state=lambda path,value:Path(path).write_bytes(value)
            def after_update(parsed):
                check(parsed.resume_exact is not None and parsed.resume_optimization is None,'Actual nested dispatch parses exact-only restore')
                check(Path(parsed.resume_exact).read_bytes()==latest.read_bytes(),'Actual archived checkpoint candidate is same opaque bytes')
                newcov=coverage(2001);newhistory=[*history,{'step':2001,'turkish_clips_seen':newcov['turkish_draws'],
                    'turkish_unique_clips_seen':newcov['turkish_unique_clips_seen'],'Turkish_domain_coverage':newcov}]
                worker.write_json(run/'training_history.json',newhistory)
                worker.write_json(run/'selection.json',{**selection,'completed_steps':2001,'Turkish_domain_coverage':newcov,
                    'turkish_epochs_completed':newcov['minimum_domain_exposure_epochs']})
            code='\n'*579+'def run(args):\n    torch.cuda.synchronize()\n    after_update(args)\n'
            rns={'torch':fake_torch,'after_update':after_update};exec(compile(code,str(here/c.WORKER),'exec'),rns);worker.run=rns['run']
            benchmark=types.ModuleType('benchmark');benchmark.write_json=worker.write_json
            preload=types.ModuleType('preload_nemo_linker_v1');preload.preload=lambda:events.append('preload') or {
                'kind':'coherent_native_cuda12_9_preload_v1','actual_binding_linker':{'version':[12,9]},'cuda_initialized':False}
            sample=lambda pid,phase:{'observed_monotonic':time.monotonic(),'observed_at_utc':c.now(),'phase':phase,
                'system_memory':{},'worker_private_memory':{},'calling_process_commit_memory':{}}
            c.shared_memory.safe_sample=sample
            c.shared_memory.creation_identity=lambda pid:{'available':True,'pid':pid,'creation_unix_seconds':time.time()}
            c.shared_memory.LiveTelemetrySampler=lambda *a,**k:types.SimpleNamespace(start=lambda:None,stop_and_join=lambda:None,
                evidence=lambda:{'thread_join_completed':True})
            c.shared_memory.UpdateTelemetry=lambda pid,sink:originals['telemetry'](pid,sink=sink,sample=sample)
            for name,module in {'benchmark':benchmark,'preload_nemo_linker_v1':preload,worker.__name__:worker}.items():
                module_originals[name]=sys.modules.get(name);sys.modules[name]=module
            def importer(name,*values,**kwargs):
                if name=='torch':events.append('mock_torch_import');return fake_torch
                return original_import(name,*values,**kwargs)
            builtins.__import__=importer
            original_awake=c.domain.guards.prior.keep_awake;original_window=c.run_window
            def mock_window(current,deadline,arguments):
                archive=Path(current['source_checkpoint']['path'])
                check(archive.read_bytes()==latest.read_bytes(),'Actual main closed-latest archive branch preserves file bytes')
                c.run_worker(current)
                current['actual_runtime_completion']=c.verify_completion(current,c.read(run/'selection.json'))
                return 0
            c.domain.guards.prior.keep_awake=lambda value:{'mocked':True};c.run_window=mock_window
            args.worker_plan=None;args.source_loop=None;args.source_receipt=None;args.validate_only=False
            args.preflight_output=None;args.gpu_window_authorized=True
            try:result=c.main(args)
            finally:c.domain.guards.prior.keep_awake=original_awake;c.run_window=original_window
            check(result==0,'Actual prepare_exact -> main archive -> fresh exact run_worker -> verify_completion completes')
            next_plan=c.read(next_loop/'plan.json');done=c.read(next_loop/'loop_status.json')['actual_runtime_completion']
            fresh=c.read(next_plan['paths']['fresh_runtime']);completion=c.read(Path(next_plan['paths']['fresh_runtime']).with_suffix('.completion.json'))
            check(done['newly_trained_updates']==1 and done['inherited_steps']==2000,'Completion counts only one actual mocked suffix update')
            check(c.read(next_plan['paths']['cache'])['cache_reclamation']['eligible_update_calls']==1,'Scoped optimizer hook suffix count is one')
            check(fresh['pid']==os.getpid()!=111 and completion['pid']==fresh['pid'],'Fresh exact proof/completion use current fixture PID, not original ready PID')
            check(events.index('preload')<events.index('mock_torch_import'),'Fresh exact native preload precedes mocked Torch import')
            check(c.record(ready)==ready_before and c.record(run/transition.PROOF_NAME)==proof_before,'Original ready and initial300 proof remain byte-identical')
            check(c.read(run/'run_config.json')['config_identity']==identity,'Scientific config identity remains unchanged')
            check(latest.read_bytes()==b'opaque-complete-own-latest-2000-Adam-sampler-RNG-counters'
                and best.read_bytes()==b'opaque-own-BEST-1500','Archived source latest and existing BEST remain unchanged')
    finally:
        builtins.__import__=original_import;c.ROOT=originals['ROOT'];c.HERE=originals['HERE'];c.source_pins=originals['source_pins']
        c.shared_memory.creation_identity=originals['creation'];c.shared_memory.safe_sample=originals['safe_sample']
        c.shared_memory.LiveTelemetrySampler=originals['sampler'];c.shared_memory.UpdateTelemetry=originals['telemetry']
        for name,module in module_originals.items():
            if module is None:sys.modules.pop(name,None)
            else:sys.modules[name]=module
    check('torch' not in sys.modules and not any(name.startswith('nemo') for name in sys.modules),'No real ML/CUDA imports or checkpoint parsing')
    check(c.record(c.__file__)==real_controller and c.record(REAL_HERE/c.WORKER)==real_worker,'Frozen actual controller/worker source hashes unchanged')
    output=Path(output).resolve();c.require(not output.exists(),'Fresh fixture output required')
    c.write(output,{'schema':1,'kind':'TSC_headlr_same_file_exact_flow_mocked_CPU_verification_v1','passed':True,
        'checks':len(checks),'check_names':checks,'controller':real_controller,'worker':real_worker,
        'fixture':c.record(__file__),'isolated_ROOT_and_HERE_under_work_temporary_directory':True,
        'actual_GPU_checkpoint_state_or_hardware_restore_verified':False,
        'mocks':['Opaque checkpoint file bytes, not tensors/Adam/RNG deserialization','ML worker.run','CUDA','DLL preload','Win32 memory/creation telemetry',
            'Prior closed runtime evidence and300 source-equivalence proof','Owned child invocation; no actual GPU subprocess',
            'External ROOT/source admission in real parsed worker.validate_arguments'],
        'actual_existing_functions_exercised':['prepare_exact','validate_plan','main closed-latest archive branch','run_worker',
            'source-backed run581 inherited cache caller','real parser/validate_arguments with external admission mocks','verify_completion','verify_pool_frontier']})
    print(json.dumps({'passed':True,'checks':len(checks),'output':str(output)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
