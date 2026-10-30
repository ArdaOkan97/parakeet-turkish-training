"""Focused stdlib-only real dispatch fixture; all ML/process state is mocked."""
import argparse,ast,hashlib,importlib.util,json,math,os,random,sys,tempfile,time
from pathlib import Path
from types import ModuleType,SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(HERE))

def main(output):
    checks=[]
    def require(ok,message):
        if not ok:raise AssertionError(message)
    def check(name,ok):require(ok,name);checks.append(name)
    def reject(name,callback):
        try:callback()
        except ValueError:checks.append(name);return
        raise AssertionError(name)
    path=HERE/'run_TSC_uniform_majority_loralr_v1.py'
    spec=importlib.util.spec_from_file_location('uniform_dispatch_fixture',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    check('guard_imports_no_Torch', 'torch' not in sys.modules)
    pins=m.source_pins()
    worker_path=HERE/m.WORKER;tree=ast.parse(worker_path.read_text())
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    check('actual_scoped_source_callback_3GB',m.caller_line(worker_path)==m.EXPECTED_SYNC_LINE
          and m.ForkCaller(SimpleNamespace(),worker_path).threshold_bytes==3000000000)
    original=ast.parse((HERE/'train_lora_batched_tsc_uniform_full_v1.py').read_text())
    old={n.name:n for n in original.body if isinstance(n,ast.FunctionDef)}
    check('all207_serializer_and_Adam_sixglyph_update_formula_retained',
          ast.dump(functions['save_training_state'],include_attributes=False)==ast.dump(old['save_training_state'],include_attributes=False)
          and 'parameter[old_size:model.config.vocab_size]' in ast.unparse(functions['run'])
          and 'old + args.new_token_lr_multiplier * (updated - old)' in ast.unparse(functions['run']))
    run=ast.unparse(functions['run'])
    check('TR_streamed_no_fullarray_audit_textlist_or_pool_dump',
          'load_indexed_pool' in run and 'read_jsonl(args.tr_manifest)' not in run
          and 'audit_tsc_training_inputs' not in run and 'annotate_tr_frame_metadata' not in run
          and 'extend_processor_and_model(processor, model)' in run and 'write_pool_summary' in run)
    check('source207_Adam_BEST_and_no_PT_grace_or_evalclock_reset',
          'optimizer.load_state_dict(resume' in run and 'best, selected_step = (math.inf, None)' not in run and "next_eval = resume['next_eval_epoch']" in run
          and 'local_updates < 1000 or gate[\'retention_gate_passed\']' in run)
    for flag in ('early_stop_patience','target_turkish_wer'):
        branches=[n for n in ast.walk(functions['run']) if isinstance(n,ast.If) and flag in ast.unparse(n.test)]
        check('no_'+flag+'_before_fullunique',len(branches)==1 and 'full_pass' in ast.unparse(branches[0].test))
    check('exact_own_inherited_history_and_selected_all6_bound',
          'source_history_prefix_sha256' in ast.unparse(functions['validate_own_state'])
          and 'retention_gate_passed' in ast.unparse(functions['validate_own_state']))
    parent_tree=ast.parse((HERE/'train_lora_batched_tsc_uniform_full_v1.py').read_text())
    parent_functions={node.name:node for node in parent_tree.body if isinstance(node,ast.FunctionDef)}
    for name in ('lr_factor','development_gate','evaluate','save_training_state','native_loss_function','annotate_cache_lengths'):
        check('unchanged_'+name,ast.dump(functions[name],include_attributes=False)==ast.dump(parent_functions[name],include_attributes=False))
    def update_body(fn):
        loop=next(node for node in fn.body if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='step')
        end=next(i for i,node in enumerate(loop.body) if isinstance(node,ast.Expr) and ast.unparse(node)=='torch.cuda.synchronize()')
        return ast.dump(ast.Module(body=loop.body[:end+1],type_ignores=[]),include_attributes=False)
    check('whole_207_backward_clip_Adam_sixboost_native_update_AST_identical',update_body(functions['run'])==update_body(parent_functions['run']))
    check('actual_prior_index_reopened_no_fresh_admission', 'prepare_approved_TR_rows(' not in ast.unparse(functions['load_indexed_pool'])
        and 'context["indexed_audit"]' in ast.get_source_segment(worker_path.read_text(),functions['load_indexed_pool']))
    check('parent_all6_BEST_not_saved_as_failed_H', 'selected_step == start_step' in run and 'source_best_checkpoint' in run)
    check('density40_2_4_parser_defaults',all(text in ast.unparse(functions['parser']) for text in ('default=0.125','default=40.0','default=2.0','default=4.0')))
    check('record_function_not_shadowed_and_trace_path_keyword_fixed', 'record as file_record' in worker_path.read_text()
        and 'checkpoint_path=str(path)' in worker_path.read_text() and 'stage_memory(memory_trace, "before_atomic207_state_save", step=step, path=' not in run)
    startup=ast.unparse(functions['run'])
    check('all207_CPUaudit_GPUcopy_Adamload_and_release_before_added350',
        startup.index('audit_parent_state(')<startup.index('model.load_state_dict(source')
        <startup.index("optimizer.load_state_dict(source['optimizer'])")
        <startup.index("del source['weights'], source['optimizer']")<startup.index('equality = compare_source_development('))
    check('history_ownership_transfer_no_deepcopy',"history, start_step = (resume['history'], resume['step'])" in startup
        and 'deepcopy(resume' not in startup and 'deepcopy(source' not in startup)
    check('source350_proof_preupdate_counts_are_required',all(text in ast.unparse(next(n for n in ast.parse(path.read_text()).body
        if isinstance(n,ast.FunctionDef) and n.name=='verify_completion')) for text in ('before_first_majority_update','all_seven_ASR_counts_exact','source_receipt')))
    state_ns={'Path':Path,'math':math,'json':json,'hashlib':hashlib,'copy':__import__('copy'),
        'canonical_identity_sha':m.canonical}
    optimizer_audits=[]
    state_ns['audit_source_optimizer']=lambda state,context,expected,**kwargs:optimizer_audits.append((state['step'],context['source_selected_step'],len(expected)))
    subset=ast.Module(body=[functions['validate_own_state'],functions['load_indexed_pool']],type_ignores=[])
    exec(compile(ast.fix_missing_locations(subset),str(worker_path),'exec'),state_ns)
    inherited=[{'step':step} for step in range(1,7622)]
    inherited[7599]['validation']={'retention_gate_passed':True,'turkish_wer':.42}
    current_identity={'domain_sampling_identity':{'opaque_declared_boundary':7621}}
    base={'schema':2,'run_id':'synthetic_majority','step':7621,'config_identity':current_identity,
        'history':inherited,'sampler':{'updates':7621,'identity':current_identity['domain_sampling_identity']},
        'selected_step':7600,'best_score':.42}
    for key in ('optimizer','python_rng','torch_rng','cuda_rng','patience','retention_failures','next_eval_epoch',
                'train_wall_seconds','audio_seconds','language_audio_seconds','evaluation_wall_seconds','initial_gate'):
        base[key]='EXPLICIT_STATE_MOCK'
    context={'source_step':7621,'uniform_origin_step':6500,'configured_end_step':25109,
        'source_history_prefix_sha256':m.canonical(inherited)}
    model=SimpleNamespace(named_parameters=lambda:[(str(i),SimpleNamespace(requires_grad=True)) for i in range(207)])
    def validate_state(value):state_ns['validate_own_state'](value,SimpleNamespace(run_id='synthetic_majority'),context,current_identity,model=model,torch_module='NO_TORCH')
    validate_state(base)
    check('own_boundary_keeps_older_eligible_BEST_and_current207_clock',optimizer_audits[-1]==(7621,7621,207))
    for name,update in (('identity',{'config_identity':{}}),('step',{'step':7620}),
        ('unqualified_selected',{'selected_step':7610}),('old_five6500_selected',{'selected_step':6500}),
        ('missing_optimizer',{'optimizer':None})):
        value=__import__('copy').deepcopy(base);value.update(update)
        if name=='missing_optimizer':del value['optimizer']
        reject('own_state_rejects_'+name,lambda value=value:validate_state(value))
    value=__import__('copy').deepcopy(base);value['history'][0]['old_lr']=0
    reject('own_state_rejects_rewritten_inherited_history',lambda:validate_state(value))
    value=__import__('copy').deepcopy(base);value.update(selected_step=None,best_score=math.inf)
    validate_state(value);check('own_none_best_is_not_oldfive_champion',value['selected_step'] is None)
    value.update(best_score=.39);reject('own_none_best_rejects_finite_score',lambda:validate_state(value))
    reopened=[];writes=[]
    state_ns['reopen_approved_TR_rows']=lambda args,directory,audit,**kwargs:(reopened.append((directory,audit,kwargs)) or 'INDEX',audit)
    state_ns['prepare_approved_TR_rows']=lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('No wholepool readmission'))
    state_ns['write_json']=lambda path,value:writes.append((path,value))
    immutable_audit={'identity':{'original_approved':True},'complete':True}
    rows,audit=state_ns['load_indexed_pool'](SimpleNamespace(resume_exact=None,indexed_tr_directory='IMMUTABLE_INDEX'),None,ROOT/'work','UNUSED_HELDOUT',{'indexed_audit':immutable_audit})
    check('actual_initial_index_reopen_uses256_cache_and_keeps_original_audit',rows=='INDEX' and reopened[-1][2]=={'cache_size':256}
        and audit==immutable_audit and audit is not immutable_audit and len(writes)==1)
    current={'available':True,'commit_used_bytes':31500000000,'commit_limit_bytes':41340000000,
             'commit_available_bytes':9840000000,'commit_fraction':31500000000/41340000000,
             'physical_available_bytes':13500000000}
    actual_sample=m.sample_memory;m.sample_memory=lambda *a:{'system_memory':current}
    check('first_material_lazy_measurement_requires8GB_not_predecessor_fit_claim',
          m.current_memory_admission(first_measurement=True)['predecessor_peak_used_as_new_policy_fit_predictor'] is False)
    for key in ('commit_available_bytes','physical_available_bytes'):
        prior=current[key];current[key]=7999999999
        reject('first_bootstrap_rejects_low_'+key,lambda:m.current_memory_admission(first_measurement=True));current[key]=prior
    m.sample_memory=actual_sample
    events=[];fake_torch=ModuleType('torch')
    fake_torch.cuda=SimpleNamespace(get_device_properties=lambda d:SimpleNamespace(total_memory=16000000000),
        set_per_process_memory_fraction=lambda f,device:events.append(('cap',f)),get_rng_state_all=lambda:['MOCK_RNG'])
    fake_torch.get_rng_state=lambda:'MOCK_CPU_RNG'
    fake_torch.save=lambda state,target:Path(target).write_text(json.dumps(state))
    fake_native=ModuleType('native_tdt_checkpointed_v4')
    fake_native.configuration_identity=lambda:{'actual_native_math':False}
    fake_native.cache_cleanup_evidence=lambda:{'available':False,'mock':True}
    fake_preload=ModuleType('preload_nemo_linker_v1')
    def preload():
        require('torch' not in sys.modules,'Preload must precede Torch')
        events.append(('preload',));sys.modules['torch']=fake_torch
        return {'kind':'coherent_native_cuda12_9_preload_v1','actual_binding_linker':{'version':[12,9]},
            'cuda_initialized':False,'module_sha256':m.sha(HERE/'preload_nemo_linker_v1.py'),
            'successful_launcher_sha256':m.sha(HERE/'launch_nemo_linker_v4.py')}
    fake_preload.preload=preload
    fake_benchmark=ModuleType('benchmark');fake_benchmark.write_json=m.write
    fake_worker=ModuleType(Path(m.WORKER).stem)
    ns={'__file__':str(worker_path),'OUTPUT':HERE,'ROOT':ROOT,'Path':Path,'json':json,'os':os,
        'torch':fake_torch,'random':random,'file_sha':m.sha,'checkpoint_state':lambda model:{str(i):'OPAQUE' for i in range(207)},
        'encoder_lora_metadata':lambda model:{'all24':True}}
    subset=ast.Module(body=[functions['native_loss_identity'],functions['save_training_state']],type_ignores=[])
    exec(compile(ast.fix_missing_locations(subset),str(worker_path),'exec'),ns)
    fake_worker.save_training_state=ns['save_training_state'];fake_worker.write_json=m.write
    originals=(fake_worker.save_training_state,fake_worker.write_json)
    modules={name:sys.modules.get(name) for name in ('torch','native_tdt_checkpointed_v4','preload_nemo_linker_v1','benchmark',fake_worker.__name__)}
    old_executable=sys.executable
    class Evidence:
        def __init__(self,*args,**kwargs):pass
        def start(self):events.append(('telemetry_start',))
        def stop_and_join(self):events.append(('join',))
        def install(self):events.append(('caller_install',))
        def restore(self):events.append(('caller_restore',))
        def evidence(self):return {'eligible_update_calls':0,'thread_join_completed':True,'mock':True}
    capacity=m.shared_memory.validate_capacity
    m.shared_memory=SimpleNamespace(__file__=str(HERE/'launch_with_memory_guard_TSC_resume_v3.py'),
        validate_capacity=capacity,creation_identity=lambda pid:{'available':True,'pid':pid},
        safe_sample=lambda pid,phase:{'observed_monotonic':time.monotonic(),'mock':True},
        UpdateTelemetry=Evidence,LiveTelemetrySampler=Evidence,telemetry_reclaimer_class=lambda *args:Evidence)
    try:
        sys.modules.update({'native_tdt_checkpointed_v4':fake_native,'preload_nemo_linker_v1':fake_preload,
            'benchmark':fake_benchmark,fake_worker.__name__:fake_worker})
        sys.executable=str(ROOT/'work/nemo-native-env-v2/Scripts/python.exe')
        with tempfile.TemporaryDirectory(dir=ROOT/'work/research',prefix='uniform_dispatch_') as directory:
            temp=Path(directory);asset=temp/'opaque.json';asset.write_text('{}')
            original_ready=None
            for exact in (False,True):
                sys.modules.pop('torch',None)
                folder=temp/('exact' if exact else 'first');folder.mkdir()
                paths={key:folder/(key+'.json') for key in ('native','identity','memory','cache','io','fresh_runtime')}
                paths.update(log=folder/'log.txt',stop=folder/'stop.txt')
                if exact:paths['native']=original_ready
                runtime=paths['fresh_runtime'] if exact else paths['native']
                argv=['--run-id','opaque_uniform_fixture','--native-backend-provenance',str(paths['native']),
                      '--uniform-full-runtime-proof',str(runtime)]
                for flag in ('--tsc-input-proof','--tsc-training-approval','--source-closure','--tr-manifest','--replay-manifest',
                    '--tr-frame-metadata','--tr-frame-metadata-proof','--data-v3-proof','--uniform-full-source-receipt','--majority-source-receipt',
                    '--portuguese-preparation-proof','--portuguese-baseline-proof','--portuguese-replay-manifest','--portuguese-dev-manifest',
                    '--resume-exact' if exact else '--resume-majority'):
                    argv.extend((flag,str(asset)))
                plan={'worker_arguments':argv,'source_sha256':pins,'run_id':'opaque_uniform_fixture',
                      'worker_arguments_sha256':'EXPLICIT_ARGUMENT_MOCK'}
                if exact:plan.update(continuation_mode='same_new_run_exact_budget_continuation',original_ready=m.record(original_ready))
                m.validate_plan=lambda p:paths # Actual source admission deliberately not executed.
                args=SimpleNamespace(run_id=plan['run_id'],tdt_implementation='nemo',native_backend_provenance=paths['native'],
                    native_max_joint_bytes=1700000000,uniform_full_runtime_proof=runtime,resume_exact=asset if exact else None)
                fake_worker.parser=lambda:SimpleNamespace(parse_args=lambda v:args)
                fake_worker.validate_arguments=lambda a:(ns['native_loss_identity'](a),events.append(('validate',)),a)[2]
                def run(a):
                    events.append(('run',))
                    fake_worker.save_training_state(folder/'opaque207.pt','MODEL',SimpleNamespace(state_dict=lambda:{'states':207}),
                        SimpleNamespace(state_dict=lambda:{'uniform':True}),run_id=a.run_id,step=6500,config_identity={'tuple':(.9,.999)},
                        history=[{'step':6500}],best_score=math.inf,selected_step=None,patience=0,retention_failures=0,next_eval_epoch=.25,
                        train_wall_seconds=1.,audio_seconds=1.,language_audio_seconds={'tr':1.,'pt':0.},evaluation_wall_seconds=1.,
                        initial_gate={'old_five_lineage':True})
                fake_worker.run=run;m.run_worker(plan)
                saved=json.loads((folder/'opaque207.pt').read_text())
                check(('exact' if exact else 'first')+'_actual_dispatch_207_serializer_no_all6_initial_BEST',
                    saved['schema']==2 and len(saved['weights'])==207 and saved['optimizer']['states']==207
                    and saved['selected_step'] is None and saved['sampler']['uniform'] is True)
                check(('exact' if exact else 'first')+'_runtime_complete_cap8_end3_and_restoration',
                    json.loads(paths['memory'].read_text())['cleanup_threshold_bytes']==3000000000
                    and json.loads(runtime.with_suffix('.completion.json').read_text())['status']=='complete'
                    and fake_worker.save_training_state is originals[0] and fake_worker.write_json is originals[1])
                check(('exact' if exact else 'first')+'_fresh_PID_lazypin_and_provenance_verified',
                    json.loads(runtime.read_text())['allocation_policy']['lazy_helper_sha256']==m.LAZY_ROWS_SHA)
                if not exact:original_ready=paths['native'];original_hash=m.sha(original_ready)
                else:check('exact_original_native_token_not_rewritten',m.sha(original_ready)==original_hash)
            check('actual_dispatch_preload_cap_validate_then_run',events.index(('preload',))<events.index(('cap',.5))
                <events.index(('validate',))<events.index(('run',)))
            paths['stop'].write_text('manual original');m.request_stop(paths['stop'],'budget')
            check('manual_stop_not_rewritten_budget',paths['stop'].read_text()=='manual original')
            for status in ('resource','manual','retention'):
                reject('no_'+status+'_automatic_successor',lambda status=status:m.exact_successor(
                    {'status':'train_window_ended','child_exit_code':0,'resource_review_required':status=='resource','stop_category':status},
                    temp/'unused',SimpleNamespace(run_id='opaque_uniform_fixture')))
    finally:
        sys.executable=old_executable
        for name,module in modules.items():
            if module is None:sys.modules.pop(name,None)
            else:sys.modules[name]=module
    require('torch' not in sys.modules,'No actual Torch import')
    result={'kind':'TSC_uniform_majority_loralr_stdlib_dispatch_verification_v1','passed':True,'checks':checks,'checks_passed':len(checks),
        'source_sha256':{path.name:m.sha(path),worker_path.name:m.sha(worker_path)},
        'actual_GPU_model_source_PT_admission_executed':False,'actual_parent_checkpoint_read':False,'all_ML_and_source_frontier_fields_explicit_mocks':True}
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({'passed':True,'checks':len(checks),'output':str(output)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
