"""One focused stdlib route fixture; actual admission/state/ML are not executed."""
import argparse
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace

ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(HERE))

def main(output):
    p=HERE/'run_TSC_bpe_shared_head_exact_v1.py'
    spec=importlib.util.spec_from_file_location('shared_exact_fixture',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    import tsc_bpe_shared_head_exact_transition_v1 as h
    checks=[]
    def check(ok,name):
        if not ok:raise AssertionError(name)
        checks.append(name)
    def reject(callback,name):
        try:callback()
        except (ValueError,KeyError):checks.append(name);return
        raise AssertionError(name+' accepted')
    check('torch' not in sys.modules,'stdlib_import_no_Torch')
    m.finalized_core();m.load_support()
    check(m.caller_line(HERE/m.WORKER)==m.EXPECTED_SYNC_LINE,'actual_worker_source_and_completed_update_callback_pin')
    old=ast.parse((HERE/'train_lora_batched_tsc_bpe_shared_head_memory_v2.py').read_text())
    new=ast.parse((HERE/m.WORKER).read_text())
    before={n.name:n for n in old.body if isinstance(n,ast.FunctionDef)};after={n.name:n for n in new.body if isinstance(n,ast.FunctionDef)}
    for name in ('batch_objective','native_loss_function','save_training_state'):
        check(ast.dump(before[name],include_attributes=False)==ast.dump(after[name],include_attributes=False),'unchanged_'+name)
    a=copy.deepcopy(next(n for n in before['run'].body if isinstance(n,ast.For) and ast.unparse(n.iter).startswith('range(start_step + 1')))
    b=copy.deepcopy(next(n for n in after['run'].body if isinstance(n,ast.For) and ast.unparse(n.iter).startswith('range(resume_step + 1')))
    a.iter=b.iter
    check(ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),'entire_native_KD_clip15_twoAdam_update_and_DEV_loop_unchanged_except_resume_start')
    source=(HERE/m.WORKER).read_text()
    check('initial_source_equivalence(' not in source and 'save(best_path, start_step' not in source
        and 'write_json(run_dir / "run_config.json"' not in source,'no_source300_BEST_or_initial_config_reset')
    check(source.index('del original, source, runtime, baseline_processor')<source.index('own_state = torch.load')
        and 'runtime = restore_runtime(adaptation, sampler, own_state' in source,
        'old_pair_CPU_objects_released_before_current_own_state_and_current_returned_runtime_only')
    history=[{'step':i} for i in range(1,10)]
    history.append({'step':10,'source_phase1_step':8,'adaptation_updates_completed':2})
    check(m.adaptation_suffix(history,9)[0]['step']==1 and history[-1]['step']==10,
        'fresh_window_progress_view_keeps_saved_cumulative_clock')
    reject(lambda:m.check_phase_arguments(['--resume-weights','x']), 'reject_reset_mode_before_dispatch')

    events=[];fake_torch=ModuleType('torch')
    fake_torch.cuda=SimpleNamespace(get_device_properties=lambda device:SimpleNamespace(total_memory=16000000000),
        set_per_process_memory_fraction=lambda fraction,device=0:events.append(('cap',fraction,device)),
        get_rng_state_all=lambda:['CUDA_RNG_OPAQUE'])
    fake_torch.get_rng_state=lambda:'CPU_RNG_OPAQUE'
    fake_torch.save=lambda state,target:Path(target).write_text(json.dumps(state),encoding='utf-8')
    native=ModuleType('native_tdt_checkpointed_v4');native.cache_cleanup_evidence=lambda:{'explicit_mock':True}
    preload_module=ModuleType('preload_nemo_linker_v1')
    def preload():
        check('torch' not in sys.modules,'coherent_preload_precedes_mock_Torch_import')
        events.append(('preload',));sys.modules['torch']=fake_torch
        return {'kind':'coherent_native_cuda12_9_preload_v1','actual_binding_linker':{'version':[12,9]},
            'cuda_initialized':False,'module_sha256':m.sha(HERE/'preload_nemo_linker_v1.py'),
            'successful_launcher_sha256':m.sha(HERE/'launch_nemo_linker_v4.py')}
    preload_module.preload=preload
    benchmark=ModuleType('benchmark');benchmark.write_json=m.write
    worker=ModuleType(Path(m.WORKER).stem)
    original_worker=ModuleType('train_lora_batched_tsc_bpe_shared_head_memory_v2')
    original_worker.native_loss_identity=lambda args:{'immutable_original_science_explicit_mock':True}
    env={'__file__':str(HERE/m.WORKER),'Path':Path,'OUTPUT':HERE,'ROOT':ROOT,'require':m.require,
        'torch':fake_torch,'random':random,'PHASE':m.PHASE,'validate_runtime_proof':h.validate_runtime_proof}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[after['native_loss_identity'],after['save_training_state']],type_ignores=[])),str(HERE/m.WORKER),'exec'),env)
    worker.save_training_state=env['save_training_state'];worker.write_json=m.write
    original_save,original_write=worker.save_training_state,worker.write_json
    modules={'native_tdt_checkpointed_v4':native,'preload_nemo_linker_v1':preload_module,
        'benchmark':benchmark,worker.__name__:worker,original_worker.__name__:original_worker}
    previous={key:sys.modules.get(key) for key in (*modules,'torch')};executable=sys.executable
    try:
        sys.modules.update(modules);sys.executable=str(ROOT/'work/nemo-native-env-v2/Scripts/python.exe')
        with tempfile.TemporaryDirectory(dir=ROOT/'work/research',prefix='shared_exact_route_') as d:
            d=Path(d);opaque=d/'opaque.json';opaque.write_text('{}',encoding='utf-8')
            original_ready=d/'original_ready.json';original_ready.write_text('{"original":true}',encoding='utf-8')
            paths={key:d/(key+'.json') for key in ('native','identity','memory','cache','io')}
            paths.update(stop=d/'stop.txt',log=d/'log.txt')
            record=h.record(opaque)
            plan={'worker_arguments':[],'source_sha256':{name:m.sha(HERE/name) for name in
                (p.name,m.WORKER,'cuda_cache_reclaim_v2.py','resource_watchdog_v2.py','paired_windows_commit_telemetry_v1.py','public_turkish_data_policy_v3.py')},
                'source_checkpoint':record,'preserved_BEST':record,'original207_checkpoint':record,'phase1_checkpoint':record,
                'original_native_ready':h.record(original_ready),'source_closure':record,
                'run_id':'OPAQUE_FIXTURE_ONLY','worker_arguments_sha256':'EXPLICIT_ARGV_MOCK',
                'original_config_identity_sha256':'EXPLICIT_STATIC_SCIENCE_MOCK',
                'row_lr':.0003,'shared_lr':.00001,'shared_updates_at_window_start':1,'phase_update_limit':4}
            m.validate_plan=lambda value:paths # Actual CLOSED admission deliberately not simulated as passed.
            capacity=m.shared_memory.validate_capacity
            class Evidence:
                def __init__(self,*a,**k):pass
                def start(self):events.append(('sampler_start',))
                def stop_and_join(self):events.append(('sampler_join',))
                def install(self):events.append(('caller_install',))
                def restore(self):events.append(('caller_restore',))
                def evidence(self):return {'eligible_update_calls':1,'thread_join_completed':True,'explicit_mock':True}
            m.shared_memory=SimpleNamespace(__file__=str(HERE/'launch_with_memory_guard_TSC_resume_v3.py'),
                validate_capacity=capacity,creation_identity=lambda pid:{'available':True,'pid':pid},
                safe_sample=lambda pid,phase:{'observed_monotonic':time.monotonic(),'phase':phase,'explicit_mock':True},
                UpdateTelemetry=Evidence,LiveTelemetrySampler=Evidence)
            m.make_caller=lambda *a:Evidence()
            args=SimpleNamespace(run_id=plan['run_id'],native_backend_provenance=original_ready,
                shared_head_exact_runtime_proof=paths['native'],shared_head_exact_source_receipt=opaque)
            worker.parser=lambda:SimpleNamespace(parse_args=lambda argv:args)
            worker.validate_arguments=lambda value:(env['native_loss_identity'](value),events.append(('validated_fresh_proof',)),value)[2]
            adaptation=SimpleNamespace(source_step=6,acquisition_updates=2,phase2_updates=2,cumulative_step=10,
                state_dict=lambda runtime_state:{'kind':m.STATE_KIND,'weights':{str(i):'OPAQUE_WEIGHT' for i in range(207)},
                    'row_optimizer':{'clock':4},'shared_optimizer':{'clock':8},'runtime_state':runtime_state})
            def run(value):
                events.append(('worker_run',))
                worker.save_training_state(d/'own.pt','OPAQUE_MODEL',adaptation,
                    SimpleNamespace(updates=10,state_dict=lambda:{'updates':10}),run_id=value.run_id,step=10,
                    config_identity={'original_static1000':True},history=[{'step':i} for i in range(1,11)],
                    best_score=.38,selected_step=9,patience=0,retention_failures=0,next_eval_epoch=.5,
                    train_wall_seconds=3.,audio_seconds=4.,language_audio_seconds={'tr':2.},
                    evaluation_wall_seconds=1.,initial_gate={'opaque_gate':True},source_selected_step=8)
            worker.run=run
            m.run_worker(plan)
            saved=json.loads((d/'own.pt').read_text())
            check(len(saved['weights'])==207 and 'row_optimizer' in saved and 'shared_optimizer' in saved
                and 'optimizer' not in saved and saved['runtime_state']['adaptation_updates_completed']==2
                and saved['source_phase1_step']==8 and saved['original_source_selected_step']==6,
                'actual_phase_serializer_atomic_commit_preserves_207_twoAdam_S_K_N_schema')
            check(events.index(('cap',.5,0))<events.index(('validated_fresh_proof',))<events.index(('worker_run',)),
                'actual_dispatch_preload_cap_fresh_native_proof_then_worker_order')
            ready=json.loads(paths['native'].read_text())
            check(ready['original_native_ready']==h.record(original_ready)
                and paths['native']!=original_ready and json.loads(original_ready.read_text())=={'original':True},
                'fresh_runtime_sidecar_separate_original_ready_never_rewritten')
            ready['pid']+=1;paths['native'].write_text(json.dumps(ready),encoding='utf-8')
            reject(lambda:h.validate_runtime_proof(args,worker_path=HERE/m.WORKER),'wrong_fresh_PID_rejected')
            check(worker.save_training_state is original_save and worker.write_json is original_write
                and ('sampler_join',) in events and ('caller_restore',) in events,
                'actual_dispatch_finally_restores_aliases_cache_callback_and_owned_sampler')
            completion=json.loads(paths['native'].with_suffix('.completion.json').read_text())
            check(completion['status']=='complete' and completion['sources_unchanged'] is True
                and json.loads(paths['io'].read_text())['legacy207_optimizer_serializer_used'] is False,
                'actual_complete_runtime_and_phase_serializer_producer_fields')
            paths['stop'].write_text('ROOT manual',encoding='utf-8');m.request_stop(paths['stop'],'budget')
            check(paths['stop'].read_text()=='ROOT manual','manual_stop_not_relabelled_budget')
    finally:
        sys.executable=executable
        for key,value in previous.items():
            if value is None:sys.modules.pop(key,None)
            else:sys.modules[key]=value
    check('torch' not in sys.modules,'no_actual_Torch_import_or_retained_mock')
    output=Path(output).resolve()
    report={'kind':'inactive_shared_head_exact_executor_stdlib_verification_v1','passed':True,
        'assertions':len(checks),'checks':checks,'controller':h.record(p),'worker':h.record(HERE/m.WORKER),
        'helper':h.record(HERE/m.TRANSITION),'actual_dispatch_function_executed':True,
        'actual_native_identity_and_phase_serializer_AST_executed':True,
        'actual_CLOSED_fullstate_admission_executed':False,'ML_state_CUDA_and_training_are_explicit_mocks':True,
        'Torch_actually_imported':False,'GPU_launched':False,'activation_ready_claimed':False}
    with output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'passed':True,'assertions':len(checks),'controller_sha256':m.sha(p),'proof_sha256':m.sha(output)}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
