"""Focused inactive controller contract/dispatch fixture; every ML call is a mock.

Uses actual controller.run_worker, worker.native_loss_identity and phase save
function. No Torch/NeMo import, real source admission, actual tensor, model,
checkpoint, corpus, CUDA API or watched run is used. Temporary opaque records
live only in a disposable work/research directory. Final source pins are checked
without real source admission. This checks interfaces, not activation readiness. Independent Adam
objects/tensors are opaque mocks; no real optimizer behavior is measured.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'outputs/parakeet_turkish'


def main(output):
    path = HERE / 'run_TSC_bpe_lora_low_lr_v1.py'
    require = lambda ok, message: None if ok else (_ for _ in ()).throw(AssertionError(message))
    spec = importlib.util.spec_from_file_location('adaptation_controller_fixture', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    checks = []
    def passed(name): checks.append(name)
    def rejected(name, callback):
        try: callback()
        except ValueError: passed(name); return
        raise AssertionError(name + ' did not reject')
    require('torch' not in sys.modules, 'Fixture must start without Torch')
    a = m.parser().parse_args(['--describe-contract'])
    require(a.adaptation_max_updates == 1000 and a.smoke_updates is None, 'Default screen/smoke')
    passed('default1000_DEV250_max1000_optional_separate_smoke1')
    rejected('exact_phase_rejected_before_admission', lambda: m.main(m.parser().parse_args(['--resume-exact', 'opaque.pt'])))
    args = ['--batch-size', '8', '--accumulation', '2', '--sampling-mode', 'domain30', '--fleurs-fraction', '.3',
            '--acquisition-lr', '.0001', '--native-max-joint-bytes', '1850000000', '--adaptation-max-updates', '1000', '--lora-lr', '.0003', '--adapt-lora-lr', '.00003']
    require(m.check_phase_arguments(args) == (1000, 0), 'Phase bound')
    rejected('legacy_optimizer_mode_rejected', lambda: m.check_phase_arguments(args + ['--resume-optimization', 'opaque.pt']))
    require(m.copied_scientific_arguments(args + ['--replay-kl', '10']) !=
            m.copied_scientific_arguments(args + ['--replay-kl', '5']), 'KD difference cannot hide behind source namespace')
    passed('all_copied_science_CLI_tokens_bind_actual_recipe')
    rows = [{'step': 1}, {'step': 2, 'adaptation_updates_completed': 1}]
    require(m.adaptation_suffix(rows, 1)[0]['step'] == 1 and rows[1]['step'] == 2, 'Local guard view')
    rejected('global_step_cannot_replace_fresh_Adam_clock', lambda: m.adaptation_suffix(
        [{'step': 1}, {'step': 2, 'adaptation_updates_completed': 2}], 1))
    passed('progress_suffix_view_does_not_rewrite_inherited_history')

    sys.path.insert(0, str(HERE)); m.load_support()
    require('torch' not in sys.modules, 'Guard support imported Torch')
    passed('actual_shared_guard_imports_are_stdlib_only')
    worker_path = HERE / m.WORKER
    tree = ast.parse(worker_path.read_text(encoding='utf-8'))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    sync = [n for n in ast.walk(functions['run']) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == 'synchronize'
            and ast.unparse(n.func.value) == 'torch.cuda']
    require(len(sync) == 1, 'One actual worker callback')
    require(m.WORKER_SHA == m.sha(worker_path) and m.EXPECTED_SYNC_LINE == sync[0].lineno,
        'Final controller worker/source frame pins must match before dispatch')
    m.finalized_core()
    require(m.caller_line(worker_path) == sync[0].lineno, 'Caller AST exact frame')
    passed('actual_worker_source_optimizer_callback_frame_admitted')

    # Compare the entire actual objective/backward/update loop and serializers.
    # Active LR is changed inside the utility AFTER loading original moments;
    # objective, clipping entry, RNG, native4, gates, and byte-checker stay exact.
    old_tree=ast.parse((HERE/'train_lora_batched_tsc_bpe_lora_native4_v2.py').read_text(encoding='utf-8'))
    old_functions={n.name:n for n in old_tree.body if isinstance(n,ast.FunctionDef)}
    for name in ('save_training_state','_protected_snapshot','_verify_smoke_phase_reload','native_loss_function'):
        require(ast.dump(functions[name],include_attributes=False)==ast.dump(old_functions[name],include_attributes=False),name+' changed')
    def update_loop(fn):
        return next(n for n in fn.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step')
    require(ast.dump(update_loop(functions['run']),include_attributes=False)==ast.dump(update_loop(old_functions['run']),include_attributes=False),'Objective/update loop changed')
    def dev_loop(fn):
        return next(n for n in fn.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_current')
    require(ast.dump(dev_loop(functions['run']),include_attributes=False)==ast.dump(dev_loop(old_functions['run']),include_attributes=False),'DEV gates changed')
    passed('actual_full_objective_backward195_clip_update_DEV_and207_twoAdam_serializer_AST_unchanged')
    rejected('active3e4_cannot_masquerade_as_lowLR',lambda:m.check_phase_arguments(m.set_option(args,'--adapt-lora-lr',3e-4)))
    rejected('dormant_original_LR_cannot_be_changed',lambda:m.check_phase_arguments(m.set_option(args,'--lora-lr',3e-5)))
    rejected('screen_bound_cannot_exceed1000',lambda:m.check_phase_arguments(m.set_option(args,'--adaptation-max-updates',1001)))
    # Actual worker parser/validator AST; only inherited parser is an explicit
    # stdlib fixture, avoiding all module-level Torch imports.
    import argparse
    old_parser=argparse.ArgumentParser()
    old_parser.add_argument('--optimization-source-receipt')
    ns={'source_parser':lambda:old_parser,'Path':Path,'ROOT':ROOT,'METADATA_PATH':ROOT/'work/research/unused_metadata','REPORT_PATH':ROOT/'work/research/unused_proof','__doc__':'mock inherited parser'}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[functions['parser']],type_ignores=[])),str(worker_path),'exec'),ns)
    parsed_worker=ns['parser']().parse_args(['--resume-adaptation','opaque.pt','--adaptation-source-receipt','opaque.json'])
    require(parsed_worker.adapt_lora_lr==3e-5 and parsed_worker.adaptation_max_updates==1000,'Actual worker options differ')
    passed('actual_worker_parser_exposes_distinct_active_LR_default1000')

    publications = []
    for node in ast.walk(functions['run']):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'write_json':
            target = node.args[0]
            if isinstance(target, ast.BinOp) and isinstance(target.op, ast.Div) and isinstance(target.right, ast.Constant):
                publications.append((node.lineno, node.col_offset, ast.unparse(target.left), target.right.value, node.args))
    publications.sort(key=lambda value: value[:2])
    small = {'run_config.json', 'training_history.json', 'selection.json'}
    for index, item in enumerate(publications):
        if item[2] == 'run_dir' and item[3] in small:
            following = publications[index + 1]
            require(following[2:4] == ('OUTPUT', item[3])
                and ast.dump(item[4][1]) == ast.dump(following[4][1]), 'Run-specific publication must precede equal shared payload')
    require(sum(item[2] == 'OUTPUT' and item[3] == 'run_config.json' for item in publications) == 1
        and sum(item[2] == 'OUTPUT' and item[3] == 'training_history.json' for item in publications) == 4
        and sum(item[2] == 'OUTPUT' and item[3] == 'selection.json' for item in publications) == 1,
        'All startup/DEV/log/final small publication sites need shared counterpart')
    require(not any(item[2] == 'OUTPUT' and item[3] == 'training_pool.json' for item in publications), 'Huge pool cannot be shared')
    passed('actual_worker_small_config_history_selection_publications_order_and_payload_match_without_shared_pool')
    # Execute only the actual paired call ASTs against disposable paths; no live OUTPUT write.
    with tempfile.TemporaryDirectory(dir=ROOT / 'work/research', prefix='acquisition_publication_fixture_') as directory:
        temp = Path(directory); (temp / 'run').mkdir(); (temp / 'shared').mkdir()
        emitted = []
        def publish(target, value):
            emitted.append((target.parent.name, target.name, value))
            target.write_text(json.dumps(value), encoding='utf-8')
        namespace = {'run_dir': temp / 'run', 'OUTPUT': temp / 'shared', 'write_json': publish,
                     'config': {'run_id': 'disposable_fixture'}, 'history': [{'step': 1}], 'selection': {'selected_step': 1}}
        calls = [ast.Expr(value=ast.Call(func=ast.Name(id='write_json', ctx=ast.Load()), args=item[4], keywords=[]))
                 for item in publications if item[3] in small and item[2] in ('run_dir', 'OUTPUT')]
        exec(compile(ast.fix_missing_locations(ast.Module(body=calls, type_ignores=[])), str(worker_path), 'exec'), namespace)
        require(len(emitted) == 12 and all(emitted[index][0] == 'run' and emitted[index + 1][0] == 'shared'
            and emitted[index][1:] == emitted[index + 1][1:] for index in range(0, len(emitted), 2)), 'Actual paired publication dispatch')
    passed('actual_publication_call_AST_executed_only_at_disposable_paths')

    events = []
    fake_torch = ModuleType('torch')
    fake_torch.cuda = SimpleNamespace(get_device_properties=lambda device: SimpleNamespace(total_memory=16_000_000_000),
        set_per_process_memory_fraction=lambda fraction, device=0: events.append(('cap', fraction, device)),
        get_rng_state_all=lambda: ['CUDA_RNG_OPAQUE_MOCK'])
    fake_torch.get_rng_state = lambda: 'CPU_RNG_OPAQUE_MOCK'
    fake_torch.save = lambda state, target: Path(target).write_text(json.dumps(state), encoding='utf-8')
    fake_native = ModuleType('native_tdt_checkpointed_v4')
    fake_native.configuration_identity = lambda: {'native_math_execution': 'explicitly mocked'}
    fake_native.cache_cleanup_evidence = lambda: {'available': False, 'mock': True}
    fake_preload = ModuleType('preload_nemo_linker_v1')
    def preload():
        require('torch' not in sys.modules, 'Preload must be before Torch')
        events.append(('preload',)); sys.modules['torch'] = fake_torch
        return {'kind': 'coherent_native_cuda12_9_preload_v1', 'actual_binding_linker': {'version': [12, 9]},
            'cuda_initialized': False, 'module_sha256': m.sha(HERE / 'preload_nemo_linker_v1.py'),
            'successful_launcher_sha256': m.sha(HERE / 'launch_nemo_linker_v4.py')}
    fake_preload.preload = preload
    fake_benchmark = ModuleType('benchmark'); fake_benchmark.write_json = m.write
    fake_worker = ModuleType(Path(m.WORKER).stem)
    env = {'__file__': str(worker_path), 'OUTPUT': HERE, 'ROOT': ROOT, 'Path': Path,
        'json': json, 'sys': sys, 'torch': fake_torch, 'random': random,
        'require': m.require, 'file_sha': m.sha, 'PHASE': m.PHASE, 'GEOMETRY': m.GEOMETRY}
    actual = ast.Module(body=[functions['native_loss_identity'], functions['save_training_state']], type_ignores=[])
    exec(compile(ast.fix_missing_locations(actual), str(worker_path), 'exec'), env)
    fake_worker.save_training_state = env['save_training_state']; fake_worker.write_json = m.write
    original_save = fake_worker.save_training_state; original_write = fake_worker.write_json
    old_modules = {name: sys.modules.get(name) for name in ('torch', 'native_tdt_checkpointed_v4', 'preload_nemo_linker_v1',
                                                          'benchmark', fake_worker.__name__)}
    old_executable = sys.executable
    sys.modules.update({'native_tdt_checkpointed_v4': fake_native, 'preload_nemo_linker_v1': fake_preload,
                        'benchmark': fake_benchmark, fake_worker.__name__: fake_worker})
    try:
        sys.executable = str(ROOT / 'work/nemo-native-env-v2/Scripts/python.exe')
        with tempfile.TemporaryDirectory(dir=ROOT / 'work/research', prefix='acquisition_dispatch_fixture_') as directory:
            temp = Path(directory); asset = temp / 'opaque_asset.json'; asset.write_text('{}', encoding='utf-8')
            paths = {key: temp / (key + '.json') for key in ('native', 'identity', 'memory', 'cache', 'io')}
            paths.update(log=temp / 'log.txt', stop=temp / 'stop.txt')
            argv = ['--run-id', 'inactive_fixture_no_watched_run', '--native-backend-provenance', str(paths['native'])]
            for flag in ('--tsc-input-proof', '--tsc-training-approval', '--source-closure', '--tr-manifest', '--replay-manifest',
                         '--tr-frame-metadata', '--tr-frame-metadata-proof', '--data-v3-proof', '--resume-adaptation',
                         '--adaptation-source-receipt', '--candidate-tr-metadata', '--candidate-tr-metadata-proof'):
                argv.extend((flag, str(asset)))
            pins = {name: m.sha(HERE / name) for name in (Path(path).name, m.WORKER, 'cuda_cache_reclaim_v2.py',
                'resource_watchdog_v2.py', 'paired_windows_commit_telemetry_v1.py', 'public_turkish_data_policy_v3.py')}
            plan = {'worker_arguments': argv, 'source_sha256': pins, 'source_checkpoint': m.record(asset),
                'run_id': 'inactive_fixture_no_watched_run', 'worker_arguments_sha256': 'EXPLICIT_ARGV_MOCK',
                'source_selected_step': 2, 'adaptation_max_updates': 1000,
                'original_source_selected_step': 1, 'selected_acquisition_updates': 1,
                'chosen_checkpoint_role': 'latest', 'original207_checkpoint': m.record(asset), 'smoke_updates': 1,
                'declared_adaptation': {'explicit_fixture_admission_mock': True}}
            m.validate_plan = lambda value: paths  # Actual CLOSED admission deliberately not executed.
            capacity = m.shared_memory.validate_capacity
            class Evidence:
                def __init__(self, *values, **keywords): pass
                def start(self): events.append(('sampler_start',))
                def stop_and_join(self): events.append(('sampler_join',))
                def install(self): events.append(('caller_install',))
                def restore(self): events.append(('caller_restore',))
                def evidence(self): return {'eligible_update_calls': 0, 'thread_join_completed': True, 'explicit_mock': True}
            m.shared_memory = SimpleNamespace(__file__=str(HERE / 'launch_with_memory_guard_TSC_resume_v3.py'),
                validate_capacity=capacity, creation_identity=lambda pid: {'available': True, 'pid': pid},
                safe_sample=lambda pid, phase: {'observed_monotonic': time.monotonic(), 'phase': phase, 'explicit_mock': True},
                UpdateTelemetry=Evidence, LiveTelemetrySampler=Evidence)
            m.make_caller = lambda *values: Evidence()
            parsed = SimpleNamespace(run_id=plan['run_id'], tdt_implementation='nemo', native_backend_provenance=paths['native'],
                                     native_max_joint_bytes=1850000000, adapt_lora_lr=3e-5, lora_lr=3e-4)
            fake_worker.parser = lambda: SimpleNamespace(parse_args=lambda values: parsed)
            fake_worker.validate_arguments = lambda value: (env['native_loss_identity'](value), events.append(('worker_validate',)), value)[2]
            phase = SimpleNamespace(source_step=1, acquisition_updates=1, phase2_updates=0, cumulative_step=2, state_dict=lambda runtime_state: {'kind': m.STATE_KIND,
                'weights': {str(i): 'OPAQUE_WEIGHT_MOCK' for i in range(207)}, 'row_optimizer': {'clock_K': 1}, 'lora_optimizer': {'clock_S': 1},
                'runtime_state': runtime_state})
            def run(value):
                events.append(('worker_run',))
                fake_worker.save_training_state(temp / 'phase.pt', 'MODEL_MOCK', phase,
                    SimpleNamespace(updates=2, state_dict=lambda: {'opaque_sampler_mock': True}),
                    run_id=value.run_id, step=2, config_identity={'phase': m.PHASE}, history=[{'step':1},{'step':2}], best_score=.4,
                    selected_step=1, patience=0, retention_failures=0, next_eval_epoch=.1, train_wall_seconds=1.,
                    audio_seconds=1., language_audio_seconds={'tr': 1.}, evaluation_wall_seconds=1.,
                    initial_gate={'opaque_gate_mock': True}, source_selected_step=2)
            fake_worker.run = run
            m.run_worker(plan)
            saved = json.loads((temp / 'phase.pt').read_text(encoding='utf-8'))
            require(saved['kind'] == m.STATE_KIND and len(saved['weights']) == 207
                and 'row_optimizer' in saved and 'lora_optimizer' in saved and 'optimizer' not in saved
                and saved['source_phase1_step']==2 and saved['original_source_selected_step']==1
                and saved['selected_acquisition_updates']==1, 'Actual phase serializer contract')
            passed('actual_phase_serializer_and_atomic_commit_keep_207_names_separate_rowK_and_LoRAS_schema')
            require(events[0][0] == 'preload' and events.index(('cap', .5, 0)) < events.index(('worker_validate',))
                < events.index(('worker_run',)), 'Actual dispatcher preload/cap/validate/worker ordering')
            passed('actual_controller_preload_cap_actual_native_identity_then_mock_worker_dispatch')
            require(fake_worker.save_training_state is original_save and fake_worker.write_json is original_write
                and ('sampler_join',) in events and ('caller_restore',) in events, 'Finally restoration')
            passed('actual_dispatch_restores_serializer_write_aliases_and_owned_telemetry')
            require(json.loads(paths['io'].read_text())['legacy207_optimizer_serializer_used'] is False
                and json.loads(paths['native'].with_suffix('.completion.json').read_text())['status'] == 'complete', 'Completion schema')
            passed('phase_runtime_completion_emits_true_producer_fields_under_explicit_mocks')
            ready = json.loads(paths['native'].read_text()); ready['geometry']['joint_width'] = 8204
            paths['native'].write_text(json.dumps(ready), encoding='utf-8')
            rejected('actual_worker_native_identity_rejects_old_geometry', lambda: env['native_loss_identity'](parsed))
            paths['stop'].write_text('ROOT original manual request', encoding='utf-8'); m.request_stop(paths['stop'], 'budget')
            require(paths['stop'].read_text() == 'ROOT original manual request', 'Manual stop overwritten')
            passed('ROOT_manual_stop_not_rewritten_as_budget')
    finally:
        sys.executable = old_executable
        for name, module in old_modules.items():
            if module is None: sys.modules.pop(name, None)
            else: sys.modules[name] = module
    # Execute the actual completion validator on a realistic producer envelope.
    # Every source/runtime/ASR record is explicitly synthetic and in memory.
    import copy
    run_id='inactive_fixture_not_a_real_run'
    run=HERE/'runs'/run_id
    paths={key:ROOT/'work/research'/('opaque_lowlr_'+key+'.json') for key in ('identity','native','io','cache','memory')}
    history=[{'step':1},{'step':2},{'step':3,'adaptation_updates_completed':1}]
    plan={'paths':paths,'run_id':run_id,'source_selected_step':2,'original_source_selected_step':1,
        'selected_acquisition_updates':1,'phase_update_limit':1,'adaptation_max_updates':250,
        'source_history_prefix_sha256':m.canonical(history[:2]),'source_recipe':{'lora_lr':3e-4},
        'source_closure':{'explicit_source_admission_mock':True},'smoke_updates':1,'worker_arguments':[]}
    selection={'run_id':run_id,'heldout_tests_evaluated':False,'adaptation_updates_completed':1,
        'source_phase1_step':2,'inherited_history_steps':2,'original_source_selected_step':1,
        'selected_acquisition_updates':1,'completed_steps':3,'row_Adam_clock':2,'LoRA_Adam_clock':2,
        'phase':m.PHASE,'external_exact_resume_supported':False,'further_unfreeze_allowed':False}
    phase={'kind':m.PHASE,'source_phase1_step':2,'original_source_selected_step':1,
        'selected_acquisition_updates':1,'active_parameter_tensors':195,'row_lr':1e-4,'lora_lr':3e-5,
        'source_original_lora_lr':3e-4,'max_updates':250,'effective_max_updates':1,
        'old15_head_moments_active':False,'original207_Adam_artifact_retained_dormant':True,
        'further_unfreeze_allowed':False,'external_exact_resume_supported':False}
    optimizer_identity={'kind':'HF_TSC_BPE512_new_rows_plus_encoder_LoRA_low_lr_identity_v1',
        'source_selected_step':1,'selected_acquisition_updates':1,'active_parameter_tensors':195,
        'fixed_shared_head_tensors':12,'old15_head_moments_active':False,'further_unfreeze_allowed':False,
        'source_original_LoRA_lr':3e-4,'active_LoRA_lr':3e-5,'source_LoRA_moments_reset':False,
        'row_group':{'lr':1e-4},'lora_group':{'lr':3e-5}}
    config={'run_id':run_id,'phase':m.PHASE,'config_identity':{'phase':m.PHASE,
        'training_arguments':{'lora_lr':3e-4,'adapt_lora_lr':3e-5},'geometry':m.GEOMETRY,
        'adaptation_phase':phase,'adaptation_optimizers':optimizer_identity}}
    proof={'kind':m.PROOF_KIND,'run_id':run_id,'passed':True,'actual_GPU_decode':True,
        'before_phase2_updates':True,'compared_fixed_DEV_utterances':300,'unsuppressed_new512_ids':True,
        'all_raw_transcripts_and_emitted_paths_exact':True,'all_six_ASR_counts_exact':True,
        'all_five_retention_gates_passed':True,'RNG_restored_after_added_evaluation':True,
        'module_training_modes_restored_after_added_evaluation':True,'source_receipt':plan['source_closure'],
        'source_selected_step':2}
    smoke={'kind':'TSC_BPE_LoRA_actual_one_update_smoke_v1','run_id':run_id,'passed':True,
        'actual_native_ASR_backward_and_two_Adam':True,'all3_new_row_deltas_nonzero':True,
        'LoRA192_gradients_finite_and_any_delta_nonzero':True,
        'protected_old_six_blank_duration_and_frozen12_rawbytes_exact':True,
        'parked_original207_and_phase1_unchanged':True,'complete207_two_optimizer_phase_save_load_passed':True,
        'row_clock_K_plus_N':2,'lora_clock_S_plus_N':2,'external_exact_resume_supported':False}
    records={paths['identity']:{'pid':77},paths['native'].with_suffix('.completion.json'):{
        'kind':'TSC_BPE_adaptation_native_completion_v1','status':'complete','phase':m.PHASE,'pid':77,
        'run_id':run_id,'sources_unchanged':True,'backend_ready_sha256':'EXPLICIT_MOCK_SHA',
        'actual_dispatch_source_sha256':'EXPLICIT_MOCK_SHA'},run/m.PROOF_NAME:proof,
        run/'run_config.json':config,run/'training_history.json':history,
        run/'adaptation_actual_one_update_smoke.json':smoke}
    for key,kind in (('io','TSC_BPE_adaptation_atomic_IO_v1'),('cache','TSC_BPE_adaptation_cache_runtime_v1'),('memory','TSC_BPE_adaptation_memory_runtime_v1')):
        records[paths[key]]={'kind':kind,'status':'complete','sources_unchanged':True}
    records[paths['cache']]['cache_reclamation']={'eligible_update_calls':1}
    records[paths['memory']].update(cap_bytes=8000000000,cleanup_threshold_bytes=4000000000,
        allocator_cap_applied=True,all_phase_owned_memory_telemetry={'thread_join_completed':True})
    original_functions={name:getattr(m,name) for name in ('read','sha','source_pins','boundary_context','record')}
    try:
        m.read=lambda path,*default:records[Path(path)]
        m.sha=lambda path:'EXPLICIT_MOCK_SHA'
        m.source_pins=lambda context:{}
        m.boundary_context=lambda argv:{}
        m.record=lambda path:{'explicit_runtime_mock':str(path)}
        require(m.verify_completion(plan,selection)['phase_checkpoint_schema']==m.STATE_KIND,'Actual completion validator')
        passed('actual_completion_validator_accepts_newutility_kind_active3e5_source3e4_and_suffixclock_envelope')
        optimizer_identity['kind']='HF_TSC_BPE512_new_rows_plus_encoder_LoRA_identity_v1'
        rejected('actual_completion_rejects_original_utility_identity_kind',lambda:m.verify_completion(plan,selection))
        optimizer_identity['kind']='HF_TSC_BPE512_new_rows_plus_encoder_LoRA_low_lr_identity_v1'
        optimizer_identity['active_LoRA_lr']=3e-4
        rejected('actual_completion_rejects_active_rate_masquerading_as_source_rate',lambda:m.verify_completion(plan,selection))
        optimizer_identity['active_LoRA_lr']=3e-5
        selection['LoRA_Adam_clock']=1
        rejected('actual_completion_rejects_reset_original_LoRA_clock',lambda:m.verify_completion(plan,selection))
        selection['LoRA_Adam_clock']=2
        records[paths['cache']]['cache_reclamation']['eligible_update_calls']=3
        rejected('actual_completion_rejects_cumulative_callback_count',lambda:m.verify_completion(plan,selection))
    finally:
        for name,function in original_functions.items():setattr(m,name,function)
    require('torch' not in sys.modules, 'Fixture retained a Torch import')
    report = {'kind': 'inactive_TSC_BPE_LoRA_low_lr_controller_dispatch_CPU_verification_v1', 'passed': True,
        'controller': m.record(path), 'worker_draft': m.record(worker_path), 'checks': checks, 'assertions': len(checks),
        'actual_controller_dispatch_function_executed': True, 'actual_worker_native_identity_AST_executed': True,
        'actual_worker_phase_serializer_AST_executed': True, 'actual_CLOSED_source_admission_executed': False,
        'ML_Torch_CUDA_PIDs_model_tensors_and_training_are_explicit_mocks': True,
        'Torch_actually_imported': False, 'GPU_launched': False, 'activation_ready_claimed': False}
    output = Path(output).resolve()
    with output.open('x', encoding='utf-8') as stream: json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps({'passed': True, 'assertions': len(checks), 'report': str(output), 'controller_sha256': m.sha(path)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
