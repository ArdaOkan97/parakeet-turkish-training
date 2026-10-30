"""INACTIVE guarded SAME-ID own-shared-state exact executor.

Original config1000/source300/native-ready stay immutable. A ROOT actual CLOSED
qualified improving latest supplies full207/two-Adam/sampler/RNG/history; an
approved stopping-bound extension is separate. One owned child only, no automatic
retry, rewind, unfreeze or successor. Full coverage requires actual unique IDs.
"""
from __future__ import annotations
import argparse
import ast
from collections import deque
from datetime import datetime, timezone
import functools
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
WORKER = 'train_lora_batched_tsc_bpe_shared_head_exact_v1.py'
TRANSITION = 'tsc_bpe_shared_head_exact_transition_v1.py'
WORKER_SHA = '881ffa3ee85941f04ecfca7f4237ba44df4e2319d3c64772e97fb62d8efcb2dc'
TRANSITION_SHA = '433d6aaa34254b81febd0ec868d2c73fdea227cd5f6b1f517da75ecf9ff45d24'
EXPECTED_SYNC_LINE = 464
PHASE = 'HF_TSC_BPE512_new_rows_plus_shared_heads_v1'
STATE_KIND = 'HF_TSC_BPE512_new_rows_plus_shared_heads_state_v1'
CONTROLLER_KIND = 'guarded_TSC_BPE_shared_heads_exact_controller_v1'
PROOF_NAME = 'TSC_BPE_shared_head_initial_source_equivalence.json'
PROOF_KIND = 'TSC_BPE512_shared_heads_before_update_source_equivalence_v1'
NATIVE_MAX_BYTES = 1_850_000_000
GEOMETRY = {'student_vocab_size': 8711, 'joint_width': 8716, 'blank_id': 8192,
    'teacher_vocab_size': 8193, 'new_BPE_rows': [8199, 8711],
    'protected_six_rows': [8193, 8199], 'duration_rows': [8711, 8716]}
UTILITY_PINS = {
    'compact_training_row_objects_v1.py': '1b3f9f0e3bdd19c285f2825a53685d7cf9fd964040c9087560f610a5ed1ec96d',
    'bpe_newrows_shared_head_adaptation_v1.py': '384d01abe9b7105761491b8acda5d3de4eb8c40ae679eee2701269079cf9967c',
    'bpe_newrows_acquisition_v1.py': 'bc26638f4874aaa5e92029e86c30bef0624a477514957d8f8d054d7e92088724',
    'tsc_bpe512_training_inputs_v1.py': '98d79657035be7ed59a9c372ac5335755391456563167b891fd883fb89b7ccf7',
    'bpe_target_sampler_transition_v1.py': 'f43f6bd46ebbad0192bafdb66daa1a5414f2ca6b1d54f451f943c9268da3140e',
}
SUPPORT_PINS = {
    'run_TSC_headlr_fork_v1.py': 'aed37d34bee878e51e9ec906023d97c132af83b11bb7f6989b0ed3cf7d1a7f8d',
    'run_hf_TSC_arm_v3.py': '7eccccd31c294e04bc962eed7cc3d0e385768386607b491064cbe21955cfc24b',
    'run_experiment_loop_continuation_v8.py': 'c65e9f37eb0db4641552d35ccda52b5604223223deb88d7d9ebfcfe61f0f051d',
    'launch_with_memory_guard_TSC_resume_v3.py': '61fec6d3ef911665bdbe876dc333f3c6a640badc012596fe92ded1b80a10c7fa',
    'cuda_cache_reclaim_v2.py': '869b9ffaac70c9b7b6e7590ba1d330363100d6b2ddb9f8f7af9b2b9865c1c7f9',
    'atomic_io_retry_v1.py': '2d3fe379d51021fd15679e562c71286f7f9501180d28dde77893e8f72c80a1b0',
    'resource_watchdog_v2.py': '64d1393e1e45cf65dac14f78ac955331c91b4a3f0eafbf7535196e2b61778641',
    'paired_windows_commit_telemetry_v1.py': '2ae9f2cdfc388a294914e000c60b62fe09cddac147700c5762851d626a95a683',
    'native_tdt_checkpointed_v3.py': '839194dddf83c3b08bf608d01cb540b2f160feee426a91373d14cfc426ab7494',
    'native_tdt_checkpointed_v4.py': '24b9cf5b8f3e11d5b043c22d0940da3735a46f8f70b6f5810b97f17d545cd90e',
    'preload_nemo_linker_v1.py': 'f8787b0f3346c9f526903ff0c06de4c266f940ca3b0f41c55907cc795e7cc2ba',
}
domain = shared_memory = ResourceGuard = sample_memory = CacheOperations = None
read = write = commit_writer = None


def require(ok, message):
    if not ok: raise ValueError(message)


def now(): return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def record(path): return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def scoped(path):
    result = Path(path).resolve()
    require(result.is_relative_to(ROOT), 'Workspace-only evidence/source path required')
    return result


def read_small(path): return json.loads(scoped(path).read_text(encoding='utf-8'))


def option(argv, flag):
    require(argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv), 'Exactly one ' + flag + ' required')
    return argv[argv.index(flag) + 1]


def set_option(argv, flag, value):
    result = list(argv)
    if flag in result:
        require(result.count(flag) == 1, 'Duplicate ' + flag)
        result[result.index(flag) + 1] = str(value)
    else: result.extend((flag, str(value)))
    return result


def remove_option(argv, flag):
    result = list(argv)
    if flag in result:
        require(result.count(flag) == 1, 'Duplicate ' + flag)
        index = result.index(flag); del result[index:index + 2]
    return result


def load_support():
    """Borrow frozen guard/cache/atomic implementations, not their old admission."""
    global domain, shared_memory, ResourceGuard, sample_memory, CacheOperations, read, write, commit_writer
    for name, digest in SUPPORT_PINS.items(): require(sha(HERE / name) == digest, 'Frozen support changed: ' + name)
    base = importlib.import_module('run_TSC_headlr_fork_v1')
    domain, shared_memory = base.domain, base.shared_memory
    ResourceGuard, sample_memory, CacheOperations = base.ResourceGuard, base.sample_memory, base.CacheOperations
    read, write, commit_writer = base.read, base.write, base.commit_writer
    require('torch' not in sys.modules, 'Controller support must remain free of ML imports')


def finalized_core():
    require(re.fullmatch('[0-9a-f]{64}', WORKER_SHA) is not None
        and re.fullmatch('[0-9a-f]{64}', TRANSITION_SHA) is not None
        and type(EXPECTED_SYNC_LINE) is int and EXPECTED_SYNC_LINE > 0,
        'Inactive draft: worker/boundary/synchronize pins are not finalized')
    for name, digest in {WORKER: WORKER_SHA, TRANSITION: TRANSITION_SHA, **UTILITY_PINS}.items():
        require(sha(HERE / name) == digest, 'Reviewed adaptation source changed: ' + name)


def source_pins(context):
    finalized_core(); load_support()
    pins = dict(domain.chain_identity()['source_sha256'])
    for name, digest in context['parent_identity']['source_sha256'].items():
        require(name not in pins or pins[name] == digest, 'Parent versus support source differs: ' + name)
        pins[name] = digest
    for name, digest in {**SUPPORT_PINS, **UTILITY_PINS, WORKER: WORKER_SHA,
                        TRANSITION: TRANSITION_SHA, Path(__file__).name: sha(__file__)}.items():
        require(name not in pins or pins[name] == digest, 'Conflicting adaptation source: ' + name)
        pins[name] = digest
    for name, digest in pins.items(): require(sha(HERE / name) == digest, 'Frozen dependency changed: ' + name)
    return pins


def caller_line(worker_path):
    require(sha(worker_path) == WORKER_SHA, 'Reviewed adaptation worker changed')
    tree = ast.parse(Path(worker_path).read_text(encoding='utf-8'))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run']
    calls = [node for fn in functions for node in ast.walk(fn) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == 'synchronize'
        and ast.unparse(node.func.value) == 'torch.cuda']
    require(len(functions) == len(calls) == 1 and not calls[0].args and not calls[0].keywords
        and calls[0].lineno == EXPECTED_SYNC_LINE, 'Single pinned completed-update cache callback required')
    return calls[0].lineno


def make_caller(cuda, worker_path, telemetry):
    class AdaptationCaller(CacheOperations):
        def __init__(self, cuda, worker_path):
            self.cuda = cuda; self.worker_path = Path(worker_path).resolve(); self.expected_sha = WORKER_SHA
            self.line = caller_line(self.worker_path); self.threshold_bytes = 3_000_000_000; self.max_events = 128
            self.events = deque(maxlen=128)
            self.eligible_calls = self.ignored_calls = self.reclamations = self.failed_reclamations = 0
            self.cleanup_seconds = 0.; self.original = self.wrapped = None
    return shared_memory.telemetry_reclaimer_class(AdaptationCaller, telemetry)(cuda, worker_path)


def check_phase_arguments(argv):
    for flag in ('--resume-exact','--resume-weights','--resume-runtime-migration','--resume-optimization',
        '--optimization-source-receipt','--resume-acquisition','--acquisition-source-receipt','--unfreeze','--smoke-updates'):
        require(flag not in argv,'Unsupported own exact state mode:'+flag)
    require(option(argv,'--batch-size')=='8' and option(argv,'--accumulation')=='2'
        and option(argv,'--sampling-mode')=='domain30' and float(option(argv,'--fleurs-fraction'))==.3
        and float(option(argv,'--row-lr')) in (1e-3,3e-4) and float(option(argv,'--shared-lr')) in (1e-5,3e-6)
        and int(option(argv,'--native-max-joint-bytes'))==NATIVE_MAX_BYTES
        and int(option(argv,'--adaptation-max-updates'))==1000,
        'Immutable initial1000/B8x2/domain30/row+shared recipe required')
    for flag in ('--resume-shared-head-exact','--shared-head-exact-source-receipt','--shared-head-exact-runtime-proof'):
        scoped(option(argv,flag))
    return 1000,0

TRANSITION_ARGUMENTS = ('--run-id', '--resume-exact', '--resume-weights', '--resume-runtime-migration',
    '--resume-optimization', '--optimization-source-receipt', '--resume-acquisition', '--acquisition-source-receipt', '--resume-shared-heads', '--shared-head-source-receipt',
    '--candidate-dir', '--candidate-tr-metadata', '--candidate-tr-metadata-proof', '--acquisition-lr', '--row-lr', '--shared-lr',
    '--acquisition-max-updates', '--adaptation-max-updates', '--smoke-updates', '--native-max-joint-bytes',
    '--native-backend-provenance', '--stop-file', '--max-wall-seconds')


def copied_scientific_arguments(argv):
    result = list(argv)
    for flag in TRANSITION_ARGUMENTS: result = remove_option(result, flag)
    return result


def boundary_context(argv):
    # Pure boundary only; no worker/Torch import before coherent preload.
    transition=importlib.import_module(Path(TRANSITION).stem)
    receipt=read_small(option(argv,'--shared-head-exact-source-receipt'))
    config=read_small(receipt['artifacts']['run_config']['path'])
    values=dict(config);values.update(config['config_identity']['training_arguments'])
    for flag in ('--resume-shared-head-exact','--shared-head-exact-source-receipt','--shared-head-exact-runtime-proof',
        '--native-backend-provenance','--stop-file'):
        values[flag[2:].replace('-','_')]=scoped(option(argv,flag))
    values.update(run_id=option(argv,'--run-id'),max_wall_seconds=float(option(argv,'--max-wall-seconds')),
        resume_exact=None,resume_weights=None,resume_runtime_migration=None,resume_optimization=None,
        optimization_source_receipt=None,resume_acquisition=None,resume_adaptation=None,smoke_updates=None)
    # Ensure the actual command, including all immutable original arguments,
    # matches its source. Only these five exact/runtime paths may differ.
    source_args=read_small(receipt['artifacts']['controller_status']['path'])['worker_arguments']
    ignored=('--resume-shared-head-exact','--shared-head-exact-source-receipt','--shared-head-exact-runtime-proof',
        '--stop-file','--max-wall-seconds')
    def stripped(items):
        for flag in ignored:items=remove_option(items,flag)
        return items
    require(stripped(list(argv))==stripped(list(source_args)), 'No scientific CLI or original native-ready change')
    return transition.preflight(SimpleNamespace(**values))

def validate_plan(plan):
    finalized_core();argv=plan['worker_arguments'];check_phase_arguments(argv)
    context=boundary_context(argv);pins=source_pins(context);receipt=context['exact_receipt']
    require(plan['kind']==CONTROLLER_KIND and plan['controller_sha256']==sha(__file__)
        and plan['source_sha256']==pins and option(argv,'--run-id')==plan['run_id']
        and domain.guards.argument_sha(argv)==plan['worker_arguments_sha256'], 'Frozen exact plan/source/arguments changed')
    require(plan['source_closure']==context['exact_receipt_record']
        and plan['source_checkpoint']==receipt['artifacts']['latest_checkpoint']
        and plan['preserved_BEST']==receipt['artifacts']['best_checkpoint']
        and plan['source_selected_step']==context['source_phase1_step']
        and plan['resume_step']==context['resume_step']
        and plan['shared_updates_at_window_start']==context['shared_updates_at_resume']
        and plan['phase_update_limit']==context['approved_total_shared_head_updates']
        and plan['original_source_selected_step']==receipt['original_step']
        and plan['selected_acquisition_updates']==receipt['source_acquisition_updates']
        and plan['original_config_identity_sha256']==canonical(context['parent_identity'])
        and plan['original_native_ready']==receipt['artifacts']['original_native_ready']
        and plan['original_source300']==receipt['artifacts']['initial_source_equivalence'], 'One exact own/latest/BEST/S-K-N/science frontier required')
    require(.1<=plan['window_hours']<=3. and math.isfinite(plan['window_hours'])
        and float(option(argv,'--max-wall-seconds'))==plan['window_hours']*3600-300,
        'Bounded3h whole-update window/grace required')
    paths={key:scoped(value) for key,value in plan['paths'].items()}
    require(len(set(paths.values()))==len(paths)
        and option(argv,'--stop-file')==str(paths['stop'])
        and option(argv,'--shared-head-exact-runtime-proof')==str(paths['native'])
        and scoped(option(argv,'--native-backend-provenance'))==scoped(plan['original_native_ready']['path'])
        and paths['native']!=scoped(plan['original_native_ready']['path']),
        'Fresh PID proof must remain separate from original scientific token')
    caller_line(HERE/WORKER);return paths

def prepare(args):
    finalized_core()
    require(math.isfinite(args.window_hours) and .1<=args.window_hours<=3. and 5<=args.poll_seconds<=15,
        'Bounded3h window/fresh5-15s polls required')
    output=scoped(args.output_dir);source_loop=scoped(args.source_loop)
    require(output.parent==HERE/'runs' and not output.exists(), 'Fresh controller window directory required')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{2,110}',args.run_id) is not None
        and (HERE/'runs'/args.run_id).is_dir() and output!=HERE/'runs'/args.run_id,
        'SAME existing worker run namespace, separate fresh window required')
    receipt_path=scoped(args.source_receipt);receipt=read_small(receipt_path)
    original_path=receipt.get('original_artifact_paths',{}).get('controller_status',receipt['artifacts']['controller_status']['path'])
    require(scoped(source_loop/'loop_status.json')==scoped(original_path),'ROOT archived source loop differs')
    source=read_small(receipt['artifacts']['controller_status']['path']);argv=list(source['worker_arguments'])
    require(args.approved_total_shared_head_updates==receipt['approved_total_shared_head_updates'],
        'Explicit CLI total must match ROOT separate extension')
    paths={key:output/f'{args.run_id}_{name}' for key,name in (('native','fresh_native_backend.json'),
        ('identity','identity.json'),('memory','memory.json'),('cache','cache.json'),('io','io.json'),('stop','stop.txt'),('log','worker.log'))}
    for flag,value in (('--resume-shared-head-exact',scoped(receipt['artifacts']['latest_checkpoint']['path'])),
        ('--shared-head-exact-source-receipt',receipt_path),('--shared-head-exact-runtime-proof',paths['native']),
        ('--stop-file',paths['stop']),('--max-wall-seconds',args.window_hours*3600-300)):
        argv=set_option(argv,flag,value)
    check_phase_arguments(argv);context=boundary_context(argv);pins=source_pins(context)
    python=scoped(source['command'][0]);require(python==ROOT/'work/nemo-native-env-v2/Scripts/python.exe' and python.is_file(),
        'Keep actual native-env-v2 interpreter')
    original=context['original_context']
    plan={'schema':1,'kind':CONTROLLER_KIND,'phase':PHASE,'run_id':args.run_id,'loop_id':output.name,
        'controller_sha256':sha(__file__),'source_loop':str(source_loop),'source_closure':context['exact_receipt_record'],
        'source_checkpoint':receipt['artifacts']['latest_checkpoint'],'preserved_BEST':receipt['artifacts']['best_checkpoint'],
        'source_selected_step':context['source_phase1_step'],'source_phase1_step':context['source_phase1_step'],
        'original_source_selected_step':receipt['original_step'],'selected_acquisition_updates':receipt['source_acquisition_updates'],
        'resume_step':context['resume_step'],'shared_updates_at_window_start':context['shared_updates_at_resume'],
        'phase_update_limit':context['approved_total_shared_head_updates'],'approved_total_shared_head_updates':context['approved_total_shared_head_updates'],
        'source_history_prefix_sha256':canonical(context['archived_history']),
        'original_config_identity_sha256':canonical(context['parent_identity']),
        'original_native_ready':receipt['artifacts']['original_native_ready'],
        'original_source300':receipt['artifacts']['initial_source_equivalence'],
        'original207_checkpoint':original['original_context']['receipt']['artifacts']['best_checkpoint'],
        'phase1_checkpoint':original['phase1_record'],'source_recipe':original['source_recipe'],
        'source_sha256':pins,'worker_arguments':argv,'worker_arguments_sha256':domain.guards.argument_sha(argv),
        'command':[str(python),str(Path(__file__).resolve()),'--worker-plan',str(output/'plan.json'),'--gpu-window-authorized'],
        'paths':{key:str(value) for key,value in paths.items()},'window_hours':args.window_hours,
        'row_lr':float(option(argv,'--row-lr')),'shared_lr':float(option(argv,'--shared-lr')),
        'adaptation_max_updates':1000,'smoke_updates':0,'inherited_scientific_coverage_horizon':44958,
        'same_phase_exact_continuation_supported':True,'original_config_or_BEST_reinitialized':False,
        'source300_repeated':False,'unfreeze_supported':False,'objective_complete':False,'heldout_TEST_evaluated':False}
    validate_plan(plan);return plan

def run_worker(plan):
    """Root-owned child: fresh12.9 preload -> cap -> same guard/cache/IO APIs."""
    paths = validate_plan(plan); argv = plan['worker_arguments']; pins = plan['source_sha256']
    for key in ('native', 'identity', 'memory', 'cache', 'io'): require(not paths[key].exists(), 'Fresh runtime evidence required: ' + key)
    live = Path(str(paths['identity']) + '.live_memory.json'); completion = paths['native'].with_suffix('.completion.json')
    require(not live.exists() and not completion.exists(), 'Never reuse an earlier PID/completion')
    require('torch' not in sys.modules, 'Actual12.9 preload must precede Torch')
    from preload_nemo_linker_v1 import preload
    preloaded = preload()
    require(preloaded.get('kind') == 'coherent_native_cuda12_9_preload_v1'
        and preloaded.get('actual_binding_linker', {}).get('version') == [12, 9]
        and preloaded.get('cuda_initialized') is False, 'Coherent fresh pre-Torch native preload required')
    transition=importlib.import_module(Path(TRANSITION).stem)
    expected=plan['source_checkpoint']
    for value in (expected,plan['preserved_BEST'],plan['original207_checkpoint'],plan['phase1_checkpoint']):
        transition.checked_record(value,checkpoint=True)
        require(sha(value['path'])==value['sha256'],'Immutable own/original/phase1 bytes changed before runtime')
    ready={'schema':1,'kind':'TSC_BPE_shared_heads_exact_fresh_preload_v1','status':'ready',
        'phase':PHASE,'geometry':GEOMETRY,'run_id':plan['run_id'],'pid':os.getpid(),'ppid':os.getppid(),
        'ready_at_utc':now(),'worker_executor':transition.record(HERE/WORKER),
        'controller_executor':transition.record(__file__),'exact_helper_sha256':TRANSITION_SHA,
        'original_native_ready':plan['original_native_ready'],'source_receipt':plan['source_closure'],
        'preload':preloaded,'allocator_policy':{'cap_bytes':8000000000,'cleanup_threshold_bytes':3000000000,'allocator_cap_applied':False},
        'compactor_sha256':UTILITY_PINS['compact_training_row_objects_v1.py'],
        'native4_adapter_sha256':SUPPORT_PINS['native_tdt_checkpointed_v4.py'],
        'immutable_original_config_identity_sha256':plan['original_config_identity_sha256'],
        'row_lr':plan['row_lr'],'shared_lr':plan['shared_lr'],
        'shared_updates_at_window_start':plan['shared_updates_at_window_start'],
        'approved_total_shared_head_updates':plan['phase_update_limit']}
    write(paths['native'],ready)
    sources = {'memory_launcher_sha256': sha(__file__), 'cache_launcher_sha256': sha(__file__),
        'cache_helper_sha256': pins['cuda_cache_reclaim_v2.py'], 'resource_helper_sha256': pins['resource_watchdog_v2.py'],
        'resource_telemetry_sha256': pins['paired_windows_commit_telemetry_v1.py'],
        'frozen_worker_sha256': WORKER_SHA, 'data_policy_sha256': pins['public_turkish_data_policy_v3.py']}
    identity = {'schema': 1, 'kind': 'owned_training_worker_identity_v1', 'pid': os.getpid(), 'ppid': os.getppid(),
        'expected_run_id': plan['run_id'], 'phase': PHASE, 'started_at_utc': now(),
        'worker_arguments_sha256': plan['worker_arguments_sha256'],
        'process_creation_identity': shared_memory.creation_identity(os.getpid()), 'live_memory_path': str(live),
        'actual_combined_dispatch_source': record(__file__), 'borrowed_implementation_sources': pins,
        'telemetry_before_torch_import': shared_memory.safe_sample(os.getpid(), 'before_torch_import'), **sources}
    write(paths['identity'], identity); identity_sha = sha(paths['identity']); lock = threading.Lock(); latest = [float('-inf')]
    def publish(value):
        observed = value.get('observed_monotonic', time.monotonic())
        with lock:
            if observed < latest[0]: return
            write(live, {'schema': 1, 'kind': 'owned_worker_live_memory_v1', 'pid': identity['pid'], 'ppid': identity['ppid'],
                'run_id': plan['run_id'], 'worker_identity_sha256': identity_sha,
                'process_creation_identity': identity['process_creation_identity'], **sources, 'sample': value}, retry_seconds=1.)
            latest[0] = observed
    publish(identity['telemetry_before_torch_import'])
    import torch
    cap = shared_memory.validate_capacity(torch.cuda.get_device_properties(0).total_memory, 8.)
    telemetry = shared_memory.UpdateTelemetry(identity['pid'], sink=publish)
    sampler = shared_memory.LiveTelemetrySampler(identity['pid'], publish)
    caller = make_caller(torch.cuda, HERE / WORKER, telemetry)
    memory = {'kind': 'TSC_BPE_adaptation_memory_runtime_v1', 'phase': PHASE, 'status': 'running', **sources, **cap,
        'cleanup_threshold_bytes': 3_000_000_000, 'native_max_joint_bytes': NATIVE_MAX_BYTES, 'run_id': plan['run_id'],
        'backend_flags_changed': False, 'allocator_scope': 'Owned process PyTorch CUDA allocator only',
        'borrowed_memory_implementation': record(shared_memory.__file__), 'started_at_utc': now()}
    cache = {'kind': 'TSC_BPE_adaptation_cache_runtime_v1', 'status': 'running', 'worker_sha256': WORKER_SHA,
        'inherited_cache_implementation': record(HERE / 'cuda_cache_reclaim_v2.py'), 'started_at_utc': now()}
    io = {'kind': 'TSC_BPE_adaptation_atomic_IO_v1', 'status': 'running', 'worker_sha256': WORKER_SHA,
        'phase_serializer_preserved': True, 'legacy207_optimizer_serializer_used': False,
        'helper': record(HERE / 'atomic_io_retry_v1.py'), 'started_at_utc': now()}
    retry_events = deque(maxlen=128); worker = None; original_save = original_write = benchmark_write = None
    outcome = 'failed'; error = None
    def event(value): retry_events.append(value)
    atomic_write = functools.partial(write, retry_seconds=30., event=event)
    for path, value in ((paths['memory'], memory), (paths['cache'], cache), (paths['io'], io)): write(path, value)
    try:
        torch.cuda.set_per_process_memory_fraction(cap['per_process_allocator_fraction'], device=0)
        memory['allocator_cap_applied'] = True; write(paths['memory'], memory)
        ready['allocator_policy']['allocator_cap_applied']=True;write(paths['native'],ready)
        sampler.start(); caller.install()
        import benchmark
        benchmark_write = benchmark.write_json; benchmark.write_json = atomic_write
        worker = importlib.import_module(Path(WORKER).stem)
        original_write = worker.write_json; worker.write_json = atomic_write; original_save = worker.save_training_state
        @functools.wraps(original_save)
        def save(path, *values, **keywords):
            return commit_writer(path, lambda target: original_save(target, *values, **keywords), retry_seconds=30., event=event)
        worker.save_training_state = save
        parsed = worker.validate_arguments(worker.parser().parse_args(argv)); worker.run(parsed)
        outcome = 'complete'
    except BaseException as exc:
        error = repr(exc); raise
    finally:
        sampler.stop_and_join(); caller.restore()
        if worker is not None:
            if original_save is not None: worker.save_training_state = original_save
            if original_write is not None: worker.write_json = original_write
        if benchmark_write is not None: benchmark.write_json = benchmark_write
        unchanged = all(sha(HERE / name) == digest for name, digest in pins.items())
        io.update(status=outcome, error=error, finished_at_utc=now(), retry_events=list(retry_events), sources_unchanged=unchanged)
        cache.update(status=outcome, error=error, finished_at_utc=now(), cache_reclamation=caller.evidence(), sources_unchanged=unchanged)
        native = sys.modules.get('native_tdt_checkpointed_v4')
        memory.update(status=outcome, error=error, finished_at_utc=now(), sources_unchanged=unchanged,
            owned_update_memory_telemetry=telemetry.evidence(), all_phase_owned_memory_telemetry=sampler.evidence(),
            native_cleanup_evidence=native.cache_cleanup_evidence() if native else {'available': False})
        memory['owned_update_memory_telemetry']['callback_scope'] = f'Pinned adaptation.run synchronize line{EXPECTED_SYNC_LINE}; observation only'
        for path, value in ((paths['memory'], memory), (paths['cache'], cache), (paths['io'], io)): write(path, value)
        write(completion, {'kind': 'TSC_BPE_adaptation_native_completion_v1', 'status': outcome, 'phase': PHASE,
            'run_id': plan['run_id'], 'pid': os.getpid(), 'sources_unchanged': unchanged,
            'backend_ready_sha256': sha(paths['native']), 'actual_dispatch_source_sha256': sha(__file__),
            'finished_at_utc': now(), 'error': error})
        require(unchanged, 'Source changed during BPE512+encoder-LoRA adaptation')

def validate_identity(value, plan, child_pid, started):
    require(value.get('kind') == 'owned_training_worker_identity_v1' and value.get('phase') == PHASE
        and value.get('expected_run_id') == plan['run_id']
        and value.get('worker_arguments_sha256') == plan['worker_arguments_sha256'], 'Actual adaptation/PID arguments differ')
    require(type(value.get('pid')) is int and value['pid'] > 0
        and (value['pid'] == child_pid or value.get('ppid') == child_pid), 'Worker is not the owned child/self')
    names = {'memory_launcher_sha256': Path(__file__).name, 'cache_launcher_sha256': Path(__file__).name,
        'cache_helper_sha256': 'cuda_cache_reclaim_v2.py', 'resource_helper_sha256': 'resource_watchdog_v2.py',
        'resource_telemetry_sha256': 'paired_windows_commit_telemetry_v1.py', 'frozen_worker_sha256': WORKER,
        'data_policy_sha256': 'public_turkish_data_policy_v3.py'}
    for key, name in names.items(): require(value.get(key) == plan['source_sha256'][name], 'Actual source differs: ' + key)
    creation = value.get('process_creation_identity', {})
    require(creation.get('available') is True and creation.get('pid') == value['pid']
        and domain.guards.stamp(started).timestamp() - 5 <= creation['creation_unix_seconds']
        <= domain.guards.stamp(value['started_at_utc']).timestamp() + 1, 'Owned creation identity differs')
    require(value.get('actual_combined_dispatch_source') == record(__file__), 'Actual phase dispatcher differs')
    return value


def adaptation_suffix(history,resume_step):
    require(all(row.get('step')==i for i,row in enumerate(history,1)) and len(history)>=resume_step,
        'Exact cumulative history prefix lost')
    suffix=history[resume_step:]
    for fresh,row in enumerate(suffix,1):
        require(row.get('adaptation_updates_completed')==row['step']-row['source_phase1_step'],
            'Exact actual localN differs')
    # Guard observation is fresh1..windowN; saved rows remain cumulative.
    return [{**row,'step':fresh} for fresh,row in enumerate(suffix,1)]

def verify_completion(plan,selection):
    paths={key:Path(value) for key,value in plan['paths'].items()};run=HERE/'runs'/plan['run_id']
    n=selection.get('adaptation_updates_completed');initial=plan['shared_updates_at_window_start']
    require(type(n) is int and initial<=n<=plan['phase_update_limit']
        and selection['run_id']==plan['run_id'] and selection['phase']==PHASE
        and selection['completed_steps']==plan['source_selected_step']+n
        and selection['source_phase1_step']==plan['source_selected_step']
        and selection['original_source_selected_step']==plan['original_source_selected_step']
        and selection['selected_acquisition_updates']==plan['selected_acquisition_updates']
        and selection['row_Adam_clock']==plan['selected_acquisition_updates']+n
        and selection['shared_Adam_clock']==plan['original_source_selected_step']+n
        and selection['external_exact_resume_supported'] is True and selection['further_unfreeze_allowed'] is False
        and selection['heldout_tests_evaluated'] is False,'Exact S-K-N/latest/BEST frontier required')
    identity=read(paths['identity']);completion=read(paths['native'].with_suffix('.completion.json'))
    require(completion.get('kind')=='TSC_BPE_adaptation_native_completion_v1' and completion.get('status')=='complete'
        and completion['sources_unchanged'] is True and completion['pid']==identity['pid']
        and completion['run_id']==plan['run_id'] and completion['backend_ready_sha256']==sha(paths['native'])
        and completion['actual_dispatch_source_sha256']==sha(__file__),'Actual owned exact completion required')
    for key,kind in (('io','TSC_BPE_adaptation_atomic_IO_v1'),('cache','TSC_BPE_adaptation_cache_runtime_v1'),('memory','TSC_BPE_adaptation_memory_runtime_v1')):
        value=read(paths[key]);require(value['kind']==kind and value['status']=='complete' and value['sources_unchanged'] is True,
            'Actual exact runtime missing:'+key)
    memory=read(paths['memory']);cache=read(paths['cache'])
    require(memory['cap_bytes']==8000000000 and memory['cleanup_threshold_bytes']==3000000000
        and memory['allocator_cap_applied'] is True and memory['all_phase_owned_memory_telemetry']['thread_join_completed'] is True
        and cache['cache_reclamation']['eligible_update_calls']==n-initial,'Cap8/end3/join/current-window suffix hook required')
    config=read(run/'run_config.json');history=read(run/'training_history.json')
    require(canonical(config['config_identity'])==plan['original_config_identity_sha256']
        and len(history)==selection['completed_steps']
        and canonical(history[:plan['resume_step']])==plan['source_history_prefix_sha256'],
        'Original config/history prefix cannot be reset or relabelled')
    require(record(plan['original_native_ready']['path'])=={key:plan['original_native_ready'][key] for key in ('path','sha256')}
        and record(run/PROOF_NAME)=={key:plan['original_source300'][key] for key in ('path','sha256')},
        'Original immutable native/source300 tokens changed')
    events=list(run.glob('shared_head_exact_execution_from_'+str(plan['resume_step'])+'_'+plan['source_closure']['sha256'][:16]+'.json'))
    require(len(events)==1,'One exact execution event required');execution=read(events[0])
    require(selection['exact_execution']==execution and execution['original_config_or_BEST_reinitialized'] is False
        and execution['source300_repeated'] is False and execution['approved_total_shared_head_updates']==plan['phase_update_limit'],
        'Separate stopping extension/executor evidence changed')
    startup=read(events[0].with_name(events[0].stem+'_startup.json'));compact=startup['training_row_compaction']
    require(startup['kind']=='shared_heads_exact_startup_compaction_v1' and startup['passed'] is True
        and compact['passed'] is True and compact['rows_verified']==253798
        and compact['all_rows_canonical_equal'] is True and compact['all_row_key_orders_equal'] is True
        and compact['same_row_list_and_row_dict_objects'] is True and compact['same_label_list_objects'] is True,
        'Actual exact cold metadata compaction required')
    source_pins(boundary_context(plan['worker_arguments']))
    return {'new_adaptation_updates':n-initial,'shared_updates_at_window_start':initial,
        'completed_shared_updates':n,'native_completion':record(paths['native'].with_suffix('.completion.json')),
        'exact_execution':record(events[0]),'original_config_unchanged':True,
        'full_unique_TRAIN_coverage':selection['full_unique_TRAIN_coverage'],'goal_complete':False}

def request_stop(path, reason):
    # Preserve an existing ROOT stop; never overwrite it or mistake it for budget.
    if not path.exists():
        with path.open('x', encoding='utf-8') as stream: stream.write('Cooperative whole-update save/DEV: ' + reason + '\n')


def run_window(state, deadline, args):
    paths = {key: Path(value) for key, value in state['paths'].items()}; loop = scoped(args.output_dir)
    child = None; stop_at = stop_reason = None; started = time.monotonic(); started_utc = now(); last_phase = None
    inherited = state['resume_step']; last_dev = inherited
    resource = ResourceGuard(); progress = domain.guards.ProgressGuard(started)
    try:
        with paths['log'].open('x', encoding='utf-8') as log:
            child = subprocess.Popen(state['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state.update(status='running', owned_launcher_pid=child.pid, started_at_utc=started_utc)
            print(f"ACTUAL owned {state['run_id']} PID{child.pid}; 15-tensor/two-Adam adaptation bound{state['phase_update_limit']}", flush=True)
            while child.poll() is None:
                history = read(HERE / 'runs' / state['run_id'] / 'training_history.json', [])
                suffix = adaptation_suffix(history, inherited) if history else []
                identity = validate_identity(read(paths['identity']), state, child.pid, started_utc) if paths['identity'].exists() else None
                phase = domain.guards.log_phase(paths['log'], suffix)
                sample = sample_memory(identity['pid'] if identity else child.pid, phase['phase'])
                reason, observation = resource.observe(sample['system_memory'], sample['worker_private_memory'],
                    observed_monotonic=sample['observed_monotonic'])
                self_reason, self_observation = domain.guards.self_memory(Path(str(paths['identity']) + '.live_memory.json'),
                    identity, paths['identity']) if identity else (None, {'available': False})
                progress_reason, progress_info = progress.observe(suffix, time.monotonic())
                reason = reason or self_reason or progress_reason
                if not identity and time.monotonic() - started > 120: reason = reason or 'owned_worker_identity_unavailable'
                if identity and not self_observation['available'] and time.monotonic() - started > 120:
                    reason = reason or 'actual_worker_self_memory_unavailable_or_stale'
                if not suffix and time.monotonic() - started > 1800: reason = reason or 'bounded_preparation_timeout'
                if paths['stop'].exists() and stop_at is None: reason = reason or 'ROOT_manual_cooperative_stop'
                if time.monotonic() >= deadline - 300: reason = reason or 'window_budget_checkpoint_boundary'
                state.update(phase=phase, new_window_updates_published=len(suffix), adaptation_updates_published=state['shared_updates_at_window_start']+len(suffix),
                    published_step=history[-1]['step'] if history else inherited,
                    system_and_owned_private_memory=observation, actual_worker_self_memory=self_observation,
                    progress=progress_info, updated_at_utc=now())
                if phase['phase'] != last_phase:
                    print(state['run_id'] + ' phase: ' + phase['phase'], flush=True); last_phase = phase['phase']
                for row in history[inherited:]:
                    if row['step'] > last_dev and 'validation' in row:
                        gate = row['validation']; print(f"{state['run_id']} DEV{row['step']}: TR{100 * gate['turkish_wer']:.4f}%, all5={gate['retention_gate_passed']}", flush=True)
                        last_dev = row['step']
                with (loop / 'memory_observations.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps({'at_utc': now(), 'phase': phase, 'memory': observation,
                        'worker_self': self_observation, 'progress': progress_info}) + '\n')
                if reason and stop_at is None:
                    stop_at = time.monotonic(); stop_reason = reason; request_stop(paths['stop'], reason)
                    state.update(cooperative_stop_reason=reason, cooperative_stop_requested_at_utc=now())
                write(loop / 'loop_status.json', state)
                if stop_at and time.monotonic() - stop_at > 300:
                    raise RuntimeError('Owned child exceeded cooperative grace; preserve and review')
                time.sleep(args.poll_seconds)
        state['child_exit_code'] = child.wait()
        require(state['child_exit_code'] == 0, 'Actual adaptation child failed; no automatic GPU retry')
        selection_path = HERE / 'runs' / state['run_id'] / 'selection.json'; selection = read(selection_path, {})
        state['actual_runtime_completion'] = verify_completion(state, selection)
        resource_stop = bool(stop_reason and stop_reason not in ('window_budget_checkpoint_boundary', 'ROOT_manual_cooperative_stop'))
        category = ('resource' if resource_stop else 'manual' if stop_reason == 'ROOT_manual_cooperative_stop'
            else 'budget' if stop_reason == 'window_budget_checkpoint_boundary' or selection.get('stop_early_reason') == 'soft_max_wall_seconds'
            else 'retention' if selection.get('stop_early_reason') == 'consecutive_retention_gate_failures'
            else 'phase_bound' if selection['adaptation_updates_completed'] == state['phase_update_limit']
            else selection.get('stop_early_reason', 'review'))
        state.update(status='resource_review_required' if resource_stop else 'train_window_ended',
            selected_step=selection['selected_step'], completed_steps=selection['completed_steps'],
            adaptation_updates_completed=selection['adaptation_updates_completed'], selection=record(selection_path),
            stop_early_reason=selection.get('stop_early_reason'), resource_review_required=resource_stop,
            stop_category=category, finished_at_utc=now(), objective_complete=False,
            next_action='ROOT reviews actual qualifying latest/full coverage before a separately approved CLOSED exact successor; no automatic reset/unfreeze')
        return 1 if resource_stop else 0
    finally:
        if child is not None and child.poll() is None:
            request_stop(paths['stop'], 'Controller exit')
            try: child.wait(timeout=300)
            except subprocess.TimeoutExpired: child.terminate(); child.wait(timeout=30)

def describe_contract():
    return {'kind':CONTROLLER_KIND,'status':'sources_pinned_only','phase':PHASE,
        'SAME_ID_qualified_improving_latest_only':True,'earlier_BEST_fork_supported':False,
        'full207_twoAdam_current_sampler_RNG_history_preserved':True,'original_config1000_native_source300_immutable':True,
        'approved_totalN_extension_separate':True,'maximum_global_horizon':44958,'budget_alone_proves_fullcoverage':False,
        'cap_bytes':8000000000,'end_cache_bytes':3000000000,'native_max_joint_bytes':NATIVE_MAX_BYTES,
        'worker_sha256':WORKER_SHA,'boundary_sha256':TRANSITION_SHA,'sync_line':EXPECTED_SYNC_LINE,
        'GPU_launched':False,'actual_state_or_fit_claimed':False,'unfreeze_supported':False}

def main(args):
    if args.describe_contract:
        print(json.dumps(describe_contract(), sort_keys=True)); return 0
    if args.worker_plan:
        require(args.gpu_window_authorized, 'ROOT exclusive GPU authorization required')
        run_worker(read_small(scoped(args.worker_plan))); return 0
    require(all((args.source_loop, args.source_receipt, args.run_id, args.output_dir)), 'Source/receipt/fresh run/output required')
    plan = prepare(args)
    if args.validate_only:
        if args.preflight_output:
            output = scoped(args.preflight_output); require(not output.exists(), 'Never overwrite a preflight receipt')
            require(not output.is_relative_to(scoped(args.output_dir)), 'Preflight must not create the future controller namespace')
            write(output, {**plan, 'passed': True, 'Torch_imported': 'torch' in sys.modules, 'GPU_launched': False})
        print(json.dumps({'passed': True, 'run_id': plan['run_id'], 'GPU_launched': False,
            'Torch_imported': 'torch' in sys.modules, 'actual_GPU_equality_or_fit_executed': False})); return 0
    require(args.gpu_window_authorized, 'ROOT must authorize sole owned GPU child after actual source closure')
    loop = scoped(args.output_dir); loop.mkdir(parents=True, exist_ok=False)
    state = {**plan, 'status': 'starting', 'absolute_deadline_utc': datetime.fromtimestamp(
        time.time() + args.window_hours * 3600, timezone.utc).isoformat()}
    deadline = time.monotonic() + args.window_hours * 3600
    state['awake_request'] = domain.guards.prior.keep_awake(True)
    try:
        write(loop / 'plan.json', plan); write(loop / 'loop_status.json', state)
        return run_window(state, deadline, args)
    except BaseException as error:
        state.update(status='failed', error=repr(error), finished_at_utc=now()); raise
    finally:
        state['awake_clear'] = domain.guards.prior.keep_awake(False); write(loop / 'loop_status.json', state)

def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--describe-contract',action='store_true');result.add_argument('--worker-plan',type=Path)
    result.add_argument('--source-loop',type=Path);result.add_argument('--source-receipt',type=Path)
    result.add_argument('--run-id');result.add_argument('--output-dir',type=Path)
    result.add_argument('--approved-total-shared-head-updates',type=int)
    result.add_argument('--window-hours',type=float,default=3.);result.add_argument('--poll-seconds',type=float,default=5.)
    result.add_argument('--validate-only',action='store_true');result.add_argument('--preflight-output',type=Path)
    result.add_argument('--gpu-window-authorized',action='store_true');return result

if __name__ == '__main__': sys.exit(main(parser().parse_args()))
