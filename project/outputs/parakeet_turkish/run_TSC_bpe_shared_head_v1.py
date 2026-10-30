"""INACTIVE one-child controller for a declared TSC BPE512 + shared-head phase.

The worker/boundary hashes and optimizer-end callback are pinned below.
--describe-contract and --help are stdlib-only; no
actual source, model, checkpoint, corpus or GPU is read by those modes.

Only new rows8199:8711 of three tables and12 shared-head tensors update.
The three-table Adam continues phase1 clockK at1e-4; a separate shared-head Adam copies
original named12 shared-head moments atclockS before the declared shared LR.
Original192 encoder-LoRA moments and original3 resized-table moments stay dormant. Initial300 unsuppressed DEV
path equality is required before phase2 updates. Exact phase continuation
and unfreezing are unsupported. ROOT alone may authorize an exclusive child. Native4 adds only a split-boundary
inactive-cache cleanup; the validated critical probe retained the same overall
reserved peak, so recurrence prevention is not claimed.
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
WORKER = 'train_lora_batched_tsc_bpe_shared_head_v1.py'
TRANSITION = 'tsc_bpe_shared_head_transition_v1.py'
WORKER_SHA = '4f632557d986393d9b14167f4a3a697e45beb6b960229e9a4c3af64c2d592b1c'
TRANSITION_SHA = '5f7a9fc31721466e60868f3c75153b6bd27591dd0fbc45e53d49ada0cf38620c'
EXPECTED_SYNC_LINE = 501
PHASE = 'HF_TSC_BPE512_new_rows_plus_shared_heads_v1'
STATE_KIND = 'HF_TSC_BPE512_new_rows_plus_shared_heads_state_v1'
CONTROLLER_KIND = 'guarded_TSC_BPE512_new_rows_plus_shared_heads_controller_v1'
PROOF_NAME = 'TSC_BPE_shared_head_initial_source_equivalence.json'
PROOF_KIND = 'TSC_BPE512_shared_heads_before_update_source_equivalence_v1'
NATIVE_MAX_BYTES = 1_850_000_000
GEOMETRY = {'student_vocab_size': 8711, 'joint_width': 8716, 'blank_id': 8192,
    'teacher_vocab_size': 8193, 'new_BPE_rows': [8199, 8711],
    'protected_six_rows': [8193, 8199], 'duration_rows': [8711, 8716]}
UTILITY_PINS = {
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
            self.line = caller_line(self.worker_path); self.threshold_bytes = 4_000_000_000; self.max_events = 128
            self.events = deque(maxlen=128)
            self.eligible_calls = self.ignored_calls = self.reclamations = self.failed_reclamations = 0
            self.cleanup_seconds = 0.; self.original = self.wrapped = None
    return shared_memory.telemetry_reclaimer_class(AdaptationCaller, telemetry)(cuda, worker_path)


def check_phase_arguments(argv):
    for flag in ('--resume-exact', '--resume-weights', '--resume-runtime-migration', '--resume-optimization',
                 '--optimization-source-receipt', '--resume-acquisition', '--acquisition-source-receipt', '--unfreeze'):
        require(flag not in argv, 'Unsupported adaptation state mode: ' + flag)
    require(option(argv, '--batch-size') == '8' and option(argv, '--accumulation') == '2'
        and option(argv, '--sampling-mode') == 'domain30'
        and float(option(argv, '--fleurs-fraction')) == .3
        and float(option(argv, '--row-lr')) in (1e-3, 3e-4)
        and float(option(argv, '--shared-lr')) in (1e-5, 3e-6)
        and float(option(argv, '--lora-lr')) == 3e-4
        and int(option(argv, '--native-max-joint-bytes')) == NATIVE_MAX_BYTES,
        'Keep B8x2/domain30 and explicit two-source Adam/bound')
    bound = int(option(argv, '--adaptation-max-updates'))
    require(0 < bound <= 1000, 'Positive bounded adaptation-only update limit required')
    smoke = int(option(argv, '--smoke-updates')) if '--smoke-updates' in argv else 0
    require(smoke in (0, 1), 'Only the separately declared one-update smoke is supported')
    return bound, smoke


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
    """No worker/ML import: use only the reviewed pure boundary's admission."""
    transition = importlib.import_module(Path(TRANSITION).stem)
    receipt = read_small(option(argv, '--shared-head-source-receipt'))
    source_config = read_small(receipt['artifacts']['run_config']['path'])
    source_arguments = read_small(receipt['artifacts']['controller_status']['path'])['worker_arguments']
    require(copied_scientific_arguments(argv) == copied_scientific_arguments(source_arguments),
        'Every inherited science/data/seed/cadence CLI token must be copied exactly; only declared phase/runtime options differ')
    values = dict(source_config); values.update(source_config['config_identity']['training_arguments'])
    values.update(run_id=option(argv, '--run-id'), resume_shared_heads=scoped(option(argv, '--resume-shared-heads')),
        shared_head_source_receipt=scoped(option(argv, '--shared-head-source-receipt')),
        candidate_dir=scoped(option(argv, '--candidate-dir')),
        candidate_tr_metadata=scoped(option(argv, '--candidate-tr-metadata')),
        candidate_tr_metadata_proof=scoped(option(argv, '--candidate-tr-metadata-proof')),
        row_lr=float(option(argv, '--row-lr')), shared_lr=float(option(argv, '--shared-lr')), native_max_joint_bytes=NATIVE_MAX_BYTES,
        adaptation_max_updates=int(option(argv, '--adaptation-max-updates')),
        smoke_updates=int(option(argv, '--smoke-updates')) if '--smoke-updates' in argv else None,
        resume_exact=None, resume_weights=None, resume_runtime_migration=None, resume_optimization=None,
        optimization_source_receipt=None, resume_acquisition=None, acquisition_source_receipt=None,
        native_backend_provenance=scoped(option(argv, '--native-backend-provenance')),
        stop_file=scoped(option(argv, '--stop-file')), max_wall_seconds=float(option(argv, '--max-wall-seconds')))
    return transition.preflight(SimpleNamespace(**values))


def validate_plan(plan):
    finalized_core()
    argv = plan['worker_arguments']; bound, smoke = check_phase_arguments(argv)
    context = boundary_context(argv); pins = source_pins(context)
    require(plan['kind'] == CONTROLLER_KIND and plan['source_sha256'] == pins
        and plan['controller_sha256'] == sha(__file__) and option(argv, '--run-id') == plan['run_id']
        and domain.guards.argument_sha(argv) == plan['worker_arguments_sha256'], 'Frozen phase plan/arguments differ')
    require(plan['source_closure'] == context['receipt_record']
        and plan['source_checkpoint'] == context['receipt']['artifacts']['chosen_checkpoint']
        and plan['source_selected_step'] == context['source_selected_step']
        and plan['original_source_selected_step'] == context['original_step']
        and plan['selected_acquisition_updates'] == context['acquisition_updates']
        and plan['source_selected_step'] == plan['original_source_selected_step'] + plan['selected_acquisition_updates']
        and plan['selected_acquisition_updates'] > 0
        and plan['chosen_checkpoint_role'] == context['receipt']['chosen_checkpoint_role']
        and plan['original207_checkpoint'] == context['original_context']['receipt']['artifacts']['best_checkpoint']
        and plan['source_recipe'] == context['source_recipe']
        and plan['row_lr'] == float(option(argv, '--row-lr')) == context['receipt']['declared_shared_heads']['row_lr']
        and plan['shared_lr'] == float(option(argv, '--shared-lr')) == context['receipt']['declared_shared_heads']['shared_lr']
        and plan['adaptation_max_updates'] == bound and plan['smoke_updates'] == smoke,
        'Use the explicit immutable CLOSED qualified acquired point; never infer LATEST equals BEST')
    require(math.isfinite(plan['window_hours']) and .1 <= plan['window_hours'] <= 3.
        and float(option(argv, '--max-wall-seconds')) == plan['window_hours'] * 3600 - 300,
        'Phase child must retain the declared bounded window and cooperative grace')
    paths = {key: scoped(value) for key, value in plan['paths'].items()}
    require(len(set(paths.values())) == len(paths) and option(argv, '--native-backend-provenance') == str(paths['native'])
        and option(argv, '--stop-file') == str(paths['stop']), 'Distinct fresh runtime paths required')
    caller_line(HERE / WORKER)
    return paths


def prepare(args):
    finalized_core()
    require(not args.resume_exact and not args.resume_closed_loop, 'Exact adaptation continuation is not wired')
    require(math.isfinite(args.window_hours) and .1 <= args.window_hours <= 3., 'Whole-update window must be bounded <=3h')
    require(5 <= args.poll_seconds <= 15, 'Fresh guard polls require5 to15 seconds')
    require(type(args.adaptation_max_updates) is int and 0 < args.adaptation_max_updates <= 1000,
        'Separate positive adaptation phase bound required')
    require(args.smoke_updates in (None, 1), 'Smoke mode is exactly one adaptation update')
    output = scoped(args.output_dir); source_loop = scoped(args.source_loop)
    require(output.parent == HERE / 'runs' and not output.exists(), 'Fresh controller directory under runs required')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{2,110}', args.run_id) is not None
        and not (HERE / 'runs' / args.run_id).exists() and output != HERE / 'runs' / args.run_id,
        'Fresh separate worker namespace required')
    for suffix in ('best.pt', 'latest.pt'):
        require(not (ROOT / 'work' / f'{args.run_id}_{suffix}').exists(), 'Worker phase checkpoint namespace already exists')
    receipt_path = scoped(args.source_receipt); receipt = read_small(receipt_path)
    artifact = receipt['artifacts']['controller_status']
    original_path = receipt.get('original_artifact_paths', {}).get('controller_status', artifact['path'])
    require((source_loop / 'loop_status.json').resolve() == scoped(original_path),
        'Source loop must identify the ROOT-archived original controller')
    require(record(artifact['path']) == {key: artifact[key] for key in ('path', 'sha256')},
        'Immutable archived ROOT-owned controller closure changed')
    source = read_small(artifact['path'])
    argv = list(source['worker_arguments']); original = list(argv)
    paths = {key: output / f'{args.run_id}_{name}' for key, name in (
        ('native', 'native_backend.json'), ('identity', 'identity.json'), ('memory', 'memory.json'),
        ('cache', 'cache.json'), ('io', 'io.json'), ('stop', 'stop.txt'), ('log', 'worker.log'))}
    for flag in ('--resume-exact', '--resume-weights', '--resume-runtime-migration', '--resume-optimization',
                 '--optimization-source-receipt', '--resume-acquisition', '--acquisition-source-receipt', '--acquisition-max-updates', '--resume-shared-heads', '--shared-head-source-receipt', '--acquisition-lr', '--row-lr', '--shared-lr', '--smoke-updates'):
        argv = remove_option(argv, flag)
    chosen = scoped(receipt['artifacts']['chosen_checkpoint']['path'])
    for flag, value in (('--run-id', args.run_id), ('--resume-shared-heads', chosen),
        ('--shared-head-source-receipt', receipt_path), ('--row-lr', args.row_lr), ('--shared-lr', args.shared_lr),
        ('--adaptation-max-updates', args.adaptation_max_updates), ('--candidate-dir', scoped(args.candidate_dir)),
        ('--candidate-tr-metadata', scoped(args.candidate_tr_metadata)),
        ('--candidate-tr-metadata-proof', scoped(args.candidate_tr_metadata_proof)),
        ('--native-max-joint-bytes', NATIVE_MAX_BYTES), ('--native-backend-provenance', paths['native']),
        ('--stop-file', paths['stop']), ('--max-wall-seconds', args.window_hours * 3600 - 300)):
        argv = set_option(argv, flag, value)
    if args.smoke_updates is not None: argv = set_option(argv, '--smoke-updates', args.smoke_updates)
    bound, smoke = check_phase_arguments(argv); context = boundary_context(argv); pins = source_pins(context)
    python = scoped(source['command'][0]); require(python.is_file(), 'Closed source native-env Python unavailable')
    plan = {'schema': 1, 'kind': CONTROLLER_KIND, 'phase': PHASE, 'run_id': args.run_id, 'loop_id': output.name,
        'controller_sha256': sha(__file__), 'source_loop': str(source_loop), 'source_closure': context['receipt_record'],
        'source_checkpoint': context['receipt']['artifacts']['chosen_checkpoint'],
        'source_selected_step': context['source_selected_step'], 'source_history_prefix_sha256': context['source_history_prefix_sha256'],
        'source_phase1_step': context['source_selected_step'], 'original_source_selected_step': context['original_step'],
        'selected_acquisition_updates': context['acquisition_updates'],
        'chosen_checkpoint_role': context['receipt']['chosen_checkpoint_role'],
        'original207_checkpoint': context['original_context']['receipt']['artifacts']['best_checkpoint'],
        'original_champion_preserved': True,
        'source_recipe': context['source_recipe'], 'source_original_worker_arguments': original,
        'source_sha256': pins, 'worker_arguments': argv, 'worker_arguments_sha256': domain.guards.argument_sha(argv),
        'command': [str(python), str(Path(__file__).resolve()), '--worker-plan', str(output / 'plan.json'), '--gpu-window-authorized'],
        'paths': {key: str(value) for key, value in paths.items()}, 'window_hours': args.window_hours,
        'adaptation_max_updates': bound, 'smoke_updates': smoke, 'phase_update_limit': smoke or bound,
        'inherited_scientific_coverage_horizon': context['source_recipe']['steps'],
        'adaptation_updates_at_start': 0, 'row_Adam_source_clock': context['acquisition_updates'],
        'shared_Adam_source_clock': context['original_step'], 'original3_resized_table_moments_loaded': False,
        'original192_encoder_LoRA_moments_loaded': False,
        'named12_original_shared_head_moments_copied': True,
        'initial_actual300_DEV_source_equality_before_shared_head_updates_required': True,
        'native_max_joint_bytes': NATIVE_MAX_BYTES, 'native_joint_bound_is_VRAM_measurement': False,
        'declared_shared_heads': importlib.import_module(Path(TRANSITION).stem).shared_head_declaration(args.row_lr, args.shared_lr),
        'row_lr': args.row_lr, 'shared_lr': args.shared_lr,
        'same_phase_exact_continuation_supported': False, 'unfreeze_supported': False,
        'heldout_TEST_evaluated': False, 'objective_complete': False, 'objective_status': 'incomplete'}
    validate_plan(plan)
    return plan


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
    flags = ('--tsc-input-proof', '--tsc-training-approval', '--source-closure', '--tr-manifest', '--replay-manifest',
        '--tr-frame-metadata', '--tr-frame-metadata-proof', '--data-v3-proof', '--resume-shared-heads',
        '--shared-head-source-receipt', '--candidate-tr-metadata', '--candidate-tr-metadata-proof')
    assets = {flag[2:].replace('-', '_'): record(option(argv, flag)) for flag in flags}
    expected = plan['source_checkpoint']
    require(assets['resume_shared_heads'] == {key: expected[key] for key in ('path', 'sha256')},
        'Actual CLOSED chosen phase1 bytes differ; never substitute BEST/latest or an older point')
    original207 = plan['original207_checkpoint']
    require(record(original207['path']) == {key: original207[key] for key in ('path', 'sha256')},
        'Original immutable207 archive differs before named LoRA moment restoration')
    ready = {'schema': 1, 'kind': 'native_tdt_training_backend_v3', 'status': 'ready', 'phase': PHASE,
        'adaptation_phase': PHASE, 'geometry': GEOMETRY,
        'declared_shared_heads': plan['declared_shared_heads'], 'run_id': plan['run_id'],
        'tdt_implementation': 'nemo', 'preload': preloaded, 'pid': os.getpid(), 'ppid': os.getppid(), 'ready_at_utc': now(),
        'trainer_path': str(HERE / WORKER), 'trainer_sha256': WORKER_SHA,
        'adapter_path': str(HERE / 'native_tdt_checkpointed_v4.py'), 'adapter_sha256': SUPPORT_PINS['native_tdt_checkpointed_v4.py'],
        'launcher_path': str(Path(__file__).resolve()), 'launcher_sha256': sha(__file__),
        'memory_launcher_path': str(Path(__file__).resolve()), 'memory_launcher_sha256': sha(__file__),
        'worker_arguments_sha256': plan['worker_arguments_sha256'], 'TSC_explicit_asset_records': assets,
        'declared_allocator_policy': {'cap_gb': 8, 'end_cache_gb': 4, 'native_inactive_threshold_bytes': 750000000},
        'native_max_joint_bytes': NATIVE_MAX_BYTES, 'vocab_size': 8711, 'joint_width': 8716, 'blank_id': 8192,
        'duration_rows': [8711, 8716], 'trainable_rows': [8199, 8711], 'new_rows_update_multiplier': 1,
        'six_character_update_multiplier': 0, 'original3_resized_table_moments_active': False,
        'original192_encoder_LoRA_moments_active': False, 'original12_shared_head_moments_active': True,
        'original207_archive_immutable': True, 'named12_original_shared_head_moments_copied': True,
        'active_parameter_tensors': 15, 'active_shared_head_tensors': 12, 'frozen_encoder_LoRA_tensors': 192,
        'row_lr': plan['row_lr'], 'shared_lr': plan['shared_lr'], 'row_Adam_source_clock': plan['selected_acquisition_updates'],
        'shared_Adam_source_clock': plan['original_source_selected_step'],
        'source_selected_step': plan['source_selected_step'], 'source_phase1_step': plan['source_selected_step'],
        'original_source_selected_step': plan['original_source_selected_step'],
        'selected_acquisition_updates': plan['selected_acquisition_updates'],
        'chosen_checkpoint_role': plan['chosen_checkpoint_role'],
        'original207_checkpoint': original207, 'adaptation_max_updates': plan['adaptation_max_updates'],
        'smoke_updates': plan['smoke_updates'], 'phase_updates_at_start': 0,
        'native_full_model_or_Lightning_trainer_equivalence_claimed': False}
    write(paths['native'], ready)
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
        'cleanup_threshold_bytes': 4_000_000_000, 'native_max_joint_bytes': NATIVE_MAX_BYTES, 'run_id': plan['run_id'],
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


def adaptation_suffix(history, source_step):
    """Read-only guard view: shared fresh-stage guard receives local1..N clocks."""
    require(all(row.get('step') == index for index, row in enumerate(history, 1)), 'Cumulative saved history is not contiguous')
    require(len(history) >= source_step, 'Inherited selected history was lost')
    suffix = history[source_step:]
    for local, row in enumerate(suffix, 1):
        require(row.get('adaptation_updates_completed') == local, 'Adaptation suffix clock differs')
    return [{**row, 'step': local} for local, row in enumerate(suffix, 1)]


def verify_completion(plan, selection):
    paths = {key: Path(value) for key, value in plan['paths'].items()}; run = HERE / 'runs' / plan['run_id']
    require(selection.get('run_id') == plan['run_id'] and selection.get('heldout_tests_evaluated') is False,
        'Durable DEV-only adaptation selection required')
    local = selection.get('adaptation_updates_completed'); inherited = plan['source_selected_step']
    require(type(local) is int and 0 <= local <= plan['phase_update_limit']
        and selection.get('source_phase1_step') == selection.get('inherited_history_steps') == inherited
        and selection.get('original_source_selected_step') == plan['original_source_selected_step']
        and selection.get('selected_acquisition_updates') == plan['selected_acquisition_updates']
        and selection.get('completed_steps') == inherited + local
        and selection.get('row_Adam_clock') == plan['selected_acquisition_updates'] + local
        and selection.get('shared_Adam_clock') == plan['original_source_selected_step'] + local,
        'Distinct phase2N / rowK+N / LoRAS+N / cumulativeS+K+N frontiers required')
    require(selection.get('phase') == PHASE and selection.get('external_exact_resume_supported') is False
        and selection.get('further_unfreeze_allowed') is False,
        'No exact continuation claim may be inferred from a phase checkpoint')
    identity = read(paths['identity'], {}); completion = read(paths['native'].with_suffix('.completion.json'), {})
    require(completion.get('kind') == 'TSC_BPE_adaptation_native_completion_v1' and completion.get('status') == 'complete'
        and completion.get('phase') == PHASE and completion.get('pid') == identity['pid']
        and completion.get('run_id') == plan['run_id'] and completion.get('sources_unchanged') is True
        and completion.get('backend_ready_sha256') == sha(paths['native'])
        and completion.get('actual_dispatch_source_sha256') == sha(__file__), 'Actual owned native completion required')
    for key, kind in (('io', 'TSC_BPE_adaptation_atomic_IO_v1'), ('cache', 'TSC_BPE_adaptation_cache_runtime_v1'),
                      ('memory', 'TSC_BPE_adaptation_memory_runtime_v1')):
        value = read(paths[key], {})
        require(value.get('kind') == kind and value.get('status') == 'complete'
            and value.get('sources_unchanged') is True, 'Actual phase runtime completion missing: ' + key)
    memory = read(paths['memory']); cache = read(paths['cache'])
    require(memory.get('cap_bytes') == 8_000_000_000 and memory.get('cleanup_threshold_bytes') == 4_000_000_000
        and memory.get('allocator_cap_applied') is True
        and memory.get('all_phase_owned_memory_telemetry', {}).get('thread_join_completed') is True,
        'Actual cap8/end4/all-phase telemetry completion required')
    require(cache['cache_reclamation']['eligible_update_calls'] == local, 'Hook must count only fresh phase2 whole updates')
    proof = read(run / PROOF_NAME, {})
    require(proof.get('kind') == PROOF_KIND and proof.get('run_id') == plan['run_id'] and proof.get('passed') is True
        and proof.get('actual_GPU_decode') is True and proof.get('before_shared_head_updates') is True
        and proof.get('compared_fixed_DEV_utterances') == 300 and proof.get('unsuppressed_new512_ids') is True
        and proof.get('all_raw_transcripts_and_emitted_paths_exact') is True and proof.get('all_six_ASR_counts_exact') is True
        and proof.get('all_five_retention_gates_passed') is True
        and proof.get('RNG_restored_after_added_evaluation') is True
        and proof.get('module_training_modes_restored_after_added_evaluation') is True
        and proof.get('source_receipt') == plan['source_closure'] and proof.get('source_selected_step') == inherited,
        'Actual unsuppressed300 source preservation before phase2 updates required')
    config = read(run / 'run_config.json'); history = read(run / 'training_history.json')
    require(canonical(history[:inherited]) == plan['source_history_prefix_sha256']
        and len(adaptation_suffix(history, inherited)) == local, 'Inherited chosen phase1 history or adaptation suffix differs')
    identity_config = config['config_identity']; actual_arguments = identity_config['training_arguments']
    require(config.get('run_id') == plan['run_id'] and config.get('phase') == identity_config.get('phase') == PHASE
        and all(actual_arguments.get(key) == value for key, value in plan['source_recipe'].items()),
        'Inherited dormant recipe metadata changed')
    require(identity_config.get('geometry') == GEOMETRY and identity_config.get('adaptation_phase') == {
        'kind': PHASE, 'source_phase1_step': inherited,
        'original_source_selected_step': plan['original_source_selected_step'],
        'selected_acquisition_updates': plan['selected_acquisition_updates'],
        'active_parameter_tensors': 15, 'active_shared_head_tensors': 12, 'frozen_encoder_LoRA_tensors': 192, 'row_lr': plan['row_lr'], 'shared_lr': plan['shared_lr'],
        'max_updates': plan['adaptation_max_updates'], 'effective_max_updates': plan['phase_update_limit'],
        'original12_shared_head_moments_active': True, 'original3_resized_table_moments_active': False, 'original192_encoder_LoRA_moments_active': False, 'original207_Adam_artifact_retained_dormant': True,
        'further_unfreeze_allowed': False,
        'external_exact_resume_supported': False},
        'Actual 15-tensor/two-Adam phase identity differs')
    optimizers = identity_config.get('adaptation_optimizers', {})
    require(optimizers.get('kind') == 'HF_TSC_BPE512_new_rows_plus_shared_heads_identity_v1'
        and optimizers.get('source_selected_step') == plan['original_source_selected_step']
        and optimizers.get('selected_acquisition_updates') == plan['selected_acquisition_updates']
        and optimizers.get('active_parameter_tensors') == 15
        and optimizers.get('active_shared_head_tensors') == 12
        and optimizers.get('frozen_encoder_LoRA_tensors') == 192
        and optimizers.get('active_row_lr') == plan['row_lr'] and optimizers.get('active_shared_lr') == plan['shared_lr']
        and optimizers.get('original12_shared_moments_active') is True
        and optimizers.get('original3_resized_table_moments_active') is False
        and optimizers.get('original192_encoder_LoRA_moments_active') is False
        and optimizers.get('further_unfreeze_allowed') is False,
        'Separate rowK / LoRAS optimizer identities and protected12 heads required')
    if plan['smoke_updates']:
        smoke = read(run / 'adaptation_actual_one_update_smoke.json', {})
        require(local == 1 and smoke.get('kind') == 'TSC_BPE_shared_heads_actual_one_update_smoke_v1'
            and smoke.get('run_id') == plan['run_id'] and smoke.get('passed') is True
            and smoke.get('actual_native_ASR_backward_and_two_Adam') is True
            and smoke.get('all3_new_row_deltas_nonzero') is True
            and smoke.get('shared12_gradients_finite_and_all_deltas_nonzero') is True
            and smoke.get('protected_old_six_blank_duration_and_frozen192_encoder_LoRA_rawbytes_exact') is True
            and smoke.get('parked_original207_and_phase1_unchanged') is True
            and smoke.get('complete207_two_optimizer_phase_save_load_passed') is True
            and smoke.get('row_clock_K_plus_N') == plan['selected_acquisition_updates'] + local
            and smoke.get('shared_clock_S_plus_N') == plan['original_source_selected_step'] + local
            and smoke.get('row_lr') == plan['row_lr'] and smoke.get('shared_lr') == plan['shared_lr']
            and smoke.get('external_exact_resume_supported') is False,
            'Separate actual one-update smoke proof is incomplete')
    source_pins(boundary_context(plan['worker_arguments']))
    return {'new_adaptation_updates': local, 'inherited_steps': inherited,
        'actual300_DEV_source_equality': record(run / PROOF_NAME), 'run_config': record(run / 'run_config.json'),
        'native_completion': record(paths['native'].with_suffix('.completion.json')),
        'phase_checkpoint_schema': STATE_KIND, 'exact_phase_continuation_supported': False}


def request_stop(path, reason):
    # Preserve an existing ROOT stop; never overwrite it or mistake it for budget.
    if not path.exists():
        with path.open('x', encoding='utf-8') as stream: stream.write('Cooperative whole-update save/DEV: ' + reason + '\n')


def run_window(state, deadline, args):
    paths = {key: Path(value) for key, value in state['paths'].items()}; loop = scoped(args.output_dir)
    child = None; stop_at = stop_reason = None; started = time.monotonic(); started_utc = now(); last_phase = None
    inherited = state['source_selected_step']; last_dev = inherited
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
                state.update(phase=phase, adaptation_updates_published=len(suffix),
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
            next_action='ROOT reviews row diagnostics, protected bytes, GPU fit and DEV; no automatic exact continuation or unfreeze')
        return 1 if resource_stop else 0
    finally:
        if child is not None and child.poll() is None:
            request_stop(paths['stop'], 'Controller exit')
            try: child.wait(timeout=300)
            except subprocess.TimeoutExpired: child.terminate(); child.wait(timeout=30)


def describe_contract():
    return {'kind': CONTROLLER_KIND, 'status': 'inactive_draft' if 'PENDING' in (WORKER_SHA, TRANSITION_SHA) else 'sources_pinned_only',
        'phase': PHASE, 'source': 'future ROOT-owned CLOSED positive-K all-five qualified chosen phase1 point; no actual source admitted',
        'source_Adam': 'original207 archive preserved; original192 LoRA and3 resized moments dormant; named12 shared-head states copied atS',
        'active_Adam': 'three acquired tables atclockK/declaredrowLR plus12 shared-head tensors atclockS/declaredsharedLR, one15 global clip, no6boost',
        'native_max_joint_bytes': NATIVE_MAX_BYTES, 'allocator_cap_bytes': 8_000_000_000,
        'default_adaptation_max_updates': 1000, 'optional_separate_smoke_updates': 1,
        'initial300_unsuppressed_DEV_equality': 'actual GPU proof required before phase2 updates',
        'exact_phase_continuation_supported': False, 'unfreeze_supported': False, 'GPU_launched': False,
        'worker_sha256': WORKER_SHA, 'boundary_sha256': TRANSITION_SHA, 'sync_line': EXPECTED_SYNC_LINE,
        'activation_ready_claimed': False}


def main(args):
    if args.describe_contract:
        print(json.dumps(describe_contract(), sort_keys=True)); return 0
    require(not args.resume_exact and not args.resume_closed_loop, 'Exact adaptation continuation is deliberately unsupported')
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
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--describe-contract', action='store_true')
    result.add_argument('--worker-plan', type=Path)
    result.add_argument('--source-loop', type=Path); result.add_argument('--source-receipt', type=Path)
    result.add_argument('--run-id'); result.add_argument('--output-dir', type=Path)
    result.add_argument('--window-hours', type=float, default=3.); result.add_argument('--poll-seconds', type=float, default=5.)
    result.add_argument('--row-lr', type=float, choices=(1e-3, 3e-4), default=1e-3)
    result.add_argument('--shared-lr', type=float, choices=(1e-5, 3e-6), default=1e-5)
    result.add_argument('--adaptation-max-updates', type=int, default=1000)
    result.add_argument('--smoke-updates', type=int)
    result.add_argument('--candidate-dir', type=Path, default=ROOT / 'work/hf_turkish_bpe_candidate_512_v1')
    result.add_argument('--candidate-tr-metadata', type=Path,
        default=ROOT / 'work/data/TSC_BPE512_TRAIN_metadata_v1/candidate_TR_frame_target_metadata.jsonl')
    result.add_argument('--candidate-tr-metadata-proof', type=Path,
        default=HERE / 'TSC_BPE512_TRAIN_target_metadata_preparation_v1.json')
    result.add_argument('--resume-exact', type=Path, help='Unsupported; rejected before dispatch')
    result.add_argument('--resume-closed-loop', type=Path, help='Unsupported; rejected before dispatch')
    result.add_argument('--validate-only', action='store_true'); result.add_argument('--preflight-output', type=Path)
    result.add_argument('--gpu-window-authorized', action='store_true')
    return result


if __name__ == '__main__': sys.exit(main(parser().parse_args()))
