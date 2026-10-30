"""INACTIVE same-ID own-phase2 exact admission and runtime restoration.

Import is stdlib-only. No original207/phase1 optimizer is a fallback for the
own-phase2 latest state. The existing utility performs all207/two-Adam tensor
audits; this module binds its otherwise opaque runtime to a ROOT CLOSED receipt.
An extension beyond the initial1000 screen is an explicit stopping-bound change,
recorded separately; the original scientific identity and original300/native
startup proofs stay immutable. No actual source is admitted by this file alone.
"""
from __future__ import annotations
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys

from tsc_bpe_lora_transition_v2 import (preflight as initial_preflight, HERE, PHASE,
    STATE_KIND, PROOF_KIND, PROOF_NAME, require, sha, canonical, scoped, read, record, checked_record)

RECEIPT_KIND = 'ROOT_owned_closed_TSC_BPE_phase2_exact_v1'
INITIAL_WORKER = 'train_lora_batched_tsc_bpe_lora_native4_v2.py'
INITIAL_CONTROLLER = 'run_TSC_bpe_lora_native4_v2.py'
INITIAL_WORKER_SHA = '15825a12c35b1262b28509c921a5a763ecf17696ee02c58446d5a52a33218be3'
INITIAL_CONTROLLER_SHA = '7fa384c0919f1780a0a415089d3cfd9287e0be3d55c51894060e2747c7670a3e'
UTILITY_SHA = '2f70811d67af5fc950fa547727dc5ae18fa895a1e5e2cb572c4ff40bbc42476b'


def validate_runtime_proof(args, *, worker_path):
    """New actual process evidence, separate from the original science token."""
    proof=read(scoped(args.phase2_exact_runtime_proof)); worker_path=scoped(worker_path)
    outer=HERE/'resume_TSC_bpe_lora_same_phase_v1.py'
    require(proof.get('kind') == 'TSC_BPE_phase2_exact_fresh_preload_v1' and proof.get('status') == 'ready'
        and proof.get('phase') == PHASE and proof.get('run_id') == args.run_id
        and proof.get('pid') == os.getpid() and proof.get('ppid') == os.getppid()
        and proof.get('worker_executor') == record(worker_path)
        and proof.get('controller_executor') == record(outer) and proof.get('exact_helper_sha256') == sha(__file__)
        and proof.get('original_native_ready') == record(args.native_backend_provenance)
        and proof.get('source_receipt') == record(args.phase2_exact_source_receipt),
        'Fresh own-PID/executor/input evidence must not be substituted by an old native-ready token')
    preload=proof.get('preload',{});cap=proof.get('allocator_policy',{})
    require(preload.get('kind') == 'coherent_native_cuda12_9_preload_v1'
        and preload.get('actual_binding_linker',{}).get('version') == [12,9]
        and preload.get('cuda_initialized') is False
        and preload.get('module_sha256') == sha(HERE/'preload_nemo_linker_v1.py')
        and preload.get('successful_launcher_sha256') == sha(HERE/'launch_nemo_linker_v4.py')
        and cap.get('cap_bytes') == 8_000_000_000 and cap.get('cleanup_threshold_bytes') == 4_000_000_000
        and cap.get('allocator_cap_applied') is True
        and proof.get('native4_adapter_sha256') == '24b9cf5b8f3e11d5b043c22d0940da3735a46f8f70b6f5810b97f17d545cd90e'
        and Path(sys.executable).resolve() == HERE.parent.parent/'work/nemo-native-env-v2/Scripts/python.exe',
        'Coherent pre-Torch12.9/native4/cap8/end4 exclusive exact executor required')
    return proof


def validate_receipt(receipt, run_id):
    require(receipt.get('schema') == 1 and receipt.get('kind') == RECEIPT_KIND
        and receipt.get('status') == 'actual_CLOSED_complete_phase2_boundary'
        and receipt.get('run_id') == run_id and receipt.get('target_run_id') == run_id
        and receipt.get('allow_same_phase2_exact_resume') is True
        and receipt.get('guards_retained') is True and receipt.get('further_unfreeze_allowed') is False,
        'Explicit ROOT same-ID CLOSED own-phase2 approval required')
    values = [receipt.get(key) for key in ('original_step', 'acquisition_updates', 'phase2_updates',
        'completed_steps', 'selected_step', 'approved_total_phase2_updates')]
    require(all(type(value) is int and value > 0 for value in values), 'Positive exact S/K/N and bound required')
    s, k, n, step, best, total = values
    require(step == s + k + n and s + k <= best <= step
        and n < total <= 44958 - s - k, 'Own phase2 cumulative/selected/remaining horizon differs')
    exits = receipt.get('owned_actual_exits', [])
    require(any(e.get('role') == 'trainer_controller' for e in exits)
        and all(e.get('exit_code') == 0 and e.get('session_id') and (e.get('chunk_id') or e.get('tool_chunk'))
            and datetime.fromisoformat(e['observed_at_utc'].replace('Z', '+00:00')).tzinfo is not None for e in exits),
        'ROOT-observed actual exit0 required; resource or failed sources are not admitted')
    require(receipt.get('source_stop_category') in ('budget', 'phase_bound'),
        'Only reviewed budget/phase-bound closure may continue; never resource/manual/retention/plateau')
    return receipt


def preflight(args):
    """Future small-record admission. Checkpoint bytes/tensors are deferred."""
    require(getattr(args, 'resume_phase2_exact', None) is not None
        and getattr(args, 'phase2_exact_source_receipt', None) is not None
        and all(getattr(args, key, None) is None for key in ('resume_exact', 'resume_weights',
            'resume_runtime_migration', 'resume_optimization', 'resume_acquisition'))
        and getattr(args, 'smoke_updates', None) is None, 'Own phase2 exact mode only; no smoke/fresh fallback')
    receipt_path = scoped(args.phase2_exact_source_receipt)
    receipt = validate_receipt(read(receipt_path), args.run_id); artifacts = receipt['artifacts']
    for key in ('latest_checkpoint', 'best_checkpoint'): checked_record(artifacts[key], checkpoint=True)
    for key in ('run_config', 'training_history', 'selection', 'controller_status', 'original_native_ready',
        'initial_source_equivalence', 'runtime_native_ready', 'runtime_completion', 'fresh_runtime', 'memory', 'cache', 'io'):
        checked_record(artifacts[key])
    require(scoped(args.resume_phase2_exact) == scoped(artifacts['latest_checkpoint']['path']),
        'Resume the complete latest own-phase2 state; preserve qualifying BEST separately')
    config = read(artifacts['run_config']['path']); identity = config['config_identity']
    selection = read(artifacts['selection']['path']); history = read(artifacts['training_history']['path'])
    phase = identity['adaptation_phase']; s, k, n = (receipt[key] for key in ('original_step', 'acquisition_updates', 'phase2_updates'))
    require(config['run_id'] == selection['run_id'] == args.run_id and config['phase'] == identity['phase'] == selection['phase'] == PHASE
        and canonical(identity) == receipt['scientific_config_identity_sha256']
        and phase['original_source_selected_step'] == s and phase['selected_acquisition_updates'] == k
        and phase['source_phase1_step'] == s + k and phase['active_parameter_tensors'] == 195
        and phase['row_lr'] == 1e-4 and phase['lora_lr'] == 3e-4
        and phase['further_unfreeze_allowed'] is False and phase['old15_head_moments_active'] is False
        and config['smoke_updates'] is None and config['steps'] == 44958,
        'Same original phase2 identity/S-K/195 parameters/constant recipe required')
    require(receipt['approved_total_phase2_updates'] <= phase['effective_max_updates']
        or receipt.get('allow_stopping_bound_extension') is True,
        'Beyond the immutable initial screen requires an explicit approved stopping-bound extension')
    require(len(history) == receipt['completed_steps'] and all(row['step'] == i for i, row in enumerate(history, 1))
        and canonical(history) == receipt['training_history_sha256']
        and all(row['adaptation_updates_completed'] == row['step'] - s - k for row in history[s + k:]),
        'Full ordered inherited history and cumulative N frontier required')
    require(selection['completed_steps'] == receipt['completed_steps'] and selection['selected_step'] == receipt['selected_step']
        and selection['adaptation_updates_completed'] == n and selection['row_Adam_clock'] == k + n
        and selection['LoRA_Adam_clock'] == s + n and selection['source_phase1_step'] == s + k
        and selection['original_source_selected_step'] == s and selection['selected_acquisition_updates'] == k,
        'Selected BEST must not replace complete latest optimizer/sampler frontier')
    require(scoped(selection['latest_weights']) == scoped(receipt['original_artifact_paths']['latest_checkpoint'])
        and scoped(selection['weights']) == scoped(receipt['original_artifact_paths']['best_checkpoint']),
        'Archive aliases must retain the actual original latest/BEST pointers')
    controller = read(artifacts['controller_status']['path'])
    require(controller['status'] == 'train_window_ended' and controller['child_exit_code'] == 0
        and controller['resource_review_required'] is False and controller['stop_category'] == receipt['source_stop_category']
        and controller['run_id'] == args.run_id and controller['completed_steps'] == receipt['completed_steps']
        and controller['selected_step'] == receipt['selected_step'], 'Actual complete nonresource owned source required')
    # Source executor is admitted independently from the immutable science token.
    executor=scoped(receipt['source_controller_path'])
    require(executor.parent == HERE and executor.name in (INITIAL_CONTROLLER, 'resume_TSC_bpe_lora_same_phase_v1.py')
        and controller['controller_sha256'] == receipt['source_controller_sha256']
        and sha(executor) == receipt['source_controller_sha256']
        and (executor.name != INITIAL_CONTROLLER or receipt['source_controller_sha256'] == INITIAL_CONTROLLER_SHA),
        'Only the frozen initial or own-phase exact executor is admitted')
    native = read(artifacts['original_native_ready']['path']); completion = read(artifacts['runtime_completion']['path'])
    worker = read(artifacts['fresh_runtime']['path'])
    require(native['run_id'] == args.run_id and native['adaptation_phase'] == PHASE
        and native['trainer_sha256'] == INITIAL_WORKER_SHA and native['launcher_sha256'] == INITIAL_CONTROLLER_SHA
        and completion['status'] == 'complete' and completion['sources_unchanged'] is True
        and completion['phase'] == PHASE and completion['run_id'] == args.run_id
        and completion['pid'] == worker['pid'] and completion['actual_dispatch_source_sha256'] == receipt['source_controller_sha256'],
        'Original scientific startup plus separate actual closed executor/PID proofs required')
    require(completion['backend_ready_sha256'] == artifacts['runtime_native_ready']['sha256'],
        'Actual window completion must bind its own fresh startup, separately from original science token')
    for key, kind in (('memory', 'TSC_BPE_adaptation_memory_runtime_v1'), ('cache', 'TSC_BPE_adaptation_cache_runtime_v1'),
                      ('io', 'TSC_BPE_adaptation_atomic_IO_v1')):
        value = read(artifacts[key]['path'])
        require(value['kind'] == kind and value['status'] == 'complete' and value['sources_unchanged'] is True,
            'Source runtime completion is missing: ' + key)
    prior_n=controller.get('phase2_updates_at_window_start',0)
    require(type(prior_n) is int and 0 <= prior_n < n
        and read(artifacts['cache']['path'])['cache_reclamation']['eligible_update_calls'] == n-prior_n,
        'Source cache hook count must be the actual fresh suffix, not inherited phase2 total')
    memory = read(artifacts['memory']['path'])
    require(memory['cap_bytes'] == 8_000_000_000 and memory['cleanup_threshold_bytes'] == 4_000_000_000
        and memory['allocator_cap_applied'] is True and memory['all_phase_owned_memory_telemetry']['thread_join_completed'] is True,
        'Existing cap8/end4/owned-memory safeguards must remain')
    proof = read(artifacts['initial_source_equivalence']['path'])
    require(proof['kind'] == PROOF_KIND and proof['run_id'] == args.run_id and proof['source_selected_step'] == s + k
        and proof['passed'] is True and proof['actual_GPU_decode'] is True and proof['before_phase2_updates'] is True
        and proof['compared_fixed_DEV_utterances'] == 300 and proof['all_raw_transcripts_and_emitted_paths_exact'] is True
        and proof['all_five_retention_gates_passed'] is True
        and identity['initial_source_equivalence_sha256'] == artifacts['initial_source_equivalence']['sha256'],
        'Original actual300 proof is preserved, never regenerated or retagged')
    arguments = identity['training_arguments']
    runtime_options = {'stop_file', 'max_wall_seconds', 'log_every', 'native_backend_provenance'}
    for key, value in arguments.items():
        if key not in runtime_options:
            actual = getattr(args, key, None)
            actual = str(actual) if isinstance(actual, Path) else actual
            require(actual == value, 'Original phase2 argument changed: ' + key)
    require(scoped(args.native_backend_provenance) == scoped(artifacts['original_native_ready']['path']),
        'Original native-ready stays an immutable scientific token; fresh PID proof must be separate')
    initial = initial_preflight(args)
    require(initial['source_selected_step'] == s + k and initial['original_step'] == s and initial['acquisition_updates'] == k,
        'Original207/phase1 pair must retain the same S-K ancestry')
    return {**initial, 'original_context': initial, 'original_config': config, 'parent_identity': identity,
        'exact_receipt': receipt, 'exact_receipt_record': record(receipt_path), 'resume_step': receipt['completed_steps'],
        'phase2_updates_at_resume': n, 'source_phase1_step': s + k,
        'approved_total_phase2_updates': receipt['approved_total_phase2_updates'],
        'archived_history': history, 'archived_selection': selection}


def validate_saved_frontier(state, context):
    """Typed state/runtime aliases remain exact; only JSON comparison is canonical."""
    receipt = context['exact_receipt']; identity = context['parent_identity']; runtime = state.get('runtime_state', {})
    s, k, n = (receipt[key] for key in ('original_step', 'acquisition_updates', 'phase2_updates'))
    require(state.get('schema') == 1 and state.get('kind') == STATE_KIND
        and state.get('complete_two_optimizer_update_boundary') is True
        and state.get('run_id') == runtime.get('run_id') == receipt['run_id']
        and state.get('step') == runtime.get('step') == state.get('cumulative_step') == s + k + n
        and state.get('phase2_updates') == runtime.get('adaptation_updates_completed') == n,
        'Only the complete own-phase2 latest S-K-N frontier is admitted')
    require(state.get('config_identity') == runtime.get('config_identity')
        and canonical(state.get('config_identity')) == canonical(identity)
        and canonical(state.get('phase_identity')) == canonical(identity['adaptation_optimizers']),
        'Same typed internal identity and JSON-equivalent declared optimizer/science identity required')
    require(state.get('source_phase1_step') == runtime.get('source_phase1_step') == s + k
        and state.get('original_source_selected_step') == runtime.get('original_source_selected_step') == runtime.get('source_selected_step') == s
        and state.get('selected_acquisition_updates') == runtime.get('selected_acquisition_updates') == k
        and state.get('selected_step') == runtime.get('selected_step') == receipt['selected_step']
        and canonical(runtime.get('history')) == receipt['training_history_sha256'], 'Opaque runtime aliases/history differ')
    require(runtime['sampler']['identity'] == identity['domain_sampling_identity'] and runtime['sampler']['updates'] == s + k + n,
        'Actual candidate sampler identity/queues/clock required')
    require(runtime['best_score'] == context['archived_selection']['turkish_validation_wer']
        and runtime['retention_failures'] == runtime['history'][-1]['selection_diagnostics']['consecutive_retention_failures']
        and runtime['patience'] == runtime['history'][-1]['selection_diagnostics']['plateau_epoch_evaluations'],
        'BEST score and failure/plateau counters must carry without reset')
    for key in ('python_rng', 'torch_rng', 'cuda_rng', 'next_eval_epoch', 'train_wall_seconds', 'audio_seconds',
                'language_audio_seconds', 'evaluation_wall_seconds', 'initial_gate'):
        require(key in runtime and runtime[key] is not None, 'Missing own-phase2 runtime: ' + key)
    return runtime


def runtime_equal(left, right, *, torch_module):
    """Typed opaque-runtime equality, including exact RNG tensor bytes."""
    torch=torch_module
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right) and left.dtype == right.dtype
            and left.shape == right.shape and torch.equal(
                left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))
    if type(left) is not type(right): return False
    if isinstance(left,dict):
        return left.keys() == right.keys() and all(runtime_equal(left[key],right[key],torch_module=torch) for key in left)
    if isinstance(left,(list,tuple)):
        return len(left) == len(right) and all(runtime_equal(a,b,torch_module=torch) for a,b in zip(left,right))
    return left == right


def restore_runtime(adaptation, sampler, state, context, *, torch_module, random_module=random):
    """Future caller-injected tensor restore; no fresh optimizer or fallback."""
    require(sha(HERE / 'bpe_newrows_lora_adaptation_v1.py') == UTILITY_SHA, 'Frozen two-Adam utility changed')
    validate_saved_frontier(state, context)
    receipt = context['exact_receipt']; s, k, n = (receipt[key] for key in ('original_step', 'acquisition_updates', 'phase2_updates'))
    require(adaptation.source_step == s and adaptation.acquisition_updates == k
        and canonical(adaptation.phase_identity()) == canonical(context['parent_identity']['adaptation_optimizers']),
        'Reconstruct the original paired utility before loading its own phase2 state')
    # Its loader audits every207 tensor, all3/192 moments, protected bytes and
    # K+N/S+N clocks. A failed load aborts; never save/retry a poisoned instance.
    runtime = adaptation.load_phase_state(state)
    require(runtime_equal(runtime,state['runtime_state'],torch_module=torch_module) and adaptation.phase2_updates == n
        and adaptation.cumulative_step == receipt['completed_steps'], 'Own-phase2 loader frontier differs')
    sampler.load_state_dict(runtime['sampler'])
    require(sampler.state_dict() == runtime['sampler'] and sampler.updates == receipt['completed_steps'],
        'Restore exact pending sampler queues/cursors/coverage/RNG, without target migration')
    # Cold construction, cache preparation and sampler validation precede this.
    random_module.setstate(runtime['python_rng'])
    torch_module.set_rng_state(runtime['torch_rng'])
    torch_module.cuda.set_rng_state_all(runtime['cuda_rng'])
    adaptation.zero_grad()
    return runtime
