"""Finite B8 quality continuation with DEV-only selection and the original deadline.

B8 changes the effective optimizer batch and starts from the qualifying DEV
winner. Actual exposure and fresh-optimizer policy are explicit. Freeze the
continuation winner before TEST; preserve prior run and export evidence.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
from types import SimpleNamespace
from atomic_io_retry_v1 import read_json as read, write_json as write

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parent.parent
BASE_REVISION = '541d1f99c6b0c3cd0b11a95167540bb8edefd82b'
STRONG_REVISION = '06f233fe06e710322aca913c1bc4249a0d71fce1'
PILOT_RUN = 'lora_20261010_01'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite_wer(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def validate_io_provenance(path, worker_name, run_id):
    evidence = read(path, {})
    expected = {'worker_sha256':sha(OUTPUT / worker_name),
                'launcher_sha256':sha(OUTPUT / 'launch_with_io_retry_v1.py'),
                'helper_sha256':sha(OUTPUT / 'atomic_io_retry_v1.py')}
    require(evidence.get('status') == 'complete' and evidence.get('training_semantics_changed') is False,
            'I/O adapter did not complete or changed training semantics.')
    for key, digest in expected.items():
        require(evidence.get(key) == digest and evidence.get(key + '_after') == digest,
                'I/O adapter source changed: ' + key)
    arguments = evidence.get('arguments', [])
    require('--run-id' in arguments and arguments[arguments.index('--run-id')+1] == run_id, 'I/O adapter wrong-run identity.')
    return evidence


def validate_reused_finalization(run_id, selected_step):
    """Validate preserved B evidence by hashes; do not parse TEST predictions."""
    audit_path = OUTPUT / 'sleep_recovery_quality_verification.json'
    audit = read(audit_path, {})
    require(audit.get('passed') is True and audit.get('run_id') == run_id
            and audit.get('selected_step') == selected_step, 'Preserved B quality audit identity differs.')
    for filename, digest in audit.get('preserved_evidence_sha256', {}).items():
        path = Path(filename).resolve()
        require(path.is_relative_to(ROOT.resolve()) and path.is_file() and sha(path) == digest,
                'Preserved B result evidence changed: ' + filename)
    directory = OUTPUT / 'runs' / run_id
    final = read(directory / 'selection.json', {})
    require(final.get('accepted') is True and final.get('merged') is True and final.get('heldout_tests_evaluated') is True
            and final.get('run_id') == run_id and final.get('selected_step') == selected_step,
            'Preserved B must have complete verified finalization.')
    marker = read(directory / 'finalization_provenance.json', {})
    candidate = Path(marker.get('candidate', '')).resolve()
    checkpoint = Path(final.get('checkpoint', '')).resolve()
    require(candidate.is_relative_to((ROOT/'work/models').resolve()) and checkpoint == (OUTPUT/'checkpoint').resolve(),
            'Preserved B export paths differ.')
    for filename, digest in marker.get('candidate_files_sha256', {}).items():
        require(Path(filename).name == filename, 'Unsafe preserved export filename.')
        for folder in (candidate, checkpoint):
            require((folder/filename).is_file() and sha(folder/filename) == digest, 'Preserved B export file changed: ' + filename)
    return final, {'run_id':run_id,'selected_step':selected_step,
                   'quality_verification_path':str(audit_path),'quality_verification_sha256':sha(audit_path),
                   'final_selection_sha256':sha(directory/'selection.json'),
                   'finalization_provenance_sha256':sha(directory/'finalization_provenance.json'),
                   'candidate_files_sha256':marker['candidate_files_sha256'],
                   'policy':'Reused original verified merge/reload/DEV/TEST export; no original B artifact rewritten or inference repeated.'}


def backlog_note(text):
    subprocess.run([str(OUTPUT / 'tools/backlog.exe'), 'task', 'edit', 'ASR-15', '--append-notes', text], cwd=OUTPUT,
                   check=True, timeout=30, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    subprocess.run([sys.executable, str(OUTPUT / 'refresh_backlog.py')], cwd=ROOT,
                   check=True, timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def keep_awake(enable):
    """Windows request is thread-local; keep the system awake, leave display alone."""
    if os.name != 'nt':
        return {'supported': False, 'enabled': False}
    flags = 0x80000000 | (0x00000001 if enable else 0)
    if not ctypes.windll.kernel32.SetThreadExecutionState(flags):
        raise OSError('SetThreadExecutionState failed')
    return {'supported': True, 'enabled': enable, 'flags': flags}


def stop_owned_child(child):
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=30)


def validate_baseline():
    directory = OUTPUT / 'results/strong_whisper_large_v3_greedy_dev_verified'
    path = directory / 'summary.json'
    baseline = read(path, {})
    audit_path = OUTPUT / 'strong_baseline_dev_verification.json'
    audit = read(audit_path, {})
    require(baseline.get('status') == 'complete' and baseline.get('repository') == 'openai/whisper-large-v3'
            and baseline.get('revision') == STRONG_REVISION and baseline.get('split') == 'validation'
            and baseline.get('language') == 'tr', 'Strong baseline must be the complete pinned Turkish DEV run.')
    metrics = baseline.get('metrics', {})
    require(finite_wer(metrics.get('wer')) and metrics.get('utterances') == 50, 'Strong baseline needs finite WER for the fixed 50 DEV clips.')
    require(audit.get('passed') is True and audit.get('summary_sha256') == sha(path)
            and audit.get('predictions_sha256') == sha(directory / 'validation_tr.jsonl')
            and audit.get('metrics_recomputed') == metrics and audit.get('all_eos_emitted') is True,
            'Strong baseline verification does not match the saved run.')
    require(baseline.get('metrics_code_sha256') == sha(OUTPUT / 'metrics.py')
            and baseline.get('script_sha256') == sha(OUTPUT / 'benchmark_strong_tr.py')
            and audit.get('script_sha256') == baseline.get('script_sha256'), 'Strong baseline code provenance changed.')
    require(baseline.get('input_audit', {}).get('manifest_sha256') == sha(ROOT / 'work/data/validation_tr.jsonl')
            and baseline['input_audit'].get('paired_baseline_dev_sha256') == sha(OUTPUT / 'results/base_v3/validation_tr.jsonl'),
            'Strong baseline does not match the fixed DEV manifest/original baseline.')
    return baseline, {'summary_path': str(path), 'summary_sha256': sha(path),
                      'predictions_sha256': audit['predictions_sha256'], 'verification_sha256': sha(audit_path),
                      'model_revision': STRONG_REVISION, 'protocol': baseline['protocol'], 'metrics': metrics}


def validate_selection(run_id):
    """Metadata and CPU checkpoint identity only; never held-out predictions."""
    directory = OUTPUT / 'runs' / run_id
    path = directory / 'training_selection.json'
    if not path.is_file():
        path = directory / 'selection.json'
    selection = read(path, {})
    require(selection.get('accepted') is True and selection.get('run_id') == run_id
            and selection.get('selection_split') == 'FLEURS validation only', 'Wrong-run or non-DEV accepted selection.')
    config = read(directory / 'run_config.json', {})
    history = read(directory / 'training_history.json', [])
    step = selection.get('selected_step')
    score = selection.get('turkish_validation_wer')
    require(config.get('run_id') == run_id and config.get('base_revision') == BASE_REVISION
            and isinstance(step, int) and step >= 0 and isinstance(history, list) and finite_wer(score),
            'Invalid run/config/selected-step/score identity.')
    if step == 0:
        gate = read(directory / 'initial_development.json', {})
    else:
        matches = [row for row in history if row.get('step') == step and 'validation' in row]
        require(len(matches) == 1, 'Selected checkpoint lacks a unique completed DEV evaluation.')
        gate = matches[0]['validation']
    require(gate.get('retention_gate_passed') is True and gate.get('turkish_wer') == score, 'Selected score/retention differs from DEV evidence.')
    if 'completed_steps' in selection:
        require(selection['completed_steps'] == (history[-1]['step'] if history else 0)
                and selection['completed_steps'] <= config['steps'], 'Selection/history completion mismatch.')
    for name, digest in config.get('code_sha256', {}).items():
        require(Path(name).name == name and sha(OUTPUT / name) == digest, f'Run code changed: {name}')
    for name, digest in config.get('input_sha256', {}).items():
        input_path = Path(name)
        if not input_path.is_absolute():
            input_path = ROOT / input_path
        require(input_path.resolve().is_relative_to(ROOT.resolve()) and sha(input_path) == digest, f'Run input changed: {name}')
    weights = Path(selection.get('weights', '')).resolve()
    require(weights.is_file() and weights.is_relative_to((ROOT / 'work').resolve()), 'Selected weights must be a workspace checkpoint.')
    import torch
    saved = torch.load(weights, map_location='cpu', weights_only=True)
    require(saved.get('run_id') == run_id and saved.get('step') == step
            and saved.get('lora_metadata') == config.get('lora_layout'), 'Checkpoint run/step/LoRA layout mismatch.')
    del saved
    return selection, {'selection_sha256': sha(path), 'run_config_sha256': sha(directory / 'run_config.json'),
                       'history_sha256': sha(directory / 'training_history.json'), 'weights_sha256': sha(weights),
                       'selected_step': step, 'code_sha256': config.get('code_sha256', {}), 'input_sha256': config.get('input_sha256', {})}


def validate_transition(args):
    path = Path(args.transition_manifest).resolve()
    require(path.is_relative_to(ROOT.resolve()) and path.is_file(), 'Transition manifest must be an existing workspace file.')
    manifest = read(path, {})
    require(manifest.get('schema') == 1 and manifest.get('kind') == 'user_prioritized_B8_transition'
            and manifest.get('user_authorized_throughput_priority') is True
            and manifest.get('intentional_transition') is True and manifest.get('prior_controller_exited') is True,
            'Explicit user-prioritized, completed intentional transition is required.')
    require(manifest['original_loop_id'] == args.original_loop_id and manifest['prior_recovery_loop_id'] == args.prior_recovery_loop_id
            and manifest['target_recovery_loop_id'] == args.loop_id, 'Transition loop identities differ.')
    require(manifest['prior_controller_sha256'] == sha(OUTPUT/'run_experiment_loop_recovery.py')
            == '5712954946ee3f68ebfc7e25075ecec2e718432f2393bf327c51d6c4905e406c', 'Frozen v1 controller bytes differ.')
    snapshot = Path(manifest['prior_status_snapshot_path']).resolve()
    prior_dir = OUTPUT/'runs'/args.prior_recovery_loop_id
    require(snapshot.is_relative_to(ROOT.resolve()) and sha(snapshot) == manifest['prior_status_snapshot_sha256']
            == sha(prior_dir/'loop_status.json'), 'Prior intentional terminal status changed.')
    prior = read(snapshot, {})
    require(prior.get('status') == 'failed' and prior.get('failure_type') == 'RuntimeError'
            and 'stop marker already exists' in prior.get('failure_message',''), 'Prior controller must have exited through the owned interlock.')
    require(prior['deadline_at_utc'] == manifest['original_deadline_at_utc'], 'Transition renewed the original deadline.')
    require(manifest.get('target_configuration') == {'batch_size':args.quality_batch_size,'accumulation':args.quality_accumulation,
                'effective_batch_size':args.quality_batch_size*args.quality_accumulation,'warmup_steps':args.warmup_steps}, 'Authorized B8 configuration differs.')
    completed = manifest['completed_C']
    expected_run = args.prior_recovery_loop_id + '_screen_C'
    require(completed['run_id'] == expected_run, 'Transition C run differs.')
    for stem, expected in [('run_config',OUTPUT/'runs'/expected_run/'run_config.json'),
                           ('training_history',OUTPUT/'runs'/expected_run/'training_history.json'),
                           ('selection',OUTPUT/'runs'/expected_run/'selection.json'),
                           ('best_checkpoint',ROOT/'work'/(expected_run+'_best.pt')),
                           ('latest_checkpoint',ROOT/'work'/(expected_run+'_latest.pt'))]:
        artifact = Path(completed[stem+'_path']).resolve()
        require(artifact == expected.resolve() and sha(artifact) == completed[stem+'_sha256'], 'Completed C artifact changed: ' + stem)
    selected = read(completed['selection_path'], {})
    require(selected.get('accepted') == completed['accepted'] and selected.get('selected_step') == completed['selected_step']
            and selected.get('completed_steps') == completed['completed_steps']
            and selected.get('stop_early_reason') == 'controller_stop_file', 'Completed C was not cooperatively saved at the recorded state.')
    expected_markers = {(ROOT/'work'/(args.prior_recovery_loop_id+'_'+name+'.stop')).resolve()
                        for name in ('screen_C','screen_B2','extend_1')}
    actual_markers = set()
    for marker in manifest['owned_guard_markers']:
        target = Path(marker['path']).resolve()
        require(target in expected_markers and sha(target) == marker['sha256'], 'Owned transition guard changed.')
        actual_markers.add(target)
    require(actual_markers == expected_markers, 'All three owned transition guards must remain preserved.')
    return manifest, prior, path


def main(args):
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.loop_id) is not None, 'Use a filename-safe loop ID of at most 64 characters.')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.original_loop_id) is not None, 'Use a safe original loop ID.')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.prior_recovery_loop_id) is not None, 'Use a safe prior recovery loop ID.')
    if args.validate_only:
        return preflight(args)
    screen_epochs = .25
    require(args.quality_batch_size == 8 and args.quality_accumulation == 2 and args.warmup_steps == 25, 'This authorized transition uses B8/accum2/effective16 and warmup25.')
    require(0 < args.epochs <= 3 and args.eval_per_epoch == 4, 'B8 continuation is capped at3 epochs with the recorded DEV cadence.')
    transition, prior_recovery, transition_path = validate_transition(args)
    origin_dir = OUTPUT / 'runs' / args.original_loop_id
    original_status_path = origin_dir / 'loop_status.json'
    original = read(original_status_path, {})
    require(original.get('loop_id') == args.original_loop_id and original.get('status') == 'strong_test_failed',
            'Recovery requires the completed, preserved failed original loop.')
    require(original.get('controller_sha256') == sha(OUTPUT / 'run_experiment_loop.py'), 'Original controller source changed.')
    require(transition['original_deadline_at_utc'] == original['deadline_at_utc'], 'Transition deadline differs from original.')
    original_deadline = datetime.fromisoformat(original['deadline_at_utc'])
    remaining_seconds = (original_deadline - datetime.now(timezone.utc)).total_seconds()
    require(remaining_seconds > args.finalization_reserve_seconds + args.stop_grace_seconds + 900,
            'Original deadline leaves insufficient bounded recovery training time.')
    reserve_seconds = args.finalization_reserve_seconds
    finalization_cap = args.finalization_cap_seconds
    stop_grace = args.stop_grace_seconds
    require(0 < reserve_seconds <= 3600 and 0 < finalization_cap <= 3600 and 0 < stop_grace <= 600,
            'Recovery reserve/cap/grace must be positive and bounded.')
    for worker_name in (args.strong_test_worker, args.audit_worker):
        require(Path(worker_name).name == worker_name and (OUTPUT / worker_name).is_file(), 'Recovery worker is unavailable or unsafe.')
    loop_dir = OUTPUT / 'runs' / args.loop_id
    loop_dir.mkdir(parents=True, exist_ok=True)
    status_path = loop_dir / 'loop_status.json'
    require(not status_path.exists(), 'Recovery ID already exists; preserve its evidence.')
    shutil.copy2(original_status_path, loop_dir / 'original_loop_status.json')
    shutil.copy2(origin_dir / 'frozen_final_selection.json', loop_dir / 'original_frozen_final_selection.json')
    shutil.copy2(transition_path, loop_dir / 'b8_transition.json')
    shutil.copy2(transition['prior_status_snapshot_path'], loop_dir / 'prior_recovery_loop_status.json')
    prior_artifacts = loop_dir / 'previous_output_artifacts'
    prior_artifacts.mkdir()
    for filename in ('adaptive_experiment_report.json','selection.json','benchmark_comparison.json',
                     'confidence_calibration.json','adapted_results_path.json','adaptive_experiment_recovery_report.json'):
        if (OUTPUT / filename).is_file():
            shutil.copy2(OUTPUT / filename, prior_artifacts / filename)
    start = time.monotonic()
    deadline = start + (original_deadline - datetime.now(timezone.utc)).total_seconds()
    reserve = reserve_seconds
    training_deadline = deadline - reserve - stop_grace
    state = {'loop_id': args.loop_id, 'status': 'validating', 'started_at_utc': now(),
             'deadline_at_utc': original['deadline_at_utc'], 'initial_window_hours': original['initial_window_hours'],
             'selection_split': 'FLEURS validation only', 'finalization_reserve_seconds': reserve,
             'stop_grace_seconds': stop_grace, 'finalization_cap_seconds': finalization_cap,
             'screen_epochs_requested': screen_epochs, 'trials': [], 'warnings': [],
             'controller_sha256': sha(OUTPUT / 'run_experiment_loop.py'),
             'recovery': {'version':2, 'controller_path': str(Path(__file__).resolve()), 'controller_sha256': sha(__file__),
                          'original_loop_id': args.original_loop_id,
                          'prior_recovery_loop_id': args.prior_recovery_loop_id,
                          'transition_manifest_snapshot_path': str(loop_dir/'b8_transition.json'),
                          'transition_manifest_snapshot_sha256': sha(loop_dir/'b8_transition.json'),
                          'transition_original_path': str(transition_path), 'transition_original_sha256': sha(transition_path),
                          'prior_recovery_status_snapshot_path': str(loop_dir/'prior_recovery_loop_status.json'),
                          'prior_recovery_status_snapshot_sha256': sha(loop_dir/'prior_recovery_loop_status.json'),
                          'original_status_snapshot': str(loop_dir / 'original_loop_status.json'),
                          'original_status_snapshot_sha256': sha(loop_dir / 'original_loop_status.json'),
                          'original_frozen_selection_snapshot_sha256': sha(loop_dir / 'original_frozen_final_selection.json'),
                          'io_retry_launcher_sha256': sha(OUTPUT / 'launch_with_io_retry_v1.py'),
                          'io_retry_helper_sha256': sha(OUTPUT / 'atomic_io_retry_v1.py'),
                          'budget_policy': 'Original absolute 12-hour deadline retained; sleep/pause/failed runs count.',
                          'lost_state_policy': 'Failed C and extension latest state is initial step0 with empty Adam; all original files preserved.',
                          'selection_policy': 'Carry original B and saved partial C; user-prioritized B8/effective16 quality continuation from the best qualifying DEV weights; DEV-only selection; prior TEST never guides choices.',
                          'backend_flags_documented': {'trainer_fresh_defaults':{'cudnn_allow_tf32':True,'cuda_matmul_allow_tf32':False},
                                                       'B8_profile':{'cudnn_allow_tf32':False,'cuda_matmul_allow_tf32':False},
                                                       'difference':'Frozen trainer uses fresh runtime default cudnn TF32 flag; numerical equivalence and timing-transfer are not claimed.'},
                          'strong_test_worker': args.strong_test_worker, 'audit_worker': args.audit_worker,
                          'previous_output_artifacts': {str(p.relative_to(prior_artifacts)):sha(p) for p in prior_artifacts.iterdir()}},
             'heldout_policy': 'Prior B TEST was already evaluated before infrastructure failure. Recovery choices use DEV only; final TEST is descriptive after frozen recovery selection.'}
    awake_requested = False

    def note(text):
        try:
            backlog_note(text)
        except Exception as exception:
            state['warnings'].append({'at_utc': now(), 'kind': 'backlog_update_failed', 'message': str(exception)})
            print(f'Backlog update warning: {exception}', flush=True)

    try:
        profile_path = OUTPUT / 'batching_speed_comparison.json'
        profile = read(profile_path, {})
        require(profile.get('passed') is True and profile.get('throughput_improved') is True, 'The loop needs verified measured throughput improvement.')
        selected = profile.get('selected_configuration', {})
        require(all(isinstance(selected.get(k), int) and selected[k] > 0 for k in ('batch_size', 'accumulation', 'chunk_size'))
                and isinstance(selected.get('encoder_checkpointing'), bool) and selected.get('tdt_implementation') in ('original', 'v2'),
                'Invalid measured batch configuration.')
        baseline, baseline_provenance = validate_baseline()
        baseline_wer = baseline['metrics']['wer']
        threshold = baseline_wer + .02
        manifest = ROOT / 'work/data/tr_expanded_train.jsonl'
        data_audit = read(OUTPUT / 'expanded_turkish_verification.json', {})
        require(data_audit.get('passed') is True and data_audit.get('manifest_sha256') == sha(manifest), 'Expanded manifest must match its completed CPU verification.')
        incoming, incoming_provenance = validate_selection(PILOT_RUN)
        carried_id = args.original_loop_id + '_screen_B'
        carried, carried_provenance = validate_selection(carried_id)
        carried_trials = [entry for entry in original['trials'] if entry['run_id'] == carried_id and entry.get('status') == 'complete']
        require(len(carried_trials) == 1, 'Original B must have one completed screening record.')
        carried_trial = dict(carried_trials[0])
        carried_trial['carried_from_original_loop'] = True
        carried_trial['selected_configuration'] = dict(selected)
        carried_trial['batch_quality_experiment'] = False
        carried_trial['batch_numerical_equivalence_proven'] = False
        carried_trial['existing_training_configuration_verified'] = True
        carried_trial['effective_batch_size'] = selected['batch_size'] * selected['accumulation']
        carried_trial['selection_provenance'] = carried_provenance
        state['trials'].append(carried_trial)
        best = {'run_id': carried_id, 'wer': carried['turkish_validation_wer'], 'weights': carried['weights'],
                'selected_step': carried['selected_step'], 'trial': 'screen_B',
                'parameters': dict(carried_trial['parameters']), 'provenance': carried_provenance,
                'selected_configuration': dict(selected), 'batch_quality_experiment': False}
        c_id = transition['completed_C']['run_id']
        c_entries = [entry for entry in prior_recovery['trials'] if entry['run_id'] == c_id]
        require(len(c_entries) == 1 and c_entries[0].get('status') == 'complete' and c_entries[0].get('exit_code') == 0,
                'Exactly one completed cooperative C trial is required.')
        c_entry = dict(c_entries[0])
        c_entry['carried_from_prior_recovery'] = True
        c_entry['prior_partial_exposure'] = True
        if c_entry.get('selection', {}).get('accepted'):
            accepted_c, c_provenance = validate_selection(c_id)
            c_entry['selection_provenance'] = c_provenance
            # Actual saved C is worse than B. Keep its evidence immutable; the
            # preserved original B is the only existing-export fallback.
            require(accepted_c['turkish_validation_wer'] >= best['wer'],
                    'This frozen transition requires the observed C score not to outrank preserved B; a new publication plan would be needed.')
        state['trials'].append(c_entry)
        state.update(status='running', strong_baseline_wer=baseline_wer, competitive_threshold_wer=threshold,
                     competitive_definition='Turkish DEV WER <= measured Whisper DEV WER + 2 percentage points and every old-language DEV gate',
                     selected_configuration=selected, profile_sha256=sha(profile_path), strong_baseline_provenance=baseline_provenance,
                     expanded_manifest_sha256=sha(manifest), expanded_data_verification_sha256=sha(OUTPUT / 'expanded_turkish_verification.json'),
                     lexical_verified_turkish_clips=data_audit.get('lexical_filter', {}).get('accepted_clips'), best_development_candidate=dict(best))
        state['awake_request'] = keep_awake(True)
        awake_requested = bool(state['awake_request'].get('enabled'))
        write(status_path, state)
        note(f'{now()}: Started recovery {args.loop_id}, retaining original deadline {original["deadline_at_utc"]}; measured DEV reference {100*baseline_wer:.4f}% WER, target {100*threshold:.4f}%. '
             f'User-prioritized B8/accum2 effective16 quality continuation starts from the DEV winner with a fresh optimizer; '
             f'up to {args.epochs:g} Turkish epochs, actual exposure and quota stops recorded. Initial DEV cadence .25/.5/1 epoch, then every quarter epoch. DEV-only selection.')

        def trial(name, parameters, weights, epochs, max_seconds, configuration=None, experimental=False):
            configuration = dict(configuration or selected)
            run_id = f'{args.loop_id}_{name}'
            log = ROOT / f'work/{run_id}.log'
            stop_file = ROOT / f'work/{run_id}.stop'
            require(not (OUTPUT / 'runs' / run_id).exists() and not stop_file.exists(), 'Trial ID/stop marker already exists.')
            stage_start = time.monotonic()
            stage_deadline = min(training_deadline, stage_start + max_seconds)
            quota = max(1., stage_deadline - stage_start)
            io_provenance = loop_dir / (run_id + '_io_retry_provenance.json')
            command = [sys.executable, '-X', 'utf8', '-u', str(OUTPUT / 'launch_with_io_retry_v1.py'),
                       '--worker', 'train_lora_batched.py', '--provenance', str(io_provenance),
                       '--expected-worker-sha', sha(OUTPUT / 'train_lora_batched.py'), '--', '--run-id', run_id,
                       '--tr-manifest', str(manifest), '--resume-weights', str(weights), '--epochs', str(epochs),
                       '--batch-size', str(configuration['batch_size']), '--accumulation', str(configuration['accumulation']),
                       '--tdt-implementation', configuration['tdt_implementation'], '--chunk-size', str(configuration['chunk_size']),
                       '--head-lr', str(parameters['head_lr']), '--lora-lr', str(parameters['lora_lr']),
                       '--new-token-lr-multiplier', str(parameters['new_token_lr_multiplier']), '--replay-kl', '5',
                       '--duration-kl', '.5', '--feature-distillation', '1', '--replay-fraction', '.5', '--warmup-steps', str(args.warmup_steps),
                       '--schedule', 'constant', '--eval-per-epoch', str(args.eval_per_epoch), '--eval-epoch-fraction', str(min(.25, epochs)),
                       '--early-stop-patience', '3', '--max-retention-failures', '2', '--min-improvement', '.002',
                       '--target-turkish-wer', str(threshold), '--max-wall-seconds', str(quota), '--stop-file', str(stop_file)]
            if not configuration['encoder_checkpointing']:
                command += ['--no-encoder-checkpoint']
            entry = {'run_id': run_id, 'name': name, 'parameters': dict(parameters), 'epochs_planned': epochs,
                     'initialization_weights': str(weights), 'initialization_sha256': sha(weights), 'optimizer_resume': False,
                     'started_at_utc': now(), 'stage_wall_quota_seconds': quota, 'command': command, 'log': str(log), 'status': 'running',
                     'io_retry_provenance_path': str(io_provenance), 'io_retry_launcher_sha256': sha(OUTPUT / 'launch_with_io_retry_v1.py'),
                     'io_retry_helper_sha256': sha(OUTPUT / 'atomic_io_retry_v1.py'),
                     'selected_configuration': configuration, 'batch_quality_experiment': experimental,
                     'batch_numerical_equivalence_proven': False,
                     'existing_training_configuration_verified': configuration == selected,
                     'batch_quality_scope': ('User-prioritized B8 effective16 quality experiment; optimizer batch changed; strict BF16 numerical gate remains failed; no numerical equivalence or timing-transfer claim.' if experimental
                                             else 'Extension inherits DEV-qualified B2 configuration; strict BF16 numerical gate remains failed.' if configuration['batch_size'] == 2
                                             else 'Existing verified B1 training configuration.'),
                     'effective_batch_size': configuration['batch_size'] * configuration['accumulation'],
                     'warmup_steps':args.warmup_steps,'eval_per_epoch':args.eval_per_epoch,
                     'warmup_sample_exposure':{'current':args.warmup_steps*configuration['batch_size']*configuration['accumulation'],'prior_B':100*4},
                     'backend_flags_documented':state['recovery']['backend_flags_documented'],
                     'quality_scope':'Different effective optimizer batch and DEV-selected warm start; not a matched numerical or identical-initialization comparison.'}
            state['trials'].append(entry)
            state['active_run_id'] = run_id
            write(status_path, state)
            child = None
            last_dev_step = None
            requested_stop = False
            environment = dict(os.environ, PYTHONUTF8='1', HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
            try:
                with log.open('wb') as stream:
                    child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    entry['pid'] = child.pid
                    while child.poll() is None:
                        history = read(OUTPUT / 'runs' / run_id / 'training_history.json', [])
                        if history:
                            entry['completed_updates'] = history[-1]['step']
                            entry['latest_training_epoch'] = history[-1].get('turkish_epoch')
                            dev = [row for row in history if 'validation' in row]
                            if dev and dev[-1]['step'] != last_dev_step:
                                record = dev[-1]
                                last_dev_step = record['step']
                                entry['latest_validation'] = record['validation']
                                entry['latest_validation_epoch'] = record.get('turkish_epoch')
                                print(json.dumps({'run_id': run_id, 'step': record['step'], 'epoch': record.get('turkish_epoch'), 'validation': record['validation']}), flush=True)
                                note(f'{now()}: {run_id} DEV step {record["step"]}, Turkish epoch {record.get("turkish_epoch")}: WER {100*record["validation"]["turkish_wer"]:.4f}%, '
                                     f'retention {record["validation"]["retention_gate_passed"]}.')
                        current = time.monotonic()
                        state['updated_at_utc'] = now()
                        state['remaining_wall_minutes'] = max(0, deadline - current) / 60
                        if current >= stage_deadline and not requested_stop:
                            stop_file.write_text('Stage quota expired; save DEV-selected evidence and exit.\n', encoding='utf-8')
                            requested_stop = True
                            entry['deadline_stop_requested_at_utc'] = now()
                        write(status_path, state)
                        if requested_stop and current >= min(stage_deadline + stop_grace, deadline - reserve):
                            entry['forced_stop'] = 'Trainer exceeded bounded cooperative-stop grace.'
                            stop_owned_child(child)
                            break
                        time.sleep(15)
                    entry['exit_code'] = child.returncode
            finally:
                stop_owned_child(child)
                entry['finished_at_utc'] = now()
                entry['wall_seconds'] = time.monotonic() - stage_start
                state.pop('active_run_id', None)
            selection = read(OUTPUT / 'runs' / run_id / 'selection.json', {})
            entry['selection'] = selection
            if io_provenance.is_file():
                entry['io_retry_provenance_sha256'] = sha(io_provenance)
                entry['io_retry_provenance'] = read(io_provenance, {})
            entry['epochs_completed'] = selection.get('turkish_epochs_completed')
            entry['equal_requested_exposure_completed'] = isinstance(entry['epochs_completed'], (int, float)) and entry['epochs_completed'] + 1e-9 >= epochs
            entry['status'] = 'complete' if entry['exit_code'] == 0 else 'failed'
            if entry['exit_code'] == 0 and selection.get('accepted'):
                try:
                    validate_io_provenance(io_provenance, 'train_lora_batched.py', run_id)
                    accepted, provenance = validate_selection(run_id)
                    entry['selection_provenance'] = provenance
                    score = accepted['turkish_validation_wer']
                    if score < best['wer']:
                        best.update(run_id=run_id, wer=score, weights=accepted['weights'], selected_step=accepted['selected_step'],
                                    trial=name, parameters=dict(parameters), provenance=provenance,
                                    selected_configuration=configuration, batch_quality_experiment=experimental,
                                    experimental_batch_configuration=configuration['batch_size'] != 1)
                except Exception as exception:
                    entry['selection_rejected'] = str(exception)
            state['best_development_candidate'] = dict(best)
            write(status_path, state)
            note(f'{now()}: {run_id} exit {entry["exit_code"]}; actual Turkish epochs {entry["epochs_completed"]}, planned {epochs:g}. '
                 f'Best qualifying DEV WER {100*best["wer"]:.4f}% ({best["run_id"]}); stop reason {selection.get("stop_early_reason", selection.get("reason"))}.')

        remaining = training_deadline - time.monotonic()
        quality_configuration = {**selected, 'batch_size':args.quality_batch_size, 'accumulation':args.quality_accumulation}
        state['priority_quality_configuration'] = quality_configuration
        state['effective_batch_change'] = {'prior_B':4,'quality_trial':16,'new_optimizer':True,
                                          'learning_rates_scaled':False,'warmup_clip_exposure_prior':400,'warmup_clip_exposure_quality':400}
        state['development_cadence'] = {'initial_epoch_thresholds':[.25,.5,1.0], 'after_epoch1_interval':.25,
                                       'bounded_final_stop_evaluation':True}
        if remaining >= 900 and best['wer'] > threshold:
            trial('b8_quality',best['parameters'],best['weights'],args.epochs,remaining,
                  configuration=quality_configuration,experimental=True)
        else:
            state['B8_quality_skipped'] = 'Insufficient original budget, or qualifying DEV target already met.'
        state.update(status='trials_finished', finished_trials_at_utc=now(), competitive_on_development=best['wer'] <= threshold,
                     best_development_candidate=dict(best), training_window_wall_hours=(time.monotonic() - start) / 3600)
        state['competitive_on_development_before_merge'] = state['competitive_on_development']
        write(status_path, state)
        write(loop_dir / 'frozen_final_selection.json', {**best, 'frozen_at_utc': now(), 'selection_split': 'FLEURS validation only',
                                                       'competitive_threshold_wer': threshold, 'baseline_provenance': baseline_provenance})
        if best['run_id'] != PILOT_RUN:
            post_deadline = min(deadline, time.monotonic() + finalization_cap)
            state['post_selection_cap_seconds'] = finalization_cap
            timeout = post_deadline - time.monotonic()
            if timeout <= 0:
                state['status'] = 'finalization_skipped_budget_exhausted'
            else:
                state.update(status='finalizing', finalization_timeout_seconds=timeout)
                write(status_path, state)
                final_start = time.monotonic()
                try:
                    if best['run_id'] == carried_id:
                        reused, reuse_provenance = validate_reused_finalization(carried_id, best['selected_step'])
                        state['reused_original_finalization'] = reuse_provenance
                        write(loop_dir/'reused_original_finalization.json', reuse_provenance)
                        result = SimpleNamespace(returncode=0)
                    else:
                        with (loop_dir / 'finalization.log').open('wb') as stream:
                            result = subprocess.run([sys.executable, '-X', 'utf8', '-u', str(OUTPUT / 'launch_with_io_retry_v1.py'),
                                                     '--worker', 'finalize_lora.py', '--provenance', str(loop_dir / 'finalizer_io_retry_provenance.json'),
                                                     '--expected-worker-sha', sha(OUTPUT / 'finalize_lora.py'),
                                                     '--', '--run-id', best['run_id']],
                                                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout,
                                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    state['finalization_exit_code'] = result.returncode
                    final = read(OUTPUT / 'runs' / best['run_id'] / 'selection.json', {})
                    state['final_selection'] = final
                    merged_gate = final.get('merged_development_gate', {})
                    merged_wer = merged_gate.get('turkish_wer')
                    state['merged_turkish_development_wer'] = merged_wer
                    state['competitive_on_development'] = merged_gate.get('passed') is True and finite_wer(merged_wer) and merged_wer <= threshold
                    success = result.returncode == 0 and final.get('run_id') == best['run_id'] and final.get('selected_step') == best['selected_step']
                    success = success and final.get('accepted') is True and final.get('merged') is True and final.get('heldout_tests_evaluated') is True
                    if success and best['run_id'] != carried_id:
                        validate_io_provenance(loop_dir / 'finalizer_io_retry_provenance.json', 'finalize_lora.py', best['run_id'])
                    state['status'] = 'finalization_complete' if success else 'finalization_failed'
                except subprocess.TimeoutExpired:
                    state['status'] = 'finalization_timed_out'
                finally:
                    state['finalization_wall_seconds'] = time.monotonic() - final_start
                    final_io_path = loop_dir / 'finalizer_io_retry_provenance.json'
                    if final_io_path.is_file():
                        state['finalizer_io_retry_provenance_path'] = str(final_io_path)
                        state['finalizer_io_retry_provenance_sha256'] = sha(final_io_path)
                if state['status'] == 'finalization_complete':
                    strong_script = OUTPUT / args.strong_test_worker
                    require(strong_script.is_file(), 'Final paired strong TEST benchmark script is unavailable.')
                    timeout = post_deadline - time.monotonic()
                    if timeout <= 0:
                        state.update(status='strong_test_budget_exhausted', strong_test_status='not_run_budget_exhausted')
                    else:
                        state.update(status='benchmarking_strong_test', strong_test_status='running',
                                     strong_test_script_sha256=sha(strong_script), strong_test_timeout_seconds=timeout)
                        write(status_path, state)
                        comparison_path = OUTPUT / 'adaptive_strong_baseline_test_comparison_v2.json'
                        strong_start = time.monotonic()
                        try:
                            with (loop_dir / 'strong_test.log').open('wb') as stream:
                                result = subprocess.run([sys.executable, '-X', 'utf8', '-u', str(strong_script),
                                                         '--frozen-selection', str(loop_dir / 'frozen_final_selection.json'),
                                                         '--selected-run-id', best['run_id']], cwd=ROOT, stdout=stream,
                                                        stderr=subprocess.STDOUT, timeout=timeout,
                                                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                            state['strong_test_exit_code'] = result.returncode
                            comparison = read(comparison_path, {})
                            valid = result.returncode == 0 and comparison.get('status') == 'complete' and comparison.get('passed') is True
                            valid = valid and comparison.get('run_id') == best['run_id'] and comparison.get('frozen_selection_sha256') == sha(loop_dir / 'frozen_final_selection.json')
                            state['strong_test_status'] = 'complete' if valid else 'failed'
                            state['strong_test_comparison'] = comparison
                            if comparison_path.is_file():
                                state['strong_test_comparison_sha256'] = sha(comparison_path)
                            state['status'] = 'complete' if valid else 'strong_test_failed'
                        except subprocess.TimeoutExpired:
                            state.update(status='strong_test_timed_out', strong_test_status='timed_out')
                        finally:
                            state['strong_test_wall_seconds'] = time.monotonic() - strong_start
                if state['status'] == 'complete':
                    verifier = OUTPUT / args.audit_worker
                    require(verifier.is_file(), 'Independent adaptive final auditor is unavailable.')
                    timeout = post_deadline - time.monotonic()
                    if timeout <= 0:
                        state['status'] = 'final_audit_budget_exhausted'
                    else:
                        state.update(status='verifying_final_results', final_audit_script_sha256=sha(verifier), final_audit_timeout_seconds=timeout)
                        write(status_path, state)
                        audit_start = time.monotonic()
                        try:
                            with (loop_dir / 'final_audit.log').open('wb') as stream:
                                result = subprocess.run([sys.executable, '-X', 'utf8', '-u', str(verifier),
                                                         '--frozen-selection', str(loop_dir / 'frozen_final_selection.json'),
                                                         '--strong-test-comparison', str(OUTPUT / 'adaptive_strong_baseline_test_comparison_v2.json'),
                                                         '--output-dir', str(loop_dir)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                                        timeout=timeout, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                            state['final_audit_exit_code'] = result.returncode
                            audit_path = loop_dir / 'adaptive_results_verification.json'
                            audit = read(audit_path, {})
                            valid = result.returncode == 0 and audit.get('passed') is True and audit.get('run_id') == best['run_id']
                            valid = valid and audit.get('loop_id') == args.loop_id and audit.get('selected_step') == best['selected_step']
                            valid = valid and audit.get('frozen_selection_sha256') == sha(loop_dir / 'frozen_final_selection.json')
                            state['status'] = 'complete' if valid else 'final_audit_failed'
                            state['final_audit'] = audit
                            if audit_path.is_file():
                                state['final_audit_sha256'] = sha(audit_path)
                        except subprocess.TimeoutExpired:
                            state['status'] = 'final_audit_timed_out'
                        finally:
                            state['final_audit_wall_seconds'] = time.monotonic() - audit_start
        else:
            state['status'] = 'complete_without_better_candidate'
        state['finished_at_utc'] = now()
        state['total_wall_hours'] = (time.monotonic() - start) / 3600
        write(status_path, state)
        write(OUTPUT / 'adaptive_experiment_recovery_v2_report.json', state)
        note(f'{now()}: Bounded loop finished {state["status"]}; best qualifying Turkish DEV WER {100*best["wer"]:.4f}% vs measured reference {100*baseline_wer:.4f}%; '
             f'provisional competitive DEV target met {state["competitive_on_development"]}. Pipeline completion and useful Turkish accuracy are separate.')
    except BaseException as exception:
        state.update(status='failed', failed_at_utc=now(), failure_type=type(exception).__name__, failure_message=str(exception),
                     total_wall_hours=(time.monotonic() - start) / 3600)
        write(status_path, state)
        write(OUTPUT / 'adaptive_experiment_recovery_v2_report.json', state)
        raise
    finally:
        if awake_requested:
            try:
                state['awake_restoration'] = keep_awake(False)
            except Exception as exception:
                state['warnings'].append({'kind': 'awake_restoration_failed', 'message': str(exception)})
        state['total_wall_hours'] = (time.monotonic() - start) / 3600
        state['finished_at_utc'] = now()
        write(status_path, state)
        write(OUTPUT / 'adaptive_experiment_recovery_v2_report.json', state)
    print(json.dumps(state, ensure_ascii=False), flush=True)


def preflight(args):
    """Read-only CPU launch guards; no inference, children, sleep, or awake request."""
    require(args.quality_batch_size == 8 and args.quality_accumulation == 2 and args.warmup_steps == 25, 'Authorized B8/effective16 warmup25 configuration required.')
    transition, prior_recovery, transition_path = validate_transition(args)
    origin_path = OUTPUT / 'runs' / args.original_loop_id / 'loop_status.json'
    origin = read(origin_path, {})
    require(origin.get('loop_id') == args.original_loop_id and origin.get('status') == 'strong_test_failed', 'Original terminal state differs.')
    require(origin.get('controller_sha256') == sha(OUTPUT/'run_experiment_loop.py'), 'Original controller source changed.')
    deadline = datetime.fromisoformat(origin['deadline_at_utc'])
    remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
    require(remaining > args.finalization_reserve_seconds + args.stop_grace_seconds + 900, 'Original deadline leaves insufficient time.')
    require(not (OUTPUT / 'runs' / args.loop_id / 'loop_status.json').exists(), 'Recovery loop ID already exists.')
    for name in ('launch_with_io_retry_v1.py', 'atomic_io_retry_v1.py', args.strong_test_worker, args.audit_worker):
        require(Path(name).name == name and (OUTPUT / name).is_file(), 'Missing or unsafe recovery worker: ' + name)
    baseline, baseline_provenance = validate_baseline()
    carried, carried_provenance = validate_selection(args.original_loop_id + '_screen_B')
    accepted_c, c_provenance = validate_selection(transition['completed_C']['run_id'])
    require(accepted_c['turkish_validation_wer'] >= carried['turkish_validation_wer'],
            'Observed preserved C must not outrank B for this frozen publication fallback policy.')
    pilot, pilot_provenance = validate_selection(PILOT_RUN)
    profile = read(OUTPUT / 'batching_speed_comparison.json', {})
    require(profile.get('passed') is True and profile.get('throughput_improved') is True, 'Verified profile is required.')
    require(profile.get('selected_configuration', {}).get('batch_size') == 1
            and profile['selected_configuration'].get('accumulation') == 4, 'Preserve verified B1 effective4 default.')
    manifest = ROOT / 'work/data/tr_expanded_train.jsonl'
    require(read(OUTPUT / 'expanded_turkish_verification.json', {}).get('manifest_sha256') == sha(manifest), 'Expanded data identity differs.')
    report = {'passed': True, 'scope': 'CPU/read-only; no training or TEST inference', 'loop_id': args.loop_id,
              'original_loop_id': args.original_loop_id, 'deadline_at_utc': origin['deadline_at_utc'],
              'remaining_wall_seconds_at_preflight': remaining, 'controller_sha256': sha(__file__),
              'original_status_sha256': sha(origin_path), 'original_controller_sha256': sha(OUTPUT/'run_experiment_loop.py'),
              'recovery_files_sha256': {name:sha(OUTPUT/name) for name in ('launch_with_io_retry_v1.py','atomic_io_retry_v1.py',args.strong_test_worker,args.audit_worker)},
              'carried_run_id': carried['run_id'], 'carried_selected_step': carried['selected_step'],
              'carried_provenance': carried_provenance, 'pilot_provenance': pilot_provenance,
              'baseline_provenance': baseline_provenance, 'selected_configuration': profile['selected_configuration'],
              'experimental_B8': {'batch_size':8,'accumulation':2,'effective_batch_size':16,'epochs_cap':args.epochs,'warmup_steps':25,
                                 'strict_BF16_numeric_gate_passed':False,'classification':'User-prioritized new DEV quality experiment; numerical equivalence/timing transfer not claimed.'},
              'transition_manifest_path':str(transition_path),'transition_manifest_sha256':sha(transition_path),
              'prior_recovery_status_sha256':transition['prior_status_snapshot_sha256'],
              'carried_C_provenance':c_provenance,'carried_C_wer':accepted_c['turkish_validation_wer'],
              'preserved_partial_C':transition['completed_C']}
    write(OUTPUT / 'recovery_controller_v2_preflight.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--loop-id', default='adaptive_20261010_recovery_b8_01')
    result.add_argument('--original-loop-id', default='adaptive_20261010_01')
    result.add_argument('--prior-recovery-loop-id',default='adaptive_20261010_recovery_01')
    result.add_argument('--transition-manifest',type=Path,default=OUTPUT/'runs/adaptive_20261010_recovery_01/b8_transition.json')
    result.add_argument('--quality-batch-size',type=int,default=8)
    result.add_argument('--quality-accumulation',type=int,default=2)
    result.add_argument('--warmup-steps',type=int,default=25)
    result.add_argument('--epochs',type=float,default=3.)
    result.add_argument('--eval-per-epoch',type=float,default=4.)
    result.add_argument('--finalization-reserve-seconds', type=float, default=1800)
    result.add_argument('--finalization-cap-seconds', type=float, default=1800)
    result.add_argument('--stop-grace-seconds', type=float, default=180)
    result.add_argument('--strong-test-worker', default='benchmark_strong_tr_test_v2.py')
    result.add_argument('--audit-worker', default='verify_recovery_results_v2.py')
    result.add_argument('--validate-only', action='store_true')
    return result


if __name__ == '__main__':
    main(parser().parse_args())
