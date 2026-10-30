"""Finite local GPU trials with DEV-only selection and a wall-clock cap.

Screens request equal Turkish exposure from the same pilot; quota truncation
and fresh-optimizer extensions are explicit. Freeze one winner before TEST.
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

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parent.parent
BASE_REVISION = '541d1f99c6b0c3cd0b11a95167540bb8edefd82b'
STRONG_REVISION = '06f233fe06e710322aca913c1bc4249a0d71fce1'
PILOT_RUN = 'lora_20261010_01'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


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


def main(args):
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.loop_id) is not None, 'Use a filename-safe loop ID of at most 64 characters.')
    require(math.isfinite(args.hours) and 0 < args.hours <= 24, 'Initial window must be positive and at most 24 hours.')
    screen_epochs = getattr(args, 'screen_epochs', .25)
    reserve_seconds = getattr(args, 'finalization_reserve_seconds', 1800.)
    finalization_cap = getattr(args, 'finalization_cap_seconds', 1800.)
    stop_grace = getattr(args, 'stop_grace_seconds', 180.)
    require(math.isfinite(screen_epochs) and 0 < screen_epochs <= 1, 'Screen exposure must be positive and at most one Turkish epoch.')
    require(math.isfinite(reserve_seconds) and reserve_seconds >= 0 and math.isfinite(finalization_cap)
            and 0 < finalization_cap <= 3600 and math.isfinite(stop_grace)
            and 0 < stop_grace <= 600, 'Reserve/grace must be finite and bounded.')
    loop_dir = OUTPUT / 'runs' / args.loop_id
    loop_dir.mkdir(parents=True, exist_ok=True)
    status_path = loop_dir / 'loop_status.json'
    require(not status_path.exists(), 'Loop ID already exists; preserve its evidence and use a fresh ID.')
    start = time.monotonic()
    deadline = start + args.hours * 3600
    reserve = min(reserve_seconds, args.hours * 3600 / 4)
    training_deadline = deadline - reserve - min(stop_grace, args.hours * 3600 / 10)
    state = {'loop_id': args.loop_id, 'status': 'validating', 'started_at_utc': now(),
             'deadline_at_utc': (datetime.now(timezone.utc) + timedelta(hours=args.hours)).isoformat(),
             'initial_window_hours': args.hours, 'selection_split': 'FLEURS validation only',
             'finalization_reserve_seconds': reserve, 'stop_grace_seconds': stop_grace,
             'finalization_cap_seconds': finalization_cap,
             'screen_epochs_requested': screen_epochs, 'trials': [], 'warnings': [], 'controller_sha256': sha(Path(__file__)),
             'heldout_policy': 'No trial reads test predictions; freeze one overall candidate before finalization.'}
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
        best = {'run_id': PILOT_RUN, 'wer': incoming['turkish_validation_wer'], 'weights': incoming['weights'],
                'selected_step': incoming['selected_step'], 'trial': None, 'provenance': incoming_provenance}
        state.update(status='running', strong_baseline_wer=baseline_wer, competitive_threshold_wer=threshold,
                     competitive_definition='Turkish DEV WER <= measured Whisper DEV WER + 2 percentage points and every old-language DEV gate',
                     selected_configuration=selected, profile_sha256=sha(profile_path), strong_baseline_provenance=baseline_provenance,
                     expanded_manifest_sha256=sha(manifest), expanded_data_verification_sha256=sha(OUTPUT / 'expanded_turkish_verification.json'),
                     lexical_verified_turkish_clips=data_audit.get('lexical_filter', {}).get('accepted_clips'), best_development_candidate=dict(best))
        state['awake_request'] = keep_awake(True)
        awake_requested = bool(state['awake_request'].get('enabled'))
        write(status_path, state)
        note(f'{now()}: Started bounded {args.hours:g}-hour loop {args.loop_id}; measured DEV reference {100*baseline_wer:.4f}% WER, target {100*threshold:.4f}%. '
             f'Screens request {screen_epochs:g} equal Turkish epochs from identical pilot weights; actual exposure/quota truncations are recorded. DEV-only selection.')

        def trial(name, parameters, weights, epochs, max_seconds):
            run_id = f'{args.loop_id}_{name}'
            log = ROOT / f'work/{run_id}.log'
            stop_file = ROOT / f'work/{run_id}.stop'
            require(not (OUTPUT / 'runs' / run_id).exists() and not stop_file.exists(), 'Trial ID/stop marker already exists.')
            stage_start = time.monotonic()
            stage_deadline = min(training_deadline, stage_start + max_seconds)
            quota = max(1., stage_deadline - stage_start)
            command = [sys.executable, '-X', 'utf8', '-u', str(OUTPUT / 'train_lora_batched.py'), '--run-id', run_id,
                       '--tr-manifest', str(manifest), '--resume-weights', str(weights), '--epochs', str(epochs),
                       '--batch-size', str(selected['batch_size']), '--accumulation', str(selected['accumulation']),
                       '--tdt-implementation', selected['tdt_implementation'], '--chunk-size', str(selected['chunk_size']),
                       '--head-lr', str(parameters['head_lr']), '--lora-lr', str(parameters['lora_lr']),
                       '--new-token-lr-multiplier', str(parameters['new_token_lr_multiplier']), '--replay-kl', '5',
                       '--duration-kl', '.5', '--feature-distillation', '1', '--replay-fraction', '.5', '--warmup-steps', '100',
                       '--schedule', 'constant', '--eval-per-epoch', '1', '--eval-epoch-fraction', str(min(.25, epochs)),
                       '--early-stop-patience', '3', '--max-retention-failures', '2', '--min-improvement', '.002',
                       '--target-turkish-wer', str(threshold), '--max-wall-seconds', str(quota), '--stop-file', str(stop_file)]
            if not selected['encoder_checkpointing']:
                command += ['--no-encoder-checkpoint']
            entry = {'run_id': run_id, 'name': name, 'parameters': dict(parameters), 'epochs_planned': epochs,
                     'initialization_weights': str(weights), 'initialization_sha256': sha(weights), 'optimizer_resume': False,
                     'started_at_utc': now(), 'stage_wall_quota_seconds': quota, 'command': command, 'log': str(log), 'status': 'running'}
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
            entry['epochs_completed'] = selection.get('turkish_epochs_completed')
            entry['equal_requested_exposure_completed'] = isinstance(entry['epochs_completed'], (int, float)) and entry['epochs_completed'] + 1e-9 >= epochs
            entry['status'] = 'complete' if entry['exit_code'] == 0 else 'failed'
            if entry['exit_code'] == 0 and selection.get('accepted'):
                try:
                    accepted, provenance = validate_selection(run_id)
                    entry['selection_provenance'] = provenance
                    score = accepted['turkish_validation_wer']
                    if score < best['wer']:
                        best.update(run_id=run_id, wer=score, weights=accepted['weights'], selected_step=accepted['selected_step'],
                                    trial=name, parameters=dict(parameters), provenance=provenance)
                except Exception as exception:
                    entry['selection_rejected'] = str(exception)
            state['best_development_candidate'] = dict(best)
            write(status_path, state)
            note(f'{now()}: {run_id} exit {entry["exit_code"]}; actual Turkish epochs {entry["epochs_completed"]}, planned {epochs:g}. '
                 f'Best qualifying DEV WER {100*best["wer"]:.4f}% ({best["run_id"]}); stop reason {selection.get("stop_early_reason", selection.get("reason"))}.')

        screening = [('screen_A', {'head_lr': 2e-5, 'lora_lr': 3e-4, 'new_token_lr_multiplier': 15}),
                     ('screen_B', {'head_lr': 5e-5, 'lora_lr': 3e-4, 'new_token_lr_multiplier': 6}),
                     ('screen_C', {'head_lr': 2e-5, 'lora_lr': 6e-4, 'new_token_lr_multiplier': 15})]
        for index, (name, parameters) in enumerate(screening):
            remaining = training_deadline - time.monotonic()
            if remaining < 900 or best['wer'] <= threshold:
                break
            trial(name, parameters, incoming['weights'], screen_epochs, min(10800, remaining / (len(screening) - index + 1)))
        for extension in range(1, 4):
            remaining = training_deadline - time.monotonic()
            if remaining <= 900 or best['trial'] is None or best['wer'] <= threshold:
                break
            previous_best = best['wer']
            trial(f'extend_{extension}', best['parameters'], best['weights'], 3, remaining - 180)
            if best['wer'] > previous_best - .002:
                break
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
                    with (loop_dir / 'finalization.log').open('wb') as stream:
                        result = subprocess.run([sys.executable, '-X', 'utf8', '-u', str(OUTPUT / 'finalize_lora.py'), '--run-id', best['run_id']],
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
                    state['status'] = 'finalization_complete' if success else 'finalization_failed'
                except subprocess.TimeoutExpired:
                    state['status'] = 'finalization_timed_out'
                finally:
                    state['finalization_wall_seconds'] = time.monotonic() - final_start
                if state['status'] == 'finalization_complete':
                    strong_script = OUTPUT / 'benchmark_strong_tr_test.py'
                    require(strong_script.is_file(), 'Final paired strong TEST benchmark script is unavailable.')
                    timeout = post_deadline - time.monotonic()
                    if timeout <= 0:
                        state.update(status='strong_test_budget_exhausted', strong_test_status='not_run_budget_exhausted')
                    else:
                        state.update(status='benchmarking_strong_test', strong_test_status='running',
                                     strong_test_script_sha256=sha(strong_script), strong_test_timeout_seconds=timeout)
                        write(status_path, state)
                        comparison_path = OUTPUT / 'adaptive_strong_baseline_test_comparison.json'
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
                    verifier = OUTPUT / 'verify_adaptive_results.py'
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
                                                         '--strong-test-comparison', str(OUTPUT / 'adaptive_strong_baseline_test_comparison.json'),
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
        write(OUTPUT / 'adaptive_experiment_report.json', state)
        note(f'{now()}: Bounded loop finished {state["status"]}; best qualifying Turkish DEV WER {100*best["wer"]:.4f}% vs measured reference {100*baseline_wer:.4f}%; '
             f'provisional competitive DEV target met {state["competitive_on_development"]}. Pipeline completion and useful Turkish accuracy are separate.')
    except BaseException as exception:
        state.update(status='failed', failed_at_utc=now(), failure_type=type(exception).__name__, failure_message=str(exception),
                     total_wall_hours=(time.monotonic() - start) / 3600)
        write(status_path, state)
        write(OUTPUT / 'adaptive_experiment_report.json', state)
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
        write(OUTPUT / 'adaptive_experiment_report.json', state)
    print(json.dumps(state, ensure_ascii=False), flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--loop-id', default='adaptive_20261010_01')
    result.add_argument('--hours', type=float, default=12)
    result.add_argument('--screen-epochs', type=float, default=.25)
    result.add_argument('--finalization-reserve-seconds', type=float, default=1800)
    result.add_argument('--finalization-cap-seconds', type=float, default=1800)
    result.add_argument('--stop-grace-seconds', type=float, default=180)
    return result


if __name__ == '__main__':
    main(parser().parse_args())
