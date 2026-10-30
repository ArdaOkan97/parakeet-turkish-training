"""Persistent sequential ROOT-authorized row-LR trials; no ML imports.

The scheduled trial starts from CLOSED phase1 step7000/K500. A real one-update
smoke, source300 equality and serialized-state checks precede its main run.
On monitor failure the queue requests a whole-update stop and waits for its
owned controller. An unobserved close blocks all successors. This new executor
never takes over a live v1 queue; ROOT must confirm closure and archive its state
before a fresh v2 namespace is authorized.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYTHON = ROOT / 'work/monitor-env/Scripts/python.exe'
REFRESH_PYTHON = ROOT / 'work/training-env/Scripts/python.exe'
SOURCE_LOOP = HERE / 'runs/adaptive_20261011_hf_TSC_BPE512_acquire_01_loop_01'
STATE = HERE / 'experiment_queue_state.json'
CONTROLLER = HERE / 'run_TSC_bpe_row_schedule_memory_v2.py'
CREATOR = HERE / 'record_TSC_BPE_row_schedule_source_v1.py'
LANGUAGES = ('tr', 'en', 'es', 'it', 'fr', 'de')
LIMITS = {'en': 69, 'es': 52, 'it': 30, 'fr': 74, 'de': 70}
PHASE = 'HF_TSC_BPE512_new_rows_scheduled_acquisition_v1'
SOURCE_CHECKPOINT_SHA = 'd228603b1394adafe55a4f7e6209983643b2e5afcaded881006436d3351943f7'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.1)


def verify_pins(manifest):
    required_sources = {Path(__file__).name, CONTROLLER.name, CREATOR.name,
                        'train_lora_batched_tsc_bpe_row_schedule_memory_v2.py',
                        'tsc_bpe_row_schedule_memory_v2.py',
                        'bpe_newrows_scheduled_acquisition_v1.py',
                        'native_tdt_checkpointed_v4.py', 'metrics.py'}
    require(set(manifest['source_sha256']) == required_sources, 'Complete queue dependency pins required')
    for name, digest in manifest['source_sha256'].items():
        require(sha(HERE / name) == digest, 'Queue dependency changed: ' + name)
    require(manifest['kind'] == 'ROOT_authorized_TSC_row_schedule_memory_queue_v2'
            and manifest['source_selected_step'] == 7000
            and manifest['source_acquisition_updates'] == 500
            and manifest['original_source_step'] == 6500
            and manifest['source_checkpoint_sha256'] == SOURCE_CHECKPOINT_SHA,
            'Queue must use the qualified acquired7000 state, not a regressed phase2 state')
    require(manifest['peak_row_lr'] in (1e-3, 3e-4)
            and manifest['main_updates'] == 1000
            and manifest['ROOT_exclusive_GPU_authorized'] is True,
            'Only the declared matched warmup50/cosine1000 schedule is authorized')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{2,110}', manifest['queue_id']) is not None,
            'Safe fresh queue namespace required')
    trial_namespace = manifest.get('trial_namespace')
    require(isinstance(trial_namespace, str)
            and re.fullmatch(r'[A-Za-z0-9_-]{2,24}', trial_namespace) is not None,
            'Safe explicitly declared trial namespace required')
    verify_closed_predecessor(manifest)



CLEANUP_GRACE_SECONDS = 600.


def option(arguments, flag):
    require(arguments.count(flag) == 1, 'One owned-controller option required: ' + flag)
    index = arguments.index(flag)
    require(index + 1 < len(arguments), 'Missing owned-controller option: ' + flag)
    return arguments[index + 1]


def cooperative_stop_path(arguments):
    """Only the exact owned GPU-controller command may name a stopfile."""
    if '--gpu-window-authorized' not in arguments:
        return None
    require(str(CONTROLLER) in arguments and '--validate-only' not in arguments,
            'GPU cleanup requires the pinned controller command')
    run_id = option(arguments, '--run-id')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{2,110}', run_id) is not None,
            'Invalid owned run namespace')
    loop = Path(option(arguments, '--output-dir')).resolve()
    require(loop.parent == (HERE / 'runs').resolve()
            and loop.name == run_id + '_loop_01', 'Cleanup stop must be in this owned controller loop')
    return loop / (run_id + '_stop.txt')


def request_owned_stop(path):
    # The controller creates its own fresh directory. Creating it here would
    # violate admission during startup; retry after it exists instead.
    if path is None or not path.parent.is_dir():
        return False
    try:
        with path.open('x', encoding='utf-8') as stream:
            stream.write('Cooperative whole-update save/DEV: queue monitor failure; ROOT review required\n')
    except FileExistsError:
        pass  # Preserve an existing ROOT/resource/budget stop byte-for-byte.
    return True


def close_owned_after_failure(child, stop_path, *, grace_seconds=CLEANUP_GRACE_SECONDS):
    """Use only our retained Popen handle; no OS enumeration or forced kill."""
    evidence = {'pid': child.pid, 'cleanup_requested': True,
                'started_at_utc': now(), 'grace_seconds': grace_seconds,
                'cooperative_stop_path': str(stop_path) if stop_path else None,
                'cooperative_stop_present_or_created': False,
                'actual_wait_observed': False, 'closure_observed': False,
                'exit_code': None, 'forced_termination_used': False, 'cleanup_errors': []}
    deadline = time.monotonic() + grace_seconds
    # wait(0), unlike a status file or missing PID, can actually observe an
    # already-closed owned child without requesting an unnecessary stop.
    timeout = 0.
    while True:
        try:
            code = child.wait(timeout=timeout)
            evidence.update(actual_wait_observed=True, closure_observed=True,
                            exit_code=code, observed_at_utc=now())
            return evidence
        except subprocess.TimeoutExpired:
            pass
        except BaseException as error:
            evidence['cleanup_errors'].append(repr(error))
        if stop_path is not None and not evidence['cooperative_stop_present_or_created']:
            try:
                evidence['cooperative_stop_present_or_created'] = request_owned_stop(stop_path)
            except BaseException as error:
                evidence['cleanup_errors'].append(repr(error))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            evidence.update(status='owned_controller_closure_unobserved', finished_at_utc=now())
            return evidence
        timeout = min(5., remaining)
        # Some handle errors return immediately. Avoid spinning on a broken
        # wait handle while retaining the same bounded cleanup deadline.
        if evidence['cleanup_errors']:
            try:
                time.sleep(min(.1, remaining))
            except BaseException as error:
                evidence['cleanup_errors'].append(repr(error))


def execute_owned(label, arguments, queue, publish, state):
    """Observe actual closure before reporting repair or admitting a successor."""
    stop_path = cooperative_stop_path(arguments)
    log = queue / (label + '.log')
    publish(status='running', current_stage=label, current_command=arguments, current_log=str(log))
    child = None
    close = None
    error = None
    try:
        with log.open('x', encoding='utf-8') as stream:
            child = subprocess.Popen(arguments, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            # Keep the handle before this fallible state publication.
            state['owned_child_pid'] = child.pid
            publish(owned_child_pid=child.pid, child_started_at_utc=now())
            while child.poll() is None:
                publish()
                time.sleep(5)
            code = child.wait()
            close = {'pid': child.pid, 'cleanup_requested': False,
                     'actual_wait_observed': True, 'closure_observed': True,
                     'exit_code': code, 'observed_at_utc': now(), 'forced_termination_used': False}
    except BaseException as caught:
        error = repr(caught)
        raise
    finally:
        if child is not None:
            if close is None:
                close = close_owned_after_failure(child, stop_path)
            close.update(kind='TSC_row_LR_queue_owned_child_closure_v2', stage=label,
                         monitoring_error=error, objective_complete=False)
            # Even if the publication that caused the failure remains broken,
            # retain independent queue-local cleanup evidence and emit it to
            # ROOT's captured stdout. Never clear the PID without actual wait.
            state['last_owned_child_cleanup'] = close
            state['owned_child_pid'] = None if close['closure_observed'] else child.pid
            if close['actual_wait_observed']:
                state['last_actual_waitpid'] = {'pid': child.pid, 'exit_code': close['exit_code'],
                                               'observed_at_utc': close['observed_at_utc']}
            try:
                write(queue / (label + '_owned_closure.json'), close)
            except BaseException as persistence_error:
                state['cleanup_evidence_persistence_error'] = repr(persistence_error)
            print(json.dumps({'owned_child_closure': close}, allow_nan=False), flush=True)
    publish(last_actual_waitpid=state['last_actual_waitpid'], owned_child_pid=None)
    require(close['exit_code'] == 0, 'Actual child failed at ' + label + '; inspect ' + str(log))

def review_closed(run_id, loop, smoke, rate):
    """Review actual child closure and independently rescore all saved DEV rows."""
    from metrics import score_transcriptions
    from bpe_newrows_scheduled_acquisition_v1 import schedule_identity, lr_for_update
    status = read(loop / 'loop_status.json')
    require(status['run_id'] == run_id and status['status'] == 'train_window_ended'
            and status['child_exit_code'] == 0 and status['resource_review_required'] is False,
            'Child did not close successfully; preserve artifacts and request repair')
    require(status['stop_category'] in ('phase_bound', 'retention', 'development_target_met'),
            'Unexpected stop category requires review')
    native = Path(status['paths']['native'])
    completion = read(native.with_suffix('.completion.json'))
    require(completion['status'] == 'complete' and completion['sources_unchanged'] is True
            and not completion.get('error') and completion['run_id'] == run_id
            and completion['backend_ready_sha256'] == sha(native),
            'Native completion is not actual, successful and bound to startup')
    run = HERE / 'runs' / run_id
    proof = read(run / 'TSC_BPE_row_schedule_initial_source_equivalence.json')
    require(proof['passed'] is True and proof['actual_GPU_decode'] is True
            and proof['compared_fixed_DEV_utterances'] == 300
            and proof['all_raw_transcripts_and_emitted_paths_exact'] is True,
            'Actual before-update fixed300 source equality is required')
    config = read(run / 'run_config.json')
    selection = read(run / 'selection.json')
    history = read(run / 'training_history.json')
    local = selection['acquisition_updates_completed']
    require(selection['phase'] == PHASE and selection['heldout_tests_evaluated'] is False
            and selection['source_selected_step'] == 7000
            and selection['source_acquisition_updates'] == 500
            and selection['row_Adam_clock'] == 500 + local
            and selection['completed_steps'] == 7000 + local == len(history),
            'Local updates and retained row-Adam clock must remain distinct')
    require(config['peak_row_lr'] == rate, 'Declared schedule peak differs')
    expected_schedule = schedule_identity(rate)
    require(selection['row_schedule'] == expected_schedule
            and selection['scheduled_acquisition_updates'] == local
            and selection['current_row_lr'] == selection['last_applied_row_lr'] == lr_for_update(local, rate),
            'Actual scheduled local clock/applied LR differs')
    for update, saved in enumerate(history[7000:], 1):
        require(saved['scheduled_acquisition_updates'] == update
                and saved['row_Adam_clock'] == 500 + update
                and saved['current_row_lr'] == saved['last_applied_row_lr'] == saved['active_row_lr'] == lr_for_update(update, rate),
                'Every applied LR must match its local scheduled update')
    if smoke:
        require(local == 1, 'Smoke must execute exactly one actual update')
        mechanical = read(run / 'acquisition_actual_one_update_smoke.json')
        require(mechanical['scheduled_state_save_load_passed'] is True
                and mechanical['row_schedule'] == expected_schedule
                and mechanical['scheduled_acquisition_updates'] == 1
                and mechanical['current_row_lr'] == lr_for_update(1, rate), 'Actual scheduled-state smoke required')
        for key in ('passed', 'actual_native_ASR_backward_and_Adam',
                    'all3_new_row_deltas_nonzero',
                    'protected_old_six_blank_duration_and_frozen204_rawbytes_exact',
                    'parked_source_unchanged', 'complete207_phase_save_load_passed',
                    'all207_reloaded_weight_bytes_exact', 'all3_Adam_moments_clocks_and_groups_exact',
                    'saved_sampler_history_config_and_actual_RNG_frontier_exact'):
            require(mechanical.get(key) is True, 'Actual smoke failed: ' + key)
    else:
        require(0 < local <= 1000, 'Main trial must have bounded positive updates')
    results = []
    for row in history[7000:]:
        if 'validation' not in row:
            continue
        step = row['step']
        result = HERE / 'results' / f'{run_id}_step_{step}'
        summary = read(result / 'summary.json')
        scores, emissions = {}, {}
        for language in LANGUAGES:
            predictions = [json.loads(line) for line in (result / f'validation_{language}.jsonl').read_text(encoding='utf-8').splitlines()]
            require(len(predictions) == 50, 'Complete fixed50 DEV rows required for every language')
            score = score_transcriptions(predictions, language=language)
            score.pop('details')
            require(score == summary[f'validation_{language}']['asr'], 'Independent DEV rescore differs')
            scores[language] = score
            require(all('tokens' in prediction.get('confidence', {}) for prediction in predictions),
                    'Actual emitted token paths are required; missing paths cannot count as zero')
            emissions[language] = sum(8199 <= token['token_id'] < 8711
                for prediction in predictions for token in prediction['confidence']['tokens'])
        all_five = all(scores[language]['word_errors'] <= cap for language, cap in LIMITS.items())
        require(all_five == row['validation']['retention_gate_passed'], 'Independent retention gates differ')
        results.append({'step': step, 'TR_WER': scores['tr']['wer'], 'scores': scores,
                        'retention_passed': all_five, 'new_BPE_emissions': emissions})
    require(results and results[-1]['step'] == selection['completed_steps'], 'Final complete DEV gate missing')
    before, after = history[6999], history[-1]
    train_time = after['train_wall_seconds'] - before['train_wall_seconds']
    require(train_time > 0 and math.isfinite(train_time), 'Phase training wall time must be positive')
    throughput = {name: (after[counter] - before[counter]) / train_time for name, counter in
                  (('mixed_audio_min_per_train_min', 'audio_seen_seconds'),
                   ('TR_audio_min_per_train_min', 'turkish_audio_seen_seconds'))}
    return {'run_id': run_id, 'passed': True, 'updates_completed': local, 'row_Adam_clock': 500 + local,
            'peak_row_lr': rate, 'row_schedule': expected_schedule, 'last_applied_row_lr': selection['last_applied_row_lr'], 'actual_native_completion_sha256': sha(native.with_suffix('.completion.json')),
            'DEV_points': results, 'last_retention_passed': results[-1]['retention_passed'],
            'throughput_excludes_startup_and_DEV': throughput, 'one_update_rate_is_not_steady': smoke,
            'selected_step': selection['selected_step'], 'goal_complete': False}


def backlog_note(text, reference):
    command = [str(HERE / 'tools/backlog.exe'), 'task', 'edit', 'ASR-15',
               '--append-notes', text, '--add-ref', str(reference)]
    subprocess.run(command, cwd=HERE, check=True, capture_output=True, text=True, encoding='utf-8')
    subprocess.run([str(REFRESH_PYTHON), '-X', 'utf8', str(HERE / 'refresh_backlog.py')],
                   cwd=ROOT, check=True, capture_output=True, text=True, encoding='utf-8')


def verify_closed_predecessor(manifest):
    predecessor = manifest.get('predecessor_closed_state', {})
    prior_path = Path(predecessor.get('path', '')).resolve()
    require(prior_path.is_relative_to(ROOT / 'work/experiment_queues')
            and sha(prior_path) == predecessor.get('sha256')
            and read(prior_path)['status'] == 'repair_required'
            and predecessor.get('ROOT_owned_exec_closed_exit_code') == 1
            and predecessor.get('ROOT_owned_exec_session_id') == 59763
            and predecessor.get('ROOT_owned_exec_exit_tool_chunk') == '68a1dc',
            'Preserve actual CLOSED1 resource predecessor; no concurrent handoff')
    approval_record = manifest['changed_memory_policy_approval']
    approval_path = Path(approval_record['path']).resolve()
    require(approval_path.is_relative_to(HERE) and sha(approval_path) == approval_record['sha256'],
            'ROOT changed-resource-policy review must be immutable')
    approval = read(approval_path)
    require(approval['kind'] == 'ROOT_actual_CLOSED_resource_stop_changed_policy_approval_v1'
            and approval['passed'] is True and approval['allow_next_changed_cache_policy'] is True
            and approval['native_complete_and_sources_unchanged'] is True
            and approval['next_controller_sha256'] == sha(CONTROLLER)
            and approval['new_end_cache_threshold_bytes'] == 3_000_000_000
            and approval['old_end_cache_threshold_bytes'] == 4_000_000_000
            and approval['cap_bytes'] == 8_000_000_000,
            'Changed cache policy and actual successful native save are required')


def fresh_memory_preflight():
    from resource_watchdog_v2 import sample_memory
    sample = sample_memory(os.getpid(), 'scheduled_memory_queue_before_GPU_dispatch')
    memory = sample['system_memory']
    historical_peak = 12_176_424_960
    projected = memory['commit_used_bytes'] + historical_peak
    require(memory['available'] is True and projected / memory['commit_limit_bytes'] < .95
            and memory['commit_limit_bytes'] - projected > 2_000_000_000,
            'Fresh baseline plus historical owned peak fails unchanged critical guard; defer GPU dispatch')
    return {'sample': memory, 'historical_owned_private_peak_bytes': historical_peak,
            'projected_fraction': projected / memory['commit_limit_bytes'],
            'prediction_is_not_actual_fit_proof': True}


def run(args):
    manifest = read(args.manifest)
    verify_pins(manifest)
    if args.validate_only:
        print(json.dumps({'passed': True, 'GPU_launched': False, 'queue': manifest['queue_id']}))
        return 0
    require(args.gpu_window_authorized, 'ROOT must authorize the sole sequential GPU queue')
    trial_namespace = manifest['trial_namespace']
    queue = ROOT / 'work/experiment_queues' / manifest['queue_id']
    require(not queue.exists() and not STATE.exists(), 'Fresh queue namespace/state required')
    queue.mkdir(parents=True)
    state = {'kind': manifest['kind'], 'queue_id': manifest['queue_id'], 'pid': os.getpid(),
             'ppid': os.getppid(), 'started_at_utc': now(), 'status': 'starting',
             'manifest': str(args.manifest.resolve()), 'manifest_sha256': sha(args.manifest),
             'trials': [], 'objective_complete': False, 'next_review': 'authorized recurring chat heartbeat'}

    def publish(**values):
        state.update(values, updated_at_utc=now())
        write(STATE, state)
        write(queue / 'state.json', state)

    def execute(label, arguments):
        verify_pins(manifest)
        if '--gpu-window-authorized' in arguments:
            publish(fresh_memory_preflight=fresh_memory_preflight())
        execute_owned(label, arguments, queue, publish, state)

    try:
        for tag, rate in (('schedule', manifest['peak_row_lr']),):
            for stage in ('smoke', 'main'):
                run_id = f'adaptive_20261011_hf_TSC_BPE512_row_schedule_{trial_namespace}_{stage}'
                loop = HERE / 'runs' / (run_id + '_loop_01')
                receipt = HERE / f'TSC_BPE_row_schedule_{trial_namespace}_{stage}_CLOSED_source_v1.json'
                archive = ROOT / 'work/closed-sources' / f'TSC_BPE_row_schedule_{trial_namespace}_{stage}_source_v1'
                execute(tag + '_' + stage + '_archive', [str(PYTHON), '-S', str(CREATOR),
                        '--source-loop', str(SOURCE_LOOP), '--chosen-checkpoint-role', 'latest',
                        '--owned-session-id', '20495', '--owned-exit-chunk', '50fde1',
                        '--owned-exit-observed-at', '2026-10-11T22:31:27Z', '--peak-row-lr', str(rate),
                        '--target-run-id', run_id, '--archive-dir', str(archive), '--receipt-output', str(receipt)])
                require(read(receipt)['artifacts']['chosen_checkpoint']['sha256'] == SOURCE_CHECKPOINT_SHA,
                        'Wrong source checkpoint admitted')
                command = [str(PYTHON), '-S', str(CONTROLLER), '--source-loop', str(SOURCE_LOOP),
                           '--source-receipt', str(receipt), '--run-id', run_id, '--output-dir', str(loop),
                           '--peak-row-lr', str(rate), '--acquisition-max-updates', '1000', '--window-hours', '3']
                if stage == 'smoke':
                    command.extend(('--smoke-updates', '1'))
                preflight = HERE / f'{run_id}_queue_preflight_v2.json'
                execute(tag + '_' + stage + '_preflight', command + ['--validate-only', '--preflight-output', str(preflight)])
                publish(current_run_id=run_id, current_controller_loop=str(loop))
                execute(tag + '_' + stage + '_train', command + ['--gpu-window-authorized'])
                review = review_closed(run_id, loop, stage == 'smoke', rate)
                report = queue / (tag + '_' + stage + '_review.json')
                write(report, review)
                state['trials'].append(review)
                publish(status='reviewed', current_stage=tag + '_' + stage + '_review')
                points = '; '.join(f"step{p['step']} TR WER{100*p['TR_WER']:.4f}% all5={p['retention_passed']} newTRtokens={p['new_BPE_emissions']['tr']}" for p in review['DEV_points'])
                backlog_note(f"Persistent local queue {manifest['queue_id']}: actual CLOSED {run_id}; peakRowLR={rate}; warmup50/cosine1000/floor0.1, encoder/shared204 frozen, acquired row moments retained. {points}. Goal remains incomplete; no held-out TEST used. Next scheduled arm is independent from the same qualified7000 anchor.", report)
                print(run_id + ': ' + points, flush=True)
                if stage == 'smoke' and not review['last_retention_passed']:
                    state['trials'].append({'arm': tag, 'main_skipped': True, 'reason': 'actual_smoke_retention_failure'})
                    publish()
                    break
        publish(status='trials_complete_review_required', finished_at_utc=now(),
                current_stage=None, current_run_id=None,
                next_action='Recurring chat heartbeat reviews learning curves and launches next quality experiment/full-pass recipe; short screens do not achieve the objective')
        backlog_note('Persistent warmup/cosine row queue completed its scheduled diagnostic trial. Objective remains incomplete. Recurring chat review is active to select and launch the next experiment; do not interpret queue completion as competitive Turkish WER or a completed full dataset pass.', STATE)
        return 0
    except BaseException as error:
        # execute_owned has already attempted cooperative closure. A timeout is
        # explicitly unknown, and this terminal repair path cannot retry/spawn.
        state.update(status='repair_required', error=repr(error), finished_at_utc=now(),
                owned_controller_closure_observed=(state.get('last_owned_child_cleanup', {}).get('closure_observed')
                    if state.get('last_owned_child_cleanup', {}).get('stage') == state.get('current_stage') else None),
                next_action='ROOT checks actual owned closure before any repair or new GPU dispatch')
        try:
            publish()
        except BaseException as persistence_error:
            state['repair_publication_error'] = repr(persistence_error)
            try:
                write(queue / 'repair_state.json', state)
            except BaseException as fallback_error:
                state['repair_fallback_error'] = repr(fallback_error)
        print(json.dumps({'repair_required': state}, allow_nan=False), flush=True)
        return 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--manifest', type=Path, required=True)
    result.add_argument('--validate-only', action='store_true')
    result.add_argument('--gpu-window-authorized', action='store_true')
    return result


if __name__ == '__main__':
    raise SystemExit(run(parser().parse_args()))
