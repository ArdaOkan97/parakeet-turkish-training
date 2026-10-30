"""Persistent sequential ROOT-authorized row-LR trials; no ML imports.

Every trial starts from CLOSED phase1 step7000/K500. A real one-update
smoke, source300 equality and serialized-state checks precede its main run.
The app heartbeat reviews this queue when it ends or requires repair.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
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
CONTROLLER = HERE / 'run_TSC_bpe_row_lr_v1.py'
CREATOR = HERE / 'record_TSC_BPE_row_LR_source_v1.py'
LANGUAGES = ('tr', 'en', 'es', 'it', 'fr', 'de')
LIMITS = {'en': 69, 'es': 52, 'it': 30, 'fr': 74, 'de': 70}
PHASE = 'HF_TSC_BPE512_new_rows_continued_acquisition_v1'
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
                        'train_lora_batched_tsc_bpe_row_lr_v1.py',
                        'tsc_bpe_row_lr_transition_v1.py',
                        'bpe_newrows_continued_acquisition_v1.py',
                        'native_tdt_checkpointed_v4.py', 'metrics.py'}
    require(set(manifest['source_sha256']) == required_sources, 'Complete queue dependency pins required')
    for name, digest in manifest['source_sha256'].items():
        require(sha(HERE / name) == digest, 'Queue dependency changed: ' + name)
    require(manifest['kind'] == 'ROOT_authorized_TSC_row_LR_queue_v1'
            and manifest['source_selected_step'] == 7000
            and manifest['source_acquisition_updates'] == 500
            and manifest['original_source_step'] == 6500
            and manifest['source_checkpoint_sha256'] == SOURCE_CHECKPOINT_SHA,
            'Queue must use the qualified acquired7000 state, not a regressed phase2 state')
    require(manifest['active_row_learning_rates'] == [1e-3, 3e-4]
            and manifest['main_updates'] == 1000
            and manifest['ROOT_exclusive_GPU_authorized'] is True,
            'Only the declared sequential two-arm schedule is authorized')


def review_closed(run_id, loop, smoke, rate):
    """Review actual child closure and independently rescore all saved DEV rows."""
    from metrics import score_transcriptions
    status = read(loop / 'loop_status.json')
    require(status['run_id'] == run_id and status['status'] == 'train_window_ended'
            and status['child_exit_code'] == 0 and status['resource_review_required'] is False,
            'Child did not close successfully; preserve artifacts and request repair')
    require(status['stop_category'] in ('phase_bound', 'retention'),
            'Unexpected stop category requires review')
    native = Path(status['paths']['native'])
    completion = read(native.with_suffix('.completion.json'))
    require(completion['status'] == 'complete' and completion['sources_unchanged'] is True
            and not completion.get('error') and completion['run_id'] == run_id
            and completion['backend_ready_sha256'] == sha(native),
            'Native completion is not actual, successful and bound to startup')
    run = HERE / 'runs' / run_id
    proof = read(run / 'TSC_BPE_row_LR_initial_source_equivalence.json')
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
    require(config['active_row_lr'] == rate, 'Active row learning rate differs')
    if smoke:
        require(local == 1, 'Smoke must execute exactly one actual update')
        mechanical = read(run / 'acquisition_actual_one_update_smoke.json')
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
            'active_row_lr': rate, 'actual_native_completion_sha256': sha(native.with_suffix('.completion.json')),
            'DEV_points': results, 'last_retention_passed': results[-1]['retention_passed'],
            'throughput_excludes_startup_and_DEV': throughput, 'one_update_rate_is_not_steady': smoke,
            'selected_step': selection['selected_step'], 'goal_complete': False}


def backlog_note(text, reference):
    command = [str(HERE / 'tools/backlog.exe'), 'task', 'edit', 'ASR-15',
               '--append-notes', text, '--add-ref', str(reference)]
    subprocess.run(command, cwd=HERE, check=True, capture_output=True, text=True, encoding='utf-8')
    subprocess.run([str(REFRESH_PYTHON), '-X', 'utf8', str(HERE / 'refresh_backlog.py')],
                   cwd=ROOT, check=True, capture_output=True, text=True, encoding='utf-8')


def run(args):
    manifest = read(args.manifest)
    verify_pins(manifest)
    if args.validate_only:
        print(json.dumps({'passed': True, 'GPU_launched': False, 'queue': manifest['queue_id']}))
        return 0
    require(args.gpu_window_authorized, 'ROOT must authorize the sole sequential GPU queue')
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
        log = queue / (label + '.log')
        publish(status='running', current_stage=label, current_command=arguments, current_log=str(log))
        with log.open('x', encoding='utf-8') as stream:
            child = subprocess.Popen(arguments, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            publish(owned_child_pid=child.pid, child_started_at_utc=now())
            while child.poll() is None:
                publish()
                time.sleep(5)
            code = child.wait()
        publish(last_actual_waitpid={'pid': child.pid, 'exit_code': code, 'observed_at_utc': now()}, owned_child_pid=None)
        require(code == 0, 'Actual child failed at ' + label + '; inspect ' + str(log))

    try:
        for tag, rate in (('1e3', 1e-3), ('3e4', 3e-4)):
            for stage in ('smoke', 'main'):
                run_id = f'adaptive_20261011_hf_TSC_BPE512_row_lr{tag}_01_{stage}'
                loop = HERE / 'runs' / (run_id + '_loop_01')
                receipt = HERE / f'TSC_BPE_row_LR_{tag}_{stage}_CLOSED_source_v1.json'
                archive = ROOT / 'work/closed-sources' / f'TSC_BPE_row_LR_{tag}_{stage}_source_v1'
                execute(tag + '_' + stage + '_archive', [str(PYTHON), '-S', str(CREATOR),
                        '--source-loop', str(SOURCE_LOOP), '--chosen-checkpoint-role', 'latest',
                        '--owned-session-id', '20495', '--owned-exit-chunk', '50fde1',
                        '--owned-exit-observed-at', '2026-10-11T22:31:27Z', '--active-row-lr', str(rate),
                        '--target-run-id', run_id, '--archive-dir', str(archive), '--receipt-output', str(receipt)])
                require(read(receipt)['artifacts']['chosen_checkpoint']['sha256'] == SOURCE_CHECKPOINT_SHA,
                        'Wrong source checkpoint admitted')
                command = [str(PYTHON), '-S', str(CONTROLLER), '--source-loop', str(SOURCE_LOOP),
                           '--source-receipt', str(receipt), '--run-id', run_id, '--output-dir', str(loop),
                           '--active-row-lr', str(rate), '--acquisition-max-updates', '1000', '--window-hours', '3']
                if stage == 'smoke':
                    command.extend(('--smoke-updates', '1'))
                preflight = HERE / f'{run_id}_queue_preflight_v1.json'
                execute(tag + '_' + stage + '_preflight', command + ['--validate-only', '--preflight-output', str(preflight)])
                publish(current_run_id=run_id, current_controller_loop=str(loop))
                execute(tag + '_' + stage + '_train', command + ['--gpu-window-authorized'])
                review = review_closed(run_id, loop, stage == 'smoke', rate)
                report = queue / (tag + '_' + stage + '_review.json')
                write(report, review)
                state['trials'].append(review)
                publish(status='reviewed', current_stage=tag + '_' + stage + '_review')
                points = '; '.join(f"step{p['step']} TR WER{100*p['TR_WER']:.4f}% all5={p['retention_passed']} newTRtokens={p['new_BPE_emissions']['tr']}" for p in review['DEV_points'])
                backlog_note(f"Persistent local queue {manifest['queue_id']}: actual CLOSED {run_id}; rowLR={rate}, encoder/shared204 frozen, acquired row moments retained. {points}. Goal remains incomplete; no held-out TEST used. Next scheduled arm is independent from the same qualified7000 anchor.", report)
                print(run_id + ': ' + points, flush=True)
                if stage == 'smoke' and not review['last_retention_passed']:
                    state['trials'].append({'arm': tag, 'main_skipped': True, 'reason': 'actual_smoke_retention_failure'})
                    publish()
                    break
        publish(status='trials_complete_review_required', finished_at_utc=now(),
                current_stage=None, current_run_id=None,
                next_action='Recurring chat heartbeat reviews learning curves and launches next quality experiment/full-pass recipe; short screens do not achieve the objective')
        backlog_note('Persistent row-LR queue completed its scheduled diagnostic trials. Objective remains incomplete. Recurring chat review is active to select and launch the next experiment; do not interpret queue completion as competitive Turkish WER or a completed full dataset pass.', STATE)
        return 0
    except BaseException as error:
        publish(status='repair_required', error=repr(error), finished_at_utc=now(),
                next_action='Recurring chat heartbeat inspects actual failure and repairs before further GPU dispatch')
        print(repr(error), flush=True)
        return 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--manifest', type=Path, required=True)
    result.add_argument('--validate-only', action='store_true')
    result.add_argument('--gpu-window-authorized', action='store_true')
    return result


if __name__ == '__main__':
    raise SystemExit(run(parser().parse_args()))
