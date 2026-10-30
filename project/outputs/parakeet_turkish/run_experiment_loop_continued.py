"""Continue the authorized interrupted loop after a checkpointed GPU benchmark.

The original controller and trainer remain byte-identical. Resume screen A
with its exact optimizer/sampler/RNG state; initialize later screens from the
same original pilot. All work shares the original absolute wall-clock deadline.
Only subprocesses created by this controller are managed by this controller.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import run_experiment_loop as original
from run_experiment_loop import (BASE_REVISION, OUTPUT, ROOT, PILOT_RUN, now, read,
    write, sha, require, finite_wer, backlog_note, keep_awake, stop_owned_child,
    validate_baseline, validate_selection)

ORIGINAL_CONTROLLER_SHA256 = '226612259c9682096bf9181fb785b8ad36e099d4626c2b136163022b1fccf4ae'
SCREENING = [('screen_A', {'head_lr': 2e-5, 'lora_lr': 3e-4, 'new_token_lr_multiplier': 15}),
             ('screen_B', {'head_lr': 5e-5, 'lora_lr': 3e-4, 'new_token_lr_multiplier': 6}),
             ('screen_C', {'head_lr': 2e-5, 'lora_lr': 6e-4, 'new_token_lr_multiplier': 15})]


def stop_owned_child(child):
    """Retain the frozen cleanup policy, with handle-based fallback if poll fails."""
    try:
        original.stop_owned_child(child)
    except Exception:
        if child is None:
            raise
        # A Popen object contains the handle for the exact child we created;
        # this does not look up or signal any host PID by number.
        child.terminate()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=30)


def local_path(value, label, *, under=None):
    path = Path(value).resolve()
    require(path.is_relative_to((under or ROOT).resolve()), label + ' must remain in its workspace directory.')
    return path


def hashed_path(manifest, key, *, under=None):
    path = local_path(manifest.get(key + '_path', ''), key, under=under)
    digest = manifest.get(key + '_sha256')
    require(path.is_file() and isinstance(digest, str) and sha(path) == digest, key + ' file/hash mismatch.')
    return path


def timestamp(value, label):
    stamp = datetime.fromisoformat(value)
    require(stamp.tzinfo is not None, label + ' must include a timezone.')
    return stamp.astimezone(timezone.utc)


def update_option(command, option, value):
    result = list(command)
    require(result.count(option) == 1, 'Original command needs one ' + option)
    index = result.index(option)
    require(index + 1 < len(result), 'Missing original command option value.')
    result[index + 1] = str(value)
    return result


def exact_resume_command(entry, checkpoint, remaining_seconds):
    command = list(entry['command'])
    require('--resume-exact' not in command and command.count('--resume-weights') == 1,
            'Original screen must have one new-optimizer warm start.')
    command[command.index('--resume-weights')] = '--resume-exact'
    command = update_option(command, '--resume-exact', checkpoint)
    command = update_option(command, '--max-wall-seconds', max(1., remaining_seconds))
    # Do not inject --steps: its original identity is null although the inferred
    # screen cap is 3,204 updates. Changing it would reject exact resume.
    return command


def validate_marker(marker, loop_id):
    roles = {'trainer_pause': 'screen_A', 'next_stage_interlock': 'screen_B'}
    require(marker.get('role') in roles, 'Unknown owned stop-marker role.')
    expected = (ROOT / 'work' / f"{loop_id}_{roles[marker['role']]}.stop").resolve()
    path = local_path(marker.get('path', ''), 'Owned stop marker', under=ROOT / 'work')
    require(path == expected and path.is_file() and sha(path) == marker.get('sha256'),
            'Owned stop marker path/content differs; refusing to remove it.')
    return path


def validate_pause(args):
    """Read-only CPU preflight. No process signals, CUDA calls or heldout predictions."""
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.loop_id) is not None, 'Unsafe loop ID.')
    loop_dir = OUTPUT / 'runs' / args.loop_id
    manifest_path = local_path(args.pause_manifest or loop_dir / 'gpu_benchmark_pause.json', 'Pause manifest', under=loop_dir)
    pause = read(manifest_path, {})
    require(pause.get('schema') == 1 and pause.get('loop_id') == args.loop_id
            and pause.get('run_id') == args.loop_id + '_screen_A', 'Wrong pause schema/loop/run.')
    require(pause.get('benchmark_completed') is True and pause.get('original_controller_exited') is True,
            'Benchmark completion and original-controller exit must be explicitly verified.')
    require(pause.get('original_controller_sha256') == ORIGINAL_CONTROLLER_SHA256
            and sha(OUTPUT / 'run_experiment_loop.py') == ORIGINAL_CONTROLLER_SHA256,
            'Original frozen controller bytes changed.')
    origin_path = hashed_path(pause, 'original_status', under=loop_dir)
    origin = read(origin_path, {})
    require(origin.get('loop_id') == args.loop_id and origin.get('controller_sha256') == ORIGINAL_CONTROLLER_SHA256
            and origin.get('selection_split') == 'FLEURS validation only', 'Original status identity differs.')
    require(origin.get('initial_window_hours') == 12., 'Continuation must retain the authorized original 12-hour window.')
    start_at = timestamp(origin['started_at_utc'], 'Original start')
    deadline_at = timestamp(origin['deadline_at_utc'], 'Original deadline')
    require(math.isclose((deadline_at - start_at).total_seconds(), 12 * 3600, abs_tol=2),
            'Original absolute deadline/window mismatch.')
    state = read(loop_dir / 'loop_status.json', {})
    after_path = hashed_path(pause, 'original_status_after_pause', under=loop_dir)
    require(sha(loop_dir / 'loop_status.json') == sha(after_path),
            'Original terminal controller status changed since its exit snapshot.')
    require(state.get('loop_id') == args.loop_id and state.get('status') == 'failed'
            and state.get('controller_sha256') == ORIGINAL_CONTROLLER_SHA256,
            'Original controller must have exited and preserved its terminal state.')
    require(not state.get('active_run_id') and not state.get('continuation'),
            'Another active/previous continuation is present.')
    require(state.get('deadline_at_utc') == origin['deadline_at_utc']
            and state.get('started_at_utc') == origin['started_at_utc'], 'Original timing changed after pause.')
    entries = [entry for entry in state.get('trials', []) if entry.get('run_id') == pause['run_id']]
    require(len(entries) == 1 and len(state['trials']) == 1, 'Expected exactly one original screen A; no later trainer may have launched.')
    entry = entries[0]
    original_entries = [record for record in origin.get('trials', []) if record.get('run_id') == pause['run_id']]
    require(len(original_entries) == 1 and entry.get('command') == original_entries[0].get('command'),
            'Original trainer command differs from the hashed pre-pause status.')
    require(entry.get('exit_code') == 0 and entry.get('status') == 'complete'
            and entry.get('name') == 'screen_A' and entry.get('parameters') == SCREENING[0][1],
            'Original screen A did not exit cooperatively and successfully.')
    for key in ('original_config_before_pause', 'paused_config', 'paused_history', 'paused_selection'):
        hashed_path(pause, key, under=loop_dir)
    config = read(Path(pause['paused_config_path']), {})
    before_config = read(Path(pause['original_config_before_pause_path']), {})
    require(config == before_config and config.get('run_id') == pause['run_id'] and config.get('steps') == 3204
            and config.get('epochs') == origin['screen_epochs_requested'], 'Paused original screen/config changed.')
    run_dir = OUTPUT / 'runs' / pause['run_id']
    for key, name in (('paused_config', 'run_config.json'), ('paused_history', 'training_history.json'), ('paused_selection', 'selection.json')):
        require(sha(run_dir / name) == pause[key + '_sha256'], 'Live paused ' + name + ' differs from its immutable snapshot.')
    checkpoint_path = hashed_path(pause, 'checkpoint', under=ROOT / 'work')
    latest = ROOT / 'work' / (pause['run_id'] + '_latest.pt')
    require(sha(latest) == pause['checkpoint_sha256'], 'Live latest checkpoint differs from archived pause state.')
    require(checkpoint_path != latest.resolve(), 'Resume from an immutable archived checkpoint, preserving pause evidence.')
    selection = read(Path(pause['paused_selection_path']), {})
    history = read(Path(pause['paused_history_path']), [])
    import torch
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    step = checkpoint.get('step')
    require(isinstance(step, int) and 0 < step < config['steps'] and checkpoint.get('schema') == 2
            and checkpoint.get('run_id') == pause['run_id'] and len(history) == step
            and checkpoint.get('history') == history and selection.get('completed_steps') == step,
            'Pause checkpoint/history/completion identity mismatch.')
    require(checkpoint.get('config_identity') == config.get('config_identity')
            and checkpoint.get('lora_metadata') == config.get('lora_layout')
            and checkpoint.get('stop_reason') == selection.get('stop_early_reason') == 'controller_stop_file',
            'Pause state is not a complete cooperative same-config checkpoint.')
    require(len(checkpoint.get('optimizer', {}).get('param_groups', [])) == 2
            and bool(checkpoint['optimizer'].get('state')) and bool(checkpoint.get('cuda_rng'))
            and checkpoint.get('selected_step') == selection.get('selected_step')
            and checkpoint.get('best_score') == selection.get('turkish_validation_wer'),
            'Pause optimizer/RNG/selected-state evidence incomplete.')
    for key in ('python_rng', 'torch_rng', 'next_eval_epoch', 'sampler', 'patience', 'retention_failures'):
        require(key in checkpoint, 'Missing exact-resume state: ' + key)
    from train_lora_batched import BalancedBucketSampler, parser as trainer_parser
    pool = read(run_dir / 'training_pool.json', {})
    sampler = BalancedBucketSampler(pool['turkish'], pool['replay'], seed=config['seed'],
        batch_size=config['batch_size'], accumulation=config['accumulation'], replay_fraction=config['replay_fraction'],
        bucket_size=config['bucket_size'])
    sampler.load_state_dict(checkpoint['sampler'])
    require(sampler.tr.draws == step * sampler.tr_per_update
            and sum(stream.draws for stream in sampler.replay.values()) == step * sampler.replay_per_update,
            'Pause sampler draws differ from completed optimizer updates.')
    command = entry['command']
    require(len(command) > 5 and Path(command[0]).resolve() == Path(sys.executable).resolve()
            and command[1:4] == ['-X', 'utf8', '-u']
            and Path(command[4]).resolve() == (OUTPUT / 'train_lora_batched.py').resolve(),
            'Original trainer command must use the current pinned local Python and frozen trainer.')
    parsed = trainer_parser().parse_args(command[5:])
    require(parsed.run_id == pause['run_id'] and parsed.steps is None and parsed.epochs == config['epochs']
            and parsed.batch_size == config['batch_size'] and parsed.accumulation == config['accumulation']
            and parsed.tdt_implementation == config['tdt_implementation'], 'Stored original trainer argv/config differs.')
    parsed_arguments = {key:str(value) if isinstance(value,Path) else value for key,value in vars(parsed).items()}
    require(all(parsed_arguments.get(key) == value for key,value in config['config_identity']['training_arguments'].items()),
            'Stored trainer hyperparameters differ from the exact checkpoint identity.')
    if selection.get('accepted'):
        validate_selection(pause['run_id'])
    for name, digest in config['code_sha256'].items():
        require(Path(name).name == name and sha(OUTPUT / name) == digest, 'Frozen training source changed: ' + name)
    for value, digest in config['input_sha256'].items():
        require(sha(local_path(value, 'Frozen training manifest')) == digest, 'Frozen training input changed: ' + value)
    markers = pause.get('owned_stop_markers', [])
    require(len(markers) == 2 and {marker.get('role') for marker in markers} == {'trainer_pause', 'next_stage_interlock'},
            'Expected the two owned stop-marker interlocks.')
    marker_paths = [validate_marker(marker, args.loop_id) for marker in markers]
    evidence = pause.get('benchmark_evidence', [])
    require(isinstance(evidence, list) and bool(evidence), 'Completed benchmark requires concrete hashed evidence.')
    benchmark_pause_path = loop_dir / 'gpu_benchmark_pause_checkpoint_verified.json'
    benchmark_pause = read(benchmark_pause_path, {})
    require(benchmark_pause_path.is_file(), 'Immutable checkpoint manifest used by GPU profiling is missing.')
    benchmark_pause_sha = sha(benchmark_pause_path)
    for key in ('loop_id', 'run_id', 'original_controller_sha256', 'original_status_sha256',
                'paused_config_sha256', 'paused_history_sha256', 'paused_selection_sha256', 'checkpoint_sha256',
                'checkpoint_path', 'owned_stop_markers'):
        require(benchmark_pause.get(key) == pause.get(key), 'Immutable profiling checkpoint evidence differs: ' + key)
    bound_profile = False
    for record in evidence:
        path = local_path(record.get('path', ''), 'Benchmark evidence')
        require(path.is_file() and sha(path) == record.get('sha256'), 'Benchmark evidence hash differs.')
        if path.suffix == '.json':
            provenance = read(path, {}).get('pause_provenance')
            if provenance is not None:
                require(Path(provenance.get('path', '')).resolve() == benchmark_pause_path.resolve()
                        and provenance.get('sha256') == benchmark_pause_sha,
                        'GPU profile does not bind the exact immutable pause checkpoint manifest.')
                bound_profile = True
    require(bound_profile, 'Completed benchmark evidence must include an actual profile bound to the immutable pause manifest.')
    del checkpoint
    return {'pause': pause, 'pause_path': manifest_path, 'origin': origin, 'state': state, 'entry': entry,
            'checkpoint': checkpoint_path, 'checkpoint_step': step, 'config': config, 'marker_paths': marker_paths,
            'benchmark_pause_path': benchmark_pause_path, 'benchmark_pause_sha256': benchmark_pause_sha,
            'start_at': start_at, 'deadline_at': deadline_at}


def fresh_command(run_id, parameters, weights, epochs, quota, selected, manifest, threshold, stop_file):
    command = [sys.executable, '-X', 'utf8', '-u', str(OUTPUT / 'train_lora_batched.py'), '--run-id', run_id,
        '--tr-manifest', str(manifest), '--resume-weights', str(weights), '--epochs', str(epochs),
        '--batch-size', str(selected['batch_size']), '--accumulation', str(selected['accumulation']),
        '--tdt-implementation', selected['tdt_implementation'], '--chunk-size', str(selected['chunk_size']),
        '--head-lr', str(parameters['head_lr']), '--lora-lr', str(parameters['lora_lr']),
        '--new-token-lr-multiplier', str(parameters['new_token_lr_multiplier']), '--replay-kl', '5', '--duration-kl', '.5',
        '--feature-distillation', '1', '--replay-fraction', '.5', '--warmup-steps', '100', '--schedule', 'constant',
        '--eval-per-epoch', '1', '--eval-epoch-fraction', str(min(.25, epochs)), '--early-stop-patience', '3',
        '--max-retention-failures', '2', '--min-improvement', '.002', '--target-turkish-wer', str(threshold),
        '--max-wall-seconds', str(max(1., quota)), '--stop-file', str(stop_file)]
    if not selected['encoder_checkpointing']:
        command += ['--no-encoder-checkpoint']
    return command


def main(args):
    preflight = validate_pause(args)
    if args.validate_only:
        print(json.dumps({'passed': True, 'cpu_only': True, 'loop_id': args.loop_id,
            'run_id': preflight['pause']['run_id'], 'checkpoint_step': preflight['checkpoint_step'],
            'checkpoint_sha256': preflight['pause']['checkpoint_sha256'],
            'deadline_at_utc': preflight['origin']['deadline_at_utc'],
            'continuation_controller_sha256': sha(Path(__file__))}, indent=2))
        return
    origin, pause = preflight['origin'], preflight['pause']
    loop_dir = OUTPUT / 'runs' / args.loop_id
    status_path = loop_dir / 'loop_status.json'
    remaining = (preflight['deadline_at'] - datetime.now(timezone.utc)).total_seconds()
    require(remaining > origin['finalization_reserve_seconds'] + origin['stop_grace_seconds'],
            'Original absolute training window is exhausted; refuse to launch a trainer.')
    monotonic_now = time.monotonic()
    deadline = monotonic_now + remaining
    elapsed = (datetime.now(timezone.utc) - preflight['start_at']).total_seconds()
    start = monotonic_now - elapsed
    reserve = origin['finalization_reserve_seconds']
    stop_grace = origin['stop_grace_seconds']
    finalization_cap = origin['finalization_cap_seconds']
    training_deadline = deadline - reserve - stop_grace
    screen_epochs = origin['screen_epochs_requested']
    state = copy.deepcopy(preflight['state'])
    failed_snapshot = loop_dir / 'original_controller_after_pause.json'
    require(not failed_snapshot.exists(), 'A continuation already preserved the interrupted controller state.')
    shutil.copyfile(status_path, failed_snapshot)
    previous_failure = {key: state.get(key) for key in ('status', 'failure_type', 'failure_message', 'failed_at_utc')}
    for key in ('failure_type', 'failure_message', 'failed_at_utc', 'finished_at_utc', 'awake_restoration', 'active_run_id'):
        state.pop(key, None)
    state['continuation'] = {'controller_path': str(Path(__file__).resolve()), 'controller_sha256': sha(Path(__file__)),
        'started_at_utc': now(), 'pause_manifest_path': str(preflight['pause_path']), 'pause_manifest_sha256': sha(preflight['pause_path']),
        'interrupted_controller_status_path': str(failed_snapshot), 'interrupted_controller_status_sha256': sha(failed_snapshot),
        'benchmark_checkpoint_manifest_path': str(preflight['benchmark_pause_path']),
        'benchmark_checkpoint_manifest_sha256': preflight['benchmark_pause_sha256'],
        'benchmark_evidence': copy.deepcopy(pause['benchmark_evidence']),
        'original_terminal_state': previous_failure, 'checkpoint_path': str(preflight['checkpoint']),
        'checkpoint_sha256': pause['checkpoint_sha256'], 'checkpoint_step': preflight['checkpoint_step'],
        'budget_policy': 'Original absolute 12-hour deadline retained; benchmark/pause time counts against it.',
        'resume_policy': 'Same screen A optimizer/sampler/RNG/config; additional cooperative-pause DEV evaluation recorded.',
        'later_screen_policy': 'B/C start from the same original step-1200 pilot, with fresh optimizers.'}
    entry = next(item for item in state['trials'] if item['run_id'] == pause['run_id'])
    entry['pause_segment'] = copy.deepcopy(preflight['entry'])
    entry['status'] = 'paused_partial'
    entry['equal_requested_exposure_completed'] = False
    entry.pop('selection_provenance', None)
    state['pre_continuation_best_development_candidate'] = copy.deepcopy(state['best_development_candidate'])
    awake_requested = False

    def note(message):
        try:
            backlog_note(message)
        except Exception as exception:
            state['warnings'].append({'at_utc': now(), 'kind': 'backlog_update_failed', 'message': str(exception)})
            print('Backlog update warning: ' + str(exception), flush=True)

    try:
        profile_path = OUTPUT / 'batching_speed_comparison.json'
        profile = read(profile_path, {})
        require(profile.get('passed') is True and profile.get('throughput_improved') is True
                and sha(profile_path) == origin['profile_sha256'], 'Original verified throughput profile changed.')
        selected = profile['selected_configuration']
        require(selected == origin['selected_configuration'], 'Original numerical/batch configuration changed.')
        baseline, baseline_provenance = validate_baseline()
        baseline_wer, threshold = baseline['metrics']['wer'], baseline['metrics']['wer'] + .02
        require(baseline_provenance == origin['strong_baseline_provenance']
                and threshold == origin['competitive_threshold_wer'], 'Original DEV benchmark/target changed.')
        manifest = ROOT / 'work/data/tr_expanded_train.jsonl'
        data_audit = read(OUTPUT / 'expanded_turkish_verification.json', {})
        require(data_audit.get('passed') is True and sha(manifest) == origin['expanded_manifest_sha256']
                and sha(OUTPUT / 'expanded_turkish_verification.json') == origin['expanded_data_verification_sha256'],
                'Original verified Turkish data changed.')
        incoming, incoming_provenance = validate_selection(PILOT_RUN)
        require(incoming['selected_step'] == 1200, 'Later screens require the original step-1200 pilot.')
        best = {'run_id': PILOT_RUN, 'wer': incoming['turkish_validation_wer'], 'weights': incoming['weights'],
                'selected_step': incoming['selected_step'], 'trial': None, 'provenance': incoming_provenance}
        state.update(status='continuing', best_development_candidate=copy.deepcopy(best))
        state['awake_request'] = keep_awake(True)
        awake_requested = bool(state['awake_request'].get('enabled'))
        # Unlink only the exact two workspace files whose identities were audited.
        for marker in pause['owned_stop_markers']:
            validate_marker(marker, args.loop_id).unlink()
        state['continuation']['owned_markers_removed_at_utc'] = now()
        write(status_path, state)
        note(f'{now()}: Continuing {args.loop_id} from exact screen A checkpoint step {preflight["checkpoint_step"]}; '
             f'original deadline {origin["deadline_at_utc"]} retained. Benchmark time is included; B/C retain identical pilot initialization.')

        def rebuild_best():
            nonlocal best
            # Refresh all provenance, including a same-score/same-run winner,
            # because exact resume rewrites history and run_config metadata.
            current, provenance = validate_selection(PILOT_RUN)
            best = {'run_id': PILOT_RUN, 'wer': current['turkish_validation_wer'], 'weights': current['weights'],
                    'selected_step': current['selected_step'], 'trial': None, 'provenance': provenance}
            for candidate in state['trials']:
                if candidate.get('status') != 'complete' or candidate.get('exit_code') != 0 or not candidate.get('selection', {}).get('accepted'):
                    continue
                try:
                    accepted, provenance = validate_selection(candidate['run_id'])
                    candidate['selection_provenance'] = provenance
                    if accepted['turkish_validation_wer'] < best['wer']:
                        best = {'run_id': candidate['run_id'], 'wer': accepted['turkish_validation_wer'], 'weights': accepted['weights'],
                                'selected_step': accepted['selected_step'], 'trial': candidate['name'],
                                'parameters': copy.deepcopy(candidate['parameters']), 'provenance': provenance}
                except Exception as exception:
                    candidate['selection_rejected'] = str(exception)
                    candidate.pop('selection_provenance', None)
            state['best_development_candidate'] = copy.deepcopy(best)

        def trial(name, parameters, weights, epochs, max_seconds, *, resume=False):
            run_id = args.loop_id + '_' + name
            run_dir = OUTPUT / 'runs' / run_id
            stop_file = ROOT / 'work' / (run_id + '.stop')
            stage_start = time.monotonic()
            if resume:
                item = next(record for record in state['trials'] if record['run_id'] == run_id)
                original_stage_end = timestamp(item['started_at_utc'], 'Original A start').timestamp() + item['stage_wall_quota_seconds']
                remaining_stage = original_stage_end - datetime.now(timezone.utc).timestamp()
                stage_deadline = min(training_deadline, stage_start + remaining_stage)
                require(stage_deadline > stage_start, 'Original screen A quota exhausted during benchmark; no exact-resume launch.')
                command = exact_resume_command(preflight['entry'], preflight['checkpoint'], stage_deadline - stage_start)
                log = ROOT / 'work' / (run_id + '_exact_resume.log')
                item.update(status='running', optimizer_resume=True, resumed_at_utc=now(), resume_command=command,
                    resume_checkpoint_sha256=pause['checkpoint_sha256'], resume_start_step=preflight['checkpoint_step'],
                    resume_wall_quota_seconds=stage_deadline-stage_start, resume_log=str(log))
            else:
                require(not run_dir.exists() and not stop_file.exists(), 'Fresh trial ID/stop marker exists.')
                stage_deadline = min(training_deadline, stage_start + max_seconds)
                command = fresh_command(run_id, parameters, weights, epochs, stage_deadline-stage_start,
                    selected, manifest, threshold, stop_file)
                log = ROOT / 'work' / (run_id + '.log')
                item = {'run_id':run_id, 'name':name, 'parameters':copy.deepcopy(parameters), 'epochs_planned':epochs,
                    'initialization_weights':str(weights), 'initialization_sha256':sha(weights), 'optimizer_resume':False,
                    'started_at_utc':now(), 'stage_wall_quota_seconds':stage_deadline-stage_start,
                    'command':command, 'log':str(log), 'status':'running'}
                state['trials'].append(item)
            require(not stop_file.exists(), 'Owned stop marker was not cleared.')
            state.update(active_run_id=run_id, status='running')
            write(status_path, state)
            child, requested_stop, last_dev_step = None, False, None
            environment = dict(os.environ, PYTHONUTF8='1', HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
            try:
                with log.open('wb') as stream:
                    child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    item['resume_pid' if resume else 'pid'] = child.pid
                    while child.poll() is None:
                        history = read(run_dir / 'training_history.json', [])
                        if history:
                            item.update(completed_updates=history[-1]['step'], latest_training_epoch=history[-1].get('turkish_epoch'))
                            dev = [row for row in history if 'validation' in row]
                            if dev and dev[-1]['step'] != last_dev_step:
                                record, last_dev_step = dev[-1], dev[-1]['step']
                                item.update(latest_validation=record['validation'], latest_validation_epoch=record.get('turkish_epoch'))
                                print(json.dumps({'run_id':run_id,'step':record['step'],'epoch':record.get('turkish_epoch'),
                                    'validation':record['validation']}), flush=True)
                                note(f'{now()}: {run_id} DEV step {record["step"]}, TR epoch {record.get("turkish_epoch")}; '
                                     f'WER {100*record["validation"]["turkish_wer"]:.4f}%, retention {record["validation"]["retention_gate_passed"]}.')
                        current = time.monotonic()
                        state.update(updated_at_utc=now(), remaining_wall_minutes=max(0., deadline-current)/60)
                        if current >= stage_deadline and not requested_stop:
                            stop_file.write_text('Original stage quota expired; save DEV-selected evidence and exit.\n', encoding='utf-8')
                            requested_stop = True
                            item['deadline_stop_requested_at_utc'] = now()
                        write(status_path, state)
                        if requested_stop and current >= min(stage_deadline+stop_grace, deadline-reserve):
                            item['forced_stop'] = 'Trainer exceeded bounded cooperative-stop grace.'
                            stop_owned_child(child)
                            break
                        time.sleep(15)
                    item['exit_code'] = child.returncode
            finally:
                stop_owned_child(child)
                item['finished_at_utc'] = now()
                item['resume_segment_wall_seconds' if resume else 'wall_seconds'] = time.monotonic()-stage_start
                state.pop('active_run_id', None)
            selection = read(run_dir / 'selection.json', {})
            item['selection'] = selection
            item['epochs_completed'] = selection.get('turkish_epochs_completed')
            item['completed_updates'] = selection.get('completed_steps')
            item['equal_requested_exposure_completed'] = (isinstance(item['epochs_completed'], (int,float))
                and item['epochs_completed']+1e-9 >= epochs)
            item['status'] = 'complete' if item['exit_code'] == 0 else 'failed'
            if resume and item['status'] == 'complete':
                config = read(run_dir / 'run_config.json', {})
                warm = config.get('warm_start', {})
                identity = config.get('config_identity')
                valid = (warm.get('mode') == 'exact_resume' and warm.get('sha256') == pause['checkpoint_sha256']
                    and warm.get('source_step') == preflight['checkpoint_step'] and identity == preflight['config']['config_identity']
                    and isinstance(selection.get('completed_steps'),int) and selection['completed_steps'] >= preflight['checkpoint_step'])
                if not valid:
                    item.update(status='failed', resume_verification_failure='Exited child did not prove exact same-state resume.')
            rebuild_best()
            write(status_path, state)
            note(f'{now()}: {run_id} exit {item["exit_code"]}; actual TR epochs {item["epochs_completed"]}, planned {epochs:g}; '
                 f'best qualifying DEV WER {100*best["wer"]:.4f}% ({best["run_id"]}).')

        # A keeps its existing stage deadline rather than receiving another
        # screen quota. Later B/C quotas derive only from the remaining window.
        trial('screen_A', SCREENING[0][1], incoming['weights'], screen_epochs, 0, resume=True)
        for index, (name, parameters) in enumerate(SCREENING[1:], start=1):
            remaining = training_deadline-time.monotonic()
            if remaining < 900 or best['wer'] <= threshold:
                break
            trial(name, parameters, incoming['weights'], screen_epochs, min(10800, remaining/(len(SCREENING)-index+1)))
        for extension in range(1,4):
            remaining = training_deadline-time.monotonic()
            if remaining <= 900 or best['trial'] is None or best['wer'] <= threshold:
                break
            previous_best = best['wer']
            trial('extend_' + str(extension), best['parameters'], best['weights'], 3, remaining-180)
            if best['wer'] > previous_best-.002:
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
    result.add_argument('--pause-manifest', type=Path)
    result.add_argument('--validate-only', action='store_true')
    return result


if __name__ == '__main__':
    main(parser().parse_args())
