"""ROOT-authorized exact-window handoff using pinned controller closure evidence.

The initial cross-job controller is not independently queried through Windows
handles. Its immutable source owns/waits its worker and publishes terminal state
only after verified completion. Every subsequent controller is this supervisor's
own Popen child and its actual exit is also required. No ML/GPU imports.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

import supervise_hf_same_science_v1 as base

V1_SHA = '85909ea7af43ea5d9996195473a5ba701b41a76bf498b58dbc2f0b0d9b806176'
HERE, ROOT = base.HERE, base.ROOT
read, write, sha, scoped, record, now, require = base.read, base.write, base.sha, base.scoped, base.record, base.now, base.require


def controller_order_proof():
    """Bind the real, immutable parent's child-close -> completion -> publish path."""
    require(sha(base.__file__) == V1_SHA and sha(base.HELPER) == base.HELPER_SHA, 'Frozen supervisor/helper source changed')
    tree = ast.parse(base.HELPER.read_text(encoding='utf-8'))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)]
    popen = [n for n in calls if isinstance(n.func, ast.Attribute) and ast.unparse(n.func) == 'subprocess.Popen']
    require(len(popen) == 1, 'Pinned controller must own exactly one training child')
    close_check = next(n for n in calls if ast.unparse(n.func) == 'require' and n.args
        and ast.unparse(n.args[0]) == 'child.returncode == 0')
    wait_loop = next(n for n in ast.walk(main) if isinstance(n, ast.While) and ast.unparse(n.test) == 'child.poll() is None')
    completion = next(n for n in ast.walk(main) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'completion' for t in n.targets))
    terminal = next(n for n in calls if ast.unparse(n.func) == 'state.update'
        and any(k.arg == 'child_exit_code' for k in n.keywords))
    outer_try = next(n for n in main.body if isinstance(n, ast.Try))
    final_publish = next(n for n in ast.walk(ast.Module(body=outer_try.finalbody, type_ignores=[]))
        if isinstance(n, ast.Call) and ast.unparse(n.func) == 'write')
    require(popen[0].lineno < wait_loop.lineno < close_check.lineno < completion.lineno < terminal.lineno < final_publish.lineno,
        'Actual child close/completion/terminal publication ordering changed')
    require(not any(n.lineno > terminal.lineno for n in popen), 'No future GPU child may spawn after terminal state')
    return {'kind': 'pinned_controller_child_close_terminal_publication_source_proof_v2',
        'controller': record(base.HELPER), 'source_lines': {'owned_child_spawn': popen[0].lineno,
        'owned_child_poll_loop': wait_loop.lineno, 'actual_returncode_zero_requirement': close_check.lineno,
        'fresh_runtime_completion_check': completion.lineno, 'terminal_state_assignment': terminal.lineno,
        'atomic_terminal_publish_in_finally': final_publish.lineno}, 'future_GPU_spawn_after_terminal': False,
        'initial_independent_OS_exit_observation': False}


def worker_proof(state, run_id, expected_launcher=None, expected_worker=None):
    identity_path = scoped(state['paths']['identity']); identity = read(identity_path, {})
    ready_path = scoped(state['paths']['fresh_runtime']); ready = read(ready_path, {})
    pid, ppid = identity.get('pid'), identity.get('ppid')
    require(type(pid) is int and type(ppid) is int and pid > 0 and ppid == state.get('owned_launcher_pid'), 'Owned worker/launcher identity differs')
    require(expected_launcher is None or ppid == expected_launcher, 'Declared ROOT launcher differs')
    require(expected_worker is None or pid == expected_worker, 'Declared ROOT worker differs')
    require(identity.get('expected_run_id') == run_id and identity.get('frozen_worker_sha256')
        == state['runtime_source_sha256']['train_lora_batched_data_v11.py'], 'Frozen self-reported worker identity differs')
    creation = identity.get('process_creation_identity', {})
    require(creation.get('available') is True and creation.get('pid') == pid and creation.get('creation_filetime_100ns', 0) > 0,
        'Worker self-reported creation identity missing')
    require(ready.get('kind') == 'HF_same_science_native_preload_resume_v1' and ready.get('status') == 'ready'
        and ready.get('run_id') == run_id and ready.get('pid') == pid and ready.get('ppid') == ppid
        and ready.get('launcher_sha256') == base.PRELOADER_SHA, 'Actual fresh preload identity differs')
    require(state.get('fresh_runtime') == ready, 'Pinned controller did not validate this fresh worker proof')
    argument_sha = hashlib.sha256(json.dumps(state['worker_arguments'], ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    require(ready.get('worker_arguments_sha256') == argument_sha, 'Actual fresh worker argument identity differs')
    original = state['original_ready']
    require(record(scoped(original['path'])) == original and ready.get('original_first_start_ready', {}).get('sha256') == original['sha256'],
        'Original immutable scientific-ready identity differs')
    for name, digest in ready['source_sha256'].items():
        require(Path(name).name == name and sha(HERE / name) == digest, 'Fresh runtime source changed: ' + name)
    return {'worker_identity': record(identity_path), 'fresh_runtime': record(ready_path), 'worker_pid': pid,
        'launcher_pid': ppid, 'worker_creation_identity_scope': 'Worker self-report, not supervisor-independent process query'}


def bootstrap(folder, run_id, launcher_pid, worker_pid):
    state = base.source_state(folder, run_id)
    require(state.get('status') == 'running', 'Bootstrap requires the current pinned controller to be running')
    timestamp = datetime.fromisoformat(state['updated_at_utc'])
    age = (datetime.now(timezone.utc) - timestamp).total_seconds()
    require(-5 <= age <= 30, 'Initial controller ownership/progress observation is stale')
    proof = worker_proof(state, run_id, launcher_pid, worker_pid)
    return {'kind': 'pinned_live_controller_metadata_bootstrap_v2', 'run_id': run_id, 'source_loop': str(folder),
        'source_state': record(folder / 'loop_status.json'), 'controller_updated_at_utc': state['updated_at_utc'],
        'controller_observation_age_seconds': age, **proof, 'controller_close_publish_order': controller_order_proof(),
        'initial_independent_OS_exit_observation': False, 'no_process_enumeration_or_permission_bypass': True,
        'initial_source_controller_state_snapshot': state, 'observed_at_utc': now()}


def closure(folder, run_id, *, owned_child=None, initial_bootstrap=None):
    # A new window is never authorized while our own controller is still live.
    owned_exit = None
    if owned_child is not None:
        require(owned_child.poll() is not None, 'Supervisor-owned controller remains live')
        owned_exit = owned_child.wait(timeout=30)
        require(owned_exit == 0, 'Supervisor-owned controller actually failed')
    state = base.source_state(folder, run_id)
    require(state.get('status') == 'train_window_ended' and state.get('child_exit_code') == 0
        and state.get('resource_review_required') is False, 'Clean pinned controller terminal state required')
    worker = worker_proof(state, run_id)
    if initial_bootstrap is not None:
        require(worker['worker_identity'] == initial_bootstrap['worker_identity']
            and worker['fresh_runtime'] == initial_bootstrap['fresh_runtime'], 'Initial ownership proof changed before closure')
    completion_path = scoped(state['paths']['fresh_runtime']).with_suffix('.completion.json')
    completion = read(completion_path, {})
    require(completion.get('kind') == 'HF_same_science_native_preload_resume_completion_v1'
        and completion.get('status') == 'complete' and completion.get('run_id') == run_id
        and completion.get('pid') == worker['worker_pid'] and completion.get('sources_unchanged') is True
        and completion.get('fresh_runtime_proof_sha256') == worker['fresh_runtime']['sha256']
        and completion.get('launcher_sha256_after') == base.PRELOADER_SHA, 'Actual current preload completion differs')
    runtime = {}
    for key in ('io', 'cache', 'memory'):
        path = scoped(state['paths'][key]); require(read(path, {}).get('status') == 'complete', 'Runtime completion missing: ' + key)
        runtime[key] = record(path)
    selection_path = HERE / 'runs' / run_id / 'selection.json'; selection = read(selection_path, {})
    require(state.get('selection') == record(selection_path) and selection.get('run_id') == run_id
        and selection.get('completed_steps') == state.get('completed_steps') and selection.get('accepted') is True,
        'Saved actual final checkpoint/DEV selection differs')
    reason, helper_reason = selection.get('stop_early_reason'), state.get('cooperative_stop_reason')
    budget = reason == 'soft_max_wall_seconds' or (reason == 'controller_stop_file' and helper_reason == 'window_budget_checkpoint_boundary')
    checkpoints = {role: record(scoped(selection[key])) for role, key in (('full_latest', 'latest_weights'), ('qualified_best', 'weights'))}
    return {'kind': 'pinned_controller_evidence_verified_same_science_window_closure_v2', 'run_id': run_id,
        'source_loop': str(folder), 'closure_observed_from_pinned_controller_evidence': True,
        'supervisor_independent_initial_OS_exit_observation': False, 'actual_supervisor_owned_controller_exit_code': owned_exit,
        'supervisor_owned_controller_pid': owned_child.pid if owned_child is not None else None,
        'controller_close_publish_order': controller_order_proof(), 'helper_final_state': record(folder / 'loop_status.json'),
        'selection': record(selection_path), 'saved_checkpoint_bytes': checkpoints,
        'fresh_runtime_completion': record(completion_path), 'runtime_completions': runtime,
        **worker, 'completed_steps': selection['completed_steps'], 'selected_step': selection['selected_step'],
        'selection_stop_reason': reason, 'helper_stop_reason': helper_reason, 'automatic_budget_handoff_allowed': budget,
        'objective_complete': False, 'verified_at_utc': now()}


def main(args):
    folder, evidence = scoped(args.source_loop), scoped(args.supervisor_dir)
    require(evidence != folder and not evidence.exists(), 'NEW supervisor namespace required')
    require(.25 <= args.max_hours <= 24 and .25 <= args.window_hours <= 6 and 5 <= args.poll_seconds <= 15, 'Bounded supervisor settings required')
    require(args.successor_prefix and Path(args.successor_prefix).name == args.successor_prefix, 'Safe successor folder prefix required')
    config = read(HERE / 'runs' / args.run_id / 'run_config.json', {})
    cap = config['config_identity']['training_arguments']['steps']; horizon = config.get('steps')
    selection = read(HERE / 'runs' / args.run_id / 'selection.json', {})
    require(type(horizon) is int and horizon > 0 and selection.get('configured_steps') == horizon
        and type(cap) is int and cap >= horizon, 'Effective configured horizon must agree')
    initial = bootstrap(folder, args.run_id, args.launcher_pid, args.worker_pid)
    evidence.mkdir(parents=True, exist_ok=False); write(evidence / 'initial_metadata_bootstrap.json', initial)
    state = {'kind': 'automatic_same_science_window_supervisor_v2', 'status': 'metadata_bootstrap_verified',
        'run_id': args.run_id, 'supervisor_pid': os.getpid(), 'supervisor_source_sha256': sha(__file__),
        'frozen_v1_source_sha256': V1_SHA, 'helper_sha256': base.HELPER_SHA, 'preloader_sha256': base.PRELOADER_SHA,
        'initial_closure_scope': 'Pinned controller child-close evidence; independent cross-job process handles unavailable',
        'configured_optimizer_horizon_unchanged': horizon, 'original_scientific_argument_step_cap_unchanged': cap,
        'window_hours': args.window_hours, 'max_hours': args.max_hours, 'objective_complete': False,
        'source_loop': str(folder), 'windows': [], 'started_at_utc': now()}
    write(evidence / 'supervisor_status.json', state)
    if args.bootstrap_only:
        print(__import__('json').dumps({'passed': True, 'metadata_bootstrap_verified': True,
            'independent_OS_process_attachment': False, 'GPU_launch': False, 'receipt': str(evidence / 'initial_metadata_bootstrap.json')})); return 0
    require(args.gpu_window_authorized, 'ROOT must authorize exclusive automatic successors')
    deadline = time.monotonic() + args.max_hours * 3600; child = None; index = 2
    try:
        while True:
            while True:
                pending = read(folder / 'loop_status.json', {})
                if child is not None and child.poll() is not None: break
                if child is None and pending.get('status') != 'running': break
                state.update(status='waiting_pinned_owned_controller_closure', current_source_loop=str(folder), updated_at_utc=now())
                write(evidence / 'supervisor_status.json', state)
                if time.monotonic() >= deadline:
                    state.update(status='supervisor_horizon_waiting_current_window', next_experiment_required=True)
                    write(evidence / 'supervisor_status.json', state); return 0
                time.sleep(args.poll_seconds)
            closed = closure(folder, args.run_id, owned_child=child, initial_bootstrap=initial if child is None else None)
            receipt_path = evidence / ('window_' + str(index-1).zfill(2) + '_controller_closure.json')
            write(receipt_path, closed); state['windows'].append(closed)
            remaining = deadline - time.monotonic()
            if closed['completed_steps'] >= horizon or not closed['automatic_budget_handoff_allowed'] or remaining < 900:
                state.update(status='next_experiment_required', stop_reason='configured_horizon' if closed['completed_steps'] >= horizon
                    else closed['selection_stop_reason'] if not closed['automatic_budget_handoff_allowed'] else 'bounded_supervisor_wall_horizon',
                    objective_complete=False, finished_at_utc=now()); write(evidence / 'supervisor_status.json', state); return 0
            target = folder.parent / (args.successor_prefix + '_' + str(index).zfill(2))
            command = base.successor_command(folder, target, args.run_id, min(args.window_hours, remaining / 3600))
            with (evidence / ('successor_' + str(index).zfill(2) + '.log')).open('x', encoding='utf-8') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state.update(status='successor_started', latest_launch={'source_verified_controller_closure': record(receipt_path),
                'target': str(target), 'command': command, 'owned_controller_Popen_pid': child.pid, 'started_at_utc': now(),
                'actual_launch_completed': True, 'actual_owned_controller_exit_code': None})
            write(evidence / 'supervisor_status.json', state)
            folder = target; index += 1
    except BaseException as error:
        state.update(status='failed_no_automatic_retry', error=repr(error), objective_complete=False, finished_at_utc=now())
        write(evidence / 'supervisor_status.json', state); raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-loop', type=Path, required=True); p.add_argument('--run-id', required=True)
    p.add_argument('--supervisor-dir', type=Path, required=True); p.add_argument('--launcher-pid', type=int, required=True)
    p.add_argument('--worker-pid', type=int, required=True); p.add_argument('--successor-prefix', required=True)
    p.add_argument('--window-hours', type=float, default=4.5); p.add_argument('--max-hours', type=float, default=12.)
    p.add_argument('--poll-seconds', type=float, default=5.); p.add_argument('--bootstrap-only', action='store_true')
    p.add_argument('--gpu-window-authorized', action='store_true'); return p


if __name__ == '__main__': sys.exit(main(parser().parse_args()))
