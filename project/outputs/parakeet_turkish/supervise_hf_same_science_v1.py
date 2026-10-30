"""ROOT-owned automatic handoff of CLOSED, same-science HF training windows.

Only declared owned processes are queried: no enumeration, command-line reads,
settings changes, process suspension or GPU/model imports. Pinned existing
controllers retain resource/progress guards and exact checkpoint restoration.
"""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from atomic_io_retry_v1 import read_json as read, write_json as write

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
HELPER = HERE / 'resume_hf_same_science_v1.py'
HELPER_SHA = 'd5518b468088abda37fae894a86e9de3f56ad3460d682ed2fb0d730cc7901db8'
PRELOADER_SHA = 'e7e8bf984f3f107723f29ab0a4f7e7e497d3e62d99a02824e382f664a38bb534'


def require(ok, message):
    if not ok: raise ValueError(message)


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
def scoped(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), 'Workspace-only evidence/source path required')
    return path
def record(path): return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


class OwnedProcess:
    """Hold one real Windows process handle, preventing PID-reuse ambiguity."""
    def __init__(self, pid, role):
        require(os.name == 'nt' and type(pid) is int and pid > 0, 'Native Windows owned PID required')
        self.pid, self.role, self.handle = pid, role, None
        k = self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]; k.OpenProcess.restype = wintypes.HANDLE
        k.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4; k.GetProcessTimes.restype = wintypes.BOOL
        k.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]; k.QueryFullProcessImageNameW.restype = wintypes.BOOL
        k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]; k.WaitForSingleObject.restype = wintypes.DWORD
        k.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]; k.GetExitCodeProcess.restype = wintypes.BOOL
        k.CloseHandle.argtypes = [wintypes.HANDLE]; k.CloseHandle.restype = wintypes.BOOL
        # Synchronize + query only; never write/terminate/change a process.
        self.handle = k.OpenProcess(0x100000 | 0x1000, False, pid)
        if not self.handle: raise ctypes.WinError(ctypes.get_last_error())
        try:
            values = [wintypes.FILETIME() for _ in range(4)]
            if not k.GetProcessTimes(self.handle, *(ctypes.byref(v) for v in values)): raise ctypes.WinError(ctypes.get_last_error())
            self.creation_ticks = (int(values[0].dwHighDateTime) << 32) | int(values[0].dwLowDateTime)
            self.creation_unix = self.creation_ticks / 1e7 - 11644473600
            length = wintypes.DWORD(32768); image = ctypes.create_unicode_buffer(length.value)
            if not k.QueryFullProcessImageNameW(self.handle, 0, image, ctypes.byref(length)): raise ctypes.WinError(ctypes.get_last_error())
            self.image = Path(image.value).resolve()
            require(self.image.name.lower() in ('python.exe', 'pythonw.exe'), 'Owned Python image required')
            self.image_sha256 = sha(self.image)
            class Basic(ctypes.Structure):
                _fields_ = [('exit_status', ctypes.c_long), ('peb', ctypes.c_void_p), ('affinity', ctypes.c_size_t),
                    ('base_priority', ctypes.c_long), ('unique_pid', ctypes.c_size_t), ('parent_pid', ctypes.c_size_t)]
            query = ctypes.WinDLL('ntdll').NtQueryInformationProcess
            query.argtypes = [wintypes.HANDLE, wintypes.ULONG, ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG)]
            query.restype = ctypes.c_long
            info, size = Basic(), wintypes.ULONG()
            status = query(self.handle, 0, ctypes.byref(info), ctypes.sizeof(info), ctypes.byref(size))
            require(status >= 0 and int(info.unique_pid) == pid, 'Owned-parent identity query failed')
            self.parent_pid = int(info.parent_pid)
        except BaseException:
            self.close(); raise

    def poll(self):
        value = self.kernel.WaitForSingleObject(self.handle, 0)
        if value == 258: return None
        require(value == 0, 'Owned process wait failed')
        code = wintypes.DWORD()
        if not self.kernel.GetExitCodeProcess(self.handle, ctypes.byref(code)): raise ctypes.WinError(ctypes.get_last_error())
        return int(code.value)

    def evidence(self):
        return {'role': self.role, 'pid': self.pid, 'parent_pid': self.parent_pid,
            'image_path': str(self.image), 'image_sha256': self.image_sha256,
            'creation_filetime_100ns': self.creation_ticks, 'creation_unix_seconds': self.creation_unix,
            'exit_code_observed_from_retained_handle': self.poll(), 'observed_at_utc': now(),
            'API': 'OpenProcess/GetProcessTimes/QueryFullProcessImageNameW/NtQueryInformationProcess/WaitForSingleObject/GetExitCodeProcess'}

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle); self.handle = None


def source_state(folder, run_id):
    state = read(folder / 'loop_status.json', {})
    require(state.get('kind') == 'HF_same_run_exact_science_continuation_v1' and state.get('run_id') == run_id,
        'Pinned same-run continuation state required')
    require(state.get('controller_sha256') == HELPER_SHA and sha(HELPER) == HELPER_SHA,
        'Frozen continuation controller changed')
    require(state.get('new_preloader', {}).get('sha256') == PRELOADER_SHA
        and sha(HERE / 'launch_same_science_resume_v1.py') == PRELOADER_SHA, 'Frozen preloader changed')
    for name, digest in state['runtime_source_sha256'].items():
        require(Path(name).name == name and sha(HERE / name) == digest, 'Frozen runtime changed: ' + name)
    require(state.get('scientific_arguments_unchanged') is True and state.get('optimizer_sampler_RNG_counters_reset') is False,
        'Same-science exact state preservation required')
    return state


def attach_window(folder, run_id, *, expected_launcher=None, expected_worker=None, process_factory=OwnedProcess, owned_child=None):
    state = source_state(folder, run_id)
    require(state.get('status') == 'running', 'Attach while actual declared source processes are live')
    identity_path = scoped(state['paths']['identity']); identity = read(identity_path, {})
    launcher_pid, worker_pid = state.get('owned_launcher_pid'), identity.get('pid')
    require(expected_launcher is None or launcher_pid == expected_launcher, 'Declared launcher PID differs')
    require(expected_worker is None or worker_pid == expected_worker, 'Declared worker PID differs')
    require(identity.get('expected_run_id') == run_id and identity.get('frozen_worker_sha256') == state['runtime_source_sha256']['train_lora_batched_data_v11.py'],
        'Actual worker identity differs')
    creation = identity.get('process_creation_identity', {})
    require(creation.get('available') is True and creation.get('pid') == worker_pid, 'Actual worker creation identity required')
    handles = []
    try:
        launcher = process_factory(launcher_pid, 'owned_native_launcher'); handles.append(launcher)
        worker = launcher if worker_pid == launcher_pid else process_factory(worker_pid, 'actual_training_worker')
        if worker is not launcher: handles.append(worker)
        require(worker.creation_ticks == creation['creation_filetime_100ns'], 'Actual worker handle/PID creation mismatch')
        require(worker is launcher or (worker.parent_pid == launcher_pid and identity.get('ppid') == launcher_pid), 'Actual worker is not owned launcher child')
        controller = process_factory(launcher.parent_pid, 'owned_same_science_controller'); handles.append(controller)
        require(controller.pid not in (launcher_pid, worker_pid) and controller.creation_ticks <= launcher.creation_ticks,
            'Derived controller parent is inconsistent')
        require(owned_child is None or controller.pid == owned_child.pid or controller.parent_pid == owned_child.pid,
            'Derived controller is not the supervisor-owned successor child')
        require(all(handle.poll() is None for handle in handles), 'All ownership handles must attach while live')
        return handles, {'kind': 'actual_owned_same_science_window_attachment_v1', 'run_id': run_id,
            'source_loop': str(folder), 'source_state_at_attachment': record(folder / 'loop_status.json'),
            'worker_identity_at_attachment': record(identity_path), 'processes': [h.evidence() for h in handles],
            'controller_parent_derived_only_from_owned_launcher': True, 'process_enumeration_or_command_lines_used': False,
            'attached_at_utc': now()}
    except BaseException:
        for handle in handles: handle.close()
        raise


def closed_window(folder, run_id, handles):
    exits = [handle.evidence() for handle in handles]
    require(all(value['exit_code_observed_from_retained_handle'] == 0 for value in exits), 'Real owned worker/launcher/controller exits must all be0')
    state = source_state(folder, run_id)
    require(state.get('status') == 'train_window_ended' and state.get('child_exit_code') == 0
        and state.get('resource_review_required') is False, 'Clean final helper closure required; resource/failed windows never auto-repeat')
    paths = {key: scoped(value) for key, value in state['paths'].items()}
    ready = read(paths['fresh_runtime'], {}); completion_path = paths['fresh_runtime'].with_suffix('.completion.json')
    completion = read(completion_path, {})
    worker_pid = read(paths['identity'], {})['pid']
    require(ready.get('status') == 'ready' and ready.get('run_id') == run_id and ready.get('pid') == worker_pid,
        'Fresh actual worker preload proof differs')
    require(completion.get('kind') == 'HF_same_science_native_preload_resume_completion_v1'
        and completion.get('status') == 'complete' and completion.get('run_id') == run_id and completion.get('pid') == worker_pid
        and completion.get('sources_unchanged') is True and completion.get('fresh_runtime_proof_sha256') == sha(paths['fresh_runtime'])
        and completion.get('launcher_sha256_after') == PRELOADER_SHA, 'Frozen fresh runtime completion differs')
    for key in ('io', 'cache', 'memory'):
        require(read(paths[key], {}).get('status') == 'complete', 'Closed runtime evidence missing: ' + key)
    selection_path = HERE / 'runs' / run_id / 'selection.json'; selection = read(selection_path, {})
    require(state.get('selection') == record(selection_path) and selection.get('run_id') == run_id
        and selection.get('completed_steps') == state.get('completed_steps') and selection.get('accepted') is True,
        'Saved final DEV selection differs')
    reason, helper_reason = selection.get('stop_early_reason'), state.get('cooperative_stop_reason')
    budget = reason == 'soft_max_wall_seconds' or (reason == 'controller_stop_file' and helper_reason == 'window_budget_checkpoint_boundary')
    return {'kind': 'real_handle_verified_same_science_window_closure_v1', 'run_id': run_id, 'source_loop': str(folder),
        'processes': exits, 'helper_final_state': record(folder / 'loop_status.json'), 'selection': record(selection_path),
        'fresh_runtime_completion': record(completion_path), 'completed_steps': selection['completed_steps'],
        'selected_step': selection['selected_step'], 'selection_stop_reason': reason, 'helper_stop_reason': helper_reason,
        'automatic_budget_handoff_allowed': budget, 'objective_complete': False, 'verified_at_utc': now()}


def successor_command(folder, target, run_id, hours):
    require(0 < hours <= 6 and sha(HELPER) == HELPER_SHA and not target.exists(), 'New bounded pinned successor required')
    return [str(ROOT / 'work/nemo-native-env-v2/Scripts/python.exe'), str(HELPER), '--source-loop', str(folder),
        '--run-id', run_id, '--output-dir', str(target), '--window-hours', str(hours),
        '--source-closed-confirmed', '--gpu-window-authorized']


def main(args):
    folder, evidence = scoped(args.source_loop), scoped(args.supervisor_dir)
    require(evidence != folder and not evidence.exists(), 'NEW supervisor namespace required')
    require(.25 <= args.max_hours <= 24 and .25 <= args.window_hours <= 6 and 5 <= args.poll_seconds <= 15, 'Bounded supervisor/window/poll settings required')
    require(args.successor_prefix and Path(args.successor_prefix).name == args.successor_prefix, 'Safe successor folder prefix required')
    config = read(HERE / 'runs' / args.run_id / 'run_config.json', {})
    declared_cap = config['config_identity']['training_arguments']['steps']
    horizon = config.get('steps')
    prior_selection = read(HERE / 'runs' / args.run_id / 'selection.json', {})
    require(type(horizon) is int and horizon > 0 and prior_selection.get('configured_steps') == horizon
        and type(declared_cap) is int and declared_cap >= horizon,
        'Actual effective run-config/selection horizon must agree; original declared cap stays unchanged')
    handles, attachment = attach_window(folder, args.run_id, expected_launcher=args.launcher_pid, expected_worker=args.worker_pid)
    evidence.mkdir(parents=True, exist_ok=False)
    write(evidence / 'initial_owned_attachment.json', attachment)
    state = {'kind': 'automatic_same_science_window_supervisor_v1', 'status': 'attached', 'run_id': args.run_id,
        'supervisor_pid': os.getpid(), 'supervisor_source_sha256': sha(__file__), 'helper_sha256': HELPER_SHA,
        'preloader_sha256': PRELOADER_SHA, 'configured_optimizer_horizon_unchanged': horizon,
        'original_scientific_argument_step_cap_unchanged': declared_cap,
        'window_hours': args.window_hours, 'max_hours': args.max_hours, 'objective_complete': False,
        'source_loop': str(folder), 'windows': [], 'started_at_utc': now()}
    write(evidence / 'supervisor_status.json', state)
    if args.attach_only:
        for handle in handles: handle.close()
        print(json.dumps({'passed': True, 'actual_owned_handle_attachment': True, 'GPU_launch': False, 'receipt': str(evidence / 'initial_owned_attachment.json')}))
        return 0
    require(args.gpu_window_authorized, 'ROOT must explicitly authorize automatic exclusive successors')
    deadline = time.monotonic() + args.max_hours * 3600
    child = None; index = 2
    try:
        while True:
            while any(handle.poll() is None for handle in handles):
                state.update(status='waiting_owned_window_closure', current_source_loop=str(folder), updated_at_utc=now())
                write(evidence / 'supervisor_status.json', state)
                if time.monotonic() >= deadline:
                    state.update(status='supervisor_horizon_waiting_current_owned_window', next_experiment_required=True)
                    write(evidence / 'supervisor_status.json', state)
                    # Never launch another child after this bound. Existing helper
                    # retains its own finite deadline/resource guards.
                    return 0
                time.sleep(args.poll_seconds)
            if child is not None:
                require(child.wait(timeout=30) == 0, 'Supervisor-owned helper returned failure')
            closure = closed_window(folder, args.run_id, handles)
            write(evidence / ('window_' + str(index-1).zfill(2) + '_real_closure.json'), closure)
            state['windows'].append(closure)
            for handle in handles: handle.close()
            handles = []
            remaining = deadline - time.monotonic()
            if closure['completed_steps'] >= horizon or not closure['automatic_budget_handoff_allowed'] or remaining < 900:
                state.update(status='next_experiment_required', stop_reason='configured_horizon' if closure['completed_steps'] >= horizon
                    else closure['selection_stop_reason'] if not closure['automatic_budget_handoff_allowed'] else 'bounded_supervisor_wall_horizon',
                    objective_complete=False, finished_at_utc=now())
                write(evidence / 'supervisor_status.json', state)
                return 0
            target = folder.parent / (args.successor_prefix + '_' + str(index).zfill(2))
            command = successor_command(folder, target, args.run_id, min(args.window_hours, remaining / 3600))
            launch = {'source_real_closure': record(evidence / ('window_' + str(index-1).zfill(2) + '_real_closure.json')),
                'target': str(target), 'command': command, 'started_at_utc': now(), 'actual_launch_completed': False}
            with (evidence / ('successor_' + str(index).zfill(2) + '.log')).open('x', encoding='utf-8') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                launch.update(actual_launch_completed=True, owned_helper_Popen_pid=child.pid)
                state.update(status='successor_started', latest_launch=launch)
                write(evidence / 'supervisor_status.json', state)
                wait_started = time.monotonic()
                while not (target / 'loop_status.json').exists() or not read(target / 'loop_status.json', {}).get('paths'):
                    require(child.poll() is None and time.monotonic() - wait_started < 600, 'Successor failed before actual runtime state')
                    time.sleep(args.poll_seconds)
                while True:
                    pending = read(target / 'loop_status.json', {})
                    if pending.get('status') == 'running' and Path(pending['paths']['identity']).exists(): break
                    require(child.poll() is None and time.monotonic() - wait_started < 600, 'Successor failed before actual owned identity')
                    time.sleep(args.poll_seconds)
                handles, attachment = attach_window(target, args.run_id, owned_child=child)
                write(evidence / ('window_' + str(index).zfill(2) + '_actual_attachment.json'), attachment)
            folder = target; index += 1
    except BaseException as error:
        state.update(status='failed_no_automatic_retry', error=repr(error), objective_complete=False, finished_at_utc=now())
        write(evidence / 'supervisor_status.json', state)
        raise
    finally:
        for handle in handles: handle.close()
        # No force-termination or relaunch on errors. Already-owned helpers keep
        # their finite local deadlines/guards; status preserves their ownership.


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-loop', type=Path, required=True); p.add_argument('--run-id', required=True)
    p.add_argument('--supervisor-dir', type=Path, required=True)
    p.add_argument('--launcher-pid', type=int, required=True); p.add_argument('--worker-pid', type=int, required=True)
    p.add_argument('--successor-prefix', required=True); p.add_argument('--window-hours', type=float, default=4.5)
    p.add_argument('--max-hours', type=float, default=12.); p.add_argument('--poll-seconds', type=float, default=5.)
    p.add_argument('--attach-only', action='store_true'); p.add_argument('--gpu-window-authorized', action='store_true')
    return p


if __name__ == '__main__': sys.exit(main(parser().parse_args()))
