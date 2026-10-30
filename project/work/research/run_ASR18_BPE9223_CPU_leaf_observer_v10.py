"""Generic9223 weak/cleaner CPU windows with two genuine aggregate exits.

Full original source, labels, geometry, frontend and provisional row protocols.
This parent owns reference and frontend handles; per-update reports never
pretend that a still-live aggregate worker has exited.
"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fastmss_BPE9223_cpu_common_v10 as core
import fastmss_owned_process_closure_v2 as process
from asr18_owned_family_observation_v4 import (owned_family_sample,
    wait_successful_launcher_after_disappearance, terminate_known_family, wait_captured_dead)

KIND = 'ASR18_actual_BPE9223_recorded_room_CPU_window_v1'
PROPOSAL = 'ASR18_BPE9223_aggregate_CPU_window_configuration_v1'
OWNER = '/root/cpu_data_preparation'
PARENT_BYTES = 128_000_000
SHIM_BYTES = 32 * 1024**2
LEAF_BYTES = 1_800_000_000
INNER_BYTES = PARENT_BYTES + SHIM_BYTES + LEAF_BYTES
FAMILY_BYTES = INNER_BYTES + SHIM_BYTES
LOGICAL_BYTES = 16 * 1024**2
PHYSICAL_BYTES = 64 * 1024**2
ALLOCATION = 512 * 1024
MAXIMUM_FILES = 64
MAXIMUM_DIRS = 12


def private_output(path):
    path = core.scoped(path)
    core.require(path.is_relative_to(core.ROOT.resolve())
        and not path.is_relative_to(core.sink.CACHE.resolve()), 'Fresh private E output only')
    return path


def inventory(root):
    """Only this fresh fixture output; reject links and charge exFAT clusters."""
    stack = [root]; total = allocated = files = dirs = 0
    while stack:
        folder = stack.pop(); stat = folder.stat(follow_symlinks=False)
        core.require(not folder.is_symlink() and not getattr(stat, 'st_file_attributes', 0) & 1024,
                     'No private fixture reparse directories')
        dirs += 1; allocated += ALLOCATION
        with os.scandir(folder) as entries:
            for entry in entries:
                item = entry.stat(follow_symlinks=False)
                core.require(not entry.is_symlink() and not getattr(item, 'st_file_attributes', 0) & 1024,
                             'No private fixture reparse members')
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                else:
                    core.require(entry.is_file(follow_symlinks=False), 'Ordinary private fixture files only')
                    files += 1; total += item.st_size
                    allocated += ((item.st_size + ALLOCATION - 1) // ALLOCATION) * ALLOCATION
        core.require(files <= MAXIMUM_FILES and dirs <= MAXIMUM_DIRS and total <= LOGICAL_BYTES
            and allocated <= PHYSICAL_BYTES, 'Private fixture file/byte/directory reservation exceeded')
    return {'files': files, 'directories': dirs, 'logical_file_bytes': total,
            'cluster_rounded_file_and_directory_bytes': allocated, 'allocation_unit_bytes': ALLOCATION}


def run_leaf(stage, request, script, root, python_pin, guard_pin, parent_pin, guard, *, task_monitor):
    """Merged-role allowances; unchanged actual disappearance/closure protocol."""
    admissions = []
    for _ in range(3):
        admissions.append(core.admit(FAMILY_BYTES, guard)); time.sleep(.2)
    folder = root / (stage + '_process'); folder.mkdir(exist_ok=False)
    monitor = folder / 'guard'; report = root / stage / 'report.json'
    arguments = [python_pin['path'], '-B', guard_pin['path'], '--folder', str(monitor),
        '--seconds', '300', '--script', script['path'], '--', '--request', request['path'], '--output', str(report)]
    env = dict(os.environ); env['CUDA_VISIBLE_DEVICES'] = '-1'
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): env[key] = '1'
    peak = 0; samples = 0; transition = None; began = time.monotonic()
    with (folder / 'worker.log').open('xb') as stream:
        child = subprocess.Popen(arguments, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        launcher = process.identity(child.pid); known = {(launcher['pid'], launcher['create_time']): launcher}
        try:
            task_monitor.enter(child, launcher, known)
            while child.poll() is None:
                try:
                    own = guard()
                    core.require((folder / 'worker.log').stat().st_size <= 1024**2, 'Bounded private leaf log1MiB')
                    sample = owned_family_sample(child, launcher, known=known, guard=guard)
                    total = own['worker_private_memory']['private_bytes'] + sample['private_bytes_sum']
                    core.require(total <= INNER_BYTES, 'Merged128M controller+unchanged1.8GB leaf+inner32MiB shim allowance')
                    peak = max(peak, total); samples += 1
                except ProcessLookupError:
                    transition = wait_successful_launcher_after_disappearance(child, launcher, known,
                        guard=guard, whole_owned_private_allowance_bytes=INNER_BYTES,
                        controller_private_reserve_bytes=PARENT_BYTES)
                    break
                core.require(time.monotonic() - began < 305, 'Same finite leaf300s timeout; no retry')
                time.sleep(.2)
            code = child.wait(timeout=5)
            core.require(type(code) is int and code == 0, 'Actual aggregate child failed; evidence remains failed')
            retirement = wait_captured_dead(known, timeout=2)
        except BaseException:
            monitor.mkdir(exist_ok=True)
            if not (monitor / 'stop.json').exists(): core.write_new(monitor / 'stop.json', {'reason': 'private_window_parent_failure'})
            terminate_known_family(child, launcher, known)
            retirement = wait_captured_dead(known, timeout=5)
            core.write_new(folder / 'failed_retirement.json', {'passed': False,
                'normal_leaf_exit0_claimed': False, 'actual_launcher_exit_code': child.poll(), 'retirement': retirement})
            raise
    task_monitor.leave(child, known)
    report_pin = core.record(report, guard); closure = core.record(monitor / 'closure.json', guard)
    actual = process.observe(child, kind='ASR18_owned_CPU_process_completion_v1', owner=OWNER,
        report=report_pin, worker_source=script, parent_source=parent_pin, guard_source=guard_pin,
        guard_closure=closure, launcher_identity=launcher, parent_guard_observation=guard(),
        output=folder / 'actual_owned_process0.json', guard=guard)
    return child, {'report': report_pin, 'actual0': actual, 'guard_closure': closure,
        'fresh_whole_family_admission_samples': admissions, 'actual_inner_family_peak_bytes': peak,
        'actual_family_samples': samples, 'actual_captured_family_retirement': retirement,
        'launcher_exit_transition': transition, 'wall_seconds': time.monotonic() - began}

