"""Private two-leaf finite-window CPU timing/byte validation; no cache writer.

The numerical contract remains the immutable CPU8 oracle contract. This NEW
parent owns two genuine aggregate child handles. No per-update normal-exit or
public prepared-owner proof is manufactured from their still-live reports.
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
import fastmss_row_serial_cpu_v8 as core
import fastmss_owned_process_closure_v2 as process
from asr18_owned_family_observation_v4 import (owned_family_sample,
    wait_successful_launcher_after_disappearance, terminate_known_family, wait_captured_dead)
import produce_FastMSS_reference_window_CPU_v1 as reference

KIND = 'ASR18_private_two_actual_CPU_leaf_full_window_hash_validation_v1'
PROPOSAL = 'ASR18_private_finite_CPU_window_hash_validation_proposal_v1'
OWNER = '/root/fastmss_integration'
PARENT_BYTES = 64 * 1024**2
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


def run_leaf(stage, request, script, root, python_pin, guard_pin, parent_pin, guard):
    """Same family caps and disappearance protocol; actual NEW source identity."""
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
            while child.poll() is None:
                try:
                    own = guard()
                    core.require((folder / 'worker.log').stat().st_size <= 1024**2, 'Bounded private leaf log1MiB')
                    sample = owned_family_sample(child, launcher, known=known, guard=guard)
                    total = own['worker_private_memory']['private_bytes'] + sample['private_bytes_sum']
                    core.require(total <= INNER_BYTES, 'Unchanged controller+one leaf+inner shim allowance')
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
    report_pin = core.record(report, guard); closure = core.record(monitor / 'closure.json', guard)
    actual = process.observe(child, kind='ASR18_owned_CPU_process_completion_v1', owner=OWNER,
        report=report_pin, worker_source=script, parent_source=parent_pin, guard_source=guard_pin,
        guard_closure=closure, launcher_identity=launcher, parent_guard_observation=guard(),
        output=folder / 'actual_owned_process0.json', guard=guard)
    return {'report': report_pin, 'actual0': actual, 'guard_closure': closure,
        'fresh_whole_family_admission_samples': admissions, 'actual_inner_family_peak_bytes': peak,
        'actual_family_samples': samples, 'actual_captured_family_retirement': retirement,
        'launcher_exit_transition': transition, 'wall_seconds': time.monotonic() - began}


def main(proposal_path, output):
    began = time.monotonic(); core.environment()
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    proposal_pin = core.record(proposal_path); proposal = core.read_bound(proposal_pin)
    core.require(proposal['kind'] == PROPOSAL and proposal['production_enabled'] is False
        and proposal['native_features_ready'] is False and proposal['GPU_or_model_job_allowed'] is False,
        'Exact inactive private proof scope')
    root = private_output(output)
    core.require(str(root) == proposal['output_root'] and not root.exists(), 'Exact fresh private output reservation')
    sources = proposal['sources']; parent_pin = core.record(__file__)
    core.require(core.same(sources['parent'], parent_pin)
        and core.same(sources['reference'], core.record(reference.__file__))
        and core.same(sources['core'], core.record(core.__file__))
        and type(proposal['freeze_after_setup']) is bool, 'Exact new finite sources and explicit GC mode')
    for pin in sources.values(): core.require(core.same(core.record(pin['path']), pin), 'Bound source differs')
    python_pin = proposal['python']; guard_pin = sources['guard']
    core.require(Path(python_pin['path']).resolve() == core.CPU_ENV / 'Scripts/python.exe'
        and core.same(core.record(python_pin['path']), python_pin)
        and core.same(guard_pin, core.record(PROJECT / 'work/research/ASR18_bounded_CPU_phase_v3.py')),
        'Exact unchanged E CPU interpreter and guard3')
    root.mkdir(parents=True, exist_ok=False)
    def guard():
        core.require(time.monotonic() - began <= 480, 'Bounded whole private window480s')
        value = core.resource_guard()
        core.require(value['worker_private_memory']['private_bytes'] <= PARENT_BYTES, 'Unchanged64MiB parent reserve')
        return value
    shared = core.read_bound(proposal['baseline_shared_features'], guard)
    core.require(shared['kind'] == 'ASR18_new_owned_immutable_first4_paired_features_v1'
        and shared['reference_and_frontend_children_all0'] is True, 'Actual complete original first4 feature oracle')
    for key in ('owner', 'manifest', 'actual_CPU_process0', 'CPU_report', 'configuration', 'source'):
        core.require(core.same(core.record(shared[key]['path'], guard), shared[key]), 'Pinned immutable baseline changed: ' + key)
    baseline = core.read_bound(shared['CPU_report'], guard)
    process.verify(core.read_bound(shared['actual_CPU_process0'], guard),
        kind='ASR18_owned_CPU_process_completion_v1', owner='/root/memory_diagnosis', report=shared['CPU_report'], guard=guard)
    entries = [core.record(Path(shared['CPU_report']['path']).parent / f'reference_request_{i:02}.json', guard) for i in range(4)]
    core.require(len(baseline['independent_reference_proofs']) == len(baseline['frontend_proofs']) == len(entries) == 4,
                 'Exactly all four actual original updates, no fresh baseline leaves')
    common_request = {'kind': reference.REQUEST, 'mode': reference.MODE,
        'production_enabled': False, 'native_features_ready': False, 'freeze_after_setup': proposal['freeze_after_setup'],
        'sources': deepcopy(sources), 'parent_source': parent_pin, 'baseline_CPU_report': shared['CPU_report'],
        'baseline_CPU_actual0': shared['actual_CPU_process0'], 'bridge': baseline['bridge'],
        'original_requests': entries, 'output_root': str(root)}
    target = root / 'reference_request.json'; core.write_new(target, common_request)
    reference_proof = run_leaf('reference', core.record(target, guard), sources['reference'], root,
                               python_pin, guard_pin, parent_pin, guard)
    inventory(root)  # Actual monitor joined before a complete private-file scan.
    frontend_request = {**common_request, 'reference_window_report': reference_proof['report'],
                        'reference_window_actual0': reference_proof['actual0']}
    target = root / 'frontend_request.json'; core.write_new(target, frontend_request)
    frontend_proof = run_leaf('frontend', core.record(target, guard), sources['frontend'], root,
                              python_pin, guard_pin, parent_pin, guard)
    inventory(root)
    frontend = core.read_bound(frontend_proof['report'], guard)
    core.require(frontend['passed'] is True and frontend['all_updates_reported'] is True
        and frontend['full_rows'] == 64 and frontend['individual_array_raw_hashes'] == 128
        and frontend['native_features_ready'] is False and frontend['new_NPY_files_written'] == 0,
        'All64 actual rows/128 actual hashes from one genuinely completed frontend leaf')
    rows = []; payloads = []
    for i, item in enumerate(frontend['updates']):
        value = core.read_bound(item['report'], guard)
        core.require(value['update_index'] == i and value['per_update_child0_claimed'] is False,
                     'Aggregate members cannot become per-update child0')
        rows.extend({'update_index': i, 'row_index': j, **row} for j, row in enumerate(value['rows']))
        payloads.append(value['prepared_payload'])
    core.require(len(rows) == 64 and len(payloads) == 4, 'Complete full ordered targets/contexts/copied payloads')
    baseline_seconds = {}
    for stage, key in (('reference', 'independent_reference_proofs'), ('frontend', 'frontend_proofs')):
        seconds = []
        for proof in baseline[key]:
            actual = core.read_bound(proof['actual0'], guard)
            seconds.append(core.read_bound(actual['guard_closure'], guard)['wall_seconds'])
        baseline_seconds[stage] = {'actual_original_four_child_guard_seconds': seconds, 'sum_seconds': sum(seconds)}
    result = {'kind': KIND, 'passed': True, 'producer': core.identity(), 'worker_source': parent_pin,
        'proposal': proposal_pin, 'baseline_shared_features': proposal['baseline_shared_features'],
        'reference_actual_aggregate0': reference_proof, 'frontend_actual_aggregate0': frontend_proof,
        'real_actual_child0_count': 2, 'per_update_child0_count_claimed': 0,
        'complete_raw_array_hash_rows': rows, 'complete_ordered_prepared_payloads': payloads,
        'baseline_stage_seconds_readonly_existing_closures': baseline_seconds,
        'new_reference_worker_timings': core.read_bound(reference_proof['report'], guard)['timings'],
        'new_frontend_worker_timings': frontend['timings'], 'freeze_after_setup': proposal['freeze_after_setup'],
        'guard_caps': {'leaf': LEAF_BYTES, 'parent': PARENT_BYTES, 'inner_family': INNER_BYTES,
                       'whole_family_with_outer_launcher': FAMILY_BYTES, 'raw_PCM': 128 * 1024**2, 'raw_jobs': 2},
        'private_inventory_before_final_report': inventory(root), 'wall_seconds': time.monotonic() - began,
        'original_arrays_and_shared_namespace_untouched': True, 'new_NPY_or_PCM_files_written': 0,
        'actual_optimizer_updates': 0, 'sampler_journal_frontier_advanced': False,
        'array_write_fsync_readback_inventory_omitted_from_private_timing': True,
        'actual_public_window_or_native_admission_claimed': False, 'actual_full_CPU_preparation_speedup_claimed': False,
        'production_enabled': False, 'actual_outer_process_exit_observed_in_child': False}
    core.require(not any(k == 'torch' or k.startswith(('torch.', 'numpy', 'transformers')) for k in sys.modules),
                 'Parent stays NumPy/Torch/frontend-free')
    core.write_new(root / 'report.json', result); guard()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--proposal', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); main(args.proposal, args.output)
