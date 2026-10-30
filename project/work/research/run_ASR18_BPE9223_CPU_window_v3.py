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
import fastmss_BPE9223_cpu_common_v3 as core
import fastmss_owned_process_closure_v2 as process
from asr18_owned_family_observation_v4 import (owned_family_sample,
    wait_successful_launcher_after_disappearance, terminate_known_family, wait_captured_dead)
import produce_FastMSS_BPE9223_reference_CPU_v3 as reference
import fastmss_BPE9223_feature_protocol_v3 as public

KIND = 'ASR18_actual_BPE9223_recorded_room_CPU_window_v1'
PROPOSAL = 'ASR18_BPE9223_aggregate_CPU_window_configuration_v1'
OWNER = '/root/cpu_data_preparation'
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
    return child, {'report': report_pin, 'actual0': actual, 'guard_closure': closure,
        'fresh_whole_family_admission_samples': admissions, 'actual_inner_family_peak_bytes': peak,
        'actual_family_samples': samples, 'actual_captured_family_retirement': retirement,
        'launcher_exit_transition': transition, 'wall_seconds': time.monotonic() - began}


def main(proposal_path, ROOT_review, output):
    began = time.monotonic(); core.environment()
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    proposal_pin = core.record(proposal_path); proposal = core.read_bound(proposal_pin)
    review_pin = core.record(ROOT_review); review = core.read_bound(review_pin, core.resource_guard)
    core.require(proposal['kind'] == PROPOSAL and proposal['production_enabled'] is False
        and proposal['native_benchmark_only'] is False and proposal['vocabulary'] == 9223
        and review['accepted'] is True and review['reviewer'] == '/root'
        and review['scope'] == 'BPE9223_recorded_noise_16room_bounded_CPU_window_v1',
        'Exact ROOT-reviewed generic9223 bounded CPU configuration')
    from asr18_BPE9223_source_context_v2 import verify_CPU_configuration, verify_CPU_prior_boundary
    reviewed_configuration = verify_CPU_configuration(proposal_pin, review_pin, guard=core.resource_guard)
    core.require(core.same(reviewed_configuration, review['configuration']), 'Exact reviewed direct configuration or authorized static template')
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
        core.require(time.monotonic() - began <= 480, 'Bounded whole CPU window480s')
        value = core.resource_guard()
        core.require(value['worker_private_memory']['private_bytes'] <= PARENT_BYTES, 'Unchanged64MiB parent reserve')
        return value
    contract_pin = proposal['source_contract']
    source, plan = public.verify_source_contract(contract_pin, guard=guard)
    numerical = core.read_bound(source['numerical_CPU_contract'], guard)
    rebuilt = core.bridge.bridge_plan(plan['coordinator_plan'], DATA_evidence=plan['DATA_evidence'],
        session_binding=plan['session'], window_index=plan['window_index'],
        prior_window_boundary=plan['prior_window_boundary'],
        verify_prior_boundary=lambda boundary: verify_CPU_prior_boundary(boundary, guard=guard), guard=guard)
    core.require(core.same(rebuilt, plan), 'Independent source/frontier/phase/full-row revalidation before any reference audio')
    core.require(core.same(source['sources']['controller'], parent_pin)
        and core.same(source['sources']['reference'], sources['reference'])
        and core.same(source['sources']['frontend'], sources['frontend'])
        and core.same(source['sources']['core'], sources['core'])
        and core.same(source['owner_validation_only'], proposal['owner_validation_only'])
        and core.same(source['checkpoint_diagnostic_only'], proposal['checkpoint_diagnostic_only'])
        and source['freeze_after_setup'] == proposal['freeze_after_setup'], 'Exact reviewed source contract')
    outer_parent = proposal['outer_parent_source']
    core.require(core.same(outer_parent, numerical['outer_parent_source'])
        and core.same(core.record(outer_parent['path'], guard), outer_parent), 'Actual external native-owner source binding')
    numerical_request = root / 'numerical_request.json'
    core.write_new(numerical_request, {'kind': core.REQUEST_KIND, 'production_enabled': False,
        'bridge': source['bridge'], 'contract': source['numerical_CPU_contract'], 'update_index': 0})
    common_request = {'kind': public.REQUEST, 'mode': public.MODE, 'production_enabled': False,
        'freeze_after_setup': proposal['freeze_after_setup'], 'source_contract': contract_pin,
        'parent_source': parent_pin, 'bridge': source['bridge'], 'numerical_request': core.record(numerical_request, guard),
        'output_root': str(root), 'ROOT_review': review_pin, 'configuration': reviewed_configuration,
        'window_configuration': proposal_pin}
    target = root / 'reference_request.json'; core.write_new(target, common_request)
    _, reference_proof = run_leaf('reference', core.record(target, guard), sources['reference'], root,
                                  python_pin, guard_pin, parent_pin, guard)
    inventory(root)
    public.verify_reference_window(reference_proof['report'], reference_proof['actual0'], contract_pin=contract_pin, guard=guard)
    owner = public.create_window(contract_pin, window_id=proposal['window_id'], frontend=numerical['frontend'],
        guard=guard, verify_prior_boundary=lambda boundary: verify_CPU_prior_boundary(boundary, guard=guard))
    frontend_request = {**common_request, 'owner': owner, 'reference_window_report': reference_proof['report'],
                        'reference_window_actual0': reference_proof['actual0']}
    target = root / 'frontend_request.json'; core.write_new(target, frontend_request)
    frontend_child, frontend_proof = run_leaf('frontend', core.record(target, guard), sources['frontend'], root,
                                             python_pin, guard_pin, parent_pin, guard)
    inventory(root)
    closure = public.observe_feature_window(frontend_child, owner_pin=owner, actual_process0=frontend_proof['actual0'],
        report=frontend_proof['report'], output=root / 'actual_frontend_window_closure.json', guard=guard)
    manifest = public.finalize_window(owner, closure, guard=guard)
    reader = public.FeatureWindowReader(owner, manifest, guard=guard)
    frontend = core.read_bound(frontend_proof['report'], guard)
    total_rows = sum(len(update['rows']) for update in plan['updates'])
    core.require(frontend['passed'] is True and frontend['all_updates_reported'] is True
        and frontend['full_rows'] == total_rows and frontend['individual_array_raw_hashes'] == 2 * total_rows
        and frontend['new_NPY_files_written'] == 2 * total_rows and frontend['native_features_ready'] is False,
        'Every full row and both original numeric arrays from the actual aggregate frontend0')
    rows, payloads, operations = [], [], []
    for index, update in enumerate(plan['updates']):
        item = core.read_bound(reader.update_record(index), guard)
        core.require(item['prepared_payload']['state'] == 'prepared_not_committed'
            and item['per_update_child0_claimed'] is False and len(item['rows']) == len(update['rows']),
            'All full copied provisional rows; no fake per-update child0')
        rows.extend({'update_index': index, 'row_index': j, **row} for j, row in enumerate(item['rows']))
        payloads.append(item['prepared_payload'])
        operations.extend({'draw': event['draw'], **operation}
            for event in item['prepared_payload']['rows'] for operation in event['augmentation'].get('operations', []))
    new_owner, new_folder, _ = public.owner_folder(owner, guard)
    store = public.engine._Store(new_folder, new_owner, guard, public.capacity_lock)
    logical = store.inventory(); actual_files = list(new_folder.iterdir())
    physical = ALLOCATION + sum(((p.stat().st_size + ALLOCATION - 1) // ALLOCATION) * ALLOCATION for p in actual_files)
    core.require(logical <= public.CAPACITY and len(actual_files) == len(new_owner['owned_names'])
        and physical <= 512 * 1024**2, 'Exact complete owned logical and cluster-rounded physical512MiB reservation')
    result = {'kind': KIND, 'passed': True, 'producer': core.identity(), 'worker_source': parent_pin,
        'parent_source': outer_parent, 'proposal': proposal_pin, 'source_contract': contract_pin,
        'owner': owner, 'manifest': manifest, 'actual_frontend_window_closure': closure,
        'bridge': source['bridge'], 'phase': plan['phase'], 'window_index': plan['window_index'],
        'reference_actual_aggregate0': reference_proof, 'frontend_actual_aggregate0': frontend_proof,
        'real_actual_child0_count': 2, 'per_update_child0_count_claimed': 0,
        'complete_raw_array_hash_rows': rows, 'complete_ordered_prepared_payloads': payloads,
        'updates': len(plan['updates']), 'full_rows': total_rows, 'individual_array_raw_hashes': 2 * total_rows,
        'new_reference_worker_timings': core.read_bound(reference_proof['report'], guard)['timings'],
        'new_frontend_worker_timings': frontend['timings'], 'freeze_after_setup': proposal['freeze_after_setup'],
        'recorded_room_operations': operations, 'recorded_room_operation_counts': {kind: sum(v['kind'] == kind for v in operations) for kind in ('noise', 'rir')},
        'actual_NEW_feature_logical_bytes': logical, 'actual_NEW_feature_cluster_rounded_bytes': physical,
        'private_output_inventory': inventory(root), 'native_features_ready': False,
        'actual_optimizer_updates': 0, 'sampler_journal_frontier_advanced': False,
        'production_enabled': False, 'native_benchmark_only': False,
        'owner_validation_only': source['owner_validation_only'],
        'checkpoint_diagnostic_only': source['checkpoint_diagnostic_only'],
        'production_source_eligible': source['production_source_eligible'],
        'actual_full_CPU_preparation_speedup_claimed': False, 'wall_seconds': time.monotonic() - began}
    core.write_new(root / 'report.json', result)
    inventory(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True); parser.add_argument('--ROOT-review', required=True)
    parser.add_argument('--folder', required=True)
    args = parser.parse_args(); main(args.config, args.ROOT_review, args.folder)
