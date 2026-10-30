"""Explicit lifetime-bounded orchestrator; original7 owns actual leaf execution.

Full original source, labels, geometry, frontend and provisional row protocols.
This parent owns reference and frontend handles; per-update reports never
pretend that a still-live aggregate worker has exited.
"""
import argparse
from copy import deepcopy
import gc
import inspect
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fastmss_BPE9223_cpu_common_v7 as core
import fastmss_owned_process_closure_v2 as process
from asr18_owned_family_observation_v4 import (owned_family_sample,
    wait_successful_launcher_after_disappearance, terminate_known_family, wait_captured_dead)
import produce_FastMSS_BPE9223_reference_CPU_v7 as reference
import fastmss_BPE9223_feature_protocol_v7 as public

import run_ASR18_BPE9223_CPU_window_v7 as legacy_leaf_executor
from asr18_BPE9223_resource_operational_compatibility_v1 import checked_compatibility

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


def checked_reference_reuse(failure_pin, proposal_pin, contract_pin, root, legacy_pin, compatibility, guard):
    """Read only the exact failed controller and its genuine completed reference."""
    from fastmss_owned_process_closure_v2 import dead
    stopped = core.read_bound(failure_pin, guard)
    core.require(core.same(failure_pin, compatibility['prior_CPU_reference_failure'])
        and stopped['kind'] == 'ASR18_operational_CPU_controller_resource_stop_after_reference0_TOOL1_v1'
        and stopped['actual_tool_exit_code'] == stopped['actual_exec_tool_exit_code'] == 1
        and stopped['all_captured_identities_dead'] is True
        and all(dead(item) for item in stopped['all_captured_identities'])
        and stopped['preserved_production_updates'] == 20
        and stopped['new_GPU_children'] == stopped['new_optimizer_updates'] == 0
        and stopped['frontend_process_started'] is False and stopped['feature_owner_created'] is False,
        'Only this genuine quiescent reference0/controller1 failure, never a whole CPU0')
    prior = core.read_bound(stopped['configuration'], guard)
    core.require(core.same(prior['CPU_controller_source'], legacy_pin)
        and core.same(prior['full_run_identity'], core.read_bound(compatibility['original_configuration'], guard)['full_run_identity']),
        'Original failed CPU7 and unchanged production identity')
    actual = stopped['actual_normal_reference0']; report = stopped['actual_reference_report']
    core.require(Path(actual['path']).resolve() == root/'reference_process/actual_owned_process0.json'
        and Path(report['path']).resolve() == root/'reference/report.json', 'Exact preserved reference paths in same p2')
    proof = core.read_bound(actual, guard)
    process.verify(proof, kind='ASR18_owned_CPU_process_completion_v1', owner=OWNER, report=report, guard=guard)
    core.require(core.same(proof['parent_source'], legacy_pin)
        and core.same(proof['worker_source'], core.record(reference.__file__, guard)), 'Actual original code7 leaf observer/reference source')
    outer = core.read_bound(stopped['owner_guard'], guard)
    core.require(outer['monitor_clean'] is True and outer['monitor_joined'] is True
        and outer['producer'] == stopped['producer'], 'Real failed outer owner joined and died')
    value = core.read_bound(report, guard)
    core.require(core.same(value['source_contract'], contract_pin) and value['passed'] is True,
        'Same complete reference source contract')
    allowed = {root/'numerical_request.json', root/'reference_request.json', root/'reference/report.json',
        root/'reference_process/actual_owned_process0.json', root/'reference_process/worker.log',
        root/'reference_process/guard/boot.json', root/'reference_process/guard/closure.json',
        root/'reference_process/guard/live.json'}
    allowed.update(Path(member['report']['path']).resolve() for member in value['updates'])
    core.require(all(path.is_relative_to(root) for path in allowed)
        and {path.resolve() for path in root.rglob('*') if path.is_file()} == allowed,
        'Only exact completed reference/request metadata; no frontend, final report or unrelated member')
    legacy_leaf_executor.inventory(root)
    for key in ('checkpoint_pointer', 'storage_state', 'durable_latest', 'journal', 'p2_plan'):
        core.require(core.same(core.record(stopped[key]['path'], guard), stopped[key]), 'Unchanged post-failure frontier: '+key)
    return {'report': report, 'actual0': actual, 'guard_closure': proof['guard_closure'],
        'reused_from_actual_failed_controller': failure_pin, 'new_reference_Popen_claimed': False}


def main(proposal_path, ROOT_review, output, operational_compatibility, resume_reference_failure=None):
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
    from asr18_BPE9223_source_context_v4 import verify_CPU_configuration, verify_CPU_prior_boundary
    reviewed_configuration = verify_CPU_configuration(proposal_pin, review_pin, guard=core.resource_guard)
    core.require(core.same(reviewed_configuration, review['configuration']), 'Exact reviewed direct configuration or authorized static template')
    root = legacy_leaf_executor.private_output(output)
    compatibility_pin = core.record(operational_compatibility)
    original = core.read_bound(core.read_bound(compatibility_pin)['original_configuration'])
    compatibility = checked_compatibility(compatibility_pin, original['full_run_identity'], guard=core.resource_guard)
    del original
    driver_pin = core.record(__file__); parent_pin = core.record(legacy_leaf_executor.__file__)
    core.require(core.same(compatibility['operational_sources']['CPU_orchestrator'], driver_pin)
        and core.same(compatibility['CPU_leaf_execution_delegate_source'], parent_pin)
        and Path(inspect.getsourcefile(legacy_leaf_executor.run_leaf)).resolve() == Path(parent_pin['path']),
        'Actual new main8 and explicitly delegated unchanged original7 Popen/observer')
    core.require(str(root) == proposal['output_root'] and (root.is_dir() if resume_reference_failure else not root.exists()),
        'Exact fresh private output, or separately authorized preserved reference-only continuation')
    sources = proposal['sources']
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
    if resume_reference_failure is None: root.mkdir(parents=True, exist_ok=False)
    def guard():
        core.require(time.monotonic() - began <= 480, 'Bounded whole CPU window480s')
        value = core.resource_guard()
        core.require(value['worker_private_memory']['private_bytes'] <= PARENT_BYTES, 'Unchanged64MiB parent reserve')
        return value
    contract_pin = proposal['source_contract']
    failure_pin = None if resume_reference_failure is None else core.record(resume_reference_failure, guard)
    reference_proof = None if failure_pin is None else checked_reference_reuse(
        failure_pin, proposal_pin, contract_pin, root, parent_pin, compatibility, guard)
    source, plan = public.verify_source_contract(contract_pin, guard=guard)
    numerical = core.read_bound(source['numerical_CPU_contract'], guard)
    rebuilt = core.bridge.bridge_plan(plan['coordinator_plan'], DATA_evidence=plan['DATA_evidence'],
        session_binding=plan['session'], window_index=plan['window_index'],
        prior_window_boundary=plan['prior_window_boundary'],
        verify_prior_boundary=lambda boundary: verify_CPU_prior_boundary(boundary, guard=guard), guard=guard)
    core.require(core.same(rebuilt, plan), 'Independent source/frontier/phase/full-row revalidation before any reference audio')
    summary = {'phase': plan['phase'], 'window_index': plan['window_index'],
        'row_counts': [len(update['rows']) for update in plan['updates']]}
    del rebuilt, plan
    gc.collect()
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
    numerical_value = {'kind': core.REQUEST_KIND, 'production_enabled': False,
        'bridge': source['bridge'], 'contract': source['numerical_CPU_contract'], 'update_index': 0}
    if reference_proof is None: core.write_new(numerical_request, numerical_value)
    else: core.require(core.same(core.read_bound(core.record(numerical_request, guard), guard), numerical_value),
        'Preserved numerical request unchanged; no overwrite')
    common_request = {'kind': public.REQUEST, 'mode': public.MODE, 'production_enabled': False,
        'freeze_after_setup': proposal['freeze_after_setup'], 'source_contract': contract_pin,
        'parent_source': parent_pin, 'bridge': source['bridge'], 'numerical_request': core.record(numerical_request, guard),
        'output_root': str(root), 'ROOT_review': review_pin, 'configuration': reviewed_configuration,
        'window_configuration': proposal_pin}
    target = root / 'reference_request.json'
    if reference_proof is None:
        core.write_new(target, common_request)
        _, reference_proof = legacy_leaf_executor.run_leaf('reference', core.record(target, guard), sources['reference'], root,
                                      python_pin, guard_pin, parent_pin, guard)
    else:
        core.require(core.same(core.read_bound(core.record(target, guard), guard), common_request),
            'Exact original source-bound reference request reused read only')
    legacy_leaf_executor.inventory(root)
    public.verify_reference_window(reference_proof['report'], reference_proof['actual0'], contract_pin=contract_pin, guard=guard)
    gc.collect()
    owner = public.create_window(contract_pin, window_id=proposal['window_id'], frontend=numerical['frontend'],
        guard=guard, verify_prior_boundary=lambda boundary: verify_CPU_prior_boundary(boundary, guard=guard))
    frontend_request = {**common_request, 'owner': owner, 'reference_window_report': reference_proof['report'],
                        'reference_window_actual0': reference_proof['actual0']}
    target = root / 'frontend_request.json'; core.write_new(target, frontend_request)
    frontend_child, frontend_proof = legacy_leaf_executor.run_leaf('frontend', core.record(target, guard), sources['frontend'], root,
                                             python_pin, guard_pin, parent_pin, guard)
    legacy_leaf_executor.inventory(root)
    closure = public.observe_feature_window(frontend_child, owner_pin=owner, actual_process0=frontend_proof['actual0'],
        report=frontend_proof['report'], output=root / 'actual_frontend_window_closure.json', guard=guard)
    manifest = public.finalize_window(owner, closure, guard=guard)
    reader = public.FeatureWindowReader(owner, manifest, guard=guard)
    frontend = core.read_bound(frontend_proof['report'], guard)
    total_rows = sum(summary['row_counts'])
    core.require(frontend['passed'] is True and frontend['all_updates_reported'] is True
        and frontend['full_rows'] == total_rows and frontend['individual_array_raw_hashes'] == 2 * total_rows
        and frontend['new_NPY_files_written'] == 2 * total_rows and frontend['native_features_ready'] is False,
        'Every full row and both original numeric arrays from the actual aggregate frontend0')
    rows, payloads, operations = [], [], []
    for index, row_count in enumerate(summary['row_counts']):
        item = core.read_bound(reader.update_record(index), guard)
        core.require(item['prepared_payload']['state'] == 'prepared_not_committed'
            and item['per_update_child0_claimed'] is False and len(item['rows']) == row_count,
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
    result = {'kind': KIND, 'passed': True, 'producer': core.identity(), 'worker_source': driver_pin,
        'legacy_leaf_execution_source': parent_pin, 'operational_compatibility': compatibility_pin,
        'reference_reuse_failure': failure_pin, 'new_aggregate_children_this_invocation': 2 if failure_pin is None else 1,
        'parent_source': outer_parent, 'proposal': proposal_pin, 'source_contract': contract_pin,
        'owner': owner, 'manifest': manifest, 'actual_frontend_window_closure': closure,
        'bridge': source['bridge'], 'phase': summary['phase'], 'window_index': summary['window_index'],
        'reference_actual_aggregate0': reference_proof, 'frontend_actual_aggregate0': frontend_proof,
        'real_actual_child0_count': 2, 'per_update_child0_count_claimed': 0,
        'complete_raw_array_hash_rows': rows, 'complete_ordered_prepared_payloads': payloads,
        'updates': len(summary['row_counts']), 'full_rows': total_rows, 'individual_array_raw_hashes': 2 * total_rows,
        'new_reference_worker_timings': core.read_bound(reference_proof['report'], guard)['timings'],
        'new_frontend_worker_timings': frontend['timings'], 'freeze_after_setup': proposal['freeze_after_setup'],
        'recorded_room_operations': operations, 'recorded_room_operation_counts': {kind: sum(v['kind'] == kind for v in operations) for kind in ('noise', 'rir')},
        'actual_NEW_feature_logical_bytes': logical, 'actual_NEW_feature_cluster_rounded_bytes': physical,
        'private_output_inventory': legacy_leaf_executor.inventory(root), 'native_features_ready': False,
        'actual_optimizer_updates': 0, 'sampler_journal_frontier_advanced': False,
        'production_enabled': False, 'native_benchmark_only': False,
        'owner_validation_only': source['owner_validation_only'],
        'checkpoint_diagnostic_only': source['checkpoint_diagnostic_only'],
        'production_source_eligible': source['production_source_eligible'],
        'actual_full_CPU_preparation_speedup_claimed': False, 'wall_seconds': time.monotonic() - began}
    core.write_new(root / 'report.json', result)
    legacy_leaf_executor.inventory(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True); parser.add_argument('--ROOT-review', required=True)
    parser.add_argument('--folder', required=True); parser.add_argument('--operational-compatibility', required=True)
    parser.add_argument('--resume-reference-failure')
    args = parser.parse_args()
    main(args.config, args.ROOT_review, args.folder, args.operational_compatibility, args.resume_reference_failure)
