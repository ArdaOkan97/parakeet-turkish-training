"""Metadata-only exact session/config assembly; never launch CPU/GPU workers.

Requires genuinely complete DATA and an explicit ROOT-reviewed CPU-family
resource contract. Its output remains gated by a SECOND exact configuration
review before the native caller can execute. No scientific frontier advances.
"""
import argparse, copy, json, os, sys
from pathlib import Path
P = Path(__file__).resolve().parents[2]; S = P/'outputs/parakeet_turkish'; sys.path.insert(0, str(S))
from atomic_io_retry_v1 import read_json, write_json
from bpe_broad_audio_v1 import record, require
from bpe_broad_production_state_v1 import digest
from bpe_final_DATA_window_planner_v4 import prepare_first_plan
from fastmss_phase_aware_plan_bridge_v5 import read_bound, validate_DATA, PURPOSE, SESSION_KIND
from fastmss_row_serial_cpu_v6 import resource_guard, verify_contract, verify_raw_review, environment
A = Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1'); F = A.parent/'FastMSS_integration_v1'


def assemble(*, DATA_evidence, source_manifest, source_review, CPU_resource_review, prior_retirement, folder):
    environment(); guard = resource_guard
    require(folder.resolve().is_relative_to(A.resolve()) and not folder.exists(), 'Fresh E-only metadata configuration namespace')
    sources = read_bound(source_manifest, guard); review = read_bound(source_review, guard)
    require(review['accepted'] is True and review['scope'] == 'source_read_review_for_future_bounded_full_DATA_two_window_diagnostic'
            and review['production_approval_claimed'] is False and review['source_manifest'] == source_manifest,
            'Complete current caller/source read-review required before configuration assembly')
    for pin in sources['bindings']: require(record(pin['path']) == pin, 'Reviewed transitive source changed before assembly')
    own = record(__file__)
    require(own in sources['bindings'], 'Configuration assembler itself must be in the reviewed source closure')
    CPU = read_bound(CPU_resource_review, guard)
    require(CPU['kind'] == 'ASR18_reviewed_row_serial_CPU_resource_contract_v1' and CPU['reviewer'] == '/root'
            and CPU['accepted'] is True and CPU['scope'] == 'CPU_preparation_only'
            and CPU['production_enabled'] is False
            and CPU['CPU_source_manifest'] == sources['CPU_owner_bundle'], 'Independent exact ROOT CPU-resource scope required')
    template = read_bound(sources['configuration_template'], guard)
    require(template['TEMPLATE_NOT_LAUNCHABLE'] is True and template['finite_diagnostic_execution_enabled'] is False,
            'Preserved unarmed template required')
    require(CPU['outer_parent_source'] == template['parent_driver_source'], 'CPU allowance applies to another native parent source')
    raw_review = copy.deepcopy(CPU['raw_lifetime_review'])
    verify_raw_review(raw_review)
    resources = CPU['resource_review']
    require(resources['passed'] is True and resources['scope'] == 'CPU_preparation_only'
            and resources['native_GPU_fit_claimed'] is False and resources['maximum_simultaneous_children'] == 1,
            'CPU review cannot imply native GPU admission')
    numeric = ('controller_private_reserve_bytes', 'shim_private_reserve_bytes', 'reference_incremental_bytes',
        'frontend_incremental_bytes', 'outer_controller_launcher_reserve_bytes', 'reference_maximum_private_bytes', 'frontend_maximum_private_bytes', 'whole_owned_private_allowance_bytes')
    require(all(type(resources[key]) is int and 0 < resources[key] <= 2_000_000_000 for key in numeric)
            and resources['whole_owned_private_allowance_bytes'] >= resources['controller_private_reserve_bytes']
                +resources['shim_private_reserve_bytes']+max(resources['reference_maximum_private_bytes'], resources['frontend_maximum_private_bytes']),
            'No invented/shrunk CPU-family allowance or non-integer bounds')
    require(resources['whole_owned_private_allowance_bytes']+resources['outer_controller_launcher_reserve_bytes']<=2_000_000_000,
            'Entire CPU family, including outer launcher, must fit explicit combined allowance')
    from asr18_reprepared_array_equivalence_v1 import validate_retirement
    require(template['first_window_repreparation_comparison']['required_before_any_GPU_launch'] is True,
            'Full original first-window deterministic comparison required')
    validate_retirement(prior_retirement,
        prior_failure=template['first_window_repreparation_comparison']['prior_failed_trial'],guard=guard)
    validated = validate_DATA(DATA_evidence, guard)
    planning = {**template['planning_arguments'], 'DATA_evidence': DATA_evidence}
    prefix = prepare_first_plan(**planning, maximum_updates=template['maximum_updates_per_window'], guard=guard)
    require(prefix['plan']['initial_committed_updates'] == 0 and prefix['actual_optimizer_updates'] == 0
            and prefix['actual_authoritative_draws_consumed'] == 0 and validated['population_mean_U'] == prefix['validated_DATA']['population_mean_U'],
            'Exact final mean/source and zero actual updates required; no diagnostic checkpoint seed')
    # Only metadata is created, after all genuine DATA and review gates pass.
    folder.mkdir(exist_ok=False)
    first_plan_path = folder/'original_first_window_plan.json'; write_json(first_plan_path, prefix['plan'])
    contract = read_json(F/'FastMSS_row_serial_CPU_contract_TEMPLATE_NOT_LAUNCHABLE_v6.json')
    require(contract['kind'] == 'FastMSS_row_serial_CPU_preparation_contract_v6'
            and contract['TEMPLATE_NOT_LAUNCHABLE'] is True, 'Exact unarmed corrected CPU contract required')
    contract.update(TEMPLATE_NOT_LAUNCHABLE=False, DATA_evidence_sha256=digest(DATA_evidence),
        outer_parent_source=template['parent_driver_source'], resource_review=copy.deepcopy(resources),
        raw_lifetime_review=raw_review, actual_ROOT_CPU_resource_review=CPU_resource_review)
    contract_path = folder/'reviewed_CPU_preparation_contract.json'; write_json(contract_path, contract)
    session = {'kind': SESSION_KIND, 'purpose': PURPOSE, 'production_enabled': False, 'diagnostic_only': True,
        'maximum_windows': 2, 'maximum_native_updates': 40,
        'initial_optimizer_updates': 0, 'initial_sampler_sha256': prefix['plan']['initial_sampler_sha256'],
        'coordinator_source': record(S/'bpe_production_window_coordinator_v4.py'),
        'augmentation_identity': template['augmentation_identity'], 'scientific_source_state': planning['snapshot_binding'],
        'preserve_original_all207_threeAdam_allRNG_and_replay_lineage': True,
        'DATA_evidence_sha256': digest(DATA_evidence), 'CPU_preparation_contract': record(contract_path),
        'configuration_assembler_source': own, 'actual_optimizer_updates': 0, 'actual_authoritative_draws_consumed': 0}
    session_path = folder/'finite_two_window_session.json'; write_json(session_path, session)
    verify_contract(record(contract_path), session, guard=guard)
    config = {**template, 'kind': 'ASR18_finite_full_DATA_two_window_diagnostic_config_v1',
        'TEMPLATE_NOT_LAUNCHABLE': False, 'finite_diagnostic_execution_enabled': True,
        'source_manifest': source_manifest, 'session': record(session_path), 'planning_arguments': planning,
        'CPU_owned_allowance_bytes': resources['whole_owned_private_allowance_bytes']+resources['outer_controller_launcher_reserve_bytes'],
        'CPU_inner_owned_allowance_bytes':resources['whole_owned_private_allowance_bytes'],
        'CPU_outer_controller_launcher_reserve_bytes':resources['outer_controller_launcher_reserve_bytes'],
        'first_window_repreparation_comparison':{**template['first_window_repreparation_comparison'],
            'prior_owned_retirement_actual0':prior_retirement},
        'configuration_assembly_source': own, 'configuration_source_read_review': source_review,
        'actual_ROOT_CPU_resource_review': CPU_resource_review, 'raw_lifetime_review': copy.deepcopy(raw_review),
        'exact_ROOT_execution_configuration_review_still_required': True,
        'CPU_preparation_contract_and_whole_family_allowance_require_separate_actual_review': False}
    config_path = folder/'finite_two_window_configuration_pending_execution_review.json'; write_json(config_path, config)
    from fastmss_owned_process_closure_v1 import identity
    result = {'kind': 'ASR18_actual_full_DATA_finite_configuration_assembly_v1', 'passed': True,
        'producer': identity(), 'worker_source': own, 'parent_source': own,
        'configuration': record(config_path), 'session': record(session_path), 'CPU_contract': record(contract_path),
        'provisional_original_first_plan': record(first_plan_path), 'source_manifest': source_manifest,
        'source_review': source_review, 'CPU_resource_review': CPU_resource_review,
        'raw_lifetime_review': copy.deepcopy(raw_review),
        'final_rows': validated['retained_rows'], 'final_token_SUM': validated['total_target_tokens'],
        'final_population_mean_U': validated['population_mean_U'], 'actual_native_or_CPU_frontend_jobs_launched': 0,
        'actual_optimizer_updates': 0, 'actual_authoritative_draws_consumed': 0,
        'shared_CACHE_created': False, 'production_enabled': False,
        'exact_final_configuration_review_and_fresh_entire_worker_admission_still_required': True}
    write_json(folder/'report.json', result); print(json.dumps({'passed': True, 'configuration': result['configuration'], 'jobs_launched': 0}), flush=True)
    return result


def bound_argument(value):
    return json.loads(value) if value.lstrip().startswith('{') else read_json(value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('DATA-evidence', 'source-manifest', 'source-review', 'CPU-resource-review', 'prior-retirement', 'folder'): parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    assemble(DATA_evidence=bound_argument(args.DATA_evidence), source_manifest=bound_argument(args.source_manifest),
        source_review=bound_argument(args.source_review), CPU_resource_review=bound_argument(args.CPU_resource_review),
        prior_retirement=bound_argument(args.prior_retirement), folder=Path(args.folder).resolve())
