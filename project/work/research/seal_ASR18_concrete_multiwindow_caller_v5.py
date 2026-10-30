"""Bind corrected CPU row/source/raw-lifetime owner bundle into NEW caller5."""
import datetime, sys
from pathlib import Path
P = Path(__file__).resolve().parents[2]; S = P/'outputs/parakeet_turkish'; R = P/'work/research'
sys.path.insert(0, str(S)); sys.path.insert(0, str(R))
from atomic_io_retry_v1 import read_json, write_json
from bpe_broad_audio_v1 import record, require
from seal_ASR18_concrete_multiwindow_caller_v1 import owned_closure
A = Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
B = A.parent/'BPE_weak_then_clean_v1'; F = A.parent/'FastMSS_integration_v1'


def main():
    lease_path = P/'work/gpu_training_owner.json'; lease = read_json(lease_path)
    require(lease['owner'] == '/root/memory_diagnosis' and lease['actual_GPU_running'] is False
            and lease.get('actual_owned_worker_pid') is None, 'Actual exclusive idle GPU, no launch')
    previous_path = A/'concrete_multiwindow_caller_source_review_manifest_v4.json'; previous = read_json(previous_path)
    for pin in previous['bindings']: require(record(pin['path']) == pin, 'Preserve reviewed caller4 and all its sources')
    old_CPU = read_json(previous['CPU_owner_bundle']['path'])
    fast_path = F/'FastMSS_bounded_raw_row_serial_CPU_source_review_manifest_v2.json'; fast = read_json(fast_path)
    for pin in fast['sources'].values(): require(record(pin['path']) == pin, 'Corrected frozen CPU source changed')
    for key in ('template', 'interface', 'actual_controls'): require(record(fast[key]['path']) == fast[key], 'CPU owner evidence changed')
    actual = read_json(fast['actual_controls']['path']); footer = read_json(actual['guard_closure']['path'])
    require(actual['owner'] == '/root/fastmss_integration' and actual['actual_native_exit_code'] == actual['actual_exec_tool_exit_code'] == 0
            and actual['check_count'] == 52 and actual['actual_tool_chunk_id'] == '285f00'
            and actual['real_DATA_audio_frontend_model_GPU_or_CACHE_proven'] is False
            and footer['monitor_clean'] is True and footer['monitor_joined'] is True
            and footer['observed']['monitor_error'] is None and footer['observed']['samples'] >= 3,
            'Actual bounded52 CPU controls only; no original audio/frontend claim')
    old_paths = {pin['path'] for pin in old_CPU['sources'].values()}
    replacements = {str(R/'run_ASR18_full_DATA_multiwindow_v4.py'): R/'run_ASR18_full_DATA_multiwindow_v5.py',
        str(R/'prepare_ASR18_full_DATA_multiwindow_configuration_v3.py'): R/'prepare_ASR18_full_DATA_multiwindow_configuration_v4.py'}
    roots = [replacements.get(pin['path'], Path(pin['path'])) for pin in previous['direct_sources'] if pin['path'] not in old_paths]
    roots += [Path(pin['path']) for pin in fast['sources'].values()]
    roots = list(dict.fromkeys(roots))
    template = read_json(previous['configuration_template']['path'])
    template.update(parent_driver_source=record(R/'run_ASR18_full_DATA_multiwindow_v5.py'),
        worker_driver_source=str(R/'run_ASR18_full_DATA_multiwindow_v5.py'),
        CPU_controller_source=fast['sources']['controller'], raw_lifetime_review=None,
        corrected_CPU_contract_kind='FastMSS_row_serial_CPU_preparation_contract_v2')
    template_path = S/'asr18_full_DATA_multiwindow.disabled.v4.json'
    require(not template_path.exists(), 'Preserve disabled successor template'); write_json(template_path, template)
    bindings = {pin['path']: pin for pin in owned_closure(roots)}
    for path in (S/'native_tdt_minimal_import_manifest_v1.json', S/'fastmss_broad_optin_v1.json', template_path, Path(fast['template']['path'])):
        bindings[str(path)] = record(path)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    value = {**previous, 'kind': 'ASR18_concrete_resident_first_full_DATA_multiwindow_caller_source_review_manifest_v5',
        'created_at_utc': now, 'previous_preserved_source_bundle': record(previous_path),
        'bindings': [bindings[key] for key in sorted(bindings)], 'direct_sources': [record(path) for path in roots],
        'configuration_template': record(template_path), 'configuration_assembler': record(R/'prepare_ASR18_full_DATA_multiwindow_configuration_v4.py'),
        'CPU_owner_bundle': record(fast_path), 'CPU_contract_template': fast['template'], 'actual_CPU_raw52_controls': fast['actual_controls'],
        'predecessor_whole_source_findings': record(A/'ROOT_concrete_multiwindow_caller4_source_read_findings_v1.json'),
        'predecessor_failure_cleanup_subset_review': record(F/'FastMSS_concrete_multiwindow_v4_final_delta_subset_read_review_v1.json'),
        'unchanged_GPU_engine': record(S/'asr18_full_DATA_native_window_v2.py'),
        'separate_preparation_only_joint_DP_physical_scalar_audit': record(A/'declared_joint_physical_geometry_source_review_manifest_v1.json'),
        'source_review_findings_in_predecessor': [
            'CPU normalized TR row omitted planner4 exact declared_frontend_geometry',
            'CPU source evidence lookup treated Alania four-digit shard as a project-relative path',
            'Raw128MiB/two-job fields lacked actual before-allocation decoder/transform/frontend reservations',
            'Inner CPU controller failure path did not observe captured descendants dead after outer launcher exit'],
        'source_remedies': {
            'row': 'Common2 reconstructs exact complete planner4 TR row including source-bound physical/valid/encoder geometry',
            'source': 'Dataset-canonical Parquet/TAR path plus actual decoder SHA agrees with source inventory and preserved index/receipt bindings',
            'raw': 'Source-declared reservation BEFORE allocation; actual channel/layout/frame gates; chunked canonical hashes; observed buffer release; explicit separate compressed/backend/spectral/workspace/private domains',
            'inner_failure': 'Controller2 retires captured actual PID/create_time family regardless launcher exit and observes identities dead',
            'ROOT_policy_binding': 'Assembler4 copies exact accepted raw_lifetime_review from ROOT full source/resource review; Parent5 rechecks exact CPU bundle/review/contract/config equality before any child',
            'preserved_native': 'Engine2, canonical loss, optimizer, sampler, coherent writer and resident/fallback algorithms remain unchanged'},
        'new_parent_v5_and_assembler_v4_AST_only_not_runtime_or_GPU_executed': True,
        'actual_original_CPU_reference_frontend_positive_this_bundle': False,
        'actual_full_DATA_positive_or_two_window_native_lifecycle_tested': False,
        'native_loss_optimizer_frontier_or_augmentation_math_changed': False,
        'production_ready': False, 'production_enabled': False}
    target = A/'concrete_multiwindow_caller_source_review_manifest_v5.json'; require(not target.exists(), 'Preserve source manifest history')
    write_json(target, value)
    carrier = read_json(B/'KIRAAT_FLAC_carrier_current_status_v1.json')
    state = read_json(A/'production_preparation_status_v1.json')
    state.update(updated_at_utc=now, current_stage='GPU_idle_corrected_CPU_bundle_and_caller5_frozen_complete_ROOT_review_full_DATA_and_resource_forecasts_pending',
        concrete_multiwindow_source_review=record(target), configuration_assembler_source=value['configuration_assembler'],
        corrected_CPU_owner_bundle=record(fast_path), actual_CPU_raw52_controls=fast['actual_controls'],
        current_caller4_whole_review_findings=value['predecessor_whole_source_findings'], current_caller4_CPU_positive_path_accepted=False,
        current_caller5_whole_source_review_accepted=False, actual_GPU_running=False, actual_owned_GPU_PID=None, actual_owned_GPU_session=None,
        DATA_carrier_snapshot={'at_utc': carrier['at_utc'], 'covered': carrier['cumulative_covered_sources'], 'required': 729},
        production_ready=False, production_enabled=False)
    readiness = A/'launch_readiness_v34.json'; require(not readiness.exists(), 'Preserve readiness history')
    write_json(A/'production_preparation_status_v1.json', state); write_json(readiness, state)
    current = read_json(A/'memory_diagnosis_status.json'); current.update(updated_at_utc=now, current_stage=state['current_stage'],
        launch_readiness=record(readiness), launch_readiness_receipt=str(readiness), actual_GPU_running=False,
        actual_owned_GPU_PID=None, actual_owned_GPU_session=None, production_ready=False)
    write_json(A/'memory_diagnosis_status.json', current)
    lease.update(updated_at_utc=now, diagnostic_phase='corrected_caller5_source_preparation_GPU_idle',
        current_source_preparation=record(target), actual_training_optimizer_running=False)
    write_json(lease_path, lease)
    print({'source_review': record(target), 'owned_bindings': len(value['bindings']),
        'parent5': record(R/'run_ASR18_full_DATA_multiwindow_v5.py'), 'assembler4': value['configuration_assembler'],
        'actual_CPU52': fast['actual_controls'], 'GPU_running': False, 'production_enabled': False})


if __name__ == '__main__': main()
