"""Freeze concrete caller sources and honest CPU-only evidence for ROOT review."""
import ast, datetime, json, sys
from pathlib import Path
P = Path(__file__).resolve().parents[2]; S = P/'outputs/parakeet_turkish'; R = P/'work/research'
sys.path.insert(0, str(S)); sys.path.insert(0, str(R))
from atomic_io_retry_v1 import read_json, write_json
from bpe_broad_audio_v1 import record, require
from seal_ASR18_DATA_and_original_schedule_preparation_v1 import actual
A = Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1'); B = A.parent/'BPE_weak_then_clean_v1'; F = A.parent/'FastMSS_integration_v1'


def owned_closure(roots):
    found = {}; pending = list(roots)
    while pending:
        path = Path(pending.pop()).resolve()
        if str(path) in found: continue
        found[str(path)] = record(path)
        if path.suffix != '.py': continue
        tree = ast.parse(path.read_text(encoding='utf-8-sig')); compile(tree, str(path), 'exec')
        for node in ast.walk(tree):
            names = [node.module] if isinstance(node, ast.ImportFrom) and node.module else [v.name for v in node.names] if isinstance(node, ast.Import) else []
            for name in names:
                for folder in (S, R):
                    child = folder/(name.split('.')[0]+'.py')
                    if child.exists(): pending.append(child); break
    return [found[key] for key in sorted(found)]


def main():
    lease = read_json(P/'work/gpu_training_owner.json')
    require(lease['owner'] == '/root/memory_diagnosis' and lease['actual_GPU_running'] is False and lease['actual_owned_worker_pid'] is None, 'Actual idle GPU, no new native work')
    proof = actual(A/'multiwindow_caller_primitives_CPU_actual_01', A/'multiwindow_caller_primitives_CPU_guard_01', 'ad5205')
    CPU = read_json(A/'multiwindow_caller_primitives_CPU_actual_01/report.json')
    for pin in CPU['bindings']: require(record(pin['path']) == pin, 'Executed CPU snapshot changed; preserve tested version')
    require(CPU['check_count'] == 27 and CPU['actual_ASR_or_GPU_updates'] == 0, 'Actual bounded CPU controls only')
    fast_path = F/'FastMSS_row_serial_CPU_controller_source_review_manifest_v1.json'; fast = read_json(fast_path)
    for pin in fast['sources'].values(): require(record(pin['path']) == pin, 'Frozen CPU owner source changed')
    config_path = S/'asr18_full_DATA_multiwindow.disabled.v1.json'; require(not config_path.exists(), 'Preserve config template')
    contract_template = read_json(F/'FastMSS_row_serial_CPU_contract_TEMPLATE_NOT_LAUNCHABLE_v1.json')
    previous = read_json(A/'geometry_forecast_CPU_actual_05/report.json')
    whole = CPU['whole_worker_proposal']
    config = {'kind': 'ASR18_finite_full_DATA_two_window_config_TEMPLATE_NOT_LAUNCHABLE_v1',
        'TEMPLATE_NOT_LAUNCHABLE': True, 'finite_diagnostic_execution_enabled': False, 'production_enabled': False, 'diagnostic_only': True,
        'maximum_windows': 2, 'maximum_total_native_updates': 40, 'maximum_updates_per_window': 20,
        'maximum_GPU_child_wall_seconds': 7200, 'maximum_CPU_phase_seconds': 3600,
        'source_manifest': None, 'session': None, 'CPU_owned_allowance_bytes': None,
        'CPU_preparation_contract_and_whole_family_allowance_require_separate_actual_review': True,
        'GPU_python': record(P/'work/nemo-native-env-v2/Scripts/python.exe'),
        'CPU_python': record(Path(r'E:\gpu_training_scratchpad\scratch\BPE_broad_CPU_preflight_env_v2\Scripts\python.exe')),
        'CPU_guard_source': record(R/'ASR18_bounded_CPU_phase_v3.py'),
        'CPU_controller_source': fast['sources']['controller'],
        'parent_driver_source': record(R/'run_ASR18_full_DATA_multiwindow_v2.py'),
        'worker_driver_source': str(R/'run_ASR18_full_DATA_multiwindow_v2.py'),
        'entire_worker_forecast': whole, 'forecast_calibration': previous['bindings'],
        'augmentation_identity': contract_template['augmentation_identity'],
        'planning_arguments': {'DATA_evidence': None,
            'snapshot_binding': record(A/'original_schedule_snapshot_CPU_actual_02/snapshot.json'),
            'snapshot_actual0': record(A/'original_schedule_snapshot_CPU_actual_02/actual_owned_CLOSED0.json'),
            'snapshot_report': record(A/'original_schedule_snapshot_CPU_actual_02/report.json'),
            'expected_original_sources': read_json(A/'original_schedule_snapshot_CPU_actual_02/snapshot.json')['source_records'],
            'replay_pool_binding': record(S/'runs/adaptive_20261014_TR_WorldSpeech_expanded_01/training_pool.json'),
            'cleaner_metadata_binding': record(A/'cleaner_metadata_actual_01/report.json')},
        'launch_requires': ['genuine final DATA/729carriers/declared-envelope and actual TOOL0 wrappers',
            'NEW exact immutable source/session/CPU resource config', 'complete ROOT source AND exact config review',
            'fresh full5,910,813,440B worker admission before GPU process', 'actual CPU preparation0 before native inputs']}
    write_json(config_path, config)
    direct = [R/'run_ASR18_full_DATA_multiwindow_v2.py', S/'asr18_full_DATA_native_window_v1.py',
        S/'bpe_final_DATA_schedule_context_v1.py', S/'bpe_final_DATA_window_planner_v4.py', S/'asr18_physical_feature_geometry_v2.py',
        S/'asr18_live_state_fingerprint_v1.py', S/'asr18_phase_private_monitor_v1.py', S/'asr18_resident_window_admission_v1.py',
        S/'asr18_calibrated_geometry_forecast_v6.py', S/'asr18_precomputed_worker_forecast_v4.py',
        R/'asr18_memory_diagnosis_v2.py', R/'ASR18_bounded_CPU_phase_v3.py', *[Path(v['path']) for v in fast['sources'].values()]]
    bindings = owned_closure(direct)
    # Data-driven/dynamic imports and non-Python policy manifests are explicit.
    extra = [S/'native_tdt_minimal_import_manifest_v1.json', S/'fastmss_broad_optin_v1.json', config_path]
    by_path = {pin['path']: pin for pin in bindings}
    for path in extra: by_path[str(path.resolve())] = record(path)
    bindings = [by_path[key] for key in sorted(by_path)]
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    value = {'kind': 'ASR18_concrete_resident_first_full_DATA_multiwindow_caller_source_review_manifest_v1',
        'owner': '/root/memory_diagnosis', 'created_at_utc': now,
        'requested_source_scope': 'source_read_review_for_future_bounded_full_DATA_two_window_diagnostic',
        'later_execution_review_scope_required': 'finite_full_DATA_two_window_diagnostic_v1',
        'exact_execution_configuration_binding_required': True,
        'direct_sources': [record(path) for path in direct], 'bindings': bindings,
        'configuration_template': record(config_path), 'CPU_owner_bundle': record(fast_path),
        'actual_27_CPU_state_forecast_controls': proof,
        'executed_parent_v1_snapshot_preserved': record(R/'run_ASR18_full_DATA_multiwindow_v1.py'),
        'new_parent_v2_changes_after_CPU_snapshot': ['dynamic next-window attestation includes actual new bridge/session',
            'E100GB reserve in live parent guard', 'exact final ROOT configuration review binding',
            'truthful process-vs-chain update counters and final state/pointer crosscheck', 'disabled-template rejection before any child'],
        'new_parent_v2_AST_only_not_runtime_executed': True,
        'architecture': {'original_sources_only': [64250,2000,62250], 'maximum_windows': 2, 'maximum_native_updates': 40,
            'resident_GPU_preferred': True, 'resident_state': 'source-bound synchronized no-hook coherent idle nonce plus <=2s parent/native witness',
            'resident_CPU_admission': 'fresh actual native private + separately reviewed whole CPU family allowance; global90/headroom2GB and whole limits unchanged',
            'fallback': 'actual owned GPU exit/restore only after failed combined admission or real final budget boundary',
            'frontiers': 'original schedule -> actual Adam -> fsynced journal -> streamed checkpoint -> saved-vs-live semantic check -> latest pointer',
            'restore': 'actual PRE-save bounded207/threeAdam/all4RNG fingerprint compared to restored live state',
            'window2_plan': 'actual checkpoint sampler; no virtual planning final state as authority',
            'features': 'row serial original decode twice for independent proof; no full14-PCM bank; 128MiB/twojob raw,512MiB features',
            'cleanup': 'separate actual0 CPU child, archive first, exact owned features only; intermediate checkpoints retain all arrays',
            'physical_frontend_frames': 'N//160+1; valid N//160; encoder ceil(valid/8); replay physical <=8*T+1',
            'pass_or_phase_barrier': 'bounded prefix stops before next pass; no automatic DEV barrier bypass or cleaner transition'},
        'proposed_native_private_bytes': 5_560_813_440, 'proposed_entire_GPU_worker_private_bytes': 5_910_813_440,
        'prospective_native40_or_resident_CPU_fit_proved': False,
        'actual_new_GPU_or_ASR_updates_this_bundle': 0, 'actual_original_CPU_reference_frontend_positive_this_bundle': False,
        'actual_full_DATA_positive_or_two_window_native_lifecycle_tested': False,
        'full_passes_or_quality_or_retention_improvement_claimed': False,
        'previous_diagnostic20_and21_remain_unqualified': True,
        'complete_final_DATA_runtime_config_and_caller_review_pending': True, 'production_enabled': False, 'production_ready': False}
    target = A/'concrete_multiwindow_caller_source_review_manifest_v1.json'; require(not target.exists(), 'Preserve source review bundle'); write_json(target, value)
    carrier = read_json(B/'KIRAAT_FLAC_carrier_current_status_v1.json')
    status = read_json(A/'production_preparation_status_v1.json'); status.update(updated_at_utc=now,
        phase='concrete_multiwindow_source_review_pending', current_stage='GPU_idle_concrete_resident_first_multiwindow_caller_and_CPU_controller_written_CPU27_passed_final_DATA_and_review_pending',
        actual_GPU_running=False, actual_owned_GPU_PID=None, actual_owned_GPU_session=None,
        concrete_multiwindow_source_review=record(target), multiwindow_CPU_primitives_actual=proof,
        actual_new_GPU_updates_this_bundle=0, next_native_forecast_approved=False,
        DATA_carrier_snapshot={'at_utc': carrier['at_utc'], 'covered': carrier['cumulative_covered_sources'], 'required':729},
        final_DATA_positive_path_exercised=False, production_ready=False, production_enabled=False)
    write_json(A/'production_preparation_status_v1.json', status); write_json(A/'launch_readiness_v28.json', status)
    current = read_json(A/'memory_diagnosis_status.json'); current.update(updated_at_utc=now,
        current_stage=status['current_stage'], launch_readiness=record(A/'launch_readiness_v28.json'),
        launch_readiness_receipt=str(A/'launch_readiness_v28.json'), actual_GPU_running=False,
        actual_owned_GPU_PID=None, actual_owned_GPU_session=None, production_ready=False)
    write_json(A/'memory_diagnosis_status.json', current)
    print(json.dumps({'source_review': record(target), 'owned_bindings':len(bindings), 'CPU27_actual':proof,
        'GPU_running':False,'production_ready':False}), flush=True)


if __name__ == '__main__': main()
