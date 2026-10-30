"""Preserve prior caller; bind outer-family/race remedies and phase timing."""
import datetime, sys
from pathlib import Path
P = Path(__file__).resolve().parents[2]; S = P/'outputs/parakeet_turkish'; R = P/'work/research'
sys.path.insert(0, str(S)); sys.path.insert(0, str(R))
from atomic_io_retry_v1 import read_json, write_json
from bpe_broad_audio_v1 import record, require
from seal_ASR18_DATA_and_original_schedule_preparation_v1 import actual
from seal_ASR18_concrete_multiwindow_caller_v1 import owned_closure
A = Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')


def main():
    previous_path = A/'concrete_multiwindow_caller_source_review_manifest_v2.json'; previous = read_json(previous_path)
    for pin in previous['bindings']: require(record(pin['path']) == pin, 'Frozen predecessor changed')
    proof = actual(A/'owned_family_exit_CPU_actual_01', A/'owned_family_exit_CPU_guard_01', '89fc54')
    require(read_json(A/'owned_family_exit_CPU_actual_01/report.json')['check_count'] == 12, 'Actual twelve real CPU Popen controls')
    replacements = {str(R/'run_ASR18_full_DATA_multiwindow_v2.py'): R/'run_ASR18_full_DATA_multiwindow_v3.py',
        str(S/'asr18_full_DATA_native_window_v1.py'): S/'asr18_full_DATA_native_window_v2.py',
        str(R/'prepare_ASR18_full_DATA_multiwindow_configuration_v1.py'): R/'prepare_ASR18_full_DATA_multiwindow_configuration_v2.py'}
    roots = [replacements.get(pin['path'], Path(pin['path'])) for pin in previous['direct_sources']]
    roots.append(S/'asr18_owned_family_observation_v1.py')
    template = read_json(previous['configuration_template']['path'])
    template.update(parent_driver_source=record(R/'run_ASR18_full_DATA_multiwindow_v3.py'),
        worker_driver_source=str(R/'run_ASR18_full_DATA_multiwindow_v3.py'),
        CPU_inner_owned_allowance_bytes=None, CPU_outer_controller_launcher_reserve_bytes=None)
    new_template = S/'asr18_full_DATA_multiwindow.disabled.v2.json'; require(not new_template.exists(), 'Preserve disabled config successor'); write_json(new_template, template)
    bindings = owned_closure(roots)
    by_path = {pin['path']: pin for pin in bindings}
    for path in (S/'native_tdt_minimal_import_manifest_v1.json', S/'fastmss_broad_optin_v1.json', new_template): by_path[str(path)] = record(path)
    value = {**previous, 'kind': 'ASR18_concrete_resident_first_full_DATA_multiwindow_caller_source_review_manifest_v3',
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'previous_preserved_source_bundle': record(previous_path), 'bindings': [by_path[key] for key in sorted(by_path)],
        'direct_sources': [record(path) for path in roots], 'configuration_template': record(new_template),
        'configuration_assembler': record(R/'prepare_ASR18_full_DATA_multiwindow_configuration_v2.py'),
        'actual_CPU_Popen_family_and_reaped_exit_controls': proof,
        'new_parent_v3_engine_v2_AST_only_not_GPU_executed': True,
        'source_review_findings_in_predecessor': [
            'Resident CPU allowance omitted outer controller venv launcher beyond frozen inner controller family',
            'Reaped native/launcher-pending transition could turn healthy exit into unavailable-telemetry failure'],
        'source_remedies': {'CPU_family': 'separate reviewed outer-launcher reserve + exact actual Popen ancestry private sum including outer shim',
            'exit_transition': 'only expected reaped native + exact successful source report permits <=10s fresh-system pending wait; still require actual Popen0',
            'failure_ownership': 'captured PID/create_time family cleanup; GPU abort also handles native live with launcher already closed',
            'timing': 'bound native/write/restore phase files, measured RTF/audio-seconds, CPU family/phase and GPU startup/restore timings'},
        'native_loss_optimizer_frontier_or_augmentation_math_changed': False,
        'actual_full_DATA_positive_or_two_window_native_lifecycle_tested': False, 'production_ready': False, 'production_enabled': False}
    target = A/'concrete_multiwindow_caller_source_review_manifest_v3.json'; require(not target.exists(), 'Preserve review manifest'); write_json(target, value)
    status = read_json(A/'production_preparation_status_v1.json')
    status.update(updated_at_utc=value['created_at_utc'], current_stage='GPU_idle_concrete_caller_outer_family_exit_race_remedies_CPU12_passed_ROOT_review_and_final_DATA_pending',
        concrete_multiwindow_source_review=record(target), configuration_assembler_source=value['configuration_assembler'],
        owned_family_exit_CPU_actual=proof, actual_GPU_running=False, actual_owned_GPU_PID=None,
        actual_owned_GPU_session=None, production_ready=False, production_enabled=False)
    write_json(A/'production_preparation_status_v1.json', status); write_json(A/'launch_readiness_v30.json', status)
    current = read_json(A/'memory_diagnosis_status.json'); current.update(updated_at_utc=value['created_at_utc'],
        current_stage=status['current_stage'], launch_readiness=record(A/'launch_readiness_v30.json'),
        launch_readiness_receipt=str(A/'launch_readiness_v30.json'), actual_GPU_running=False,
        actual_owned_GPU_PID=None, actual_owned_GPU_session=None, production_ready=False)
    write_json(A/'memory_diagnosis_status.json', current)
    print({'source_review': record(target), 'owned_bindings': len(value['bindings']), 'CPU12_actual': proof, 'GPU_running': False})


if __name__ == '__main__': main()
