"""Preserve v3 and bind actual orphan-retirement CPU evidence and new sources."""
import datetime, sys
from pathlib import Path
P = Path(__file__).resolve().parents[2]; S = P/'outputs/parakeet_turkish'; R = P/'work/research'
sys.path.insert(0, str(S)); sys.path.insert(0, str(R))
from atomic_io_retry_v1 import read_json, write_json
from bpe_broad_audio_v1 import record, require
from seal_ASR18_DATA_and_original_schedule_preparation_v1 import actual
from seal_ASR18_concrete_multiwindow_caller_v1 import owned_closure
A = Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
B = A.parent/'BPE_weak_then_clean_v1'; F = A.parent/'FastMSS_integration_v1'


def main():
    lease = read_json(P/'work/gpu_training_owner.json')
    require(lease['owner'] == '/root/memory_diagnosis' and lease['actual_GPU_running'] is False
            and lease.get('actual_owned_worker_pid') is None, 'Actual idle GPU lease, no launch')
    previous_path = A/'concrete_multiwindow_caller_source_review_manifest_v3.json'; previous = read_json(previous_path)
    for pin in previous['bindings']: require(record(pin['path']) == pin, 'Preserve frozen predecessor')
    proof = actual(A/'owned_orphan_retirement_CPU_actual_01', A/'owned_orphan_retirement_CPU_guard_01', '19c52d')
    report = read_json(A/'owned_orphan_retirement_CPU_actual_01/report.json')
    require(report['check_count'] == 9 and report['captured_identity_retirement']['all_captured_owned_identities_observed_dead'] is True
            and report['normal_leaf_exit0_or_clean_leaf_footer_claimed'] is False, 'Actual failure-path retirement, no invented leaf0')
    replacements = {str(R/'run_ASR18_full_DATA_multiwindow_v3.py'): R/'run_ASR18_full_DATA_multiwindow_v4.py',
        str(S/'asr18_owned_family_observation_v1.py'): S/'asr18_owned_family_observation_v2.py',
        str(R/'prepare_ASR18_full_DATA_multiwindow_configuration_v2.py'): R/'prepare_ASR18_full_DATA_multiwindow_configuration_v3.py'}
    roots = [replacements.get(pin['path'], Path(pin['path'])) for pin in previous['direct_sources']]
    template = read_json(previous['configuration_template']['path'])
    template.update(parent_driver_source=record(R/'run_ASR18_full_DATA_multiwindow_v4.py'),
        worker_driver_source=str(R/'run_ASR18_full_DATA_multiwindow_v4.py'))
    template_path = S/'asr18_full_DATA_multiwindow.disabled.v3.json'
    require(not template_path.exists(), 'Preserve configuration successor'); write_json(template_path, template)
    bindings = {pin['path']: pin for pin in owned_closure(roots)}
    for path in (S/'native_tdt_minimal_import_manifest_v1.json', S/'fastmss_broad_optin_v1.json', template_path):
        bindings[str(path)] = record(path)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    value = {**previous, 'kind': 'ASR18_concrete_resident_first_full_DATA_multiwindow_caller_source_review_manifest_v4',
        'created_at_utc': now, 'previous_preserved_source_bundle': record(previous_path),
        'bindings': [bindings[key] for key in sorted(bindings)], 'direct_sources': [record(path) for path in roots],
        'configuration_template': record(template_path), 'configuration_assembler': record(R/'prepare_ASR18_full_DATA_multiwindow_configuration_v3.py'),
        'predecessor_subset_review': record(F/'FastMSS_concrete_multiwindow_v3_delta_subset_read_review_v1.json'),
        'actual_CPU_orphan_retirement_controls': proof,
        'new_parent_v4_failure_cleanup_AST_only_not_GPU_executed': True,
        'source_review_findings_in_predecessor': [
            'CPU failure cleanup omitted captured descendants if the outer launcher had already exited',
            'GPU abort could release ownership after launcher wait before observing native identity dead'],
        'source_remedies': {
            'CPU_failure': 'Always terminate captured matching PID/create_time identities, including after launcher exit; wait until all are dead before family retirement',
            'GPU_failure': 'After cooperative bounded stop/termination, observe actual native identity dead before lease release',
            'GPU_startup_failure': 'Capture actual owned launcher ancestry during handshake; verified native added only after source/PID/PPID/create_time validation; observe captured identities dead before releasing any acquired lease',
            'scope': 'The real CPU orphan control tests the shared retirement helper; GPU abort/startup remains source/AST reviewed only'},
        'native_loss_optimizer_frontier_or_augmentation_math_changed': False,
        'actual_full_DATA_positive_or_two_window_native_lifecycle_tested': False,
        'production_ready': False, 'production_enabled': False}
    target = A/'concrete_multiwindow_caller_source_review_manifest_v4.json'
    require(not target.exists(), 'Preserve source review manifest'); write_json(target, value)
    carrier = read_json(B/'KIRAAT_FLAC_carrier_current_status_v1.json')
    status = read_json(A/'production_preparation_status_v1.json')
    status.update(updated_at_utc=now, current_stage='GPU_idle_concrete_caller4_failure_retirement_CPU9_passed_complete_ROOT_review_and_final_DATA_pending',
        concrete_multiwindow_source_review=record(target), configuration_assembler_source=value['configuration_assembler'],
        owned_orphan_retirement_CPU_actual=proof, actual_GPU_running=False, actual_owned_GPU_PID=None, actual_owned_GPU_session=None,
        DATA_carrier_snapshot={'at_utc': carrier['at_utc'], 'covered': carrier['cumulative_covered_sources'], 'required': 729},
        production_ready=False, production_enabled=False)
    write_json(A/'production_preparation_status_v1.json', status)
    readiness = A/'launch_readiness_v32.json'; require(not readiness.exists(), 'Preserve readiness history'); write_json(readiness, status)
    current = read_json(A/'memory_diagnosis_status.json'); current.update(updated_at_utc=now,
        current_stage=status['current_stage'], launch_readiness=record(readiness), launch_readiness_receipt=str(readiness),
        actual_GPU_running=False, actual_owned_GPU_PID=None, actual_owned_GPU_session=None, production_ready=False)
    write_json(A/'memory_diagnosis_status.json', current)
    print({'source_review': record(target), 'owned_bindings': len(value['bindings']), 'actual_CPU9': proof,
           'parent': record(R/'run_ASR18_full_DATA_multiwindow_v4.py'), 'helper': record(S/'asr18_owned_family_observation_v2.py'), 'GPU_running': False})


if __name__ == '__main__': main()
