"""Seal actual scalar controls and preserve the first fixture connection error."""
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
    actual0 = actual(A/'declared_joint_physical_CPU_actual_02', A/'declared_joint_physical_CPU_guard_02', '7d02c4')
    guard1 = read_json(A/'declared_joint_physical_CPU_guard_01/closure.json')
    require(type(guard1['actual_exit_code']) is int and guard1['actual_exit_code'] == 1, 'Preserve actual fixture1 failure')
    roots = [S/'bpe_declared_joint_budget_audit_v2.py', R/'audit_ASR18_final_declared_joint_envelope_CPU_v1.py',
        R/'validate_ASR18_declared_joint_physical_geometry_CPU_v2.py']
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    value = {'kind': 'ASR18_declared_joint_physical_geometry_source_review_manifest_v1', 'created_at_utc': now,
        'scope': 'CPU_only_logical_JOINT_DP_and_physical_feature_scalar_audit', 'actual_CPU15': actual0,
        'direct_sources': [record(path) for path in roots], 'bindings': owned_closure(roots),
        'preserved_first_fixture_failure': {'actual_exec_tool_exit_code': 1, 'actual_tool_chunk_id': 'fc139a',
            'guard': record(A/'declared_joint_physical_CPU_guard_01/closure.json'),
            'fixture_source': record(R/'validate_ASR18_declared_joint_physical_geometry_CPU_v1.py'),
            'cause': 'Fixture SQLite context manager committed but kept writer connection open; exact read-only final lock correctly rejected WinError32',
            'supported_fixture_only_change': 'New fixture2 explicitly closes each committed writer connection before final read lock; auditor source unchanged'},
        'all_full_population_positive_executed': False, 'raw_lifetime_or_encoder_decoder_or_worker_fit_approved': False,
        'actual_new_GPU_or_ASR_updates': 0, 'production_ready': False, 'production_enabled': False}
    target = A/'declared_joint_physical_geometry_source_review_manifest_v1.json'
    require(not target.exists(), 'Preserve source review receipt'); write_json(target, value)
    state = read_json(A/'production_preparation_status_v1.json')
    state.update(updated_at_utc=now, current_stage='GPU_idle_caller4_preserved_NEW_CPU_row_source_raw_lifetime_successor_and_final_DATA_pending',
        declared_joint_physical_scalar_preparation=record(target),
        current_caller4_whole_review_findings=record(A/'ROOT_concrete_multiwindow_caller4_source_read_findings_v1.json'),
        current_caller4_cleanup_subset_review=record(A.parent/'FastMSS_integration_v1/FastMSS_concrete_multiwindow_v4_final_delta_subset_read_review_v1.json'),
        current_caller4_CPU_positive_path_accepted=False, actual_GPU_running=False, actual_owned_GPU_PID=None,
        actual_owned_GPU_session=None, production_ready=False, production_enabled=False)
    readiness = A/'launch_readiness_v33.json'; require(not readiness.exists(), 'Preserve readiness history')
    write_json(A/'production_preparation_status_v1.json', state); write_json(readiness, state)
    current = read_json(A/'memory_diagnosis_status.json'); current.update(updated_at_utc=now, current_stage=state['current_stage'],
        launch_readiness=record(readiness), launch_readiness_receipt=str(readiness), actual_GPU_running=False,
        actual_owned_GPU_PID=None, actual_owned_GPU_session=None, production_ready=False)
    write_json(A/'memory_diagnosis_status.json', current)
    print({'source_review': record(target), 'CPU15_actual': actual0, 'full_final_envelope_audited': False})


if __name__ == '__main__': main()
