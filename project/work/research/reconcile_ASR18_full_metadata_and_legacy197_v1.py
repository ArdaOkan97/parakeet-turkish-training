"""Seal finite legacy receipts and refresh compact pointers; never launch media work."""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone, timedelta

from ASR18_continuous_data_guard_v5 import DataIOGuard, AREA as A
P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json

g = DataIOGuard('remaining197_geometry_metadata_reconciliation', heavy=False, maximum_bytes=32_000_000)
g.wait()


def pin(path):
    path = Path(path)
    stat = path.stat()
    assert stat.st_size < 16_000_000
    g.pace(stat.st_size)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert (path.stat().st_size, path.stat().st_mtime_ns) == (stat.st_size, stat.st_mtime_ns)
    return dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest)


def read(path):
    path = Path(path)
    assert path.stat().st_size < 16_000_000
    g.wait()
    return json.loads(path.read_text(encoding='utf-8-sig'))


front = A / 'remaining_legacy_declared_geometry_sources_prepared_v1.json'
frontier = read(front)
expected = {item['job']['sha256'] for item in frontier['selected']}
assert len(expected) == frontier['remaining_sources_count'] == 197
sources = set()
phases = []
rows = payload = charged = 0
for number in range(2, 9):
    closure_path = A / f'KIRAAT_declared_geometry_completion{number:02d}_actual_CLOSED0_v1.json'
    closure = read(closure_path)
    wrapper_path = A / f'legacy_declared_geometry{number:02d}_review_owned_tool_actual_CLOSED0_v1.json'
    wrapper = read(wrapper_path)
    review = read(wrapper['published_marker']['path'])
    assert closure['actual_exit_code'] == 0 and closure['actual_tool_chunk_id']
    assert closure['no_owned_carrier_child_running'] is True
    assert wrapper['actual_exit_code'] == 0 and wrapper['actual_tool_chunk_id']
    assert wrapper['published_marker'] == pin(wrapper['published_marker']['path'])
    assert review['passed'] is True and review['metadata_verifier_audio_payload_bytes_read'] == 0
    assert review['unresolved_declared_geometry_rows'] == closure['unresolved_declared_geometry_rows'] == 0
    assert review['actual_phase'] == pin(closure_path)
    selected = read(closure['completed_sources']['path'])
    assert len(selected) == closure['phase_completed_sources'] == review['source_count']
    for item in selected:
        sha = item['job']['sha256']
        assert sha in expected and sha not in sources
        sources.add(sha)
    rows += closure['physical_rows']
    payload += closure['all_record_and_payload_SHA_verified_bytes']
    charged += closure['phase_paced_IO_bytes']
    phases.append(dict(phase=number, source_count=len(selected), actual_phase=pin(closure_path),
                       independent_review_actual0=pin(wrapper_path)))
assert sources == expected
report = dict(kind='ASR18_acknowledged_remaining197_declared_geometry_completion_v1',
    owner='/root/cpu_data_preparation', created_at_utc=datetime.now(timezone.utc).isoformat(),
    frontier=pin(front), complete_acknowledged197=True, completed_unique_sources=197,
    total_physical_rows=rows, original_payload_bytes_reverified=payload,
    total_native_paced_IO_bytes=charged, phases=phases, source_SHA256s=sorted(sources),
    all_native_owned_exits_and_independent_mapping_reviews_actual0=True,
    metadata_verifier_audio_payload_bytes_read=0, original_payloads_unchanged=True,
    all729_source_coverage=False, carrier_sources_at_closed_legacy_boundary=326,
    declared_sources_at_closed_legacy_boundary=326,
    declared_forecast_not_full_PCM_validation=True, TRAIN_admitted=False)
write_json(A / 'remaining_legacy197_declared_geometry_actual_completion_v1.json', report)
h = read(A / 'DATA_owner_compact_handoff_v4.json')
current = read(A / 'KIRAAT_FLAC_carrier_current_status_v1.json')
assert current['supervisor_PID'] == 17656 and current['prior_covered_sources'] == 326
h['owned_sessions']['final_two_pass_population_metadata'].update(
    actual_exit_code=0, actual_tool_chunk_id='471f15', process_alive=False)
h.update(updated_at_utc=datetime.now(timezone.utc).isoformat(),
    final_two_pass_population_metadata_actual_closed=pin(A / 'ASR18_final_two_pass_population_metadata_owned_tool_actual_CLOSED0_v1.json'),
    latest_actual_declared_geometry_independent_review=pin(A / 'legacy_declared_geometry08_review_owned_tool_actual_CLOSED0_v1.json'),
    remaining197_legacy_geometry_actual_completion=pin(A / 'remaining_legacy197_declared_geometry_actual_completion_v1.json'),
    prepared_final_DATA_population_publisher=pin(A / 'final_DATA_population_metadata_publisher_prepared_v2.json'),
    current_carrier_status_snapshot=current)
h['full_boundary_actions'] = [
    'All784 compressed sources/51TAR indexes/full730 catalog/full8711 targets/denominator-only two-pass metadata are genuinely actual0. N2631115/SUM101470757/meanU38.56568679058118. No download or light writer remains.',
    'All197 acknowledged legacy sources have genuine source-bound phase and independent mapping/sidecar review0; initial32 proof preserved. Closed legacy frontier has326 carrier and326 declared sources, not729.',
    'Supervise SAME carrier12 own86819/supervisor17656 unchanged. Consume actual exit; only actual0 permits seal_ASR18_carrier_batch_and_pin_one_v4.py to seal12/pin ONE separately capped successor. Finite manual admissions only, one child, no GPU permit/rate/concurrency change.',
    'After all729 source pairs finish, run carrier coverage producer with fresh forecast and actual wrapper; then declared envelope producer using legacy01..08 actual closures, preserve actual0 four-field wrapper. No fullPCM scan or quality selection.',
    'After declared envelope actual0, preserve metadata76ba5e/actual471f15 and run NEW ASR18_final_two_pass_population_metadata_with_declared_geometry_v1.json using unchanged producer --with-declared-geometry; observe actual0 and matching *_with_declared_geometry_owned_tool_actual_CLOSED0_v1.json four-field wrapper.',
    'Only then raw DATA publisher v2 may publish canonical aggregate and genuine actual wrapper. ROOT/memoryowner bridge/native forecast/production arm remain separate; TRAINfalse.'
]
write_json(A / 'DATA_owner_compact_handoff_v4.json', h)
eta = read(A / 'current_data_completion_ETA_v1.json')
remaining = 729 - current['cumulative_covered_sources']
seconds = remaining * 24.505697916666577
finish = (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()
eta.update(updated_at_utc=datetime.now(timezone.utc).isoformat(),
    carrier_sources_complete=current['cumulative_covered_sources'],
    declared_geometry_sources_complete=current['cumulative_covered_sources'],
    remaining_carrier_sources=remaining, remaining_legacy_header_sources_at_last_sealed_boundary=0,
    conditional_remaining_carrier_seconds=seconds, conditional_remaining_legacy_seconds=0,
    all729_carrier_and_declared_geometry_conditional_ETA_UTC=finish,
    conditional_carrier_plus_declared_work_finish_UTC=finish)
eta['carrier'].update(actual_closed_sources=326,
    actual_individual_child_audited_sources=current['cumulative_covered_sources'], remaining_sources=remaining,
    conditional_remaining_hours=seconds / 3600, current_outer_phase_actual_exit_pending=True)
eta['legacy_declared'].update(sealed_completed_from197=197, remaining_sources=0,
    actual_complete197=True, conditional_remaining_hours=0)
write_json(A / 'current_data_completion_ETA_v1.json', eta)
pointer_path = P / 'outputs/parakeet_turkish/current_CPU_data_preparation_v1.json'
pointer = read(pointer_path)
pointer['ASR18_BPE_weak_then_clean_DATA'] = dict(owner='/root/cpu_data_preparation',
    updated_at_utc=datetime.now(timezone.utc).isoformat(),
    direct_human_authorization_after_prior_cancellation=True,
    prior_cancelled_reference_ASR_phases_not_resumed=True,
    current_authoritative_dispatch=str(A / 'current_data_dispatch_v2.json'),
    current_authoritative_status=str(A / 'current_data_preparation_status_v2.json'),
    current_authoritative_handoff=str(A / 'DATA_owner_compact_handoff_v4.json'),
    completed_original_compressed_sources=784, total_original_compressed_source_bytes=718862913959,
    source_inventory_actual_closed=pin(A / 'completed_source_inventory_owned_tool_actual_CLOSED0_v1.json'),
    full_final_metadata_rows=2631115, total_final_target_tokens=101470757,
    full_population_mean_U=101470757 / 2631115,
    two_pass_population_metadata_actual_closed=pin(A / 'ASR18_final_two_pass_population_metadata_owned_tool_actual_CLOSED0_v1.json'),
    ordering_owner='/root/memory_diagnosis', ordering_representation='virtual CompletePassCursor',
    all197_legacy_geometry_actual_completion=pin(A / 'remaining_legacy197_declared_geometry_actual_completion_v1.json'),
    carrier_live_status=str(A / 'KIRAAT_FLAC_carrier_current_status_v1.json'),
    current_carrier_owned_session=86819, current_carrier_supervisor_PID=17656,
    carrier_sources_individually_audited_at_observation=current['cumulative_covered_sources'],
    required_KIRAAT_carrier_sources=729, full_carrier_and_declared_envelope_ready=False,
    TRAIN_admitted=False, production_training_started=False)
write_json(pointer_path, pointer)
print(dict(remaining197_actual_closed_metadata=pin(A / 'remaining_legacy197_declared_geometry_actual_completion_v1.json'),
    current_carrier_sources=current['cumulative_covered_sources'],
    current_carrier_active_child=current.get('active_child_PID'), conditional_carrier_finish_UTC=finish))
