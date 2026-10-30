"""Seal a compact handoff after genuine final DATA producer exits; never arm training."""
import os
from pathlib import Path
import sys

from ASR18_continuous_data_guard_v5 import AREA, DataIOGuard
from cpu_preparation_guard_v2 import now
from finalize_ASR18_declared_training_geometry_v1 import MetadataIO
from refresh_ASR18_canonical_producer_dispatch_v2 import PROOFS

P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import read_json, write_json


def main():
    guard = DataIOGuard('final_DATA_preparation_handoff', heavy=False, maximum_bytes=64_000_000)
    guard.wait()
    io = MetadataIO(guard)
    output = AREA / 'ASR18_final_DATA_preparation_handoff_v1.json'
    assert not output.exists()
    actual = {}
    markers = {}
    for role, name in PROOFS.items():
        path = AREA / name
        proof = io.json(path)
        assert proof['kind'] == 'ASR18_source_bound_actual_tool_exit_v1'
        assert proof['owner'] == '/root/cpu_data_preparation'
        assert type(proof['actual_exit_code']) is int and proof['actual_exit_code'] == 0
        assert proof['native_PID'] > 0 and proof['actual_tool_chunk_id']
        marker = io.binding(proof['published_marker']['path'])
        assert all(marker[key] == value for key, value in proof['published_marker'].items())
        actual[role] = dict(actual_wrapper=io.binding(path), published_marker=marker,
            owned_session=proof['owned_session'], native_PID=proof['native_PID'],
            actual_exit_code=0, actual_tool_chunk_id=proof['actual_tool_chunk_id'])
        markers[role] = io.json(marker['path'])
    raw = markers['final_raw_DATA_population']
    geometry = markers['final_declared_geometry']
    carriers = markers['final_carrier_coverage']
    two_pass = markers['final_two_pass_with_declared_geometry']
    assert raw['retained_rows'] == geometry['population']['retained_rows'] == two_pass['retained_rows'] == 2631115
    assert raw['total_target_tokens'] == two_pass['total_target_tokens'] == 101470757
    assert raw['unresolved_geometry_rows'] == geometry['unknown_invalid_counts']['retained_unresolved_rows'] == 0
    assert carriers['source_files'] == geometry['KIRAAT_source_sidecars']['sources_count'] == 729
    assert raw['ordering'] == two_pass['ordering']
    assert raw['quality_filters'] == raw['language_filters'] == []
    hpath = AREA / 'DATA_owner_compact_handoff_v4.json'
    h = read_json(hpath)
    assert h['final_metadata_chain_complete'] is True and h['current_final_metadata_producer'] is None
    assert all(role.get('process_alive') is not True for role in h['owned_sessions'].values())
    report = dict(kind='ASR18_final_DATA_preparation_handoff_v1', schema_version=1,
        owner='/root/cpu_data_preparation', at_utc=now(), PID=os.getpid(), complete_DATA_preparation=True,
        completed_original_sources=784, total_original_compressed_source_bytes=718862913959,
        finalized_catalog_rows=2631115, finalized_target_tokens=101470757,
        population_mean_U=101470757 / 2631115, vocabulary=8711, pass_count=2,
        complete_KIRAAT_source_carriers=729, all_physical_KIRAAT_rows=carriers['total_physical_rows'],
        independently_audited_original_FLAC_bytes=carriers['total_original_FLAC_bytes'],
        declared_geometry_population=geometry['population'],
        unresolved_declared_geometry_rows=0, canonical_producers_actual_closed=actual,
        ordering=raw['ordering'], quality_filters=[], language_filters=[],
        raw_labels_flags_and_historical_exception_outcomes_preserved=True,
        original_source_audio_unchanged=True, original_compressed_bytes_copied_without_transcoding=True,
        full_PCM_pool_or_full_corpus_PCM_decode_created=False,
        all_owned_DATA_media_and_metadata_producers_actually_closed=True,
        no_successor_carrier_phase25=True,
        remaining_before_production=['Independent DATA verifier/evidence review by /root/memory_diagnosis',
            'ROOT-reviewed CPU resource/config/native lifecycle and production admission',
            'Runtime heldout PCM and native/resampled/encoder-length match before each consumed draw; versioned reconciliation on mismatch'],
        reference_PCM_binding_is_not_a_full_TRAIN_fuzzy_or_crosscodec_scan=True,
        gross_metadata_hours_are_not_unique_new_accepted_TRAIN_hours=True,
        new_accepted_TRAIN_hours=0, native_envelope_or_forecast_approved_by_DATA=False,
        TRAIN_admitted=False, production_training_started=False,
        authoritative_current_dispatch=str(AREA / 'current_data_dispatch_v2.json'))
    io.unchanged()
    write_json(output, report)
    pin = io.binding(output)
    h.update(updated_at_utc=now(), final_DATA_preparation_handoff=pin,
        current_final_metadata_producer=None, final_metadata_chain_complete=True,
        full_boundary_actions=['All DATA canonical producers have observed owned-tool0 and immutable bindings. No DATA worker or phase25 remains. Independent verifier/native-resource/config/ROOT production admission remains with memory owner/ROOT; runtime heldout and decoded geometry checks remain mandatory.'],
        TRAIN_admitted=False)
    write_json(hpath, h)
    for name in ('current_data_dispatch_v2.json', 'current_data_readiness_v1.json', 'current_data_preparation_status_v2.json'):
        path = AREA / name
        current = read_json(path)
        current.update(updated_at_utc=now(), status='canonical_DATA_preparation_actual_CLOSED0',
            final_DATA_preparation_handoff=pin, final_metadata_chain_complete=True,
            final_metadata_producer=None, current_final_metadata_producer=None,
            final_metadata_IO_running=False, heavy_IO_running=False,
            all729_carrier_final_coverage=True, full_declared_geometry_ready=True,
            independent_final_DATA_validation_pending=True, TRAIN_admitted=False,
            production_training_ready=False, production_training_started=False)
        write_json(path, current)
    eta_path = AREA / 'current_data_completion_ETA_v1.json'
    eta = read_json(eta_path)
    eta.update(updated_at_utc=now(), remaining_network_bytes=0, remaining_carrier_sources=0,
        remaining_carrier_source_stage_pairs=0, remaining_legacy_declared_sidecar_sources=0,
        carrier_sources_complete=729, declared_geometry_sources_complete=729,
        full_carrier_coverage_ready=True, full_declared_geometry_ready=True,
        canonical_metadata_finalization_in_progress=None,
        full_DATA_preparation_complete=True, full_launch_readiness=False,
        conditional_remaining_carrier_seconds=0, conditional_remaining_legacy_seconds=0,
        conditional_carrier_plus_declared_work_finish_UTC=None,
        all729_carrier_and_declared_geometry_conditional_ETA_UTC=None,
        final_DATA_preparation_handoff=pin,
        ETA_note='All original downloads, all729 carriers/sidecars and canonical DATA metadata producers genuinely closed0. Preparation ETA is complete; independent validation and ROOT-reviewed production admission are separate, with no launch ETA claimed.')
    eta['carrier'].update(actual_closed_sources=729, remaining_sources=0,
        actual_individual_child_audited_sources=729, current_outer_phase_actual_exit_pending=False,
        conditional_remaining_hours=0)
    eta['legacy_declared'].update(remaining_sources=0, current_phase=8, current_phase_not_counted_as_sealed=False)
    write_json(eta_path, eta)
    project_path = P / 'outputs/parakeet_turkish/current_CPU_data_preparation_v1.json'
    project = read_json(project_path)
    section = project['ASR18_BPE_weak_then_clean_DATA']
    section.update(updated_at_utc=now(), status='canonical_DATA_preparation_actual_CLOSED0',
        final_DATA_preparation_handoff=pin, canonical_producers_actual_closed=actual,
        current_carrier_owned_session=None, current_carrier_supervisor_PID=None,
        historical_final_carrier_owned_session=1316, historical_final_carrier_supervisor_PID=37620,
        carrier_sources_individually_audited_at_observation=729,
        full_carrier_and_declared_envelope_ready=True,
        two_pass_population_metadata_actual_closed=actual['final_two_pass_with_declared_geometry']['actual_wrapper'],
        no_owned_DATA_worker_running=True, independent_final_DATA_validation_pending=True,
        TRAIN_admitted=False, production_training_started=False)
    write_json(project_path, project)
    assert not any(name in sys.modules for name in ('torch', 'av', 'soundfile', 'scipy', 'pyarrow', 'transformers'))
    print(dict(final_DATA_preparation_handoff=pin, all_canonical_DATA_producers_actual0=True, TRAIN_admitted=False))


if __name__ == '__main__':
    main()
