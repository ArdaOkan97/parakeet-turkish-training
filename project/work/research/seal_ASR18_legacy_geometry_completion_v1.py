"""Seal only a genuinely observed finite header-completion tool exit."""
import argparse
from pathlib import Path
import sys
import psutil

from ASR18_continuous_data_guard_v5 import AREA, ROOT
from cpu_preparation_guard_v2 import now
from normalize_ASR18_FLAC_stage_provenance_v1 import binding, checked
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, required=True)
    parser.add_argument('--owned-session', type=int, required=True)
    parser.add_argument('--actual-exit-code', type=int, choices=[0], required=True)
    parser.add_argument('--actual-tool-chunk', required=True)
    args = parser.parse_args()
    phase = AREA / f'KIRAAT_declared_geometry_completion{args.phase:02d}_v1'
    native_path = phase / 'native_phase_CLOSED0.json'
    native = read_json(native_path)
    assert native['kind'] == 'ASR18_native_bounded_declared_geometry_completion_CLOSED0_v1'
    assert type(native['native_return_code']) is int and native['native_return_code'] == 0
    assert not psutil.pid_exists(native['supervisor_PID']), 'Observe actual tool exit before sealing'
    plan = checked(native['pinned_plan'])
    finished = checked(native['completed_sources'])
    assert 1 <= len(finished) == native['phase_source_files'] == len(plan['selected']) <= 32
    assert native['phase_new_payload_bytes'] == 0
    assert native['phase_paced_IO_bytes'] <= 96_000_000_000
    rows = unresolved = original_bytes = 0
    peaks = []
    for item in finished:
        proof = checked(item['geometry_completion_actual_exit'])
        assert proof['mode'] == 'declared_geometry_completion'
        assert type(proof['actual_exit_code']) is int and proof['actual_exit_code'] == 0
        assert proof['parent_observed_wait'] is True
        footer = checked(proof['native_closure'])
        assert type(footer['native_return_code']) is int and footer['native_return_code'] == 0
        assert footer['PID'] == proof['native_PID'] and footer['watchdog_clean'] is True
        assert not psutil.pid_exists(footer['PID'])
        assert footer['peak_private_bytes'] < footer['CPU_private_cap_bytes'] == 2_000_000_000
        result = item['geometry_result']
        assert result['source_SHA256'] == item['job']['sha256']
        assert result['all_record_SHA_and_payload_SHA_verified'] is True
        assert result['original_source_and_carrier_files_modified'] is False
        geometry = result['declared_STREAMINFO_geometry']
        assert geometry['sidecar'] == binding(geometry['sidecar']['path'])
        rows += geometry['physical_rows']
        unresolved += geometry['unresolved_declared_geometry_rows']
        original_bytes += result['existing_FLAC_payload_bytes_read_and_SHA_verified']
        peaks.append(footer['peak_private_bytes'])
    assert original_bytes <= 16_000_000_000
    target = AREA / f'KIRAAT_declared_geometry_completion{args.phase:02d}_actual_CLOSED0_v1.json'
    assert not target.exists()
    value = dict(kind='ASR18_bounded_legacy_geometry_completion_actual_tool_CLOSED0_v1',
        owner='/root/cpu_data_preparation', at_utc=now(), owned_session=args.owned_session,
        supervisor_PID=native['supervisor_PID'], actual_exit_code=0,
        actual_tool_chunk_id=args.actual_tool_chunk, no_owned_carrier_child_running=True,
        native_phase_closure=binding(native_path), completed_sources=binding(phase / 'completed_sources.json'),
        pinned_plan=native['pinned_plan'], source_pins=native['source_pins'], phase_completed_sources=len(finished),
        physical_rows=rows, all_record_and_payload_SHA_verified_bytes=original_bytes,
        phase_paced_IO_bytes=native['phase_paced_IO_bytes'], maximum_child_private_bytes=max(peaks),
        unresolved_declared_geometry_rows=unresolved, carrier_coverage_unchanged=plan['prior_covered_sources'],
        prior_geometry_covered_sources=plan['prior_geometry_covered_sources'],
        cumulative_geometry_covered_sources=plan['prior_geometry_covered_sources']+len(finished),
        declared_forecast_not_full_decoded_validation=True, full729_geometry_coverage=False,
        TRAIN_admitted=False)
    write_json(target, value)
    handoff = read_json(AREA / 'DATA_owner_compact_handoff_v4.json')
    handoff['updated_at_utc'] = now()
    handoff['owned_sessions']['declared_geometry_completion'].update(
        actual_exit_code=0, actual_tool_chunk_id=args.actual_tool_chunk, process_alive=False)
    handoff['latest_actual_declared_geometry_completion'] = binding(target)
    handoff['current_declared_geometry_completion_snapshot'] = read_json(phase / 'status.json')
    write_json(AREA / 'DATA_owner_compact_handoff_v4.json', handoff)
    print(dict(actual_closure=binding(target), geometry_coverage=value['cumulative_geometry_covered_sources'],
        carrier_coverage_unchanged=value['carrier_coverage_unchanged']), flush=True)


if __name__ == '__main__':
    main()
