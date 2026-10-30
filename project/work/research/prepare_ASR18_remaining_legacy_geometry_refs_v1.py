"""Metadata-only provenance/frontier for the remaining pre-sidecar carriers.

No job is launched here. Each eventual <=32-source phase needs a separate plan,
actual current boundary, fresh guard admission and actual supervised closure.
"""
import json
from pathlib import Path
import sys

from ASR18_continuous_data_guard_v5 import DataIOGuard, AREA, ROOT
from ASR18_FLAC_carrier_guard_v3 import CACHE, signature
from cpu_preparation_guard_v2 import now
from finalize_ASR18_FLAC_carrier_coverage_v1 import collect, original_outer
from normalize_ASR18_FLAC_stage_provenance_v1 import binding, checked, normalize_owned, normalize_delegated_fixture
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import read_json, write_json


def main():
    guard = DataIOGuard('remaining_legacy_geometry_metadata_preparation', heavy=False, maximum_bytes=64_000_000)
    guard.wait()
    target = AREA / 'remaining_legacy_declared_geometry_sources_prepared_v1.json'
    assert not target.exists()
    base = AREA / 'KIRAAT_FLAC_carrier_batch07_actual_CLOSED0_v1.json'
    initial = collect(base)
    completed_path = AREA / 'KIRAAT_declared_geometry_completion01_actual_CLOSED0_v1.json'
    completed = read_json(completed_path)
    assert completed['actual_exit_code'] == 0 and completed['actual_tool_chunk_id']
    already = {item['job']['sha256'] for item in checked(completed['completed_sources'])}
    plan = read_json(AREA / 'preparation_plan_v1.json')
    jobs = {item['sha256']: item for item in plan['download_jobs']
            if item['repository'] == 'serdarcaglar/kiraat'
            and Path(item['path']).name.startswith(('train-', 'rest-'))}
    report = ROOT / 'analysis/ASR18_memory_diagnosis_v1/fixture_FLAC_carriers_actual_01/report.json'
    fixture = read_json(report)
    for sha, job in jobs.items():
        if sha in initial:
            continue
        matches = [item for item in fixture['actual_owned_results'] if item['source'] == job['destination']
                   and item.get('phase') in ('prepare', 'audit') and item.get('actual_exit_code') == 0]
        if len(matches) == 2 and {item['phase'] for item in matches} == {'prepare', 'audit'}:
            initial[sha] = (None, str(report))
    assert len(initial) == 229 and len(already) == 32 and already.issubset(initial)
    remaining = []
    normalized = AREA / 'legacy_geometry_original_stage_provenance_v1'
    correction = AREA / 'carrier_phase_v1_actual_retirement_v1.json'
    for sha in sorted(set(initial) - already, key=lambda value: jobs[value]['path']):
        guard.pace(262144)
        job, (item, outer) = jobs[sha], initial[sha]
        folder = CACHE / sha
        receipt_path = folder / 'receipt.json'
        receipt = read_json(receipt_path)
        assert receipt['complete'] is True and receipt['source']['source_sha256'] == sha
        assert signature(job['destination']) == {
            'path': job['destination'], 'bytes': receipt['source']['source_size'],
            'mtime_ns': receipt['source']['source_mtime_ns']}
        stages = []
        if item is None:
            for mode in ('prepare', 'audit'):
                stages.append(normalize_delegated_fixture(report, job['destination'], sha, mode,
                    receipt_path if mode == 'prepare' else folder / 'validation.json', normalized))
        else:
            assert item['job'] == job
            for key in ('prepared_actual_exit', 'audited_actual_exit'):
                raw = checked(item[key])
                proof_path = Path(item[key]['path'])
                footer = checked(raw['native_closure'])
                stages.append(normalize_owned(proof_path, proof_path.parent / 'result.json', normalized,
                    parent_actual_tool_closure=original_outer(proof_path, outer),
                    PID_correction_provenance=correction if raw['native_PID'] != footer['PID'] else None))
        assert stages[0]['mode'] in ('prepare', 'completion_verify') and stages[1]['mode'] == 'audit'
        assert stages[0]['native_PID'] != stages[1]['native_PID']
        remaining.append(dict(job=job, source_signature=signature(job['destination']),
            source_receipt=binding(Path(job['destination']).with_suffix('.parquet.receipt.json')),
            initial_payload_bytes=receipt['records']['bytes'], existing_payload_bytes=receipt['records']['bytes'],
            original_prepared_actual_exit=stages[0]['actual_exit_proof'],
            original_audited_actual_exit=stages[1]['actual_exit_proof']))
    assert len(remaining) == 197
    value = dict(kind='ASR18_remaining_pre_sidecar_original_carriers_metadata_frontier_v1', owner='/root/cpu_data_preparation',
        created_at_utc=now(), PID=guard.process.pid, prepared_only_not_launched=True,
        original_229_boundary=binding(base), completed_legacy32_actual=binding(completed_path),
        remaining_sources_count=197, remaining_existing_payload_bytes=sum(item['existing_payload_bytes'] for item in remaining),
        selected=remaining, normalized_stage_proofs_preserve_original_nonzero_or_unknown_outcomes=True,
        no_audio_or_payload_read=True, maximum_per_phase_sources=32, maximum_per_phase_existing_input_bytes=16_000_000_000,
        maximum_per_phase_IO_bytes=96_000_000_000, maximum_phase_new_audio_payload_bytes=0,
        maximum_IO_bytes_s=256*1024**2, combined_cache_cap_bytes=400_000_000_000,
        reserve_all_remaining_download_bytes_and_E100GB=True, TRAIN_admitted=False)
    write_json(target, value)
    print(dict(prepared_remaining_refs=binding(target), remaining_sources=197,
        existing_payload_bytes=value['remaining_existing_payload_bytes'], launched=False), flush=True)


if __name__ == '__main__':
    main()
