"""Fail-closed final729 metadata producer. Never decode/rehash audio payloads."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

from ASR18_continuous_data_guard_v5 import DataIOGuard, AREA, ROOT
from cpu_preparation_guard_v2 import memory, now
from ASR18_FLAC_carrier_guard_v3 import CACHE, signature
from normalize_ASR18_FLAC_stage_provenance_v1 import (
    binding, checked, normalize_owned, normalize_delegated_fixture)

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import read_json, write_json


def base(pin):
    return {key: pin[key] for key in ('path', 'bytes', 'sha256')}


def collect(closure_path):
    closure = read_json(closure_path)
    assert type(closure['actual_exit_code']) is int and closure['actual_exit_code'] == 0
    assert closure['owner'] == '/root/cpu_data_preparation' and closure['no_owned_carrier_child_running'] is True
    assert closure.get('actual_tool_chunk') or closure.get('actual_tool_chunk_id')
    if closure['kind'] == 'ASR18_original32_carrier_pilot_actual_tool_CLOSED0_v1':
        entries = checked(closure['completed_sources']) + checked(closure['fixture_completion_verifications'])
        return {item['job']['sha256']: (item, str(closure_path)) for item in entries}
    assert closure['kind'] == 'ASR18_bounded_carrier_batch_actual_tool_CLOSED0_v1'
    prior = closure['prior_actual_closure']
    checked(prior)
    values = collect(prior['path'])
    for item in checked(closure['completed_sources']):
        assert item['job']['sha256'] not in values
        values[item['job']['sha256']] = (item, str(closure_path))
    return values


def original_outer(proof_path, fallback):
    text = str(proof_path)
    names = {
        'KIRAAT_FLAC_carrier_phase_v1': 'carrier_phase_v1_actual_retirement_v1.json',
        'KIRAAT_FLAC_carrier_continuation_v2': 'carrier_continuation_v2_actual_CLOSED0.json',
        'KIRAAT_FLAC_carrier_continuation_v4': 'carrier_continuation_v4_actual_failure_v1.json'}
    for component, target in names.items():
        if component in Path(text).parts:
            return AREA / target
    return Path(fallback)


def observed_wrapper(path, marker):
    value = read_json(path)
    assert value['kind'] == 'ASR18_source_bound_actual_tool_exit_v1'
    assert value['owner'] == '/root/cpu_data_preparation'
    assert type(value['actual_exit_code']) is int and value['actual_exit_code'] == 0
    assert value['actual_tool_chunk_id']
    assert value['published_marker'] == base(binding(marker))
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--last-actual-phase', required=True)
    args = parser.parse_args()
    assert Path(args.last_actual_phase).resolve().is_relative_to(ROOT.resolve())
    guard = DataIOGuard('final729_carrier_metadata_producer', heavy=False, maximum_bytes=8_000_000_000)
    guard.wait()
    target = AREA / 'KIRAAT_original_FLAC_carrier_coverage_v1.json'
    assert not target.exists(), 'Never overwrite a final coverage receipt'
    plan_path = AREA / 'preparation_plan_v1.json'
    contract_path = AREA / 'production_finalization_contract_v5.json'
    inventory_path = AREA / 'completed_source_inventory_v1.json'
    inventory = read_json(inventory_path)
    assert inventory['kind'] == 'ASR18_completed_source_inventory_v1' and inventory['sources_count'] == 784
    assert inventory['finalization_contract'] == base(binding(contract_path))
    assert inventory['preparation_plan'] == base(binding(plan_path))
    observed_wrapper(AREA / 'completed_source_inventory_owned_tool_actual_CLOSED0_v1.json', inventory_path)
    metadata_marker = AREA / 'broad_metadata_actual_CLOSED0_v3.json'
    observed_wrapper(AREA / 'broad_metadata_owned_tool_actual_CLOSED0_v3.json', metadata_marker)
    meta = read_json(metadata_marker)
    assert meta['actual_exit_code'] == 0 and meta['committed_metadata_files'] == 730
    catalog = Path(meta['catalog_path'])
    assert catalog.resolve().is_relative_to(ROOT.resolve())
    plan = read_json(plan_path)
    jobs = {job['sha256']: job for job in plan['download_jobs']
            if job['repository'] == 'serdarcaglar/kiraat'
            and Path(job['path']).name.startswith(('train-', 'rest-'))}
    assert len(jobs) == 729
    definitions = collect(args.last_actual_phase)
    fixture_report = ROOT / 'analysis/ASR18_memory_diagnosis_v1/fixture_FLAC_carriers_actual_01/report.json'
    report = read_json(fixture_report)
    delegated = {}
    for sha, job in jobs.items():
        if sha in definitions:
            continue
        matches = [item for item in report['actual_owned_results'] if item['source'] == job['destination']
                   and item.get('phase') in ('prepare', 'audit')
                   and type(item.get('actual_exit_code')) is int and item['actual_exit_code'] == 0]
        if len(matches) == 2 and {item['phase'] for item in matches} == {'prepare', 'audit'}:
            delegated[sha] = matches
    assert set(definitions).union(delegated) == set(jobs), 'All729 actual preparation/completion AND audit pairs required; partial phase cannot finalize'
    expected_inventory = {item['destination']: item for item in inventory['sources']}
    assert all(job['destination'] in expected_inventory and expected_inventory[job['destination']]['source_SHA256'] == sha
               for sha, job in jobs.items())
    fresh = 0
    while fresh < 3:
        guard.wait()
        state = memory()
        limit = state['commit_headroom_bytes'] / (1 - state['commit_ratio'])
        admitted = state['commit_ratio'] + 750_000_000 / limit < .9 and state['commit_headroom_bytes'] - 750_000_000 >= 2_000_000_000
        fresh = fresh + 1 if admitted else 0
        if not admitted:
            guard.publish('yielding_projected_metadata_grouping_memory', reserved_peak_plus_margin_bytes=750_000_000, memory=state)
        time.sleep(.1 if admitted else .25)
    assert guard.sha(catalog) == meta['catalog_sha256']
    db = sqlite3.connect(catalog.as_uri() + '?mode=ro&immutable=1', uri=True)
    db.execute('PRAGMA cache_size=-4096')
    db.execute('PRAGMA temp_store=MEMORY')
    previous_io = guard.process.io_counters().read_bytes
    def pace_sql():
        nonlocal previous_io
        current = guard.process.io_counters().read_bytes
        if current > previous_io:
            guard.pace(current - previous_io)
        previous_io = current
        guard.wait()
        return 0
    db.set_progress_handler(pace_sql, 5000)
    files = {name: (sha, seen, retained) for name, sha, seen, retained in
             db.execute('SELECT file,source_sha256,rows_seen,rows_retained FROM files')}
    ranges = {name: (minimum, maximum, count) for name, minimum, maximum, count in db.execute(
        "SELECT shard,MIN(row_index),MAX(row_index),COUNT(*) FROM clips WHERE dataset='kiraat' GROUP BY shard")}
    db.close()
    normalized = AREA / 'carrier_provenance_normalized_full_v1'
    results = []
    correction = AREA / 'carrier_phase_v1_actual_retirement_v1.json'
    for sha, job in sorted(jobs.items(), key=lambda pair: pair[1]['path']):
        guard.wait()
        source = Path(job['destination'])
        folder = CACHE / sha
        receipt_path = folder / 'receipt.json'
        receipt = read_json(receipt_path)
        assert receipt['kind'] == 'ASR18_original_compressed_FLAC_records_v1' and receipt['complete'] is True
        original = receipt['source']
        assert original['source_sha256'] == sha and original['revision'] == job['revision']
        assert signature(source) == {'path': str(source.resolve()), 'bytes': original['source_size'], 'mtime_ns': original['source_mtime_ns']}
        acquired_path = source.with_suffix(source.suffix + '.receipt.json')
        acquired = read_json(acquired_path)
        assert acquired['sha256'] == sha and acquired['revision'] == job['revision']
        assert base(binding(acquired_path)) == expected_inventory[job['destination']]['source_acquisition_receipt']
        if sha in definitions:
            item, outer = definitions[sha]
            assert item['job'] == job and signature(source) == item['source_signature']
            stages = []
            raw_stages = []
            for key in ('prepared_actual_exit', 'audited_actual_exit'):
                checked(item[key])
                proof_path = Path(item[key]['path'])
                raw = read_json(proof_path)
                assert type(raw['actual_exit_code']) is int and raw['actual_exit_code'] == 0
                raw_stages.append(raw)
                footer = checked(raw['native_closure'])
                assert type(footer['native_return_code']) is int and footer['native_return_code'] == 0
                assert type(footer['PID']) is int and footer['PID'] > 0
                stages.append(normalize_owned(proof_path, proof_path.parent / 'result.json', normalized,
                    parent_actual_tool_closure=original_outer(proof_path, outer),
                    PID_correction_provenance=correction if raw['native_PID'] != footer['PID'] else None))
            audit_result = read_json(Path(item['audited_actual_exit']['path']).parent / 'result.json')
            audit_path = Path(audit_result['validation']['path'])
            checked(audit_result['validation'])
            original_outcome = {'actual_exit_code': 0, 'reason': 'Original owned preparation observed0',
                                'preserved_provenance': [binding(item['prepared_actual_exit']['path'])]}
            if raw_stages[0]['mode'] == 'completion_verify':
                assert job['path'] == 'data/rest-00020-of-00137.parquet'
                old_failure = ROOT / 'analysis/ASR18_memory_diagnosis_v1/FLAC_carrier_CPU_guard_01/closure.json'
                assert read_json(old_failure)['actual_exit_code'] == 1
                original_outcome = {'actual_exit_code': 1, 'reason': 'Original late dependency assertion failure preserved; new read-only completion0 plus separate audit0',
                                    'preserved_provenance': [binding(old_failure), binding(folder / 'validation.json')]}
            elif job['path'] == 'data/rest-00006-of-00137.parquet':
                original_outcome = {'actual_exit_code': None, 'reason': 'Original v4 source006 exit unobserved after parent sharing error; fresh supported prepare0/audit0, no relabeling',
                                    'preserved_provenance': [binding(AREA / 'carrier_continuation_v4_actual_failure_v1.json'), binding(AREA / 'carrier_unobserved_stage_IO_reservation_v1.json')]}
        else:
            audit_path = folder / 'validation.json'
            stages = [normalize_delegated_fixture(fixture_report, job['destination'], sha, mode,
                       receipt_path if mode == 'prepare' else audit_path, normalized) for mode in ('prepare', 'audit')]
            original_outcome = {'actual_exit_code': 0, 'reason': 'Delegated original preparation actual0 with observed parent wait',
                                'preserved_provenance': [binding(fixture_report)]}
        assert stages[0]['mode'] in ('prepare', 'completion_verify') and stages[1]['mode'] == 'audit'
        assert stages[0]['native_PID'] != stages[1]['native_PID']
        audit = read_json(audit_path)
        assert audit['passed'] is True and audit['source_sha256'] == sha
        assert audit['every_compressed_byte_and_physical_row_identical'] is True
        assert audit['full_physical_rows'] == receipt['physical_rows']
        assert audit['full_original_FLAC_bytes_compared'] == receipt['records']['bytes']
        assert base(binding(receipt_path)) == audit['carrier_receipt']
        assert files[job['path']][0] == sha and files[job['path']][1] == receipt['physical_rows']
        assert 0 <= files[job['path']][2] <= receipt['physical_rows']
        if files[job['path']][2]:
            low, high, count = ranges[job['destination']]
            assert 0 <= low <= high < receipt['physical_rows'] and count == files[job['path']][2]
        media = receipt['records']
        assert signature(media['path']) == {key: media[key] for key in ('path', 'bytes', 'mtime_ns')}
        index = receipt['index']
        assert signature(index['path']) == {key: index[key] for key in ('path', 'bytes', 'mtime_ns')}
        assert guard.sha(index['path']) == index['sha256']
        results.append({'source_sha256': sha, 'source_signature': signature(source),
            'source_acquisition_receipt': binding(acquired_path), 'carrier_receipt': binding(receipt_path),
            'payload': dict(media), 'row_index': dict(index), 'audit_receipt': binding(audit_path),
            'physical_rows': receipt['physical_rows'], 'original_FLAC_bytes': media['bytes'],
            'original_preparation_outcome': original_outcome,
            'preparation_or_completion_actual_closed': stages[0], 'independent_audit_actual_closed': stages[1]})
        write_json(AREA / 'final729_carrier_producer_status_v1.json', {'at_utc': now(), 'PID': guard.process.pid,
            'validated_sources': len(results), 'required_sources': 729, 'TRAIN_admitted': False})
    assert len(results) == 729 and len({item['source_sha256'] for item in results}) == 729
    cache_bytes = sum(path.stat().st_size for path in CACHE.rglob('*') if path.is_file())
    assert cache_bytes <= 400_000_000_000
    value = {'kind': 'ASR18_original_FLAC_carrier_coverage_v1', 'schema_version': 1, 'complete': True,
        'created_at_utc': now(), 'owner': '/root/cpu_data_preparation',
        'preparation_plan': binding(plan_path), 'finalization_contract': binding(contract_path),
        'completed_source_inventory': binding(inventory_path),
        'coverage_contract': binding(AREA / 'KIRAAT_FLAC_carrier_coverage_contract_v1.json'),
        'scope': {'repository': 'serdarcaglar/kiraat', 'revision': 'f8afe95abedc19ec6392c77cf470f5d3dcd2788f',
                  'splits': ['train', 'rest'], 'expected_source_files': 729},
        'source_files': 729, 'total_physical_rows': sum(item['physical_rows'] for item in results),
        'total_original_FLAC_bytes': sum(item['original_FLAC_bytes'] for item in results),
        'total_cache_bytes': cache_bytes, 'sources': results,
        'all_physical_rows_and_original_compressed_bytes_independently_verified': True,
        'decoded_PCM_pool_created': False, 'TRAIN_admitted': False}
    guard.wait()
    assert len(json.dumps(value, ensure_ascii=False, indent=2).encode('utf-8')) + 64 < 8_000_000
    write_json(target, value)
    assert target.stat().st_size < 8_000_000, 'Keep exact final receipt within strict bounded metadata parser'
    print({'final_coverage': binding(target), 'source_files': 729, 'payload_rehash_claimed': False,
           'actual_owned_tool_exit_still_required': True}, flush=True)


if __name__ == '__main__':
    main()
