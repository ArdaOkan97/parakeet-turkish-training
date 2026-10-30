"""After one observed finite batch exit, seal it and pin at most one next batch."""
import argparse
import hashlib
from pathlib import Path
import sys

import psutil
from ASR18_continuous_data_guard_v5 import DataIOGuard, AREA
from ASR18_FLAC_carrier_guard_v3 import CACHE, signature
from cpu_preparation_guard_v2 import now

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import read_json, write_json


def binding(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def checked(pin):
    assert binding(pin['path']) == pin
    return read_json(pin['path'])


def prior_hashes(pin):
    receipt = checked(pin)
    assert receipt['actual_exit_code'] == 0 and receipt['no_owned_carrier_child_running'] is True
    if receipt['kind'] == 'ASR18_original32_carrier_pilot_actual_tool_CLOSED0_v1':
        values = {item['job']['sha256'] for item in checked(receipt['completed_sources'])}
        values.update(item['source_sha256'] for item in receipt['fixture_source_bindings'])
        assert len(values) == receipt['audited_total_unique_sources'] == 37
        return values
    assert receipt['kind'] == 'ASR18_bounded_carrier_batch_actual_tool_CLOSED0_v1'
    values = prior_hashes(receipt['prior_actual_closure'])
    fresh = {item['job']['sha256'] for item in checked(receipt['completed_sources'])}
    assert not values.intersection(fresh)
    values.update(fresh)
    assert sorted(values) == receipt['covered_source_hashes']
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--owned-session', type=int, required=True)
    parser.add_argument('--actual-exit-code', type=int, required=True)
    parser.add_argument('--actual-tool-chunk', required=True)
    parser.add_argument('--maximum-next-sources', type=int, default=32)
    args = parser.parse_args()
    assert 1 <= args.maximum_next_sources <= 32
    assert 2 <= args.batch <= 30 and args.actual_exit_code == 0
    guard = DataIOGuard('carrier_metadata_boundary_seal', heavy=False, maximum_bytes=32_000_000)
    guard.wait()
    phase = AREA / f'KIRAAT_FLAC_carrier_batch{args.batch:02d}_v6'
    plan_path = AREA / f'KIRAAT_FLAC_carrier_batch{args.batch:02d}_plan_v6.json'
    plan = read_json(plan_path)
    native = read_json(phase / 'native_phase_CLOSED0.json')
    assert native['native_return_code'] == 0 and checked(native['pinned_plan']) == plan
    finished = checked(native['completed_sources'])
    assert len(finished) == native['phase_source_files'] == len(plan['selected']) <= 32
    assert native['phase_new_payload_bytes'] <= 16_000_000_000
    assert native['phase_paced_IO_bytes'] + 32_000_000 <= 96_000_000_000
    for process in psutil.process_iter(['pid', 'cmdline']):
        try:
            command = process.info['cmdline'] or []
            assert not any('prepare_ASR18_KIRAAT_FLAC_records_v' in part for part in command), 'A carrier child remains alive'
            assert not any('run_ASR18_FLAC_carrier_batch_v' in part for part in command), 'Previous carrier supervisor remains alive'
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    for item in finished:
        guard.wait()
        for name, mode in (('prepared_actual_exit', 'prepare'), ('audited_actual_exit', 'audit')):
            proof = checked(item[name])
            assert proof['actual_exit_code'] == 0 and proof['mode'] == mode
            assert proof['source_sha256'] == item['job']['sha256'] and proof['parent_observed_wait'] is True
            closed = checked(proof['native_closure'])
            assert closed['PID'] == proof['native_PID'] and closed['native_return_code'] == 0 and closed['watchdog_clean'] is True
    values = prior_hashes(plan['prior_actual_closure'])
    fresh = {item['job']['sha256'] for item in finished}
    assert not values.intersection(fresh)
    values.update(fresh)
    assert len(values) == plan['prior_covered_sources'] + len(finished)
    closure_path = AREA / f'KIRAAT_FLAC_carrier_batch{args.batch:02d}_actual_CLOSED0_v1.json'
    assert not closure_path.exists()
    closure = {'kind': 'ASR18_bounded_carrier_batch_actual_tool_CLOSED0_v1',
               'at_utc': now(), 'owner': '/root/cpu_data_preparation',
               'owned_session': args.owned_session, 'supervisor_PID': native['supervisor_PID'],
               'actual_exit_code': 0, 'actual_tool_chunk': args.actual_tool_chunk,
               'no_owned_carrier_child_running': True,
               'prior_actual_closure': plan['prior_actual_closure'],
               'native_phase_closure': binding(phase / 'native_phase_CLOSED0.json'),
               'completed_sources': binding(phase / 'completed_sources.json'),
               'covered_source_hashes': sorted(values), 'covered_source_files': len(values),
               'phase_source_files': len(finished), 'phase_new_payload_bytes': native['phase_new_payload_bytes'],
               'phase_paced_IO_bytes': native['phase_paced_IO_bytes'],
               'full729_coverage': False, 'TRAIN_admitted': False}
    write_json(closure_path, closure)
    original_plan = read_json(AREA / 'preparation_plan_v1.json')
    selected = []
    projected = 0
    unresolved_existing = []
    jobs = [job for job in original_plan['download_jobs'] if job['repository'] == 'serdarcaglar/kiraat'
            and Path(job['path']).name.startswith(('train-', 'rest-'))]
    assert len(jobs) == 729 and values.issubset({job['sha256'] for job in jobs})
    for job in sorted(jobs, key=lambda value: value['path']):
        guard.wait()
        if job['sha256'] in values:
            continue
        source = Path(job['destination'])
        receipt_path = source.with_suffix(source.suffix + '.receipt.json')
        if not source.exists() or not receipt_path.exists():
            continue
        receipt = read_json(receipt_path)
        assert source.stat().st_size == job['size'] and receipt['sha256'] == job['sha256']
        assert receipt['revision'] == job['revision']
        folder = CACHE / job['sha256']
        if (folder / 'receipt.json').exists():
            unresolved_existing.append(job['path'])
            continue
        if projected + job['size'] > 15_900_000_000:
            break
        selected.append({'job': job, 'source_signature': signature(source),
                         'source_receipt': binding(receipt_path),
                         'initial_payload_bytes': (folder / 'records.bin').stat().st_size if (folder / 'records.bin').exists() else 0})
        projected += job['size']
        if len(selected) == args.maximum_next_sources:
            break
    assert not unresolved_existing, 'Existing completed carrier lacks observed closure; reconcile, never overwrite/exclude'
    if not selected:
        print({'actual_closure': binding(closure_path), 'next_batch_pinned': False,
               'covered_sources': len(values), 'all729_actual_stage_proofs_available': len(values) == 729,
               'full_final_coverage_receipt_still_required': True})
        return
    assert projected * 6 + 32_000_000 < 96_000_000_000
    number = args.batch + 1
    successor = PROJECT / 'work/research/run_ASR18_FLAC_carrier_batch_v9.py'
    plan = {'kind': 'ASR18_separately_pinned_bounded_FLAC_batch_plan_v6',
            'created_at_utc': now(), 'owner': '/root/cpu_data_preparation',
            'phase': str(AREA / f'KIRAAT_FLAC_carrier_batch{number:02d}_v6'),
            'prior_actual_closure': binding(closure_path), 'prior_covered_sources': len(values),
            'selected': selected, 'projected_source_bytes': projected,
            'conservative_six_pass_IO_bytes': projected * 6,
            'maximum_new_payload_bytes': 16_000_000_000, 'maximum_IO_bytes': 96_000_000_000,
            'maximum_IO_bytes_s': 256 * 1024**2, 'aggregate_cache_cap_bytes': 400_000_000_000,
            'reserve_all_missing_fixed_downloads_plus_E100GB': True,
            'one_native_carrier_child': True,
            'ROOT_acknowledgment': binding(AREA / 'ROOT_KIRAAT_FLAC_sweep_acknowledgment_v2.json'),
            'declared_geometry_acknowledgment': binding(AREA / 'ROOT_KIRAAT_declared_geometry_audit_acknowledgment_v1.json'),
            'successor_source': binding(successor), 'no_GPU_or_idle_permission_gate': True,
            'TRAIN_admitted': False}
    target = AREA / f'KIRAAT_FLAC_carrier_batch{number:02d}_plan_v6.json'
    assert not target.exists()
    write_json(target, plan)
    print({'actual_closure': binding(closure_path), 'next_pinned_plan': binding(target),
           'selected_sources': len(selected), 'prior_covered_sources': len(values),
           'projected_source_bytes': projected, 'native_launch_performed': False})


if __name__ == '__main__':
    main()
