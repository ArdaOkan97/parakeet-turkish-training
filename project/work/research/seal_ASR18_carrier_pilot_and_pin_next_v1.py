"""Seal an observed pilot exit and pin one finite next batch; no native launch."""
import argparse
import datetime as dt
import hashlib
from pathlib import Path
import sys

import psutil

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import read_json, write_json

AREA = Path(r'E:\gpu_training_scratchpad\analysis\BPE_weak_then_clean_v1')
CACHE = Path(r'E:\gpu_training_scratchpad\cache\KIRAAT_original_FLAC_records_v1')


def binding(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def signature(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'mtime_ns': path.stat().st_mtime_ns}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actual-exit-code', type=int, required=True)
    parser.add_argument('--actual-tool-chunk', required=True)
    args = parser.parse_args()
    assert args.actual_exit_code == 0
    phase = AREA / 'KIRAAT_FLAC_carrier_continuation_v5'
    native = read_json(phase / 'native_phase_CLOSED0.json')
    assert native['native_return_code'] == 0 and native['original_pilot_sources'] == 32
    completed = read_json(phase / 'completed_sources.json')
    fixtures = read_json(phase / 'fixture_completion_verifications.json')
    assert len(completed) == 32 and len(fixtures) == 1
    for item in completed + fixtures:
        for field in ('prepared_actual_exit', 'audited_actual_exit'):
            proof_pin = item[field]
            assert binding(proof_pin['path']) == proof_pin
            proof = read_json(proof_pin['path'])
            assert proof['actual_exit_code'] == 0 and proof['source_sha256'] == item['job']['sha256']
    assert read_json(fixtures[0]['prepared_actual_exit']['path'])['mode'] == 'completion_verify'
    for process in psutil.process_iter(['pid', 'cmdline']):
        try:
            command = process.info['cmdline'] or []
            assert not any('prepare_ASR18_KIRAAT_FLAC_records_v6.py' in part for part in command), 'Carrier child remains alive'
            assert not any('run_ASR18_FLAC_carrier_continuation_v5.py' in part for part in command), 'Prior supervisor remains alive'
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    original_plan = read_json(AREA / 'preparation_plan_v1.json')
    covered = {item['job']['sha256'] for item in completed}
    fixture_sources = []
    for folder in CACHE.iterdir():
        receipt_path = folder / 'receipt.json'
        if not receipt_path.exists():
            continue
        receipt = read_json(receipt_path)
        source_hash = receipt['source']['source_sha256']
        if source_hash not in covered:
            fixture_sources.append({'source_sha256': source_hash, 'carrier_receipt': binding(receipt_path)})
            covered.add(source_hash)
    assert len(covered) == 37 and len(fixture_sources) == 5
    reservation_path = AREA / 'carrier_unobserved_stage_IO_reservation_v1.json'
    reservation = read_json(reservation_path)
    charged_upper = native['cumulative_original_pilot_paced_IO_bytes'] + reservation['conservative_reserved_IO_bytes'] + 32_000_000
    assert charged_upper <= 96_000_000_000
    milestone = {'kind': 'ASR18_original32_carrier_pilot_actual_tool_CLOSED0_v1',
                 'at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                 'owner': '/root/cpu_data_preparation', 'owned_session': 68048,
                 'supervisor_PID': native['supervisor_PID'], 'actual_exit_code': 0,
                 'actual_tool_chunk': args.actual_tool_chunk,
                 'no_owned_carrier_child_running': True,
                 'native_phase_closure': binding(phase / 'native_phase_CLOSED0.json'),
                 'completed_sources': binding(phase / 'completed_sources.json'),
                 'fixture_completion_verifications': binding(phase / 'fixture_completion_verifications.json'),
                 'original32_sources': 32, 'existing_fixture_sources': 5,
                 'audited_total_unique_sources': 37,
                 'fixture_source_bindings': fixture_sources,
                 'original_preparation_outcomes_preserved': True,
                 'unknown_original_IO_reservation': binding(reservation_path),
                 'known_completed_child_paced_IO_bytes': native['cumulative_original_pilot_paced_IO_bytes'],
                 'conservative_pilot_IO_upper_charge_bytes': charged_upper,
                 'original_pilot_payload_bytes': native['original_pilot_new_payload_bytes'],
                 'full729_coverage': False, 'TRAIN_admitted': False}
    closure_path = AREA / 'KIRAAT_FLAC_original32_pilot_actual_CLOSED0_v1.json'
    assert not closure_path.exists()
    write_json(closure_path, milestone)
    selected = []
    projected = 0
    for job in sorted(original_plan['download_jobs'], key=lambda item: item['path']):
        if job['repository'] != 'serdarcaglar/kiraat' or not Path(job['path']).name.startswith(('train-', 'rest-')):
            continue
        if job['sha256'] in covered:
            continue
        source = Path(job['destination'])
        receipt_path = source.with_suffix(source.suffix + '.receipt.json')
        if not source.exists() or not receipt_path.exists():
            continue
        receipt = read_json(receipt_path)
        assert source.stat().st_size == job['size']
        assert receipt['sha256'] == job['sha256'] and receipt['revision'] == job['revision']
        if projected + job['size'] > 15_900_000_000:
            break
        folder = CACHE / job['sha256']
        assert not (folder / 'receipt.json').exists()
        selected.append({'job': job, 'source_signature': signature(source),
                         'source_receipt': binding(receipt_path),
                         'initial_payload_bytes': (folder / 'records.bin').stat().st_size if (folder / 'records.bin').exists() else 0})
        projected += job['size']
        if len(selected) == 32:
            break
    assert selected and len(selected) <= 32 and projected * 6 < 96_000_000_000 - 32_000_000
    successor = PROJECT / 'work/research/run_ASR18_FLAC_carrier_batch_v6.py'
    plan = {'kind': 'ASR18_separately_pinned_bounded_FLAC_batch_plan_v6',
            'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'owner': '/root/cpu_data_preparation',
            'phase': str(AREA / 'KIRAAT_FLAC_carrier_batch02_v6'),
            'prior_actual_closure': binding(closure_path),
            'prior_covered_sources': 37, 'selected': selected,
            'projected_source_bytes': projected, 'conservative_six_pass_IO_bytes': projected * 6,
            'maximum_new_payload_bytes': 16_000_000_000,
            'maximum_IO_bytes': 96_000_000_000, 'maximum_IO_bytes_s': 256 * 1024**2,
            'aggregate_cache_cap_bytes': 400_000_000_000,
            'reserve_all_missing_fixed_downloads_plus_E100GB': True,
            'one_native_carrier_child': True,
            'ROOT_acknowledgment': binding(AREA / 'ROOT_KIRAAT_FLAC_sweep_acknowledgment_v2.json'),
            'successor_source': binding(successor),
            'no_GPU_or_idle_permission_gate': True, 'TRAIN_admitted': False}
    plan_path = AREA / 'KIRAAT_FLAC_carrier_batch02_plan_v6.json'
    assert not plan_path.exists()
    write_json(plan_path, plan)
    print({'actual_pilot_closure': binding(closure_path), 'next_pinned_plan': binding(plan_path),
           'selected_sources': len(selected), 'projected_source_bytes': projected})


if __name__ == '__main__':
    main()
