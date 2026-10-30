"""One pre-pinned finite carrier batch; no automatic batch extension."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

from ASR18_continuous_data_guard_v5 import DataIOGuard, ROOT, AREA
from ASR18_FLAC_carrier_guard_v3 import CACHE
from ASR18_FLAC_geometry_sidecar_v1 import GEOMETRY_CACHE
from cpu_preparation_guard_v2 import now

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import read_json, write_json

WORKER = PROJECT / 'work/research/complete_ASR18_legacy_FLAC_declared_geometry_v2.py'
RUNTIME = ROOT / 'scratch/CPU_validation_v2_env/Scripts/python.exe'


def binding(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def metadata(path):
    result = read_json(path)
    if result is None:
        raise RuntimeError('Required immutable receipt unavailable: ' + str(path))
    return result


def cache_bytes():
    return sum(path.stat().st_size for root in (CACHE, GEOMETRY_CACHE)
               for path in root.rglob('*') if path.is_file())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    assert plan_path.is_relative_to(ROOT.resolve())
    config = metadata(plan_path)
    plan_binding = binding(plan_path)
    phase = Path(config['phase']).resolve()
    assert phase.is_relative_to(ROOT.resolve()) and not phase.exists()
    prior = metadata(config['prior_actual_closure']['path'])
    assert binding(config['prior_actual_closure']['path']) == config['prior_actual_closure']
    assert prior['actual_exit_code'] == 0 and prior['no_owned_carrier_child_running'] is True
    assert 1 <= len(config['selected']) <= 32
    assert len({item['job']['sha256'] for item in config['selected']}) == len(config['selected'])
    assert config['maximum_new_payload_bytes'] == 0
    assert sum(item['existing_payload_bytes'] for item in config['selected']) <= 16_000_000_000
    assert config['maximum_IO_bytes'] == 96_000_000_000
    assert config['maximum_IO_bytes_s'] == 256 * 1024**2
    assert binding(Path(__file__)) == config['successor_source']
    assert binding(AREA / 'ROOT_KIRAAT_declared_geometry_audit_acknowledgment_v1.json') == config['declared_geometry_acknowledgment']
    guard = DataIOGuard('FLAC_phase_supervision_v6', heavy=False, maximum_bytes=32_000_000)
    guard.wait()
    phase.mkdir()
    sources = {str(path): binding(path) for path in (
        Path(__file__), WORKER,
        PROJECT / 'work/research/normalize_ASR18_FLAC_stage_provenance_v1.py',
        PROJECT / 'work/research/ASR18_FLAC_geometry_sidecar_v1.py',
        PROJECT / 'work/research/prepare_ASR18_KIRAAT_FLAC_records_v8.py',
        PROJECT / 'work/research/ASR18_FLAC_streaminfo_geometry_v1.py',
        PROJECT / 'work/research/ASR18_FLAC_carrier_guard_v3.py',
        PROJECT / 'work/research/ASR18_FLAC_carrier_guard_v6.py',
        PROJECT / 'work/research/ASR18_continuous_data_guard_v5.py',
        PROJECT / 'outputs/parakeet_turkish/atomic_io_retry_v1.py')}
    write_json(phase / 'admitted_source_pins.json', sources)
    original_plan = metadata(AREA / 'preparation_plan_v1.json')
    fixed_jobs = {job['sha256']: job for job in original_plan['download_jobs']}
    for item in config['selected']:
        assert item['job'] == fixed_jobs[item['job']['sha256']]
        assert item['job']['repository'] == 'serdarcaglar/kiraat'
        assert Path(item['job']['path']).name.startswith(('train-', 'rest-'))
    charged = 0
    finished = []
    started = time.monotonic()

    def growth():
        return sum(max(0, (CACHE / item['job']['sha256'] / 'records.bin').stat().st_size
                       - item['initial_payload_bytes']) for item in config['selected']
                   if (CACHE / item['job']['sha256'] / 'records.bin').exists())

    def publish(status, **extra):
        state = {'kind': 'ASR18_bounded_declared_geometry_completion_status_v1', 'at_utc': now(),
                 'supervisor_PID': os.getpid(), 'status': status,
                 'declared_selected_sources': len(config['selected']),
                 'phase_geometry_completed_sources': len(finished),
                 'prior_covered_sources': config['prior_covered_sources'],
                 'cumulative_covered_sources': config['prior_covered_sources'],
                 'prior_geometry_covered_sources': config['prior_geometry_covered_sources'],
                 'cumulative_geometry_covered_sources': config['prior_geometry_covered_sources']+len(finished),
                 'phase_new_payload_bytes': growth(),
                 'completed_child_paced_IO_bytes': charged,
                 'maximum_phase_new_payload_bytes': config['maximum_new_payload_bytes'],
                 'maximum_phase_IO_bytes': config['maximum_IO_bytes'],
                 'elapsed_seconds': time.monotonic() - started,
                 'carrier_IO_ceiling_bytes_s': config['maximum_IO_bytes_s'],
                 'pinned_plan': plan_binding, 'supervisor_source': str(Path(__file__).resolve()), 'full729_coverage': False,
                 'TRAIN_admitted': False, **extra}
        write_json(phase / 'status.json', state)
        write_json(AREA / 'KIRAAT_declared_geometry_completion_current_status_v1.json', state)

    publish('pinned_batch_before_work', active_child_PID=None, active_launcher_PID=None)
    for index, item in enumerate(config['selected']):
        assert (CACHE / item['job']['sha256'] / 'receipt.json').exists()
        preparation_proof = None
        for mode in ('declared_geometry_completion',):
            guard.wait()
            for path, source_pin in sources.items():
                assert binding(path) == source_pin, 'Do not mutate admitted source'
            assert binding(plan_path) == plan_binding, 'Do not mutate admitted plan'
            run = phase / f'{index:03d}_{mode}'
            run.mkdir()
            child_config = {**item, 'mode': mode, 'run_folder': str(run),
                            'cache_bytes_before': cache_bytes(),
                            'remaining_download_bytes_reserved': sum(job['size'] for job in original_plan['download_jobs'] if not Path(job['destination']).exists()),
                            'remaining_IO_bytes': config['maximum_IO_bytes'] - 32_000_000 - charged,
                            'remaining_new_payload_bytes': config['maximum_new_payload_bytes'] - growth(),
                            'expected_peak_private_bytes': 128_000_000}
            assert child_config['remaining_IO_bytes'] > 0 and child_config['remaining_new_payload_bytes'] >= 0
            config_path = run / 'config.json'
            write_json(config_path, child_config)
            with (run / 'stdout.log').open('wb') as log:
                child = subprocess.Popen([str(RUNTIME), '-X', 'utf8', '-B', str(WORKER), '--config', str(config_path)],
                                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         env=os.environ.copy(), cwd=PROJECT)
                last = 0
                while child.poll() is None:
                    if time.monotonic() - last > 2:
                        live = read_json(run / 'live.json', default={})
                        publish(mode, active_child_PID=live.get('PID'), active_launcher_PID=child.pid,
                                current_source=item['job']['path'], run_folder=str(run), child_live=live)
                        last = time.monotonic()
                    time.sleep(.25)
                actual = child.wait()
            native_path = run / 'native_closure.json'
            native = metadata(native_path) if native_path.exists() else None
            proof = {'kind': 'ASR18_actual_owned_FLAC_child_exit_v1', 'observed_at_utc': now(),
                     'native_PID': native['PID'] if native else None, 'launcher_PID': child.pid,
                     'mode': mode, 'actual_exit_code': actual, 'source_sha256': item['job']['sha256'],
                     'parent_observed_wait': True, 'worker_source': sources[str(WORKER)],
                     'native_closure': binding(native_path) if native else None}
            write_json(run / 'actual_owned_exit.json', proof)
            if actual != 0 or native is None:
                publish('stopped_preserved_prefix', active_child_PID=None, active_launcher_PID=None,
                        failed_mode=mode, actual_exit_code=actual)
                raise RuntimeError('Stopped at actual child boundary; preserve originals and prefix, no automatic retry')
            assert native['native_return_code'] == 0 and native['watchdog_clean'] is True
            charged += native['actual_paced_IO_bytes']
            assert charged + 32_000_000 <= config['maximum_IO_bytes']
            assert growth() <= config['maximum_new_payload_bytes'] and cache_bytes() <= 400_000_000_000
            finished.append({'job': item['job'], 'source_signature': item['source_signature'],
                             'source_receipt': item['source_receipt'],
                             'original_prepared_actual_exit': item['original_prepared_actual_exit'],
                             'original_audited_actual_exit': item['original_audited_actual_exit'],
                             'geometry_completion_actual_exit': binding(run / 'actual_owned_exit.json'),
                             'geometry_result': metadata(run / 'result.json')})
            write_json(phase / 'completed_sources.json', finished)
            publish('source_declared_geometry_completed_and_payload_reverified', active_child_PID=None,
                    active_launcher_PID=None, current_source=item['job']['path'])
            print({'source': item['job']['path'], 'phase_completed_sources': len(finished),
                   'phase_paced_IO_bytes': charged}, flush=True)
    assert len(finished) == len(config['selected'])
    publish('completed_declared_bounded_geometry_completion_phase', active_child_PID=None,
            active_launcher_PID=None, actual_exit_code=0)
    write_json(phase / 'native_phase_CLOSED0.json', {
        'kind': 'ASR18_native_bounded_declared_geometry_completion_CLOSED0_v1', 'at_utc': now(),
        'supervisor_PID': os.getpid(), 'native_return_code': 0, 'pinned_plan': binding(plan_path),
        'source_pins': sources, 'completed_sources': binding(phase / 'completed_sources.json'),
        'phase_source_files': len(finished), 'phase_new_payload_bytes': growth(),
        'phase_paced_IO_bytes': charged, 'full729_coverage': False, 'TRAIN_admitted': False})


if __name__ == '__main__':
    main()
