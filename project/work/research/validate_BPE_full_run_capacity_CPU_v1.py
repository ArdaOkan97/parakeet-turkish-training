"""Bounded small-metadata capacity controls; no native/audio or cleanup."""
import argparse
from copy import deepcopy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P/'outputs/parakeet_turkish'))
from bpe_full_run_capacity_v1 import E, allocation, derive, from_pinned_metadata, small_metadata, require
from fastmss_row_serial_cpu_v6 import environment, resource_guard, admit
from atomic_io_retry_v1 import write_json


def guard():
    sample = resource_guard()
    require(sample['worker_private_memory']['private_bytes'] < 100_000_000, 'CPU100MB fixture cap')
    return sample


def pin(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main(config_path, output):
    began = time.monotonic()
    environment()
    admission = admit(100_000_000, guard=guard)
    config = json.loads(Path(config_path).read_text())
    require(config['kind'] == 'ROOT_capacity_small_metadata_fixture_configuration_v1', 'Exact fixture configuration')
    for source in config['sources']:
        require(pin(source['path']) == source, 'Frozen fixture/source changed')
    output = Path(output).resolve()
    require(output.is_relative_to(E.resolve()) and not output.exists(), 'Fresh E-only report namespace')
    output.mkdir()
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    sectors = ctypes.c_ulong(); sector_bytes = ctypes.c_ulong()
    free_clusters = ctypes.c_ulong(); all_clusters = ctypes.c_ulong()
    require(kernel.GetDiskFreeSpaceW('E:\\', ctypes.byref(sectors), ctypes.byref(sector_bytes),
        ctypes.byref(free_clusters), ctypes.byref(all_clusters)), 'Actual E allocation unit')
    disk = {'allocation_unit_bytes': sectors.value*sector_bytes.value,
        'free_bytes': shutil.disk_usage(E).free, 'observed_at_unix_seconds': time.time(),
        'scope': 'Actual E only; existing artifacts already included in observed free bytes'}
    result = from_pinned_metadata(inputs=config['inputs'], limits=config['limits'], disk=disk)
    values = {name: small_metadata(item) for name, item in config['inputs'].items()}
    checks = []; samples = [guard()]
    def check(name, value):
        require(value, name); checks.append(name); samples.append(guard())
    def reject(name, operation):
        try:
            operation()
        except (ValueError, TypeError, KeyError):
            checks.append(name); samples.append(guard()); return
        raise ValueError('Expected rejection: '+name)
    check('Exact401072 full native updates', result['total_native_updates'] == 401072)
    check('Both broad-pass tails retained in normal publication bound', result['maximum_normal_publications'] == 95230)
    check('Worst physical frame bounds preserve final zero-masked frame',
        result['physical_feature_frame_bounds'] == {'weak': 15925, 'replay': 2609, 'cleaner': 2897})
    check('Four weak and20 cleaner complete updates fit conditional window bound',
        result['minimum_full_normal_window_updates'] == {'weak': 4, 'cleaner': 20})
    check('Pointer allocation rounded independently', result['allocation_rounded_artifact_caps']['pointer']
        == allocation(config['limits']['artifact_caps_bytes']['pointer'], disk['allocation_unit_bytes']))
    check('Single byte consumes one real nonzero cluster', allocation(1, disk['allocation_unit_bytes']) == disk['allocation_unit_bytes'])
    changed = deepcopy(values); changed['population']['retained_rows'] = True
    reject('Boolean population count rejected', lambda: derive(**changed, limits=config['limits'], disk=disk))
    changed = deepcopy(values); changed['geometry']['unknown_invalid_counts']['retained_unresolved_rows'] = 1
    reject('Unresolved geometry rejected', lambda: derive(**changed, limits=config['limits'], disk=disk))
    caps = deepcopy(config['limits']); caps['feature_cache_bytes'] = 0
    reject('Zero feature capacity rejected', lambda: derive(**values, limits=caps, disk=disk))
    bad = deepcopy(config['inputs']['population']); bad['sha256'] = '0'*64
    reject('Wrong small metadata hash rejected', lambda: small_metadata(bad))
    caps = deepcopy(config['limits']); caps['maximum_emergency_publications'] += 1
    more = derive(**values, limits=caps, disk=disk)
    check('Additional emergency consumes retained publication and auxiliary space',
        more['maximum_publications'] == result['maximum_publications']+1
        and more['whole_required_bytes_including_reserve'] > result['whole_required_bytes_including_reserve'])
    caps = deepcopy(config['limits']); caps['maximum_retained_owned_directories'] += result['maximum_normal_publications']
    per_window_dirs = derive(**values, limits=caps, disk=disk)
    check('Per-window directories explicitly consume cluster-rounded space',
        per_window_dirs['whole_required_bytes_including_reserve']-result['whole_required_bytes_including_reserve']
        == result['maximum_normal_publications']*disk['allocation_unit_bytes'])
    check('Arithmetic cannot declare launch ready', result['launch_ready'] is False
        and result['production_enabled'] is False and not result['actual_audio_native_GPU_or_storage_lifecycle_proved'])
    write_json(output/'capacity.json', result)
    write_json(output/'per_window_directory_counterexample.json', per_window_dirs)
    report = {'kind': 'ASR18_bounded_small_metadata_full_run_capacity_controls_v1',
        'passed': True, 'producer': {'pid': os.getpid(), 'ppid': os.getppid()},
        'source': pin(__file__), 'configuration': pin(config_path), 'checks': checks,
        'capacity': pin(output/'capacity.json'),
        'per_window_directory_counterexample': pin(output/'per_window_directory_counterexample.json'),
        'observed_private_peak_bytes': max(v['worker_private_memory']['private_bytes'] for v in samples),
        'admission': admission, 'wall_seconds': time.monotonic()-began,
        'new_database_audio_payload_or_GPU_reads': 0,
        'capacity_or_production_policy_accepted': False}
    write_json(output/'report.json', report)
    print(json.dumps({'passed': True, 'checks': len(checks), 'report': pin(output/'report.json')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args.config, args.output)
