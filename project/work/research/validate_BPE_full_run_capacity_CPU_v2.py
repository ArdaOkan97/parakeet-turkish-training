"""Only the new directory-growth delta; prior13 controls are not rerun."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time

P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P/'outputs/parakeet_turkish'))
sys.path.insert(0, str(P/'work/research'))
from bpe_full_run_capacity_v1 import E, derive as derive1, small_metadata, require
from bpe_full_run_capacity_v2 import derive, from_pinned_metadata
from validate_BPE_full_run_capacity_CPU_v1 import guard, pin
from fastmss_row_serial_cpu_v6 import environment, admit
from atomic_io_retry_v1 import write_json


def main(config_path, output):
    began = time.monotonic(); environment(); admission = admit(100_000_000, guard=guard)
    config = json.loads(Path(config_path).read_text())
    require(config['kind'] == 'ROOT_capacity_directory_growth_delta_configuration_v2', 'Exact new delta config')
    for source in config['sources']:
        require(pin(source['path']) == source, 'Frozen source changed')
    output = Path(output).resolve()
    require(output.is_relative_to(E.resolve()) and not output.exists(), 'Fresh E-only delta namespace')
    output.mkdir()
    old = small_metadata(config['prior_capacity'])
    disk = dict(old['disk_observation'], free_bytes=shutil.disk_usage(E).free,
        observed_at_unix_seconds=time.time())
    # Use the prior ACTUAL E allocation-unit observation; never guess or inspect
    # another volume. A production admission must observe it freshly again.
    require(disk['allocation_unit_bytes'] == 524288, 'Pinned actual E512KiB observation')
    new = from_pinned_metadata(inputs=config['inputs'], limits=config['limits'], disk=disk)
    values = {key: small_metadata(value) for key, value in config['inputs'].items()}
    base = derive1(**values, limits=config['limits'], disk=disk)
    checks = []
    def check(name, value):
        guard(); require(value, name); checks.append(name)
    check('Exact extra4GiB included in total and margin',
        new['whole_required_bytes_including_reserve']-base['whole_required_bytes_including_reserve'] == 4*1024**3
        and base['conditional_margin_bytes']-new['conditional_margin_bytes'] == 4*1024**3)
    check('Base and entry directory components are distinct',
        new['directory_base_allocation_bytes']+new['directory_entry_growth_bytes'] == new['directory_allocation_bytes'])
    check('Original scientific updates/publication bounds unchanged',
        new['total_native_updates'] == 401072 and new['maximum_publications'] == 95742)
    limits = deepcopy(config['limits']); limits['maximum_directory_entry_growth_bytes'] = 1
    try:
        derive(**values, limits=limits, disk=disk)
    except ValueError:
        checks.append('Unaccounted directory-entry growth rejected')
    else:
        raise ValueError('Missing growth rejection')
    check('Conditional arithmetic is never launch readiness', new['launch_ready'] is False and new['production_enabled'] is False)
    write_json(output/'capacity.json', new)
    sample = guard()
    report = {'kind': 'ASR18_small_metadata_directory_growth_delta_controls_v2', 'passed': True,
        'source': pin(__file__), 'configuration': pin(config_path), 'capacity': pin(output/'capacity.json'),
        'prior13_controls_reexecuted': False, 'checks': checks, 'admission': admission,
        'worker_private_snapshot_bytes': sample['worker_private_memory']['private_bytes'],
        'wall_seconds': time.monotonic()-began, 'actual_native_audio_payload_or_cleanup': False,
        'production_enabled': False, 'whole_namespace_layout_or_capacity_policy_approved': False}
    write_json(output/'report.json', report)
    print(json.dumps({'passed': True, 'checks': len(checks), 'report': pin(output/'report.json')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); main(a.config, a.output)
