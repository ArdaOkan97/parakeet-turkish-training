"""Record one already launched, explicitly owned finite carrier phase; never launch work."""
import argparse
import hashlib
from pathlib import Path
import sys

import psutil
from ASR18_continuous_data_guard_v5 import DataIOGuard, AREA
from cpu_preparation_guard_v2 import now

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import read_json, write_json


def binding(path):
    path = Path(path).resolve()
    assert path.stat().st_size < 8_000_000
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--owned-session', type=int, required=True)
    parser.add_argument('--initial-tool-chunk', required=True)
    args = parser.parse_args()
    assert 6 <= args.batch <= 30 and args.owned_session > 0
    guard = DataIOGuard('finite_carrier_dispatch_metadata', heavy=False, maximum_bytes=4_000_000)
    guard.wait()
    status = read_json(AREA / 'KIRAAT_FLAC_carrier_current_status_v1.json')
    plan_path = AREA / f'KIRAAT_FLAC_carrier_batch{args.batch:02d}_plan_v6.json'
    plan_pin = binding(plan_path)
    assert status['pinned_plan'] == plan_pin
    plan = read_json(plan_path)
    source = Path(status['supervisor_source'])
    assert binding(source) == plan['successor_source']
    assert status['status'] != 'completed_declared_bounded_carrier_batch'
    process = psutil.Process(status['supervisor_PID'])
    assert process.is_running() and any(str(source) == arg or source.name == Path(arg).name
                                       for arg in process.cmdline())
    prior_pin = plan['prior_actual_closure']
    assert binding(prior_pin['path']) == prior_pin
    prior = read_json(prior_pin['path'])
    assert prior['actual_exit_code'] == 0 and prior['no_owned_carrier_child_running'] is True
    worker = PROJECT / 'work/research/prepare_ASR18_KIRAAT_FLAC_records_v7.py'
    target = AREA / f'owned_FLAC_carrier_batch{args.batch:02d}_dispatch_v8.json'
    assert not target.exists(), 'Preserve an existing ownership receipt'
    timestamp = now()
    receipt = {'kind': 'ASR18_owned_bounded_carrier_dispatch_v8',
               'owner': '/root/cpu_data_preparation', 'created_at_utc': timestamp,
               'owned_session': args.owned_session, 'supervisor_PID': process.pid,
               'initial_tool_chunk': args.initial_tool_chunk,
               'source': binding(source), 'worker_source': binding(worker),
               'pinned_plan': plan_pin, 'prior_actual_CLOSED0': prior_pin,
               'current_status_path': str(AREA / 'KIRAAT_FLAC_carrier_current_status_v1.json'),
               'one_native_child': True, 'full729_coverage': False, 'TRAIN_admitted': False}
    write_json(target, receipt)
    handoff_path = AREA / 'DATA_owner_compact_handoff_v4.json'
    handoff = read_json(handoff_path)
    handoff['updated_at_utc'] = timestamp
    handoff['owned_sessions']['carrier'].update(
        session=args.owned_session, supervisor_PID=process.pid, phase=args.batch, source=source.name)
    handoff['current_carrier_dispatch'] = binding(target)
    handoff['current_carrier_status_snapshot'] = status
    handoff['latest_actual_carrier_phase'] = prior_pin
    handoff['prepared_boundary_seal_and_pin_source'] = binding(
        PROJECT / 'work/research/seal_ASR18_carrier_batch_and_pin_one_v3.py')
    handoff['full_boundary_actions'][1] = (
        f'Carrier: consume OWN{args.owned_session} actual exit before sealing. On actual0 run '
        f'seal helperv3 --batch{args.batch} with actual session/toolchunk; admit only one next '
        'separately pinned <=32-source/16GB/96GB phase with supervisor8/worker7 and unchanged '
        'guard6. Native workload never autoextends. Worker7 computes independent full-payload '
        'SHA during every original-row/byte comparison, preserving every check while avoiding '
        'one redundant payload read. Final729 coverage still requires the genuine final producer '
        'and its observed owned-tool exit after full source inventory/catalog prerequisites.')
    write_json(handoff_path, handoff)
    print({'dispatch': binding(target), 'owned_session': args.owned_session,
           'supervisor_PID': process.pid, 'prior_covered_sources': plan['prior_covered_sources'],
           'selected_sources': len(plan['selected']), 'no_launch_performed_by_recorder': True})


if __name__ == '__main__':
    main()
