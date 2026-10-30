"""Mutable compact pointers after one actual carrier boundary; no worker launches."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone, timedelta
from ASR18_continuous_data_guard_v5 import DataIOGuard, AREA
P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, required=True)
parser.add_argument('--owned-session', type=int, required=True)
args = parser.parse_args()
guard = DataIOGuard('post_metadata_carrier_boundary_pointers', heavy=False, maximum_bytes=8_000_000)
guard.wait()


def read(name):
    path = AREA / name
    assert path.stat().st_size < 8_000_000
    guard.wait()
    return json.loads(path.read_text(encoding='utf-8-sig'))


def pin(path):
    path = Path(path)
    stat = path.stat()
    assert stat.st_size < 8_000_000
    guard.pace(stat.st_size)
    return dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


h = read('DATA_owner_compact_handoff_v4.json')
c = read('KIRAAT_FLAC_carrier_current_status_v1.json')
assert h['owned_sessions']['carrier']['session'] == args.owned_session
assert h['owned_sessions']['carrier']['phase'] == args.batch
assert h['owned_sessions']['carrier']['supervisor_PID'] == c['supervisor_PID']
assert c['pinned_plan']['path'] == str(AREA / f'KIRAAT_FLAC_carrier_batch{args.batch:02d}_plan_v6.json')
current = c['cumulative_covered_sources']
remaining = 729 - current
assert 0 <= remaining <= 403
eta = read('current_data_completion_ETA_v1.json')
seconds = remaining * eta['carrier']['observed_seconds_per_source']
finish = (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()
eta.update(updated_at_utc=datetime.now(timezone.utc).isoformat(), carrier_sources_complete=current,
    declared_geometry_sources_complete=current, remaining_carrier_sources=remaining,
    conditional_remaining_carrier_seconds=seconds,
    all729_carrier_and_declared_geometry_conditional_ETA_UTC=finish,
    conditional_carrier_plus_declared_work_finish_UTC=finish)
eta['carrier'].update(actual_closed_sources=c['prior_covered_sources'],
    actual_individual_child_audited_sources=current, remaining_sources=remaining,
    conditional_remaining_hours=seconds / 3600, current_outer_phase_actual_exit_pending=True)
write_json(AREA / 'current_data_completion_ETA_v1.json', eta)
h.update(updated_at_utc=datetime.now(timezone.utc).isoformat(), current_carrier_status_snapshot=c,
    conditional_separate_completion_ETA=pin(AREA / 'current_data_completion_ETA_v1.json'))
h['full_boundary_actions'][2] = (
    'Supervise SAME current carrier owned session/supervisor in owned_sessions.carrier unchanged. Consume actual exit; '
    'only actual0 permits seal_ASR18_carrier_batch_and_pin_one_v4.py to seal/pin ONE separately capped successor. '
    'Finite manual admissions only, one child, no GPU permit/rate/concurrency change.')
write_json(AREA / 'DATA_owner_compact_handoff_v4.json', h)
pointer_path = P / 'outputs/parakeet_turkish/current_CPU_data_preparation_v1.json'
pointer = json.loads(pointer_path.read_text(encoding='utf-8-sig'))
pointer['ASR18_BPE_weak_then_clean_DATA'].update(updated_at_utc=datetime.now(timezone.utc).isoformat(),
    current_carrier_owned_session=args.owned_session, current_carrier_supervisor_PID=c['supervisor_PID'],
    carrier_sources_individually_audited_at_observation=current)
write_json(pointer_path, pointer)
print(dict(current_carrier_phase=args.batch, owned_session=args.owned_session,
    supervisor_PID=c['supervisor_PID'], individual_source_pairs=current,
    conditional_carrier_plus_declared_ETA_UTC=finish, final_producer_and_native_admission_time_excluded=True))
