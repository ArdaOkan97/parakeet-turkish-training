"""Update only mutable DATA pointers from observed phase exits and live producers."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from ASR18_continuous_data_guard_v5 import AREA
from cpu_preparation_guard_v2 import now

P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import read_json, write_json


def pin(path):
    path = Path(path).resolve()
    s = path.stat()
    assert s.st_size < 32_000_000
    return dict(path=str(path), bytes=s.st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), mtime_ns=s.st_mtime_ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--role', required=True)
    ap.add_argument('--session', type=int, required=True)
    ap.add_argument('--PID', type=int, required=True)
    ap.add_argument('--source', required=True)
    ap.add_argument('--status', required=True)
    args = ap.parse_args()
    assert args.session > 0 and args.PID > 0
    hpath = AREA / 'DATA_owner_compact_handoff_v4.json'
    h = read_json(hpath)
    phase = read_json(AREA / 'KIRAAT_FLAC_carrier_batch24_actual_CLOSED0_v1.json')
    assert phase['actual_exit_code'] == 0 and phase['no_owned_carrier_child_running'] is True
    h['owned_sessions']['carrier'].update(session=1316, supervisor_PID=37620, phase=24,
        actual_exit_code=0, actual_tool_chunk_id='f3846f', process_alive=False)
    h['owned_sessions']['declared_geometry_completion'].update(session=36246,
        supervisor_PID=34856, phase=8, actual_exit_code=0, actual_tool_chunk_id='dd510a', process_alive=False)
    h['latest_actual_carrier_phase'] = pin(AREA / 'KIRAAT_FLAC_carrier_batch24_actual_CLOSED0_v1.json')
    h['current_carrier_status_snapshot'] = read_json(AREA / 'KIRAAT_FLAC_carrier_current_status_v1.json')
    role = dict(session=args.session, PID=args.PID, source=args.source,
        status_path=args.status, process_alive=True, actual_exit_code=None,
        source_binding=pin(args.source), metadata_only=True, GPU_used=False)
    h['owned_sessions'][args.role] = role
    h['updated_at_utc'] = now()
    h['full_boundary_actions'] = [x for x in h['full_boundary_actions'] if not x.startswith(('Carrier: consume OWN', 'Supervise SAME current carrier'))]
    h['full_boundary_actions'].insert(1, 'All729 source carrier stage pairs and all legacy declared-sidecar stages have actual observed0; no phase25 is selected. Canonical full coverage/envelope/geometry-bound two-pass/raw DATA producer chain remains separately guarded and actual-exit-bound; no TRAIN admission.')
    write_json(hpath, h)
    subprocess.run([sys.executable, '-B', str(P / 'work/research/refresh_ASR18_data_dispatch_v4.py')], check=True)
    dpath = AREA / 'current_data_dispatch_v2.json'
    d = read_json(dpath)
    d['updated_at_utc'] = now()
    d['final_metadata_producer'] = role
    d['heavy_successors'] = [x for x in d.get('heavy_successors', []) if not x.get('final_metadata_producer')]
    d['heavy_successors'].append(dict(role, final_metadata_producer=True, role=args.role))
    d['heavy_IO_running'] = False
    d['final_metadata_IO_running'] = True
    d['carrier_stage_pairs_actual_complete'] = 729
    write_json(dpath, d)
    eta_path = AREA / 'current_data_completion_ETA_v1.json'
    eta = read_json(eta_path)
    eta.update(updated_at_utc=now(), remaining_network_bytes=0,
        remaining_carrier_source_stage_pairs=0, remaining_legacy_declared_sidecar_sources=0,
        canonical_metadata_finalization_in_progress=args.role,
        full_DATA_readiness=False, TRAIN_admitted=False,
        remaining_completion_ETA_seconds=None,
        ETA_note='Original downloads and all729 carrier/sidecar native source stages have actual closures. Canonical metadata producers and independent downstream review remain; no transfer ETA is a launch-readiness estimate.')
    write_json(eta_path, eta)
    print(json.dumps(dict(updated_at_utc=now(), final_metadata_producer=role, carrier_stage_pairs_actual_complete=729)))


if __name__ == '__main__':
    main()
