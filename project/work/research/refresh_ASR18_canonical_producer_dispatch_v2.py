"""Mutable current pointers; canonical receipts and historical outcomes stay immutable."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import psutil

from ASR18_continuous_data_guard_v5 import AREA
from cpu_preparation_guard_v2 import now

P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import read_json, write_json

PROOFS = {
    'final_carrier_coverage': 'KIRAAT_FLAC_carrier_coverage_owned_tool_actual_CLOSED0_v1.json',
    'final_declared_geometry': 'ASR18_final_declared_training_geometry_owned_tool_actual_CLOSED0_v1.json',
    'final_CPU_resource_envelope': 'ASR18_final_declared_CPU_resource_envelope_owned_tool_actual_CLOSED0_v1.json',
    'final_two_pass_with_declared_geometry': 'ASR18_final_two_pass_population_metadata_with_declared_geometry_owned_tool_actual_CLOSED0_v1.json',
    'final_raw_DATA_population': 'ASR18_final_data_population_for_bounded_preparation_owned_tool_actual_CLOSED0_v1.json',
}


def pin(path):
    path = Path(path).resolve()
    stat = path.stat()
    assert stat.st_size < 32_000_000
    return dict(path=str(path), bytes=stat.st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), mtime_ns=stat.st_mtime_ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--role', choices=PROOFS)
    ap.add_argument('--session', type=int)
    ap.add_argument('--PID', type=int)
    ap.add_argument('--source')
    ap.add_argument('--status')
    args = ap.parse_args()
    if args.role:
        assert args.session > 0 and args.PID > 0 and args.source and args.status
        assert str(Path(args.source).resolve()) in psutil.Process(args.PID).cmdline()
    hpath = AREA / 'DATA_owner_compact_handoff_v4.json'
    h = read_json(hpath)
    closed = {}
    for role, name in PROOFS.items():
        path = AREA / name
        if not path.exists():
            continue
        proof = read_json(path)
        assert proof['kind'] == 'ASR18_source_bound_actual_tool_exit_v1'
        assert proof['owner'] == '/root/cpu_data_preparation'
        assert type(proof['actual_exit_code']) is int and proof['actual_exit_code'] == 0
        assert proof['actual_tool_chunk_id'] and proof['native_PID'] > 0
        bound = pin(proof['published_marker']['path'])
        assert all(bound[k] == v for k, v in proof['published_marker'].items())
        closed[role] = dict(actual_wrapper=pin(path), published_marker=bound)
        h['owned_sessions'].setdefault(role, {}).update(session=proof['owned_session'],
            PID=proof['native_PID'], process_alive=False, actual_exit_code=0,
            actual_tool_chunk_id=proof['actual_tool_chunk_id'], actual_wrapper=pin(path))
    live = None
    if args.role:
        assert args.role not in closed
        live = dict(role=args.role, session=args.session, PID=args.PID, source=args.source,
            source_binding=pin(args.source), status_path=args.status, process_alive=True,
            actual_exit_code=None, metadata_only=True, GPU_used=False, final_metadata_producer=True)
        h['owned_sessions'][args.role] = live
    h.update(updated_at_utc=now(), canonical_actual_closed=closed,
        all729_carrier_final_coverage='final_carrier_coverage' in closed,
        final_metadata_chain_complete=len(closed) == len(PROOFS), TRAIN_admitted=False)
    h['current_final_metadata_producer'] = live
    write_json(hpath, h)
    subprocess.run([sys.executable, '-B', str(P / 'work/research/refresh_ASR18_data_dispatch_v4.py')], check=True)
    dpath = AREA / 'current_data_dispatch_v2.json'
    d = read_json(dpath)
    d['heavy_successors'] = [x for x in d.get('heavy_successors', []) if not x.get('final_metadata_producer')]
    if live:
        d['heavy_successors'].append(live)
    d.update(updated_at_utc=now(), final_metadata_producer=live,
        final_metadata_IO_running=live is not None, heavy_IO_running=False,
        carrier_stage_pairs_actual_complete=729, canonical_actual_closed=closed,
        final_metadata_chain_complete=len(closed) == len(PROOFS), TRAIN_admitted=False)
    write_json(dpath, d)
    for name in ('current_data_readiness_v1.json', 'current_data_preparation_status_v2.json'):
        path = AREA / name
        value = read_json(path)
        value.update(updated_at_utc=now(), canonical_actual_closed=closed,
            current_final_metadata_producer=live,
            all729_carrier_final_coverage='final_carrier_coverage' in closed,
            final_metadata_chain_complete=len(closed) == len(PROOFS),
            TRAIN_admitted=False, production_training_ready=False)
        write_json(path, value)
    print(dict(updated_at_utc=now(), canonical_actual_closed=closed, current_final_metadata_producer=live))


if __name__ == '__main__':
    main()
