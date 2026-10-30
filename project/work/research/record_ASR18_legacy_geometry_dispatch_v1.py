"""Record one actually launched finite legacy-header supervisor, never launch it."""
import argparse
from pathlib import Path
import sys
import psutil

from ASR18_continuous_data_guard_v5 import AREA
from cpu_preparation_guard_v2 import now
from normalize_ASR18_FLAC_stage_provenance_v1 import binding, checked
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, required=True)
    parser.add_argument('--owned-session', type=int, required=True)
    parser.add_argument('--initial-tool-chunk', required=True)
    args = parser.parse_args()
    phase = AREA / f'KIRAAT_declared_geometry_completion{args.phase:02d}_v1'
    state = read_json(phase / 'status.json')
    plan = checked(state['pinned_plan'])
    source = binding(state['supervisor_source'])
    assert {key: source[key] for key in ('path', 'bytes', 'sha256')} == plan['successor_source']
    process = psutil.Process(state['supervisor_PID'])
    assert any(Path(source['path']).name in argument for argument in process.cmdline())
    assert args.owned_session > 0 and args.initial_tool_chunk
    target = AREA / f'owned_declared_geometry_completion{args.phase:02d}_dispatch_v1.json'
    assert not target.exists()
    value = dict(kind='ASR18_owned_finite_legacy_geometry_completion_dispatch_v1', owner='/root/cpu_data_preparation',
        at_utc=now(), owned_session=args.owned_session, supervisor_PID=process.pid,
        initial_tool_chunk_id=args.initial_tool_chunk, pinned_plan=state['pinned_plan'],
        supervisor_source=source, status_path=str(AREA / 'KIRAAT_declared_geometry_completion_current_status_v1.json'),
        read_dynamic_child_PID_from_status=True, actual_started=True, first_observed_state=state,
        all729_geometry_complete=False, TRAIN_admitted=False)
    write_json(target, value)
    handoff_path = AREA / 'DATA_owner_compact_handoff_v4.json'
    handoff = read_json(handoff_path)
    handoff['updated_at_utc'] = now()
    handoff['owned_sessions']['declared_geometry_completion'] = dict(session=args.owned_session,
        supervisor_PID=process.pid, source=source['path'], phase=args.phase,
        status_path=value['status_path'], read_dynamic_child_PID_from_status=True)
    handoff['owned_sessions']['carrier']['process_alive'] = False
    handoff['current_declared_geometry_completion_dispatch'] = binding(target)
    handoff['current_declared_geometry_completion_snapshot'] = state
    handoff['remaining_legacy_scope_acknowledged'] = binding(AREA / 'ROOT_remaining_legacy_declared_geometry_acknowledgment_v2.json')
    handoff['full_boundary_actions'][1] = ('Finish current finite legacy-header phase; consume its exact owned tool exit. '
        'After actual0 seal its frontier and run the independent complete metadata/mapping/sidecar verifier. '
        'Only after both actual0 closures pin/admit ONE next <=32/16GBinput/96GBIO phase within acknowledged197. '
        'No native autoextension. Carrier12 remains pinned for after legacy completion; download/writers continue.')
    write_json(handoff_path, handoff)
    print(dict(dispatch=binding(target), owned_session=args.owned_session,
        supervisor_PID=process.pid, phase_completed_sources=state['phase_geometry_completed_sources'],
        positive_current_child_IO_bytes=state.get('child_live', {}).get('paced_IO_bytes')), flush=True)


if __name__ == '__main__':
    main()
