"""Pin exactly one finite successor from the acknowledged197-source frontier."""
import argparse
from pathlib import Path
import sys

from ASR18_continuous_data_guard_v5 import DataIOGuard, AREA
from cpu_preparation_guard_v2 import now
from normalize_ASR18_FLAC_stage_provenance_v1 import binding, checked
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import read_json, write_json


def base(pin):
    return {key: pin[key] for key in ('path', 'bytes', 'sha256')}


def legacy_chain(path):
    value = read_json(path)
    assert value['kind'] == 'ASR18_bounded_legacy_geometry_completion_actual_tool_CLOSED0_v1'
    assert type(value['actual_exit_code']) is int and value['actual_exit_code'] == 0
    assert value['no_owned_carrier_child_running'] is True and value['actual_tool_chunk_id']
    plan = checked(value['pinned_plan'])
    prior = plan.get('previous_legacy_actual_closure')
    completed = legacy_chain(prior['path']) if prior else {}
    if prior:
        checked(prior)
    for item in checked(value['completed_sources']):
        sha = item['job']['sha256']
        assert sha not in completed
        completed[sha] = item
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, required=True)
    parser.add_argument('--current-carrier-actual', required=True)
    parser.add_argument('--last-legacy-actual', required=True)
    parser.add_argument('--last-legacy-review-actual', required=True)
    args = parser.parse_args()
    assert 2 <= args.phase <= 8
    guard = DataIOGuard('pin_one_legacy_geometry_successor', heavy=False, maximum_bytes=16_000_000)
    guard.wait()
    ack_path = AREA / 'ROOT_remaining_legacy_declared_geometry_acknowledgment_v2.json'
    ack = read_json(ack_path)
    assert ack['acknowledged'] is True and ack['owner'] == '/root'
    frontier_path = Path(ack['frontier']['path'])
    assert binding(frontier_path)['sha256'] == ack['frontier']['sha256']
    frontier = read_json(frontier_path)
    assert frontier['remaining_sources_count'] == 197
    allowed = {item['job']['sha256']: item for item in frontier['selected']}
    assert len(allowed) == 197
    carrier = read_json(args.current_carrier_actual)
    assert carrier['kind'] == 'ASR18_bounded_carrier_batch_actual_tool_CLOSED0_v1'
    assert carrier['actual_exit_code'] == 0 and carrier['no_owned_carrier_child_running'] is True
    assert carrier['actual_tool_chunk']
    covered_carriers = set(carrier['covered_source_hashes'])
    assert set(allowed).issubset(covered_carriers)
    previous = read_json(args.last_legacy_actual)
    previous_pin = binding(args.last_legacy_actual)
    completed = legacy_chain(args.last_legacy_actual)
    completed_from197 = set(completed).intersection(allowed)
    assert len(set(completed) - set(allowed)) == 32, 'The original independently reviewed legacy32 remains separate'
    proof = read_json(args.last_legacy_review_actual)
    assert proof['kind'] == 'ASR18_source_bound_actual_tool_exit_v1'
    assert proof['owner'] == '/root/cpu_data_preparation' and proof['actual_exit_code'] == 0
    assert proof['actual_tool_chunk_id']
    report = checked(proof['published_marker'])
    assert report['passed'] is True and report['actual_phase'] == previous_pin
    assert report['source_count'] == previous['phase_completed_sources']
    pending = [item for sha, item in allowed.items() if sha not in completed_from197]
    pending.sort(key=lambda item: item['job']['path'])
    selected, total = [], 0
    for item in pending:
        if len(selected) >= 32 or total + item['existing_payload_bytes'] > 16_000_000_000:
            break
        selected.append(item)
        total += item['existing_payload_bytes']
    if not selected:
        assert len(completed_from197) == 197
        print({'remaining197_completed': True, 'no_native_worker_launched': True}, flush=True)
        return
    prepared = AREA / 'legacy_declared_geometry_completion02_selection_prepared_v1.json'
    acknowledged_selection = read_json(prepared)
    assert binding(prepared)['sha256'] == ack['next_prepared_selection']['sha256']
    for key in ('worker_source', 'supervisor_source'):
        assert binding(acknowledged_selection[key]['path']) == acknowledged_selection[key]
    if args.phase == 2:
        assert not completed_from197 and selected == acknowledged_selection['selected']
    else:
        assert len(completed_from197) == 32 * (args.phase - 2)
    current_carrier_plan = checked(carrier['native_phase_closure'])
    source_plan = checked(current_carrier_plan['pinned_plan'])
    phase = AREA / f'KIRAAT_declared_geometry_completion{args.phase:02d}_v1'
    out = AREA / f'KIRAAT_declared_geometry_completion{args.phase:02d}_plan_v2.json'
    assert not phase.exists() and not out.exists()
    prior_path = args.current_carrier_actual if carrier['at_utc'] > previous['at_utc'] else args.last_legacy_actual
    plan = {**source_plan, 'kind': 'ASR18_separately_pinned_bounded_legacy_geometry_completion_plan_v2',
        'created_at_utc': now(), 'phase': str(phase), 'selected': selected,
        'prior_actual_closure': base(binding(prior_path)),
        'previous_legacy_actual_closure': previous_pin,
        'previous_phase_independent_review_actual_closed': binding(args.last_legacy_review_actual),
        'prior_covered_sources': len(covered_carriers),
        'prior_geometry_covered_sources': len(covered_carriers)-197+len(completed_from197),
        'remaining_legacy_completed_before_phase': len(completed_from197),
        'remaining_legacy_acknowledgment': binding(ack_path),
        'acknowledged_remaining_legacy_frontier': binding(frontier_path),
        'maximum_new_payload_bytes': 0, 'existing_payload_input_bytes': total,
        'successor_source': base(acknowledged_selection['supervisor_source'])}
    assert plan['maximum_IO_bytes'] == 96_000_000_000 and plan['maximum_IO_bytes_s'] == 256*1024**2
    guard.pace(len(str(plan).encode('utf-8')))
    write_json(out, plan)
    print(dict(pinned_one_plan=binding(out), sources=len(selected), existing_input_bytes=total,
        prior_carrier_sources=plan['prior_covered_sources'],
        prior_declared_geometry_sources=plan['prior_geometry_covered_sources'], native_launched=False), flush=True)


if __name__ == '__main__':
    main()
