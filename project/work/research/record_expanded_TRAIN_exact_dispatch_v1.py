"""Record an already launched exact window using bounded metadata only."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/parakeet_turkish'
sys.path.insert(0, str(OUT))
from atomic_io_retry_v1 import write_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def record(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path.resolve()), sha256=digest, bytes=path.stat().st_size)


def main(args):
    rid = args.run_id
    loop = OUT / 'runs' / f'{rid}_loop_01_window_{args.window}_journal_01'
    state = read(loop / 'loop_status.json')
    assert state['run_id'] == rid and state['status'] == 'running'
    assert state['resume_step'] == args.resume_step
    identity = read(state['paths']['identity'])
    closed_path = OUT / f'WorldSpeech_expanded_actual_CLOSED{args.resume_step}_preservation_v1.json'
    closed = read(closed_path)
    assert closed['passed'] and closed['run_id'] == rid and closed['data_arm'] == 'expanded'
    assert closed['actual_owned_exit'] == dict(session_id=args.prior_owned_session, exit_code=0, tool_chunk=args.prior_exit_tool_chunk)
    assert closed['native_child_exit_code'] == 0 and closed['native_sources_unchanged']
    selection = read(closed['artifacts']['selection']['archived']['path'])
    counts = {lang: score['word_errors'] for lang, score in closed['counts'].items()}
    epoch = closed['data_expansion_epoch']
    world = selection['Turkish_domain_coverage']['groups']['worldspeech']['unique_clips_seen']
    now = datetime.now(timezone.utc).isoformat()
    proof = dict(kind=f'expanded_TRAIN_actual_window{args.window}_same_ID_dispatch_v1',
        observed_at_utc=now, run_id=rid, actual_owned_session_id=args.owned_session,
        actual_launch_tool_chunk=args.launch_tool_chunk, native_launcher_pid=state['owned_launcher_pid'],
        worker_pid=identity['pid'], resume_step=args.resume_step, original_data_source_step=40750,
        closed_source=record(closed_path), closed_full207_checkpoint=closed['artifacts']['latest_checkpoint']['archived'],
        plan=record(loop / 'plan.json'), real_preflight=record(OUT / f'expanded_TRAIN_window{args.window}_{args.resume_step}_preflight_v1.json'),
        fresh_strict_memory_admission=state['fresh_memory_admission'], source350_repeated=False,
        initial_original_source350_preserved=True, same_full207_Adam_RNG_sampler_history_and_coverage_restoration_pending=True,
        actual_positive_updates_this_window=0, unchanged_scientific_identity=True,
        unchanged_configured_end_step=65948, remaining_updates=65948-args.resume_step,
        no_pagefile_or_guard_or_dataset_changes=True)
    path = OUT / f'expanded_TRAIN_actual_window{args.window}_{args.resume_step}_dispatch_v1.json'
    assert not path.exists()
    write_json(path, proof)
    receipt = record(path)
    lp = ROOT / 'work/gpu_training_owner.json'
    lease = read(lp)
    assert lease['target_run_id'] == rid
    lease[f'prior_expanded_window{args.window-1}'] = dict(actual_owned_exit=closed['actual_owned_exit'],
        closed_receipt=record(closed_path), complete_DEV_counts=counts, completed_step=args.resume_step, data_expansion_epoch=epoch)
    for key in ('actual_exit_code', 'actual_closed_tool_chunk', 'actual_owned_exit', 'native_child_exit_code', 'actual_closed_status', 'actual_native_child_exit_code'):
        lease.pop(key, None)
    lease.update(status=f'actual_expanded_TRAIN_window{args.window}_RUNNING_restore_pending', updated_at_utc=now,
        actual_GPU_running=True, actual_training_optimizer_running=False, actual_GPU_launched=True,
        running=True, reserved=True, actual_launch_tool_chunk=args.launch_tool_chunk,
        actual_controller_session_id=args.owned_session, actual_owned_native_launcher_pid=state['owned_launcher_pid'],
        actual_owned_worker_pid=identity['pid'], actual_controller_loop=str(loop.resolve()),
        actual_completed_step=args.resume_step, source_frontier=args.resume_step,
        actual_resume_window_start_step=args.resume_step, actual_current_window_positive_updates=0,
        actual_remaining_updates=65948-args.resume_step, actual_dispatch_receipt=receipt,
        actual_closed_checkpoint_for_current_window=closed['artifacts']['latest_checkpoint']['archived'],
        actual_fresh_strict_memory_admission=state['fresh_memory_admission'],
        own_fullstate_restore_and_first_update_pending=True, source350_and_first_optimizer_pending=False,
        restored_source_retention_counter=closed['actual_monitored_retention_counter'],
        restored_source_retention_counter_scope=f'Immutable CLOSED{args.resume_step} entry to window {args.window}; current monitoring counter is separate.',
        actual_retention_counter=closed['actual_monitored_retention_counter'],
        actual_encoder_schedule_updates=args.resume_step-40750, current_matched_cycle_unique=epoch['unique'],
        new_coverage_local_unique=epoch['unique'], current_new_WorldSpeech_clips_seen=world,
        current_matched_cycle_complete=False, actual_last_complete_DEV_step=args.resume_step,
        actual_last_complete_DEV_counts=counts,
        current_complete_DEV_all_six_passed=all(counts[k] <= v for k, v in dict(en=69, es=52, it=30, fr=74, de=70, pt=66).items()))
    write_json(lp, lease)
    pp = OUT / 'experiment_plan.json'
    plan = read(pp)
    plan['updated_at_utc'] = now
    plan['actual_GPU_running'] = True
    plan['status_authority'] = (f'Actual normal CLOSED0/native0 at {args.resume_step}; same-ID expanded window {args.window} '
        f'launched as owned {args.owned_session}, native {state["owned_launcher_pid"]}, worker {identity["pid"]}. '
        'Fresh strict admission passed. Own 207 restore and positive updates pending; source 350 remains immutable.')
    if plan['global_qualified_champion']['step'] == closed['selected_qualified_BEST_step']:
        plan['global_qualified_champion']['preserved_full207_checkpoint'] = closed['artifacts']['best_checkpoint']['archived']
    current = plan['current_training']
    current.update(status=f'actual_exact_window{args.window}_launched_restore_pending',
        actual_owned_session=args.owned_session, native_launcher_pid=state['owned_launcher_pid'], worker_pid=identity['pid'],
        current_frontier=args.resume_step, actual_resume_frontier=args.resume_step,
        new_optimizer_updates=args.resume_step-40750, new_epoch_unique=epoch['unique'], new_WorldSpeech_unique=world,
        last_complete_DEV_step=args.resume_step, last_DEV_counts=counts,
        last_complete_DEV_TR_WER=counts['tr']/825, last_complete_DEV_all_six_passed=lease['current_complete_DEV_all_six_passed'],
        selected_qualified_BEST_step=closed['selected_qualified_BEST_step'], own207_restore_pending=True,
        actual_controller_loop=str(loop.resolve()), current_window_dispatch=receipt)
    plan['actual_current_training'] = dict(current)
    plan['current_post_epoch_trial'] = dict(current)
    write_json(pp, plan)
    pp = OUT / 'post_epoch_experiment_protocol.json'
    protocol = read(pp)
    protocol['updated_at_utc'] = now
    for item in protocol['ordered_next_work']:
        if item['order'] == 6:
            item.update(status=f'expanded_full_cycle_window{args.window}_actually_launched_restore_pending',
                current_step=args.resume_step, new_optimizer_updates=args.resume_step-40750,
                new_epoch_unique=epoch['unique'], current_window_dispatch=receipt,
                latest_complete_DEV_step=args.resume_step, latest_complete_DEV_counts=counts,
                latest_complete_DEV=closed['summary'])
    write_json(pp, protocol)
    print(json.dumps(dict(receipt=receipt, session=args.owned_session, native=state['owned_launcher_pid'],
        worker=identity['pid'], fresh_memory_admission=state['fresh_memory_admission'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='adaptive_20261014_TR_WorldSpeech_expanded_01')
    for key in ('window', 'resume-step', 'owned-session', 'prior-owned-session'):
        parser.add_argument('--'+key, type=int, required=True)
    for key in ('launch-tool-chunk', 'prior-exit-tool-chunk'):
        parser.add_argument('--'+key, required=True)
    main(parser.parse_args())
