"""Bounded observation of an already running exact window; no ML or checkpoint reads."""
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


def tail(path, bound=524288):
    path = Path(path)
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-bound))
        return stream.read().decode('utf-8', errors='replace').splitlines()


def main(args):
    rid = args.run_id
    run = OUT / 'runs' / rid
    loop = OUT / 'runs' / f'{rid}_loop_01_window_{args.window}_journal_01'
    state = read(loop/'loop_status.json')
    assert state['run_id'] == rid and state['status'] == 'running' and state['resume_step'] == args.resume_step
    pid = read(state['paths']['identity'])['pid']
    row = [json.loads(line) for line in tail(state['paths']['log']) if line.startswith('{"step":')][-1]
    n = row['step']-args.resume_step
    assert n > 0
    closed = read(OUT/f'WorldSpeech_expanded_actual_CLOSED{args.resume_step}_preservation_v1.json')
    epoch = row['data_expansion_epoch']
    assert epoch['source_step'] == 40750 and epoch['repeat_visits'] == 0
    assert epoch['unique'] == closed['data_expansion_epoch']['unique']+14*n
    stages = []
    for line in tail(run/'stage_memory.jsonl'):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if value.get('pid') != pid:
            continue
        if value.get('stage') in ('after_own207_GPU_restore_CPU_payload_release', 'after_all207_Adam_restore', 'after_CPU_source_release_and_bounded_TR_startup') or (
                value.get('stage') == 'after_completed_207_Adam_update' and value.get('step') in (args.resume_step+1, args.resume_step+2)):
            stages.append(value)
    assert any(x['stage'] == 'after_own207_GPU_restore_CPU_payload_release' for x in stages)
    assert any(x['stage'] == 'after_all207_Adam_restore' for x in stages)
    assert all(any(x['stage'] == 'after_completed_207_Adam_update' and x['step'] == step for x in stages) for step in (args.resume_step+1, args.resume_step+2))
    config = record(run/'run_config.json')
    equality = record(run/'majority_initial_source_equivalence.json')
    assert config['sha256'] == closed['artifacts']['run_config']['archived']['sha256']
    assert equality['sha256'] == closed['artifacts']['initial_source_equivalence']['archived']['sha256']
    now = datetime.now(timezone.utc).isoformat()
    proof = dict(kind=f'expanded_TRAIN_actual_window{args.window}_own207_restore_positive_v1', passed=True,
        observed_at_utc=now, run_id=rid, actual_owned_session_id=args.owned_session,
        native_launcher_pid=state['owned_launcher_pid'], worker_pid=pid,
        closed_source_step=args.resume_step, current_step=row['step'], new_window_optimizer_updates=n,
        new_data_epoch_updates=row['data_expansion_updates'], original_data_source_step=40750,
        data_expansion_epoch=epoch, original_closed_epoch=closed['data_expansion_epoch'],
        same_full207_weights_Adam_clocks_RNG_history_sampler_own_state_admission_executed=True,
        actual_restore_and_positive_stage_evidence=stages, original_config_bytes_unchanged=config,
        original_source350_bytes_unchanged=equality, source350_repeated=False, coverage_rewind_or_new_queue=False,
        encoder_LR_schedule_reset=False, actual_encoder_lr=row['lora_lr'], actual_head_lr=row['head_lr'],
        GPU_peak_GB=row['peak_allocated_vram_gb'], latest_observed_optimizer_record=row,
        fresh_runtime=record(state['paths']['fresh_runtime']), startup=record(run/f'uniform_full_startup_from_{args.resume_step}.json'),
        actual_fresh_strict_memory_admission=state['fresh_memory_admission'], own_checkpoint_payload_reread_for_this_receipt=False,
        configured_end_step=65948, strict_integer_gates_monitor_only=True)
    path = OUT/f'expanded_TRAIN_actual_window{args.window}_{args.resume_step}_207_restore_positive_v1.json'
    assert not path.exists()
    write_json(path, proof)
    receipt = record(path)
    lp = ROOT/'work/gpu_training_owner.json'
    lease = read(lp)
    assert lease['actual_controller_session_id'] == args.owned_session and lease['target_run_id'] == rid
    lease.update(status=f'actual_expanded_TRAIN_window{args.window}_RUNNING_own207_restored_positive', updated_at_utc=now,
        actual_training_optimizer_running=True, own_fullstate_restore_and_first_update_pending=False,
        actual_completed_step=row['step'], actual_current_window_positive_updates=n,
        actual_encoder_schedule_updates=row['data_expansion_updates'], actual_remaining_updates=65948-row['step'],
        current_matched_cycle_unique=epoch['unique'], new_coverage_local_unique=epoch['unique'],
        current_new_WorldSpeech_clips_seen=row['Turkish_domain_coverage']['groups']['worldspeech']['unique_clips_seen'],
        actual_applied_encoder_lr=row['lora_lr'], actual_head_lr=row['head_lr'],
        latest_owned_worker_self_system_commit_observation=state['actual_worker_self_memory'])
    lease[f'actual_window{args.window}_207_restore_positive'] = receipt
    write_json(lp, lease)
    pp = OUT/'experiment_plan.json'
    plan = read(pp)
    plan['updated_at_utc'] = now
    plan['status_authority'] = (f'Healthy sole expanded TRAIN window {args.window}: actual all-207 exact restore and {n} '
        f'optimizer updates from {args.resume_step} to {row["step"]}; fresh unique {epoch["unique"]}/352768, no repeats. '
        'Source 350/config unchanged; full 65948 budget and base control remain incomplete.')
    current = plan['current_training']
    current.update(status=f'actual_exact_window{args.window}_own207_restored_positive', current_frontier=row['step'],
        new_optimizer_updates=row['data_expansion_updates'], new_epoch_unique=epoch['unique'],
        new_WorldSpeech_unique=row['Turkish_domain_coverage']['groups']['worldspeech']['unique_clips_seen'],
        applied_encoder_lr=row['lora_lr'], own207_restore_pending=False, current_window_207_restore_positive=receipt)
    plan['actual_current_training'] = dict(current)
    plan['current_post_epoch_trial'] = dict(current)
    write_json(pp, plan)
    pp = OUT/'post_epoch_experiment_protocol.json'
    protocol = read(pp)
    protocol['updated_at_utc'] = now
    for item in protocol['ordered_next_work']:
        if item['order'] == 6:
            item.update(status=f'expanded_full_cycle_window{args.window}_actual_own207_positive', current_step=row['step'],
                new_optimizer_updates=row['data_expansion_updates'], new_epoch_unique=epoch['unique'], current_window_207_restore_positive=receipt)
    write_json(pp, protocol)
    print(json.dumps(dict(receipt=receipt, current_step=row['step'], current_window_positive_updates=n,
        unique=epoch['unique'], WorldSpeech=row['Turkish_domain_coverage']['groups']['worldspeech']['unique_clips_seen'], GPU_peak_GB=row['peak_allocated_vram_gb'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='adaptive_20261014_TR_WorldSpeech_expanded_01')
    for key in ('window', 'resume-step', 'owned-session'):
        parser.add_argument('--'+key, type=int, required=True)
    main(parser.parse_args())
