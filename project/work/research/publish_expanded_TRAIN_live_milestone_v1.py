"""Bounded metadata bookkeeping only; never imports ML or opens checkpoints."""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/parakeet_turkish'
sys.path.insert(0, str(OUT))
from atomic_io_retry_v1 import write_json

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def record(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path.resolve()), 'sha256': digest, 'bytes': path.stat().st_size}

def main(args):
    run = OUT / 'runs' / args.run_id
    current_lease = read(ROOT / 'work/gpu_training_owner.json')
    assert current_lease['target_run_id'] == args.run_id
    assert current_lease['actual_controller_session_id'] == args.owned_session
    loop = Path(current_lease['actual_controller_loop']).resolve()
    assert loop.parent == (OUT / 'runs').resolve() and loop.name.startswith(args.run_id + '_loop_')
    state = read(loop / 'loop_status.json')
    assert state['status'] == 'running' and state['run_id'] == args.run_id
    summary = OUT / 'results' / f'{args.run_id}_step_{args.dev_step}' / 'summary.json'
    scores = read(summary)
    refs = dict(tr=825, en=1045, es=1289, it=1158, fr=1072, de=1004, pt=1095)
    limits = dict(en=69, es=52, it=30, fr=74, de=70, pt=66)
    counts = {}
    for lang, reference in refs.items():
        value = scores['validation_' + lang]['asr']
        assert value['utterances'] == 50 and value['reference_words'] == reference
        assert type(value['word_errors']) is int and value['wer'] == value['word_errors'] / reference
        counts[lang] = value['word_errors']
    gates = {lang: counts[lang] <= limit for lang, limit in limits.items()}
    log = Path(state['paths']['log'])
    with log.open('rb') as stream:
        stream.seek(max(0, log.stat().st_size - 524288))
        lines = stream.read().decode('utf-8', errors='replace').splitlines()
    rows = [json.loads(line) for line in lines if line.startswith('{"step":')]
    row = rows[-1]
    assert row['step'] >= args.dev_step
    dev_lines = [line for line in lines if line.startswith(f'dev step{args.dev_step} ')]
    assert dev_lines, 'Requested completed DEV must remain in the bounded native log tail'
    selected = int(re.search(r'best(\d+)', dev_lines[-1]).group(1))
    history = run / 'training_history.json'
    with history.open('rb') as stream:
        stream.seek(max(0, history.stat().st_size - 262144))
        snapshot_tail = stream.read().decode('utf-8', errors='replace')
    start = snapshot_tail.rfind('\n  {\n')
    assert start >= 0
    fragment = snapshot_tail[start:].strip()
    assert fragment.endswith(']')
    frontier = json.loads(fragment[:-1].strip())
    assert frontier['step'] == args.dev_step, 'Use the actual complete DEV snapshot frontier'
    counter = frontier['selection_diagnostics']['consecutive_retention_failures']
    epoch = row['data_expansion_epoch']
    worldspeech = row['Turkish_domain_coverage']['groups'].get('worldspeech', {}).get('unique_clips_seen', 0)
    now = datetime.now(timezone.utc).isoformat()
    proof = dict(kind='expanded_TRAIN_actual_live_complete_DEV_milestone_v1', observed_at_utc=now,
        run_id=args.run_id, owned_session=args.owned_session, native_pid=state['owned_launcher_pid'],
        worker_pid=read(Path(state['paths']['identity']))['pid'], complete_DEV_step=args.dev_step,
        counts=counts, strict_limits=limits, gates=gates, all_six_passed=all(gates.values()),
        summary=record(summary), actual_worker_DEV_line=dev_lines[-1], selected_BEST_step=selected,
        actual_retention_monitoring_counter=counter, complete_snapshot_frontier_read_bytes=262144,
        current_step=row['step'], new_updates=row['data_expansion_updates'], new_epoch=epoch,
        new_WorldSpeech_unique=worldspeech, encoder_lr=row['lora_lr'], head_lr=row['head_lr'],
        GPU_peak_GB=row['peak_allocated_vram_gb'], resource_state=state['system_and_owned_private_memory'],
        window_peak_private_bytes=state['peak_observed_owned_private_bytes'],
        full_matched_cycle_continues_unchanged=True, WER_gain_or_data_causality_claimed=False)
    path = OUT / f'expanded_TRAIN_actual_DEV{args.dev_step}_milestone_v1.json'
    assert not path.exists(), 'Fresh milestone receipt required'
    write_json(path, proof)
    receipt = record(path)
    lease_path = ROOT / 'work/gpu_training_owner.json'
    lease = read(lease_path)
    assert lease['actual_controller_session_id'] == args.owned_session and lease['target_run_id'] == args.run_id
    lease.update(updated_at_utc=now, actual_completed_step=row['step'],
        actual_encoder_schedule_updates=row['data_expansion_updates'],
        actual_current_window_positive_updates=row['step'] - state['resume_step'],
        actual_remaining_updates=65948-row['step'], current_matched_cycle_unique=epoch['unique'],
        current_matched_cycle_complete=epoch['full_pass_complete'], new_coverage_local_unique=epoch['unique'],
        current_new_WorldSpeech_clips_seen=worldspeech, actual_applied_encoder_lr=row['lora_lr'],
        actual_last_complete_DEV_step=args.dev_step, actual_last_complete_DEV_counts=counts,
        current_complete_DEV_all_six_passed=all(gates.values()), actual_current_milestone=receipt,
        actual_retention_counter=counter)
    write_json(lease_path, lease)
    plan_path = OUT / 'experiment_plan.json'
    plan = read(plan_path)
    plan['updated_at_utc'] = now
    plan['status_authority'] = (f'Healthy sole expanded TRAIN at {row["step"]}, '
        f'+{row["data_expansion_updates"]} updates and {epoch["unique"]}/{epoch["pool"]} fresh unique; '
        f'{worldspeech} WorldSpeech clips. Complete DEV{args.dev_step}: TR{counts["tr"]}/825, '
        f'counts {counts}, all six {all(gates.values())}, actual saved BEST{selected}. '
        'Full matched budget to65948 continues unchanged; no data benefit or plateau claim.')
    current = plan['current_training']
    current.update(current_frontier=row['step'], new_optimizer_updates=row['data_expansion_updates'],
        new_epoch_unique=epoch['unique'], new_WorldSpeech_unique=worldspeech, applied_encoder_lr=row['lora_lr'],
        last_complete_DEV_step=args.dev_step, last_DEV_counts=counts,
        last_complete_DEV_TR_WER=counts['tr']/825, last_complete_DEV_all_six_passed=all(gates.values()),
        current_milestone=receipt, selected_qualified_BEST_step=selected)
    if 'historical_phase_aliases' not in plan:
        plan['historical_phase_aliases'] = {key: plan[key] for key in
            ('actual_current_training', 'current_post_epoch_trial') if key in plan}
    plan['actual_current_training'] = dict(current)
    plan['current_post_epoch_trial'] = dict(run_id=args.run_id, family='matched_original_vs_expanded_TRAIN',
        data_arm=row['data_arm'], status='actual_full_cycle_running', source_step=40750,
        current_step=row['step'], configured_end_step=65948, latest_complete_DEV_step=args.dev_step,
        latest_complete_DEV_counts=counts, latest_complete_DEV_all_six_passed=all(gates.values()),
        base_control_pending=True)
    research = ROOT / 'work/research/Turkish_scale_research_v1/research_conclusions.json'
    if research.is_file():
        plan.setdefault('future_data_expansion_policy', {})['scale_research_only_not_admitted'] = dict(
            record=record(research), report_page='https://chatgpt.com/space/page_19829e912f50819195037c1c70bedf1a',
            current_trial_dataset_tokenizer_augmentation_unchanged=True)
    write_json(plan_path, plan)
    protocol_path = OUT / 'post_epoch_experiment_protocol.json'
    protocol = read(protocol_path)
    protocol['updated_at_utc'] = now
    for item in protocol['ordered_next_work']:
        if item['order'] == 6:
            item.update(current_step=row['step'], new_optimizer_updates=row['data_expansion_updates'],
                new_epoch_unique=epoch['unique'], new_WorldSpeech_unique=worldspeech,
                latest_complete_DEV=receipt, accepted_new_TRAIN_hours=408.2420753472222)
    write_json(protocol_path, protocol)
    print(json.dumps(dict(receipt=receipt, current_step=row['step'], new_updates=row['data_expansion_updates'],
        unique=epoch['unique'], WorldSpeech_unique=worldspeech, counts=counts,
        all_six=all(gates.values()), selected_BEST=selected)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='adaptive_20261014_TR_WorldSpeech_expanded_01')
    parser.add_argument('--owned-session', type=int, default=65641)
    parser.add_argument('--dev-step', type=int, required=True)
    main(parser.parse_args())
