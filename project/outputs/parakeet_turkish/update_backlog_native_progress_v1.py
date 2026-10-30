"""Record the live extension's first DEV result without changing training."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def main():
    loop_path = OUT / 'runs/adaptive_20261011_native_memory_fix_01/loop_status.json'
    state = json.loads(loop_path.read_text(encoding='utf-8'))
    run_id = 'adaptive_20261011_native_memory_fix_01_public_winner_extension'
    history_path = OUT / 'runs' / run_id / 'training_history.json'
    history = json.loads(history_path.read_text(encoding='utf-8'))
    first_dev = next(row for row in history if row['step'] == 250 and 'validation' in row)
    last = history[-1]
    gate = first_dev['validation']
    note = (
        f'{datetime.now(timezone.utc).isoformat()}: Live native-loss extension {run_id} '
        f'has published update {last["step"]}, B8 x accumulation2, '
        f'{last["audio_minutes_per_wall_minute"]:.2f} combined and '
        f'{last["turkish_audio_seen_seconds"] / last["train_wall_seconds"]:.2f} '
        'Turkish audio min per training minute. '
        f'First DEV at250: Turkish WER {100 * gate["turkish_wer"]:.4f}%, '
        'two fewer word errors than the64.9697% warmstart. '
        f'Retention passes={gate["retention_gate_passed"]}; Spanish delta '
        f'{100 * gate["retention_wer_deltas"]["es"]:.4f}pp exceeds its '
        f'{100 * gate["retention_wer_limits"]["es"]:.4f}pp limit. '
        'This is one consecutive retention failure, so the checkpoint does not qualify as a new best. '
        f'Expanded Turkish pool coverage is {100 * last["turkish_epoch"]:.2f}% of one clip epoch. '
        'Training is advancing, but quality progress is small and the competitive objective remains unmet. '
        'Focused saved-DEV review finds broad word/segmentation errors rather than a diacritic-only bug; '
        'first short control CER18.62% versus WER65.09%, all-diacritic folding WER63.88%, '
        'forgiving exact word split/merge boundaries WER58.79%. '
        'Trainer plateau patience already counts full epochs; no plateau patch is needed. '
        'Prepare a longer Turkish-exposure arm with firstDEV250 then every1000; '
        'retain existing-language and resource guards. No changes to this live run, TEST, or export.')
    for task in ('ASR-15', 'ASR-16'):
        subprocess.run(
            [str(OUT / 'tools/backlog.exe'), 'task', 'edit', task, '--append-notes', note,
             '--add-ref', str(history_path.relative_to(OUT)),
             '--add-ref', 'native_long_schedule_review_v1.json'],
            cwd=OUT, check=True, capture_output=True, text=True, encoding='utf-8')
    subprocess.run([str(ROOT / 'work/training-env/Scripts/python.exe'),
                    str(OUT / 'refresh_backlog.py')], cwd=ROOT, check=True)
    print(json.dumps({'backlog_updated': True, 'step': last['step'],
                      'DEV250': gate, 'observed_loop_status': state['status']}))


if __name__ == '__main__':
    main()
