"""Record completed DEV screens, the live extension and corrected tracking."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    loop = OUT / 'runs/adaptive_20261011_native_memory_fix_01'
    state = read(loop / 'loop_status.json')
    assert state['status'] == 'running' and not state['resource_review_required']
    assert state['active_run_id'].endswith('_public_winner_extension')
    values = []
    for trial in state['trials'][:2]:
        assert trial['status'] == 'complete' and trial['completed_steps'] == 250
        run_id = trial['run_id']
        history = read(OUT / 'runs' / run_id / 'training_history.json')
        last = history[-1]
        assert last['step'] == 250 and last['validation']['retention_gate_passed']
        values.append((run_id, last))
    repair = read(OUT / 'trackio_step0_namespace_repair_v2/repair_receipt.json')
    assert repair['passed'] and repair['applied'] and repair['rows_deleted'] == 0
    control, challenger = values[0][1], values[1][1]
    note = (
        f'{datetime.now(timezone.utc).isoformat()}: Both native-loss DEV screens completed250 updates without a resource stop. '
        f'Control50% replay TR WER{100 * control["validation"]["turkish_wer"]:.2f}%; '
        f'challenger25% replay TR WER{100 * challenger["validation"]["turkish_wer"]:.2f}%; '
        'both pass allfive foreign-language retention gates and neither improves the existing64.97% best. '
        f'Control throughput{control["audio_minutes_per_wall_minute"]:.2f} combined / '
        f'{control["turkish_audio_seen_seconds"] / control["train_wall_seconds"]:.2f} Turkish audio min/min; '
        f'challenger{challenger["audio_minutes_per_wall_minute"]:.2f} combined / '
        f'{challenger["turkish_audio_seen_seconds"] / challenger["train_wall_seconds"]:.2f} Turkish min/min. '
        'Different replay mixes change audio lengths; this is a measured exposure comparison, not an isolated kernel speedup. '
        f'TR corpus coverage was only{control["turkish_epoch"]:.4f} and{challenger["turkish_epoch"]:.4f} epochs. '
        f'Actual live extension {state["active_run_id"]} starts from the qualifying best, batch8x2/native loss/early cleanup, '
        'DEV every250 and continuation within the current three-hour window. '
        'Next adaptation should assess a longer75%-Turkish run because it nearly doubled Turkish audio throughput while passing retention; '
        'the shared unchanged initial checkpoint is not evidence that50% replay is the better long-term recipe. '
        'Dashboard correction preserved two original base-reference rows and moved their metric keys into reference/*; '
        'warmstart DEV at localstep0 is64.97%, avoiding the false average with base_v3. '
        'Mirrorv3 session99627 is live; all training data and model-selection scores were unchanged. '
        'No TEST/export and competitive quality remains unmet. Goal active.')
    for task in ('ASR-15', 'ASR-16'):
        subprocess.run([str(OUT / 'tools/backlog.exe'), 'task', 'edit', task, '--append-notes', note,
                        '--add-ref', str((loop / 'loop_status.json').relative_to(OUT)),
                        '--add-ref', 'trackio_step0_namespace_repair_v2/repair_receipt.json'],
                       cwd=OUT, check=True, capture_output=True, text=True, encoding='utf-8')
    subprocess.run([str(ROOT / 'work/training-env/Scripts/python.exe'), str(OUT / 'refresh_backlog.py')],
                   cwd=ROOT, check=True)
    print(json.dumps({'backlog_updated': True, 'screens_completed': 2, 'live_extension': state['active_run_id']}))


if __name__ == '__main__':
    main()
