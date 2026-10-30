"""Record the actual step63 continuation after the measured memory repair."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def main():
    loop = OUT / 'runs/adaptive_20261011_native_memory_fix_01'
    state = json.loads((loop / 'loop_status.json').read_text(encoding='utf-8'))
    assert state['status'] == 'running' and not state['resource_review_required']
    run_id = state['active_run_id']
    trial = state['trials'][-1]
    assert trial['initial_step'] == 63 and trial['exact_optimizer_sampler_RNG_continuation']
    directory = OUT / 'runs' / run_id
    config = json.loads((directory / 'run_config.json').read_text(encoding='utf-8'))
    history = json.loads((directory / 'training_history.json').read_text(encoding='utf-8'))
    old_history = json.loads((OUT / 'runs/adaptive_20261011_native_data_01_public_rank8_control/training_history.json').read_text(encoding='utf-8'))
    assert history[:len(old_history)] == old_history and history[-1]['step'] > 63
    assert config['batch_size'] == 8 and config['accumulation'] == 2 and config['tdt_implementation'] == 'nemo'
    backend = json.loads((loop / (run_id + '_native_backend.json')).read_text(encoding='utf-8'))
    assert backend['status'] == 'ready' and backend['declared_allocator_policy']['cap_gb'] == 9
    note = (
        f'{datetime.now(timezone.utc).isoformat()}: ACTUAL native-loss training resumed in {run_id}, '
        f'root session40244; latest published step{history[-1]["step"]} (after inherited63). '
        'Complete optimizer, sampler, RNG and history were resumed from the immutable63 checkpoint; '
        'the entire original history prefix is unchanged. The best qualifying step0 checkpoint is preserved. '
        'Same batch8 with accumulation2, learning rates, losses, replay fraction and57.3h Turkish TRAIN pool. '
        'Official NeMo TDT loss now clears inactive CUDA cache around per-clip evaluation and backward recomputation; '
        'tested cap9GB, end cleanup4GB, original95%/2GB memory guards enabled. '
        'A pre-child namespace rejection was repaired by archiving the redundant preseeded latest file and failed-launch artifacts; '
        'no training source changed and no optimizer step ran in that failed launch. '
        'First next DEV evaluation at250, followed by the replay25 challenger and winner extension in the adaptive loop. '
        'The hybrid HF/LoRA loop remains custom; the full NeMo Lightning trainer has not been tested. '
        'Optional future work removes a redundant FP32 packed-joint copy around the same official native kernel; '
        'CPU parity passed but GPU validation is pending, so this optimization is not in the active run. '
        'No TEST/export and no new quality claim. The competitive Turkish WER objective remains active.')
    for task in ('ASR-15', 'ASR-16'):
        subprocess.run([str(OUT / 'tools/backlog.exe'), 'task', 'edit', task, '--append-notes', note,
                        '--add-ref', str((directory / 'run_config.json').relative_to(OUT)),
                        '--add-ref', 'native_memory_fix_01_prechild_failure_v1/recovery_receipt.json'],
                       cwd=OUT, check=True, capture_output=True, text=True, encoding='utf-8')
    subprocess.run([str(ROOT / 'work/training-env/Scripts/python.exe'), str(OUT / 'refresh_backlog.py')],
                   cwd=ROOT, check=True)
    print(json.dumps({'backlog_updated': True, 'actual_step': history[-1]['step'], 'resumed_exact63': True}))


if __name__ == '__main__':
    main()
