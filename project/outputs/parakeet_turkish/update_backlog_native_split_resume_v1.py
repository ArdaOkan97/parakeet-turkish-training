"""Record observed new training updates after the step358 resource recovery."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    run_id = 'adaptive_20261011_native_split_resume_01'
    source_id = 'adaptive_20261011_native_memory_fix_01_public_winner_extension'
    history = read(OUT / 'runs' / run_id / 'training_history.json')
    source_history = read(OUT / 'runs' / source_id / 'training_history.json')
    assert len(source_history) == 358 and history[:358] == source_history
    last, before = history[-1], source_history[-1]
    assert last['step'] > 358
    seconds = last['train_wall_seconds'] - before['train_wall_seconds']
    assert seconds > 0
    combined_rate = (last['audio_seen_seconds'] - before['audio_seen_seconds']) / seconds
    tr_rate = (last['turkish_audio_seen_seconds'] - before['turkish_audio_seen_seconds']) / seconds
    stress = read(OUT / 'native_long_memory_cap8_v4.json')
    gradient = read(OUT / 'native_split_actual_GPU_comparison_v1.json')
    assert stress['passed'] and gradient['passed']
    note = (
        f'{datetime.now(timezone.utc).isoformat()}: Correction to the prior live-extension observation: '
        'the old extension stopped at step358 after a real transient Windows commit event at update354. '
        'At that sample, active CUDA memory was2.844GB and reserved memory6.363GB; immediate cache reclamation reduced reserve to2.896GB. '
        'Earlier per-native-loss cleanup did not cover unused allocations from encoder backward. '
        f'New run {run_id} has actually completed update {last["step"]}, '
        'resuming all358 prior updates with weights, Adam, sampler, RNG, and history preserved. '
        'Batch8 x accumulation2, same50% replay and LR/KD/DEV cadence; cap8GB. Native3 splits the joint before FP32 casting '
        'and trainer10 also reclaims unused cache after each completed microbatch backward and Adam. '
        'No guard threshold waiver, data crop/drop, or optimizer reset. '
        f'One complete16-TRAIN stress update with the longest clips passed: peak active memory {stress["peak_allocated_bytes"] / 1e9:.3f}GB, '
        f'peak reserve {stress["peak_reserved_bytes"] / 1e9:.3f}GB, minimum self commit headroom '
        f'{stress["minimum_owned_self_headroom_bytes"] / 1e9:.3f}GB. '
        'Old stress peak active memory8.149GB; new6.347GB, about22% lower in this resource check, '
        'not a representative training throughput benchmark. '
        'Two real TRAIN clips in FP32 and BF16 had bit-exact native loss and full joint logit gradients between old/new wrappers. '
        f'Post-resume measured training interval ({last["step"] - 358} updates): '
        f'{combined_rate:.2f} combined / {tr_rate:.2f} Turkish audio min/min; excludes startup and DEV. '
        'Best qualifying Turkish DEV remains64.9697%; old final358 DEV65.3333% passed retention. '
        'Next DEV at500. Competitive quality remains unmet; training/experiment goal remains active. '
        'Considering a larger physical batch through a matched whole-training benchmark. Current runtime uses official NeMo TDT loss '
        'in the HF LoRA loop; full NeMo Lightning trainer adoption is pending. Current25-step warmup and constant LR remain for recovery. '
        'Warmup with cosine decay and FLEURS TRAIN source balancing are next declared recipe options.')
    for task in ('ASR-15', 'ASR-16'):
        subprocess.run([str(OUT / 'tools/backlog.exe'), 'task', 'edit', task,
                        '--append-notes', note, '--add-ref', f'runs/{run_id}/training_history.json',
                        '--add-ref', 'native_long_memory_cap8_v4.json'],
                       cwd=OUT, check=True, capture_output=True, text=True, encoding='utf-8')
    subprocess.run([str(ROOT / 'work/training-env/Scripts/python.exe'), str(OUT / 'refresh_backlog.py')],
                   cwd=ROOT, check=True)
    print(json.dumps({'backlog_updated': True, 'run_id': run_id, 'step': last['step'],
                      'post_resume_combined_audio_min_per_min': combined_rate,
                      'post_resume_Turkish_audio_min_per_min': tr_rate}))


if __name__ == '__main__':
    main()
