"""Append the measured training comparison and verified quality-loop restart."""
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]

def main():
    comparison = json.loads((OUT / 'native_training_pilot_comparison_v1.json').read_text(encoding='utf-8'))
    assert comparison['status'] == 'complete' and comparison['common_measured_updates'] == 20
    assert all(arm['completed_updates'] == 20 for arm in comparison['arms'].values())
    pause = json.loads((OUT / 'public_data_02_preparation_exit_receipt_v1.json').read_text(encoding='utf-8'))
    assert pause['worker_absent_confirmed'] and pause['published_optimizer_steps'] == 0
    mask = json.loads((OUT / 'TR_metadata_actual_GPU_masks_v1.json').read_text(encoding='utf-8'))
    assert mask['passed'] and mask['training_updates'] == 0
    state = json.loads((OUT / 'runs/adaptive_20261011_public_data_03/loop_status.json').read_text(encoding='utf-8'))
    assert state['status'] == 'running' and state['active_run_id']
    custom = comparison['custom']
    native = comparison['native']
    stamp = datetime.now(timezone.utc).isoformat()
    notes = {
        'ASR-16': f'{stamp}: Actual matched TRAIN benchmark complete. Root custom session84983 EXIT0 and native session67979 EXIT0, 20 optimizer updates each /320 distinct TRAIN clips, same source1871, B8xaccum2, 50% replay, initialization/batches/Adam/learning-rate schedule/normalization. Whole updates timed including reads, frontend, forward/backward, grad clipping, optimizer and allocator cleanup; JIT/cold start, warmup and serialization excluded. Custom loss {custom["combined_audio_minutes_per_wall_minute"]:.6f} combined audio min/min ({custom["turkish_audio_minutes_per_wall_minute"]:.6f} Turkish); NeMo native loss {native["combined_audio_minutes_per_wall_minute"]:.6f} combined ({native["turkish_audio_minutes_per_wall_minute"]:.6f} Turkish). Native ratio {comparison["native_over_custom_training_speed_ratio"]:.8f}, only0.56% difference: no meaningful throughput win from this short pilot. Peak allocated VRAM custom5.442GB/native4.584GB. Scope is native NeMo TDT CUDA loss inside the same HF/LoRA training loop, not a full official Lightning-trainer comparison. No heldout rows. Full training-gradient equivalence not measured in this pilot; scalar agreement does not prove it. Keep existing loss for quality loop; native memory advantage may justify a later explicit candidate.',
        'ASR-15': f'{stamp}: public_data_02 unused Turkish teacher preparation interrupted by root before any optimizer update; root95658 EXIT1, actual worker5212 absent/API87, original stale running JSON preserved and bound by separate immutable root receipt. Source1871 unchanged. NEW frozen controller6 restarted root-owned session40682 in public_data_03, actual current phase {state.get("current_phase")}; B8x2,250-update DEV cadence,57.3h public Turkish TRAIN, replay50% control then replay25% challenger and DEV-selected extension. Trainer6 uses verified Turkish frame metadata, eliminates unused Turkish teacher encoding for49,224 uncached clips/38.378h audio; genuine replay teacher features kept. All74,883 frame counts and25,659 old cache lengths verified; actual GPU16TRAIN masks passed. Explicit full-audit reuse binds prior completed PCM audit plus rechecked fingerprints/headers; no fresh full-bytehash claim. NEW preparation Trackio mirror root64431. No TEST/export; quality target remains unmet and adaptive goal active.',
    }
    for task, note in notes.items():
        subprocess.run([str(OUT / 'tools/backlog.exe'), 'task', 'edit', task,
                        '--append-notes', note, '--add-ref', 'native_training_pilot_comparison_v1.json'],
                       cwd=OUT, check=True, capture_output=True, text=True, encoding='utf-8')
    subprocess.run([str(ROOT / 'work/training-env/Scripts/python.exe'), str(OUT / 'refresh_backlog.py')],
                   cwd=ROOT, check=True)
    print(json.dumps({'backlog_updated': True, 'comparison_updates': 20, 'quality_loop': state['loop_id'],
                      'phase': state.get('current_phase')}))

if __name__ == '__main__':
    main()
