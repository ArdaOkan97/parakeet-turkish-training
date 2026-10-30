"""Append actual native Trainer attempts and hardware results to the canonical backlog."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    failed = OUT / 'runs/full_nemo_turkish_20261011_pilot_02'
    progress = read(failed / 'native_training_progress.json')
    outcome = read(failed / 'trial_outcome.json')
    native = read(failed / 'native_trial_provenance.json')
    assert progress['step'] == outcome['actual_completed_optimizer_updates'] == 20
    assert not native['custom_training_step_or_optimizer_override']
    observations = outcome['resource_telemetry']
    fractions = [x['observation']['system_memory']['commit_fraction'] for x in observations]
    headroom = [x['observation']['calling_process_commit_memory']['ullAvailPageFile_bytes'] for x in observations]
    baseline = read(failed / 'development_gate_step_0.json')
    next_run = OUT / 'runs/full_nemo_turkish_20261011_pilot_03'
    assert next_run.joinpath('resolved_trial_config.json').is_file()
    current = read(next_run / 'resolved_trial_config.json')
    assert current['allocator_cap_gb'] == 9
    note = (
        f'{datetime.now(timezone.utc).isoformat()}: Full native NeMo/Lightning training is now actually being tested. '
        'The previous HF/LoRA run saved its latest checkpoint and full history at update862 before root GPU handoff; '
        'best qualifying Turkish DEV remains64.9697%, so the quality goal is active. '
        'Native pilot01 failed before model initialization when the official archive extractor was denied a sandbox temporary-file write. '
        'The pinned archive was then byte-preservingly extracted into normalized workspace paths; official SaveRestoreConnector.model_extracted_dir fixes restore. '
        'Native pilot02 restored/resized the original v3 model, installed rank8 attention LoRA, and used inherited NeMo training_step, '
        'official data loaders, configure_optimizers, Lightning backward/optimizer, released TDT mean_volume/sigma.02/omega.1, '
        'AdamW and25-update warmup plus9361-update cosine horizon. B8 x accumulation2; native joint subbatch1; supervised50% replay, no teacher KD. '
        f'Initial native DEV TR WER {100*baseline["turkish_wer"]:.4f}%; all five existing-language gates passed. '
        f'Actual first20 optimizer updates: {progress["audio_minutes_per_wall_minute"]:.2f} combined audio min/min, '
        f'{progress["train_wall_seconds"]:.3f} measured training seconds for {progress["audio_seen_seconds"]:.3f} audio seconds. '
        'This includes steady data loading/H2D/backward/optimizer/callback time; startup, DEV, and the first batch loading are excluded. '
        'Native-base initialization/objective differ from the HF warmstart; this is not an isolated same-checkpoint speed comparison. '
        'At replay microbatch41, stock NeMo RNNTLoss.forward needed another334MiB for its FP32 joint cast and hit the8GB allocator cap. '
        'Native joint subbatch1 still retains all sub-loss graphs until backward. '
        f'Highest sampled global commit {100*max(fractions):.2f}%; minimum owned self headroom {min(headroom)/1e9:.3f}GB; '
        'the resource guard did not trigger. '
        'Root observed pilot02 EXIT1 and started pilot03 with the same frozen v2 entry, data, objective and B8x2, using a9GB allocator cap; '
        'Windows resource thresholds remain unchanged. Both native attempts and actual training/dev points are mirrored into the same parakeet-turkish Trackio project. '
        'Full-pool continuation manifests preserve the pilot prefix and visit72,883 new Turkish IDs afterward; no final clips dropped. '
        'No TEST inference or export performed.')
    for task in ('ASR-15', 'ASR-16'):
        subprocess.run([str(OUT / 'tools/backlog.exe'), 'task', 'edit', task, '--append-notes', note,
                        '--add-ref', 'runs/full_nemo_turkish_20261011_pilot_02/trial_outcome.json',
                        '--add-ref', 'full_nemo_training_continuation_v1.json'],
                       cwd=OUT, check=True, capture_output=True, text=True, encoding='utf-8')
    subprocess.run([str(ROOT / 'work/training-env/Scripts/python.exe'), str(OUT / 'refresh_backlog.py')],
                   cwd=ROOT, check=True)
    print(json.dumps({'backlog_updated': True, 'full_native_completed_updates': 20,
                      'next_native_run': next_run.name}))


if __name__ == '__main__':
    main()
