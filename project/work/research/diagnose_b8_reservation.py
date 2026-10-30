"""Read-only CPU snapshot of the active B8 memory/timing evidence."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'outputs/parakeet_turkish'
sys.path.insert(0, str(OUTPUT))
from atomic_io_retry_v1 import read_bytes_shared, write_json


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    run_id = 'adaptive_20261010_recovery_b8_01_b8_quality'
    run_dir = OUTPUT / 'runs' / run_id
    raw = read_bytes_shared(run_dir / 'training_history.json')
    history = json.loads(raw)
    raw_config = read_bytes_shared(run_dir / 'run_config.json')
    config = json.loads(raw_config)
    cache_dir = ROOT / 'work/encoder-cache' / config['encoder_cache_namespace']
    cache_meta = {}
    rows = []
    chosen = {1, 10, 50, 100, 150, 400, 790, 800, 801, 802, 810, 820, 830, 840, 850, 860, 870, 880, 890, 900, len(history)}
    for index, row in enumerate(history):
        if row['step'] not in chosen:
            continue
        previous = history[max(0, index-10)]
        interval_seconds = row['train_wall_seconds'] - previous['train_wall_seconds']
        interval_audio = row['audio_seen_seconds'] - previous['audio_seen_seconds']
        item = {key: row[key] for key in ('step', 'mean_microbatch_lattice_padding_factor',
                  'peak_allocated_vram_gb', 'current_allocated_vram_gb',
                  'current_reserved_vram_gb', 'audio_minutes_per_wall_minute', 'turkish_epoch')}
        item.update(interval_steps=index-max(0, index-10), interval_training_seconds=interval_seconds,
                    interval_audio_seconds=interval_audio,
                    interval_audio_min_per_wall_min=interval_audio/interval_seconds if interval_seconds else None)
        microbatches = []
        for start in range(0, len(row['sampled_source_ids']), config['batch_size']):
            selected = row['sampled_source_ids'][start:start+config['batch_size']]
            shapes = []
            for source_id in selected:
                if source_id not in cache_meta:
                    path = cache_dir / (hashlib.sha256(source_id.encode()).hexdigest() + '.pt')
                    cached = torch.load(path, map_location='cpu', weights_only=True)
                    assert cached['source_id'] == source_id
                    cache_meta[source_id] = (int(cached['features'].shape[0]), int(cached['labels'].numel()))
                shapes.append(cache_meta[source_id])
            t = max(shape[0] for shape in shapes)
            u = max(shape[1] for shape in shapes)
            microbatches.append({'frames': [shape[0] for shape in shapes],
                                 'labels': [shape[1] for shape in shapes],
                                 'max_frames': t, 'max_labels': u,
                                 'padded_lattice_cells': len(shapes)*t*(u+1)})
        item['sampled_step_microbatches'] = microbatches
        rows.append(item)
    checkpoint = ROOT / 'work' / (run_id + '_latest.pt')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    checkpoint_evidence = {'path': str(checkpoint), 'sha256': sha(checkpoint),
                          'run_id': saved['run_id'], 'step': saved['step'],
                          'selected_step': saved['selected_step'], 'best_score': saved['best_score'],
                          'optimizer_state_count': len(saved['optimizer']['state']),
                          'trainable_tensor_count': len(saved['weights']),
                          'history_rows': len(saved['history']), 'stop_reason': saved['stop_reason'],
                          'next_eval_epoch': saved['next_eval_epoch'],
                          'config_identity_matches_live': saved['config_identity'] == config['config_identity'],
                          'restore_fields_present': all(key in saved for key in
                              ('optimizer', 'sampler', 'python_rng', 'torch_rng', 'cuda_rng'))}
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'run_id': run_id,
              'scope': 'CPU-only read-only TRAIN/history/checkpoint/source diagnosis; no GPU or heldout predictions',
              'latest_history_step_at_snapshot': history[-1]['step'],
              'history_sha256_at_snapshot': hashlib.sha256(raw).hexdigest(),
              'run_config_sha256_at_snapshot': hashlib.sha256(raw_config).hexdigest(),
              'source_sha256': {name: sha(OUTPUT/name) for name in
                  ('train_lora_batched.py', 'batched_lora.py', 'efficient_tdt_v2.py',
                   'run_experiment_loop_recovery_v2.py', 'launch_with_io_retry_v1.py', 'atomic_io_retry_v1.py')},
              'rows': rows, 'latest_durable_checkpoint': checkpoint_evidence,
              'development_only': [{'step': row['step'], **row['validation']} for row in history if 'validation' in row],
              'findings': [
                  'Live CUDA tensor allocation remains about2.85GB while allocator reservation expands above physical16GB.',
                  'DEV801 cache clearing reduces reservation to4.977GB at802; reservation then grows to31.931GB by860.',
                  'The slowdown correlates with growing allocator reservation; these records do not directly prove driver paging or its causal mechanism.',
                  'Current allocated/peak metrics, bounded4-shape metadata cache, scalar history and explicit batch/result deletion do not show an accumulating live computation-graph leak.',
                  'Per-update sampled lattice lengths vary; padding factors alone do not explain persistent10-update intervals near140–256seconds with late padding near1.3–1.5.'
              ],
              'safe_options': [
                  'Preserve process ownership; cooperative stop is checked at update boundaries, empties allocator cache before final DEV, and saves full optimizer/sampler/RNG state.',
                  'A hard kill loses all updates after the latest durable DEV checkpoint; current durable step is recorded above.',
                  'If continuation is authorized within the original deadline, use a new versioned wrapper with bounded periodic torch.cuda.empty_cache calls and exact state resume; unchanged model/objective/source/sampler identity must be audited.',
                  'Collect memory_stats/memory_snapshot and operating-system dedicated/shared GPU counters before asserting fragmentation or paging as the cause.'
              ]}
    target = OUTPUT / 'b8_memory_slowdown_diagnosis.json'
    write_json(target, report)
    print(json.dumps({'report': str(target), 'history_step': history[-1]['step'],
                      'durable_step': saved['step'], 'adam_states': len(saved['optimizer']['state'])}))


if __name__ == '__main__':
    main()
