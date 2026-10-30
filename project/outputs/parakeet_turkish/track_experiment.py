"""Mirror existing experiment JSON logs into a local Trackio dashboard.

Run with work/monitor-env/Scripts/python.exe. This process does not import the
training runtime or change the active model. It can backfill and follow logs.
"""
import argparse
import json
import os
import time
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
os.environ['TRACKIO_DIR'] = str(OUTPUT / 'trackio')
os.environ['HF_HOME'] = str(ROOT / 'work' / 'monitor-hf-cache')
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['TRACKIO_PLOT_ORDER'] = 'dev/wer_tr_percent,train/tdt_loss,train/replay_kl,dev,retention,resources,gpu'
for key in ('TRACKIO_SPACE_ID', 'TRACKIO_SERVER_URL', 'TRACKIO_DATASET_ID',
            'TRACKIO_BUCKET_ID', 'TRACKIO_WEBHOOK_URL'):
    os.environ.pop(key, None)

import trackio


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def metrics_for(record):
    result = {
        'train/tdt_loss': record['tdt_loss'],
        'train/replay_kl': record['replay_kl'],
        'train/gradient_norm': record['gradient_norm'],
        'resources/elapsed_minutes': record['elapsed_seconds'] / 60,
        'resources/peak_torch_allocated_vram_gb': record['peak_allocated_vram_gb'],
    }
    for name in ('replay_duration_kl', 'replay_feature_distillation'):
        if name in record:
            result[f'train/{name}'] = record[name]
    for name in ('current_allocated_vram_gb', 'current_reserved_vram_gb'):
        if name in record:
            result[f'resources/{name}'] = record[name]
    if 'turkish_unique_clips_seen' in record:
        result['data/turkish_unique_clips_seen'] = record['turkish_unique_clips_seen']
    for name in ('audio_minutes_per_wall_minute', 'train_wall_seconds', 'evaluation_wall_seconds',
                 'process_wall_seconds'):
        if name in record:
            result[f'resources/{name}'] = record[name]
    for name in ('turkish_epoch', 'turkish_clips_seen', 'replay_clips_seen',
                 'audio_seen_seconds', 'turkish_audio_seen_seconds', 'replay_audio_seen_seconds',
                 'mean_microbatch_lattice_padding_factor'):
        if name in record:
            result[f'data/{name}'] = record[name]
    for name in ('head_lr', 'lora_lr', 'new_token_update_multiplier'):
        if name in record:
            result[f'train/{name}'] = record[name]
    if 'validation' in record:
        validation = record['validation']
        result['retention/all_dev_gates_passed'] = int(validation['retention_gate_passed'])
        for lang, delta in validation['retention_wer_deltas'].items():
            result[f'retention/wer_delta_{lang}_pp'] = 100 * delta
    return result


def dev_metrics(run_id, step):
    summary = read_json(OUTPUT / f'results/{run_id}_step_{step}/summary.json', {})
    # A partial evaluation is allowed: finished languages appear as they arrive.
    return {f'dev/wer_{key.removeprefix("validation_")}_percent': 100 * value['asr']['wer']
            for key, value in summary.items() if key.startswith('validation_')}


def follow(args):
    state_path = ROOT / 'work' / 'trackio_mirror_state.json'
    state = read_json(state_path, {})
    active = None
    finished = {key for key, value in state.items() if value.get('finished')}
    while True:
        config_path = OUTPUT / 'run_config.json'
        history_path = OUTPUT / 'training_history.json'
        config = read_json(config_path, {})
        history = read_json(history_path, [])
        run_id = config.get('run_id')
        if run_id and history and history_path.stat().st_mtime >= config_path.stat().st_mtime:
            if active != run_id and run_id not in finished:
                if active is not None:
                    trackio.finish()
                trackio.init(project=args.project, name=run_id, group='Turkish adaptation',
                             config=config, resume='allow', gpu_log_interval=10.0)
                active = run_id
                state.setdefault(run_id, {})
                baseline = read_json(OUTPUT / 'results/base_v3/summary.json', {})
                if 'baseline' not in state[run_id]:
                    trackio.log({f'dev/wer_{key.removeprefix("validation_")}_percent': 100 * value['asr']['wer']
                                 for key, value in baseline.items() if key.startswith('validation_')}, step=0)
                    state[run_id]['baseline'] = True
                print(f'Mirroring {run_id}', flush=True)
                warm_start_dev = dev_metrics(run_id, 0)
                if warm_start_dev:
                    trackio.log(warm_start_dev, step=0)
            if active == run_id:
                for record in history:
                    step = record['step']
                    values = metrics_for(record)
                    values.update(dev_metrics(run_id, step))
                    old = state[run_id].setdefault(str(step), {})
                    changed = {key: value for key, value in values.items() if old.get(key) != value}
                    if changed:
                        trackio.log(changed, step=step)
                        old.update(changed)
                state_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
                selected = read_json(OUTPUT / 'selection.json', {})
                result = read_json(OUTPUT / 'adapted_results_path.json', {})
                if result.get('tag') == f'{run_id}_adapted':
                    final = read_json(OUTPUT / f'results/{run_id}_adapted/summary.json', {})
                    if all(f'test_{lang}' in final for lang in ('en', 'es', 'it', 'fr', 'de', 'tr')):
                        final_metrics = {f'test/wer_{key.removeprefix("test_")}_percent': 100 * value['asr']['wer']
                                         for key, value in final.items() if key.startswith('test_')}
                        final_metrics['selection/best_step'] = selected.get('selected_step', 0)
                        trackio.log(final_metrics, step=history[-1]['step'])
                        trackio.finish()
                        state[run_id]['finished'] = 'heldout-evaluation-complete'
                        state_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
                        finished.add(run_id)
                        active = None
                        print(f'Final held-out scores imported for {run_id}', flush=True)
                elif (selected.get('accepted') is False
                      and (OUTPUT / 'selection.json').stat().st_mtime >= config_path.stat().st_mtime
                      and history[-1]['step'] >= config.get('steps', float('inf'))):
                    trackio.log({'selection/accepted': 0,
                                 'selection/report': trackio.Markdown(selected.get('reason', 'Rejected on development gates.'))},
                                step=history[-1]['step'])
                    trackio.finish()
                    state[run_id]['finished'] = 'rejected-on-development'
                    state_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
                    finished.add(run_id)
                    active = None
                    print(f'Rejected development outcome imported for {run_id}', flush=True)
        if not args.follow:
            if active is not None:
                trackio.finish()
            return
        time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default='parakeet-turkish')
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--dashboard', action='store_true')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    if args.dashboard:
        trackio.show(project=args.project, host='127.0.0.1', server_port=args.port,
                     open_browser=False, share=False, block_thread=True)
    else:
        try:
            follow(args)
        finally:
            trackio.finish()
