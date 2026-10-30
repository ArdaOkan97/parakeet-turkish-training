"""Mirror actual native Lightning records into the existing local Trackio project."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import track_experiment as tracking


def read(path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--follow', action='store_true')
    args = parser.parse_args()
    run_dir = (tracking.OUTPUT / 'runs' / args.run_id).resolve()
    if not run_dir.is_relative_to((tracking.OUTPUT / 'runs').resolve()):
        raise ValueError('Run must remain in the experiment workspace')
    state_path = tracking.ROOT / 'work' / f'trackio_full_nemo_{args.run_id}.json'
    state = read(state_path, {})
    initialized = False
    try:
        while True:
            config = read(run_dir / 'resolved_trial_config.json')
            if config and not initialized:
                tracking.trackio.init(project='parakeet-turkish', name=args.run_id,
                                      group='Full NeMo training', config=config,
                                      resume='allow', gpu_log_interval=10.)
                initialized = True
                print(f'Mirroring actual full-NeMo trial {args.run_id}', flush=True)
            if initialized:
                values_by_step = {}
                telemetry_path = run_dir / 'training_telemetry.jsonl'
                if telemetry_path.exists():
                    for line in telemetry_path.read_text(encoding='utf-8').splitlines():
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue  # A live writer may have an unfinished last record.
                        values_by_step.setdefault(row['step'], {}).update(row['metrics'])
                for path in run_dir.glob('development_gate_step_*.json'):
                    gate = read(path)
                    if not gate:
                        continue
                    values = values_by_step.setdefault(gate['step'], {})
                    values.update({f'dev/wer_{lang}_percent': 100 * wer
                                   for lang, wer in gate['per_language_wer'].items()})
                    values['retention/gate_passed'] = int(gate['retention_gate_passed'])
                for step, values in sorted(values_by_step.items()):
                    prior = state.setdefault(str(step), {})
                    changed = {k: v for k, v in values.items() if prior.get(k) != v}
                    if changed:
                        tracking.trackio.log(changed, step=step)
                        prior.update(changed)
                state_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
                outcome = read(run_dir / 'trial_outcome.json')
                if outcome:
                    tracking.trackio.log({
                        'native/completed_optimizer_updates': outcome['actual_completed_optimizer_updates'],
                        'native/trial_finished': 1,
                        'native/trial_failed': int(outcome['status'] == 'failed'),
                        'native/stop_reason': tracking.trackio.Markdown(outcome.get('stop_reason') or outcome['status']),
                    }, step=outcome['actual_completed_optimizer_updates'])
                    print(json.dumps({'mirror_finished': True, 'run_id': args.run_id,
                                      'outcome': outcome['status']}), flush=True)
                    return
            if not args.follow:
                return
            time.sleep(5)
    finally:
        if initialized:
            tracking.trackio.finish()


if __name__ == '__main__':
    main()
