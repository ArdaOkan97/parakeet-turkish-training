"""Reclassify two confirmed original-base rows; retain every row and value.

Root must stop the old tracking mirror before --apply. Training is unaffected.
Default is read-only. This is deliberately scoped to the two native-memory runs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
RUNS = [
    'adaptive_20261011_native_memory_fix_01_native_resume',
    'adaptive_20261011_native_memory_fix_01_public_rank8_replay25_screen',
]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dev(summary):
    return {f'dev/wer_{k.removeprefix("validation_")}_percent': 100*v['asr']['wer']
            for k, v in summary.items() if k.startswith('validation_')}


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    baseline_path = OUTPUT/'results/base_v3/summary.json'
    expected_base = dev(read(baseline_path))
    report_dir = OUTPUT/'trackio_step0_namespace_repair_v1'
    receipt_path = report_dir/'repair_receipt.json'
    state_path = ROOT/'work/trackio_continuation_mirror_state_v3.json'
    if args.apply and (receipt_path.exists() or state_path.exists()):
        raise RuntimeError('Preserve existing receipt/state; this scoped repair is one-time')
    db_path = OUTPUT/'trackio/parakeet-turkish.db'
    con = sqlite3.connect(str(db_path) if args.apply else db_path.as_uri()+'?mode=ro',
                          uri=not args.apply, timeout=5)
    con.row_factory = sqlite3.Row
    con.execute('BEGIN IMMEDIATE' if args.apply else 'BEGIN')
    changes, checked = [], []
    state_source = ROOT/'work/trackio_continuation_mirror_state_v2.json'
    mirror_state = read(state_source)
    try:
        for run in RUNS:
            summary_path = OUTPUT/f'results/{run}_step_0/summary.json'
            rows = [dict(r) for r in con.execute(
                'SELECT * FROM metrics WHERE run_name=? AND step=0 ORDER BY id', (run,))]
            if not rows:
                checked.append({'run_id': run, 'status': 'not_logged_yet; v3 will classify correctly'})
                continue
            actual = dev(read(summary_path))
            initial = read(OUTPUT/f'runs/{run}/initial_development.json')
            assert actual['dev/wer_tr_percent'] == 100*initial['turkish_wer']
            assert any(all(json.loads(r['metrics']).get(k) == v for k, v in actual.items())
                       for r in rows), 'Actual warmstart DEV row missing'
            wrong = [r for r in rows if json.loads(r['metrics']) == expected_base]
            assert len(wrong) == 1, 'Expected exactly one confirmed original-base row'
            row = wrong[0]
            after_metrics = {k.replace('dev/wer_', 'reference/base_v3_wer_', 1): v
                             for k, v in expected_base.items()}
            assert sorted(after_metrics.values()) == sorted(expected_base.values())
            after = dict(row, metrics=encoded(after_metrics))
            changes.append({'run_id': run, 'before': row, 'after': after,
                            'actual_step0_summary_path': str(summary_path),
                            'actual_step0_summary_sha256': sha(summary_path),
                            'actual_step0_dev': actual})
            mirror_state.setdefault(run, {})['reference_base_v3'] = True
            mirror_state[run].setdefault('0', {}).update(actual)
            if args.apply:
                count = con.execute('UPDATE metrics SET metrics=? WHERE id=? AND run_name=? AND metrics=?',
                                    (after['metrics'], row['id'], run, row['metrics'])).rowcount
                assert count == 1, 'Concurrent row changed; transaction aborted'
                current = dict(con.execute('SELECT * FROM metrics WHERE id=?', (row['id'],)).fetchone())
                assert current == after, 'Row identity/timestamp/source fields changed'
            checked.append({'run_id': run, 'status': 'confirmed', 'step0_rows': len(rows),
                            'base_row_id': row['id'], 'actual_TR_step0_percent': actual['dev/wer_tr_percent']})
        report = {'schema': 1, 'kind': 'scoped_original_base_DEV_namespace_repair_v1',
                  'observed_at_utc': datetime.now(timezone.utc).isoformat(),
                  'applied': args.apply, 'passed': True, 'database_path': str(db_path),
                  'scope': RUNS, 'baseline_source_path': str(baseline_path),
                  'baseline_source_sha256': sha(baseline_path), 'checked': checked,
                  'changes': changes, 'rows_deleted': 0, 'values_changed': 0,
                  'training_files_changed': False, 'historical_runs_outside_scope_changed': False,
                  'new_mirror_path': str(OUTPUT/'track_continuation_v3.py'),
                  'new_mirror_sha256': sha(OUTPUT/'track_continuation_v3.py'),
                  'mirror_state_seed_path': str(state_path),
                  'previous_mirror_state_path': str(state_source),
                  'previous_mirror_state_sha256': sha(state_source)}
        if args.apply:
            report_dir.mkdir(exist_ok=False)
            # Persist the complete original rows before any commit for reversal.
            (report_dir/'original_rows.json').write_text(encoded(changes)+'\n', encoding='utf-8')
            state_path.write_text(json.dumps(mirror_state, indent=2)+'\n', encoding='utf-8')
            con.commit()
            receipt_path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        else:
            con.rollback()
        print(json.dumps({k: report[k] for k in ('applied', 'passed', 'checked', 'rows_deleted', 'values_changed')}))
        if args.apply:
            print(str(receipt_path))
    except BaseException:
        con.rollback()
        raise
    finally:
        con.close()


if __name__ == '__main__':
    main()
