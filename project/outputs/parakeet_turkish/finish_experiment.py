"""Finite local completion worker for one already-authorized LoRA experiment.

Wait for actual completed training, finalize a development-selected checkpoint,
run the small confidence/speech checks, and independently verify final outputs.
This is a one-shot process, not a recurring automation. GPU work starts only
after the run-specific trainer writes its completed history and selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
LANGUAGES = ('en', 'es', 'it', 'fr', 'de', 'tr')
LOSS_FIELDS = ('tdt_loss', 'replay_kl', 'replay_duration_kl',
               'replay_feature_distillation', 'gradient_norm', 'elapsed_seconds')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_history(history, configured_steps):
    if not isinstance(history, list) or len(history) > configured_steps:
        raise RuntimeError('Training history shape or length differs from the configured run')
    for expected, record in enumerate(history, 1):
        if not isinstance(record, dict) or record.get('step') != expected:
            raise RuntimeError('Training history must contain each actual optimizer step exactly once')
        if any(not isinstance(record.get(field), (int, float)) or not math.isfinite(record[field])
               for field in LOSS_FIELDS):
            raise RuntimeError(f'Nonfinite/incomplete recorded training values at step {expected}')
    return history[-1]['step'] if history else 0


def validate_selection(selection, run_id, history, configured_steps):
    if selection.get('run_id') != run_id or type(selection.get('accepted')) is not bool:
        raise RuntimeError('Completed selection must identify this exact run and an explicit acceptance result')
    selected = selection.get('selected_step')
    if selection['accepted']:
        if type(selected) is not int or not 1 <= selected <= configured_steps:
            raise RuntimeError('Accepted selection has an invalid selected step')
        validation = history[selected-1].get('validation', {})
        if not validation.get('retention_gate_passed'):
            raise RuntimeError('Selected step did not pass the recorded development retention gate')
        if validation.get('turkish_wer') != selection.get('turkish_validation_wer'):
            raise RuntimeError('Selected Turkish development score differs from recorded training history')
        if not Path(selection['weights']).is_file():
            raise RuntimeError('Selected trainable checkpoint is missing')
    elif selected is not None:
        raise RuntimeError('Rejected selection unexpectedly identifies an accepted checkpoint')


def write_status(run_dir, **fields):
    path = run_dir/'completion_status.json'
    current = read_json(path) if path.exists() else {}
    write_json(path, {**current, **fields, 'updated_at_utc': utc_now()})


def wait_for_training(run_dir, run_id, configuration, config_sha, deadline, poll_seconds):
    history_path = run_dir/'training_history.json'
    config_path = run_dir/'run_config.json'
    config_mtime = config_path.stat().st_mtime_ns
    last_reported = None
    while True:
        if sha256_file(config_path) != config_sha:
            raise RuntimeError('Run configuration changed after the completion worker started')
        history = read_json(history_path) if history_path.exists() else []
        completed = validate_history(history, configuration['steps'])
        if history_path.exists() and history_path.stat().st_mtime_ns < config_mtime:
            raise RuntimeError('Training history predates this run configuration')
        if completed != last_reported:
            write_status(run_dir, stage='waiting_for_training', completed_steps=completed)
            print(f'{run_id}: actual completed optimizer steps {completed}/{configuration["steps"]}', flush=True)
            last_reported = completed
        training_selection = run_dir/'training_selection.json'
        selection_path = training_selection if training_selection.exists() else run_dir/'selection.json'
        if completed >= configuration['steps'] and selection_path.exists():
            if selection_path.stat().st_mtime_ns < config_mtime:
                raise RuntimeError('Selection predates this run configuration')
            selection = read_json(selection_path)
            validate_selection(selection, run_id, history, configuration['steps'])
            return history, selection
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f'Training did not complete within the bounded wait; only {completed} steps are recorded')
        time.sleep(min(poll_seconds, remaining))


def run_child(run_dir, stage, command, deadline, *, cwd=ROOT, stdout_path=None):
    remaining = deadline-time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f'Completion deadline reached before {stage}')
    log_dir = run_dir/'completion_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = Path(stdout_path) if stdout_path else log_dir/f'{stage}.stdout.log'
    stderr_path = log_dir/f'{stage}.stderr.log'
    write_status(run_dir, stage=stage, last_command=command, stdout_log=str(stdout_path), stderr_log=str(stderr_path))
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    with stdout_path.open('wb') as output, stderr_path.open('wb') as error:
        completed = subprocess.run(command, cwd=cwd, stdout=output, stderr=error,
                                   timeout=remaining, creationflags=flags, check=False)
    if completed.returncode:
        raise RuntimeError(f'{stage} failed with exit {completed.returncode}; retained logs: {stdout_path}, {stderr_path}')


def build_report(run_id, configuration, history, selection, status, *, reason=None, verification=False):
    run_dir = OUTPUT/'runs'/run_id
    completed = validate_history(history, configuration['steps'])
    report = {
        'schema_version': 1, 'run_id': run_id, 'status': status, 'updated_at_utc': utc_now(),
        'configured_steps': configuration['steps'], 'completed_steps': completed,
        'selected_step': selection.get('selected_step'),
        'development_selection_accepted': bool(selection.get('accepted', False)),
        'heldout_tests_evaluated': bool(selection.get('heldout_tests_evaluated', False)),
        'checkpoint': selection.get('checkpoint') if selection.get('accepted') else None,
        'baseline_results_path': str(OUTPUT/'results/base_v3/summary.json'),
        'adapted_results_path': None, 'development_gate': selection.get('merged_development_gate'),
        'per_language_test': {}, 'confidence_calibration': {},
        'pilot_improvement_and_retention_gates_passed': None,
        'verification': {'path': str(run_dir/'final_results_verification.json'), 'passed': verification},
        'artifacts': {
            'configuration': str(run_dir/'run_config.json'), 'history': str(run_dir/'training_history.json'),
            'selection': str(run_dir/'selection.json'),
            'optional_full_split_evaluation': str(OUTPUT/'optional_full_split_evaluation.txt'),
        },
        'limitations': [
            'FLEURS pilot: 50 development and 100 heldout test recordings per language; small samples do not establish full-split accuracy.',
            'Retention is measured only for English, Spanish, Italian, French and German; other languages were not tested.',
            'Checkpoint selection and confidence temperatures use development data only; heldout results never select a checkpoint.',
            'A development-accepted checkpoint does not guarantee heldout retention or useful Turkish accuracy.',
            f'Training uses {configuration.get("turkish_clips_used", "a limited number of")} unique Turkish clips; repeated optimizer draws are not additional unique data.',
            'The ten-English-development confidence perturbation probe is descriptive and does not establish broad robustness.',
        ],
    }
    if reason:
        report['reason'] = reason
    if history:
        report['turkish_unique_training_clips_seen'] = history[-1].get('turkish_unique_clips_seen')
    if report['heldout_tests_evaluated']:
        pointer = read_json(run_dir/'adapted_results_path.json')
        baseline = read_json(OUTPUT/'results/base_v3/summary.json')
        adapted = read_json(pointer['summary'])
        comparison = read_json(run_dir/'benchmark_comparison.json')
        report['adapted_results_path'] = pointer['summary']
        report['artifacts'].update(final_results_tag=pointer['tag'],
            paired_bootstrap=str(run_dir/'benchmark_comparison.json'),
            calibration=str(run_dir/'confidence_calibration.json'),
            finalization_provenance=str(run_dir/'finalization_provenance.json'),
            selected_trainables=selection.get('selected_trainables'))
        report['confidence_calibration'] = read_json(run_dir/'confidence_calibration.json')
        retention = True
        for language in LANGUAGES:
            before, after = baseline[f'test_{language}']['asr'], adapted[f'test_{language}']['asr']
            if before['utterances'] != 100 or after['utterances'] != 100:
                raise RuntimeError(f'Final pilot summary must cover 100 matched test recordings: {language}')
            delta = after['wer']-before['wer']
            heldout_gate = delta <= max(.005, .1*before['wer']) if language != 'tr' else None
            if heldout_gate is not None:
                retention = retention and heldout_gate
            report['per_language_test'][language] = {
                'baseline_asr': before, 'adapted_asr': after,
                'wer_delta_percentage_points': 100*delta,
                'cer_delta_percentage_points': 100*(after['cer']-before['cer']),
                'paired_bootstrap': comparison[language],
                'heldout_retention_gate_passed': heldout_gate,
            }
        turkish = report['per_language_test']['tr']
        report['turkish_test_improved'] = turkish['adapted_asr']['wer'] < turkish['baseline_asr']['wer']
        report['heldout_retention_gate_passed'] = retention
        report['pilot_improvement_and_retention_gates_passed'] = retention and report['turkish_test_improved']
        wer = turkish['adapted_asr']['wer']
        report['turkish_accuracy_assessment'] = (
            f'Turkish ASR remains highly inaccurate: measured heldout pilot WER is {100*wer:.3f}%. This is a research checkpoint.'
            if wer >= .5 else f'Measured Turkish pilot WER is {100*wer:.3f}%; suitability requires more data and evaluation.')
    return report


def save_report(report):
    run_dir = OUTPUT/'runs'/report['run_id']
    text = [f'Parakeet Turkish pilot: {report["run_id"]}',
            f'Status: {report["status"]}. Actual completed optimizer steps: {report["completed_steps"]}/{report["configured_steps"]}.',
            f'Development-selected step: {report["selected_step"]}; accepted: {report["development_selection_accepted"]}.',
            f'Final heldout benchmark completed: {report["heldout_tests_evaluated"]}.']
    if report.get('reason'):
        text.append(report['reason'])
    if report['per_language_test']:
        text += ['', 'Heldout FLEURS pilot: 100 matched recordings per language; WER/CER are percentages.',
                 'Language  Baseline WER/CER  Selected WER/CER  WER change (percentage points; 95% paired sentence-cluster interval)']
        for language, result in report['per_language_test'].items():
            before, after = result['baseline_asr'], result['adapted_asr']
            low, high = result['paired_bootstrap']['wer']['ci_percentage_points']
            text.append(f'{language:8} {100*before["wer"]:7.3f}/{100*before["cer"]:7.3f}  '
                        f'{100*after["wer"]:7.3f}/{100*after["cer"]:7.3f}  '
                        f'{result["wer_delta_percentage_points"]:+.3f} [{low:+.3f},{high:+.3f}]')
        text += ['', report['turkish_accuracy_assessment'],
                 f'Turkish improved on this pilot test: {report["turkish_test_improved"]}.',
                 f'All five heldout retention gates passed: {report["heldout_retention_gate_passed"]}.',
                 f'Pilot relative-improvement and retention gates passed: {report["pilot_improvement_and_retention_gates_passed"]}.',
                 'A lower Turkish error rate can still leave highly inaccurate ASR; assess the measured WER/CER directly.',
                 f'Published local HF model/processor: {report["checkpoint"]}.']
    text += ['', f'Independent final-output audit passed: {report["verification"]["passed"]}.', '', 'Limitations:']
    text.extend('- '+value for value in report['limitations'])
    for folder in (run_dir, OUTPUT):
        write_json(folder/'experiment_report.json', report)
        (folder/'final_report.txt').write_text('\n'.join(text)+'\n', encoding='utf-8')


def update_backlog(run_dir, report, deadline):
    binary = str(OUTPUT/'tools/backlog.exe')
    notes = (f'{utc_now()}: Run {report["run_id"]} actually completed {report["completed_steps"]}/{report["configured_steps"]} '
             f'optimizer steps. Completion status: {report["status"]}. Development selection accepted: '
             f'{report["development_selection_accepted"]}; selected step {report["selected_step"]}. '
             f'Heldout benchmark completed: {report["heldout_tests_evaluated"]}; independent final audit passed: '
             f'{report["verification"]["passed"]}. Pilot relative-improvement and retention gates passed on heldout data: '
             f'{report["pilot_improvement_and_retention_gates_passed"]}. See run-specific experiment_report.json/final_report.txt for '
             'measured Turkish WER/CER, all preservation-language deltas, confidence calibration, paired uncertainty '
             'and limitations. Completion is separate from useful Turkish accuracy or measured retention. ')
    complete = report['status'] == 'complete' and report['verification']['passed']
    if complete:
        research_criterion = 'Save/load a functional research model and processor; verify transcription and publish measured accuracy with configuration and limitations'
        research_notes = notes+report['turkish_accuracy_assessment']+' Functional reload/transcription checks do not establish useful Turkish accuracy.'
        run_child(run_dir, 'backlog_research_export_scope', [binary,'task','edit','ASR-5','--remove-ac','4',
            '--ac',research_criterion,'--append-notes',research_notes,
            '--add-ref',f'runs/{report["run_id"]}/experiment_report.json'], deadline, cwd=OUTPUT)
        run_child(run_dir, 'backlog_research_export_complete', [binary,'task','edit','ASR-5','--status','Done',
            '--check-ac','3','--check-ac','4'], deadline, cwd=OUTPUT)
    command = [binary, 'task', 'edit', 'ASR-11', '--status', 'Done' if complete else 'In Progress',
               '--append-notes', notes, '--add-ref', f'runs/{report["run_id"]}/experiment_report.json']
    if complete:
        command += ['--check-ac', '4', '--check-ac', '5']
    run_child(run_dir, 'backlog_training', command, deadline, cwd=OUTPUT)
    command = [binary, 'task', 'edit', 'ASR-8', '--status', 'Done' if complete else 'In Progress',
               '--append-notes', notes, '--add-ref', f'runs/{report["run_id"]}/experiment_report.json']
    if complete:
        for index in range(1,5):
            command += ['--check-ac', str(index)]
    run_child(run_dir, 'backlog_final_evaluation', command, deadline, cwd=OUTPUT)
    run_child(run_dir, 'backlog_refresh', [sys.executable, '-X', 'utf8', str(OUTPUT/'refresh_backlog.py')], deadline, cwd=OUTPUT)


def run(args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', args.run_id):
        raise ValueError('run-id must be a short filename-safe identifier')
    if not math.isfinite(args.max_wait_seconds) or args.max_wait_seconds <= 0 or not 1 <= args.poll_seconds <= 60:
        raise ValueError('Wait must be finite and positive; poll interval must be 1–60 seconds')
    run_dir = OUTPUT/'runs'/args.run_id
    configuration = read_json(run_dir/'run_config.json')
    if configuration.get('run_id') != args.run_id or type(configuration.get('steps')) is not int or configuration['steps'] < 1:
        raise RuntimeError('A valid existing run-specific configuration is required')
    config_sha = sha256_file(run_dir/'run_config.json')
    deadline = time.monotonic()+args.max_wait_seconds
    history, selection = [], {}
    write_status(run_dir, run_id=args.run_id, started_at_utc=utc_now(), configured_steps=configuration['steps'],
                 config_sha256=config_sha, worker_sha256=sha256_file(Path(__file__)), finite_one_shot=True)
    python = [sys.executable, '-X', 'utf8']
    try:
        history, selection = wait_for_training(run_dir, args.run_id, configuration, config_sha, deadline, args.poll_seconds)
        if selection['accepted']:
            run_child(run_dir, 'finalize_lora', python+[str(OUTPUT/'finalize_lora.py'), '--run-id', args.run_id], deadline)
            selection = read_json(run_dir/'selection.json')
        if not selection['accepted']:
            report = build_report(args.run_id, configuration, history, selection, 'rejected', reason=selection.get('reason'))
            save_report(report)
            update_backlog(run_dir, report, deadline)
            write_status(run_dir, stage='rejected', finished_at_utc=utc_now())
            return 0
        if not selection.get('heldout_tests_evaluated') or not selection.get('merged'):
            raise RuntimeError('Finalizer did not complete the merged export and heldout evaluation')
        checkpoint = Path(selection['checkpoint']).resolve()
        if checkpoint != (OUTPUT/'checkpoint').resolve():
            raise RuntimeError('Finalizer did not publish the expected local checkpoint')
        base = (ROOT/'work/base_model_path.txt').read_text(encoding='utf-8').strip()
        probe_path = run_dir/'confidence_robustness.json'
        run_child(run_dir, 'confidence_robustness', python+[str(OUTPUT/'confidence_robustness.py'),
            '--manifest', str(ROOT/'work/data/validation_en.jsonl'), '--limit', '10',
            '--model', 'v3='+base, '--model', 'selected='+str(checkpoint), '--output', str(probe_path)], deadline)
        pointer = read_json(run_dir/'adapted_results_path.json')
        run_child(run_dir, 'verify_confidence_robustness', python+[str(OUTPUT/'verify_confidence_robustness.py'),
            '--input', str(probe_path), '--reference', 'v3='+str(OUTPUT/'results/base_v3'),
            '--reference', 'selected='+str(OUTPUT/'results'/pointer['tag'])], deadline)
        probe_audit_path = probe_path.with_name(probe_path.stem+'_independent_audit.json')
        probe_audit = read_json(probe_audit_path)
        if not probe_audit.get('passed') or not probe_audit.get('all_checks_passed'):
            raise RuntimeError('Independent confidence perturbation audit did not pass')
        tr = [json.loads(line) for line in (ROOT/'work/data/validation_tr.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()][0]
        smoke_path = run_dir/'transcription_smoke.json'
        run_child(run_dir, 'transcription_smoke', python+[str(OUTPUT/'transcribe.py'), tr['audio_filepath'],
            '--language', 'tr', '--model-path', str(checkpoint),
            '--calibration-path', str(run_dir/'confidence_calibration.json')], deadline, stdout_path=smoke_path)
        smoke = read_json(smoke_path)
        if not isinstance(smoke.get('transcript'), str) or not isinstance(smoke.get('words'), list):
            raise RuntimeError('Transcription smoke did not return a valid transcript and word-confidence JSON')
        predictions = [json.loads(line) for line in (OUTPUT/'results'/pointer['tag']/'validation_tr.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        matching = [row for row in predictions if row['source_id'] == tr['source_id']]
        calibration = read_json(run_dir/'confidence_calibration.json')['tr']['temperature']
        if len(matching) != 1 or smoke['transcript'].strip() != matching[0]['hypothesis'].strip():
            raise RuntimeError('Published-checkpoint smoke differs from the matching merged development transcript')
        if smoke.get('confidence_temperature') != calibration:
            raise RuntimeError('Published-checkpoint smoke did not use the selected development-fitted Turkish temperature')
        for word in smoke['words']:
            if any(not isinstance(word.get(key), (int,float)) or not math.isfinite(word[key]) or not 0 <= word[key] <= 1
                   for key in ('raw_confidence','calibrated_confidence','conditional_nonblank_confidence')):
                raise RuntimeError('Published-checkpoint smoke returned invalid word confidence')
        write_json(run_dir/'transcription_smoke_reference.json', {
            'source_id': tr['source_id'], 'reference': tr['text'], 'audio_filepath': tr['audio_filepath'],
            'wav_sha256': tr['wav_sha256'], 'selection': 'First fixed Turkish development manifest row; no selection by transcription quality'})
        report = build_report(args.run_id, configuration, history, selection, 'verification_pending')
        report['artifacts'].update(confidence_robustness=str(probe_path),
                                  confidence_robustness_verification=str(probe_audit_path),
                                  transcription_smoke=str(smoke_path))
        save_report(report)
        run_child(run_dir, 'verify_final_results', python+[str(OUTPUT/'verify_final_results.py'),
            '--run-id', args.run_id, '--report', str(run_dir/'experiment_report.json')], deadline)
        audit = read_json(run_dir/'final_results_verification.json')
        if not audit.get('passed'):
            raise RuntimeError('Independent final-output audit did not pass')
        report.update(status='complete', verification={'path': str(run_dir/'final_results_verification.json'), 'passed': True})
        save_report(report)
        # The auditor hashes its report input. Verify the final saved report
        # after adding completion/pass fields so that its durable hash is exact;
        # do not rewrite the report after this pass.
        run_child(run_dir, 'verify_completed_report', python+[str(OUTPUT/'verify_final_results.py'),
            '--run-id', args.run_id, '--report', str(run_dir/'experiment_report.json')], deadline)
        if not read_json(run_dir/'final_results_verification.json').get('passed'):
            raise RuntimeError('The saved completed report did not pass its independent audit')
        update_backlog(run_dir, report, deadline)
        write_status(run_dir, stage='complete', finished_at_utc=utc_now(),
                     pilot_improvement_and_retention_gates_passed=report['pilot_improvement_and_retention_gates_passed'])
        print(f'Finite completion finished: {run_dir/"final_report.txt"}', flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        # Preserve observed partial progress and every stage log; planned steps
        # or a partially published checkpoint are never reported as completion.
        if (run_dir/'training_history.json').exists():
            history = read_json(run_dir/'training_history.json')
        if (run_dir/'selection.json').exists():
            selection = read_json(run_dir/'selection.json')
        try:
            report = build_report(args.run_id, configuration, history, selection, 'failed', reason=f'{type(error).__name__}: {error}')
            save_report(report)
            # Failure bookkeeping is local CPU work with its own short bound,
            # including when the training/completion deadline was exhausted.
            try:
                update_backlog(run_dir, report, time.monotonic()+60)
            except Exception as backlog_error:
                write_json(run_dir/'completion_backlog_failure.json', {'error':str(backlog_error),'at_utc':utc_now()})
        except Exception as reporting_error:
            write_json(run_dir/'completion_failure.json', {'run_id':args.run_id,'error':str(error),'report_error':str(reporting_error),
                       'last_completed_step':history[-1].get('step') if history else 0,'at_utc':utc_now()})
        write_status(run_dir, stage='failed', error=f'{type(error).__name__}: {error}', finished_at_utc=utc_now())
        print(f'Completion failed; retained run status and logs: {error}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='lora_20261010_01')
    parser.add_argument('--max-wait-seconds', type=float, default=4*60*60)
    parser.add_argument('--poll-seconds', type=float, default=30)
    raise SystemExit(run(parser.parse_args()))
