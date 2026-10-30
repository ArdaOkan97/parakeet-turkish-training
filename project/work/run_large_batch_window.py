"""Finite, sequential, isolated physical-batch GPU benchmark queue."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs/parakeet_turkish'
WORK = ROOT / 'work/large_batches'
LOOP = OUT / 'runs/adaptive_20261010_01'
SCRIPT = OUT / 'profile_large_batches.py'
PAUSE = LOOP / 'gpu_benchmark_pause_checkpoint_verified.json'
MATRIX = OUT / 'large_batch_matrix.json'
STATUS = OUT / 'large_batch_window_status.json'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)

def now():
    return datetime.now(timezone.utc)

def main():
    pause = json.loads(PAUSE.read_text())
    assert pause['status'] == 'paused_checkpoint_verified'
    original = json.loads((LOOP / 'loop_status.json').read_text())
    assert original['status'] == 'failed' and not original.get('active_run_id')
    assert not MATRIX.exists() and not STATUS.exists()
    WORK.mkdir(parents=True, exist_ok=True)
    script_sha = sha(SCRIPT)
    deadline = datetime.fromisoformat(pause['paused_at_utc']) + timedelta(minutes=45)
    state = {'status': 'running', 'started_at_utc': now().isoformat(),
             'deadline_at_utc': deadline.isoformat(), 'script_sha256': script_sha,
             'pause_provenance_path': str(PAUSE), 'pause_provenance_sha256': sha(PAUSE),
             'effective_batch': 16, 'jobs': [], 'all_cases_requested': 30,
             'scope': 'Separate processes; training stopped; one GPU job at a time. No adoption or DEV quality result.'}
    environment = dict(os.environ, PYTHONUTF8='1', HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    write(STATUS, state)
    for family in ('typical', 'long'):
        jobs = [('fp32-reference', 1, 'math', None)]
        fp32_ref = WORK / f'large_batch_{family}_fp32_reference_math_b1.pt'
        jobs += [('fp32-parity', batch, 'math', fp32_ref) for batch in (2,4,8,16)]
        for batch in (1,2,4,8,16):
            ref = WORK / f'large_batch_{family}_bf16_reference_default_b{batch}.pt'
            jobs += [('bf16-reference', batch, 'default', None), ('timing', batch, 'default', ref)]
        for mode, batch, backend, reference in jobs:
            remaining = (deadline - now()).total_seconds()
            if remaining < 90:
                state['status'] = 'budget_exhausted'
                break
            assert sha(SCRIPT) == script_sha, 'Profiler source changed; refuse mixed-provenance jobs.'
            case_id = f'large_batch_{family}_{mode.replace("-", "_")}_{backend}_b{batch}'
            command = [sys.executable, '-X', 'utf8', '-u', str(SCRIPT), '--case', family,
                       '--batch-size', str(batch), '--mode', mode, '--backend', backend,
                       '--updates', '10' if family == 'typical' else '3', '--warmup', '2',
                       '--gpu-context', 'solo', '--max-gpu-gb', '14.5',
                       '--pause-provenance', str(PAUSE), '--matrix-output', str(MATRIX)]
            if reference:
                command += ['--reference-path', str(reference)]
            if mode == 'timing':
                command += ['--exploratory-timing']
            log = WORK / f'{case_id}.log'
            job = {'case_id':case_id,'command':command,'started_at_utc':now().isoformat(),
                   'log':str(log),'status':'running'}
            state['jobs'].append(job)
            write(STATUS, state)
            if reference and not reference.exists():
                job.update(status='skipped_missing_reference', reference_path=str(reference))
            else:
                started = time.monotonic()
                try:
                    with log.open('wb') as stream:
                        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                                                stderr=subprocess.STDOUT, timeout=min(300, remaining-45),
                                                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    job.update(status='finished',exit_code=result.returncode)
                except subprocess.TimeoutExpired:
                    job.update(status='timed_out', timeout_seconds=min(300,remaining-45))
                job['wall_seconds'] = time.monotonic()-started
            job['finished_at_utc'] = now().isoformat()
            profile = OUT / f'{case_id}.json'
            if profile.exists():
                report = json.loads(profile.read_text())
                job.update(report_status=report.get('status'), report_path=str(profile), report_sha256=sha(profile))
                if report.get('timing'):
                    job['throughput'] = report['timing'].get('end_to_end',{}).get('audio_min_per_wall_min')
            write(STATUS,state)
            print(json.dumps({key:job.get(key) for key in ('case_id','status','exit_code','report_status','throughput','wall_seconds')}), flush=True)
        if state['status'] != 'running':
            break
    if state['status'] == 'running':
        state['status'] = 'all_jobs_finished'
    state['finished_at_utc'] = now().isoformat()
    write(STATUS,state)
    print(json.dumps({'status':state['status'],'jobs':len(state['jobs']),'matrix':str(MATRIX)}),flush=True)

if __name__ == '__main__':
    main()
