"""Bind completed, independently audited GPU profiles to the authorized resume."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs/parakeet_turkish'
LOOP = OUT / 'runs/adaptive_20261010_01'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    status = read(OUT / 'large_batch_window_status.json')
    assert status['status'] == 'all_jobs_finished'
    assert len(status['jobs']) == 30
    assert all(job['status'] == 'finished' and job['exit_code'] == 0 for job in status['jobs'])
    audit = read(OUT / 'large_batch_results_verification.json')
    assert audit['passed'] is True and audit['status'] == 'complete'
    assert audit['matrix_sha256_at_audit'] == sha(OUT / 'large_batch_matrix.json')
    assert audit['adoption_eligible'] is False
    publication = read(OUT / 'large_batch_trackio_publication.json')
    assert publication['audit_sha256'] == sha(OUT / 'large_batch_results_verification.json')
    assert publication['matrix_sha256'] == sha(OUT / 'large_batch_matrix.json')
    assert len(publication['runs']) == 2
    original = read(LOOP / 'loop_status.json')
    assert original['status'] == 'failed' and not original.get('active_run_id')
    assert original['failure_message'] == 'Trial ID/stop marker already exists.'
    assert sha(LOOP / 'loop_status.json') == sha(LOOP / 'original_status_after_pause.json')
    manifest_path = LOOP / 'gpu_benchmark_pause.json'
    manifest = read(manifest_path)
    assert manifest['benchmark_completed'] is False
    immutable = LOOP / 'gpu_benchmark_pause_checkpoint_verified.json'
    assert sha(immutable) == '4d13d9aacac1cac36819042a565252b93d20fc5701493a69563d0b3906920f32'
    evidence = [OUT / name for name in (
        'large_batch_window_status.json', 'large_batch_matrix.json',
        'large_batch_results_verification.json', 'large_batch_speed_report.json',
        'large_batch_trackio_publication.json',
        'large_batch_timing_arithmetic_independent_audit.json',
        'large_batch_typical_timing_default_b8.json',
        'large_batch_long_timing_default_b4.json',
        'batch_math_objective_independent_audit.json',
        'continuation_controller_cpu_verification.json')]
    assert all(path.is_file() for path in evidence)
    finished = datetime.now(timezone.utc)
    manifest.update(benchmark_completed=True, original_controller_exited=True,
        benchmark_evidence=[{'path': str(path.resolve()), 'sha256': sha(path)} for path in evidence],
        benchmark_verified_at_utc=finished.isoformat(),
        elapsed_pause_wall_seconds=(finished - datetime.fromisoformat(manifest['paused_at_utc'])).total_seconds(),
        benchmark_quality_scope='Fixed TRAIN microbenchmarks; no new ASR DEV/TEST predictions or batch adoption.',
        numerical_summary='All FP32 MATH larger-batch gates passed. Strict same-shape BF16 gradient gates failed including B1; exploratory timings retained.',
        resume_configuration='Same B1x4 screen A, exact optimizer/sampler/RNG restoration; original absolute deadline.')
    temporary = manifest_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    temporary.replace(manifest_path)
    print(json.dumps({'benchmark_completed': True, 'evidence_count': len(evidence),
        'elapsed_pause_minutes': manifest['elapsed_pause_wall_seconds'] / 60,
        'checkpoint_step': manifest['paused_step'], 'immutable_pause_sha256': sha(immutable)}))

if __name__ == '__main__':
    main()
