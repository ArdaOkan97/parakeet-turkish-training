"""One owned, time-bounded CPU audit; no training routing changes."""
from pathlib import Path
import ctypes, datetime, hashlib, json, os, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'outputs/parakeet_turkish'
REPORT = OUTPUT / 'training_input_audit_reuse_v2_actual_measurement.json'
LOG = ROOT / 'work/training_input_audit_reuse_v2_actual_measurement.log'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(value):
    REPORT.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def worker():
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    started = time.perf_counter()
    priority = {'requested': 'BELOW_NORMAL_PRIORITY_CLASS', 'applied': False}
    if os.name == 'nt':
        api = ctypes.WinDLL('kernel32', use_last_error=True)
        api.GetCurrentProcess.restype = ctypes.c_void_p
        api.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        api.SetPriorityClass.restype = ctypes.c_int
        api.GetPriorityClass.argtypes = [ctypes.c_void_p]
        api.GetPriorityClass.restype = ctypes.c_ulong
        handle = api.GetCurrentProcess()
        priority['applied'] = bool(api.SetPriorityClass(handle, 0x4000))
        priority['actual_priority_class'] = int(api.GetPriorityClass(handle))
        if not priority['applied']:
            priority['error'] = ctypes.get_last_error()
    sys.path.insert(0, str(OUTPUT))
    import public_turkish_data_policy_v3 as policy
    from training_input_audit_reuse_v2 import audit_public_training_inputs_reused
    audio_calls = []

    def forbidden_audio(*args, **kwargs):
        audio_calls.append('unexpected_soundfile_audio_API')
        raise RuntimeError('Stat-only measurement attempted a WAV header/PCM read')

    # The frozen policy imports soundfile, but its False branch must not use it.
    policy.sf.SoundFile = forbidden_audio
    policy.sf.info = forbidden_audio
    policy.sf.read = forbidden_audio
    inputs = {'manifest': ROOT / 'work/data/tr_public_v3_train.jsonl',
              'replay_manifest': ROOT / 'work/data/replay_train.jsonl',
              'preparation_proof': OUTPUT / 'tr_public_v3_preparation.json',
              'reuse_receipt': OUTPUT / 'public_data_full_audit_reuse_v1/reuse_receipt.json'}
    result = {'schema': 1, 'kind': 'training_input_audit_reuse_v2_actual_stat_only_measurement',
              'started_at_utc': started_at, 'priority': priority,
              'scope': {'CPU_only': True, 'full_file_count_once': True,
                        'concurrent_main_training': True, 'controlled_speed_comparison': False,
                        'routing_changes': False, 'audio_changes': False,
                        'soundfile_dependency_imported_by_frozen_policy': True,
                        'no_source_or_extraction_parquet_hash_checks_removed': True},
              'helper_path': str(OUTPUT / 'training_input_audit_reuse_v2.py'),
              'helper_sha256': sha(OUTPUT / 'training_input_audit_reuse_v2.py'),
              'requested_inputs': {key: str(path) for key, path in inputs.items()},
              'caveats': ['One concurrent duration measurement, not an A/B speedup test.',
                          'Filesystem cache and concurrent training CPU/I/O load are uncontrolled.',
                          'Other runs reported 72.359 s and 334.831 s for normal header audits; those are not matched controls.',
                          'Existing fingerprint metadata limitation remains: same-size/same-mtime content replacement is not detected.']}
    try:
        audit = audit_public_training_inputs_reused(**inputs, check_audio_headers=False)
        result.update(status='complete', passed=audit['passed'], audit=audit)
    except Exception as error:
        result.update(status='failed', passed=False, error_type=type(error).__name__, error=str(error))
    result.update(completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  process_elapsed_seconds=time.perf_counter() - started,
                  soundfile_audio_calls=len(audio_calls), Torch_imported='torch' in sys.modules,
                  NeMo_imported=any(name == 'nemo' or name.startswith('nemo.') for name in sys.modules))
    write(result)
    print(json.dumps({'status': result['status'], 'passed': result['passed'],
                      'process_elapsed_seconds': result['process_elapsed_seconds'],
                      'soundfile_audio_calls': len(audio_calls), 'Torch_imported': 'torch' in sys.modules}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        raise SystemExit(worker())
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with LOG.open('w', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'],
                                 cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        try:
            code = child.wait(timeout=180)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
            write({'schema': 1, 'kind': 'training_input_audit_reuse_v2_actual_stat_only_measurement',
                   'status': 'timeout', 'passed': False, 'timeout_seconds': 180,
                   'owned_CPU_worker_pid': child.pid, 'GPU_work': False,
                   'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
            code = 124
    print(json.dumps({'exit_code': code, 'report': str(REPORT), 'log': str(LOG)}))
    raise SystemExit(code)
