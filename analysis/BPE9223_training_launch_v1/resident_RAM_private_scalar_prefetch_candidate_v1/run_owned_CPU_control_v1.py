"""One reviewed inactive CPU control, with independent actual process monitoring."""
import hashlib, importlib.util, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
import psutil

N = Path(r'E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1')
O = N/'resident_RAM_private_scalar_prefetch_candidate_v1'
REVIEW = N/'ROOT_private_scalar_prefetch_CPU_control_source_read_review_v1.json'
OUT = N/'resident_RAM_private_scalar_prefetch_actual_01'
ROOT = Path(r'E:/gpu_training_scratchpad')
P = Path(r'C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
S = P/'outputs/parakeet_turkish'
sys.path.insert(0, str(S))


def pin(p):
    p = Path(p).resolve()
    with p.open('rb') as f:
        sha = hashlib.file_digest(f, 'sha256').hexdigest()
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha}


def save(p, value):
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def identity(p):
    return {'pid': p.pid, 'create_time': p.create_time(), 'ppid': p.ppid()}


def alive(w):
    try:
        return psutil.Process(w['pid']).create_time() == w['create_time']
    except psutil.NoSuchProcess:
        return False


def preflight():
    require(sys.flags.utf8_mode == 1 and os.environ.get('PYTHONUTF8') == '1' and
            os.environ.get('CUDA_VISIBLE_DEVICES') == '-1', 'CPU-only upfront UTF8')
    require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),
            'All four thread caps exactly1')
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    require(shutil.disk_usage(ROOT).free >= 100000000000, 'E100GB reserve')
    require(pin(REVIEW)['sha256'] == 'cbdfee0d324be230a6d6f61a4c80e46c31f23193b84fb9abe043efc2205671ea', 'Exact ROOT one-control source review')
    h = json.loads((O/'proposal_handoff.json').read_text(encoding='utf-8'))
    pins = [h[k] for k in ('helper_source','unexecuted_CPU_control_source','integration_and_lifetime_proposal',
                            'admitted_driver_source','configuration')]+h['dependencies']
    pins += [v for pair in h['proposed_exact_frozen_plan_pairs'] for v in pair]
    require(len(pins) == 19 and all(pin(v['path']) == v for v in pins), 'All19 independently reviewed bindings unchanged')
    return h


if '--child' in sys.argv:
    try:
        h = preflight()
        # Verify the actual original exception types before running the frozen
        # fixture's negative assertions; neither original source is changed.
        from bpe_broad_audio_v1 import require as audio_require
        from bpe_broad_production_state_v1 import require as state_require
        negatives = []
        for name, f in (('audio.require', audio_require), ('state.require', state_require)):
            try:
                f(False, 'owned negative exception-type preflight')
            except Exception as error:
                require(type(error) is ValueError, 'Frozen fixture expects actual original ValueError')
                negatives.append({'guard': name, 'exception_type': type(error).__name__})
            else:
                raise AssertionError('Original guard did not reject false assertion')
        source = h['unexecuted_CPU_control_source']['path']
        spec = importlib.util.spec_from_file_location('approved_frozen_full_plan_control', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        began = time.monotonic()
        report = module.run(h['configuration'], h['proposed_exact_frozen_plan_pairs'], OUT/'fixture')
        require(not any(t.name == 'private-broad-row-prefetch' for t in __import__('threading').enumerate()),
                'No private worker remains after genuine joined closure')
        save(OUT/'child_completion.json', {'passed': True, 'producer': identity(psutil.Process()),
            'ROOT_source_review': pin(REVIEW), 'report': report, 'negative_exception_preflight': negatives,
            'private_worker_thread_still_alive': False, 'full_original_context_finally_returned': True,
            'elapsed_seconds': time.monotonic()-began, 'actual_GPU_or_audio_work': False})
        print(json.dumps({'passed': True, 'child_completion': pin(OUT/'child_completion.json')}), flush=True)
    except BaseException as error:
        save(OUT/'child_failure.json', {'error': repr(error), 'traceback': traceback.format_exc(),
            'ROOT_source_review': pin(REVIEW), 'producer': identity(psutil.Process())})
        raise
else:
    h = preflight()
    require(not OUT.exists(), 'Fresh actualCPU01 output; no silent retry')
    OUT.mkdir()
    captured = {}
    peaks = {}
    samples = 0
    errors = []
    began = time.monotonic()
    from resource_watchdog_v2 import sample_memory
    proc = None
    try:
        with (OUT/'stdout.log').open('wb') as stdout, (OUT/'stderr.log').open('wb') as stderr:
            proc = subprocess.Popen([sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), '--child'],
                                    stdout=stdout, stderr=stderr, env=os.environ.copy(),
                                    creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            parent = psutil.Process(proc.pid)
            captured[(parent.pid, parent.create_time())] = identity(parent)
            save(OUT/'actual_start.json', {'ROOT_source_review': pin(REVIEW), 'wrapper_source': pin(__file__),
                'owned_Popen': identity(parent), 'wrapper': identity(psutil.Process()),
                'limits': {'elapsed_seconds': 180, 'each_owned_process_private_bytes': 536870912,
                           'commit_headroom_bytes': 2000000000, 'physical_headroom_bytes': 1500000000,
                           'additional_prefetch_admission_bytes': 268435456}})
            while proc.poll() is None:
                require(time.monotonic()-began < 180, 'Actual independent180s timeout')
                family = [parent]
                try:
                    family += parent.children(recursive=True)
                except psutil.NoSuchProcess:
                    pass
                for member in family:
                    try:
                        w = identity(member)
                        captured[(w['pid'], w['create_time'])] = w
                        s = sample_memory(member.pid, 'owned_private_scalar_prefetch_CPU_control')
                    except psutil.NoSuchProcess:
                        continue
                    samples += 1
                    peak = s['worker_private_memory']['private_bytes']
                    peaks[str(member.pid)] = max(peaks.get(str(member.pid), 0), peak)
                    require(peak <= 512*1024**2 and
                            s['system_memory']['commit_available_bytes'] >= 2000000000 and
                            s['system_memory']['physical_available_bytes'] >= 1500000000,
                            'Actual independent512MiB private and unchanged system headroom')
                time.sleep(.25)
            actual = proc.wait()
        require(actual == 0, 'Actual reviewed CPU child nonzero; preserve this failure')
        completion = json.loads((OUT/'child_completion.json').read_text(encoding='utf-8'))
        require(completion['passed'] and pin(completion['report']['path']) == completion['report'], 'Actual joined control report')
    except BaseException as error:
        errors.append({'error': repr(error), 'traceback': traceback.format_exc()})
        if proc is not None and proc.poll() is None:
            for w in reversed(list(captured.values())):
                if alive(w):
                    try:
                        psutil.Process(w['pid']).terminate()
                    except psutil.NoSuchProcess:
                        pass
            proc.wait(timeout=15)
        raise
    finally:
        deadline = time.monotonic()+10
        while any(alive(w) for w in captured.values()) and time.monotonic() < deadline:
            time.sleep(.1)
        result = {'kind': 'ASR18_actual_owned_private_scalar_prefetch_CPU_control_closure_v1',
            'ROOT_source_review': pin(REVIEW), 'wrapper_source': pin(__file__),
            'actual_Popen_wait_exit_code': None if proc is None else proc.poll(),
            'captured_identities': list(captured.values()),
            'all_captured_identities_dead': [{'identity': w, 'same_identity_alive': alive(w)} for w in captured.values()],
            'guard_samples': samples, 'maximum_sampled_private_bytes_by_PID': peaks,
            'guard_errors': errors, 'elapsed_seconds': time.monotonic()-began,
            'training_child_source_environment_DB_changed': False, 'production_admission_claimed': False}
        for name in ('child_completion.json','child_failure.json','fixture/report.json','stdout.log','stderr.log'):
            if (OUT/name).is_file():
                result[name] = pin(OUT/name)
        save(OUT/'actual_closure.json', result)
        print(json.dumps({'closure': pin(OUT/'actual_closure.json'), 'actual_exit': result['actual_Popen_wait_exit_code'],
                          'errors': len(errors)}), flush=True)
