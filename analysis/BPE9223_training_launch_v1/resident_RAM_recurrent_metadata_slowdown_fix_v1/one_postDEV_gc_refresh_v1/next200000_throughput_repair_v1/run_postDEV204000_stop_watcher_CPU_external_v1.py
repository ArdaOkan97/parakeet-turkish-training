"""External bounded guard for ONE isolated watcher CPU proof, never a live arm."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import psutil

P = Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
PROTECTED = {19320, 39856, 21928, 42068, 40536, 26892, 30128}
CAP = 512 * 1024**2


def require(value, message):
    if not value:
        raise ValueError(message)


def identity(pid=None):
    proc = psutil.Process(os.getpid() if pid is None else pid)
    return {'pid': proc.pid, 'create_time': proc.create_time(), 'ppid': proc.ppid()}


def alive(who):
    try:
        proc = psutil.Process(who['pid'])
        return abs(proc.create_time() - who['create_time']) < .001 and proc.ppid() == who['ppid'] and proc.is_running()
    except psutil.NoSuchProcess:
        return False


def pin(path):
    path = Path(path)
    require(path.stat().st_size <= 32 * 1024**2, 'Bounded source/metadata pin')
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def write(path, value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':')).encode() + b'\n'
    require(len(raw) <= 32 * 1024**2, 'Bounded external output')
    with Path(path).open('xb') as file:
        file.write(raw)
        file.flush()
        os.fsync(file.fileno())
    return pin(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ROOT-review', required=True)
    args = parser.parse_args()
    began = time.monotonic()
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    scope = json.loads(Path(args.ROOT_review).read_text(encoding='utf-8-sig'))
    require(scope['kind'] == 'ROOT_postDEV204000_stop_watcher_finite_CPU_exact_ONCE_execution_scope_v1'
            and scope['accepted'] is True and scope['reviewer'] == '/root'
            and scope['external_source'] == pin(__file__), 'Separate exact finite CPU ROOT scope')
    output = Path(scope['external_output'])
    fixture_output = Path(scope['fixture_output'])
    require(output.drive.lower() == fixture_output.drive.lower() == 'e:'
            and not output.exists() and not fixture_output.exists(), 'Fresh E-only actual namespace')
    output.mkdir()
    me, parent = identity(), identity(os.getppid())
    report = {'kind': 'ASR18_postDEV204000_watcher_finite_CPU_external_v1',
              'external': me, 'external_interpreter_launcher_parent': parent,
              'ROOT_review': pin(args.ROOT_review), 'source': pin(__file__),
              'Popen_returncode': None, 'guard_error': None, 'cleanup': [],
              'descendant_identities': [], 'family_private_peak_bytes': 0,
              'sampled_family_cap_not_hard_allocation_cap': True,
              'parent_sampled_but_never_waited_or_terminated': True}
    process = None
    popen_identity = None
    stdout = stderr = None
    sys.path.insert(0, str(P / 'outputs/parakeet_turkish'))
    from resource_watchdog_v2 import sample_memory

    def guard(label, prospective=False):
        value = sample_memory(os.getpid(), label)
        family = {me['pid']: me, parent['pid']: parent}
        require(alive(me) and alive(parent), 'Exact external interpreter and launcher parent')
        for proc in psutil.Process(me['pid']).children(recursive=True):
            who = identity(proc.pid)
            family[who['pid']] = who
            if who not in report['descendant_identities']:
                report['descendant_identities'].append(who)
        require(len(report['descendant_identities']) <= 32, 'Finite owned identity count')
        private = 0
        for who in family.values():
            if alive(who):
                private += psutil.Process(who['pid']).memory_info().private
        report['family_private_peak_bytes'] = max(report['family_private_peak_bytes'], private)
        if prospective:
            write(output / 'actual_initial_full_numeric_sample_before_predicate.json',
                  {'sample': value, 'family': list(family.values()), 'private_bytes': private,
                   'required_commit_bytes': 5055306368, 'required_physical_bytes': 2305306368})
        mem = value['system_memory']
        require(value['paired_sampling_within_one_second'] is True and private < CAP
                and mem['commit_available_bytes'] >= (5055306368 if prospective else 2268435456)
                and mem['physical_available_bytes'] >= (2305306368 if prospective else 1768435456)
                and shutil.disk_usage('E:/').free >= 100000000000
                and time.monotonic() - began < 120, 'Original full prospective/runtime/family/E/wall guards')
        return {'sample': value, 'private_bytes': private, 'family': list(family.values())}

    try:
        report['initial_sample'] = guard('watcher_finite_initial', True)
        before = [pin(item['path']) for item in scope['input_pins']]
        require(before == scope['input_pins'], 'All immutable finite CPU sources/inputs')
        report['input_pins_before'] = before
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
                and all(os.environ.get(key) == '1' for key in
                        ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')),
                'CPU-hidden original one-thread caps')
        require(os.environ.get('TORCH_EXTENSIONS_DIR', '').lower().startswith('e:'), 'Process-local E storage environment')
        for who in scope['protected_live_identities']:
            require(alive(who), 'Frozen current training/reader identities, no automatic rebind')
        stdout = (output / 'stdout.log').open('xb')
        stderr = (output / 'stderr.log').open('xb')
        command = [sys.executable, '-X', 'utf8', '-B', scope['fixture_source']['path'],
                   '--ROOT-review', args.ROOT_review, '--output', str(fixture_output)]
        report['command'] = command
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
        popen_identity = identity(process.pid)
        report['Popen_launcher_identity'] = popen_identity
        require(process.pid not in PROTECTED and popen_identity['ppid'] == me['pid'], 'Only own exact Popen child')
        while process.poll() is None:
            report['last_sample'] = guard('watcher_finite_live')
            require((output / 'stdout.log').stat().st_size <= 4 * 1024**2
                    and (output / 'stderr.log').stat().st_size <= 4 * 1024**2, 'Bounded finite logs')
            time.sleep(.1)
        report['Popen_returncode'] = process.wait()
        report['last_sample'] = guard('watcher_finite_after_exit')
        report['input_pins_after'] = [pin(item['path']) for item in scope['input_pins']]
        require(report['input_pins_after'] == before, 'All immutable inputs unchanged')
        require(report['Popen_returncode'] == 0, 'Actual finite fixture Popen0 required')
        report['fixture_report'] = pin(fixture_output / 'report.json')
        require(json.loads((fixture_output / 'report.json').read_text(encoding='utf-8'))['passed'] is True,
                'Actual finite proof passed, not merely process0')
    except BaseException as exc:
        report['guard_error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        raise
    finally:
        if process is not None and process.poll() is None and popen_identity is not None and alive(popen_identity):
            root = psutil.Process(popen_identity['pid'])
            require(root.ppid() == me['pid'] and root.pid not in PROTECTED, 'Exact owned root before cleanup')
            owned = [identity(proc.pid) for proc in root.children(recursive=True)]
            for who in reversed(owned):
                if alive(who) and who['pid'] not in PROTECTED:
                    require(popen_identity['pid'] in [proc.pid for proc in psutil.Process(who['pid']).parents()],
                            'Rechecked live owned ancestry before descendant termination')
                    psutil.Process(who['pid']).kill()
                    report['cleanup'].append({'identity': who, 'action': 'owned_descendant_kill'})
            if alive(popen_identity):
                process.kill()
                report['cleanup'].append({'identity': popen_identity, 'action': 'owned_Popen_kill'})
            try:
                report['Popen_returncode'] = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                report['Popen_returncode'] = None
        for file in (stdout, stderr):
            if file is not None:
                file.close()
        report['wall_seconds'] = time.monotonic() - began
        report['final_external_identity'] = identity()
        result = write(output / 'external_result.json', report)
        print(json.dumps({'external_result': result, 'Popen_returncode': report['Popen_returncode'],
                          'guard_error': report['guard_error']}), flush=True)


if __name__ == '__main__':
    main()
