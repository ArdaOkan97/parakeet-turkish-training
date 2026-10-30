"""Private CPU-only controls of frozen worker-pool byte/ownership protocols.

The one real leaf is CPU-only. A disclosed in-memory adapter maps ONLY the
archive call's GPU-kind expectation to the original CPU-proof verifier. No GPU
proof is produced; this cannot establish native/public archive admission.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
import asr18_worker_control_pool_v1 as module
import fastmss_owned_process_closure_v2 as process
from atomic_io_retry_v1 import write_json
from resource_watchdog_v2 import ResourceGuard, sample_memory

EXPECTED = '3230bd55b4478c2fc61f85e4c470d74aa449e63e6e93ad3199d2ad250c6d53a5'
ROOT = module.ROOT
NAMESPACE = ROOT / 'runs/ASR18_worker_pool_fixture_01'
OUTPUT = ROOT / 'analysis/FastMSS_integration_v1/worker_control_pool_CPU_actual_01'
POLICY = {'kind': 'PRIVATE_SYNTHETIC_CPU_FIXTURE_NO_PRODUCTION_AUTHORITY',
          'fixed_control_allocation_cap_bytes': 1073741824,
          'total_allocated_cap_bytes': 4294967296,
          'maximum_GPU_worker_directories': 48,
          'reserved_each_GPU_worker_allocation_bytes': 67108864}
IDENTITY = {'kind': 'PRIVATE_CPU_WORKER_CONTROL_FIXTURE_ONLY',
            'production_enabled': False, 'actual_native_GPU_authority': False}
watch = ResourceGuard(); peak = 0; family_peak = 0; admissions = []; adapters = []
known = {}


def require(value, message):
    if not value:
        raise ValueError(message)


def guard():
    global peak, family_peak
    value = sample_memory(os.getpid(), 'worker_pool_private_CPU_fixture')
    require(value['paired_sampling_within_one_second']
            and value['worker_private_memory']['available'] and value['system_memory']['available'], 'Fresh CPU observations')
    stop, _ = watch.observe(value['system_memory'], value['worker_private_memory'])
    require(stop is None, 'Unchanged resource guard: ' + str(stop))
    private = value['worker_private_memory']['private_bytes']; peak = max(peak, private)
    require(private < 256_000_000, 'Fixture private256MB inside unchanged2GB hard guard')
    import psutil
    total = private
    for child in psutil.Process().children(recursive=True):
        try:
            item = process.identity(child.pid); known[(item['pid'], item['create_time'])] = item
            observed = sample_memory(child.pid, 'worker_pool_actual_CPU_descendant')
            if not observed['worker_private_memory']['available']:
                require(not child.is_running(), 'Still-live CPU descendant has unavailable telemetry')
                continue
            total += observed['worker_private_memory']['private_bytes']
        except psutil.NoSuchProcess:
            pass
    family_peak = max(family_peak, total)
    require(total < 512_000_000, 'Entire owned CPU family512MB')
    require(shutil.disk_usage(ROOT).free > 100_000_000_000, 'Unchanged E100GB reserve')
    return value


@contextmanager
def metadata_phase(name):
    require(name == 'read_index', 'Only metadata/index phase')
    guard()
    try:
        yield
    finally:
        guard()


def allocated(size):
    return ((max(size, 1) + 524287) // 524288) * 524288


def control_admission(path, prospective_bytes):
    guard(); path = Path(path).resolve()
    require(path.is_relative_to(NAMESPACE) and type(prospective_bytes) is int, 'Prospective owned fixture storage')
    total = 0
    for base in (NAMESPACE, OUTPUT):
        if not base.exists():
            continue
        for root, dirs, names in os.walk(base):
            total += len(dirs) * 524288
            for name in names:
                p = Path(root) / name
                try:
                    total += allocated(p.stat().st_size)
                except FileNotFoundError:
                    require(name.startswith('owned_member_') or '.atomic-v1-' in name, 'Only own temporary disappearance')
    old = allocated(path.stat().st_size) if path.exists() else 0
    require(total + max(allocated(prospective_bytes) - old, 0) + 2 * 1024**2 < 256 * 1024**2,
            'Whole fixture cluster-rounded prospective256MiB')
    return total


def authority(request):
    require(request['identity'] == IDENTITY and request['owner']['path'].startswith(str(NAMESPACE)), 'Private fixture authority only')
    evidence = request['evidence']; module.bound(evidence)
    value = json.loads(Path(evidence['path']).read_text())
    require(value['kind'] == 'PRIVATE_SYNTHETIC_CPU_AUTHORITY_NOT_NATIVE'
            and value['production_enabled'] is False and value['native_GPU_claimed'] is False, 'No fabricated GPU/coherent checkpoint authority')
    guard()
    return True


def closed_authority(request):
    authority(request)
    require(request['normal_coherent_boundary_and_mapping_release_required'] is True
            and request['crash_tail_rollback_or_original_artifact_deletion_permitted'] is False, 'Actual helper requests strict production obligations')
    # Those scientific obligations are NOT exercised by this private fixture.
    original_verify(json.loads(Path(request['process0']['path']).read_text()),
                    kind='ASR18_owned_CPU_process_completion_v1', owner='/root/memory_diagnosis', guard=guard)
    return True


original_verify = process.verify


def CPU_fixture_adapter(value, *, kind, owner, report=None, guard=lambda: None):
    require(kind == 'ASR18_owned_GPU_process_completion_v1'
            and value['kind'] == 'ASR18_owned_CPU_process_completion_v1', 'Explicit CPU-only fixture expectation adapter')
    adapters.append({'requested_kind': kind, 'actual_proof_kind': value['kind'], 'GPU_claimed': False})
    return original_verify(value, kind=value['kind'], owner=owner, report=report, guard=guard)


def leaf(folder):
    folder = Path(folder).resolve()
    require(folder.is_relative_to(NAMESPACE), 'Fresh owned CPU leaf only')
    require(os.environ['CUDA_VISIBLE_DEVICES'] == '-1', 'No GPU')
    (folder / 'empty.log').write_bytes(b'')
    (folder / 'unicode.txt').write_bytes('exact UTF-8: üşğ — bytes preserved\n'.encode())
    write_json(folder / 'report.json', {'passed': True, 'producer': process.identity(),
        'worker_source': module.pin(__file__), 'parent_source': module.pin(__file__),
        'CPU_only': True, 'GPU_activity': False, 'synthetic_protocol_fixture': True})
    time.sleep(.7)


def main():
    began = time.monotonic(); checks = []
    require(module.pin(module.__file__)['sha256'] == EXPECTED, 'Frozen helper unchanged')
    require(os.environ['CUDA_VISIBLE_DEVICES'] == '-1' and not NAMESPACE.exists() and not OUTPUT.exists(), 'Fresh exact E fixture/no GPU')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        require(os.environ[name] == '1', 'Four thread caps1')
    for _ in range(3):
        sample = guard(); system = sample['system_memory']
        require(system['commit_used_bytes'] + 512_000_000 < .9 * system['commit_limit_bytes']
                and system['commit_available_bytes'] - 512_000_000 > 2_000_000_000, 'Fresh whole512MB admission')
        admissions.append(sample); time.sleep(.1)
    OUTPUT.mkdir(); NAMESPACE.mkdir()
    source = module.pin(__file__)
    def check(name, condition):
        require(condition, name); checks.append({'name': name, 'passed': True}); guard()
    def reject(name, operation):
        try:
            operation()
        except (ValueError, KeyError, TypeError, OSError, module.zlib.error) as error:
            checks.append({'name': name, 'passed': True, 'expected_rejection': type(error).__name__, 'message': str(error)[:300]}); guard(); return
        raise ValueError('Missing expected rejection: ' + name)
    evidence_path = NAMESPACE / 'private_authority.json'
    write_json(evidence_path, {'kind': 'PRIVATE_SYNTHETIC_CPU_AUTHORITY_NOT_NATIVE',
        'production_enabled': False, 'native_GPU_claimed': False})
    evidence = module.pin(evidence_path)
    control = NAMESPACE / 'control'; control.mkdir()
    external = OUTPUT / 'external_configuration'; external.mkdir()
    external_pin = external / 'config.json'; write_json(external_pin, {'private': True})
    def open_pool(create=False):
        return module.WorkerControlPool(control, identity=IDENTITY, policy=POLICY,
            authority_source=source, guard=guard, metadata_phase=metadata_phase,
            admit_control=control_admission, create=create)
    pool = open_pool(True)
    child = None
    try:
        pool.protect([], files=[module.pin(external_pin)], evidence=evidence, verify_protection=authority)
        check('Real external configuration is protection-only, not a registered archive member', not pool.owns_member(module.pin(external_pin)))
        with pool._temporary() as temp:
            temp.write(b'tracked'); temp.flush()
            check('Exact active owned_member temporary coexists with strict registry scan', pool._load()['archive_count'] == 0)
        db = pool._db()
        try:
            db.execute('BEGIN IMMEDIATE'); db.execute('CREATE TABLE fixture_rollback_only (i INTEGER)')
            check('Own active SQLite rollback journal coexists with strict scan', pool._load()['archive_count'] == 0)
            db.rollback()
        finally:
            db.close(); pool._sqlite_writers -= 1
        orphan = pool.folder / 'owned_member_ORPHAN.tmp'; orphan.write_bytes(b'unknown')
        reject('Untracked similarly-named orphan is rejected', pool._load)
        check('Orphan preserved at rejection', orphan.read_bytes() == b'unknown'); orphan.unlink()
        config = control / 'native_configuration_000001.json'; write_json(config, {'private_CPU_leaf': True})
        owner = NAMESPACE / 'owner_01'; owner.mkdir(); folder = owner / 'native_000001'
        pool.reserve(folder, configuration=module.pin(config), worker_source=source, parent_source=source)
        folder.mkdir(); guard_source = PROJECT / 'work/research/ASR18_bounded_CPU_phase_v3.py'
        python = ROOT / 'scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe'
        log = (folder / 'worker.log').open('xb')
        child = subprocess.Popen([str(python), '-X', 'utf8', '-B', str(guard_source), '--folder', str(folder / 'guard'),
            '--seconds', '30', '--script', str(Path(__file__).resolve()), '--', '--leaf', str(folder)],
            env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        launcher = process.identity(child.pid)
        known[(launcher['pid'], launcher['create_time'])] = launcher
        while child.poll() is None:
            guard(); require(time.monotonic() - began < 140, 'Finite fixture'); time.sleep(.1)
        require(child.wait() == 0, 'Actual bounded CPU leaf0'); log.close()
        actual = process.observe(child, kind='ASR18_owned_CPU_process_completion_v1', owner='/root/memory_diagnosis',
            report=module.pin(folder / 'report.json'), worker_source=source, parent_source=source,
            guard_source=module.pin(guard_source), guard_closure=module.pin(folder / 'guard/closure.json'),
            launcher_identity=launcher, parent_guard_observation=guard(), output=folder / 'actual_CPU_process0.json', guard=guard)
        empty = module.pin(folder / 'empty.log'); text = module.pin(folder / 'unicode.txt')
        process_copy = OUTPUT / 'genuine_CPU_leaf_actual_process0.json'; shutil.copyfile(actual['path'], process_copy)
        original_report = OUTPUT / 'genuine_CPU_leaf_report.json'; shutil.copyfile(folder / 'report.json', original_report)
        original_guard = OUTPUT / 'genuine_CPU_leaf_guard_closure.json'; shutil.copyfile(folder / 'guard/closure.json', original_guard)
        pool.protect([folder], files=[module.pin(external_pin)], evidence=evidence, verify_protection=authority)
        with patch.object(process, 'verify', CPU_fixture_adapter):
            reject('Protected CURRENT worker remains physical', lambda: pool.archive(folder, process0=actual,
                paired_files=[module.pin(config)], evidence=evidence, verify_closed=closed_authority))
        check('Protected rejection leaves complete process report and empty log unchanged', module.pin(folder / 'empty.log') == empty)
        pool.protect([], files=[module.pin(external_pin)], evidence=evidence, verify_protection=authority)
        unknown = folder / 'unknown.bin'; unknown.write_bytes(b'preserve unknown')
        with patch.object(process, 'verify', CPU_fixture_adapter):
            reject('Unknown worker member rejects before unlink', lambda: pool.archive(folder, process0=actual,
                paired_files=[module.pin(config)], evidence=evidence, verify_closed=closed_authority))
        check('Unknown member preserved at rejection', unknown.read_bytes() == b'preserve unknown'); unknown.unlink()
        original_lstat = Path.lstat
        def reparse_metadata(path, *args, **kwargs):
            info = original_lstat(path, *args, **kwargs)
            if str(path) == str(folder / 'unicode.txt'):
                return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=getattr(info, 'st_file_attributes', 0) | 0x400)
            return info
        with patch.object(Path, 'lstat', reparse_metadata):
            reject('Explicit synthetic reparse attribute rejected (exFAT has no actual link creation)', lambda: module.safe(folder / 'unicode.txt', folder))
        with patch.object(process, 'verify', CPU_fixture_adapter):
            result = pool.archive(folder, process0=actual, paired_files=[module.pin(config)],
                                  evidence=evidence, verify_closed=closed_authority)
        check('Actual full-byte archive and exact owned unlink returned', result['passed'] and not folder.exists() and not config.exists())
        check('Paired config and empty original are exact indexed members', pool.owns_member(empty) and pool.owns_member(text))
        with pool.resolve_member(empty) as stream:
            check('Empty original member resolves with truthful zero bytes/SHA', stream.read(1) == b'')
        with pool.resolve_member(text) as stream:
            check('UTF-8 original raw bytes and EOF exact', stream.read(1024) == 'exact UTF-8: üşğ — bytes preserved\n'.encode() and stream.read(1) == b'')
        before = pool.snapshot_binding(); pool.close()
        pool = open_pool(False); after = pool.snapshot_binding()
        check('Actual close/reopen preserves owner/index/archive/registry and counters', before == after)
        check('Prior owner roots remain registered after obsolete worker removal', owner in pool.owner_folders() and not pool.live_folders())
        reject('Fresh supervisor cannot reserve before fresh protection authority', lambda: pool.reserve(owner / 'native_000002',
            configuration=module.pin(external_pin), worker_source=source, parent_source=source))
        pool.protect([], files=[module.pin(external_pin)], evidence=evidence, verify_protection=authority)
        for i in range(2, 50):
            path = control / f'native_configuration_{i:06}.json'; write_json(path, {'private_reservation_only': i})
            pool.reserve(owner / f'native_{i:06}', configuration=module.pin(path), worker_source=source, parent_source=source)
        check('Cumulative reservation counter preserved after reopen, not reset', pool.snapshot_binding()['reservation_count'] == 49 and len(pool.live_folders()) == 48)
        path = control / 'native_configuration_000050.json'; write_json(path, {'private': 50})
        reject('49th LIVE reservation fails without increasing cumulative count', lambda: pool.reserve(owner / 'native_000050',
            configuration=module.pin(path), worker_source=source, parent_source=source))
        check('Live ceiling and cumulative counter remain unchanged on failure', pool.snapshot_binding()['reservation_count'] == 49)
        with pool.resolve_member(text) as stream:
            check('Exact archive lookup still works after reopen and later reservations', stream.read(1024).startswith(b'exact UTF-8:'))
        with pool.archive_path.open('r+b') as stream:
            stream.seek(result['descriptor']['offset'] + module.HEADER.size
                        + max(20, result['descriptor']['compressed_bytes'] // 2))
            old = stream.read(1); stream.seek(-1, 1); stream.write(bytes([old[0] ^ 1])); stream.flush(); os.fsync(stream.fileno())
        def corrupt_read():
            with pool.resolve_member(text) as stream:
                stream.read(1)
        reject('Actual corrupted fixture frame rejected on exact lookup', corrupt_read)
        with pool.archive_path.open('ab') as stream:
            stream.write(b'PRIVATE_UNKNOWN_TAIL'); stream.flush(); os.fsync(stream.fileno())
        reject('Uncommitted archive suffix rejects and remains preserved', pool._load)
        pool.close(); pool = None
        reject('Fresh constructor cannot repair/adopt unknown suffix', lambda: open_pool(False))
        check('Frozen helper and original validator source unchanged', module.pin(module.__file__)['sha256'] == EXPECTED
              and process.verify is original_verify)
        report = {'kind': 'ASR18_worker_control_pool_private_CPU_controls_v1', 'passed': True,
            'producer': process.identity(), 'worker_source': source, 'parent_source': source,
            'helper': module.pin(module.__file__), 'checks': checks, 'check_count': len(checks),
            'actual_CPU_leaf0': module.pin(process_copy), 'actual_CPU_leaf_guard': module.pin(original_guard),
            'actual_CPU_leaf_report_copy': module.pin(original_report),
            'CPU_leaf_original_proof_members_archived_in_fixture': True,
            'CPU_fixture_proof_adapter': adapters, 'GPU_process_proof_produced': False,
            'public_native_or_production_admission_claimed': False, 'synthetic_authority': evidence,
            'namespace': str(NAMESPACE), 'forecast_bytes': 512_000_000, 'private_peak_bytes': peak,
            'owned_family_peak_bytes': family_peak, 'fresh_three_admissions': admissions,
            'wall_seconds': time.monotonic() - began, 'expected_corrupt_and_tail_fixture_artifacts_preserved': True}
        write_json(OUTPUT / 'report.json', report)
    finally:
        if pool is not None:
            pool.close()
        if child is not None and (child.poll() is None or any(not process.dead(x) for x in known.values())):
            from asr18_owned_family_observation_v4 import terminate_known_family
            terminate_known_family(child, locals().get('launcher'), known)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--leaf')
    args = parser.parse_args()
    if args.leaf:
        leaf(args.leaf)
    else:
        main()
