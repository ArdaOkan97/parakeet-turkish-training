"""Private synthetic inventory comparison; no arrays/frontend/native admission.

The real frozen sink6 and inactive sink7 inventory methods see exactly the same
growing first4/20 trees and scan cadence. Tiny named byte files stand in for NPY
files ONLY for directory accounting, not a feature/owner/public protocol proof.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path(r'E:\gpu_training_scratchpad')
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
import fastmss_phase_aware_feature_window_v6 as old
import fastmss_phase_aware_feature_window_v7 as new
from atomic_io_retry_v1 import write_json
from fastmss_owned_process_closure_v2 import identity
from resource_watchdog_v2 import ResourceGuard, sample_memory

watch = ResourceGuard()
checks = []
private_peak = 0
began = time.monotonic()


def require(value, message):
    if not value:
        raise ValueError(message)


def pin(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT) or path.is_relative_to(PROJECT), 'E/project only')
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha}


def guard():
    global private_peak
    value = sample_memory(os.getpid(), 'private_sink7_inventory_controls')
    w, s = value['worker_private_memory'], value['system_memory']
    require(value['paired_sampling_within_one_second'] and w['available'] and s['available'], 'Fresh paired telemetry')
    stop, _ = watch.observe(s, w)
    private_peak = max(private_peak, w['private_bytes'])
    require(stop is None and private_peak <= 128_000_000 and time.monotonic() - began < 180,
            'Unchanged system/CPU2GB guards plus proposed128MB/180s fixture')
    require(shutil.disk_usage(ROOT).free >= 100_000_000_000, 'E100GB reserve')
    return value


def check(name, value, **detail):
    require(value, name)
    checks.append({'name': name, 'passed': True, **detail})
    guard()


def rejects(name, operation):
    try:
        operation()
    except (ValueError, OSError) as error:
        check(name, True, actual_exception=type(error).__name__)
    else:
        raise ValueError('Expected rejection: ' + name)


def names_for(updates):
    names = ['owner.json', 'manifest.json']
    for i in range(updates):
        names += [f'active_{i:02}.json', f'update_{i:02}.json']
        for j in range(16):
            names += [f'u{i:02}_r{j:02}_features.npy', f'u{i:02}_r{j:02}_mask.npy']
    return names


def private_stores(folder, names):
    folder.mkdir()
    owner = {'owned_names': names}
    return (old._Store(folder, owner, guard, nullcontext),
            new._Store(folder, owner, guard, nullcontext))


def benchmark(folder, updates):
    previous, candidate = private_stores(folder, names_for(updates))
    count = 0
    expected = 0
    seconds = {'sink6': 0., 'sink7': 0.}
    entry_visits = 0
    files = 0

    def scan():
        nonlocal count, entry_visits
        guard()
        arms = [('sink6', previous), ('sink7', candidate)]
        if count % 2:
            arms.reverse()  # Each arm goes first equally often: no warm-cache privilege.
        for key, store in arms:
            started = time.perf_counter()
            actual = store.inventory()
            seconds[key] += time.perf_counter() - started
            require(actual == expected, 'Exact whole-directory byte totals agree')
        count += 1
        entry_visits += files

    def write(name):
        nonlocal expected, files
        data = b'synthetic_inventory_bytes_only\n'
        with (folder / name).open('xb') as stream:
            stream.write(data)
        expected += len(data)
        files += 1

    write('owner.json')
    for i in range(updates):
        scan()  # write_metadata(active): same preallocation scan.
        write(f'active_{i:02}.json')
        for j in range(16):
            for kind in ('features', 'mask'):
                scan()
                write(f'u{i:02}_r{j:02}_{kind}.npy')
                scan()
        scan()  # write_metadata(update): same preallocation scan.
        write(f'update_{i:02}.json')
    scan()  # Same window finalization inventory; no fabricated manifest.
    expected_visits = sum(35 + 68 * i + sum(12 + 136 * i + 8 * j for j in range(16))
                          for i in range(updates)) + 1 + 34 * updates
    result = {'updates': updates, 'scans_per_arm': count, 'entry_visits_per_arm': entry_visits,
              'files': files, 'logical_bytes': expected, 'inventory_seconds': seconds,
              'candidate_over_baseline': seconds['sink7'] / seconds['sink6'],
              'payload_content': 'tiny synthetic bytes, not NPY/feature validation',
              'timing_scope': 'inventory only; alternating arm order; real external watchdog/guard between calls'}
    # Preserve completed timing evidence even if a later counter/negative fails.
    write_json(folder.parent / f'first{updates}_inventory_timing.json', result)
    check(f'first{updates}: original scan count/entry visits/bytes preserved',
          count == 66 * updates + 1 and entry_visits == expected_visits
          and files == 1 + 34 * updates and len(candidate._names_set) == 2 + 34 * updates,
          scans_per_arm=count, entry_visits_per_arm=entry_visits, logical_bytes=expected)
    return result


def negatives(root):
    previous, candidate = private_stores(root / 'negative', ['a', 'b'])
    (candidate.folder / 'a').write_bytes(b'abc')
    check('Regular bytes match frozen sink6', previous.inventory() == candidate.inventory() == 3)
    with patch.object(new, 'CAPACITY', 2):
        rejects('Exact whole-directory quota exceeded', candidate.inventory)
    (candidate.folder / 'foreign').write_bytes(b'x')
    rejects('Unknown name rejected', candidate.inventory)
    (candidate.folder / 'foreign').unlink()  # Exact fixture-owned negative cleanup only.
    (candidate.folder / 'b').mkdir()
    rejects('Owned-name directory rejected', candidate.inventory)
    (candidate.folder / 'b').rmdir()
    candidate.owner['owned_names'].append('foreign')
    rejects('In-place ownership-list mutation rejected', candidate.inventory)
    candidate.owner['owned_names'].pop()
    candidate.owner['owned_names'] = ['a', 'b']
    rejects('Replaced ownership-list object rejected', candidate.inventory)
    candidate.owner['owned_names'] = candidate._names_object
    check('Unmodified original list can be read after explicit negative fixture', candidate.inventory() == 3)
    class ReparseEntry:
        name = 'a'
        def stat(self, *, follow_symlinks):
            require(follow_symlinks is False, 'No-follow actual member API')
            return SimpleNamespace(st_mode=stat.S_IFREG, st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT,
                                   st_size=3)
    class FakeScan:
        def __enter__(self): return iter([ReparseEntry()])
        def __exit__(self, *args): pass
    with patch.object(new.os, 'scandir', lambda path: FakeScan()):
        rejects('Explicit simulated Windows file reparse flag rejected', candidate.inventory)
    root_before = candidate.folder
    moved = root / 'negative_original_preserved'
    root_before.rename(moved)
    root_before.mkdir()
    rejects('Actual root directory replacement rejected', candidate.inventory)
    check('Original root content preserved after replacement rejection', (moved / 'a').read_bytes() == b'abc')
    other_previous, other = private_stores(root / 'changed_during_scan', ['a'])
    (other.folder / 'a').write_bytes(b'a')
    counter = [0]
    original_stat = other._directory_stat
    def changed_after():
        counter[0] += 1
        result = original_stat()
        return result if counter[0] == 1 else (result[0], result[1] + 1)
    with patch.object(other, '_directory_stat', changed_after):
        rejects('Explicit simulated root replacement after enumeration rejected', other.inventory)
    return {'simulated_cases': ['file reparse flag', 'mid-scan root identity change'],
            'actual_cases': ['root replacement', 'unknown file', 'subdirectory', 'name mutation', 'quota'],
            'reparse_scope': 'Explicit typed no-follow Windows reparse metadata adapter; no actual reparse creation on designated exFAT, no other storage used',
            'negative_artifacts_preserved': True}


def main(output):
    require(os.name == 'nt', 'Exact Windows directory metadata semantics')
    output = Path(output).resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), 'Fresh E fixture namespace only')
    require(json.loads((ROOT / 'storage_config.json').read_text())['allow_fallback_outside_root'] is False,
            'No E fallback')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        require(os.environ[key] == '1', 'Four thread caps1')
    require(os.environ['CUDA_VISIBLE_DEVICES'] == '-1', 'CUDA masked')
    admissions = []
    for _ in range(3):
        v = guard(); s = v['system_memory']
        require(s['commit_used_bytes'] + 192_000_000 < .90 * s['commit_limit_bytes']
                and s['commit_available_bytes'] - 192_000_000 >= 2_000_000_000, 'Full192MB fresh fixture admission')
        admissions.append(v)
        time.sleep(.1)
    output.mkdir()
    sources = {'baseline': pin(old.__file__), 'candidate': pin(new.__file__), 'fixture': pin(__file__)}
    timings = [benchmark(output / f'first{n}', n) for n in (4, 20)]
    negative_scope = negatives(output)
    check('No NumPy/Torch/frontend/model imported', not any(k == 'numpy' or k.startswith('numpy.')
          or k == 'torch' or k.startswith('torch.') or k == 'transformers' or k.startswith('transformers.') for k in sys.modules))
    check('Frozen and candidate source bytes unchanged', pin(old.__file__) == sources['baseline'] and pin(new.__file__) == sources['candidate'])
    cluster = 512 * 1024
    logical = 0
    file_count = 0
    directory_count = 1
    # Do not follow the deliberately preserved junction.
    for parent, dirs, files in os.walk(output, followlinks=False):
        dirs[:] = [d for d in dirs if not os.lstat(Path(parent) / d).st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT]
        directory_count += len(dirs)
        for name in files:
            logical += (Path(parent) / name).stat().st_size
            file_count += 1
    rounded_bound = (file_count + directory_count + 8) * cluster
    require(logical < 16 * 1024**2 and rounded_bound < 512 * 1024**2, 'Bounded logical16MiB/cluster-rounded512MiB fixture')
    report = {'kind': 'ASR18_sink7_private_synthetic_inventory_controls_v1', 'passed': True,
        'producer': identity(), 'worker_source': sources['fixture'],
        'parent_source': pin(PROJECT / 'work/research/ASR18_bounded_CPU_phase_v3.py'),
        'sources': sources, 'checks': checks, 'check_count': len(checks), 'benchmarks': timings,
        'negative_scope': negative_scope, 'logical_fixture_bytes': logical,
        'cluster_rounded_fixture_bound_bytes': rounded_bound, 'file_count': file_count,
        'directory_count': directory_count, 'private_peak_bytes': private_peak,
        'fresh_three_admissions': admissions, 'forecast_family_bytes': 192_000_000,
        'wall_seconds': time.monotonic() - began, 'public_owner_constructor_exercised': False,
        'real_feature_or_audio_bytes_used': False, 'actual_frontend_speedup_claimed': False,
        'production_enabled': False}
    write_json(output / 'report.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
