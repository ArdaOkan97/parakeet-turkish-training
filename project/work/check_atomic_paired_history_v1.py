"""CPU-only disposable correctness/ABBA publication benchmark; no watched writes."""
import argparse
import ast
import ctypes
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'outputs/parakeet_turkish'
sys.path.insert(0, str(HERE))
import atomic_io_retry_v1 as atomic
import atomic_paired_history_v1 as pair
import resource_watchdog_v2 as watchdog


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def record(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': sha(path), 'bytes': path.stat().st_size}


def controls():
    checks = 0
    with tempfile.TemporaryDirectory(prefix='paired_correctness_', dir=ROOT / 'work') as folder:
        folder = Path(folder)
        a, b, old = folder/'run.json', folder/'shared.json', folder/'legacy.json'
        value = [{'step': 1, 'text': 'Türkçe ı ş ç é', 'loss': 1.25,
                  'zero': -0.0, 'optional': None, 'nested': {'a': [True, False]}},
                 {'step': 2, 'validation': {'validation_tr': {'wer': 352/825}},
                  'unknown_flag': {'retained': True}}]
        atomic.write_json(old, value)
        publisher = pair.PairedHistoryPublisher(a, b, chunk_bytes=31)
        result = publisher.publish(value)
        assert a.read_bytes() == b.read_bytes() == old.read_bytes(); checks += 1
        assert json.loads(a.read_text(encoding='utf-8')) == value; checks += 1
        assert result['status'] == 'complete' and result['run_committed'] and result['shared_committed']; checks += 1
        assert result['serialized_bytes'] == result['copied_bytes'] == old.stat().st_size; checks += 1
        assert all(math.isfinite(result[key]) and result[key] >= 0 for key in pair.TIMERS); checks += 1
        assert publisher.counters_snapshot()['complete_pairs'] == 1; checks += 1
        original = atomic.retry_replace
        for failure_at in (1, 2):
            a.write_bytes(b'old run'); b.write_bytes(b'old shared')
            calls = []
            def fail(source, target, **options):
                calls.append(str(Path(target).resolve()))
                if len(calls) == failure_at:
                    raise OSError('injected commit failure')
                return original(source, target, **options)
            atomic.retry_replace = fail
            publisher = pair.PairedHistoryPublisher(a, b)
            try:
                publisher.publish(value)
                raise AssertionError('failure was not propagated')
            except OSError:
                pass
            finally:
                atomic.retry_replace = original
            assert publisher.last_report['status'] == 'failed'; checks += 1
            assert calls[0] == str(a.resolve()); checks += 1
            if failure_at == 1:
                assert a.read_bytes() == b'old run' and b.read_bytes() == b'old shared'; checks += 1
                assert len(publisher.last_report['pending_temporaries']) == 2; checks += 1
            else:
                assert a.read_bytes() == old.read_bytes() and b.read_bytes() == b'old shared'; checks += 1
                assert calls[1] == str(b.resolve()) and len(publisher.last_report['pending_temporaries']) == 1; checks += 1
            assert publisher.counters_snapshot()['complete_pairs'] == 0; checks += 1
        a.write_bytes(b'protected'); b.write_bytes(b'protected shared')
        try:
            pair.PairedHistoryPublisher(a, b).publish([{'not_finite': float('nan')}])
            raise AssertionError('NaN accepted')
        except ValueError:
            pass
        assert a.read_bytes() == b'protected' and b.read_bytes() == b'protected shared'; checks += 1
        for options in ({'chunk_bytes': 1048577}, {'retry_seconds': 61}):
            try:
                pair.PairedHistoryPublisher(a, b, **options)
                raise AssertionError('unbounded option accepted')
            except ValueError:
                checks += 1
    return {'passed': True, 'assertions': checks,
            'scope': 'Actual stdlib filesystem/format/failure controls in disposable work paths; injected commit failures, no real sharing failure.'}


def samples_from_closed_history(path):
    wanted = {22630, 22750, 23630, 25109}
    rows = {}; seen = 0; block = None
    before = path.stat()
    # Atomic pretty-print format: object entries start/end at indentation2.
    # Retain four complete row objects, never the original25109-row list.
    with path.open('r', encoding='utf-8') as stream:
        for line in stream:
            if block is None:
                if line.rstrip() == '  {':
                    block = [line]
            else:
                block.append(line)
                if line.rstrip() in ('  },', '  }'):
                    block[-1] = block[-1].rstrip().rstrip(',') + '\n'
                    row = json.loads(''.join(block)); block = None; seen += 1
                    assert row['step'] == seen
                    if seen in wanted:
                        rows[seen] = row
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino)
    assert seen == 25109 and rows.keys() == wanted
    return rows, {'path': str(path.resolve()), 'bytes': before.st_size,
                  'mtime_ns': before.st_mtime_ns, 'verified_contiguous_records': seen,
                  'retained_representative_records': sorted(rows), 'source_not_modified': True}


def legacy_pair(a, b, value):
    output = {'mode': 'legacy_serialize_twice', 'serializations': 2,
              'copies': 0, 'commits': 2, 'stream_copy_seconds': 0.0,
              'serialize_and_buffered_write_seconds': 0.0, 'flush_seconds': 0.0,
              'fsync_seconds': 0.0, 'replace_seconds': 0.0}
    started = time.perf_counter()
    # Same code path/format as frozen atomic.write_json, split only for timing.
    for destination in (a, b):
        private = atomic.temporary_path(destination)
        with private.open('w', encoding='utf-8', newline='\n') as stream:
            before = time.perf_counter()
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            output['serialize_and_buffered_write_seconds'] += time.perf_counter()-before
            before = time.perf_counter(); stream.flush()
            output['flush_seconds'] += time.perf_counter()-before
            before = time.perf_counter(); os.fsync(stream.fileno())
            output['fsync_seconds'] += time.perf_counter()-before
        before = time.perf_counter(); atomic.retry_replace(private, destination)
        output['replace_seconds'] += time.perf_counter()-before
    output['pair_total_seconds'] = time.perf_counter()-started
    return output


def optimized_pair(publisher, value):
    value = publisher.publish(value)
    return {'mode': 'optimized_serialize_once', 'serializations': 1, 'copies': 1, 'commits': 2,
            'serialize_and_buffered_write_seconds': value['serialize_and_buffered_write_seconds'],
            'stream_copy_seconds': value['stream_copy_seconds'],
            'flush_seconds': value['first_flush_seconds']+value['second_flush_seconds'],
            'fsync_seconds': value['first_fsync_seconds']+value['second_fsync_seconds'],
            'replace_seconds': value['run_replace_seconds']+value['shared_replace_seconds'],
            'pair_total_seconds': value['pair_total_seconds'], 'replace_retries': value['replace_retries']}


def verify_output(path, expected):
    count = 0; block = None
    with path.open('r', encoding='utf-8') as stream:
        for line in stream:
            if block is None:
                if line.rstrip() == '  {':
                    block = [line]
            else:
                block.append(line)
                if line.rstrip() in ('  },', '  }'):
                    block[-1] = block[-1].rstrip().rstrip(',') + '\n'
                    assert json.loads(''.join(block)) == expected[count]
                    block = None; count += 1
    assert count == len(expected)
    return count


def profile(history_path):
    memory = []
    def observe(phase):
        sample = watchdog.sample_memory(os.getpid(), phase)
        system = sample['system_memory']
        if not system.get('available') or system['commit_fraction'] >= .90 or system['commit_available_bytes'] < 2_000_000_000:
            raise RuntimeError('CPU benchmark requires measured global commit below90% and2GB headroom')
        memory.append(sample)
    if os.name == 'nt':
        api = ctypes.WinDLL('kernel32', use_last_error=True)
        api.GetCurrentProcess.restype = ctypes.c_void_p
        api.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        assert api.SetPriorityClass(api.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL_PRIORITY_CLASS
    observe('CPU_ABBA_before_shape_sampling')
    rows, source = samples_from_closed_history(history_path)
    # Synthetic workload, authentic representative nested row shapes/UTF8 and
    # production record count. Nested objects are shared, not copied to25109.
    # Never published under OUTPUT/runs or interpreted as training evidence.
    value = [dict(rows[22750 if i % 250 == 0 else 22630], step=i) for i in range(1, 25110)]
    observe('CPU_ABBA_after_bounded_shape_carrier')
    results = []; expected_sha = None
    with tempfile.TemporaryDirectory(prefix='paired_ABBA_', dir=ROOT / 'work') as folder:
        folder = Path(folder); a, b = folder/'run.json', folder/'shared.json'
        publisher = pair.PairedHistoryPublisher(a, b)
        for order, mode in enumerate(('A', 'B', 'B', 'A'), 1):
            observe('CPU_ABBA_before_' + mode + str(order))
            result = legacy_pair(a, b, value) if mode == 'A' else optimized_pair(publisher, value)
            observe('CPU_ABBA_after_publication_' + mode + str(order))
            digest = sha(a); assert sha(b) == digest
            if expected_sha is None:
                expected_sha = digest
            assert digest == expected_sha
            result.update({'order': order, 'AB_mode': mode, 'file_bytes': a.stat().st_size,
                           'same_run_and_shared_SHA256': digest,
                           'parsed_records_and_every_row_equal': verify_output(a, value)})
            results.append(result)
    means = {mode: statistics.mean(r['pair_total_seconds'] for r in results if r['AB_mode'] == mode) for mode in ('A', 'B')}
    return {'passed': True, 'source_shape_sample': source, 'actual_training_history_used_as_output': False,
            'own_PID_memory_and_global_guard_samples': memory,
            'memory_measurement_limit': 'Before/after-operation own-private samples and API peakworking-set; actual transient private peak not continuously sampled.',
            'carrier': 'Synthetic25109-row list, four retained actual CLOSED rows; shallow row dicts and shared nested objects. No full decoded actual history list or giant encoded text/bytes retained.',
            'results': results, 'mean_total_seconds': means,
            'mean_pair_time_reduction_percent': 100*(means['A']-means['B'])/means['A'],
            'limitations': ['One ABBA sequence on this filesystem/cache state, not a robust performance distribution.',
                           'json.dump timing includes interleaved buffered writes; it is not pure JSON CPU time.',
                           'Authentic row-shape production-sized synthetic workload; no actual worker/GPU/control benchmark yet.',
                           'No historical metric publication, source checkpoint load or Trackio DB mutation.']}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--profile', action='store_true'); p.add_argument('--output', required=True)
    args = p.parse_args()
    report = {'schema': 1, 'kind': 'paired_history_publication_CPU_verification_v1',
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'helper': record(HERE/'atomic_paired_history_v1.py'),
              'atomic_source': record(HERE/'atomic_io_retry_v1.py'),
              'fixture_source': record(__file__), 'correctness': controls(),
              'GPU_Torch_checkpoint_import_or_read': False, 'existing_source_logger_or_DB_modified': False}
    if args.profile:
        report['matched_CPU_ABBA'] = profile(HERE/'runs/adaptive_20261012_TR_uniform_full_epoch_01/training_history.json')
    report['passed'] = True
    path = Path(args.output)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
    print(json.dumps({'report': record(path), 'correctness': report['correctness'],
                      'CPU_ABBA': report.get('matched_CPU_ABBA')}))


if __name__ == '__main__':
    main()
