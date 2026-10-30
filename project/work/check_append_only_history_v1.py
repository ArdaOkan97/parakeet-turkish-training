"""Stdlib-only INACTIVE journal correctness + synthetic 37k publication timing.

No production histories, ML, Trackio, model state, or GPU. Disposable files stay
under work. The actual frozen json.dump publisher is the timing/format baseline.
"""
from __future__ import annotations
import argparse
import copy
import ctypes
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import statistics
import tempfile
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/parakeet_turkish/append_only_history_v1.py'
DEFAULT_REPORT = ROOT / 'outputs/parakeet_turkish/append_only_history_CPU_verification_v2.json'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for payload in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(payload)
    return result.hexdigest()


def host_sample():
    if os.name != 'nt':
        return {'available': False}
    class Performance(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_uint32)] + [(key, ctypes.c_size_t) for key in
            ('CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal', 'PhysicalAvailable',
             'SystemCache', 'KernelTotal', 'KernelPaged', 'KernelNonpaged', 'PageSize')] + [
            ('HandleCount', ctypes.c_uint32), ('ProcessCount', ctypes.c_uint32),
            ('ThreadCount', ctypes.c_uint32)]
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    value = Performance()
    value.cb = ctypes.sizeof(value)
    if not psapi.GetPerformanceInfo(ctypes.byref(value), value.cb):
        return {'available': False, 'error': ctypes.get_last_error()}
    total, limit = value.CommitTotal * value.PageSize, value.CommitLimit * value.PageSize
    return {'available': True, 'system_commit_bytes': total, 'system_commit_limit_bytes': limit,
            'system_commit_fraction': total / limit, 'physical_available_bytes':
            value.PhysicalAvailable * value.PageSize}


def below_normal():
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
            raise OSError('Cannot apply BelowNormal priority')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--skip-timing', action='store_true')
    args = parser.parse_args()
    below_normal()
    module = load(SOURCE, 'inactive_history_candidate')
    checks, rejections = [], []
    temp = Path(tempfile.mkdtemp(prefix='history-journal-cpu-v1-', dir=ROOT / 'work'))
    checkpoint = {'path': 'OPAQUE_NOT_OPENED_source_checkpoint.pt', 'sha256': '1' * 64, 'bytes': 123}
    def check(name, value):
        if not value:
            raise AssertionError(name)
        checks.append(name)
    def reject(name, call, exception=module.JournalError):
        try:
            call()
        except exception as error:
            rejections.append({'name': name, 'message': str(error)})
        else:
            raise AssertionError('Expected rejection: ' + name)
    def row(step):
        return {'step': step, 'loss': step / 7, 'negative_zero': -0.0,
                'sampled_source_ids': ['TR:a', 'PT:b'], 'unicode': 'ı ş Türkçe\u2028line\u2029end',
                'coverage': {'nested': {'draws': step * 14, 'complete': False},
                             'unknown_flags': [True, None, 1.2345678901234567]},
                'retention_counter': 3}
    def new(name, **kwargs):
        return module.HistoryJournal(temp / name, **kwargs)
    history = [row(i) for i in range(1, 4)]
    journal = new('basic')
    seed_report = journal.seed(history, checkpoint)
    check('seed binds checkpoint and complete ordered prefix', seed_report['max_step'] == 3
          and seed_report['checkpoint_record'] == checkpoint)
    history += [row(i) for i in range(4, 14)]
    report = journal.publish(history)
    check('periodic publication appends only 10 records', report['records_written'] == 10)
    check('reconstruction equals all nested metrics', list(journal.records()) == history)
    observer = module.HistoryJournal(journal.directory)
    history[-1]['validation'] = {'tr': {'WER': .445, 'S': 3, 'I': 2, 'D': 1},
                                 'pt': {'retention_pass': False}}
    history[-1]['selection_diagnostics'] = {'best_step': 8250, 'counter': 4}
    amended = journal.publish(history, publication='DEV')
    check('latest DEV amendment writes exactly one upsert', amended['records_written'] == 1)
    seq = journal.event_id
    check('identical retry is idempotent', journal.publish(history, publication='DEV')['idempotent']
          and journal.event_id == seq)
    check('observer incremental refresh sees amended latest only', observer.refresh()
          and observer.get_step(13) == history[-1] and observer.frontier()['event_id'] == seq)
    check('bounded observer suffix includes only requested rows',
          list(observer.iter_latest(12)) == history[11:] and not observer.refresh())
    snapshot = temp / 'complete.json'
    journal.snapshot(snapshot)
    expected = json.dumps(history, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    check('snapshot byte-identical to frozen legacy format incl Unicode separators',
          snapshot.read_bytes() == expected.encode('utf-8'))
    decoded = json.loads(snapshot.read_text(encoding='utf-8'))
    check('parsed snapshot exact fields and float sign', decoded == history
          and repr(decoded[0]['negative_zero']) == '-0.0')
    restarted = module.HistoryJournal(journal.directory)
    check('restart verifies checksums and complete reconstruction', list(restarted.records()) == history)
    bad = copy.deepcopy(history)
    bad[-1]['loss'] += 1
    reject('latest immutable metric conflict', lambda: journal.publish(bad, publication='DEV'))
    bad = copy.deepcopy(history)
    bad[-1]['validation']['tr']['WER'] += .01
    reject('unlabelled latest DEV change', lambda: journal.publish(bad))
    bad = history + [row(15)]
    reject('gap in appended optimizer steps', lambda: journal.publish(bad))
    bad = history + [row(14), row(14)]
    reject('duplicate appended optimizer step', lambda: journal.publish(bad))
    bad = history + [row(15), row(14)]
    reject('out of order appended optimizer steps', lambda: journal.publish(bad))
    reject('rollback requires explicit authoritative reconciliation', lambda: journal.publish(history[:3]))
    reject('invalid checkpoint binding', lambda: new('bad-binding').seed([row(1)], {}))
    reject('boolean optimizer step rejected', lambda: new('bool-step').seed([{'step': True}], checkpoint))
    bounded = new('bounded', max_record_bytes=1024)
    reject('bounded payload cannot allocate full history JSON', lambda: bounded.seed(
        [{'step': 1, 'payload': 'a' * 1200}], checkpoint))
    reject('JSON escaping expansion rejected before whole token allocation', lambda: bounded.seed(
        [{'step': 1, 'payload': '\x01' * 300}], checkpoint))
    reject('nonfinite metric rejected', lambda: new('nonfinite').seed(
        [{'step': 1, 'loss': float('nan')}], checkpoint), ValueError)
    check('failed record writes did not commit HEAD', not bounded.head_path.exists())

    # Interrupted/torn tails are uncommitted private files; committed references
    # must reject corruption/truncation, not silently roll back a valid event.
    orphan = journal.directory / 'uncommitted-crash.tmp'
    orphan.write_bytes(module.MAGIC + b'torn frame')
    check('torn uncommitted tail ignored and retained',
          list(module.HistoryJournal(journal.directory).records()) == history and orphan.exists())
    corrupt_dir = temp / 'corrupt'
    import shutil
    shutil.copytree(journal.directory, corrupt_dir)
    committed = corrupt_dir / journal.head['last_transaction']['name']
    raw = bytearray(committed.read_bytes())
    raw[-20] ^= 1
    committed.write_bytes(raw)
    reject('corrupt committed footer rejected', lambda: module.HistoryJournal(corrupt_dir))
    truncate_dir = temp / 'truncate'
    shutil.copytree(journal.directory, truncate_dir)
    committed = truncate_dir / journal.head['last_transaction']['name']
    with committed.open('r+b') as stream:
        stream.truncate(committed.stat().st_size - 3)
    reject('truncated committed data rejected', lambda: module.HistoryJournal(truncate_dir))

    failed = new('HEAD-failure')
    two = [row(1), row(2)]
    failed.seed(two, checkpoint)
    original_write = failed.atomic.write_json
    def fail_head(path, value, **kwargs):
        raise OSError('injected before atomic HEAD commit')
    failed.atomic.write_json = fail_head
    three = two + [row(3)]
    reject('HEAD failure propagated', lambda: failed.publish(three), OSError)
    failed.atomic.write_json = original_write
    after_failure = module.HistoryJournal(failed.directory)
    check('restart excludes fully written but uncommitted transaction', after_failure.max_step == 2
          and len(list(failed.directory.glob('*.txn'))) == 2)
    after_failure.publish(three)
    check('retry deterministic commit no duplicate event or row', after_failure.event_id == 2
          and list(after_failure.records()) == three)
    check('repeat postcommit retry remains no-op', after_failure.publish(three)['idempotent'])

    old_snapshot = snapshot.read_bytes()
    original_replace = journal.atomic.retry_replace
    def fail_replace(source, destination, **kwargs):
        raise OSError('injected snapshot replacement failure')
    journal.atomic.retry_replace = fail_replace
    reject('snapshot commit failure propagated', lambda: journal.snapshot(snapshot), OSError)
    journal.atomic.retry_replace = original_replace
    check('failed snapshot atomic replacement preserved old complete JSON and tempfile',
          snapshot.read_bytes() == old_snapshot
          and bool(list(snapshot.parent.glob(snapshot.name + '.atomic-v1-*.tmp'))))

    branch_journal = new('checkpoint-reconcile')
    original_history = [row(i) for i in range(1, 7)]
    branch_journal.seed(original_history[:3], checkpoint)
    branch_journal.publish(original_history)
    reject('orphan committed suffix cannot silently disappear', lambda: branch_journal.reconcile_checkpoint(
        original_history[:4], checkpoint))
    wrong_prefix = copy.deepcopy(original_history[:4])
    wrong_prefix[1]['loss'] = 100
    reject('authoritative older prefix conflict', lambda: branch_journal.reconcile_checkpoint(
        wrong_prefix, checkpoint, preserve_orphan_suffix=True))
    restored_history = copy.deepcopy(original_history[:4])
    restored_history[-1]['validation'] = {'tr': {'WER': .3}}
    reconciliation = branch_journal.reconcile_checkpoint(restored_history, checkpoint,
                                                        preserve_orphan_suffix=True)
    branch = reconciliation['orphan_branch']
    check('model checkpoint cuts active timeline only after explicit commit',
          branch_journal.max_step == 4 and list(branch_journal.records()) == restored_history)
    check('failed historical suffix retained and independently readable',
          list(branch_journal.branch_records(branch)) == original_history
          and branch['failed_step_range'] == [5, 6] and branch['used_as_training_state'] is False)
    restored_history += [row(5), row(6), row(7)]
    branch_journal.publish(restored_history)
    check('same checkpoint continuation has no duplicated steps or silent skipped rows',
          list(branch_journal.records()) == restored_history
          and [r['step'] for r in branch_journal.records()] == list(range(1, 8)))
    check('restart retains active path and failed historical branch',
          list(module.HistoryJournal(branch_journal.directory).records()) == restored_history
          and len(module.HistoryJournal(branch_journal.directory).orphan_branches) == 1)
    # Journal lag behind an authoritative checkpoint may include a missed latest
    # DEV amendment. Admission verifies the prefix and then fills the exact tail.
    lagged = new('lagged-checkpoint')
    lagged.seed(restored_history[:3], checkpoint)
    authoritative = copy.deepcopy(restored_history)
    authoritative[2]['validation'] = {'tr': {'WER': .7}}
    lagged.reconcile_checkpoint(authoritative, checkpoint)
    check('checkpoint ahead recovers missed DEV and exact suffix', list(lagged.records()) == authoritative)
    check('same authoritative restart is idempotent',
          lagged.reconcile_checkpoint(authoritative, checkpoint)['idempotent'])
    # Nested branch cutoffs containing zero rows are also fully recoverable.
    nested = branch_journal.reconcile_checkpoint(restored_history[:5], checkpoint,
                                                 preserve_orphan_suffix=True)
    nested2 = branch_journal.reconcile_checkpoint(restored_history[:4], checkpoint,
                                                  preserve_orphan_suffix=True)
    check('nested zero-frame orphan branches remain readable',
          list(branch_journal.branch_records(nested2['orphan_branch'])) == restored_history[:5]
          and list(branch_journal.branch_records(nested['orphan_branch'])) == restored_history)

    timing = {'executed': False}
    sample_before = host_sample()
    if not args.skip_timing:
        if sample_before.get('available') and (sample_before['system_commit_fraction'] >= .90
                or sample_before['physical_available_bytes'] < 2 * 1024 ** 3):
            timing = {'executed': False, 'deferred_for_current_training_headroom': True,
                      'host_sample': sample_before}
        else:
            # Tiny rows with real nested shapes and 37k optimizer steps. Shared
            # immutable nested objects reduce fixture RAM; no production input.
            shared = {'Turkish_domain_coverage': {'uniform_pass': {'draws': 253800,
                       'unique': 253798, 'complete': True}, 'domains': {'fleurs': {'draws': 5}}},
                      'retention': {'en': True, 'es': False, 'pt': False},
                      'nested_unknown_flags': [True, False, None, .12345678901234568]}
            synthetic = [{'step': i, 'loss': i / 37, 'head_lr': .00001, 'lora_lr': .0001,
                          'training_seconds': .99, 'audio_seconds': 87.2, 'metrics': shared}
                         for i in range(1, 37001)]
            measured = new('timing-37k')
            tracemalloc.start()
            started = time.perf_counter()
            measured.seed(synthetic, checkpoint)
            seed_seconds = time.perf_counter() - started
            seed_peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            atomic = module._atomic()
            samples = []
            for mode in ('legacy', 'journal', 'journal', 'legacy'):
                start_step = len(synthetic) + 1
                synthetic.extend({'step': i, 'loss': i / 37, 'head_lr': .00001, 'lora_lr': .0001,
                                  'training_seconds': .99, 'audio_seconds': 87.2, 'metrics': shared}
                                 for i in range(start_step, start_step + 10))
                started = time.perf_counter()
                if mode == 'legacy':
                    run = temp / 'legacy-run.json'
                    paired = temp / 'legacy-shared.json'
                    atomic.write_json(run, synthetic)
                    atomic.write_json(paired, synthetic)
                    operation = {'full_snapshots_serialized': 2, 'rows_per_snapshot': len(synthetic),
                                 'published_bytes': run.stat().st_size + paired.stat().st_size}
                else:
                    operation = measured.publish(synthetic)
                seconds = time.perf_counter() - started
                samples.append({'mode': mode, 'rows': len(synthetic), 'seconds': seconds,
                                'allocation_tracing_enabled_during_timing': False,
                                'operation': operation})
                if mode == 'legacy':
                    # Keep the exact same journal timeline; this untimed update
                    # is not silently omitted from the measured operation scope.
                    tracemalloc.start()
                    measured.publish(synthetic)
                    allocation_peak = tracemalloc.get_traced_memory()[1]
                    tracemalloc.stop()
                    samples[-1]['untimed_journal_delta_allocation_probe_peak_bytes'] = allocation_peak
            snapshots = [measured.snapshot(temp / name) for name in ('journal-run.json', 'journal-shared.json')]
            check('37k final journal snapshots exactly equal both frozen full publications',
                  digest(temp / 'journal-run.json') == digest(temp / 'legacy-run.json')
                  == digest(temp / 'journal-shared.json') == digest(temp / 'legacy-shared.json'))
            check('37k compact index bounded and whole row reconstruction equal',
                  measured.index_bytes == 52 * len(synthetic)
                  and all(a == b for a, b in zip(measured.records(), synthetic)))
            averages = {mode: statistics.mean(r['seconds'] for r in samples if r['mode'] == mode)
                        for mode in ('legacy', 'journal')}
            timing = {'executed': True, 'scope': 'synthetic_37000_step_tiny_nested_rows_ABBA_periodic_publication_only',
                      'sequence': ['legacy', 'journal', 'journal', 'legacy'], 'samples': samples,
                      'allocation_tracing_enabled_during_ABBA_timing': False,
                      'mean_seconds': averages,
                      'periodic_seconds_reduction_percent': 100 * (1 - averages['journal'] / averages['legacy']),
                      'seed_seconds_one_time': seed_seconds, 'seed_peak_traced_incremental_bytes': seed_peak,
                      'allocation_tracing_enabled_during_seed': True,
                      'final_complete_pair_snapshot_operations': snapshots,
                      'rows_final': len(synthetic), 'index_bytes': measured.index_bytes,
                      'max_record_payload_bound_bytes': measured.max_record_bytes,
                      'full_payload_in_memory': False,
                      'candidate_changes_cadence': True,
                      'not_end_to_end_training_or_production_gain': True,
                      'legacy_original_publisher': str(ROOT / 'outputs/parakeet_turkish/atomic_io_retry_v1.py')}
    sample_after = host_sample()
    result = {'schema': 1, 'kind': 'inactive_append_only_history_CPU_verification_v1',
              'passed': True, 'source': {'path': str(SOURCE), 'sha256': digest(SOURCE),
                                       'bytes': SOURCE.stat().st_size},
              'test_source': {'path': str(Path(__file__).resolve()), 'sha256': digest(__file__)},
              'checks_passed': len(checks), 'checks': checks, 'rejections_passed': len(rejections),
              'rejections': rejections, 'timing': timing, 'host_before': sample_before,
              'host_after': sample_after, 'scratch_directory': str(temp),
              'actual_production_history_read': False, 'Torch_Trackio_GPU_imports': False,
              'adopted_or_native_rollout': False, 'training_RNG_or_sampler_touched': False,
              'limits': ['Sole writer only; a separate adapter must bind actual checkpoint/scientific identity.',
                         'publish(history) inspects only the frontier and new suffix; older rows must be immutable.',
                         'Full authoritative prefix verification occurs at seed/exact checkpoint reconciliation.',
                         'Legacy JSON observers need journal frontier/suffix support between full snapshots.',
                         'FS fsync/atomic replace are verified at process level; physical power-loss durability unmeasured.',
                         'Transaction/shard archival retention policy is not implemented; failed suffixes are preserved.']}
    module._atomic().write_json(args.report, result)
    print(json.dumps({'report': str(args.report), 'sha256': digest(args.report),
                      'source_sha256': result['source']['sha256'], 'passed': result['passed'],
                      'checks': len(checks), 'rejections': len(rejections), 'timing': timing.get('mean_seconds'),
                      'periodic_reduction_percent': timing.get('periodic_seconds_reduction_percent')}))


if __name__ == '__main__':
    main()
