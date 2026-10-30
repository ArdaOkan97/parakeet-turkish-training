"""Finite window-scoped independent full-PCM references; no native admission.

One actual CPU leaf covers <=20 immutable ordered updates. Per-update files are
reports from a still-live process, not child0 claims. Only the real parent's
aggregate Popen0 can close this window. Original CPU8 readers remain unchanged.
"""
import argparse
from contextlib import contextmanager
import gc
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
import fastmss_row_serial_cpu_v8 as core
import fastmss_owned_process_closure_v2 as process

REQUEST = 'FastMSS_private_finite_CPU_window_request_v1'
REFERENCE = 'FastMSS_independent_reference_window_v1'
REFERENCE_UPDATE = 'FastMSS_independent_reference_update_within_live_window_v1'
MODE = 'private_hash_only_window_validation'


class Timings:
    def __init__(self):
        self.values = {}
        self.gc_events = []
        self.phase = 'setup'
        self.original_collect = None

    @contextmanager
    def stage(self, name):
        old = self.phase; self.phase = name
        wall, cpu = time.perf_counter(), time.process_time()
        try:
            yield
        finally:
            value = self.values.setdefault(name, {'wall_seconds': 0., 'process_CPU_seconds': 0., 'calls': 0})
            value['wall_seconds'] += time.perf_counter() - wall
            value['process_CPU_seconds'] += time.process_time() - cpu
            value['calls'] += 1
            self.phase = old

    def install_gc_timer(self):
        core.require(self.original_collect is None, 'One process-local GC timer')
        self.original_collect = gc.collect
        def timed(*args, **kwargs):
            wall, cpu = time.perf_counter(), time.process_time()
            result = self.original_collect(*args, **kwargs)
            core.require(len(self.gc_events) < 2048, 'Bounded <=20-update GC event table')
            self.gc_events.append({'phase': self.phase, 'wall_seconds': time.perf_counter() - wall,
                'process_CPU_seconds': time.process_time() - cpu, 'collected_objects': result})
            return result
        gc.collect = timed

    def close(self):
        if self.original_collect is not None:
            gc.collect = self.original_collect


def load_window_request(path, stage):
    guard = core.resource_guard
    request_pin = core.record(path, guard)
    request = core.read_bound(request_pin, guard)
    core.require(request['kind'] == REQUEST and request['mode'] == MODE
        and request['production_enabled'] is False and request['native_features_ready'] is False,
        'Explicit private validation-only window, never a public owner alias')
    core.require(stage in ('reference', 'frontend') and type(request['freeze_after_setup']) is bool,
                 'Exact finite stage/GC mode')
    for key, pin in request['sources'].items():
        core.require(core.same(core.record(pin['path'], guard), pin), 'Exact new window source: ' + key)
    core.require(core.same(request['sources']['core'], core.record(core.__file__, guard))
        and core.same(request['sources']['reference'], core.record(__file__, guard)), 'Exact CPU8 numerical core/new reference source')
    baseline = core.read_bound(request['baseline_CPU_report'], guard)
    process.verify(core.read_bound(request['baseline_CPU_actual0'], guard),
        kind='ASR18_owned_CPU_process_completion_v1', owner='/root/memory_diagnosis',
        report=request['baseline_CPU_report'], guard=guard)
    core.require(baseline['passed'] is True and baseline['all_children_actually_closed0'] is True
        and baseline['actual_optimizer_updates'] == 0 and baseline['sampler_journal_frontier_advanced'] is False,
        'Actual old full CPU window0 is the immutable oracle')
    entries = request['original_requests']
    core.require(type(entries) is list and 1 <= len(entries) <= 20
        and len(entries) == len(baseline['independent_reference_proofs']) == len(baseline['frontend_proofs']),
        'Exactly the bounded complete original window')
    first, plan, _, contract, validated = core.load_request(entries[0], guard)
    core.require(first['update_index'] == 0 and core.same(first['bridge'], baseline['bridge'])
        and core.same(request['bridge'], first['bridge'])
        and core.same(first['contract'], baseline['contract']) and len(plan['updates']) == len(entries),
        'No selected-row/window shortening or changed numerical contract')
    oracle = []
    for index, pin in enumerate(entries):
        item = core.read_bound(pin, guard)
        core.require(item['kind'] == core.REQUEST_KIND and item['production_enabled'] is False
            and type(item['update_index']) is int and item['update_index'] == index
            and core.same(item['bridge'], first['bridge']) and core.same(item['contract'], first['contract']),
            'Exact original ordered request/bridge/session/contract')
        ref, feature = baseline['independent_reference_proofs'][index], baseline['frontend_proofs'][index]
        process.verify(core.read_bound(ref['actual0'], guard), kind='ASR18_owned_CPU_process_completion_v1',
            owner='/root/memory_diagnosis', report=ref['report'], guard=guard)
        process.verify(core.read_bound(feature['actual0'], guard), kind='ASR18_owned_CPU_process_completion_v1',
            owner='/root/memory_diagnosis', report=feature['report'], guard=guard)
        closed = core.sink.verify_feature_closure(feature['feature_closure'], baseline['owner'], success=True, guard=guard)
        saved = core.read_bound(closed['update_record'], guard)
        old_refs = core.read_bound(ref['report'], guard)
        core.require(saved['update_index'] == index and old_refs['update_index'] == index
            and core.same(old_refs['request'], pin)
            and len(saved['rows']) == len(old_refs['references']) == len(plan['updates'][index]['rows']),
            'Complete actually closed baseline rows/references')
        oracle.append({'reference_pin': ref['report'], 'references': old_refs['references'],
            'feature_update_pin': closed['update_record'], 'feature_update': saved})
    core.require(core.same(request['sources']['parent'], request['parent_source'])
        and core.same(request['sources']['guard'], contract['sources']['guard']), 'Real new parent/unchanged guard source')
    return request_pin, request, plan, contract, validated, oracle


def output_folder(request, output, stage):
    output = core.scoped(output)
    root = core.scoped(request['output_root'])
    core.require(output == root / stage / 'report.json' and not output.is_relative_to(core.sink.CACHE.resolve()),
                 'Fresh private output, never the shared feature slot')
    output.parent.mkdir(parents=True, exist_ok=False)
    return output.parent


def closed_reference_window(request, request_pin, plan, guard):
    report_pin, actual_pin = request['reference_window_report'], request['reference_window_actual0']
    value = core.read_bound(report_pin, guard)
    process.verify(core.read_bound(actual_pin, guard), kind='ASR18_owned_CPU_process_completion_v1',
                   owner='/root/fastmss_integration', report=report_pin, guard=guard)
    core.require(value['kind'] == REFERENCE and value['passed'] is True and value['mode'] == MODE
        and core.same(value['bridge'], request['bridge']) and value['all_updates_reported'] is True
        and value['actual_optimizer_updates'] == 0 and value['sampler_journal_frontier_advanced'] is False
        and len(value['updates']) == len(plan['updates']) and value['raw_lifetime_receipt']['passed'] is True,
        'Genuine complete reference-window process0 before any frontend')
    old_request = core.read_bound(value['request'], guard)
    core.require(core.same(old_request['baseline_CPU_report'], request['baseline_CPU_report'])
        and core.same(old_request['original_requests'], request['original_requests'])
        and core.same(old_request['sources'], request['sources']), 'Reference/frontend window requests differ')
    results = []
    for index, entry in enumerate(value['updates']):
        item = core.read_bound(entry['report'], guard)
        core.require(type(entry['update_index']) is int and entry['update_index'] == index
            and item['kind'] == REFERENCE_UPDATE and item['update_index'] == index
            and core.same(item['producer'], value['producer']) and core.same(item['bridge'], request['bridge'])
            and item['phase'] == plan['phase'] and item['per_update_child0_claimed'] is False
            and len(item['references']) == len(plan['updates'][index]['rows']),
            'Ordered selected update is a member of the actual aggregate0, never a fabricated child0')
        results.append(item)
    return value, results


def main(request_path, output):
    began = time.monotonic(); core.environment(); timings = Timings()
    with timings.stage('immutable_metadata_and_oracle_validation'):
        request_pin, request, plan, contract, validated, oracle = load_window_request(request_path, 'reference')
    core.require(core.same(request['sources']['reference'], core.record(__file__)), 'Actual new reference worker source')
    guard = core.stage_guard(contract, 'reference')
    admission = core.admit(contract['resource_review']['reference_incremental_bytes'], guard)
    folder = output_folder(request, output, 'reference')
    timings.install_gc_timer(); reader = None; updates = []
    try:
        with timings.stage('NumPy_import_and_reader_setup'):
            np, _ = core.numpy_runtime()
            reader = core.OriginalRows(validated, raw_contract=contract['raw_lifetime_review'],
                private_limit_bytes=contract['resource_review']['reference_maximum_private_bytes'],
                frontend_enabled=False, guard=guard)
        for index, update in enumerate(plan['updates']):
            guard(); core.require(reader.active_job is None and not reader.ledger.jobs, 'Empty raw ledger at every update boundary')
            refs = []
            for raw, expected in zip(update['coordinator_update']['rows'], update['rows']):
                with timings.stage('original_decode_resample_hash_geometry_heldout'):
                    wave = reader(raw, expected, update['phase'])
                    refs.append(core.reference_entry(raw, expected, reader))
                    del wave
                with timings.stage('outer_row_gc'):
                    gc.collect()
                with timings.stage('raw_weakref_finish_including_original_inner_gc'):
                    reader.finish_raw_row()
            core.require(core.same(refs, oracle[index]['references']), 'Every full reference field must match actual old independent references')
            target = folder / f'update_{index:02}.json'
            value = {'kind': REFERENCE_UPDATE, 'passed': True, 'producer': core.identity(),
                'worker_source': core.record(__file__, guard), 'parent_source': request['parent_source'],
                'request': request_pin, 'bridge': request['bridge'], 'update_index': index, 'phase': update['phase'],
                'references': refs, 'baseline_reference': oracle[index]['reference_pin'],
                'per_update_child0_claimed': False, 'aggregate_actual_exit_required': True,
                'actual_optimizer_updates': 0, 'sampler_journal_frontier_advanced': False}
            core.write_new(target, value)
            updates.append({'update_index': index, 'report': core.record(target, guard)})
        receipt = reader.raw_receipt()
    finally:
        if reader is not None: reader.close()
        timings.close()
    core.require(not any(k == 'torch' or k.startswith(('torch.', 'transformers', 'pyarrow')) for k in sys.modules),
                 'Reference window still imports no Torch/frontend/model/Arrow')
    result = {'kind': REFERENCE, 'mode': MODE, 'passed': True, 'producer': core.identity(),
        'worker_source': core.record(__file__, guard), 'parent_source': request['parent_source'],
        'request': request_pin, 'bridge': request['bridge'], 'updates': updates, 'all_updates_reported': True,
        'raw_lifetime_receipt': receipt, 'raw_PCM_files_written': 0, 'maximum_simultaneous_rows': 1,
        'actual_optimizer_updates': 0, 'sampler_journal_frontier_advanced': False,
        'actual_process_exit_observed_in_child': False, 'actual_aggregate_exit_required': True,
        'NumPy': np.__version__, 'timings': timings.values, 'gc_events': timings.gc_events,
        'wall_seconds': time.monotonic() - began, 'CPU_stage_admission': admission, 'production_enabled': False}
    core.write_new(core.scoped(output), result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--request', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); main(args.request, args.output)
