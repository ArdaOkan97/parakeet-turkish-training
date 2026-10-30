"""Full-byte aggregate CPU references; new public protocol, review-required generic9223 recorded-room slot.

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
import fastmss_BPE9223_cpu_common_v11 as core
import fastmss_owned_process_closure_v2 as process
import fastmss_BPE9223_feature_protocol_v11 as public

REQUEST, REFERENCE, REFERENCE_UPDATE, MODE = public.REQUEST, public.REFERENCE, public.REFERENCE_MEMBER, public.MODE


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
    request_pin = core.record(path, guard); request = core.read_bound(request_pin, guard)
    core.require(request['kind'] == REQUEST and request['mode'] == MODE
        and request['production_enabled'] is False and stage in ('reference', 'frontend'),
        'Exact source-bound aggregate preparation request, no private hash-mode alias')
    from asr18_BPE9223_source_context_v8 import verify_CPU_configuration
    reviewed = verify_CPU_configuration(request['window_configuration'], request['ROOT_review'], guard=guard)
    core.require(core.same(reviewed, request['configuration']), 'Actual direct or template ROOT authorization before numerical import/audio')
    source, public_plan = public.verify_source_contract(request['source_contract'], guard=guard)
    first, plan, _, contract, validated = core.load_request(request['numerical_request'], guard)
    core.require(first['update_index'] == 0 and core.same(plan, public_plan)
        and core.same(first['bridge'], source['bridge']) and core.same(request['bridge'], first['bridge'])
        and core.same(first['contract'], source['numerical_CPU_contract']),
        'Exact unchanged numerical CPU8 contract and full original bridge')
    core.require(core.same(request['parent_source'], source['sources']['controller'])
        and type(request['freeze_after_setup']) is bool
        and request['freeze_after_setup'] == source['freeze_after_setup']
        and core.same(core.record(__file__, guard), source['sources']['reference']),
        'Actual new source/parent/explicit GC policy, no old controller authority alias')
    return request_pin, request, plan, contract, validated, None


def output_folder(request, output, stage):
    output = core.scoped(output)
    root = core.scoped(request['output_root'])
    core.require(output == root / stage / 'report.json' and not output.is_relative_to(core.sink.CACHE.resolve()),
                 'Fresh private output, never the shared feature slot')
    output.parent.mkdir(parents=True, exist_ok=False)
    return output.parent


def closed_reference_window(request, request_pin, plan, guard):
    return public.verify_reference_window(request['reference_window_report'], request['reference_window_actual0'],
        contract_pin=request['source_contract'], guard=guard)


def main(request_path, output):
    began = time.monotonic(); core.environment(); timings = Timings()
    with timings.stage('immutable_metadata_and_oracle_validation'):
        request_pin, request, plan, contract, validated, oracle = load_window_request(request_path, 'reference')
    source, _ = public.verify_source_contract(request['source_contract'], guard=core.resource_guard)
    core.require(core.same(source['sources']['reference'], core.record(__file__)), 'Actual new reference-window worker source')
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
            target = folder / f'update_{index:02}.json'
            value = {'kind': REFERENCE_UPDATE, 'passed': True, 'producer': core.identity(),
                'worker_source': core.record(__file__, guard), 'parent_source': request['parent_source'],
                'request': request_pin, 'source_contract': request['source_contract'], 'bridge': request['bridge'], 'update_index': index, 'phase': update['phase'],
                'references': refs,
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
        'request': request_pin, 'source_contract': request['source_contract'], 'bridge': request['bridge'], 'updates': updates, 'all_updates_reported': True, 'per_update_child0_claimed': False,
        'raw_lifetime_receipt': receipt, 'raw_PCM_files_written': 0, 'maximum_simultaneous_rows': 1,
        'actual_optimizer_updates': 0, 'sampler_journal_frontier_advanced': False,
        'actual_process_exit_observed_in_child': False, 'actual_aggregate_exit_required': True,
        'NumPy': np.__version__, 'timings': timings.values, 'gc_events': timings.gc_events,
        'wall_seconds': time.monotonic() - began, 'CPU_stage_admission': admission, 'production_enabled': False}
    core.write_new(core.scoped(output), result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--request', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); main(args.request, args.output)
