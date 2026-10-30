"""INACTIVE finite pure-CPU equivalence, limits, lifetime and ABBA timing proof.
Plans/metadata/native operations are synthetic METHOD_STUBs, never production
data. Original canonical() is the real source-bound function. No unpickle, GC,
Torch/model/audio/GPU, SQLite/corpus query or production operation is performed.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import threading
import time
import traceback
import tracemalloc
import weakref

import psutil

P = Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
sys.path.insert(0, str(P / 'outputs/parakeet_turkish'))
from bpe_broad_production_state_v1 import canonical, digest
import plan_binary_validation_v2 as binary
from main_path_scalar_attribution_v1 import MainPathScalarAttribution, LABELS


def require(value, message):
    if not value:
        raise AssertionError(message)


def pin(path):
    path = Path(path)
    raw = path.read_bytes()
    require(len(raw) <= 32 * 1024**2, 'Bounded source/metadata')
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def identity(pid=None):
    proc = psutil.Process(os.getpid() if pid is None else pid)
    return {'pid': proc.pid, 'create_time': proc.create_time(), 'ppid': proc.ppid()}


def write(path, value):
    raw = canonical(value) + b'\n'
    require(len(raw) <= 32 * 1024**2, 'Bounded proof scalar report')
    with Path(path).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return pin(path)


def witness(fn=canonical, **kwargs):
    return binary.PlanBinaryWitness(fn,
        admitted_extra_reservation_bytes=binary.EXTRA_RESERVATION_BYTES, **kwargs)


def synthetic_plan(seed=17):
    rng = random.Random(seed)  # Separate deterministic fixture RNG, never global authority.
    updates = []
    rows = {}
    phrase = 'Özgün Türkçe çalışma, kaynak ve tam hedef. '
    for index in range(20):
        draws = []
        for slot in range(16):
            ordinal = index * 16 + slot + 1
            source_id = 'METHOD_STUB_' + str(ordinal).zfill(10)
            row = {'ordinal': ordinal, 'source_id': source_id, 'language': 'tr' if slot < 14 else 'en',
                   'text': phrase * 118 + str(ordinal),
                   'tr_metadata_labels': [rng.randrange(8192) for _ in range(99)],
                   'declared_geometry': {'clean16k_samples': 210288, 'encoder_frames': 165,
                       'physical_samples': 630864, 'target_tokens': 99,
                       'source_path': 'E:/METHOD_STUB/not_read/source_' + source_id + '.tar'},
                   'augmentation_role': 'student_tr' if slot < 14 else 'replay'}
            rows[ordinal] = row
            draws.append({'ordinal': ordinal, 'source_id': source_id, 'language': row['language']})
        updates.append({'update_index': index, 'batch_id': 'METHOD_STUB_batch_' + str(index),
                        'native_transaction_sha256': hashlib.sha256(str(index).encode()).hexdigest(),
                        'before_sampler_sha256': hashlib.sha256(('before' + str(index)).encode()).hexdigest(),
                        'after_sampler_sha256': hashlib.sha256(('after' + str(index)).encode()).hexdigest(),
                        'draws': draws})
    return {'kind': 'METHOD_STUB_complete20_plan_for_CPU_equivalence_only',
            'initial_committed_updates': 204000, 'updates': updates, 'rows': rows,
            'final_private_planning_sampler': {'broad': {'cursor': 2855997},
                'turkish_rng': rng.getstate(), 'replay_updates': 204020}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ROOT-review', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    scope = json.loads(Path(args.ROOT_review).read_text(encoding='utf-8-sig'))
    require(scope['kind'] == 'ROOT_plan_binary_and_main_path_finite_CPU_exact_ONCE_execution_scope_v1'
            and scope['accepted'] is True and scope['reviewer'] == '/root'
            and scope['fixture_source'] == pin(__file__)
            and scope['binary_helper_source'] == pin(binary.__file__), 'Separate exact ROOT finite CPU admission')
    out = Path(args.output)
    require(out.drive.lower() == 'e:' and str(out) == scope['fixture_output'] and not out.exists(), 'One fresh E fixture')
    out.mkdir()
    began = time.monotonic()
    report = {'kind': 'ASR18_plan_binary_and_main_path_finite_CPU_v1', 'passed': False,
              'CPU_performance_supports_activation': False, 'cases': [], 'benchmarks': [],
              'fixture': identity(), 'fixture_launcher': identity(os.getppid()),
              'scientific_owners': 'METHOD_STUB synthetic only', 'real_original_canonical': True,
              'no_production_operation_unpickle_GC_model_audio_GPU': True,
              'python': sys.version, 'recursion_limit_before': sys.getrecursionlimit()}
    global_rng_before = random.getstate()
    workers = []

    def case(name, evidence=None):
        report['cases'].append({'name': name, 'passed': True, 'evidence': evidence})

    def original_failure_identity(value, *, initial=False):
        observed = []
        def observing_original(arg):
            try:
                return canonical(arg)
            except BaseException as exc:
                observed.append(exc)
                raise
        cache = witness(observing_original)
        plan = value if initial else {'valid': [1, 2]}
        if not initial:
            cache.digest(plan)
            plan['mutated'] = value
        try:
            cache.digest(plan)
        except BaseException as exc:
            require(len(observed) == 1 and exc is observed[0], 'Actual original exception object passes through unchanged')
            evidence = {'type': type(exc).__name__, 'message': str(exc), 'same_original_object': True}
        else:
            raise AssertionError('Expected original canonical failure')
        finally:
            cache.close()
            observed.clear()
        return evidence

    try:
        before = [pin(item['path']) for item in scope['input_pins']]
        require(before == scope['input_pins'], 'All frozen proof inputs')
        report['input_pins_before'] = before
        plan = {'rows': {12: {'text': 'Türkçe é 🐟', 'ids': [1, 8193, 9222]},
                         2: {'text': 'ç', 'ids': [3]}}, 'rng': (3, (1, 2, 3), None), 'float': -0.0}
        count = [0]
        def counted_original(value):
            count[0] += 1
            return canonical(value)
        cache = witness(counted_original)
        expected = digest(plan)
        for _ in range(60):
            require(cache.digest(plan) == expected, 'All original60 mutation boundaries compare full current plan')
        stats = cache.snapshot_and_reset()
        require(count[0] == 1 and stats['costs']['original_baseline']['count'] == 1
                and stats['costs']['original_baseline']['bytes'] == len(canonical(plan))
                and stats['costs']['binary_validation']['count'] == 60
                and stats['counts']['binary_equal'] == 59, 'Original first call once; all60 fresh binary visits')
        case('unicode_integer_routes_tuple_floats_and_all60_fresh_boundaries', stats)
        baseline = copy.deepcopy(plan)
        mutations = [
            ('scalar', lambda p: p.update(float=1.5)),
            ('nested_list', lambda p: p['rows'][12]['ids'].append(5)),
            ('tuple_to_list_same_canonical', lambda p: p.update(rng=[3, [1, 2, 3], None])),
            ('dict_order_same_canonical', lambda p: p.update(rows=dict(reversed(list(p['rows'].items()))))),
            ('new_key', lambda p: p.update(extra='new'))]
        for name, mutate in mutations:
            changed = copy.deepcopy(baseline)
            mutate(changed)
            require(cache.digest(changed) == digest(changed), 'Actual mutated canonical result at each boundary')
            result = cache.snapshot_and_reset()
            require(result['counts']['binary_mismatch'] == 1
                    and result['costs']['original_fallback']['count'] == 1, 'No baseline refresh or skipped fallback')
            case('mutation_' + name, result)
        require(cache.digest(baseline) == expected, 'Returning to exact original tree may witness equality')
        cache.close()

        reductions = [0]
        class Custom:
            def __reduce_ex__(self, protocol):
                reductions[0] += 1
                raise AssertionError('Custom reduction must never execute')
        class CustomList(list):
            def __reduce_ex__(self, protocol):
                reductions[0] += 1
                raise AssertionError('Subclass reduction must never execute')
        class CustomInt(int):
            def __reduce_ex__(self, protocol):
                reductions[0] += 1
                raise AssertionError('Subclass reduction must never execute')
        for name, value in [('custom', Custom()), ('bytes', b'a'), ('set', {1}),
                            ('frozenset', frozenset((1,))), ('bytearray', bytearray(b'a')),
                            ('PickleBuffer', binary._pickle.PickleBuffer(b'a'))]:
            case('unsupported_' + name, original_failure_identity(value))
            require(reductions[0] == 0, 'No custom reduction even on forbidden builtin bypass cases')
            if name == 'PickleBuffer':
                value.release()
        for value in (CustomList([1, 2]), CustomInt(3)):
            c = witness()
            require(c.digest({'x': value}) == digest({'x': value}), 'Original JSON-supported subclass outcome')
            state = c.snapshot_and_reset()
            require(state['counts']['baseline_unsupported'] == 1, 'Unsupported subclass never cached')
            c.close()
        require(reductions[0] == 0, 'All subclass reducers remain uncalled')
        case('JSON_supported_subclasses_original_fallback_no_reduction')
        for name, value in [('nan', float('nan')), ('positive_inf', float('inf')), ('negative_inf', -float('inf'))]:
            case('first_boundary_original_exception_' + name, original_failure_identity({'x': value}, initial=True))
            case('mutated_original_exception_' + name, original_failure_identity(value))
        cycle = []
        cycle.append(cycle)
        try:
            case('cycle_original_exception_identity', original_failure_identity(cycle))
        finally:
            cycle.clear()
        shared = [1, 2, 3]
        original = {'left': shared, 'right': shared}
        c = witness()
        c.digest(original)
        copied = {'left': list(shared), 'right': list(shared)}
        require(c.digest(copied) == digest(copied), 'Alias topology change takes correct original canonical')
        require(c.snapshot_and_reset()['counts']['binary_mismatch'] == 1, 'Alias changes do not refresh baseline')
        c.close()
        case('alias_topology_equality_only_safe_fallback')

        for name, changed in [('container_cap', {'x': [0] * (binary.MAX_CONTAINER_ITEMS + 1)}),
                              ('Unicode_cap', {'x': 'ç' * (binary.MAX_STRING_CODEPOINTS + 1)}),
                              ('integer_cap', {'x': 1 << (binary.MAX_INTEGER_BITS + 1)})]:
            c = witness()
            c.digest({'x': 1})
            require(c.digest(changed) == digest(changed), 'Capacity is original fallback, never scientific truncation')
            require(c.snapshot_and_reset()['counts']['binary_capacity'] == 1, 'Actual expected capacity branch')
            c.close()
            case(name)
        large = {'x': [{'a': index, 'b': index + 1, 'c': index + 2} for index in range(65536)]}
        tracemalloc.start(1)
        try:
            try:
                binary.complete_binary(large)
            except binary.BinaryCapacity as exc:
                cap_evidence = {'type': type(exc).__name__, 'message': str(exc)}
            else:
                raise AssertionError('Adversarial fullvisit/wire case must meet an actual fixed cap')
            current, peak = tracemalloc.get_traced_memory()
            require(peak < binary.EXTRA_RESERVATION_BYTES, 'Actual adversarial fresh serializer tracked allocation below64MiB')
            cap_evidence.update(traced_peak_bytes=peak, traced_current_bytes=current,
                                tracing_fixture_only_not_production=True)
        finally:
            tracemalloc.stop()
        del large
        case('actual_adversarial_visit_or_wire_cap_and_allocation_bound', cap_evidence)
        deep = []
        cursor = deep
        for _ in range(700):
            nxt = []
            cursor.append(nxt)
            cursor = nxt
        c = witness()
        require(c.digest(deep) == digest(deep), 'Original result despite stricter C recursion eligibility')
        c.close()
        del cursor, nxt, deep
        case('original_recursion_limit_not_changed_and_safe_original_fallback')

        original_pickler = binary._StrictPickler
        sentinel = MemoryError('METHOD_STUB original resource/allocation denial')
        def fail_memory(writer):
            raise sentinel
        try:
            binary._StrictPickler = fail_memory
            c = witness()
            try:
                c.digest({'x': 1})
            except MemoryError as exc:
                require(exc is sentinel, 'Actual MemoryError identity propagates, no fallback retry')
            else:
                raise AssertionError('MemoryError must propagate')
            require(c.snapshot_and_reset()['costs']['original_baseline']['count'] == 1, 'Only one actual original baseline')
            c.close()
        finally:
            binary._StrictPickler = original_pickler
        case('METHOD_STUB_MemoryError_no_retry_no_guard_relaxation')
        def unavailable(writer):
            raise ValueError('METHOD_STUB constructor unavailable')
        try:
            binary._StrictPickler = unavailable
            c = witness()
            require(c.digest({'x': 1}) == digest({'x': 1}), 'Known constructor failure preserves original result')
            require(c.snapshot_and_reset()['counts']['baseline_unsupported'] == 1, 'Constructor failure cannot create cache')
            c.close()
        finally:
            binary._StrictPickler = original_pickler
        case('METHOD_STUB_constructor_failure_original_result_and_restoration')
        class WeakList(list):
            pass
        payload = WeakList([1, 2, 3])
        ref = weakref.ref(payload)
        owned = {'payload': payload}
        c = witness()
        c.digest(owned)
        owned.clear()
        del payload
        require(ref() is None, 'No cache/memo reference retains unsupported original payload')
        c.close()
        exact = {'nested': [1, 2, 3]}
        before_refs = (sys.getrefcount(exact), sys.getrefcount(exact['nested']))
        c = witness()
        c.digest(exact)
        c.digest(exact)
        require(before_refs == (sys.getrefcount(exact), sys.getrefcount(exact['nested'])), 'Fresh Pickler memo released on success')
        c.close()
        case('real_weakref_and_exact_builtin_reference_counts_no_retained_plan_or_memo')

        def bad_clock():
            raise OSError('METHOD_STUB clock read failed')
        c = witness(wall_clock=bad_clock, thread_clock=bad_clock)
        require(c.digest({'x': 1}) == digest({'x': 1}), 'Broken clock cannot change original result')
        require(c.snapshot_and_reset()['costs']['original_baseline']['clock_errors'] > 0, 'Clock failure explicit')
        c.close()
        case('binary_clock_failure_passthrough')
        original_clock_failure = ValueError('METHOD_STUB original canonical exception with broken clocks')
        original_clock_calls = [0]
        def failing_original_with_bad_clocks(value):
            original_clock_calls[0] += 1
            raise original_clock_failure
        c = witness(failing_original_with_bad_clocks, wall_clock=bad_clock, thread_clock=bad_clock)
        try:
            c.digest({'x': 1})
        except ValueError as exc:
            require(exc is original_clock_failure and original_clock_calls[0] == 1,
                    'Broken clocks never mask/duplicate actual original canonical failure')
        else:
            raise AssertionError('Original canonical error must propagate')
        c.close()
        case('binary_clock_failure_same_original_exception_once')
        c = witness()
        wrong_thread_errors = []
        def wrong_thread():
            try:
                c.digest({'x': 1})
            except ValueError as exc:
                wrong_thread_errors.append(str(exc))
        thread = threading.Thread(target=wrong_thread, name='METHOD_STUB_wrong_owner')
        workers.append(thread)
        thread.start()
        thread.join(5)
        require(not thread.is_alive() and len(wrong_thread_errors) == 1, 'No other-thread scientific witness')
        c.close()
        case('actual_MainThread_confinement')

        clock = MainPathScalarAttribution()
        events = []
        returned = object()
        argument = object()
        def operation(name, arg):
            events.append((name, arg))
            return returned
        stamp = clock.start('authoritative_propose')
        raised = True
        try:
            result = operation('propose', argument)
            raised = False
        finally:
            clock.finish(stamp, raised=raised)
        require(result is returned and events == [('propose', argument)], 'Same original call/order/argument/result identity')
        failure = ValueError('METHOD_STUB original operation exception')
        stamp = clock.start('native_and_commit')
        raised = True
        try:
            events.append(('native', argument))
            raise failure
        except ValueError as exc:
            require(exc is failure, 'Original operation exception object unchanged')
        finally:
            clock.finish(stamp, raised=raised)
        outer = clock.start('native_and_commit')
        inner = clock.start('metadata_binding')
        clock.finish(inner, raised=False)
        clock.finish(outer, raised=False)
        for index in range(20):
            clock.copy_preparation_scalar(index / 10)
        observed = clock.snapshot_and_reset()
        require(set(observed['phases']) == set(LABELS)
                and observed['phases']['native_and_commit']['raised'] == 1
                and observed['phases']['metadata_binding']['nested_calls'] == 1
                and observed['existing_packet_preparation_seconds']['count'] == 20,
                'Fixed phases/counts/exception/nested overlap and separate original packet scalar')
        require(len(canonical(observed)) < 16384, 'Actual bounded fixed scalar row')
        require(len(canonical({'main_path': observed, 'binary': stats})) < 16384,
                'Actual combined new nested original20-row scalar record bound')
        case('actual_inline_call_order_args_return_exception_nested_and_fixed_bounds', observed)
        del events, argument, returned, result
        for fail_clocks in (False, True):
            c = MainPathScalarAttribution(wall_clock=bad_clock if fail_clocks else time.perf_counter,
                                          thread_clock=bad_clock if fail_clocks else time.thread_time)
            fail = RuntimeError('METHOD_STUB original failure with clock failure')
            calls = [0]
            stamp = c.start('packet_take')
            try:
                calls[0] += 1
                raise fail
            except RuntimeError as exc:
                require(exc is fail and calls[0] == 1, 'No swallowed/duplicated original call or masked exception')
            finally:
                c.finish(stamp, raised=True)
            scalar = c.snapshot_and_reset()
            require(scalar['phases']['packet_take']['raised'] == 1, 'One original raised observation')
            case('inline_original_exception_identity_clockfail_' + str(fail_clocks), scalar)
        payload = WeakList([1])
        ref = weakref.ref(payload)
        c = MainPathScalarAttribution()
        stamp = c.start('archive_drop_release')
        del payload
        c.finish(stamp, raised=False)
        require(ref() is None, 'Clock receipt does not retain packet/payload')
        case('inline_stamp_no_packet_or_frame_retention')
        for name in ('serialization', 'fsync'):
            c = MainPathScalarAttribution()
            stamp = c.start('progress_emit')
            try:
                if name == 'serialization':
                    canonical({'invalid': float('nan')})
                else:
                    with (out / 'owned_fsync_failure.tmp').open('xb') as file:
                        file.write(b'owned scalar fixture\n')
                        file.flush()
                        raise OSError('METHOD_STUB fsync failed before durable publication')
            except (ValueError, OSError) as exc:
                observed_error = {'type': type(exc).__name__, 'message': str(exc)}
            else:
                raise AssertionError('Expected original write/serialization failure')
            finally:
                c.finish(stamp, raised=True)
            require(c.snapshot_and_reset()['phases']['progress_emit']['raised'] == 1, 'Original failure observed exactly once')
            case('original_' + name + '_failure_scalar_passthrough', observed_error)

        # One frozen ABBA workload, no rate-selected repeats or trial tuning.
        # Each trial has20 original updates *3 full-plan checks, plus the exact
        # candidate's FIRST original canonical baseline. No production data.
        for trial_index, mode in enumerate(('original', 'candidate', 'candidate', 'original')):
            p = synthetic_plan(17)
            raw_bytes = len(canonical(p))
            require(1500000 <= raw_bytes <= 3500000, 'Realistic bounded original20-plan-shaped synthetic metadata')
            ready = threading.Event()
            begin_worker = threading.Event()
            worker_result = {}
            def scalar_worker():
                ready.set()
                begin_worker.wait(5)
                wall0, cpu0 = time.perf_counter(), time.thread_time()
                checksum = hashlib.sha256()
                for index in range(4000):
                    row = {'ordinal': index, 'source_id': 'METHOD_STUB_scalar_' + str(index),
                           'labels': list(range(40)), 'geometry': {'samples': 210288, 'T': 165}}
                    checksum.update(canonical(row))
                worker_result.update(wall_seconds=time.perf_counter() - wall0,
                                     thread_CPU_seconds=time.thread_time() - cpu0,
                                     fixed_rows=4000, checksum=checksum.hexdigest())
            worker = threading.Thread(target=scalar_worker, name='METHOD_STUB_fixed_scalar_metadata')
            workers.append(worker)
            worker.start()
            require(ready.wait(5), 'Fixed concurrent metadata thread ready')
            cache = witness() if mode == 'candidate' else None
            reference = digest(p)
            begin_worker.set()
            wall0, cpu0 = time.perf_counter(), time.thread_time()
            values = [cache.digest(p) if cache is not None else digest(p) for _ in range(60)]
            loop_wall, loop_cpu = time.perf_counter() - wall0, time.thread_time() - cpu0
            require(all(value == reference for value in values), 'All actual60 trial checks original equivalent')
            worker.join(10)
            require(not worker.is_alive() and worker_result.get('fixed_rows') == 4000, 'One fixed worker completes, no extra work')
            cost = None if cache is None else cache.snapshot_and_reset()
            if cache is not None:
                require(cost['counts']['baseline_eligible'] == 1 and cost['counts']['binary_equal'] == 59,
                        'Benchmark candidate really uses strict fast path, not hidden fallback')
                cache.close()
            report['benchmarks'].append({'trial_index': trial_index, 'predeclared_ABBA_mode': mode,
                'original_updates': 20, 'original_boundary_calls': 60, 'canonical_plan_bytes': raw_bytes,
                'main_loop_wall_seconds': loop_wall, 'main_loop_thread_CPU_seconds': loop_cpu,
                'concurrent_worker': worker_result, 'candidate_costs': cost})
        old = [row for row in report['benchmarks'] if row['predeclared_ABBA_mode'] == 'original']
        new = [row for row in report['benchmarks'] if row['predeclared_ABBA_mode'] == 'candidate']
        old_wall = sum(row['main_loop_wall_seconds'] for row in old)
        new_wall = sum(row['main_loop_wall_seconds'] for row in new)
        old_cpu = sum(row['main_loop_thread_CPU_seconds'] for row in old)
        new_cpu = sum(row['main_loop_thread_CPU_seconds'] for row in new)
        report['CPU_performance_supports_activation'] = new_wall < .98 * old_wall and new_cpu < .98 * old_cpu
        report['predeclared_performance_comparison'] = {'old_two_trial_wall_seconds': old_wall,
            'new_two_trial_wall_seconds': new_wall, 'old_two_trial_thread_CPU_seconds': old_cpu,
            'new_two_trial_thread_CPU_seconds': new_cpu,
            'requires_at_least2percent_better_both_wall_and_threadCPU': True,
            'slower_or_unclear_candidate_must_not_promote': True,
            'CPU_only_not_production_throughput_or_recurrent_cause_proof': True}
        require(random.getstate() == global_rng_before, 'Actual isolated Python global RNG unchanged')
        require(sys.getrecursionlimit() == report['recursion_limit_before'], 'Original recursion policy unchanged')
        require(all(not worker.is_alive() for worker in workers), 'Every owned fixture thread genuinely joined')
        require(binary._StrictPickler is original_pickler, 'All METHOD_STUB factory aliases restored')
        report['input_pins_after'] = [pin(item['path']) for item in scope['input_pins']]
        require(report['input_pins_after'] == before, 'Every frozen input unchanged')
        report['actual_global_RNG_unchanged'] = True
        report['all_fixture_threads_joined'] = True
        report['passed'] = True
    except BaseException as exc:
        report['error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        raise
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        report['wall_seconds'] = time.monotonic() - began
        report['fixture_threads_remaining'] = [thread.name for thread in workers if thread.is_alive()]
        result = write(out / 'report.json', report)
        print(json.dumps({'report': result, 'passed': report['passed'],
                          'cases': len(report['cases']),
                          'CPU_performance_supports_activation': report['CPU_performance_supports_activation']}), flush=True)


if __name__ == '__main__':
    main()
