"""INACTIVE finite proof: actual original row/propose/cursor/replay, fixture DBs.

Scientific population admission, replay metadata and numerical commit are honest
METHOD_STUBs. Actual original read-only SQLite, deny-write handles, signatures,
row/declaration/cache body, CompletePassCursor, CarriedReplay/BucketCycler and
ScheduledTransaction are exercised. No production data, audio or GPU is read.
"""
import argparse
from collections import OrderedDict
from copy import deepcopy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sqlite3
import struct
import sys
import threading
import time
import traceback
import weakref

import psutil

P = Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
S = P / 'outputs/parakeet_turkish'
W = Path(__file__).parent
CAP = 512 * 1024**2


def require(value, message):
    if not value:
        raise ValueError(message)


def pin(path):
    path = Path(path)
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def write(path, value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode() + b'\n'
    require(len(raw) < 8 * 1024**2, 'Bounded finite metadata output')
    with Path(path).open('xb') as file:
        file.write(raw);file.flush();os.fsync(file.fileno())
    return pin(path)


def identity():
    me = psutil.Process()
    return {'pid': me.pid, 'create_time': me.create_time(), 'ppid': me.ppid()}


def main():
    a = argparse.ArgumentParser();a.add_argument('--ROOT-review', required=True);a.add_argument('--output', required=True)
    args = a.parse_args();began = time.monotonic()
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    output = Path(args.output)
    require(output.drive.lower() == 'e:' and not output.exists(), 'Fresh E-only finite namespace')
    output.mkdir()
    scope = json.loads(Path(args.ROOT_review).read_text(encoding='utf-8-sig'))
    require(scope['kind'] == 'ROOT_current_plan_identity_view_finite_CPU_exact_ONCE_execution_scope_v1'
        and scope['accepted'] is True and scope['reviewer'] == '/root'
        and scope['fixture_source'] == pin(__file__), 'Separate exact finite CPU scope')
    sys.path.insert(0, str(S))
    from resource_watchdog_v2 import sample_memory
    report = {'kind': 'ASR18_current_plan_identity_view_actual_CPU_v1', 'passed': False,
              'fixture_identity': identity(), 'cases': [], 'trials': [], 'guard_error': None,
              'scientific_population_and_replay_rows_and_commit_METHOD_STUB': True,
              'real_production_or_audio_or_GPU_operation': False}
    contexts = []

    def guard(label, initial=False):
        value = sample_memory(os.getpid(), label)
        if initial:
            write(output / 'actual_initial_numeric_before_predicate.json', {'sample': value, 'identity': identity(),
                'commit_required': 5130803840, 'physical_required': 2380803840})
        m = value['system_memory']
        require(m['commit_available_bytes'] >= (5130803840 if initial else 2343932928)
                and m['physical_available_bytes'] >= (2380803840 if initial else 1843932928)
                and value['worker_private_memory']['private_bytes'] < CAP
                and shutil.disk_usage('E:/').free >= 100000000000 and time.monotonic() - began < 100,
                'Original full guards plus separately reserved64+8MiB, no freed credit')
        return value

    def case(name, **extra):
        report['cases'].append({'name': name, 'passed': True, **extra})

    try:
        report['initial_sample'] = guard('identity_fixture_initial', True)
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '-1' and all(os.environ.get(k) == '1' for k in
            ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')), 'CPU-hidden caps1')
        require(os.environ.get('TORCH_EXTENSIONS_DIR', '').lower().startswith('e:'), 'E storage environment')
        before_pins = [pin(p['path']) for p in scope['input_pins']]
        require(before_pins == scope['input_pins'], 'All frozen source/metadata inputs')
        report['input_pins_before'] = before_pins
        # Only now import the reviewed candidate and original pure metadata code.
        import asr18_BPE9223_source_context_v2 as ctxmod
        from bpe_broad_production_state_v1 import CarriedReplay, canonical, digest, LANGUAGES
        from bpe_replay_bucket_no_runtime_v1 import BucketCycler
        from bpe_production_window_coordinator_v4 import make_plan
        from bpe_training_schedule_v1 import WEAK
        spec = importlib.util.spec_from_file_location('identity_candidate', scope['helper_source']['path'])
        helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        fixture_rng = random.getstate()
        replay_rows = [{'source_id': 'METHOD_STUB_' + lang + '_' + str(i), 'language': lang,
                        'encoder_frames': 165, 'target_tokens': 66, 'augmentation_role': 'replay'}
                       for lang in LANGUAGES for i in range(8)]

        def replay():
            rng, pt = random.Random(781), random.Random(782)
            streams = {lang: BucketCycler([r for r in replay_rows if r['language'] == lang], pt if lang == 'pt' else rng, bucket_size=4)
                       for lang in LANGUAGES}
            snap = {'source_step': 64250, 'source_identity': {'METHOD_STUB': 'original lineage interface'},
                'replay_language_counts': {l: 0 for l in LANGUAGES}, 'replay_rng': rng.getstate(),
                'Portuguese_rng': pt.getstate(), 'replay': {l: s.state_dict() for l, s in streams.items()}}
            return CarriedReplay(replay_rows, snap, BucketCycler)

        def schedule():
            return ctxmod.TrainingSchedule9223(broad_identity={'METHOD_STUB': 'complete sparse fixture'},
                cleaner_identity={'METHOD_STUB': 'never opened'}, source_turkish_rng=random.Random(783).getstate(), replay=replay())

        sched = schedule();cursor = deepcopy(sched.broad);ordinals = []
        for _ in range(20):
            draws = cursor.preview(14);ordinals.extend(d['ordinal'] for d in draws);cursor.commit(draws)
        require(len(set(ordinals)) == 280, 'Actual original bijective current20 prefix')
        data = output / 'fixture_metadata';data.mkdir()
        dbs = [data / n for n in ('catalog.sqlite', 'labels.sqlite', 'geometry.sqlite')]
        connections = [sqlite3.connect(p) for p in dbs]
        try:
            connections[0].execute('CREATE TABLE clips(ordinal INTEGER PRIMARY KEY,dataset TEXT,unique_id TEXT,source_id TEXT,start REAL,end REAL,text TEXT,duration REAL)')
            connections[1].execute('CREATE TABLE labels(ordinal INTEGER PRIMARY KEY,target_ids_u32 BLOB,target_tokens INTEGER,text_sha256 TEXT)')
            connections[2].execute('CREATE TABLE geometry(ordinal INTEGER PRIMARY KEY,source_component INTEGER,native_sr INTEGER,native_samples INTEGER,clean16k_samples INTEGER,mel_frames INTEGER,encoder_frames INTEGER,target_tokens INTEGER,status TEXT)')
            for n, ordinal in enumerate(ordinals):
                text = 'Özgün tam kaynak ve hedef ' + str(ordinal) + '. '
                u = 66 + n % 35
                connections[0].execute('INSERT INTO clips VALUES(?,?,?,?,?,?,?,?)', (ordinal, 'alania_yodas3', 'METHOD_STUB_clip_' + str(ordinal), 'METHOD_STUB_recording_' + str(ordinal), 0., 13.2, text, 13.2))
                connections[1].execute('INSERT INTO labels VALUES(?,?,?,?)', (ordinal, struct.pack('<' + 'I' * u, *range(1, u + 1)), u, hashlib.sha256(text.encode()).hexdigest()))
                connections[2].execute('INSERT INTO geometry VALUES(?,?,?,?,?,?,?,?,?)', (ordinal, 0, 48000, 633600, 211200, 1320, 165, u, 'declared'))
            for c in connections:c.commit()
        finally:
            for c in connections:c.close()
        bindings = {k: pin(p) for k, p in zip(helper.KEYS, dbs)}
        bindings['declared_geometry_index']['mtime_ns'] = dbs[2].stat().st_mtime_ns
        validated = {'kind': ctxmod.KIND, 'passed': True, 'complete': True, 'bindings': bindings}
        count = {'guard': 0, 'signature': 0, 'sql': 0}
        def original_guard():count['guard'] += 1

        def context():
            c = ctxmod.FullRunSourceContext9223.__new__(ctxmod.FullRunSourceContext9223)
            c.closed = False;c.validated = deepcopy(validated);c.guard = original_guard
            c.resolver = ctxmod.FinalBroadRows9223(c.validated, guard=original_guard)
            c.cleaner_rows = None  # METHOD_STUB, never touched by broad eligible view.
            contexts.append(c)
            for db in [*c.resolver.connections, c.resolver.geometry.connection]:
                db.set_trace_callback(lambda sql: count.__setitem__('sql', count['sql'] + int(sql.startswith('SELECT'))))
            return c

        c = context();population = write(data / 'population_METHOD_STUB.json', {'complete': True, 'METHOD_STUB_sparse_rows': 280})
        def resolve_rows(phase, draws, carried):
            return [c.resolver.row(d['ordinal']) for d in draws] + [deepcopy(r) for r in carried]
        plan = make_plan(sched, clone_schedule=schedule, resolve_identity=c.resolve_identity, resolve_rows=resolve_rows,
            feature_bytes_upper_bound=lambda phase, rows: 4096, finalized_population=population,
            verify_finalized_population=lambda p: p == population, maximum_updates=20)
        fresh = digest(plan);authority = helper.OriginalSourceAuthority(c);entry = plan['updates'][0]
        plan_before = canonical(plan);state_before = sched.state_dict();old_cache = c.resolver.cache
        tx = sched.propose(optimizer_updates=0, resolve_TR=c.resolve_identity)
        expected = {'identity': tx.identity_digest, 'draws': deepcopy(tx.TR_draws), 'TR': deepcopy(tx.TR_identity), 'after_replay': deepcopy(tx.after_replay)}
        tx.abandon();del tx
        with helper.current_plan_identity_view(c, sched, plan, 0, entry, authority=authority, fresh_plan_sha256=fresh, packet_plan_sha256=fresh, optimizer_updates=0) as eligible:
            require(eligible, 'Exact original admitted current entry is eligible')
            view_ref = weakref.ref(c.resolver.cache)
            tx = sched.propose(optimizer_updates=0, resolve_TR=c.resolve_identity)
            require({'identity': tx.identity_digest, 'draws': tx.TR_draws, 'TR': tx.TR_identity, 'after_replay': tx.after_replay} == expected, 'Actual original propose/cursor/replay equivalence')
            tx.abandon();del tx
        require(c.resolver.cache is old_cache and view_ref() is None and canonical(plan) == plan_before and sched.state_dict() == state_before and random.getstate() == fixture_rng, 'Original cache restored, view released, all inputs/RNG unchanged')
        case('actual_original_row_propose_cursor_replay_RNG_fullrow_equivalence_and_view_lifetime')
        first_ordinal = entry['TR_draws'][0]['ordinal']
        with helper.current_plan_identity_view(c, sched, plan, 0, entry, authority=authority, fresh_plan_sha256=fresh, packet_plan_sha256=fresh, optimizer_updates=0) as eligible:
            got = c.resolver.row(first_ordinal);got['tr_metadata_labels'][0] = 777
            require(entry['rows'][0]['tr_metadata_labels'][0] == 1, 'Original returned full-row deepcopy')
        case('fullrow_deepcopy_no_current_plan_mutation')
        # Actual source aliases stay bound to the same original signature
        # function, with scalar observers only for this finite call-count case.
        import bpe_declared_geometry_index_v1 as geometry_module
        sig0, gsig0 = ctxmod.signature, geometry_module.signature
        signature_calls = {'row': 0, 'geometry': 0}
        def row_signature(path):
            signature_calls['row'] += 1
            return sig0(path)
        def geometry_signature(path):
            signature_calls['geometry'] += 1
            return gsig0(path)
        try:
            ctxmod.signature = row_signature;geometry_module.signature = geometry_signature
            c.resolver.cache.clear();old_counts = dict(count)
            original_tx = sched.propose(optimizer_updates=0, resolve_TR=c.resolve_identity);original_tx.abandon()
            miss = {k: count[k] - old_counts[k] for k in ('guard', 'sql')}
            miss.update(signature_calls)
            require(miss == {'guard': 28, 'sql': 42, 'row': 28, 'geometry': 14}, 'Actual original14 miss counts')
            signature_calls.update(row=0, geometry=0);old_counts = dict(count)
            with helper.current_plan_identity_view(c, sched, plan, 0, entry, authority=authority, fresh_plan_sha256=fresh, packet_plan_sha256=fresh, optimizer_updates=0) as eligible:
                actual_tx = sched.propose(optimizer_updates=0, resolve_TR=c.resolve_identity);actual_tx.abandon()
            hit = {k: count[k] - old_counts[k] for k in ('guard', 'sql')};hit.update(signature_calls)
            require(hit == {'guard': 14, 'sql': 0, 'row': 28, 'geometry': 0}, 'Actual original14 cache-hit guard/twofresh signature counts')
        finally:
            ctxmod.signature = sig0;geometry_module.signature = gsig0
        case('actual_original_percall_guard_twofresh_signature_and_SQL_counts', original_miss=miss, temporary_cache_hit=hit,
             skipped_original_miss_geometry_and_SQL_opportunities_explicit=True)
        raw_bindings = deepcopy(c.validated['bindings'])
        try:
            for k in helper.KEYS:c.validated['bindings'][k]['path'] = c.validated['bindings'][k]['path'].replace(chr(92), '/')
            slash_authority = helper.OriginalSourceAuthority(c)
            require(slash_authority.agrees(c), 'Original normalized path ownership supports admitted slash serialization')
        finally:c.validated['bindings'] = raw_bindings
        case('slash_serialization_normalized_actual_lock_ownership_rawpins_preserved')
        for name, change in (
            ('wrong_plan_sha', lambda p: None),
            ('wrong_index', lambda p: p['updates'][0].__setitem__('local_update_index', 1)),
            ('wrong_phase', lambda p: p.__setitem__('phase', 'cleaner_data_fine_tuning')),
            ('wrong_ordinal', lambda p: p['updates'][0]['TR_draws'][0].__setitem__('ordinal', 1)),
            ('wrong_source_identity', lambda p: p['updates'][0]['TR_identity'][0].__setitem__('source_id', 'WRONG')),
            ('wrong_before_state', lambda p: p['updates'][0].__setitem__('before_sampler_sha256', '0' * 64)),
            ('wrong_order', lambda p: p['updates'][0]['TR_identity'].reverse()),
            ('missing_row', lambda p: p['updates'][0]['rows'].pop(0))):
            bad = deepcopy(plan);change(bad);bf = digest(bad)
            with helper.current_plan_identity_view(c, sched, bad, 0, bad['updates'][0], authority=authority,
                    fresh_plan_sha256=bf, packet_plan_sha256='f' * 64 if name == 'wrong_plan_sha' else bf, optimizer_updates=0) as eligible:
                require(not eligible, 'Unproved current entry delegates original before propose')
                actual = sched.propose(optimizer_updates=0, resolve_TR=c.resolve_identity)
                require(actual.identity_digest == expected['identity'], 'Fallback is one original propose');actual.abandon()
            case(name + '_original_fallback')
        require(not helper._bounded_builtin_tree([0] * 50000) and not helper._bounded_builtin_tree({str(i): 0 for i in range(12000)}), 'Wide frontier rejected before expansion')
        alias = [];require(not helper._bounded_builtin_tree([alias, alias]), 'Shared containers are unsupported')
        cycle = [];cycle.append(cycle);require(not helper._bounded_builtin_tree(cycle), 'Cycle falls back')
        case('large_wide_dict_list_alias_cycle_strict_bound')
        saved_handles = c.resolver.metadata_locks.handles[:]
        c.resolver.metadata_locks.handles.pop()
        with helper.current_plan_identity_view(c, sched, plan, 0, entry, authority=authority, fresh_plan_sha256=fresh, packet_plan_sha256=fresh, optimizer_updates=0) as eligible:require(not eligible, 'Changed continuous lock owner rejects view')
        c.resolver.metadata_locks.handles[:] = saved_handles
        case('wrong_continuous_handle_binding_fallback')
        sentinel = RuntimeError('METHOD_STUB original guard exception identity')
        saved_guard = c.resolver.guard
        guard_calls = [0]
        def fail_guard():
            guard_calls[0] += 1
            if guard_calls[0] == 2:raise sentinel
        c.resolver.guard = fail_guard
        try:
            with helper.current_plan_identity_view(c, sched, plan, 0, entry, authority=authority, fresh_plan_sha256=fresh, packet_plan_sha256=fresh, optimizer_updates=0) as eligible:
                sched.propose(optimizer_updates=0, resolve_TR=c.resolve_identity)
        except RuntimeError as exc:require(exc is sentinel and c.resolver.poisoned and guard_calls[0] == 2, 'Exact original second-lookup exception and poison owner')
        else:raise AssertionError('Original injected exception not propagated')
        c.resolver.guard = saved_guard;c.resolver.poisoned = False
        require(c.resolver.cache is old_cache and sched.state_dict() == state_before, 'Partial propose is not retried and state unchanged')
        case('original_exception_object_poison_partialpropose_no_retry_and_finally_restore')
        # Unknown ordinal during the eligible view uses the same original row
        # method once. Pick a later actual fixture row, so it is a real SQL miss.
        unknown = plan['updates'][1]['TR_draws'][0]['ordinal']
        with helper.current_plan_identity_view(c, sched, plan, 0, entry, authority=authority, fresh_plan_sha256=fresh, packet_plan_sha256=fresh, optimizer_updates=0) as eligible:
            count_before = count['sql'];row = c.resolver.row(unknown)
            require(row['ordinal'] == unknown and count['sql'] == count_before + 3, 'Unexpected lookup delegates original miss once')
        case('unexpected_ordinal_original_SQL_miss_once')
        saved_sig = c.resolver.signatures[0].copy();c.resolver.signatures[0]['mtime_ns'] -= 1
        try:
            c.resolver.row(first_ordinal)
        except ValueError:require(c.resolver.poisoned, 'Original fresh signature mutation poisons')
        else:raise AssertionError('Original source mutation check missing')
        c.resolver.signatures[0] = saved_sig;c.resolver.poisoned = False
        case('actual_original_fresh_signature_mismatch_poison')
        for db in dbs:
            try:
                with db.open('r+b') as f:f.write(b'x')
            except OSError:pass
            else:raise AssertionError('Original actual deny-write handle allowed mutation')
        case('actual_three_original_denywrite_handles')
        # Frozen ABBA x BOTH conditions. Timing includes eligibility/copies,
        # original real signatures and SQLite; no favorable shape selection.
        # Cursor/replay state advances here only in generated fixture metadata.
        for concurrent in (False, True):
            for order, mode in enumerate(('original', 'view', 'view', 'original')):
                guard('identity_trial_before')
                run_schedule = schedule();c.resolver.cache.clear();worker_errors = []
                stop = threading.Event()
                def scalar_worker():
                    try:
                        for _ in range(6000):
                            if stop.is_set():break
                            for db in dbs:ctxmod.signature(db)
                    except BaseException as e:worker_errors.append(str(e))
                worker = threading.Thread(target=scalar_worker, name='METHOD_STUB_fixed_scalar_signature_worker') if concurrent else None
                if worker is not None:worker.start()
                wall, cpu = time.perf_counter(), time.thread_time();n = 0
                try:
                    for i, e in enumerate(plan['updates']):
                        # Fresh existing plan digest cost is common; three
                        # original production call sites are NOT removed.
                        first_plan_digest = digest(plan)
                        second_plan_digest = digest(plan)
                        pf = digest(plan)
                        require(first_plan_digest == second_plan_digest == pf == fresh, 'All THREE original fresh canonical boundary checks remain common')
                        if mode == 'view':
                            with helper.current_plan_identity_view(c, run_schedule, plan, i, e, authority=authority, fresh_plan_sha256=pf, packet_plan_sha256=pf, optimizer_updates=run_schedule.committed_updates()) as eligible:
                                require(eligible, 'All fixed benchmark entries eligible')
                                t = run_schedule.propose(optimizer_updates=run_schedule.committed_updates(), resolve_TR=c.resolve_identity)
                        else:t = run_schedule.propose(optimizer_updates=run_schedule.committed_updates(), resolve_TR=c.resolve_identity)
                        require(t.identity_digest == e['native_transaction_sha256'], 'Every actual original transaction exact')
                        draws, after = deepcopy(t.TR_draws), deepcopy(t.after_replay);t.abandon()
                        run_schedule.active_cursor.commit(draws);run_schedule.replay.load_state_dict(after);n += 1
                finally:
                    elapsed_wall, elapsed_cpu = time.perf_counter() - wall, time.thread_time() - cpu
                    stop.set()
                    if worker is not None:worker.join(timeout=5);require(not worker.is_alive(), 'Owned fixed scalar thread joined')
                require(not worker_errors and n == 20 and run_schedule.state_dict() == plan['final_private_planning_sampler'], 'All original cursor/replay final states and bounded20/scalar complete')
                report['trials'].append({'concurrent_scalar': concurrent, 'order': order, 'mode': mode, 'updates': n,
                    'wall_seconds': elapsed_wall, 'MainThread_CPU_seconds': elapsed_cpu})
        performance = []
        for concurrent in (False, True):
            group = [t for t in report['trials'] if t['concurrent_scalar'] == concurrent]
            old = {k: sum(t[k] for t in group if t['mode'] == 'original') / 2 for k in ('wall_seconds', 'MainThread_CPU_seconds')}
            new = {k: sum(t[k] for t in group if t['mode'] == 'view') / 2 for k in old}
            performance.append({'concurrent_scalar': concurrent, 'original': old, 'view': new,
                'at_least2percent_both': all(new[k] <= .98 * old[k] for k in old)})
        report['performance'] = performance
        report['CPU_performance_supports_activation'] = all(p['at_least2percent_both'] for p in performance)
        require(canonical(plan) == plan_before and random.getstate() == fixture_rng, 'No plan/global RNG mutation across entire proof')
        report['input_pins_after'] = [pin(p['path']) for p in scope['input_pins']]
        require(report['input_pins_after'] == before_pins, 'All source pins unchanged')
        report['actual_fixture_DBs'] = [pin(p) for p in dbs]
        report['original_calls'] = count
        report['final_sample'] = guard('identity_fixture_final')
        report['passed'] = True
    except BaseException as error:
        report['guard_error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
        raise
    finally:
        for c in contexts:
            c.resolver.close();c.closed = True
        report['original_fixture_readers_and_all_lock_handles_closed'] = all(not c.resolver.metadata_locks.handles for c in contexts)
        report['wall_seconds'] = time.monotonic() - began
        report['source'] = pin(__file__)
        actual = write(output / 'report.json', report)
        print(json.dumps({'report': actual, 'passed': report['passed']}), flush=True)


if __name__ == '__main__':main()
