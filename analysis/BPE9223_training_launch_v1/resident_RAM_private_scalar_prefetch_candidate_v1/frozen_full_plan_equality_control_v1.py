"""INACTIVE CPU-only control source; not executed or admitted by this file.

Constructs the genuine original source context in its own future CPU fixture.
No training context/DB is changed. All fixture output must be a fresh E folder.
"""
from copy import deepcopy
import json, os, random, sys, threading, time
from pathlib import Path

P = Path(r'C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
S = P/'outputs/parakeet_turkish'
sys.path.insert(0, str(S))
from atomic_io_retry_v1 import read_json, write_json
from bpe_broad_audio_v1 import record, require
from bpe_broad_production_state_v1 import canonical, digest
from bpe_final_DATA_schedule_context_v2 import normalize_schedule_state
from bpe_production_window_coordinator_v4 import make_plan
from asr18_BPE9223_lifecycle_v3 import next_boundary_request
from asr18_BPE9223_source_context_v2 import FullRunSourceContext9223
from asr18_BPE9223_private_scalar_prefetch_v1 import PrivateScalarPrefetch, ADDITIONAL_PRIVATE_ADMISSION_BYTES


def run(configuration_pin, frozen_pairs, output):
    """Proposed180s/512MiB fixture; source review must precede actual invocation."""
    import psutil
    from resource_watchdog_v2 import sample_memory
    require(sys.flags.utf8_mode == 1 and os.environ.get('PYTHONUTF8') == '1' and
            os.environ.get('CUDA_VISIBLE_DEVICES') == '-1', 'CPU-only upfront UTF8')
    output = Path(output).resolve()
    require(output.is_relative_to(Path(r'E:/gpu_training_scratchpad')) and not output.exists(), 'Fresh E fixture only')
    output.mkdir()
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    began = time.monotonic()
    peak, samples = 0, 0
    def guard():
        nonlocal peak, samples
        require(time.monotonic()-began < 180, 'Bounded180s metadata-only control')
        s = sample_memory(os.getpid(), 'private_scalar_prefetch_frozen_plan_equality')
        samples += 1
        peak = max(peak, s['worker_private_memory']['private_bytes'])
        require(peak <= 512*1024**2 and s['system_memory']['commit_available_bytes'] >= 2000000000 and
                s['system_memory']['physical_available_bytes'] >= 1500000000, 'Actual private/headroom caps unchanged')
        return s
    def admit(name, increment):
        s = guard()
        require(increment == ADDITIONAL_PRIVATE_ADMISSION_BYTES and
                s['system_memory']['commit_available_bytes']-increment >= 2000000000 and
                s['system_memory']['physical_available_bytes']-increment >= 1500000000,
                'Full256MiB additional host allowance, no unmeasured freed-memory credit')
        return {'name': name, 'increment': increment, 'sample': s}
    require(record(configuration_pin['path']) == configuration_pin, 'Exact current immutable configuration')
    c = read_json(configuration_pin['path'])
    original_rng = random.getstate()
    coordinator_pin = record(S/'bpe_production_window_coordinator_v4.py')
    context = FullRunSourceContext9223(c['planning_arguments'], guard=guard)
    prefetch = None
    results = []
    try:
        prefetch = PrivateScalarPrefetch(context, guard=guard, admit=admit)
        for previous_pin, expected_pin in frozen_pairs:
            require(record(previous_pin['path']) == previous_pin and record(expected_pin['path']) == expected_pin,
                    'Immutable existing adjacent complete plans')
            previous, expected = read_json(previous_pin['path']), read_json(expected_pin['path'])
            before = normalize_schedule_state(previous['final_private_planning_sampler'])
            require(digest(before) == expected['initial_sampler_sha256'] and len(expected['updates']) == 20,
                    'Real prior plan yields exact next frozen sampler frontier')
            schedule = context.new_schedule()
            schedule.load_state_dict(deepcopy(before))
            action = next_boundary_request(schedule.state_dict(), 0)
            require(action['kind'] == 'bounded_training_window' and action['maximum_updates'] == 20,
                    'Past ordinary broad plan, no artificial fullpass/periodic bypass')
            started = time.monotonic()
            old = context.plan(schedule, completed_boundaries=0)
            original_seconds = time.monotonic()-started
            require(canonical(old) == canonical(expected), 'Full original plan equals genuine frozen plan, including replay/RNG/targets')
            ticket = prefetch.submit(schedule.state_dict(), maximum_updates=20)
            prepared = prefetch.take(ticket, wait=True)
            started = time.monotonic()
            with prefetch.planning_rows(prepared, schedule):
                new = context.plan(schedule, completed_boundaries=0)
            main_adoption_seconds = time.monotonic()-started
            require(canonical(new) == canonical(old) == canonical(expected) and schedule.state_dict() == before,
                    'Every full row, target, payload bound, both cursor/replay RNG states and transaction digest byteequivalent')
            results.append({'previous_plan': previous_pin, 'frozen_plan': expected_pin,
                            'full_canonical_plan_sha256': digest(new), 'full_plan_equal': True,
                            'original_main_planning_seconds': original_seconds,
                            'worker_prefetch_seconds': prepared['preparation_seconds'],
                            'main_adoption_planning_seconds': main_adoption_seconds,
                            'no_pipeline_speed_claim': True})
            # Negative: exact future-state adoption and original MainThread gate stay strict.
            wrong = deepcopy(before)
            wrong['broad_cursor']['draws'] += 14
            wrong_schedule = context.new_schedule()
            try:
                wrong_schedule.load_state_dict(wrong)
            except ValueError:
                pass  # Original schedule already rejects replay/frontier mismatch.
            else:
                raise AssertionError('Original schedule accepted mismatched actual replay count')
            errors = []
            def forbidden_original_planner():
                try:
                    make_plan(None, clone_schedule=None, resolve_identity=None, resolve_rows=None,
                              feature_bytes_upper_bound=None, finalized_population=None, verify_finalized_population=None)
                except ValueError as e:
                    errors.append(str(e))
            t = threading.Thread(target=forbidden_original_planner)
            t.start(); t.join(3)
            require(not t.is_alive() and errors == ['Serial window/cursor use only'], 'Unchanged original coordinator rejects worker planning')
            del prepared, new, old, schedule
        closed = prefetch.close()
        prefetch = None
        require(random.getstate() == original_rng and record(coordinator_pin['path']) == coordinator_pin and
                'torch' not in sys.modules and 'numpy' not in sys.modules, 'Original RNG/source guards, no model/audio/GPU')
        report = {'kind': 'ASR18_inactive_private_scalar_prefetch_full_frozen_plan_control_v1', 'passed': True,
                  'source': record(__file__), 'helper': record(S/'asr18_BPE9223_private_scalar_prefetch_v1.py'),
                  'configuration': configuration_pin, 'original_coordinator': coordinator_pin,
                  'results': results, 'worker_closure': closed, 'samples': samples, 'private_peak_bytes': peak,
                  'actual_authoritative_sampler_native_or_journal_advance': False,
                  'production_source_or_training_process_changed': False,
                  'scope': 'Exact frozen-plan equivalence only; shared OS cache and no running-native overlap measurement',
                  'elapsed_seconds': time.monotonic()-began}
        write_json(output/'report.json', report)
        return record(output/'report.json')
    finally:
        try:
            if prefetch is not None:
                prefetch.close(discard_unconsumed=True)
        finally:
            context.close()
