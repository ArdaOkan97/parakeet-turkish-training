"""INACTIVE one-first-periodic-DEV refresh; no operation at module import.

This is a changed GC lifecycle hypothesis, not a speed/fit/retention guarantee.
The production caller must first genuinely join/drop the empty scalar worker,
persist the full checkpoint/DEV/CPU closure, and drop declared dead aliases.
Scientific capture is a separately reviewed caller callback. In particular,
identity/version equality alone must never be described as tensor-byte equality.
"""
import gc
import hashlib
import json
import sys
import threading
import time
import types

ORIGINAL_COLLECT = gc.collect
ORIGINAL_UNFREEZE = gc.unfreeze
ORIGINAL_FREEZE = gc.freeze
BOOKKEEPING_RESERVE_BYTES = 16 * 1024**2
FULL_COMMIT_THRESHOLD_BYTES = 4518435456
FULL_PHYSICAL_THRESHOLD_BYTES = 1768435456
MAX_CAPTURE_BYTES = 4 * 1024**2
POLICY_KIND = 'ASR18_one_first_periodic_DEV_quiescent_gc_refresh_policy_v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def checked_builtin(value, name):
    require(isinstance(value, types.BuiltinFunctionType)
            and value.__name__ == name and value.__module__ == 'gc',
            'Captured actual original gc builtin ' + name)


def check_barrier(barrier, expected_update):
    require(type(barrier) is dict and barrier.get('kind') == 'ASR18_actual_first_periodic_DEV_refresh_barrier_v1',
            'Exact explicit periodic DEV barrier, never startup/fullpass substitution')
    require(type(barrier.get('updates')) is int and barrier['updates'] == expected_update
            and expected_update >= 4000 and expected_update % 4000 == 0,
            'Only the first genuine scheduled periodic DEV after this restoration')
    required_true = ('genuine_periodic_DEV_returned_and_recorded', 'checkpoint_full_persistence_same_frontier',
                     'feature_CPU_actual_wait0_joined_pipes_and_both_credit_sets_closed',
                     'DEV_preservation_and_phase_monitor_join_succeeded', 'optimizer_idle_unpoisoned',
                     'scalar_empty_joined_closed_errorNone_worker_DB_and_locks_closed',
                     'old_scalar_object_and_frame_dropped', 'parent_original_context_and_locks_continuous',
                     'no_live_packet_or_arrays', 'checkpoint_writer_and_completed_aliases_dropped',
                     'queued_original_same_object_and_pin', 'main_Monitor_active')
    require(set(barrier) == {'kind', 'updates', *required_true, 'stream_is_None',
                            'future_ticket_is_None', 'resolver_is_original', 'capture_contract'},
            'Complete bounded barrier fields without unknown closure assumptions')
    require(all(barrier[k] is True for k in required_true)
            and barrier['stream_is_None'] is True and barrier['future_ticket_is_None'] is True
            and barrier['resolver_is_original'] is True,
            'True joined/dropped idle boundary; no live worker, packet, writer, poison or unknown closure')
    require(barrier['capture_contract'] == 'original_bounded_live_fingerprint_and_identity_manifest_v1',
            'Independent original full-state fingerprint required, not version-only substitution')


class RefreshSequence:
    """Operation state machine; injected operations are METHOD_STUB controls only.

    Production uses OnePostDEVRefresh with captured real builtins below. No
    exception/frame history is retained; failure metadata is only scalars.
    """
    def __init__(self, restored_updates):
        require(type(restored_updates) is int and restored_updates >= 0, 'Exact restored optimizer frontier')
        self.expected_update = (restored_updates // 4000 + 1) * 4000
        self.attempted = False
        self.completed = False
        self.operations_completed = []
        self.failed_phase = None

    def execute(self, barrier, *, capture, compare, guard, admit, operations, emit):
        require(threading.current_thread() is threading.main_thread(), 'Refresh authority is MainThread only')
        require(not self.attempted and not self.completed, 'One event only; no fallback or second attempt')
        check_barrier(barrier, self.expected_update)
        require(set(operations) == {'unfreeze', 'collect', 'freeze', 'get_freeze_count'}
                and all(callable(v) for v in operations.values()), 'Exact operation set')
        require(all(callable(v) for v in (capture, compare, guard, admit, emit)), 'Reviewed bounded caller callbacks')
        self.attempted = True
        report = {'kind': 'ASR18_actual_one_postDEV_gc_refresh_event_v1', 'passed': False,
                  'optimizer_updates': self.expected_update, 'policy_collect_is_additional': True,
                  'policy_collect_counted_as_original_per_update_observation': False,
                  'bookkeeping_reserve_bytes': BOOKKEEPING_RESERVE_BYTES, 'host_credit_bytes': 0,
                  'operations_completed': self.operations_completed,
                  'retention_of_opaque_framework_cycles_excluded': False,
                  'fresh_scalar_worker_created_only_after_event': True}
        began = time.monotonic()
        try:
            self.failed_phase = 'guard_and_additional_prospective_admission'
            guard()
            report['admission'] = admit(BOOKKEEPING_RESERVE_BYTES)
            require(type(report['admission']) is dict and report['admission'].get('passed') is True,
                    'Actual full original thresholds plus separate bookkeeping reserve before event')
            self.failed_phase = 'capture_before'
            before = capture()
            before_bytes = canonical(before)
            require(len(before_bytes) <= MAX_CAPTURE_BYTES, 'Bounded scalar/hash capture; never trim state')
            report['before_capture_sha256'] = hashlib.sha256(before_bytes).hexdigest()
            report['before_frozen_objects'] = operations['get_freeze_count']()
            require(type(report['before_frozen_objects']) is int and report['before_frozen_objects'] > 0,
                    'Original startup-frozen generation must be present')
            guard()
            self.failed_phase = 'unfreeze'
            operations['unfreeze']()
            self.operations_completed.append('unfreeze')
            guard()
            self.failed_phase = 'captured_original_full_collect'
            report['policy_collect_objects'] = operations['collect']()
            self.operations_completed.append('captured_original_full_collect')
            guard()
            self.failed_phase = 'freeze'
            operations['freeze']()
            self.operations_completed.append('freeze')
            guard()
            report['after_frozen_objects'] = operations['get_freeze_count']()
            require(type(report['after_frozen_objects']) is int and report['after_frozen_objects'] > 0,
                    'Refreshed frozen generation exists; no repeated freeze fallback')
            self.failed_phase = 'capture_after_and_compare'
            after = capture()
            after_bytes = canonical(after)
            compare(before, after)
            require(len(after_bytes) <= MAX_CAPTURE_BYTES and after_bytes == before_bytes,
                    'Exact independent scientific fingerprint and all identity/frontier invariants')
            report['after_capture_sha256'] = hashlib.sha256(after_bytes).hexdigest()
            report['before_after_capture_exact'] = True
            del before, after, before_bytes, after_bytes
            guard()
            self.failed_phase = 'persist_success_receipt'
            report['wall_seconds'] = time.monotonic() - began
            report['passed'] = True
            emit(report)
            self.completed = True
            self.failed_phase = None
            return report
        except BaseException as error:
            report['passed'] = False
            report['failed_phase'] = self.failed_phase
            report['error_type'] = type(error).__name__
            report['error'] = str(error)
            report['wall_seconds'] = time.monotonic() - began
            report['original_teardown_still_authoritative'] = True
            try:
                emit(report)
            except BaseException:
                pass  # Never replace the original guard/callback exception.
            raise


class OnePostDEVRefresh(RefreshSequence):
    """Production binding to actual builtins; construction/import does no GC work."""
    def execute_real(self, barrier, *, capture, compare, guard, admit, original_collect_observer, emit):
        for value, name in ((ORIGINAL_COLLECT, 'collect'), (ORIGINAL_UNFREEZE, 'unfreeze'),
                            (ORIGINAL_FREEZE, 'freeze')):
            checked_builtin(value, name)
        module = sys.modules.get('asr18_observe_original_gc_collect_v2')
        require(module is not None and type(original_collect_observer) is module.CollectScalarObservation
                and original_collect_observer.original is ORIGINAL_COLLECT
                and gc.collect is original_collect_observer
                and gc.unfreeze is ORIGINAL_UNFREEZE and gc.freeze is ORIGINAL_FREEZE,
                'Original active collect observation and actual unchanged freeze/unfreeze builtins')
        return self.execute(barrier, capture=capture, compare=compare, guard=guard, admit=admit, emit=emit,
                            operations={'unfreeze': ORIGINAL_UNFREEZE, 'collect': ORIGINAL_COLLECT,
                                        'freeze': ORIGINAL_FREEZE, 'get_freeze_count': gc.get_freeze_count})
