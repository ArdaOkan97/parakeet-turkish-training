"""INACTIVE temporary original-cache-hit owner for a fully checked CURRENT plan.

No source hash is cached and no original row check is replaced. Eligibility is
decided before one authoritative propose; unexpected lookups use its original
row implementation once, including a real miss when necessary. Original LRU
contents are restored, so cache contents intentionally differ during the view.
No decoder, framework, transport or production entry point is imported here.
"""
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
import math
import sys
import threading

from bpe_broad_production_state_v1 import digest
from bpe_training_schedule_v1 import WEAK
from bpe_windows_final_metadata_lock_v1 import FinalMetadataReadLocks
from asr18_BPE9223_source_context_v2 import (FinalBroadRows9223,
    FullRunSourceContext9223, TrainingSchedule9223)

ADDITIONAL_HOST_BYTES = 8 * 1024**2
MAX_OWNED_TREE_BYTES = 1024**2
MAX_NODES = 16384
MAX_DEPTH = 48
KEYS = ('final_catalog', 'final_labels', 'declared_geometry_index')
_ACTIVE = set()  # MainThread-only object IDs, never plan/row/packet references.


def _main():
    return threading.current_thread() is threading.main_thread()


def _bounded_builtin_tree(value):
    """Eligibility only; reject a wide child frontier BEFORE allocating it.

    <=16384 total visited+pending nodes, <=one MiB exact object-size sum,
    <=48 depth and no shared/cyclic containers. Pending tuples and seen IDs
    each stay bounded by MAX_NODES; two full copies/memos and this metadata
    are conservatively covered by the separate eight-MiB reservation.
    """
    pending = [(value, 0)]
    seen = set()
    total = nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        t = type(item)
        if t not in (dict, list, tuple, str, int, float, bool, type(None)):
            return False
        total += sys.getsizeof(item)
        if nodes > MAX_NODES or depth > MAX_DEPTH or total > MAX_OWNED_TREE_BYTES:
            return False
        if t is float and not math.isfinite(item):
            return False
        if t in (dict, list, tuple):
            if id(item) in seen:
                return False  # Unsupported alias/cycle uses original path.
            seen.add(id(item))
            children = 2 * len(item) if t is dict else len(item)
            if nodes + len(pending) + children > MAX_NODES:
                return False  # No enumerate/extend before this frontier gate.
            if t is dict:
                if any(type(k) not in (str, int) for k in item):
                    return False
                pending.extend((k, depth + 1) for k in item)
                pending.extend((v, depth + 1) for v in item.values())
            else:
                pending.extend((v, depth + 1) for v in item)
    return True


class OriginalSourceAuthority:
    """Scalar identity witness of the already full-SHA-admitted main context.

    This is not a new full-file verification. Continuous deny-write handle
    ownership is supplied by the unchanged original constructor and teardown.
    No file, connection, lock or context is closed or reopened by this helper.
    """
    def __init__(self, context):
        if not _main() or type(context) is not FullRunSourceContext9223:
            raise ValueError('Original MainThread source owner required')
        original = context.resolver
        locks = original.metadata_locks
        if type(original) is not FinalBroadRows9223 or type(locks) is not FinalMetadataReadLocks:
            raise ValueError('Exact original row and deny-write lock owners required')
        bindings = context.validated['bindings']
        paths = tuple(str(locks.paths[i]) for i in range(3))
        if len(locks.handles) != 3 or len(locks.paths) != 3 or len(set(locks.handles)) != 3:
            raise ValueError('Three actual continuous original deny-write handles required')
        if paths != tuple(bindings[k]['path'] for k in KEYS):
            raise ValueError('Actual original lock paths and full-SHA bindings required')
        self.context_id = id(context)
        self.original_id = id(original)
        self.locks_id = id(locks)
        self.handles = tuple(locks.handles)
        self.paths = paths
        self.connections = tuple(id(c) for c in original.connections)
        self.geometry_id = id(original.geometry)
        self.population = original.population
        self.signatures = tuple(tuple(sorted(p.items())) for p in original.signatures)
        self.geometry_signature = tuple(sorted(original.geometry.signature.items()))
        self.binding_pins = tuple(tuple((n, bindings[k][n]) for n in ('path', 'bytes', 'sha256')) for k in KEYS)

    def agrees(self, context):
        if not _main() or type(context) is not FullRunSourceContext9223 or id(context) != self.context_id or context.closed:
            return False
        original = context.resolver
        if type(original) is not FinalBroadRows9223 or id(original) != self.original_id or original.poisoned:
            return False
        locks = original.metadata_locks
        bindings = context.validated['bindings']
        return (type(locks) is FinalMetadataReadLocks and id(locks) == self.locks_id
            and tuple(locks.handles) == self.handles and tuple(str(p) for p in locks.paths) == self.paths
            and tuple(id(c) for c in original.connections) == self.connections
            and id(original.geometry) == self.geometry_id and original.population == self.population
            and tuple(tuple(sorted(p.items())) for p in original.signatures) == self.signatures
            and tuple(sorted(original.geometry.signature.items())) == self.geometry_signature
            and tuple(tuple((n, bindings[k][n]) for n in ('path', 'bytes', 'sha256')) for k in KEYS) == self.binding_pins)


def _eligible(context, schedule, plan, index, entry, *, authority,
              fresh_plan_sha256, packet_plan_sha256, optimizer_updates):
    # The driver passes the just-computed THIRD ORIGINAL digest(plan) result.
    # Its original call/count/order/errors remain visible; no fourth digest or
    # mutable plan hash cache is introduced by this helper.
    if not authority.agrees(context) or type(schedule) is not TrainingSchedule9223 or schedule.poisoned:
        return False
    if type(plan) is not dict or type(entry) is not dict or type(index) is not int:
        return False
    if (type(fresh_plan_sha256) is not str or len(fresh_plan_sha256) != 64
            or fresh_plan_sha256 != packet_plan_sha256 or any(c not in '0123456789abcdef' for c in fresh_plan_sha256)):
        return False
    if type(optimizer_updates) is not int or optimizer_updates != schedule.committed_updates():
        return False
    if (plan['kind'] != 'ASR18_phase_aware_provisional_production_window_plan_v1'
            or plan['phase'] != WEAK or schedule.phase != WEAK or not 0 <= index < len(plan['updates'])
            or plan['updates'][index] is not entry or entry['local_update_index'] != index
            or plan['initial_committed_updates'] + index != optimizer_updates
            or entry['global_update'] != optimizer_updates + 1 or entry['phase'] != WEAK
            or entry['prepared_is_committed'] is not False):
        return False
    before = schedule.state_dict()
    if digest(before) != entry['before_sampler_sha256']:
        return False
    cursor = schedule.active_cursor
    count = min(14, cursor.remaining, cursor.population - cursor.draws % cursor.population)
    draws = cursor.preview(count)
    identities, rows = entry['TR_identity'], entry['rows']
    if (entry['TR_draws'] != draws or type(identities) is not list or type(rows) is not list
            or len(identities) != count or len(rows) != count + 2 or not _bounded_builtin_tree(rows[:count])):
        return False
    if len({d['ordinal'] for d in draws}) != count:
        return False
    for draw, identity, row in zip(draws, identities, rows):
        if (type(row) is not dict or type(identity) is not dict
                or identity != {'ordinal': draw['ordinal'], 'source_id': row['source_id'], 'language': 'tr'}
                or row['ordinal'] != draw['ordinal'] or row['language'] != 'tr'
                or row['augmentation_role'] != 'student_tr' or row['source_id'] != row['unique_id']
                or row['recording_source_id'] != row['broad_original']['source_id']
                or row['broad_original']['ordinal'] != draw['ordinal']
                or row['training_input_type'] != 'TR_frame_target_metadata_v1'
                or row['runtime_full_audio_geometry_and_heldout_check_required'] is not True
                or row['target_tokens'] != len(row['tr_metadata_labels']) or row['target_tokens'] <= 0):
            return False
    return True


@contextmanager
def current_plan_identity_view(context, schedule, plan, index, entry, *, authority,
                               fresh_plan_sha256, packet_plan_sha256, optimizer_updates):
    """One unchanged propose, with actual ORIGINAL row cache-hit semantics.

    Unsupported eligibility delegates the ENTIRE propose to the original path.
    An unexpected ordinal during an eligible propose still runs original row
    once, using its normal cache/miss/poison/error behavior; never retry propose.
    The original LRU object/contents are restored, with no cache policy change.
    """
    eligible = False
    try:
        eligible = (id(context) not in _ACTIVE and _eligible(context, schedule, plan, index, entry,
            authority=authority, fresh_plan_sha256=fresh_plan_sha256,
            packet_plan_sha256=packet_plan_sha256, optimizer_updates=optimizer_updates))
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        eligible = False
    if not eligible:
        yield False
        return
    original = context.resolver
    original_cache = original.cache
    view = OrderedDict((draw['ordinal'], deepcopy(row)) for draw, row in zip(entry['TR_draws'], entry['rows']))
    _ACTIVE.add(id(context))
    original.cache = view
    try:
        yield True
    finally:
        # Original row sets THIS original owner's poison flag and propagates
        # the original exception object. This cleanup never catches/retries it.
        original.cache = original_cache
        view.clear()
        _ACTIVE.discard(id(context))
        del original_cache, view, original
