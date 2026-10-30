"""INACTIVE broad-row prefetch candidate; no training entry or sampler commits.

Only the original readonly row method runs on the worker. Original coordinator
and lifecycle planning remain on MainThread. Adoption is an explicit temporary
input view while that unchanged planner constructs its independent private plan.
"""
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import queue
import threading
import time

from bpe_broad_production_state_v1 import CompletePassCursor, canonical, digest, require
from bpe_declared_geometry_index_v1 import DeclaredGeometryIndex, open_final, signature
from bpe_training_schedule_v1 import WEAK
from bpe_windows_final_metadata_lock_v1 import FinalMetadataReadLocks
from asr18_BPE9223_source_context_v2 import FullRunSourceContext9223, FinalBroadRows9223, BROAD_N

MAX_ROWS = 280
MAX_ROW_BYTES = 8 * 1024**2
ADDITIONAL_PRIVATE_ADMISSION_BYTES = 256 * 1024**2
KEYS = ('final_catalog', 'final_labels', 'declared_geometry_index')


def main_thread():
    require(threading.current_thread() is threading.main_thread(), 'Authoritative planning remains MainThread-only')


class _ThreadOwnedBroadRows(FinalBroadRows9223):
    """Original row/declaration implementations, separate actual readonly handles.

    This constructor does NOT claim to repeat original whole-file SHA admission.
    Its source authority is the continuously locked, already SHA-validated main
    context, followed by independently acquired deny-write handles and signatures.
    """
    def __init__(self, bindings, expected_signatures, *, guard):
        require(threading.current_thread() is not threading.main_thread(), 'Worker-owned SQLite objects only')
        self.guard = guard
        self.cache = OrderedDict()
        self.poisoned = False
        self.population = BROAD_N
        self.connections = []
        self.metadata_locks = FinalMetadataReadLocks([bindings[k]['path'] for k in KEYS])
        try:
            require([signature(bindings[k]['path']) for k in KEYS] == expected_signatures,
                    'Same continuously locked immutable metadata, no replacement or changed bytes')
            self.signatures = deepcopy(expected_signatures[:2])
            for key in KEYS[:2]:
                self.connections.append(open_final(bindings[key]['path']))
            geometry = DeclaredGeometryIndex.__new__(DeclaredGeometryIndex)
            geometry.guard = guard
            geometry.path = Path(bindings[KEYS[2]]['path']).resolve()
            geometry.signature = deepcopy(expected_signatures[2])
            geometry.poisoned = False
            geometry.connection = open_final(geometry.path)
            self.geometry = geometry
            require(all(c.execute('PRAGMA query_only').fetchone()[0] == 1 and
                        c.execute('PRAGMA cache_size').fetchone()[0] == -4096
                        for c in [*self.connections, geometry.connection]),
                    'Original readonly4MiB caches, no pagecache-policy change')
        except BaseException:
            self.close()
            raise


class _RowsView:
    def __init__(self, result):
        self.rows = result['rows']
        self.used = set()

    def row(self, ordinal):
        main_thread()
        require(type(ordinal) is int and ordinal in self.rows, 'Only the exact prefetched original ordinal')
        self.used.add(ordinal)
        return deepcopy(self.rows[ordinal])


class PrivateScalarPrefetch:
    """One pending request/result, no replay RNG, decode, native or disk outputs.

    Caller must keep the original context open through joined close(). The
    worker never receives that context, schedule, GPU guard or SQLite connection.
    Main-thread guard/admission remain responsible for real whole-process memory.
    """
    def __init__(self, context, *, guard, admit):
        main_thread()
        require(type(context) is FullRunSourceContext9223 and not context.closed and
                type(context.resolver) is FinalBroadRows9223 and not context.resolver.poisoned,
                'Original already-admitted open source context')
        self.context = context
        self.original = context.resolver
        self.guard = guard
        self.bindings = deepcopy({k: context.validated['bindings'][k] for k in KEYS})
        self.signatures = [signature(self.bindings[k]['path']) for k in KEYS]
        self._check_main_authority()
        self.admission = admit('private_scalar_prefetch_additional_host_state', ADDITIONAL_PRIVATE_ADMISSION_BYTES)
        self.requests = queue.Queue(maxsize=1)
        self.results = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.worker_status = {'error': None, 'closed': False}
        self.closed = False
        self.sequence = 0
        self.outstanding = None
        self.held_result = None
        self.thread = threading.Thread(target=PrivateScalarPrefetch._run,
            args=(deepcopy(self.bindings), deepcopy(self.signatures), self.requests, self.results, self.stop, self.worker_status),
            name='private-broad-row-prefetch', daemon=True)
        self.thread.start()

    def _check_main_authority(self):
        main_thread()
        self.guard()
        require(not self.context.closed and self.context.resolver is self.original and not self.original.poisoned and
                len(self.original.metadata_locks.handles) == 3 and
                self.original.metadata_locks.paths == [Path(self.bindings[k]['path']).resolve() for k in KEYS] and
                [signature(self.bindings[k]['path']) for k in KEYS] == self.signatures and
                self.original.signatures == self.signatures[:2] and
                self.original.geometry.signature == self.signatures[2],
                'Continuously held genuine full-SHA-admitted original source authority')

    @staticmethod
    def _run(bindings, signatures, requests, results, stop, status):
        def worker_guard():
            if stop.is_set():
                raise InterruptedError('Closing unconsumed private scalar work')
        reader = None
        try:
            reader = _ThreadOwnedBroadRows(bindings, signatures, guard=worker_guard)
            while not stop.is_set():
                try:
                    request = requests.get(timeout=.1)
                except queue.Empty:
                    continue
                began = time.monotonic()
                cursor_state = request['cursor']
                cursor = CompletePassCursor(population=cursor_state['population'], identity=cursor_state['identity'],
                                            seed=cursor_state['seed'], passes=cursor_state['passes'])
                cursor.load_state_dict(cursor_state)
                initial_epoch = cursor.draws // cursor.population
                rows, draws, encoded_bytes = {}, [], 0
                for _ in range(request['maximum_updates']):
                    worker_guard()
                    require(time.monotonic() - began <= 120, 'Bounded private metadata request; no silent stale adoption')
                    if not cursor.remaining or cursor.draws // cursor.population != initial_epoch:
                        break
                    batch = cursor.preview(min(14, cursor.remaining, cursor.population - cursor.draws % cursor.population))
                    for draw in batch:
                        ordinal = draw['ordinal']
                        require(ordinal not in rows and len(rows) < MAX_ROWS, 'At most one exact280-row private prefix')
                        row = reader.row(ordinal)
                        encoded_bytes += len(canonical(row))
                        require(encoded_bytes <= MAX_ROW_BYTES, 'Original8MiB metadata limit; no row trimming')
                        rows[ordinal] = row
                    draws.append(batch)
                    cursor.commit(batch)  # Independent scalar preview cursor only; no schedule/replay exists here.
                result = {'sequence': request['sequence'], 'before_sampler_sha256': request['before_sampler_sha256'],
                          'phase': WEAK, 'rows': rows, 'draws': draws, 'bindings_sha256': digest(bindings),
                          'rows_sha256': digest(rows),
                          'metadata_bytes': encoded_bytes, 'preparation_seconds': time.monotonic() - began}
                while not stop.is_set():
                    try:
                        results.put(result, timeout=.1)
                        break
                    except queue.Full:
                        pass
                del rows, result
        except InterruptedError:
            if not stop.is_set():
                status['error'] = RuntimeError('Unexpected private metadata interruption')
        except BaseException as error:
            status['error'] = error
        finally:
            if reader is not None:
                try:
                    reader.close()  # All worker connections and deny-write handles close on their owning thread.
                except BaseException as error:
                    status['error'] = error
            status['closed'] = True

    def submit(self, future_schedule_state, *, maximum_updates):
        main_thread()
        self._check_main_authority()
        require(not self.closed and self.outstanding is None and self.held_result is None and self.worker_status['error'] is None and
                self.thread.is_alive() and future_schedule_state['phase'] == WEAK and
                type(maximum_updates) is int and 1 <= maximum_updates <= 20,
                'One private broad-only future prefix; all original pass/periodic bounds are supplied by MainThread')
        cursor = deepcopy(future_schedule_state['broad_cursor'])
        current = self.context.new_schedule().broad.state_dict()
        require({k: v for k, v in cursor.items() if k != 'draws'} ==
                {k: v for k, v in current.items() if k != 'draws'}, 'Same original broad population/seed/bijection')
        self.sequence += 1
        request = {'sequence': self.sequence, 'cursor': cursor,
                   'before_sampler_sha256': digest(future_schedule_state), 'maximum_updates': maximum_updates}
        self.requests.put_nowait(request)
        self.outstanding = {'sequence': self.sequence, 'before_sampler_sha256': request['before_sampler_sha256']}
        return deepcopy(self.outstanding)

    def ready(self):
        main_thread()
        self.guard()
        if self.worker_status['error'] is not None:
            raise RuntimeError('Private scalar prefetch failed') from self.worker_status['error']
        return not self.results.empty()

    def take(self, ticket, *, wait=False):
        main_thread()
        require(ticket == self.outstanding and not self.closed, 'Exact ordered single outstanding request')
        while not self.ready():
            require(wait, 'Private metadata not ready; continue current native window')
            require(self.thread.is_alive(), 'Private worker terminated before result')
            time.sleep(.02)
        result = self.results.get_nowait()
        self._check_main_authority()
        require(all(result[k] == ticket[k] for k in ticket) and result['bindings_sha256'] == digest(self.bindings),
                'Result belongs to this immutable source and exact future sampler state')
        self.outstanding = None
        self.held_result = result
        return result

    @contextmanager
    def planning_rows(self, result, schedule):
        """Only around the unchanged original MainThread context.plan call."""
        main_thread()
        self._check_main_authority()
        before = deepcopy(schedule.state_dict())
        require(self.held_result is result and before['phase'] == WEAK and
                digest(before) == result['before_sampler_sha256'] and result['rows_sha256'] == digest(result['rows']) and
                result['bindings_sha256'] == digest(self.bindings), 'Exact private schedule/source at ordered adoption')
        view = _RowsView(result)
        self.context.resolver = view
        try:
            yield view
        finally:
            require(self.context.resolver is view, 'No overlapping resolver/adoption scope')
            self.context.resolver = self.original
            require(schedule.state_dict() == before, 'Original planner leaves private input schedule unchanged')
            self._check_main_authority()
            self.held_result = None

    def close(self, *, discard_unconsumed=False):
        main_thread()
        require(not self.closed and (self.held_result is None or discard_unconsumed) and
                (self.outstanding is None or discard_unconsumed) and self.context.resolver is self.original,
                'Explicit unconsumed-only cancellation; no consumed draw/native authority')
        self.stop.set()
        self.thread.join(timeout=30)
        require(not self.thread.is_alive() and self.worker_status['closed'],
                'Actual joined private worker before original context closes; timeout is owner failure, not normal0')
        self.closed = True
        self.held_result = None
        while not self.results.empty():
            self.results.get_nowait()
        if self.worker_status['error'] is not None:
            raise RuntimeError('Private scalar prefetch closed after failure') from self.worker_status['error']
        return {'joined': True, 'worker_connections_closed_on_worker': True,
                'discarded_unconsumed_request': self.outstanding is not None,
                'actual_sampler_replay_or_native_advance': False}
