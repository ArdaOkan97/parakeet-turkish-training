"""INACTIVE persistence candidate: committed step upserts and legacy snapshots.

No training, RNG, Trackio, or ML imports. A sole external writer supplies its
already resident history. Journals are never a source of model/optimizer state.
Every transaction is immutable; atomic HEAD replacement is the commit point.
Unreferenced interrupted files are retained, never replayed. A verified model
checkpoint may explicitly cut the active timeline while retaining the previous
committed head as a failed historical branch. No native adapter is admitted here.

Cadence change, if separately adopted: journal at the existing 10-update points;
complete compatible JSON at each DEV/checkpoint/end. Observers between snapshots
must read the committed journal HEAD, not infer progress from old JSON snapshots.
"""
from __future__ import annotations

from array import array
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import struct
import tempfile
import time

KIND = 'committed_optimizer_step_history_journal_v1'
MAGIC = b'ASRHJ1\r\n'
ROW = struct.Struct('<4sQQ32s')
END = struct.Struct('<4sQQ32s')
DEFAULT_AMENDMENT_KEYS = ('validation', 'selection_diagnostics',
                          'development_selection_only', 'final_development_evaluation')
ATOMIC_SHA256 = '2d3fe379d51021fd15679e562c71286f7f9501180d28dde77893e8f72c80a1b0'
NAME = re.compile(r'^[0-9]{20}-[0-9a-f]{64}\.txn$')
ENCODER = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(',', ':'))


class JournalError(ValueError):
    pass


def _atomic():
    path = Path(__file__).with_name('atomic_io_retry_v1.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != ATOMIC_SHA256:
        raise JournalError('Frozen atomic publisher source changed')
    spec = importlib.util.spec_from_file_location('_history_journal_atomic', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compact(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(',', ':')).encode('utf-8')


def _payload(value, limit):
    # Only one bounded record, never the full history, is encoded in memory.
    # JSONEncoder emits a complete string token at once. Reject an oversized
    # string before that allocation, including its UTF-8/JSON escaping expansion.
    def preflight(item, depth=0):
        if depth > 128:
            raise JournalError('Record nesting exceeds the declared bound')
        if isinstance(item, str):
            if len(item) > limit:
                raise JournalError('String exceeds the declared payload bound')
            if len(item) * 6 > limit:
                encoded_length = 2
                for i in range(0, len(item), 4096):
                    encoded_length += len(json.dumps(item[i:i + 4096],
                        ensure_ascii=False).encode('utf-8')) - 2
                    if encoded_length > limit:
                        raise JournalError('Escaped string exceeds the declared payload bound')
        elif isinstance(item, dict):
            for key, member in item.items():
                if not isinstance(key, str):
                    raise JournalError('Journal metrics require JSON string keys')
                preflight(key, depth + 1)
                preflight(member, depth + 1)
        elif isinstance(item, (list, tuple)):
            for member in item:
                preflight(member, depth + 1)
    preflight(value)
    result = bytearray()
    for chunk in ENCODER.iterencode(value):
        encoded = chunk.encode('utf-8')
        if len(result) + len(encoded) > limit:
            raise JournalError('Record exceeds the declared payload bound')
        result.extend(encoded)
    return bytes(result)


def _step(row):
    if not isinstance(row, dict) or type(row.get('step')) is not int or row['step'] < 1:
        raise JournalError('Each history row requires a positive integer optimizer step')
    return row['step']


def _checkpoint(record):
    if (not isinstance(record, dict) or not isinstance(record.get('path'), str)
            or not re.fullmatch('[0-9a-f]{64}', str(record.get('sha256', '')))
            or type(record.get('bytes')) is not int or record['bytes'] < 1):
        raise JournalError('Caller must supply a hash-bound authoritative checkpoint record')
    # Opaque binding only: this utility never opens a checkpoint or admits science.
    return dict(record)


class HistoryJournal:
    """Single-writer durable history with a compact per-step disk location index.

    len/history steps must be 1..H, in original order. publish checks the latest
    committed row and new suffix only; callers must not mutate older rows. Full
    prefix verification is mandatory at seed/reconcile_checkpoint, not per update.
    Exact continuation restores its model checkpoint first and explicitly calls
    reconcile_checkpoint(checkpoint_history, verified_checkpoint_record).
    """
    def __init__(self, directory, *, max_record_bytes=4 * 1024 * 1024,
                 amendment_keys=DEFAULT_AMENDMENT_KEYS, retry_seconds=30.0):
        if type(max_record_bytes) is not int or not 1024 <= max_record_bytes <= 16 * 1024 * 1024:
            raise JournalError('Payload bound must be 1 KiB..16 MiB')
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.head_path = self.directory / 'HEAD.json'
        self.max_record_bytes = max_record_bytes
        self.amendment_keys = tuple(amendment_keys)
        if len(set(self.amendment_keys)) != len(self.amendment_keys) or not all(
                isinstance(k, str) and k != 'step' for k in self.amendment_keys):
            raise JournalError('Invalid amendment field declaration')
        self.retry_seconds = retry_seconds
        self.atomic = _atomic()
        self.tx = []
        self.locations = array('Q')
        self.offsets = array('Q')
        self.lengths = array('I')
        self.digests = bytearray()
        self.orphan_branches = []
        self.last_report = {}
        self._busy = False
        self._invalid = False
        self.head = self._read_head()
        if self.head is not None:
            self._rebuild()

    @property
    def max_step(self):
        return len(self.locations)

    @property
    def event_id(self):
        return self.head['seq'] if self.head else 0

    @property
    def index_bytes(self):
        return (self.locations.itemsize * len(self.locations)
                + self.offsets.itemsize * len(self.offsets)
                + self.lengths.itemsize * len(self.lengths) + len(self.digests))

    def _record_path(self, record):
        if (not isinstance(record, dict) or not NAME.fullmatch(str(record.get('name', '')))
                or record.get('sha256') != record['name'][21:85]
                or type(record.get('bytes')) is not int or record['bytes'] < 1):
            raise JournalError('Invalid immutable transaction reference')
        return self.directory / record['name']

    def _read_head(self):
        if not self.head_path.exists():
            return None
        if self.head_path.stat().st_size > 65536:
            raise JournalError('Oversized HEAD')
        try:
            payload = self.atomic.read_bytes_shared(self.head_path)
            if len(payload) > 65536:
                raise JournalError('Oversized HEAD')
            head = json.loads(payload)
            if (head['kind'] != KIND or head['schema'] != 1
                    or type(head['seq']) is not int or head['seq'] < 1
                    or type(head['max_step']) is not int or head['max_step'] < 1
                    or not re.fullmatch('[0-9a-f]{64}', str(head.get('latest_payload_sha256', '')))
                    or head['max_record_bytes'] != self.max_record_bytes
                    or tuple(head['amendment_keys']) != self.amendment_keys):
                raise JournalError('HEAD identity/format mismatch')
            self._record_path(head['last_transaction'])
            return head
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise JournalError('Corrupt committed HEAD') from exc

    def _header(self, record):
        path = self._record_path(record)
        with path.open('rb') as stream:
            if stream.read(8) != MAGIC:
                raise JournalError('Invalid committed transaction magic')
            raw = stream.read(4)
            if len(raw) != 4:
                raise JournalError('Truncated committed header')
            length = struct.unpack('<I', raw)[0]
            if not 0 < length <= 65536:
                raise JournalError('Invalid committed header length')
            payload = stream.read(length)
            if len(payload) != length:
                raise JournalError('Truncated committed header payload')
            try:
                return json.loads(payload)
            except (ValueError, UnicodeError) as exc:
                raise JournalError('Corrupt committed header') from exc

    def _scan(self, record, callback):
        path = self._record_path(record)
        if path.stat().st_size != record['bytes']:
            raise JournalError('Truncated/extended committed transaction')
        full = hashlib.sha256()
        body = hashlib.sha256()
        count = 0
        with path.open('rb') as stream:
            def read_exact(length, *, is_body=True):
                value = stream.read(length)
                if len(value) != length:
                    raise JournalError('Truncated committed transaction frame')
                full.update(value)
                if is_body:
                    body.update(value)
                return value
            if read_exact(8) != MAGIC:
                raise JournalError('Invalid committed transaction magic')
            length = struct.unpack('<I', read_exact(4))[0]
            if not 0 < length <= 65536:
                raise JournalError('Invalid header bound')
            header = json.loads(read_exact(length))
            if (header.get('kind') != KIND or header.get('schema') != 1
                    or header.get('max_record_bytes') != self.max_record_bytes
                    or tuple(header.get('amendment_keys', ())) != self.amendment_keys):
                raise JournalError('Committed transaction identity mismatch')
            while True:
                marker = read_exact(4, is_body=False)
                if marker == b'END1':
                    footer = marker + read_exact(END.size - 4, is_body=False)
                    _, seq, expected_count, digest = END.unpack(footer)
                    if seq != header['seq'] or expected_count != count or digest != body.digest():
                        raise JournalError('Committed transaction footer/digest mismatch')
                    if stream.read(1):
                        raise JournalError('Unexpected committed trailing bytes')
                    break
                if marker != b'ROW1':
                    raise JournalError('Unknown committed frame marker')
                body.update(marker)
                frame = marker + read_exact(ROW.size - 4)
                _, step, payload_length, digest = ROW.unpack(frame)
                if not 0 < payload_length <= self.max_record_bytes:
                    raise JournalError('Committed record exceeds bound')
                offset = stream.tell()
                payload = read_exact(payload_length)
                if hashlib.sha256(payload).digest() != digest:
                    raise JournalError('Committed row digest mismatch')
                row = json.loads(payload)
                if _step(row) != step:
                    raise JournalError('Committed row step mismatch')
                callback(header, step, offset, payload_length, digest, row)
                count += 1
        if full.hexdigest() != record['sha256']:
            raise JournalError('Committed file SHA mismatch')
        return header, count

    def _set_location(self, step, txid, offset, length, digest):
        if step == self.max_step + 1:
            self.locations.append(txid)
            self.offsets.append(offset)
            self.lengths.append(length)
            self.digests.extend(digest)
        elif step == self.max_step and self.max_step:
            self.locations[-1] = txid
            self.offsets[-1] = offset
            self.lengths[-1] = length
            self.digests[-32:] = digest
        else:
            raise JournalError('Duplicate/gap/out-of-order committed optimizer step')

    def _truncate(self, step):
        del self.locations[step:]
        del self.offsets[step:]
        del self.lengths[step:]
        del self.digests[step * 32:]

    def _apply_transaction(self, record):
        header = self._header(record)
        if header.get('previous_max_step') != self.max_step:
            raise JournalError('Committed step frontier mismatch')
        cutoff = header.get('truncate_to')
        if cutoff is not None:
            if type(cutoff) is not int or not 1 <= cutoff < self.max_step:
                raise JournalError('Invalid historical branch cutoff')
            branch = header.get('orphan_branch')
            if (not isinstance(branch, dict) or branch.get('max_step') != self.max_step
                    or branch.get('last_transaction') != header.get('previous')
                    or branch.get('active_cutoff') != cutoff):
                raise JournalError('Orphan suffix must be explicitly preserved')
            _checkpoint(branch.get('authoritative_checkpoint'))
            self.orphan_branches.append(branch)
            self._truncate(cutoff)
        txid = len(self.tx)
        self.tx.append(record)
        seen = set()
        def receive(h, step, offset, length, digest, row):
            if step in seen:
                raise JournalError('Duplicate row in committed transaction')
            seen.add(step)
            if step == self.max_step and self.max_step:
                self._amendment(self.record(step), row)
            self._set_location(step, txid, offset, length, digest)
        self._scan(record, receive)

    def _rebuild(self, *, verify_head_digest=True):
        chain = []
        reference = self.head['last_transaction']
        expected_seq = self.head['seq']
        while reference is not None:
            header = self._header(reference)
            if header.get('seq') != expected_seq or expected_seq < 1:
                raise JournalError('Broken committed event order')
            chain.append(reference)
            reference = header.get('previous')
            expected_seq -= 1
        if expected_seq != 0:
            raise JournalError('Committed event gap')
        for record in reversed(chain):
            self._apply_transaction(record)
        if self.max_step != self.head['max_step']:
            raise JournalError('HEAD/committed optimizer frontier mismatch')
        if verify_head_digest and self.digests[-32:].hex() != self.head['latest_payload_sha256']:
            raise JournalError('HEAD/latest payload mismatch')

    def refresh(self):
        """Observer admits only new committed events, never rereads the seed rows."""
        if self._invalid:
            raise JournalError('Failed observer admission requires a new validated instance')
        new_head = self._read_head()
        if new_head == self.head:
            return False
        if new_head is None or (self.head and new_head['seq'] <= self.event_id):
            raise JournalError('HEAD moved backwards or conflicted')
        chain = []
        reference = new_head['last_transaction']
        expected = new_head['seq']
        boundary = self.head['last_transaction'] if self.head else None
        while reference != boundary:
            if reference is None or expected <= self.event_id:
                raise JournalError('Observer committed chain diverged')
            header = self._header(reference)
            if header.get('seq') != expected:
                raise JournalError('Observer event sequence gap')
            chain.append(reference)
            reference = header.get('previous')
            expected -= 1
        if expected != self.event_id:
            raise JournalError('Observer event chain boundary mismatch')
        try:
            for record in reversed(chain):
                self._apply_transaction(record)
            if (self.max_step != new_head['max_step']
                    or self.digests[-32:].hex() != new_head['latest_payload_sha256']):
                raise JournalError('Observer HEAD/frontier mismatch')
            self.head = new_head
        except BaseException:
            self._invalid = True
            raise
        return True

    def frontier(self, *, refresh=True):
        if refresh:
            self.refresh()
        return {'event_id': self.event_id, 'max_step': self.max_step,
                'last_transaction': self.head['last_transaction'] if self.head else None,
                'preserved_orphan_branches': len(self.orphan_branches),
                'journal_is_training_state': False}

    def get_step(self, step):
        return self.record(step)

    def iter_latest(self, start_step=1, *, refresh=True):
        if refresh:
            self.refresh()
        yield from self.records(start_step=start_step)

    def record(self, step):
        if self._invalid:
            raise JournalError('Invalid observer must be reopened')
        if type(step) is not int or not 1 <= step <= self.max_step:
            raise JournalError('History step is outside the active timeline')
        i = step - 1
        path = self._record_path(self.tx[self.locations[i]])
        with path.open('rb') as stream:
            stream.seek(self.offsets[i])
            payload = stream.read(self.lengths[i])
        if hashlib.sha256(payload).digest() != self.digests[i * 32:(i + 1) * 32]:
            raise JournalError('Committed payload changed after admission')
        row = json.loads(payload)
        if _step(row) != step:
            raise JournalError('Disk index/optimizer step mismatch')
        return row

    def records(self, *, start_step=1):
        # Opening once per run of frames avoids 37k file handles during snapshots.
        current_tx = None
        stream = None
        if self._invalid or type(start_step) is not int or not 1 <= start_step <= self.max_step + 1:
            raise JournalError('Invalid observer/start step')
        try:
            for i in range(start_step - 1, self.max_step):
                txid = self.locations[i]
                if current_tx != txid:
                    if stream:
                        stream.close()
                    stream = self._record_path(self.tx[txid]).open('rb')
                    current_tx = txid
                stream.seek(self.offsets[i])
                payload = stream.read(self.lengths[i])
                if hashlib.sha256(payload).digest() != self.digests[i * 32:(i + 1) * 32]:
                    raise JournalError('Committed payload changed after admission')
                row = json.loads(payload)
                if _step(row) != i + 1:
                    raise JournalError('Disk index/step mismatch')
                yield row
        finally:
            if stream:
                stream.close()

    def _amendment(self, before, after):
        if _step(before) != _step(after):
            raise JournalError('Amendment changes optimizer step')
        core_before = {k: v for k, v in before.items() if k not in self.amendment_keys}
        core_after = {k: v for k, v in after.items() if k not in self.amendment_keys}
        if _compact(core_before) != _compact(core_after):
            raise JournalError('Latest amendment conflicts with immutable training metrics')

    def _commit(self, records, *, operation, publication, checkpoint_record=None, cutoff=None):
        if self._busy or self._invalid:
            raise JournalError('Reentrant/concurrent writer unsupported')
        self._busy = True
        before_token = self.atomic.read_bytes_shared(self.head_path) if self.head_path.exists() else None
        old_step = self.max_step
        sequence = self.event_id + 1
        previous = self.head['last_transaction'] if self.head else None
        header = {'schema': 1, 'kind': KIND, 'seq': sequence, 'previous': previous,
                  'previous_max_step': old_step, 'operation': operation,
                  'publication': publication, 'max_record_bytes': self.max_record_bytes,
                  'amendment_keys': list(self.amendment_keys), 'truncate_to': cutoff,
                  'checkpoint_record': checkpoint_record}
        if cutoff is not None:
            header['orphan_branch'] = {'kind': 'preserved_failed_publication_suffix_v1',
                'last_transaction': previous, 'event_id': self.event_id,
                'max_step': old_step, 'active_cutoff': cutoff,
                'failed_step_range': [cutoff + 1, old_step],
                'authoritative_checkpoint': checkpoint_record,
                'used_as_training_state': False}
        encoded_header = _payload(header, 65536)
        descriptor, temporary_name = tempfile.mkstemp(prefix='uncommitted-', suffix='.tmp', dir=self.directory)
        temporary = Path(temporary_name)
        full = hashlib.sha256()
        body = hashlib.sha256()
        pending_steps = array('Q')
        pending_offsets = array('Q')
        pending_lengths = array('I')
        pending_digests = bytearray()
        frontier = cutoff if cutoff is not None else old_step
        count = 0
        serialize_seconds = 0.0
        started = time.perf_counter()
        try:
            with os.fdopen(descriptor, 'wb', buffering=1024 * 1024) as stream:
                def write(value, *, is_body=True):
                    stream.write(value)
                    full.update(value)
                    if is_body:
                        body.update(value)
                write(MAGIC)
                write(struct.pack('<I', len(encoded_header)))
                write(encoded_header)
                previous_step = None
                for row in records:
                    step = _step(row)
                    if step == previous_step or not (step == frontier + 1 or (count == 0 and step == frontier)):
                        raise JournalError('Duplicate/gap/order error in new transaction')
                    began = time.perf_counter()
                    payload = _payload(row, self.max_record_bytes)
                    serialize_seconds += time.perf_counter() - began
                    digest = hashlib.sha256(payload).digest()
                    write(ROW.pack(b'ROW1', step, len(payload), digest))
                    pending_steps.append(step)
                    pending_offsets.append(stream.tell())
                    pending_lengths.append(len(payload))
                    pending_digests.extend(digest)
                    write(payload)
                    frontier = step
                    previous_step = step
                    count += 1
                if not count and cutoff is None:
                    raise JournalError('Empty transactions must not be committed')
                write(END.pack(b'END1', sequence, count, body.digest()), is_body=False)
                stream.flush()
                fsync_started = time.perf_counter()
                os.fsync(stream.fileno())
                fsync_seconds = time.perf_counter() - fsync_started
            record = {'name': f'{sequence:020d}-{full.hexdigest()}.txn',
                      'sha256': full.hexdigest(), 'bytes': temporary.stat().st_size}
            immutable_path = self._record_path(record)
            if immutable_path.exists():
                if immutable_path.stat().st_size != record['bytes']:
                    raise JournalError('Immutable retry transaction size conflict')
                self._scan(record, lambda *unused: None)
                # Identical deterministic retry is retained; private file is also
                # retained if failed. This unlink targets only our checked tempfile.
                temporary.unlink()
            else:
                self.atomic.retry_replace(temporary, immutable_path, retry_seconds=self.retry_seconds)
            if (self.atomic.read_bytes_shared(self.head_path) if self.head_path.exists() else None) != before_token:
                raise JournalError('Concurrent writer changed HEAD; sole writer required')
            latest_digest = (pending_digests[-32:].hex() if count
                             else self.digests[(frontier - 1) * 32:frontier * 32].hex())
            head = {'schema': 1, 'kind': KIND, 'seq': sequence, 'last_transaction': record,
                    'max_step': frontier, 'latest_payload_sha256': latest_digest,
                    'max_record_bytes': self.max_record_bytes,
                    'amendment_keys': list(self.amendment_keys),
                    'snapshot_format': 'json_indent2_utf8_noASCII_noNaN_finalLF',
                    'candidate_cadence': 'journal_at_10_updates_snapshots_at_DEV_checkpoint_end'}
            head_started = time.perf_counter()
            self.atomic.write_json(self.head_path, head, retry_seconds=self.retry_seconds)
            head_seconds = time.perf_counter() - head_started
            if cutoff is not None:
                self.orphan_branches.append(header['orphan_branch'])
                self._truncate(cutoff)
            txid = len(self.tx)
            self.tx.append(record)
            for i, step in enumerate(pending_steps):
                self._set_location(step, txid, pending_offsets[i], pending_lengths[i],
                                   pending_digests[i * 32:(i + 1) * 32])
            self.head = head
            self.last_report = {'committed': True, 'event_id': sequence, 'max_step': self.max_step,
                'records_written': count, 'transaction_bytes': record['bytes'],
                'serialize_seconds': serialize_seconds, 'transaction_fsync_seconds': fsync_seconds,
                'HEAD_commit_seconds': head_seconds, 'total_seconds': time.perf_counter() - started,
                'index_bytes': self.index_bytes, 'checkpoint_record': checkpoint_record,
                'orphan_branch': header.get('orphan_branch'), 'publication': publication}
            return dict(self.last_report)
        finally:
            self._busy = False

    def seed(self, history, checkpoint_record):
        if self.head or self.max_step:
            raise JournalError('Seed requires a new empty journal; reconcile an existing one')
        return self._commit(history, operation='seed', publication='checkpoint',
                            checkpoint_record=_checkpoint(checkpoint_record))

    def publish(self, history, *, publication='periodic'):
        if publication not in ('periodic', 'DEV', 'checkpoint', 'end'):
            raise JournalError('Unknown publication boundary')
        if not self.head:
            raise JournalError('Seed or checkpoint reconciliation required before publication')
        if len(history) < self.max_step:
            raise JournalError('History behind journal requires explicit checkpoint reconciliation')
        old = self.max_step
        boundary = history[old - 1]
        if _step(boundary) != old:
            raise JournalError('History is not keyed in optimizer-step order')
        payload = _payload(boundary, self.max_record_bytes)
        changed = hashlib.sha256(payload).digest() != self.digests[-32:]
        if changed:
            self._amendment(self.record(old), boundary)
            if publication == 'periodic':
                raise JournalError('Latest DEV amendment needs an explicit DEV/checkpoint/end boundary')
        if not changed and len(history) == old:
            return {'committed': False, 'idempotent': True, 'event_id': self.event_id,
                    'max_step': old, 'records_written': 0, 'transaction_bytes': 0}
        def suffix():
            if changed:
                yield boundary
            for i in range(old, len(history)):
                if _step(history[i]) != i + 1:
                    raise JournalError('Gap/order conflict in supplied history suffix')
                yield history[i]
        return self._commit(suffix(), operation='upsert', publication=publication)

    def reconcile_checkpoint(self, history, checkpoint_record, *, preserve_orphan_suffix=False):
        """Verify caller's checkpoint history, never restore from journal metrics.

        A journal ahead of the model checkpoint requires explicit preservation.
        Its prior HEAD and immutable chain remain readable via branch_records().
        Earlier core/metric conflicts are rejected; only boundary DEV fields may
        differ, because the model checkpoint can precede the final DEV publication.
        """
        binding = _checkpoint(checkpoint_record)
        if not self.head:
            return self.seed(history, binding)
        h = len(history)
        if h < 1:
            raise JournalError('Empty authoritative checkpoint history unsupported')
        limit = min(h, self.max_step)
        boundary_changed = False
        for i in range(limit):
            row = history[i]
            if _step(row) != i + 1:
                raise JournalError('Authoritative checkpoint step gap/order conflict')
            digest = hashlib.sha256(_payload(row, self.max_record_bytes)).digest()
            expected = self.digests[i * 32:(i + 1) * 32]
            if digest != expected:
                if i + 1 != limit:
                    raise JournalError('Committed prefix conflicts with authoritative checkpoint')
                self._amendment(self.record(i + 1), row)
                boundary_changed = True
        cutoff = h if self.max_step > h else None
        if cutoff is not None and not preserve_orphan_suffix:
            raise JournalError('Committed suffix beyond checkpoint needs explicit failed-branch preservation')
        if cutoff is None and h == self.max_step and not boundary_changed:
            return {'committed': False, 'idempotent': True, 'event_id': self.event_id,
                    'max_step': h, 'checkpoint_record': binding, 'orphan_branch': None}
        def suffix():
            if boundary_changed:
                yield history[limit - 1]
            for i in range(limit, h):
                if _step(history[i]) != i + 1:
                    raise JournalError('Checkpoint history suffix gap/order conflict')
                yield history[i]
        return self._commit(suffix(), operation='checkpoint_reconcile', publication='checkpoint',
                            checkpoint_record=binding, cutoff=cutoff)

    def branch_records(self, branch):
        """Read preserved failed-branch metrics; never apply them to training."""
        if branch not in self.orphan_branches:
            raise JournalError('Unknown failed historical branch')
        view = object.__new__(HistoryJournal)
        view.__dict__ = dict(self.__dict__)
        view.tx = []
        view.locations, view.offsets, view.lengths = array('Q'), array('Q'), array('I')
        view.digests, view.orphan_branches = bytearray(), []
        header = self._header(branch['last_transaction'])
        view.head = {'last_transaction': branch['last_transaction'], 'seq': header['seq'],
                     'max_step': branch['max_step']}
        view._rebuild(verify_head_digest=False)
        yield from view.records()

    @staticmethod
    def stream_legacy_json(stream, rows):
        encoder = json.JSONEncoder(indent=2, ensure_ascii=False, allow_nan=False)
        iterator = iter(rows)
        first = next(iterator, None)
        if first is None:
            stream.write('[]\n')
            return 0
        stream.write('[\n')
        count = 0
        row = first
        while True:
            at_start = True
            for chunk in encoder.iterencode(row):
                # Only literal LF is structural. Unicode U+2028/U+2029 inside
                # strings must remain byte-identical to the frozen json.dump.
                parts = chunk.split('\n')
                for i, part in enumerate(parts):
                    if at_start and (part or i < len(parts) - 1):
                        stream.write('  ')
                    stream.write(part)
                    if i < len(parts) - 1:
                        stream.write('\n')
                        at_start = True
                    elif part:
                        at_start = False
            count += 1
            row = next(iterator, None)
            if row is None:
                stream.write('\n]\n')
                return count
            stream.write(',\n')

    def snapshot(self, path):
        """Stream/fsync/atomically replace one complete legacy-compatible JSON."""
        path = Path(path).resolve()
        if path == self.head_path or path.parent == self.directory:
            raise JournalError('Snapshot must not overwrite journal-owned files')
        temporary = self.atomic.temporary_path(path)
        started = time.perf_counter()
        with temporary.open('w', encoding='utf-8', newline='\n', buffering=1024 * 1024) as stream:
            count = self.stream_legacy_json(stream, self.records())
            stream.flush()
            fsync_started = time.perf_counter()
            os.fsync(stream.fileno())
            fsync_seconds = time.perf_counter() - fsync_started
        serialize_seconds = time.perf_counter() - started - fsync_seconds
        commit_started = time.perf_counter()
        retries = self.atomic.retry_replace(temporary, path, retry_seconds=self.retry_seconds)
        return {'path': str(path), 'event_id': self.event_id, 'rows': count,
                'bytes': path.stat().st_size, 'stream_serialize_seconds': serialize_seconds,
                'fsync_seconds': fsync_seconds, 'commit_seconds': time.perf_counter() - commit_started,
                'total_seconds': time.perf_counter() - started, 'retries': retries,
                'cadence_changed': True, 'legacy_schema_and_format_preserved': True}

