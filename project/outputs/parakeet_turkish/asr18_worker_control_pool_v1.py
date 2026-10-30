"""Inactive, future-owned worker controls: bounded live slots and exact archives.

This module launches nothing. An independently reviewed owner proves normal
process completion, current recovery/quiescence and the complete protection set.
It can retire only directories reserved here while absent, plus explicitly pinned
files in the same future control root. Existing/unknown/crash-tail artifacts are
never adopted or repaired. Archive/index bytes remain in the fixed 1 GiB domain.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import struct
import tempfile
import zlib

from atomic_io_retry_v1 import read_json, write_json

ROOT = Path(r'E:\gpu_training_scratchpad')
KIND = 'ASR18_future_owned_worker_control_pool_v1'
MAGIC = b'ASR18_WORKER_CONTROLS_1\n'
CHUNK = 1024 * 1024
HEADER = struct.Struct('<QQ32s')
CAPS = {
    'fixed_control_bytes': 1073741824, 'whole_control_bytes': 4294967296,
    'live_slots': 48, 'per_live_slot_bytes': 67108864,
    'archive_stream_bytes': 134217728, 'index_bytes': 16777216,
    'registry_bytes': 1048576, 'archive_frame_bytes': 8388608,
    'raw_frame_bytes': 134217728, 'header_bytes': 262144,
    'files_per_archive': 64, 'directories_per_worker': 8,
    'worker_files': 32, 'paired_files': 32, 'member_bytes': 16777216,
    'archive_count': 4096, 'member_rows': 65536, 'owner_folders': 513,
}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def same(a, b):
    return a == b and digest(a) == digest(b)


def safe(value, parent=ROOT):
    path = Path(value).absolute()
    parent = Path(parent).resolve()
    require(path.is_relative_to(parent) and path != parent, 'Exact E owned path')
    for item in (path, *path.parents):
        if item.exists():
            info = item.lstat()
            require(not item.is_symlink() and not (getattr(info, 'st_file_attributes', 0) & 0x400),
                    'No symlink/reparse component')
        if item == parent:
            break
    require(path.resolve() == path, 'Canonical absolute path, no traversal/alias')
    return path


def pin(path, guard=lambda: None):
    path = Path(path).resolve()
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and not path.is_symlink()
            and not (getattr(info, 'st_file_attributes', 0) & 0x400), 'Regular exact file')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            guard(); block = stream.read(CHUNK)
            if not block:
                break
            h.update(block)
    after = path.stat()
    require((info.st_size, info.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            'File changed during complete byte hash')
    return {'path': str(path), 'bytes': after.st_size, 'sha256': h.hexdigest()}


def bound(value, guard=lambda: None, maximum=None):
    require(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}
            and type(value['bytes']) is int and value['bytes'] >= 0
            and type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['sha256']),
            'Exact typed record3, including truthful zero-byte logs')
    if maximum is not None:
        require(value['bytes'] <= maximum, 'Bounded metadata/member')
    require(same(pin(value['path'], guard), value), 'Exact existing record3 bytes')
    return Path(value['path']).resolve()


def _callback(function, source, request):
    actual = inspect.unwrap(function)
    require(callable(function) and Path(inspect.getsourcefile(actual)).resolve() == Path(source['path']),
            'Exact reviewed owner callback source')
    bound(source, maximum=262144)
    require(function(deepcopy(request)) is True, 'Independent owner authority rejected')


class _LimitedReader:
    def __init__(self, stream, guard):
        self.stream, self.guard = stream, guard

    def read(self, count):
        require(type(count) is int and 0 < count <= CHUNK, 'Only bounded binary reads')
        self.guard()
        return self.stream.read(count)


class WorkerControlPool:
    """One locked future pool, reopened without resetting any cumulative quota.

    authority_source is the immutable controller/parent callback implementation.
    protect() must be called and actually verified in each supervisor before
    reserve/archive. Caller must independently validate normal-pause/outer0 before
    opening create=False; this class additionally rejects any partial pool work.
    """
    def __init__(self, control, *, identity, policy, authority_source, guard,
                 metadata_phase, admit_control, create=False):
        self.control = safe(control, ROOT / 'runs')
        self.folder = self.control / 'worker_control_pool'
        self.identity = deepcopy(identity); self.policy = deepcopy(policy)
        self.authority_source = deepcopy(authority_source)
        self.guard, self.metadata_phase, self.admit_control = guard, metadata_phase, admit_control
        self._lock = None; self._locked = False; self._retiring_paths = frozenset()
        self._temporary_paths = set(); self._sqlite_writers = 0
        self._verified_protection = False
        self._verify_protection = None; self._protection_evidence = None
        require(callable(guard) and callable(metadata_phase) and callable(admit_control)
                and type(identity) is dict and identity,
                'Explicit scientific identity and live metadata guards')
        require(policy['fixed_control_allocation_cap_bytes'] == CAPS['fixed_control_bytes']
                and policy['total_allocated_cap_bytes'] == CAPS['whole_control_bytes']
                and policy['maximum_GPU_worker_directories'] == CAPS['live_slots']
                and policy['reserved_each_GPU_worker_allocation_bytes'] == CAPS['per_live_slot_bytes'],
                'Unchanged fixed1GiB +48LIVE*64MiB = whole4GiB policy')
        bound(authority_source, guard, 262144)
        require(Path(inspect.getsourcefile(inspect.unwrap(admit_control))).resolve() == Path(authority_source['path']),
                'Prospective fixed-control admission uses exact owner-parent source')
        require(self.control.is_dir(), 'Caller-created exact future control root')
        if create:
            require(not self.folder.exists(), 'Fresh pool only, no old directory adoption')
            self._admit(self.folder / 'archives.bin', 20 * 1024**2)
            self.folder.mkdir()
        require(self.folder.is_dir(), 'Existing exact pool required on reopen')
        safe(self.folder, self.control)
        self.owner_path = self.folder / 'owner.json'
        self.registry_path = self.folder / 'registry.json'
        self.archive_path = self.folder / 'archives.bin'
        self.index_path = self.folder / 'index.sqlite'
        try:
            self._acquire()
            owner = {'kind': KIND, 'control': str(self.control), 'identity': self.identity,
                     'policy': self.policy, 'authority_source': authority_source,
                     'component_source': pin(__file__), 'caps': CAPS,
                     'existing_or_original_artifact_adoption_permitted': False}
            if create:
                write_json(self.owner_path, owner)
                with self.archive_path.open('xb') as stream:
                    stream.write(MAGIC); stream.flush(); os.fsync(stream.fileno())
                db = self._db()
                try:
                    db.execute('CREATE TABLE archives (id INTEGER PRIMARY KEY, descriptor TEXT NOT NULL, status TEXT NOT NULL)')
                    db.execute('CREATE TABLE members (path TEXT NOT NULL, bytes INTEGER NOT NULL, sha256 TEXT NOT NULL, archive_id INTEGER NOT NULL, PRIMARY KEY(path,bytes,sha256))')
                    db.commit()
                finally:
                    db.close(); self._sqlite_writers -= 1
                write_json(self.registry_path, {'kind': KIND, 'revision': 0, 'live': [], 'owner_folders': [],
                    'protected_folders': [], 'protected_files': [], 'protection_evidence': None,
                    'archive_count': 0, 'member_count': 0, 'archive_bytes': len(MAGIC),
                    'reservation_count': 0, 'pending_retirement': None})
            require(same(read_json(self.owner_path), owner), 'Same immutable pool/source/identity/policy')
            self.owner_pin = pin(self.owner_path, guard)
            self._state = self._load()
        except BaseException:
            self.close()
            raise

    def _acquire(self):
        import msvcrt
        path = safe(self.folder / 'writer.lock', self.control)
        self._lock = path.open('a+b')
        if path.stat().st_size == 0:
            self._lock.write(b'0'); self._lock.flush(); os.fsync(self._lock.fileno())
        require(path.stat().st_size == 1, 'Exact persistent pool lock')
        self._lock.seek(0)
        msvcrt.locking(self._lock.fileno(), msvcrt.LK_NBLCK, 1)
        self._locked = True

    def _db(self):
        if self.index_path.exists():
            require(self.index_path.stat().st_size <= CAPS['index_bytes'], 'Bounded pool SQLite index')
        db = sqlite3.connect(self.index_path); self._sqlite_writers += 1
        db.execute('PRAGMA journal_mode=DELETE'); db.execute('PRAGMA synchronous=FULL')
        db.execute('PRAGMA cache_size=-2048'); db.execute('PRAGMA temp_store=MEMORY')
        db.execute('PRAGMA max_page_count=4096'); db.execute('PRAGMA busy_timeout=1000')
        return db

    def _read_db(self):
        require(self.index_path.is_file() and self.index_path.stat().st_size <= CAPS['index_bytes'],
                'Existing bounded pool index; lookup never creates a database')
        db = sqlite3.connect(self.index_path.as_uri() + '?mode=ro', uri=True, timeout=1)
        db.execute('PRAGMA cache_size=-2048'); db.execute('PRAGMA query_only=ON')
        return db

    def _admit(self, path, prospective_bytes):
        self.guard(); bound(self.authority_source, self.guard, 262144)
        require(type(prospective_bytes) is int and prospective_bytes >= 0, 'Typed prospective allocation')
        self.admit_control(Path(path), prospective_bytes)

    @contextmanager
    def _temporary(self):
        # Windows TemporaryFile uses a private name and removes only this newly
        # created file. Track that exact live name; an orphan at reopen is not
        # accepted merely because its spelling resembles a temporary.
        temp = tempfile.TemporaryFile(dir=self.folder, prefix='owned_member_', suffix='.tmp')
        name = str(Path(temp.name).absolute())
        self._temporary_paths.add(name)
        try:
            yield temp
        finally:
            temp.close(); self._temporary_paths.discard(name)

    def _load(self):
        self.guard()
        require(same(pin(self.owner_path), self.owner_pin) if hasattr(self, 'owner_pin') else True,
                'Pool ownership unchanged')
        require(self.registry_path.stat().st_size <= CAPS['registry_bytes'], 'Bounded live pool registry')
        state = read_json(self.registry_path)
        require(type(state) is dict and state['kind'] == KIND and state['pending_retirement'] is None,
                'No automatic adoption of partial archive/deletion or crash tail')
        require(all(type(state[k]) is int and state[k] >= 0 for k in
                    ('revision', 'archive_count', 'member_count', 'archive_bytes', 'reservation_count')),
                'Exact integer cumulative pool counters')
        require(type(state['live']) is list and len(state['live']) <= CAPS['live_slots']
                and len({x['folder'] for x in state['live']}) == len(state['live']), 'Bounded unique live reservation table')
        require(type(state['owner_folders']) is list
                and len(state['owner_folders']) == len(set(state['owner_folders'])) <= CAPS['owner_folders']
                and all(str(Path(x['folder']).parent) in state['owner_folders'] for x in state['live']),
                'Persistent bounded previous/current owner-root inventory')
        require(self.archive_path.stat().st_size == state['archive_bytes'] <= CAPS['archive_stream_bytes']
                and 0 <= state['archive_count'] <= CAPS['archive_count']
                and 0 <= state['member_count'] <= CAPS['member_rows'], 'Exact committed archive tail/cumulative limits')
        expected = {'owner.json', 'writer.lock', 'registry.json', 'archives.bin', 'index.sqlite'}
        expected.update(Path(p).name for p in self._temporary_paths if Path(p).exists())
        if self._sqlite_writers and (self.folder / 'index.sqlite-journal').exists():
            expected.add('index.sqlite-journal')
        require(set(p.name for p in self.folder.iterdir()) == expected,
                'Unknown/orphan pool files preserved and rejected')
        for path in self.folder.iterdir():
            safe(path, self.control)
            require(path.is_file(), 'No pool directories or reparse artifacts')
        db = self._read_db()
        try:
            row = db.execute('SELECT COUNT(*),MAX(id) FROM archives WHERE status="complete"').fetchone()
            require(row[0] == state['archive_count'] and (row[1] or 0) == state['archive_count']
                    and db.execute('SELECT COUNT(*) FROM archives WHERE status!="complete"').fetchone()[0] == 0
                    and db.execute('SELECT COUNT(*) FROM members').fetchone()[0] == state['member_count'],
                    'SQLite/catalog and registry commit frontier match; no repair')
        finally:
            db.close()
        return state

    def _save(self, state):
        require(len(canonical(state)) + 4096 <= CAPS['registry_bytes'], 'Bounded registry before atomic write')
        state = deepcopy(state); state['revision'] += 1
        self._admit(self.registry_path, CAPS['registry_bytes'] + 2 * 1024**2)
        write_json(self.registry_path, state)
        self._state = state

    def _request(self, **extra):
        return {'identity': deepcopy(self.identity), 'owner': self.owner_pin,
                'revision': self._state['revision'], 'live': deepcopy(self._state['live']),
                'protected_folders': deepcopy(self._state['protected_folders']),
                'protected_files': deepcopy(self._state['protected_files']), **extra}

    def live_folders(self):
        return tuple(Path(item['folder']) for item in self._state['live'])

    def owner_folders(self):
        return tuple(Path(path) for path in self._state['owner_folders'])

    @property
    def state_pin(self):
        return pin(self.registry_path, self.guard)

    @property
    def revision(self):
        return self._state['revision']

    def snapshot_binding(self):
        self._state = self._load()
        return {'kind': 'ASR18_exact_worker_control_pool_snapshot_v1',
                'owner': self.owner_pin, 'registry': self.state_pin,
                'index': pin(self.index_path, self.guard),
                'archive_prefix': pin(self.archive_path, self.guard),
                'archive_bytes': self._state['archive_bytes'],
                'archive_count': self._state['archive_count'],
                'reservation_count': self._state['reservation_count'],
                'live_folders': [str(x) for x in self.live_folders()],
                'owner_folders': [str(x) for x in self.owner_folders()]}

    def protect(self, folders, *, files=(), evidence, verify_protection):
        """Persist and independently verify the COMPLETE replacement protection set."""
        self._state = self._load(); bound(evidence, self.guard, CAPS['member_bytes'])
        values = sorted(str(safe(path)) for path in folders)
        require(len(values) == len(set(values)) <= CAPS['live_slots']
                and set(values) <= {x['folder'] for x in self._state['live']}, 'Only registered protected worker folders')
        pins = list(files)
        require(len(pins) <= 256, 'Bounded explicit protected control pins')
        for item in pins:
            path = bound(item, self.guard, CAPS['member_bytes'])
            # Exact E config/session/assembly pins may be protected without
            # being owned. Neither reserve nor archive gains authority over
            # those external paths; archive's separate owned-scope test stays.
            safe(path)
        request = self._request(operation='verify_complete_worker_control_protection',
                                folders=values, files=pins, evidence=evidence)
        _callback(verify_protection, self.authority_source, request)
        state = deepcopy(self._state); state.update(protected_folders=values,
            protected_files=deepcopy(pins), protection_evidence=deepcopy(evidence))
        self._save(state); self._verified_protection = True
        self._verify_protection, self._protection_evidence = verify_protection, deepcopy(evidence)
        return {'owner': self.owner_pin, 'revision': self._state['revision'],
                'protected_folders': values, 'protected_files': deepcopy(pins)}

    def reserve(self, folder, *, configuration, worker_source, parent_source):
        self._state = self._load()
        require(self._verified_protection, 'Fresh complete protection verification in each supervisor')
        path = safe(folder)
        require(not path.exists() and path.parent.is_dir() and len(self._state['live']) < CAPS['live_slots']
                and re.fullmatch(r'(?:native|recovery)_[0-9]{6}|DEV_[A-Za-z0-9_]{1,80}', path.name),
                'Fresh bounded worker reservation before any Popen/directory creation')
        require(path.parent.is_relative_to(ROOT / 'analysis') or path.parent.is_relative_to(ROOT / 'runs'),
                'Worker invocation root only in reviewed analysis/runs domain')
        config_path = bound(configuration, self.guard, CAPS['member_bytes'])
        require(config_path.is_relative_to(self.control) and not config_path.is_relative_to(self.folder),
                'Exact paired immutable owned worker configuration')
        bound(worker_source, self.guard, 262144); bound(parent_source, self.guard, 262144)
        require(same(parent_source, self.authority_source), 'Actual reservation parent is the reviewed authority source')
        require(str(path) not in {x['folder'] for x in self._state['live']}, 'No live worker directory reuse')
        item = {'folder': str(path), 'configuration': deepcopy(configuration),
                'worker_source': deepcopy(worker_source), 'parent_source': deepcopy(parent_source),
                'reservation': self._state['reservation_count'] + 1}
        state = deepcopy(self._state); state['live'].append(item); state['reservation_count'] += 1
        if str(path.parent) not in state['owner_folders']:
            require(len(state['owner_folders']) < CAPS['owner_folders'], 'Bounded persistent invocation-root count')
            state['owner_folders'].append(str(path.parent))
        self._save(state)
        return deepcopy(item)

    def _inventory(self, folder, paired):
        files, directories = [], []
        for root, dirs, names in os.walk(folder, followlinks=False):
            self.guard()
            for name in dirs:
                path = safe(Path(root) / name, folder)
                require(path.is_dir(), 'Only exact bounded owned directories')
                directories.append(str(path))
            require(len(directories) <= CAPS['directories_per_worker'], 'Bounded owned worker directories')
            for name in names:
                path = safe(Path(root) / name, folder)
                require(path.suffix.lower() in ('.json', '.jsonl', '.log', '.txt'), 'No unknown/temp/model/audio worker members')
                item = pin(path, self.guard)
                require(item['bytes'] <= (8388608 if name in ('worker.log', 'stages.jsonl') else 1048576),
                        'Original source per-worker metadata/log cap')
                files.append(item)
            require(len(files) <= CAPS['worker_files'], 'Original bounded worker file table')
        for item in paired:
            path = bound(item, self.guard, CAPS['member_bytes'])
            require((path.is_relative_to(self.control) or str(path.parent) in self._state['owner_folders'])
                    and not path.is_relative_to(self.folder)
                    and path.suffix.lower() in ('.json', '.jsonl', '.log', '.txt'), 'Only explicitly pinned paired owned control metadata')
            files.append(deepcopy(item))
        require(len(files) <= CAPS['files_per_archive'] and len({x['path'] for x in files}) == len(files),
                'Unique complete bounded byte inventory')
        require(sum(x['bytes'] for x in files) + CAPS['header_bytes'] <= CAPS['raw_frame_bytes'], 'Raw archive admission before allocation')
        return sorted(files, key=lambda x: x['path']), sorted(directories)

    def _protected(self, folder, files):
        require(str(folder) not in self._state['protected_folders'], 'Protected current/recovery/DEV/winner worker cannot retire')
        protected = {x['path'] for x in self._state['protected_files']}
        require(not any(x['path'] in protected for x in files), 'Protected paired configuration/pause/proof cannot retire')

    def archive(self, folder, *, process0, paired_files=(), evidence, verify_closed):
        """Normal CLOSED0 only. Any crash between append/index/unlink stays blocked."""
        self._state = self._load()
        require(self._verified_protection, 'Fresh independently verified protection before archive')
        path = safe(folder); matches = [x for x in self._state['live'] if x['folder'] == str(path)]
        require(len(matches) == 1 and path.is_dir(), 'Only a registered future-owned worker directory')
        require(len(paired_files) <= CAPS['paired_files'], 'Bounded explicit paired control list')
        bound(process0, self.guard, CAPS['member_bytes']); bound(evidence, self.guard, CAPS['member_bytes'])
        from fastmss_owned_process_closure_v2 import verify
        completed = read_json(process0['path'])
        verify(completed, kind='ASR18_owned_GPU_process_completion_v1', owner='/root/memory_diagnosis', guard=self.guard)
        require(same(completed['worker_source'], matches[0]['worker_source'])
                and same(completed['parent_source'], matches[0]['parent_source'])
                and Path(process0['path']).resolve().is_relative_to(path)
                and Path(completed['report']['path']).resolve().is_relative_to(path),
                'Actual dead/source-bound process and canonical report belong to this reservation')
        files, directories = self._inventory(path, paired_files); self._protected(path, files)
        require(matches[0]['configuration'] in files, 'Explicit paired reservation configuration must be preserved in archive')
        request = self._request(operation='verify_closed_worker_control_retirement', reservation=matches[0],
            process0=process0, evidence=evidence, files=files, directories=directories,
            normal_coherent_boundary_and_mapping_release_required=True,
            crash_tail_rollback_or_original_artifact_deletion_permitted=False)
        _callback(self._verify_protection, self.authority_source,
                  self._request(operation='reverify_current_worker_control_protection', evidence=self._protection_evidence))
        _callback(verify_closed, self.authority_source, request)
        count = self._state['archive_count'] + 1
        require(count <= CAPS['archive_count'] and self._state['member_count'] + len(files) <= CAPS['member_rows'],
                'Cumulative archive/index quota never resets on supervisor restart')
        manifest = {'kind': KIND, 'owner': self.owner_pin, 'archive_id': count,
                    'reservation': matches[0], 'process0': process0, 'authority_evidence': evidence,
                    'protection_evidence': self._protection_evidence, 'files': files, 'directories': directories}
        header = canonical(manifest)
        require(len(header) <= CAPS['header_bytes'], 'Complete archive header, no omitted fields')
        raw_bytes = 4 + len(header) + sum(x['bytes'] for x in files)
        raw_hash = hashlib.sha256(); compressor = zlib.compressobj(6, zlib.DEFLATED, 31)
        # Charge maximum new frame + simultaneous compressed staging + full
        # SQLite rollback/index growth + bounded atomic registry scratch before
        # creating any of them. This is conservative fixed-domain admission,
        # not an increase to the archive's own 128 MiB durable cap.
        self._admit(self.archive_path, self._state['archive_bytes'] + HEADER.size
                    + 2 * CAPS['archive_frame_bytes'] + CAPS['index_bytes'] + 2 * 1024**2)
        with self.metadata_phase('read_index'):
            with self._temporary() as temp:
                def emit(raw):
                    self.guard(); raw_hash.update(raw); block = compressor.compress(raw)
                    require(temp.tell() + len(block) <= CAPS['archive_frame_bytes'], 'Compression hard cap before write')
                    temp.write(block)
                emit(struct.pack('<I', len(header))); emit(header)
                for item in files:
                    require(same(pin(item['path'], self.guard), item), 'Unchanged source before archival read')
                    with Path(item['path']).open('rb') as source:
                        while True:
                            block = source.read(CHUNK)
                            if not block:
                                break
                            emit(block)
                tail = compressor.flush()
                require(temp.tell() + len(tail) <= CAPS['archive_frame_bytes'], 'Complete bounded compression')
                temp.write(tail); temp.flush(); os.fsync(temp.fileno())
                length = temp.tell(); offset = self._state['archive_bytes']
                frame = HEADER.pack(length, raw_bytes, raw_hash.digest())
                require(offset + HEADER.size + length <= CAPS['archive_stream_bytes'], 'Fixed-control archive capacity before append')
                temp.seek(0)
                with self.archive_path.open('ab') as target:
                    target.write(frame)
                    while True:
                        self.guard(); block = temp.read(CHUNK)
                        if not block:
                            break
                        target.write(block)
                    target.flush(); os.fsync(target.fileno())
        descriptor = {'archive_id': count, 'offset': offset, 'compressed_bytes': length,
                      'raw_bytes': raw_bytes, 'raw_sha256': raw_hash.hexdigest()}
        self._read_archive(descriptor, expected=manifest)
        _callback(verify_closed, self.authority_source, request)
        require(self._inventory(path, paired_files) == (files, directories), 'Exact inventory stable immediately before retirement')
        self._protected(path, files)
        state = deepcopy(self._state); state['pending_retirement'] = descriptor
        self._save(state)
        db = self._db()
        try:
            db.execute('INSERT INTO archives VALUES (?,?,?)', (count, canonical(descriptor).decode(), 'retiring'))
            db.executemany('INSERT INTO members VALUES (?,?,?,?)',
                [(x['path'], x['bytes'], x['sha256'], count) for x in files]); db.commit()
            self._retiring_paths = frozenset(x['path'] for x in files)
            for item in files:
                self.guard(); require(same(pin(item['path'], self.guard), item), 'Only exact archived bytes may unlink')
                Path(item['path']).unlink()
            for directory in sorted(directories, key=lambda x: len(Path(x).parts), reverse=True):
                safe(directory, path).rmdir()
            path.rmdir()
            db.execute('UPDATE archives SET status="complete" WHERE id=?', (count,)); db.commit()
        finally:
            db.close(); self._sqlite_writers -= 1
        state['live'] = [x for x in state['live'] if x['folder'] != str(path)]
        state['archive_count'] = count; state['member_count'] += len(files)
        state['archive_bytes'] = offset + HEADER.size + length; state['pending_retirement'] = None
        self._save(state)
        self._retiring_paths = frozenset()
        return {'kind': 'ASR18_exact_closed_worker_controls_archived_v1', 'passed': True,
                'owner': self.owner_pin, 'descriptor': descriptor, 'original_folder': str(path),
                'all_original_member_bytes_verified_before_exact_unlink': True,
                'protected_original_or_checkpoint_files_changed': False,
                'live_folders': [str(x) for x in self.live_folders()]}

    def _read_archive(self, descriptor, *, expected=None, selected=None, output=None):
        """Bounded GZIP framing; consume+hash ALL members before yielding one."""
        require(descriptor['compressed_bytes'] <= CAPS['archive_frame_bytes']
                and descriptor['raw_bytes'] <= CAPS['raw_frame_bytes'], 'Bounded indexed frame')
        with self.archive_path.open('rb') as source:
            source.seek(descriptor['offset']); frame = source.read(HEADER.size)
            require(len(frame) == HEADER.size, 'Complete archive framing')
            length, raw_length, checksum = HEADER.unpack(frame)
            require(length == descriptor['compressed_bytes'] and raw_length == descriptor['raw_bytes']
                    and checksum.hex() == descriptor['raw_sha256'], 'Exact indexed frame header')
            decoder = zlib.decompressobj(31); remaining = length; buffer = b''
            consumed = 0; whole = hashlib.sha256()
            def take(count):
                nonlocal remaining, buffer, consumed
                require(0 < count <= CHUNK, 'Bounded archive read')
                self.guard()
                while len(buffer) < count:
                    self.guard()
                    if decoder.unconsumed_tail:
                        data = decoder.unconsumed_tail
                    elif remaining:
                        data = source.read(min(65536, remaining)); require(data, 'Compressed EOF'); remaining -= len(data)
                    else:
                        break
                    block = decoder.decompress(data, min(CHUNK, count - len(buffer)))
                    buffer += block
                    require(consumed + len(buffer) <= raw_length, 'Decompressed cumulative hard cap')
                    if decoder.eof:
                        require(not decoder.unused_data and not remaining, 'No trailing stream/member')
                        break
                result, buffer = buffer[:count], buffer[count:]
                require(len(result) == count, 'Complete exact raw member')
                whole.update(result); consumed += len(result)
                return result
            header_length = struct.unpack('<I', take(4))[0]
            require(0 < header_length <= CAPS['header_bytes'], 'Bounded complete manifest')
            manifest = json.loads(take(header_length).decode('utf-8'))
            require(manifest['kind'] == KIND and same(manifest['owner'], self.owner_pin)
                    and manifest['archive_id'] == descriptor['archive_id']
                    and len(manifest['files']) <= CAPS['files_per_archive'], 'Exact source-owned archive manifest')
            if expected is not None:
                require(same(manifest, expected), 'Readback preserved complete archive metadata')
            found = 0
            for item in manifest['files']:
                self.guard()
                require(type(item['bytes']) is int and 0 <= item['bytes'] <= CAPS['member_bytes'], 'Bounded raw member')
                h = hashlib.sha256(); left = item['bytes']; choose = selected is not None and same(item, selected)
                if choose:
                    found += 1
                while left:
                    block = take(min(left, CHUNK)); left -= len(block); h.update(block)
                    if choose:
                        output.write(block)
                require(h.hexdigest() == item['sha256'], 'Lossless full original member hash')
            # Force GZIP trailer processing even when last exact read ended at
            # its output boundary; no ignored trailer, extra stream or raw byte.
            while not decoder.eof:
                self.guard()
                data = decoder.unconsumed_tail
                if not data and remaining:
                    data = source.read(min(65536, remaining)); require(data, 'GZIP trailer EOF'); remaining -= len(data)
                require(data, 'Missing complete GZIP trailer')
                require(decoder.decompress(data, 1) == b'', 'No extra unmanifested raw byte')
            require(not decoder.unused_data and remaining == 0 and not buffer
                    and consumed == raw_length and whole.hexdigest() == descriptor['raw_sha256'],
                    'Entire exact frame EOF/hash verified')
            if selected is not None:
                require(found == 1, 'Unique exact indexed member')
            return manifest

    def _member_descriptor(self, binding):
        require(type(binding) is dict and set(binding) == {'path', 'bytes', 'sha256'}
                and type(binding['bytes']) is int and 0 <= binding['bytes'] <= CAPS['member_bytes']
                and type(binding['sha256']) is str and re.fullmatch('[0-9a-f]{64}', binding['sha256']),
                'Bounded exact original metadata binding')
        path = safe(binding['path'])
        db = self._read_db()
        try:
            row = db.execute('SELECT a.descriptor FROM members m JOIN archives a ON a.id=m.archive_id '
                'WHERE m.path=? AND m.bytes=? AND m.sha256=? AND a.status="complete"',
                (str(path), binding['bytes'], binding['sha256'])).fetchone()
            return None if row is None else json.loads(row[0])
        finally:
            db.close()

    def owns_member(self, binding):
        """Only a full exact committed record3 lookup, never a path-only claim."""
        return self._member_descriptor(binding) is not None

    def is_retiring_path(self, path):
        return str(Path(path).absolute()) in self._retiring_paths

    def is_archived_path(self, path):
        """Narrow missing-stat observer aid; never authority to read/delete a file."""
        path = safe(path)
        db = self._read_db()
        try:
            return db.execute('SELECT 1 FROM members m JOIN archives a ON a.id=m.archive_id '
                'WHERE m.path=? AND a.status="complete" LIMIT 1', (str(path),)).fetchone() is not None
        finally:
            db.close()

    @contextmanager
    def resolve_member(self, binding):
        """Exact original record3 only; no arbitrary path or history materialization."""
        descriptor = self._member_descriptor(binding)
        require(descriptor is not None, 'Only exact committed source-owned archived member lookup')
        path = safe(binding['path'])
        if path.exists():
            bound(binding, self.guard, CAPS['member_bytes'])
            with path.open('rb') as stream:
                yield _LimitedReader(stream, self.guard)
            bound(binding, self.guard, CAPS['member_bytes'])
            return
        self._admit(self.archive_path, self.archive_path.stat().st_size + binding['bytes'] + 1024**2)
        with self.metadata_phase('read_index'):
            with self._temporary() as temp:
                self._read_archive(descriptor, selected=binding, output=temp)
                temp.flush(); require(temp.tell() == binding['bytes'], 'Exact selected size'); temp.seek(0)
                yield _LimitedReader(temp, self.guard)

    def close(self):
        if self._lock is not None:
            import msvcrt
            try:
                if self._locked:
                    self._lock.seek(0); msvcrt.locking(self._lock.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                self._lock.close(); self._lock = None; self._locked = False
