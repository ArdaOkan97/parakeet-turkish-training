"""Inactive bounded original/archived metadata resolver for the sole owner.

Only exact registered auxiliary members or an explicitly archived byte-identical
temporary control-plan alias can replace a missing original metadata path. This
does not search history, recreate old paths, retire artifacts or mutate a store.
Any temporary selected-member spool is newly owned under the explicit E control
directory and is released by its own handle on context exit.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

from bpe_broad_production_state_v1 import digest, require
from bpe_broad_audio_v1 import record


ROOT = Path(r'E:\gpu_training_scratchpad')
MAXIMUM = 16 * 1024**2
CHUNK = 1024**2
KEY = re.compile(r'p[0-9]{9}')


def _path(value):
    require(type(value) is str, 'Exact absolute E path required')
    path = Path(value)
    require(path.is_absolute() and path.resolve().is_relative_to(ROOT.resolve())
            and str(path.resolve()) == value, 'No fallback or aliased metadata path')
    for part in (path, *path.parents):
        if part.exists():
            info = part.lstat()
            require(not part.is_symlink() and not getattr(info, 'st_file_attributes', 0)
                    & stat.FILE_ATTRIBUTE_REPARSE_POINT, 'No reparse ownership escape')
        if part == ROOT.resolve(): break
    return path


def _pin(pin, maximum=MAXIMUM):
    require(type(pin) is dict and set(pin) == {'path', 'bytes', 'sha256'}
            and type(pin['bytes']) is int and 0 < pin['bytes'] <= maximum
            and type(pin['sha256']) is str and re.fullmatch(r'[0-9a-f]{64}', pin['sha256']),
            'Exact bounded original record3')
    _path(pin['path'])
    return pin


def _json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'Duplicate alias JSON key'); out[key] = value
        return out
    def invalid(value): raise ValueError('Nonfinite alias JSON: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


class _ReadOnlyChunks:
    def __init__(self, stream, guard): self._stream = stream; self._guard = guard
    def read(self, count):
        require(type(count) is int and 0 <= count <= CHUNK, 'Bounded binary metadata read only')
        self._guard()
        return self._stream.read(count)


class MetadataResolver:
    def __init__(self, store, *, control_root, guard, metadata_phase):
        from asr18_BPE9223_indexed_checkpoint_storage_v1 import IndexedOwnedFutureCheckpointStore
        require(type(store) is IndexedOwnedFutureCheckpointStore
                and callable(guard) and callable(metadata_phase), 'Exact public indexed3 store and guarded phase API')
        source = Path(__file__).with_name('asr18_BPE9223_indexed_checkpoint_storage_v1.py')
        require(record(source)['sha256'] == 'f1f37f963fce6324e484826ed8e61b024b084919e74ea71a8b0bd52637b7c415',
                'Frozen indexed3 archive implementation')
        self.store = store; self.guard = guard; self.metadata_phase = metadata_phase
        self.control_root = _path(str(control_root))
        require(self.control_root.is_dir(), 'Explicit existing E-only control/spool directory')
        self.namespace = _path(str(store.folder))
        _pin(store.marker_pin)
        require(Path(store.marker_pin['path']) == self.namespace/'ownership.json', 'Exact registered storage owner')

    def _key(self, original):
        path = _path(original['path']); transient = self.namespace/'transient'
        if path.is_relative_to(transient):
            parts = path.relative_to(transient).parts
            require(len(parts) >= 2 and KEY.fullmatch(parts[0]), 'Canonical exact registered auxiliary member path')
            return parts[0], False
        require(path.parent == self.control_root and path.name.endswith('.plan.json'),
                'Missing metadata is neither registered auxiliary nor exact owned control plan')
        key = path.name[:-len('.plan.json')]
        require(KEY.fullmatch(key) and path == self.control_root/(key + '.plan.json'), 'Canonical temporary control plan identity')
        return key, True

    def _manifest(self, key):
        # Same sole owner; no retained whole-history map or arbitrary filesystem lookup.
        with self.store.locked():
            self.store._load()
            item = self.store._auxiliary(key, verify_archive=True)
            require(item['id'] == key and item['status'] in ('archived', 'retired')
                    and Path(item['workspace']) == self.namespace/'transient'/key,
                    'Only exactly registered and verified completed auxiliary archives')
            manifest = self.store._verify_auxiliary_archive(item)
        return item, manifest

    @contextmanager
    def _archived(self, key, member):
        _pin(member)
        # Anonymous, explicitly E-local, selected-member spool only. No old
        # artifact is unlinked, overwritten, adopted or given a new identity.
        with tempfile.TemporaryFile(mode='w+b', dir=str(self.control_root),
                                    prefix='owned_member_', suffix='.tmp') as temporary:
            count = 0; hashed = hashlib.sha256()
            def consume(block):
                nonlocal count
                self.guard()
                require(type(block) is bytes and 0 < len(block) <= CHUNK
                        and count + len(block) <= member['bytes'] <= MAXIMUM,
                        'Bounded selected original member bytes before allocation/write')
                require(temporary.write(block) == len(block), 'Complete owned temporary metadata write')
                count += len(block); hashed.update(block)
            proof = self.store.stream_archived_auxiliary_member(key, member, consume)
            require(proof['passed'] is True and proof['original_logical_file'] == member
                    and proof['bytes_delivered'] == count == member['bytes']
                    and hashed.hexdigest() == member['sha256']
                    and proof['original_path_recreated'] is False and proof['any_file_deleted_or_adopted'] is False,
                    'Exact registered original member SHA/length/EOF, not a path substitution')
            temporary.flush(); temporary.seek(0)
            yield _ReadOnlyChunks(temporary, self.guard)

    def _alias_member(self, key, original, item, manifest):
        workspace = self.namespace/'transient'/key
        aliases = [row['file'] for row in manifest['files']
                   if row['file']['path'] == str(workspace/'plan_alias.json')]
        require(len(aliases) == 1, 'One explicitly registered control-plan alias receipt')
        alias_pin = _pin(aliases[0], 65536)
        with self._archived(key, alias_pin) as source:
            raw = source.read(alias_pin['bytes']); require(source.read(1) == b'', 'Exact complete alias EOF')
        alias = _json(raw)
        require(type(alias) is dict and alias['kind'] == 'ASR18_exact_owned_control_plan_archive_alias_v1'
                and alias['publication_id'] == key and alias['original'] == original
                and alias['original'] == item['plan']
                and Path(alias['original']['path']) == self.control_root/(key + '.plan.json'),
                'Exact registered original control-plan alias, no metadata adoption')
        copied = _pin(alias['copied'])
        require(copied['path'] == str(workspace/'coordinator_plan_copy.json')
                and copied['bytes'] == original['bytes'] and copied['sha256'] == original['sha256']
                and sum(row['file'] == copied and digest(row['file']) == digest(copied)
                        for row in manifest['files']) == 1,
                'One archived byte-identical complete copy of this exact original plan')
        return copied

    @contextmanager
    def __call__(self, binding):
        with self.metadata_phase('read_index'):
            self.guard(); descriptor = None; original = binding
            if type(binding) is dict and binding.get('kind') == 'ASR18_indexed_auxiliary_original_member_v1':
                require(set(binding) == {'kind', 'original', 'storage_owner', 'auxiliary_id'}
                        and binding['storage_owner'] == self.store.marker_pin,
                        'Exact source-bound original-member descriptor/registered owner')
                descriptor = deepcopy(binding); original = descriptor['original']
            original = _pin(original); path = _path(original['path'])
            if path.exists() and descriptor is None:
                require(path.is_file(), 'No directory as metadata')
                with path.open('rb') as source:
                    from asr18_BPE9223_indexed_checkpoint_storage_v1 import _IndexedProtocol
                    creation = _IndexedProtocol._creation_identity(source)
                    info = os.fstat(source.fileno()); before_stat = (info.st_size, info.st_mtime_ns)
                    before = source.seek(0, 2); source.seek(0); hashed = hashlib.sha256(); count = 0
                    require(before == original['bytes'], 'Original regular metadata byte count')
                    while block := source.read(CHUNK):
                        self.guard(); count += len(block); hashed.update(block)
                        require(count <= original['bytes'], 'No growing or additional original metadata')
                    require(count == original['bytes'] and hashed.hexdigest() == original['sha256'],
                            'Actual original regular metadata SHA')
                    source.seek(0); yield _ReadOnlyChunks(source, self.guard)
                    require(_IndexedProtocol._creation_identity(source) == creation
                            and source.seek(0, 2) == before, 'Open original metadata identity/size changed')
                    after = os.fstat(source.fileno())
                    require((after.st_size, after.st_mtime_ns) == before_stat, 'Original metadata mutated during read')
                    with path.open('rb') as current:
                        require(_IndexedProtocol._creation_identity(current) == creation,
                                'Original metadata path replaced during read')
                return
            key, alias = self._key(original)
            if descriptor is not None:
                require(descriptor['auxiliary_id'] == key, 'Descriptor auxiliary ID differs from exact original path')
            item, manifest = self._manifest(key)
            member = self._alias_member(key, original, item, manifest) if alias else original
            require(sum(row['file'] == member and digest(row['file']) == digest(member)
                        for row in manifest['files']) == 1, 'Only one exact registered lossless original member')
            with self._archived(key, member) as stream:
                yield stream
