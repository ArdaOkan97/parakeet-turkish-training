"""Run-local append-only journals with a shared byte quota and exact resume.

No model, optimizer, sampler, or process launch. A reviewed caller supplies the
actual native-exit verifier. Unknown tails are preserved, never truncated or
replayed. Reuse avoids copying the growing journal at every clean DEV boundary.
"""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
from bpe_broad_production_state_v1 import canonical, require
from bpe_broad_audio_v1 import record
from atomic_io_retry_v1 import read_json, write_json

ROOT = Path(r'E:\gpu_training_scratchpad').resolve()
QUOTA = 16 * 1024**3
OTHER_CONTROL_QUOTA = 4 * 1024**3
MAX_JOURNALS = 513
CHUNK = 1024**2


def safe(path, parent):
    """Reject reparse points before following a path under the owned namespace."""
    raw = Path(path).absolute()
    require(raw.is_relative_to(parent), 'Journal path outside exact owned namespace')
    cursor = raw
    while cursor != parent.parent:
        if cursor.exists():
            require(not (getattr(cursor.lstat(), 'st_file_attributes', 0) & 1024),
                    'Reparse journal paths are unsupported')
        if cursor == parent: break
        cursor = cursor.parent
    require(raw.resolve() == raw, 'Canonical journal path required')
    return raw


def prefix(path, count, guard):
    require(type(count) is int and count >= 0, 'Typed nonnegative prefix length')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        remaining = count
        while remaining:
            guard(); block = stream.read(min(CHUNK, remaining))
            require(block, 'Truncated journal prefix')
            remaining -= len(block); digest.update(block)
    return digest


class JournalNamespace:
    """One source-bound writer; registry changes only when creating a branch.

    Startup observes every registered file (at most 513). Per-update quota work
    is constant-size and checks the live journal size plus immutable other-file
    sizes captured at startup. The caller must retain sole-writer ownership.
    """
    def __init__(self, root, *, identity, guard, create=False):
        self.root = safe(root, ROOT / 'runs')
        self.guard = guard
        self.identity = deepcopy(identity)
        require(type(identity) is dict and identity and callable(guard),
                'Explicit run identity and live caller guard')
        self.registry = self.root / 'journal_registry.json'
        self.owner = self.root / 'journal_owner.json'
        if create:
            require(not self.root.exists(), 'Fresh owned journal namespace only')
            self.root.mkdir(parents=True)
            write_json(self.owner, {'kind': 'ASR18_bounded_journal_owner_v1',
                'identity': identity, 'source': record(__file__),
                'journal_quota_bytes': QUOTA, 'separate_other_control_quota_bytes': OTHER_CONTROL_QUOTA,
                'maximum_journals': MAX_JOURNALS})
            write_json(self.registry, {'kind': 'ASR18_bounded_journal_registry_v1', 'paths': []})
        value = read_json(self.owner)
        require(value == {'kind': 'ASR18_bounded_journal_owner_v1', 'identity': identity,
                'source': record(__file__), 'journal_quota_bytes': QUOTA,
                'separate_other_control_quota_bytes': OTHER_CONTROL_QUOTA,
                'maximum_journals': MAX_JOURNALS}, 'Exact source-bound journal owner')
        self.owner_pin = record(self.owner)
        self._load()

    def _load(self):
        value = read_json(self.registry)
        require(set(value) == {'kind', 'paths'} and value['kind'] == 'ASR18_bounded_journal_registry_v1'
                and type(value['paths']) is list and len(value['paths']) <= MAX_JOURNALS
                and len(set(value['paths'])) == len(value['paths']), 'Bounded unique journal registry')
        self.paths = [safe(path, self.root) for path in value['paths']]
        require(all(path.parent == self.root and path.name.startswith('commits_')
                    and path.suffix == '.jsonl' and path.is_file() for path in self.paths),
                'Only explicitly registered direct journal files')
        self.sizes = {str(path): path.stat().st_size for path in self.paths}
        require(sum(self.sizes.values()) <= QUOTA, 'Shared journal quota already exceeded')
        self.registry_pin = record(self.registry)

    def check(self):
        self.guard()
        require(record(self.owner) == self.owner_pin and record(self.registry) == self.registry_pin,
                'Source owner or branch registry changed under the sole writer')

    def register(self, name, *, prospective_bytes=0):
        self.check()
        require(type(prospective_bytes) is int and prospective_bytes >= 0
                and len(self.paths) < MAX_JOURNALS, 'Bounded branch count and bytes')
        path = safe(self.root / name, self.root)
        require(path.parent == self.root and path.name.startswith('commits_')
                and path.suffix == '.jsonl' and not path.exists(), 'Fresh named journal branch')
        actual = sum(item.stat().st_size for item in self.paths)
        require(actual + prospective_bytes <= QUOTA, 'Branch copy exceeds aggregate journal quota')
        with path.open('xb') as stream:
            stream.flush(); os.fsync(stream.fileno())
        write_json(self.registry, {'kind': 'ASR18_bounded_journal_registry_v1',
                                  'paths': [str(item) for item in self.paths] + [str(path)]})
        self._load()
        return path

    def writer(self, path):
        self.active_path = safe(path, self.root)
        return BoundedCommitJournal(self, self.active_path)

    def resume(self, saved, *, destination_name, actual_exit, verify_quiescence):
        """Reuse a whole coherent journal, otherwise create one bounded prefix.

        verify_quiescence(actual_exit, saved) must verify exact source-bound real
        wait0/dead PID+create-time and the native report's final checkpoint.
        Metadata existence or a caller-supplied boolean is not that evidence.
        """
        self.check()
        require(type(saved) is dict and set(saved) == {'path', 'offset', 'sha256'}
                and type(saved['offset']) is int and saved['offset'] >= 0,
                'Exact checkpoint journal prefix')
        source = safe(saved['path'], self.root)
        require(source in self.paths and callable(verify_quiescence), 'Registered source and explicit exit verifier')
        require(verify_quiescence(deepcopy(actual_exit), deepcopy(saved)) is True,
                'Actual coherent native0/dead/source/current-checkpoint proof required')
        before = source.stat()
        require(saved['offset'] <= before.st_size, 'Saved prefix is longer than journal')
        hashed = prefix(source, saved['offset'], self.guard)
        require(hashed.hexdigest() == saved['sha256'], 'Saved prefix SHA differs')
        require((before.st_size, before.st_mtime_ns) == (source.stat().st_size, source.stat().st_mtime_ns),
                'Closed journal changed during verification')
        reused = saved['offset'] == before.st_size
        if reused:
            destination = source
        else:
            destination = self.register(destination_name, prospective_bytes=saved['offset'])
            with source.open('rb') as original, destination.open('ab') as output:
                remaining = saved['offset']
                while remaining:
                    self.check(); block = original.read(min(CHUNK, remaining))
                    require(block, 'Source prefix truncated during branch copy')
                    output.write(block); remaining -= len(block)
                output.flush(); os.fsync(output.fileno())
            require((before.st_size, before.st_mtime_ns) == (source.stat().st_size, source.stat().st_mtime_ns),
                    'Source mutated during bounded branch copy')
            self._load()
        journal = self.writer(destination)
        require(journal.binding()['offset'] == saved['offset']
                and journal.binding()['sha256'] == saved['sha256'], 'Authoritative resume prefix differs')
        return journal, {'kind': 'ASR18_bounded_journal_resume_v1', 'source': record(__file__),
            'identity': deepcopy(self.identity), 'source_checkpoint_prefix': deepcopy(saved),
            'actual_exit': deepcopy(actual_exit), 'same_whole_closed_journal_reused': reused,
            'copied_bytes': 0 if reused else saved['offset'],
            'unknown_tail_preserved_bytes': before.st_size - saved['offset'],
            'original_journal_modified': False, 'tail_replayed_or_sampler_advanced': False,
            'journal': journal.binding(), 'journal_quota_bytes': QUOTA}


class BoundedCommitJournal:
    def __init__(self, namespace, path):
        namespace.check()
        require(path in namespace.paths, 'Journal must already be explicitly registered')
        self.namespace, self.path = namespace, path
        self._offset = path.stat().st_size
        self._hash = prefix(path, self._offset, namespace.guard)
        self.other_bytes = sum(size for key, size in namespace.sizes.items() if key != str(path))

    def append(self, event):
        self.namespace.check(); self.binding()
        require(self.namespace.active_path == self.path,
                'An older writer cannot append after selecting a different branch')
        raw = canonical(event) + b'\n'
        require(len(raw) <= 128 * 1024, 'Native event exceeds explicit bounded serialization')
        require(self.other_bytes + self._offset + len(raw) <= QUOTA,
                'Stop before journal write: aggregate16GiB journal quota')
        with self.path.open('ab') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        self._hash.update(raw); self._offset += len(raw)
        return self.binding()

    def binding(self):
        require(self.path.stat().st_size == self._offset, 'Concurrent journal writer is prohibited')
        return {'path': str(self.path), 'offset': self._offset, 'sha256': self._hash.hexdigest()}

    def verify(self, value):
        require(set(value) == {'path', 'offset', 'sha256'} and Path(value['path']).resolve() == self.path
                and type(value['offset']) is int and 0 <= value['offset'] <= self.path.stat().st_size,
                'Exact registered checkpoint prefix required')
        return prefix(self.path, value['offset'], self.namespace.guard).hexdigest() == value['sha256']
