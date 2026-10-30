"""Inactive production CURRENT recovery and durable persistence-prefix proofs.

No tensors, model, runtime, process launch, cache mutation or historical GPU
restore. The parent supplies an independently source-pinned bounded metadata
resolver. Completed window frames and actual journal prefixes are streamed;
one current fresh recovery discharges the pending production prefix only.
Finite diagnostic every-capture recovery is deliberately not accepted here.
"""
from copy import deepcopy
from asr18_BPE9223_resource_operational_compatibility_v1 import checked_compatibility, checked_resource_stop
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import stat
import struct

ROOT = Path(r'E:\gpu_training_scratchpad')
PROJECT = Path(__file__).resolve().parents[2]
OWNER = '/root/memory_diagnosis'
MAX_METADATA = 16 * 1024**2
MAX_FRAME = 65536
CHUNK = 1024**2
MAX_PUBLICATIONS = 95742
MAX_JOURNAL = 16 * 1024**3
SOURCE_KEYS = frozenset(('prefix_verifier', 'metadata_resolver', 'native_worker', 'native_parent',
    'recovery_worker', 'recovery_parent', 'persistence_checker', 'process_validator',
    'durable_adapter', 'indexed_storage', 'streamed_semantics'))
FIXED = {
 'persistence_checker': ('asr18_streamed_persistence_fingerprint_v1.py', 'a7a58cd99e2805cf4997734d75b7f7ca9996177dcb2a4a5a7f7fd9892dfcd3e3'),
 'process_validator': ('fastmss_owned_process_closure_v2.py', 'a720292ce528931a63b4f83e1e7e7360cdd50ebe30ad27986f42066c5c3e3bc8'),
 'durable_adapter': ('asr18_BPE9223_durable_owner_adapter_v3.py', '82871d243432210c50393f68a4aa3b2eac93fbb1289bc086476c420bb5afca56'),
 'indexed_storage': ('asr18_BPE9223_indexed_checkpoint_storage_v1.py', 'f1f37f963fce6324e484826ed8e61b024b084919e74ea71a8b0bd52637b7c415'),
 'streamed_semantics': ('bpe_streaming_checkpoint_semantics_v2.py', 'ce721bcbeaa85b9c59d5230dc81f5a6fd85320b938c6c4d8910c2eec1ff2d309')}


def require(value, message):
    if not value: raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf8')


def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def same(a, b): return a == b and digest(a) == digest(b)
def sha(value): return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def integer(value, low=0, high=None):
    require(type(value) is int and value >= low and (high is None or value <= high), 'Typed bounded integer required')
    return value


def path_in_scope(value, *, source=False):
    path = Path(value).resolve()
    require(path.is_relative_to(ROOT.resolve()) or source and path.is_relative_to(PROJECT), 'E artifacts/existing project source only')
    require(str(path) == value, 'Exact resolved absolute path required')
    for parent in (path, *path.parents):
        if parent.exists():
            require(not parent.is_symlink() and not getattr(parent.lstat(), 'st_file_attributes', 0)
                    & stat.FILE_ATTRIBUTE_REPARSE_POINT, 'No reparse ownership escape')
        if parent in (ROOT.resolve(), PROJECT): break
    return path


def record3(value):
    require(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'} and sha(value['sha256']), 'Exact original record3 required')
    integer(value['bytes'], 1)
    path_in_scope(value['path'], source=True)
    return value


def _json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'Duplicate JSON key'); out[key] = value
        return out
    def invalid(value): raise ValueError('Nonfinite JSON: '+value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def _source(pin, guard):
    record3(pin); require(pin['bytes'] <= 256 * 1024, 'Bounded source identity')
    path = path_in_scope(pin['path'], source=True); before = path.stat(); hashed = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(65536): guard(); hashed.update(block)
    require(before.st_size == pin['bytes'] and hashed.hexdigest() == pin['sha256']
            and (before.st_size, before.st_mtime_ns) == (path.stat().st_size, path.stat().st_mtime_ns), 'Source binding changed')


def diagnostic_identity(identity):
    """Explicit qualification namespace; never infer a missing boolean."""
    require(type(identity) is dict and type(identity.get('owner_validation_only')) is bool,
            'Typed immutable owner_validation_only required')
    return identity['owner_validation_only']


class _Metadata:
    def __init__(self, identity, resolve_member, guard, operational_compatibility):
        self.diagnostic = diagnostic_identity(identity)
        sources = identity['persistent_prefix_sources']
        require(type(sources) is dict and set(sources) == SOURCE_KEYS, 'Exact immutable ROOT-reviewed prefix source map required')
        for pin in sources.values(): _source(pin, guard)
        require(Path(sources['prefix_verifier']['path']) == Path(__file__).with_name('asr18_BPE9223_persistence_prefix_v4.py').resolve()
                and sources['prefix_verifier']['sha256'] == hashlib.sha256(Path(sources['prefix_verifier']['path']).read_bytes()).hexdigest(), 'Exact prefix verifier source')
        for key, (name, checksum) in FIXED.items():
            require(Path(sources[key]['path']) == Path(__file__).with_name(name).resolve()
                    and sources[key]['sha256'] == checksum, 'Frozen proof dependency differs: '+key)
        self.compatibility = deepcopy(operational_compatibility)
        self.operational = checked_compatibility(self.compatibility, identity, guard)
        require(same(self.operational['operational_sources']['prefix'], _record_self()), 'Actual operational prefix source')
        function = inspect.unwrap(resolve_member)
        if not inspect.isfunction(function) and not inspect.ismethod(function):
            function = inspect.unwrap(function.__call__)
        actual_resolver_source = self.operational['operational_sources']['controller']
        _source(actual_resolver_source, guard)
        require(Path(inspect.getsourcefile(function)).resolve() == Path(actual_resolver_source['path']),
                'Actual operational controller resolver is separate from immutable historical callback map')
        self.sources, self.resolve, self.guard = sources, resolve_member, guard

    def read(self, binding, maximum=MAX_METADATA):
        pin = binding
        if type(binding) is dict and binding.get('kind') == 'ASR18_indexed_auxiliary_original_member_v1':
            require(set(binding) == {'kind', 'original', 'storage_owner', 'auxiliary_id'}, 'Exact archived-member descriptor')
            record3(binding['storage_owner']); require(type(binding['auxiliary_id']) is str and binding['auxiliary_id'], 'Exact auxiliary ID')
            pin = binding['original']
        record3(pin); integer(pin['bytes'], 1, maximum)
        path_in_scope(pin['path'])
        out = bytearray(); hashed = hashlib.sha256()
        with self.resolve(deepcopy(binding)) as stream:
            while len(out) < pin['bytes']:
                self.guard(); block = stream.read(min(CHUNK, pin['bytes'] - len(out)))
                require(type(block) is bytes and 0 < len(block) <= min(CHUNK, pin['bytes'] - len(out)), 'Bounded complete binary metadata stream')
                out.extend(block); hashed.update(block)
            require(stream.read(1) == b'', 'No trailing member bytes')
        require(hashed.hexdigest() == pin['sha256'], 'Resolved original metadata SHA differs')
        return _json(out)


def _descriptor(value):
    require(type(value) is dict and set(value) == {'kind', 'path', 'offset', 'framed_bytes', 'sha256', 'sequence'}
            and value['kind'] == 'ASR18_logical_immutable_durable_head_descriptor_v2' and sha(value['sha256']), 'Exact logical durable descriptor')
    path_in_scope(value['path']); integer(value['offset']); integer(value['framed_bytes'], 41, MAX_FRAME); integer(value['sequence'], 1)
    return value


def _frame(pin, identity_sha, owner, guard):
    _descriptor(pin)
    with path_in_scope(pin['path']).open('rb') as stream:
        stream.seek(pin['offset']); header = stream.read(40); raw = stream.read(pin['framed_bytes'] - 40)
    guard()
    require(len(header) == 40 and len(raw) == pin['framed_bytes'] - 40
            and struct.unpack('<Q', header[:8])[0] + 40 == pin['framed_bytes']
            and hashlib.sha256(raw).digest() == header[8:] and hashlib.sha256(raw).hexdigest() == pin['sha256'], 'Complete exact immutable frame')
    value = _json(raw); head = value['head']
    pending = head['pending']
    require(type(pending) is dict and set(pending) == {'windows', 'publications', 'chain_sha256', 'first_optimizer_update', 'last_optimizer_update'}
            and sha(pending['chain_sha256']), 'Exact typed pending persistence counters')
    integer(pending['windows'], 0, MAX_PUBLICATIONS); integer(pending['publications'], 0, MAX_PUBLICATIONS)
    require(value['kind'] == 'ASR18_packed_durable_event_and_head_record_v2'
            and head['kind'] == 'ASR18_packed_durable_full_run_lifecycle_head_v2'
            and integer(value['sequence'], 1) == integer(head['sequence'], 1) == pin['sequence']
            and value['identity_sha256'] == head['identity_sha256'] == identity_sha
            and same(value['owner'], owner) and same(head['owner'], owner)
            and head['production_enabled'] is False and head['TEST_complete'] is False and head['export_complete'] is False
            and integer(head['total_event_bytes'], 1) == pin['offset'] + pin['framed_bytes'], 'Source/owner/identity/full frame frontier')
    return value


def _next_frame(path, offset, sequence):
    with path_in_scope(path).open('rb') as stream:
        stream.seek(offset); header = stream.read(40)
    require(len(header) == 40, 'Missing chronological durable frame')
    framed = integer(struct.unpack('<Q', header[:8])[0] + 40, 41, MAX_FRAME)
    return {'kind': 'ASR18_logical_immutable_durable_head_descriptor_v2', 'path': path,
            'offset': offset, 'framed_bytes': framed, 'sha256': header[8:].hex(), 'sequence': sequence}


class _JournalPrefixes:
    """Linear prefix verification per encountered closed branch, not per window."""
    def __init__(self, guard):
        self.guard = guard; self.stream = None; self.closed = set(); self.total_sizes = 0
        self.name = None; self.offset = 0; self.hashed = None; self.before = None; self.creation = None

    def close(self):
        if self.stream is not None:
            stream = self.stream
            try: self._unchanged()
            finally: self.stream = None; stream.close()

    def _unchanged(self):
        from asr18_BPE9223_indexed_checkpoint_storage_v1 import _IndexedProtocol
        now = os.fstat(self.stream.fileno())
        require((now.st_size, now.st_mtime_ns) == self.before
                and same(_IndexedProtocol._creation_identity(self.stream), self.creation), 'Closed journal mutated during verification')
        with path_in_scope(self.name).open('rb') as current:
            require(same(_IndexedProtocol._creation_identity(current), self.creation), 'Journal path was replaced during verification')

    def verify(self, prefix):
        require(type(prefix) is dict and set(prefix) == {'path', 'offset', 'sha256'} and sha(prefix['sha256']), 'Exact authoritative journal prefix')
        target = integer(prefix['offset'], 0, MAX_JOURNAL)
        path = path_in_scope(prefix['path'])
        if self.name != str(path):
            self.close()
            require(str(path) not in self.closed and len(self.closed) < 513, 'Journal branch may not revisit an older closed writer')
            self.closed.add(str(path)); self.name = str(path); self.stream = path.open('rb')
            info = os.fstat(self.stream.fileno()); self.before = (info.st_size, info.st_mtime_ns)
            self.total_sizes += info.st_size; require(self.total_sizes <= MAX_JOURNAL, 'Declared16GiB registered journal domain')
            from asr18_BPE9223_indexed_checkpoint_storage_v1 import _IndexedProtocol
            self.creation = _IndexedProtocol._creation_identity(self.stream)
            self.offset = 0; self.hashed = hashlib.sha256()
        require(self.offset <= target <= self.before[0], 'Ordered journal prefixes cannot roll back or exceed actual file')
        while self.offset < target:
            self.guard(); block = self.stream.read(min(CHUNK, target - self.offset))
            require(block, 'Truncated journal prefix'); self.hashed.update(block); self.offset += len(block)
        require(self.hashed.hexdigest() == prefix['sha256'], 'Actual chronological journal prefix hash differs')
        self._unchanged()

    def finish_current(self):
        require(self.stream is not None and self.offset == self.before[0], 'Current closed journal has unknown suffix; preserve and reconcile before DEV')
        self.close()


def _capture(capture, pointer_pin, io, *, identity_sha, previous_sampler, previous_updates, journal):
    from asr18_streamed_persistence_fingerprint_v1 import normalize_fingerprint
    from bpe_streaming_checkpoint_semantics_v2 import select, decode_scalar_tree, verify_saved_metadata
    require(type(capture) is dict and same(capture['pointer'], pointer_pin)
            and capture['checkpoint_persistence_verified'] is True and capture['exact_GPU_restore_performed'] is False
            and capture['publication_role'] == 'completed_window_end'
            and capture['current_latest_recovery_required_before_phase_DEV_winner_advance'] is True, 'One honest normal window-end persistence capture')
    pointer = io.read(pointer_pin); update = integer(pointer['committed_updates'], 1, 401072)
    require(pointer['kind'] == 'ASR18_production_window_completed_checkpoint_pointer_v2'
            and pointer['diagnostic_only'] is io.diagnostic and pointer['writer_returned_and_payload_verified'] is True
            and pointer['all_window_updates_committed'] is True and update > previous_updates
            and 1 <= update - previous_updates <= 20
            and integer(pointer['committed_window_updates'], 1, 20) == update - previous_updates
            and same(pointer['checkpoint'], capture['checkpoint']), 'Exact returned non-diagnostic complete-window pointer')
    window = pointer['identity']['window']; caller = pointer['identity']['caller_identity']
    require(caller['run_identity_sha256'] == identity_sha and same(caller['pre_save_fingerprint'], capture['pre_save_fingerprint'])
            and same(caller['next_draw_packet'], capture['next_draw_packet']) and integer(window['optimizer_updates'], 1) == update
            and window['sampler_state_sha256'] == pointer['sampler_state_sha256']
            and same(window['committed_journal'], pointer['journal_prefix']), 'Immutable actual caller/window/source/PRE/nextdraw identity')
    metadata = io.read(pointer['checkpoint'], 2 * 1024**2)
    require(same(metadata['payload'], pointer['payload']), 'Actual saved checkpoint/payload pair identity')
    sampler = decode_scalar_tree(select(metadata['tree'], 'sampler'))
    require(digest(sampler) == pointer['sampler_state_sha256'], 'Saved full sampler/RNG does not match pointer')
    semantic = verify_saved_metadata(metadata, identity=pointer['identity'], optimizer_updates=update,
        sampler_state=sampler, journal_prefix=pointer['journal_prefix'], diagnostic_only=io.diagnostic)
    require(same(semantic, pointer['saved_semantic_state_verified']), 'Actual saved scalar semantics differ from publication')
    pre = normalize_fingerprint(io.read(capture['pre_save_fingerprint']))
    state = pre['state']
    require(integer(state['updates'], 1) == update and len(state['weights']) == 207
            and type(state['Adam_partitions']) is list and len(state['Adam_partitions']) == 3
            and same(state['clocks'], [64250 + update, 2000 + update, 62250 + update])
            and set(state['RNG']) == {'python', 'numpy', 'torch', 'CUDA'}, 'Complete actual207/threeAdam/all4RNG PRE-save')
    proof = io.read(capture['checkpoint_persistence_proof'])
    expected_worker = (io.sources['native_worker'] if update <= 20 else io.operational['operational_sources']['runtime'])
    require(proof['kind'] == 'ASR18_actual_streamed_saved_state_matches_independent_pre_save_v1'
            and proof['passed'] is True and same(proof['source'], io.sources['persistence_checker'])
            and same(proof['worker_source'], expected_worker)
            and same(proof['checkpoint_pointer'], pointer_pin) and same(proof['checkpoint'], pointer['checkpoint'])
            and same(proof['payload'], pointer['payload']) and integer(proof['optimizer_updates'], 1) == update
            and integer(proof['weights_verified'], 1) == 207 and integer(proof['Adam_states_verified'], 207) >= 207
            and 1 <= integer(proof['named_tensors_verified'], 1) <= 4096
            and proof['independent_pre_save_state_sha256'] == proof['saved_state_sha256'] == pre['state_sha256']
            and proof['live_after_state_sha256'] == pre['state_sha256'] and proof['live_unchanged_after_persistence'] is True
            and proof['actual_GPU_restore_performed'] is False and proof['live_model_optimizer_or_RNG_mutated_here'] is False
            and same(proof['saved_semantic_state_verified'], semantic) and proof['sampler_state_sha256'] == digest(sampler)
            and same(proof['journal_prefix'], pointer['journal_prefix'])
            and same(proof['pre_save_fingerprint'], capture['pre_save_fingerprint'])
            and same(proof['next_draw_packet'], capture['next_draw_packet'])
            and integer(proof['maximum_payload_read_bytes'], 1) == 4 * 1024**2,
            'Source-bound actual PRE/saved/live207/threeAdam/RNG persistence equality; no new payload read or restore')
    packet = io.read(capture['next_draw_packet'])
    require(integer(packet['optimizer_updates'], 1) == update, 'Actual provisional next draw frontier')
    if update == 401072:
        require(packet['kind'] == 'ASR18_full_run_complete_no_further_draw_packet_v1'
                and packet['sampler_state_sha256'] == digest(sampler) and packet['further_native_or_authoritative_draws_allowed'] is False
                and integer(packet['draws_consumed']) == 0, 'Explicit complete schedule terminal packet; no fabricated next draw')
    else:
        require(packet['kind'] == 'ASR18_actual_checkpoint_provisional_next_draw_packet_v1'
                and packet['before_sampler_sha256'] == digest(sampler) and integer(packet['draws_consumed']) == 0
                and packet['replay_frontier_advanced'] is False and packet['audio_or_augmentation_applied'] is False,
                'Actual source-bound next draw, zero sampler/replay/augmentation advancement')
    journal.verify(pointer['journal_prefix'])
    return pointer, sampler


def _record_self():
    raw = Path(__file__).read_bytes()
    return {'path': str(Path(__file__).resolve()), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def _current(evidence, io, request, pointer, capture):
    from fastmss_owned_process_closure_v2 import verify
    resource = evidence['kind'] == 'ASR18_full_run_resource_stop_current_recovery_prefix_inputs_v1'
    require(evidence['kind'] in ('ASR18_full_run_current_recovery_prefix_inputs_v1',
            'ASR18_full_run_resource_stop_current_recovery_prefix_inputs_v1')
            and evidence['owner'] == OWNER and evidence['diagnostic_only'] is io.diagnostic
            and evidence['run_identity_sha256'] == digest(request['identity'])
            and same(evidence['pending_head'], request['head']) and same(evidence['current_checkpoint_capture'], capture)
            and same(evidence['operational_compatibility'], io.compatibility), 'Exact current operational recovery evidence')
    recovery_process = io.read(evidence['actual_recovery_process0'])
    verify(recovery_process, kind='ASR18_owned_GPU_process_completion_v1', owner=OWNER,
           report=evidence['recovery_report'], guard=io.guard)
    op = io.operational['operational_sources']
    require(same(recovery_process['worker_source'], op['runtime'])
            and same(recovery_process['parent_source'], op['parent']), 'Actual reviewed fresh operational recovery0')
    if resource:
        native = checked_resource_stop(evidence['current_native_resource_closure'], identity=request['identity'],
            compatibility=io.compatibility, guard=io.guard)
        require(same(native['current_checkpoint_pointer'], capture['pointer']) and pointer['committed_updates'] == 20
                and all(not same(item, recovery_process['producer']) for item in native['all_captured_identities']),
                'Fresh recovery is distinct from every closed abnormal source')
    else:
        native_process = io.read(evidence['current_native_process0'])
        verify(native_process, kind='ASR18_owned_GPU_process_completion_v1', owner=OWNER, guard=io.guard)
        require(same(native_process['worker_source'], op['runtime'])
                and same(native_process['parent_source'], op['parent'])
                and not same(native_process['producer'], recovery_process['producer']), 'Actual distinct operational native0/recovery0')
        native = io.read(native_process['report'])
        require(native['passed'] is True and native['diagnostic_only'] is io.diagnostic
                and native['run_identity_sha256'] == digest(request['identity'])
                and same(native['current_checkpoint'], capture)
                and integer(native['final_optimizer_updates'], 1) == pointer['committed_updates']
                and native['all_exact_GPU_checkpoint_restores_deferred'] is True, 'Actual normal native0 current capture')
    recovery = io.read(evidence['recovery_report'])
    expected_kind = ('ASR18_full_run_actual_CURRENT_checkpoint_recovered_after_resource_stop_v1' if resource
                     else 'ASR18_full_run_actual_CURRENT_checkpoint_recovered_v2')
    source_key = 'current_native_resource_closure' if resource else 'current_native_process0'
    require(recovery['kind'] == expected_kind and recovery['passed'] is True
            and same(recovery['source'], op['current_recovery'])
            and recovery['diagnostic_only'] is io.diagnostic and recovery['current_latest_checkpoint_recovered'] is True
            and recovery['historical_fresh_GPU_restores_claimed'] is False
            and recovery['run_identity_sha256'] == digest(request['identity'])
            and same(recovery[source_key], evidence[source_key])
            and same(recovery['operational_compatibility'], io.compatibility)
            and integer(recovery['actual_native_updates_in_recovery_process']) == 0
            and integer(recovery['actual_draws_consumed_in_recovery']) == 0
            and type(recovery['recovered_checkpoints']) is list and len(recovery['recovered_checkpoints']) == 1,
            'Actual sole-current fresh recovery, zero optimization/draws, explicit source lineage')
    recovered_pin = recovery['recovered_checkpoints'][0]; recovered = io.read(recovered_pin)
    require(recovered['passed'] is True and same(recovered['pointer'], capture['pointer'])
            and same(recovered['checkpoint'], pointer['checkpoint'])
            and same(recovered['pre_save_fingerprint'], capture['pre_save_fingerprint'])
            and same(recovered['next_draw_packet'], capture['next_draw_packet'])
            and recovered['all207_threeAdam_all4RNG_match_actual_pre_save'] is True
            and recovered['both_samplers_journal_prefix_and_next_draw_exact'] is True
            and integer(recovered['native_updates_performed']) == 0 and integer(recovered['draws_consumed']) == 0
            and recovered['separate_fresh_GPU_recovery'] is True, 'Actual all854/all207/Adam/RNG/sampler/prefix/nextdraw equality')
    return recovered_pin


def _coverage(request, *, resolve_member, guard):
    io = _Metadata(request['identity'], resolve_member, guard, request['operational_compatibility']); identity_sha = digest(request['identity'])
    record3(request['owner']); record3(request['policy'])
    namespace = path_in_scope(request['namespace']); require(namespace != ROOT.resolve(), 'Exact future durable namespace')
    owner = io.read(request['owner']); policy = io.read(request['policy'])
    require(Path(request['owner']['path']) == namespace/'owner.json'
            and owner['kind'] == 'ASR18_future_packed_durable_lifecycle_owner_v2'
            and same(owner['policy'], request['policy']) and same(owner['identity'], request['identity'])
            and same(owner['source'], io.sources['durable_adapter']) and owner['adopt_existing_artifacts'] is False
            and policy['kind'] == 'ROOT_inactive_packed_durable_full_run_policy_review_v2'
            and policy['accepted'] is True and policy['reviewer'] == '/root'
            and policy['scope'] == 'full_run_integration_validation' and policy['production_enabled'] is False
            and policy['namespace'] == str(namespace) and policy['run_identity_sha256'] == identity_sha
            and same(policy['source'], io.sources['durable_adapter'])
            and same(owner['proof_sources'], policy['proof_callback_sources'])
            and same(owner['proof_sources']['verify_current_recovery_and_persistence_prefix'], io.sources['prefix_verifier']),
            'Actual ROOT-reviewed namespace/identity/source-bound sole owner, not an unowned frame stream')
    tip = _descriptor(request['head']); require(Path(tip['path']) == namespace/'records.bin', 'Exact owned durable stream namespace/name')
    current = _frame(tip, identity_sha, request['owner'], guard)
    require(same(current['head'], request['state']), 'Caller snapshot equals actual immutable pending head')
    pending = current['head']['pending']; windows = integer(pending['windows'], 1, MAX_PUBLICATIONS)
    require(integer(pending['publications'], 1, MAX_PUBLICATIONS) == windows, 'Production normal window-end cadence only')
    # Discover the prior acknowledged recovery without retaining a history list.
    first = tip; scanned = 0
    while integer(current['head']['pending']['windows']):
        require(current['operation'] == 'native_window_pending_exact_recovery', 'Unsupported metadata crash tail is preserved, not invented as a native window')
        scanned += 1; require(scanned <= windows, 'Pending prefix count mismatch')
        prev = _descriptor(current['previous_head'])
        require(prev['path'] == first['path'] and prev['sequence'] + 1 == first['sequence']
                and prev['offset'] + prev['framed_bytes'] == first['offset'], 'Contiguous previous durable frame')
        baseline = _frame(prev, identity_sha, request['owner'], guard)
        if baseline['head']['pending']['windows'] == 0: break
        first, current = prev, baseline
    require(scanned == windows and baseline['head']['pending']['publications'] == 0, 'Complete prefix since last genuine acknowledged recovery')
    previous = prev; previous_head = baseline['head']; chain = digest([])
    require(same(previous_head['pending'], {'windows': 0, 'publications': 0, 'chain_sha256': digest([]),
            'first_optimizer_update': None, 'last_optimizer_update': None}), 'Exact cleared previous persistence prefix')
    if previous_head['lifecycle']['last_pointer'] is None:
        require(baseline['operation'] == 'genesis' and baseline['sequence'] == 1, 'Only actual original0 genesis lacks a current pointer')
        previous_updates = 0
    else:
        prior_pointer = io.read(previous_head['lifecycle']['last_pointer'])
        previous_updates = integer(prior_pointer['committed_updates'], 1)
        require(previous_head['last_recovery'] is not None, 'Earlier prefix requires retained actual current recovery acknowledgement')
        prior_ack = io.read(previous_head['last_recovery'])
        prior_coverage = io.read(prior_ack['persistence_prefix_coverage_index'])
        require(prior_ack['kind'] in ('ASR18_full_run_current_latest_recovery_and_verified_persistence_prefix_v1',
                    'ASR18_resource_stop_current_latest_recovery_and_verified_persistence_prefix_v1')
                and prior_ack['diagnostic_only'] is io.diagnostic and prior_ack['run_identity_sha256'] == identity_sha
                and same(prior_ack['final_checkpoint_pointer'], previous_head['lifecycle']['last_pointer'])
                and prior_coverage['kind'] == 'ASR18_verified_durable_persistence_prefix_coverage_v1'
                and prior_coverage['passed'] is True and (same(prior_coverage['source'], io.sources['prefix_verifier']) or
                    same(prior_coverage['source'], _record_self()) and same(prior_coverage['operational_compatibility'], io.compatibility))
                and prior_coverage['run_identity_sha256'] == identity_sha
                and same(prior_coverage['pending_head'], prior_ack['pending_head'])
                and same(prior_coverage['final_checkpoint_pointer'], previous_head['lifecycle']['last_pointer'])
                and prior_coverage['all_completed_window_PRE_saved_live_persistence_verified'] is True
                and prior_coverage['historical_fresh_GPU_restores_claimed'] is False,
                'Retained source-bound previous exact coverage chain; no implicit historical re-restore')
    baseline_update = previous_updates; baseline_sampler = previous_head['lifecycle']['last_sampler_sha256']
    journal = _JournalPrefixes(guard); final_capture = None; pointer = None
    try:
        for index in range(windows):
            pin = first if index == 0 else _next_frame(tip['path'], previous['offset'] + previous['framed_bytes'], previous['sequence'] + 1)
            value = _frame(pin, identity_sha, request['owner'], guard)
            require(same(value['previous_head'], previous) and value['operation'] == 'native_window_pending_exact_recovery', 'One chronological completed-window frame')
            completion = io.read(value['evidence']['completion']); plan = io.read(value['evidence']['plan'])
            require(completion['passed'] is True and completion['diagnostic_only'] is io.diagnostic
                    and completion['run_identity_sha256'] == identity_sha
                    and completion['lifecycle_parent_head_sha256'] == previous['sha256']
                    and completion['all_checkpoint_persistence_verified'] is True and completion['exact_GPU_restores_deferred'] is True
                    and type(completion['checkpoints']) is list and len(completion['checkpoints']) == 1,
                    'One actual non-diagnostic normal publication, never a prepared or finite diagnostic completion')
            final_capture = completion['checkpoints'][0]
            require(same(final_capture['pointer'], completion['latest_pointer']) and plan['initial_sampler_sha256'] == baseline_sampler
                    and integer(plan['initial_committed_updates']) == previous_updates and 1 <= len(plan['updates']) <= 20
                    and integer(completion['actual_new_updates'], 1) == len(plan['updates']), 'Exact actual window source prefix')
            pointer, sampler = _capture(final_capture, completion['latest_pointer'], io, identity_sha=identity_sha,
                previous_sampler=baseline_sampler, previous_updates=previous_updates, journal=journal)
            require(integer(completion['global_new_updates'], 1) == pointer['committed_updates']
                    and same(completion['coordinator_plan'], value['evidence']['plan'])
                    and completion['sampler_state_sha256'] == pointer['sampler_state_sha256']
                    and same(completion['journal_prefix'], pointer['journal_prefix'])
                    and pointer['identity']['window']['plan_sha256'] == digest(plan)
                    and pointer['identity']['window']['last_transaction_sha256'] == plan['updates'][-1]['native_transaction_sha256']
                    and digest(plan['final_private_planning_sampler']) == digest(sampler)
                    and pointer['committed_updates'] == previous_updates + len(plan['updates'])
                    and same(value['head']['lifecycle']['last_pointer'], completion['latest_pointer'])
                    and value['head']['lifecycle']['last_sampler_sha256'] == digest(sampler), 'Actual saved/full-plan/frontier is the immutable durable head')
            chain = digest({'previous': chain, 'completion': value['evidence']['completion'], 'captures': completion['checkpoints']})
            expected_pending = {'windows': index + 1, 'publications': index + 1, 'chain_sha256': chain,
                'first_optimizer_update': baseline_update + 1, 'last_optimizer_update': pointer['committed_updates']}
            require(same(value['head']['pending'], expected_pending), 'Actual chronological pending-chain digest/frontiers')
            previous_updates = pointer['committed_updates']; baseline_sampler = digest(sampler); previous = pin
        require(same(previous, tip) and same(expected_pending, pending), 'Exact tip/prefix end, no hidden or omitted window')
        journal.finish_current()
    finally: journal.close()
    evidence = io.read(request['current_recovery_evidence'])
    recovered_pin = _current(evidence, io, request, pointer, final_capture)
    return {'kind': 'ASR18_verified_durable_persistence_prefix_coverage_v1', 'passed': True,
        'source': _record_self(), 'original_semantic_prefix_source': deepcopy(io.sources['prefix_verifier']),
        'operational_compatibility': deepcopy(io.compatibility), 'run_identity_sha256': identity_sha,
        'pending_head': deepcopy(tip), 'pending': deepcopy(pending),
        'final_checkpoint_pointer': deepcopy(final_capture['pointer']), 'current_checkpoint_capture_sha256': digest(final_capture),
        'all_completed_window_PRE_saved_live_persistence_verified': True, 'historical_fresh_GPU_restores_claimed': False,
        'native_updates_consumed_in_verification': 0, 'current_recovery_evidence': request['current_recovery_evidence'],
        'current_latest_exact_restore': recovered_pin, 'prior_acknowledged_head': prev,
        'verified_new_window_count': windows, 'maximum_metadata_read_bytes': MAX_METADATA,
        'maximum_durable_frame_bytes': MAX_FRAME, 'maximum_journal_chunk_bytes': CHUNK,
        'tensor_payload_bytes_read': 0, 'whole_history_materialized': False, 'any_native_or_sampler_operation': False}


def create_coverage(request, *, resolve_member, guard):
    """After actual native0/current recovery0, before durable recovery ACK."""
    path = path_in_scope(request['coverage_output']); namespace = path_in_scope(request['namespace'])
    require(path.is_relative_to(namespace) and path.parent.is_dir() and not path.exists(), 'Fresh owned flat coverage output only')
    value = _coverage(deepcopy(request), resolve_member=resolve_member, guard=guard)
    raw = canonical(value); require(len(raw) <= MAX_FRAME, 'Bounded compact coverage receipt')
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


class OwnerPrefixVerifier:
    """Concrete source-bound durable2 callback; resolver and guard are retained."""
    def __init__(self, *, resolve_member, guard, operational_compatibility):
        self.resolve_member, self.guard = resolve_member, guard
        self.operational_compatibility = deepcopy(operational_compatibility)

    def verify_current_recovery_and_persistence_prefix(self, request):
        require(request['event_chain_must_be_streamed'] is True
                and ((request.get('actual_resource_native_quiescence_required') is True
                      and request['actual_native_and_recovery_process0_required'] is False)
                     or request.get('actual_resource_native_quiescence_required', False) is False
                      and request['actual_native_and_recovery_process0_required'] is True)
                and request['current_latest_all207_threeAdam_all4RNG_sampler_prefix_nextdraw_required'] is True
                and request['historical_PRE_persistence_prefix_required'] is True
                and request['historical_fresh_GPU_restores_claimed'] is False, 'Exact durable2 owner callback requirements')
        io = _Metadata(request['identity'], self.resolve_member, self.guard, self.operational_compatibility); ack = io.read(request['proof'])
        require(ack['kind'] in ('ASR18_full_run_current_latest_recovery_and_verified_persistence_prefix_v1',
                    'ASR18_resource_stop_current_latest_recovery_and_verified_persistence_prefix_v1')
                and ack['diagnostic_only'] is io.diagnostic and ack['run_identity_sha256'] == digest(request['identity'])
                and same(ack['pending_head'], request['head']) and same(ack['pending'], request['state']['pending'])
                and ack['fresh_recovery_scope'] == 'current_latest_only'
                and ack['all_persistence_verified_through_pending_prefix'] is True
                and ack['all_historical_checkpoints_freshly_restored'] is False
                and ack['current_latest_all207_threeAdam_all4RNG_sampler_prefix_nextdraw_match_actual_pre_save'] is True
                and integer(ack['actual_recovery_native_updates']) == 0 and integer(ack['actual_recovery_draws_consumed']) == 0,
                'Exact production current recovery/persistence acknowledgement')
        resource = ack['kind'] == 'ASR18_resource_stop_current_latest_recovery_and_verified_persistence_prefix_v1'
        require(resource == (request.get('actual_resource_native_quiescence_required') is True)
                and same(ack['operational_compatibility'], self.operational_compatibility), 'Explicit normal/resource acknowledgement distinction')
        if resource:
            require(ack['actual_resource_quiescence_proven'] is True and ack['actual_native_normal_exit0_claimed'] is False, 'No abnormal native0 relabel')
        recompute = deepcopy(request); recompute['operational_compatibility'] = deepcopy(self.operational_compatibility)
        recompute['current_recovery_evidence'] = ack['current_recovery_evidence']
        expected = _coverage(recompute, resolve_member=self.resolve_member, guard=self.guard)
        evidence = io.read(ack['current_recovery_evidence'])
        require(same(io.read(ack['persistence_prefix_coverage_index']), expected)
                and same(ack['final_checkpoint_pointer'], expected['final_checkpoint_pointer'])
                and same(ack['actual_closed_native_index'], evidence['current_native_resource_closure' if resource else 'current_native_process0'])
                and same(ack['actual_closed_recovery_index'], evidence['actual_recovery_process0'])
                and same(ack['current_latest_exact_restore'], expected['current_latest_exact_restore']), 'Actual independently reverified current coverage receipt')
        return True
