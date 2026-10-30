"""Inactive packed durability successor; no model/native/DEV executor.

Reuses v1's explicit pending-persistence/current-only-recovery transitions but
replaces its per-event files with one framed stream and a bounded SQLite index.
Logical descriptors identify immutable byte ranges, NOT whole-file record3s.
An actual newer coherent native checkpoint can reconcile before restored-state
equality. No native replay, sampler rollback or unverified orphan adoption.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import struct

from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record, require
from bpe_broad_production_state_v1 import digest
from asr18_BPE9223_lifecycle_v3 import (FullRunLifecycle, _LifecycleProtocol,
    bound, read_bound, progress, validate_window_plan)
from asr18_BPE9223_durable_transitions_v3 import (DurableLifecycleAdapter as Transitions,
    OwnerProofCallbacks, same, integer, new_E, E, diagnostic_identity, poison_on_failure)
from asr18_BPE9223_resource_operational_compatibility_v1 import (
    checked_compatibility, checked_resource_stop, OperationalOwnerProofCallbacks)

HEAD = 'ASR18_packed_durable_full_run_lifecycle_head_v2'
RECORD = 'ASR18_packed_durable_event_and_head_record_v2'
DESCRIPTOR = 'ASR18_logical_immutable_durable_head_descriptor_v2'
POINTER = 'ASR18_atomic_logical_durable_head_pointer_v2'
HEADER = 40  # uint64 payload length, then SHA256 payload bytes


def derive_orphan_protocol_state(*, durable_state, before_state, plan, pointer_binding, pointer, after_state, identity):
    """Pure metadata transition for an independently PROVEN actual orphan.

    Used before constructing a restored FullRunLifecycle at the actual current
    schedule. It performs no native update or sampler mutation. Native-owner
    verification of checkpoint/PRE-save/closure/source is still mandatory.
    """
    require(durable_state['last_sampler_sha256'] == digest(before_state), 'Exact old durable sampler required')
    candidate = _LifecycleProtocol(before_state, identity, None, restored=durable_state)
    candidate.accept_window(before_state, plan, after_state, pointer_binding, pointer)
    return deepcopy(candidate.state)


class PackedDurableLifecycleAdapter(Transitions):
    """Same normal transition API as v1; new typed physical backing/policy.

    A complete record is capped at64KiB including framing. Head, event and
    evidence are copied into that single bounded record. Whole stream/index
    capacity and E100GB reserve are checked separately. Native checkpoint parent
    binding uses descriptor.sha256, which remains valid after future appends.

    Only an explicit source-verified single normal orphan window is supported.
    Unknown/truncated tails, partial-window emergencies and missing actual state
    are preserved and rejected, never erased or automatically rerun.
    """
    def __init__(self, lifecycle, folder, *, policy, proofs, create=False, resume_proof=None,
                 orphan_native=None, operational_compatibility=None, resource_resume=None):
        require(isinstance(lifecycle, FullRunLifecycle) and isinstance(proofs, OwnerProofCallbacks),
                'Frozen lifecycle protocol and source-bound owner verifiers')
        require(create is False and orphan_native is None,
                'Operational successor only reopens the original creator namespace, no genesis/orphan adoption')
        self.operational_compatibility = deepcopy(operational_compatibility)
        review = checked_compatibility(operational_compatibility, lifecycle.identity)
        require(isinstance(proofs, OperationalOwnerProofCallbacks)
                and same(proofs.compatibility, operational_compatibility)
                and same(proofs.sources, review['operational_proof_callback_sources'])
                and same(lifecycle.callbacks.sources, review['operational_lifecycle_callback_sources'])
                and same(review['operational_sources']['durable_adapter'], record(__file__)),
                'Actual operational source/callback implementations distinct from creator contract')
        creator_source = record(Path(__file__).with_name('asr18_BPE9223_durable_owner_adapter_v3.py'))
        resource = None if resource_resume is None else checked_resource_stop(resource_resume,
            identity=lifecycle.identity, compatibility=operational_compatibility)
        self.lifecycle = lifecycle; self.identity = deepcopy(lifecycle.identity); self.proofs = proofs
        self.folder = new_E(folder); self.policy_pin = deepcopy(policy); self.policy = read_bound(policy)
        self.creator_pid = os.getpid(); self.poisoned = False; self.lock = None
        self.head = None; self.head_pin = None; self.latest_pin = None
        self._reconcile_tail = False
        p = self.policy
        require(p['kind'] == 'ROOT_inactive_packed_durable_full_run_policy_review_v2'
                and p['accepted'] is True and p['reviewer'] == '/root'
                and p['scope'] == 'full_run_integration_validation' and p['production_enabled'] is False
                and p['source'] == creator_source
                and p['transition_source'] == record(Path(__file__).with_name('asr18_BPE9223_durable_transitions_v3.py'))
                and p['namespace'] == str(self.folder) and p['run_identity_sha256'] == digest(self.identity)
                and same(p['proof_callback_sources'], proofs.legacy_contract_sources)
                and same(p['lifecycle_callback_sources'], proofs.legacy_lifecycle_contract_sources),
                'Exact inactive packed policy/source/namespace/callback authority required')
        for key in ('maximum_records', 'maximum_stream_bytes', 'maximum_index_bytes',
                    'maximum_record_bytes', 'maximum_pending_windows', 'maximum_pending_publications'):
            integer(p[key], 1)
        require(HEADER < p['maximum_record_bytes'] <= 65536
                and p['normal_publication_cadence'] == 'completed_window_end_only'
                and p['pending_capacity_supports_complete_longest_pass'] is True
                and p['maximum_pending_windows'] >= integer(self.identity['maximum_pass_window_publications'], 1)
                and p['maximum_pending_publications'] >= self.identity['maximum_pass_window_publications']
                and integer(p['maximum_emergency_publications']) == 0,
                'Bounded complete-pass normal cadence; no diagnostic2/40 or hidden emergency support')
        if create:
            require(resume_proof is None and orphan_native is None, 'No existing state adoption at genesis')
            self.folder.mkdir(exist_ok=False)
        require(self.folder.is_dir(), 'Exact future-owned packed namespace')
        self._acquire_lock()
        try:
            if create:
                write_json(self.folder/'owner.json', {'kind': 'ASR18_future_packed_durable_lifecycle_owner_v2',
                    'policy': policy, 'source': record(__file__), 'transition_source': p['transition_source'],
                    'identity': self.identity, 'proof_sources': proofs.sources, 'adopt_existing_artifacts': False})
                self.owner = record(self.folder/'owner.json')
                with (self.folder/'records.bin').open('xb') as stream: stream.flush(); os.fsync(stream.fileno())
                db = self._db()
                try:
                    db.execute('CREATE TABLE heads (seq INTEGER PRIMARY KEY, offset INTEGER UNIQUE NOT NULL, framed_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL)')
                    db.commit()
                finally: db.close()
                self._append('genesis', lifecycle.state_dict(), self._empty_pending(), {}, None)
            else:
                self.owner = record(self.folder/'owner.json'); owner = read_bound(self.owner)
                require(owner['kind'] == 'ASR18_future_packed_durable_lifecycle_owner_v2'
                        and same(owner['policy'], policy) and same(owner['identity'], self.identity)
                        and same(owner['proof_sources'], proofs.legacy_contract_sources) and owner['source'] == creator_source
                        and same(owner['transition_source'], p['transition_source'])
                        and owner['adopt_existing_artifacts'] is False, 'Exact prior packed owner')
                self.latest_pin = record(self.folder/'latest.json'); latest = read_bound(self.latest_pin)
                require(latest['kind'] == POINTER and same(latest['owner'], self.owner), 'Exact packed latest pointer')
                self.head_pin = latest['head']; self.head = self._checked_head(self._read_descriptor(self.head_pin)['head'])
                self._audit_chain()
                # This source-bound check is deliberately BEFORE actual/restored
                # protocol equality. A newer ACTUAL checkpoint is not rolled back.
                if resource is not None:
                    require(resume_proof is None
                            and self.head['pending']['windows'] == 1
                            and self.head['pending']['publications'] == 1
                            and self.head['pending']['first_optimizer_update'] == 1
                            and self.head['pending']['last_optimizer_update'] == 20
                            and self.head['last_recovery'] is None
                            and same(self.head['lifecycle']['last_pointer'], resource['current_checkpoint_pointer'])
                            and same(lifecycle.state_dict(), self.head['lifecycle'])
                            and digest(lifecycle.schedule.state_dict()) == self.head['lifecycle']['last_sampler_sha256'],
                            'Initial abnormal20 lane only reopens the exact pending p1 scalar authority')
                    self.proofs.verify('verify_durable_resume', self._request(
                        resource_resume=resource_resume, restore_proof=None,
                        actual_native_normal_exit0_claimed=False,
                        fresh_recovery_still_required=True, no_authoritative_advance=True))
                else:
                    proofs.verify('verify_durable_resume', self._request(restore_proof=resume_proof,
                        orphan_native=orphan_native, actual_newer_checkpoint_reconciliation_before_equality=True,
                        native_replay_or_sampler_rollback_permitted=False))
                if orphan_native is not None:
                    self._reconcile_orphan_native(orphan_native)
                else:
                    require((self.folder/'records.bin').stat().st_size == self.head['total_event_bytes'],
                            'Unacknowledged stream tail requires explicit owner reconciliation')
                    self._require_index_frontier(self.head['sequence'])
                require(same(lifecycle.state_dict(), self.head['lifecycle'])
                        and digest(lifecycle.schedule.state_dict()) == self.head['lifecycle']['last_sampler_sha256'],
                        'Actual authoritative restored protocol must equal verified durable/reconciled state')
        except BaseException:
            self.close(); raise

    def _db(self):
        db = sqlite3.connect(self.folder/'heads.sqlite')
        db.execute('PRAGMA journal_mode=DELETE'); db.execute('PRAGMA synchronous=FULL')
        db.execute('PRAGMA cache_size=-1024'); db.execute('PRAGMA temp_store=MEMORY')
        page = db.execute('PRAGMA page_size').fetchone()[0]
        pages = self.policy['maximum_index_bytes']//page
        require(pages >= 16 and db.execute('PRAGMA max_page_count='+str(pages)).fetchone()[0] <= pages,
                'Hard owned-index file cap before SQLite allocation')
        return db

    def _require_index_frontier(self, sequence):
        db = self._db()
        try: count, last = db.execute('SELECT count(*), max(seq) FROM heads').fetchone()
        finally: db.close()
        require(count == sequence and last == sequence, 'No unknown indexed tail or missing immutable record')

    def _read_descriptor(self, pin):
        require(type(pin) is dict and set(pin) == {'kind', 'path', 'offset', 'framed_bytes', 'sha256', 'sequence'}
                and pin['kind'] == DESCRIPTOR and Path(pin['path']).resolve() == self.folder/'records.bin',
                'Logical immutable range descriptor; never a whole mutable-file record3')
        offset = integer(pin['offset']); framed = integer(pin['framed_bytes'], HEADER+1)
        require(framed <= self.policy['maximum_record_bytes'] and integer(pin['sequence'], 1), 'Bounded exact frame geometry')
        with (self.folder/'records.bin').open('rb') as stream:
            stream.seek(offset); header = stream.read(HEADER)
            require(len(header) == HEADER, 'Full framed header required')
            size = struct.unpack('<Q', header[:8])[0]
            require(size+HEADER == framed, 'Length descriptor/header mismatch before read')
            raw = stream.read(size)
        require(len(raw) == size and hashlib.sha256(raw).digest() == header[8:]
                and hashlib.sha256(raw).hexdigest() == pin['sha256'], 'Exact immutable bounded record bytes')
        value = json.loads(raw)
        require(value['kind'] == RECORD and value['sequence'] == pin['sequence']
                and same(value['owner'], self.owner) and value['identity_sha256'] == digest(self.identity),
                'Exact logical record source/owner/sequence')
        return value

    def _checked_head(self, value):
        require(value['kind'] == HEAD and same(value['owner'], self.owner)
                and value['identity_sha256'] == digest(self.identity)
                and value['production_enabled'] is False and value['TEST_complete'] is False
                and value['export_complete'] is False, 'Exact packed scalar head')
        require(integer(value['sequence'], 1) <= self.policy['maximum_records']
                and integer(value['total_event_bytes'], 1) <= self.policy['maximum_stream_bytes']
                and integer(value['pending']['windows']) <= self.policy['maximum_pending_windows']
                and integer(value['pending']['publications']) <= self.policy['maximum_pending_publications'],
                'Declared packed/pending capacity')
        return value

    def _check(self):
        require(not self.poisoned and self.lock is not None and os.getpid() == self.creator_pid,
                'Exact sole current packed writer')
        if self.latest_pin is not None:
            require(record(self.folder/'latest.json') == self.latest_pin
                    and same(self._read_descriptor(self.head_pin)['head'], self.head), 'Packed durable authority changed')

    def _audit_chain(self):
        pin = self.head_pin; expected = self.head['sequence']; end = self.head['total_event_bytes']
        db = self._db()
        try:
            while pin is not None:
                self.proofs.verify('verify_writer_authority', self._request(operation='startup_packed_history_audit', record=pin))
                value = self._read_descriptor(pin)
                require(pin['sequence'] == expected and pin['offset']+pin['framed_bytes'] == end,
                        'Exact ordered contiguous packed prefix')
                row = db.execute('SELECT offset,framed_bytes,sha256 FROM heads WHERE seq=?', (expected,)).fetchone()
                require(row == (pin['offset'], pin['framed_bytes'], pin['sha256']), 'Index disagrees with actual immutable bytes')
                end = pin['offset']; pin = value['previous_head']; expected -= 1
            require(expected == 0 and end == 0, 'Full actual acknowledged prefix coverage')
        finally: db.close()

    def _append(self, operation, state, pending, evidence, last_recovery):
        self._check()
        self.proofs.verify('verify_writer_authority', self._request(operation=operation,
            proposed_lifecycle=state, proposed_pending=pending, proposed_evidence=evidence))
        require(digest(self.lifecycle.schedule.state_dict()) == state['last_sampler_sha256'], 'Actual sampler changed at metadata authority boundary')
        sequence = 1 if self.head is None else self.head['sequence']+1
        offset = 0 if self.head is None else self.head['total_event_bytes']
        head = {'kind': HEAD, 'owner': self.owner, 'identity_sha256': digest(self.identity),
            'sequence': sequence, 'total_event_bytes': offset, 'lifecycle': deepcopy(state),
            'pending': deepcopy(pending), 'last_recovery': deepcopy(last_recovery),
            'production_enabled': False, 'TEST_complete': False, 'export_complete': False}
        value = {'kind': RECORD, 'sequence': sequence, 'owner': self.owner,
            'identity_sha256': digest(self.identity), 'previous_head': self.head_pin,
            'operation': operation, 'evidence': deepcopy(evidence), 'head': head}
        # The encoded prefix length appears in the head; solve its few decimal
        # digits before writing any bytes, with a fixed bounded iteration count.
        for _ in range(16):
            raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf8')
            length = offset+HEADER+len(raw)
            if length == head['total_event_bytes']: break
            head['total_event_bytes'] = length
        else: raise ValueError('Packed length did not stabilize')
        self._checked_head(head)
        framed = HEADER+len(raw)
        require(framed <= self.policy['maximum_record_bytes'], 'Unsupported metadata size before packed allocation')
        require(shutil.disk_usage(E).free >= 100_000_000_000+2*self.policy['maximum_record_bytes']+4*1024**2,
                'Unchanged E100GB reserve and bounded index/pointer workspace')
        sha = hashlib.sha256(raw).digest(); blob = struct.pack('<Q', len(raw))+sha+raw
        pin = {'kind': DESCRIPTOR, 'path': str(self.folder/'records.bin'), 'offset': offset,
               'framed_bytes': framed, 'sha256': sha.hex(), 'sequence': sequence}
        try:
            size = (self.folder/'records.bin').stat().st_size
            if size == offset:
                with (self.folder/'records.bin').open('ab') as stream:
                    stream.write(blob); stream.flush(); os.fsync(stream.fileno())
            else:
                require(self._reconcile_tail and size == offset+framed,
                        'Unknown/truncated tail preserved; no implicit truncate, overwrite or retry')
                with (self.folder/'records.bin').open('rb') as stream: stream.seek(offset); actual = stream.read(framed)
                require(actual == blob, 'Existing orphan bytes differ from actual source-verified transition')
            require(same(self._read_descriptor(pin), value), 'Read-back packed byte identity')
            db = self._db()
            try:
                row = db.execute('SELECT offset,framed_bytes,sha256 FROM heads WHERE seq=?', (sequence,)).fetchone()
                expected = (offset, framed, sha.hex())
                if row is None: db.execute('INSERT INTO heads VALUES (?,?,?,?)', (sequence, *expected))
                else: require(self._reconcile_tail and row == expected, 'Unknown index sequence conflict')
                db.commit()
            finally: db.close()
            require((self.folder/'heads.sqlite').stat().st_size <= self.policy['maximum_index_bytes'], 'Declared bounded index capacity')
            write_json(self.folder/'latest.json', {'kind': POINTER, 'owner': self.owner, 'head': pin})
            self.head = deepcopy(head); self.head_pin = pin; self.latest_pin = record(self.folder/'latest.json')
            self.lifecycle.protocol.state = deepcopy(state)
            self._require_index_frontier(sequence)
            return deepcopy(pin)
        except BaseException:
            self.poisoned = True; raise

    def _reconcile_orphan_native(self, packet_pin):
        packet = read_bound(packet_pin)
        require(packet['kind'] == 'ASR18_actual_orphan_committed_window_for_durable_reconciliation_v2'
                and packet['diagnostic_only'] is diagnostic_identity(self.identity) and same(packet['prior_head'], self.head_pin)
                and packet['run_identity_sha256'] == digest(self.identity), 'Exact source-verified actual orphan packet')
        before = read_bound(packet['before_sampler_state']); planned = read_bound(packet['coordinator_plan'])
        completion = read_bound(packet['actual_completion']); pointer_pin = completion['latest_pointer']; pointer = read_bound(pointer_pin)
        after = self.lifecycle.schedule.state_dict()
        require(completion['passed'] is True and completion['diagnostic_only'] is diagnostic_identity(self.identity)
                and completion['run_identity_sha256'] == digest(self.identity)
                and completion['lifecycle_parent_head_sha256'] == self.head_pin['sha256']
                and integer(completion['actual_new_updates'], 1) == len(planned['updates'])
                and integer(completion['global_new_updates'], 1) == pointer['committed_updates']
                and completion['all_checkpoint_persistence_verified'] is True
                and completion['exact_GPU_restores_deferred'] is True,
                'Actual orphan must bind old authority and new coherent native frontier')
        captures = completion['checkpoints']; require(type(captures) is list and len(captures) == 1,
                'Only one actual normal completed-window orphan; emergencies unresolved')
        capture = captures[0]
        require(same(capture['pointer'], pointer_pin) and capture['publication_role'] == 'completed_window_end'
                and capture['checkpoint_persistence_verified'] is True and capture['exact_GPU_restore_performed'] is False
                and capture['current_latest_recovery_required_before_phase_DEV_winner_advance'] is True,
                'Actual final persistence cannot be relabeled recovery')
        for key in ('pointer', 'checkpoint', 'checkpoint_persistence_proof', 'pre_save_fingerprint', 'next_draw_packet'): bound(capture[key])
        candidate = derive_orphan_protocol_state(durable_state=self.head['lifecycle'], before_state=before,
            plan=planned, pointer_binding=pointer_pin, pointer=pointer, after_state=after, identity=self.identity)
        require(same(candidate, self.lifecycle.state_dict()),
                'Actual restored lifecycle candidate must equal recomputed original transition, never fabricated old state')
        self.proofs.verify('verify_pending_window', self._request(before_state=before, plan=packet['coordinator_plan'],
            completion=packet['actual_completion'], expected_after_state=after, all_captures=captures,
            actual_orphan_reconciliation=packet_pin, no_native_reexecution_or_sampler_rollback=True))
        old = self.head['pending']
        pending = {'windows': old['windows']+1, 'publications': old['publications']+1,
            'chain_sha256': digest({'previous': old['chain_sha256'], 'completion': packet['actual_completion'], 'captures': captures}),
            'first_optimizer_update': old['first_optimizer_update'] if old['windows'] else progress(before)['optimizer_updates']+1,
            'last_optimizer_update': pointer['committed_updates']}
        self._reconcile_tail = True
        try:
            self._append('native_window_pending_exact_recovery', candidate, pending,
                         {'plan': packet['coordinator_plan'], 'completion': packet['actual_completion']}, self.head['last_recovery'])
        finally: self._reconcile_tail = False


    @poison_on_failure
    def acknowledge_recovery(self, proof):
        value = read_bound(proof)
        if value.get('kind') == 'ASR18_cleanup_join_failure_current40_recovery_and_verified_persistence_prefix_v1':
            return self._acknowledge_cleanup_join_recovery(proof, value)
        if value.get('kind') != 'ASR18_resource_stop_current_latest_recovery_and_verified_persistence_prefix_v1':
            return Transitions.acknowledge_recovery.__wrapped__(self, proof)
        pending = self.head['pending']
        require(pending['publications'] == 1 and pending['windows'] == 1
                and pending['first_optimizer_update'] == 1 and pending['last_optimizer_update'] == 20
                and self.head['last_recovery'] is None
                and value['diagnostic_only'] is diagnostic_identity(self.identity)
                and value['run_identity_sha256'] == digest(self.identity)
                and same(value['operational_compatibility'], self.operational_compatibility)
                and same(value['pending'], pending) and same(value['pending_head'], self.head_pin)
                and same(value['final_checkpoint_pointer'], self.lifecycle.state_dict()['last_pointer'])
                and value['fresh_recovery_scope'] == 'current_latest_only'
                and value['all_persistence_verified_through_pending_prefix'] is True
                and value['all_historical_checkpoints_freshly_restored'] is False
                and value['current_latest_all207_threeAdam_all4RNG_sampler_prefix_nextdraw_match_actual_pre_save'] is True
                and value['actual_resource_quiescence_proven'] is True
                and value['actual_native_normal_exit0_claimed'] is False
                and integer(value['actual_recovery_native_updates']) == 0
                and integer(value['actual_recovery_draws_consumed']) == 0,
                'Real abnormal20 quiescence followed by actual fresh p1 exact recovery, never fabricated native0')
        for key in ('actual_closed_native_index', 'actual_closed_recovery_index',
                    'persistence_prefix_coverage_index', 'current_latest_exact_restore'):
            bound(value[key])
        self.proofs.verify('verify_current_recovery_and_persistence_prefix', self._request(proof=proof,
            event_chain_must_be_streamed=True, actual_native_and_recovery_process0_required=False,
            actual_resource_native_quiescence_required=True,
            current_latest_all207_threeAdam_all4RNG_sampler_prefix_nextdraw_required=True,
            historical_PRE_persistence_prefix_required=True, historical_fresh_GPU_restores_claimed=False))
        return self._append('current_latest_recovery_and_persistence_prefix_acknowledged', self.lifecycle.state_dict(),
                            self._empty_pending(), {'recovery': proof}, proof)


    def _acknowledge_cleanup_join_recovery(self, proof, value):
        """One explicit20-to40 completed writer failure; no native0 relabel."""
        from asr18_BPE9223_persistence_prefix_v9 import checked_cleanup_join_failure
        from asr18_BPE9223_operational_metadata_v2 import read_bound as original_or_archived_read
        failed = checked_cleanup_join_failure(value['current_cleanup_join_failure'], identity=self.identity,
            operational_compatibility=self.operational_compatibility, guard=lambda: None,
            reader=original_or_archived_read)
        pending = self.head['pending']
        require(pending['publications'] == pending['windows'] == 1
            and pending['first_optimizer_update'] == 21 and pending['last_optimizer_update'] == 40
            and self.head['last_recovery'] is not None
            and value['diagnostic_only'] is diagnostic_identity(self.identity)
            and value['run_identity_sha256'] == digest(self.identity)
            and same(value['operational_compatibility'], self.operational_compatibility)
            and same(value['pending'], pending) and same(value['pending_head'], self.head_pin)
            and same(value['final_checkpoint_pointer'], failed['checkpoint_pointer'])
            and same(value['final_checkpoint_pointer'], self.lifecycle.state_dict()['last_pointer'])
            and value['fresh_recovery_scope'] == 'current_latest_only'
            and value['all_persistence_verified_through_pending_prefix'] is True
            and value['all_historical_checkpoints_freshly_restored'] is False
            and value['current_latest_all207_threeAdam_all4RNG_sampler_prefix_nextdraw_match_actual_pre_save'] is True
            and value['actual_cleanup_join_failure_quiescence_proven'] is True
            and value['actual_native_normal_exit0_claimed'] is False
            and same(value['actual_closed_native_index'], value['actual_closed_recovery_index'])
            and integer(value['actual_recovery_native_updates']) == 0
            and integer(value['actual_recovery_draws_consumed']) == 0,
            'Exact pending21-to40 plus actual fresh40 recovery, no rollback or failed writer normal0 claim')
        for key in ('actual_closed_native_index', 'actual_closed_recovery_index', 'persistence_prefix_coverage_index',
                    'current_latest_exact_restore', 'cleanup_reconciliation_process0', 'cleanup_reconciliation_report'):
            bound(value[key])
        self.proofs.verify('verify_current_recovery_and_persistence_prefix', self._request(proof=proof,
            event_chain_must_be_streamed=True, actual_native_and_recovery_process0_required=False,
            actual_resource_native_quiescence_required=False, actual_cleanup_join_native_quiescence_required=True,
            current_latest_all207_threeAdam_all4RNG_sampler_prefix_nextdraw_required=True,
            historical_PRE_persistence_prefix_required=True, historical_fresh_GPU_restores_claimed=False))
        return self._append('current_latest_recovery_and_persistence_prefix_acknowledged', self.lifecycle.state_dict(),
                            self._empty_pending(), {'recovery': proof}, proof)
