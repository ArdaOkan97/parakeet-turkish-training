"""Inactive scalar lifecycle durability over the frozen lifecycle2 state machine.

No launcher, model, optimizer, audio, preparation or checkpoint tensor access.
The native owner supplies separately reviewed proof verifiers. Persisted native
state is PENDING boundary recovery; only actual native exit plus separate exact
CURRENT/LATEST recovery and verification of the durable persistence prefix
discharge it. This does not assert historical checkpoints were freshly restored.
Existing finite diagnostic checkpoints are rejected and their every-capture
recovery requirements are unchanged.
"""
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import stat
from functools import wraps

from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record, require
from bpe_broad_production_state_v1 import digest
from asr18_BPE9223_lifecycle_v2 import (FullRunLifecycle, _LifecycleProtocol,
    bound, read_bound, official_DEV, progress, validate_window_plan)

E = Path(r'E:\gpu_training_scratchpad')
KIND = 'ASR18_durable_full_run_lifecycle_head_v1'
EVENT = 'ASR18_durable_full_run_lifecycle_event_v1'
PROOF_CALLBACKS = frozenset(('verify_pending_window', 'verify_current_recovery_and_persistence_prefix',
                           'verify_durable_resume', 'verify_writer_authority'))


def same(a, b):
    return a == b and digest(a) == digest(b)


def integer(value, low=0):
    require(type(value) is int and value >= low, 'Typed bounded nonnegative integer')
    return value


def new_E(path):
    path = Path(path).resolve()
    require(path.is_relative_to(E.resolve()) and path != E.resolve(), 'Future owned E namespace only')
    for parent in (path, *path.parents):
        if parent.exists():
            require(not parent.is_symlink() and not getattr(parent.lstat(), 'st_file_attributes', 0)
                    & stat.FILE_ATTRIBUTE_REPARSE_POINT, 'No reparse/symlink ownership escape')
        if parent == E.resolve(): break
    return path


def poison_on_failure(method):
    @wraps(method)
    def checked(self, *args, **kwargs):
        self._check()
        try:
            return method(self, *args, **kwargs)
        except BaseException:
            self.poisoned = True; self.lifecycle.protocol.poisoned = True
            raise
    return checked


def diagnostic_identity(identity):
    """Explicit qualification namespace; never infer a missing boolean."""
    require(type(identity) is dict and type(identity.get('owner_validation_only')) is bool,
            'Typed immutable owner_validation_only required')
    return identity['owner_validation_only']


class OwnerProofCallbacks:
    """Source identity only; actual authority and resource checks stay external.

    verify_pending_window independently verifies current all207/threeAdam/RNG,
    saved semantic/payload/PRE-save/next-draw identity, original schedule/rows,
    exact positive native frontier and every capture's persistence. It MUST NOT
    return True because a prepared window, idle marker or pointer exists.

    verify_current_recovery_and_persistence_prefix independently streams the
    exact durable persistence prefix, verifies actual native process0 and
    distinct recovery process0, and matches CURRENT/LATEST all207/threeAdam/
    all4RNG/sampler/prefix/next draw. It checks every historical publication's
    independent PRE-save/persistence link without claiming fresh restore of
    obsolete historical checkpoints. Diagnostic every-capture recovery is not
    accepted through this production-only interface.

    verify_durable_resume verifies actual authoritative checkpoint restore and
    durable head/DEV/winner/journal reconciliation, including orphan events.
    verify_writer_authority checks the sole current owner, fresh guards, exact
    unchanged head and separately reviewed metadata/storage growth allowance.
    """
    def __init__(self, sources, functions):
        require(set(sources) == set(functions) == PROOF_CALLBACKS, 'All exact owner proof callbacks required')
        self.sources = deepcopy(sources); self.functions = dict(functions)
        for name, function in functions.items():
            bound(sources[name])
            require(str(Path(inspect.getsourcefile(function)).resolve()) == sources[name]['path'],
                    'Callback source path mismatch')

    def verify(self, name, request):
        bound(self.sources[name])
        require(self.functions[name](deepcopy(request)) is True, 'Independent owner proof failed: '+name)


class DurableLifecycleAdapter:
    """One future namespace, one locked writer, bounded scalar hot state.

    Native callbacks execute outside this adapter. record_pending_window() is
    called only AFTER their positive update/checkpoint. It records the actual
    sampler frontier without asserting restore. acknowledge_recovery() may run
    at an earlier coherent chunk or the mandatory pass/DEV/final boundary.
    Complete DEV and winner acceptance remain blocked until that current-state
    recovery and persistence-prefix verification. Normal production publication
    is window-end only; diagnostic local1/10 captures are rejected. Every append
    is durable before the in-memory protocol advances.

    create/open require a NEW exact ROOT policy; no such accepted policy or
    concrete native callbacks are provided here. Resume audits immutable event
    files serially with bounded RAM; full-horizon startup I/O is unmeasured.
    """
    def __init__(self, lifecycle, folder, *, policy, proofs, create=False, resume_proof=None):
        require(isinstance(lifecycle, FullRunLifecycle) and isinstance(proofs, OwnerProofCallbacks),
                'Frozen lifecycle2 plus exact owner proofs required')
        self.lifecycle = lifecycle; self.proofs = proofs; self.folder = new_E(folder)
        self.policy_pin = deepcopy(policy); self.policy = read_bound(policy)
        p = self.policy; self.identity = deepcopy(lifecycle.identity)
        require(p['kind'] == 'ROOT_inactive_durable_full_run_adapter_policy_review_v1'
                and p['accepted'] is True and p['reviewer'] == '/root'
                and p['production_enabled'] is False and p['scope'] == 'full_run_integration_validation'
                and p['source'] == record(__file__) and p['namespace'] == str(self.folder)
                and p['run_identity_sha256'] == digest(self.identity)
                and same(p['proof_callback_sources'], proofs.sources)
                and same(p['lifecycle_callback_sources'], lifecycle.callbacks.sources),
                'Exact separately reviewed inactive namespace/source/identity/callback policy required')
        for key in ('maximum_events', 'maximum_total_event_bytes', 'maximum_pending_windows',
                    'maximum_pending_publications', 'maximum_event_bytes', 'maximum_head_bytes'):
            integer(p[key], 1)
        require(p['maximum_event_bytes'] <= 8*1024**2 and p['maximum_head_bytes'] <= 4*1024**2,
                'Bounded metadata-only records; no tensor/archive payloads')
        require(p['normal_publication_cadence'] == 'completed_window_end_only'
                and p['pending_capacity_supports_complete_longest_pass'] is True
                and integer(p['maximum_pending_windows'], 1) >= integer(self.identity['maximum_pass_window_publications'], 1)
                and p['maximum_pending_publications'] >= self.identity['maximum_pass_window_publications']
                and integer(p['maximum_emergency_publications']) == 0,
                'Reviewed whole-pass pending capacity; this adapter does not implement emergency partial-window publication')
        self.poisoned = False; self.creator_pid = os.getpid(); self.lock = None
        if create:
            self.folder.mkdir(exist_ok=False); (self.folder/'events').mkdir(); (self.folder/'heads').mkdir()
        require(self.folder.is_dir() and (self.folder/'events').is_dir() and (self.folder/'heads').is_dir(),
                'Exact owned future namespace')
        self._acquire_lock()
        self.head = None; self.head_pin = None; self.latest_pin = None
        try:
            if create:
                require(resume_proof is None, 'Creation cannot adopt a prior native/diagnostic state')
                write_json(self.folder/'owner.json', {'kind': 'ASR18_future_durable_lifecycle_owner_v1',
                    'policy': policy, 'source': record(__file__), 'identity': self.identity,
                    'proof_sources': proofs.sources, 'adopt_existing_artifacts': False})
                self.owner = record(self.folder/'owner.json')
                self._append('genesis', lifecycle.state_dict(), self._empty_pending(), {}, None)
            else:
                self.owner = record(self.folder/'owner.json'); owner = read_bound(self.owner)
                require(same(owner['policy'], policy) and same(owner['identity'], self.identity)
                        and same(owner['proof_sources'], proofs.sources)
                        and owner['source'] == record(__file__) and owner['adopt_existing_artifacts'] is False,
                        'Exact prior owner/source/policy required')
                self.latest_pin = record(self.folder/'latest.json'); latest = read_bound(self.latest_pin)
                require(latest['kind'] == 'ASR18_atomic_durable_full_run_head_pointer_v1'
                        and same(latest['owner'], self.owner), 'Exact atomic head pointer')
                self.head_pin = latest['head']; self.head = self._checked_head(read_bound(self.head_pin))
                self._audit_chain()
                require(same(lifecycle.state_dict(), self.head['lifecycle']), 'Restored protocol differs from durable authority')
                proofs.verify('verify_durable_resume', self._request(restore_proof=resume_proof,
                    exact_native_restore_and_orphan_reconciliation_required=True))
        except BaseException:
            self.close(); raise

    def _acquire_lock(self):
        require(os.name == 'nt', 'Reviewed Windows future-owner file lock only')
        import msvcrt
        self.lock = (self.folder/'writer.lock').open('a+b')
        try:
            size = self.lock.seek(0, 2)
            require(size in (0, 1), 'Only the exact one-byte future-owned writer lock')
            if size == 0:
                self.lock.write(b'0'); self.lock.flush(); os.fsync(self.lock.fileno())
            self.lock.seek(0); msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
        except BaseException:
            self.lock.close(); self.lock = None; raise

    def close(self):
        if self.lock is not None:
            import msvcrt
            try:
                self.lock.seek(0); msvcrt.locking(self.lock.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                self.lock.close(); self.lock = None

    @staticmethod
    def _empty_pending():
        return {'windows': 0, 'publications': 0, 'chain_sha256': digest([]),
                'first_optimizer_update': None, 'last_optimizer_update': None}

    def _request(self, **extra):
        return {'identity': self.identity, 'namespace': str(self.folder), 'policy': self.policy_pin,
                'owner': self.owner, 'head': self.head_pin, 'state': deepcopy(self.head), **extra}

    def _checked_head(self, value):
        require(value['kind'] == KIND and same(value['owner'], self.owner)
                and value['identity_sha256'] == digest(self.identity)
                and value['production_enabled'] is False and value['TEST_complete'] is False
                and value['export_complete'] is False, 'Exact scalar durable head')
        integer(value['sequence'], 1); integer(value['total_event_bytes'], 1)
        require(value['sequence'] <= self.policy['maximum_events']
                and value['total_event_bytes'] <= self.policy['maximum_total_event_bytes'], 'Declared durable growth cap')
        require(value['pending']['windows'] <= self.policy['maximum_pending_windows']
                and value['pending']['publications'] <= self.policy['maximum_pending_publications'], 'Pending persistence index cap')
        integer(value['pending']['windows']); integer(value['pending']['publications'])
        return value

    def _check(self):
        require(not self.poisoned and self.lock is not None and os.getpid() == self.creator_pid,
                'Live exact sole durable writer; failures need reconciliation')
        if self.latest_pin is not None:
            require(record(self.folder/'latest.json') == self.latest_pin
                    and record(self.head_pin['path']) == self.head_pin, 'Authoritative durable head changed')

    def _audit_chain(self):
        pin = self.head['event']; expected = self.head['sequence']; total = 0
        while pin is not None:
            value = read_bound(pin); total += pin['bytes']
            require(value['kind'] == EVENT and value['sequence'] == expected
                    and same(value['owner'], self.owner)
                    and value['identity_sha256'] == digest(self.identity), 'Exact contiguous immutable event chain')
            require(Path(pin['path']).resolve() == self.folder/'events'/(str(expected).zfill(12)+'.json'),
                    'Event outside exact owned sequence')
            pin = value['previous_event']; expected -= 1
        require(expected == 0 and total == self.head['total_event_bytes'], 'Complete durable history coverage')
        last = read_bound(self.head['event'])
        require(same(last['lifecycle'], self.head['lifecycle']) and same(last['pending'], self.head['pending']),
                'Head must match actual final immutable event')

    def _append(self, operation, state, pending, evidence, last_recovery):
        self._check()
        self.proofs.verify('verify_writer_authority', self._request(operation=operation,
            proposed_lifecycle=state, proposed_pending=pending, proposed_evidence=evidence))
        require(digest(self.lifecycle.schedule.state_dict()) == state['last_sampler_sha256'],
                'Metadata authority callback changed actual sampler/frontier')
        sequence = 1 if self.head is None else self.head['sequence']+1
        previous = None if self.head is None else self.head['event']
        event = {'kind': EVENT, 'sequence': sequence, 'owner': self.owner,
            'identity_sha256': digest(self.identity), 'previous_event': previous,
            'operation': operation, 'lifecycle': deepcopy(state), 'pending': deepcopy(pending),
            'evidence': deepcopy(evidence), 'last_recovery': deepcopy(last_recovery),
            'production_enabled': False}
        encoded = (json.dumps(event, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode('utf8')
        require(len(encoded) <= self.policy['maximum_event_bytes'], 'Unresolved event geometry before write')
        reserve = self.policy['maximum_event_bytes']+2*self.policy['maximum_head_bytes']
        require(shutil.disk_usage(E).free >= 100_000_000_000+reserve, 'Unchanged E100GB reserve before metadata publication')
        path = self.folder/'events'/(str(sequence).zfill(12)+'.json')
        head_path = self.folder/'heads'/(str(sequence).zfill(12)+'.json')
        require(not path.exists() and not head_path.exists(),
                'Orphan/conflicting event/head requires independent durable reconciliation')
        event_pin = {'path': str(path), 'bytes': len(encoded), 'sha256': hashlib.sha256(encoded).hexdigest()}
        total = event_pin['bytes']+(0 if self.head is None else self.head['total_event_bytes'])
        head = {'kind': KIND, 'owner': self.owner, 'identity_sha256': digest(self.identity),
            'sequence': sequence, 'event': event_pin, 'total_event_bytes': total,
            'lifecycle': deepcopy(state), 'pending': deepcopy(pending), 'last_recovery': deepcopy(last_recovery),
            'production_enabled': False, 'TEST_complete': False, 'export_complete': False}
        self._checked_head(head)
        require(len((json.dumps(head, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode('utf8'))
                <= self.policy['maximum_head_bytes'], 'Bounded durable hot head before write')
        try:
            write_json(path, event); require(record(path) == event_pin, 'Exact independently predicted event bytes')
            write_json(head_path, head); head_pin = record(head_path)
            write_json(self.folder/'latest.json', {'kind': 'ASR18_atomic_durable_full_run_head_pointer_v1',
                       'owner': self.owner, 'head': head_pin})
            self.head = head; self.head_pin = head_pin; self.latest_pin = record(self.folder/'latest.json')
            self.lifecycle.protocol.state = deepcopy(state)
            return deepcopy(self.head_pin)
        except BaseException:
            self.poisoned = True; raise

    def next_action(self):
        self._check(); action = self.lifecycle.protocol.request(self.lifecycle.schedule.state_dict())
        pending = self.head['pending']
        capacity_barrier = (pending['windows'] >= self.policy['maximum_pending_windows']
                            or pending['publications']+1 > self.policy['maximum_pending_publications'])
        if action['kind'] == 'bounded_training_window' and capacity_barrier:
            return {'kind': 'unsupported_pending_index_capacity_before_preparation',
                    'pending': deepcopy(pending), 'blocked_action': action, 'head': self.head_pin,
                    'forced_GPU_reload_or_training_shortening_authorized': False}
        if pending['publications'] and action['kind'] != 'bounded_training_window':
            return {'kind': 'actual_native_exit_then_current_latest_recovery_and_persistence_prefix_required',
                    'pending': deepcopy(self.head['pending']), 'blocked_action': action, 'head': self.head_pin}
        return {'action': action, 'head': self.head_pin, 'pending_recovery': deepcopy(self.head['pending'])}

    @poison_on_failure
    def record_pending_window(self, *, before_state, plan, completion):
        """Called after native success; never returns a restored/overall-pass flag."""
        self._check(); before = deepcopy(before_state); action = self.lifecycle.protocol.request(before)
        planned = read_bound(plan); actual = read_bound(completion)
        validate_window_plan(before, action, planned)
        pointer_pin = actual['latest_pointer']; pointer = read_bound(pointer_pin)
        after = self.lifecycle.schedule.state_dict()
        require(actual['passed'] is True and actual['diagnostic_only'] is diagnostic_identity(self.identity)
                and actual['lifecycle_parent_head_sha256'] == self.head_pin['sha256']
                and actual['run_identity_sha256'] == digest(self.identity)
                and integer(actual['actual_new_updates'], 1) == len(planned['updates'])
                and integer(actual['global_new_updates'], 1) == pointer['committed_updates']
                and actual['all_checkpoint_persistence_verified'] is True
                and actual['exact_GPU_restores_deferred'] is True,
                'Actual non-diagnostic native positive persistence; never finite checkpoint adoption')
        captures = actual['checkpoints']; require(type(captures) is list and len(captures) == 1,
                'Normal production window-end only; diagnostic local1/10 captures are not production cadence')
        for capture in captures:
            require(capture['checkpoint_persistence_verified'] is True
                    and capture['exact_GPU_restore_performed'] is False
                    and capture['publication_role'] == 'completed_window_end'
                    and capture['current_latest_recovery_required_before_phase_DEV_winner_advance'] is True,
                    'Persistence and restore remain distinct')
            for key in ('pointer', 'checkpoint', 'checkpoint_persistence_proof', 'pre_save_fingerprint', 'next_draw_packet'):
                bound(capture[key])
        capture_updates = [integer(read_bound(c['pointer'])['committed_updates'], 1) for c in captures]
        require(capture_updates == sorted(set(capture_updates)) and capture_updates[-1] == pointer['committed_updates']
                and capture_updates[0] > progress(before)['optimizer_updates']
                and same(captures[-1]['pointer'], pointer_pin), 'Every actual capture ordered through final pointer')
        self.proofs.verify('verify_pending_window', self._request(before_state=before, plan=plan,
            completion=completion, expected_after_state=after, all_captures=captures))
        require(digest(self.lifecycle.schedule.state_dict()) == digest(after), 'Persistence verifier changed actual sampler')
        candidate = _LifecycleProtocol(before, self.identity, None, restored=self.lifecycle.state_dict())
        candidate.accept_window(before, planned, after, pointer_pin, pointer)
        old = self.head['pending']; count = old['windows']+1; publication_count = old['publications']+1
        pending = {'windows': count, 'publications': publication_count,
            'chain_sha256': digest({'previous': old['chain_sha256'], 'completion': completion, 'captures': captures}),
            'first_optimizer_update': old['first_optimizer_update'] if old['windows'] else progress(before)['optimizer_updates']+1,
            'last_optimizer_update': pointer['committed_updates']}
        require(count <= self.policy['maximum_pending_windows'] and publication_count <= self.policy['maximum_pending_publications'],
                'Unresolved pending persistence capacity; never omit a publication')
        return self._append('native_window_pending_exact_recovery', candidate.state, pending,
                            {'plan': plan, 'completion': completion}, self.head['last_recovery'])

    @poison_on_failure
    def acknowledge_recovery(self, proof):
        self._check(); value = read_bound(proof); pending = self.head['pending']
        require(pending['publications'] > 0 and value['kind'] == 'ASR18_full_run_current_latest_recovery_and_verified_persistence_prefix_v1'
                and value['diagnostic_only'] is diagnostic_identity(self.identity) and value['run_identity_sha256'] == digest(self.identity)
                and same(value['pending'], pending) and value['pending_head'] == self.head_pin
                and same(value['final_checkpoint_pointer'], self.lifecycle.state_dict()['last_pointer'])
                and value['fresh_recovery_scope'] == 'current_latest_only'
                and value['all_persistence_verified_through_pending_prefix'] is True
                and value['all_historical_checkpoints_freshly_restored'] is False
                and value['current_latest_all207_threeAdam_all4RNG_sampler_prefix_nextdraw_match_actual_pre_save'] is True
                and integer(value['actual_recovery_native_updates']) == 0
                and integer(value['actual_recovery_draws_consumed']) == 0,
                'Current/latest recovery plus exact durable persistence prefix; no all-history fresh-recovery claim')
        for key in ('actual_closed_native_index', 'actual_closed_recovery_index', 'persistence_prefix_coverage_index',
                    'current_latest_exact_restore'):
            bound(value[key])
        self.proofs.verify('verify_current_recovery_and_persistence_prefix', self._request(proof=proof,
            event_chain_must_be_streamed=True, actual_native_and_recovery_process0_required=True,
            current_latest_all207_threeAdam_all4RNG_sampler_prefix_nextdraw_required=True,
            historical_PRE_persistence_prefix_required=True, historical_fresh_GPU_restores_claimed=False))
        return self._append('current_latest_recovery_and_persistence_prefix_acknowledged', self.lifecycle.state_dict(),
                            self._empty_pending(), {'recovery': proof}, proof)

    @poison_on_failure
    def acknowledge_DEV_and_winner(self, *, report, winner):
        """Accept an already genuinely sealed DEV/candidate; no DEV launch here.

        Native owner preserves/returns the same immutable report and candidate
        after a crash. Re-acknowledgment must never rerun optimizer or consume a
        draw. Failed/equal candidates must pass the exact old winner artifact.
        """
        self._check(); require(diagnostic_identity(self.identity) is False, 'Validation checkpoints cannot acknowledge production DEV or winner'); state = self.lifecycle.schedule.state_dict(); action = self.lifecycle.protocol.request(state)
        require(action['kind'] == 'checkpoint_then_serial350DEV' and self.head['pending']['publications'] == 0
                and self.head['last_recovery'] is not None, 'No DEV acceptance before complete exact recovery')
        pointer = self.lifecycle.state_dict()['last_pointer']; callback = self.lifecycle.callbacks
        callback_request = {'report': report, 'checkpoint_pointer': pointer, 'boundary': action['boundary'],
            'identity': self.identity, 'all207_weights_and_all_source_clocks_unchanged_required': True,
            'all4RNG_and_replay_queues_unchanged_required': True}
        require(callback.call('verify_DEV', callback_request) is True, 'Actual complete optimizer-free350 DEV required')
        actual_report = read_bound(report)
        require(actual_report['production_recovery_scope'] == 'current_latest_only_with_verified_persistence_prefix'
                and actual_report['historical_fresh_GPU_restore_claimed'] is False
                and same(actual_report['production_recovery_boundary'], self.head['last_recovery']),
                'NEW production DEV proof must honestly bind current recovery/persistence prefix; frozen DEV6 unchanged')
        score = official_DEV(actual_report, pointer, read_bound(self.identity['scorer_policy']))
        candidate = _LifecycleProtocol(state, self.identity, None, restored=self.lifecycle.state_dict())
        old = candidate.state['winner']
        change = score['strict_all_six_passed'] and (old is None or score['word_errors']['tr'] < old['word_errors']['tr'])
        if change:
            bound(winner)
            require(callback.call('verify_winner', {'winner': winner, 'DEV': report, 'score': score,
                'checkpoint_pointer': pointer, 'identity': self.identity}) is True, 'Immutable actual strict-qualified candidate proof')
        result = candidate.accept_DEV(state, report, score, winner)
        require(digest(self.lifecycle.schedule.state_dict()) == digest(state), 'DEV/selection changed authoritative sampler')
        head = self._append('complete_DEV_and_winner_acknowledged', candidate.state, self._empty_pending(),
                            {'DEV': report, 'winner': winner, 'score': score}, self.head['last_recovery'])
        return {'head': head, **result}

    def state_dict(self):
        self._check()
        require(same(self.lifecycle.state_dict(), self.head['lifecycle'])
                and digest(self.lifecycle.schedule.state_dict()) == self.head['lifecycle']['last_sampler_sha256'],
                'Protocol/actual sampler differs from durable head')
        return deepcopy(self.head)
