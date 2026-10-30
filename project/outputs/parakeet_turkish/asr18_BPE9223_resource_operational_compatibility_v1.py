"""Read-only operational compatibility and abnormal-quiescence proofs.

The original scientific identity, creator records and failed exit are retained.
New implementation sources are explicit, never stamped as the old callbacks.
This module does not load tensor payloads, mutate journals or launch processes.
"""
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path

from bpe_broad_audio_v1 import record, require
from bpe_broad_production_state_v1 import digest
from bpe_streaming_checkpoint_semantics_v2 import decode_scalar_tree, select, verify_saved_metadata
from asr18_BPE9223_durable_transitions_v3 import OwnerProofCallbacks, PROOF_CALLBACKS
from fastmss_owned_process_closure_v2 import dead

ROOT = Path(r'E:\gpu_training_scratchpad')
HERE = Path(__file__).resolve().parent
MAX_JSON = 16 * 1024**2
REVIEW_KIND = 'ROOT_BPE9223_resource_stop_operational_compatibility_review_v1'
RESOURCE_KIND = 'ASR18_real_BPE9223_resource_stop_after_coherent20_actual_TOOL1_v1'


def same(left, right):
    return left == right and digest(left) == digest(right)


def read_bound(pin, guard=lambda: None):
    require(type(pin) is dict and set(pin) == {'path', 'bytes', 'sha256'}
            and type(pin['bytes']) is int and 0 < pin['bytes'] <= MAX_JSON,
            'Bounded exact source/metadata record required')
    path = Path(pin['path']).resolve()
    require(path.is_relative_to(ROOT.resolve()) or path.is_relative_to(HERE.parents[1]),
            'Existing project or designated E metadata only')
    guard(); require(record(path) == pin, 'Reviewed metadata/source binding changed')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def checked_compatibility(pin, identity, guard=lambda: None):
    value = read_bound(pin, guard)
    require(value['kind'] == REVIEW_KIND and value['accepted'] is True
            and value['reviewer'] == '/root'
            and value['scope'] == 'operational_metadata_lifetime_and_exact_resource_recovery_only'
            and value['scientific_identity_preserved'] is True
            and value['scientific_identity_sha256'] == digest(identity)
            and value['numerical_recipe_changed'] is False
            and value['guard_limits_changed'] is False
            and value['source_optimizer_RNG_sampler_reset_permitted'] is False,
            'Explicit ROOT operational transition, never a scientific reset or guard relaxation')
    old = read_bound(value['original_configuration'], guard)
    require(same(old['full_run_identity'], identity) and old['vocabulary'] == 9223
            and old['owner_validation_only'] is False and old['diagnostic_only'] is False
            and old['planned_native_updates'] == 401072
            and same(old['source_manifest'], value['original_source_manifest'])
            and same(identity['full_run_source_contract'], value['original_source_manifest']),
            'Original production identity and full schedule remain immutable')
    original = read_bound(value['original_source_manifest'], guard)
    operational = read_bound(value['operational_source_manifest'], guard)
    original_sources = {item['path']: item for item in original['bindings']}
    operational_sources = {item['path']: item for item in operational['bindings']}
    require(len(original_sources) == len(original['bindings'])
            and len(operational_sources) == len(operational['bindings'])
            and all(operational_sources.get(path) == item for path, item in original_sources.items())
            and len(operational_sources) <= len(original_sources) + 32,
            'All original executable bindings retained; only bounded explicit operational successors')
    for role, item in value['operational_sources'].items():
        require(type(role) is str and operational_sources.get(item['path']) == item
                and record(item['path']) == item, 'Exact reviewed operational source')
    require(same(value['operational_sources']['compatibility'], record(__file__)),
            'Actual compatibility implementation is source pinned')
    old_policy = read_bound(old['durable_policy'], guard)
    require(same(value['legacy_proof_callback_sources'], old_policy['proof_callback_sources'])
            and same(value['legacy_lifecycle_callback_sources'], old_policy['lifecycle_callback_sources'])
            and set(value['operational_proof_callback_sources']) == PROOF_CALLBACKS
            and set(value['operational_lifecycle_callback_sources']) == set(old_policy['lifecycle_callback_sources']),
            'Historical and actual operational callbacks are separate exact maps')
    for mapping in ('legacy_proof_callback_sources', 'legacy_lifecycle_callback_sources',
                    'operational_proof_callback_sources', 'operational_lifecycle_callback_sources'):
        for item in value[mapping].values():
            require(operational_sources.get(item['path']) == item and record(item['path']) == item,
                    'Callback source not present in reviewed unchanged/operational closure')
    return value


def checked_resource_stop(resource_pin, *, identity, compatibility, guard=lambda: None):
    """Prove the abnormal source is quiescent before first fresh p1 recovery.

    Requires the whole journal at this boundary. Future training may append only
    after the separately verified recovery; this is not a normal-exit receipt.
    """
    review = checked_compatibility(compatibility, identity, guard)
    require(same(resource_pin, review['actual_resource_stop']), 'Exact reviewed real resource stop')
    value = read_bound(resource_pin, guard)
    require(value['kind'] == RESOURCE_KIND and value['owner'] == '/root/memory_diagnosis'
            and type(value['actual_tool_exit_code']) is int and value['actual_tool_exit_code'] == 1
            and type(value['actual_exec_tool_exit_code']) is int and value['actual_exec_tool_exit_code'] == 1
            and value['actual_tool_chunk_id'] == '07d337' and value['owned_session'] == 57534
            and value['run_identity_sha256'] == digest(identity)
            and type(value['actual_committed_optimizer_updates']) is int
            and value['actual_committed_optimizer_updates'] == 20
            and type(value['actual_authoritative_TR_draws']) is int and value['actual_authoritative_TR_draws'] == 280
            and type(value['actual_authoritative_replay_updates']) is int and value['actual_authoritative_replay_updates'] == 20
            and value['normal40_or_normal_TOOL0_claimed'] is False
            and value['actual_native0_or_CURRENT_fresh0_claimed'] is False,
            'Truthful abnormal20 boundary, never a fabricated normal0 or fresh recovery')
    require(value['all_captured_identities_dead'] is True
            and len(value['all_captured_identities']) == 4
            and all(dead(item) for item in value['all_captured_identities']),
            'All original parent/native/launcher identities must actually be dead')
    joined = read_bound(value['owner_guard'], guard)
    require(joined['monitor_clean'] is True and joined['monitor_joined'] is True
            and joined['producer'] == value['producer'] and joined['samples'] >= 3,
            'Original monitor completed and joined')
    pointer = read_bound(value['current_checkpoint_pointer'], guard)
    metadata = read_bound(value['current_checkpoint_metadata'], guard)
    require(same(pointer['checkpoint'], value['current_checkpoint_metadata'])
            and same(metadata['payload'], value['current_payload_binding_not_rehashed'])
            and pointer['committed_updates'] == 20 and len(metadata['tensors']) == 854
            and pointer['diagnostic_only'] is False,
            'Exact real20 coherent saved checkpoint, not a diagnostic source')
    scalar = lambda *path: decode_scalar_tree(select(metadata['tree'], *path))
    sampler = scalar('sampler')
    require(same(scalar('identity'), pointer['identity'])
            and same(pointer['identity']['caller_identity']['full_run_lifecycle_identity'], identity)
            and scalar('optimizer', 'updates') == 20
            and same(scalar('optimizer', 'clocks'), [64270, 2020, 62270])
            and scalar('optimizer', 'appended512_Adam_clock') == 20,
            'Original mature clocks and appended20 are preserved')
    semantics = verify_saved_metadata(metadata, identity=pointer['identity'], optimizer_updates=20,
        sampler_state=sampler, journal_prefix=pointer['journal_prefix'], diagnostic_only=False)
    require(same(semantics, value['current_saved_semantics']), 'Saved scalar and journal semantics differ')
    journal = value['closed_registered_journal']; prefix = pointer['journal_prefix']
    require(value['checkpoint_prefix_equals_whole_journal'] is True
            and journal['path'] == prefix['path'] and journal['bytes'] == prefix['offset']
            and journal['sha256'] == prefix['sha256'] and record(journal['path']) == journal,
            'Actual entire closed registered p1 journal, no unknown suffix or rewind')
    payload = metadata['payload']; path = Path(payload['path']).resolve()
    require(path.is_relative_to(ROOT.resolve()) and path.stat().st_size == payload['bytes'],
            'Existing p1 payload is preserved; full hash belongs to fresh native recovery')
    reserved = value['reserved_unconsumed_p2']
    require(reserved['id'] == 'p000000002' and value['p2_tensor_pointer_metadata_and_writing_files_absent'] is True
            and value['second_CPU_child_started'] is False
            and all(not Path(reserved[key]).exists() for key in
                    ('metadata_path', 'payload_path', 'pointer_path', 'temporary_path')),
            'p2 is only a retained reservation, never a consumed or published update')
    return value


class OperationalOwnerProofCallbacks(OwnerProofCallbacks):
    """Actual callback sources plus separate immutable creator contract.

    .sources remains the truthful source of the functions that execute. The
    legacy maps are used only to validate the historical owner and policy.
    """
    def __init__(self, *, compatibility, identity, sources, functions):
        review = checked_compatibility(compatibility, identity)
        require(same(sources, review['operational_proof_callback_sources']),
                'Exact actual operational callback map')
        super().__init__(sources, functions)
        self.compatibility = deepcopy(compatibility)
        self.identity = deepcopy(identity)
        self.legacy_contract_sources = deepcopy(review['legacy_proof_callback_sources'])
        self.legacy_lifecycle_contract_sources = deepcopy(review['legacy_lifecycle_callback_sources'])
        self.operational_lifecycle_sources = deepcopy(review['operational_lifecycle_callback_sources'])

    def verify(self, name, request):
        checked_compatibility(self.compatibility, self.identity)
        value = deepcopy(request)
        value['operational_compatibility'] = deepcopy(self.compatibility)
        return super().verify(name, value)
