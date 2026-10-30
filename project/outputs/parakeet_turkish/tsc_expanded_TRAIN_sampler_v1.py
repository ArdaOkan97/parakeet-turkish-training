"""Matched fresh TRAIN coverage epochs, carrying the six replay streams exactly.

Initial admission strict-loads the frozen original matched-family40750 state on
the ORIGINAL three-group catalog. Its TR cursor/seen/order are parked by digest
and immutable source checkpoint, never presented as an expanded-epoch prefix.
The new indexed queue uses the copied current TR RNG and shuffles on first draw.
Exact own-state admission needs the original new-run sampling identity, not the
original40750 tensor checkpoint. No model/checkpoint/corpus reads occur here.
"""
from __future__ import annotations
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
import re

from tsc_uniform_full_sampler_v1 import (
    IndexedBucketCycler, IndexedIDSet, CatalogGroups, OLD_LANGUAGES, canonical, require,
)
from tsc_uniform_encoder_schedule_sampler_v1 import (
    TSCUniformEncoderScheduleSampler, KIND as PARENT_KIND,
)
from tsc_uniform_majority_ptfocus_sampler_v1 import KIND as FOCUS_KIND
from tsc_uniform_majority_sampler_v1 import KIND as MAJORITY_KIND

KIND = 'TSC_uniform_expanded_TRAIN_epoch_sampler_v1'
SOURCE_STEP = 40750
UPDATES = 25198
END_STEP = 65948
BASE_POOL = 253798
EXPANDED_POOL = 352768
UNIFORM_ORIGIN = 6500
QUOTA_ORIGIN = 7621
PT_ORIGIN = 8000
MATCHED_ORIGIN = 25109
MATCHED_UPDATES = 18129
MATCHED_END = 43238
PARENT_SOURCE_SHA256 = '8177ae6195ee9ddda042258437c00358f4a6758d18225e3bc7c3735301391ba9'
LANGUAGES = (*OLD_LANGUAGES, 'pt')


def streamed_sha(value):
    digest = hashlib.sha256()
    for chunk in json.JSONEncoder(sort_keys=True, separators=(',', ':'),
                                  ensure_ascii=False, allow_nan=False).iterencode(value):
        digest.update(chunk.encode('utf-8'))
    return digest.hexdigest()


def _record(value):
    require(isinstance(value, dict) and isinstance(value.get('path'), str)
            and re.fullmatch('[0-9a-f]{64}', str(value.get('sha256', '')))
            and type(value.get('bytes')) is int and value['bytes'] > 0,
            'Hash-bound immutable chosen source checkpoint record required')
    return deepcopy(value)


class _Domains(Mapping):
    def __init__(self, groups): self.groups = groups
    def __len__(self): return len(self.groups)
    def __iter__(self): return iter(self.groups)
    def __getitem__(self, source):
        return 'fleurs' if self.groups[source] == 'fleurs' else 'non_fleurs_TR'


class TSCExpandedTrainSampler:
    def __init__(self, original_tr, active_tr, replay, *, portuguese_replay,
                 portuguese_seed, data_arm, **kwargs):
        require(data_arm in ('base', 'expanded') and kwargs.get('sampling_mode') == 'uniform'
                and kwargs.get('batch_size') == 8 and kwargs.get('accumulation') == 2
                and kwargs.get('replay_fraction') == .125,
                'Explicit matched base/expanded uniform14TR2RP physical8x2 required')
        self._parent = TSCUniformEncoderScheduleSampler(original_tr, replay,
            portuguese_replay=portuguese_replay, portuguese_seed=portuguese_seed, **kwargs)
        self._base_identity = deepcopy(self._parent.identity)
        self.original_catalog = original_tr
        self.catalog = active_tr
        self.data_arm = data_arm
        require(len(original_tr) == BASE_POOL, 'Exact original accepted253798 catalog required')
        for name in ('source_ids', 'index_by_source_id', 'encoder_frames', 'target_tokens',
                     'durations', 'groups', 'length_key', 'fingerprint', 'ids_sha', 'receipt'):
            require(hasattr(active_tr, name), 'Complete indexed active TRAIN catalog required: ' + name)
        receipt = active_tr.receipt
        provenance = receipt.get('provenance', {})
        counts = dict(Counter(active_tr.groups))
        require(receipt.get('complete') is True and receipt.get('rows') == len(active_tr)
                and provenance.get('training_admission_complete') is True
                and provenance.get('all_rows_verified') is True
                and provenance.get('verified_clips') == len(active_tr)
                and receipt.get('fingerprint') == active_tr.fingerprint
                and receipt.get('source_ids_sha256') == active_tr.ids_sha
                and receipt.get('source_group_counts') == counts,
                'Active catalog must bind completed whole-row admission and scalar identity')
        old_counts = dict(original_tr.receipt['source_group_counts'])
        if data_arm == 'base':
            require(len(active_tr) == BASE_POOL and active_tr.fingerprint == original_tr.fingerprint
                    and active_tr.ids_sha == original_tr.ids_sha and counts == old_counts,
                    'Base arm must use the identical accepted original catalog')
        else:
            require(len(active_tr) == EXPANDED_POOL and set(counts) == {'fleurs', 'cv', 'tsc', 'worldspeech'}
                    and all(counts[name] == value for name, value in old_counts.items())
                    and counts['worldspeech'] == EXPANDED_POOL - BASE_POOL
                    and provenance.get('original_carrier_fingerprint') == original_tr.fingerprint
                    and provenance.get('original_rows_preserved_value_for_value_in_same_prefix_order') is True
                    and receipt.get('full_original_row_values_and_key_order_preserved') is True,
                    'Expanded arm must preserve every original row and truthful WorldSpeech suffix')
        for row in (*replay, *portuguese_replay):
            require(row['source_id'] not in active_tr.index_by_source_id,
                    'Replay and all active TRAIN source IDs must be disjoint')
        self.tr_rng, self.replay_rng, self.pt_rng = self._parent.tr_rng, self._parent.replay_rng, self._parent.pt_rng
        self.replay = self._parent.replay
        self.replay_language_counts = {name: 0 for name in LANGUAGES}
        self.batch_size, self.accumulation, self.effective_batch = 8, 2, 16
        self.tr_per_update, self.replay_per_update = 14, 2
        self.sampling_mode = 'uniform'
        self.tr_by_id = active_tr.index_by_source_id
        self.source_groups = CatalogGroups(active_tr)
        self.source_domains = _Domains(self.source_groups)
        self.group_pool_sizes = counts
        self.domain_pool_sizes = {'fleurs': counts['fleurs'],
                                  'non_fleurs_TR': len(active_tr) - counts['fleurs']}
        self._static_identity = deepcopy(self._base_identity)
        self._static_identity.update(sampler_family=KIND, source_sampling_mode=PARENT_KIND,
            turkish_pool_fingerprint=active_tr.fingerprint, turkish_uniform_pool_size=len(active_tr),
            source_group_pool_counts=counts.copy(), indexed_TR_fingerprint=active_tr.fingerprint,
            indexed_TR_ids_sha256=active_tr.ids_sha, replay_per_update=2,
            second_domain='uniform_all_accepted_TR', data_arm=data_arm,
            original_TR_fingerprint=original_tr.fingerprint, original_TR_ids_sha256=original_tr.ids_sha,
            source_boundary_step=SOURCE_STEP, original_uniform_origin_step=UNIFORM_ORIGIN,
            quota_boundary_step=QUOTA_ORIGIN, PT_focus_boundary_step=PT_ORIGIN,
            additional_update_budget=UPDATES, configured_end_step=END_STEP,
            new_TR_epoch_local_zero=True, parked_TR_prefix_not_claimed_as_new=True)
        self.identity = deepcopy(self._static_identity)
        self.data_expansion_transition = None
        self.source_prefix = None
        self.updates = SOURCE_STEP
        self._fresh_active_queue()

    def _fresh_active_queue(self):
        self.turkish_streams = {'uniform': IndexedBucketCycler(self.catalog, self.tr_rng,
                                                              self._base_identity['bucket_size'])}
        self.group_draws = {name: 0 for name in self.group_pool_sizes}
        self.group_audio_seconds = {name: 0. for name in self.group_pool_sizes}
        self.group_seen = {name: IndexedIDSet(self.catalog) for name in self.group_pool_sizes}

    @property
    def source_step(self): return SOURCE_STEP
    @property
    def remaining_update_budget(self): return UPDATES
    @property
    def configured_end_step(self): return END_STEP
    @property
    def original_uniform_origin(self): return UNIFORM_ORIGIN
    @property
    def transition_source_step(self): return QUOTA_ORIGIN
    @property
    def focus_source_step(self): return PT_ORIGIN
    @property
    def uniform_pass_updates(self): return UPDATES
    @property
    def tr_draws(self):
        prior = sum(self.source_prefix['prior_source_group_draws'].values()) if self.source_prefix else 0
        return prior + self.turkish_streams['uniform'].draws
    @property
    def tr_unique(self):
        if self.source_prefix is None:
            return 0
        # The strict source loader proved all original rows were already seen.
        return BASE_POOL + (len(self.group_seen['worldspeech']) if 'worldspeech' in self.group_seen else 0)
    @property
    def local_turkish_epoch(self): return self.turkish_streams['uniform'].draws / len(self.catalog)
    @property
    def turkish_epoch(self):
        base = self.source_prefix['source_turkish_epoch'] if self.source_prefix else 0.
        return base + self.local_turkish_epoch

    def _parent_identities(self, parent_identity):
        quota = deepcopy(parent_identity.get('majority_quota_transition'))
        pt = deepcopy(parent_identity.get('PT_focus_transition'))
        matched = deepcopy(parent_identity.get('matched_cycle_execution_extension'))
        require(isinstance(quota, dict) and isinstance(pt, dict) and isinstance(matched, dict)
                and parent_identity.get('source_boundary_step') == UNIFORM_ORIGIN
                and quota.get('source_step') == QUOTA_ORIGIN
                and quota.get('original_uniform_source_step') == UNIFORM_ORIGIN
                and quota.get('configured_end_step') == MATCHED_ORIGIN
                and pt.get('source_step') == PT_ORIGIN
                and matched.get('source_step') == MATCHED_ORIGIN
                and matched.get('updates') == MATCHED_UPDATES
                and matched.get('configured_matched_end_step') == MATCHED_END
                and matched.get('cycle_target_clips') == BASE_POOL
                and matched.get('source_uniform_draws') == BASE_POOL + 2,
                'Frozen original uniform/quota/PT/matched lineage and actual catalog must match')
        majority = deepcopy(self._base_identity)
        majority.update(source_boundary_step=UNIFORM_ORIGIN,
            source_domain_identity_sha256=parent_identity.get('source_domain_identity_sha256'),
            source_sampler_sha256=parent_identity.get('source_sampler_sha256'),
            sampler_family=MAJORITY_KIND, replay_per_update=2, majority_quota_transition=quota)
        focus = deepcopy(majority)
        focus.update(sampler_family=FOCUS_KIND, PT_focus_transition=pt)
        expected = deepcopy(focus)
        expected.update(sampler_family=PARENT_KIND, matched_cycle_execution_extension=matched)
        require(expected == parent_identity and pt.get('source_identity_sha256') == canonical(majority)
                and matched.get('source_identity_sha256') == canonical(focus),
                'Original metadata/labels/seeds/replay identity or frozen lineage digest changed')
        return majority, focus, quota, pt, matched

    def _context(self, context, parent_identity):
        require(context.get('source_step') == SOURCE_STEP and context.get('data_arm') == self.data_arm
                and context.get('data_expansion_updates') == UPDATES
                and context.get('configured_end_step') == END_STEP == SOURCE_STEP + UPDATES
                and context.get('TR_pool_clips') == len(self.catalog)
                and context.get('parent_identity', {}).get('domain_sampling_identity') == parent_identity,
                'Exact qualified40750 source, matched arm/pool and25198-update bound required')
        self._parent_identities(parent_identity)
        return _record(context.get('source_checkpoint'))

    def _install_transition(self, lineage, parent_identity, checkpoint):
        transition = {'kind': 'declared_fresh_uniform_TRAIN_epoch_matched_data_arm_v1',
            'source_step': SOURCE_STEP, 'data_arm': self.data_arm,
            'additional_updates': UPDATES, 'configured_end_step': END_STEP,
            'source_checkpoint': checkpoint, 'parent_identity_sha256': canonical(parent_identity),
            'parent_sampler_source_sha256': PARENT_SOURCE_SHA256,
            'original_pool_fingerprint': self.original_catalog.fingerprint,
            'active_pool_fingerprint': self.catalog.fingerprint, 'active_ids_sha256': self.catalog.ids_sha,
            'active_pool_clips': len(self.catalog), 'active_source_group_counts': self.group_pool_sizes.copy(),
            'TR_per_update': 14, 'Portuguese_per_update': 1, 'old_five_per_update': 1,
            'new_queue_shuffled_at_admission': False, 'TR_RNG_copied': True,
            'six_replay_queues_and_RNG_copied': True,
            'old_TR_queue_parked_not_reinterpreted': True,
            'source_lineage': deepcopy(lineage)}
        self.data_expansion_transition = transition
        self.source_prefix = deepcopy(lineage)
        self.identity = deepcopy(self._static_identity)
        self.identity['data_expansion_transition'] = deepcopy(transition)

    def migrate_closed_state(self, state, parent_identity, context):
        require(self.data_expansion_transition is None and self._parent is not None,
                'One initial CLOSED source migration only')
        checkpoint = self._context(context, parent_identity)
        require(state.get('schema') == 1 and state.get('kind') == PARENT_KIND
                and state.get('updates') == SOURCE_STEP and state.get('identity') == parent_identity,
                'Actual frozen own matched-family40750 sampler required')
        majority, focus, quota, pt, matched = self._parent_identities(parent_identity)
        prefix = state.get('source_prefix')
        require(streamed_sha(prefix) == quota.get('source_original_prefix_sha256')
                and state.get('quota_transition') == quota and state.get('PT_focus_transition') == pt
                and state.get('matched_cycle_transition') == matched,
                'Actual parked prefix/quota/PT/matched state differs from source identity')
        validator = self._parent
        validator.source_prefix = prefix
        validator.quota_transition = quota
        validator.PT_focus_transition = pt
        validator.parent_majority_identity = majority
        validator.original_parent_identity = focus
        validator.matched_cycle_transition = matched
        validator.identity = deepcopy(parent_identity)
        validator.load_state_dict(state)  # Actual frozen strict loader; no source projection shortcut.
        stream = validator.turkish_streams['uniform']
        require(len(stream.seen) == BASE_POOL and stream.draws == BASE_POOL + 2 + 14 * (SOURCE_STEP - MATCHED_ORIGIN),
                'Actual full original pool and chosen40750 TR frontier required')
        lineage = {'kind': 'parked_genuine40750_original_TR_lineage_v1',
            'source_step': SOURCE_STEP, 'source_sampler_sha256': streamed_sha(state),
            'source_identity_sha256': canonical(parent_identity),
            'source_prefix_sha256': streamed_sha(prefix),
            'source_uniform_stream_sha256': streamed_sha(state['turkish_streams']['uniform']),
            'source_replay_state_sha256': streamed_sha(state['replay']),
            'source_turkish_rng_sha256': canonical(state['turkish_rng']),
            'source_replay_rng_sha256': canonical(state['replay_rng']),
            'source_Portuguese_rng_sha256': canonical(state['Portuguese_rng']),
            'original_uniform_origin_step': UNIFORM_ORIGIN, 'quota_boundary_step': QUOTA_ORIGIN,
            'PT_focus_boundary_step': PT_ORIGIN, 'prior_uniform_draws': stream.draws,
            'prior_uniform_unique': len(stream.seen), 'source_turkish_epoch': validator.turkish_epoch,
            'prior_replay_language_counts': validator.replay_language_counts.copy(),
            'prior_domain_draws': validator.domain_draws.copy(),
            'prior_domain_audio_seconds': validator.domain_audio_seconds.copy(),
            'prior_source_group_draws': validator.group_draws.copy(),
            'prior_source_group_audio_seconds': validator.group_audio_seconds.copy(),
            'prior_source_group_unique': {name: len(value) for name, value in validator.group_seen.items()}}
        self.tr_rng, self.replay_rng, self.pt_rng = validator.tr_rng, validator.replay_rng, validator.pt_rng
        self.replay = validator.replay
        self.replay_language_counts = validator.replay_language_counts.copy()
        self.updates = SOURCE_STEP
        self._fresh_active_queue()  # Constructor consumes no entropy; first draw does the shuffle.
        self._install_transition(lineage, parent_identity, checkpoint)
        self._parent = None  # No dormant original permutation/seen/source-state object retained.
        return {'kind': 'actual40750_to_matched_TRAIN_epoch_projection_v1', 'passed': True,
                'data_expansion_transition': deepcopy(self.data_expansion_transition),
                'new_TR_draws': 0, 'new_TR_unique': 0,
                'replay_state_and_RNG_preserved': True,
                'physical_microbatch_composition': [[8, 0], [6, 2]]}

    def initialize_from_own_state(self, state, context, parent_identity):
        require(self.data_expansion_transition is None, 'Own-state bootstrap only on a fresh sampler')
        checkpoint = self._context(context, parent_identity)
        own_identity = context.get('own_sampling_identity')
        require(isinstance(own_identity, dict) and state.get('identity') == own_identity,
                'Immutable original new-run own sampling identity must authorize exact bootstrap')
        transition = own_identity.get('data_expansion_transition', {})
        lineage = transition.get('source_lineage')
        require(isinstance(lineage, dict) and lineage.get('source_step') == SOURCE_STEP
                and lineage.get('source_identity_sha256') == canonical(parent_identity)
                and lineage.get('prior_uniform_draws') == BASE_POOL + 2 + 14 * (SOURCE_STEP - MATCHED_ORIGIN)
                and lineage.get('prior_uniform_unique') == BASE_POOL
                and lineage.get('source_turkish_epoch') == lineage['prior_uniform_draws'] / BASE_POOL,
                'Authorized parked original40750 coverage/RNG lineage required')
        self._install_transition(lineage, parent_identity, checkpoint)
        require(self.identity == own_identity, 'Own arm/catalog/source/seeds/transition identity changed')
        self.load_state_dict(state)
        self._parent = None
        return {'passed': True, 'own_sampler_state_restored': True, 'original40750_checkpoint_reloaded': False,
                'source_step': SOURCE_STEP, 'local_updates': self.updates - SOURCE_STEP}

    def next_batches(self):
        require(self.data_expansion_transition is not None and self.updates < self.configured_end_step,
                'No unsourced draw or extension beyond matched25198-update budget')
        tr = [self.turkish_streams['uniform'].next() for _ in range(14)]
        for row in tr:
            index = self.catalog.index_by_source_id[row['source_id']]
            group = self.catalog.groups[index]
            self.group_draws[group] += 1
            self.group_seen[group].add_index(index)
            self.group_audio_seconds[group] += float(row['duration'])
        old = self.replay_rng.choice(OLD_LANGUAGES)
        replay = [self.replay[old].next(), self.replay['pt'].next()]
        self.replay_language_counts[old] += 1
        self.replay_language_counts['pt'] += 1
        self.updates += 1
        return [tr[:8], [*tr[8:], *replay]]

    def coverage(self):
        require(self.data_expansion_transition is not None, 'Source admission required before coverage')
        stream = self.turkish_streams['uniform']
        n, unique = self.updates - SOURCE_STEP, len(stream.seen)
        epoch = {'source_step': SOURCE_STEP, 'data_arm': self.data_arm, 'local_updates': n,
            'draws': stream.draws, 'unique': unique, 'pool': len(self.catalog),
            'fraction': unique / len(self.catalog), 'full_pass_complete': unique == len(self.catalog),
            'unique_clips_seen': unique, 'pool_clips': len(self.catalog),
            'fraction_unique_seen': unique / len(self.catalog), 'repeat_visits': stream.draws - unique,
            'epoch_clock': self.local_turkish_epoch, 'new_epoch_not_inherited_prefix': True}
        groups = {name: {'pool_clips': count, 'draws': self.group_draws[name],
            'unique_clips_seen': len(self.group_seen[name]), 'audio_seconds': self.group_audio_seconds[name],
            'fraction_unique_seen': len(self.group_seen[name]) / count}
            for name, count in self.group_pool_sizes.items()}
        return {'sampler_family': KIND, 'sampling_mode': 'uniform', 'completed_updates': self.updates,
            'turkish_epoch': self.turkish_epoch, 'local_turkish_epoch': self.local_turkish_epoch,
            'epoch_clock': 'source_offset_plus_new_uniform_draw_equivalent',
            'data_expansion_epoch': epoch, 'uniform_pass': deepcopy(epoch), 'groups': groups,
            'replay': {name: {'draws': value.draws, 'unique_clips_seen': len(value.seen),
                              'pool_clips': len(value.rows)} for name, value in self.replay.items()},
            'Portuguese_replay_draws': self.replay['pt'].draws,
            'parked_source_lineage': deepcopy(self.source_prefix),
            'cumulative_TR_draws': self.tr_draws, 'cumulative_TR_unique_clips_seen': self.tr_unique,
            'cumulative_TR_audio_seconds': math.fsum(self.source_prefix['prior_source_group_audio_seconds'].values())
                                           + math.fsum(self.group_audio_seconds.values()),
            'cumulative_source_group_draws': {name: self.source_prefix['prior_source_group_draws'].get(name, 0) + self.group_draws[name]
                                             for name in self.group_pool_sizes},
            'physical_microbatch_composition': [[8, 0], [6, 2]]}

    def state_dict(self):
        require(self.data_expansion_transition is not None, 'No unsourced own sampler serialization')
        return {'schema': 1, 'kind': KIND, 'identity': deepcopy(self.identity), 'updates': self.updates,
            'data_expansion_transition': deepcopy(self.data_expansion_transition),
            'source_prefix': deepcopy(self.source_prefix),
            'turkish_streams': {'uniform': self.turkish_streams['uniform'].state_dict()},
            'replay': {name: value.state_dict() for name, value in self.replay.items()},
            'replay_language_counts': self.replay_language_counts.copy(),
            'turkish_rng': self.tr_rng.getstate(), 'replay_rng': self.replay_rng.getstate(),
            'Portuguese_rng': self.pt_rng.getstate(), 'source_group_draws': self.group_draws.copy(),
            'source_group_seen_counts': {name: len(value) for name, value in self.group_seen.items()},
            'source_group_audio_seconds': self.group_audio_seconds.copy()}

    def load_state_dict(self, state):
        require(self.data_expansion_transition is not None and state.get('schema') == 1
                and state.get('kind') == KIND and state.get('identity') == self.identity
                and state.get('data_expansion_transition') == self.data_expansion_transition
                and state.get('source_prefix') == self.source_prefix,
                'Own exact arm/catalog/source/seed identity and compact parked lineage must match')
        step = state.get('updates')
        require(type(step) is int and SOURCE_STEP <= step <= END_STEP
                and tuple(state.get('replay', {})) == LANGUAGES
                and list(state.get('turkish_streams', {})) == ['uniform'],
                'Exact own global frontier and six replay stream order required')
        n = step - SOURCE_STEP
        stream = self.turkish_streams['uniform']
        stream.load_state_dict(state['turkish_streams']['uniform'])
        require(stream.draws == 14 * n, 'Own local TR draws must equal14N')
        before = self.source_prefix['prior_replay_language_counts']
        for name, value in self.replay.items():
            value.load_state_dict(state['replay'][name])
        require(state['replay_language_counts'] == {name: value.draws for name, value in self.replay.items()}
                and self.replay['pt'].draws == before['pt'] + n
                and sum(self.replay[name].draws - before[name] for name in OLD_LANGUAGES) == n
                and all(self.replay[name].draws >= before[name] for name in LANGUAGES),
                'Exact carried replay+N PT/+N old-five counters required')
        draws = {name: max(0, stream.cycles_started - 1) * count for name, count in self.group_pool_sizes.items()}
        audio = {name: 0. for name in self.group_pool_sizes}
        if stream.cycles_started > 1:
            for index in range(len(self.catalog)):
                audio[self.catalog.groups[index]] += float(self.catalog.durations[index])
            audio = {name: value * (stream.cycles_started - 1) for name, value in audio.items()}
        for index in (stream.rows[:stream.position] if stream.cycles_started else ()):
            name = self.catalog.groups[index]
            draws[name] += 1
            audio[name] += float(self.catalog.durations[index])
        seen = {name: IndexedIDSet(self.catalog) for name in self.group_pool_sizes}
        for index in stream.seen.indices():
            seen[self.catalog.groups[index]].add_index(index)
        require(state['source_group_draws'] == draws
                and state['source_group_seen_counts'] == {name: len(value) for name, value in seen.items()},
                'Exact active source-group visit/unique counts differ')
        saved_audio = state['source_group_audio_seconds']
        require(set(saved_audio) == set(audio) and all(type(saved_audio[name]) in (int, float)
                and math.isfinite(saved_audio[name]) and saved_audio[name] >= 0
                and math.isclose(saved_audio[name], total, rel_tol=1e-10, abs_tol=1e-6)
                for name, total in audio.items()), 'Active finite source-group audio differs from actual indexed visits')
        self.tr_rng.setstate(state['turkish_rng'])
        self.replay_rng.setstate(state['replay_rng'])
        self.pt_rng.setstate(state['Portuguese_rng'])
        self.updates = step
        self.replay_language_counts = state['replay_language_counts'].copy()
        self.group_draws, self.group_seen = draws, seen
        self.group_audio_seconds = saved_audio.copy()
