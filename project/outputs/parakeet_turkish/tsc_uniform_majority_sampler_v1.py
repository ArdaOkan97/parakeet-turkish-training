"""Declared 14-TR/2-replay continuation of an existing uniform TRAIN queue.

Admission of the CLOSED checkpoint belongs to the caller. This sampler checks
its frozen 8/8 source state, carries every queue/RNG/counter, and records the
quota boundary. It never opens a checkpoint, reshuffles on migration, or resets
the original uniform coverage clock. Exact restores require the same original
CLOSED source migration first. Physical batches remain 8TR then 6TR+2replay.
"""
from __future__ import annotations

from copy import deepcopy
import math

from tsc_uniform_full_sampler_v1 import (
    TSCUniformFullSampler, IndexedIDSet, OLD_LANGUAGES, KIND as SOURCE_KIND,
    canonical, require,
)

KIND = 'TSC_uniform_majority_TR_six_replay_sampler_v1'
SOURCE_SAMPLER_SHA256 = '8ebd56d20e47a568db4a7cfe50ed172ad86af253275b3b1172ad78a34b60439a'


class TSCUniformMajoritySampler(TSCUniformFullSampler):
    def __init__(self, tr, replay, *, portuguese_replay, portuguese_seed, **kwargs):
        require(kwargs.get('batch_size') == 8 and kwargs.get('accumulation') == 2
                and kwargs.get('replay_fraction') == .125,
                'Declared physical8 accumulation2 with14TR/2replay required')
        super().__init__(tr, replay, portuguese_replay=portuguese_replay,
                         portuguese_seed=portuguese_seed, **kwargs)
        self.quota_transition = None

    @property
    def transition_source_step(self):
        require(self.quota_transition is not None, 'CLOSED uniform source migration required')
        return self.quota_transition['source_step']

    @property
    def remaining_update_budget(self):
        require(self.quota_transition is not None, 'CLOSED uniform source migration required')
        return self.quota_transition['remaining_update_budget']

    @property
    def configured_end_step(self):
        return self.transition_source_step + self.remaining_update_budget

    @property
    def uniform_pass_updates(self):
        # This means remaining updates from H, never a fresh full-pool restart.
        return self.remaining_update_budget

    def migrate_source_state(self, state, source_identity):
        raise ValueError('A majority transition requires existing own-uniform state, not a fresh domain-to-uniform migration')

    def migrate_closed_state(self, state, source_identity):
        require(self.quota_transition is None and state.get('schema') == 1
                and state.get('kind') == SOURCE_KIND and state.get('identity') == source_identity,
                'One immutable existing8TR/8replay uniform source required')
        prefix = state.get('source_prefix', {})
        start, boundary = prefix.get('updates'), state.get('updates')
        require(type(start) is int and start > 0 and type(boundary) is int and boundary > start,
                'Actual original uniform source and advanced CLOSED frontier required')
        expected = deepcopy(self.identity)
        expected['replay_per_update'] = 8
        expected.update(source_boundary_step=start,
            source_domain_identity_sha256=canonical(prefix.get('source_identity')),
            source_sampler_sha256=prefix.get('source_sampler_sha256'))
        require(source_identity == expected, 'Source pool/catalog/seeds/labels/layout or original uniform identity changed')
        # Validate through the actual immutable source loader before carrying its
        # state. Reuse stream objects without making a second TR permutation.
        validator = object.__new__(TSCUniformFullSampler)
        validator.__dict__ = self.__dict__.copy()
        validator.identity = deepcopy(source_identity)
        validator.replay_per_update, validator.tr_per_update = 8, 8
        validator.source_prefix = deepcopy(prefix)
        TSCUniformFullSampler.load_state_dict(validator, state)
        stream = validator.turkish_streams['uniform']
        prior = stream.draws
        require(prior == 8 * (boundary - start) and 0 < prior < len(self.catalog)
                and len(stream.seen) == prior and stream.cycles_started == 1,
                'Continue the unfinished first uniform pass without reset or hidden repeats')
        for key in ('turkish_streams', 'replay', 'tr_rng', 'replay_rng', 'pt_rng',
                    'updates', 'replay_language_counts', 'domain_draws', 'domain_seen',
                    'domain_audio_seconds', 'group_draws', 'group_seen', 'group_audio_seconds', 'source_prefix'):
            setattr(self, key, getattr(validator, key))
        remaining = len(self.catalog) - prior
        budget = math.ceil(remaining / 14)
        self.quota_transition = {
            'kind': 'declared_existing_uniform_8_8_to_14_2_quota_v1',
            'source_step': boundary, 'original_uniform_source_step': start,
            'prior_uniform_updates': boundary - start, 'prior_uniform_draws': prior,
            'prior_replay_draws': sum(value.draws for value in self.replay.values()),
            'prior_TR_per_update': 8, 'prior_replay_per_update': 8,
            'new_TR_per_update': 14, 'new_replay_per_update': 2,
            'remaining_TR_clips_at_boundary': remaining,
            'remaining_update_budget': budget, 'configured_end_step': boundary + budget,
            'last_whole_update_next_cycle_repeats': budget * 14 - remaining,
            'source_uniform_sampler_sha256': canonical(state),
            'source_uniform_identity_sha256': canonical(source_identity),
            'source_original_prefix_sha256': canonical(self.source_prefix),
            'source_sampler_source_sha256': SOURCE_SAMPLER_SHA256,
            'fresh_queue_shuffle_or_rng_reset': False,
        }
        self.identity = deepcopy(source_identity)
        self.identity.update(sampler_family=KIND, replay_per_update=2,
                             majority_quota_transition=deepcopy(self.quota_transition))
        self.replay_per_update, self.tr_per_update = 2, 14
        return {'kind': 'actual_sampler_8_8_to_14_2_state_projection_v1', 'passed': True,
            'quota_transition': deepcopy(self.quota_transition),
            'TR_order_cursor_seen_and_rng_preserved': True,
            'all_six_replay_queues_counts_and_rng_preserved': True,
            'original_source_prefix_and_cumulative_counters_preserved': True,
            'PT_grace_or_optimizer_history_counters_reset': False,
            'future_TR_visits_are_original_queue_suffix': True,
            'physical_microbatch_composition': [[8, 0], [6, 2]]}

    def _segments(self, updates):
        source = self.quota_transition
        require(source is not None and type(updates) is int
                and source['source_step'] <= updates <= source['configured_end_step'],
                'Completed frontier outside approved remaining first pass')
        n = updates - source['source_step']
        return [
            {'first_step': source['original_uniform_source_step'] + 1,
             'last_step': source['source_step'], 'updates': source['prior_uniform_updates'],
             'TR_per_update': 8, 'replay_per_update': 8,
             'TR_draws': source['prior_uniform_draws'], 'replay_draws': 8 * source['prior_uniform_updates']},
            {'first_step': source['source_step'] + 1, 'last_step': updates, 'updates': n,
             'TR_per_update': 14, 'replay_per_update': 2, 'TR_draws': 14 * n, 'replay_draws': 2 * n},
        ]

    def next_batches(self):
        require(self.quota_transition is not None and self.updates < self.configured_end_step,
                'No unsourced draw or hidden extension beyond the remaining fullpass')
        return super().next_batches()

    def coverage(self):
        require(self.quota_transition is not None, 'CLOSED source required for majority coverage')
        value = super().coverage()
        n = self.updates - self.transition_source_step
        value['uniform_pass']['quota_segments'] = self._segments(self.updates)
        value['majority_transition'] = {**deepcopy(self.quota_transition), 'local_updates': n,
            'TR_draws_after_transition': n * 14, 'replay_draws_after_transition': n * 2,
            'physical_microbatch_composition': [[8, 0], [6, 2]]}
        return value

    def state_dict(self):
        require(self.quota_transition is not None, 'No unsourced majority serialization')
        value = super().state_dict()
        value.update(kind=KIND, quota_transition=deepcopy(self.quota_transition),
                     quota_segments=self._segments(self.updates))
        return value

    def load_state_dict(self, state):
        require(self.quota_transition is not None and state.get('schema') == 1
                and state.get('kind') == KIND and state.get('identity') == self.identity
                and state.get('quota_transition') == self.quota_transition
                and canonical(state.get('source_prefix')) == canonical(self.source_prefix),
                'Own majority identity and original CLOSED boundary must remain exact')
        updates = state.get('updates')
        require(state.get('quota_segments') == self._segments(updates), 'Piecewise8/8 then14/2 segment layout changed')
        require(list(state['turkish_streams']) == ['uniform']
                and tuple(state['replay']) == (*OLD_LANGUAGES, 'pt'), 'Uniform/six replay stream order changed')
        stream = self.turkish_streams['uniform']
        stream.load_state_dict(state['turkish_streams']['uniform'])
        for name, value in self.replay.items():
            value.load_state_dict(state['replay'][name])
        n = updates - self.transition_source_step
        require(stream.draws == self.quota_transition['prior_uniform_draws'] + 14 * n
                and sum(value.draws for value in self.replay.values()) == self.quota_transition['prior_replay_draws'] + 2 * n
                and state['replay_language_counts'] == {name: value.draws for name, value in self.replay.items()},
                'Actual piecewise TR/replay draw frontiers differ')
        draws = self.source_prefix['domain_draws'].copy()
        groups = self.source_prefix['source_group_draws'].copy()
        seen = {name: IndexedIDSet(self.catalog, ids) for name, ids in self.source_prefix['domain_seen'].items()}
        group_seen = {name: IndexedIDSet(self.catalog, ids) for name, ids in self.source_prefix['source_group_seen'].items()}
        for name, count in self.domain_pool_sizes.items():
            draws[name] += max(0, stream.cycles_started - 1) * count
        for name, count in self.group_pool_sizes.items():
            groups[name] += max(0, stream.cycles_started - 1) * count
        for index in (stream.rows[:stream.position] if stream.cycles_started else []):
            group = self.catalog.groups[index]
            domain = 'fleurs' if group == 'fleurs' else 'cv_tsc'
            draws[domain] += 1
            groups[group] += 1
        for index in stream.seen.indices():
            group = self.catalog.groups[index]
            seen['fleurs' if group == 'fleurs' else 'cv_tsc'].add_index(index)
            group_seen[group].add_index(index)
        require(state['domain_draws'] == draws and state['source_group_draws'] == groups
                and state['domain_seen'] == {name: sorted(ids) for name, ids in seen.items()}
                and state['source_group_seen'] == {name: sorted(ids) for name, ids in group_seen.items()},
                'Carried historical plus actual uniform source coverage changed')
        for key, pools in (('domain_audio_seconds', self.domain_pool_sizes),
                           ('source_group_audio_seconds', self.group_pool_sizes)):
            require(set(state[key]) == set(pools) and all(type(value) in (int, float)
                    and math.isfinite(value) and value >= self.source_prefix[key][name]
                    for name, value in state[key].items()), 'Carried finite cumulative audio counters required')
        require(math.isclose(math.fsum(state['domain_audio_seconds'].values()),
                math.fsum(state['source_group_audio_seconds'].values()), rel_tol=1e-10, abs_tol=1e-7),
                'Domain and source audio counters disagree')
        self.tr_rng.setstate(state['turkish_rng'])
        self.replay_rng.setstate(state['replay_rng'])
        self.pt_rng.setstate(state['Portuguese_rng'])
        self.updates = updates
        self.replay_language_counts = state['replay_language_counts'].copy()
        self.domain_draws, self.domain_seen = draws, seen
        self.domain_audio_seconds = state['domain_audio_seconds'].copy()
        self.group_draws, self.group_seen = groups, group_seen
        self.group_audio_seconds = state['source_group_audio_seconds'].copy()
