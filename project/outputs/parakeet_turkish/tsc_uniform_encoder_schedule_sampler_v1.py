"""Matched encoder schedule over the carried second uniform shuffle cycle.

Strict frozen own-source25109 restoration precedes admission. Cycle2 already
contains2 visits;18129 whole14TR/2RP updates visit its remaining253796 entries
and10 entries from cycle3. Original first-pass quota/end25109 stays immutable.
No queue, sampler RNG, seen/audio counter or replay stream is reset. The caller
owns CLOSED all207/recipe/source admission and schedule math; this class only
extends matched exposure and independently reports carried/new/tail visits.
"""
from __future__ import annotations

from copy import deepcopy

from tsc_uniform_majority_ptfocus_sampler_v1 import (
    TSCUniformMajorityPTFocusSampler, KIND as PARENT_KIND, RUNTIME_FIELDS,
)
from tsc_uniform_majority_sampler_v1 import TSCUniformMajoritySampler
from tsc_uniform_full_sampler_v1 import canonical, require

KIND = 'TSC_uniform_matched_encoder_next_cycle_sampler_v1'
SOURCE_END_STEP = 25109
MATCHED_UPDATES = 18129
TARGET_POOL_CLIPS = 253798
PARENT_SOURCE_SHA256 = '61f0c253c09abe49a6dbbf47f6ab4cbf3b0d6fdbf75d6f4490d59f2f381c2a2e'


class TSCUniformEncoderScheduleSampler(TSCUniformMajorityPTFocusSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.matched_cycle_transition = None
        self._strict_closed_source_sha256 = None
        self.original_parent_identity = None

    @property
    def original_full_pass_end_step(self):
        return super().configured_end_step

    @property
    def configured_end_step(self):
        original = self.original_full_pass_end_step
        return original if self.matched_cycle_transition is None else self.matched_cycle_transition['configured_matched_end_step']

    def load_state_dict(self, state):
        if self.matched_cycle_transition is None:
            require(state.get('updates') == SOURCE_END_STEP
                    and self.original_full_pass_end_step == SOURCE_END_STEP,
                    'Strict CLOSED completed original25109 required before technical extension')
            TSCUniformMajorityPTFocusSampler.load_state_dict(self, state)
            stream = self.turkish_streams['uniform']
            require(len(self.catalog) == TARGET_POOL_CLIPS and len(stream.seen) == len(self.catalog)
                    and stream.draws == len(self.catalog)+2
                    and stream.cycles_started == 2 and stream.position == 2,
                    'Actual completed pool253798 and carried second-cycle position2 required')
            self._strict_closed_source_sha256 = canonical(state)
            return

        require(state.get('schema') == 1 and state.get('kind') == KIND
                and state.get('identity') == self.identity
                and state.get('matched_cycle_transition') == self.matched_cycle_transition
                and state.get('quota_transition') == self.quota_transition
                and state.get('PT_focus_transition') == self.PT_focus_transition
                and canonical(state.get('source_prefix')) == canonical(self.source_prefix),
                'Own technical identity/original quota/selector/prefix changed')
        step = state.get('updates')
        require(type(step) is int and SOURCE_END_STEP <= step <= self.configured_end_step
                and state.get('quota_segments') == self._segments(step),
                'Technical completed frontier or explicit segments changed')
        n = step-SOURCE_END_STEP
        require(state['turkish_streams']['uniform']['draws']
                == self.matched_cycle_transition['source_uniform_draws']+14*n
                and sum(v['draws'] for v in state['replay'].values())
                == self.matched_cycle_transition['source_replay_draws']+2*n,
                'Technical14TR/2replay draw clocks changed')

        # Validate unchanged queues/RNG/seen/audio and piecewise counts through
        # the frozen real parent. The copied bound is only a validation view;
        # never replace the durable original quota or parent source identity.
        quota = deepcopy(self.quota_transition)
        quota['configured_end_step'] = self.configured_end_step
        quota['remaining_update_budget'] += MATCHED_UPDATES
        majority_identity = deepcopy(self.parent_majority_identity)
        majority_identity['majority_quota_transition'] = deepcopy(quota)
        focus_identity = deepcopy(self.original_parent_identity)
        focus_identity['majority_quota_transition'] = deepcopy(quota)
        validator = object.__new__(TSCUniformMajorityPTFocusSampler)
        validator.__dict__ = self.__dict__.copy()
        validator.quota_transition = quota
        validator.parent_majority_identity = majority_identity
        validator.identity = focus_identity
        view = {**state, 'kind': PARENT_KIND, 'identity': focus_identity,
                'quota_transition': quota,
                'quota_segments': TSCUniformMajoritySampler._segments(validator, step)}
        TSCUniformMajorityPTFocusSampler.load_state_dict(validator, view)
        for key in RUNTIME_FIELDS:
            if key not in ('quota_transition', 'source_prefix'):
                setattr(self, key, getattr(validator, key))

    def begin_matched_cycle(self, context):
        require(self.matched_cycle_transition is None and self._strict_closed_source_sha256 is not None,
                'Actual frozen own-source loader must succeed first; one technical extension only')
        require(context.get('source_step') == SOURCE_END_STEP == self.updates
                and context.get('encoder_schedule_updates') == MATCHED_UPDATES
                and context['parent_identity']['domain_sampling_identity'] == self.identity,
                'Only explicitly bound same-source18129 next-cycle updates admitted')
        require(self.original_full_pass_end_step == SOURCE_END_STEP
                and context.get('configured_end_step') == SOURCE_END_STEP+MATCHED_UPDATES,
                'Original end25109 and matched end43238 must be declared separately')
        stream = self.turkish_streams['uniform']
        remaining = len(self.catalog)-stream.position
        require((remaining+13)//14 == MATCHED_UPDATES, 'Exact whole-update next-cycle bound differs')
        self.original_parent_identity = deepcopy(self.identity)
        self.matched_cycle_transition = {
            'kind': 'declared_carried_next_shuffle_cycle_encoder_schedule_control_v1',
            'source_step': SOURCE_END_STEP, 'updates': MATCHED_UPDATES,
            'original_full_pass_end_step': SOURCE_END_STEP,
            'configured_matched_end_step': SOURCE_END_STEP+MATCHED_UPDATES,
            'source_sampler_sha256': self._strict_closed_source_sha256,
            'source_identity_sha256': canonical(self.original_parent_identity),
            'parent_sampler_source_sha256': PARENT_SOURCE_SHA256,
            'source_uniform_draws': stream.draws,
            'source_uniform_unique_clips': len(stream.seen),
            'cycle_index': 2, 'carried_cycle_visits': 2,
            'cycle_target_clips': len(self.catalog), 'remaining_cycle_visits': remaining,
            'declared_next_cycle_tail_draws': MATCHED_UPDATES*14-remaining,
            'source_replay_draws': sum(v.draws for v in self.replay.values()),
            'source_checkpoint': deepcopy(context.get('source_checkpoint')),
            'TR_per_update': 14, 'replay_per_update': 2,
            'PT_per_update': 1, 'old_five_per_update': 1,
            'queue_shuffle_rng_or_seen_reset': False,
            'additional_unique_first_pass_claimed': False,
        }
        self.identity = deepcopy(self.original_parent_identity)
        self.identity.update(sampler_family=KIND,
                             matched_cycle_execution_extension=deepcopy(self.matched_cycle_transition))
        return deepcopy(self.matched_cycle_transition)

    def next_batches(self):
        require(self.matched_cycle_transition is not None,
                'No technical draw before explicitly approved extension')
        return super().next_batches()

    def _segments(self, updates):
        if self.matched_cycle_transition is None:
            return super()._segments(updates)
        require(type(updates) is int and self.transition_source_step <= updates <= self.configured_end_step,
                'Technical step outside declared18129 next-cycle extension')
        segments = TSCUniformMajoritySampler._segments(self, min(updates, SOURCE_END_STEP))
        if updates > SOURCE_END_STEP:
            n = updates-SOURCE_END_STEP
            segments.append({'first_step': SOURCE_END_STEP+1, 'last_step': updates,
                             'updates': n, 'TR_per_update': 14, 'replay_per_update': 2,
                             'TR_draws': 14*n, 'replay_draws': 2*n,
                             'matched_cycle': True})
        return segments

    def state_dict(self):
        require(self.matched_cycle_transition is not None, 'No unsourced technical serialization')
        value = super().state_dict()
        value.update(kind=KIND, matched_cycle_transition=deepcopy(self.matched_cycle_transition))
        return value

    def coverage(self):
        value = super().coverage()
        if self.matched_cycle_transition is not None:
            n = self.updates-SOURCE_END_STEP
            draws = 14*n
            remaining = self.matched_cycle_transition['remaining_cycle_visits']
            new_cycle_visits = min(draws, remaining)
            carried = self.matched_cycle_transition['carried_cycle_visits']
            value['matched_cycle'] = {
                **deepcopy(self.matched_cycle_transition), 'local_updates': n,
                'additional_TR_draws': draws, 'additional_replay_draws': 2*n,
                'new_cycle_visits': new_cycle_visits,
                'cycle_unique': carried+new_cycle_visits,
                'full_cycle_complete': new_cycle_visits == remaining,
                'target': self.matched_cycle_transition['cycle_target_clips'],
                'tail_draws': max(0, draws-remaining),
                'new_unique_first_pass_clips': 0,
                'carried_visits_not_claimed_new': True,
            }
        return value
