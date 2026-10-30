"""Declared100-update technical I/O benchmark after an actual complete pass.

The original PT-focused loader first validates the CLOSED source25109. Original
quota/end/prefix/selector dictionaries remain immutable lineage. The extension
uses the same next TR/replay queues and RNG; it is not a second promised unique
pass or a scientific full-pass continuation. Only a temporary validation view
extends the frozen parent's count bound when loading this new own-state kind.
"""
from __future__ import annotations

from copy import deepcopy

from tsc_uniform_majority_ptfocus_sampler_v1 import (
    TSCUniformMajorityPTFocusSampler, KIND as PARENT_KIND, RUNTIME_FIELDS,
)
from tsc_uniform_majority_sampler_v1 import TSCUniformMajoritySampler
from tsc_uniform_full_sampler_v1 import canonical, require

KIND = 'TSC_uniform_same_future_batches_technical_IO_sampler_v1'
SOURCE_END_STEP = 25109
TECHNICAL_UPDATES = 100
PARENT_SOURCE_SHA256 = '61f0c253c09abe49a6dbbf47f6ab4cbf3b0d6fdbf75d6f4490d59f2f381c2a2e'


class TSCUniformIOSampler(TSCUniformMajorityPTFocusSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.technical_extension = None
        self._strict_closed_source_sha256 = None
        self.original_parent_identity = None

    @property
    def original_full_pass_end_step(self):
        return super().configured_end_step

    @property
    def configured_end_step(self):
        original = self.original_full_pass_end_step
        return original if self.technical_extension is None else self.technical_extension['configured_technical_end_step']

    def load_state_dict(self, state):
        if self.technical_extension is None:
            require(state.get('updates') == SOURCE_END_STEP
                    and self.original_full_pass_end_step == SOURCE_END_STEP,
                    'Strict CLOSED completed original25109 required before technical extension')
            TSCUniformMajorityPTFocusSampler.load_state_dict(self, state)
            stream = self.turkish_streams['uniform']
            require(len(stream.seen) == len(self.catalog)
                    and len(self.catalog) <= stream.draws < len(self.catalog)+14,
                    'Source must have complete unique coverage and only declared whole-update tail')
            self._strict_closed_source_sha256 = canonical(state)
            return

        require(state.get('schema') == 1 and state.get('kind') == KIND
                and state.get('identity') == self.identity
                and state.get('technical_extension') == self.technical_extension
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
                == self.technical_extension['source_uniform_draws']+14*n
                and sum(v['draws'] for v in state['replay'].values())
                == self.technical_extension['source_replay_draws']+2*n,
                'Technical14TR/2replay draw clocks changed')

        # Validate unchanged queues/RNG/seen/audio and piecewise counts through
        # the frozen real parent. The copied bound is only a validation view;
        # never replace the durable original quota or parent source identity.
        quota = deepcopy(self.quota_transition)
        quota['configured_end_step'] = self.configured_end_step
        quota['remaining_update_budget'] += TECHNICAL_UPDATES
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

    def begin_technical_extension(self, context):
        require(self.technical_extension is None and self._strict_closed_source_sha256 is not None,
                'Actual frozen own-source loader must succeed first; one technical extension only')
        require(context.get('source_step') == SOURCE_END_STEP == self.updates
                and context.get('technical_extension_updates') == TECHNICAL_UPDATES
                and context['parent_identity']['domain_sampling_identity'] == self.identity,
                'Only explicitly bound same-source100 technical updates admitted')
        require(self.original_full_pass_end_step == SOURCE_END_STEP,
                'Original completed fullpass end must stay25109')
        stream = self.turkish_streams['uniform']
        self.original_parent_identity = deepcopy(self.identity)
        self.technical_extension = {
            'kind': 'declared_same_future_batches_100_update_IO_control_v1',
            'source_step': SOURCE_END_STEP, 'updates': TECHNICAL_UPDATES,
            'original_full_pass_end_step': SOURCE_END_STEP,
            'configured_technical_end_step': SOURCE_END_STEP+TECHNICAL_UPDATES,
            'source_sampler_sha256': self._strict_closed_source_sha256,
            'source_identity_sha256': canonical(self.original_parent_identity),
            'parent_sampler_source_sha256': PARENT_SOURCE_SHA256,
            'source_uniform_draws': stream.draws,
            'source_uniform_unique_clips': len(stream.seen),
            'source_replay_draws': sum(v.draws for v in self.replay.values()),
            'source_checkpoint': deepcopy(context.get('source_checkpoint')),
            'TR_per_update': 14, 'replay_per_update': 2,
            'PT_per_update': 1, 'old_five_per_update': 1,
            'queue_shuffle_rng_or_seen_reset': False,
            'additional_unique_first_pass_claimed': False,
        }
        self.identity = deepcopy(self.original_parent_identity)
        self.identity.update(sampler_family=KIND,
                             technical_execution_extension=deepcopy(self.technical_extension))
        return deepcopy(self.technical_extension)

    def next_batches(self):
        require(self.technical_extension is not None,
                'No technical draw before explicitly approved extension')
        return super().next_batches()

    def _segments(self, updates):
        if self.technical_extension is None:
            return super()._segments(updates)
        require(type(updates) is int and self.transition_source_step <= updates <= self.configured_end_step,
                'Technical step outside declared100 extension')
        segments = TSCUniformMajoritySampler._segments(self, min(updates, SOURCE_END_STEP))
        if updates > SOURCE_END_STEP:
            n = updates-SOURCE_END_STEP
            segments.append({'first_step': SOURCE_END_STEP+1, 'last_step': updates,
                             'updates': n, 'TR_per_update': 14, 'replay_per_update': 2,
                             'TR_draws': 14*n, 'replay_draws': 2*n,
                             'technical_IO_extension': True})
        return segments

    def state_dict(self):
        require(self.technical_extension is not None, 'No unsourced technical serialization')
        value = super().state_dict()
        value.update(kind=KIND, technical_extension=deepcopy(self.technical_extension))
        return value

    def coverage(self):
        value = super().coverage()
        if self.technical_extension is not None:
            n = self.updates-SOURCE_END_STEP
            value['technical_IO_extension'] = {
                **deepcopy(self.technical_extension), 'completed_updates': n,
                'additional_TR_visits': 14*n, 'additional_replay_visits': 2*n,
                'new_unique_first_pass_clips': 0,
            }
        return value
