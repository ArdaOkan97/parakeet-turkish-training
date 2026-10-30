"""Selector-only PT repair: retain14TR/2replay, choose one old-five plus one PT.

The caller admits a genuine CLOSED all207 frontier. This projection validates
its original majority native state; no queue, cursor, seen map, RNG, quota or
uniform coverage clock is reset. Own exact restores validate the fixed selector
boundary in addition to the unchanged piecewise8/8 then14/2 count invariants.
"""
from __future__ import annotations
from copy import deepcopy

from tsc_uniform_majority_sampler_v1 import TSCUniformMajoritySampler, KIND as PARENT_KIND
from tsc_uniform_full_sampler_v1 import OLD_LANGUAGES, canonical, require

KIND='TSC_uniform_majority_PT_focus_six_replay_sampler_v1'
PARENT_SAMPLER_SHA256='19adad00ed5d2c00ad3c07e9d31c221c8d171e1202b1ea83348176e276d61976'
RUNTIME_FIELDS=('turkish_streams','replay','tr_rng','replay_rng','pt_rng','updates',
    'replay_language_counts','domain_draws','domain_seen','domain_audio_seconds',
    'group_draws','group_seen','group_audio_seconds','source_prefix','quota_transition')


class TSCUniformMajorityPTFocusSampler(TSCUniformMajoritySampler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.PT_focus_transition=None
        self.parent_majority_identity=None

    @property
    def focus_source_step(self):
        require(self.PT_focus_transition is not None,'CLOSED own-majority selector source required')
        return self.PT_focus_transition['source_step']

    @property
    def focus_remaining_update_budget(self):
        return self.configured_end_step-self.focus_source_step

    def _validate_parent(self,state,identity):
        require(state.get('schema')==1 and state.get('kind')==PARENT_KIND
                and state.get('identity')==identity,'Actual own-majority parent kind/identity required')
        prefix=state['source_prefix'];quota=state['quota_transition']
        require(canonical(prefix)==quota.get('source_original_prefix_sha256'),
                'Historical original source prefix must match the immutable quota digest')
        expected=deepcopy(self.identity)
        expected.update(source_boundary_step=prefix['updates'],
            source_domain_identity_sha256=canonical(prefix['source_identity']),
            source_sampler_sha256=prefix['source_sampler_sha256'],sampler_family=PARENT_KIND,
            replay_per_update=2,majority_quota_transition=deepcopy(quota))
        require(expected==identity,'Original catalog/labels/pools/seeds/14-2 quota identity changed')
        validator=object.__new__(TSCUniformMajoritySampler)
        validator.__dict__=self.__dict__.copy()
        validator.identity=deepcopy(identity);validator.source_prefix=deepcopy(prefix)
        validator.quota_transition=deepcopy(quota)
        TSCUniformMajoritySampler.load_state_dict(validator,state)
        return validator

    def migrate_closed_state(self,state,source_identity):
        require(self.PT_focus_transition is None,'One immutable selector boundary only')
        validator=self._validate_parent(state,source_identity)
        require(validator.transition_source_step<validator.updates<validator.configured_end_step
                and validator.turkish_streams['uniform'].draws<len(self.catalog),
                'Advanced unfinished same-pass source required; no completed-pass restart')
        for key in RUNTIME_FIELDS:setattr(self,key,getattr(validator,key))
        self.parent_majority_identity=deepcopy(source_identity)
        self.PT_focus_transition={
            'kind':'declared_majority_one_PT_one_old_five_selector_v1','source_step':self.updates,
            'quota_boundary_step':self.transition_source_step,'uniform_origin_step':self.source_prefix['updates'],
            'configured_end_step':self.configured_end_step,
            'source_sampler_sha256':canonical(state),'source_identity_sha256':canonical(source_identity),
            'parent_sampler_source_sha256':PARENT_SAMPLER_SHA256,
            'prior_Portuguese_draws':self.replay['pt'].draws,
            'prior_old_five_draws':sum(self.replay[name].draws for name in OLD_LANGUAGES),
            'Portuguese_per_update':1,'old_five_per_update':1,'TR_per_update':14,'replay_per_update':2,
            'old_five_selection':'uniform_choice_from_en_es_it_fr_de_using_carried_replay_rng',
            'new_queue_shuffle_or_rng_reset':False,'renewed_Portuguese_grace':False,
            'future_replay_sequence_changes_explicitly':True,
        }
        self.identity=deepcopy(source_identity)
        self.identity.update(sampler_family=KIND,PT_focus_transition=deepcopy(self.PT_focus_transition))
        return {'kind':'actual_majority_PT_focus_selector_projection_v1','passed':True,
            'PT_focus_transition':deepcopy(self.PT_focus_transition),
            'all_original_queues_cursors_seen_RNG_and_quota_preserved':True,
            'physical_microbatch_composition':[[8,0],[6,2]],
            'second_microbatch_replay_order':['one_uniform_old_five','one_Portuguese']}

    def next_batches(self):
        require(self.PT_focus_transition is not None and self.updates<self.configured_end_step,
                'No unsourced or beyond-fullpass PT-focused draws')
        tr=[self.turkish_streams['uniform'].next() for _ in range(14)]
        for row in tr:
            source=row['source_id'];domain=self.source_domains[source];group=self.source_groups[source]
            self.domain_draws[domain]+=1;self.domain_seen[domain].add(source)
            self.domain_audio_seconds[domain]+=float(row['duration'])
            self.group_draws[group]+=1;self.group_seen[group].add(source)
            self.group_audio_seconds[group]+=float(row['duration'])
        old=self.replay_rng.choice(OLD_LANGUAGES)
        replay=[self.replay[old].next(),self.replay['pt'].next()]
        self.replay_language_counts[old]+=1;self.replay_language_counts['pt']+=1
        self.updates+=1
        return [tr[:8],[*tr[8:],*replay]]

    def coverage(self):
        require(self.PT_focus_transition is not None,'CLOSED selector origin required')
        result=super().coverage();n=self.updates-self.focus_source_step
        result['PT_focus_transition']={**deepcopy(self.PT_focus_transition),'local_updates':n,
            'Portuguese_draws_after_transition':n,'old_five_draws_after_transition':n}
        return result

    def state_dict(self):
        require(self.PT_focus_transition is not None,'No unsourced focused serialization')
        value=super().state_dict()
        value.update(kind=KIND,PT_focus_transition=deepcopy(self.PT_focus_transition))
        return value

    def load_state_dict(self,state):
        require(self.PT_focus_transition is not None and state.get('schema')==1 and state.get('kind')==KIND
                and state.get('identity')==self.identity and state.get('PT_focus_transition')==self.PT_focus_transition,
                'Own PT-focused identity and immutable selector boundary must remain exact')
        step=state.get('updates')
        require(type(step) is int and self.focus_source_step<=step<=self.configured_end_step,
                'PT-focused completed frontier outside same fullpass')
        n=step-self.focus_source_step
        require(state['replay']['pt']['draws']==self.PT_focus_transition['prior_Portuguese_draws']+n
                and sum(state['replay'][name]['draws'] for name in OLD_LANGUAGES)==self.PT_focus_transition['prior_old_five_draws']+n,
                'One PT plus one old-five draw per completed new update required')
        # The parent checks unchanged TR/quota/native queue/RNG invariants on a
        # validation view. Selector-specific kind/identity/counters were checked
        # above; neither the input nor durable own-kind state is rewritten.
        view={**state,'kind':PARENT_KIND,'identity':self.parent_majority_identity}
        validator=object.__new__(TSCUniformMajoritySampler)
        validator.__dict__=self.__dict__.copy();validator.identity=deepcopy(self.parent_majority_identity)
        TSCUniformMajoritySampler.load_state_dict(validator,view)
        for key in RUNTIME_FIELDS:setattr(self,key,getattr(validator,key))
