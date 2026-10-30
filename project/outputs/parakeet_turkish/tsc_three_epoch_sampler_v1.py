"""Continue selected queues unchanged; explicitly extend execution to three passes."""
from copy import deepcopy
from tsc_bpe_full_decoder_sampler_v1 import _ExecutionBoundParent
from tsc_three_epoch_transition_v1 import PHASE, SOURCE_STEP, END, POOL, declaration, require, canonical


class ThreeEpochSampler:
    def __init__(self,*args,**kwargs):
        self.parent=_ExecutionBoundParent(*args,**kwargs)
        self.identity=None

    def _restore(self,state,context):
        original=deepcopy(context["source_sampler_context"])
        domain=context["parent_identity"]["domain_sampling_identity"]
        original["own_sampling_identity"]=domain
        # Only the private execution-bound validation view changes. Old source
        # catalog/lineage identities, every queue and RNG stay exact.
        self.parent.decoder_execution_end=END
        result=self.parent.initialize_from_own_state(state,original,original["parent_identity"]["domain_sampling_identity"])
        self.identity=dict(kind=PHASE,source_domain_identity=deepcopy(domain),
            source_checkpoint=deepcopy(context["source_checkpoint"]),declaration=declaration())
        return result

    def migrate_closed_state(self,state,context):
        require(self.identity is None and state["updates"]==SOURCE_STEP,"Exact selected source sampler required")
        before=canonical(state)
        result=self._restore(state,context)
        require(canonical(state)==before and self.parent.coverage()["data_expansion_epoch"]["draws"]==301000
                and self.parent.coverage()["data_expansion_epoch"]["unique"]==301000,
                "Source coverage or queues mutated/reset at extension")
        return dict(passed=True,original_sampler_restored=result,source_state_sha256=before,
            new_training_draws=0,permutation_RNG_cursor_seen_counters_preserved=True)

    def initialize_from_own_state(self,state,context,identity):
        require(self.identity is None,"Fresh own-state bootstrap required")
        result=self._restore(state["continued_parent_sampler"],context)
        require(canonical(self.identity)==canonical(identity)==canonical(state["identity"]),"Own extension identity changed")
        self.load_state_dict(state)
        return dict(passed=True,own_sampler_restored=result,no_sampler_or_RNG_reset=True)

    @property
    def updates(self):return self.parent.updates

    def next_batches(self):
        require(self.identity is not None and self.updates<END,"Only declared three-pass bound")
        return self.parent.next_batches()

    def coverage(self):
        value=self.parent.coverage()
        draws=value["data_expansion_epoch"]["draws"]
        value["three_epoch_extension"]=dict(source_step=SOURCE_STEP,source_TR_draws=301000,
            new_updates=self.updates-SOURCE_STEP,new_TR_visits=draws-301000,
            accepted_TR_pool=POOL,total_TR_visits=draws,complete_permutation_cycles=draws//POOL,
            current_cycle_visits=draws%POOL,target_cycles=3,declared_end_step=END,
            original_source_prefix_and_all_completed_prior_branches_preserved=True)
        return value

    def state_dict(self):
        return dict(kind=PHASE,identity=deepcopy(self.identity),updates=self.updates,
            continued_parent_sampler=self.parent.state_dict())

    def load_state_dict(self,state):
        require(state["kind"]==PHASE and canonical(state["identity"])==canonical(self.identity)
                and SOURCE_STEP<=state["updates"]<=END,"Own exact sampler extension differs")
        self.parent.load_state_dict(state["continued_parent_sampler"])
        require(self.updates==state["updates"],"Saved parent/extension clocks disagree")
