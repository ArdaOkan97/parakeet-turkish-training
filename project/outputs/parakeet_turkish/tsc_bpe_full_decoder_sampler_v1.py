"""Inactive same-future-ID decoder screen; stdlib and frozen indexed sampler only.

The chosen expanded-study sampler is strictly restored first. Its permutations,
old T/U sorting metadata, six replay queues and RNGs are continued unchanged.
Candidate label rewriting happens downstream, never inside the sampling catalog.
The new 2,000-update execution bound is separate from the old study's 65,948 cap.
"""
from __future__ import annotations
from copy import deepcopy
from types import FunctionType

from tsc_expanded_TRAIN_sampler_v1 import TSCExpandedTrainSampler, END_STEP, require, streamed_sha

KIND = "TSC_BPE_full_decoder_same_future_ID_sampler_v1"
UPDATES = 2000


class _ExecutionBoundParent(TSCExpandedTrainSampler):
    def __init__(self, *args, **kwargs):
        self.decoder_execution_end = END_STEP
        super().__init__(*args, **kwargs)

    @property
    def configured_end_step(self):
        return self.decoder_execution_end

    def load_state_dict(self, state):
        if self.decoder_execution_end == END_STEP:
            return super().load_state_dict(state)
        original = TSCExpandedTrainSampler.load_state_dict
        # A private validation view changes only the execution-bound assertion.
        # Frozen module globals and durable old-study identity remain untouched.
        function = FunctionType(original.__code__, {**original.__globals__, "END_STEP": self.decoder_execution_end},
                                original.__name__, original.__defaults__, original.__closure__)
        return function(self, state)


class BPEFullDecoderSampler:
    def __init__(self, *args, **kwargs):
        self.parent = _ExecutionBoundParent(*args, **kwargs)
        self.transition = None
        self.identity = None

    def migrate_closed_state(self, state, parent_identity, context):
        require(self.transition is None, "One genuine source admission per sampler")
        step = context["source_step"]
        require(type(step) is int and 40750 <= step <= END_STEP and state.get("updates") == step
                and context["decoder_updates"] == UPDATES
                and context["configured_end_step"] == step + UPDATES,
                "Chosen coherent completed-study frontier and declared 2000-update screen required")
        original = deepcopy(context["source_sampler_context"])
        original["own_sampling_identity"] = parent_identity["domain_sampling_identity"]
        evidence = self.parent.initialize_from_own_state(state, original,
                                                        original["parent_identity"]["domain_sampling_identity"])
        require(self.parent.identity == parent_identity["domain_sampling_identity"], "Chosen source sampling identity differs")
        source_coverage = self.parent.coverage()["data_expansion_epoch"]
        require(source_coverage["draws"] == context["chosen_source_original_data_epoch_draws"]
                and source_coverage["unique"] == context["chosen_source_original_data_epoch_unique"],
                "Genuine earlier BEST coverage must not inherit the completed later branch exposure")
        self._install(context, parent_identity)
        return {"passed": True, "strict_original_sampler_restore": evidence,
                "source_state_sha256": streamed_sha(state), "source_step": step,
                "TR_reshuffled_or_reseeded": False, "replay_queues_or_RNG_changed": False,
                "new_phase_draws": 0, "sampling_target_lengths_remain_original8199": True}

    def _install(self, context, parent_identity):
        source = context["source_checkpoint"]
        require(set(source) >= {"path", "sha256", "bytes"}, "Immutable chosen full-state source record required")
        self.source_step = context["source_step"]
        self.end_step = self.source_step + UPDATES
        self.transition = {"kind": KIND, "source_step": self.source_step,
                           "source_checkpoint": deepcopy(source), "updates": UPDATES,
                           "configured_end_step": self.end_step, "old_study_execution_end": END_STEP,
                           "source_sampling_identity_sha256": streamed_sha(parent_identity["domain_sampling_identity"]),
                           "candidate_target_lengths_do_not_change_sampling": True,
                           "physical_microbatches": [[8, 0], [6, 2]], "TR_PT_oldfive_per_update": [14, 1, 1]}
        self.identity = {"kind": KIND, "transition": deepcopy(self.transition),
                         "source_sampling_identity": deepcopy(parent_identity["domain_sampling_identity"])}
        self.parent.decoder_execution_end = self.end_step

    def initialize_from_own_state(self, state, context, parent_identity):
        require(self.transition is None, "Own exact admission on a fresh sampler only")
        immutable = context.get("own_sampling_identity")
        require(isinstance(immutable, dict) and state.get("identity") == immutable,
                "Original admitted new-run configuration must bind own exact sampler identity")
        continued = state.get("continued_parent_sampler", {})
        # Install original source identity through the frozen bootstrap; it needs
        # only the saved compact lineage and current queues, not the original .pt.
        original = deepcopy(context["source_sampler_context"])
        original["own_sampling_identity"] = parent_identity["domain_sampling_identity"]
        self.parent.decoder_execution_end = context["source_step"] + UPDATES
        evidence = self.parent.initialize_from_own_state(continued, original,
                                                        original["parent_identity"]["domain_sampling_identity"])
        self._install(context, parent_identity)
        require(self.identity == immutable, "Own exact source/catalog/execution bound changed")
        self.load_state_dict(state)
        return {"passed": True, "own_exact_sampler_restored": evidence,
                "source_checkpoint_reloaded_for_sampling": False, "RNG_or_coverage_reset": False}

    @property
    def updates(self): return self.parent.updates
    @property
    def local_updates(self): return self.updates - self.source_step
    @property
    def configured_end_step(self): return self.end_step
    @property
    def remaining_update_budget(self): return UPDATES
    @property
    def uniform_pass_updates(self): return UPDATES
    @property
    def turkish_epoch(self): return self.parent.turkish_epoch
    @property
    def tr_draws(self): return self.parent.tr_draws
    @property
    def tr_unique(self): return self.parent.tr_unique
    @property
    def replay(self): return self.parent.replay
    @property
    def counts(self): return self.parent.replay_language_counts
    @property
    def replay_language_counts(self): return self.parent.replay_language_counts
    @property
    def source_prefix(self): return self.parent.source_prefix

    def next_batches(self):
        require(self.transition is not None and 0 <= self.local_updates < UPDATES,
                "No draw before source admission or beyond the declared decoder screen")
        return self.parent.next_batches()

    def coverage(self):
        require(self.transition is not None, "Genuine source sampler required")
        result = self.parent.coverage()
        result["sampler_family"] = KIND
        result["decoder_adaptation"] = {"source_step": self.source_step,
            "local_updates": self.local_updates, "TR_visits": 14 * self.local_updates,
            "PT_visits": self.local_updates, "oldfive_visits": self.local_updates,
            "update_budget": UPDATES, "screen_complete": self.local_updates == UPDATES,
            "original_study_coverage_retained_as_lineage": True}
        return result

    def state_dict(self):
        require(self.transition is not None, "No unsourced decoder sampler save")
        return {"kind": KIND, "identity": deepcopy(self.identity), "updates": self.updates,
                "decoder_adaptation_updates": self.local_updates,
                "continued_parent_sampler": self.parent.state_dict()}

    def load_state_dict(self, state):
        require(state.get("kind") == KIND and state.get("identity") == self.identity,
                "Own exact decoder sampler identity mismatch")
        count = state.get("decoder_adaptation_updates")
        require(type(count) is int and 0 <= count <= UPDATES and state.get("updates") == self.source_step + count,
                "Own exact source/local/global sampler clocks differ")
        self.parent.load_state_dict(state["continued_parent_sampler"])
        require(self.updates == state["updates"], "Parent and new phase frontier differ")
