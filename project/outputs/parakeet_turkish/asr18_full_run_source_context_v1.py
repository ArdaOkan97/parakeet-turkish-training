"""Inactive full-population source resolver and pass-bounded owner planner.

Original source RNG, two broad passes and one cleaner pass are authoritative.
This adds no selection, sampler mutation, model construction or launch entrypoint.
"""
from copy import deepcopy
from bpe_broad_audio_v1 import require
from bpe_broad_production_state_v1 import digest
from bpe_training_schedule_v1 import WEAK,CLEANER,COMPLETE
from bpe_cleaner_data_v1 import CleanerRows
from bpe_final_DATA_schedule_context_v2 import ScheduleContext as OriginalContext,normalize_schedule_state
from bpe_final_DATA_window_planner_v4 import feature_upper_bound as broad_feature_upper_bound
from bpe_full_run_lifecycle_v2 import make_pass_bounded_plan,progress,next_boundary_request,validate_window_plan
from bpe_production_window_coordinator_v4 import make_plan
from asr18_checkpoint_next_draw_v1 import next_draw_packet as regular_next_draw_packet
from fastmss_phase_aware_plan_bridge_v7 import read_bound
from bpe_streaming_checkpoint_semantics_v2 import select,decode_scalar_tree,verify_saved_metadata
from bpe_broad_audio_v1 import record


class FullRunSourceContext(OriginalContext):
    def __init__(self,arguments,*,guard):
        super().__init__(arguments,guard=guard)
        self.cleaner_rows=CleanerRows()
        require(self.cleaner_rows.identity==self.cleaner['identity']
                and self.cleaner_rows.mean_U==11972886/352768,
                'Unchanged full352768 cleaner source and population denominator')

    def resolve_identity(self,phase,ordinal):
        require(not self.closed and phase in (WEAK,CLEANER),'Original full source phase')
        if phase==WEAK:return super().resolve_identity(phase,ordinal)
        row=self.cleaner_rows.row(ordinal)
        return {'ordinal':ordinal,'source_id':row['source_id'],'language':'tr'}

    def resolve_rows(self,phase,draws,replay):
        require(not self.closed and phase in (WEAK,CLEANER),'Explicit original production source phase')
        # The frozen broad resolver adds the same complete replay labels/proofs.
        # Request an empty broad prefix to reuse exactly that replay binding path.
        if phase==WEAK:return super().resolve_rows(phase,draws,replay)
        rows=[self.cleaner_rows.row(draw['ordinal']) for draw in draws]
        rows.extend(super().resolve_rows(WEAK,[],replay))
        require(all(row['language']=='tr' for row in rows[:len(draws)])
                and len(rows)==len(draws)+2,'Every actual cleaner tail keeps exact carried pair')
        return rows

    def feature_upper_bound(self,phase,rows):
        require(3<=len(rows)<=16 and phase in (WEAK,CLEANER),'Original complete/tail TR plus replay')
        if phase==WEAK:return broad_feature_upper_bound(phase,rows)
        # CPU8 enforces actual mono16k header-derived T BEFORE raw allocation.
        # N <=1280*T+159, so physical frontend length <=8*T+1.
        return sum((8*row['encoder_frames']+1)*(128*4+1)+8192 for row in rows)

    def plan(self,schedule,*,completed_boundaries):
        before=deepcopy(schedule.state_dict());progress(before)
        plan=make_pass_bounded_plan(schedule,completed_boundaries=completed_boundaries,
            clone_schedule=self.new_schedule,resolve_identity=self.resolve_identity,resolve_rows=self.resolve_rows,
            feature_bytes_upper_bound=self.feature_upper_bound,
            finalized_population=self.arguments['DATA_evidence']['population'],
            verify_finalized_population=lambda pin:pin==self.arguments['DATA_evidence']['population'])
        require(digest(schedule.state_dict())==digest(before),'Planner never commits original/replay RNG')
        return plan

    def validation_plan(self,schedule,*,completed_boundaries,maximum_updates):
        """Explicit off-cycle bounded integration, never pass completion fiction."""
        require(type(maximum_updates) is int and 1<=maximum_updates<=20,'Reviewed finite owner-validation budget')
        before=deepcopy(schedule.state_dict());request=next_boundary_request(before,completed_boundaries)
        require(request['kind']=='bounded_training_window','No off-cycle bypass of an actual pending DEV barrier')
        plan=make_plan(schedule,clone_schedule=self.new_schedule,resolve_identity=self.resolve_identity,
            resolve_rows=self.resolve_rows,feature_bytes_upper_bound=self.feature_upper_bound,
            finalized_population=self.arguments['DATA_evidence']['population'],
            verify_finalized_population=lambda pin:pin==self.arguments['DATA_evidence']['population'],
            maximum_updates=min(maximum_updates,request['maximum_updates']))
        validate_window_plan(before,request,plan)
        require(digest(schedule.state_dict())==digest(before),'Validation preparation consumes no draw')
        return plan

    def close(self):
        if not self.closed:
            self.cleaner_rows.close()
            super().close()


def actual_saved_full_run_schedule(pointer_binding,*,identity,guard):
    pointer=read_bound(pointer_binding,guard)
    require(pointer['kind']=='ASR18_production_window_completed_checkpoint_pointer_v2'
            and pointer['diagnostic_only'] is False and pointer['identity']==identity
            and pointer['all_window_updates_committed'] is True
            and pointer['writer_returned_and_payload_verified'] is True,
            'Actual original full-run coherent checkpoint, never diagnostic promotion')
    metadata=read_bound(pointer['checkpoint'],guard)
    state=decode_scalar_tree(select(metadata['tree'],'sampler'))
    state=normalize_schedule_state(state);p=progress(state)
    verify_saved_metadata(metadata,identity=identity,optimizer_updates=p['optimizer_updates'],
        sampler_state=state,journal_prefix=pointer['journal_prefix'],diagnostic_only=False)
    require(p['optimizer_updates']==pointer['committed_updates'] and digest(state)==pointer['sampler_state_sha256']
            and record(metadata['payload']['path'])==metadata['payload']==pointer['payload'],
            'Actual full-run source state/payload/sampler frontier')
    return pointer,state


def full_run_next_draw_packet(schedule,context,*,optimizer_updates,guard):
    if schedule.phase!=COMPLETE:
        return regular_next_draw_packet(schedule,context,optimizer_updates=optimizer_updates,guard=guard)
    guard();state=schedule.state_dict();p=progress(state)
    require(optimizer_updates==p['optimizer_updates']==schedule.committed_updates()==401072
            and state['carried_replay']['updates']==401072 and not schedule.poisoned,
            'Exactly two full broad passes and one full cleaner pass, no invented terminal draw')
    return {'kind':'ASR18_full_run_complete_no_further_draw_packet_v1','optimizer_updates':401072,
        'sampler_state_sha256':digest(state),'carried_replay_state_sha256':digest(state['carried_replay']),
        'phase':COMPLETE,'draws_consumed':0,'replay_frontier_advanced':False,
        'further_native_or_authoritative_draws_allowed':False,'TEST_used':False,
        'production_enabled':False}
