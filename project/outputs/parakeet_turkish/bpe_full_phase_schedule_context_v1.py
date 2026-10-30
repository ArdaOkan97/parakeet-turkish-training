"""Inactive original-source resolver for two broad passes and one cleaner pass.

No CLI, native executor, resource approval, DEV acknowledgement or production
enablement is supplied here. Every plan is provisional and pass-bounded.
"""
from copy import deepcopy
from bpe_broad_audio_v1 import require
from bpe_broad_production_state_v1 import digest
from bpe_final_DATA_schedule_context_v1 import ScheduleContext
from bpe_final_DATA_window_planner_v4 import feature_upper_bound as weak_feature_upper_bound
from bpe_training_schedule_v1 import WEAK,CLEANER
from bpe_cleaner_data_v1 import CleanerRows
from bpe_replay_cache_labels_v1 import full_replay_labels
from asr18_physical_feature_geometry_v2 import from_encoder_upper_bound
from bpe_full_run_lifecycle_v2 import make_pass_bounded_plan
from fastmss_phase_aware_plan_bridge_v5 import actual0,read_bound


def full_phase_feature_upper_bound(phase,rows):
    """Conditional physical-file bound; actual clean header T must match first."""
    require(phase in (WEAK,CLEANER) and 3<=len(rows)<=16,'One full/tail TR batch plus unchanged replay pair')
    if phase==WEAK:return weak_feature_upper_bound(phase,rows)
    total=0
    for row in rows:
        frames=row['encoder_frames']
        require(type(frames) is int and frames>0,'Original declared encoder T required')
        physical=from_encoder_upper_bound(frames)['physical_feature_frames_upper_bound']
        total+=physical*(128*4+1)+2*4096
    return total


class FullPhaseScheduleContext(ScheduleContext):
    """Keep original cursors/RNG/clocks/labels; add explicit clean source route."""
    def __init__(self,arguments,*,cleaner_resource,cleaner_resource_actual0,guard):
        self.cleaner_rows=None
        super().__init__(arguments,guard=guard)
        try:
            actual0(cleaner_resource_actual0,cleaner_resource,guard,owner='/root/cpu_data_preparation')
            resource=read_bound(cleaner_resource,guard)
            require(resource['kind']=='ASR18_full_cleaner_conditional_declared_T_resource_envelope_v1'
                and resource['complete_scalar_population'] is True and type(resource['rows']) is int
                and resource['rows']==352768 and type(resource['full_target_tokens']) is int
                and resource['full_target_tokens']==11972886
                and resource['population_mean_U']==11972886/352768
                and resource['counts']['nonpositive_T']==0 and resource['counts']['weak_transform_possible']==0,
                'Genuine full cleaner scalar denominator and conditional geometry, never partial mean')
            self.cleaner_rows=CleanerRows()
            require(digest(self.cleaner_rows.identity)==digest(self.cleaner['identity'])
                and self.cleaner_rows.population==352768 and self.cleaner_rows.total_target_tokens==11972886,
                'Exact original full cleaner source/labels, no new selection or rewrite')
            self.cleaner_resource={'receipt':cleaner_resource,'actual0':cleaner_resource_actual0,
                'geometry_is_conditional_on_original_mono16k_header_T_match_before_allocation':True,
                'exact_all352768_N_or_channels_known':False,'native_or_CPU_private_fit_claimed':False,
                'allocation_workspace_policy_approval_supplied':False}
        except BaseException:
            self.close();raise

    def resolve_identity(self,phase,ordinal):
        require(not self.closed,'Closed context cannot resolve rows')
        if phase==WEAK:return super().resolve_identity(phase,ordinal)
        require(phase==CLEANER,'No implicit phase or completed-schedule draw')
        row=self.cleaner_rows.row(ordinal)
        require(row['cleaner_original_ordinal']==ordinal and row['language']=='tr','Original cleaner ordinal/source')
        return {'ordinal':ordinal,'source_id':row['source_id'],'language':'tr'}

    def resolve_rows(self,phase,draws,replay):
        require(not self.closed and phase in (WEAK,CLEANER),'Explicit original phase before preparation')
        if phase==WEAK:return super().resolve_rows(phase,draws,replay)
        rows=[self.cleaner_rows.row(draw['ordinal']) for draw in draws]
        require(1<=len(rows)<=14 and len(replay)==2,'Whole or exact tail TR rows plus original carried replay')
        for row,draw in zip(rows,draws):
            require(type(draw['ordinal']) is int and row['cleaner_original_ordinal']==draw['ordinal']
                and row['augmentation_role']=='student_tr' and row['language']=='tr'
                and type(row['encoder_frames']) is int and row['encoder_frames']>0
                and type(row['target_tokens']) is int and row['target_tokens']==len(row['tr_metadata_labels']),
                'Unmodified full cleaner BPE labels and stored T; no duration/quality cap')
        for original in replay:
            self.guard()
            from bpe_broad_audio_v1 import record
            cache=record(original['cache_path'])
            labels,proof=full_replay_labels(original,cache,guard=self.guard)
            key=original['source_id']
            require(key not in self.replay_evidence or self.replay_evidence[key]==proof,
                'Original replay source/full-target identity changed')
            self.replay_evidence[key]=proof
            rows.append({**original,'augmentation_role':'replay','clean_target_ids':labels,
                'full_replay_label_binding':proof})
        return rows

    def plan_for_next_pass(self,schedule,*,completed_boundaries):
        """DEV acknowledgements come from the separately reviewed lifecycle."""
        require(not self.closed and type(completed_boundaries) is int and 0<=completed_boundaries<3,
            'Explicit acknowledged pass-boundary prefix')
        before=deepcopy(schedule.state_dict())
        plan=make_pass_bounded_plan(schedule,completed_boundaries=completed_boundaries,
            clone_schedule=self.new_schedule,resolve_identity=self.resolve_identity,
            resolve_rows=self.resolve_rows,feature_bytes_upper_bound=full_phase_feature_upper_bound,
            finalized_population=self.arguments['DATA_evidence']['population'],
            verify_finalized_population=lambda pin:pin==self.arguments['DATA_evidence']['population'])
        require(digest(schedule.state_dict())==digest(before)
            and plan['initial_sampler_sha256']==digest(before),'No authoritative cursor/replay/RNG change on preparation')
        return plan

    def close(self):
        if self.cleaner_rows is not None:
            self.cleaner_rows.close();self.cleaner_rows=None
        super().close()
