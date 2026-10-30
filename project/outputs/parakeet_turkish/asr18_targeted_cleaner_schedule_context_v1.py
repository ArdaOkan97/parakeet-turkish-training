"""One honest cleaner diagnostic after actual22; no production phase jump.

The entire actual broad schedule is immutable ancestry. A separately named
two-row diagnostic cleaner cursor and the exact carried replay advance only
after one real23 Adam commit. This state is not a production schedule seed.
"""
from copy import deepcopy
from pathlib import Path
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import CarriedReplay,digest
from bpe_training_schedule_v1 import ScheduledTransaction,CLEANER
from bpe_replay_bucket_no_runtime_v1 import BucketCycler
from bpe_cleaner_data_v1 import CleanerRows
from bpe_replay_cache_labels_v1 import full_replay_labels
from bpe_production_window_coordinator_v4 import make_plan
from bpe_final_DATA_schedule_context_v2 import ScheduleContext as SourceContext,actual_saved_schedule,normalize_schedule_state
from asr18_cleaner_fixture_cursor_v1 import DiagnosticCleanerCursor
from asr18_verified_pipeline_ancestor_v2 import verify_ancestor

KIND='ASR18_targeted_cleaner_after_actual22_diagnostic_schedule_v1'


def normalize_diagnostic_state(value):
    value=deepcopy(value)
    value['continued_full_population_schedule']=normalize_schedule_state(value['continued_full_population_schedule'])
    replay=normalize_schedule_state({'carried_replay':value['carried_replay']})['carried_replay']
    value['carried_replay']=replay
    return value


class DiagnosticCleanerSchedule:
    def __init__(self,context):
        self.context=context;self.poisoned=False
        self.original=deepcopy(context.original_state)
        original=context.source.new_schedule();original.load_state_dict(normalize_schedule_state(self.original))
        self.broad=original.broad
        self.cleaner=DiagnosticCleanerCursor(ordinals=[1,352768],identity=context.cursor_identity,seed=context.cursor_seed)
        self.replay=CarriedReplay(context.source.pool,context.source.snapshot,BucketCycler)
        self.replay.load_state_dict(normalize_schedule_state(self.original)['carried_replay'])
        self.means=deepcopy(original.means)

    @property
    def phase(self):return CLEANER

    @property
    def active_cursor(self):return self.cleaner

    def committed_updates(self):
        require(self.cleaner.draws in (0,2),'One diagnostic two-row whole-tail boundary only')
        return 22+(1 if self.cleaner.draws==2 else 0)

    def state_dict(self):
        require(digest(self.broad.state_dict())==digest(self.original['broad_cursor']),
                'Diagnostic cleaner code cannot advance the original broad source cursor')
        return {'kind':KIND,'diagnostic_only':True,'production_phase_transition_claimed':False,
            'initial_source_optimizer_updates':22,'phase':CLEANER,
            'continued_full_population_schedule':deepcopy(self.original),'broad_cursor':self.broad.state_dict(),
            'diagnostic_cleaner_cursor':self.cleaner.state_dict(),'carried_replay':deepcopy(self.replay.state_dict()),
            'population_mean_U':deepcopy(self.means)}

    def load_state_dict(self,value):
        value=normalize_diagnostic_state(value)
        require(not self.poisoned and value['kind']==KIND and value['diagnostic_only'] is True
                and value['production_phase_transition_claimed'] is False and value['initial_source_optimizer_updates']==22
                and value['phase']==CLEANER and value['population_mean_U']==self.means
                and digest(value['continued_full_population_schedule'])==digest(self.original)
                and digest(value['broad_cursor'])==digest(self.original['broad_cursor']),
                'Actual22 full broad schedule remains unchanged; no completed broad passes invented')
        try:
            self.cleaner.load_state_dict(value['diagnostic_cleaner_cursor']);self.replay.load_state_dict(value['carried_replay'])
            require(self.replay.updates==self.committed_updates(),'Actual23 diagnostic/replay/Adam frontier required')
        except BaseException:self.poisoned=True;raise

    def propose(self,*,optimizer_updates,resolve_TR):
        require(self.cleaner.remaining>0,'Completed one-update diagnostic cannot plan additional cleaner draws')
        return ScheduledTransaction(self,optimizer_updates=optimizer_updates,resolve_TR=resolve_TR)


class CleanerContext:
    def __init__(self,arguments,ancestor,*,guard):
        self.guard=guard;self.closed=False;self.arguments=deepcopy(arguments);self.ancestor=deepcopy(ancestor)
        verify_ancestor(ancestor,guard=guard)
        self.pointer,self.original_state=actual_saved_schedule(ancestor['checkpoint_pointer'],guard=guard)
        self.source=SourceContext(arguments,guard=guard);self.snapshot=self.source.snapshot
        source_schedule=self.source.new_schedule();source_schedule.load_state_dict(normalize_schedule_state(self.original_state))
        require(source_schedule.committed_updates()==source_schedule.replay.updates==22
                and self.original_state['cleaner_cursor']['draws']==0,
                'Actual22 original schedule, no completed broad/cleaner phase fabricated')
        self.cleaner=CleanerRows();self.rows={ordinal:self.cleaner.row(ordinal) for ordinal in (1,352768)}
        require(self.cleaner.identity==self.source.cleaner['identity'] and self.cleaner.mean_U==11972886/352768,
                'Same full352768 cleaner source and complete target denominator')
        self.cursor_identity={'cleaner_catalog':deepcopy(self.cleaner.identity),'ordinals':[1,352768],
            'ancestor_checkpoint':ancestor['checkpoint_pointer'],'is_production_cleaner_pass':False}
        self.cursor_seed=digest([self.snapshot['turkish_rng'],CLEANER,'targeted_after_actual22_diagnostic_v1'])

    def new_schedule(self):return DiagnosticCleanerSchedule(self)

    def resolve_identity(self,phase,ordinal):
        require(phase==CLEANER and ordinal in self.rows,'Explicit bounded real cleaner rows only')
        row=self.rows[ordinal];return {'ordinal':ordinal,'source_id':row['source_id'],'language':'tr'}

    def resolve_rows(self,phase,draws,replay):
        require(not self.closed and phase==CLEANER,'No weak augmentation/implicit production phase transition')
        rows=[deepcopy(self.rows[draw['ordinal']]) for draw in draws]
        for original in replay:
            self.guard();cache=record(original['cache_path']);ids,proof=full_replay_labels(original,cache,guard=self.guard)
            rows.append({**original,'augmentation_role':'replay','clean_target_ids':ids,'full_replay_label_binding':proof})
        return rows

    def feature_upper_bound(self,phase,rows):
        require(phase==CLEANER and len(rows)==4,'Exactly2 real cleaner TR +actual carried old-five/PT pair')
        return sum((8*row['encoder_frames']+1)*513+8192 for row in rows)

    def plan(self,schedule,*,maximum_updates):
        require(type(maximum_updates) is int and maximum_updates==1 and schedule.committed_updates()==22,
                'Only one NEW cleaner23 diagnostic update')
        return make_plan(schedule,clone_schedule=self.new_schedule,resolve_identity=self.resolve_identity,
            resolve_rows=self.resolve_rows,feature_bytes_upper_bound=self.feature_upper_bound,
            finalized_population=self.arguments['DATA_evidence']['population'],
            verify_finalized_population=lambda pin:pin==self.arguments['DATA_evidence']['population'],maximum_updates=1)

    def close(self):
        if not self.closed:self.cleaner.close();self.source.close();self.closed=True


def completed_diagnostic_next_draw_packet(schedule,context,*,optimizer_updates,guard):
    """Explicit terminal diagnostic packet; never fabricate a production draw."""
    guard();state=schedule.state_dict()
    require(optimizer_updates==schedule.committed_updates()==schedule.replay.updates==23
            and schedule.cleaner.remaining==0 and state['kind']==KIND
            and digest(state['continued_full_population_schedule'])==digest(context.original_state),
            'One actual cleaner diagnostic finished; original fullpopulation source remains ancestry')
    return {'kind':'ASR18_completed_targeted_cleaner23_diagnostic_no_further_draw_packet_v1',
        'optimizer_updates':23,'sampler_state_sha256':digest(state),
        'carried_replay_state_sha256':digest(state['carried_replay']),
        'unchanged_original_fullpopulation_ancestor_sha256':digest(context.original_state),
        'diagnostic_cleaner_remaining':0,'further_native_or_authoritative_draws_allowed':False,
        'production_phase_advance_claimed':False,'production_source_eligible':False}
