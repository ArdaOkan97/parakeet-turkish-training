"""One explicit effective16 hard-envelope diagnostic from original source0.

The first13 already executed original TRAIN ordinals are followed by the full
retained T1991/U574 row, then the exact original carried replay pair. This is
a named diagnostic order, never the full production permutation or a seed.
"""
from copy import deepcopy
from pathlib import Path
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import CarriedReplay,digest
from bpe_training_schedule_v1 import ScheduledTransaction,WEAK
from bpe_replay_bucket_no_runtime_v1 import BucketCycler
from bpe_final_DATA_schedule_context_v2 import ScheduleContext as OriginalContext,normalize_schedule_state
from bpe_production_window_coordinator_v4 import make_plan
from fastmss_phase_aware_plan_bridge_v6 import read_bound
from fastmss_owned_process_closure_v2 import verify as verify_process

KIND='ASR18_original_source_hard_envelope_effective16_diagnostic_schedule_v1'
CURSOR='ASR18_explicit_fourteen_original_TR_hard_diagnostic_cursor_v1'
CONTEXT='ASR18_original_source_hard_envelope_diagnostic_bridge_context_v1'
ORDINALS=[763141,1198967,2210155,2449137,2600439,962666,2289117,161546,1045199,846082,481277,701563,1691089,1051133]

def normalize_diagnostic_state(value):
    value=deepcopy(value)
    value['continued_full_population_schedule']=normalize_schedule_state(value['continued_full_population_schedule'])
    value['carried_replay']=normalize_schedule_state({'carried_replay':value['carried_replay']})['carried_replay']
    return value

def verified_prior_rows(evidence,guard):
    require(set(evidence)=={'actual_TOOL0','report','first_plan'},'Exact prior closed20 row provenance')
    actual=read_bound(evidence['actual_TOOL0'],guard);report=read_bound(evidence['report'],guard)
    plan=read_bound(evidence['first_plan'],guard)
    require(actual['kind']=='ASR18_fullpopulation_firstwindow20_conv_operator_observed_TOOL_CLOSED0_v1'
            and actual['owner']=='/root/memory_diagnosis' and type(actual['actual_tool_exit_code']) is int
            and actual['actual_tool_exit_code']==0 and actual['actual_tool_completion_observed'] is True
            and actual['actual_tool_chunk_id']=='65f92a' and actual['report']==evidence['report']
            and actual['actual_native_updates']==20 and report['passed'] is True
            and report['all_deferred_recovery_obligations_discharged'] is True
            and len(report['windows'])==1 and report['windows'][0]['coordinator_plan']==evidence['first_plan'],
            'Genuine previously executed full original first20, not newly claimed PCM/weights')
    for key in ('actual_native_process0','actual_recovery_process0'):
        proof=read_bound(actual[key],guard)
        verify_process(proof,kind='ASR18_owned_GPU_process_completion_v1',owner='/root/memory_diagnosis',guard=guard)
    require(plan['initial_committed_updates']==0 and len(plan['updates'])==20
            and [v['ordinal'] for v in plan['updates'][0]['TR_draws'][:13]]==ORDINALS[:13]
            and evidence['first_plan']['sha256']=='6bec4cb6781805def767d0cd4b1e98363845941d6ee96ed4aa18623014e2b36b',
            'Exact13 previously executed original source ordinals; hard row is separately added')
    return plan['updates'][0]['rows'][:13]

class ExplicitHardCursor:
    def __init__(self,identity,seed):self.identity=deepcopy(identity);self.seed=seed;self.population=14;self.passes=1;self.draws=0
    @property
    def remaining(self):return 14-self.draws
    def preview(self,count):
        require(type(count) is int and 0<=count<=self.remaining,'Bounded explicit diagnostic preview')
        return [{'ordinal':ORDINALS[index],'epoch':0,'draw_index':index} for index in range(self.draws,self.draws+count)]
    def commit(self,draws):
        require(draws==self.preview(len(draws)),'Exact ordered diagnostic rows only')
        self.draws+=len(draws)
    def state_dict(self):
        return {'kind':CURSOR,'population':14,'passes':1,'draws':self.draws,'ordinals':list(ORDINALS),
            'identity':deepcopy(self.identity),'seed':self.seed,'original_full_population_permutation_claimed':False}
    def load_state_dict(self,value):
        expected=self.state_dict();expected['draws']=value['draws']
        require(type(value['draws']) is int and value['draws'] in (0,14)
                and digest(expected)==digest(value),'Exact unqualified diagnostic cursor and complete update boundary')
        self.draws=value['draws']

class DiagnosticHardSchedule:
    def __init__(self,context):
        self.context=context;self.poisoned=False;self.original=deepcopy(context.original_state)
        self.broad=ExplicitHardCursor(context.cursor_identity,context.cursor_seed)
        original=context.source.new_schedule();original.load_state_dict(normalize_schedule_state(self.original))
        self.cleaner=original.cleaner;self.replay=CarriedReplay(context.source.pool,context.source.snapshot,BucketCycler)
        self.replay.load_state_dict(normalize_schedule_state(self.original)['carried_replay']);self.means=deepcopy(original.means)
    @property
    def phase(self):return WEAK
    @property
    def active_cursor(self):return self.broad
    def committed_updates(self):
        require(self.broad.draws in (0,14),'One complete effective16 diagnostic update only')
        return self.broad.draws//14
    def state_dict(self):
        require(digest(self.cleaner.state_dict())==digest(self.original['cleaner_cursor']),'No cleaner/source phase advancement')
        return {'kind':KIND,'diagnostic_only':True,'production_source_eligible':False,'production_phase_transition_claimed':False,
            'initial_source_optimizer_updates':0,'phase':WEAK,'continued_full_population_schedule':deepcopy(self.original),
            'broad_cursor':self.broad.state_dict(),'cleaner_cursor':self.cleaner.state_dict(),
            'carried_replay':deepcopy(self.replay.state_dict()),'population_mean_U':deepcopy(self.means)}
    def load_state_dict(self,value):
        value=normalize_diagnostic_state(value)
        immutable=('kind','diagnostic_only','production_source_eligible','production_phase_transition_claimed',
                   'initial_source_optimizer_updates','phase','continued_full_population_schedule','population_mean_U')
        before=self.state_dict()
        require(not self.poisoned and set(value)==set(before)
                and all(digest(value[key])==digest(before[key]) for key in immutable),'Original0 source/scientific ancestry is immutable')
        try:
            self.broad.load_state_dict(value['broad_cursor']);self.cleaner.load_state_dict(value['cleaner_cursor'])
            self.replay.load_state_dict(value['carried_replay'])
            require(self.replay.updates==self.committed_updates(),'One carried pair per actual diagnostic Adam')
        except BaseException:self.poisoned=True;raise
    def propose(self,*,optimizer_updates,resolve_TR):
        require(self.broad.remaining>0,'Terminal hard diagnostic may not plan another draw')
        return ScheduledTransaction(self,optimizer_updates=optimizer_updates,resolve_TR=resolve_TR)

class HardEnvelopeContext:
    def __init__(self,arguments,prior_closed20,*,guard):
        self.arguments=deepcopy(arguments);self.guard=guard;self.closed=False
        prior=verified_prior_rows(prior_closed20,guard)
        self.source=OriginalContext(arguments,guard=guard);self.snapshot=self.source.snapshot
        self.original_state=self.source.new_schedule().state_dict()
        require(self.original_state['carried_replay']['updates']==0 and self.original_state['broad_cursor']['draws']==0
                and self.original_state['cleaner_cursor']['draws']==0,'Original full0 schedule, never diagnostic checkpoint weights')
        self.rows={ordinal:self.source.resolver.row(ordinal) for ordinal in ORDINALS}
        require(all(digest(self.rows[ordinal])==digest(row) for ordinal,row in zip(ORDINALS[:13],prior)),
                'The13 raw source/labels/declared geometry equal prior real first20 preparation')
        hard=self.rows[1051133]
        require(hard['source_id']=='kiraat:src01792-00990' and hard['encoder_frames']==1991
                and hard['target_tokens']==len(hard['tr_metadata_labels'])==574
                and hard['declared_geometry']['clean16k_samples']==2547840,
                'Actual complete retained maximum source/encoder/targets/audio, no crop/cap/filter')
        self.cursor_identity={'full_final_population':deepcopy(self.source.identity),'ordered_original_ordinals':list(ORDINALS),
            'prior_executed_row_provenance':deepcopy(prior_closed20),'hard_original_ordinal':1051133,
            'diagnostic_only':True,'production_source_eligible':False}
        self.cursor_seed=digest([self.snapshot['turkish_rng'],'one_explicit_hard_envelope_diagnostic_v1'])
    def new_schedule(self):return DiagnosticHardSchedule(self)
    def resolve_identity(self,phase,ordinal):
        require(phase==WEAK and ordinal in self.rows,'Explicit original retained diagnostic row only')
        return {'ordinal':ordinal,'source_id':self.rows[ordinal]['source_id'],'language':'tr'}
    def resolve_rows(self,phase,draws,replay):
        require(not self.closed and phase==WEAK,'One honest weak-phase diagnostic, no production phase advancement')
        rows=[deepcopy(self.rows[draw['ordinal']]) for draw in draws]
        for original in replay:
            from bpe_replay_cache_labels_v1 import full_replay_labels
            self.guard();cache=record(original['cache_path']);ids,proof=full_replay_labels(original,cache,guard=self.guard)
            rows.append({**original,'augmentation_role':'replay','clean_target_ids':ids,'full_replay_label_binding':proof})
        return rows
    def feature_upper_bound(self,phase,rows):
        require(phase==WEAK and len(rows)==16,'Exactly14 original TR+2 carried replay, physical1 serial native')
        from bpe_final_DATA_window_planner_v4 import feature_upper_bound
        return feature_upper_bound(phase,rows)
    def plan(self,schedule,*,maximum_updates):
        require(type(maximum_updates) is int and maximum_updates==1 and schedule.committed_updates()==0,'Exactly one new effective16 diagnostic')
        return make_plan(schedule,clone_schedule=self.new_schedule,resolve_identity=self.resolve_identity,
            resolve_rows=self.resolve_rows,feature_bytes_upper_bound=self.feature_upper_bound,
            finalized_population=self.arguments['DATA_evidence']['population'],
            verify_finalized_population=lambda pin:pin==self.arguments['DATA_evidence']['population'],maximum_updates=1)
    def close(self):
        if not self.closed:self.source.close();self.closed=True

def verify_bridge_state(plan,context_pin,*,population,guard):
    """Real metadata plan reconstruction BEFORE audio; no model/payload load."""
    value=read_bound(context_pin,guard)
    require(set(value)=={'kind','planning_arguments','prior_closed20','initial_diagnostic_sampler','schedule_adapter_source'}
            and value['kind']==CONTEXT and value['schedule_adapter_source']==record(__file__),
            'Exact source-bound hard diagnostic bridge context')
    context=HardEnvelopeContext(value['planning_arguments'],value['prior_closed20'],guard=guard)
    try:
        require(context.source.validated['retained_rows']==population==2631115,'Full final unfiltered population, no hard-only denominator')
        schedule=context.new_schedule();initial=schedule.state_dict()
        require(digest(initial)==digest(value['initial_diagnostic_sampler'])==plan['initial_sampler_sha256'],
                'Original source0 RNG/queues/means and diagnostic cursor initial state')
        expected=context.plan(schedule,maximum_updates=1);provided=deepcopy(plan)
        require(provided.pop('hard_envelope_diagnostic_context')==context_pin and digest(expected)==digest(provided)
                and digest(schedule.state_dict())==digest(initial),'Full ordered rows/targets/transaction/replay/finalstate; no metadata draw consumed')
        require(expected['final_private_planning_sampler']['production_source_eligible'] is False
                and expected['final_private_planning_sampler']['carried_replay']['updates']==1,'One diagnostic update, never a production pass/DEV barrier')
        return {'passed':True,'mode':'one_hard_envelope_diagnostic','production_source_eligible':False,
            'maximum_windows':1,'maximum_native_updates':1,'TR_rows':14,'replay_rows':2,
            'population_pass_or_DEV_barrier_claimed':False}
    finally:context.close()

def completed_diagnostic_next_draw_packet(schedule,context,*,optimizer_updates,guard):
    guard();state=schedule.state_dict()
    require(optimizer_updates==schedule.committed_updates()==schedule.replay.updates==1 and schedule.broad.remaining==0,
            'Actual single hard diagnostic committed; no fabricated next production draw')
    return {'kind':'ASR18_completed_hard_effective16_diagnostic_no_further_draw_packet_v1','optimizer_updates':1,
        'sampler_state_sha256':digest(state),'carried_replay_state_sha256':digest(state['carried_replay']),
        'unchanged_original_fullpopulation_ancestor_sha256':digest(context.original_state),'further_native_or_authoritative_draws_allowed':False,
        'production_source_eligible':False,'production_phase_advance_claimed':False}
