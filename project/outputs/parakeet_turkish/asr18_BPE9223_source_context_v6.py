"""Source-bound9223 label overlay, original audio/order/replay, bounded readers.

No model, numerical package, audio decode or Popen is imported here. New label
proofs extend the sealed audio/heldout foundation; old target means are never
projected onto the new vocabulary.
"""
from collections import OrderedDict
from copy import deepcopy
import hashlib,json,sqlite3,struct
from pathlib import Path
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import CarriedReplay,CompletePassCursor,digest
from bpe_training_schedule_v1 import TrainingSchedule,WEAK,CLEANER,COMPLETE
from bpe_replay_bucket_no_runtime_v1 import BucketCycler
from bpe_original_schedule_snapshot_v1 import load_original_snapshot
from bpe_replay_cache_labels_v1 import full_replay_labels
from bpe_declared_geometry_index_v1 import DeclaredGeometryIndex,open_final,signature
from bpe_windows_final_metadata_lock_v1 import FinalMetadataReadLocks
from bpe_final_DATA_window_planner_v4 import feature_upper_bound
from fastmss_phase_aware_plan_bridge_v3 import validate_DATA,actual0,read_bound
from asr18_BPE9223_operational_metadata_v2 import read_bound

ROOT=Path(r'E:\gpu_training_scratchpad')
N=ROOT/'analysis/BPE9223_training_launch_v1'
PROJECT=Path(__file__).resolve().parents[2]
BROAD_N,BROAD_TOKENS=2631115,90746310
CLEAN_N,CLEAN_TOKENS=352768,10621773
KIND='ASR18_validated_9223_training_population_v1'


def exact(pin,guard=lambda:None,*,metadata=False):
    require(type(pin) is dict and set(pin) in ({'path','bytes','sha256'},{'path','bytes','sha256','mtime_ns'}),'Exact record3/4 required')
    p=Path(pin['path']);guard()
    require(record(p)=={k:pin[k] for k in ('path','bytes','sha256')}
        and ('mtime_ns' not in pin or p.stat().st_mtime_ns==pin['mtime_ns']),'Pinned9223 artifact differs')
    if metadata:
        require(0<p.stat().st_size<=16*1024**2,'Bounded metadata only')
        return json.loads(p.read_text(encoding='utf-8'))
    return {k:pin[k] for k in ('path','bytes','sha256')}


def closed(proof_pin,report_pin,guard,*,kind):
    value=exact(proof_pin,guard,metadata=True);report=exact(report_pin,guard,metadata=True)
    require(value['kind']==kind and value['owner']=='/root/label_bpe_audit'
        and type(value['actual_exit_code']) is int and value['actual_exit_code']==0
        and type(value['actual_tool_chunk']) is str and bool(value['actual_tool_chunk'])
        and value['actual_status']['passed'] is True and value['actual_status']['CPU_only'] is True
        and value['actual_status']['GPU_used'] is False,'Actual source-bound label producer/verifier0')
    require(value['all_captured_identities_dead'] and all(x['same_identity_alive'] is False for x in value['all_captured_identities_dead']),
        'Captured exact label identities actually closed')
    import psutil
    for x in value['all_captured_identities_dead']:
        try:require(psutil.Process(x['PID']).create_time()!=x['create_time'],'Exact label producer remains alive')
        except psutil.NoSuchProcess:pass
    link='report' if 'report' in value else 'producer_population'
    require(value[link]==report_pin and report['PID']==value['actual_status']['PID']
        and report['create_time']==value['actual_status']['create_time'],'Same actual producer/report identity')
    return report


def validate_CPU_population(evidence,guard=lambda:None):
    if evidence.get('kind')==KIND:evidence=evidence['population_evidence']
    require(all(k in evidence for k in ('original_DATA_evidence','BPE9223_population','BPE9223_actual0','BPE9223_report')),
        'Complete old audio foundation plus actual NEW label population')
    old=validate_DATA(evidence['original_DATA_evidence'],guard)
    runtime=exact(evidence['BPE9223_population'],guard,metadata=True)
    report=closed(evidence['BPE9223_actual0'],evidence['BPE9223_report'],guard,
        kind='Actual_independent_BPE9223_label_DB_verifier_CLOSED0_v1')
    population=closed(runtime['actual_producer_closure'],runtime['complete_label_population'],guard,
        kind='Actual_BPE9223_label_population_producer_CLOSED0_v1')
    require(runtime['kind']=='BPE9223_complete_training_label_population_runtime_bindings_v1'
        and runtime['complete'] is True and runtime['vocabulary']==9223 and runtime['blank_id']==8192
        and runtime['joint_width']==9228 and runtime['duration_rows_half_open']==[9223,9228]
        and runtime['independent_full_DB_verification']==evidence['BPE9223_report']
        and runtime['databases']==report['databases']==population['databases'], 'Exact new runtime/report/database set')
    require(report['passed'] is True and report['broad_rows']==BROAD_N and report['broad_tokens']==BROAD_TOKENS
        and report['cleaner_rows']==CLEAN_N and report['cleaner_tokens']==CLEAN_TOKENS
        and runtime['broad_mean_U']==report['broad_mean_U']==BROAD_TOKENS/BROAD_N
        and runtime['cleaner_mean_U']==report['cleaner_mean_U']==CLEAN_TOKENS/CLEAN_N
        and population['broad']['tokens']==BROAD_TOKENS and population['cleaner']['tokens']==CLEAN_TOKENS,
        'Both COMPLETE new target denominators, no partial or old means')
    require(all(report[k] is True for k in ('all_2983883_stored_BLOB_labels_match_prior_actual_Rust_digest',
        'broad_sourceID_ordinal_textSHA_matches_old_exactly','cleaner_original8199_source_metadata_preserved_all_rows',
        'all_non_target_broad_geometry_fields_preserved_all_rows','row_counts_ordinals_blob_widths_token_ranges_hashes_and_means_complete',
        'protected1914_replay_teacher_DEV_TEST_audio_untouched','no_partial_mean_armed')),'Full independent exact label/source verification')
    require(runtime['tokenizer']['sha256']=='f2d80ffee7ed1b006a83b52cc77bfb28bed4ba850352b5222bb3a49d6cfbac4a'
        and {k:runtime['source_catalog_unchanged'][k] for k in ('path','bytes','sha256')}==old['bindings']['final_catalog'],
        'Approved9223 tokenizer and unchanged full audio/source catalog')
    for pin in [runtime['source_catalog_unchanged'],runtime['cleaner_source_catalog_unchanged'],*runtime['databases'].values()]:
        require(signature(pin['path'])=={k:pin[k] for k in ('path','bytes','mtime_ns')},'Current exact immutable source/label signature')
    geometry=exact(runtime['declared_geometry'],guard,metadata=True)
    require(geometry['complete'] is True and geometry['geometry_index']==runtime['databases']['declared_geometry']
        and geometry['broad_labels']==runtime['databases']['broad_labels']
        and geometry['all_non_target_geometry_fields_exactly_unchanged'] is True
        and geometry['population']['retained_rows']==BROAD_N and geometry['population']['total_target_tokens']==BROAD_TOKENS
        and report['broad_MAX_T']==1991 and report['broad_MAX_U']==517 and report['broad_MAX_T_times_U_plus_1']==1031338,
        'New exact target geometry joins, no oldU inference')
    bindings=deepcopy(old['bindings']);bindings.update(final_labels={k:runtime['databases']['broad_labels'][k] for k in ('path','bytes','sha256')},
        declared_geometry_index=runtime['databases']['declared_geometry'],cleaner_labels=runtime['databases']['cleaner_labels'])
    declared={'passed':True,'TRAIN_admitted':False,'native_allocation_or_encoder_fit_approved':False,
        'scalar_audit':{'passed':True,'population':geometry['population'],
            'source_signatures':[signature(bindings[k]['path']) for k in ('final_catalog','final_labels','declared_geometry_index')]},
        'new_geometry_binding':runtime['declared_geometry']}
    return {'kind':KIND,'passed':True,'complete':True,'vocabulary':9223,'blank_id':8192,
        'retained_rows':BROAD_N,'label_rows':BROAD_N,'frontier':BROAD_N,'total_target_tokens':BROAD_TOKENS,
        'population_mean_U':BROAD_TOKENS/BROAD_N,'cleaner_rows':CLEAN_N,'cleaner_target_tokens':CLEAN_TOKENS,
        'cleaner_mean_U':CLEAN_TOKENS/CLEAN_N,'bindings':bindings,'declared_geometry':declared,
        'runtime_bindings':deepcopy(runtime),'new_population':evidence['BPE9223_population'],
        'population_evidence':deepcopy(evidence),'heldout_reference_PCM_binding_complete':True,'production_enabled':False,
        'TRAIN_admitted':False,'native_envelope_or_forecast_approved':False,'actual_optimizer_updates':0}


class FinalBroadRows9223:
    def __init__(self,validated,*,guard=lambda:None):
        require(validated['kind']==KIND and validated['passed'] and validated['complete'],'Actual complete9223 population')
        self.guard=guard;self.cache=OrderedDict();self.poisoned=False;self.connections=[];self.population=BROAD_N
        b=validated['bindings'];self.signatures=[]
        self.metadata_locks=FinalMetadataReadLocks([b[k]['path'] for k in ('final_catalog','final_labels','declared_geometry_index')])
        try:
            for key in ('final_catalog','final_labels'):
                exact(b[key],guard);self.signatures.append(signature(b[key]['path']));self.connections.append(open_final(b[key]['path']))
            self.geometry=DeclaredGeometryIndex(b['declared_geometry_index'],guard=guard)
        except BaseException:self.close();raise
    def row(self,ordinal):
        require(not self.poisoned and type(ordinal) is int and 1<=ordinal<=self.population,'Exact full source ordinal')
        try:
            self.guard();require(all(signature(p['path'])==p for p in self.signatures),'Source DB mutation before draw')
            if ordinal in self.cache:self.cache.move_to_end(ordinal);return deepcopy(self.cache[ordinal])
            raw=self.connections[0].execute('SELECT * FROM clips WHERE ordinal=?',(ordinal,)).fetchone()
            label=self.connections[1].execute('SELECT target_ids_u32,target_tokens,text_sha256 FROM labels WHERE ordinal=?',(ordinal,)).fetchone()
            require(raw is not None and label is not None,'No missing retained row or target')
            raw=dict(raw);blob,u,text_sha=label
            require(hashlib.sha256(raw['text'].encode()).hexdigest()==text_sha and type(u) is int and u>0
                and type(blob) is bytes and len(blob)==4*u,'Exact fulltext/newtargetBLOB identity')
            ids=[x[0] for x in struct.iter_unpack('<I',blob)]
            require(all(0<=x<9223 and x!=8192 for x in ids),'Full9223 target IDs')
            d=self.geometry.declaration(raw,full_target_tokens=u)
            from asr18_physical_feature_geometry_v2 import from_samples
            row={**raw,'broad_original':deepcopy(raw),'recording_source_id':raw['source_id'],'source_id':raw['unique_id'],
                'language':'tr','augmentation_role':'student_tr','encoder_frames':d['encoder_frames'],'target_tokens':u,
                'tr_metadata_labels':ids,'training_input_type':'TR_frame_target_metadata_v1','declared_geometry':d,
                'runtime_full_audio_geometry_and_heldout_check_required':True,'declared_frontend_geometry':from_samples(d['clean16k_samples'])}
            self.cache[ordinal]=deepcopy(row)
            while len(self.cache)>32:self.cache.popitem(last=False)
            return row
        except BaseException:self.poisoned=True;raise
    def close(self):
        for c in self.connections:c.close()
        if hasattr(self,'geometry'):self.geometry.close()
        if hasattr(self,'metadata_locks'):self.metadata_locks.close()


class CleanerRows9223:
    def __init__(self,population_evidence,*,guard=lambda:None):
        from expanded_training_rows_v1 import reopen_approved_expanded_TR_rows
        self.guard=guard;self.validated=validate_CPU_population(population_evidence,guard);runtime=self.validated['runtime_bindings']
        self.source=reopen_approved_expanded_TR_rows(PROJECT/'work/data/TR_expanded_WorldSpeech_index_v1',cache_size=256)
        self.database=runtime['databases']['cleaner_labels'];exact(self.database,guard);self.signature=signature(self.database['path'])
        self.connection=open_final(self.database['path']);self.cache=OrderedDict();self.population=CLEAN_N
        total=self.connection.execute('SELECT COUNT(*),SUM(new_u) FROM labels').fetchone()
        require(tuple(total)==(CLEAN_N,CLEAN_TOKENS) and len(self.source)==CLEAN_N,'Complete fixed cleaner/new denominator')
        self.total_target_tokens=CLEAN_TOKENS;self.mean_U=CLEAN_TOKENS/CLEAN_N
        self.identity={'kind':'ASR18_cleaner352768_BPE9223_catalog_binding_v1','source_catalog':runtime['original_cleaner_index_provenance'],
            'BPE_database':self.database,'population_binding':self.validated['new_population'],'rows':CLEAN_N,
            'full_target_tokens':CLEAN_TOKENS,'population_mean_U':self.mean_U,'hours':self.source.receipt['hours'],
            'source_ids_sha256':self.source.ids_sha,'quality_or_language_selection_added':False,'augmentation_phase':CLEANER}
    def row(self,ordinal):
        require(type(ordinal) is int and 1<=ordinal<=CLEAN_N,'One-based original cleaner ordinal')
        self.guard();require(signature(self.database['path'])==self.signature,'Cleaner labels changed')
        source=self.source[ordinal-1]
        value=self.connection.execute('SELECT text_sha,frames,old_u,old_labels_sha,new_u,new_labels_sha,label_ids FROM labels WHERE ordinal=? AND source_id=?',
            (ordinal-1,source['source_id'])).fetchone();require(value is not None,'Exact original cleaner source ID')
        text_sha,t,old_u,old_sha,u,new_sha,blob=value
        from bpe_training_label_catalog_v1 import labels_sha
        require(text_sha==hashlib.sha256(source['text'].encode()).hexdigest() and t==source['encoder_frames']
            and old_u==source['target_tokens'] and old_sha==labels_sha(source['tr_metadata_labels'])
            and type(u) is int and u>0 and len(blob)==2*u,'Original8199 cleaner fields and complete newBLOB')
        labels=list(struct.unpack('<'+'H'*u,blob));require(labels_sha(labels)==new_sha and all(0<=x<9223 and x!=8192 for x in labels),'Exact9223 cleaner targets')
        return {**source,'original_target_tokens':old_u,'sampling_target_tokens':old_u,'target_tokens':u,'tr_metadata_labels':labels,
            'source_tr_frame_metadata_sha256':source.get('tr_frame_metadata_sha256'),'tr_frame_metadata_sha256':self.database['sha256'],
            'BPE_target_catalog_kind':'ASR18_BPE9223_same_ID_target_overlay_v1','BPE_target_catalog_sha256':self.database['sha256'],
            'cleaner_original_ordinal':ordinal,'augmentation_role':'student_tr'}
    def close(self):self.connection.close();self.source.close()


class TrainingSchedule9223(TrainingSchedule):
    def __init__(self,*,broad_identity,cleaner_identity,source_turkish_rng,replay):
        self.broad=CompletePassCursor(population=BROAD_N,identity=broad_identity,seed=digest(source_turkish_rng),passes=2)
        self.cleaner=CompletePassCursor(population=CLEAN_N,identity=cleaner_identity,seed=digest([source_turkish_rng,CLEANER]),passes=1)
        self.means={WEAK:BROAD_TOKENS/BROAD_N,CLEANER:CLEAN_TOKENS/CLEAN_N};self.replay=replay;self.poisoned=False


class FullRunSourceContext9223:
    def __init__(self,arguments,*,guard):
        self.arguments=deepcopy(arguments);self.guard=guard;self.closed=False
        self.validated=validate_CPU_population(arguments['DATA_evidence'],guard)
        actual0(arguments['snapshot_actual0'],arguments['snapshot_report'],guard,owner='/root/memory_diagnosis')
        report=read_bound(arguments['snapshot_report'],guard)
        require(report['passed'] and report['snapshot']==arguments['snapshot_binding'] and report['all_original_RNG_queues_counts_exact']
            and report['actual_optimizer_updates']==report['draws_consumed']==0,'Untouched original scientific snapshot')
        self.snapshot=load_original_snapshot(arguments['snapshot_binding'],expected_sources=arguments['expected_original_sources'],guard=guard)
        pool=exact(arguments['replay_pool_binding'],guard,metadata=True);self.pool=pool['replay'];require(len(self.pool)==1914,'Full original replay pool')
        require(arguments['replay_pool_binding']['sha256']=='00b839c717762aba74326e9867969d85b7ffaaabd7ead7ddbe732fe6be8c6cf4','Exact original replay lineage')
        self.cleaner_rows=CleanerRows9223(self.validated,guard=guard);self.cleaner={'identity':self.cleaner_rows.identity}
        self.identity={'kind':'ASR18_final_unfiltered_BPE9223_broad_identity_v1','population':self.validated['new_population'],
            **{k:self.validated['bindings'][k] for k in ('final_catalog','final_labels','declared_geometry_index','completed_source_inventory')}}
        self.resolver=FinalBroadRows9223(self.validated,guard=guard);self.replay_evidence={}
    def new_schedule(self):
        require(not self.closed,'Open9223 context')
        return TrainingSchedule9223(broad_identity=self.identity,cleaner_identity=self.cleaner_rows.identity,
            source_turkish_rng=self.snapshot['turkish_rng'],replay=CarriedReplay(self.pool,self.snapshot,BucketCycler))
    def resolve_identity(self,phase,ordinal):
        row=self.resolver.row(ordinal) if phase==WEAK else self.cleaner_rows.row(ordinal)
        require(phase in (WEAK,CLEANER) and not self.closed,'Explicit original curriculum phase')
        return {'ordinal':ordinal,'source_id':row['source_id'],'language':'tr'}
    def resolve_rows(self,phase,draws,replay):
        require(not self.closed and phase in (WEAK,CLEANER),'Explicit full phase')
        resolver=self.resolver if phase==WEAK else self.cleaner_rows
        rows=[resolver.row(d['ordinal']) for d in draws]
        for original in replay:
            self.guard();cache=record(original['cache_path']);labels,proof=full_replay_labels(original,cache,guard=self.guard)
            key=original['source_id'];require(key not in self.replay_evidence or self.replay_evidence[key]==proof,'Exact original replay cached targets')
            self.replay_evidence[key]=proof;rows.append({**original,'augmentation_role':'replay','clean_target_ids':labels,'full_replay_label_binding':proof})
        return rows
    def feature_upper_bound(self,phase,rows):
        if phase==WEAK:return feature_upper_bound(phase,rows)
        require(phase==CLEANER and 3<=len(rows)<=16,'Full cleaner tail plus unchanged carried pair')
        return sum((8*r['encoder_frames']+1)*(128*4+1)+8192 for r in rows)
    def plan(self,schedule,*,completed_boundaries):
        from asr18_BPE9223_lifecycle_v3 import make_pass_bounded_plan
        before=deepcopy(schedule.state_dict())
        p=make_pass_bounded_plan(schedule,completed_boundaries=completed_boundaries,clone_schedule=self.new_schedule,
            resolve_identity=self.resolve_identity,resolve_rows=self.resolve_rows,feature_bytes_upper_bound=self.feature_upper_bound,
            finalized_population=self.validated['new_population'],verify_finalized_population=lambda p:p==self.validated['new_population'])
        require(schedule.state_dict()==before,'No planning frontier/RNG mutation');return p
    def validation_plan(self,schedule,*,completed_boundaries,maximum_updates):
        from asr18_BPE9223_lifecycle_v3 import next_boundary_request,validate_window_plan
        from bpe_production_window_coordinator_v4 import make_plan
        before=deepcopy(schedule.state_dict());request=next_boundary_request(before,completed_boundaries)
        require(request['kind']=='bounded_training_window' and type(maximum_updates) is int and 1<=maximum_updates<=20,'Bounded validation cannot bypass realpassDEV')
        p=make_plan(schedule,clone_schedule=self.new_schedule,resolve_identity=self.resolve_identity,resolve_rows=self.resolve_rows,
            feature_bytes_upper_bound=self.feature_upper_bound,finalized_population=self.validated['new_population'],
            verify_finalized_population=lambda p:p==self.validated['new_population'],maximum_updates=min(maximum_updates,request['maximum_updates']))
        validate_window_plan(before,request,p);require(schedule.state_dict()==before,'No validation frontier mutation');return p
    def close(self):
        if not self.closed:self.resolver.close();self.cleaner_rows.close();self.closed=True


def full_run_next_draw_packet(schedule,context,*,optimizer_updates,guard):
    from asr18_checkpoint_next_draw_v1 import next_draw_packet
    if schedule.phase!=COMPLETE:return next_draw_packet(schedule,context,optimizer_updates=optimizer_updates,guard=guard)
    require(optimizer_updates==schedule.committed_updates()==401072 and not schedule.poisoned,'Exact full2broad+1cleaner completion')
    return {'kind':'ASR18_full_run_complete_no_further_draw_packet_v1','optimizer_updates':401072,
        'sampler_state_sha256':digest(schedule.state_dict()),'carried_replay_state_sha256':digest(schedule.replay.state_dict()),
        'phase':COMPLETE,'draws_consumed':0,'replay_frontier_advanced':False,'further_native_or_authoritative_draws_allowed':False,
        'TEST_used':False,'production_enabled':False}


def verify_CPU_configuration(proposal_pin,ROOT_review,guard=lambda:None):
    proposal=exact(proposal_pin,guard,metadata=True);review=exact(ROOT_review,guard,metadata=True)
    require(review['accepted'] is True and review['reviewer']=='/root'
        and review['scope']=='BPE9223_recorded_noise_16room_bounded_CPU_window_v1','Exact external ROOT CPU admission')
    reviewed=review['configuration'];template=exact(reviewed,guard,metadata=True)
    if exact(proposal_pin,guard)==exact(reviewed,guard):return reviewed
    require(template['kind']=='ASR18_BPE9223_CPU_window_template_v1'
        and proposal['template_configuration']==reviewed and template['source_context_source']==record(__file__),
        'Only exact reviewed static CPU template may derive future windows')
    contract=exact(proposal['source_contract'],guard,metadata=True)
    planned=exact(contract['bridge'],guard,metadata=True)
    validate_CPU_population(planned['DATA_evidence'],guard)
    from fastmss_BPE9223_feature_protocol_v9 import verify_window_derivative
    require(verify_window_derivative(template,proposal,guard) is True,'Source-bound fullrun window derivative')
    return reviewed


def verify_CPU_prior_boundary(boundary,guard=lambda:None):
    """Independent metadata cross-check in addition to full bridge validation."""
    from bpe_streaming_checkpoint_semantics_v2 import select,decode_scalar_tree,verify_saved_metadata
    from fastmss_owned_process_closure_v2 import verify_resident_idle,verify
    from fastmss_BPE9223_resource_CPU_boundary_v2 import configure_session,verify_resource_boundary
    require(boundary['kind']=='FastMSS_actual_coherent_full_run_CPU_window_boundary_v2','Exact actual coherent prior boundary')
    session=read_bound(boundary['session'],guard);configure_session(session,guard)
    pointer=read_bound(boundary['coherent_checkpoint_pointer'],guard)
    require(session['source_context_source']==record(__file__) and session['label_population_contract']['sha256']
        ==pointer['identity']['caller_identity']['BPE9223_population_sha256']
        and pointer['diagnostic_only']==session['checkpoint_diagnostic_only']
        and pointer['identity']['caller_identity']['run_identity_sha256']==session['full_run_identity_sha256'],
        'Same newpopulation/runidentity/typed qualification, no diagnostic adoption')
    metadata=read_bound(pointer['checkpoint'],guard);state=decode_scalar_tree(select(metadata['tree'],'sampler'))
    updates=pointer['committed_updates']
    verify_saved_metadata(metadata,identity=pointer['identity'],optimizer_updates=updates,sampler_state=state,
        journal_prefix=pointer['journal_prefix'],diagnostic_only=session['checkpoint_diagnostic_only'])
    require(decode_scalar_tree(select(metadata['tree'],'optimizer','kind'))=='BPE9223_broad_full207_three_AdamW_mixed_fresh_clock_v1'
        and type(decode_scalar_tree(select(metadata['tree'],'optimizer','appended512_Adam_clock'))) is int
        and decode_scalar_tree(select(metadata['tree'],'optimizer','appended512_Adam_clock'))==updates
        and digest(state)==boundary['sampler_state_sha256'],'Actual9223 saved fullstate/newrow clock')
    require(sum(key in boundary for key in ('actual_native_closed','resident_native_boundary','actual_native_resource_closure'))==1,'One honest normal/live/resource boundary')
    if 'actual_native_resource_closure' in boundary:
        require(verify_resource_boundary(boundary,pointer=pointer,session=session,guard=guard) is True,'Exact resource recovery authority before CPU draw')
    elif 'resident_native_boundary' in boundary:
        verify_resident_idle(read_bound(boundary['resident_native_boundary'],guard),pointer=pointer,
            proof_binding=boundary['resident_native_boundary'],guard=guard)
    else:
        value=read_bound(boundary['actual_native_closed'],guard)
        verify(value,kind='ASR18_owned_GPU_process_completion_v1',owner='/root/memory_diagnosis',guard=guard)
    return True
