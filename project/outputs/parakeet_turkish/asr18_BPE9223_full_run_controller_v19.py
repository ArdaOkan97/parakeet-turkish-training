"""Concrete inactive full-run owner body; bounded validation uses its stop budget.

Normal windows remain resident and publish persistence once at window-end.
Actual native exit and CURRENT fresh recovery precede durable prefix ack, DEV,
winner selection, and phase advance. Validation never fabricates a pass barrier.
"""
from copy import deepcopy
from contextlib import contextmanager
import gc,hashlib,inspect,json,os,secrets,shutil,time,struct,zlib
from pathlib import Path
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import canonical,digest
from asr18_BPE9223_lifecycle_v3 import FullRunLifecycle,SourceBoundOwnerCallbacks,CALLBACKS,REFERENCES,LIMITS,progress
from asr18_BPE9223_resource_operational_compatibility_v1 import OperationalOwnerProofCallbacks as OwnerProofCallbacks,checked_compatibility,checked_resource_stop
from asr18_BPE9223_durable_owner_adapter_v10 import PackedDurableLifecycleAdapter
from asr18_BPE9223_indexed_checkpoint_storage_v3 import IndexedOwnedFutureCheckpointStore
from asr18_BPE9223_indexed_checkpoint_storage_v1 import AUX_ROLES
from bpe_bounded_production_journal_v4 import JournalNamespace
from asr18_BPE9223_source_context_v2 import FullRunSourceContext9223 as FullRunSourceContext,full_run_next_draw_packet
from asr18_BPE9223_persistence_prefix_v13 import OwnerPrefixVerifier,create_coverage,_Metadata,_capture,_JournalPrefixes,verify_acknowledged_resource_reentry
from asr18_BPE9223_metadata_resolver_v3 import MetadataResolver
from bpe_final_DATA_schedule_context_v2 import normalize_schedule_state
from bpe_training_schedule_v1 import WEAK
from bpe_streaming_checkpoint_semantics_v2 import select,decode_scalar_tree,verify_saved_metadata
from asr18_BPE9223_operational_metadata_v2 import read_bound,configure as configure_metadata
from fastmss_owned_process_closure_v2 import identity,verify,verify_resident_idle
ROOT=Path(r'E:\gpu_training_scratchpad');OWNER='/root/memory_diagnosis'
WITNESS_ZDICT=b'ASR18 owner native_checkpoint producer sha256 bytes path source configuration ownership state_revision catalog coherent_latest_pointer protected_owned_winners nonce checkpoint_mappings_released candidate_payload_handles_released checkpoint_write_inflight restore_in_progress actual_current_checkpoint_pointer native_release_ack resident_native_boundary current_exact_recovery publication_cadence_evidence incremental_authoritative_journal creation_identity writer_epoch source session_owner_nonce auxiliary_manifest_sha256 all_CPU_auxiliary_producers_actually_closed0 native_auxiliary_writer_contexts_returned all_auxiliary_handles_released E:\\\\gpu_training_scratchpad\\\\analysis\\\\ASR18_memory_diagnosis_v1\\\\ E:\\\\gpu_training_scratchpad\\\\runs\\\\ASR18_full_owner_validation_08_ C:\\\\Users\\\\ardao\\\\Documents\\\\Codex\\\\2026-10-10\\\\i-x20\\\\outputs\\\\parakeet_turkish\\\\ '
WITNESS_ZDICT_V2=b'ASR18 owner native_checkpoint producer sha256 bytes path source configuration ownership state_revision catalog coherent_latest_pointer protected_owned_winners nonce checkpoint_mappings_released candidate_payload_handles_released checkpoint_write_inflight restore_in_progress actual_current_checkpoint_pointer native_release_ack resident_native_boundary current_exact_recovery publication_cadence_evidence incremental_authoritative_journal creation_identity writer_epoch source session_owner_nonce auxiliary_manifest_sha256 all_CPU_auxiliary_producers_actually_closed0 native_auxiliary_writer_contexts_returned all_auxiliary_handles_released E:\\\\gpu_training_scratchpad\\\\analysis\\\\ASR18_memory_diagnosis_v1\\\\ E:\\\\gpu_training_scratchpad\\\\runs\\\\ASR18_full_owner_validation_08_ C:\\\\Users\\\\ardao\\\\Documents\\\\Codex\\\\2026-10-10\\\\i-x20\\\\outputs\\\\parakeet_turkish\\\\  merged_CPU_live_task_boundary merged_parent_process_exit0_claimed metadata_task_completed_no_CPU_inflight every_preserved_control_attempt_closed0_claimed legacy_all_CPU_auxiliary_producers_actually_closed0_scope actual_numerical_reference_frontend_and_external_cleanup_only producer_completion_scopes actual_owner actual_current_checkpoint_pointer active_auxiliary_workspaces committed_window_updates retained_metadata_bytes auxiliary_archives auxiliary_bytes archive_bytes live_payload_bytes live_pairs exact_pass_tail file_index history last_frame accounting checkpoint_mappings_released candidate_payload_handles_released checkpoint_write_inflight restore_in_progress native_release_ack native_work_quiescent resident_native_boundary current_exact_recovery publication_cadence_evidence incremental_authoritative_journal writer_source writer_epoch sole_current_writer prior_bytes_never_rewritten_or_truncated producer_PID producer_create_time session_owner_nonce all_CPU_auxiliary_producers_actually_closed0 native_auxiliary_writer_contexts_returned all_auxiliary_handles_released auxiliary_manifest_sha256 "merged_CPU_live_task_boundary": "merged_parent_process_exit0_claimed": "metadata_task_completed_no_CPU_inflight": "every_preserved_control_attempt_closed0_claimed": "legacy_all_CPU_auxiliary_producers_actually_closed0_scope": "actual_numerical_reference_frontend_and_external_cleanup_only": "producer_completion_scopes": "actual_owner": "actual_current_checkpoint_pointer": "active_auxiliary_workspaces": "committed_window_updates": "retained_metadata_bytes": "auxiliary_archives": "auxiliary_bytes": "archive_bytes": "live_payload_bytes": "live_pairs": "exact_pass_tail": "file_index": "history": "last_frame": "accounting": "checkpoint_mappings_released": "candidate_payload_handles_released": "checkpoint_write_inflight": "restore_in_progress": "native_release_ack": "native_work_quiescent": "resident_native_boundary": "current_exact_recovery": "publication_cadence_evidence": "incremental_authoritative_journal": "writer_source": "writer_epoch": "sole_current_writer": "prior_bytes_never_rewritten_or_truncated": "producer_PID": "producer_create_time": "session_owner_nonce": "all_CPU_auxiliary_producers_actually_closed0": "native_auxiliary_writer_contexts_returned": "all_auxiliary_handles_released": "auxiliary_manifest_sha256": \\"merged_CPU_live_task_boundary\\": \\"merged_parent_process_exit0_claimed\\": \\"metadata_task_completed_no_CPU_inflight\\": \\"every_preserved_control_attempt_closed0_claimed\\": \\"legacy_all_CPU_auxiliary_producers_actually_closed0_scope\\": \\"actual_numerical_reference_frontend_and_external_cleanup_only\\": \\"producer_completion_scopes\\": \\"actual_owner\\": \\"actual_current_checkpoint_pointer\\": \\"active_auxiliary_workspaces\\": \\"committed_window_updates\\": \\"retained_metadata_bytes\\": \\"auxiliary_archives\\": \\"auxiliary_bytes\\": \\"archive_bytes\\": \\"live_payload_bytes\\": \\"live_pairs\\": \\"exact_pass_tail\\": \\"file_index\\": \\"history\\": \\"last_frame\\": \\"accounting\\": \\"checkpoint_mappings_released\\": \\"candidate_payload_handles_released\\": \\"checkpoint_write_inflight\\": \\"restore_in_progress\\": \\"native_release_ack\\": \\"native_work_quiescent\\": \\"resident_native_boundary\\": \\"current_exact_recovery\\": \\"publication_cadence_evidence\\": \\"incremental_authoritative_journal\\": \\"writer_source\\": \\"writer_epoch\\": \\"sole_current_writer\\": \\"prior_bytes_never_rewritten_or_truncated\\": \\"producer_PID\\": \\"producer_create_time\\": \\"session_owner_nonce\\": \\"all_CPU_auxiliary_producers_actually_closed0\\": \\"native_auxiliary_writer_contexts_returned\\": \\"all_auxiliary_handles_released\\": \\"auxiliary_manifest_sha256\\":'

def witness_decoder(packed):
    require(type(packed)is bytes and len(packed)>=6,'Complete dictionary zlib header')
    cmf,flg=packed[0],packed[1]
    require(cmf&15==8 and cmf>>4<=7 and (cmf*256+flg)%31==0 and flg&32,'Source dictionary witness stream')
    dictionary_id=int.from_bytes(packed[2:6],'big')
    dictionaries={zlib.adler32(WITNESS_ZDICT):WITNESS_ZDICT,zlib.adler32(WITNESS_ZDICT_V2):WITNESS_ZDICT_V2}
    require(len(dictionaries)==2 and dictionary_id in dictionaries,'Only explicit legacy/V2 generic schema DICTIDs')
    return zlib.decompressobj(zdict=dictionaries[dictionary_id])



def checked_normal_resume_request(config_pin,review,resume_pin,*,parent_source,guard):
    """Pure bounded metadata/process checks BEFORE namespace mutation or Popen.

    Only a normally closed owner whose actual TOOL0 was independently observed
    may be resumed. Unknown suffixes, inflight work and live old owners stop.
    """
    from fastmss_owned_process_closure_v2 import dead
    from asr18_full_run_serial_DEV_9223_v3 import read_logical_durable_head
    from asr18_BPE9223_operational_metadata_v2 import read_bound as physical_or_configured
    c=physical_or_configured(config_pin,guard);request=physical_or_configured(resume_pin,guard)
    compatibility=checked_compatibility(c['operational_compatibility'],c['full_run_identity'],guard)
    @contextmanager
    def phase(name):
        require(name=='read_index','Only bounded readonly startup metadata');guard();yield;guard()
    from asr18_BPE9223_indexed_checkpoint_storage_v2 import IndexedOwnedFutureCheckpointStore as ReadonlyStore
    readonly=c['process_readonly_metadata_authority']
    local_store=ReadonlyStore(c['operational_storage_owner'],guard=guard,
        operational_compatibility=readonly['operational_compatibility'])
    local_resolver=MetadataResolver(local_store,control_root=Path(c['operational_control_root']),guard=guard,
        metadata_phase=phase,identity=c['full_run_identity'],operational_compatibility=readonly['operational_compatibility'],
        current_native_resource_closure=None)
    def read_bound(pin,unused_guard=guard):
        if Path(pin['path']).is_file() and Path(pin['path']).stat().st_size==pin['bytes']:return physical_or_configured(pin,guard)
        require(type(pin) is dict and set(pin)=={'path','bytes','sha256'}
            and type(pin['bytes']) is int and 0<pin['bytes']<=16*1024**2,'Exact bounded original JSON metadata')
        with local_resolver(pin) as stream:
            raw=bytearray()
            while len(raw)<pin['bytes']:
                guard();block=stream.read(min(1024**2,pin['bytes']-len(raw)))
                require(block,'Complete archived startup metadata');raw.extend(block)
            require(stream.read(1)==b'','Complete original metadata EOF')
        def unique(pairs):
            value={}
            for key,item in pairs:
                require(key not in value,'Duplicate original metadata key');value[key]=item
            return value
        def invalid(value):raise ValueError('Nonfinite original metadata: '+value)
        return json.loads(raw,object_pairs_hook=unique,parse_constant=invalid)
    original=read_bound(compatibility['original_configuration'],guard)
    require(compatibility['legacy_guard_authority_source']==original['parent_driver_source'],
        'Original monitor authority remains explicit, no source stamp rewrite')
    require(review['resume_request']==resume_pin and review['invocation']=='normal_boundary_resume'
        and request['kind']=='ASR18_exact_normal_boundary_owner_resume_request_v1'
        and request['configuration']==config_pin and request['source_manifest']==c['source_manifest']
        and request['parent_source']==parent_source==c['parent_driver_source']
        and request['controller_source']==record(__file__)
        and request['run_identity_sha256']==digest(c['full_run_identity']),
        'Exact separately reviewed normal-resume request/source/configuration')
    snapshot=read_bound(request['normal_pause_snapshot'],guard)
    prior=read_bound(request['prior_owner_report'],guard);actual=read_bound(request['prior_owner_actual_TOOL0'],guard)
    require(snapshot['kind']=='ASR18_normal_coherent_owner_pause_snapshot_v1'
        and snapshot['configuration']==config_pin and snapshot['run_identity_sha256']==digest(c['full_run_identity'])
        and snapshot['parent_source']==parent_source and snapshot['controller_source']==record(__file__)
        and snapshot['namespace']==c['run_namespace'] and snapshot['normal_coherent_pause'] is True,
        'Same immutable genesis identity, no historical namespace adoption')
    read_bound(snapshot['worker_control_pool_state'],guard)
    require(prior['kind']=='ASR18_full_owner_normal_boundary_pause_report_v1' and prior['passed'] is True
        and prior['normal_pause_snapshot']==request['normal_pause_snapshot'] and prior['producer']==snapshot['producer']
        and prior['worker_source']==record(__file__) and prior['parent_source']==parent_source
        and prior['actual_optimizer_updates']==snapshot['actual_optimizer_updates']
        and prior['configuration']==config_pin and actual['kind']=='ASR18_observed_normal_owner_pause_TOOL0_v1'
        and actual['actual_tool_exit_code']==0 and actual['report']==request['prior_owner_report']
        and actual['normal_pause_snapshot']==request['normal_pause_snapshot']
        and actual['producer']==prior['producer'] and actual['parent_source']==parent_source
        and actual['configuration']==config_pin and dead(prior['producer']),
        'Actual normal owner TOOL0 and exact old producer dead, never file-existence inference')
    joined=read_bound(request['prior_owner_guard'],guard)
    require(actual['owner_guard']==request['prior_owner_guard']
        and joined['kind']=='ASR18_actual_outer_owner_guard_joined_completion_v1'
        and joined['producer']==prior['producer']
        and joined['source']==compatibility['legacy_guard_authority_source']
        and joined['monitor_joined'] is True and joined['monitor_clean'] is True
        and type(joined['samples']) is int and joined['samples']>=3,
        'Actual old owner clean joined guard plus separately observed outer TOOL0')
    for key in ('closed_native','closed_recovery'):
        proof=read_bound(snapshot[key],guard);verify(proof,kind='ASR18_owned_GPU_process_completion_v1',owner=OWNER,guard=guard)
        require(proof['parent_source']==parent_source and proof['worker_source']==c['worker_driver_source'],
                'Same genesis native/recovery execution sources')
    pointer=read_bound(snapshot['current_capture']['pointer'],guard)
    release=read_bound(snapshot['current_release'],guard);response=snapshot['current_response']
    native_proof=read_bound(snapshot['closed_native'],guard)
    native_report=read_bound(native_proof['report'],guard)
    require(native_report['current_checkpoint']==snapshot['current_capture']
        and release['producer']==native_proof['producer'] and release['worker_source']==c['worker_driver_source']
        and release['request_nonce']==response['nonce'] and release['writer_intent']==snapshot['current_intent']
        and response['window_completion']==snapshot['current_completion']
        and response['checkpoint_storage_release_ack']==snapshot['current_release']
        and release['checkpoint_pointer']==snapshot['current_capture']['pointer']
        and all(release[k] is True for k in ('writer_context_returned','persistence_reader_context_returned',
                                            'startup_restore_context_returned')),
        'Exact closed native response nonce/current capture/release contexts; no old idle proof promoted')
    ack=read_bound(snapshot['current_ack'],guard);coverage=read_bound(snapshot['current_coverage'],guard)
    require(ack['kind']=='ASR18_full_run_current_latest_recovery_and_verified_persistence_prefix_v1'
        and ack['final_checkpoint_pointer']==snapshot['current_capture']['pointer']
        and ack['current_latest_all207_threeAdam_all4RNG_sampler_prefix_nextdraw_match_actual_pre_save'] is True
        and ack['actual_closed_native_index']==snapshot['closed_native']
        and ack['actual_closed_recovery_index']==snapshot['closed_recovery']
        and ack['persistence_prefix_coverage_index']==snapshot['current_coverage']
        and coverage['passed'] is True and coverage['historical_fresh_GPU_restores_claimed'] is False,
        'Actual current restore and complete PRE/persistence prefix ACK')
    latest=read_json(Path(c['run_namespace'])/'durable/latest.json')
    require(latest['head']==snapshot['durable_head'] and latest['owner']==snapshot['durable_owner'],
            'No stale ACK/head, no newer durable state silently dropped')
    head=read_logical_durable_head(snapshot['durable_head'])['head']
    require(head['identity_sha256']==digest(c['full_run_identity']) and head['owner']==snapshot['durable_owner']
        and head['pending']['windows']==head['pending']['publications']==0
        and head['lifecycle']==snapshot['lifecycle_state']
        and head['last_recovery']==snapshot['current_ack'],
        'Entire bounded lifecycle/winner and pending0 are exact packed authority')
    state=read_json(Path(c['checkpoint_storage_namespace'])/'state.json')
    require(state['owner']==snapshot['storage_owner'] and state['latest']==snapshot['current_intent']['id']
        and state['pending_publication'] is None and state['pending_retirement'] is None
        and state['catalog']==snapshot['storage_catalog'],
        'No pending publication/retirement/inflight workspace at normal pause')
    import sqlite3
    catalog=Path(c['checkpoint_storage_namespace'])/'catalog.sqlite'
    db=sqlite3.connect('file:'+catalog.as_posix()+'?mode=ro',uri=True)
    try:require(db.execute("SELECT id FROM auxiliary WHERE status='preparing' LIMIT 1").fetchone() is None,
                'No preparing auxiliary producer workspace on normal resume')
    finally:db.close()
    require(not Path(snapshot['current_intent']['temporary_path']).exists(),'No inflight tensor writer temporary on normal resume')
    sampler=snapshot['sampler_state'];frontier=progress(sampler)
    require(digest(sampler)==pointer['sampler_state_sha256']==head['lifecycle']['last_sampler_sha256']
        and frontier['optimizer_updates']==pointer['committed_updates']==snapshot['actual_optimizer_updates']
        and 0<frontier['optimizer_updates']<=c['planned_native_updates'],
        'Exact positive original counter/sampler frontier, never a reset or guessed reconstruction')
    if c['owner_validation_only']:
        require(frontier['optimizer_updates']==4 and frontier['broad_draws']==56 and frontier['cleaner_draws']==0
            and snapshot['actual_owner_invocation_count']==1,'Separately bounded split4/4 validation config')
    packet=read_bound(snapshot['current_capture']['next_draw_packet'],guard)
    require(digest(packet)==snapshot['next_draw_packet_sha256'],'Exact saved next-draw packet')
    prefix=pointer['journal_prefix'];path=Path(prefix['path']).resolve()
    require(path.parent==Path(c['run_namespace'])/'journals' and path.stat().st_size==prefix['offset']
        and prefix==snapshot['journal_prefix'],'Whole closed registered journal only; unknown suffix preserved/rejected')
    sha=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(1024**2):guard();sha.update(block)
    require(sha.hexdigest()==prefix['sha256'],'Exact complete saved journal bytes')
    require(type(snapshot['native_sequence']) is int and 1<=snapshot['native_sequence']<513
        and type(snapshot['recovery_sequence']) is int and 1<=snapshot['recovery_sequence']<513
        and type(snapshot['recovery_control_sequence']) is int
        and 1<=snapshot['recovery_sequence']<=snapshot['recovery_control_sequence']<513
        and type(snapshot['last_window_index']) is int and snapshot['last_window_index']>=0
        and type(snapshot['actual_owner_invocation_count']) is int and 1<=snapshot['actual_owner_invocation_count']<513,
        'Exact cumulative normal-boundary counters, never reset on fresh owner')
    return snapshot




def _original_proof_sequence(pin,guard):
    """Exact original bounded witness frames; no guessed/reset proof counter."""
    require(record(pin['path'])==pin and pin['bytes']<=805306368,'Exact prior witness stream')
    previous=0
    with Path(pin['path']).open('rb') as stream:
        while header:=stream.read(40):
            guard();require(len(header)==40,'Whole acknowledged witness header')
            packed_size,raw_size=struct.unpack('<II',header[:8])
            require(0<packed_size<=2048 and 0<raw_size<=65536,'Frozen witness frame bounds')
            packed=stream.read(packed_size);decoder=witness_decoder(packed)
            raw=decoder.decompress(packed,65537)
            require(len(packed)==packed_size and len(raw)==raw_size and decoder.eof
                and not decoder.unused_data and not decoder.unconsumed_tail
                and hashlib.sha256(raw).digest()==header[8:],'Complete exact witness frame bytes')
            value=json.loads(raw);original=value['original'];path=Path(original['path'])
            require(path.parent==Path(pin['path']).parent and '.proof.' in path.name
                and path.name.endswith('.json'),'Exact original owned proof path')
            sequence=int(path.name.rsplit('.proof.',1)[1][:-5])
            require(sequence==previous+1,'Continuous original proof counter; no reset')
            previous=sequence
    require(record(pin['path'])==pin and previous==2,'Only the reviewed real p1 two-proof stream')
    return previous




































class FullOwner:
    def __init__(self,config_path,review_path,folder,*,parent_source,ParentGuard,GPUService,run_CPU,cleanup_script,
                 storage,fresh_admit,execute_owned_DEV,native_boundary_DEV,install_owner,
                 verify_worker_control_protection,verify_closed_worker_control,resume_request=None):
        storage();self.config_pin=record(config_path);self.config=read_json(config_path);self.review_pin=record(review_path)
        self.review=read_json(review_path);self.parent_source=parent_source;self.GPUService=GPUService;self.run_CPU=run_CPU
        self.cleanup_script=cleanup_script;self.fresh_admit=fresh_admit;self.producer=identity();self.gpu=None
        self.folder=Path(folder).resolve();self.root=Path(self.config['run_namespace']).resolve();self.control=self.root/'control'
        self.identity=deepcopy(self.config['full_run_identity']);self.closed_native=None;self.closed_recovery=None
        self.current_capture=None;self.current_completion=None;self.current_release=None;self.current_idle=None
        self.current_recovery_report=None;self.current_ack=None;self.current_coverage=None;self.prior_boundary=None
        self.current_packet=None;self.last_window_index=-1;self.resume_checked=False;self.DEV=None
        self.native_sequence=0;self.recovery_sequence=0;self.recovery_control_sequence=0;self.proof_sequence=0;self.boundary_state=None
        self.resume_request=resume_request;self.resume_snapshot=None;self.actual_owner_invocation_count=1
        self.merged_CPU_inflight=False
        c=self.config;r=self.review
        require(c['kind']=='ASR18_full_run_native_owner_configuration_v1' and c['production_enabled'] is False
                and type(c['owner_validation_only']) is bool and c['diagnostic_only']==c['owner_validation_only']
                and c['vocabulary']==9223 and c['planned_native_updates']==401072
                and c['parent_driver_source']==parent_source and record(parent_source['path'])==parent_source,
                'Exact original full-run owner and immutable source configuration')
        require(r['accepted'] is True and r['reviewer']=='/root' and r['configuration']==self.config_pin
                and r['source_manifest']==c['source_manifest'] and r['production_approval_claimed'] is False,
                'Exact separate source/resource/execution review')
        if c['owner_validation_only']:
            require(r['scope']=='finite_full_owner_first4_normal_pause_resume_next4_equivalence_v1'
                    and c['maximum_owner_validation_updates']==8 and c['maximum_validation_updates_per_window']==4
                    and self.identity['owner_validation_only'] is True
                    and self.identity['production_source_eligible'] is False,
                    'Exact original first8/two-window owner validation, no diagnostic seed promotion')
        else:
            require(c['full_run_launch_enabled'] is True and r['full_run_launch_enabled'] is True
                    and r['scope']=='reviewed_full_two_broad_one_cleaner_operational_continuation_v1'
                    and self.identity['owner_validation_only'] is False,
                    'Long launch requires separately exact human-authorized configuration; no CLI override')
        require(c['normal_boundary_resume_enabled'] is True
                and (not c['owner_validation_only'] or c['owner_validation_pause_after_updates']==4),
                'Resume-enabled immutable genesis config required from first invocation')
        self.committed60_initial=self.prepared60_initial=self.ACKed60_initial=False
        self.reserved_committed60=self.ACKed40_initial=self.pending40_initial=self.pending40_finalization=False
        self.resource_initial_pending=self.acknowledged_resource_pending=False
        self.resource_closure=None;self.reserved_preparation=None;self.prepared_CPU_resume=None;self.pending40_inputs=None
        checked_compatibility(c['operational_compatibility'],self.identity)
        if resume_request is not None:
            request=read_bound(resume_request)
            require(request['kind']=='ASR18_exact_normal_boundary_owner_resume_request_v1','Fresh RUN03 supports genuine normal continuation only')
            self.resume_snapshot=checked_normal_resume_request(self.config_pin,r,resume_request,parent_source=parent_source,guard=lambda:None)
            require(self.root.is_dir() and self.control.is_dir(),'Only exact normally closed same namespace may reopen')
        else:
            require(r['invocation']=='genesis_authorized_production' and r['fresh_genesis_authorized'] is True
                and r['fresh_genesis_namespace']==str(self.root) and self.root.is_dir()
                and {p.name for p in self.root.iterdir()}=={'checkpoints'},
                'Explicit empty precreated store only; never reset/adopt RUN02')
            empty=c['fresh_empty_storage_bindings']
            require(empty==r['fresh_empty_storage_bindings'] and empty['owner']==c['operational_storage_owner']
                and all(record(empty[k]['path'])==empty[k] for k in ('owner','state','catalog','actual0')),
                'Exact real empty-storage creation0 and all ownership/state/catalog pins')
            actual=read_bound(empty['actual0']);state=read_bound(empty['state'])
            require(actual['actual_tool_exit_code']==actual['actual_exec_tool_exit_code']==0
                and state['owner']==empty['owner'] and state['revision']==state['publications']==0
                and state['latest'] is state['pending_publication'] is state['pending_retirement'] is None
                and state['catalog']['revision']==0 and state['catalog']['publications']==0
                and state['history']['offset']==0 and state['winners']==[],
                'Genuine metadata-only storage0; no lifecycle/journal/model or prior training adoption')
        administrative=r.get('administrative_maximum_new_updates')
        require(administrative is None or (not c['owner_validation_only'] and type(administrative) is int
            and 4<=administrative<=c['planned_native_updates']),'Explicit reviewed invocation-only administrative budget')
        self.administrative_maximum_new_updates=administrative
        require(self.folder.is_relative_to(ROOT) and self.root.is_relative_to(ROOT/'runs')
                and not self.folder.exists(),'Fresh actual outer supervisor output folder per invocation')
        self.folder.mkdir()
        if self.resume_snapshot is None:self.control.mkdir()
        self.parent=ParentGuard(self.folder)
        self.parent.configure_control_quota(self.control,c['control_log_policy'],identity=self.identity,
            create=self.resume_snapshot is None,operational_compatibility=c['operational_compatibility'])
        self.parent.configure_merged_CPU(c['merged_CPU_policy'],c['operational_compatibility'])
        if self.resume_snapshot is not None:
            require(self.parent.control_pool.state_pin==self.resume_snapshot['worker_control_pool_state']
                and self.parent.control_pool.snapshot_binding()==self.resume_snapshot['worker_control_pool_binding'],
                'Same complete worker-pool quota/protection/archive counters; no reset on fresh owner')
        self.began=time.monotonic();self.current_worker_config=None
        self.context=FullRunSourceContext(c['planning_arguments'],guard=self.parent.check);self.schedule=self.context.new_schedule()
        require(digest(self.schedule.state_dict())==c['initial_sampler_sha256'] and self.schedule.committed_updates()==0,
                'Original source BPE64250/LoRA62250 RNG/sampler frontier0')
        for i in range(3):
            for extra in (c['entire_worker_forecast']['entire_worker_private_forecast_bytes']+384000000+64000000,
                          c['CPU_owned_allowance_bytes']+128000000):
                require(fresh_admit(extra)['passed'],'Three fresh whole-domain admissions before namespace/producer work')
            time.sleep(.2)
        resuming=self.resume_snapshot is not None
        self.journal_namespace=JournalNamespace(self.root/'journals',identity=self.identity,guard=self.parent.check,create=not resuming)
        if resuming:
            snapshot=checked_normal_resume_request(self.config_pin,r,resume_request,parent_source=parent_source,guard=self.parent.check)
            self.schedule.load_state_dict(normalize_schedule_state(snapshot['sampler_state']))
            for key in ('current_capture','current_completion','current_response','current_release','current_recovery_report','current_ack',
                'current_coverage','current_packet','current_CPU0','current_CPU_report','current_bridge',
                'current_cleanup0','current_cleanup','current_workspace','current_original_plan','current_intent',
                'closed_native','closed_recovery','current_cadence','native_sequence','recovery_sequence','recovery_control_sequence','proof_sequence','last_window_index'):
                setattr(self,key,deepcopy(snapshot[key]))
            self.actual_owner_invocation_count=snapshot['actual_owner_invocation_count']+1
            self.reserved_preparation=deepcopy(snapshot.get('reserved_preparation'))
            self.prepared_CPU_resume=deepcopy(snapshot.get('prepared_CPU'))
            self.pending40_inputs=deepcopy(snapshot.get('pending40'))
            self.initial_journal=Path(snapshot['journal_prefix']['path'])
            require(self.initial_journal in self.journal_namespace.paths,'Same exact registered journal, no branch/copy/reset')
            require(digest(full_run_next_draw_packet(self.schedule,self.context,
                optimizer_updates=self.schedule.committed_updates(),guard=self.parent.check))==snapshot['next_draw_packet_sha256'],
                    'Actual source-context next draw matches normal pause BEFORE any prepare/Popen')
        else:self.initial_journal=self.journal_namespace.register('commits_000000.jsonl')
        self.invocation_initial_updates=self.schedule.committed_updates()
        lifecycle_functions={name:getattr(self,name) for name in CALLBACKS}
        callbacks=SourceBoundOwnerCallbacks(sources={name:record(inspect.getsourcefile(function)) for name,function in lifecycle_functions.items()},callbacks=lifecycle_functions)
        self.lifecycle=FullRunLifecycle(self.schedule,identity=self.identity,callbacks=callbacks,
            qualified_winner=deepcopy(c['qualified_source_winner']),
            restored=None if not resuming else snapshot['lifecycle_state'],
            restore_proof=(c['prior_committed60_CPU_memory_failure'] if self.committed60_initial else
                self.resource_closure if self.resource_initial_pending else self.current_ack))
        prefix=OwnerPrefixVerifier(resolve_member=self.resolve_member,guard=self.parent.check,operational_compatibility=c['operational_compatibility'])
        methods={'verify_pending_window':self.verify_pending_window,'verify_writer_authority':self.verify_writer_authority,
            'verify_durable_resume':self.verify_durable_resume,
            'verify_current_recovery_and_persistence_prefix':prefix.verify_current_recovery_and_persistence_prefix}
        proofs=OwnerProofCallbacks(compatibility=c['operational_compatibility'],identity=self.identity,sources={name:record(inspect.getsourcefile(function)) for name,function in methods.items()},functions=methods)
        with self.parent.metadata_phase('read_index'):
            self.durable=PackedDurableLifecycleAdapter(self.lifecycle,self.root/'durable',policy=c['durable_policy'],proofs=proofs,create=not resuming,
                resume_proof=None if not resuming else self.current_ack,
                operational_compatibility=c['operational_compatibility'])
            require(self.durable.owner==read_bound(c['session'],self.parent.check)['durable_owner'],
                    'Actual newly written deterministic owner equals session expected binding, never an invented actual receipt')
            self.store=IndexedOwnedFutureCheckpointStore(snapshot['storage_owner'] if resuming else c['operational_storage_owner'],
                guard=self.parent.check,operational_compatibility=c['operational_compatibility'])
        # CPU bridge/context require this exact frozen session binding in the
        # SAME process.  It is read-only; self.store/proof mutation authority
        # remains the separately checked CURRENT operational compatibility.
        readonly=deepcopy(c['process_readonly_metadata_authority'])
        current=checked_compatibility(c['operational_compatibility'],self.identity,self.parent.check)
        require(readonly==read_bound(c['session'],self.parent.check)['operational_metadata']
            ==current['process_readonly_metadata_authority'],'Exact ROOT/session read-only authority from first configure')
        self.resolver=configure_metadata(**readonly,guard=self.parent.check,metadata_phase=self.parent.metadata_phase)
        # Prefix's resolver is this exact source-bound wrapper, delegated to the
        # separately frozen concrete resolver; identity pins this controller too.
        self.prefix=prefix;self.proofs=proofs;install_owner(self)
        self.CPU_execution_delegate_source=c['CPU_execution_delegate_source']
        compatibility=checked_compatibility(c['operational_compatibility'],self.identity,self.parent.check)
        require(self.CPU_execution_delegate_source==compatibility['CPU_execution_delegate_source']
            and Path(inspect.getsourcefile(self.run_CPU)).resolve()==Path(self.CPU_execution_delegate_source['path']),
            'Actual CPU Popen executor is the explicitly reviewed original code6 delegate')
        self.CPU_leaf_execution_delegate_source=c['CPU_leaf_execution_delegate_source']
        import run_ASR18_BPE9223_CPU_leaf_observer_v9 as CPU_leaf_delegate
        require(self.CPU_leaf_execution_delegate_source==compatibility['CPU_leaf_execution_delegate_source']
            and record(inspect.getsourcefile(CPU_leaf_delegate.run_leaf))==self.CPU_leaf_execution_delegate_source
            and c['CPU_controller_source']==compatibility['operational_sources']['CPU_orchestrator'],
            'Overall orchestrator8 and actual leaf Popen observer7 are separately source bound')
        self.pool_verify_protection=verify_worker_control_protection
        self.pool_verify_closed=verify_closed_worker_control
        self.worker_control_archive_result=[]
        self._protect_worker_controls()
        from asr18_full_run_serial_DEV_9223_v3 import SerialDEVCallbacks
        self.DEV=SerialDEVCallbacks(ROOT_policy_review=c['ROOT_DEV_policy_review'],execution_sources=c['DEV_execution_sources'],
            execute_owned=execute_owned_DEV,execute_owned_source=parent_source,native_boundary=native_boundary_DEV,
            native_boundary_source=parent_source,native_boundary_sources=c['DEV_native_boundary_sources'],output_directory=self._DEV_folder())
        write_json(self.folder/'launch.json',{'configuration':self.config_pin,'review':self.review_pin,'producer':self.producer,
            'run_identity_sha256':digest(self.identity),'resume_request':resume_request,
            'actual_owner_invocation_count':self.actual_owner_invocation_count,'production_enabled':False})
        if self.resource_initial_pending or self.acknowledged_resource_pending:self.resume_snapshot={'resource_transition':True}

    def _new(self,path,value):
        path=Path(path).resolve()
        require((path.is_relative_to(self.root) or path.is_relative_to(self.folder)) and not path.exists(),
                'Fresh immutable exact owned control artifact')
        if path.is_relative_to(self.control):self.parent.control_guard(path,len(canonical(value))+4096)
        write_json(path,value);return record(path)

    def _retire_witness(self,pin,mutation):
        # An immutable range in ONE bounded source-owned control stream keeps
        # every original witness byte/pin, including retirement operations whose
        # store event itself only contains the original witness binding.
        require(record(pin['path'])==pin,'Same returned-operation witness before exact owned retirement')
        path=Path(pin['path']).resolve()
        require(path.parent==self.control and '.proof.' in path.name,'Only exact new operation witness')
        original=path.read_bytes()
        event={'kind':'ASR18_exact_archived_operation_witness_v1','original':pin,
            'mutation':mutation,'raw_UTF8':original.decode('utf-8')}
        raw=canonical(event);encoder=zlib.compressobj(level=9,zdict=WITNESS_ZDICT_V2)
        packed=encoder.compress(raw)+encoder.flush()
        require(len(raw)<=65536 and len(packed)<=2048,'Complete lossless witness frame cap; no field dropping')
        log=self.control/'storage_operation_witnesses.bin'
        offset=log.stat().st_size if log.exists() else 0
        frame=struct.pack('<II',len(packed),len(raw))+hashlib.sha256(raw).digest()+packed
        require(offset+len(frame)<=805306368,'Entire declared operation-witness control horizon')
        self.parent.control_guard(log,offset+len(frame))
        with log.open('ab') as stream:stream.write(frame);stream.flush();os.fsync(stream.fileno())
        with log.open('rb') as stream:stream.seek(offset);actual=stream.read(len(frame))
        decoder=witness_decoder(actual[40:])
        decoded=decoder.decompress(actual[40:],65537)
        require(decoder.eof and not decoder.unconsumed_tail and not decoder.unused_data,
                'Complete fixed source-dictionary witness EOF, no trailing/unconsumed compressed data')
        require(actual==frame and decoded==raw and json.loads(decoded)['raw_UTF8'].encode('utf-8')==original
            and hashlib.sha256(original).hexdigest()==pin['sha256'] and len(original)==pin['bytes'],
            'Actual fsynced immutable complete witness range/readback before single owned unlink')
        path.unlink();require(not path.exists(),'Actual exact transient witness retirement')

    def _DEV_folder(self):
        path=self.root/'DEV'
        if self.resume_snapshot is None:path.mkdir(exist_ok=False)
        else:require(path.is_dir(),'Same owned DEV output root on explicit normal resume')
        return path

    @contextmanager
    def resolve_member(self,binding):
        require(hasattr(self,'resolver'),'Archive resolver unavailable before namespace initialization')
        if (type(binding) is dict and set(binding)=={'path','bytes','sha256'}
            and not Path(binding['path']).exists() and self.parent.control_pool.owns_member(binding)):
            with self.parent.control_pool.resolve_member(binding) as stream:yield stream
        else:
            with self.resolver(binding) as stream:yield stream

    def _worker_control_protection(self):
        state=read_bound(self.parent.control_pool.state_pin,self.parent.check);folders=set()
        if self.gpu is not None:folders.add(str(self.gpu.folder))
        for key in ('closed_native','closed_recovery'):
            pin=getattr(self,key,None)
            if pin is not None:folders.add(str(Path(pin['path']).parent))
        folders.update(x['folder']for x in state['live']if Path(x['folder']).name.startswith('DEV_'))
        folders&={x['folder']for x in state['live']}
        files=[self.config_pin,self.config['session'],self.config['operational_compatibility']]
        files.extend(x['configuration']for x in state['live']if x['folder']in folders)
        if self.resume_request is not None:
            request=read_bound(self.resume_request,self.parent.check)
            require(request['kind']=='ASR18_exact_normal_boundary_owner_resume_request_v1','No abnormal-source continuation in fresh RUN03')
            files.append(self.resume_request)
            files.extend(request[k]for k in ('normal_pause_snapshot','prior_owner_report','prior_owner_actual_TOOL0','prior_owner_guard'))
        unique={x['path']:x for x in files}
        return sorted(folders),[unique[key]for key in sorted(unique)]

    def _protect_worker_controls(self):
        folders,files=self._worker_control_protection()
        evidence=self.current_ack if self.current_ack is not None else self.config_pin
        self.parent.control_pool.protect(folders,files=files,evidence=evidence,
            verify_protection=self.pool_verify_protection)
        self.parent.worker_slots=list(self.parent.control_pool.live_folders())
        self.parent.control_guard()

    def verify_worker_control_protection(self,request):
        self.check();pool=self.parent.control_pool
        require(request['identity']==self.identity and request['owner']==pool.owner_pin
            and request['revision']==pool.revision,'Exact current pool owner/identity/revision')
        folders,files=self._worker_control_protection()
        supplied=(request['folders'],request['files']) if request['operation']=='verify_complete_worker_control_protection' else (
            request['protected_folders'],request['protected_files'])
        require(supplied==(folders,files) and request['evidence']==(self.current_ack if self.current_ack is not None else self.config_pin),
            'Complete exact live/current native+recovery/DEV and immutable input protection')
        return True

    def verify_closed_worker_control(self,request):
        self.check();pool=self.parent.control_pool
        require(self.gpu is None and self.current_ack is not None and request['evidence']==self.current_ack
            and request['identity']==self.identity and request['owner']==pool.owner_pin
            and request['revision']==pool.revision
            and request['normal_coherent_boundary_and_mapping_release_required'] is True
            and request['crash_tail_rollback_or_original_artifact_deletion_permitted'] is False,
            'Only normal current-recovered source-owned worker-control retirement')
        self._verify_closed(self.closed_native);self._verify_closed(self.closed_recovery)
        ack=read_bound(self.current_ack,self.parent.check);release=read_bound(self.current_release,self.parent.check)
        require(self.durable.head['pending']['publications']==self.durable.head['pending']['windows']==0
            and self.durable.head['last_recovery']==self.current_ack
            and ack['final_checkpoint_pointer']==self.current_capture['pointer']
            and release['checkpoint_pointer']==self.current_capture['pointer']
            and release['request_nonce']==self.current_response['nonce']
            and all(release[k] is True for k in ('writer_context_returned','persistence_reader_context_returned','startup_restore_context_returned')),
            'Actual CURRENT restore/prefix ACK and returned candidate contexts before retirement')
        proof=self._verify_closed(request['process0']);reservation=request['reservation']
        folders,files=self._worker_control_protection()
        prior=read_bound(reservation['configuration'],self.parent.check)
        compatibility=checked_compatibility(self.config['operational_compatibility'],self.identity,self.parent.check)
        lineage=[]
        for role in ('original_source_manifest','operational_source_manifest','prior_operational_source_manifest','actual_ACKed60_source_lineage_manifest'):
            if role in compatibility:lineage.extend(read_bound(compatibility[role],self.parent.check)['bindings'])
        require(prior['full_run_identity']==self.identity
            and prior['worker_driver_source']==reservation['worker_source'] in lineage
            and prior['parent_driver_source']==reservation['parent_source'] in lineage,
            'Original paired configuration/scientific identity and ROOT-reviewed actual historical source lineage')
        require(reservation in request['live'] and reservation['folder'] not in folders
            and proof['worker_source']==reservation['worker_source']
            and proof['parent_source']==reservation['parent_source']
            and Path(request['process0']['path']).parent==Path(reservation['folder'])
            and reservation['configuration'] in request['files']
            and not any(x in request['files'] for x in files),
            'Exact obsolete registered CLOSED0 reservation, complete controls and no protected files')
        return True

    def _archive_obsolete_worker_controls(self):
        require(self.gpu is None and self.current_ack is not None,'Only actual coherent normal boundary')
        self._protect_worker_controls();pool=self.parent.control_pool
        state=read_bound(pool.state_pin,self.parent.check);protected=set(state['protected_folders'])
        obsolete=[x for x in state['live'] if x['folder'] not in protected]
        require(len(obsolete)<=48,'Bounded registered live set, no history inventory')
        results=[]
        for item in obsolete:
            folder=Path(item['folder']);proof=record(folder/'actual_owned_native_process0.json')
            # helper verifies every member/EOF before exact owned unlink. This
            # parent then independently exercises registered original lookup.
            originals=(proof,item['configuration'])
            result=pool.archive(folder,process0=proof,paired_files=[item['configuration']],
                evidence=self.current_ack,verify_closed=self.pool_verify_closed)
            for original in originals:
                require(not Path(original['path']).exists() and pool.owns_member(original),
                        'Exact retired worker/config registered original lookup')
                sha=hashlib.sha256();count=0
                with self.resolve_member(original) as stream:
                    while block:=stream.read(1024**2):self.parent.check();sha.update(block);count+=len(block)
                require(count==original['bytes'] and sha.hexdigest()==original['sha256'],'All selected original bytes roundtrip after retirement')
            results.append(result)
            self.parent.worker_slots=list(pool.live_folders());self.parent.control_guard()
        self.worker_control_archive_result=results[-2:]

    def check(self):
        self.parent.check();require(identity()==self.producer and record(self.config_pin['path'])==self.config_pin
            and record(self.parent_source['path'])==self.parent_source,'Actual same source-bound owner process/configuration')
        lease=read_json(Path(self.parent_source['path']).resolve().parents[2]/'work/gpu_training_owner.json')
        require(lease['owner']==OWNER,'Exclusive native owner unchanged')
        if self.gpu is None:require(not lease['actual_GPU_running'] and not lease.get('actual_owned_worker_pid'),'No unknown live GPU child')
        else:require(lease['actual_GPU_running'] and lease['actual_owned_worker_pid']==self.gpu.pid,'Exact actual owned GPU child')
        return True

    def verify_writer_authority(self,request):
        self.check();require(request['identity']==self.identity and request['namespace']==str(self.root/'durable')
            and request['policy']==self.config['durable_policy'],'Exact sole durable writer authority')
        return True

    def verify_pending_window(self,request):
        self.check();require((self.pending40_finalization or self.gpu is not None and self.current_idle is not None) and request['identity']==self.identity
            and request['plan']==self.current_packet['coordinator_plan'] and request['completion']==self.current_completion,
            'Actual current owned window, not report existence alone')
        pointer=read_bound(self.current_capture['pointer'],self.parent.check)
        if self.pending40_finalization:self._verify_pending40_quiescence()
        else:verify_resident_idle(read_bound(self.current_idle,self.parent.check),pointer=pointer,phase=self.current_packet['phase'],
            proof_binding=self.current_idle,guard=self.parent.check)
        io=_Metadata(self.identity,self.resolve_member,self.parent.check,self.config['operational_compatibility']);journal=_JournalPrefixes(self.parent.check)
        try:
            _,after=_capture(self.current_capture,self.current_capture['pointer'],io,identity_sha=digest(self.identity),
                previous_sampler=digest(request['before_state']),previous_updates=progress(request['before_state'])['optimizer_updates'],journal=journal)
            require(digest(after)==digest(request['expected_after_state'])==digest(self.schedule.state_dict())
                    and request['all_captures']==[self.current_capture],'Exact PRE/persistence current actual sampler prefix')
        finally:journal.close()
        return True

    def verify_durable_resume(self,request):
        self.check();require(self.gpu is None and self.current_ack is not None and request['restore_proof']==self.current_ack
            and request['identity']==self.identity and request['namespace']==str(self.root/'durable')
            and request.get('native_replay_or_sampler_rollback_permitted') is False,'Exact current recovery before same-run normal reopen')
        value=read_bound(self.current_ack,self.parent.check)
        require(request['state']['last_recovery']==self.current_ack and request['state']['pending']['publications']==0
            and value['final_checkpoint_pointer']==self.current_capture['pointer'],'No rollback or abnormal owner0 fiction')
        self._verify_closed(self.closed_native);self._verify_closed(self.closed_recovery);return True

    def verify_resume(self,request):
        if self.committed60_initial:
            require(request['proof']==self.config['prior_committed60_CPU_memory_failure'] and request['identity']==self.identity
                and request['lifecycle']==self.resume_snapshot['lifecycle_state']
                and request['schedule']==self.schedule.state_dict() and self.schedule.committed_updates()==60,
                'Actual saved60 lifecycle/sampler only before mandatory fresh60 recovery')
            return True
        if self.resource_initial_pending:
            require(request['proof']==self.resource_closure and request['identity']==self.identity
                and request['lifecycle']==self.resume_snapshot['lifecycle_state']
                and request['schedule']==self.resume_snapshot['sampler_state'],
                'Only exact saved production20 metadata before mandatory fresh recovery')
            checked_resource_stop(self.resource_closure,identity=self.identity,
                compatibility=self.config['operational_compatibility'],guard=self.parent.check)
            return True
        require(self.current_ack is not None and request['proof']==self.current_ack and request['identity']==self.identity,
            'Actual exact current recovery before lifecycle reconstruction')
        expected=(self.resume_snapshot['lifecycle_state'] if not hasattr(self,'lifecycle') else self.lifecycle.state_dict())
        value=read_bound(self.current_ack,self.parent.check)
        require(request['lifecycle']==expected and request['schedule']==self.schedule.state_dict()
                and request['lifecycle']['last_pointer']==value['final_checkpoint_pointer']
                and digest(request['schedule'])==request['lifecycle']['last_sampler_sha256'],'Current authoritative lifecycle/schedule')
        return True

    def verify_winner(self,request):
        winner=request['winner']
        if winner==self.config['qualified_source_winner']:
            summary=read_bound(self.config['qualified_source_DEV_summary'],self.parent.check)
            errors={lang:summary['validation_'+lang]['asr']['word_errors'] for lang in REFERENCES}
            require(winner['strict_qualified'] is True and winner['word_errors']==errors and winner['reference_words']==REFERENCES
                and all(summary['validation_'+l]['asr']['reference_words']==n for l,n in REFERENCES.items())
                and all(errors[l]<=cap for l,cap in LIMITS.items())
                and winner['artifact']==self.config['planning_arguments']['expected_original_sources']['parked_parent'],
                'Exact qualified original source/winner; no diagnostic promotion')
            return True
        artifact=winner.get('artifact') if type(winner) is dict and 'artifact' in winner else winner
        report=request.get('DEV',winner.get('DEV') if type(winner) is dict else None)
        require(report is not None and self.current_ack is not None,
                'Actual independently sealed qualified winner, never a diagnostic candidate')
        actual=read_bound(report,self.parent.check)
        source_request=read_bound(actual['source_request'],self.parent.check)
        from asr18_full_run_serial_DEV_9223_v3 import verified_completed_DEV
        require(verified_completed_DEV(actual['source_request'],actual['actual_worker_report'],actual['actual_owned_DEV_process0'])==actual,
                'Same sealed actual official350 execution, never a retention fixture')
        from asr18_BPE9223_lifecycle_v3 import official_DEV
        score=official_DEV(actual,artifact,read_bound(self.identity['scorer_policy'],self.parent.check))
        require(score['strict_all_six_passed'] and source_request['identity']==self.identity
                and source_request['checkpoint_pointer']==artifact,
                'Exact integer all-six retention and original immutable candidate/run identity')
        if type(winner) is dict and 'artifact' in winner:
            with self.parent.metadata_phase('read_index'):
                with self.store.locked():
                    state=self.store._load()[0]
                    entries=[self.store._entry(key,verify_files=True) for key in state['winners']]
                require(any(entry['pointer']==artifact and entry.get('qualification') for entry in entries),
                        'Historical qualified winner remains actually registered/protected with its original immutable DEV')
        else:
            require(artifact==self.current_capture['pointer'],'Only current recovered candidate can be NEWLY promoted')
        if type(winner) is dict and 'artifact' in winner:
            require(winner['strict_qualified'] and winner['word_errors']==score['word_errors']
                and winner['reference_words']==REFERENCES,'Stored winner exact official score')
        return True

    def admit_operation(self,request):self.check();return True
    def prepare_window(self,request):
        raise ValueError('Owner drives lifecycle.request + pass-bounded plan directly; lifecycle.advance is intentionally unused')
    def execute_window(self,request):
        raise ValueError('Owner drives actual resident persistence + durable pending callback, never fake per-window restore')
    def verify_checkpoint(self,request):
        require(request['pointer']==self.current_capture['pointer'] and request['completion']==self.current_completion,'Current actual checkpoint');return True
    def checkpoint_storage_boundary(self,request):return self.config['ROOT_storage_boundary_receipt']
    def serial_DEV(self,request):return self.DEV.serial_DEV(request)
    def verify_DEV(self,request):return self.DEV.verify_DEV(request)
    def preserve_winner(self,request):
        if not request['replace']:return request['old_winner']['artifact']
        require(request['score']['strict_all_six_passed'] and self.gpu is None and self.current_ack is not None,
                'Actual current fresh recovery and official strict qualification before winner protection')
        self.verify_winner({'winner':request['candidate_pointer'],'DEV':request['DEV']})
        with self.parent.metadata_phase('read_index'):
            proof=self._owner_proof(extra={'qualification':request['DEV'],'strict_all_six_qualified':True,
                'qualified_pointer':self.current_capture['pointer']})
            self.store.protect_qualified_winner(self.current_intent['id'],proof,self.verify_current_owner)
            self._retire_witness(proof,{'kind':'actual_qualified_winner_protected','checkpoint_id':self.current_intent['id']})
        return request['candidate_pointer']

    def _verified_current_merged_CPU_task(self):
        if self.config.get('prior_prepared60_UTF8_failure') is not None and self.current_CPU0==read_bound(self.config['prior_prepared60_UTF8_failure'],self.parent.check)['actual_prepared_CPU_task0']:
            require(self.merged_CPU_inflight is False,'No current CPU task in flight during carried validation')
            return checked_prepared60_task_carryforward(self.config,self.current_packet,guard=self.parent.check)
        from run_ASR18_BPE9223_CPU_window_v11 import checked_task_completion
        task=read_bound(self.current_CPU0,self.parent.check)
        require(task['kind']=='ASR18_completed_merged_CPU_task_no_inflight_v1' and self.merged_CPU_inflight is False,
            'Typed returned LIVE-parent CPU task, not aggregate process0')
        checked_task_completion(self.current_CPU0,self.current_CPU_report,task['configuration'],
            self.config['operational_compatibility'],guard=self.parent.check)
        require(task['producer']==self.producer and task['source']==self.config['CPU_controller_source']
            and task['owner_binding']==self.current_packet['feature_owner'] and task['manifest']==self.current_packet['feature_manifest'],
            'Actual same-owner/window task and exact features at archive boundary')
        return task

    def _merged_CPU_producer_table(self):
        task=self._verified_current_merged_CPU_task();report=read_bound(self.current_CPU_report,self.parent.check)
        normal=[]
        for role,key in (('reference','reference_actual_aggregate0'),('frontend','frontend_actual_aggregate0')):
            item=report[key];proof=read_bound(item['actual0'],self.parent.check)
            verify(proof,kind='ASR18_owned_CPU_process_completion_v1',owner='/root/cpu_data_preparation',report=item['report'],guard=self.parent.check)
            normal.append({'role':role,'actual_process0':item['actual0'],'report':item['report'],'actual_exit_code':0})
        verify(read_bound(self.current_cleanup0,self.parent.check),kind='ASR18_owned_CPU_process_completion_v1',owner=OWNER,
            report=self.current_cleanup,guard=self.parent.check)
        normal.append({'role':'external_feature_cleanup','actual_process0':self.current_cleanup0,'report':self.current_cleanup,'actual_exit_code':0})
        return self._new(Path(self.current_workspace['workspace'])/'merged_CPU_producer_completion_scopes.json',{
            'kind':'ASR18_explicit_numerical_leaf0_and_LIVE_merged_task_completion_scopes_v1','source':record(__file__),
            'configuration':self.config_pin,'actual_numerical_reference_frontend_and_cleanup_process0':normal,
            'completed_merged_CPU_task':self.current_CPU0,'completed_merged_CPU_task_report':self.current_CPU_report,
            'merged_parent_producer':task['producer'],'current_supervisor_producer':self.producer,'merged_parent_process_exit0_claimed':False,
            'legacy_all_CPU_auxiliary_producers_actually_closed0_scope':'actual_numerical_reference_frontend_and_external_cleanup_only',
            'metadata_task_completed_no_CPU_inflight':True,'every_preserved_control_attempt_closed0_claimed':False,
            'preserved_prior_preparation_failure':None})





    def _verify_closed(self,pin):
        proof=read_bound(pin,self.parent.check)
        verify(proof,kind=proof['kind'],owner=OWNER,guard=self.parent.check);return proof

    def _read_state(self):
        with self.store.locked():return self.store._load(allow_preparing_auxiliary=True)[0]

    def _current_capture_writer_source(self):
        return self.config['worker_driver_source']

    def _owner_proof(self,*,extra=None):
        self.check();require(self.current_release is not None,'Actual returned checkpoint writer/read/restore contexts')
        state=self._read_state();release=read_bound(self.current_release,self.parent.check)
        require(release['worker_source']==self._current_capture_writer_source()
                and release['request_nonce']==self.current_response['nonce'] and release['writer_intent']==self.current_intent
                and release['checkpoint_pointer']==self.current_capture['pointer']
                and all(release[k] is True for k in ('writer_context_returned','persistence_reader_context_returned','startup_restore_context_returned')),
                'Exact candidate checkpoint contexts returned before publication/archive')
        if self.gpu is not None:
            self.gpu.check();require(release['producer']==self.gpu.producer,'Same actual live native writer')
        else:
            require(self.current_ack is not None,'Closed-state mutation requires current exact recovery')
            if self.pending40_finalization:self._verify_pending40_quiescence()
            self._verify_closed(self.closed_native);self._verify_closed(self.closed_recovery)
        latest=None if state['latest'] is None else self.store._entry(state['latest'],verify_files=False)['pointer']
        value={'kind':'ASR18_owned_checkpoint_mapping_release_quiescence_v1','owner':OWNER,'ownership':self.store.marker_pin,
            'state_revision':state['revision'],'catalog':state['catalog'],'coherent_latest_pointer':latest,
            'protected_owned_winners':state['winners'],'nonce':secrets.token_hex(32),
            'native_work_quiescent':True,'checkpoint_mappings_released':True,'candidate_payload_handles_released':True,
            'checkpoint_write_inflight':False,'restore_in_progress':False,'native_release_ack':self.current_release,
            'actual_current_checkpoint_pointer':self.current_capture['pointer'],'actual_owner':self.producer,
            'configuration':self.config_pin,'resident_native_boundary':self.current_idle if self.gpu is not None else None,
            'closed_native':self.closed_native if self.gpu is None else None,
            'current_exact_recovery':(self.current_recovery_report if self.pending40_finalization else self.current_ack) if self.gpu is None else None,
            'publication_cadence_evidence':self.current_cadence}
        if self.gpu is not None:
            journal=read_bound(self.current_capture['pointer'],self.parent.check)['journal_prefix']
            with Path(journal['path']).open('rb') as stream:creation=self.store._creation_identity(stream)
            value['incremental_authoritative_journal']={'kind':'ASR18_current_sole_append_journal_writer_identity_v1',
                'writer_source':self.config['journal_writer_source'],'path':journal['path'],'creation_identity':creation,
                'sole_current_writer':True,'append_operation':'open_ab_write_flush_fsync_only',
                'prior_bytes_never_rewritten_or_truncated':True,'writer_epoch':{
                    'kind':'ASR18_verified_continuous_append_writer_epoch_v1','writer_source':self.config['journal_writer_source'],
                    'producer_PID':self.gpu.producer['pid'],'producer_create_time':self.gpu.producer['create_time'],
                    'session_owner_nonce':self.current_worker_nonce,'configuration':self.current_worker_config}}
        if extra:value.update(extra)
        self.proof_sequence+=1;path=self.control/(self.current_intent['id']+'.proof.'+str(self.proof_sequence)+'.json')
        self.current_owner_proof=self._new(path,value);return self.current_owner_proof

    def verify_current_owner(self,request):
        self.check();require(request['proof']==self.current_owner_proof and request['ownership']==self.store.marker_pin,
            'Exact current source-bound owner operation witness')
        proof=read_bound(request['proof'],self.parent.check);release=read_bound(proof['native_release_ack'],self.parent.check)
        require(proof['actual_owner']==self.producer and proof['configuration']==self.config_pin
            and proof['actual_current_checkpoint_pointer']==self.current_capture['pointer'] and release['checkpoint_pointer']==self.current_capture['pointer']
            and release['request_nonce']==self.current_response['nonce'],'No stale/foreign current release witness')
        pointer=read_bound(self.current_capture['pointer'],self.parent.check)
        if proof.get('merged_CPU_live_task_boundary') is True:
            table=read_bound(proof['producer_completion_scopes'],self.parent.check)
            require(table['kind']=='ASR18_explicit_numerical_leaf0_and_LIVE_merged_task_completion_scopes_v1'
                and table['source']==record(__file__) and table['configuration']==self.config_pin
                and table['completed_merged_CPU_task']==self.current_CPU0
                and table['merged_parent_process_exit0_claimed'] is False
                and table['legacy_all_CPU_auxiliary_producers_actually_closed0_scope']=='actual_numerical_reference_frontend_and_external_cleanup_only'
                and len(table['actual_numerical_reference_frontend_and_cleanup_process0'])==3
                and all(row['actual_exit_code']==0 for row in table['actual_numerical_reference_frontend_and_cleanup_process0']),
                'Exact true numerical-leaf0 scopes; merged supervisor is still LIVE')
            self._verified_current_merged_CPU_task()
            for row in table['actual_numerical_reference_frontend_and_cleanup_process0']:
                value=read_bound(row['actual_process0'],self.parent.check)
                verify(value,kind='ASR18_owned_CPU_process_completion_v1',owner=value['owner'],report=row['report'],guard=self.parent.check)
        if self.pending40_finalization and 'producer_completion_scopes' in proof:
            table=read_bound(proof['producer_completion_scopes'],self.parent.check)
            require(table['every_preserved_control_attempt_closed0_claimed'] is False
                and table['configuration']==self.config_pin and len(table['active_numerical_and_successful_window_producers'])==3
                and all(row['actual_exit_code']==0 for row in table['active_numerical_and_successful_window_producers'])
                and all(row['actual_exit_code']==1 and row['normal_process0_claimed'] is False for row in table['preserved_failed_metadata_controls']),
                'Exact preverified numerical0/failed-control1 producer scope table')
        if self.gpu is not None:
            require(release['producer']==self.gpu.producer and self.gpu.command_state=='coherent_idle_CPU_preparation',
                    'Actual continuous sole native writer is coherent idle')
            verify_resident_idle(read_bound(self.current_idle,self.parent.check),pointer=pointer,phase=self.current_packet['phase'],
                proof_binding=self.current_idle,guard=self.parent.check)
            witness=proof['incremental_authoritative_journal'];epoch=witness['writer_epoch']
            require(witness['path']==pointer['journal_prefix']['path'] and epoch['configuration']==self.current_worker_config
                and epoch['session_owner_nonce']==self.current_worker_nonce and epoch['producer_PID']==self.gpu.producer['pid']
                and epoch['producer_create_time']==self.gpu.producer['create_time']
                and witness['writer_source']==record(Path(__file__).with_name('bpe_bounded_production_journal_v4.py')),
                'Exact source-bound live append writer, not mere saved metadata')
        else:
            require(proof['current_exact_recovery']==(self.current_recovery_report if self.pending40_finalization else self.current_ack) and proof['closed_native']==self.closed_native,
                    'No live native writer during current-recovered qualification')
            self._verify_closed(self.closed_native);self._verify_closed(self.closed_recovery)
            if self.pending40_finalization:self._verify_pending40_quiescence()
        return True

    def _wait_ready(self,path):
        began=time.monotonic()
        while not path.exists():
            self.gpu.check();require(time.monotonic()-began<self.config['maximum_GPU_child_wall_seconds'],'Actual worker ready timeout');time.sleep(.1)
        return read_json(path)

    def _admit_CPU_family(self,output):
        require(self.parent.phase=='merged_CPU' and self.parent.merged_CPU_active is True
            and self.parent.current_cap==128000000 and self.config['CPU_owned_allowance_bytes']==1867108864,
            'Reviewed merged supervisor/CPU128M domain; independent parent lifetime384M remains guarded')
        admissions=[]
        for _ in range(3):
            own=self.parent.check()['worker_private_memory']['private_bytes']
            require(own<=128000000,'Actual current merged supervisor/orchestrator stays inside128M')
            if self.gpu is None:
                remaining=max(128000000-own,0)
                context={'GPU_absent':True,'actual_parent_private_bytes':own,'remaining_family_bytes':remaining,
                    'merged_parent_private_cap_bytes':128000000,'additional_GPU_shim_allowance_bytes':0}
            else:
                self.gpu.check()
                require(self.gpu.command_state=='coherent_idle_CPU_preparation','No native forward/Adam during CPU preparation')
                context=self.gpu._family_guard()
                require(time.monotonic()-context['observed_monotonic']<=1,'Exact fresh available resident-family context')
                remaining=context['remaining_parent_and_allshim_allowance_bytes']
            admission=self.fresh_admit(self.config['CPU_owned_allowance_bytes']+remaining)
            require(admission['passed'],'Fresh wholeCPU plus unused parent/allGPUshims does not fit; no child launch')
            admissions.append({'admission':admission,'exact_family_context':context});time.sleep(.1)
        self._new(output,{'kind':'ASR18_actual_pre_CPU_full_family_admission_v1','passed':True,
            'configuration':self.config_pin,'source':record(__file__),'CPU_child_subtree_allowance_bytes':1867108864,
            'merged_parent_private_cap_bytes':128000000,'CPU_whole_allowance_bytes':1995108864,
            'separate_CPU_orchestrator_process_exists':False,
            'resident_actual_memory_already_in_global_baseline':True,'samples':admissions})

    def _verify_prior_prefix_before_reservation(self):
        self._refresh_boundary();boundary=deepcopy(self.boundary_state)
        require(boundary['durable_head']==self.durable.head_pin and boundary['coherent_checkpoint_pointer']==self.current_capture['pointer'],
                'Same actual current durable/capture before reservation')
        io=_Metadata(self.identity,self.resolve_member,self.parent.check,self.config['operational_compatibility']);journal=_JournalPrefixes(self.parent.check)
        try:
            prior_plan=io.read(self.current_original_plan)
            pointer,saved=_capture(self.current_capture,self.current_capture['pointer'],io,
                identity_sha=digest(self.identity),previous_sampler=prior_plan['initial_sampler_sha256'],
                previous_updates=prior_plan['initial_committed_updates'],journal=journal)
            require(digest(saved)==digest(self.schedule.state_dict())==boundary['sampler_state_sha256'],
                    'Independent prior PRE/persistence/sampler/journal while store has NO preparing workspace')
        finally:journal.close()
        return {'boundary':boundary,'pointer':pointer,'sampler_sha256':digest(saved),'phase':prior_plan['phase']}


    def _bridge_plan(self,*args,**kwargs):
        import fastmss_BPE9223_plan_bridge_v9 as bridge
        require(record(bridge.__file__)==self.config['CPU_bridge_source'],'Exact operational CPU bridge')
        return bridge.bridge_plan(*args,**kwargs)

    def _prepare_metadata(self,plan,index):
        require(not self.pending40_initial,'No new prepare before actual fresh40/publication/ACK')
        if getattr(self,'prepared_CPU_resume',None) is not None:
            prepared=self.prepared_CPU_resume
            if 'carried_packet' in prepared:
                require(self.prepared60_initial and index==3 and self.gpu is None and self.current_ack is not None
                    and self.schedule.committed_updates()==60 and digest(plan)==digest(read_bound(prepared['plan_pin'],self.parent.check))
                    and self._read_state()['pending_publication']==prepared['intent'],'Exact completed unconsumed p4; no reserve/CPU rerun')
                self.current_intent=deepcopy(prepared['intent']);self.prepared_CPU_resume=None
                return prepared
            require(index==1 and self.gpu is None and self.current_ack is not None
                and digest(plan)==digest(read_bound(prepared['plan_pin'],self.parent.check))
                and self._read_state()['pending_publication']==prepared['intent'],
                'Exact parsed original unconsumed p2 after actual acknowledged resource recovery')
            self.current_intent=deepcopy(prepared['intent'])
            self.prepared_CPU_resume=None
            return prepared
        if getattr(self,'reserved_preparation',None) is None:
            return self._prepare_new_metadata(plan,index)
        if self.reserved_committed60:
            return self._prepare_reserved_committed60_window(plan,index)
        if self.ACKed40_initial:
            return self._prepare_reserved_reader_window(plan,index)
        require(not self.resource_initial_pending and self.current_ack is not None
            and self.gpu is None and self.closed_native==self.resource_closure,
            'Fresh exact p1 recovery and resource ACK BEFORE reserved p2 continuation')
        reserved=self.reserved_preparation;intent=reserved['intent'];workspace=reserved['workspace']
        plan_pin=reserved['plan'];root=Path(workspace['workspace']);cpu=root/'CPU'
        require(index==1 and intent['id']=='p000000002' and workspace['id']==intent['id']
            and digest(plan)==digest(read_bound(plan_pin,self.parent.check))
            and plan['initial_sampler_sha256']==digest(self.schedule.state_dict())
            and self._read_state()['pending_publication']==intent,
            'Recomputed next source20 plan equals EXACT original unconsumed p2 reservation')
        self._refresh_boundary()
        self.prior_boundary=self._new(root/'operational_prior_boundary_000002.json',self.boundary_state)
        bridge=self._bridge_plan(plan_pin,DATA_evidence=self.config['planning_arguments']['DATA_evidence'],
            session_binding=self.config['session'],window_index=index,prior_window_boundary=self.prior_boundary,
            verify_prior_boundary=self.verify_prior_boundary,guard=self.parent.check)
        bp=self._new(root/'operational_bridge_000002.json',bridge)
        self._new(root/'operational_bridge_binding_000002.json',bp)
        from asr18_BPE9223_CPU_window_configuration_v8 import derive_contract,derive_configuration
        template=self.config['CPU_template_configuration']
        sc=self._new(root/'operational_CPU_source_contract_000002.json',derive_contract(template,bp,self.parent.check))
        proposal=self._new(root/'operational_CPU_window_configuration_000002.json',
            derive_configuration(template,sc,index,intent['id'],str(cpu/'payload'),self.parent.check))
        self.current_intent=deepcopy(intent)
        self.reserved_preparation=None
        return {'phase':plan['phase'],'plan_pin':plan_pin,'window_index':index,'intent':intent,
            'workspace':workspace,'bridge_pin':bp,'CPU_contract':sc,'CPU_folder':str(cpu),
            'arguments':['--config',proposal['path'],'--ROOT-review',self.config['ROOT_CPU_execution_review']['path'],
                         '--folder',str(cpu/'payload')]}


    def _prepare_reserved_committed60_window(self,plan,index):
        reserved=self.reserved_preparation;intent=reserved['intent'];workspace=reserved['workspace']
        root=Path(workspace['workspace']);cpu=root/'CPU';plan_pin=reserved['plan']
        require(not self.committed60_initial and self.reserved_committed60 and self.gpu is None
            and self.closed_native==self.closed_recovery and self.current_ack is not None
            and self.schedule.committed_updates()==60 and index==3 and intent['id']==workspace['id']=='p000000004'
            and digest(plan)==digest(read_bound(plan_pin,self.parent.check))
            and plan['initial_sampler_sha256']==digest(self.schedule.state_dict())
            and self._read_state()['pending_publication']==intent and not cpu.exists()
            and all(record(pin['path'])==pin for pin in reserved['old_prepared_metadata'].values()),
            'Preserve exact actual60/unconsumed prepared p4; no new reservation, old metadata overwrite or CPU draw')
        self._refresh_boundary();suffix='merged_CPU_owner_'+str(self.actual_owner_invocation_count).zfill(6)
        self.prior_boundary=self._new(root/(suffix+'_prior_boundary.json'),self.boundary_state)
        bridge=self._bridge_plan(plan_pin,DATA_evidence=self.config['planning_arguments']['DATA_evidence'],
            session_binding=self.config['session'],window_index=index,prior_window_boundary=self.prior_boundary,
            verify_prior_boundary=self.verify_prior_boundary,guard=self.parent.check)
        bp=self._new(root/(suffix+'_bridge.json'),bridge);self._new(root/(suffix+'_bridge_binding.json'),bp)
        from asr18_BPE9223_CPU_window_configuration_v8 import derive_contract,derive_configuration
        sc=self._new(root/(suffix+'_CPU_source_contract.json'),derive_contract(self.config['CPU_template_configuration'],bp,self.parent.check))
        proposal=self._new(root/(suffix+'_CPU_window_configuration.json'),derive_configuration(
            self.config['CPU_template_configuration'],sc,index,intent['id'],str(cpu/'payload'),self.parent.check))
        self.current_intent=deepcopy(intent);self.reserved_preparation=None;self.reserved_committed60=False;self.ACKed60_initial=False
        return {'phase':plan['phase'],'plan_pin':plan_pin,'window_index':index,'intent':intent,'workspace':workspace,
            'bridge_pin':bp,'CPU_contract':sc,'CPU_folder':str(cpu),'CPU_proposal':proposal,
            'arguments':['--config',proposal['path'],'--ROOT-review',self.config['ROOT_CPU_execution_review']['path'],'--folder',str(cpu/'payload')]}


    def _prepare_reserved_reader_window(self,plan,index):
        reserved=self.reserved_preparation;intent=reserved['intent'];workspace=reserved['workspace']
        plan_pin=reserved['plan'];root=Path(workspace['workspace']);cpu=root/'CPU'
        require(index==2 and self.schedule.committed_updates()==40 and self.gpu is None
            and self.current_ack is not None and intent['id']==workspace['id']=='p000000003'
            and digest(plan)==digest(read_bound(plan_pin,self.parent.check))
            and plan['initial_sampler_sha256']==digest(self.schedule.state_dict())
            and self._read_state()['pending_publication']==intent and not cpu.exists(),
            'Exact original reserved p3/source40 plan, no new reserve/crop/replay or CPU overwrite')
        self._refresh_boundary();self.prior_boundary=reserved['prior_window_boundary']
        require(read_bound(self.prior_boundary,self.parent.check)==self.boundary_state,
            'Original failed-attempt p3 prior boundary exactly equals actual ACK40')
        bridge=self._bridge_plan(plan_pin,DATA_evidence=self.config['planning_arguments']['DATA_evidence'],
            session_binding=self.config['session'],window_index=index,prior_window_boundary=self.prior_boundary,
            verify_prior_boundary=self.verify_prior_boundary,guard=self.parent.check)
        suffix='reader_authority_owner_'+str(self.actual_owner_invocation_count).zfill(6)
        bp=self._new(root/(suffix+'_bridge.json'),bridge)
        self._new(root/(suffix+'_bridge_binding.json'),bp)
        from asr18_BPE9223_CPU_window_configuration_v8 import derive_contract,derive_configuration
        template=self.config['CPU_template_configuration']
        sc=self._new(root/(suffix+'_CPU_source_contract.json'),derive_contract(template,bp,self.parent.check))
        proposal=self._new(root/(suffix+'_CPU_window_configuration.json'),
            derive_configuration(template,sc,index,intent['id'],str(cpu/'payload'),self.parent.check))
        self.current_intent=deepcopy(intent);self.reserved_preparation=None;self.ACKed40_initial=False
        return {'phase':plan['phase'],'plan_pin':plan_pin,'window_index':index,'intent':intent,
            'workspace':workspace,'bridge_pin':bp,'CPU_contract':sc,'CPU_folder':str(cpu),
            'arguments':['--config',proposal['path'],'--ROOT-review',self.config['ROOT_CPU_execution_review']['path'],
                         '--folder',str(cpu/'payload')]}

    def _prepare_new_metadata(self,plan,index):
        self.check()
        verified=self._verify_prior_prefix_before_reservation() if index>0 else None
        intent=self.store.reserve_publication(publication_role='completed_window_end');self.current_intent=intent
        pp=self.control/(intent['id']+'.plan.json');require(not pp.exists(),'One fresh exact owned control plan')
        write_json(pp,plan);plan_pin=record(pp)
        workspace=self.store.reserve_auxiliary_workspace(intent,plan_pin);root=Path(workspace['workspace'])
        copied=root/'coordinator_plan_copy.json';shutil.copyfile(pp,copied)
        require(record(pp)['sha256']==record(copied)['sha256'] and pp.stat().st_size==copied.stat().st_size,'Exact complete source-owned plan bytecopy')
        write_json(root/'plan_alias.json',{'kind':'ASR18_exact_owned_control_plan_archive_alias_v1','publication_id':intent['id'],'original':plan_pin,'copied':record(copied)})
        if index>0:
            self._refresh_boundary()
            self.prior_boundary=self._new(root/'prior_window_boundary.json',self.boundary_state)
        bridge=self._bridge_plan(plan_pin,DATA_evidence=self.config['planning_arguments']['DATA_evidence'],session_binding=self.config['session'],
            window_index=index,prior_window_boundary=self.prior_boundary,verify_prior_boundary=self.verify_prior_boundary,guard=self.parent.check)
        bp=root/'bridge.json';write_json(bp,bridge);ba=root/'bridge_binding.json';write_json(ba,record(bp))
        from asr18_BPE9223_CPU_window_configuration_v8 import derive_contract,derive_configuration
        cpu=root/'CPU';template=self.config['CPU_template_configuration']
        sc=self._new(root/'CPU_source_contract.json',derive_contract(template,record(bp),self.parent.check))
        proposal=self._new(root/'CPU_window_configuration.json',derive_configuration(template,sc,index,intent['id'],str(cpu/'payload'),self.parent.check))
        arguments=['--config',proposal['path'],'--ROOT-review',self.config['ROOT_CPU_execution_review']['path'],'--folder',str(cpu/'payload')]
        if index>0:
            boundary=read_bound(self.prior_boundary,self.parent.check)
            self.verify_prior_boundary(boundary)
            require(boundary==verified['boundary'] and self.durable.head_pin==boundary['durable_head']
                and digest(self.schedule.state_dict())==verified['sampler_sha256'],
                'After new bridge, recheck SAME previously verified old head/capture/sampler')
            pointer=read_bound(self.current_capture['pointer'],self.parent.check)
            require(pointer==verified['pointer'],'No prior pointer mutation between archive verification and CPU handoff')
            if self.gpu is not None:
                self.gpu.check()
                verify_resident_idle(read_bound(self.current_idle,self.parent.check),pointer=pointer,
                    proof_binding=self.current_idle,phase=verified['phase'],
                    guard=self.parent.check)
            else:
                require(self.current_ack is not None,'Closed native handoff requires actual current recovery ACK')
                self._verify_closed(self.closed_native);self._verify_closed(self.closed_recovery)
            attestation={'kind':'FastMSS_native_owner_prior_window_CPU_handoff_v1','owner':OWNER,'passed':True,
                'boundary_sha256':digest(boundary),'actual_all_state_prefix_checkpoint_verified':True,
                'bridge':record(bp),'session':self.config['session'],'parent_source':self.parent_source,
                'durable_head':self.durable.head_pin,'current_durable_owner_head_and_persistence_prefix_verified':True,
                'current_checkpoint_capture':self.current_capture,'source':record(__file__),
                'saved_sampler_sha256':verified['sampler_sha256'],'historical_fresh_GPU_restore_claimed':False}
            ap=root/'prior_boundary_verification.json';self._new(ap,attestation)
            ab=root/'prior_boundary_verification_binding.json';self._new(ab,record(ap))
            # Exact bridge already binds this freshly reverified prior boundary;
            # generic CPU3 reconstructs it independently through the context callback.
            # Retain the actual handoff attestation as archive metadata, not a fake CLI proof.
        # No parsed bridge, prior verification dictionaries or decoded pointer
        # escapes this helper frame. Only original immutable pins/scalars do.
        return {'phase':plan['phase'],'plan_pin':plan_pin,'window_index':index,
            'intent':intent,'workspace':workspace,'bridge_pin':record(bp),
            'CPU_contract':sc,'CPU_folder':str(cpu),'arguments':arguments}

    def _execute_prepared_CPU(self,prepared):
        if 'carried_packet' in prepared:
            packet=deepcopy(prepared['carried_packet'])
            checked_prepared60_task_carryforward(self.config,packet,guard=self.parent.check)
            require(self.gpu is None and self.merged_CPU_inflight is False and self.schedule.committed_updates()==60,
                'No task/audio/GPU/draw while reopening prepared p4')
            self.current_packet=packet;self.current_CPU0=packet['CPU_task_completion'];self.current_CPU_report=packet['CPU_task_report']
            self.current_bridge=prepared['bridge_pin'];self.current_workspace=prepared['workspace'];self.current_original_plan=prepared['plan_pin']
            self.prepared60_initial=False;self.ACKed60_initial=False;self.reserved_committed60=False
            return packet
        # Full collection changes metadata lifetime only. The SAME actual64M
        # entry and live CPU gates remain authoritative; no release is assumed.
        before=self.parent.check()['worker_private_memory']['private_bytes']
        collected=gc.collect()
        after=self.parent.check()['worker_private_memory']['private_bytes']
        root=Path(prepared['workspace']['workspace']);cpu=Path(prepared['CPU_folder'])
        self._new(root/('metadata_builder_frame_release_owner_'+str(self.actual_owner_invocation_count).zfill(6)+'.json'),{
            'kind':'ASR18_actual_metadata_builder_frame_release_v1','source':record(__file__),
            'parsed_bridge_verified_boundary_pointer_frames_returned':True,
            'prior_run_plan_completion_response_packet_locals_released':True,
            'full_collection_generation':2,'objects_collected':collected,
            'actual_before_private_bytes':before,'actual_after_private_bytes':after,
            'CPU_current_cap_bytes':128000000,'separate_CPU_orchestrator_process_eliminated':True,'release_or_resource_credit_assumed':False})
        from asr18_BPE9223_parent_SQLite_cache_release_v1 import release_owned_readonly_caches
        compatibility=checked_compatibility(self.config['operational_compatibility'],self.identity,self.parent.check)
        require(compatibility['operational_sources']['parent_SQLite_cache_release']==record(Path(__file__).with_name('asr18_BPE9223_parent_SQLite_cache_release_v1.py')),
            'Exact reviewed metadata page-cache release source')
        before_sampler=digest(self.schedule.state_dict())
        released=release_owned_readonly_caches(self.context,guard=self.parent.check,
            measure=lambda:self.parent.check()['worker_private_memory']['private_bytes'])
        require(digest(self.schedule.state_dict())==before_sampler,'Cache release never changes sampler/RNG/counters')
        released.update(sampler_state_sha256=before_sampler)
        self._new(root/('SQLite_cache_release_owner_'+str(self.actual_owner_invocation_count).zfill(6)+'.json'),released)
        del released
        gc.collect()
        del compatibility
        with self.parent.CPU_phase():
            self._admit_CPU_family(root/('CPU_remaining_family_admission_owner_'+str(self.actual_owner_invocation_count).zfill(6)+'.json'))
            require(not cpu.exists(),'Fresh exact p4 CPU task metadata, no old output adoption');cpu.mkdir()
            proposal=prepared.get('CPU_proposal')
            if proposal is None:
                arguments=prepared['arguments'];proposal=record(arguments[arguments.index('--config')+1])
            from run_ASR18_BPE9223_CPU_window_v11 import run_merged_task,checked_task_completion
            require(record(__import__('inspect').getsourcefile(run_merged_task))==self.config['CPU_controller_source'],
                'Actual in-process merged task code, not legacy external-controller0')
            self.merged_CPU_inflight=True
            try:
                proof,report_pin=run_merged_task(proposal['path'],self.config['ROOT_CPU_execution_review']['path'],
                    str(cpu/'payload'),self.config['operational_compatibility']['path'],
                    parent_guard=self.parent,actual_parent_source=self.parent_source)
            finally:self.merged_CPU_inflight=False
            checked_task_completion(proof,report_pin,proposal,self.config['operational_compatibility'],guard=self.parent.check)
            report=read_bound(report_pin,self.parent.check);self._verify_aggregate_CPU_report(report,prepared['CPU_contract'])
        packet={'window_index':prepared['window_index'],'phase':prepared['phase'],
            'coordinator_plan':prepared['plan_pin'],'feature_owner':report['owner'],'feature_manifest':report['manifest'],
            'session':self.config['session'],'lifecycle_parent_head':self.durable.head_pin,
            'checkpoint_intent':prepared['intent'],'auxiliary_workspace':prepared['workspace'],
            'CPU_task_completion':proof,'CPU_task_report':report_pin}
        self.current_packet=packet;self.current_CPU0=proof;self.current_CPU_report=report_pin
        self.current_bridge=prepared['bridge_pin'];self.current_workspace=prepared['workspace']
        self.current_original_plan=prepared['plan_pin']
        return packet

    def _seal_fresh_CPU_task_handoff(self,packet):
        from run_ASR18_BPE9223_CPU_window_v11 import checked_task_completion
        task=read_bound(packet['CPU_task_completion'],self.parent.check)
        carried=(self.config.get('prior_prepared60_UTF8_failure') is not None and packet['CPU_task_completion']==read_bound(self.config['prior_prepared60_UTF8_failure'],self.parent.check)['actual_prepared_CPU_task0'])
        if carried:checked_prepared60_task_carryforward(self.config,packet,guard=self.parent.check)
        else:
            checked_task_completion(packet['CPU_task_completion'],packet['CPU_task_report'],task['configuration'],
                self.config['operational_compatibility'],guard=self.parent.check)
        require(self.gpu is not None and self.merged_CPU_inflight is False and self.parent.merged_CPU_active is False
            and (carried or task['producer']==self.producer),'Exact carried old task OR same current producer, current parent live/no CPU inflight')
        witness={'kind':'ASR18_fresh_live_parent_no_CPU_inflight_native_task_handoff_v1','source':record(__file__),
            'parent_source':self.parent_source,'producer':self.producer,'task_nonce':task['task_nonce'],
            'prior_prepared60_UTF8_failure':self.config['prior_prepared60_UTF8_failure'] if carried else None,
            'original_task_producer':task['producer'],
            'CPU_task_completion':packet['CPU_task_completion'],'packet_semantic_sha256':digest(packet),
            'actual_CPU_leaf_inflight':False,'task_family_observer_joined':True,
            'optimizer_updates':self.schedule.committed_updates(),'sampler_sha256':digest(self.schedule.state_dict()),
            'observed_monotonic':time.monotonic()}
        packet['CPU_task_handoff_witness']=self._new(Path(self.current_workspace['workspace'])/'fresh_merged_CPU_native_handoff.json',witness)
        return packet

    def _verify_aggregate_CPU_report(self,report,contract_pin=None):
        if self.config.get('prior_prepared60_UTF8_failure') is not None:
            failed=read_bound(self.config['prior_prepared60_UTF8_failure'],self.parent.check)
            if report.get('proposal')==failed['prepared_CPU']['CPU_proposal']:
                require(digest(report)==digest(read_bound(failed['actual_prepared_CPU_report'],self.parent.check)),'Exact original carried report bytes/semantics')
                checked_prepared60_task_carryforward(self.config,failed['prepared_packet'],guard=self.parent.check)
                if contract_pin is not None:require(report['source_contract']==contract_pin,'Exact carried source contract')
                return report
        require(report['passed'] is True and report['real_actual_child0_count']==2
            and report['per_update_child0_count_claimed']==0 and report['actual_optimizer_updates']==0
            and report['worker_source']==self.config['CPU_controller_source']
            and report['parent_source']==self.parent_source
            and report['legacy_leaf_execution_source']==self.CPU_leaf_execution_delegate_source
            and report['operational_compatibility']==self.config['operational_compatibility']
            and report['kind']=='ASR18_actual_BPE9223_recorded_room_merged_CPU_task_window_v1'
            and report['merged_parent_remains_live'] is True and report['aggregate_process_exit0_claimed'] is False,'Actual aggregate controller0 under explicit code6 CPU observer, no source stamp rewrite')
        if contract_pin is not None:require(report['source_contract']==contract_pin,'Exact derived actual source contract')
        contract=read_bound(report['source_contract'],self.parent.check)
        for key,role in (('reference_actual_aggregate0','reference'),('frontend_actual_aggregate0','frontend')):
            item=report[key];proof=read_bound(item['actual0'],self.parent.check)
            verify(proof,kind='ASR18_owned_CPU_process_completion_v1',owner='/root/cpu_data_preparation',
                report=item['report'],guard=self.parent.check)
            require(proof['worker_source']==contract['sources'][role]
                and proof['parent_source']==self.CPU_leaf_execution_delegate_source,'Actual original7 source-bound aggregate leaf0')
        return report

    def verify_prior_boundary(self,boundary):
        self.check();require(self.prior_boundary is not None and boundary==read_bound(self.prior_boundary,self.parent.check)
            and boundary['durable_head']==self.durable.head_pin,'Exact previous owned coherent/cleanup/durable authority')
        return True

    def _start_native(self,packet):
        require(not self.pending40_initial,'No training/source20 restore before exact pending40 completion')
        manifest=read_bound(packet['feature_manifest'],self.parent.check)
        require(self.native_sequence<513,'Bounded actual native restart/branch quota; no unchanged retry')
        self.native_sequence+=1;self.current_worker_nonce=secrets.token_hex(32);self.native_started=time.monotonic()
        worker={**self.config,'resource_journal_handoff':False,'worker_mode':'resident_training','first_feature_update':manifest['updates'][0]['update'],
            'journal_namespace':str(self.root/'journals'),'initial_journal_path':str(self.initial_journal),
            'journal_resume_sequence':self.native_sequence,'session_owner_nonce':self.current_worker_nonce,
            'restore_parent_pointer':None if self.current_capture is None else self.current_capture['pointer'],
            'actual_supervisor_identity':self.producer}
        if self.current_capture is not None:
            require(self.current_ack is not None and self.closed_native is not None,'No restart before exact current recovery')
            worker.update(source_native_process0=self.closed_native,source_checkpoint_capture=self.current_capture)
        path=self.control/('native_configuration_'+str(self.native_sequence).zfill(6)+'.json')
        self.current_worker_config=self._new(path,worker)
        self.gpu=self.GPUService.launch(path,self.folder/('native_'+str(self.native_sequence).zfill(6)),self.parent);self.parent.GPU=self.gpu
        ready=self._wait_ready(self.gpu.folder/'worker_ready.json')
        require(ready['optimizer_updates']==self.schedule.committed_updates(),
                'Original0 or actual current all-state startup restore BEFORE next draw')

    def _cleanup(self,completion,*,native0=None):
        pointer=read_bound(self.current_capture['pointer'],self.parent.check);CPU=read_json(self.current_CPU_report['path'])
        root=Path(self.current_workspace['workspace']);request={'owner':self.current_packet['feature_owner'],
            'closure':CPU['actual_frontend_window_closure'],'manifest':self.current_packet['feature_manifest'],'parent_source':self.CPU_execution_delegate_source,
            'archive_path':str(root/'retained_feature_metadata.json'),'boundary':{
                'kind':'coherent_checkpoint_boundary','owner_sha256':self.current_packet['feature_owner']['sha256'],
                'session':self.config['session'],'phase':self.current_packet['phase'],
                'committed_window_updates':pointer['committed_window_updates'],
                'window_manifest_sha256':self.current_packet['feature_manifest']['sha256'],
                'checkpoint':pointer['checkpoint'],'coherent_checkpoint_pointer':self.current_capture['pointer']}}
        request['boundary']['resident_native_boundary' if native0 is None else 'actual_native_closed']=self.current_idle if native0 is None else native0
        path=root/'cleanup_request.json';write_json(path,request);out=root/'cleanup'
        with self.parent.CPU_phase():
            self._admit_CPU_family(root/'cleanup_remaining_family_admission.json')
            proof=self.run_CPU(self.config,self.cleanup_script,['--cleanup','--request',str(path),'--folder',str(out)],out,
                guard=self.parent.check if self.gpu is None else self.gpu.check)
        result=read_json(out/'report.json');require(result['passed'] is True,'Actual exact owned featurecleanup0')
        self.current_cleanup0=proof;self.current_cleanup=record(out/'report.json')
        return result

    def _seal_closed_empty_CPU_logs(self,plan):
        if self.pending40_finalization:return self._seal_reconciled_empty_CPU_logs()
        # Only logs opened by the exact actually CLOSED0 CPU producers. Empty
        # content is retained verbatim as SHA(empty)/size0 in immutable receipts.
        import stat
        root=Path(self.current_workspace['workspace'])
        report=read_bound(self.current_CPU_report,self.parent.check)
        contract=read_bound(report['source_contract'],self.parent.check)
        self._verify_aggregate_CPU_report(report)
        require(report['bridge']==self.current_bridge,'Same actual aggregate CPU window before empty-log sealing')
        sources=contract['sources'];allowed=[]
        def add(path,proof_pin,worker,parent):
            proof=read_bound(proof_pin,self.parent.check)
            verify(proof,kind='ASR18_owned_CPU_process_completion_v1',owner=OWNER if parent in (self.parent_source,self.CPU_execution_delegate_source) else '/root/cpu_data_preparation',guard=self.parent.check)
            require(proof['kind']=='ASR18_owned_CPU_process_completion_v1'
                and proof['worker_source']==worker and proof['parent_source']==parent,
                'Exact source-bound CLOSED0 producer of this optional log')
            path=Path(path)
            require(path.is_relative_to(root) and path.name=='worker.log','Only this new registered AUX optional worker.log')
            require(Path(proof_pin['path'])==path.parent/'actual_owned_process0.json',
                'Log and actual producer completion share the exact owned directory')
            for ancestor in (path,*path.parents):
                info=ancestor.lstat()
                require(not (getattr(info,'st_file_attributes',0)&0x400) and not ancestor.is_symlink(),
                    'No reparse/symlink log or ancestor')
                if ancestor==ROOT:break
            info=path.lstat();require(stat.S_ISREG(info.st_mode),'Exact regular owned optional log')
            if info.st_size==0:
                original=record(path)
                require(original['bytes']==0 and original['sha256']==hashlib.sha256(b'').hexdigest(),
                    'Only genuinely empty bytes, never discard output')
                allowed.append({'original':original,'actual_process0':proof_pin,
                    'producer':proof['producer'],'worker_source':worker,'parent_source':parent})
        self._verified_current_merged_CPU_task()
        require(not (root/'CPU/worker.log').exists(),'Merged task has no external controller worker.log or process0')
        add(root/'cleanup/worker.log',self.current_cleanup0,self.parent_source,self.CPU_execution_delegate_source)
        for kind,key in (('reference','reference_actual_aggregate0'),('frontend','frontend_actual_aggregate0')):
            item=report[key]
            add(root/'CPU/payload'/(kind+'_process')/'worker.log',item['actual0'],
                sources[kind],self.CPU_leaf_execution_delegate_source)
        require(len(allowed)<=4,'Two actual aggregate leaf logs plus controller/cleanup only')
        require(self.gpu is not None,'Empty-log sealing only at actual resident idle, no fabricated closed boundary')
        self.gpu.check();pointer=read_bound(self.current_capture['pointer'],self.parent.check)
        verify_resident_idle(read_bound(self.current_idle,self.parent.check),pointer=pointer,
            proof_binding=self.current_idle,phase=plan['phase'],guard=self.parent.check)
        release=read_bound(self.current_release,self.parent.check)
        require(release['producer']==self.gpu.producer and release['worker_source']==self.config['worker_driver_source']
            and release['request_nonce']==self.current_response['nonce'] and release['writer_intent']==self.current_intent
            and release['checkpoint_pointer']==self.current_capture['pointer']
            and all(release[k] is True for k in ('writer_context_returned','persistence_reader_context_returned',
                                                'startup_restore_context_returned')),
            'Actual native idle and returned candidate contexts before optional empty-log unlink')
        intent=self._new(root/'empty_CPU_log_retirement_intent.json',{
            'kind':'ASR18_exact_closed_empty_CPU_log_retirement_intent_v1','owner':OWNER,
            'source':record(__file__),'configuration':self.config_pin,'workspace':self.current_workspace,
            'native_idle':self.current_idle,'native_release_ack':self.current_release,
            'empty_logs':allowed,'nonempty_log_content_retirement_permitted':False})
        for row in allowed:
            path=Path(row['original']['path']);info=path.lstat()
            require(stat.S_ISREG(info.st_mode) and not (getattr(info,'st_file_attributes',0)&0x400)
                and record(path)==row['original'] and info.st_size==0,'Same exact regular empty log immediately before unlink')
            require(record(intent['path'])==intent,'Durable complete empty-byte/source/closure receipt before unlink')
            path.unlink();require(not path.exists(),'Exact new optional empty log retired')
        return self._new(root/'empty_CPU_log_retirement_completed.json',{
            'kind':'ASR18_exact_closed_empty_CPU_log_retirement_completed_v1','passed':True,
            'intent':intent,'retired_exact_empty_logs':len(allowed),'discarded_content_bytes':0,
            'all_original_empty_path_SHA_and_producer_proofs_retained':True})

    def _archive_and_publish(self,plan,completion):
        pointer=read_bound(self.current_capture['pointer'],self.parent.check)
        metadata=read_bound(pointer['checkpoint'],self.parent.check);sampler=decode_scalar_tree(select(metadata['tree'],'sampler'))
        last=plan['updates'][-1];self.current_cadence={'kind':'ASR18_source_bound_actual_completed_window_cadence_v1',
            'phase':plan['phase'],'committed_window_updates':len(plan['updates']),'exact_pass_tail':bool(last['TR_draws'][-1]['draw_index']+1==
                (last['TR_draws'][-1]['epoch']+1)*(sampler['broad_cursor']['population'] if plan['phase']==WEAK else sampler['cleaner_cursor']['population'])),
            'coordinator_plan':self.current_original_plan}
        workspace=Path(self.current_workspace['workspace']);rows=[]
        self._seal_closed_empty_CPU_logs(plan)
        producer_table=self._pending40_producer_table() if self.pending40_finalization else self._merged_CPU_producer_table()
        with self.store.locked():inventory=self.store._workspace_inventory(self.current_workspace)
        directories=inventory['directories']
        for path in sorted(inventory['files'].values()):
            self.parent.check()
            if path==Path(self.current_workspace['workspace_marker']['path']):continue
            require(path.suffix.lower() in ('.json','.jsonl','.log','.txt','.sha256'),
                    'Reject unexpected audio/tensor/model BEFORE any full file hash')
            role='CPU_provenance'
            if path.name=='actual_pre_save_fingerprint.json':role='pre_save'
            elif path.name=='actual_next_draw_packet.json':role='next_draw'
            elif path.name=='persistence_verified.json':role='persistence'
            elif path.name in ('coordinator_plan_copy.json','plan_alias.json'):role='window_plan'
            elif path.name=='completion.json':role='native_completion'
            rows.append({'role':role,'file':record(path)})
        rows.sort(key=lambda row:Path(row['file']['path']).relative_to(workspace).as_posix())
        manifest={'kind':'ASR18_future_owned_window_auxiliary_manifest_v1','workspace_marker':self.current_workspace['workspace_marker'],
            'coordinator_plan':self.current_original_plan,'run_identity_sha256':digest(self.identity),
            'tensor_payloads_audio_features_and_models_included':False,'files':rows,'directories':sorted(directories),'journal_prefix':pointer['journal_prefix']}
        require(len(rows)<=self.store.policy['maximum_auxiliary_files'] and len(canonical(manifest))<=self.store.policy['maximum_auxiliary_header_bytes'],
            'New plan copy/alias fully counted, no silent archive cap expansion')
        self.schedule.load_state_dict(normalize_schedule_state(sampler))
        extra={'auxiliary_manifest_sha256':digest(manifest),'all_CPU_auxiliary_producers_actually_closed0':True,
            'native_auxiliary_writer_contexts_returned':True,'all_auxiliary_handles_released':True}
        if not self.pending40_finalization:
            extra.update(producer_completion_scopes=producer_table,merged_CPU_live_task_boundary=True,
                legacy_all_CPU_auxiliary_producers_actually_closed0_scope='actual_numerical_reference_frontend_and_external_cleanup_only',
                merged_parent_process_exit0_claimed=False,metadata_task_completed_no_CPU_inflight=True,
                every_preserved_control_attempt_closed0_claimed=False)
        if self.pending40_finalization:
            extra.update(producer_completion_scopes=producer_table,
                legacy_all_CPU_auxiliary_producers_actually_closed0_scope='active_reference_frontend_and_successful_feature_window_payload_producers_only',
                every_preserved_control_attempt_closed0_claimed=False,preserved_failed_metadata_controls_verified_closed_nonzero=True)
        with self.parent.metadata_phase('archive_aux'):
            proof=self._owner_proof(extra=extra)
            archived=self.store.archive_auxiliary(self.current_intent['id'],manifest,proof,self.verify_current_owner)
            self._retire_witness(proof,{'kind':'actual_auxiliary_archive_verified','archive':archived['archive']})
        expectation={'identity':pointer['identity'],'optimizer_updates':pointer['committed_updates'],'sampler_state':sampler,
            'journal_prefix':pointer['journal_prefix'],'run_identity':self.identity,'publication_role':'completed_window_end',
            'publication_cadence_evidence':self.current_cadence}
        with self.parent.metadata_phase('read_index'):
            proof=self._owner_proof();published=self.store.complete_publication(self.current_intent,self.current_capture['pointer'],expectation,proof,self.verify_current_owner)
            self._retire_witness(proof,{'kind':'actual_checkpoint_pair_publication_verified','archive':published['archive']})
            self.durable.record_pending_window(before_state=self.before_window,plan=self.current_original_plan,completion=completion)
        # Exact registered archive + immutable pending acknowledgment precede
        # retiring ONLY this newly created outside-store temporary control plan.
        with self.resolve_member({'kind':'ASR18_indexed_auxiliary_original_member_v1','original':self.current_original_plan,
                'storage_owner':self.store.marker_pin,'auxiliary_id':self.current_intent['id']}) as stream:
            h=hashlib.sha256();count=0
            while True:
                block=stream.read(1024**2)
                if not block:break
                h.update(block);count+=len(block)
        require(count==self.current_original_plan['bytes'] and h.hexdigest()==self.current_original_plan['sha256']
            and record(self.current_original_plan['path'])==self.current_original_plan,'Exact archive and actual owned original before single unlink')
        Path(self.current_original_plan['path']).unlink()
        if not self.pending40_finalization:
            self._refresh_boundary()
            self._retire_obsolete()

    def _recover(self):
        require(self.recovery_sequence<513,'Bounded coherent restart/current-recovery quota')
        require(self.recovery_control_sequence<512,'Bounded cumulative control attempts, never reset')
        self.recovery_control_sequence+=1;name='recovery_'+str(self.recovery_sequence+1).zfill(6)
        controls=self.control/('recovery_'+str(self.recovery_control_sequence).zfill(6));controls.mkdir(exist_ok=False)
        self.closed_native=self.gpu.finish();self.gpu=None;self.parent.GPU=None
        request={'kind':'ASR18_full_run_CURRENT_checkpoint_fresh_recovery_request_v1','identity':self.identity,
            'planning_arguments':self.config['planning_arguments'],'augmentation_identity':self.config['augmentation_identity'],
            'current_native_process0':self.closed_native,'current_checkpoint_capture':self.current_capture,
            'native_worker_source':self.config['worker_driver_source'],'native_parent_source':self.parent_source,
            'recovery_worker_source':self.config['worker_driver_source'],'recovery_parent_source':self.parent_source,
            'operational_compatibility':self.config['operational_compatibility']}
        path=controls/'request.json';self._new(path,request)
        config={**self.config,'resource_journal_handoff':False,'worker_mode':'restore_only','current_recovery_request':record(path),
            'source_native_process0':self.closed_native,'source_checkpoint_capture':self.current_capture,'journal_namespace':str(self.root/'journals'),
            'journal_resume_sequence':self.recovery_sequence+1,'entire_worker_forecast':self.config['final_recovery_worker_forecast']}
        cp=controls/'worker_configuration.json';self._new(cp,config)
        self.gpu=self.GPUService.launch(cp,self.folder/name,self.parent);self.parent.GPU=self.gpu
        self.recovery_sequence+=1  # An actual admitted GPU child now exists; failed control creation alone is not recovery.
        ready=self._wait_ready(self.gpu.folder/'current_recovery_ready.json');self.current_recovery_report=ready['report']
        self.closed_recovery=self.gpu.finish();self.gpu=None;self.parent.GPU=None
        require(read_bound(self.closed_recovery,self.parent.check)['report']==self.current_recovery_report,'SAME canonical current recovery report in actualprocess0')
        evidence={'kind':'ASR18_full_run_current_recovery_prefix_inputs_v1','owner':OWNER,'diagnostic_only':self.config['owner_validation_only'],'run_identity_sha256':digest(self.identity),
            'pending_head':self.durable.head_pin,'current_checkpoint_capture':self.current_capture,'current_native_process0':self.closed_native,
            'actual_recovery_process0':self.closed_recovery,'recovery_report':self.current_recovery_report,
            'operational_compatibility':self.config['operational_compatibility']}
        ep=controls/'evidence.json';self._new(ep,evidence)
        request=self.durable._request(operational_compatibility=self.config['operational_compatibility'],current_recovery_evidence=record(ep),coverage_output=str(self.root/'durable'/('coverage_'+str(self.recovery_sequence).zfill(6)+'.json')))
        with self.parent.metadata_phase('read_index'):
            coverage=create_coverage(request,resolve_member=self.resolve_member,guard=self.parent.check)
            self.current_coverage=coverage;v=read_bound(coverage,self.parent.check)
            ack={'kind':'ASR18_full_run_current_latest_recovery_and_verified_persistence_prefix_v1','diagnostic_only':self.config['owner_validation_only'],
                'run_identity_sha256':digest(self.identity),'pending':deepcopy(self.durable.head['pending']),'pending_head':self.durable.head_pin,
                'final_checkpoint_pointer':self.current_capture['pointer'],'fresh_recovery_scope':'current_latest_only',
                'all_persistence_verified_through_pending_prefix':True,'all_historical_checkpoints_freshly_restored':False,
                'current_latest_all207_threeAdam_all4RNG_sampler_prefix_nextdraw_match_actual_pre_save':True,
                'actual_recovery_native_updates':0,'actual_recovery_draws_consumed':0,'actual_closed_native_index':self.closed_native,
                'actual_closed_recovery_index':self.closed_recovery,'persistence_prefix_coverage_index':coverage,
                'current_latest_exact_restore':v['current_latest_exact_restore'],'current_recovery_evidence':record(ep),
                'operational_compatibility':self.config['operational_compatibility']}
            ap=controls/'acknowledgement.json';self.current_ack=self._new(ap,ack)
            self.durable.acknowledge_recovery(self.current_ack)
        self._refresh_boundary()
        self._archive_obsolete_worker_controls()

    def _refresh_boundary(self):
        if self.current_capture is None:return
        cleanup=read_bound(self.current_cleanup,self.parent.check)
        value={'kind':'FastMSS_actual_coherent_full_run_CPU_window_boundary_v2','session':self.config['session'],
            'completed_window_index':self.current_packet['window_index'],
            'bridge':self.current_bridge,'committed_window_updates':read_bound(self.current_capture['pointer'],self.parent.check)['committed_window_updates'],
            'optimizer_updates':self.schedule.committed_updates(),'sampler_state_sha256':digest(self.schedule.state_dict()),
            'coherent_checkpoint_pointer':self.current_capture['pointer'],'durable_head':self.durable.head_pin,
            'owned_cleanup_actual_closed':self.current_cleanup0,'owned_cleanup_receipt':self.current_cleanup,
            'preserved_metadata':cleanup['preserved_metadata'],'originals_checkpoints_and_journals_untouched':True,
            'owned_feature_window_retired':True}
        if self.gpu is None:
            if self.closed_native==self.resource_closure:
                require(self.current_ack is not None and self.closed_recovery is not None,'Resource boundary needs actual fresh0 and prefix ACK')
                value.update(actual_native_resource_closure=self.resource_closure,
                    actual_fresh_recovery_process0=self.closed_recovery,resource_recovery_report=self.current_recovery_report,
                    resource_recovery_ack=self.current_ack,operational_compatibility=self.config['operational_compatibility'])
            else:value['actual_native_closed']=self.closed_native
        else:value['resident_native_boundary']=self.current_idle
        self.boundary_state=value

    def _retire_obsolete(self):
        # Bounded live B-tree sets only, never a Python full-history inventory.
        with self.parent.metadata_phase('read_index'):
            with self.store.locked():
                state=self.store._load()[0]
                protected=sorted(set(state['winners']+([state['latest']] if state['latest'] else [])))
                db=self.store._db()
                try:
                    live=[row[0] for row in db.execute('SELECT id FROM checkpoints WHERE retired=0 ORDER BY id LIMIT 6')]
                    aux=[row[0] for row in db.execute("SELECT id FROM auxiliary WHERE status='archived' ORDER BY id LIMIT 6")]
                finally:db.close()
                require(len(live)<=5,'Exact bounded B-tree live set; never iterate _Rows')
                for key in live:self.store._entry(key,verify_files=True)
                obsolete=[key for key in live if key not in protected]
            require(len(obsolete)<=5 and len(aux)<=5,'Bounded exact active owned sets')
            oldaux=[key for key in aux if key not in protected]
            if oldaux:
                proof=self._owner_proof(extra={'retire_auxiliary_ids':oldaux,'protected_active_auxiliary_ids':protected,
                    'all_auxiliary_handles_released':True})
                result=self.store.retire_obsolete_auxiliary(oldaux,proof,self.verify_current_owner)
                self._retire_witness(proof,{'kind':'exact_obsolete_auxiliary_retirement_returned','result':result})
            if obsolete:
                proof=self._owner_proof(extra={'retire_checkpoint_ids':obsolete})
                self.store.retire_obsolete_payloads(obsolete,proof,self.verify_current_owner)
                self._retire_witness(proof,{'kind':'exact_obsolete_payload_retirement_returned','checkpoint_ids':obsolete})

    def make_DEV_boundary(self,pointer):
        require(self.current_ack is not None and pointer==self.current_capture['pointer'] and self.gpu is None,'Actual CURRENT recovery before DEV')
        p=read_bound(pointer,self.parent.check)
        value={'kind':'ASR18_full_run_native_exit_and_CURRENT_checkpoint_recovery_boundary_for_serial_DEV_v2','owner':OWNER,'passed':True,
            'diagnostic_only':False,'source':self.parent_source,'production_recovery_scope':'CURRENT_LATEST_ONLY_WITH_VERIFIED_DURABLE_PERSISTENCE_PREFIX',
            'checkpoint_pointer':pointer,'run_identity_sha256':digest(self.identity),'optimizer_updates':p['committed_updates'],
            'sampler_state_sha256':p['sampler_state_sha256'],'journal_prefix':p['journal_prefix'],
            'all207_threeAdam_all4RNG_exact_checkpoint_restore_verified':True,'current_checkpoint_capture':self.current_capture,
            'current_native_process0':self.closed_native,'actual_recovery_process0':self.closed_recovery,'recovery_report':self.current_recovery_report,
            'production_recovery_boundary':self.current_ack,'durable_persistence_prefix_verification':self.current_coverage,'durable_head':self.durable.head_pin}
        path=self.control/('DEV_boundary_'+str(len(self.lifecycle.state_dict()['completed_boundaries']))+'.json');return self._new(path,value)

    def execute_DEV(self,request_pin):
        require(self.gpu is None and not self.config['owner_validation_only'],'Only actual declared pass-tail DEV, no validation fake barrier')
        from asr18_full_run_DEV_runtime_9223_v3 import resource_proposal
        request=read_bound(request_pin,self.parent.check);name=request['action']['boundary']['name']
        folder=self.folder/('DEV_'+name)
        config={'kind':'ASR18_optimizer_free_serial350DEV_worker_configuration_v1','worker_mode':'serial350DEV_optimizer_free',
            'production_enabled':False,'diagnostic_only':False,'parent_driver_source':self.parent_source,
            'source_manifest':self.config['source_manifest'],'worker_driver_source':self.config['DEV_execution_sources']['worker_source'],
            'GPU_python':self.config['GPU_python'],'resource_policy':resource_proposal(),
            'entire_worker_forecast':{'entire_worker_private_forecast_bytes':5450000000},
            'base_assets':self.config['DEV_base_assets'],'source_bindings':self.config['DEV_source_bindings'],
            'request':request_pin,'owned_worker_admission_path':str(folder/'owned_worker_admission.json'),
            'ROOT_execution_review':self.config['ROOT_DEV_static_execution_review'],
            'ROOT_resource_review':self.config['ROOT_DEV_resource_review'],
            'maximum_GPU_child_wall_seconds':self.config['maximum_DEV_child_wall_seconds']}
        path=self.control/('DEV_configuration_'+name+'.json');self._new(path,config)
        self.gpu=self.GPUService.launch(path,folder,self.parent);self.parent.GPU=self.gpu
        actual0=self.gpu.finish();self.gpu=None;self.parent.GPU=None
        worker_report=read_bound(actual0,self.parent.check)['report']
        require(Path(worker_report['path'])==folder/'serial350_worker_report.json',
                'SAME actual autonomous inner report in process0; release envelope independently sealed')
        return {'worker_report':worker_report,'actual_process0':actual0}

    def _normal_budget_barrier(self):
        if self.gpu is None or self.current_capture is None:return False
        # Whole-window budget boundary only: never interrupt Adam or shorten a
        # pass. Preserve PRE/state/journal then actual0/current restore/prefix ACK.
        reserve=(self.config['maximum_CPU_phase_seconds']+self.config['maximum_native_window_seconds']
                 +self.config['maximum_archive_phase_seconds']+self.config['maximum_checkpoint_phase_seconds']
                 +self.config['maximum_native_exit_seconds'])
        require(0<reserve<self.config['maximum_GPU_child_wall_seconds'],
                'Reviewed native wall ceiling must support one complete next window plus closure')
        return time.monotonic()-self.native_started+reserve>=self.config['maximum_GPU_child_wall_seconds']

    def _normal_owner_pause(self):
        require(0<self.schedule.committed_updates()<=self.config['planned_native_updates']
            and self.actual_owner_invocation_count<513,'Positive bounded cumulative normal pause, no crash-tail recovery')
        if self.gpu is not None:self._recover()
        else:
            require(self.current_ack is not None,'Closed normal pause needs actual current recovery ACK')
            self._verify_closed(self.closed_native);self._verify_closed(self.closed_recovery)
        require(self.gpu is None and self.durable.state_dict()['pending']['publications']==0,
                'Actual native0/CURRENT recovery0/prefix ACK before normal owner pause')
        state=self._read_state();require(state['pending_publication'] is None and state['pending_retirement'] is None,
                'No storage mutation in flight at normal owner pause')
        pointer=read_bound(self.current_capture['pointer'],self.parent.check)
        fields=('current_capture','current_completion','current_response','current_release','current_recovery_report','current_ack',
            'current_coverage','current_packet','current_CPU0','current_CPU_report','current_bridge',
            'current_cleanup0','current_cleanup','current_workspace','current_original_plan','current_intent',
            'closed_native','closed_recovery','current_cadence','native_sequence','recovery_sequence','recovery_control_sequence','proof_sequence','last_window_index')
        value={key:deepcopy(getattr(self,key)) for key in fields}
        value.update(kind='ASR18_normal_coherent_owner_pause_snapshot_v1',configuration=self.config_pin,
            run_identity_sha256=digest(self.identity),namespace=str(self.root),producer=self.producer,
            parent_source=self.parent_source,controller_source=record(__file__),normal_coherent_pause=True,
            actual_optimizer_updates=self.schedule.committed_updates(),actual_owner_invocation_count=self.actual_owner_invocation_count,
            sampler_state=deepcopy(self.schedule.state_dict()),lifecycle_state=deepcopy(self.lifecycle.state_dict()),
            durable_head=deepcopy(self.durable.head_pin),durable_owner=self.durable.owner,storage_owner=self.store.marker_pin,
            worker_control_pool_state=self.parent.control_pool.state_pin,
            worker_control_pool_binding=self.parent.control_pool.snapshot_binding(),
            storage_catalog=deepcopy(state['catalog']),journal_prefix=deepcopy(pointer['journal_prefix']),
            next_draw_packet_sha256=digest(read_bound(self.current_capture['next_draw_packet'],self.parent.check)))
        snapshot=self._new(self.control/('normal_pause_'+str(self.actual_owner_invocation_count).zfill(6)+'.json'),value)
        report={'kind':'ASR18_full_owner_normal_boundary_pause_report_v1','passed':True,'producer':self.producer,
            'worker_source':record(__file__),'parent_source':self.parent_source,'configuration':self.config_pin,
            'run_identity_sha256':digest(self.identity),'actual_optimizer_updates':self.schedule.committed_updates(),
            'actual_owner_invocation_count':self.actual_owner_invocation_count,'normal_pause_snapshot':snapshot,
            'normal_internal_native_and_recovery_processes_actually_closed0':True,
            'outer_actual_TOOL0_still_must_be_observed':True,
            'completed_passes':len(self.lifecycle.state_dict()['completed_boundaries']),'production_enabled':False}
        write_json(self.folder/'report.json',report);return report

    def run(self):
        while True:
            self.check()
            owner_reserve=(self.config['maximum_CPU_phase_seconds']+self.config['maximum_native_window_seconds']
                +self.config['maximum_archive_phase_seconds']+self.config['maximum_checkpoint_phase_seconds']
                +self.config['maximum_cleanup_phase_seconds']+self.config['maximum_native_exit_seconds']
                +self.config['maximum_recovery_phase_seconds'])
            require(0<owner_reserve<self.config['maximum_owner_wall_seconds'],'One complete future window plus coherent owner closure reserve')
            if (self.administrative_maximum_new_updates is not None and self.current_capture is not None
                and self.schedule.committed_updates()-self.invocation_initial_updates>=self.administrative_maximum_new_updates
                and self.schedule.committed_updates()<self.config['planned_native_updates']):
                return self._normal_owner_pause()
            if (self.current_capture is not None and self.schedule.committed_updates()<self.config['planned_native_updates']
                and time.monotonic()-self.began+owner_reserve>=self.config['maximum_owner_wall_seconds']):
                return self._normal_owner_pause()
            require(time.monotonic()-self.began<self.config['maximum_owner_wall_seconds'],'Reviewed outer owner wall budget')
            if self._normal_budget_barrier():
                self._recover();continue
            action=self.durable.next_action()
            if action.get('kind')=='actual_native_exit_then_current_latest_recovery_and_persistence_prefix_required':
                self._recover();continue
            action=action.get('action',action)
            if action['kind']=='training_and_declared_DEV_complete':break
            if action['kind']=='checkpoint_then_serial350DEV':
                report=self.serial_DEV({'action':action,'checkpoint_pointer':self.current_capture['pointer'],'identity':self.identity,
                    'optimizer_updates':self.schedule.committed_updates(),'split':'validation','clips':350})
                from asr18_BPE9223_lifecycle_v3 import official_DEV
                score=official_DEV(read_bound(report,self.parent.check),self.current_capture['pointer'],read_bound(self.identity['scorer_policy'],self.parent.check))
                old=self.lifecycle.state_dict()['winner'];change=score['strict_all_six_passed'] and (old is None or score['word_errors']['tr']<old['word_errors']['tr'])
                winner=self.preserve_winner({'replace':change,'old_winner':old,'candidate_pointer':self.current_capture['pointer'],'DEV':report,'score':score})
                self.durable.acknowledge_DEV_and_winner(report=report,winner=winner);self._refresh_boundary();continue
            require(action['kind']=='bounded_training_window','Unsupported tail preserved, no guessed recovery')
            self.before_window=deepcopy(self.schedule.state_dict());count=len(self.lifecycle.state_dict()['completed_boundaries'])
            plan=(self.context.validation_plan(self.schedule,completed_boundaries=count,maximum_updates=4)
                if self.config['owner_validation_only'] else self.context.plan(self.schedule,completed_boundaries=count))
            if (self.administrative_maximum_new_updates is not None and self.current_capture is not None
                and self.schedule.committed_updates()-self.invocation_initial_updates+len(plan['updates'])>self.administrative_maximum_new_updates):
                return self._normal_owner_pause()
            prepared=self._prepare_metadata(plan,self.last_window_index+1)
            del plan
            packet=self._execute_prepared_CPU(prepared)
            if self.gpu is None:self._start_native(packet)
            packet=self._seal_fresh_CPU_task_handoff(packet)
            response=self.gpu.command('run_prepared_window',window=packet);self.current_response=response
            self.current_completion=response['window_completion'];completion=read_bound(self.current_completion,self.parent.check)
            self.current_capture=completion['checkpoints'][0];self.current_release=response['checkpoint_storage_release_ack']
            self.current_idle=self.gpu.seal_idle(response)
            self._cleanup(completion)
            plan=read_bound(prepared['plan_pin'],self.parent.check)
            self._archive_and_publish(plan,self.current_completion);self.last_window_index+=1
            del plan,prepared,completion,response,packet
            gc.collect()
            if self.config['owner_validation_only']:
                actual=self.schedule.committed_updates()
                require(actual in (4,8),'Same original productionbody, two actual complete4-update windows')
                if actual==4 and self.actual_owner_invocation_count==1:return self._normal_owner_pause()
                if actual==8:
                    self._recover();break
            self._refresh_boundary()

        if self.gpu is not None:self._recover()
        # Actually reopen the packed authority after current exact recovery.
        restored=self.lifecycle.state_dict();self.durable.close()
        lifecycle=FullRunLifecycle(self.schedule,identity=self.identity,callbacks=self.lifecycle.callbacks,
            qualified_winner=self.config['qualified_source_winner'],restored=restored,restore_proof=self.current_ack)
        self.lifecycle=lifecycle
        self.durable=PackedDurableLifecycleAdapter(lifecycle,self.root/'durable',policy=self.config['durable_policy'],proofs=self.proofs,
            create=False,resume_proof=self.current_ack,operational_compatibility=self.config['operational_compatibility'])
        require(self.durable.state_dict()['pending']['publications']==0,'Actual current recovery and durable reopen0')
        result={'kind':'ASR18_actual_full_run_owner_pipeline_validation_v1','passed':True,'producer':self.producer,
            'worker_source':record(__file__),'parent_source':self.parent_source,'run_identity_sha256':digest(self.identity),
            'actual_owner_invocation_count':self.actual_owner_invocation_count,'actual_normal_resume_request':self.resume_request,
            'actual_optimizer_updates':self.schedule.committed_updates(),'completed_passes':len(self.lifecycle.state_dict()['completed_boundaries']),
            'current_checkpoint':self.current_capture,'current_native_process0':self.closed_native,'current_recovery_process0':self.closed_recovery,
            'current_recovery_report':self.current_recovery_report,'current_persistence_prefix_coverage':self.current_coverage,
            'durable_current_recovery_ack':self.current_ack,'durable_reopened_head':self.durable.head_pin,
            'actual_obsolete_worker_control_archives':self.worker_control_archive_result,
            'worker_control_pool':self.parent.control_pool.snapshot_binding(),
            'qualified_source_winner_preserved':self.lifecycle.state_dict()['winner']==self.config['qualified_source_winner'],
            'owner_validation_only':self.config['owner_validation_only'],'production_source_eligible':not self.config['owner_validation_only'],
            'no_fake_pass_or_DEV_barrier':True,'production_enabled':False,'wall_seconds':time.monotonic()-self.began}
        write_json(self.folder/'report.json',result);return result

    def close(self):
        if getattr(self,'gpu',None) is not None:self.gpu.abort();self.gpu=None;self.parent.GPU=None
        if hasattr(self,'store'):self.store.close()
        if hasattr(self,'durable'):self.durable.close()
        if hasattr(self,'context'):self.context.close()
        if hasattr(self,'parent'):self.parent.close()


def run(config_path,review_path,folder,**services):
    owner=object.__new__(FullOwner)
    try:
        FullOwner.__init__(owner,config_path,review_path,folder,**services);return owner.run()
    finally:
        owner.close()


def main():
    import argparse,os,sys,inspect
    require(os.environ.get('PYTHONUTF8')=='1' and sys.flags.utf8_mode==1,'Required UTF8 at NEW process entry before owner construction')
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'work/research'))
    import run_ASR18_BPE9223_full_run_owner_v19 as frozen
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--review',required=True)
    parser.add_argument('--folder',required=True);parser.add_argument('--resume-request');a=parser.parse_args()
    c=read_bound(record(a.config));comp=checked_compatibility(c['operational_compatibility'],c['full_run_identity'])
    require(c['full_owner_controller_source']==comp['operational_sources']['controller']==record(__file__)
        and c['parent_driver_source']==comp['operational_sources']['parent']==record(frozen.__file__)
        and c['operational_entry_mode']==comp['operational_entry_mode']=='direct_new_controller_with_frozen_parent19_services'
        and c['parent_process_UTF8_required'] is comp['parent_process_UTF8_required'] is True,
        'Actual fresh controller19 CLI entry distinct from exact frozen parent19 service authority')
    require(record(inspect.getsourcefile(frozen.GPUService.__init__))==c['parent_driver_source'],
        'Actual frozen19 GPU service code; never call old convenience controller17')
    class FreshParentGuard(frozen.ParentGuard):
        def configure_control_quota(self,control,policy,*,identity,create,operational_compatibility):
            require(policy==frozen.CONTROL_POLICY,'Same exact control quotas')
            self.control=Path(control).resolve()
            require(self.control.is_relative_to(ROOT/'runs'),'Exact new E namespace')
            from asr18_worker_control_pool_v3 import WorkerControlPool
            self.control_pool=WorkerControlPool(self.control,identity=identity,policy=policy,
                authority_source=record(frozen.frozen_parent.__file__),guard=self.pool_guard,metadata_phase=self.metadata_phase,
                admit_control=self.control_guard,operational_compatibility=operational_compatibility,create=create)
            self.worker_slots=list(self.control_pool.live_folders());self.control_guard()
    return run(a.config,a.review,Path(a.folder).resolve(),parent_source=record(frozen.__file__),ParentGuard=FreshParentGuard,
        GPUService=frozen.GPUService,run_CPU=frozen.frozen_CPU_executor.run_CPU,cleanup_script=Path(frozen.__file__).resolve(),
        storage=frozen.storage,fresh_admit=frozen.fresh_admit,execute_owned_DEV=frozen.execute_owned_DEV,
        native_boundary_DEV=frozen.native_boundary_DEV,install_owner=frozen.install_owner,
        verify_worker_control_protection=frozen.frozen_parent.verify_worker_control_protection,
        verify_closed_worker_control=frozen.frozen_parent.verify_closed_worker_control,resume_request=None if a.resume_request is None else record(a.resume_request))

if __name__=='__main__':main()
