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
from asr18_BPE9223_durable_owner_adapter_v7 import PackedDurableLifecycleAdapter
from asr18_BPE9223_indexed_checkpoint_storage_v2 import IndexedOwnedFutureCheckpointStore
from asr18_BPE9223_indexed_checkpoint_storage_v1 import AUX_ROLES
from bpe_bounded_production_journal_v4 import JournalNamespace
from asr18_BPE9223_source_context_v2 import FullRunSourceContext9223 as FullRunSourceContext,full_run_next_draw_packet
from asr18_BPE9223_persistence_prefix_v10 import OwnerPrefixVerifier,create_coverage,_Metadata,_capture,_JournalPrefixes,verify_acknowledged_resource_reentry
from asr18_BPE9223_metadata_resolver_v3 import MetadataResolver
from bpe_final_DATA_schedule_context_v2 import normalize_schedule_state
from bpe_training_schedule_v1 import WEAK
from bpe_streaming_checkpoint_semantics_v2 import select,decode_scalar_tree,verify_saved_metadata
from asr18_BPE9223_operational_metadata_v2 import read_bound,configure as configure_metadata
from fastmss_owned_process_closure_v2 import identity,verify,verify_resident_idle
ROOT=Path(r'E:\gpu_training_scratchpad');OWNER='/root/memory_diagnosis'
WITNESS_ZDICT=b'ASR18 owner native_checkpoint producer sha256 bytes path source configuration ownership state_revision catalog coherent_latest_pointer protected_owned_winners nonce checkpoint_mappings_released candidate_payload_handles_released checkpoint_write_inflight restore_in_progress actual_current_checkpoint_pointer native_release_ack resident_native_boundary current_exact_recovery publication_cadence_evidence incremental_authoritative_journal creation_identity writer_epoch source session_owner_nonce auxiliary_manifest_sha256 all_CPU_auxiliary_producers_actually_closed0 native_auxiliary_writer_contexts_returned all_auxiliary_handles_released E:\\\\gpu_training_scratchpad\\\\analysis\\\\ASR18_memory_diagnosis_v1\\\\ E:\\\\gpu_training_scratchpad\\\\runs\\\\ASR18_full_owner_validation_08_ C:\\\\Users\\\\ardao\\\\Documents\\\\Codex\\\\2026-10-10\\\\i-x20\\\\outputs\\\\parakeet_turkish\\\\ '


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
    local_store=IndexedOwnedFutureCheckpointStore(c['operational_storage_owner'],guard=guard,
        operational_compatibility=c['operational_compatibility'])
    local_resolver=MetadataResolver(local_store,control_root=Path(c['operational_control_root']),guard=guard,
        metadata_phase=phase,identity=c['full_run_identity'],operational_compatibility=c['operational_compatibility'],
        current_native_resource_closure=None)
    def read_bound(pin,unused_guard=guard):
        if Path(pin['path']).exists():return physical_or_configured(pin,guard)
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
        and snapshot['recovery_sequence']<snapshot['recovery_control_sequence']<513
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
            packed=stream.read(packed_size);decoder=zlib.decompressobj(zdict=WITNESS_ZDICT)
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


def _checked_prior_operational_failure(pin,*,config,compatibility,resource,guard):
    """Exact dead supervisor failure before any GPU child, never recovery0."""
    from fastmss_owned_process_closure_v2 import dead
    require(pin==config['prior_operational_failure']==compatibility['prior_operational_failure'],
        'Separately ROOT-bound prior operational failure, never a normal0 receipt')
    failed=read_bound(pin,guard);prior=read_bound(failed['configuration'],guard)
    require(failed['kind']=='ASR18_operational_supervisor_failed_before_GPU_reservation_actual_TOOL1_v1'
        and failed['actual_tool_exit_code']==failed['actual_exec_tool_exit_code']==1
        and failed['actual_identity_dead'] is True and dead(failed['producer'])
        and failed['source']==prior['parent_driver_source'] and failed['controller']==prior['full_owner_controller_source']
        and prior['full_run_identity']==config['full_run_identity']
        and prior['resource_native_closure']==config['resource_native_closure']
        and prior['original_genesis_configuration']==compatibility['original_configuration'],
        'Actual failed old operational parent/source/configuration and exact scientific lineage')
    for key in ('actual_GPU_children_started','actual_current_fresh_restores','new_authoritative_optimizer_updates'):
        require(type(failed[key]) is int and failed[key]==0,'No invented prior GPU launch/recovery/update: '+key)
    require(failed['preserved_production_optimizer_updates']==20
        and failed['no_GPU_worker_directory_created'] is True and failed['no_new_genesis_or_scientific_reset'] is True
        and failed['checkpoint_pointer']==resource['current_checkpoint_pointer']
        and failed['storage_state']==resource['storage_state']
        and failed['journal']==resource['closed_registered_journal']
        and failed['p2_reservation_preserved']==resource['reserved_unconsumed_p2'],
        'Real20/p1/whole journal and reserved unconsumed p2 survived failure exactly')
    joined=read_bound(failed['owner_guard'],guard)
    require(joined['kind']=='ASR18_actual_outer_owner_guard_joined_completion_v1'
        and joined['producer']==failed['producer'] and joined['source']==compatibility['legacy_guard_authority_source']
        and joined['monitor_joined'] is True and joined['monitor_clean'] is True
        and type(joined['samples']) is int and joined['samples']>=3,'Actual clean joined original guard, failed supervisor dead')
    registry=read_bound(failed['pool_registry'],guard)
    require(registry['revision']==failed['pool_revision']==3
        and registry['reservation_count']==failed['pool_reservation_count']==1
        and len(registry['live'])==failed['pool_live_count']==1
        and registry['pending_retirement'] is None,
        'Protection-only pool revision3; no prior new GPU reservation or counter reset')
    for field in ('checkpoint_pointer','storage_state','journal','durable_latest','pool_registry'):
        require(record(failed[field]['path'])==failed[field],'Exact unchanged failure frontier: '+field)
    controls=Path(config['operational_control_root'])/'recovery_000001'
    for key,name in (('request','request.json'),('configuration','worker_configuration.json')):
        member=failed['recovery_control_metadata_preserved'][key]
        require(Path(member['path'])==controls/name and record(member['path'])==member,
            'Preserved failed control attempt metadata; never overwritten/reused')
    require(not (Path(pin['path']).parent/'recovery_000001').exists(),
        'No prior failed operational GPU worker directory or child inferred')
    return failed


def checked_resource_resume_request(config_pin,review,request_pin,*,guard):
    """Read-only exact abnormal frontier; this is NOT a normal0 pause proof."""
    c=read_bound(config_pin,guard);request=read_bound(request_pin,guard)
    compatibility=checked_compatibility(c['operational_compatibility'],c['full_run_identity'],guard)
    resource=checked_resource_stop(c['resource_native_closure'],identity=c['full_run_identity'],
        compatibility=c['operational_compatibility'],guard=guard)
    require(review['resource_resume_request']==request_pin
        and request['kind']=='ASR18_exact_coherent_resource_stop_resume_request_v1'
        and request['original_configuration']==compatibility['original_configuration']
        and request['operational_configuration']==config_pin
        and request['operational_compatibility']==c['operational_compatibility']
        and request['current_native_resource_closure']==c['resource_native_closure']
        and request['run_identity_sha256']==digest(c['full_run_identity'])
        and request['no_source_optimizer_RNG_sampler_or_history_reset'] is True,
        'Exact reviewed operational request retains immutable genesis and scientific identity')
    failed=_checked_prior_operational_failure(request['prior_operational_failure'],config=c,
        compatibility=compatibility,resource=resource,guard=guard)
    pointer=read_bound(resource['current_checkpoint_pointer'],guard)
    metadata=read_bound(pointer['checkpoint'],guard)
    saved=decode_scalar_tree(select(metadata['tree'],'sampler'))
    snapshot=deepcopy(request['owner_state'])
    snapshot['sampler_state']=normalize_schedule_state(snapshot['sampler_state'])
    require(snapshot['sampler_state']==saved and digest(saved)==pointer['sampler_state_sha256']
        and snapshot['actual_owner_invocation_count']==2
        and snapshot['native_sequence']==1 and snapshot['recovery_sequence']==0
        and type(snapshot['recovery_control_sequence']) is int and snapshot['recovery_control_sequence']==1
        and type(snapshot['proof_sequence']) is int and snapshot['proof_sequence']==_original_proof_sequence(request['prior_operation_witness_stream'],guard)
        and snapshot['last_window_index']==0
        and snapshot['current_capture']['pointer']==resource['current_checkpoint_pointer']
        and snapshot['current_capture']['checkpoint']==pointer['checkpoint']
        and snapshot['current_capture']['pre_save_fingerprint']==pointer['identity']['caller_identity']['pre_save_fingerprint']
        and snapshot['current_capture']['next_draw_packet']==pointer['identity']['caller_identity']['next_draw_packet']
        and snapshot['journal_prefix']==pointer['journal_prefix']
        and snapshot['next_draw_packet_record_sha256']==resource['next_draw_packet_archived_binding']['sha256']
        and snapshot['current_ack'] is None and snapshot['current_recovery_report'] is None
        and snapshot['closed_native']==c['resource_native_closure'] and snapshot['closed_recovery'] is None,
        'Saved real20 scientific frontier and cumulative invocation/native/recovery counters, no invented recovery')
    # The saved native response is physical and remains distinct from the
    # archived completion/PRE bytes, which the registered resolver checks later.
    response=read_bound(resource['old_response'],guard);command=read_bound(resource['old_command'],guard)
    require(snapshot['current_response']==response and snapshot['current_completion']==response['window_completion']
        and snapshot['current_release']==response['checkpoint_storage_release_ack']
        and snapshot['current_packet']==command['window']
        and snapshot['current_intent']==command['window']['checkpoint_intent']
        and snapshot['current_workspace']==command['window']['auxiliary_workspace']
        and snapshot['current_original_plan']==command['window']['coordinator_plan'],
        'Exact original response nonce/release/completion/window links, prior idle never restored as live')
    from asr18_full_run_serial_DEV_9223_v3 import read_logical_durable_head
    head=read_logical_durable_head(snapshot['durable_head'])['head']
    require(head['lifecycle']==snapshot['lifecycle_state']
        and head['lifecycle']['last_pointer']==resource['current_checkpoint_pointer']
        and head['lifecycle']['last_sampler_sha256']==digest(saved)
        and head['pending']['publications']==head['pending']['windows']==1
        and head['last_recovery'] is None and head['owner']==snapshot['durable_owner'],
        'Original exact pending p1 lifecycle, not a fabricated normal acknowledged head')
    state=read_bound(resource['storage_state'],guard)
    reservation=request['reserved_preparation']
    require(state['latest']=='p000000001' and state['normal_publications']==1
        and state['pending_publication']==reservation['intent']==resource['reserved_unconsumed_p2']
        and state['pending_retirement'] is None and state['catalog']==snapshot['storage_catalog']
        and reservation['workspace']['id']=='p000000002'
        and reservation['workspace']['plan']==reservation['plan']
        and record(reservation['plan']['path'])==reservation['plan']
        and not (Path(reservation['workspace']['workspace'])/'CPU/payload').exists(),
        'Original reserved unconsumed p2 is preserved exactly; no cancel/reset/new reservation')
    require(snapshot['worker_control_pool_state']==failed['pool_registry'],
        'Exact revised pool snapshot follows the failed operational invocation')
    snapshot['reserved_preparation']=deepcopy(reservation)
    return snapshot


def checked_acknowledged_resource_resume_request(config_pin,review,request_pin,*,guard):
    """Actual already-recovered p1 plus failed CPU entry, never owner TOOL0."""
    from fastmss_owned_process_closure_v2 import dead
    from asr18_full_run_serial_DEV_9223_v3 import read_logical_durable_head
    c=read_bound(config_pin,guard);request=read_bound(request_pin,guard)
    compatibility=checked_compatibility(c['operational_compatibility'],c['full_run_identity'],guard)
    stopped_pin=c['prior_acknowledged_resource_stop'];stopped=read_bound(stopped_pin,guard)
    prior=read_bound(stopped['configuration'],guard)
    require(review['resource_resume_request']==request_pin
        and request['kind']=='ASR18_exact_ACKed_resource_reference0_CPU_resume_request_v1'
        and request['operational_configuration']==config_pin
        and request['original_configuration']==compatibility['original_configuration']
        and request['operational_compatibility']==c['operational_compatibility']
        and request['prior_operational_compatibility']==compatibility['prior_operational_compatibility']==prior['operational_compatibility']
        and request['prior_acknowledged_resource_stop']==stopped_pin==compatibility['prior_acknowledged_resource_stop']
        and request['current_native_resource_closure']==c['resource_native_closure']
        and request['no_source_optimizer_RNG_sampler_or_history_reset'] is True
        and request['run_identity_sha256']==digest(c['full_run_identity']),
        'Exact separately reviewed already-ACKed resource reentry and unchanged scientific identity')
    require(stopped['kind']=='ASR18_operational_supervisor_CPU_entry_resource_stop_after_actual_CURRENT20_recovery_TOOL1_v1'
        and stopped['actual_tool_exit_code']==stopped['actual_exec_tool_exit_code']==1
        and stopped['all_captured_identities_dead'] is True
        and all(dead(item) for item in stopped['all_captured_identities'])
        and prior['full_run_identity']==c['full_run_identity']
        and prior['resource_native_closure']==c['resource_native_closure']
        and stopped['source']==stopped['parent_source']==prior['parent_driver_source']
        and stopped['controller_source']==prior['full_owner_controller_source']
        and stopped['first_GPU_fresh_recovery_passed'] is True
        and stopped['preserved_production_updates']==20
        and all(type(stopped[k]) is int and stopped[k]==0 for k in ('new_training_GPU_children','new_CPU_children','new_optimizer_updates')),
        'Genuine dead owner TOOL1 after fresh20, no new training draw or normal-owner0 relabel')
    joined=read_bound(stopped['owner_guard'],guard)
    require(joined['producer']==stopped['producer'] and joined['source']==compatibility['legacy_guard_authority_source']
        and joined['monitor_clean'] is True and joined['monitor_joined'] is True,
        'Prior source2 monitor actually joined cleanly; whole owner nevertheless failed')
    failed_pin=c['prior_CPU_reference_failure'];failed=read_bound(failed_pin,guard)
    require(failed_pin==request['prior_CPU_reference_failure']==compatibility['prior_CPU_reference_failure']
        and failed['kind']=='ASR18_operational_CPU_controller_resource_stop_after_reference0_TOOL1_v1'
        and failed['actual_tool_exit_code']==failed['actual_exec_tool_exit_code']==1
        and failed['all_captured_identities_dead'] is True
        and all(dead(item) for item in failed['all_captured_identities'])
        and failed['new_GPU_children']==failed['new_optimizer_updates']==0
        and failed['preserved_production_updates']==20
        and failed['frontend_process_started'] is False and failed['feature_owner_created'] is False
        and failed['prior_genuine_current20_recovery0']==stopped['actual_normal_fresh_recovery0']
        and failed['prior_resource_ACK']==stopped['resource_prefix_acknowledgement'],
        'Distinct actual controller1/reference0 intervening failure, no fabricated normal CPU0')
    failed_config=read_bound(failed['configuration'],guard)
    require(failed_config['full_run_identity']==c['full_run_identity']
        and failed_config['CPU_controller_source']==c['CPU_leaf_execution_delegate_source']
        and failed['parent_source']==failed_config['parent_driver_source']
        and failed['controller_source']==failed_config['full_owner_controller_source'],
        'Truthful prior failed parent/controller/leaf-delegate source lineage')
    failed_guard=read_bound(failed['owner_guard'],guard)
    require(failed_guard['monitor_clean'] is True and failed_guard['monitor_joined'] is True
        and failed_guard['producer']==failed['producer'], 'Intervening failed owner monitor actually joined')
    for key in ('checkpoint_pointer','storage_state','durable_latest','journal','pool_registry','p2_plan'):
        require(record(failed[key]['path'])==failed[key], 'Exact unchanged latest post-stop frontier: '+key)
        if key!='pool_registry':require(failed[key]==stopped[key], 'Real committed20 frontier unchanged: '+key)
    for pin in stopped['p2_metadata_files_preserved']:
        require(Path(pin['path']).parent==Path(c['checkpoint_storage_namespace'])/'transient/p000000002'
            and record(pin['path'])==pin,'Every prior p2 preparation artifact retained exactly')
    resource=checked_resource_stop(c['resource_native_closure'],identity=c['full_run_identity'],compatibility=c['operational_compatibility'],guard=guard)
    pointer=read_bound(stopped['checkpoint_pointer'],guard);metadata=read_bound(pointer['checkpoint'],guard)
    sampler=normalize_schedule_state(decode_scalar_tree(select(metadata['tree'],'sampler')))
    snapshot=deepcopy(request['owner_state']);snapshot['sampler_state']=normalize_schedule_state(snapshot['sampler_state'])
    require(snapshot['sampler_state']==sampler and digest(sampler)==pointer['sampler_state_sha256']
        and snapshot['actual_owner_invocation_count']==failed['actual_owner_invocation_count']==4
        and snapshot['native_sequence']==stopped['native_sequence']==1
        and snapshot['recovery_sequence']==stopped['recovery_sequence']==1
        and snapshot['recovery_control_sequence']==stopped['recovery_control_sequence']==2
        and snapshot['last_window_index']==0
        and snapshot['proof_sequence']==_original_proof_sequence(request['prior_operation_witness_stream'],guard)
        and snapshot['journal_prefix']==pointer['journal_prefix']
        and snapshot['current_capture']['pointer']==stopped['checkpoint_pointer']==resource['current_checkpoint_pointer']
        and snapshot['current_capture']['checkpoint']==pointer['checkpoint']
        and snapshot['current_capture']['pre_save_fingerprint']==pointer['identity']['caller_identity']['pre_save_fingerprint']
        and snapshot['current_capture']['next_draw_packet']==pointer['identity']['caller_identity']['next_draw_packet']
        and snapshot['next_draw_packet_record_sha256']==snapshot['current_capture']['next_draw_packet']['sha256']
        and type(snapshot['next_draw_packet_sha256']) is str and len(snapshot['next_draw_packet_sha256'])==64
        and all(ch in '0123456789abcdef' for ch in snapshot['next_draw_packet_sha256'])
        and snapshot['current_ack']==request['acknowledged_recovery']==stopped['resource_prefix_acknowledgement']
        and snapshot['current_recovery_report']==stopped['actual_current_recovery_report']
        and snapshot['closed_native']==c['resource_native_closure']
        and snapshot['closed_recovery']==stopped['actual_normal_fresh_recovery0']
        and snapshot['current_coverage']==stopped['coverage'],
        'Coherent committed p1 state/cumulative counters preserved; provisional p2 is separately prepared')
    ack=read_bound(snapshot['current_ack'],guard);proof=read_bound(snapshot['closed_recovery'],guard)
    report=read_bound(snapshot['current_recovery_report'],guard);old=checked_compatibility(request['prior_operational_compatibility'],c['full_run_identity'],guard)
    verify(proof,kind='ASR18_owned_GPU_process_completion_v1',owner=OWNER,report=snapshot['current_recovery_report'],guard=guard)
    require(proof['parent_source']==old['operational_sources']['parent'] and proof['worker_source']==old['operational_sources']['runtime']
        and report['passed'] is True and report['source']==old['operational_sources']['current_recovery']
        and report['operational_compatibility']==request['prior_operational_compatibility']
        and ack['actual_closed_recovery_index']==snapshot['closed_recovery']
        and ack['actual_closed_native_index']==c['resource_native_closure']
        and ack['current_latest_all207_threeAdam_all4RNG_sampler_prefix_nextdraw_match_actual_pre_save'] is True,
        'Actual old recovery runtime7/parent6 source preserved, never stamped as the new implementation')
    latest=read_bound(stopped['durable_latest'],guard);head=read_logical_durable_head(latest['head'])['head']
    require(snapshot['durable_head']==latest['head'] and snapshot['durable_owner']==latest['owner']
        and snapshot['lifecycle_state']==head['lifecycle'] and head['last_recovery']==snapshot['current_ack']
        and head['pending']['windows']==head['pending']['publications']==0
        and head['lifecycle']['last_pointer']==snapshot['current_capture']['pointer'],
        'Actual already-ACKed head3/pending0, not original pending1 or a fabricated normal pause')
    response=read_bound(resource['old_response'],guard);command=read_bound(resource['old_command'],guard)
    require(snapshot['current_response']==response and snapshot['current_completion']==response['window_completion']
        and snapshot['current_release']==response['checkpoint_storage_release_ack']
        and snapshot['current_packet']==command['window']
        and snapshot['current_intent']==command['window']['checkpoint_intent']
        and snapshot['current_workspace']==command['window']['auxiliary_workspace']
        and snapshot['current_original_plan']==command['window']['coordinator_plan'],
        'Coherent p1 release/response/packet identity, never promote provisional prepared p2 to a commit')
    state=read_bound(stopped['storage_state'],guard);prepared=deepcopy(request['prepared_CPU'])
    require(set(prepared)=={'phase','plan_pin','window_index','intent','workspace','bridge_pin','CPU_contract','CPU_folder','CPU_report','arguments'}
        and prepared['window_index']==1 and prepared['intent']==state['pending_publication']==resource['reserved_unconsumed_p2']
        and prepared['workspace']['id']=='p000000002' and prepared['plan_pin']==stopped['p2_plan']
        and prepared['workspace']['plan']==prepared['plan_pin']
        and state['latest']=='p000000001' and state['pending_retirement'] is None
        and state['catalog']==snapshot['storage_catalog'] and state['owner']==snapshot['storage_owner'],
        'Exact original unconsumed p2 and its existing CPU preparation, no new reservation or overwrite')
    workspace=Path(prepared['workspace']['workspace']);proposal_path=workspace/'operational_CPU_window_configuration_000002.json'
    proposal=read_bound(record(proposal_path),guard);contract=read_bound(prepared['CPU_contract'],guard)
    bridge=read_bound(prepared['bridge_pin'],guard);plan=read_bound(prepared['plan_pin'],guard)
    require(Path(prepared['CPU_folder'])==workspace/'CPU_resume_000005' and not Path(prepared['CPU_folder']).exists()
        and Path(prepared['CPU_report']).resolve()==(workspace/'CPU/payload/report.json').resolve()
        and not Path(prepared['CPU_report']).exists()
        and prepared['arguments']==['--config',str(proposal_path),'--ROOT-review',c['ROOT_CPU_execution_review']['path'],'--folder',str(workspace/'CPU/payload'),'--resume-reference-failure',failed_pin['path']]
        and proposal['source_contract']==prepared['CPU_contract'] and contract['bridge']==prepared['bridge_pin']
        and bridge['coordinator_plan']==prepared['plan_pin'] and bridge['session']==c['session']
        and proposal['outer_parent_source']==c['CPU_execution_delegate_source']==compatibility['CPU_execution_delegate_source']
        and c['CPU_execution_delegate_source']==old['operational_sources']['parent']
        and prepared['phase']==plan['phase'] and plan['initial_sampler_sha256']==digest(sampler),
        'Same exact old CPU observer6/proposal/session and parsed source20 plan; new GPU parent remains independently bound')
    require(snapshot['worker_control_pool_state']==failed['pool_registry'],
        'Exact post-stop pool; archived nextdraw semantic check deferred until actual resolver exists')
    reference=read_bound(failed['actual_normal_reference0'],guard)
    verify(reference,kind='ASR18_owned_CPU_process_completion_v1',owner='/root/cpu_data_preparation',
        report=failed['actual_reference_report'],guard=guard)
    require(reference['parent_source']==c['CPU_leaf_execution_delegate_source']
        and Path(failed['actual_normal_reference0']['path']).resolve()==workspace/'CPU/payload/reference_process/actual_owned_process0.json'
        and Path(failed['actual_reference_report']['path']).resolve()==workspace/'CPU/payload/reference/report.json',
        'Genuine original7 full reference0 retained separately from failed controller')
    require(not (workspace/'CPU/payload/frontend_request.json').exists()
        and not (workspace/'CPU/payload/frontend_process').exists(), 'Frontend never started, no overwrite/retry')
    snapshot['prepared_CPU']=prepared;snapshot['reserved_preparation']=None
    return snapshot


def checked_pending40_resume_request(config_pin,review,request_pin,*,guard):
    """Prior durable20 is protocol history; completed40 is actual scientific state."""
    from asr18_BPE9223_persistence_prefix_v10 import checked_cleanup_join_failure
    c=read_bound(config_pin,guard);request=read_bound(request_pin,guard)
    compatibility=checked_compatibility(c['operational_compatibility'],c['full_run_identity'],guard)
    failed_pin=c['current_cleanup_join_failure']
    failed=checked_cleanup_join_failure(failed_pin,identity=c['full_run_identity'],
        operational_compatibility=c['operational_compatibility'],guard=guard)
    require(request['kind']=='ASR18_exact_completed40_pending_publication_cleanup_join_resume_request_v1'
        and review['resource_resume_request']==request_pin and request['operational_configuration']==config_pin
        and request['operational_compatibility']==c['operational_compatibility']
        and request['current_cleanup_join_failure']==failed_pin
        and request['no_source_optimizer_RNG_sampler_or_history_reset'] is True
        and request['authoritative_current_optimizer_updates']==40
        and request['prior_durable_protocol_optimizer_updates']==20,
        'Typed publication completion, never actual current20, a replay or normal TOOL0')
    # This failed supervisor consumed owner6/control3, but never reached GPU
    # Popen.  The completed40 scientific frontier is still solely b949.
    from fastmss_owned_process_closure_v2 import dead
    latest_pin=c['prior_pending40_preGPU_failure']
    require(latest_pin==request['prior_pending40_preGPU_failure']==compatibility['prior_pending40_preGPU_failure'],
        'Explicit latest failed pre-GPU invocation, never reset/reuse control3')
    latest=read_bound(latest_pin,guard);prior_config=read_bound(latest['configuration'],guard)
    require(latest['kind']=='ASR18_pending40_supervisor_configuration_path_type_failed_before_GPU_actual_TOOL1_v1'
        and latest['actual_tool_exit_code']==latest['actual_exec_tool_exit_code']==1
        and latest['all_captured_identities_dead'] is True
        and all(dead(item) for item in latest['all_captured_identities'])
        and latest['original_cleanup_join_failure']==failed_pin
        and prior_config['full_run_identity']==c['full_run_identity']
        and prior_config['current_cleanup_join_failure']==failed_pin
        and latest['source']==prior_config['parent_driver_source']
        and latest['controller_source']==prior_config['full_owner_controller_source']
        and latest['runtime_source']==prior_config['worker_driver_source'],
        'Actual failed owner/source lineage and same completed40, no scientific rollback')
    require(all(type(latest[k]) is int and latest[k]==0 for k in
        ('new_GPU_children','new_CPU_children','new_fresh_recoveries','new_optimizer_updates'))
        and latest['authoritative_current_optimizer_updates']==40
        and latest['actual_failure']['type']=='TypeError'
        and latest['actual_failure']['Popen_reached'] is False
        and latest['new_GPU_worker_directory_absent'] is True,
        'Preserved actual path-type failure before any GPU child/restore/update')
    joined=read_bound(latest['owner_guard'],guard)
    require(joined['producer']==latest['producer'] and dead(latest['producer'])
        and joined['source']==compatibility['legacy_guard_authority_source']
        and joined['monitor_clean'] is True and joined['monitor_joined'] is True,
        'Actual latest supervisor dead and clean joined original guard')
    require(all(latest['state_frontier_unchanged'][k]==failed[k]
        and record(failed[k]['path'])==failed[k] for k in
        ('checkpoint_pointer','checkpoint_metadata','PRE_save_fingerprint','next_draw_packet',
         'current_registered_journal','storage_state','durable_latest','p2_plan')),
        'Whole saved40/state/head/journal remains exact; no reset or new publication')
    controls=Path(c['operational_control_root'])/'recovery_000003'
    for key,name in (('request','request.json'),('configuration','worker_configuration.json')):
        pin=latest['preserved_failed_recovery_control'][key]
        require(Path(pin['path'])==controls/name and record(pin['path'])==pin,
            'Keep exact failed control3 metadata; next attempt must use control4')
    require(not (Path(latest_pin['path']).parent/'recovery_000002').exists(),
        'Actual failed attempt created no GPU worker directory or implicit process0')
    old_request=read_bound(latest['resume_request'],guard)
    expected=deepcopy(old_request['prior_durable_owner_state'])
    counters={'actual_owner_invocation_count':6,'native_sequence':2,'recovery_sequence':1,
        'recovery_control_sequence':3,'proof_sequence':2}
    require(all(type(latest[k]) is int and latest[k]==v for k,v in counters.items()),
        'Exact cumulative owner6/native2/recovery1/control3/proof2 before next owner7/control4')
    expected.update(counters)
    pool_state=latest['worker_control_pool_state_after_protection']
    expected['worker_control_pool_state']=pool_state
    expected['worker_control_pool_binding']=request['prior_durable_owner_state']['worker_control_pool_binding']
    expected['closed_native']=expected['closed_recovery']
    snapshot=deepcopy(request['prior_durable_owner_state'])
    require(digest(snapshot)==digest(expected), 'Only explicitly declared protocol counter/pool/current closed-authority substitutions')
    snapshot['sampler_state']=normalize_schedule_state(snapshot['sampler_state'])
    require(progress(snapshot['sampler_state'])['optimizer_updates']==20
        and snapshot['current_ack']==read_bound(c['prior_acknowledged_resource_stop'],guard)['resource_prefix_acknowledgement'],
        'Exact old acknowledged protocol20, not a source restore or draw')
    require(record(pool_state['path'])==pool_state,'Exact latest failed supervisor protection revision')
    registry=read_bound(pool_state,guard);pool=Path(pool_state['path']).parent
    binding={'kind':'ASR18_exact_worker_control_pool_snapshot_v1','owner':record(pool/'owner.json'),
        'registry':record(pool/'registry.json'),'index':record(pool/'index.sqlite'),'archive_prefix':record(pool/'archives.bin'),
        'archive_bytes':registry['archive_bytes'],'archive_count':registry['archive_count'],
        'reservation_count':registry['reservation_count'],'live_folders':[row['folder'] for row in registry['live']],
        'owner_folders':registry['owner_folders']}
    require(snapshot['worker_control_pool_binding']==binding and snapshot['worker_control_pool_state']==pool_state,
        'Exact present pool state, no discarded reservation or cumulative counter')
    response=read_bound(failed['actual_native_response'],guard);completion=read_bound(failed['actual_completed_native_window'],guard)
    command=read_bound(record(Path(failed['actual_native_response']['path']).parent/'command.json'),guard)
    completed=request['completed40']
    require(completed=={'response':response,'completion':failed['actual_completed_native_window'],
        'packet':command['window'],'capture':completion['checkpoints'][0],
        'release':response['checkpoint_storage_release_ack'],'CPU0':failed['actual_CPU_controller0'],
        'CPU_report':failed['actual_CPU_report'],'bridge':read_bound(failed['actual_CPU_report'],guard)['bridge'],
        'cleanup_reconciliation_process0':request['cleanup_reconciliation_process0'],
        'cleanup_reconciliation_report':request['cleanup_reconciliation_report']}
        and response['sequence']==command['sequence'] and response['nonce']==command['nonce']
        and response['window_completion']==failed['actual_completed_native_window']
        and completed['capture']['pointer']==failed['checkpoint_pointer'], 'Exact completed40 packet/capture/returned writer, no regenerated plan')
    reconciliation=read_bound(completed['cleanup_reconciliation_report'],guard)
    verify(read_bound(completed['cleanup_reconciliation_process0'],guard),kind='ASR18_owned_CPU_process_completion_v1',
        owner=OWNER,report=completed['cleanup_reconciliation_report'],guard=guard)
    require(reconciliation['kind']=='ASR18_readonly_postmortem_completed_feature_cleanup_reconciliation_v1'
        and reconciliation['passed'] is True and reconciliation['physical_cleanup_performed_in_this_invocation'] is False
        and reconciliation['current_cleanup_join_failure']==failed_pin
        and reconciliation['original_cleanup_report']==failed['actual_cleanup_report'],
        'Genuine new readonly verification0, never old cleanup process0 or repeated deletion')
    snapshot['pending40']=deepcopy(completed);snapshot['prepared_CPU']=None;snapshot['reserved_preparation']=None
    return snapshot


def ACKed40_owner_state(failed,*,next_draw_packet_sha256,guard):
    """Reconstruct committed40 metadata, never a fabricated normal-pause receipt."""
    require(type(next_draw_packet_sha256) is str and len(next_draw_packet_sha256)==64
        and all(c in '0123456789abcdef' for c in next_draw_packet_sha256),'Saved semantic nextdraw digest')
    old_request=read_bound(failed['resume_request'],guard);data=old_request['completed40']
    pointer=read_bound(failed['checkpoint_pointer'],guard);metadata=read_bound(pointer['checkpoint'],guard)
    saved=decode_scalar_tree(select(metadata['tree'],'sampler'))
    require(progress(saved)['optimizer_updates']==40 and digest(saved)==pointer['sampler_state_sha256'],
        'Actual saved40 sampler, no source20 restore or replay')
    witness=failed['storage_operation_witnesses'];require(record(witness['path'])==witness,'Exact postACK witness stream')
    sequence=0;cadence=None
    with Path(witness['path']).open('rb') as stream:
        while header:=stream.read(40):
            guard();require(len(header)==40,'Whole actual witness header')
            size,raw_size=struct.unpack('<II',header[:8]);require(0<size<=2048 and 0<raw_size<=65536,'Bounded original witness')
            packed=stream.read(size);decoder=zlib.decompressobj(zdict=WITNESS_ZDICT);raw=decoder.decompress(packed,65537)
            require(len(packed)==size and len(raw)==raw_size and decoder.eof and not decoder.unused_data
                and not decoder.unconsumed_tail and hashlib.sha256(raw).digest()==header[8:],'Exact complete witness frame')
            event=json.loads(raw);original=event['original'];name=Path(original['path']).name
            require(Path(original['path']).parent==Path(witness['path']).parent
                and int(name.rsplit('.proof.',1)[1][:-5])==sequence+1,'Continuous actual proof counter')
            sequence+=1;source_raw=event['raw_UTF8'].encode('utf8')
            require(len(source_raw)==original['bytes'] and hashlib.sha256(source_raw).hexdigest()==original['sha256'],
                'Complete original proof bytes retained, no reconstructed witness')
            proof=json.loads(source_raw)
            if proof.get('actual_current_checkpoint_pointer')==failed['checkpoint_pointer']:
                require(proof['configuration']==failed['configuration'] and proof['actual_owner']==failed['producer'],
                    'Actual producer7 publication/retirement proof lineage')
                if cadence is None:cadence=proof['publication_cadence_evidence']
                require(cadence==proof['publication_cadence_evidence'],'Same actual completed-window cadence')
    require(sequence==failed['proof_sequence']==6 and cadence is not None,'Actual6 witnesses/cadence preserved')
    pool=Path(failed['pool_registry']['path']).parent;registry=read_bound(failed['pool_registry'],guard)
    state=read_bound(failed['storage_state'],guard);latest=read_bound(failed['durable_latest'],guard)
    snapshot=deepcopy(old_request['prior_durable_owner_state'])
    snapshot.update(current_capture=data['capture'],current_completion=data['completion'],current_response=data['response'],
        current_release=data['release'],current_recovery_report=failed['actual_current_recovery_report'],
        current_ack=failed['resource_prefix_acknowledgement'],current_coverage=failed['coverage'],current_packet=data['packet'],
        current_CPU0=data['CPU0'],current_CPU_report=data['CPU_report'],current_bridge=data['bridge'],
        current_cleanup0=data['cleanup_reconciliation_process0'],current_cleanup=data['cleanup_reconciliation_report'],
        current_workspace=data['packet']['auxiliary_workspace'],current_original_plan=data['packet']['coordinator_plan'],
        current_intent=data['packet']['checkpoint_intent'],closed_native=failed['actual_normal_fresh_recovery0'],
        closed_recovery=failed['actual_normal_fresh_recovery0'],current_cadence=cadence,sampler_state=saved,
        lifecycle_state=deepcopy(failed['lifecycle_state']),durable_head=latest['head'],durable_owner=latest['owner'],
        storage_owner=state['owner'],storage_catalog=state['catalog'],journal_prefix=pointer['journal_prefix'],
        next_draw_packet_sha256=next_draw_packet_sha256,next_draw_packet_record_sha256=data['capture']['next_draw_packet']['sha256'],
        worker_control_pool_state=failed['pool_registry'],actual_owner_invocation_count=7,native_sequence=2,
        recovery_sequence=2,recovery_control_sequence=4,proof_sequence=6,last_window_index=1)
    snapshot['worker_control_pool_binding']={'kind':'ASR18_exact_worker_control_pool_snapshot_v1',
        'owner':record(pool/'owner.json'),'registry':record(pool/'registry.json'),'index':record(pool/'index.sqlite'),
        'archive_prefix':record(pool/'archives.bin'),'archive_bytes':registry['archive_bytes'],'archive_count':registry['archive_count'],
        'reservation_count':registry['reservation_count'],'live_folders':[r['folder'] for r in registry['live']],
        'owner_folders':registry['owner_folders']}
    return snapshot


def checked_ACKed40_resume_request(config_pin,review,request_pin,*,guard):
    """Actual abnormal supervisor1 after genuine fresh40/publication/ACK; no new GPU authority draw."""
    from fastmss_owned_process_closure_v2 import dead
    from asr18_full_run_serial_DEV_9223_v3 import read_logical_durable_head
    c=read_bound(config_pin,guard);request=read_bound(request_pin,guard)
    compatibility=checked_compatibility(c['operational_compatibility'],c['full_run_identity'],guard)
    pin=c['prior_acknowledged_cleanup_join_stop'];failed=read_bound(pin,guard)
    require(pin==request['prior_acknowledged_cleanup_join_stop']==compatibility['prior_acknowledged_cleanup_join_stop']
        and review['resource_resume_request']==request_pin
        and request['kind']=='ASR18_exact_ACKed40_post_worker_control_retirement_failure_resume_request_v1'
        and request['operational_configuration']==config_pin and request['operational_compatibility']==c['operational_compatibility']
        and request['run_identity_sha256']==digest(c['full_run_identity'])
        and request['no_source_optimizer_RNG_sampler_or_history_reset'] is True,'Exact separately reviewed abnormal ACKed40 reentry')
    prior=read_bound(failed['configuration'],guard)
    require(failed['kind']=='ASR18_post_ACK40_worker_control_retirement_source_join_failure_actual_TOOL1_v1'
        and failed['actual_tool_exit_code']==failed['actual_exec_tool_exit_code']==1
        and failed['normal_owner_TOOL0_claimed'] is False and failed['all_captured_identities_dead'] is True
        and all(dead(item) for item in failed['all_captured_identities'])
        and prior['full_run_identity']==c['full_run_identity'] and failed['source']==prior['parent_driver_source']
        and failed['controller_source']==prior['full_owner_controller_source'] and failed['runtime_source']==prior['worker_driver_source']
        and failed['original_cleanup_join_failure']==c['current_cleanup_join_failure']
        and failed['preserved_production_updates']==40 and failed['new_training_GPU_children']==failed['new_CPU_children']==failed['new_optimizer_updates']==0,
        'Real dead owner7 actual1 after ACK40, never normal0 or a source20 reset')
    joined=read_bound(failed['owner_guard'],guard)
    require(joined['producer']==failed['producer'] and joined['source']==compatibility['legacy_guard_authority_source']
        and joined['monitor_joined'] is True and joined['monitor_clean'] is True,'Actual original monitor clean joined')
    for key in ('checkpoint_pointer','checkpoint_metadata','current_registered_journal','storage_state','durable_latest',
        'pool_registry','pool_index','pool_archive_prefix','storage_operation_witnesses'):
        require(record(failed[key]['path'])==failed[key],'Exact postACK40 frontier/counters: '+key)
    proof=read_bound(failed['actual_normal_fresh_recovery0'],guard)
    verify(proof,kind='ASR18_owned_GPU_process_completion_v1',owner=OWNER,report=failed['actual_current_recovery_report'],guard=guard)
    recovered=read_bound(failed['actual_current_recovery_report'],guard);ack=read_bound(failed['resource_prefix_acknowledgement'],guard)
    coverage=read_bound(failed['coverage'],guard);pointer=read_bound(failed['checkpoint_pointer'],guard)
    require(proof['worker_source']==prior['worker_driver_source'] and proof['parent_source']==prior['parent_driver_source']
        and recovered['passed'] is True and recovered['current_checkpoint']['pointer']==failed['checkpoint_pointer']
        and recovered['actual_native_updates_in_recovery_process']==recovered['actual_draws_consumed_in_recovery']==0
        and recovered['operational_compatibility']==prior['operational_compatibility']
        and ack['kind']=='ASR18_cleanup_join_failure_current40_recovery_and_verified_persistence_prefix_v1'
        and ack['operational_compatibility']==prior['operational_compatibility']
        and ack['actual_closed_native_index']==ack['actual_closed_recovery_index']==failed['actual_normal_fresh_recovery0']
        and ack['final_checkpoint_pointer']==failed['checkpoint_pointer']
        and ack['persistence_prefix_coverage_index']==failed['coverage'] and coverage['passed'] is True
        and ack['current_latest_all207_threeAdam_all4RNG_sampler_prefix_nextdraw_match_actual_pre_save'] is True,
        'Genuine fresh40/closed authority/full ACK40 in original source lineage; no repeated standalone recovery')
    latest=read_bound(failed['durable_latest'],guard);head=read_logical_durable_head(latest['head'])['head']
    state=read_bound(failed['storage_state'],guard);pool=read_bound(failed['pool_registry'],guard)
    require(latest['head']['sequence']==5 and head['last_recovery']==failed['resource_prefix_acknowledgement']
        and head['pending']['windows']==head['pending']['publications']==0 and head['lifecycle']==failed['lifecycle_state']
        and state['latest']=='p000000002' and state['pending_publication'] is None and state['pending_retirement'] is None
        and pool['revision']==13 and pool['archive_count']==0 and pool['archive_bytes']==24 and pool['pending_retirement'] is None,
        'Actual ACKed40/no pending archive tail; no rollback to earlier source frontier')
    snapshot=deepcopy(request['owner_state'])
    expected=ACKed40_owner_state(failed,next_draw_packet_sha256=snapshot['next_draw_packet_sha256'],guard=guard)
    require(digest(snapshot)==digest(expected),'Entire exact current40 owner/sampler/history/counter reconstruction')
    snapshot['sampler_state']=normalize_schedule_state(snapshot['sampler_state'])
    require(digest(snapshot['sampler_state'])==pointer['sampler_state_sha256']
        and snapshot['lifecycle_state']['last_pointer']==failed['checkpoint_pointer'],'Exact current40 schedule/lifecycle')
    snapshot['prepared_CPU']=None;snapshot['reserved_preparation']=None
    return snapshot


def ACKed40_reserved_owner_state(latest,*,guard=lambda:None):
    """Exact owner8/current40 plus its own reserved, wholly unconsumed p3."""
    import sqlite3
    prior_request=read_bound(latest['resume_request'],guard)
    require(digest(prior_request['owner_state'])==digest(latest['previous_request_owner_state_preserved']),
        'Exact previously verified current40 state, never a source20 protocol snapshot')
    snapshot=deepcopy(latest['previous_request_owner_state_preserved'])
    state=read_bound(latest['storage_state'],guard);pool=Path(latest['pool_registry']['path']).parent
    registry=read_bound(latest['pool_registry'],guard);reserved=latest['reserved_unconsumed_preparation']
    plan=reserved['plan'];root=Path(reserved['workspace_path'])
    require(state['latest']=='p000000002' and state['pending_publication']==reserved['intent']
        and state['pending_publication']['id']=='p000000003' and state['pending_retirement'] is None,
        'Existing actual p3 reservation only, no cancel/new reserve/namespace adoption')
    db=sqlite3.connect('file:'+(Path(latest['storage_state']['path']).parent/'catalog.sqlite').as_posix()+'?mode=ro',uri=True)
    try:
        db.execute('PRAGMA query_only=ON');row=db.execute('SELECT value FROM auxiliary WHERE id=?',('p000000003',)).fetchone()
    finally:db.close()
    require(row is not None,'Exact registered p3 auxiliary row');workspace=json.loads(row[0])
    require(workspace['id']=='p000000003' and workspace['status']=='preparing' and workspace['plan']==plan
        and Path(workspace['workspace'])==root and read_bound(workspace['workspace_marker'],guard)==workspace['workspace_ownership']
        and all(record(reserved[k]['path'])==reserved[k] for k in
            ('plan','coordinator_plan_copy','plan_alias','prior_window_boundary'))
        and reserved['coordinator_plan_copy']['sha256']==plan['sha256']
        and not (root/'bridge.json').exists() and not (root/'CPU').exists(),
        'Owned unconsumed p3 marker/plan/copy/alias/boundary unchanged; no CPU child or overwrite')
    require(all(type(latest[k]) is int and latest[k]==value for k,value in
        {'actual_owner_invocation_count':8,'native_sequence':2,'recovery_sequence':2,
         'recovery_control_sequence':4,'proof_sequence':6,'last_window_index':1}.items()),
        'Actual owner8/native2/recovery2/control4/proof6 preserved')
    snapshot.update(actual_owner_invocation_count=8,worker_control_pool_state=latest['pool_registry'],
        storage_catalog=state['catalog'])
    snapshot['worker_control_pool_binding']={'kind':'ASR18_exact_worker_control_pool_snapshot_v1',
        'owner':record(pool/'owner.json'),'registry':record(pool/'registry.json'),'index':record(pool/'index.sqlite'),
        'archive_prefix':record(pool/'archives.bin'),'archive_bytes':registry['archive_bytes'],'archive_count':registry['archive_count'],
        'reservation_count':registry['reservation_count'],'live_folders':[r['folder'] for r in registry['live']],
        'owner_folders':registry['owner_folders']}
    snapshot['reserved_preparation']={'intent':reserved['intent'],'workspace':workspace,'plan':plan,
        'prior_window_boundary':reserved['prior_window_boundary']}
    snapshot['prepared_CPU']=None
    return snapshot


def checked_ACKed40_reader_resume_request(config_pin,review,request_pin,*,guard):
    from fastmss_owned_process_closure_v2 import dead
    from asr18_full_run_serial_DEV_9223_v3 import read_logical_durable_head
    c=read_bound(config_pin,guard);request=read_bound(request_pin,guard)
    compatibility=checked_compatibility(c['operational_compatibility'],c['full_run_identity'],guard)
    pin=c['prior_CPU_metadata_reader_failure'];latest=read_bound(pin,guard)
    require(pin==request['prior_CPU_metadata_reader_failure']==compatibility['prior_CPU_metadata_reader_failure']
        and review['resource_resume_request']==request_pin
        and request['kind']=='ASR18_exact_ACKed40_reserved_p3_readonly_authority_resume_request_v1'
        and request['operational_configuration']==config_pin and request['operational_compatibility']==c['operational_compatibility']
        and request['run_identity_sha256']==digest(c['full_run_identity'])
        and request['no_source_optimizer_RNG_sampler_or_history_reset'] is True,
        'Typed actual40/unconsumedp3 reentry after readonly-reader source join failure')
    prior=read_bound(latest['configuration'],guard)
    require(latest['kind']=='ASR18_ACKed40_parent_CPU_session_readonly_authority_join_failure_actual_TOOL1_v1'
        and latest['actual_tool_exit_code']==latest['actual_exec_tool_exit_code']==1
        and latest['normal_owner_TOOL0_claimed'] is False and latest['all_captured_identities_dead'] is True
        and all(dead(item) for item in latest['all_captured_identities'])
        and latest['source']==prior['parent_driver_source'] and latest['controller_source']==prior['full_owner_controller_source']
        and latest['runtime_source']==prior['worker_driver_source'] and prior['full_run_identity']==c['full_run_identity']
        and all(type(latest[k]) is int and latest[k]==0 for k in ('new_GPU_children','new_CPU_children','new_optimizer_updates','new_standalone_fresh_recoveries'))
        and latest['preserved_production_updates']==40 and latest['prior_acknowledged_cleanup_join_stop']==c['prior_acknowledged_cleanup_join_stop'],
        'Real dead owner8/no child/no newdraw; original ACK40 remains authority')
    joined=read_bound(latest['owner_guard'],guard)
    require(joined['producer']==latest['producer'] and joined['source']==compatibility['legacy_guard_authority_source']
        and joined['monitor_joined'] is True and joined['monitor_clean'] is True,'Original guard actually joined cleanly')
    for key in ('checkpoint_pointer','checkpoint_metadata','current_registered_journal','storage_state','durable_latest',
        'pool_registry','pool_index','pool_archive_prefix','storage_operation_witnesses'):
        require(record(latest[key]['path'])==latest[key],'Exact reservedp3/ACK40/counter frontier: '+key)
    stopped=read_bound(c['prior_acknowledged_cleanup_join_stop'],guard)
    require(c['prior_acknowledged_cleanup_join_stop']==compatibility['prior_acknowledged_cleanup_join_stop']
        and all(latest[k]==stopped[k] for k in ('resource_prefix_acknowledgement','coverage','actual_normal_fresh_recovery0',
            'actual_current_recovery_report','checkpoint_pointer','checkpoint_metadata','journal_prefix')),
        'Same immutable real fresh40/ACK40 lineage, no repeated standalone recovery')
    proof=read_bound(latest['actual_normal_fresh_recovery0'],guard)
    verify(proof,kind='ASR18_owned_GPU_process_completion_v1',owner=OWNER,report=latest['actual_current_recovery_report'],guard=guard)
    ack=read_bound(latest['resource_prefix_acknowledgement'],guard);pointer=read_bound(latest['checkpoint_pointer'],guard)
    head=read_logical_durable_head(read_bound(latest['durable_latest'],guard)['head'])['head']
    require(pointer['committed_updates']==40 and head['last_recovery']==latest['resource_prefix_acknowledgement']
        and head['pending']['windows']==head['pending']['publications']==0
        and ack['actual_closed_native_index']==ack['actual_closed_recovery_index']==latest['actual_normal_fresh_recovery0'],
        'Exact acknowledged40/no native or optimizer draw since recovery')
    authority=c['process_readonly_metadata_authority'];session=read_bound(c['session'],guard)
    require(authority==session['operational_metadata']==compatibility['process_readonly_metadata_authority']
        and authority['storage_owner']==c['operational_storage_owner'] and authority['control_root']==c['operational_control_root']
        and authority['identity']==c['full_run_identity'] and authority['current_native_resource_closure'] is None,
        'One frozen session2 readonly authority, mutable operations separately current ROOT authority')
    readonly=checked_compatibility(authority['operational_compatibility'],c['full_run_identity'],guard)
    require(all(readonly['operational_sources'][role]==compatibility['operational_sources'][role]
        for role in ('indexed_storage','metadata_resolver','operational_metadata')),
        'Same exact executing read-only components, no old mutable callback authority granted')
    snapshot=deepcopy(request['owner_state']);expected=ACKed40_reserved_owner_state(latest,guard=guard)
    require(digest(snapshot)==digest(expected),'Complete real40/p3/counter snapshot, only declared metadata substitutions')
    snapshot['sampler_state']=normalize_schedule_state(snapshot['sampler_state'])
    require(digest(snapshot['sampler_state'])==pointer['sampler_state_sha256']
        and snapshot['lifecycle_state']==head['lifecycle'],'Exact saved40 sampler/lifecycle preserved')
    return snapshot



def committed60_owner_state(failure_pin,*,guard=lambda:None):
    """Reconstruct actual9/source60/head6 plus exact unconsumed p4, never40."""
    import sqlite3
    from asr18_full_run_serial_DEV_9223_v3 import read_logical_durable_head
    failed=read_bound(failure_pin,guard)
    prior_request=read_bound(failed['resume_request'],guard)
    snapshot=deepcopy(prior_request['owner_state'])
    pointer=read_bound(failed['checkpoint_pointer'],guard)
    metadata=read_bound(pointer['checkpoint'],guard)
    sampler=normalize_schedule_state(decode_scalar_tree(select(metadata['tree'],'sampler')))
    saved=verify_saved_metadata(metadata,identity=pointer['identity'],optimizer_updates=60,
        sampler_state=sampler,journal_prefix=pointer['journal_prefix'],diagnostic_only=False)
    require(pointer['committed_updates']==60 and digest(sampler)==pointer['sampler_state_sha256']
        and saved==pointer['saved_semantic_state_verified'],'Actual854-descriptor saved60 scalar/sampler identity')
    latest=read_bound(failed['durable_latest'],guard);head=read_logical_durable_head(latest['head'])['head']
    require(head['sequence']==6 and head['lifecycle']['last_pointer']==failed['checkpoint_pointer']
        and head['lifecycle']['last_sampler_sha256']==digest(sampler)
        and head['pending']['windows']==head['pending']['publications']==1
        and head['pending']['first_optimizer_update']==41 and head['pending']['last_optimizer_update']==60,
        'Existing actual protocol60/pending41..60/previousACK40, no protocol rollback')
    completion=read_bound(failed['current_native_completion'],guard)
    response=read_bound(failed['last_resident_response'],guard)
    command=read_bound(record(Path(failed['native_boot']['path']).parent/'command.json'),guard)
    packet=command['window'];capture=completion['checkpoints'][0]
    require(capture==failed['current_checkpoint_capture'] and completion['global_new_updates']==60
        and completion['actual_new_updates']==20 and packet['window_index']==2
        and packet['coordinator_plan']==completion['coordinator_plan']
        and packet['feature_owner']==completion['feature_owner'] and packet['feature_manifest']==completion['feature_manifest']
        and response['window_completion']==failed['current_native_completion'],'Actual p3 packet/completion/source60 capture')
    CPU_report=record(Path(failed['actual_current_p3_CPU0']['path']).parent/'payload/report.json')
    cleanup_report=read_bound(failed['actual_current_p3_cleanup0'],guard)['report']
    prior_boundary=read_bound(failed['reserved_unconsumed_preparation']['files']['prior_window_boundary.json'],guard)
    state=read_bound(failed['storage_state'],guard);root=Path(failed['reserved_unconsumed_preparation']['workspace_path'])
    db=sqlite3.connect('file:'+(Path(failed['storage_state']['path']).parent/'catalog.sqlite').as_posix()+'?mode=ro',uri=True)
    try:
        db.execute('PRAGMA query_only=ON');row=db.execute('SELECT value FROM auxiliary WHERE id=?',('p000000004',)).fetchone()
    finally:db.close()
    require(row is not None,'Existing owned p4 AUX');workspace=json.loads(row[0]);plan=workspace['plan']
    require(state['latest']=='p000000003' and state['pending_publication']==failed['reserved_unconsumed_preparation']['intent']
        and state['pending_publication']['id']==workspace['id']=='p000000004' and workspace['status']=='preparing'
        and Path(workspace['workspace'])==root and not (root/'CPU').exists()
        and all(record(pin['path'])==pin for pin in failed['reserved_unconsumed_preparation']['files'].values()),
        'Actual complete unconsumed prepared p4, no new reservation or overwritten metadata')
    cadence=None;sequence=0
    with Path(failed['storage_operation_witnesses']['path']).open('rb') as stream:
        while header:=stream.read(40):
            guard();size,raw_size=struct.unpack('<II',header[:8]);decoder=zlib.decompressobj(zdict=WITNESS_ZDICT)
            raw=decoder.decompress(stream.read(size),65537)
            require(len(raw)==raw_size and decoder.eof and hashlib.sha256(raw).digest()==header[8:],'Whole exact witness frame')
            event=json.loads(raw);sequence+=1
            require(event['original']['path'].endswith('.proof.'+str(sequence)+'.json'),'Actual continuous proof counter')
            proof=json.loads(event['raw_UTF8'])
            if proof.get('publication_cadence_evidence') is not None:cadence=proof['publication_cadence_evidence']
    require(sequence==failed['proof_sequence']==10 and cadence is not None,'Actual10 witnesses and cadence')
    pool=Path(failed['pool_registry']['path']).parent;registry=read_bound(failed['pool_registry'],guard)
    snapshot.update(actual_optimizer_updates=60,sampler_state=sampler,lifecycle_state=head['lifecycle'],durable_head=latest['head'],
        journal_prefix=pointer['journal_prefix'],next_draw_packet_sha256=digest(read_bound(capture['next_draw_packet'],guard)),
        next_draw_packet_record_sha256=capture['next_draw_packet']['sha256'],
        current_capture=capture,current_completion=failed['current_native_completion'],current_response=response,
        current_release=response['checkpoint_storage_release_ack'],current_recovery_report=None,current_ack=head['last_recovery'],
        current_coverage=read_bound(head['last_recovery'],guard)['persistence_prefix_coverage_index'],
        current_packet=packet,current_CPU0=failed['actual_current_p3_CPU0'],current_CPU_report=CPU_report,
        current_bridge=prior_boundary['bridge'],current_cleanup0=failed['actual_current_p3_cleanup0'],current_cleanup=cleanup_report,
        current_workspace=packet['auxiliary_workspace'],current_original_plan=packet['coordinator_plan'],
        current_intent=packet['checkpoint_intent'],closed_native=failure_pin,closed_recovery=None,
        current_cadence=cadence,native_sequence=3,recovery_sequence=2,recovery_control_sequence=4,proof_sequence=10,
        last_window_index=2,actual_owner_invocation_count=9,storage_catalog=state['catalog'],worker_control_pool_state=failed['pool_registry'])
    snapshot['worker_control_pool_binding']={'kind':'ASR18_exact_worker_control_pool_snapshot_v1',
        'owner':record(pool/'owner.json'),'registry':record(pool/'registry.json'),'index':record(pool/'index.sqlite'),
        'archive_prefix':record(pool/'archives.bin'),'archive_bytes':registry['archive_bytes'],'archive_count':registry['archive_count'],
        'reservation_count':registry['reservation_count'],'live_folders':[v['folder'] for v in registry['live']],
        'owner_folders':registry['owner_folders']}
    snapshot['reserved_preparation']={'intent':state['pending_publication'],'workspace':workspace,'plan':plan,
        'old_prepared_metadata':failed['reserved_unconsumed_preparation']['files']}
    snapshot['prepared_CPU']=None
    return snapshot


def committed60_preGPU_owner_state(latest_failure_pin,*,guard=lambda:None):
    """Only actual owner10 pre-GPU failure; preserve genuine60 and control4."""
    from fastmss_owned_process_closure_v2 import dead
    latest=read_bound(latest_failure_pin,guard)
    require(latest['kind']=='ASR18_actual_committed60_parent_control_protection_preGPU_failure_TOOL1_v1'
        and type(latest['actual_tool_exit_code']) is int and latest['actual_tool_exit_code']==1
        and type(latest['actual_exec_tool_exit_code']) is int and latest['actual_exec_tool_exit_code']==1
        and latest['actual_failure']=='KeyError: normal_pause_snapshot'
        and latest['failing_function']=='FullOwner._worker_control_protection'
        and latest['source_failure_before_GPU_Popen'] is True
        and latest['source_failure_before_control5_creation'] is True
        and latest['GPU_children_started']==latest['CPU_children_started']==latest['optimizer_updates_this_invocation']==0
        and latest['preserved_production_updates']==60 and latest['parent_owner_invocation_attempt']==10
        and latest['persisted_owner_boot_written'] is False
        and latest['all_captured_identities_dead'] is True
        and all(dead(p)for p in latest['all_captured_identities']),
        'Truthful owner10 pre-GPU1, no GPU/control5/native0 claim')
    joined=read_bound(latest['owner_guard'],guard);prior=read_bound(latest['configuration'],guard)
    request=read_bound(latest['resume_request'],guard)
    require(joined['producer']==latest['producer'] and joined['monitor_clean'] is True and joined['monitor_joined'] is True
        and request['operational_configuration']==latest['configuration']
        and request['kind']=='ASR18_exact_committed60_CPU_memory_and_merged_topology_resume_request_v1'
        and prior['prior_committed60_CPU_memory_failure']==latest['current_committed60_CPU_failure']
        and prior['parent_driver_source']==latest['source'] and prior['full_owner_controller_source']==latest['controller_source']
        and prior['worker_driver_source']==latest['runtime_source'], 'Exact failed invocation/config/request/guard source lineage')
    failed=read_bound(latest['current_committed60_CPU_failure'],guard)
    keys=('storage_state','durable_latest','current_registered_journal','pool_registry','pool_index','pool_archive_prefix','storage_operation_witnesses')
    require(all(latest['all7frontier_bindings_unchanged'][k] is True
        and latest['state_frontier_unchanged'][k]==failed[k]==record(failed[k]['path'])for k in keys)
        and latest['recovery_control5_absent'] is True
        and not (Path(prior['run_namespace'])/'control/recovery_000005').exists(),
        'All seven actual60/head6/p4/pool16 authorities unchanged; no control5 consumed')
    snapshot=committed60_owner_state(latest['current_committed60_CPU_failure'],guard=guard)
    require(digest(snapshot)==digest(latest['previous_request_owner_state_preserved'])==digest(request['owner_state'])
        and {k:latest[k]for k in ('native_sequence','recovery_sequence','recovery_control_sequence','proof_sequence','last_window_index')}
        =={'native_sequence':3,'recovery_sequence':2,'recovery_control_sequence':4,'proof_sequence':10,'last_window_index':2},
        'Complete original60 state/history unchanged; only failed owner attempt consumed')
    snapshot['actual_owner_invocation_count']=10
    return snapshot


def checked_committed60_merged_resume_request(config_pin,review,request_pin,*,guard):
    from asr18_BPE9223_persistence_prefix_v10 import checked_committed60_CPU_failure
    c=read_bound(config_pin,guard);request=read_bound(request_pin,guard)
    comp=checked_compatibility(c['operational_compatibility'],c['full_run_identity'],guard)
    pin=c['prior_committed60_CPU_memory_failure']
    require(pin==comp['prior_committed60_CPU_memory_failure']==request['prior_committed60_CPU_memory_failure']
        and request['kind'] in ('ASR18_exact_committed60_CPU_memory_and_merged_topology_resume_request_v1',
            'ASR18_exact_committed60_preGPU_protection_failure_and_merged_topology_resume_request_v1')
        and request['operational_configuration']==config_pin and review['resource_resume_request']==request_pin
        and request['operational_compatibility']==c['operational_compatibility']
        and request['run_identity_sha256']==digest(c['full_run_identity'])
        and request['no_source_optimizer_RNG_sampler_or_history_reset'] is True,
        'Typed exact actual60 and resource-role merge, never normal-owner0 or source40')
    stopped=checked_committed60_CPU_failure(pin,identity=c['full_run_identity'],operational_compatibility=c['operational_compatibility'],guard=guard)
    for key in ('storage_state','durable_latest','current_registered_journal','pool_registry','pool_index','pool_archive_prefix','storage_operation_witnesses'):
        require(record(stopped[key]['path'])==stopped[key]==request['current_stopped_state_bindings'][key], 'Exact whole stopped60 authority: '+key)
    if request['kind']=='ASR18_exact_committed60_preGPU_protection_failure_and_merged_topology_resume_request_v1':
        latest=c['prior_committed60_preGPU_failure']
        require(latest==comp['prior_committed60_preGPU_failure']==request['prior_committed60_preGPU_failure'],
            'Exact separately ROOT-bound failed owner10, no generic abnormal retry')
        expected=committed60_preGPU_owner_state(latest,guard=guard)
    else:
        require(c.get('prior_committed60_preGPU_failure') is None,'Prior failed invocation cannot be omitted')
        expected=committed60_owner_state(pin,guard=guard)
    snapshot=deepcopy(request['owner_state'])
    snapshot['sampler_state']=normalize_schedule_state(snapshot['sampler_state'])
    require(digest(snapshot)==digest(expected) and snapshot['actual_optimizer_updates']==60
        and c['CPU_owned_allowance_bytes']==1867108864 and c['CPU_inner_owned_allowance_bytes']==1833554432
        and c['merged_CPU_policy']==comp['merged_CPU_topology']
        and c['merged_CPU_policy']['whole_owned_CPU_allowance_bytes']==1995108864,
        'Saved60/owner9/native3/recovery2/control4/proof10/p4 and whole<2GB exactly preserved')
    require(c['process_readonly_metadata_authority']==read_bound(c['session'],guard)['operational_metadata']
        ==comp['process_readonly_metadata_authority'],'One stable reviewed read-only metadata authority')
    return snapshot


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
        require(resume_request is not None,'Operational owner requires an explicit existing-state request')
        request=read_bound(resume_request)
        require(request.get('kind') in ('ASR18_exact_committed60_CPU_memory_and_merged_topology_resume_request_v1',
            'ASR18_exact_committed60_preGPU_protection_failure_and_merged_topology_resume_request_v1',
            'ASR18_exact_normal_boundary_owner_resume_request_v1'),'Only exact committed60 or future normal continuation')
        self.committed60_initial=request.get('kind') in ('ASR18_exact_committed60_CPU_memory_and_merged_topology_resume_request_v1',
            'ASR18_exact_committed60_preGPU_protection_failure_and_merged_topology_resume_request_v1')
        self.reserved_committed60=self.committed60_initial
        self.ACKed40_initial=request.get('kind')=='ASR18_exact_ACKed40_reserved_p3_readonly_authority_resume_request_v1'
        self.pending40_initial=request.get('kind')=='ASR18_exact_completed40_pending_publication_cleanup_join_resume_request_v1'
        self.pending40_finalization=False
        self.resource_initial_pending=request.get('kind')=='ASR18_exact_coherent_resource_stop_resume_request_v1'
        self.acknowledged_resource_pending=request.get('kind')=='ASR18_exact_ACKed_resource_reference0_CPU_resume_request_v1'
        self.resource_closure=c['resource_native_closure']
        checked_compatibility(c['operational_compatibility'],self.identity)
        self.resume_snapshot=(checked_committed60_merged_resume_request(self.config_pin,r,resume_request,guard=lambda:None)
            if self.committed60_initial else checked_ACKed40_reader_resume_request(self.config_pin,r,resume_request,guard=lambda:None)
            if self.ACKed40_initial else checked_pending40_resume_request(self.config_pin,r,resume_request,guard=lambda:None)
            if self.pending40_initial else checked_acknowledged_resource_resume_request(self.config_pin,r,resume_request,guard=lambda:None)
            if self.acknowledged_resource_pending else checked_resource_resume_request(self.config_pin,r,resume_request,guard=lambda:None)
            if self.resource_initial_pending else checked_normal_resume_request(self.config_pin,r,resume_request,
                parent_source=parent_source,guard=lambda:None))
        require(self.root.is_dir() and self.control.is_dir(),'Existing original production namespace only')
        administrative=r.get('administrative_maximum_new_updates')
        require(administrative is None or (not c['owner_validation_only'] and type(administrative) is int
            and 4<=administrative<=c['planned_native_updates']),'Explicit reviewed invocation-only administrative budget')
        self.administrative_maximum_new_updates=administrative
        require(self.folder.is_relative_to(ROOT) and self.root.is_relative_to(ROOT/'runs')
                and not self.folder.exists(),'Fresh actual outer supervisor output folder per invocation')
        self.folder.mkdir()
        if self.resume_snapshot is None:self.root.mkdir();self.control.mkdir()
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
            snapshot=(checked_committed60_merged_resume_request(self.config_pin,r,resume_request,guard=self.parent.check)
                if self.committed60_initial else checked_ACKed40_reader_resume_request(self.config_pin,r,resume_request,guard=self.parent.check)
                if self.ACKed40_initial else checked_pending40_resume_request(self.config_pin,r,resume_request,guard=self.parent.check)
                if self.pending40_initial else checked_acknowledged_resource_resume_request(self.config_pin,r,resume_request,guard=self.parent.check)
                if self.acknowledged_resource_pending else checked_resource_resume_request(self.config_pin,r,resume_request,guard=self.parent.check)
                if self.resource_initial_pending else checked_normal_resume_request(self.config_pin,r,resume_request,
                    parent_source=parent_source,guard=self.parent.check))
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
            self.durable=PackedDurableLifecycleAdapter(self.lifecycle,self.root/'durable',policy=c['durable_policy'],proofs=proofs,create=False,
                resume_proof=None if self.resource_initial_pending or self.committed60_initial else self.current_ack,
                operational_compatibility=c['operational_compatibility'],
                resource_resume=self.resource_closure if self.resource_initial_pending else None,
                committed60_CPU_resume=c['prior_committed60_CPU_memory_failure'] if self.committed60_initial else None)
            require(self.durable.owner==read_bound(c['session'],self.parent.check)['durable_owner'],
                    'Actual newly written deterministic owner equals session expected binding, never an invented actual receipt')
            self.store=(IndexedOwnedFutureCheckpointStore(snapshot['storage_owner'],guard=self.parent.check,operational_compatibility=c['operational_compatibility']) if resuming else
                IndexedOwnedFutureCheckpointStore.create(c['checkpoint_storage_namespace'],c['checkpoint_storage_policy'],
                    ROOT_review=c['ROOT_storage_review'],run_identity=self.identity,verifier_source=record(__file__),
                    protected=c['protected_external'],guard=self.parent.check))
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
        import run_ASR18_BPE9223_CPU_leaf_observer_v8 as CPU_leaf_delegate
        require(self.CPU_leaf_execution_delegate_source==compatibility['CPU_leaf_execution_delegate_source']
            and record(inspect.getsourcefile(CPU_leaf_delegate.run_leaf))==self.CPU_leaf_execution_delegate_source
            and c['CPU_controller_source']==compatibility['operational_sources']['CPU_orchestrator'],
            'Overall orchestrator8 and actual leaf Popen observer7 are separately source bound')
        if self.acknowledged_resource_pending:
            from asr18_BPE9223_full_run_controller_v6 import FullOwner as OriginalController6
            self.acknowledged_reentry=verify_acknowledged_resource_reentry(identity=self.identity,
                operational_compatibility=c['operational_compatibility'],resolve_member=self.resolve_member,
                legacy_resolve_member=OriginalController6.resolve_member.__get__(self,type(self)),guard=self.parent.check)
            require(digest(read_bound(self.current_capture['next_draw_packet'],self.parent.check))==snapshot['next_draw_packet_sha256'],
                'Actual archived saved nextdraw semantic digest after source-bound resolver initialization, before CPU/GPU')
        if self.resource_initial_pending:
            completion=read_bound(self.current_completion,self.parent.check)
            require(completion['checkpoints']==[self.current_capture]
                and completion['global_new_updates']==self.schedule.committed_updates(),
                'Actual archived completed p1 and exact current capture')
            release=read_bound(self.current_release,self.parent.check)
            require(release['request_nonce']==self.current_response['nonce']
                and release['checkpoint_pointer']==self.current_capture['pointer'],
                'Original archived closed-state response nonce and release checkpoint')
            require(digest(read_bound(self.current_capture['next_draw_packet'],self.parent.check))==snapshot['next_draw_packet_sha256'],
                'Canonical semantic nextdraw digest independently checked from exact archived bytes')
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
        raw=canonical(event);encoder=zlib.compressobj(level=9,zdict=WITNESS_ZDICT)
        packed=encoder.compress(raw)+encoder.flush()
        require(len(raw)<=65536 and len(packed)<=2048,'Complete lossless witness frame cap; no field dropping')
        log=self.control/'storage_operation_witnesses.bin'
        offset=log.stat().st_size if log.exists() else 0
        frame=struct.pack('<II',len(packed),len(raw))+hashlib.sha256(raw).digest()+packed
        require(offset+len(frame)<=805306368,'Entire declared operation-witness control horizon')
        self.parent.control_guard(log,offset+len(frame))
        with log.open('ab') as stream:stream.write(frame);stream.flush();os.fsync(stream.fileno())
        with log.open('rb') as stream:stream.seek(offset);actual=stream.read(len(frame))
        decoder=zlib.decompressobj(zdict=WITNESS_ZDICT)
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
        pool=self.parent.control_pool;state=read_bound(pool.state_pin,self.parent.check)
        folders=set()
        if self.gpu is not None:folders.add(str(self.gpu.folder))
        for key in ('closed_native','closed_recovery'):
            value=getattr(self,key,None)
            if value is not None:folders.add(str(Path(value['path']).parent))
        if self.config.get('prior_committed60_CPU_memory_failure') is not None:
            stopped60=read_bound(self.config['prior_committed60_CPU_memory_failure'],self.parent.check)
            folders.add(str(Path(stopped60['native_boot']['path']).parent))
        resource=read_bound(self.resource_closure,self.parent.check)
        folders.add(str(Path(resource['old_native_boot']['path']).parent))
        # At most three real DEV boundaries. Keep their canonical process/report
        # paths physical; public qualification validators require those paths.
        folders.update(x['folder'] for x in state['live'] if Path(x['folder']).name.startswith('DEV_'))
        folders&={x['folder'] for x in state['live']}
        files=[self.config_pin,self.config['session'],self.config['prior_operational_failure'],self.config['prior_acknowledged_resource_stop'],self.config['prior_CPU_reference_failure']]
        if self.config.get('current_cleanup_join_failure') is not None:
            failed=read_bound(self.config['current_cleanup_join_failure'],self.parent.check)
            folders.add(str(Path(failed['actual_native_boot']['path']).parent))
            files.append(self.config['current_cleanup_join_failure'])
        files.extend(x['configuration'] for x in state['live'] if x['folder'] in folders)
        if self.resume_request is not None:
            request=read_bound(self.resume_request,self.parent.check);files.append(self.resume_request)
            if request['kind'] in ('ASR18_exact_committed60_CPU_memory_and_merged_topology_resume_request_v1',
                'ASR18_exact_committed60_preGPU_protection_failure_and_merged_topology_resume_request_v1',
                'ASR18_exact_coherent_resource_stop_resume_request_v1','ASR18_exact_ACKed_resource_reference0_CPU_resume_request_v1','ASR18_exact_completed40_pending_publication_cleanup_join_resume_request_v1','ASR18_exact_ACKed40_post_worker_control_retirement_failure_resume_request_v1','ASR18_exact_ACKed40_reserved_p3_readonly_authority_resume_request_v1'):
                old=read_bound(self.resource_closure,self.parent.check)
                folders.add(str(Path(old['old_native_boot']['path']).parent))
                files.extend((self.resource_closure,self.config['operational_compatibility']))
                if request['kind'].startswith('ASR18_exact_committed60_'):
                    files.append(self.config['prior_committed60_CPU_memory_failure'])
            elif request['kind']=='ASR18_exact_normal_boundary_owner_resume_request_v1':
                files.extend(request[k] for k in ('normal_pause_snapshot','prior_owner_report','prior_owner_actual_TOOL0','prior_owner_guard'))
            else:require(False,'Unsupported worker-protection request kind; no implicit normal fallback')
        if self.config.get('prior_committed60_preGPU_failure') is not None:
            failed_pin=self.config['prior_committed60_preGPU_failure'];failed=read_bound(failed_pin,self.parent.check)
            files.extend((failed_pin,failed['configuration'],failed['resume_request'],failed['owner_guard']))
        if self.config.get('prior_CPU_metadata_reader_failure') is not None:
            files.append(self.config['prior_CPU_metadata_reader_failure'])
        if self.config.get('prior_acknowledged_cleanup_join_stop') is not None:
            files.append(self.config['prior_acknowledged_cleanup_join_stop'])
        if self.config.get('prior_pending40_preGPU_failure') is not None:
            latest_pin=self.config['prior_pending40_preGPU_failure'];latest=read_bound(latest_pin,self.parent.check)
            files.append(latest_pin);files.extend(latest['preserved_failed_recovery_control'].values())
        if self.config.get('cleanup_reconciliation_process0') is not None:
            reconcile=self.config['cleanup_reconciliation_process0'];proof=read_bound(reconcile,self.parent.check)
            require(Path(reconcile['path']).resolve().is_relative_to(ROOT/'analysis/BPE9223_training_launch_v1/cleanup_join_reconciliation_v2'),
                'Exact external owned reconciliation evidence remains physical through ACK/next CPU')
            files.extend((reconcile,proof['report'],proof['guard_closure']))
        unique={x['path']:x for x in files}
        return sorted(folders),[unique[key] for key in sorted(unique)]

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
        for role in ('original_source_manifest','operational_source_manifest','prior_operational_source_manifest'):
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
        if self.committed60_initial:
            require(request['committed60_CPU_resume']==self.config['prior_committed60_CPU_memory_failure']
                and request['fresh_recovery_still_required'] is True and request['no_authoritative_advance'] is True
                and request['state']['lifecycle']==self.resume_snapshot['lifecycle_state']
                and request['state']['pending']['first_optimizer_update']==41 and request['state']['pending']['last_optimizer_update']==60,
                'Exact real60 pending-head6 reopen only, no old40 protocol reset')
            from asr18_BPE9223_persistence_prefix_v10 import checked_committed60_CPU_failure
            checked_committed60_CPU_failure(request['committed60_CPU_resume'],identity=self.identity,
                operational_compatibility=self.config['operational_compatibility'],guard=self.parent.check)
            return True
        if self.resource_initial_pending:
            require(request['resource_resume']==self.resource_closure
                and request['fresh_recovery_still_required'] is True
                and request['no_authoritative_advance'] is True
                and request['state']['lifecycle']==self.resume_snapshot['lifecycle_state']
                and request['state']['pending']['publications']==1
                and request['state']['last_recovery'] is None,
                'Exact resource-pending20 reopen, never normal recovery acknowledgement')
            checked_resource_stop(self.resource_closure,identity=self.identity,
                compatibility=self.config['operational_compatibility'],guard=self.parent.check)
            return True
        self.check();require(self.gpu is None and self.current_ack is not None and request['restore_proof']==self.current_ack
            and request['identity']==self.identity and request['namespace']==str(self.root/'durable')
            and request.get('native_replay_or_sampler_rollback_permitted') is False,'Exact current recovery before same-owner durable reopen')
        value=read_bound(self.current_ack,self.parent.check)
        require(request['state']['last_recovery']==self.current_ack and request['state']['pending']['publications']==0
                and value['final_checkpoint_pointer']==self.current_capture['pointer'],'No older lifecycle restore or unreconciled orphan')
        self._verify_closed(self.closed_native);self._verify_closed(self.closed_recovery)
        return True

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
        from run_ASR18_BPE9223_CPU_window_v10 import checked_task_completion
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
            'merged_parent_producer':self.producer,'merged_parent_process_exit0_claimed':False,
            'legacy_all_CPU_auxiliary_producers_actually_closed0_scope':'actual_numerical_reference_frontend_and_external_cleanup_only',
            'metadata_task_completed_no_CPU_inflight':True,'every_preserved_control_attempt_closed0_claimed':False,
            'preserved_prior_preparation_failure':self.config['prior_committed60_CPU_memory_failure']})

    def _pending40_producer_table(self):
        from fastmss_owned_process_closure_v2 import dead
        report=read_bound(self.current_CPU_report,self.parent.check);self._verify_aggregate_CPU_report(report)
        normal=[]
        for role,key in (('reference','reference_actual_aggregate0'),('frontend','frontend_actual_aggregate0')):
            item=report[key];proof=read_bound(item['actual0'],self.parent.check)
            verify(proof,kind='ASR18_owned_CPU_process_completion_v1',owner='/root/cpu_data_preparation',report=item['report'],guard=self.parent.check)
            normal.append({'role':role,'actual_process0':item['actual0'],'report':item['report'],'actual_exit_code':0})
        proof=read_bound(self.current_CPU0,self.parent.check)
        verify(proof,kind='ASR18_owned_CPU_process_completion_v1',owner=OWNER,report=self.current_CPU_report,guard=self.parent.check)
        normal.append({'role':'successful_feature_window_orchestrator','actual_process0':self.current_CPU0,'report':self.current_CPU_report,'actual_exit_code':0})
        failed=[]
        for pin in self.config['preserved_failed_metadata_controls']:
            value=read_bound(pin,self.parent.check)
            require(type(value['actual_tool_exit_code']) is int and type(value['actual_exec_tool_exit_code']) is int
                and value['actual_tool_exit_code']==value['actual_exec_tool_exit_code']==1
                and value['all_captured_identities_dead'] is True and all(dead(i) for i in value['all_captured_identities']),
                'Preserved metadata/orchestrator failure actual1/dead; never classified payload producer0')
            failed.append({'role':'preserved_failed_metadata_control','actual_failure':pin,'actual_exit_code':1,
                'all_captured_identities_dead':True,'normal_process0_claimed':False})
        require(len(failed)==2,'Only original failed CPU7 metadata and failed readonly helper1 controls in this AUX')
        return self._new(Path(self.current_workspace['workspace'])/'metadata_producer_completion_scopes.json',{
            'kind':'ASR18_explicit_payload_producer_and_failed_control_completion_scopes_v1','source':record(__file__),
            'configuration':self.config_pin,'active_numerical_and_successful_window_producers':normal,
            'preserved_failed_metadata_controls':failed,'readonly_cleanup_reconciliation_process0':self.current_cleanup0,
            'legacy_all_CPU_auxiliary_producers_actually_closed0_scope':'active_reference_frontend_and_successful_feature_window_payload_producers_only',
            'every_preserved_control_attempt_closed0_claimed':False})

    def _verify_pending40_quiescence(self):
        require(self.pending40_finalization and self.gpu is None and self.closed_native==self.closed_recovery,
            'Only explicit pending40 finalization under one genuine current restored-process0')
        from asr18_BPE9223_persistence_prefix_v10 import checked_cleanup_join_failure
        failed=checked_cleanup_join_failure(self.config['current_cleanup_join_failure'],identity=self.identity,
            operational_compatibility=self.config['operational_compatibility'],guard=self.parent.check,
            reader=lambda pin:read_bound(pin,self.parent.check))
        report=read_bound(self.current_recovery_report,self.parent.check)
        verify(read_bound(self.closed_recovery,self.parent.check),kind='ASR18_owned_GPU_process_completion_v1',
            owner=OWNER,report=self.current_recovery_report,guard=self.parent.check)
        require(report['kind']=='ASR18_full_run_actual_CURRENT40_checkpoint_recovered_after_cleanup_join_v1'
            and report['current_checkpoint']==self.current_capture
            and report['current_cleanup_join_failure']==self.config['current_cleanup_join_failure']
            and self.current_capture['pointer']==failed['checkpoint_pointer']
            and report['cleanup_reconciliation_process0']==self.current_cleanup0
            and report['cleanup_reconciliation_report']==self.current_cleanup,
            'Actual fresh40 exact recovery and distinct physical-cleanup reconciliation0 before metadata writes')
        reconciliation=read_bound(self.current_cleanup,self.parent.check)
        verify(read_bound(self.current_cleanup0,self.parent.check),kind='ASR18_owned_CPU_process_completion_v1',
            owner=OWNER,report=self.current_cleanup,guard=self.parent.check)
        require(reconciliation['physical_cleanup_performed_in_this_invocation'] is False
            and reconciliation['original_cleanup_report']==failed['actual_cleanup_report'],
            'No old cleanup handle or re-deletion fabricated')
        return True

    def _seal_reconciled_empty_CPU_logs(self):
        import stat
        self._verify_pending40_quiescence();root=Path(self.current_workspace['workspace']);rows=[]
        report=read_bound(self.current_CPU_report,self.parent.check);self._verify_aggregate_CPU_report(report)
        contract=read_bound(report['source_contract'],self.parent.check)
        normal=[(Path(self.current_CPU0['path']).parent/'worker.log',self.current_CPU0,self.config['CPU_controller_source'],self.CPU_execution_delegate_source)]
        for kind,key in (('reference','reference_actual_aggregate0'),('frontend','frontend_actual_aggregate0')):
            normal.append((root/'CPU/payload'/(kind+'_process')/'worker.log',report[key]['actual0'],
                contract['sources'][kind],self.CPU_leaf_execution_delegate_source))
        for path,pin,worker,parent in normal:
            proof=read_bound(pin,self.parent.check)
            verify(proof,kind='ASR18_owned_CPU_process_completion_v1',owner=proof['owner'],guard=self.parent.check)
            require(proof['worker_source']==worker and proof['parent_source']==parent
                and Path(pin['path']).parent==path.parent, 'Actual producer0 of optional original empty log')
            require(path.is_relative_to(root) and path.name=='worker.log' and not path.is_symlink(), 'Exact workspace ordinary log')
            if path.stat().st_size==0:rows.append({'original':record(path),'actual_process0':pin,'authority_scope':'actual_original_producer0'})
        reconciliation=read_bound(self.current_cleanup,self.parent.check)
        old=reconciliation['original_cleanup_empty_worker_log'];path=Path(old['path'])
        require(path==root/'cleanup/worker.log' and record(path)==old and old['bytes']==0
            and old['sha256']==hashlib.sha256(b'').hexdigest() and not path.is_symlink(),
            'Exact historical zero-byte log independently reconciled; new helper did not produce this log')
        rows.append({'original':old,'readonly_reconciliation_process0':self.current_cleanup0,
            'readonly_reconciliation_report':self.current_cleanup,'authority_scope':'verified_historical_physical_cleanup_only'})
        require(len(rows)<=4,'Existing four optional zero-byte log reservation')
        intent=self._new(root/'empty_CPU_log_retirement_intent.json',{
            'kind':'ASR18_exact_reconciled_closed_empty_CPU_log_retirement_intent_v1','owner':OWNER,
            'source':record(__file__),'configuration':self.config_pin,'workspace':self.current_workspace,
            'current_cleanup_join_failure':self.config['current_cleanup_join_failure'],
            'actual_fresh40_process0':self.closed_recovery,'cleanup_reconciliation_process0':self.current_cleanup0,
            'empty_logs':rows,'old_cleanup_process0_claimed':False,'nonempty_log_content_retirement_permitted':False})
        for row in rows:
            path=Path(row['original']['path'])
            for ancestor in (path,*path.parents):
                info=ancestor.lstat();require(not ancestor.is_symlink() and not getattr(info,'st_file_attributes',0)&0x400,'No reparse owned log or ancestor')
                if ancestor==ROOT:break
            info=path.lstat()
            require(stat.S_ISREG(info.st_mode) and not info.st_file_attributes&0x400
                and record(path)==row['original'] and info.st_size==0 and record(intent['path'])==intent,
                'Same exact empty bytes and durable producer/reconciliation metadata before unlink')
            path.unlink();require(not path.exists(),'Exact optional zero-byte member retired')
        return self._new(root/'empty_CPU_log_retirement_completed.json',{
            'kind':'ASR18_exact_reconciled_closed_empty_CPU_log_retirement_completed_v1','passed':True,
            'intent':intent,'retired_exact_empty_logs':len(rows),'discarded_content_bytes':0,
            'all_original_empty_path_SHA_and_producer_proofs_retained':True})

    def _finish_pending40(self):
        require(self.pending40_initial and self.schedule.committed_updates()==20 and self.gpu is None,
            'Only reopen old protocol20 to finish already-completed40; no training or new prepare')
        data=self.pending40_inputs;self.before_window=deepcopy(self.schedule.state_dict())
        self.current_response=deepcopy(data['response']);self.current_completion=data['completion']
        self.current_capture=deepcopy(data['capture']);self.current_release=data['release'];self.current_idle=None
        self.current_packet=deepcopy(data['packet']);self.current_intent=self.current_packet['checkpoint_intent']
        self.current_workspace=self.current_packet['auxiliary_workspace'];self.current_original_plan=self.current_packet['coordinator_plan']
        self.current_CPU0=data['CPU0'];self.current_CPU_report=data['CPU_report'];self.current_bridge=data['bridge']
        self.current_cleanup0=data['cleanup_reconciliation_process0'];self.current_cleanup=data['cleanup_reconciliation_report']
        require(self.recovery_control_sequence<512 and self.recovery_sequence<513,'Unchanged cumulative recovery quotas')
        self.recovery_control_sequence+=1;controls=self.control/('recovery_'+str(self.recovery_control_sequence).zfill(6));controls.mkdir(exist_ok=False)
        request={'kind':'ASR18_full_run_cleanup_join_CURRENT40_checkpoint_fresh_recovery_request_v1',
            'identity':self.identity,'planning_arguments':self.config['planning_arguments'],
            'augmentation_identity':self.config['augmentation_identity'],
            'current_cleanup_join_failure':self.config['current_cleanup_join_failure'],
            'current_checkpoint_capture':self.current_capture,'cleanup_reconciliation_process0':self.current_cleanup0,
            'cleanup_reconciliation_report':self.current_cleanup,
            'native_worker_source':read_bound(self.config['current_cleanup_join_failure'],self.parent.check)['runtime_source'],
            'native_parent_source':read_bound(self.config['current_cleanup_join_failure'],self.parent.check)['parent_source'],
            'recovery_worker_source':self.config['worker_driver_source'],'recovery_parent_source':self.parent_source,
            'operational_compatibility':self.config['operational_compatibility']}
        rp=self._new(controls/'request.json',request)
        worker={**self.config,'resource_journal_handoff':False,'cleanup_join_journal_handoff':True,
            'worker_mode':'restore_only','current_recovery_request':rp,'source_checkpoint_capture':self.current_capture,
            'journal_namespace':str(self.root/'journals'),'journal_resume_sequence':self.recovery_sequence+1,
            'entire_worker_forecast':self.config['final_recovery_worker_forecast']}
        cp=self._new(controls/'worker_configuration.json',worker)
        self.gpu=self.GPUService.launch(cp['path'],self.folder/('recovery_'+str(self.recovery_sequence+1).zfill(6)),self.parent)
        self.parent.GPU=self.gpu;self.recovery_sequence+=1
        ready=self._wait_ready(self.gpu.folder/'current_recovery_ready.json');self.current_recovery_report=ready['report']
        self.closed_recovery=self.gpu.finish();self.gpu=None;self.parent.GPU=None
        self.closed_native=self.closed_recovery  # Genuine CURRENT restored40 authority; original8376 remains abnormal in b949.
        self.pending40_finalization=True;self._verify_pending40_quiescence()
        plan=read_bound(self.current_original_plan,self.parent.check)
        self._archive_and_publish(plan,self.current_completion);del plan
        evidence={'kind':'ASR18_full_run_cleanup_join_current40_recovery_prefix_inputs_v1','owner':OWNER,
            'diagnostic_only':False,'run_identity_sha256':digest(self.identity),
            'pending_head':self.durable.head_pin,'current_checkpoint_capture':self.current_capture,
            'current_cleanup_join_failure':self.config['current_cleanup_join_failure'],
            'cleanup_reconciliation_process0':self.current_cleanup0,'cleanup_reconciliation_report':self.current_cleanup,
            'actual_recovery_process0':self.closed_recovery,'recovery_report':self.current_recovery_report,
            'operational_compatibility':self.config['operational_compatibility']}
        ep=self._new(controls/'evidence.json',evidence)
        request=self.durable._request(operational_compatibility=self.config['operational_compatibility'],
            current_recovery_evidence=ep,coverage_output=str(self.root/'durable'/('coverage_'+str(self.recovery_sequence).zfill(6)+'.json')))
        with self.parent.metadata_phase('read_index'):
            coverage=create_coverage(request,resolve_member=self.resolve_member,guard=self.parent.check)
            self.current_coverage=coverage;value=read_bound(coverage,self.parent.check)
            ack={'kind':'ASR18_cleanup_join_failure_current40_recovery_and_verified_persistence_prefix_v1',
                'diagnostic_only':False,'run_identity_sha256':digest(self.identity),'pending':deepcopy(self.durable.head['pending']),
                'pending_head':self.durable.head_pin,'final_checkpoint_pointer':self.current_capture['pointer'],
                'fresh_recovery_scope':'current_latest_only','all_persistence_verified_through_pending_prefix':True,
                'all_historical_checkpoints_freshly_restored':False,
                'current_latest_all207_threeAdam_all4RNG_sampler_prefix_nextdraw_match_actual_pre_save':True,
                'actual_recovery_native_updates':0,'actual_recovery_draws_consumed':0,
                'actual_cleanup_join_failure_quiescence_proven':True,'actual_native_normal_exit0_claimed':False,
                'resource_cause':False,'current_cleanup_join_failure':self.config['current_cleanup_join_failure'],
                'actual_closed_native_index':self.closed_recovery,'actual_closed_recovery_index':self.closed_recovery,
                'current_closed_authority_role':'genuine_fresh_restored40_process0_never_original_training_exit0',
                'cleanup_reconciliation_process0':self.current_cleanup0,'cleanup_reconciliation_report':self.current_cleanup,
                'persistence_prefix_coverage_index':coverage,'current_latest_exact_restore':value['current_latest_exact_restore'],
                'current_recovery_evidence':ep,'operational_compatibility':self.config['operational_compatibility']}
            self.current_ack=self._new(controls/'acknowledgement.json',ack);self.durable.acknowledge_recovery(self.current_ack)
        self.last_window_index=1;self.invocation_initial_updates=40
        require(self.schedule.committed_updates()==40 and self.durable.head['pending']['publications']==0,
            'Actual completed40 fully published/ACKed before any next CPU/native draw')
        self.pending40_finalization=False;self.pending40_initial=False;self.pending40_inputs=None
        self._refresh_boundary();self._retire_obsolete();self._archive_obsolete_worker_controls();gc.collect()

    def _verify_closed(self,pin):
        if pin==self.config.get('prior_committed60_CPU_memory_failure'):
            from asr18_BPE9223_persistence_prefix_v10 import checked_committed60_CPU_failure
            return checked_committed60_CPU_failure(pin,identity=self.identity,operational_compatibility=self.config['operational_compatibility'],guard=self.parent.check)
        if pin==self.resource_closure:
            return checked_resource_stop(pin,identity=self.identity,compatibility=self.config['operational_compatibility'],guard=self.parent.check)
        proof=read_bound(pin,self.parent.check);verify(proof,kind=proof['kind'],owner=OWNER,guard=self.parent.check);return proof

    def _read_state(self):
        with self.store.locked():return self.store._load(allow_preparing_auxiliary=True)[0]

    def _current_capture_writer_source(self):
        if self.config.get('prior_committed60_CPU_memory_failure') is not None:
            failed=read_bound(self.config['prior_committed60_CPU_memory_failure'],self.parent.check)
            if self.current_capture is not None and self.current_capture['pointer']==failed['checkpoint_pointer']:
                return failed['runtime_source']
        if self.config.get('current_cleanup_join_failure') is not None:
            failed=read_bound(self.config['current_cleanup_join_failure'],self.parent.check)
            if self.current_capture is not None and self.current_capture['pointer']==failed['checkpoint_pointer']:
                return failed['runtime_source']
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
        import fastmss_BPE9223_plan_bridge_v8 as bridge
        require(record(bridge.__file__)==self.config['CPU_bridge_source'],'Exact operational CPU bridge')
        return bridge.bridge_plan(*args,**kwargs)

    def _prepare_metadata(self,plan,index):
        require(not self.pending40_initial,'No new prepare before actual fresh40/publication/ACK')
        if getattr(self,'prepared_CPU_resume',None) is not None:
            prepared=self.prepared_CPU_resume
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
        from asr18_BPE9223_CPU_window_configuration_v7 import derive_contract,derive_configuration
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

    def _recover_committed60(self):
        require(self.committed60_initial and self.gpu is None and self.schedule.committed_updates()==60
            and self.actual_owner_invocation_count==(11 if self.config.get('prior_committed60_preGPU_failure') is not None else 10)
            and self.native_sequence==3
            and self.recovery_sequence==2 and self.recovery_control_sequence==4,
            'Mandatory fresh saved60 is first GPU operation; actual counters never reset')
        failed=read_bound(self.config['prior_committed60_CPU_memory_failure'],self.parent.check)
        self.recovery_control_sequence+=1
        controls=self.control/('recovery_'+str(self.recovery_control_sequence).zfill(6));controls.mkdir(exist_ok=False)
        request={'kind':'ASR18_full_run_committed60_CPU_memory_CURRENT_checkpoint_fresh_recovery_request_v1',
            'identity':self.identity,'planning_arguments':self.config['planning_arguments'],
            'augmentation_identity':self.config['augmentation_identity'],
            'current_committed60_CPU_failure':self.config['prior_committed60_CPU_memory_failure'],
            'current_checkpoint_capture':self.current_capture,'operational_compatibility':self.config['operational_compatibility'],
            'native_worker_source':failed['runtime_source'],'native_parent_source':failed['source'],
            'recovery_worker_source':self.config['worker_driver_source'],'recovery_parent_source':self.parent_source}
        request_pin=self._new(controls/'request.json',request)
        worker={**self.config,'resource_journal_handoff':False,'cleanup_join_journal_handoff':False,
            'committed60_CPU_journal_handoff':True,'worker_mode':'restore_only','current_recovery_request':request_pin,
            'journal_namespace':str(self.root/'journals'),'journal_resume_sequence':self.recovery_sequence+1,
            'entire_worker_forecast':self.config['final_recovery_worker_forecast']}
        cp=self._new(controls/'worker_configuration.json',worker)
        self.gpu=self.GPUService.launch(cp['path'],self.folder/('recovery_'+str(self.recovery_sequence+1).zfill(6)),self.parent)
        self.parent.GPU=self.gpu;self.recovery_sequence+=1
        ready=self._wait_ready(self.gpu.folder/'current_recovery_ready.json');self.current_recovery_report=ready['report']
        self.closed_recovery=self.gpu.finish();self.gpu=None;self.parent.GPU=None
        self.closed_native=self.closed_recovery
        report=read_bound(self.current_recovery_report,self.parent.check)
        require(report['kind']=='ASR18_full_run_actual_CURRENT60_checkpoint_recovered_after_CPU_memory_stop_v1'
            and report['current_checkpoint']==self.current_capture
            and report['current_committed60_CPU_failure']==self.config['prior_committed60_CPU_memory_failure']
            and report['actual_native_updates_in_recovery_process']==report['actual_draws_consumed_in_recovery']==0,
            'Actual distinct fresh60 exactstate0, never original42192 normal0')
        evidence={'kind':'ASR18_full_run_committed60_CPU_memory_recovery_prefix_inputs_v1','owner':OWNER,
            'diagnostic_only':False,'run_identity_sha256':digest(self.identity),'pending_head':self.durable.head_pin,
            'current_checkpoint_capture':self.current_capture,
            'current_committed60_CPU_failure':self.config['prior_committed60_CPU_memory_failure'],
            'actual_recovery_process0':self.closed_recovery,'recovery_report':self.current_recovery_report,
            'operational_compatibility':self.config['operational_compatibility']}
        ep=self._new(controls/'evidence.json',evidence)
        request=self.durable._request(operational_compatibility=self.config['operational_compatibility'],
            current_recovery_evidence=ep,coverage_output=str(self.root/'durable'/('coverage_'+str(self.recovery_sequence).zfill(6)+'.json')))
        with self.parent.metadata_phase('read_index'):
            self.current_coverage=create_coverage(request,resolve_member=self.resolve_member,guard=self.parent.check)
            value=read_bound(self.current_coverage,self.parent.check)
            ack={'kind':'ASR18_committed60_CPU_memory_stop_current_recovery_and_verified_persistence_prefix_v1',
                'diagnostic_only':False,'run_identity_sha256':digest(self.identity),'pending':deepcopy(self.durable.head['pending']),
                'pending_head':self.durable.head_pin,'final_checkpoint_pointer':self.current_capture['pointer'],
                'fresh_recovery_scope':'current_latest_only','all_persistence_verified_through_pending_prefix':True,
                'all_historical_checkpoints_freshly_restored':False,
                'current_latest_all207_threeAdam_all4RNG_sampler_prefix_nextdraw_match_actual_pre_save':True,
                'actual_recovery_native_updates':0,'actual_recovery_draws_consumed':0,
                'actual_CPU_memory_failure_quiescence_proven':True,'actual_native_normal_exit0_claimed':False,'resource_cause':True,
                'current_committed60_CPU_failure':self.config['prior_committed60_CPU_memory_failure'],
                'actual_closed_native_index':self.closed_recovery,'actual_closed_recovery_index':self.closed_recovery,
                'current_closed_authority_role':'genuine_fresh_restored60_process0_never_original_training_exit0',
                'persistence_prefix_coverage_index':self.current_coverage,'current_latest_exact_restore':value['current_latest_exact_restore'],
                'current_recovery_evidence':ep,'operational_compatibility':self.config['operational_compatibility']}
            self.current_ack=self._new(controls/'acknowledgement.json',ack);self.durable.acknowledge_recovery(self.current_ack)
        self.committed60_initial=False;self.invocation_initial_updates=60
        require(self.schedule.committed_updates()==60 and self.durable.head['pending']['publications']==0,
            'Actual fresh60/prefixACK60 before p4 CPU or any native draw')
        self.resume_snapshot={'existing_exact_resume_verified':True};self._refresh_boundary();gc.collect()

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
        from asr18_BPE9223_CPU_window_configuration_v7 import derive_contract,derive_configuration
        sc=self._new(root/(suffix+'_CPU_source_contract.json'),derive_contract(self.config['CPU_template_configuration'],bp,self.parent.check))
        proposal=self._new(root/(suffix+'_CPU_window_configuration.json'),derive_configuration(
            self.config['CPU_template_configuration'],sc,index,intent['id'],str(cpu/'payload'),self.parent.check))
        self.current_intent=deepcopy(intent);self.reserved_preparation=None;self.reserved_committed60=False
        return {'phase':plan['phase'],'plan_pin':plan_pin,'window_index':index,'intent':intent,'workspace':workspace,
            'bridge_pin':bp,'CPU_contract':sc,'CPU_folder':str(cpu),'CPU_proposal':proposal,
            'arguments':['--config',proposal['path'],'--ROOT-review',self.config['ROOT_CPU_execution_review']['path'],'--folder',str(cpu/'payload')]}

    def _recover_resource_stop(self):
        require(self.resource_initial_pending and self.gpu is None and self.recovery_sequence==0,
            'One exact first fresh20 recovery before CPU/new native/draw')
        require(self.recovery_control_sequence==1,'Exact preserved failed first control attempt')
        self.recovery_control_sequence+=1;name='recovery_'+str(self.recovery_sequence+1).zfill(6)
        controls=self.control/('recovery_'+str(self.recovery_control_sequence).zfill(6));controls.mkdir(exist_ok=False)
        original=read_bound(self.resource_closure,self.parent.check)
        request={'kind':'ASR18_full_run_resource_stop_CURRENT_checkpoint_fresh_recovery_request_v1',
            'identity':self.identity,'planning_arguments':self.config['planning_arguments'],
            'augmentation_identity':self.config['augmentation_identity'],
            'current_native_resource_closure':self.resource_closure,
            'operational_compatibility':self.config['operational_compatibility'],
            'current_checkpoint_capture':self.current_capture,
            'native_worker_source':original['native_worker_source'],'native_parent_source':original['parent_source'],
            'recovery_worker_source':self.config['worker_driver_source'],'recovery_parent_source':self.parent_source}
        path=controls/'request.json';self._new(path,request)
        config={**self.config,'worker_mode':'restore_only','current_recovery_request':record(path),
            'resource_journal_handoff':True,'source_checkpoint_capture':self.current_capture,
            'journal_namespace':str(self.root/'journals'),'journal_resume_sequence':self.recovery_sequence+1,
            'entire_worker_forecast':self.config['final_recovery_worker_forecast']}
        cp=controls/'worker_configuration.json';self._new(cp,config)
        self.gpu=self.GPUService.launch(cp,self.folder/name,self.parent);self.parent.GPU=self.gpu
        self.recovery_sequence+=1  # An actual admitted GPU child now exists; failed control creation alone is not recovery.
        ready=self._wait_ready(self.gpu.folder/'current_recovery_ready.json');self.current_recovery_report=ready['report']
        self.closed_recovery=self.gpu.finish();self.gpu=None;self.parent.GPU=None
        require(read_bound(self.closed_recovery,self.parent.check)['report']==self.current_recovery_report,
            'SAME actual canonical fresh20 report in real normal recovery process0')
        evidence={'kind':'ASR18_full_run_resource_stop_current_recovery_prefix_inputs_v1','owner':OWNER,
            'diagnostic_only':False,'run_identity_sha256':digest(self.identity),'pending_head':self.durable.head_pin,
            'current_checkpoint_capture':self.current_capture,'current_native_resource_closure':self.resource_closure,
            'actual_recovery_process0':self.closed_recovery,'recovery_report':self.current_recovery_report,
            'operational_compatibility':self.config['operational_compatibility']}
        ep=controls/'evidence.json';self._new(ep,evidence)
        request=self.durable._request(operational_compatibility=self.config['operational_compatibility'],
            current_recovery_evidence=record(ep),coverage_output=str(self.root/'durable'/('coverage_'+str(self.recovery_sequence).zfill(6)+'.json')))
        with self.parent.metadata_phase('read_index'):
            coverage=create_coverage(request,resolve_member=self.resolve_member,guard=self.parent.check)
            self.current_coverage=coverage;value=read_bound(coverage,self.parent.check)
            ack={'kind':'ASR18_resource_stop_current_latest_recovery_and_verified_persistence_prefix_v1',
                'diagnostic_only':False,'run_identity_sha256':digest(self.identity),
                'pending':deepcopy(self.durable.head['pending']),'pending_head':self.durable.head_pin,
                'final_checkpoint_pointer':self.current_capture['pointer'],'fresh_recovery_scope':'current_latest_only',
                'all_persistence_verified_through_pending_prefix':True,'all_historical_checkpoints_freshly_restored':False,
                'current_latest_all207_threeAdam_all4RNG_sampler_prefix_nextdraw_match_actual_pre_save':True,
                'actual_recovery_native_updates':0,'actual_recovery_draws_consumed':0,
                'actual_closed_native_index':self.resource_closure,'actual_closed_recovery_index':self.closed_recovery,
                'actual_resource_quiescence_proven':True,'actual_native_normal_exit0_claimed':False,
                'persistence_prefix_coverage_index':coverage,'current_latest_exact_restore':value['current_latest_exact_restore'],
                'current_recovery_evidence':record(ep),'operational_compatibility':self.config['operational_compatibility']}
            ap=controls/'acknowledgement.json';self.current_ack=self._new(ap,ack)
            self.durable.acknowledge_recovery(self.current_ack)
        self.resource_initial_pending=False
        self._refresh_boundary();self._protect_worker_controls()

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
        from asr18_BPE9223_CPU_window_configuration_v7 import derive_contract,derive_configuration
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
        from asr18_BPE9223_CPU_window_configuration_v7 import derive_contract,derive_configuration
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
            from run_ASR18_BPE9223_CPU_window_v10 import run_merged_task,checked_task_completion
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
        from run_ASR18_BPE9223_CPU_window_v10 import checked_task_completion
        task=read_bound(packet['CPU_task_completion'],self.parent.check)
        checked_task_completion(packet['CPU_task_completion'],packet['CPU_task_report'],task['configuration'],
            self.config['operational_compatibility'],guard=self.parent.check)
        require(self.gpu is not None and self.merged_CPU_inflight is False and self.parent.merged_CPU_active is False
            and task['producer']==self.producer,'Source-bound SAME live parent, no CPU task inflight')
        witness={'kind':'ASR18_fresh_live_parent_no_CPU_inflight_native_task_handoff_v1','source':record(__file__),
            'parent_source':self.parent_source,'producer':self.producer,'task_nonce':task['task_nonce'],
            'CPU_task_completion':packet['CPU_task_completion'],'packet_semantic_sha256':digest(packet),
            'actual_CPU_leaf_inflight':False,'task_family_observer_joined':True,
            'optimizer_updates':self.schedule.committed_updates(),'sampler_sha256':digest(self.schedule.state_dict()),
            'observed_monotonic':time.monotonic()}
        packet['CPU_task_handoff_witness']=self._new(Path(self.current_workspace['workspace'])/'fresh_merged_CPU_native_handoff.json',witness)
        return packet

    def _verify_aggregate_CPU_report(self,report,contract_pin=None):
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
        worker={**self.config,'resource_journal_handoff':self.closed_native==self.resource_closure,'worker_mode':'resident_training','first_feature_update':manifest['updates'][0]['update'],
            'journal_namespace':str(self.root/'journals'),'initial_journal_path':str(self.initial_journal),
            'journal_resume_sequence':self.native_sequence,'session_owner_nonce':self.current_worker_nonce,
            'restore_parent_pointer':None if self.current_capture is None else self.current_capture['pointer']}
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
        if self.committed60_initial:self._recover_committed60()
        if self.pending40_initial:self._finish_pending40()
        if self.resource_initial_pending:self._recover_resource_stop()
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
