"""Read-only abnormal20 + genuine fresh CURRENT recovery gate for CPU handoff.

An abnormal source is never a normal native0. No audio/tensor payload is read,
no draw is consumed and no journal, checkpoint or ownership record is changed.
"""
from contextlib import contextmanager
from pathlib import Path
from bpe_broad_audio_v1 import require,record
from bpe_broad_production_state_v1 import digest
from fastmss_owned_process_closure_v2 import verify
from asr18_BPE9223_resource_operational_compatibility_v1 import checked_compatibility,checked_resource_stop
import asr18_BPE9223_operational_metadata_v2 as metadata

RESOURCE_FIELDS=frozenset({'actual_native_resource_closure','actual_fresh_recovery_process0',
    'resource_recovery_report','resource_recovery_ack','operational_compatibility'})


def same(a,b):return a==b and digest(a)==digest(b)


def configure_session(session,guard):
    """Keep process read authority stable; separately bind current callbacks."""
    if 'operational_metadata' not in session:return False
    value=session['operational_metadata']
    require(type(value) is dict and set(value)=={'storage_owner','control_root','identity','operational_compatibility','current_native_resource_closure'}
        and value['current_native_resource_closure'] is None
        and digest(value['identity'])==session['full_run_identity_sha256'],'Exact immutable operational metadata authority')
    current=checked_compatibility(session['operational_compatibility'],value['identity'],guard)
    require(same(value,current['process_readonly_metadata_authority'])
        and current['operational_sources']['CPU_resource_boundary']==record(__file__),
        'Current ROOT authority binds exact stable readonly value and actual boundary2 source')
    readonly=checked_compatibility(value['operational_compatibility'],value['identity'],guard)
    require(same(current['original_configuration'],readonly['original_configuration'])
        and same(current['original_source_manifest'],readonly['original_source_manifest'])
        and readonly['operational_sources']['CPU_resource_boundary']==record(Path(__file__).with_name('fastmss_BPE9223_resource_CPU_boundary_v1.py')),
        'Separate historical readonly boundary1 and unchanged original science lineage')
    for role in ('operational_metadata','indexed_storage','metadata_resolver'):
        require(same(current['operational_sources'][role],readonly['operational_sources'][role]),
            'Readonly resolver implementation cannot change with mutable callbacks: '+role)
    original=metadata.read_bound(session['original_session'],guard)
    for key in ('full_run_identity_sha256','initial_optimizer_updates','initial_sampler_sha256','label_population_contract','DATA_evidence_sha256',
        'augmentation_identity','scientific_source_state','durable_owner','planned_native_updates','owner_validation_only','checkpoint_diagnostic_only','production_source_eligible'):
        require(same(original[key],session[key]),'Operational session cannot alter original science/frontiers: '+key)
    require(same(session['original_session'],metadata.read_bound(readonly['original_configuration'],guard)['session']), 'Reviewed actual original session lineage')
    @contextmanager
    def phase(name):
        require(name=='read_index','Readonly bounded CPU metadata only');guard();yield;guard()
    metadata.configure(**value,guard=guard,metadata_phase=phase)
    return True

def verify_resource_boundary(boundary,*,pointer,session,guard):
    require(RESOURCE_FIELDS<=set(boundary) and 'actual_native_closed' not in boundary and 'resident_native_boundary' not in boundary,
        'One explicit abnormal closure, never combined with normal0/live authority')
    require(configure_session(session,guard) is True and same(boundary['operational_compatibility'],session['operational_compatibility']), 'New operational session required for resource handoff')
    identity=pointer['identity']['caller_identity']['full_run_lifecycle_identity']
    require(digest(identity)==session['full_run_identity_sha256'],'Exact unchanged scientific identity')
    review=checked_compatibility(boundary['operational_compatibility'],identity,guard)
    stopped=checked_resource_stop(boundary['actual_native_resource_closure'],identity=identity,compatibility=boundary['operational_compatibility'],guard=guard)
    require(same(stopped['current_checkpoint_pointer'],boundary['coherent_checkpoint_pointer']) and pointer['committed_updates']==20
        and boundary['optimizer_updates']==20 and boundary['sampler_state_sha256']==pointer['sampler_state_sha256'],'Only exact stopped current20/p1 sampler')
    read=lambda pin:metadata.read_bound(pin,guard)
    process=read(boundary['actual_fresh_recovery_process0']);report=read(boundary['resource_recovery_report']);ack=read(boundary['resource_recovery_ack'])
    verify(process,kind='ASR18_owned_GPU_process_completion_v1',owner='/root/memory_diagnosis',report=boundary['resource_recovery_report'],guard=guard)
    op=review['operational_sources']
    require(same(process['worker_source'],op['runtime']) and same(process['parent_source'],op['parent'])
        and all(not same(process['producer'],old) for old in stopped['all_captured_identities']),'Actually fresh reviewed operational recovery0, not stopped source')
    require(report['kind']=='ASR18_full_run_actual_CURRENT_checkpoint_recovered_after_resource_stop_v1' and report['passed'] is True
        and same(report['source'],op['current_recovery']) and report['diagnostic_only'] is False and report['current_latest_checkpoint_recovered'] is True
        and report['historical_fresh_GPU_restores_claimed'] is False and report['run_identity_sha256']==digest(identity)
        and same(report['current_native_resource_closure'],boundary['actual_native_resource_closure'])
        and same(report['operational_compatibility'],boundary['operational_compatibility'])
        and all(type(report[k]) is int and report[k]==0 for k in ('actual_native_updates_in_recovery_process','actual_draws_consumed_in_recovery'))
        and type(report['recovered_checkpoints']) is list and len(report['recovered_checkpoints'])==1,'Genuine sole-current resource recovery, zero new updates/draws')
    require(ack['kind']=='ASR18_resource_stop_current_latest_recovery_and_verified_persistence_prefix_v1' and ack['diagnostic_only'] is False
        and ack['run_identity_sha256']==digest(identity) and same(ack['operational_compatibility'],boundary['operational_compatibility'])
        and ack['actual_resource_quiescence_proven'] is True and ack['actual_native_normal_exit0_claimed'] is False
        and same(ack['actual_closed_native_index'],boundary['actual_native_resource_closure'])
        and same(ack['actual_closed_recovery_index'],boundary['actual_fresh_recovery_process0'])
        and same(ack['final_checkpoint_pointer'],boundary['coherent_checkpoint_pointer']) and ack['fresh_recovery_scope']=='current_latest_only'
        and ack['all_persistence_verified_through_pending_prefix'] is True and ack['all_historical_checkpoints_freshly_restored'] is False
        and ack['current_latest_all207_threeAdam_all4RNG_sampler_prefix_nextdraw_match_actual_pre_save'] is True
        and all(type(ack[k]) is int and ack[k]==0 for k in ('actual_recovery_native_updates','actual_recovery_draws_consumed')),'Exact resource ACK, no normal-exit0 relabel or frontier advancement')
    evidence=read(ack['current_recovery_evidence']);coverage=read(ack['persistence_prefix_coverage_index'])
    require(evidence['kind']=='ASR18_full_run_resource_stop_current_recovery_prefix_inputs_v1'
        and same(evidence['current_native_resource_closure'],boundary['actual_native_resource_closure'])
        and same(evidence['actual_recovery_process0'],boundary['actual_fresh_recovery_process0'])
        and same(evidence['recovery_report'],boundary['resource_recovery_report'])
        and same(evidence['operational_compatibility'],boundary['operational_compatibility'])
        and same(evidence['current_checkpoint_capture']['pointer'],boundary['coherent_checkpoint_pointer'])
        and coverage['passed'] is True and same(coverage['source'],op['prefix'])
        and same(coverage['current_recovery_evidence'],ack['current_recovery_evidence'])
        and same(coverage['final_checkpoint_pointer'],boundary['coherent_checkpoint_pointer'])
        and same(coverage['pending_head'],ack['pending_head']) and coverage['all_completed_window_PRE_saved_live_persistence_verified'] is True,
        'Source-pinned complete prefix ACK/evidence/coverage joins')
    require(type(coverage['native_updates_consumed_in_verification']) is int and coverage['native_updates_consumed_in_verification']==0
        and coverage['historical_fresh_GPU_restores_claimed'] is False,'Pure full-prefix verification, never historical fresh restores or updates')
    recovered=read(report['recovered_checkpoints'][0]);capture=evidence['current_checkpoint_capture']
    require(recovered['passed'] is True and same(recovered['pointer'],boundary['coherent_checkpoint_pointer'])
        and same(recovered['checkpoint'],pointer['checkpoint']) and same(recovered['pre_save_fingerprint'],capture['pre_save_fingerprint'])
        and same(recovered['next_draw_packet'],capture['next_draw_packet'])
        and same(ack['current_latest_exact_restore'],report['recovered_checkpoints'][0])
        and same(coverage['current_latest_exact_restore'],report['recovered_checkpoints'][0])
        and recovered['all207_threeAdam_all4RNG_match_actual_pre_save'] is True and recovered['both_samplers_journal_prefix_and_next_draw_exact'] is True
        and recovered['separate_fresh_GPU_recovery'] is True
        and all(type(recovered[k]) is int and recovered[k]==0 for k in ('native_updates_performed','draws_consumed')),'Actual full saved state/PRE/all RNG/sampler/nextdraw recovered with no draws')
    return True
